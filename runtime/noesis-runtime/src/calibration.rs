//! Startup calibration — per-machine drip ceiling.
//!
//! Spec: `HYPOTHESES.md` §H1 "Startup calibration protocol". The
//! runtime's ambient drip rate is not a hard-coded constant; it is
//! derived per machine from `tokens_per_cpu_second` (measured
//! throughput) × `fan_safe_cpu_percent` (measured or user-declared
//! silence budget) × `n_cores` × `safety_margin`.
//!
//! Phase-1 scope (this file):
//!   - Data model + TOML persistence.
//!   - Fingerprint (kernel + backend + cpu_model + n_cores) with
//!     invalidation triggers so a stale file is rejected instead of
//!     silently miscalibrating the drip loop.
//!   - Derived drip formula (`drip_rate_tokens_per_sec`).
//!   - Safe defaults (`fallback`) when no calibration exists yet.
//!
//! Deferred to a follow-up (spec'd in H1 but not implemented here):
//!   - Actual throughput measurement (needs a backend-specific
//!     `generate_burst` hook — differs for rwkv-cpp in-process vs
//!     Ollama HTTP).
//!   - Auto-detect fan-safe threshold via `hwmon` fan RPM sweep.
//!   - Interactive `noesis calibrate --interactive` CLI subcommand.
//!
//! Until measurement lands the supervisor emits a `calibration_state`
//! event on startup marking the file as `defaulted` so downstream
//! consumers know they are running against a placeholder.

// The thermal probe is exercised by its own tests + live-read test.
// Production wiring lands with the thermal-sweep commit, so the outer
// build won't call any of its symbols yet.
#[allow(dead_code)]
pub mod thermal;

use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Max age of a cached calibration before we treat it as stale.
/// Matches H1 §"Invalidation triggers" (c).
pub const CALIBRATION_TTL: Duration = Duration::from_secs(30 * 24 * 60 * 60);

/// Default buffer below the fan-safe threshold. Matches H1
/// §"Derived drip formula": `safety_margin` default `0.6`.
pub const DEFAULT_SAFETY_MARGIN: f64 = 0.6;

/// Conservative fallback used when no calibration file exists yet
/// (silence must not depend on the measurement having landed). Matches
/// the historical pilot fallback in H1 §"Reference numbers".
pub const FALLBACK_FAN_SAFE_CPU_PERCENT: f64 = 1.0;

/// Conservative fallback throughput. Matches i5-1235U + Ollama pilot
/// number from H1 §"Reference numbers"; low enough to guarantee silence
/// on any hardware, will be replaced the first time measurement runs.
pub const FALLBACK_TOKENS_PER_CPU_SECOND: f64 = 9.4;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Calibration {
    /// Tokens produced per full core-second of CPU time consumed by
    /// the inference process (summed across cores; a burst that pins
    /// 4 cores for 1 wall-second consumes 4 CPU-seconds).
    pub tokens_per_cpu_second: f64,

    /// Package CPU% at which the fans stay silent under sustained
    /// load. `top` reports package CPU on a 0..100 scale regardless
    /// of core count, matching what a user hears.
    pub fan_safe_cpu_percent: f64,

    pub cpu_model: String,
    pub n_cores: u32,
    pub kernel: String,
    pub backend: String,

    /// ISO-8601 UTC timestamp of when this calibration was produced.
    pub measured_at: String,

    /// True when values are conservative defaults rather than real
    /// measurements. Emitted so downstream consumers can flag runs
    /// that are throttled by lack-of-data rather than lack-of-headroom.
    #[serde(default)]
    pub defaulted: bool,
}

impl Calibration {
    /// Fallback calibration when no file exists on disk yet. Values
    /// are conservative — chosen so drip rate stays silent on any
    /// hardware while real measurement is still deferred.
    pub fn fallback(fingerprint: &SystemFingerprint, backend: &str) -> Self {
        Self {
            tokens_per_cpu_second: FALLBACK_TOKENS_PER_CPU_SECOND,
            fan_safe_cpu_percent: FALLBACK_FAN_SAFE_CPU_PERCENT,
            cpu_model: fingerprint.cpu_model.clone(),
            n_cores: fingerprint.n_cores,
            kernel: fingerprint.kernel.clone(),
            backend: backend.into(),
            measured_at: iso8601_now(),
            defaulted: true,
        }
    }

    /// Derived ambient drip rate in tokens per wall-second.
    ///
    /// Model: at drip rate `r` tokens/s the backend consumes
    /// `r × cpu_sec_per_token = r / tokens_per_cpu_second` CPU-seconds
    /// per wall-second. Divided by `n_cores` this is the fraction of
    /// package CPU used; expressed as a percent it must stay below
    /// `fan_safe_cpu_percent` (times `safety_margin`). Solving:
    ///
    ///     r_max = fan_safe_pct/100 × n_cores × tokens_per_cpu_second × margin
    ///
    /// Note: `HYPOTHESES.md` §H1 text version omits `× n_cores`,
    /// which reads if `fan_safe_cpu_percent` is interpreted in
    /// per-core-percent units (100% = one full core). We use package
    /// percent throughout because that matches what `top` reports and
    /// what a user actually hears. The doc will be corrected in a
    /// follow-up.
    pub fn drip_rate_tokens_per_sec(&self, safety_margin: f64) -> f64 {
        (self.fan_safe_cpu_percent / 100.0)
            * self.n_cores as f64
            * self.tokens_per_cpu_second
            * safety_margin
    }

    pub fn drip_rate_default(&self) -> f64 {
        self.drip_rate_tokens_per_sec(DEFAULT_SAFETY_MARGIN)
    }

    /// True when this calibration is invalidated by a system change.
    /// Reasons listed in returned enum for logging.
    pub fn invalidation_reason(
        &self,
        current_fp: &SystemFingerprint,
        current_backend: &str,
        now: SystemTime,
    ) -> Option<InvalidationReason> {
        if self.kernel != current_fp.kernel {
            return Some(InvalidationReason::KernelChanged {
                old: self.kernel.clone(),
                new: current_fp.kernel.clone(),
            });
        }
        if self.backend != current_backend {
            return Some(InvalidationReason::BackendChanged {
                old: self.backend.clone(),
                new: current_backend.into(),
            });
        }
        if self.n_cores != current_fp.n_cores || self.cpu_model != current_fp.cpu_model {
            return Some(InvalidationReason::CpuChanged);
        }
        match parse_iso8601(&self.measured_at) {
            Some(t) => {
                if let Ok(age) = now.duration_since(t) {
                    if age > CALIBRATION_TTL {
                        return Some(InvalidationReason::Aged { age });
                    }
                }
                None
            }
            None => Some(InvalidationReason::BadTimestamp),
        }
    }
}

#[derive(Debug, Clone)]
pub enum InvalidationReason {
    KernelChanged { old: String, new: String },
    BackendChanged { old: String, new: String },
    CpuChanged,
    Aged { age: Duration },
    BadTimestamp,
}

impl std::fmt::Display for InvalidationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KernelChanged { old, new } => {
                write!(f, "kernel changed ({old} -> {new})")
            }
            Self::BackendChanged { old, new } => {
                write!(f, "inference backend changed ({old} -> {new})")
            }
            Self::CpuChanged => write!(f, "CPU model or core count changed"),
            Self::Aged { age } => write!(f, "calibration older than TTL ({}s)", age.as_secs()),
            Self::BadTimestamp => write!(f, "calibration measured_at could not be parsed"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemFingerprint {
    pub cpu_model: String,
    pub n_cores: u32,
    pub kernel: String,
}

impl SystemFingerprint {
    pub fn detect() -> Self {
        Self {
            cpu_model: read_cpu_model().unwrap_or_else(|| "unknown".into()),
            n_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(1),
            kernel: read_kernel_release().unwrap_or_else(|| "unknown".into()),
        }
    }
}

/// TOML wrapper so the on-disk file has a `[calibration]` section
/// (matches H1 example layout — future keys can go under sibling
/// sections without breaking parse).
#[derive(Debug, Serialize, Deserialize)]
struct CalibrationFile {
    calibration: Calibration,
}

/// Unused until the throughput-measurement path lands; kept public so
/// tests can exercise round-trip and the follow-up commit that wires
/// measurement has a place to write results.
#[allow(dead_code)]
pub fn save(path: &Path, cal: &Calibration) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let text = toml::to_string_pretty(&CalibrationFile {
        calibration: cal.clone(),
    })
    .context("serialising calibration")?;
    fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Load a stored calibration, returning `Ok(None)` when the file is
/// absent, unreadable, unparseable, or invalidated by a fingerprint
/// mismatch. Each miss logs at `info` with the reason so ops can trace
/// why a fresh measurement is being requested.
pub fn load(
    path: &Path,
    current_fp: &SystemFingerprint,
    current_backend: &str,
) -> Result<Option<Calibration>> {
    if !path.exists() {
        info!(path = %path.display(), "no calibration file — using fallback");
        return Ok(None);
    }
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            warn!(path = %path.display(), error = %e, "calibration read failed");
            return Ok(None);
        }
    };
    let parsed: CalibrationFile = match toml::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            warn!(path = %path.display(), error = %e, "calibration parse failed");
            return Ok(None);
        }
    };
    let cal = parsed.calibration;
    if let Some(reason) = cal.invalidation_reason(current_fp, current_backend, SystemTime::now()) {
        info!(path = %path.display(), reason = %reason, "calibration invalidated");
        return Ok(None);
    }
    Ok(Some(cal))
}

/// Load if present + valid, otherwise return the fallback. Convenience
/// wrapper for supervisor startup where a `None` from `load` is not
/// actionable — silence must never depend on measurement having landed.
pub fn load_or_fallback(
    path: &Path,
    current_fp: &SystemFingerprint,
    current_backend: &str,
) -> Calibration {
    match load(path, current_fp, current_backend) {
        Ok(Some(cal)) => cal,
        Ok(None) => Calibration::fallback(current_fp, current_backend),
        Err(e) => {
            warn!(path = %path.display(), error = %e, "calibration load errored — using fallback");
            Calibration::fallback(current_fp, current_backend)
        }
    }
}

/// One measurement burst — the closure-caller runs `n_tokens_requested`
/// tokens through the actual backend and reports what came back.
/// `gen_tokens` may differ from the request if the backend stopped
/// early (EOS, error). CPU time is measured by `measure_throughput`
/// around the closure call — the callee does not need to time itself.
pub struct BurstSample {
    pub gen_tokens: usize,
}

/// Warm-up burst + `n_measurement_bursts` bursts of `burst_gen_tokens`
/// each; returns the median `tokens_per_cpu_second` across the
/// measurement bursts. Matches H1 §"Startup calibration protocol"
/// steps 1 (warm-up) and 2 (throughput measurement).
///
/// The closure is called with a token count and must run the backend
/// synchronously; CPU time is captured via `/proc/self/stat` around
/// each call. This works because `spawn_blocking`-hosted backends
/// (rwkv-cpp in-process, or an Ollama call that pins the process
/// waiting on socket IO) accumulate their CPU time under the caller
/// process — the whole calibration path runs on the same PID.
///
/// A backend that offloads to another PID (e.g. a separately-spawned
/// ollama serve child not in our cgroup) would NOT show up in
/// `/proc/self/stat`; that case needs `cgroup.stat` cpu accounting
/// instead, which is deferred until we actually need it.
#[allow(dead_code)] // wired by the background-calibration commit
pub fn measure_throughput<F>(
    warmup_gen_tokens: usize,
    burst_gen_tokens: usize,
    n_measurement_bursts: usize,
    mut burst_fn: F,
) -> Result<f64>
where
    F: FnMut(usize) -> Result<BurstSample>,
{
    if n_measurement_bursts == 0 {
        anyhow::bail!("measure_throughput requires at least one measurement burst");
    }
    burst_fn(warmup_gen_tokens).context("warm-up burst failed")?;

    let mut samples = Vec::with_capacity(n_measurement_bursts);
    for i in 0..n_measurement_bursts {
        let cpu_start = read_process_cpu_time()?;
        let burst = burst_fn(burst_gen_tokens)
            .with_context(|| format!("measurement burst {i} failed"))?;
        let cpu_end = read_process_cpu_time()?;
        let cpu_delta = cpu_end.saturating_sub(cpu_start);
        if cpu_delta.is_zero() {
            anyhow::bail!("burst {i} consumed no measurable CPU time");
        }
        if burst.gen_tokens == 0 {
            anyhow::bail!("burst {i} produced no tokens");
        }
        samples.push(burst.gen_tokens as f64 / cpu_delta.as_secs_f64());
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = samples.len() / 2;
    let median = if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) / 2.0
    } else {
        samples[mid]
    };
    Ok(median)
}

/// Process-wide CPU time (utime + stime) via `/proc/self/stat`.
/// Result is in seconds; resolution is `1/USER_HZ` (10ms on mainline
/// Linux where `USER_HZ = 100`).
#[allow(dead_code)] // used by measure_throughput, wired later
fn read_process_cpu_time() -> Result<Duration> {
    let text = fs::read_to_string("/proc/self/stat").context("reading /proc/self/stat")?;
    parse_stat_cpu_ticks(&text).map(user_hz_ticks_to_duration)
}

/// Extracts `utime + stime` (fields 14 + 15 per proc(5)) in raw jiffies
/// from the contents of `/proc/self/stat`. The `comm` field (index 2)
/// is wrapped in parens and may contain whitespace/parens; we cut past
/// the *last* `)` and index into the remainder to sidestep that.
#[allow(dead_code)] // used by read_process_cpu_time, wired later
fn parse_stat_cpu_ticks(text: &str) -> Result<u64> {
    let paren_end = text
        .rfind(')')
        .ok_or_else(|| anyhow::anyhow!("malformed /proc/self/stat: no ')' delimiter"))?;
    let tail = &text[paren_end + 1..];
    let fields: Vec<&str> = tail.split_whitespace().collect();
    // proc(5) 1-indexed fields; comm is field 2, so tail index 0 == field 3.
    // utime is field 14 → tail index 11.
    // stime is field 15 → tail index 12.
    let utime: u64 = fields
        .get(11)
        .ok_or_else(|| anyhow::anyhow!("/proc/self/stat missing utime"))?
        .parse()
        .context("parsing utime")?;
    let stime: u64 = fields
        .get(12)
        .ok_or_else(|| anyhow::anyhow!("/proc/self/stat missing stime"))?
        .parse()
        .context("parsing stime")?;
    Ok(utime + stime)
}

/// `USER_HZ` on Linux mainline is 100 (10ms per tick). This is a
/// user-space constant fixed by `_SC_CLK_TCK`; the kernel's `CONFIG_HZ`
/// (100/250/1000) affects scheduling granularity, not this value.
#[allow(dead_code)] // used by user_hz_ticks_to_duration, wired later
const USER_HZ_TICKS_PER_SEC: u64 = 100;

#[allow(dead_code)] // used by read_process_cpu_time, wired later
fn user_hz_ticks_to_duration(ticks: u64) -> Duration {
    Duration::from_millis(ticks * (1000 / USER_HZ_TICKS_PER_SEC))
}

fn read_cpu_model() -> Option<String> {
    let text = fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in text.lines() {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim() == "model name" {
                return Some(v.trim().to_string());
            }
        }
    }
    None
}

fn read_kernel_release() -> Option<String> {
    fs::read_to_string("/proc/sys/kernel/osrelease")
        .ok()
        .map(|s| s.trim().to_string())
}

fn iso8601_now() -> String {
    // Minimal RFC-3339 formatter — avoids pulling `chrono`/`time` into
    // the runtime just for one timestamp. Precision: seconds, UTC.
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format_iso8601(secs)
}

/// Formats a UNIX timestamp (seconds) as `YYYY-MM-DDTHH:MM:SSZ` in
/// UTC. Julian-day arithmetic per Fliegel-Van Flandern, valid for
/// dates 1970-01-01 through year 9999.
fn format_iso8601(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let sod = (secs % 86_400) as u32;
    let hour = sod / 3600;
    let min = (sod / 60) % 60;
    let sec = sod % 60;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Parses `YYYY-MM-DDTHH:MM:SSZ` back to `SystemTime`. Returns `None`
/// on malformed input; the caller treats malformed as an invalidation
/// signal rather than a hard error.
fn parse_iso8601(s: &str) -> Option<SystemTime> {
    let bytes = s.as_bytes();
    if bytes.len() != 20 || bytes[4] != b'-' || bytes[7] != b'-' || bytes[10] != b'T'
        || bytes[13] != b':' || bytes[16] != b':' || bytes[19] != b'Z'
    {
        return None;
    }
    let y: i32 = s[0..4].parse().ok()?;
    let m: u32 = s[5..7].parse().ok()?;
    let d: u32 = s[8..10].parse().ok()?;
    let h: u32 = s[11..13].parse().ok()?;
    let mi: u32 = s[14..16].parse().ok()?;
    let se: u32 = s[17..19].parse().ok()?;
    if m == 0 || m > 12 || d == 0 || d > 31 || h > 23 || mi > 59 || se > 60 {
        return None;
    }
    let days = days_from_civil(y, m, d);
    let secs = days as i64 * 86_400 + (h as i64 * 3600 + mi as i64 * 60 + se as i64);
    if secs < 0 {
        return None;
    }
    Some(UNIX_EPOCH + Duration::from_secs(secs as u64))
}

/// Howard Hinnant's `days_from_civil` — returns days since 1970-01-01.
/// See `https://howardhinnant.github.io/date_algorithms.html`.
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u32;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era as i64 * 146_097 + doe as i64 - 719_468
}

/// Inverse of `days_from_civil`.
fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i32 + era as i32 * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fp() -> SystemFingerprint {
        SystemFingerprint {
            cpu_model: "12th Gen Intel(R) Core(TM) i5-1235U".into(),
            n_cores: 12,
            kernel: "6.11.0-9-generic".into(),
        }
    }

    fn base_cal() -> Calibration {
        Calibration {
            tokens_per_cpu_second: 9.4,
            fan_safe_cpu_percent: 6.0,
            cpu_model: fp().cpu_model,
            n_cores: fp().n_cores,
            kernel: fp().kernel,
            backend: "ollama-0.3.14".into(),
            measured_at: iso8601_now(),
            defaulted: false,
        }
    }

    #[test]
    fn drip_formula_matches_h1_reference_numbers() {
        // H1 §11 reference: fan_safe=6% package on 12-core i5-1235U,
        // tokens_per_cpu_second=9.4 → ~3.4 tok/s at safety_margin=0.6.
        // Our formula: 0.06 × 12 × 9.4 × 0.6 = 4.06.
        // Close to the doc's ~3.4; both are order-of-magnitude
        // estimates rather than exact.
        let cal = base_cal();
        let r = cal.drip_rate_default();
        assert!(r > 3.0 && r < 5.0, "expected 3..5 tok/s, got {r}");
    }

    #[test]
    fn fallback_is_conservative() {
        let cal = Calibration::fallback(&fp(), "ollama-0.3.14");
        assert!(cal.defaulted);
        assert_eq!(cal.fan_safe_cpu_percent, FALLBACK_FAN_SAFE_CPU_PERCENT);
        // With 1% × 12 × 9.4 × 0.6 = 0.677 tok/s.
        let r = cal.drip_rate_default();
        assert!(r < 1.0, "fallback drip must stay well under 1 tok/s, got {r}");
    }

    #[test]
    fn roundtrip_toml() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let cal = base_cal();
        save(tmp.path(), &cal).unwrap();
        let loaded = load(tmp.path(), &fp(), &cal.backend).unwrap().unwrap();
        assert_eq!(loaded, cal);
    }

    #[test]
    fn invalidated_by_kernel_change() {
        let cal = base_cal();
        let mut new_fp = fp();
        new_fp.kernel = "6.20.0-1-generic".into();
        assert!(matches!(
            cal.invalidation_reason(&new_fp, &cal.backend, SystemTime::now()),
            Some(InvalidationReason::KernelChanged { .. })
        ));
    }

    #[test]
    fn invalidated_by_backend_change() {
        let cal = base_cal();
        assert!(matches!(
            cal.invalidation_reason(&fp(), "rwkv-cpp-0.1", SystemTime::now()),
            Some(InvalidationReason::BackendChanged { .. })
        ));
    }

    #[test]
    fn invalidated_by_cpu_change() {
        let cal = base_cal();
        let mut new_fp = fp();
        new_fp.n_cores = 8;
        assert!(matches!(
            cal.invalidation_reason(&new_fp, &cal.backend, SystemTime::now()),
            Some(InvalidationReason::CpuChanged)
        ));
    }

    #[test]
    fn invalidated_by_ttl() {
        let mut cal = base_cal();
        // 31 days ago → past the 30-day TTL.
        let old_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            - 31 * 24 * 60 * 60;
        cal.measured_at = format_iso8601(old_secs);
        assert!(matches!(
            cal.invalidation_reason(&fp(), &cal.backend, SystemTime::now()),
            Some(InvalidationReason::Aged { .. })
        ));
    }

    #[test]
    fn fresh_calibration_not_invalidated() {
        let cal = base_cal();
        assert!(cal
            .invalidation_reason(&fp(), &cal.backend, SystemTime::now())
            .is_none());
    }

    #[test]
    fn missing_file_returns_none() {
        let path = std::path::Path::new("/nonexistent/noesis-cal-does-not-exist.toml");
        assert!(load(path, &fp(), "ollama-0.3.14").unwrap().is_none());
    }

    #[test]
    fn malformed_toml_returns_none() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "not [valid toml here").unwrap();
        assert!(load(tmp.path(), &fp(), "ollama-0.3.14").unwrap().is_none());
    }

    #[test]
    fn load_or_fallback_returns_fallback_when_missing() {
        let path = std::path::Path::new("/nonexistent/noesis-cal-does-not-exist.toml");
        let cal = load_or_fallback(path, &fp(), "ollama-0.3.14");
        assert!(cal.defaulted);
    }

    #[test]
    fn iso8601_roundtrips() {
        // Sanity: today's timestamp parses back to (roughly) itself.
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let s = format_iso8601(now);
        let parsed = parse_iso8601(&s).unwrap();
        let back = parsed.duration_since(UNIX_EPOCH).unwrap().as_secs();
        assert_eq!(back, now, "roundtrip mismatch: {s}");
    }

    #[test]
    fn iso8601_parse_rejects_garbage() {
        assert!(parse_iso8601("").is_none());
        assert!(parse_iso8601("2026-07-25").is_none());
        assert!(parse_iso8601("2026-07-25T12:00:00").is_none()); // missing Z
        assert!(parse_iso8601("2026-13-01T00:00:00Z").is_none()); // bad month
    }

    #[test]
    fn iso8601_known_epoch() {
        assert_eq!(format_iso8601(0), "1970-01-01T00:00:00Z");
        assert_eq!(
            format_iso8601(1_700_000_000),
            "2023-11-14T22:13:20Z"
        );
    }

    #[test]
    fn parse_stat_cpu_ticks_extracts_utime_plus_stime() {
        // Real `/proc/self/stat` shape: pid (comm) state ppid pgrp ...
        // Fields 14/15 are utime/stime (1-indexed per proc(5)).
        // Below: 40 utime + 60 stime = 100 ticks.
        let line = "1234 (noesis-runtime) S 1 1234 1234 0 -1 0 0 0 0 0 \
                    40 60 0 0 20 0 4 0 100 0 0 0 0 0 0 0 0 0 0 0 0 0 \
                    0 0 0 0 0 0 0 0 0 0 0";
        assert_eq!(parse_stat_cpu_ticks(line).unwrap(), 100);
    }

    #[test]
    fn parse_stat_cpu_ticks_survives_parens_in_comm() {
        // `comm` may legally contain `(` and `)`; rfind(')') skips past.
        let line = "1 (weird ) name) S 0 1 1 0 -1 0 0 0 0 0 \
                    5 7 0 0 20 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 \
                    0 0 0 0 0 0 0 0 0 0 0";
        assert_eq!(parse_stat_cpu_ticks(line).unwrap(), 12);
    }

    #[test]
    fn parse_stat_cpu_ticks_errors_on_missing_paren() {
        assert!(parse_stat_cpu_ticks("nothing here").is_err());
    }

    #[test]
    fn parse_stat_cpu_ticks_errors_on_truncated_fields() {
        // Enough to satisfy rfind(')') but not enough fields for utime.
        let line = "1 (short) S 0 1 1";
        assert!(parse_stat_cpu_ticks(line).is_err());
    }

    #[test]
    fn user_hz_conversion_matches_100hz_mainline() {
        // 100 ticks × 10ms/tick = 1s on mainline USER_HZ=100.
        assert_eq!(user_hz_ticks_to_duration(100), Duration::from_secs(1));
        assert_eq!(user_hz_ticks_to_duration(1), Duration::from_millis(10));
        assert_eq!(user_hz_ticks_to_duration(0), Duration::from_millis(0));
    }

    #[test]
    fn measure_throughput_rejects_zero_bursts() {
        let r = measure_throughput(10, 20, 0, |_| {
            Ok(BurstSample { gen_tokens: 20 })
        });
        assert!(r.is_err());
    }

    #[test]
    fn measure_throughput_runs_warmup_then_n_bursts() {
        // Verify: closure called `1 + n_measurement_bursts` times, and
        // the first call receives `warmup_gen_tokens` while subsequent
        // calls receive `burst_gen_tokens`. Each closure invocation
        // burns a bit of CPU so `cpu_delta` clears the USER_HZ resolution
        // (10ms) — otherwise `measure_throughput` correctly bails on the
        // first zero-delta burst.
        use std::cell::RefCell;
        let calls: RefCell<Vec<usize>> = RefCell::new(Vec::new());
        let _ = measure_throughput(7, 42, 3, |n| {
            calls.borrow_mut().push(n);
            burn_cpu_for(Duration::from_millis(30));
            Ok(BurstSample { gen_tokens: n })
        });
        assert_eq!(*calls.borrow(), vec![7, 42, 42, 42]);
    }

    fn burn_cpu_for(d: Duration) {
        let start = std::time::Instant::now();
        // Sum-of-squares busy-loop; the black_box prevents the optimiser
        // from constant-folding the whole thing away.
        let mut acc: u64 = 0;
        while start.elapsed() < d {
            for i in 0..10_000u64 {
                acc = acc.wrapping_add(i.wrapping_mul(i));
            }
            std::hint::black_box(acc);
        }
    }

    #[test]
    fn measure_throughput_propagates_burst_error() {
        let r = measure_throughput(10, 20, 2, |_| -> Result<BurstSample> {
            anyhow::bail!("simulated backend failure")
        });
        assert!(r.is_err());
    }

    #[test]
    fn measure_throughput_errors_on_zero_token_burst() {
        // Warm-up returns tokens (so it doesn't fail early on CPU); the
        // first measurement burst reports zero tokens, which must fail.
        use std::cell::Cell;
        let call = Cell::new(0);
        let r = measure_throughput(10, 20, 1, |_| {
            let n = call.get();
            call.set(n + 1);
            Ok(BurstSample {
                gen_tokens: if n == 0 { 20 } else { 0 },
            })
        });
        assert!(r.is_err());
    }
}
