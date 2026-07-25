//! Thermal probe — coretemp read helpers for the thermal sweep leg
//! of startup calibration (H1). Primary signal for "silent budget"
//! detection because it's readable without special privileges on any
//! Intel/AMD Linux box, unlike RAPL (`energy_uj` is root-only since
//! CVE-2020-8694 aka Platypus) and unlike hwmon fan_input (often
//! returns EIO on laptop ACPI stacks).
//!
//! Signal model: package die temperature is a monotonic proxy for
//! the fan curve. Fans respond to temperature, so temperature change
//! *predicts* the fan response before it happens. We read a baseline
//! at idle, apply a controlled CPU load, and compare temperature
//! delta + absolute peak to decide whether that load level is inside
//! the silence budget.
//!
//! Sensor precedence:
//!   1. `coretemp` hwmon → `temp*_input` labelled `Package id N` —
//!      the aggregated die temperature Intel firmware itself uses.
//!   2. If no Package label, fall back to `max(Core N)` across all
//!      per-core inputs (matches turbostat's "PkgTmp" fallback).
//!   3. If neither, return `None` — thermal sweep unavailable, will
//!      trigger the same interactive-fallback path as EIO fans.
//!
//! Deferred to follow-ups:
//!   - RAPL package power sampler (bonus signal, needs udev/caps —
//!     see Task 17). Would let us measure joules/token and cap on
//!     watts instead of just temp.
//!   - Apple Silicon path (macOS `powermetrics`, no coretemp file).
//!   - Fanless thermal-throttle detection (needs `IA32_THERM_STATUS`
//!     MSR, less accessible than coretemp).

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

const HWMON_ROOT: &str = "/sys/class/hwmon";

/// A resolved set of coretemp inputs discovered by `probe_default`.
/// Read `max_temp_c()` to get the current worst-case reading; that's
/// the signal the sweep watches during load bursts.
#[derive(Debug, Clone)]
pub struct CoretempProbe {
    /// Preferred single-source input (the "Package id N" label) when
    /// available. Cheapest read path.
    package_input: Option<PathBuf>,
    /// Per-core inputs; used as fallback when there's no Package
    /// label, and as sanity check when Package is stale/frozen.
    core_inputs: Vec<PathBuf>,
}

impl CoretempProbe {
    /// Enumerate hwmon, pick the first coretemp instance, resolve
    /// Package + Core inputs from its labels.
    ///
    /// Returns `Ok(None)` when no coretemp hwmon exists (AMD Zen
    /// uses `k10temp` instead — a follow-up can add it); the caller
    /// treats absent coretemp identically to "sweep unavailable →
    /// interactive fallback required".
    pub fn probe_default() -> Result<Option<Self>> {
        let hwmon_dir = match find_hwmon_by_name("coretemp")? {
            Some(d) => d,
            None => return Ok(None),
        };
        let (package_input, core_inputs) = classify_temp_inputs(&hwmon_dir)?;
        if package_input.is_none() && core_inputs.is_empty() {
            return Ok(None);
        }
        Ok(Some(Self {
            package_input,
            core_inputs,
        }))
    }

    /// Current worst-case die temperature (°C). Reads Package when
    /// available; otherwise `max(Core N)`. Silently drops any input
    /// that transiently errors (some kernels return -ENODATA during
    /// C-state transitions) — the caller retries at the next sample.
    pub fn max_temp_c(&self) -> Result<f64> {
        if let Some(path) = &self.package_input {
            if let Ok(t) = read_temp_c(path) {
                return Ok(t);
            }
        }
        let mut best: Option<f64> = None;
        for p in &self.core_inputs {
            if let Ok(t) = read_temp_c(p) {
                best = Some(match best {
                    Some(b) if b >= t => b,
                    _ => t,
                });
            }
        }
        best.ok_or_else(|| anyhow::anyhow!("all coretemp inputs unreadable"))
    }

    /// Reported in `calibration_result` events so operators can tell
    /// whether the sweep used Intel's aggregated package sensor or
    /// the max-of-cores fallback.
    pub fn has_package_sensor(&self) -> bool {
        self.package_input.is_some()
    }

    pub fn n_core_sensors(&self) -> usize {
        self.core_inputs.len()
    }
}

fn find_hwmon_by_name(target: &str) -> Result<Option<PathBuf>> {
    let root = Path::new(HWMON_ROOT);
    if !root.exists() {
        return Ok(None);
    }
    for entry in fs::read_dir(root).with_context(|| format!("reading {HWMON_ROOT}"))? {
        let entry = entry?;
        let name_file = entry.path().join("name");
        let Ok(name) = fs::read_to_string(&name_file) else {
            continue;
        };
        if name.trim() == target {
            return Ok(Some(entry.path()));
        }
    }
    Ok(None)
}

fn classify_temp_inputs(hwmon_dir: &Path) -> Result<(Option<PathBuf>, Vec<PathBuf>)> {
    let mut package: Option<PathBuf> = None;
    let mut cores: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(hwmon_dir)
        .with_context(|| format!("reading {}", hwmon_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let Some(fname) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        // temp{N}_label → paired with temp{N}_input.
        let Some(n) = fname
            .strip_prefix("temp")
            .and_then(|s| s.strip_suffix("_label"))
        else {
            continue;
        };
        let Ok(label) = fs::read_to_string(&path) else {
            continue;
        };
        let label = label.trim();
        let input_path = hwmon_dir.join(format!("temp{n}_input"));
        if !input_path.exists() {
            continue;
        }
        if label.starts_with("Package id ") {
            package = Some(input_path);
        } else if label.starts_with("Core ") {
            cores.push(input_path);
        }
    }
    Ok((package, cores))
}

fn read_temp_c(path: &Path) -> Result<f64> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let mdeg: i64 = raw
        .trim()
        .parse()
        .with_context(|| format!("parsing millidegrees from {}", path.display()))?;
    // hwmon convention: value in millidegrees Celsius (48000 = 48.0 °C).
    Ok(mdeg as f64 / 1000.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_temp_c_parses_millidegrees() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "48000\n").unwrap();
        let t = read_temp_c(tmp.path()).unwrap();
        assert!((t - 48.0).abs() < 1e-9, "expected 48.0, got {t}");
    }

    #[test]
    fn read_temp_c_handles_no_trailing_newline() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "72500").unwrap();
        let t = read_temp_c(tmp.path()).unwrap();
        assert!((t - 72.5).abs() < 1e-9, "expected 72.5, got {t}");
    }

    #[test]
    fn read_temp_c_errors_on_garbage() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), "not a number").unwrap();
        assert!(read_temp_c(tmp.path()).is_err());
    }

    #[test]
    fn classify_finds_package_and_cores_in_synthetic_hwmon() {
        let dir = tempfile::tempdir().unwrap();
        // temp1 = Package
        std::fs::write(dir.path().join("temp1_label"), "Package id 0").unwrap();
        std::fs::write(dir.path().join("temp1_input"), "48000").unwrap();
        // temp2, temp10 = Cores
        std::fs::write(dir.path().join("temp2_label"), "Core 0").unwrap();
        std::fs::write(dir.path().join("temp2_input"), "47000").unwrap();
        std::fs::write(dir.path().join("temp10_label"), "Core 8").unwrap();
        std::fs::write(dir.path().join("temp10_input"), "46000").unwrap();
        // Unrelated label — must be ignored.
        std::fs::write(dir.path().join("temp3_label"), "PCH").unwrap();
        std::fs::write(dir.path().join("temp3_input"), "55000").unwrap();

        let (pkg, cores) = classify_temp_inputs(dir.path()).unwrap();
        assert!(pkg.is_some());
        assert_eq!(cores.len(), 2);
    }

    #[test]
    fn classify_skips_label_without_matching_input() {
        // Missing temp1_input for the labelled temp1 → not returned.
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("temp1_label"), "Package id 0").unwrap();
        // no temp1_input file
        let (pkg, cores) = classify_temp_inputs(dir.path()).unwrap();
        assert!(pkg.is_none());
        assert!(cores.is_empty());
    }

    #[test]
    fn max_temp_c_prefers_package_over_cores() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("temp1_label"), "Package id 0").unwrap();
        std::fs::write(dir.path().join("temp1_input"), "80000").unwrap();
        std::fs::write(dir.path().join("temp2_label"), "Core 0").unwrap();
        std::fs::write(dir.path().join("temp2_input"), "60000").unwrap();
        let (pkg, cores) = classify_temp_inputs(dir.path()).unwrap();
        let probe = CoretempProbe {
            package_input: pkg,
            core_inputs: cores,
        };
        assert!((probe.max_temp_c().unwrap() - 80.0).abs() < 1e-9);
    }

    #[test]
    fn max_temp_c_falls_back_to_max_core_when_no_package() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("temp2_label"), "Core 0").unwrap();
        std::fs::write(dir.path().join("temp2_input"), "62000").unwrap();
        std::fs::write(dir.path().join("temp3_label"), "Core 4").unwrap();
        std::fs::write(dir.path().join("temp3_input"), "71000").unwrap();
        std::fs::write(dir.path().join("temp4_label"), "Core 8").unwrap();
        std::fs::write(dir.path().join("temp4_input"), "65000").unwrap();
        let (pkg, cores) = classify_temp_inputs(dir.path()).unwrap();
        assert!(pkg.is_none());
        let probe = CoretempProbe {
            package_input: pkg,
            core_inputs: cores,
        };
        assert!((probe.max_temp_c().unwrap() - 71.0).abs() < 1e-9);
    }

    /// End-to-end read against the real hwmon — only runs when the host
    /// actually has a coretemp instance. Skipped silently on machines
    /// without it (AMD Zen with only k10temp, ARM boards, etc.) so CI
    /// on non-Intel hosts stays green.
    #[test]
    fn live_coretemp_read_is_plausible_or_absent() {
        match CoretempProbe::probe_default() {
            Ok(Some(probe)) => {
                let t = probe.max_temp_c().expect("live read");
                // Very loose bounds: below freezing = probe is lying;
                // above 120°C = machine is on fire. Real range on any
                // running laptop/desktop is 30..100°C.
                assert!(
                    (0.0..=120.0).contains(&t),
                    "implausible live temp {t}°C"
                );
            }
            Ok(None) => { /* no coretemp on this host — fine */ }
            Err(e) => panic!("probe_default errored on live hwmon: {e}"),
        }
    }
}
