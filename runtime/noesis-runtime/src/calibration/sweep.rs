//! Thermal sweep — finds the highest sustained CPU load level at
//! which package die temperature stays within a safety envelope.
//!
//! Signal model (from `calibration::thermal` module docs): temperature
//! predicts fan response. If a 30% package load doesn't raise ΔT more
//! than a few degrees over baseline and stays below an absolute ceiling,
//! fans won't ramp during ambient drip at that level either.
//!
//! Algorithm:
//!   1. Baseline: read package temp every 500 ms for `baseline_secs`,
//!      take the median. Median (vs. mean) rejects a stray thermal
//!      spike from a background task.
//!   2. For each step in ascending order:
//!      a. Spawn `load_threads` busy-loop workers at duty cycle
//!         `step_pct/100` per 100 ms window.
//!      b. Wait `settle_secs` for temp to catch up to the new load.
//!      c. Sample temp every 500 ms for the remaining window; take max.
//!      d. Kill workers.
//!      e. Decision: if ΔT > `delta_ceiling_c` OR peak > `absolute_ceiling_c`,
//!         this step is *unsafe*. Return the previous safe step (or 1
//!         if even the smallest step is unsafe — silence budget too tight
//!         to auto-detect, caller falls back to fixed conservative).
//!   3. All steps passed → return the largest step tried.
//!
//! Non-goals for this file:
//!   - Actually running the sweep during startup (see the background-
//!     calibrate wire commit).
//!   - Interpreting the result — the drip formula in `Calibration`
//!     consumes `fan_safe_cpu_percent` regardless of how it was derived.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::Serialize;

use super::thermal::CoretempProbe;

/// Tunables for the sweep. Defaults picked for a laptop-class silence
/// budget; overridable via `noesis-runtime calibrate --interactive`
/// once that CLI lands.
#[derive(Debug, Clone)]
pub struct SweepConfig {
    /// Ascending list of package CPU% levels to try (e.g., `[10, 20, 30]`).
    pub steps: Vec<u32>,
    /// How long to sample baseline temp before starting the sweep.
    pub baseline_secs: u64,
    /// Wait time after ramping a step before we start sampling.
    /// Temp lags load by several seconds — sampling too early would
    /// read pre-load temperature and pass every step spuriously.
    pub settle_secs: u64,
    /// Total per-step duration (must be >= settle_secs).
    pub step_secs: u64,
    /// A step is unsafe if peak sampled temp exceeds
    /// `baseline + delta_ceiling_c`.
    pub delta_ceiling_c: f64,
    /// A step is unsafe if peak sampled temp exceeds this absolute
    /// value, regardless of baseline. Guards against a machine that
    /// starts already-warm (idle temp ≈ 70 °C on some poor thermal
    /// designs) where a small ΔT still crosses fan-ramp threshold.
    pub absolute_ceiling_c: f64,
    /// Number of OS threads to spawn for the load. Defaults to the
    /// core count; using fewer just means each thread runs at higher
    /// duty cycle to hit the same package %.
    pub load_threads: usize,
}

impl Default for SweepConfig {
    fn default() -> Self {
        let n_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            steps: vec![10, 20, 30],
            baseline_secs: 10,
            settle_secs: 5,
            step_secs: 30,
            delta_ceiling_c: 8.0,
            absolute_ceiling_c: 75.0,
            load_threads: n_cores,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct StepResult {
    pub pct: u32,
    pub peak_temp_c: f64,
    pub delta_c: f64,
    pub within_delta: bool,
    pub within_absolute: bool,
}

impl StepResult {
    pub fn is_safe(&self) -> bool {
        self.within_delta && self.within_absolute
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SweepResult {
    pub baseline_temp_c: f64,
    pub steps: Vec<StepResult>,
    /// Highest step that stayed inside both ceilings, or `1` when even
    /// the smallest step failed (caller then holds to the fallback
    /// silence budget rather than raise it based on a hot sample).
    pub safe_percent: u32,
    pub used_package_sensor: bool,
}

/// Full sweep. Blocks the calling thread; the caller should host this
/// on `tokio::task::spawn_blocking` if it needs to keep the async
/// runtime responsive.
///
/// `cfg.step_secs` bounds each step's wall time; the whole sweep
/// runs at most `baseline_secs + steps.len() * step_secs` seconds
/// (real-world: ~2 minutes on default config with 3 steps).
pub fn run(probe: &CoretempProbe, cfg: &SweepConfig) -> Result<SweepResult> {
    if cfg.step_secs < cfg.settle_secs {
        anyhow::bail!("step_secs ({}) must be >= settle_secs ({})",
                      cfg.step_secs, cfg.settle_secs);
    }
    if cfg.steps.is_empty() {
        anyhow::bail!("sweep requires at least one step");
    }

    let baseline_temp_c = sample_median(probe, Duration::from_secs(cfg.baseline_secs))
        .context("sampling baseline")?;

    let mut step_results = Vec::with_capacity(cfg.steps.len());
    for &pct in &cfg.steps {
        let step = run_step(probe, pct, cfg)
            .with_context(|| format!("running step {pct}%"))?;
        let delta = step.peak_temp_c - baseline_temp_c;
        let r = StepResult {
            pct,
            peak_temp_c: step.peak_temp_c,
            delta_c: delta,
            within_delta: delta <= cfg.delta_ceiling_c,
            within_absolute: step.peak_temp_c <= cfg.absolute_ceiling_c,
        };
        let unsafe_step = !r.is_safe();
        step_results.push(r);
        if unsafe_step {
            break;
        }
    }

    let safe_percent = decide_safe_percent(&step_results);
    Ok(SweepResult {
        baseline_temp_c,
        steps: step_results,
        safe_percent,
        used_package_sensor: probe.has_package_sensor(),
    })
}

/// Pure decision logic — pulled out so tests can exercise it without
/// running actual load. Returns `1` (the conservative fallback level)
/// when no step succeeded; otherwise the pct of the last safe step in
/// order (results are assumed to be in ascending pct order, which is
/// how `run` produces them).
pub fn decide_safe_percent(results: &[StepResult]) -> u32 {
    let mut best: u32 = 1;
    for r in results {
        if r.is_safe() {
            best = r.pct;
        } else {
            break;
        }
    }
    best
}

struct StepSample {
    peak_temp_c: f64,
}

fn run_step(probe: &CoretempProbe, pct: u32, cfg: &SweepConfig) -> Result<StepSample> {
    let stop = Arc::new(AtomicBool::new(false));
    let mut workers = Vec::with_capacity(cfg.load_threads);
    for _ in 0..cfg.load_threads {
        let stop = Arc::clone(&stop);
        workers.push(thread::spawn(move || {
            duty_cycle_loop(pct, &stop);
        }));
    }

    // Settle window: don't sample. Temp lags load; sampling here reads
    // pre-load values and spuriously passes hot steps.
    thread::sleep(Duration::from_secs(cfg.settle_secs));

    let sample_window = Duration::from_secs(cfg.step_secs - cfg.settle_secs);
    let peak = sample_peak(probe, sample_window)?;

    stop.store(true, Ordering::Relaxed);
    for w in workers {
        let _ = w.join();
    }
    Ok(StepSample { peak_temp_c: peak })
}

/// Duty-cycle busy loop: for each 100 ms window, work for `pct` ms and
/// sleep for `(100 - pct)` ms. Simpler than trying to sample process
/// CPU% and PID-control it — the goal is a rough steady-state package
/// load, not precision.
fn duty_cycle_loop(pct: u32, stop: &AtomicBool) {
    let window = Duration::from_millis(100);
    let work_ms = pct.min(100) as u64;
    let work_dur = Duration::from_millis(work_ms);
    while !stop.load(Ordering::Relaxed) {
        let window_start = Instant::now();
        let mut acc: u64 = 0;
        while window_start.elapsed() < work_dur && !stop.load(Ordering::Relaxed) {
            for i in 0..10_000u64 {
                acc = acc.wrapping_add(i.wrapping_mul(i));
            }
            std::hint::black_box(acc);
        }
        if stop.load(Ordering::Relaxed) {
            return;
        }
        let elapsed = window_start.elapsed();
        if elapsed < window {
            thread::sleep(window - elapsed);
        }
    }
}

fn sample_median(probe: &CoretempProbe, dur: Duration) -> Result<f64> {
    let mut samples = collect_samples(probe, dur)?;
    if samples.is_empty() {
        anyhow::bail!("no temperature samples collected");
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = samples.len() / 2;
    Ok(if samples.len().is_multiple_of(2) {
        (samples[mid - 1] + samples[mid]) / 2.0
    } else {
        samples[mid]
    })
}

fn sample_peak(probe: &CoretempProbe, dur: Duration) -> Result<f64> {
    let samples = collect_samples(probe, dur)?;
    samples
        .into_iter()
        .fold(None::<f64>, |acc, x| Some(acc.map_or(x, |a| a.max(x))))
        .ok_or_else(|| anyhow::anyhow!("no temperature samples collected"))
}

fn collect_samples(probe: &CoretempProbe, dur: Duration) -> Result<Vec<f64>> {
    let interval = Duration::from_millis(500);
    let deadline = Instant::now() + dur;
    let mut samples = Vec::new();
    while Instant::now() < deadline {
        if let Ok(t) = probe.max_temp_c() {
            samples.push(t);
        }
        thread::sleep(interval);
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(pct: u32, within_delta: bool, within_absolute: bool) -> StepResult {
        StepResult {
            pct,
            peak_temp_c: 0.0,
            delta_c: 0.0,
            within_delta,
            within_absolute,
        }
    }

    #[test]
    fn decide_returns_highest_safe_step() {
        let results = vec![
            mk(10, true, true),
            mk(20, true, true),
            mk(30, true, true),
        ];
        assert_eq!(decide_safe_percent(&results), 30);
    }

    #[test]
    fn decide_stops_at_first_unsafe_step() {
        let results = vec![
            mk(10, true, true),
            mk(20, true, true),
            mk(30, false, true), // delta exceeded
        ];
        assert_eq!(decide_safe_percent(&results), 20);
    }

    #[test]
    fn decide_returns_fallback_when_first_step_unsafe() {
        // Even 10% pushed the box past ceiling → can't trust auto-sweep
        // to raise fan_safe%, so hold on the caller's fallback (1%).
        let results = vec![mk(10, false, true)];
        assert_eq!(decide_safe_percent(&results), 1);
    }

    #[test]
    fn decide_treats_absolute_ceiling_as_hard_stop() {
        // Machine started at 70°C (idle), ΔT was small but peak > 75°C
        // absolute — fans probably already ramped, refuse to raise safe%.
        let results = vec![mk(10, true, false)];
        assert_eq!(decide_safe_percent(&results), 1);
    }

    #[test]
    fn decide_returns_fallback_on_empty_result() {
        assert_eq!(decide_safe_percent(&[]), 1);
    }

    #[test]
    fn is_safe_requires_both_within_delta_and_within_absolute() {
        assert!(mk(10, true, true).is_safe());
        assert!(!mk(10, false, true).is_safe());
        assert!(!mk(10, true, false).is_safe());
        assert!(!mk(10, false, false).is_safe());
    }

    #[test]
    fn run_rejects_step_secs_less_than_settle_secs() {
        let probe = match CoretempProbe::probe_default() {
            Ok(Some(p)) => p,
            _ => return, // no live coretemp, skip
        };
        let mut cfg = SweepConfig::default();
        cfg.step_secs = 3;
        cfg.settle_secs = 5;
        assert!(run(&probe, &cfg).is_err());
    }

    #[test]
    fn run_rejects_empty_steps() {
        let probe = match CoretempProbe::probe_default() {
            Ok(Some(p)) => p,
            _ => return,
        };
        let cfg = SweepConfig {
            steps: vec![],
            ..SweepConfig::default()
        };
        assert!(run(&probe, &cfg).is_err());
    }

    /// Live end-to-end sweep — actually runs load on the machine.
    /// Ignored by default because it takes ~30s on a fast config and
    /// perturbs any other work happening on the box. Run explicitly
    /// with `cargo test -- --ignored live_sweep`.
    #[test]
    #[ignore = "runs real CPU load — invoke with --ignored"]
    fn live_sweep_produces_plausible_result() {
        let probe = CoretempProbe::probe_default()
            .expect("probe_default")
            .expect("coretemp not available");
        let cfg = SweepConfig {
            baseline_secs: 3,
            settle_secs: 2,
            step_secs: 5,
            ..SweepConfig::default()
        };
        let result = run(&probe, &cfg).expect("sweep run");
        eprintln!("live sweep: {result:?}");
        assert!(result.baseline_temp_c > 0.0 && result.baseline_temp_c < 120.0);
        assert!([1, 10, 20, 30].contains(&result.safe_percent));
    }
}
