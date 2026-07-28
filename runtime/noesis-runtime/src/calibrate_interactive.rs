//! `noesis-runtime calibrate --interactive` subcommand.
//!
//! Used when the background auto-calibration is unavailable (no coretemp
//! sensor, or the host is idle-hot so every sweep returns <1%). The user
//! runs the model under load, watches the fans, and enters the threshold
//! CPU% they observed. The result is written to calibration.toml and takes
//! effect on the next daemon restart.
//!
//! Flow:
//!
//! 1. Load config + model (required for throughput measurement).
//! 2. Run a burst to measure `tokens_per_cpu_second`.
//! 3. Print the measured rate and prompt the user.
//! 4. User enters `fan_safe_cpu_percent` (or runs another burst while
//!    watching fans).
//! 5. Save calibration.toml.

use std::io::{self, Write as _};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tracing::info;

use crate::calibration::{
    self, BurstFn, BurstSample, Calibration, SystemFingerprint,
    DEFAULT_SAFETY_MARGIN, THROUGHPUT_BURST_TOKENS, THROUGHPUT_N_BURSTS,
    THROUGHPUT_WARMUP_TOKENS,
};
use crate::inference;

/// Sweep thread counts [1, 2, 4, 6, 8, 12] and print a table showing
/// wall tok/s and estimated CPU% for each count. Requires a loaded model.
/// Reads fan_safe_cpu_percent from existing calibration (or fallback) to
/// annotate which counts are fan-safe.
pub async fn run_thread_sweep(cfg: &crate::Config) -> Result<()> {
    let fingerprint = SystemFingerprint::detect();
    let backend_id = crate::backend_identifier(cfg);

    println!("=== noesis-runtime thread sweep ===");
    println!("CPU model : {}", fingerprint.cpu_model);
    println!("Cores     : {}", fingerprint.n_cores);
    println!("Backend   : {backend_id}");
    println!();

    let loaded = crate::load_rwkv_if_configured(cfg).await
        .ok_or_else(|| anyhow::anyhow!(
            "model required for thread sweep — check NOESIS_CONFIG model_path"
        ))?;

    let cal = calibration::load_or_fallback(&cfg.calibration_path, &fingerprint, &backend_id);
    let fan_safe = cal.fan_safe_cpu_percent;
    if cal.defaulted {
        println!("Note: no calibration file found — using fallback fan_safe_cpu_percent={fan_safe:.1}%.");
        println!("Run `calibrate --interactive` first for a meaningful fan-safe column.");
        println!();
    } else {
        println!("Fan-safe threshold: {fan_safe:.1}% (from calibration)");
        println!();
    }

    let candidates: Vec<u32> = [1u32, 2, 4, 6, 8, 12]
        .iter()
        .copied()
        .filter(|&t| t <= fingerprint.n_cores)
        .collect();

    println!("{:>7}  {:>12}  {:>12}  {:>8}  {}", "threads", "wall tok/s", "tok/CPU-s", "~CPU%", "fan-safe?");
    println!("{}", "-".repeat(60));

    let mut results: Vec<(u32, f64, f64, f64)> = Vec::new(); // (threads, wall, cpu_s, cpu_pct)

    for t in &candidates {
        let t = *t;
        let ctx = match loaded.runtime.ctx.clone_for_parallel(t) {
            Ok(c) => c,
            Err(e) => {
                println!("{t:>7}  (clone failed: {e})");
                continue;
            }
        };
        let tok = Arc::clone(&loaded.runtime.tok);
        let prompt = loaded.heartbeat_prompt.clone();

        print!("{t:>7}  measuring… ");
        io::stdout().flush().ok();

        let tok_cpu_s = tokio::task::spawn_blocking(move || {
            let mut f: BurstFn = Box::new(move |n: usize| -> Result<BurstSample> {
                let r = crate::inference::generate_once(
                    &ctx,
                    &tok,
                    &prompt,
                    n,
                    &noesis_rwkv::SamplingParams::greedy(),
                    &[],
                );
                if r.gen_tokens == 0 {
                    anyhow::bail!("burst produced no tokens");
                }
                Ok(BurstSample { gen_tokens: r.gen_tokens })
            });
            calibration::measure_throughput(
                THROUGHPUT_WARMUP_TOKENS,
                THROUGHPUT_BURST_TOKENS,
                THROUGHPUT_N_BURSTS,
                &mut *f,
            )
        })
        .await
        .context("thread sweep task join")?
        .with_context(|| format!("throughput measurement at {t} threads"))?;

        let wall = tok_cpu_s * t as f64;
        let cpu_pct = t as f64 / fingerprint.n_cores as f64 * 100.0;
        let ok = if cpu_pct <= fan_safe { "✓" } else { "✗" };

        // Overwrite the "measuring…" line.
        print!("\r");
        println!("{t:>7}  {wall:>12.1}  {tok_cpu_s:>12.2}  {cpu_pct:>7.1}%  {ok}");

        results.push((t, wall, tok_cpu_s, cpu_pct));
    }

    if results.is_empty() {
        println!("No results — all clones failed.");
        return Ok(());
    }

    println!();

    // Ambient: highest thread count where ~CPU% ≤ fan_safe
    let ambient = results.iter().filter(|r| r.3 <= fan_safe).last();
    // Interactive: peak wall tok/s
    let interactive = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    println!("Recommendations:");
    match ambient {
        Some((t, wall, _, cpu)) => {
            println!("  threads      = {t}   # ambient drip: {wall:.1} tok/s @ ~{cpu:.0}% CPU (fan-safe)");
        }
        None => {
            println!("  threads      = 1   # all counts exceed fan_safe threshold — use 1 as conservative default");
        }
    }
    if let Some((t, wall, _, cpu)) = interactive {
        println!("  http_threads = {t}   # interactive: {wall:.1} tok/s @ ~{cpu:.0}% CPU (max throughput)");
    }
    println!();
    println!("Set these in your NOESIS_CONFIG [rwkv_cpp] section.");

    Ok(())
}

pub async fn run(cfg: &crate::Config) -> Result<()> {
    let fingerprint = SystemFingerprint::detect();
    let backend_id = crate::backend_identifier(cfg);

    println!("=== noesis-runtime interactive calibration ===");
    println!("CPU model : {}", fingerprint.cpu_model);
    println!("Cores     : {}", fingerprint.n_cores);
    println!("Kernel    : {}", fingerprint.kernel);
    println!("Backend   : {backend_id}");
    println!();

    // ── Step 1: measure throughput ─────────────────────────────────────────
    let (tok_per_cpu_s, n_threads) = measure_throughput_interactive(cfg).await?;
    // Wall-clock tok/s ≈ tok/CPU-s × n_threads (threads run in parallel on the
    // matrix multiplications; the drip formula uses tok/CPU-s which is the right
    // axis for the thermal model, but users expect to see wall-clock speed).
    let wall_tok_s = tok_per_cpu_s * n_threads as f64;
    println!();
    println!("Measured throughput: {wall_tok_s:.1} tok/s  ({tok_per_cpu_s:.2} tok/CPU-s × {n_threads} threads)");
    println!();

    // ── Step 2: ask for fan threshold ──────────────────────────────────────
    println!("Now enter the CPU% at which your fans become audible under sustained load.");
    println!("If you are unsure, run:");
    println!("  watch -n1 'cat /sys/class/hwmon/hwmon*/temp*_input'");
    println!("in another terminal while the model bursts, then lower the threshold");
    println!("until the fans stay quiet.");
    println!();
    let fan_safe_pct = prompt_fan_safe_percent()?;

    // ── Step 3: save ───────────────────────────────────────────────────────
    let cal = Calibration {
        tokens_per_cpu_second: tok_per_cpu_s,
        fan_safe_cpu_percent: fan_safe_pct,
        cpu_model: fingerprint.cpu_model.clone(),
        n_cores: fingerprint.n_cores,
        kernel: fingerprint.kernel.clone(),
        backend: backend_id,
        measured_at: iso8601_now(),
        defaulted: false,
    };

    let drip = cal.drip_rate_tokens_per_sec(DEFAULT_SAFETY_MARGIN);
    println!();
    println!("Derived drip rate: {drip:.2} tok/s");
    println!(
        "Formula: {:.0}% / 100 × {} cores × {:.2} tok/CPU-s × {:.1} margin",
        fan_safe_pct,
        fingerprint.n_cores,
        tok_per_cpu_s,
        DEFAULT_SAFETY_MARGIN,
    );

    calibration::save(&cfg.calibration_path, &cal)
        .with_context(|| format!("saving calibration to {}", cfg.calibration_path.display()))?;

    info!(
        path = %cfg.calibration_path.display(),
        fan_safe_cpu_percent = cal.fan_safe_cpu_percent,
        tokens_per_cpu_second = cal.tokens_per_cpu_second,
        drip_tokens_per_sec = %format!("{drip:.2}"),
        "interactive calibration saved",
    );
    println!();
    println!("Saved to: {}", cfg.calibration_path.display());
    println!("Restart noesis-runtime to apply the new drip ceiling.");

    Ok(())
}

/// Load the model and run `measure_throughput`. If the model is not
/// configured, falls back to a CPU-burn stub so the user still gets a
/// throughput number (it will be meaningless for inference but lets the
/// formula produce a drip rate).
/// Returns `(tok_per_cpu_s, n_threads)` so the caller can compute wall-clock tok/s.
async fn measure_throughput_interactive(cfg: &crate::Config) -> Result<(f64, u32)> {
    let loaded = crate::load_rwkv_if_configured(cfg).await;

    let n_threads = loaded.as_ref().map(|l| l.runtime.n_threads).unwrap_or(1);

    if loaded.is_none() {
        println!("No model configured — using CPU-burn stub for throughput.");
        println!("(The drip rate will be approximate; load a model for a real number.)");
        println!();
    }

    print!("Running throughput burst ({} warm-up + {} × {} measurement tokens)… ",
        THROUGHPUT_WARMUP_TOKENS,
        THROUGHPUT_N_BURSTS,
        THROUGHPUT_BURST_TOKENS,
    );
    io::stdout().flush().ok();

    // Burst function: real model if available, CPU-burn stub otherwise.
    let mut burst_fn: BurstFn = match loaded {
        Some(l) => {
            let ctx = match l.runtime.ctx.clone_for_parallel(l.runtime.n_threads) {
                Ok(c) => c,
                Err(e) => {
                    println!("(clone failed: {e}; using CPU-burn stub)");
                    return cpu_burn_throughput().map(|t| (t, 1));
                }
            };
            let tok = Arc::clone(&l.runtime.tok);
            let prompt = l.heartbeat_prompt.clone();
            Box::new(move |n: usize| -> Result<BurstSample> {
                let r = inference::generate_once(
                    &ctx,
                    &tok,
                    &prompt,
                    n,
                    &noesis_rwkv::SamplingParams::greedy(),
                    &[],
                );
                if r.gen_tokens == 0 {
                    anyhow::bail!("burst produced no tokens");
                }
                Ok(BurstSample { gen_tokens: r.gen_tokens })
            })
        }
        None => Box::new(|n: usize| -> Result<BurstSample> {
            cpu_burn_for(Duration::from_millis(50 * n as u64));
            Ok(BurstSample { gen_tokens: n })
        }),
    };

    let tok_per_cpu_s = tokio::task::spawn_blocking(move || {
        calibration::measure_throughput(
            THROUGHPUT_WARMUP_TOKENS,
            THROUGHPUT_BURST_TOKENS,
            THROUGHPUT_N_BURSTS,
            &mut *burst_fn,
        )
    })
    .await
    .context("throughput task join")?
    .context("throughput measurement")?;

    println!("done.");
    Ok((tok_per_cpu_s, n_threads))
}

fn cpu_burn_throughput() -> Result<f64> { // returns tok/CPU-s (n_threads=1 for stub)
    calibration::measure_throughput(
        THROUGHPUT_WARMUP_TOKENS,
        THROUGHPUT_BURST_TOKENS,
        THROUGHPUT_N_BURSTS,
        |n: usize| {
            cpu_burn_for(Duration::from_millis(50 * n as u64));
            Ok(BurstSample { gen_tokens: n })
        },
    )
}

fn cpu_burn_for(d: Duration) {
    let start = std::time::Instant::now();
    let mut acc: u64 = 0;
    while start.elapsed() < d {
        for i in 0..10_000u64 {
            acc = acc.wrapping_add(i.wrapping_mul(i));
        }
        std::hint::black_box(acc);
    }
}

fn prompt_fan_safe_percent() -> Result<f64> {
    loop {
        print!("Fan-safe CPU% (package CPU, 0–100): ");
        io::stdout().flush().ok();
        let mut line = String::new();
        io::stdin().read_line(&mut line)?;
        let trimmed = line.trim();
        match trimmed.parse::<f64>() {
            Ok(v) if v >= 0.0 && v <= 100.0 => return Ok(v),
            Ok(_) => println!("Must be between 0 and 100."),
            Err(_) => println!("Invalid number, try again."),
        }
    }
}

fn iso8601_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Borrow the private helper via the calibration module's public API —
    // we re-export through Calibration::fallback's measured_at; for the
    // interactive path we simply format it ourselves using the same algorithm.
    let days = (secs / 86_400) as i64;
    let sod = (secs % 86_400) as u32;
    let h = sod / 3600;
    let m = (sod / 60) % 60;
    let s = sod % 60;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i32 + era as i32 * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}
