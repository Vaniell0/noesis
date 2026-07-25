//! noesis-runtime — supervisor process.
//!
//! Responsibilities (skeleton, will grow):
//!   1. Load runtime config from `$NOESIS_CONFIG` (TOML).
//!   2. Open all four zone stores under `state_path`.
//!   3. Spawn the inference child (rwkv-cpp).
//!   4. Sit on a supervised loop, handling SIGTERM cleanly.
//!
//! Everything past step 2 is TODO for the Phase B skeleton; the point right
//! now is that the systemd unit stays alive, opens the stores without error,
//! and can be verified with `journalctl --user -u noesis-runtime`.

mod calibration;
mod collectors;
mod inference;
mod retention;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use noesis_schema::EventInput;
use noesis_store::Store;
use serde::Deserialize;
use serde_json::json;
use tokio::signal::unix::{signal, SignalKind};
use tracing::{info, warn};

#[derive(Debug, Deserialize)]
struct Config {
    state_path: PathBuf,
    #[serde(default)]
    model_path: Option<PathBuf>,
    #[serde(default = "default_backend")]
    inference_backend: String,
    #[serde(default = "default_calibration_path")]
    calibration_path: PathBuf,
    #[serde(default)]
    rwkv_cpp: Option<RwkvCppSection>,
}

fn default_calibration_path() -> PathBuf {
    PathBuf::from("/var/lib/noesis/calibration.toml")
}

#[derive(Debug, Deserialize)]
struct RwkvCppSection {
    /// Path to the rwkv.cpp .bin model. Falls back to top-level
    /// `model_path` when omitted.
    #[serde(default)]
    model_path: Option<PathBuf>,
    #[serde(default)]
    threads: Option<u32>,
    #[serde(default = "default_heartbeat_prompt")]
    heartbeat_prompt: String,
    #[serde(default = "default_heartbeat_secs")]
    heartbeat_secs: u64,
    #[serde(default = "default_rwkv_max_gen")]
    max_gen_tokens: usize,
    /// `host:port` for the Ollama-shape HTTP shim. When absent the shim
    /// is disabled and the backend only runs the heartbeat loop.
    #[serde(default)]
    http_bind: Option<String>,
}

fn default_rwkv_max_gen() -> usize {
    20
}

fn default_heartbeat_prompt() -> String {
    "You are noesis, a persistent cognitive runtime. Report your status in one sentence.".into()
}

fn default_heartbeat_secs() -> u64 {
    300
}

fn default_backend() -> String {
    "rwkv-cpp".into()
}

/// Loaded rwkv-cpp state + parsed section values, produced once at
/// startup. Kept together so we can hand the same `RwkvRuntime` to the
/// inference supervisor (heartbeat + HTTP shim) and the calibration
/// burst hook (throughput measurement) without paying the multi-second
/// mmap twice.
struct LoadedRwkv {
    runtime: inference::RwkvRuntime,
    model_path: PathBuf,
    heartbeat_prompt: String,
    heartbeat: Duration,
    max_gen_tokens: usize,
    http_bind: Option<SocketAddr>,
}

async fn load_rwkv_if_configured(cfg: &Config) -> Option<LoadedRwkv> {
    if cfg.inference_backend != "rwkv-cpp" {
        warn!(backend = %cfg.inference_backend, "unknown inference backend name");
        return None;
    }
    let s = cfg.rwkv_cpp.as_ref()?;
    let model_path = s.model_path.clone().or_else(|| cfg.model_path.clone());
    let model_path = match model_path {
        Some(p) => p,
        None => {
            warn!("rwkv-cpp backend has no model_path — supervisor idle");
            return None;
        }
    };
    let n_threads = s.threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get().min(4) as u32)
            .unwrap_or(2)
    });
    let http_bind = s.http_bind.as_ref().and_then(|s| match s.parse() {
        Ok(addr) => Some(addr),
        Err(e) => {
            warn!(http_bind = %s, error = %e,
                  "invalid rwkv_cpp.http_bind — HTTP shim disabled");
            None
        }
    });

    let model_path_open = model_path.clone();
    let runtime = match tokio::task::spawn_blocking(move || {
        inference::open_rwkv(&model_path_open, n_threads)
    })
    .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            warn!(error = ?e, "rwkv open failed — supervisor idle");
            return None;
        }
        Err(e) => {
            warn!(error = %e, "rwkv open join errored — supervisor idle");
            return None;
        }
    };

    Some(LoadedRwkv {
        runtime,
        model_path,
        heartbeat_prompt: s.heartbeat_prompt.clone(),
        heartbeat: Duration::from_secs(s.heartbeat_secs),
        max_gen_tokens: s.max_gen_tokens,
        http_bind,
    })
}

fn inference_backend_from(loaded: Option<&LoadedRwkv>) -> inference::Backend {
    match loaded {
        Some(l) => inference::Backend::RwkvCpp {
            runtime: l.runtime.clone(),
            model_path: l.model_path.clone(),
            heartbeat_prompt: l.heartbeat_prompt.clone(),
            heartbeat: l.heartbeat,
            max_gen_tokens: l.max_gen_tokens,
            http_bind: l.http_bind,
        },
        None => inference::Backend::Unspecified,
    }
}

/// Build a burst closure the calibration job uses to measure real
/// tokens/CPU-second. We clone the ctx via `clone_for_parallel` so
/// the burst doesn't contend with the heartbeat loop for scratch
/// buffers; the mmap'd weights are shared. Returns `None` when there
/// is no runtime or the clone fails — calibration then keeps the
/// fallback throughput number.
fn build_burst_fn(loaded: Option<&LoadedRwkv>) -> Option<calibration::BurstFn> {
    let l = loaded?;
    let burst_ctx = match l.runtime.ctx.clone_for_parallel(l.runtime.n_threads) {
        Ok(c) => c,
        Err(e) => {
            warn!(error = ?e,
                  "rwkv clone_for_parallel failed — calibration on fallback throughput");
            return None;
        }
    };
    let tok = Arc::clone(&l.runtime.tok);
    let prompt = l.heartbeat_prompt.clone();
    Some(Box::new(move |n: usize| -> Result<calibration::BurstSample> {
        let r = inference::generate_once(&burst_ctx, &tok, &prompt, n);
        if r.gen_tokens == 0 {
            anyhow::bail!("burst produced no tokens");
        }
        Ok(calibration::BurstSample {
            gen_tokens: r.gen_tokens,
        })
    }))
}

/// Backend fingerprint for calibration invalidation. Only rwkv-cpp is
/// supported as a model backend; swapping GGUF quantisation levels or
/// model families invalidates measured throughput too.
fn backend_identifier(cfg: &Config) -> String {
    match cfg.inference_backend.as_str() {
        "rwkv-cpp" => {
            let model = cfg
                .rwkv_cpp
                .as_ref()
                .and_then(|s| s.model_path.as_ref())
                .or(cfg.model_path.as_ref())
                .map(|p| p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string())
                .unwrap_or_else(|| "no-model".into());
            format!("rwkv-cpp:{model}")
        }
        other => format!("unknown:{other}"),
    }
}

fn load_config() -> Result<Config> {
    let path = std::env::var("NOESIS_CONFIG")
        .context("NOESIS_CONFIG env var not set")?;
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading config {path}"))?;
    let cfg: Config = toml::from_str(&text)
        .with_context(|| format!("parsing config {path}"))?;
    Ok(cfg)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    info!("noesis-runtime starting");

    let cfg = load_config()?;
    info!(state_path = %cfg.state_path.display(),
          backend = %cfg.inference_backend,
          "config loaded");

    let store = Arc::new(
        Store::open(&cfg.state_path)
            .with_context(|| format!("opening store at {}", cfg.state_path.display()))?,
    );
    info!("all zone stores open");

    let fingerprint = calibration::SystemFingerprint::detect();
    let backend_id = backend_identifier(&cfg);
    let cal = calibration::load_or_fallback(&cfg.calibration_path, &fingerprint, &backend_id);
    let drip = cal.drip_rate_default();
    info!(
        source = if cal.defaulted { "fallback" } else { "measured" },
        tokens_per_cpu_second = cal.tokens_per_cpu_second,
        fan_safe_cpu_percent = cal.fan_safe_cpu_percent,
        n_cores = cal.n_cores,
        drip_tokens_per_sec = %format!("{drip:.2}"),
        "ambient drip ceiling",
    );
    if let Err(e) = store.system_obs.insert(&EventInput {
        kind: "calibration_state".into(),
        payload: json!({
            "source": if cal.defaulted { "fallback" } else { "measured" },
            "tokens_per_cpu_second": cal.tokens_per_cpu_second,
            "fan_safe_cpu_percent": cal.fan_safe_cpu_percent,
            "n_cores": cal.n_cores,
            "cpu_model": cal.cpu_model,
            "kernel": cal.kernel,
            "backend": cal.backend,
            "measured_at": cal.measured_at,
            "drip_tokens_per_sec": drip,
            "safety_margin": calibration::DEFAULT_SAFETY_MARGIN,
        }),
        refs: vec![],
    }) {
        warn!(error = %e, "calibration_state insert failed");
    }

    let loaded_rwkv = load_rwkv_if_configured(&cfg).await;
    let burst_fn = build_burst_fn(loaded_rwkv.as_ref());
    let inference_cfg = inference::InferenceConfig {
        backend: inference_backend_from(loaded_rwkv.as_ref()),
        ..inference::InferenceConfig::default()
    };
    let shutdown_flag = Arc::clone(&inference_cfg.shutdown);
    let inference_handle = tokio::spawn(inference::run(Arc::clone(&store), inference_cfg));
    let retention_handle = tokio::spawn(retention::run(
        Arc::clone(&store),
        retention::RetentionConfig::default(),
    ));
    // Background thermal sweep + throughput measurement — no-op when
    // a valid measured calibration is already on disk. On first boot /
    // after fingerprint invalidation, waits `startup_grace`, runs a
    // ~2-minute sweep on the blocking pool for `fan_safe_cpu_percent`,
    // and (when the burst hook is available) drives a few short bursts
    // through the parallel rwkv context for `tokens_per_cpu_second`.
    let calibrate_handle = tokio::spawn(calibration::run_background_job(
        Arc::clone(&store),
        fingerprint.clone(),
        cal.clone(),
        calibration::BackgroundJobConfig {
            startup_grace: Duration::from_secs(30),
            sweep: calibration::sweep::SweepConfig::default(),
            calibration_path: cfg.calibration_path.clone(),
            backend_id: backend_id.clone(),
        },
        burst_fn,
    ));
    let collector_handles = vec![
        (
            "system_obs",
            tokio::spawn(collectors::system_obs::run(
                Arc::clone(&store),
                collectors::system_obs::SystemObsConfig::default(),
            )),
        ),
        (
            "proc_stat",
            tokio::spawn(collectors::proc_stat::run(
                Arc::clone(&store),
                collectors::proc_stat::ProcStatConfig::default(),
            )),
        ),
        (
            "proc_net",
            tokio::spawn(collectors::proc_net::run(
                Arc::clone(&store),
                collectors::proc_net::ProcNetConfig::default(),
            )),
        ),
        (
            "journal",
            tokio::spawn(collectors::journal::run(
                Arc::clone(&store),
                collectors::journal::JournalConfig::default(),
            )),
        ),
        (
            "proc_self",
            tokio::spawn(collectors::proc_self::run(
                Arc::clone(&store),
                collectors::proc_self::ProcSelfConfig::default(),
            )),
        ),
        (
            "evdev",
            tokio::spawn(collectors::evdev::run(
                Arc::clone(&store),
                collectors::evdev::EvdevConfig::default(),
            )),
        ),
    ];
    info!(count = collector_handles.len(), "collectors spawned");

    let mut sigterm = signal(SignalKind::terminate())?;
    let mut sigint = signal(SignalKind::interrupt())?;
    let mut heartbeat = tokio::time::interval(Duration::from_secs(60));

    loop {
        tokio::select! {
            _ = sigterm.recv() => { info!("SIGTERM — shutting down"); break; }
            _ = sigint.recv()  => { info!("SIGINT — shutting down"); break; }
            _ = heartbeat.tick() => { info!("heartbeat"); }
        }
    }

    // Flip the cooperative shutdown flag first: rwkv-cpp runs on a
    // spawn_blocking OS thread that ignores `abort()` mid-eval; the flag
    // lets it exit at the next heartbeat boundary.
    shutdown_flag.store(true, Ordering::SeqCst);
    for (name, handle) in &collector_handles {
        handle.abort();
        let _ = name;
    }
    inference_handle.abort();
    retention_handle.abort();
    calibrate_handle.abort();
    for (name, handle) in collector_handles {
        match handle.await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => warn!(collector = name, error = %e, "collector exited with error"),
            Err(e) if e.is_cancelled() => info!(collector = name, "collector cancelled"),
            Err(e) => warn!(collector = name, error = %e, "collector join error"),
        }
    }
    match inference_handle.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => warn!(component = "inference", error = %e, "exited with error"),
        Err(e) if e.is_cancelled() => info!(component = "inference", "cancelled"),
        Err(e) => warn!(component = "inference", error = %e, "join error"),
    }
    match retention_handle.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => warn!(component = "retention", error = %e, "exited with error"),
        Err(e) if e.is_cancelled() => info!(component = "retention", "cancelled"),
        Err(e) => warn!(component = "retention", error = %e, "join error"),
    }
    match calibrate_handle.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => warn!(component = "calibrate", error = %e, "exited with error"),
        Err(e) if e.is_cancelled() => info!(component = "calibrate", "cancelled"),
        Err(e) => warn!(component = "calibrate", error = %e, "join error"),
    }

    info!("noesis-runtime stopped");
    Ok(())
}
