//! noesis-runtime — supervisor process.
//!
//! Responsibilities (skeleton, will grow):
//!   1. Load runtime config from `$NOESIS_CONFIG` (TOML).
//!   2. Open all four zone stores under `state_path`.
//!   3. Spawn the inference child (rwkv-cpp or ollama) — TODO.
//!   4. Sit on a supervised loop, handling SIGTERM cleanly.
//!
//! Everything past step 2 is TODO for the Phase B skeleton; the point right
//! now is that the systemd unit stays alive, opens the stores without error,
//! and can be verified with `journalctl --user -u noesis-runtime`.

mod collectors;
mod inference;
mod retention;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use noesis_store::Store;
use serde::Deserialize;
use tokio::signal::unix::{signal, SignalKind};
use tracing::{info, warn};

#[derive(Debug, Deserialize)]
struct Config {
    state_path: PathBuf,
    #[serde(default)]
    model_path: Option<PathBuf>,
    #[serde(default = "default_backend")]
    inference_backend: String,
    #[serde(default)]
    rwkv_cpp: Option<RwkvCppSection>,
    #[serde(default)]
    ollama: Option<OllamaSection>,
}

#[derive(Debug, Deserialize)]
struct RwkvCppSection {
    binary: String,
    #[serde(default)]
    threads: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OllamaSection {
    endpoint: String,
    #[serde(default)]
    model: Option<String>,
    #[serde(default = "default_heartbeat_prompt")]
    heartbeat_prompt: String,
    #[serde(default = "default_heartbeat_secs")]
    heartbeat_secs: u64,
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

fn inference_config_from(cfg: &Config) -> inference::InferenceConfig {
    let backend = match cfg.inference_backend.as_str() {
        "rwkv-cpp" => match &cfg.rwkv_cpp {
            Some(s) => inference::Backend::RwkvCpp {
                binary: s.binary.clone(),
            },
            None => inference::Backend::Unspecified,
        },
        "ollama" => match &cfg.ollama {
            Some(s) => inference::Backend::Ollama {
                endpoint: s.endpoint.clone(),
                model: s.model.clone(),
                heartbeat_prompt: s.heartbeat_prompt.clone(),
                heartbeat: Duration::from_secs(s.heartbeat_secs),
            },
            None => inference::Backend::Unspecified,
        },
        other => {
            warn!(backend = other, "unknown inference backend name");
            inference::Backend::Unspecified
        }
    };
    inference::InferenceConfig {
        backend,
        ..inference::InferenceConfig::default()
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

    let inference_cfg = inference_config_from(&cfg);
    let inference_handle = tokio::spawn(inference::run(Arc::clone(&store), inference_cfg));
    let retention_handle = tokio::spawn(retention::run(
        Arc::clone(&store),
        retention::RetentionConfig::default(),
    ));
    let _ = cfg.model_path;

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

    for (name, handle) in &collector_handles {
        handle.abort();
        let _ = name;
    }
    inference_handle.abort();
    retention_handle.abort();
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

    info!("noesis-runtime stopped");
    Ok(())
}
