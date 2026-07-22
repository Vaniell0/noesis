//! Inference child supervisor (Phase B skeleton).
//!
//! Two backend flavours:
//!
//! - **rwkv-cpp**: a CPU inference binary is spawned as a child process.
//!   For now we only verify the binary is executable and log its version
//!   / help output; real prompt/response wiring lands with C1 event-
//!   stream ingestion.
//! - **ollama**: an already-running HTTP daemon on the local host. We
//!   just TCP-check the listening port every tick and emit a health
//!   event into `system_obs`. Real generation lands via HTTP later.
//!
//! Failure discipline mirrors the L1 journal collector: if the backend
//! is genuinely absent (rwkv-cpp binary missing, ollama daemon never
//! listens), we log once at WARN and shut down cleanly instead of
//! busy-restarting.

use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};

use noesis_schema::EventInput;
use noesis_store::Store;
use serde::Deserialize;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpStream;
use tokio::process::Command;
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub enum Backend {
    RwkvCpp {
        binary: String,
    },
    Ollama {
        endpoint: String,
        model: Option<String>,
        heartbeat_prompt: String,
        heartbeat: Duration,
    },
    Unspecified,
}

pub struct InferenceConfig {
    pub backend: Backend,
    pub health_interval: Duration,
    pub connect_timeout: Duration,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Unspecified,
            health_interval: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(3),
        }
    }
}

pub async fn run(store: Arc<Store>, cfg: InferenceConfig) -> anyhow::Result<()> {
    match cfg.backend {
        Backend::RwkvCpp { binary } => run_rwkv_cpp(store, binary).await,
        Backend::Ollama {
            endpoint,
            model,
            heartbeat_prompt,
            heartbeat,
        } => {
            run_ollama(
                store,
                endpoint,
                model,
                heartbeat_prompt,
                heartbeat,
                cfg.health_interval,
                cfg.connect_timeout,
            )
            .await
        }
        Backend::Unspecified => {
            warn!("inference backend unspecified — supervisor idle");
            Ok(())
        }
    }
}

/// Ollama: no child process; the daemon is already system-wide. Two
/// interleaved ticks share one task:
///
/// - **Health tick** (`health_interval`): TCP-connect probe; result lands
///   in `system_obs` as `inference_health`.
/// - **Heartbeat tick** (`heartbeat`, opt-in via `model`): actual
///   `/api/generate` round-trip; response text + timing lands in
///   `session_scratch` as `ollama_generation`.
///
/// The heartbeat proves the model round-trip works end-to-end. Without a
/// configured model the runtime stays health-check-only.
#[allow(clippy::too_many_arguments)]
async fn run_ollama(
    store: Arc<Store>,
    endpoint: String,
    model: Option<String>,
    heartbeat_prompt: String,
    heartbeat: Duration,
    health_interval: Duration,
    connect_timeout: Duration,
) -> anyhow::Result<()> {
    let addr = parse_host_port(&endpoint)?;
    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    let mut health_ticker = tokio::time::interval(health_interval);
    health_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut heartbeat_ticker = tokio::time::interval(heartbeat);
    heartbeat_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            _ = health_ticker.tick() => {
                let ok = tcp_probe(&addr, connect_timeout).await;
                let payload = json!({
                    "backend": "ollama",
                    "endpoint": endpoint,
                    "reachable": ok,
                });
                if let Err(e) = store.system_obs.insert(&EventInput {
                    kind: "inference_health".into(),
                    payload,
                    refs: vec![],
                }) {
                    warn!(error = %e, "inference_health insert failed");
                }
            }
            _ = heartbeat_ticker.tick() => {
                let Some(model_name) = model.as_deref() else { continue; };
                match ollama_generate(&http, &endpoint, model_name, &heartbeat_prompt).await {
                    Ok(result) => {
                        info!(
                            model = model_name,
                            eval_count = result.eval_count.unwrap_or(0),
                            eval_ms = result.eval_duration_ms(),
                            "ollama heartbeat ok",
                        );
                        let payload = json!({
                            "backend": "ollama",
                            "endpoint": endpoint,
                            "model": model_name,
                            "prompt": heartbeat_prompt,
                            "response": result.response,
                            "eval_count": result.eval_count,
                            "eval_duration_ns": result.eval_duration,
                            "prompt_eval_count": result.prompt_eval_count,
                            "prompt_eval_duration_ns": result.prompt_eval_duration,
                            "total_duration_ns": result.total_duration,
                            "wall_ms": result.wall_ms,
                        });
                        if let Err(e) = store.session_scratch.insert(&EventInput {
                            kind: "ollama_generation".into(),
                            payload,
                            refs: vec![],
                        }) {
                            warn!(error = %e, "ollama_generation insert failed");
                        }
                    }
                    Err(e) => warn!(error = %e, model = model_name, "ollama heartbeat failed"),
                }
            }
        }
    }
}

async fn tcp_probe(addr: &str, timeout: Duration) -> bool {
    match tokio::time::timeout(timeout, TcpStream::connect(addr)).await {
        Ok(Ok(_)) => true,
        Ok(Err(e)) => {
            warn!(error = %e, addr = %addr, "ollama TCP connect failed");
            false
        }
        Err(_) => {
            warn!(addr = %addr, "ollama TCP connect timed out");
            false
        }
    }
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateResponse {
    #[serde(default)]
    response: String,
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    eval_duration: Option<u64>,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    prompt_eval_duration: Option<u64>,
    #[serde(default)]
    total_duration: Option<u64>,
    #[serde(skip)]
    wall_ms: u64,
}

impl OllamaGenerateResponse {
    fn eval_duration_ms(&self) -> u64 {
        self.eval_duration.map(|ns| ns / 1_000_000).unwrap_or(0)
    }
}

async fn ollama_generate(
    http: &reqwest::Client,
    endpoint: &str,
    model: &str,
    prompt: &str,
) -> anyhow::Result<OllamaGenerateResponse> {
    let url = format!("{}/api/generate", endpoint.trim_end_matches('/'));
    let started = Instant::now();
    let resp = http
        .post(&url)
        .json(&json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
        }))
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!("ollama /api/generate returned HTTP {}", resp.status());
    }
    let mut parsed: OllamaGenerateResponse = resp.json().await?;
    parsed.wall_ms = started.elapsed().as_millis() as u64;
    Ok(parsed)
}

/// rwkv-cpp: try to invoke the binary with `--help` to verify it's
/// executable, log stdout/stderr, then exit. Real serving mode comes
/// later — rwkv.cpp's example CLIs are one-shot, not daemons.
async fn run_rwkv_cpp(_store: Arc<Store>, binary: String) -> anyhow::Result<()> {
    let mut cmd = Command::new(&binary);
    cmd.arg("--help")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            warn!(binary = %binary, "rwkv-cpp binary not found — supervisor idle");
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };

    if let Some(stdout) = child.stdout.take() {
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                info!(source = "rwkv-cpp.stdout", "{line}");
            }
        });
    }
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                info!(source = "rwkv-cpp.stderr", "{line}");
            }
        });
    }

    let status = child.wait().await?;
    info!(status = ?status, "rwkv-cpp --help completed");
    // Real serving loop lands in Phase C1. For now the supervisor just
    // idles after the probe — long-lived sleep so tokio doesn't reap us.
    std::future::pending::<()>().await;
    Ok(())
}

fn parse_host_port(endpoint: &str) -> anyhow::Result<String> {
    let s = endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
        .unwrap_or(endpoint);
    let host_port = s.split('/').next().unwrap_or(s);
    if !host_port.contains(':') {
        anyhow::bail!("endpoint {endpoint} missing :port");
    }
    Ok(host_port.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_host_port_strips_scheme_and_path() {
        assert_eq!(parse_host_port("http://127.0.0.1:11434").unwrap(), "127.0.0.1:11434");
        assert_eq!(parse_host_port("https://ollama.local:8080/api").unwrap(), "ollama.local:8080");
        assert_eq!(parse_host_port("127.0.0.1:11434").unwrap(), "127.0.0.1:11434");
    }

    #[test]
    fn parse_host_port_requires_port() {
        assert!(parse_host_port("http://127.0.0.1").is_err());
        assert!(parse_host_port("localhost").is_err());
    }
}
