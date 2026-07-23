//! Inference child supervisor (Phase B skeleton).
//!
//! Two backend flavours:
//!
//! - **rwkv-cpp**: linked in-process via `noesis-rwkv`. On each heartbeat
//!   we spin a fresh `RwkvSession` off the shared context, encode a probe
//!   prompt, run `eval_sequence` + a short greedy `eval` loop, and log
//!   both the health signal and the generated text into the store.
//!   When `http_bind` is set, an Ollama-shape HTTP shim runs alongside
//!   on a `clone_for_parallel` context so external clients (e.g. a
//!   Claude-Code-style CLI) can drive `/api/generate` without contending
//!   with the heartbeat.
//! - **ollama**: an already-running HTTP daemon on the local host. TCP
//!   health check on `health_interval`; `/api/generate` heartbeat on the
//!   backend-specific interval.
//!
//! Both backends land uniform `inference_health` events in `system_obs`
//! and backend-specific generation events in `session_scratch`
//! (`rwkv_generation` / `ollama_generation`).

mod rwkv_http;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use noesis_rwkv::{argmax, tokenizer::WorldTokenizer, RwkvContext, RwkvSession};
use noesis_schema::EventInput;
use noesis_store::Store;
use serde::Deserialize;
use serde_json::json;
use tokio::net::TcpStream;
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub enum Backend {
    RwkvCpp {
        model_path: PathBuf,
        n_threads: u32,
        heartbeat_prompt: String,
        heartbeat: Duration,
        max_gen_tokens: usize,
        http_bind: Option<SocketAddr>,
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
    /// Cooperative shutdown flag. The rwkv-cpp loop runs on a
    /// `spawn_blocking` thread that cannot be cancelled mid-eval; main
    /// flips this on SIGTERM and the loop breaks between heartbeats.
    pub shutdown: Arc<AtomicBool>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Unspecified,
            health_interval: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(3),
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }
}

pub async fn run(store: Arc<Store>, cfg: InferenceConfig) -> anyhow::Result<()> {
    match cfg.backend {
        Backend::RwkvCpp {
            model_path,
            n_threads,
            heartbeat_prompt,
            heartbeat,
            max_gen_tokens,
            http_bind,
        } => {
            run_rwkv_cpp(
                store,
                model_path,
                n_threads,
                heartbeat_prompt,
                heartbeat,
                max_gen_tokens,
                http_bind,
                cfg.shutdown,
            )
            .await
        }
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

/// rwkv-cpp: load the model in-process via `noesis-rwkv`, then run the
/// heartbeat loop and (optionally) the Ollama-shape HTTP shim on a
/// cloned C-level context.
///
/// Both heartbeat and HTTP handlers do their eval work on the tokio
/// blocking pool (rwkv.cpp is synchronous CPU work with no cooperative
/// yield points; store inserts hit sqlite synchronously). Shutdown is
/// cooperative via the `shutdown` `AtomicBool` — main flips it on
/// SIGTERM and both paths exit at their next boundary.
#[allow(clippy::too_many_arguments)]
async fn run_rwkv_cpp(
    store: Arc<Store>,
    model_path: PathBuf,
    n_threads: u32,
    heartbeat_prompt: String,
    heartbeat: Duration,
    max_gen_tokens: usize,
    http_bind: Option<SocketAddr>,
    shutdown: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    // Blocking init: open ctx + build tokenizer. Neither is Send-safe
    // to await on, so we do it inside spawn_blocking and unpack after.
    let load_started = Instant::now();
    let init = tokio::task::spawn_blocking({
        let model_path = model_path.clone();
        move || -> anyhow::Result<(RwkvContext, WorldTokenizer)> {
            let ctx = RwkvContext::open(&model_path, n_threads, 0)
                .map_err(|e| anyhow::anyhow!("rwkv open failed: {e:?}"))?;
            let tok = WorldTokenizer::new()
                .map_err(|e| anyhow::anyhow!("tokenizer init failed: {e}"))?;
            Ok((ctx, tok))
        }
    })
    .await?;
    let (ctx, tok) = match init {
        Ok(v) => v,
        Err(e) => {
            warn!(model = %model_path.display(), error = %e, "rwkv init failed — supervisor idle");
            return Ok(());
        }
    };
    info!(
        model = %model_path.display(),
        load_ms = load_started.elapsed().as_millis() as u64,
        n_vocab = ctx.n_vocab(),
        n_embed = ctx.n_embed(),
        n_layer = ctx.n_layer(),
        state_len = ctx.state_len(),
        n_threads,
        "rwkv.cpp loaded",
    );
    let tok = Arc::new(tok);

    // Optional HTTP shim on a cloned rwkv_context. `clone_for_parallel`
    // gives us a second C-level context sharing the weight mmap but with
    // its own scratch buffers — safe to run concurrently with the
    // heartbeat loop.
    let http_task = if let Some(bind) = http_bind {
        match ctx.clone_for_parallel(n_threads) {
            Ok(http_ctx) => {
                let tok = Arc::clone(&tok);
                let shutdown = Arc::clone(&shutdown);
                Some(tokio::spawn(rwkv_http::serve(
                    http_ctx,
                    tok,
                    bind,
                    max_gen_tokens,
                    shutdown,
                )))
            }
            Err(e) => {
                warn!(error = ?e, "rwkv clone_for_parallel failed — HTTP shim disabled");
                None
            }
        }
    } else {
        None
    };

    // Heartbeat loop on the original ctx, on the blocking pool.
    let heartbeat_task = {
        let ctx = ctx.clone();
        let tok = Arc::clone(&tok);
        let store = Arc::clone(&store);
        let shutdown = Arc::clone(&shutdown);
        let model_path = model_path.clone();
        tokio::task::spawn_blocking(move || {
            heartbeat_loop(
                store,
                ctx,
                tok,
                model_path,
                heartbeat_prompt,
                heartbeat,
                max_gen_tokens,
                shutdown,
            );
        })
    };

    let _ = heartbeat_task.await;
    if let Some(h) = http_task {
        let _ = h.await;
    }
    Ok(())
}

fn heartbeat_loop(
    store: Arc<Store>,
    ctx: RwkvContext,
    tok: Arc<WorldTokenizer>,
    model_path: PathBuf,
    heartbeat_prompt: String,
    heartbeat: Duration,
    max_gen_tokens: usize,
    shutdown: Arc<AtomicBool>,
) {
    let model_path_str = model_path.display().to_string();
    loop {
        if shutdown.load(Ordering::Relaxed) {
            info!("rwkv heartbeat shutdown");
            return;
        }
        let round_started = Instant::now();
        let result = generate_once(&ctx, &tok, &heartbeat_prompt, max_gen_tokens);

        let payload_health = json!({
            "backend": "rwkv-cpp",
            "model_path": &model_path_str,
            "ok": result.ok,
            "prompt_tokens": result.prompt_tokens,
            "gen_tokens": result.gen_tokens,
            "wall_ms": round_started.elapsed().as_millis() as u64,
        });
        if let Err(e) = store.system_obs.insert(&EventInput {
            kind: "inference_health".into(),
            payload: payload_health,
            refs: vec![],
        }) {
            warn!(error = %e, "inference_health insert failed");
        }

        if result.ok {
            let tok_per_s = if result.gen_ms > 0 {
                result.gen_tokens as f64 / (result.gen_ms as f64 / 1000.0)
            } else {
                0.0
            };
            info!(
                prompt_tokens = result.prompt_tokens,
                gen_tokens = result.gen_tokens,
                prompt_ms = result.prompt_ms,
                gen_ms = result.gen_ms,
                tok_per_s = %format!("{tok_per_s:.1}"),
                "rwkv heartbeat ok",
            );
            let payload = json!({
                "backend": "rwkv-cpp",
                "model_path": &model_path_str,
                "prompt": &heartbeat_prompt,
                "response": result.response,
                "prompt_tokens": result.prompt_tokens,
                "gen_tokens": result.gen_tokens,
                "prompt_ms": result.prompt_ms,
                "gen_ms": result.gen_ms,
                "wall_ms": round_started.elapsed().as_millis() as u64,
            });
            if let Err(e) = store.session_scratch.insert(&EventInput {
                kind: "rwkv_generation".into(),
                payload,
                refs: vec![],
            }) {
                warn!(error = %e, "rwkv_generation insert failed");
            }
        } else {
            warn!(model = %model_path_str, "rwkv heartbeat failed");
        }

        interruptible_sleep(heartbeat, &shutdown);
    }
}

/// One prompt-in / response-out round on the given context. Shared by
/// the heartbeat loop and the HTTP shim so timing/failure semantics
/// stay identical.
pub(super) struct GenerateResult {
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    pub prompt_ms: u64,
    pub gen_ms: u64,
    pub response: String,
    /// True iff prompt ingestion succeeded and the full `max_gen`
    /// budget was produced (partial generation ⇒ false so the caller
    /// can distinguish a stalled decode from a healthy round).
    pub ok: bool,
}

pub(super) fn generate_once(
    ctx: &RwkvContext,
    tok: &WorldTokenizer,
    prompt: &str,
    max_gen: usize,
) -> GenerateResult {
    let prompt_ids = tok.encode(prompt);
    let mut session = RwkvSession::new(ctx.clone());
    let t_prompt = Instant::now();
    let prompt_ok = session.eval_sequence(&prompt_ids).is_ok();
    let prompt_ms = t_prompt.elapsed().as_millis() as u64;

    let mut generated: Vec<u32> = Vec::with_capacity(max_gen);
    let t_gen = Instant::now();
    if prompt_ok {
        let mut last = *prompt_ids.last().unwrap_or(&0);
        for _ in 0..max_gen {
            match session.eval(last) {
                Ok(logits) => {
                    let next = argmax(logits);
                    generated.push(next);
                    last = next;
                }
                Err(e) => {
                    warn!(error = ?e, "rwkv eval failed mid-gen");
                    break;
                }
            }
        }
    } else {
        warn!("rwkv eval_sequence failed");
    }
    let gen_ms = t_gen.elapsed().as_millis() as u64;
    let response = tok
        .decode(&generated)
        .unwrap_or_else(|e| format!("[decode error: {e}]"));
    let ok = prompt_ok && generated.len() == max_gen;
    GenerateResult {
        prompt_tokens: prompt_ids.len(),
        gen_tokens: generated.len(),
        prompt_ms,
        gen_ms,
        response,
        ok,
    }
}

/// Sleep in 100 ms slices so the rwkv-cpp loop can honour a shutdown
/// flip mid-pause. Returns as soon as the flag flips or the total
/// elapsed time reaches `total`.
fn interruptible_sleep(total: Duration, shutdown: &AtomicBool) {
    let slice = Duration::from_millis(100);
    let deadline = Instant::now() + total;
    while Instant::now() < deadline {
        if shutdown.load(Ordering::Relaxed) {
            return;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        std::thread::sleep(remaining.min(slice));
    }
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
