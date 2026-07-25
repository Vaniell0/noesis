//! Inference child supervisor (Phase B skeleton).
//!
//! One model backend: **rwkv-cpp**, linked in-process via `noesis-rwkv`.
//! On each heartbeat we spin a fresh `RwkvSession` off the shared
//! context, encode a probe prompt, run `eval_sequence` + a short greedy
//! `eval` loop, and log both the health signal and the generated text
//! into the store. When `http_bind` is set, an Ollama-shape HTTP shim
//! runs alongside on a `clone_for_parallel` context so external clients
//! (e.g. a Claude-Code-style CLI) can drive `/api/generate` without
//! contending with the heartbeat.
//!
//! The Ollama HTTP surface is a **client wire format** we expose, not a
//! model backend — the runtime never delegates generation to an external
//! Ollama daemon. See `inference::rwkv_http` for the shim.
//!
//! Health signals land in `system_obs` as `inference_health`; per-round
//! generations land in `session_scratch` as `rwkv_generation`.

mod rwkv_http;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use noesis_rwkv::{argmax, tokenizer::WorldTokenizer, RwkvContext, RwkvSession};
use noesis_schema::EventInput;
use noesis_store::Store;
use serde_json::json;
use tracing::{info, warn};

/// The loaded rwkv-cpp state — shared between the inference supervisor,
/// the HTTP shim, and the background calibration burst hook so we open
/// the model **once** at startup instead of paying multi-second mmap +
/// tokenizer init on each consumer.
///
/// Both fields are cheap to clone: `RwkvContext` is `Clone` (refcounted
/// C-level handle); `Arc<WorldTokenizer>` is trivially shareable.
#[derive(Clone)]
pub struct RwkvRuntime {
    pub ctx: RwkvContext,
    pub tok: Arc<WorldTokenizer>,
    pub n_threads: u32,
}

/// Load model + tokenizer synchronously. Callers on the async runtime
/// wrap in `spawn_blocking` — this is a multi-second mmap on real
/// weights, so it must not block a tokio worker.
pub fn open_rwkv(model_path: &Path, n_threads: u32) -> anyhow::Result<RwkvRuntime> {
    let ctx = RwkvContext::open(model_path, n_threads, 0)
        .map_err(|e| anyhow::anyhow!("rwkv open failed: {e:?}"))?;
    let tok = WorldTokenizer::new()
        .map_err(|e| anyhow::anyhow!("tokenizer init failed: {e}"))?;
    Ok(RwkvRuntime {
        ctx,
        tok: Arc::new(tok),
        n_threads,
    })
}

#[derive(Clone)]
pub enum Backend {
    RwkvCpp {
        runtime: RwkvRuntime,
        model_path: PathBuf,
        heartbeat_prompt: String,
        heartbeat: Duration,
        max_gen_tokens: usize,
        http_bind: Option<SocketAddr>,
    },
    Unspecified,
}

pub struct InferenceConfig {
    pub backend: Backend,
    /// Cooperative shutdown flag. The rwkv-cpp loop runs on a
    /// `spawn_blocking` thread that cannot be cancelled mid-eval; main
    /// flips this on SIGTERM and the loop breaks between heartbeats.
    pub shutdown: Arc<AtomicBool>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Unspecified,
            shutdown: Arc::new(AtomicBool::new(false)),
        }
    }
}

pub async fn run(store: Arc<Store>, cfg: InferenceConfig) -> anyhow::Result<()> {
    match cfg.backend {
        Backend::RwkvCpp {
            runtime,
            model_path,
            heartbeat_prompt,
            heartbeat,
            max_gen_tokens,
            http_bind,
        } => {
            run_rwkv_cpp(
                store,
                runtime,
                model_path,
                heartbeat_prompt,
                heartbeat,
                max_gen_tokens,
                http_bind,
                cfg.shutdown,
            )
            .await
        }
        Backend::Unspecified => {
            warn!("inference backend unspecified — supervisor idle");
            Ok(())
        }
    }
}

/// rwkv-cpp: spawn the heartbeat loop on the blocking pool and,
/// optionally, the Ollama-shape HTTP shim on a `clone_for_parallel`
/// context so external clients don't contend with the heartbeat.
///
/// The model is already loaded — see `open_rwkv` (main.rs calls it
/// before spawning this task so the same `RwkvRuntime` can also be
/// handed to `calibration::run_background_job` for the throughput
/// burst without a second multi-second mmap).
///
/// Both heartbeat and HTTP handlers do their eval work on the tokio
/// blocking pool (rwkv.cpp is synchronous CPU work with no cooperative
/// yield points; store inserts hit sqlite synchronously). Shutdown is
/// cooperative via the `shutdown` `AtomicBool` — main flips it on
/// SIGTERM and both paths exit at their next boundary.
#[allow(clippy::too_many_arguments)]
async fn run_rwkv_cpp(
    store: Arc<Store>,
    runtime: RwkvRuntime,
    model_path: PathBuf,
    heartbeat_prompt: String,
    heartbeat: Duration,
    max_gen_tokens: usize,
    http_bind: Option<SocketAddr>,
    shutdown: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let RwkvRuntime { ctx, tok, n_threads } = runtime;
    info!(
        model = %model_path.display(),
        n_vocab = ctx.n_vocab(),
        n_embed = ctx.n_embed(),
        n_layer = ctx.n_layer(),
        state_len = ctx.state_len(),
        n_threads,
        "rwkv.cpp inference supervisor starting",
    );

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
/// the heartbeat loop, the HTTP shim, and the calibration burst hook
/// so timing/failure semantics stay identical.
pub struct GenerateResult {
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

pub fn generate_once(
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

