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

pub mod lens;
mod rwkv_http;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use noesis_rwkv::{sample, tokenizer::WorldTokenizer, RwkvContext, RwkvSession, SamplingParams, SplitMix64};
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

/// Derive a user-facing model name from its file path.
///
/// Resolves symlinks first (`model.bin` → `rwkv7-g1i-2.9b-q5_1.bin`) so the
/// quant suffix is visible. For nix-store paths where the resolved stem is
/// still `"model"`, falls back to stripping the hash prefix from the parent.
pub fn model_display_name(path: &Path) -> String {
    let resolved = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let stem = resolved.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if stem.is_empty() || stem == "model" {
        resolved.parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .and_then(|n| n.splitn(2, '-').nth(1))
            .unwrap_or("noesis-model")
            .to_string()
    } else {
        stem.to_string()
    }
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
        /// Thread count for the interactive/HTTP clone context. Defaults
        /// to `runtime.n_threads` (heartbeat count) when not set separately.
        http_threads: u32,
    },
    Unspecified,
}

pub struct InferenceConfig {
    pub backend: Backend,
    /// Cooperative shutdown flag. The rwkv-cpp loop runs on a
    /// `spawn_blocking` thread that cannot be cancelled mid-eval; main
    /// flips this on SIGTERM and the loop breaks between heartbeats.
    pub shutdown: Arc<AtomicBool>,
    /// True while at least one interactive request is being served
    /// through the HTTP shim. Read by ambient consumers (calibration,
    /// future drip pacer) so they can back off during user work — see
    /// `docs/policies.md § CPU / thermal / interactive regime`.
    pub interactive: Arc<AtomicBool>,
    /// Root directory for per-lens WKV snapshots (plan §5). `None` means
    /// the shim's `/lens/*` endpoints return `HTTP 501`.
    pub lens_root: Option<PathBuf>,
    /// Context transform config (plan §10). Passed to the HTTP shim.
    /// Defaults to `TransformConfig::default()` (tail_turns=4, no preamble).
    pub transform_config: noesis_http::TransformConfig,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Unspecified,
            shutdown: Arc::new(AtomicBool::new(false)),
            interactive: Arc::new(AtomicBool::new(false)),
            lens_root: None,
            transform_config: noesis_http::TransformConfig::default(),
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
            http_threads,
        } => {
            run_rwkv_cpp(
                store,
                runtime,
                model_path,
                heartbeat_prompt,
                heartbeat,
                max_gen_tokens,
                http_bind,
                http_threads,
                cfg.shutdown,
                cfg.interactive,
                cfg.lens_root,
                cfg.transform_config,
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
    http_threads: u32,
    shutdown: Arc<AtomicBool>,
    interactive: Arc<AtomicBool>,
    lens_root: Option<PathBuf>,
    transform_config: noesis_http::TransformConfig,
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
    // heartbeat loop. Uses `http_threads` (possibly higher than the
    // heartbeat's `n_threads`) to reduce interactive TTFT.
    let model_name: Arc<str> = model_display_name(&model_path).into();

    // Always clone_for_parallel so HTTP gets http_threads (not the heartbeat
    // thread count). When heartbeat is disabled the original ctx is unused
    // after this — weights remain mmap-shared through the clone.
    let http_ctx_opt = match ctx.clone_for_parallel(http_threads) {
        Ok(c) => {
            if heartbeat == Duration::ZERO {
                info!("heartbeat disabled — substrate serves HTTP on demand only");
            }
            Some(c)
        }
        Err(e) => {
            warn!(error = ?e, "rwkv clone_for_parallel failed — HTTP shim disabled");
            None
        }
    };

    let http_task = if let (Some(bind), Some(http_ctx)) = (http_bind, http_ctx_opt) {
        let tok = Arc::clone(&tok);
        let shutdown = Arc::clone(&shutdown);
        let interactive = Arc::clone(&interactive);
        let composer = Arc::new(noesis_composer::Composer::new(
            noesis_composer::ComposerConfig::default(),
            Arc::clone(&store),
        ));
        Some(tokio::spawn(rwkv_http::serve(
            http_ctx,
            tok,
            bind,
            max_gen_tokens,
            shutdown,
            interactive,
            lens_root.clone(),
            transform_config,
            composer,
            Arc::clone(&model_name),
        )))
    } else {
        None
    };

    // Heartbeat loop — skip entirely when disabled (secs=0).
    // When enabled, runs on the original ctx (HTTP uses the clone above).
    if heartbeat > Duration::ZERO {
        let hb_ctx = ctx.clone();
        let tok = Arc::clone(&tok);
        let store = Arc::clone(&store);
        let shutdown = Arc::clone(&shutdown);
        let model_path = model_path.clone();
        let heartbeat_task = tokio::task::spawn_blocking(move || {
            heartbeat_loop(
                store, hb_ctx, tok, model_path,
                heartbeat_prompt, heartbeat, max_gen_tokens, shutdown,
            );
        });
        let _ = heartbeat_task.await;
    }

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
        // Heartbeat stays deterministic — the health probe compares tokens/s
        // and success rate across ticks, so any sampling stochasticity would
        // just add noise to the signal.
        let result = generate_once(
            &ctx,
            &tok,
            &heartbeat_prompt,
            max_gen_tokens,
            &SamplingParams::greedy(),
            &[],
        );

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

/// Why generation stopped. Mirrors Ollama's `done_reason` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Hit the `max_gen` budget.
    Length,
    /// Matched one of the caller-supplied stop strings on the decoded
    /// tail — the matched suffix is trimmed from `response`.
    StopSequence,
    /// Prompt ingestion or a mid-generation eval failed; generation
    /// terminated early via an error path rather than a clean stop.
    Error,
}

impl StopReason {
    pub fn as_ollama_str(self) -> &'static str {
        match self {
            StopReason::Length => "length",
            StopReason::StopSequence => "stop",
            StopReason::Error => "error",
        }
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
    pub stop_reason: StopReason,
    /// True iff prompt ingestion succeeded and the run terminated
    /// cleanly (either budget filled or stop sequence matched). False
    /// on any eval error — the caller distinguishes stalled from healthy.
    pub ok: bool,
}

/// Wrapper for the common case (no lens): fresh session, run
/// `generate_on_session`, drop the session. Everything with a lens goes
/// through [`generate_on_session`] directly so the caller can persist
/// the ended session's state.
pub fn generate_once(
    ctx: &RwkvContext,
    tok: &WorldTokenizer,
    prompt: &str,
    max_gen: usize,
    params: &SamplingParams,
    stops: &[String],
) -> GenerateResult {
    let mut session = RwkvSession::new(ctx.clone());
    generate_on_session(&mut session, tok, prompt, max_gen, params, stops, None)
}

/// Prompt-in / response-out on a caller-owned session. Session state
/// is mutated in place — for lens flow the caller loads state before
/// this call and saves it after.
///
/// `on_delta`, when `Some`, is called after each generated token with
/// the UTF-8 string delta since the previous call. The stop-string
/// suffix is never included in any delta. Used by the streaming HTTP
/// path; pass `None` for the buffered path.
pub fn generate_on_session(
    session: &mut RwkvSession,
    tok: &WorldTokenizer,
    prompt: &str,
    max_gen: usize,
    params: &SamplingParams,
    stops: &[String],
    mut on_delta: Option<&mut dyn FnMut(&str)>,
) -> GenerateResult {
    let prompt_ids = tok.encode(prompt);
    let t_prompt = Instant::now();
    let prompt_ok = session.eval_sequence(&prompt_ids).is_ok();
    let prompt_ms = t_prompt.elapsed().as_millis() as u64;

    let seed = params
        .seed
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0xDEADBEEF)
        });
    let mut rng = SplitMix64::new(seed);

    // Precompute the byte window we need to keep in the rolling tail
    // to reliably match any stop string. Any stop longer than what's
    // been generated so far cannot match.
    let stop_tail_bytes = stops.iter().map(|s| s.len()).max().unwrap_or(0);

    let streaming = on_delta.is_some();
    let mut generated: Vec<u32> = Vec::with_capacity(max_gen);
    let mut stop_reason = StopReason::Length;
    let mut eval_error = false;
    let mut trim_bytes = 0usize;
    // Byte offset into the decoded text that has already been emitted
    // as deltas. Only meaningful when `streaming` is true.
    let mut emitted_bytes = 0usize;

    let t_gen = Instant::now();
    if prompt_ok {
        let mut last = *prompt_ids.last().unwrap_or(&0);
        for _ in 0..max_gen {
            match session.eval(last) {
                Ok(logits) => {
                    let next = sample(logits, &generated, params, &mut rng);
                    generated.push(next);
                    last = next;

                    // Decode when we need to check stops or emit a delta.
                    if !stops.is_empty() || streaming {
                        let text = tok.decode(&generated).unwrap_or_default();

                        // Stop-string check.
                        if !stops.is_empty() {
                            let cmp_window = if stop_tail_bytes == 0
                                || text.len() <= stop_tail_bytes
                            {
                                text.as_str()
                            } else {
                                let cut = text.len() - stop_tail_bytes;
                                let cut = (cut..text.len())
                                    .find(|&i| text.is_char_boundary(i))
                                    .unwrap_or(text.len());
                                &text[cut..]
                            };
                            if let Some(hit) =
                                stops.iter().find(|s| cmp_window.ends_with(s.as_str()))
                            {
                                stop_reason = StopReason::StopSequence;
                                trim_bytes = hit.len();
                                // Emit the pre-stop delta (without the matched suffix).
                                if let Some(cb) = on_delta.as_mut() {
                                    let trimmed_len = {
                                        let n = text.len().saturating_sub(trim_bytes);
                                        (0..=n)
                                            .rev()
                                            .find(|&i| text.is_char_boundary(i))
                                            .unwrap_or(0)
                                    };
                                    if trimmed_len > emitted_bytes
                                        && text.is_char_boundary(emitted_bytes)
                                    {
                                        let s = text[emitted_bytes..trimmed_len].replace('\x00', "");
                                        if !s.is_empty() { cb(&s); }
                                    }
                                }
                                break;
                            }
                        }

                        // Emit newly decoded bytes as a delta, holding back the
                        // last `stop_tail_bytes` to avoid emitting a partial
                        // stop sequence that completes on the next token.
                        if let Some(cb) = on_delta.as_mut() {
                            let safe_end = if stop_tail_bytes > 0 {
                                let e = text.len().saturating_sub(stop_tail_bytes);
                                (0..=e).rev()
                                    .find(|&i| text.is_char_boundary(i))
                                    .unwrap_or(emitted_bytes)
                            } else {
                                text.len()
                            };
                            if safe_end > emitted_bytes
                                && text.is_char_boundary(emitted_bytes)
                            {
                                let s = text[emitted_bytes..safe_end].replace('\x00', "");
                                if !s.is_empty() { cb(&s); }
                                emitted_bytes = safe_end;
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(error = ?e, "rwkv eval failed mid-gen");
                    eval_error = true;
                    break;
                }
            }
        }
    } else {
        warn!("rwkv eval_sequence failed");
    }
    // Flush any held-back delta bytes (the stop_tail_bytes window) when the
    // loop ended for a reason other than a stop-sequence match.
    if stop_reason != StopReason::StopSequence {
        if let Some(cb) = on_delta.as_mut() {
            let flush_text = tok.decode(&generated).unwrap_or_default().replace('\x00', "");
            if flush_text.len() > emitted_bytes
                && flush_text.is_char_boundary(emitted_bytes)
            {
                let s = flush_text[emitted_bytes..].replace('\x00', "");
                if !s.is_empty() { cb(&s); }
            }
        }
    }
    let gen_ms = t_gen.elapsed().as_millis() as u64;

    let mut response = tok
        .decode(&generated)
        .unwrap_or_else(|e| format!("[decode error: {e}]"))
        .replace('\x00', "");
    if trim_bytes > 0 && response.len() >= trim_bytes {
        let new_len = response.len() - trim_bytes;
        // Cut back to the last char boundary to guarantee valid UTF-8.
        let new_len = (0..=new_len)
            .rev()
            .find(|&i| response.is_char_boundary(i))
            .unwrap_or(0);
        response.truncate(new_len);
    }

    if !prompt_ok || eval_error {
        stop_reason = StopReason::Error;
    }
    let ok = prompt_ok && !eval_error;
    GenerateResult {
        prompt_tokens: prompt_ids.len(),
        gen_tokens: generated.len(),
        prompt_ms,
        gen_ms,
        response,
        stop_reason,
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

