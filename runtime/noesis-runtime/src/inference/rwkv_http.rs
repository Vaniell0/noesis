//! Ollama-shape + Anthropic-shape HTTP shim on top of the in-process rwkv.cpp context.
//!
//! Endpoints:
//!
//! - `GET  /api/version`  — trivial identity string (Ollama compat).
//! - `GET  /api/tags`     — one-model catalogue (Ollama compat).
//! - `POST /api/show`     — model metadata blob (Ollama compat).
//! - `POST /api/generate` — buffered (`stream: false`) and NDJSON streaming
//!   (`stream: true` or omitted). Per-token `{response, done}` objects
//!   followed by a final `done: true` object with timing.
//! - `POST /v1/messages`  — Anthropic Messages API adapter. Translates
//!   `{model, messages, system, max_tokens, stream}` to a generate call
//!   and returns SSE (`text/event-stream`) or a buffered JSON response.
//!   Enables `ANTHROPIC_BASE_URL=http://127.0.0.1:11435` for claude-cli.
//!   Context transform (plan §10) is not yet applied — last user message
//!   is used as the prompt directly until the composer lands.
//!
//! Concurrency: rwkv.cpp forbids concurrent `rwkv_eval` on the same
//! context, so the shim serialises HTTP requests through a
//! `std::sync::Mutex<()>` acquired inside `spawn_blocking`. The
//! heartbeat runs on a *different* cloned context (see
//! `RwkvContext::clone_for_parallel` in the caller), so heartbeat and
//! HTTP never contend.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use bytes::Bytes;
use noesis_rwkv::{tokenizer::WorldTokenizer, RwkvContext, RwkvSession, SamplingParams};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{info, warn};

use noesis_composer::{Composer, ComposerConfig};
use noesis_http::{ChatTurn, ContextTransform, RetrievalSlot, TransformConfig};

use super::lens::{self, LensDir};
use super::generate_on_session;


/// Conservative sampling defaults for the HTTP shim.
///
/// The primary use-case of the in-process model is structured utility work
/// (record writing, emit-gate, classification). For that, deterministic
/// argmax is correct — callers that want creative variance must supply
/// an explicit `temperature`. This overrides the Ollama-compat T=0.8
/// defaults in `SamplingParams::default()` for all HTTP paths.
fn http_sampling_defaults() -> SamplingParams {
    SamplingParams {
        temperature: 0.0, // argmax; short-circuits to greedy path in `sample()`
        top_k: 0,
        top_p: 1.0,
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        seed: None,
    }
}

#[derive(Clone)]
struct HttpState {
    ctx: RwkvContext,
    tok: Arc<WorldTokenizer>,
    eval_lock: Arc<Mutex<()>>,
    default_max_gen: usize,
    /// User-facing model name advertised via /api/tags and /v1/models.
    /// Derived from the model path at startup — never the raw file path.
    model_name: Arc<str>,
    /// Interactive-regime signal: shared with the supervisor so ambient
    /// consumers (calibration, future drip pacer) can back off while a
    /// client request is in flight. Set true on `/api/generate` entry,
    /// false on exit via `InteractiveGuard` (RAII, exception-safe).
    interactive: Arc<AtomicBool>,
    /// Where per-lens snapshots live. `None` disables `/lens/*` endpoints
    /// and the `lens_id` field on `/api/generate` — set at supervisor
    /// startup from `config.lens_root`.
    lens_dir: Option<LensDir>,
    /// Context transform (plan §10). Converts the client's `messages` array
    /// into a substrate prompt: tail_turns + retrieval slot + preamble.
    ctx_transform: Arc<ContextTransform>,
    /// Composer: renders preamble + retrieval snippet per request.
    composer: Arc<Composer>,
}

pub(super) async fn serve(
    ctx: RwkvContext,
    tok: Arc<WorldTokenizer>,
    bind: SocketAddr,
    default_max_gen: usize,
    shutdown: Arc<AtomicBool>,
    interactive: Arc<AtomicBool>,
    lens_root: Option<PathBuf>,
    transform_config: TransformConfig,
    composer: Arc<Composer>,
    model_name: Arc<str>,
) -> anyhow::Result<()> {
    let state = HttpState {
        ctx,
        tok,
        eval_lock: Arc::new(Mutex::new(())),
        default_max_gen,
        interactive,
        lens_dir: lens_root.map(LensDir::new),
        ctx_transform: Arc::new(ContextTransform::new(transform_config)),
        composer,
        model_name,
    };
    let router = Router::new()
        .route("/api/version", get(handle_version))
        .route("/api/tags", get(handle_tags))
        .route("/api/ps", get(handle_ps))
        .route("/api/show", post(handle_show))
        .route("/api/generate", post(handle_generate))
        .route("/api/chat", post(handle_api_chat))
        .route("/v1/models", get(handle_v1_models))
        .route("/v1/messages", post(handle_v1_messages))
        .route("/v1/chat/completions", post(handle_v1_chat_completions))
        .route("/lens/save", post(handle_lens_save))
        .route("/lens/load", post(handle_lens_load))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    info!(bind = %bind, "rwkv-cpp HTTP shim listening");
    axum::serve(listener, router)
        .with_graceful_shutdown(async move {
            // Poll the cooperative shutdown flag on the same 200 ms cadence
            // the blocking loops use.
            while !shutdown.load(Ordering::Relaxed) {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            info!("rwkv HTTP shim: shutdown flag observed, draining");
        })
        .await?;
    Ok(())
}

async fn handle_version() -> Json<Value> {
    Json(json!({ "version": "noesis-rwkv-shim/0.1" }))
}

async fn handle_tags(State(s): State<HttpState>) -> Json<Value> {
    Json(json!({
        "models": [{
            "name": s.model_name.as_ref(),
            "model": s.model_name.as_ref(),
            "modified_at": "2026-07-23T00:00:00Z",
            "size": 0,
            "digest": "",
            "details": {
                "format": "rwkv.cpp",
                "family": "rwkv7",
                "parameter_size": "unknown",
                "quantization_level": "unknown"
            }
        }]
    }))
}

async fn handle_show(
    State(s): State<HttpState>,
    Json(_req): Json<Value>,
) -> Json<Value> {
    Json(json!({
        "modelfile": format!("# {}\n", s.model_name),
        "parameters": "",
        "template": "",
        "details": {
            "format": "rwkv.cpp",
            "family": "rwkv7",
            "parameter_size": "unknown",
            "quantization_level": "unknown"
        }
    }))
}

/// `GET /api/ps` — Ollama "running models" endpoint. Returns the single
/// resident model so Ollama-compat clients see it as already loaded.
async fn handle_ps(State(s): State<HttpState>) -> Json<Value> {
    Json(json!({
        "models": [{
            "name": s.model_name.as_ref(),
            "model": s.model_name.as_ref(),
            "size": 0,
            "digest": "",
            "details": {
                "format": "rwkv.cpp",
                "family": "rwkv7",
                "parameter_size": "unknown",
                "quantization_level": "unknown"
            },
            "expires_at": "0001-01-01T00:00:00Z",
            "size_vram": 0
        }]
    }))
}

/// `GET /v1/models` — OpenAI-format model list. Required by Open WebUI
/// and any client using the OpenAI wire format (OPENAI_BASE_URL=...).
async fn handle_v1_models(State(s): State<HttpState>) -> Json<Value> {
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Json(json!({
        "object": "list",
        "data": [{
            "id": s.model_name.as_ref(),
            "object": "model",
            "created": created,
            "owned_by": "noesis"
        }]
    }))
}

#[derive(Deserialize)]
struct GenerateRequest {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    options: Option<GenerateOptions>,
    /// Ollama also accepts `stop` as a top-level field on `/api/generate`
    /// as well as under `options`. Honour both.
    #[serde(default)]
    stop: Option<Vec<String>>,
    /// Convenience passthrough: some callers set `seed` at the top level.
    #[serde(default)]
    seed: Option<u64>,
    /// **noesis extension** (not in Ollama's schema). When present, the
    /// generation session is hydrated from `<lens_root>/<lens_id>/`
    /// before the prompt runs, and the updated state is saved back
    /// afterwards. Requires `lens_root` set at supervisor startup;
    /// otherwise the request errors with 501.
    #[serde(default)]
    lens_id: Option<String>,
}

/// Body of `/lens/save` and `/lens/load`.
#[derive(Deserialize)]
struct LensRequest {
    lens_id: String,
    /// For `/lens/save`: prompt to run on a fresh session before saving
    /// the resulting state. Required on `/lens/save`, ignored on
    /// `/lens/load`.
    #[serde(default)]
    prompt: Option<String>,
    /// Sampling seed for `/lens/save` — priming with prompt only ingests
    /// the prompt (no generation), so `seed` is unused today; kept in the
    /// schema for parity with `/api/generate`.
    #[serde(default)]
    seed: Option<u64>,
}

/// Subset of Ollama's `options` block. Fields we don't yet consume
/// (mirostat, tfs_z, typical_p, penalize_newline, num_ctx, …) are
/// intentionally left out — `#[serde(default)]` on `options` still lets
/// the request deserialise, and the surviving fields drive `SamplingParams`
/// and stop handling.
#[derive(Deserialize)]
struct GenerateOptions {
    #[serde(default)]
    num_predict: Option<i64>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_k: Option<i64>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    repeat_last_n: Option<i64>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<Vec<String>>,
}

impl GenerateOptions {
    fn sampling(&self) -> SamplingParams {
        let d = http_sampling_defaults();
        SamplingParams {
            temperature: self.temperature.unwrap_or(d.temperature).max(0.0),
            top_k: self
                .top_k
                .map(|k| if k < 0 { 0 } else { k as usize })
                .unwrap_or(d.top_k),
            top_p: self
                .top_p
                .unwrap_or(d.top_p)
                .clamp(0.0, 1.0),
            repeat_penalty: self.repeat_penalty.unwrap_or(d.repeat_penalty).max(1.0),
            repeat_last_n: self
                .repeat_last_n
                .map(|n| if n < 0 { 0 } else { n as usize })
                .unwrap_or(d.repeat_last_n),
            seed: self.seed,
        }
    }
}

/// RAII flag guard — flips `interactive` on construction, back on drop.
/// Uses `swap` so re-entrant handlers do not toggle the flag twice; the
/// guard only clears when it actually took the lift.
struct InteractiveGuard {
    flag: Arc<AtomicBool>,
    owned: bool,
}

impl InteractiveGuard {
    fn acquire(flag: Arc<AtomicBool>) -> Self {
        let owned = !flag.swap(true, Ordering::AcqRel);
        if owned {
            info!("interactive regime: entered");
        }
        Self { flag, owned }
    }
}

impl Drop for InteractiveGuard {
    fn drop(&mut self) {
        if self.owned {
            self.flag.store(false, Ordering::Release);
            info!("interactive regime: exited");
        }
    }
}

async fn handle_generate(
    State(s): State<HttpState>,
    Json(req): Json<GenerateRequest>,
) -> Response<Body> {
    // stream defaults to true (Ollama behaviour). Explicit false → buffered.
    let streaming = req.stream.unwrap_or(true);

    // Merge sampling knobs + stop-strings.
    let opts = req.options;
    let mut sampling = opts
        .as_ref()
        .map(GenerateOptions::sampling)
        .unwrap_or_default();
    if let Some(seed) = req.seed {
        sampling.seed = Some(seed);
    }
    let max_gen = opts
        .as_ref()
        .and_then(|o| o.num_predict)
        .map(|n| n.max(1) as usize)
        .unwrap_or(s.default_max_gen);
    let mut stops: Vec<String> = opts
        .as_ref()
        .and_then(|o| o.stop.clone())
        .unwrap_or_default();
    if let Some(top_stops) = req.stop {
        stops.extend(top_stops);
    }
    stops.retain(|s| !s.is_empty());

    let model_name = req
        .model
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| s.model_name.to_string());
    let prompt = req.prompt;

    // Validate lens_id before entering spawn_blocking.
    let lens_binding = match (req.lens_id.as_deref(), s.lens_dir.as_ref()) {
        (Some(id), Some(dir)) => match lens::sanitize_lens_id(id) {
            Ok(_) => Some((id.to_string(), dir.clone())),
            Err(e) => {
                return err_response(StatusCode::BAD_REQUEST, format!("invalid lens_id: {e}"));
            }
        },
        (Some(_), None) => {
            return err_response(
                StatusCode::NOT_IMPLEMENTED,
                "lens_id sent but supervisor started without lens_root".into(),
            );
        }
        (None, _) => None,
    };

    let ctx = s.ctx.clone();
    let tok = Arc::clone(&s.tok);
    let lock = Arc::clone(&s.eval_lock);
    let _regime = InteractiveGuard::acquire(Arc::clone(&s.interactive));
    let started = Instant::now();

    if streaming {
        // Channel capacity: enough to absorb a burst of tokens without
        // blocking the blocking thread when the async consumer is slow.
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(128);
        let mn = model_name.clone();

        tokio::task::spawn_blocking(move || {
            let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
            let mut session = build_session(&lens_binding, &ctx);
            let mn2 = mn.clone();
            let mut on_delta = |delta: &str| {
                let line = ndjson_line(&json!({
                    "model": &mn2,
                    "response": delta,
                    "done": false,
                }));
                let _ = tx.blocking_send(Ok(Bytes::from(line)));
            };
            let result = generate_on_session(
                &mut session, &tok, &prompt, max_gen, &sampling, &stops,
                Some(&mut on_delta),
            );
            save_lens_if_needed(&lens_binding, &session);
            let total_ns = started.elapsed().as_nanos() as u64;
            if !result.ok {
                warn!(model = %mn, "rwkv HTTP generate (stream): partial result");
            }
            let done_line = ndjson_line(&json!({
                "model": mn,
                "response": "",
                "done": true,
                "done_reason": result.stop_reason.as_ollama_str(),
                "total_duration": total_ns,
                "load_duration": 0u64,
                "prompt_eval_count": result.prompt_tokens,
                "prompt_eval_duration": result.prompt_ms * 1_000_000,
                "eval_count": result.gen_tokens,
                "eval_duration": result.gen_ms * 1_000_000,
            }));
            let _ = tx.blocking_send(Ok(Bytes::from(done_line)));
            // tx drop → ReceiverStream ends
        });

        let stream = ReceiverStream::new(rx);
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/x-ndjson")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        // Buffered path.
        let result = tokio::task::spawn_blocking(move || {
            let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
            let mut session = build_session(&lens_binding, &ctx);
            let r = generate_on_session(
                &mut session, &tok, &prompt, max_gen, &sampling, &stops, None,
            );
            save_lens_if_needed(&lens_binding, &session);
            r
        })
        .await;

        match result {
            Ok(result) => {
                let total_ns = started.elapsed().as_nanos() as u64;
                if !result.ok {
                    warn!(model = %model_name, "rwkv HTTP generate: partial result");
                }
                Json(json!({
                    "model": model_name,
                    "response": result.response,
                    "done": true,
                    "done_reason": result.stop_reason.as_ollama_str(),
                    "total_duration": total_ns,
                    "load_duration": 0u64,
                    "prompt_eval_count": result.prompt_tokens,
                    "prompt_eval_duration": result.prompt_ms * 1_000_000,
                    "eval_count": result.gen_tokens,
                    "eval_duration": result.gen_ms * 1_000_000,
                }))
                .into_response()
            }
            Err(e) => err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("join error: {e}"),
            ),
        }
    }
}

/// Hydrate a session from a lens snapshot if `lens_binding` is `Some` and
/// the snapshot exists; otherwise return a fresh session.
fn build_session(
    lens_binding: &Option<(String, LensDir)>,
    ctx: &RwkvContext,
) -> RwkvSession {
    match lens_binding {
        Some((id, dir)) if dir.exists(id) => match lens::load_session(dir, id, ctx.clone()) {
            Ok(sess) => sess,
            Err(e) => {
                warn!(lens_id = %id, error = %e,
                      "lens hydrate failed — falling back to fresh session");
                RwkvSession::new(ctx.clone())
            }
        },
        _ => RwkvSession::new(ctx.clone()),
    }
}

/// Save the session back to its lens directory if `lens_binding` is `Some`.
fn save_lens_if_needed(lens_binding: &Option<(String, LensDir)>, session: &RwkvSession) {
    if let Some((id, dir)) = lens_binding {
        if let Err(e) = lens::save_session(dir, id, session) {
            warn!(lens_id = %id, error = %e, "lens save-back failed");
        }
    }
}

// ── Ollama /api/chat ──────────────────────────────────────────────────────

/// `POST /api/chat` — Ollama chat endpoint. Open WebUI uses this when the
/// connection type is "Ollama External". Takes a `messages` array (same shape
/// as OpenAI), renders to ChatML prompt via the context transform, runs
/// generation, returns Ollama chat streaming NDJSON or buffered JSON.
async fn handle_api_chat(
    State(s): State<HttpState>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Response<Body> {
    let streaming = req.stream.unwrap_or(true);
    let model_name = req
        .model
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| s.model_name.to_string());

    let turns: Vec<ChatTurn> = req.messages.iter()
        .map(|m| ChatTurn {
            role: m.role.clone(),
            content: m.content.clone().unwrap_or_default(),
        })
        .collect();
    let query = turns.iter().rfind(|t| t.role == "user").map(|t| t.content.as_str()).unwrap_or("");
    let (preamble, snippet) = s.composer.compose(query, s.ctx_transform.config.retrieval_bytes);
    let mut transform_cfg = s.ctx_transform.config.clone();
    transform_cfg.system_preamble = preamble;
    let transform = ContextTransform::new(transform_cfg);
    let slot = if snippet.is_empty() { RetrievalSlot::Empty } else { RetrievalSlot::Snippet(&snippet) };
    let prompt = transform.build_prompt(&turns, slot);

    let mut sampling = http_sampling_defaults();
    if let Some(t) = req.temperature { sampling.temperature = t.max(0.0); }
    if let Some(p) = req.top_p       { sampling.top_p = p.clamp(0.0, 1.0); }
    // Interactive chat needs a larger default than the ambient heartbeat default (20).
    let max_gen = req.max_tokens.unwrap_or(512);
    let mut stops: Vec<String> = match req.stop {
        Some(serde_json::Value::String(s)) => vec![s],
        Some(serde_json::Value::Array(arr)) => arr.into_iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string())).collect(),
        _ => vec![],
    };
    for &stop in MESSAGES_DEFAULT_STOPS {
        if !stops.iter().any(|x| x == stop) { stops.push(stop.to_string()); }
    }

    let created_at = chrono_now_str();
    let ctx = s.ctx.clone();
    let tok = Arc::clone(&s.tok);
    let lock = Arc::clone(&s.eval_lock);
    let _regime = InteractiveGuard::acquire(Arc::clone(&s.interactive));
    let started = Instant::now();

    if streaming {
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(128);
        let mn = model_name.clone();
        let cat = created_at.clone();

        tokio::task::spawn_blocking(move || {
            let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
            let mut session = RwkvSession::new(ctx.clone());

            let mut on_delta = |delta: &str| {
                let line = ndjson_line(&json!({
                    "model": &mn,
                    "created_at": &cat,
                    "message": {"role": "assistant", "content": delta},
                    "done": false,
                }));
                let _ = tx.blocking_send(Ok(Bytes::from(line)));
            };
            let result = generate_on_session(
                &mut session, &tok, &prompt, max_gen, &sampling, &stops,
                Some(&mut on_delta),
            );
            let total_ns = started.elapsed().as_nanos() as u64;
            let done_reason = match result.stop_reason {
                super::StopReason::Length => "length",
                _ => "stop",
            };
            let done_line = ndjson_line(&json!({
                "model": mn,
                "created_at": cat,
                "message": {"role": "assistant", "content": ""},
                "done": true,
                "done_reason": done_reason,
                "total_duration": total_ns,
                "eval_count": result.gen_tokens,
            }));
            let _ = tx.blocking_send(Ok(Bytes::from(done_line)));
        });

        let stream = ReceiverStream::new(rx);
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "application/x-ndjson")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        let result = tokio::task::spawn_blocking(move || {
            let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
            let mut session = RwkvSession::new(ctx.clone());
            generate_on_session(&mut session, &tok, &prompt, max_gen, &sampling, &stops, None)
        }).await;

        match result {
            Ok(result) => {
                let total_ns = started.elapsed().as_nanos() as u64;
                let done_reason = match result.stop_reason {
                    super::StopReason::Length => "length",
                    _ => "stop",
                };
                Json(json!({
                    "model": model_name,
                    "created_at": created_at,
                    "message": {"role": "assistant", "content": result.response},
                    "done": true,
                    "done_reason": done_reason,
                    "total_duration": total_ns,
                    "eval_count": result.gen_tokens,
                })).into_response()
            }
            Err(e) => err_response(StatusCode::INTERNAL_SERVER_ERROR, format!("join error: {e}")),
        }
    }
}

fn chrono_now_str() -> String {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{secs}")
}

/// Serialize `v` as a single NDJSON line (JSON + `\n`).
fn ndjson_line(v: &Value) -> Vec<u8> {
    let mut buf = serde_json::to_vec(v).unwrap_or_default();
    buf.push(b'\n');
    buf
}

/// Build a plain-text error response with `status`.
fn err_response(status: StatusCode, msg: String) -> Response<Body> {
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(Body::from(msg))
        .unwrap()
}

/// Guard both `/lens/*` handlers. Returns 501 if the supervisor was
/// started without `lens_root`, 400 if the id is bad.
fn require_lens_dir<'a>(
    s: &'a HttpState,
    lens_id: &str,
) -> Result<&'a LensDir, (StatusCode, String)> {
    let dir = s.lens_dir.as_ref().ok_or((
        StatusCode::NOT_IMPLEMENTED,
        "lens_root not configured on this supervisor".into(),
    ))?;
    lens::sanitize_lens_id(lens_id)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid lens_id: {e}")))?;
    Ok(dir)
}

/// `POST /lens/save` — build state under `<lens_root>/<lens_id>/` from a
/// prompt. Runs `eval_sequence(prompt)` on a fresh session (or on a
/// hydrated session if the lens already exists), then writes
/// `wkv.snapshot` + `meta.json`. Returns the updated meta.
async fn handle_lens_save(
    State(s): State<HttpState>,
    Json(req): Json<LensRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let _seed = req.seed;
    let dir = require_lens_dir(&s, &req.lens_id)?.clone();
    let prompt = req.prompt.ok_or((
        StatusCode::BAD_REQUEST,
        "prompt required on /lens/save".into(),
    ))?;
    let lens_id = req.lens_id.clone();

    let ctx = s.ctx.clone();
    let tok = Arc::clone(&s.tok);
    let lock = Arc::clone(&s.eval_lock);
    // Interactive-regime guard: /lens/save is user-visible, not ambient.
    let _regime = InteractiveGuard::acquire(Arc::clone(&s.interactive));

    let started = Instant::now();
    let (meta, prompt_tokens, prompt_ms) = tokio::task::spawn_blocking(move || {
        let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut session = if dir.exists(&lens_id) {
            match lens::load_session(&dir, &lens_id, ctx.clone()) {
                Ok(sess) => sess,
                Err(e) => {
                    warn!(lens_id = %lens_id, error = %e,
                          "lens hydrate on save failed — starting fresh");
                    RwkvSession::new(ctx.clone())
                }
            }
        } else {
            RwkvSession::new(ctx.clone())
        };
        let ids = tok.encode(&prompt);
        let t = Instant::now();
        // eval_sequence chunks the prompt through rwkv.cpp's fast path.
        let ok = session.eval_sequence(&ids).is_ok();
        let elapsed = t.elapsed().as_millis() as u64;
        if !ok {
            return Err(anyhow::anyhow!("eval_sequence failed while priming lens"));
        }
        let meta = lens::save_session(&dir, &lens_id, &session)?;
        Ok((meta, ids.len(), elapsed))
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join error: {e}")))?
    .map_err(|e: anyhow::Error| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let total_ns = started.elapsed().as_nanos() as u64;
    info!(lens_id = %meta.lens_id, save_count = meta.save_count,
          prompt_tokens, "lens saved");
    Ok(Json(json!({
        "lens_id": meta.lens_id,
        "created_at": meta.created_at,
        "last_saved": meta.last_saved,
        "state_len": meta.state_len,
        "state_bytes_on_disk": meta.state_bytes_on_disk,
        "save_count": meta.save_count,
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": prompt_ms * 1_000_000,
        "total_duration": total_ns,
    })))
}

// ── Anthropic /v1/messages ────────────────────────────────────────────────

/// One turn in an Anthropic Messages request.
#[derive(Deserialize)]
struct AnthropicMessage {
    role: String,
    /// Content may be a plain string or an array of content blocks.
    /// We accept both; only the text portions are extracted.
    content: AnthropicContent,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicBlock>),
}

#[derive(Deserialize)]
struct AnthropicBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
}

impl AnthropicContent {
    fn as_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .filter(|b| b.kind == "text")
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// Anthropic `POST /v1/messages` request body (minimal subset).
#[derive(Deserialize)]
struct MessagesRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<AnthropicMessage>,
    /// Optional top-level system prompt. Anthropic SDK sends this as either
    /// a plain string or an array of content blocks — accept both.
    #[serde(default)]
    system: Option<AnthropicContent>,
    max_tokens: usize,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<i64>,
    #[serde(default)]
    stop_sequences: Option<Vec<String>>,
}

/// Default stop sequences for `/v1/messages` — prevent the model from
/// continuing past the first assistant turn in ChatML format.
const MESSAGES_DEFAULT_STOPS: &[&str] = &[
    "<|im_end|>",
    "<|endoftext|>",
    "\n<|im_start|>",
];

/// Convert `AnthropicMessage` slice → `ChatTurn` slice for the context transform.
fn anthropic_to_turns(system: Option<&str>, messages: &[AnthropicMessage]) -> Vec<ChatTurn> {
    let mut turns: Vec<ChatTurn> = Vec::new();
    if let Some(s) = system {
        if !s.is_empty() {
            turns.push(ChatTurn { role: "system".into(), content: s.to_string() });
        }
    }
    for m in messages {
        let text = m.content.as_text();
        if text.is_empty() { continue; }
        turns.push(ChatTurn { role: m.role.clone(), content: text });
    }
    turns
}

/// Format one SSE event as `event: <name>\ndata: <json>\n\n`.
fn sse_event(name: &str, data: &Value) -> Bytes {
    let mut buf = format!("event: {name}\ndata: ");
    buf.push_str(&serde_json::to_string(data).unwrap_or_default());
    buf.push_str("\n\n");
    Bytes::from(buf)
}

/// `POST /v1/messages` — Anthropic Messages API adapter.
///
/// Enables `ANTHROPIC_BASE_URL=http://127.0.0.1:11435` for claude-cli and
/// any other Anthropic-SDK client. Streaming (`stream: true` or omitted,
/// matching the SDK default) returns `text/event-stream` SSE; `stream: false`
/// returns the single-object Anthropic response JSON.
///
/// Context transform (plan §10) is not yet applied — the messages array is
/// rendered to a prompt directly. When noesis-composer lands, the transform
/// replaces this render step.
async fn handle_v1_messages(
    State(s): State<HttpState>,
    Json(req): Json<MessagesRequest>,
) -> Response<Body> {
    // Anthropic SDK defaults stream=true; omitting it means streaming.
    let streaming = req.stream.unwrap_or(true);
    let model_name = req
        .model
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| s.model_name.to_string());

    let system_text = req.system.as_ref().map(|c| c.as_text());
    let turns = anthropic_to_turns(system_text.as_deref(), &req.messages);
    let query = turns.iter().rfind(|t| t.role == "user").map(|t| t.content.as_str()).unwrap_or("");
    let (preamble, snippet) = s.composer.compose(query, s.ctx_transform.config.retrieval_bytes);
    let mut transform_cfg = s.ctx_transform.config.clone();
    transform_cfg.system_preamble = preamble;
    let transform = ContextTransform::new(transform_cfg);
    let slot = if snippet.is_empty() { RetrievalSlot::Empty } else { RetrievalSlot::Snippet(&snippet) };
    let prompt = transform.build_prompt(&turns, slot);
    let mut sampling = http_sampling_defaults();
    if let Some(t) = req.temperature {
        sampling.temperature = t.max(0.0);
    }
    if let Some(p) = req.top_p {
        sampling.top_p = p.clamp(0.0, 1.0);
    }
    if let Some(k) = req.top_k {
        sampling.top_k = if k < 0 { 0 } else { k as usize };
    }
    let max_gen = req.max_tokens;
    let mut stops: Vec<String> = req.stop_sequences.unwrap_or_default();
    for &s in MESSAGES_DEFAULT_STOPS {
        if !stops.iter().any(|x| x == s) {
            stops.push(s.to_string());
        }
    }
    let msg_id = format!("msg_{:016x}", fastrand_u64());

    let ctx = s.ctx.clone();
    let tok = Arc::clone(&s.tok);
    let lock = Arc::clone(&s.eval_lock);
    let _regime = InteractiveGuard::acquire(Arc::clone(&s.interactive));
    let started = Instant::now();

    if streaming {
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(128);
        let mn = model_name.clone();
        let mid = msg_id.clone();

        tokio::task::spawn_blocking(move || {
            let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
            let mut session = RwkvSession::new(ctx.clone());

            // message_start
            let _ = tx.blocking_send(Ok(sse_event("message_start", &json!({
                "type": "message_start",
                "message": {
                    "id": &mid,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": &mn,
                    "stop_reason": null,
                    "stop_sequence": null,
                    "usage": { "input_tokens": 0, "output_tokens": 0 }
                }
            }))));
            // content_block_start
            let _ = tx.blocking_send(Ok(sse_event("content_block_start", &json!({
                "type": "content_block_start",
                "index": 0,
                "content_block": { "type": "text", "text": "" }
            }))));

            let mut on_delta = |delta: &str| {
                let _ = tx.blocking_send(Ok(sse_event("content_block_delta", &json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": { "type": "text_delta", "text": delta }
                }))));
            };
            let result = generate_on_session(
                &mut session, &tok, &prompt, max_gen, &sampling, &stops,
                Some(&mut on_delta),
            );

            let stop_reason = match result.stop_reason {
                super::StopReason::Length => "max_tokens",
                super::StopReason::StopSequence => "stop_sequence",
                super::StopReason::Error => "end_turn",
            };

            // content_block_stop
            let _ = tx.blocking_send(Ok(sse_event("content_block_stop", &json!({
                "type": "content_block_stop",
                "index": 0
            }))));
            // message_delta with usage
            let _ = tx.blocking_send(Ok(sse_event("message_delta", &json!({
                "type": "message_delta",
                "delta": { "stop_reason": stop_reason, "stop_sequence": null },
                "usage": { "output_tokens": result.gen_tokens }
            }))));
            // message_stop
            let _ = tx.blocking_send(Ok(sse_event("message_stop", &json!({
                "type": "message_stop"
            }))));

            if !result.ok {
                warn!(model = %mn, "v1/messages (stream): partial result");
            }
            let _ = started.elapsed(); // suppress unused warning
        });

        let stream = ReceiverStream::new(rx);
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
            .header("cache-control", "no-cache")
            .header("x-accel-buffering", "no")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        let result = tokio::task::spawn_blocking(move || {
            let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
            let mut session = RwkvSession::new(ctx.clone());
            generate_on_session(&mut session, &tok, &prompt, max_gen, &sampling, &stops, None)
        })
        .await;

        match result {
            Ok(result) => {
                let stop_reason = match result.stop_reason {
                    super::StopReason::Length => "max_tokens",
                    super::StopReason::StopSequence => "stop_sequence",
                    super::StopReason::Error => "end_turn",
                };
                Json(json!({
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [{ "type": "text", "text": result.response }],
                    "model": model_name,
                    "stop_reason": stop_reason,
                    "stop_sequence": null,
                    "usage": {
                        "input_tokens": result.prompt_tokens,
                        "output_tokens": result.gen_tokens
                    }
                }))
                .into_response()
            }
            Err(e) => err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("join error: {e}"),
            ),
        }
    }
}

// ── OpenAI /v1/chat/completions ────────────────────────────────────────────

/// One turn in an OpenAI chat-completions request.
#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    /// May be string or null (tool-call roles); we coerce null → "".
    #[serde(default)]
    content: Option<String>,
}

/// `POST /v1/chat/completions` request body (minimal subset).
#[derive(Deserialize)]
struct ChatCompletionsRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stop: Option<serde_json::Value>,
}

/// `POST /v1/chat/completions` — OpenAI chat-completions adapter.
///
/// Enables `OPENAI_BASE_URL=http://127.0.0.1:11435` for tools that speak
/// the OpenAI wire format. Accepts `stream: true` (SSE) or `stream: false`
/// (buffered). The same ChatML prompt render as `/v1/messages` is used.
///
/// Note: the `system` prompt in OpenAI format lives as a message with
/// `role: "system"` inside `messages`, not as a top-level field.
async fn handle_v1_chat_completions(
    State(s): State<HttpState>,
    Json(req): Json<ChatCompletionsRequest>,
) -> Response<Body> {
    let streaming = req.stream.unwrap_or(false);
    let model_name = req
        .model
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| s.model_name.to_string());

    // Separate system message from the rest; convert to AnthropicMessage for
    // reuse of messages_to_prompt (same ChatML format).
    let turns: Vec<ChatTurn> = req.messages.iter()
        .map(|m| ChatTurn {
            role: m.role.clone(),
            content: m.content.clone().unwrap_or_default(),
        })
        .collect();
    let query = turns.iter().rfind(|t| t.role == "user").map(|t| t.content.as_str()).unwrap_or("");
    let (preamble, snippet) = s.composer.compose(query, s.ctx_transform.config.retrieval_bytes);
    let mut transform_cfg = s.ctx_transform.config.clone();
    transform_cfg.system_preamble = preamble;
    let transform = ContextTransform::new(transform_cfg);
    let slot = if snippet.is_empty() { RetrievalSlot::Empty } else { RetrievalSlot::Snippet(&snippet) };
    let prompt = transform.build_prompt(&turns, slot);
    let mut sampling = http_sampling_defaults();
    if let Some(t) = req.temperature {
        sampling.temperature = t.max(0.0);
    }
    if let Some(p) = req.top_p {
        sampling.top_p = p.clamp(0.0, 1.0);
    }
    let max_gen = req.max_tokens.unwrap_or(512);

    // `stop` may be a string or an array of strings.
    let mut stops: Vec<String> = match req.stop {
        Some(serde_json::Value::String(s)) => vec![s],
        Some(serde_json::Value::Array(arr)) => arr
            .into_iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect(),
        _ => vec![],
    };
    for &s in MESSAGES_DEFAULT_STOPS {
        if !stops.iter().any(|x| x == s) {
            stops.push(s.to_string());
        }
    }

    let completion_id = format!("chatcmpl-{:016x}", fastrand_u64());
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let ctx = s.ctx.clone();
    let tok = Arc::clone(&s.tok);
    let lock = Arc::clone(&s.eval_lock);
    let _regime = InteractiveGuard::acquire(Arc::clone(&s.interactive));

    if streaming {
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(128);
        let mn = model_name.clone();
        let cid = completion_id.clone();

        tokio::task::spawn_blocking(move || {
            let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
            let mut session = RwkvSession::new(ctx.clone());

            // OpenAI SSE: each chunk is `data: {…}\n\n`, terminated by `data: [DONE]\n\n`.
            let make_chunk = |delta_obj: serde_json::Value, finish_reason: Option<&str>| -> Bytes {
                let chunk = json!({
                    "id": &cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": &mn,
                    "choices": [{
                        "index": 0,
                        "delta": delta_obj,
                        "finish_reason": finish_reason,
                    }]
                });
                Bytes::from(format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap_or_default()))
            };

            // First chunk carries the role; subsequent carry only content.
            let _ = tx.blocking_send(Ok(make_chunk(json!({"role": "assistant", "content": ""}), None)));

            let mut on_delta = |delta: &str| {
                let _ = tx.blocking_send(Ok(make_chunk(json!({"content": delta}), None)));
            };
            let result = generate_on_session(
                &mut session, &tok, &prompt, max_gen, &sampling, &stops,
                Some(&mut on_delta),
            );

            let finish_reason = match result.stop_reason {
                super::StopReason::Length => "length",
                super::StopReason::StopSequence => "stop",
                super::StopReason::Error => "stop",
            };
            let _ = tx.blocking_send(Ok(make_chunk(json!({}), Some(finish_reason))));
            let _ = tx.blocking_send(Ok(Bytes::from("data: [DONE]\n\n")));

            if !result.ok {
                warn!(model = %mn, "v1/chat/completions (stream): partial result");
            }
        });

        let stream = ReceiverStream::new(rx);
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
            .header("cache-control", "no-cache")
            .header("x-accel-buffering", "no")
            .body(Body::from_stream(stream))
            .unwrap()
    } else {
        let result = tokio::task::spawn_blocking(move || {
            let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
            let mut session = RwkvSession::new(ctx.clone());
            generate_on_session(&mut session, &tok, &prompt, max_gen, &sampling, &stops, None)
        })
        .await;

        match result {
            Ok(result) => {
                let finish_reason = match result.stop_reason {
                    super::StopReason::Length => "length",
                    super::StopReason::StopSequence => "stop",
                    super::StopReason::Error => "stop",
                };
                Json(json!({
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.response,
                        },
                        "finish_reason": finish_reason,
                    }],
                    "usage": {
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.gen_tokens,
                        "total_tokens": result.prompt_tokens + result.gen_tokens,
                    }
                }))
                .into_response()
            }
            Err(e) => err_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("join error: {e}"),
            ),
        }
    }
}

/// Simple non-cryptographic u64 for message IDs — timestamp nanos XOR pid.
fn fastrand_u64() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xDEAD_BEEF_CAFE_1234);
    let pid = std::process::id() as u64;
    nanos ^ pid.wrapping_mul(0x9e3779b97f4a7c15)
}

// ── Lens endpoints ─────────────────────────────────────────────────────────

/// `POST /lens/load` — metadata read, no session-side effect. A client
/// uses this to check "does this lens exist?" and to see how large its
/// state is before deciding whether to hydrate. The actual hydration
/// happens inside `/api/generate` when `lens_id` is set.
async fn handle_lens_load(
    State(s): State<HttpState>,
    Json(req): Json<LensRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let dir = require_lens_dir(&s, &req.lens_id)?;
    let meta = lens::read_meta(dir, &req.lens_id)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    match meta {
        Some(m) => Ok(Json(json!({
            "lens_id": m.lens_id,
            "created_at": m.created_at,
            "last_saved": m.last_saved,
            "state_len": m.state_len,
            "state_bytes_on_disk": m.state_bytes_on_disk,
            "save_count": m.save_count,
            "exists": true,
        }))),
        None => Err((
            StatusCode::NOT_FOUND,
            format!("lens {} not found", req.lens_id),
        )),
    }
}
