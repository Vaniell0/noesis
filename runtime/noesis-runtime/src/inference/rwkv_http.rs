//! Ollama-shape HTTP shim on top of the in-process rwkv.cpp context.
//!
//! Endpoints (subset of Ollama's REST surface):
//!
//! - `GET /api/version` — trivial identity string.
//! - `GET /api/tags`    — one-model catalogue.
//! - `POST /api/show`   — model metadata blob.
//! - `POST /api/generate` — non-streaming completion. Streaming
//!   (`stream: true`) returns HTTP 501 for the MVP; it's a follow-up
//!   once we need SSE/NDJSON on the client side.
//!
//! Concurrency: rwkv.cpp forbids concurrent `rwkv_eval` on the same
//! context, so the shim serialises HTTP requests through a
//! `std::sync::Mutex<()>` acquired inside `spawn_blocking`. The
//! heartbeat runs on a *different* cloned context (see
//! `RwkvContext::clone_for_parallel` in the caller), so heartbeat and
//! HTTP never contend.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use noesis_rwkv::{tokenizer::WorldTokenizer, RwkvContext};
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{info, warn};

use super::generate_once;

/// Identifier the shim advertises via `/api/tags` and echoes on
/// `/api/generate` responses. Not the file path — this is the
/// user-facing model name a CLI would type.
const MODEL_NAME: &str = "noesis-rwkv7-0.4b";

#[derive(Clone)]
struct HttpState {
    ctx: RwkvContext,
    tok: Arc<WorldTokenizer>,
    eval_lock: Arc<Mutex<()>>,
    default_max_gen: usize,
}

pub(super) async fn serve(
    ctx: RwkvContext,
    tok: Arc<WorldTokenizer>,
    bind: SocketAddr,
    default_max_gen: usize,
    shutdown: Arc<AtomicBool>,
) -> anyhow::Result<()> {
    let state = HttpState {
        ctx,
        tok,
        eval_lock: Arc::new(Mutex::new(())),
        default_max_gen,
    };
    let router = Router::new()
        .route("/api/version", get(handle_version))
        .route("/api/tags", get(handle_tags))
        .route("/api/show", post(handle_show))
        .route("/api/generate", post(handle_generate))
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

async fn handle_tags(State(_s): State<HttpState>) -> Json<Value> {
    Json(json!({
        "models": [{
            "name": MODEL_NAME,
            "modified_at": "2026-07-23T00:00:00Z",
            "size": 0,
            "digest": "",
            "details": {
                "format": "rwkv.cpp",
                "family": "rwkv7",
                "parameter_size": "0.4B",
                "quantization_level": "FP16"
            }
        }]
    }))
}

async fn handle_show(
    State(_s): State<HttpState>,
    Json(_req): Json<Value>,
) -> Json<Value> {
    Json(json!({
        "modelfile": format!("# {}\n", MODEL_NAME),
        "parameters": "",
        "template": "",
        "details": {
            "format": "rwkv.cpp",
            "family": "rwkv7",
            "parameter_size": "0.4B",
            "quantization_level": "FP16"
        }
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
}

#[derive(Deserialize)]
struct GenerateOptions {
    #[serde(default)]
    num_predict: Option<i64>,
}

async fn handle_generate(
    State(s): State<HttpState>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    if req.stream.unwrap_or(false) {
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            "streaming responses are not yet supported by the noesis shim".into(),
        ));
    }
    let max_gen = req
        .options
        .and_then(|o| o.num_predict)
        .map(|n| n.max(1) as usize)
        .unwrap_or(s.default_max_gen);
    let model_name = req
        .model
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| MODEL_NAME.to_string());
    let prompt = req.prompt;
    let ctx = s.ctx.clone();
    let tok = Arc::clone(&s.tok);
    let lock = Arc::clone(&s.eval_lock);
    let started = Instant::now();
    let result = tokio::task::spawn_blocking(move || {
        // Poison-tolerant: we don't store any state behind the mutex,
        // it only serialises rwkv_eval calls on the shared context.
        let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        generate_once(&ctx, &tok, &prompt, max_gen)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join error: {e}")))?;
    let total_ns = started.elapsed().as_nanos() as u64;
    if !result.ok {
        warn!(model = %model_name, "rwkv HTTP generate: partial result");
    }
    Ok(Json(json!({
        "model": model_name,
        "response": result.response,
        "done": true,
        "done_reason": "stop",
        "total_duration": total_ns,
        "load_duration": 0u64,
        "prompt_eval_count": result.prompt_tokens,
        "prompt_eval_duration": result.prompt_ms * 1_000_000,
        "eval_count": result.gen_tokens,
        "eval_duration": result.gen_ms * 1_000_000,
    })))
}
