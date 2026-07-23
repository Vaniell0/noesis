//! Safe wrapper over rwkv.cpp.
//!
//! The interesting surface for noesis is small:
//!
//! - [`RwkvContext`] — loaded model; expensive; shared by clone across
//!   sessions (rwkv.cpp is thread-safe *between* eval calls, not
//!   *during*, so a per-session context via `rwkv_clone_context` is the
//!   right tool for parallel sessions).
//! - [`RwkvSession`] — owns the WKV state buffer. `eval` mutates it in
//!   place. This is the object noesis will snapshot when a lens is
//!   suspended (H11) and restore when it resumes.
//!
//! Not implemented here (yet):
//!
//! - sampling — greedy / top-p / temperature. Small and orthogonal;
//!   goes in a sibling module once we need it. For skeleton wiring the
//!   caller can implement its own argmax on the returned logits slice.
//!
//! Tokenizer wrapper (WORLD) lives in the [`tokenizer`] module and
//! delegates to the `rwkv-tokenizer` crate.
//!
//! Design notes:
//!
//! - Every fallible call reads `rwkv_get_last_error(ctx)` on error. The
//!   flag encodes both category (upper byte) and subcategory (lower
//!   byte), which we surface verbatim so caller can inspect either.
//! - `RwkvContext` is `Send + Sync`: upstream documents rwkv_context as
//!   safe to move between threads and safe to *share* as long as callers
//!   don't concurrently invoke `rwkv_eval` on the same context. The
//!   type system can't enforce that; the callers we care about hold one
//!   session per zone/lens.

use std::ffi::CString;
use std::path::Path;
use std::sync::Arc;

use noesis_rwkv_sys as sys;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RwkvError {
    #[error("path contains an interior NUL byte")]
    NulPath,
    #[error("rwkv_init_from_file failed (flags = 0x{0:x})")]
    InitFailed(u32),
    #[error("rwkv_eval failed (flags = 0x{0:x})")]
    EvalFailed(u32),
    #[error("rwkv_eval_sequence failed (flags = 0x{0:x})")]
    EvalSequenceFailed(u32),
    #[error("rwkv_clone_context returned NULL")]
    CloneFailed,
}

pub type Result<T> = std::result::Result<T, RwkvError>;

/// A loaded RWKV model. Cheap to clone (it just bumps an Arc); expensive
/// to create (the underlying `rwkv_init_from_file` mmaps GB of weights).
///
/// Clone this per session, then create a `RwkvSession` off each clone.
#[derive(Clone)]
pub struct RwkvContext {
    inner: Arc<RwkvContextInner>,
}

struct RwkvContextInner {
    ptr: *mut sys::rwkv_context,
}

// SAFETY: `rwkv_context` is documented as thread-safe between eval
// calls. Concurrent eval on the *same* context is UB — we don't do
// that; callers get their own via `RwkvSession::new_cloned`.
unsafe impl Send for RwkvContextInner {}
unsafe impl Sync for RwkvContextInner {}

impl Drop for RwkvContextInner {
    fn drop(&mut self) {
        // SAFETY: we own the pointer; caller has released all sessions.
        unsafe { sys::rwkv_free(self.ptr) };
    }
}

impl RwkvContext {
    /// Load a model file. `n_threads` must be positive. `n_gpu_layers`
    /// is 0 on CPU-only builds (which is what nix/rwkv-cpp.nix ships).
    pub fn open(path: &Path, n_threads: u32, n_gpu_layers: u32) -> Result<Self> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| RwkvError::NulPath)?;
        // SAFETY: c_path lives across the FFI call; the returned
        // pointer is either NULL or a valid rwkv_context.
        let ptr = unsafe {
            sys::rwkv_init_from_file(c_path.as_ptr(), n_threads, n_gpu_layers)
        };
        if ptr.is_null() {
            let flags = unsafe { sys::rwkv_get_last_error(std::ptr::null_mut()) };
            return Err(RwkvError::InitFailed(flags));
        }
        Ok(RwkvContext { inner: Arc::new(RwkvContextInner { ptr }) })
    }

    /// Number of FP32 elements in a state buffer for this model.
    pub fn state_len(&self) -> usize {
        unsafe { sys::rwkv_get_state_len(self.inner.ptr) }
    }

    /// Number of FP32 elements in a logits buffer (equals vocabulary size).
    pub fn logits_len(&self) -> usize {
        unsafe { sys::rwkv_get_logits_len(self.inner.ptr) }
    }

    pub fn n_vocab(&self) -> usize {
        unsafe { sys::rwkv_get_n_vocab(self.inner.ptr) }
    }

    pub fn n_embed(&self) -> usize {
        unsafe { sys::rwkv_get_n_embed(self.inner.ptr) }
    }

    pub fn n_layer(&self) -> usize {
        unsafe { sys::rwkv_get_n_layer(self.inner.ptr) }
    }

    /// Clone the underlying rwkv_context so a second thread can eval in
    /// parallel. `n_threads` is the thread count for the *new* context's
    /// internal ops, not for the caller.
    pub fn clone_for_parallel(&self, n_threads: u32) -> Result<Self> {
        let ptr = unsafe { sys::rwkv_clone_context(self.inner.ptr, n_threads) };
        if ptr.is_null() {
            return Err(RwkvError::CloneFailed);
        }
        Ok(RwkvContext { inner: Arc::new(RwkvContextInner { ptr }) })
    }
}

/// One conversation / lens. Owns its WKV state buffer; the buffer is
/// what H8/H9/H11 want to snapshot.
pub struct RwkvSession {
    ctx: RwkvContext,
    state: Vec<f32>,
    logits: Vec<f32>,
}

impl RwkvSession {
    /// Fresh session with a zero-initialised state (via `rwkv_init_state`
    /// — plain zero would produce NaNs per upstream note).
    pub fn new(ctx: RwkvContext) -> Self {
        let state_len = ctx.state_len();
        let logits_len = ctx.logits_len();
        let mut state = vec![0.0f32; state_len];
        unsafe { sys::rwkv_init_state(ctx.inner.ptr, state.as_mut_ptr()) };
        RwkvSession {
            ctx,
            state,
            logits: vec![0.0f32; logits_len],
        }
    }

    /// Resume a session from a previously-snapshotted state.
    /// Length must match `ctx.state_len()`.
    pub fn from_state(ctx: RwkvContext, state: Vec<f32>) -> Self {
        assert_eq!(state.len(), ctx.state_len(), "state length mismatch");
        let logits_len = ctx.logits_len();
        RwkvSession { ctx, state, logits: vec![0.0f32; logits_len] }
    }

    /// Snapshot the current WKV state. Cheap-ish: a `Vec<f32>` clone.
    /// For H11 lens snapshotting the caller stores this next to session
    /// metadata; for a hot loop the state is already in `session.state`.
    pub fn snapshot_state(&self) -> Vec<f32> {
        self.state.clone()
    }

    /// Evaluate a single token, updating internal state and logits
    /// buffers. Returns a slice view into the fresh logits.
    pub fn eval(&mut self, token: u32) -> Result<&[f32]> {
        // rwkv_eval reads `state_in`, writes `state_out`. We use the
        // same buffer for both — upstream API supports aliasing (it
        // reads the whole state before writing it).
        let state_ptr = self.state.as_mut_ptr();
        let ok = unsafe {
            sys::rwkv_eval(
                self.ctx.inner.ptr,
                token,
                state_ptr,
                state_ptr,
                self.logits.as_mut_ptr(),
            )
        };
        if !ok {
            let flags = unsafe { sys::rwkv_get_last_error(self.ctx.inner.ptr) };
            return Err(RwkvError::EvalFailed(flags));
        }
        Ok(&self.logits)
    }

    /// Evaluate a sequence of tokens using rwkv.cpp's chunked path —
    /// much faster than looping `eval` for prompt ingestion. Uses
    /// `chunk_size = 16` per upstream recommendation.
    pub fn eval_sequence(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        if tokens.is_empty() {
            return Ok(&self.logits);
        }
        let state_ptr = self.state.as_mut_ptr();
        let ok = unsafe {
            sys::rwkv_eval_sequence_in_chunks(
                self.ctx.inner.ptr,
                tokens.as_ptr(),
                tokens.len(),
                16,
                state_ptr,
                state_ptr,
                self.logits.as_mut_ptr(),
            )
        };
        if !ok {
            let flags = unsafe { sys::rwkv_get_last_error(self.ctx.inner.ptr) };
            return Err(RwkvError::EvalSequenceFailed(flags));
        }
        Ok(&self.logits)
    }

    pub fn context(&self) -> &RwkvContext {
        &self.ctx
    }

    pub fn state(&self) -> &[f32] {
        &self.state
    }
}

/// Greedy sampling — returns argmax(logits). Simplest possible policy;
/// noesis will replace with temperature/top-p once the skeleton runs.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

pub mod tokenizer {
    //! Thin wrapper over `rwkv-tokenizer`'s WORLD implementation.
    //!
    //! rwkv.cpp's C API takes `u32` tokens; the WORLD vocabulary fits
    //! in `u16` (n_vocab = 65536), which is what the upstream crate
    //! returns. We convert at the boundary and expose `u32` throughout
    //! noesis so it lines up with the FFI.
    use rwkv_tokenizer::WorldTokenizer as UpstreamTokenizer;
    use std::str::Utf8Error;

    /// WORLD tokenizer with the built-in v20230424 vocab (matches
    /// n_vocab = 65536 RWKV-5/6/7 models).
    pub struct WorldTokenizer {
        inner: UpstreamTokenizer,
    }

    impl WorldTokenizer {
        /// Load the built-in vocab (embedded in the `rwkv-tokenizer`
        /// crate — no filesystem access needed at runtime).
        pub fn new() -> std::io::Result<Self> {
            Ok(Self { inner: UpstreamTokenizer::new(None)? })
        }

        pub fn encode(&self, text: &str) -> Vec<u32> {
            self.inner.encode(text).into_iter().map(u32::from).collect()
        }

        pub fn decode(&self, tokens: &[u32]) -> Result<String, Utf8Error> {
            let narrow: Vec<u16> = tokens.iter().map(|&t| t as u16).collect();
            self.inner.decode(narrow)
        }
    }
}
