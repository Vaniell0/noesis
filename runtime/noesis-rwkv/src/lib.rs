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
use std::io::{Read, Write};
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
    #[error("state snapshot i/o: {0}")]
    SnapshotIo(#[from] std::io::Error),
    #[error("state snapshot magic mismatch (got {got:?}, expected NWKV)")]
    SnapshotMagic { got: [u8; 4] },
    #[error("state snapshot version {found} unsupported (expected {expected})")]
    SnapshotVersion { found: u32, expected: u32 },
    #[error("state snapshot length {found} does not match model state_len {expected}")]
    SnapshotLength { found: u32, expected: u32 },
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

/// Snapshot file header magic — used by `save_state` / `load_state` on
/// `RwkvSession` and by lens hydration under `/var/lib/noesis/lenses/<id>/`.
pub const SNAPSHOT_MAGIC: [u8; 4] = *b"NWKV";
/// Snapshot format version. Bumped on any layout change; older files then
/// fail loud (`RwkvError::SnapshotVersion`) instead of silently
/// corrupting a session.
pub const SNAPSHOT_VERSION: u32 = 1;

/// Write an FP32 state slice with a magic/version/length header.
/// Format (little-endian):
///   4 bytes `NWKV` · 4 bytes `SNAPSHOT_VERSION` · 4 bytes state_len (u32)
///   · state_len × 4 bytes FP32-LE payload.
///
/// The header lets the reader refuse mismatched or corrupted files before
/// they reach the model — a wrong-shape state buffer otherwise yields NaN
/// logits after the first eval, which is much harder to trace than a
/// load-time error.
pub fn write_state_snapshot<W: Write>(state: &[f32], mut writer: W) -> Result<()> {
    writer.write_all(&SNAPSHOT_MAGIC)?;
    writer.write_all(&SNAPSHOT_VERSION.to_le_bytes())?;
    let state_len_u32: u32 = state
        .len()
        .try_into()
        .expect("state_len does not fit in u32");
    writer.write_all(&state_len_u32.to_le_bytes())?;
    let mut buf = [0u8; 4];
    for &v in state {
        buf.copy_from_slice(&v.to_le_bytes());
        writer.write_all(&buf)?;
    }
    Ok(())
}

/// Read a state snapshot; verifies magic, version, and length against
/// `expected_len` before allocating.
pub fn read_state_snapshot<R: Read>(expected_len: usize, mut reader: R) -> Result<Vec<f32>> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if magic != SNAPSHOT_MAGIC {
        return Err(RwkvError::SnapshotMagic { got: magic });
    }
    let mut buf4 = [0u8; 4];
    reader.read_exact(&mut buf4)?;
    let version = u32::from_le_bytes(buf4);
    if version != SNAPSHOT_VERSION {
        return Err(RwkvError::SnapshotVersion {
            found: version,
            expected: SNAPSHOT_VERSION,
        });
    }
    reader.read_exact(&mut buf4)?;
    let stored_len = u32::from_le_bytes(buf4);
    let expected_len_u32: u32 = expected_len
        .try_into()
        .expect("state_len does not fit in u32");
    if stored_len != expected_len_u32 {
        return Err(RwkvError::SnapshotLength {
            found: stored_len,
            expected: expected_len_u32,
        });
    }
    let mut state = vec![0.0f32; stored_len as usize];
    for slot in state.iter_mut() {
        reader.read_exact(&mut buf4)?;
        *slot = f32::from_le_bytes(buf4);
    }
    Ok(state)
}

impl RwkvSession {
    /// Serialise the current WKV state — see [`write_state_snapshot`].
    pub fn save_state<W: Write>(&self, writer: W) -> Result<()> {
        write_state_snapshot(&self.state, writer)
    }

    /// Load a snapshot into a fresh session on the given `ctx`. Length is
    /// checked against `ctx.state_len()`; a mismatched checkpoint returns
    /// an error rather than producing a silently-broken session.
    pub fn load_state<R: Read>(ctx: RwkvContext, reader: R) -> Result<Self> {
        let state = read_state_snapshot(ctx.state_len(), reader)?;
        let logits = vec![0.0f32; ctx.logits_len()];
        Ok(RwkvSession { ctx, state, logits })
    }
}

/// Greedy sampling — returns argmax(logits). Used by the heartbeat loop
/// (deterministic health check) and as the temperature-0 branch of
/// [`sample`].
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

/// Ollama-compatible sampling knobs. Defaults match Ollama's own
/// defaults so a client that sends nothing gets the behaviour it
/// expects. `temperature = 0.0` short-circuits to [`argmax`].
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    /// llama.cpp-style penalty applied to the last `repeat_last_n`
    /// tokens: `logit /= penalty` if `logit > 0`, else `logit *= penalty`.
    /// `1.0` disables the penalty.
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    /// `None` = seed from wall clock; deterministic when `Some`.
    pub seed: Option<u64>,
}

impl SamplingParams {
    /// Deterministic argmax — heartbeat / calibration / anywhere we want
    /// the same next token for the same prefix.
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            seed: None,
        }
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: None,
        }
    }
}

/// SplitMix64 — a small, seedable PRNG. One line to advance, adequate
/// for token sampling. Kept in-crate to avoid dragging `rand` into the
/// runtime's dep graph.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        // Guard against a zero seed producing degenerate sequences.
        Self {
            state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed },
        }
    }

    /// Uniform u64.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // 24-bit precision (mantissa width). Right-shift by 40 for the
        // top 24 bits, divide by 2^24.
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }
}

/// Sample one token from `logits` under [`SamplingParams`], applying
/// (in order): repeat penalty against `recent_tokens`, temperature scaling,
/// top-K truncation, top-p (nucleus) truncation, then a categorical draw
/// from the surviving softmax.
///
/// `recent_tokens` is the tail of the generated sequence used for the
/// repeat penalty; anything older than `params.repeat_last_n` is ignored
/// by the caller before passing in, but the function also caps internally
/// for safety.
pub fn sample(logits: &[f32], recent_tokens: &[u32], params: &SamplingParams, rng: &mut SplitMix64) -> u32 {
    if params.temperature <= 0.0 {
        return argmax(logits);
    }

    let mut scored: Vec<f32> = logits.to_vec();

    // Repeat penalty.
    if params.repeat_penalty > 1.0 && !recent_tokens.is_empty() && params.repeat_last_n > 0 {
        let start = recent_tokens
            .len()
            .saturating_sub(params.repeat_last_n);
        for &tok in &recent_tokens[start..] {
            let idx = tok as usize;
            if idx < scored.len() {
                let v = scored[idx];
                scored[idx] = if v > 0.0 { v / params.repeat_penalty } else { v * params.repeat_penalty };
            }
        }
    }

    // Temperature.
    if (params.temperature - 1.0).abs() > f32::EPSILON {
        let inv = 1.0 / params.temperature;
        for v in scored.iter_mut() {
            *v *= inv;
        }
    }

    // Sort (index, logit) desc by logit for top-K / top-p.
    let mut idx: Vec<usize> = (0..scored.len()).collect();
    idx.sort_unstable_by(|&a, &b| scored[b].partial_cmp(&scored[a]).unwrap_or(std::cmp::Ordering::Equal));

    // Top-K.
    let k_limit = if params.top_k == 0 { idx.len() } else { params.top_k.min(idx.len()) };
    idx.truncate(k_limit);

    // Softmax over survivors (numerically-stable subtract-max).
    let max_l = scored[idx[0]];
    let mut probs: Vec<f32> = idx.iter().map(|&i| (scored[i] - max_l).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return idx[0] as u32;
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // Top-p (nucleus): keep prefix whose cumulative prob >= top_p.
    if params.top_p < 1.0 && params.top_p > 0.0 {
        let mut cum = 0.0f32;
        let mut cutoff = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= params.top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
        idx.truncate(cutoff);
        let s: f32 = probs.iter().sum();
        if s > 0.0 {
            for p in probs.iter_mut() {
                *p /= s;
            }
        }
    }

    // Categorical draw.
    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc {
            return idx[i] as u32;
        }
    }
    // Fell through due to float rounding — return the top choice.
    idx[0] as u32
}

pub mod tokenizer {
    //! Thin wrapper over `rwkv-tokenizer`'s WORLD implementation.
    //!
    //! rwkv.cpp's C API takes `u32` tokens; the WORLD vocabulary fits
    //! in `u16` (n_vocab = 65536), which is what the upstream crate
    //! returns. We convert at the boundary and expose `u32` throughout
    //! noesis so it lines up with the FFI.
    use rwkv_tokenizer::WorldTokenizer as UpstreamTokenizer;
    use std::io::{self, Write};
    use std::path::PathBuf;
    use std::str::Utf8Error;
    use std::sync::OnceLock;

    /// v20230424 WORLD vocab, embedded at compile time. rwkv-tokenizer 0.9.1
    /// only knows how to read the vocab from disk (its `Option<&str>` path
    /// argument), so we materialise the embedded copy into a stable cache
    /// file once per process and hand the path to the upstream constructor.
    const EMBEDDED_VOCAB: &str = include_str!("../assets/rwkv_vocab_v20230424.txt");

    /// WORLD tokenizer with the built-in v20230424 vocab (matches
    /// n_vocab = 65536 RWKV-5/6/7 models).
    pub struct WorldTokenizer {
        inner: UpstreamTokenizer,
    }

    fn ensure_vocab_on_disk() -> io::Result<PathBuf> {
        static PATH: OnceLock<PathBuf> = OnceLock::new();
        if let Some(p) = PATH.get() {
            return Ok(p.clone());
        }
        let dir = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .unwrap_or_else(std::env::temp_dir)
            .join("noesis");
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("rwkv_vocab_v20230424.txt");
        if !path.exists() || std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0) != EMBEDDED_VOCAB.len() as u64 {
            let mut f = std::fs::File::create(&path)?;
            f.write_all(EMBEDDED_VOCAB.as_bytes())?;
            f.sync_all()?;
        }
        let _ = PATH.set(path.clone());
        Ok(path)
    }

    impl WorldTokenizer {
        /// Materialise the embedded vocab into `$XDG_CACHE_HOME/noesis/`
        /// (once per process) and hand the path to the upstream tokenizer.
        pub fn new() -> std::io::Result<Self> {
            let path = ensure_vocab_on_disk()?;
            let path_str = path.to_str().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "vocab cache path is not valid UTF-8")
            })?;
            Ok(Self { inner: UpstreamTokenizer::new(Some(path_str))? })
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

#[cfg(test)]
mod tests {
    //! Snapshot format is testable without loading a real model — the
    //! session-facing wrappers just delegate to `write_state_snapshot` /
    //! `read_state_snapshot`. The full "same next-token after save/load"
    //! determinism check lives in `examples/state_roundtrip.rs`, which
    //! needs actual weights.
    use super::*;

    #[test]
    fn snapshot_roundtrips_bitexact() {
        let state: Vec<f32> = (0..1024).map(|i| (i as f32).sin() * 3.5 - 1.25).collect();
        let mut buf = Vec::<u8>::new();
        write_state_snapshot(&state, &mut buf).unwrap();
        // 12-byte header + payload.
        assert_eq!(buf.len(), 12 + state.len() * 4);
        assert_eq!(&buf[0..4], &SNAPSHOT_MAGIC);
        let loaded = read_state_snapshot(state.len(), buf.as_slice()).unwrap();
        assert_eq!(loaded.len(), state.len());
        for (i, (a, b)) in state.iter().zip(loaded.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "mismatch at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn snapshot_rejects_bad_magic() {
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(b"XXXX");
        buf.extend_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        let err = read_state_snapshot(0, buf.as_slice()).unwrap_err();
        assert!(matches!(err, RwkvError::SnapshotMagic { .. }), "{err:?}");
    }

    #[test]
    fn snapshot_rejects_bad_version() {
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(&SNAPSHOT_MAGIC);
        buf.extend_from_slice(&(SNAPSHOT_VERSION + 99).to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        let err = read_state_snapshot(0, buf.as_slice()).unwrap_err();
        assert!(matches!(err, RwkvError::SnapshotVersion { .. }), "{err:?}");
    }

    #[test]
    fn snapshot_rejects_length_mismatch() {
        let state = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut buf = Vec::<u8>::new();
        write_state_snapshot(&state, &mut buf).unwrap();
        // Ask for length that doesn't match the stored length.
        let err = read_state_snapshot(5, buf.as_slice()).unwrap_err();
        assert!(matches!(err, RwkvError::SnapshotLength { .. }), "{err:?}");
    }

    #[test]
    fn snapshot_truncated_payload_errors() {
        let state = vec![1.0f32; 16];
        let mut buf = Vec::<u8>::new();
        write_state_snapshot(&state, &mut buf).unwrap();
        buf.truncate(buf.len() - 3); // chop last FP32 in half
        let err = read_state_snapshot(state.len(), buf.as_slice()).unwrap_err();
        assert!(matches!(err, RwkvError::SnapshotIo(_)), "{err:?}");
    }

    #[test]
    fn sample_temperature_zero_is_argmax() {
        let logits = vec![0.1, 0.2, 5.0, 0.3, 0.05];
        let mut rng = SplitMix64::new(42);
        let params = SamplingParams::greedy();
        for _ in 0..8 {
            assert_eq!(sample(&logits, &[], &params, &mut rng), 2);
        }
    }

    #[test]
    fn sample_seeded_is_deterministic() {
        // Two identical seeds → identical draws over a burst.
        let logits = vec![1.0f32, 1.05, 1.02, 0.95, 1.1, 0.8];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            seed: Some(12345),
        };
        let seq_a: Vec<u32> = {
            let mut rng = SplitMix64::new(12345);
            (0..16).map(|_| sample(&logits, &[], &params, &mut rng)).collect()
        };
        let seq_b: Vec<u32> = {
            let mut rng = SplitMix64::new(12345);
            (0..16).map(|_| sample(&logits, &[], &params, &mut rng)).collect()
        };
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn sample_top_k_1_collapses_to_argmax() {
        let logits = vec![0.4, 0.1, 0.9, 0.2, 0.7];
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 1,
            top_p: 1.0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            seed: Some(7),
        };
        let mut rng = SplitMix64::new(7);
        for _ in 0..8 {
            assert_eq!(sample(&logits, &[], &params, &mut rng), 2);
        }
    }

    #[test]
    fn sample_repeat_penalty_pushes_off_recent_token() {
        // Token 3 is the max logit. With a strong repeat penalty and
        // token 3 in the recent buffer, the softmax should shift mass
        // enough that greedy-under-penalty picks something else.
        let logits = vec![0.1f32, 0.2, 0.3, 5.0, 0.15];
        let params_no_pen = SamplingParams {
            temperature: 0.001, // ≈ greedy
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            seed: Some(1),
        };
        let mut rng = SplitMix64::new(1);
        assert_eq!(sample(&logits, &[3], &params_no_pen, &mut rng), 3, "no penalty → argmax");

        let params_pen = SamplingParams {
            repeat_penalty: 100.0,
            repeat_last_n: 4,
            ..params_no_pen
        };
        let mut rng = SplitMix64::new(1);
        // 5.0 / 100.0 = 0.05, so 0.3 becomes the new max. Confirm the
        // answer is *not* 3.
        let out = sample(&logits, &[3], &params_pen, &mut rng);
        assert_ne!(out, 3, "penalty should have suppressed token 3 (got {out})");
    }

    #[test]
    fn sample_top_p_narrow_focuses_on_dominant_class() {
        // One class has ~all the mass under softmax. Top-p tiny → we
        // should never see the low-mass classes.
        let logits = vec![10.0f32, 0.0, 0.0, 0.0, 0.0];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.5,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            seed: Some(99),
        };
        let mut rng = SplitMix64::new(99);
        for _ in 0..32 {
            assert_eq!(sample(&logits, &[], &params, &mut rng), 0);
        }
    }

    #[test]
    fn splitmix64_f32_in_unit_interval() {
        let mut rng = SplitMix64::new(1);
        for _ in 0..1000 {
            let x = rng.next_f32();
            assert!((0.0..1.0).contains(&x), "out of range: {x}");
        }
    }
}
