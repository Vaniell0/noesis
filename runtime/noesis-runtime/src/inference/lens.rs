//! Per-lens WKV snapshot storage on disk.
//!
//! Layout (plan §5, cf. `docs/memory-lenses.md`):
//! ```
//! <lens_root>/<lens_id>/
//! ├── wkv.snapshot   # noesis_rwkv snapshot format (NWKV magic + FP32 LE)
//! └── meta.json      # LensMeta (see below)
//! ```
//!
//! `scratch.slice`, `history.jsonl`, and the richer `lens.toml` land later
//! (blocked on composer + tool-call dispatcher). Everything here is only
//! what unblocks `/lens/save` / `/lens/load` + the `lens_id` on
//! `/api/generate` — enough for a client to prime a state under a name
//! and continue from it.
//!
//! Path safety: `lens_id` is a URL/JSON-provided string that becomes a
//! directory name, so a permissive regex would let a caller write
//! `/etc/passwd`. See [`sanitize_lens_id`] for the allowed shape.
//! Reject-on-parse means callers get a 400 rather than an fs error.

use std::fs;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context, Result};
use noesis_rwkv::{RwkvContext, RwkvSession};
use serde::{Deserialize, Serialize};

/// Persistent metadata for a lens. Written alongside `wkv.snapshot`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LensMeta {
    pub lens_id: String,
    /// Unix seconds when the directory was first created.
    pub created_at: u64,
    /// Unix seconds of the most recent successful `save`.
    pub last_saved: u64,
    /// State length (FP32 count) in the snapshot on disk. A mismatch
    /// against `ctx.state_len()` on load means the checkpoint changed
    /// under us and the lens is no longer portable — reject loudly.
    pub state_len: usize,
    /// Byte size of `wkv.snapshot` on disk (payload + 12 B header).
    pub state_bytes_on_disk: u64,
    /// How many `/lens/save` cycles have written this lens. Handy for
    /// debugging churn without keeping a full history.jsonl yet.
    #[serde(default)]
    pub save_count: u64,
}

/// The maximum accepted lens_id length. sha256[:16] is 16 chars — the
/// plan's canonical form. We allow up to 64 so human-readable ids for
/// testing and manual pins are OK.
const MAX_LENS_ID_LEN: usize = 64;

/// Accept only `[A-Za-z0-9_-]{1,64}`. Rejects `/`, `\`, `..`, `.`, `~`,
/// whitespace, NUL, and empty. Returns the original string on success —
/// no normalisation (case, unicode) so id ↔ dir maps 1:1.
pub fn sanitize_lens_id(raw: &str) -> Result<&str> {
    if raw.is_empty() {
        return Err(anyhow!("lens_id must not be empty"));
    }
    if raw.len() > MAX_LENS_ID_LEN {
        return Err(anyhow!(
            "lens_id longer than {} chars", MAX_LENS_ID_LEN
        ));
    }
    for c in raw.chars() {
        let ok = c.is_ascii_alphanumeric() || c == '_' || c == '-';
        if !ok {
            return Err(anyhow!(
                "lens_id contains disallowed char {c:?} (allowed: [A-Za-z0-9_-])"
            ));
        }
    }
    Ok(raw)
}

/// Path helpers scoped to a single `lens_root`.
#[derive(Debug, Clone)]
pub struct LensDir {
    pub root: PathBuf,
}

impl LensDir {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn dir_for(&self, lens_id: &str) -> Result<PathBuf> {
        let id = sanitize_lens_id(lens_id)?;
        Ok(self.root.join(id))
    }

    pub fn snapshot_path(&self, lens_id: &str) -> Result<PathBuf> {
        Ok(self.dir_for(lens_id)?.join("wkv.snapshot"))
    }

    pub fn meta_path(&self, lens_id: &str) -> Result<PathBuf> {
        Ok(self.dir_for(lens_id)?.join("meta.json"))
    }

    pub fn exists(&self, lens_id: &str) -> bool {
        self.snapshot_path(lens_id)
            .map(|p| p.exists())
            .unwrap_or(false)
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Atomically write a file: write to `<path>.tmp`, then rename over the
/// target. Prevents a torn `wkv.snapshot` if the process crashes mid-
/// write — the reader either sees the previous good file or the new
/// good file, never a truncated one.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension(format!(
        "{}.tmp",
        path.extension().and_then(|s| s.to_str()).unwrap_or("out")
    ));
    {
        let f = fs::File::create(&tmp)
            .with_context(|| format!("create {}", tmp.display()))?;
        let mut w = BufWriter::new(f);
        w.write_all(bytes)?;
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

/// Persist a session's WKV state under `<lens_root>/<lens_id>/`.
/// Preserves `created_at` and `save_count` across saves by reading any
/// existing meta. Returns the updated meta so the HTTP layer can echo
/// it in the response.
pub fn save_session(
    dir: &LensDir,
    lens_id: &str,
    session: &RwkvSession,
) -> Result<LensMeta> {
    let d = dir.dir_for(lens_id)?;
    fs::create_dir_all(&d)
        .with_context(|| format!("create {}", d.display()))?;

    let snapshot_path = dir.snapshot_path(lens_id)?;
    let mut snapshot_bytes: Vec<u8> = Vec::new();
    session
        .save_state(&mut snapshot_bytes)
        .map_err(|e| anyhow!("save_state failed: {e}"))?;
    write_atomic(&snapshot_path, &snapshot_bytes)?;

    let existing = read_meta(dir, lens_id).ok().flatten();
    let created_at = existing.as_ref().map(|m| m.created_at).unwrap_or_else(now_secs);
    let save_count = existing.as_ref().map(|m| m.save_count + 1).unwrap_or(1);

    let meta = LensMeta {
        lens_id: lens_id.to_string(),
        created_at,
        last_saved: now_secs(),
        state_len: session.state().len(),
        state_bytes_on_disk: snapshot_bytes.len() as u64,
        save_count,
    };
    let meta_path = dir.meta_path(lens_id)?;
    let meta_json = serde_json::to_vec_pretty(&meta)?;
    write_atomic(&meta_path, &meta_json)?;
    Ok(meta)
}

/// Load a lens snapshot into a fresh session on the given ctx. Length
/// mismatch (checkpoint changed under us) returns an error — no silent
/// zero-fill or truncate.
pub fn load_session(
    dir: &LensDir,
    lens_id: &str,
    ctx: RwkvContext,
) -> Result<RwkvSession> {
    let snapshot_path = dir.snapshot_path(lens_id)?;
    let f = fs::File::open(&snapshot_path)
        .with_context(|| format!("open {}", snapshot_path.display()))?;
    let reader = BufReader::new(f);
    let session = RwkvSession::load_state(ctx, reader)
        .map_err(|e| anyhow!("load_state failed: {e}"))?;
    Ok(session)
}

/// Read `meta.json` if present. Returns `Ok(None)` for a lens that has
/// never been saved (dir may or may not exist). Any other io/parse
/// error surfaces as `Err` so a corrupted meta doesn't get silently
/// masked as "not there".
pub fn read_meta(dir: &LensDir, lens_id: &str) -> Result<Option<LensMeta>> {
    let path = dir.meta_path(lens_id)?;
    let bytes = match fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(anyhow::Error::from(e).context(format!("read {}", path.display()))),
    };
    let meta: LensMeta = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse {}", path.display()))?;
    Ok(Some(meta))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_accepts_reasonable_ids() {
        assert!(sanitize_lens_id("a").is_ok());
        assert!(sanitize_lens_id("abc123").is_ok());
        assert!(sanitize_lens_id("proj_foo-bar").is_ok());
        // sha256[:16]-style hex
        assert!(sanitize_lens_id("0123456789abcdef").is_ok());
    }

    #[test]
    fn sanitize_rejects_path_traversal() {
        assert!(sanitize_lens_id("").is_err());
        assert!(sanitize_lens_id("..").is_err());
        assert!(sanitize_lens_id(".").is_err());
        assert!(sanitize_lens_id("../etc").is_err());
        assert!(sanitize_lens_id("a/b").is_err());
        assert!(sanitize_lens_id("a\\b").is_err());
        assert!(sanitize_lens_id("~root").is_err());
        assert!(sanitize_lens_id("a b").is_err());
        assert!(sanitize_lens_id("a\0b").is_err());
        assert!(sanitize_lens_id(&"a".repeat(65)).is_err());
    }
}
