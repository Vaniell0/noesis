//! Full state save/load round-trip against a real model.
//!
//! Usage:
//!     cargo run --example state_roundtrip -- <model.bin> ["prompt"]
//!
//! What it does:
//!   1. Load model, tokenize prompt, `eval_sequence` it into a session.
//!   2. Record the current logits' argmax (would-be next token) — call it
//!      `next_before`.
//!   3. `save_state` to a tempfile.
//!   4. Fresh clone of the context, `load_state` from tempfile into a new
//!      session, run `eval` on the last prompt token to produce fresh
//!      logits — call the argmax `next_after`.
//!   5. Assert `next_before == next_after`. This is the load-bearing check
//!      for lens persistence: a saved WKV state must resume decode on a
//!      new session without drift.
//!
//! Not a benchmark, not a stress test — just proof that the shipped
//! save/load path preserves the exact reasoning frontier the session held.

use std::fs;
use std::path::PathBuf;

use noesis_rwkv::{argmax, tokenizer::WorldTokenizer, RwkvContext, RwkvSession};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = PathBuf::from(
        args.next().ok_or_else(|| anyhow::anyhow!("model path required"))?,
    );
    let prompt = args
        .next()
        .unwrap_or_else(|| "The Wright brothers".to_string());

    let n_threads: u32 = std::env::var("RWKV_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(
            std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4),
        );

    let ctx = RwkvContext::open(&model_path, n_threads, 0)
        .map_err(|e| anyhow::anyhow!("open failed: {e:?}"))?;
    eprintln!(
        "loaded {}: state_len={}",
        model_path.display(),
        ctx.state_len()
    );

    let tok = WorldTokenizer::new()?;
    let prompt_ids = tok.encode(&prompt);
    if prompt_ids.is_empty() {
        anyhow::bail!("prompt tokenised to zero tokens");
    }
    let (last_prompt_id, before_last) = (
        *prompt_ids.last().unwrap(),
        &prompt_ids[..prompt_ids.len() - 1],
    );

    // Session A: absorb everything but the last token; record argmax of
    // logits after feeding the last token — this is what the model would
    // pick as the next token.
    let mut session_a = RwkvSession::new(ctx.clone());
    if !before_last.is_empty() {
        session_a
            .eval_sequence(before_last)
            .map_err(|e| anyhow::anyhow!("eval_sequence failed: {e:?}"))?;
    }
    let logits_a = session_a
        .eval(last_prompt_id)
        .map_err(|e| anyhow::anyhow!("eval a failed: {e:?}"))?;
    let next_before = argmax(logits_a);

    // Snapshot AFTER the same eval; we then verify that re-loading this
    // snapshot produces a session whose *next* eval matches what session_a
    // would produce next.
    let mut snap_path = std::env::temp_dir();
    snap_path.push(format!("noesis-state-{}.snap", std::process::id()));
    {
        let f = fs::File::create(&snap_path)?;
        let bw = std::io::BufWriter::new(f);
        session_a
            .save_state(bw)
            .map_err(|e| anyhow::anyhow!("save_state failed: {e:?}"))?;
    }
    eprintln!("snapshot written: {}", snap_path.display());

    // What session_a will produce next, feeding `next_before` back in.
    let logits_a2 = session_a
        .eval(next_before)
        .map_err(|e| anyhow::anyhow!("eval a2 failed: {e:?}"))?;
    let next_after_a = argmax(logits_a2);

    // Session B: fresh, load snapshot, feed the *same* next_before token,
    // check argmax matches next_after_a.
    let ctx_b = ctx
        .clone_for_parallel(n_threads)
        .map_err(|e| anyhow::anyhow!("clone_for_parallel failed: {e:?}"))?;
    let f = fs::File::open(&snap_path)?;
    let br = std::io::BufReader::new(f);
    let mut session_b = RwkvSession::load_state(ctx_b, br)
        .map_err(|e| anyhow::anyhow!("load_state failed: {e:?}"))?;
    let logits_b = session_b
        .eval(next_before)
        .map_err(|e| anyhow::anyhow!("eval b failed: {e:?}"))?;
    let next_after_b = argmax(logits_b);

    // Reporting.
    let dec = |t: u32| -> String {
        tok.decode(&[t]).unwrap_or_else(|e| format!("[?{e}]"))
    };
    println!("prompt: {prompt:?} ({} tokens)", prompt_ids.len());
    println!("next_before      = {next_before}  ({:?})", dec(next_before));
    println!(
        "next_after_a     = {next_after_a}  ({:?})",
        dec(next_after_a)
    );
    println!(
        "next_after_b     = {next_after_b}  ({:?})  [reloaded]",
        dec(next_after_b)
    );

    let _ = fs::remove_file(&snap_path);

    if next_after_a != next_after_b {
        anyhow::bail!(
            "save/load round-trip drift: session_a picked {next_after_a}, \
             session_b (reloaded) picked {next_after_b}"
        );
    }
    println!("OK — state save/load preserved next-token decision.");
    Ok(())
}
