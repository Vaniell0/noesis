//! End-to-end smoke test for the rwkv.cpp skeleton.
//!
//! Loads the model at `$1` (or `./result/model.bin`), tokenizes "$2"
//! (default: "The Wright brothers"), evaluates the prompt with
//! `eval_sequence`, then greedily samples 20 tokens with `eval` and
//! prints the resulting text.
//!
//! This is the minimum end-to-end proof that
//!   `noesis-model → noesis-rwkv-sys → noesis-rwkv → tokenizer → argmax`
//! is wired up.  Not a benchmark, not a correctness test — just:
//! do we get real tokens out?

use std::path::PathBuf;
use std::time::Instant;

use noesis_rwkv::{argmax, tokenizer::WorldTokenizer, RwkvContext, RwkvSession};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = PathBuf::from(
        args.next().unwrap_or_else(|| "result/model.bin".to_string()),
    );
    let prompt = args
        .next()
        .unwrap_or_else(|| "The Wright brothers".to_string());
    let n_generate: usize = args
        .next()
        .as_deref()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let n_threads: u32 = std::env::var("RWKV_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(4));
    let t0 = Instant::now();
    let ctx = RwkvContext::open(&model_path, n_threads, 0)
        .map_err(|e| anyhow::anyhow!("open failed: {e:?}"))?;
    eprintln!("using n_threads={n_threads}");
    eprintln!(
        "loaded {} in {:?}: n_vocab={} n_embed={} n_layer={} state_len={}",
        model_path.display(),
        t0.elapsed(),
        ctx.n_vocab(),
        ctx.n_embed(),
        ctx.n_layer(),
        ctx.state_len(),
    );

    let tok = WorldTokenizer::new()?;
    let prompt_ids = tok.encode(&prompt);
    eprintln!("prompt: {:?}", &prompt);
    eprintln!("prompt_ids ({}): {:?}", prompt_ids.len(), prompt_ids);

    let mut session = RwkvSession::new(ctx);

    let t1 = Instant::now();
    let _ = session
        .eval_sequence(&prompt_ids)
        .map_err(|e| anyhow::anyhow!("eval_sequence failed: {e:?}"))?;
    eprintln!("prompt eval: {:?}", t1.elapsed());

    let mut generated: Vec<u32> = Vec::with_capacity(n_generate);
    let t2 = Instant::now();
    for _ in 0..n_generate {
        // Sample from the last logits produced by the previous call.
        let logits = session.state(); // placeholder to satisfy borrow-checker
        let _ = logits; // discard — real logits come from eval below
        // First: re-run eval with the last token so we have fresh logits.
        // Simpler: on each step, feed the *last generated* token
        // (or the last prompt token on the very first iteration).
        let last = *generated.last().unwrap_or(prompt_ids.last().unwrap());
        let logits = session
            .eval(last)
            .map_err(|e| anyhow::anyhow!("eval failed: {e:?}"))?;
        let next = argmax(logits);
        generated.push(next);
    }
    eprintln!(
        "generated {} tokens in {:?} ({:.1} tok/s)",
        n_generate,
        t2.elapsed(),
        n_generate as f64 / t2.elapsed().as_secs_f64(),
    );

    let text = tok
        .decode(&generated)
        .unwrap_or_else(|e| format!("[decode error: {e}]"));
    println!("{prompt}{text}");
    Ok(())
}
