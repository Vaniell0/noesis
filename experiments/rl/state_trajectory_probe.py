#!/usr/bin/env python3
"""state_trajectory_probe.py — per-token WKV state-norm trajectory,
read (prompt prefill) vs generate (self-feed), per layer.

Answers a question left open all session: we've measured state content
(IPC, geometry) and decoded M-loop content (what tokens get chosen), but
never how the state itself actually MOVES token-by-token, or whether
read-phase motion looks different from generate-phase (self-feed)
motion. `distill_step`'s generate phase already calls
`loaded.forward_stateful` one token at a time (so per-token state is
already "free" there) — this script does the same token-by-token call
for the READ phase too (normally done in one batched call for training
speed), specifically to get matching per-token granularity on both
sides. Deliberately a separate diagnostic script, not baked into
train_think_distill.py's hot loop — same convention as
`_diag_think_content.py`/`probes.py`: instrumentation for understanding,
not something that should slow down every real training step.

Usage:
    python experiments/rl/state_trajectory_probe.py \\
        --model models/rwkv7-g1i-2.9b-....pth \\
        --out experiments/rl/results/state_trajectory_<name>.json \\
        [--lora-r 32 --lora-alpha 64 --resume path/to/ckpt_stepN]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from experiments.rl.loader import load_rwkv7
from experiments.rl.checkpoint import load_checkpoint
from experiments.rl.wkv_loop import _last_vec
from training.state_reg import DEFAULT_WORK_LAYERS

# Same 4 fixed prompts as the earlier one-off diagnostic decoder
# (_diag_think_content.py, deleted after use) — matrix addition,
# wordsearch, arithmetic sequence, XOR — kept identical so results are
# directly comparable to that session's qualitative findings.
PROMPTS = {
    "matrix_addition": "You are a precise reasoning assistant. Work step by step.\n\n"
        "Add these two 2x2 matrices:\nA = [[1, 2], [3, 4]]\nB = [[5, 6], [7, 8]]\n\n<think>\n",
    "wordsearch": "You are a precise reasoning assistant. Work step by step.\n\n"
        "Below is a 4x4 letter matrix. Rows are separated by newlines; letters within a row are "
        "separated by single spaces.\n\nC A T S\nD O G X\nB I R D\nF I S H\n\n"
        "Find the word CAT reading in any direction (horizontal, vertical, or diagonal).\n\n<think>\n",
    "arithmetic_sequence": "You are a precise reasoning assistant. Work step by step.\n\n"
        "What is the next number in this sequence: 2, 4, 6, 8, ?\n\n<think>\n",
    "xor": "You are a precise reasoning assistant. Work step by step.\n\n"
        "Compute the bitwise XOR of 1010 and 0110.\n\n<think>\n",
}

MAX_GENERATE_TOKENS = 40  # generous over the corpus's real ~8-token phase budget, to see saturation


def _state_norms(wkv, layers) -> dict[int, float]:
    return {L: float(torch.linalg.vector_norm(wkv[L].float().flatten()).item()) for L in layers}


def trace_prompt(loaded, prompt_text: str, layers, tok) -> dict:
    ids = tok.encode(prompt_text)
    state = loaded.new_state(batch=1)
    read_trace = []
    for pos, tid in enumerate(ids):
        x = torch.tensor([[tid]], device=loaded.device)
        logits, state = loaded.forward_stateful(x, state)
        read_trace.append({"pos": pos, "token_id": tid, "norms": _state_norms(state.wkv, layers)})

    gen_trace = []
    next_id = None
    for step in range(MAX_GENERATE_TOKENS):
        v = _last_vec(logits)
        next_id = int(v.argmax().item())
        gen_trace.append({"step": step, "token_id": next_id,
                           "norms": _state_norms(state.wkv, layers)})
        if next_id == 0:  # EOS — stop tracing, nothing more to feed
            break
        x = torch.tensor([[next_id]], device=loaded.device)
        logits, state = loaded.forward_stateful(x, state)

    return {"read": read_trace, "generate": gen_trace, "prompt_n_tokens": len(ids)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--resume", type=Path, default=None,
                     help="Optional checkpoint dir to load on top of --model (LoRA or full-FT).")
    ap.add_argument("--lora-r", type=int, default=0)
    ap.add_argument("--lora-alpha", type=int, default=0)
    ap.add_argument("--work-layers", default=",".join(str(x) for x in DEFAULT_WORK_LAYERS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    layers = tuple(int(x) for x in args.work_layers.split(","))
    loaded = load_rwkv7(args.model, device=args.device, backend="peft",
                         lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    if args.resume is not None:
        step = load_checkpoint(args.resume, loaded, None, None)
        print(f"[state_trajectory_probe] resumed from {args.resume} at step {step}")

    tok = loaded.tokenizer
    results = {}
    with torch.no_grad():
        for name, prompt_text in PROMPTS.items():
            print(f"[state_trajectory_probe] tracing {name} ...")
            results[name] = trace_prompt(loaded, prompt_text, layers, tok)
            n_read = len(results[name]["read"])
            n_gen = len(results[name]["generate"])
            hit_eos = results[name]["generate"][-1]["token_id"] == 0 if n_gen else False
            print(f"  read={n_read} tok, generate={n_gen} tok, hit_eos={hit_eos}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": str(args.model), "resume": str(args.resume) if args.resume else None,
        "layers": list(layers), "results": results,
    }, indent=2))
    print(f"[state_trajectory_probe] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
