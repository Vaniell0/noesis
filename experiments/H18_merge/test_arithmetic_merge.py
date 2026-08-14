#!/usr/bin/env python3
"""H18 sub-claim 2: arithmetic merge of WKV states.

Protocol (H18 sub-claim 2):
  1. Prime trunk state on context A (math problem).
  2. Fork → branch state.
  3. Prime branch on context B (clarification / narrative).
  4. Weighted average: merged = α*trunk + (1-α)*branch, α ∈ {0.3, 0.5, 0.7}.
  5. Decode from merged — check coherence with both contexts.
  Compare with Lucas's Linear merge (candidate d): α=0.5 should give the
  smoothest coherence. α=0.7 (trunk-heavy) should skew toward A.

Usage:
  python3 experiments/H18_merge/test_arithmetic_merge.py \
      --model ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
      --device cpu \
      --out experiments/H18_merge/results/h18_merge_g1d_04b.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import torch

# ── Add repo root to path ─────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "0")

try:
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE
except ImportError:
    print("[H18] ERROR: rwkv package not found. Run from venv with pip install rwkv.", file=sys.stderr)
    sys.exit(1)

# ── Context pairs ─────────────────────────────────────────────────────────────
CONTEXT_PAIRS = [
    {
        "id": "math_narrative",
        "context_A": "What is 17 * 23? Let me compute step by step. 17 * 20 = 340. 17 * 3 = 51. Total: 391.",
        "context_B": "Actually, I meant to ask about a different problem. The farmer has 17 fields and each grows 23 crops.",
        "probe": "So the answer",
        "expected_A": "391",
        "expected_B": "field",
    },
    {
        "id": "code_history",
        "context_A": "The function returns sum(lst) // len(lst) to compute the average of a list of integers.",
        "context_B": "Historically, averaging was done by hand, using tallies and fractions before calculators existed.",
        "probe": "This means",
        "expected_A": "integer",
        "expected_B": "history",
    },
    {
        "id": "science_recipe",
        "context_A": "Sodium chloride (NaCl) dissolves in water because the polar water molecules surround the Na+ and Cl- ions.",
        "context_B": "To make a simple salt solution, add one teaspoon of salt to a glass of warm water and stir until clear.",
        "probe": "The result is",
        "expected_A": "ion",
        "expected_B": "solution",
    },
]

ALPHAS = [0.3, 0.5, 0.7]


def load_model(path: str, device: str):
    torch.set_grad_enabled(False)
    model = RWKV(model=path, strategy=f"{device} bf16")
    pipeline = PIPELINE(model, "rwkv_vocab_v20230424")
    return model, pipeline


def prime_state(model, pipeline, text: str, state_in=None):
    """Feed full text at once, return final (logits, state)."""
    tokens = pipeline.encode(text)
    with torch.no_grad():
        logits, state = model.forward(tokens, state_in)
    return logits, state


def clone_state(state):
    if isinstance(state, (list, tuple)):
        return [s.clone() for s in state]
    return state.clone()


def merge_states(state_a, state_b, alpha: float):
    """merged = alpha * A + (1 - alpha) * B, element-wise."""
    if isinstance(state_a, (list, tuple)):
        return [alpha * a + (1 - alpha) * b for a, b in zip(state_a, state_b)]
    return alpha * state_a + (1 - alpha) * state_b


def decode_tokens(model, pipeline, start_logits, start_state, probe_text: str, n: int = 24) -> str:
    """Feed probe_text then greedily decode n tokens."""
    probe_toks = pipeline.encode(probe_text)
    state = clone_state(start_state)
    with torch.no_grad():
        if probe_toks:
            logits, state = model.forward(probe_toks, state)
        else:
            logits = start_logits
        out = []
        for _ in range(n):
            tok = int(torch.argmax(logits).item())
            out.append(tok)
            if tok == 0:
                break
            logits, state = model.forward([tok], state)
    return pipeline.decode(out)


def run_pair(model, pipeline, pair: dict) -> dict:
    print(f"  pair={pair['id']}")

    # Trunk: prime on A from scratch
    t0 = time.time()
    logits_a, state_a = prime_state(model, pipeline, pair["context_A"])
    print(f"    A primed in {time.time()-t0:.1f}s")

    # Branch: start from A's state, then prime on B
    t0 = time.time()
    logits_b, state_b = prime_state(model, pipeline, pair["context_B"],
                                    state_in=clone_state(state_a))
    print(f"    B primed in {time.time()-t0:.1f}s")

    # Pure A baseline
    text_pure_a = decode_tokens(model, pipeline, logits_a, state_a, pair["probe"])
    text_pure_b = decode_tokens(model, pipeline, logits_b, state_b, pair["probe"])
    print(f"    pure_A: '{text_pure_a[:60]}'")
    print(f"    pure_B: '{text_pure_b[:60]}'")

    results_alpha = []
    for alpha in ALPHAS:
        merged = merge_states(state_a, state_b, alpha)
        text_merged = decode_tokens(model, pipeline, logits_a, merged, pair["probe"])
        has_a = pair["expected_A"].lower() in text_merged.lower()
        has_b = pair["expected_B"].lower() in text_merged.lower()
        results_alpha.append({
            "alpha": alpha,
            "text": text_merged,
            "has_expected_A": has_a,
            "has_expected_B": has_b,
        })
        print(f"    α={alpha}: '{text_merged[:60]}' A={has_a} B={has_b}")

    return {
        "pair_id": pair["id"],
        "context_A": pair["context_A"],
        "context_B": pair["context_B"],
        "probe": pair["probe"],
        "pure_A": text_pure_a,
        "pure_B": text_pure_b,
        "alpha_results": results_alpha,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[H18] loading {args.model}")
    model, pipeline = load_model(args.model, args.device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for pair in CONTEXT_PAIRS:
        r = run_pair(model, pipeline, pair)
        results.append(r)

    with open(out_path, "w") as f:
        json.dump({"model": args.model, "results": results}, f, indent=2)
    print(f"[H18] saved → {out_path}")


if __name__ == "__main__":
    main()
