#!/usr/bin/env python3
"""think_geometry.py — per-token WKV state delta analysis, think vs non-think.

Measures how much each token changes the WKV state (delta norm, stable rank
of the delta matrix, sigma1) at key layers. Tokens inside <think>...</think>
are labelled separately from tokens outside.

Prediction (H8): think-span tokens produce larger, higher-rank WKV deltas
than non-think tokens — the model routes active computation there.

Usage:
    python think_geometry.py \
        --model ~/.libs/models/rwkv7/rwkv7-g1h-2.9b-step9b-e1.pth \
        --layers 4,16,31 \
        --out results/think_vs_nonthink_step9b_e1.json

    # Optionally compare two checkpoints:
    python think_geometry.py \
        --model ~/.libs/models/rwkv7/rwkv7-g1h-2.9b-20260710-ctx10240.pth \
        --compare ~/.libs/models/rwkv7/rwkv7-g1h-2.9b-step9b-e1.pth \
        --layers 4,16,31 \
        --out results/think_geometry_base_vs_step9b.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common import registry
from experiments._common.layers import default_layers
from experiments._common.model import load_model
from experiments._common.results import save_result

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

DEFAULT_PROMPT = (
    "Solve step by step: A train leaves city A at 9:00 at 60 km/h. "
    "Another train leaves city B (300 km away) at 10:00 at 90 km/h. "
    "When do they meet?\n"
    "<think>\n"
    "Distance from A: 60*(t+1), distance from B: 90*t where t = hours after 10:00.\n"
    "They meet when 60*(t+1) + 90*t = 300.\n"
    "60t + 60 + 90t = 300 → 150t = 240 → t = 1.6 hours.\n"
    "They meet at 11:36.\n"
    "</think>\n"
    "They meet at 11:36 AM."
)


def _delta_stats(delta: torch.Tensor) -> Dict[str, float]:
    """Fast stats on (n_head, H, H) delta tensor — no SVD, just norms."""
    d = delta.float()
    frob = float(d.norm())
    # Per-head mean absolute delta — proxy for update magnitude
    mean_abs = float(d.abs().mean())
    # Rough rank proxy: ratio of max singular value of mean head to frob
    # Use only first head for sigma1 to keep cost low
    try:
        sv = torch.linalg.svdvals(d[0])
        s1 = float(sv[0])
        sr = (frob ** 2) / (s1 ** 2 * d.shape[0] + 1e-12)
    except Exception:
        s1, sr = 0.0, 0.0
    return {"sigma1": s1, "stable_rank": sr, "frob": frob, "mean_abs": mean_abs}


def _find_think_spans(token_ids: List[int], tokenizer) -> List[bool]:
    """Return per-token boolean: True = inside <think>...</think>."""
    # Tokenize the markers to find their ids
    think_open_ids = tokenizer(THINK_OPEN)["input_ids"]
    think_close_ids = tokenizer(THINK_CLOSE)["input_ids"]

    inside = [False] * len(token_ids)
    in_think = False

    def _matches(ids: List[int], pos: int, target: List[int]) -> bool:
        return ids[pos : pos + len(target)] == target

    i = 0
    while i < len(token_ids):
        if _matches(token_ids, i, think_open_ids):
            in_think = True
            for j in range(i, min(i + len(think_open_ids), len(token_ids))):
                inside[j] = True
            i += len(think_open_ids)
        elif _matches(token_ids, i, think_close_ids):
            for j in range(i, min(i + len(think_close_ids), len(token_ids))):
                inside[j] = True
            in_think = False
            i += len(think_close_ids)
        else:
            inside[i] = in_think
            i += 1

    return inside


def _analyze(model, tokenizer, layers: Optional[List[int]], prompt: str) -> Dict:
    model.eval()

    enc = tokenizer(prompt)
    token_ids: List[int] = enc["input_ids"]
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.squeeze().tolist()

    think_mask = _find_think_spans(token_ids, tokenizer)

    tokens_decoded = [tokenizer.decode([t]) for t in token_ids]

    per_token: List[Dict] = []
    prev_state: Optional[List[torch.Tensor]] = None

    with torch.no_grad():
        state = None
        for pos, tok_id in enumerate(token_ids):
            _, state = model.forward([tok_id], state)

            if layers is None:
                # First token's state reveals real n_layer — resolve once,
                # by fractional depth, instead of a hardcoded list (was
                # "4,16,31", silently skipped out-of-range layers on
                # smaller models with no warning — fixed 2026-08-18).
                layers = default_layers(len(state) // 3)

            # Extract WKV state per target layer: state[3*L + 1]
            wkv_now: Dict[int, torch.Tensor] = {}
            for L in layers:
                idx = 3 * L + 1
                if idx < len(state):
                    wkv_now[L] = state[idx].float().cpu()

            # Compute delta vs previous step
            layer_stats: Dict[str, Dict] = {}
            for L in layers:
                if L not in wkv_now:
                    continue
                s_cur = wkv_now[L]  # (n_head, H, H)
                if prev_state is not None and L in prev_state:
                    delta = s_cur - prev_state[L]
                else:
                    delta = s_cur  # first token: delta = state itself

                stats = _delta_stats(delta)
                layer_stats[str(L)] = stats

            per_token.append({
                "pos": pos,
                "token_id": tok_id,
                "token": tokens_decoded[pos],
                "in_think": think_mask[pos],
                "layers": layer_stats,
            })

            # Save current WKV state for next delta
            prev_state = {L: wkv_now[L].clone() for L in layers if L in wkv_now}

    # Aggregate: mean stats inside vs outside think
    def _mean(lst):
        return sum(lst) / len(lst) if lst else None

    per_layer_summary: Dict[str, Dict] = {}
    for L in layers:
        key = str(L)
        think_frobs = [t["layers"][key]["frob"] for t in per_token
                       if key in t["layers"] and t["in_think"]]
        nothink_frobs = [t["layers"][key]["frob"] for t in per_token
                         if key in t["layers"] and not t["in_think"]]
        think_srs = [t["layers"][key]["stable_rank"] for t in per_token
                     if key in t["layers"] and t["in_think"]]
        nothink_srs = [t["layers"][key]["stable_rank"] for t in per_token
                       if key in t["layers"] and not t["in_think"]]

        think_mean, nothink_mean = _mean(think_frobs), _mean(nothink_frobs)
        ratio = (think_mean or 0) / (nothink_mean or 1e-9)
        per_layer_summary[key] = {
            "think_frob": think_mean, "nothink_frob": nothink_mean, "ratio": ratio,
            "think_stable_rank": _mean(think_srs), "nothink_stable_rank": _mean(nothink_srs),
        }
        print(f"\nLayer {L}:")
        print(f"  mean delta frob  — think: {think_mean:.4f}  "
              f"non-think: {nothink_mean:.4f}  ratio: {ratio:.2f}×")
        print(f"  mean stable_rank — think: {_mean(think_srs):.4f}  "
              f"non-think: {_mean(nothink_srs):.4f}")

    return {
        "prompt_len": len(token_ids),
        "think_tokens": sum(think_mask),
        "non_think_tokens": len(think_mask) - sum(think_mask),
        "layers": layers,
        "per_layer_summary": per_layer_summary,
        "per_token": per_token,
        "_summary": {f"L{L} think/non-think ratio": f"{per_layer_summary[str(L)]['ratio']:.2f}×" for L in layers},
    }


def run_probe(model_path: str, layers: List[int], prompt: str,
              device: str = "cpu") -> Dict:
    """Standalone entry point: loads its own model (for base/--compare CLI usage)."""
    model, tokenizer = load_model(model_path, device=device)
    result = _analyze(model, tokenizer, layers, prompt)
    return {"model_path": str(model_path), **result}


def _add_think_geometry_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--layers", default=None,
                     help="Comma-separated layer indices (default: picked by fractional "
                          "depth from the loaded model)")
    # Prefixed, not shared with ipc_analysis's --prompt: different default,
    # different contract (must contain <think>...</think>), so the shared
    # parser's conflict_handler="resolve" assumption (shared name = shared
    # semantics, see run.py) does not hold here.
    ap.add_argument("--tg-prompt", dest="tg_prompt", default=None,
                     help="Custom prompt (must contain <think>...</think>)")


@registry.probe(
    "think_geometry", hypothesis=["H8"],
    description="Per-token WKV state delta (frob, stable rank) inside vs outside <think> spans",
    add_args=_add_think_geometry_args,
)
def run(model, tokenizer, args) -> Dict:
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    prompt = args.tg_prompt or DEFAULT_PROMPT
    result = _analyze(model, tokenizer, layers, prompt)
    return {"model": args.model, **result}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--compare", default=None, help="Optional second checkpoint")
    ap.add_argument("--layers", default=None,
                     help="Comma-separated layer indices (default: picked by fractional "
                          "depth from the loaded model)")
    ap.add_argument("--prompt", default=None, help="Custom prompt (must contain <think>...</think>)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")] if args.layers else None
    prompt = args.prompt or DEFAULT_PROMPT
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== think_geometry: {args.model} ===")
    result_a = run_probe(args.model, layers, prompt, args.device)

    result = {"model_a": result_a}

    if args.compare:
        print(f"\n=== think_geometry: {args.compare} ===")
        result_b = run_probe(args.compare, layers, prompt, args.device)
        result["model_b"] = result_b

        print("\n=== Delta frob ratio comparison (model_b / model_a) ===")
        for L in layers:
            key = str(L)
            def _think_mean(res):
                vals = [t["layers"][key]["frob"] for t in res["per_token"]
                        if key in t["layers"] and t["in_think"]]
                return sum(vals) / len(vals) if vals else 0.0
            ra = _think_mean(result_a)
            rb = _think_mean(result_b)
            print(f"  L{L}: base={ra:.4f}  trained={rb:.4f}  ratio={rb/(ra+1e-9):.2f}×")

    save_result(
        out, result, experiment="think_geometry", hypothesis=["H8"],
        model=args.compare or args.model, script=__file__,
    )
    print(f"\nSaved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
