#!/usr/bin/env python3
"""Rich WKV-state probe — per-head spectral analysis.

CPU-compatible (BlinkDL rwkv package, no triton). Serves as a
TransformerLens-equivalent for RWKV7 when GPU is not available.

Fixed 2026-08-18: the original docstring also promised a third metric,
"gradient saliency (R-lens proxy)" — `gradient_saliency()` was defined but
never called from `probe_prompt`/`run_probes`, the same
promised-vs-actual gap pattern found in `jlens_probe.py` the same night.
Removed rather than wired up (nobody asked for it; `rlens_probe.py`
already covers saliency via a different, working mechanism). `--work-layers`
CLI flag removed for the same reason — defined, never read; every layer is
already reported unconditionally by `per_head_spectrum`.

## Metrics

### Per-head spectral analysis (new vs. probe.py mean+std pooling)
For each prompt, at each layer l, for each head h:
  - `sigma1[l,h]`        : largest singular value of WKV[l,h] (64×64 matrix)
  - `stable_rank[l,h]`   : (||WKV||_F / ||WKV||_2)^2 — multi-slot capacity
  - `entropy[l,h]`        : entropy of normalised squared singular values
                             — head "specialisation" signal

### State trajectory (cross-layer)
  - `layer_delta[l]`     : ||WKV[l] - WKV[l-1]||_F / ||WKV[l-1]||_F
                             — how much state changes per layer

Both metrics can be compared across base vs trained checkpoints.

## Usage

    # Single model, dump per-layer per-head stats
    python rich_probe.py \\
        --model ~/.libs/models/rwkv7/rwkv7-g1h-2.9b-20260710-ctx10240.pth \\
        --prompts prompts.txt \\
        --out results/rich_base.json

    # Base vs trained comparison
    python rich_probe.py \\
        --model ~/.libs/models/rwkv7/rwkv7-g1h-2.9b-20260710-ctx10240.pth \\
        --trained ~/.libs/models/rwkv7/step8/rwkv-step8-epoch0.pth \\
        --out results/rich_base_vs_step8.json

    # H21/H22 items — saliency per premise token
    python rich_probe.py \\
        --model ... \\
        --items ../premise_validator/items_v4_clean.jsonl \\
        --out results/rich_h21_saliency.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments._common import registry
from experiments._common.model import load_model
from experiments._common.results import save_result
from experiments.A0_state_probe.probe import _extract_wkv_per_layer


# ── WKV spectral analysis ──────────────────────────────────────────────────────

def per_head_spectrum(wkv_per_layer: List[torch.Tensor]) -> Dict[str, np.ndarray]:
    """Compute per-layer, per-head spectral metrics from WKV state tensors.

    Args:
        wkv_per_layer: list of tensors, each [n_head, head_size, head_size]

    Returns dict with arrays of shape [n_layer, n_head]:
        sigma1        — largest singular value
        stable_rank   — (frobenius / spectral)^2
        sv_entropy    — entropy of squared sv / sum(sq sv)
    """
    n_layer = len(wkv_per_layer)
    n_head = wkv_per_layer[0].shape[0]

    sigma1 = np.zeros((n_layer, n_head), dtype=np.float32)
    stable_rank = np.zeros((n_layer, n_head), dtype=np.float32)
    sv_entropy = np.zeros((n_layer, n_head), dtype=np.float32)

    for l, wkv in enumerate(wkv_per_layer):
        m = wkv.float().numpy()  # [n_head, head_size, head_size]
        for h in range(n_head):
            sv = np.linalg.svd(m[h], compute_uv=False)  # descending
            s1 = float(sv[0])
            frob = float(np.sqrt((sv ** 2).sum()))
            sr = (frob / s1) ** 2 if s1 > 1e-9 else 0.0
            p = (sv ** 2) / max((sv ** 2).sum(), 1e-30)
            ent = float(-np.sum(p * np.log(p + 1e-30)))
            sigma1[l, h] = s1
            stable_rank[l, h] = sr
            sv_entropy[l, h] = ent

    return {"sigma1": sigma1, "stable_rank": stable_rank, "sv_entropy": sv_entropy}


def layer_trajectory(wkv_per_layer: List[torch.Tensor]) -> np.ndarray:
    """Relative Frobenius delta between consecutive layers. Shape: [n_layer-1]."""
    deltas = []
    for i in range(1, len(wkv_per_layer)):
        prev = wkv_per_layer[i - 1].float()
        curr = wkv_per_layer[i].float()
        delta = float((curr - prev).norm()) / max(float(prev.norm()), 1e-9)
        deltas.append(delta)
    return np.array(deltas, dtype=np.float32)


# ── Item probing ──────────────────────────────────────────────────────────────

def probe_prompt(model, tokenizer, prompt: str) -> Dict:
    """Full probe on a single prompt: spectrum + trajectory.

    Uses tokenizer(prompt)["input_ids"], not tokenizer.encode(prompt) —
    _TokenizerAdapter (experiments/_common/model.py) only implements the
    HF-style __call__ + decode(), no bare .encode(). Found 2026-08-18: this
    was broken (AttributeError) the first time it actually ran through the
    shared loader — probe_prompt/run_probes were written against a
    different tokenizer interface and never exercised end-to-end since.
    """
    ids = tokenizer(prompt)["input_ids"]
    t0 = time.time()
    logits, state = model.forward(ids, None)
    elapsed = time.time() - t0

    wkv_layers = _extract_wkv_per_layer(state)
    spectrum = per_head_spectrum(wkv_layers)
    trajectory = layer_trajectory(wkv_layers)

    # Aggregate per-layer means across heads
    return {
        "n_tokens": len(ids),
        "elapsed_s": round(elapsed, 2),
        "sigma1_by_layer": spectrum["sigma1"].mean(axis=1).tolist(),
        "stable_rank_by_layer": spectrum["stable_rank"].mean(axis=1).tolist(),
        "sv_entropy_by_layer": spectrum["sv_entropy"].mean(axis=1).tolist(),
        "sigma1_per_head": spectrum["sigma1"].tolist(),   # [n_layer, n_head]
        "layer_trajectory": trajectory.tolist(),
    }


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_checkpoints(base_stats: Dict, trained_stats: Dict,
                        prompt_ids: List[str]) -> Dict:
    """Compute per-layer sigma1 and stable_rank diffs (trained - base)."""
    diffs = {}
    for pid in prompt_ids:
        if pid not in base_stats or pid not in trained_stats:
            continue
        b = np.array(base_stats[pid]["sigma1_by_layer"])
        t = np.array(trained_stats[pid]["sigma1_by_layer"])
        diffs[pid] = {
            "sigma1_delta": (t - b).tolist(),
            "sigma1_ratio": (t / (b + 1e-9)).tolist(),
        }
    return diffs


# ── Main ──────────────────────────────────────────────────────────────────────

def run_probes(model, tokenizer, prompts: List[Tuple[str, str]]) -> Dict:
    """prompts: list of (id, prompt_text). Returns dict id→stats."""
    results = {}
    for i, (pid, prompt) in enumerate(prompts):
        print(f"[rich] {i+1}/{len(prompts)} {pid} ...", flush=True)
        try:
            stats = probe_prompt(model, tokenizer, prompt)
            results[pid] = stats
            mid_L = min(16, len(stats["sigma1_by_layer"]) - 1)
            print(f"[rich]   sigma1_L{mid_L}={stats['sigma1_by_layer'][mid_L]:.3f} "
                  f"traj_mean={np.mean(stats['layer_trajectory']):.3f} "
                  f"t={stats['elapsed_s']:.1f}s")
        except Exception as e:
            print(f"[rich]   ERROR: {e}")
            results[pid] = {"error": str(e)}
    return results


DEFAULT_PROMPTS = [
    ("reasoning", (
        "<think>\nLet me analyse carefully. The document describes a meeting on "
        "Tuesday where Alice and Bob discussed the project timeline. Deadline: March 15, "
        "budget: $50,000, team lead: Carol.\n</think>"
    )),
    ("narrative", (
        "The old lighthouse keeper had watched over the bay for thirty years. "
        "Every evening he climbed the spiral stairs, lit the great lamp, and "
        "scanned the horizon for ships in distress."
    )),
    ("math", (
        "<think>\nTo solve 37 × 48: 37 × 48 = 37 × 50 - 37 × 2 = 1850 - 74 = 1776."
        "\n</think>\nAnswer: 1776"
    )),
]


def _load_prompt_list(items_path: Optional[str], prompts_path: Optional[str]) -> List[Tuple[str, str]]:
    prompts: List[Tuple[str, str]] = []
    if items_path:
        with open(items_path) as f:
            for line in f:
                it = json.loads(line)
                pid = it.get("id", f"item_{len(prompts)}")
                text = it.get("prompt") or it.get("text", "")
                if text:
                    prompts.append((pid, text))
    elif prompts_path:
        with open(prompts_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    prompts.append((f"prompt_{i}", line))
    else:
        prompts = DEFAULT_PROMPTS
    return prompts


def _add_rich_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--items", default=None,
                    help="JSONL with {id, prompt} or H21/H22 items.jsonl (uses 'prompt' field).")
    ap.add_argument("--prompts", default=None, help="Plain text file, one prompt per line.")


@registry.probe(
    "rich", hypothesis=["H8"],
    description="Per-head WKV spectral analysis (sigma1/stable_rank/sv_entropy) + cross-layer trajectory.",
    add_args=_add_rich_args,
)
def run(model, tokenizer, args) -> Dict:
    prompts = _load_prompt_list(args.items, args.prompts)
    base_stats = run_probes(model, tokenizer, prompts)
    return {
        "model": args.model,
        "base": base_stats,
        "_summary": {pid: f"sigma1={np.mean(s['sigma1_by_layer']):.3f} "
                          f"stable_rank={np.mean(s['stable_rank_by_layer']):.1f}"
                     for pid, s in base_stats.items() if "error" not in s},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Rich per-head WKV spectral probe.")
    ap.add_argument("--model", required=True, help="Path to .pth (base or single model).")
    ap.add_argument("--trained", default=None, help="Path to .pth trained checkpoint for diff.")
    ap.add_argument("--items", default=None,
                    help="JSONL with {id, prompt} or H21/H22 items.jsonl (uses 'prompt' field).")
    ap.add_argument("--prompts", default=None, help="Plain text file, one prompt per line.")
    ap.add_argument("--out", required=True, help="Output JSON path.")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    prompts = _load_prompt_list(args.items, args.prompts)

    print(f"[rich] loading base model {args.model}")
    model, tokenizer = load_model(args.model, device=args.device)

    base_stats = run_probes(model, tokenizer, prompts)

    output: Dict = {"base": base_stats, "model": args.model}

    if args.trained:
        print(f"[rich] loading trained model {args.trained}")
        model_t, tokenizer_t = load_model(args.trained, device=args.device)
        trained_stats = run_probes(model_t, tokenizer_t, prompts)
        output["trained"] = trained_stats
        output["trained_model"] = args.trained
        output["diff"] = compare_checkpoints(base_stats, trained_stats,
                                              [p[0] for p in prompts])
        del model_t  # free RAM

    out_path = save_result(
        args.out, output, experiment="rich", hypothesis=["H8"],
        model=args.trained or args.model, script=__file__,
    )
    print(f"[rich] wrote {out_path}")

    # Print summary
    print("\n=== Summary (sigma1 mean across layers) ===")
    for pid, stats in base_stats.items():
        if "error" in stats:
            continue
        s1_mean = np.mean(stats["sigma1_by_layer"])
        sr_mean = np.mean(stats["stable_rank_by_layer"])
        print(f"  {pid}: sigma1={s1_mean:.3f}  stable_rank={sr_mean:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
