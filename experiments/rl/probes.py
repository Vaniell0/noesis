"""probes.py — inline diagnostic suite for WKV-loop RL training.

Called after each checkpoint (or every K batches). No external deps beyond
what loader.py and metrics.py already provide.

Three probes:

  stable_rank_probe(loaded, layers)
    Run three fixed diagnostic prompts through the model, capture WKV state,
    compute stable-rank at requested layers. Returns dict[prompt_name → SR].
    Measures how much orientation diversity WKV heads use — should rise during
    word-search RL as model learns multi-directional scanning.

  effort_frontier(rollouts)
    Histogram of M (loop steps) and exit_reason over the batch.
    Tracks whether the model is learning to resolve tasks in fewer steps.

  shortcut_score(rollouts)
    Fraction of rollouts that exit "commit" at M ≤ 1. High value = model
    collapsed to memorised token (SHORTCUT condition in monitor.py).

  run_inline_probes(loaded, rollouts, layers, label)
    Convenience wrapper. Returns compact dict ready for JSON logging.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional

import torch

from experiments.rl.loader import LoadedModel
from experiments.rl.wkv_loop import WKVLoopRollout
from experiments.A0_state_probe.metrics import stable_rank


_DIAG_PROMPTS = {
    "reasoning":
        "Let me analyse carefully. The document describes a meeting on Tuesday "
        "where Alice and Bob discussed the project timeline. Deadline: March 15, "
        "budget: $50,000.",
    "math":
        "To solve 37 × 48: 37 × 50 minus 37 × 2 equals 1850 minus 74 equals 1776.",
    "narrative":
        "The old lighthouse keeper had watched over the bay for thirty years. "
        "Every evening he climbed the spiral stairs and lit the great lamp.",
}

# Default layers to probe (matches rl-track.md §stable_rank tracking)
_DEFAULT_LAYERS = [4, 16, 28]


# ------------------------------------------------------------------
# 1. stable_rank probe

def stable_rank_probe(
    loaded: LoadedModel,
    layers: Optional[List[int]] = None,
) -> Dict[str, Dict[int, float]]:
    """Compute mean stable-rank per requested layer for each diagnostic prompt.

    Returns:
        {prompt_name: {layer_idx: mean_SR_over_heads}}
    """
    target_layers = layers or _DEFAULT_LAYERS
    results: Dict[str, Dict[int, float]] = {}

    for name, text in _DIAG_PROMPTS.items():
        ids = loaded.tokenizer.encode(text)
        state = loaded.new_state(batch=1)
        if loaded.backend == "peft":
            inp = torch.tensor([ids], dtype=torch.long, device=loaded.device)
        else:
            inp = ids
        with torch.no_grad():
            _, state = loaded.forward_stateful(inp, state)

        # wkv_stack returns [L, n_head, head_size, head_size]
        wkv = loaded.wkv_stack(state)   # [L, n_head, h, h]
        # stable_rank expects list of [n_head, h, h] tensors
        wkv_layers = [wkv[i] for i in range(wkv.shape[0])]

        # Compute SR per layer for the requested layers
        sr_per_layer = stable_rank(wkv_layers)   # list[list[float]]: [layer][head]

        layer_mean: Dict[int, float] = {}
        for l in target_layers:
            if l < len(sr_per_layer):
                heads = sr_per_layer[l]
                layer_mean[l] = sum(heads) / len(heads) if heads else 0.0
            else:
                layer_mean[l] = float("nan")

        results[name] = layer_mean

    return results


# ------------------------------------------------------------------
# 2. effort frontier

def effort_frontier(rollouts: List[WKVLoopRollout]) -> Dict:
    """M distribution + exit_reason histogram over a rollout batch."""
    M_values = [r.M for r in rollouts]
    exit_counts = Counter(r.exit_reason for r in rollouts)

    if not M_values:
        return {"M_mean": float("nan"), "M_max": 0, "M_min": 0,
                "M_hist": {}, "exit_reason": {}}

    return {
        "M_mean":   sum(M_values) / len(M_values),
        "M_max":    max(M_values),
        "M_min":    min(M_values),
        "M_hist":   _hist(M_values, bins=8),
        "exit_reason": dict(exit_counts),
    }


# ------------------------------------------------------------------
# 3. shortcut score

def shortcut_score(rollouts: List[WKVLoopRollout]) -> float:
    """Fraction of rollouts with commit exit at M ≤ 1 (shortcut alarm)."""
    if not rollouts:
        return 0.0
    n = sum(1 for r in rollouts if r.exit_reason == "commit" and r.M <= 1)
    return n / len(rollouts)


# ------------------------------------------------------------------
# Convenience wrapper

def run_inline_probes(
    loaded: LoadedModel,
    rollouts: List[WKVLoopRollout],
    *,
    layers: Optional[List[int]] = None,
    label: str = "",
) -> Dict:
    """Run all three probes and return a flat dict for JSON logging."""
    sr = stable_rank_probe(loaded, layers=layers)
    ef = effort_frontier(rollouts)
    sc = shortcut_score(rollouts)

    out: Dict = {
        "label": label,
        "shortcut_score": sc,
        "M_mean":    ef["M_mean"],
        "M_max":     ef["M_max"],
        "M_min":     ef["M_min"],
        "M_hist":    ef["M_hist"],
        "exit_reason": ef["exit_reason"],
    }
    for prompt_name, layer_dict in sr.items():
        for layer_idx, sr_val in layer_dict.items():
            out[f"sr_{prompt_name}_L{layer_idx}"] = sr_val

    return out


# ------------------------------------------------------------------
# Internal

def _hist(values: List[int], bins: int = 8) -> Dict[str, int]:
    if not values:
        return {}
    lo, hi = min(values), max(values)
    if lo == hi:
        return {str(lo): len(values)}
    step = max(1, math.ceil((hi - lo + 1) / bins))
    counts: Dict[str, int] = {}
    for v in values:
        bucket = ((v - lo) // step) * step + lo
        key = f"{bucket}-{bucket + step - 1}"
        counts[key] = counts.get(key, 0) + 1
    return counts
