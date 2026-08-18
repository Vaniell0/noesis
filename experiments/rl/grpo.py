#!/usr/bin/env python3
"""grpo.py — GRPO advantage normalisation + the H12b.i aux-loss primitive.

Trimmed 2026-08-18: this file used to also hold `grpo_loss` (the PPO-clip
surrogate + KL loss) and its log-prob recomputation helpers
(`recompute_log_probs`/`_sequential_log_probs`), all built around the
pre-WKV-loop `RolloutGroup` structure (prompt_ids + output_ids as one
flat sequence, `</think>`-token rollouts). Deleted along with
`rollout.py` (which defined `RolloutGroup`) and `train_wordsearch.py`
(the only caller) — none of it is reachable from the live path.
`train_wkv_loop.py::wkv_grpo_loss` is the WKV-loop-native replacement
(operates on `WKVLoopRollout`, has its own `_recompute_wkv_log_probs`).

`compute_advantages` below is genuinely shared (both the old and new loss
functions used it, it doesn't care about rollout structure at all).
`h12bi_aux_loss` was extracted out of the deleted `grpo_loss` rather than
deleted with it — it's a self-contained regulariser (only needs `model`
and a scalar loss to add to, no `RolloutGroup` dependency at all) that
`wkv_grpo_loss` doesn't currently call. Not wired in — extracted so the
capability isn't lost, not because it's been decided this belongs in the
live loss; that's a call for whoever next tunes the RL reward.
"""
from __future__ import annotations

import torch


def compute_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalise rewards within a group to get advantages. Shape [G]."""
    mean = rewards.mean()
    std  = rewards.std() + eps
    return (rewards - mean) / std


def h12bi_aux_loss(model, policy_loss: torch.Tensor, h12bi_weight: float) -> torch.Tensor:
    """H12b.i LoRA rank-entropy regulariser (prevents rank collapse during RL).

    Adds `h12bi_weight * compute_h12bi_aux(lora_pairs)` to `policy_loss` and
    returns the result. No-op (returns `policy_loss` unchanged) if
    `h12bi_weight <= 0`, if the model has no `*lora_A`/`*lora_B` parameters,
    or if `training/state_reg.py` isn't importable.

    Not currently called from `train_wkv_loop.py::wkv_grpo_loss` — this is
    the RL-level H12b.i implementation from before the WKV-loop rewrite,
    kept as a ready-to-use primitive rather than lost when the rest of its
    surrounding (RolloutGroup-based) code was deleted.
    """
    if h12bi_weight <= 0.0:
        return policy_loss
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    if str(_root / "training") not in sys.path:
        sys.path.insert(0, str(_root / "training"))
    try:
        from state_reg import compute_h12bi_aux
    except ImportError:
        return policy_loss  # state_reg not on path — skip silently

    lora_A = {n[:-len("lora_A")]: p
              for n, p in model.named_parameters() if n.endswith("lora_A")}
    lora_B = {n[:-len("lora_B")]: p
              for n, p in model.named_parameters() if n.endswith("lora_B")}
    lora_pairs = [(a, lora_B[k]) for k, a in lora_A.items() if k in lora_B]
    if not lora_pairs:
        return policy_loss
    return policy_loss + h12bi_weight * compute_h12bi_aux(lora_pairs)
