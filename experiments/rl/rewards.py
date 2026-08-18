#!/usr/bin/env python3
"""rewards.py — reward computation for GRPO / WKV-loop RL.

  from experiments.rl.rewards import compute_wkv_loop_rewards
  rewards, diag = compute_wkv_loop_rewards(rollouts, rubric)

  r = r_correct − β·M − γ·Σ_t ReLU(ΔH_t) [+ δ·stability_bonus]

Trimmed 2026-08-18 alongside `rollout.py`/`train_wordsearch.py`'s
deletion: the legacy `compute_rewards(group: RolloutGroup, ...)` combiner
and the `<think>`/`</think>`-span entropy reward (`_entropy_reward`,
`_find_think_span`, `_THINK_*` constants) are gone — both were built
around token spans that don't exist in the WKV-loop's M-step design
(see `docs/rl-track.md` §Deferred). `_infonce_reward` was kept, paired
with `clipo_head.py` — CLIPO is explicitly flagged there as a real
"revisit later" item, not dead code, even though its current integration
(this file's deleted `compute_rewards`) doesn't apply anymore.
"""
from __future__ import annotations

import math
import re
from typing import List

import torch
import torch.nn.functional as F

from experiments.rl.wkv_loop import WKVLoopRollout


# ── r_correct ─────────────────────────────────────────────────────────────────

_FORMAT_POSITION = re.compile(r"row\s*=\s*\d+\b[^0-9]*col\s*=\s*\d+\b", re.IGNORECASE)


def _score_correct(text: str, rubric: dict) -> float:
    rtype = rubric.get("type")
    if rtype == "regex":
        pattern = rubric.get("value", "")
        if re.search(pattern, text, re.IGNORECASE):
            return 1.0
        # correct format but wrong value → 0.0, prevents reward collapse when
        # model learns to structure output but not yet solve the task
        if "col" in pattern and _FORMAT_POSITION.search(text):
            return 0.0
        return -1.0
    if rtype == "exact":
        value = rubric.get("value", "")
        return 1.0 if value.upper() in text.upper() else -1.0
    return 0.0


# ── r_clipo ───────────────────────────────────────────────────────────────────

def _infonce_reward(
    states: torch.Tensor,    # [G, D] — projected WKV states
    correct_mask: torch.Tensor,  # [G] bool
    tau: float = 0.05,
    clamp_min: float = -0.5,
) -> torch.Tensor:
    """InfoNCE contrastive reward per rollout.

    For each correct rollout i:
        anchor  = states[i]
        pos     = states[j] where j≠i and correct[j]
        neg     = states[k] where ~correct[k]
    r_con_i = -InfoNCE(anchor, pos, neg)

    Incorrect rollouts get r_con = 0 (no contrastive signal).
    Returns tensor [G] of rewards, clamped to [clamp_min, 0].
    """
    G = states.shape[0]
    rewards = torch.zeros(G)

    correct_idx = correct_mask.nonzero(as_tuple=True)[0]
    wrong_idx = (~correct_mask).nonzero(as_tuple=True)[0]

    if len(correct_idx) < 2 or len(wrong_idx) == 0:
        return rewards  # not enough contrast pairs

    norm = F.normalize(states, dim=-1)

    for i in correct_idx:
        anchor = norm[i]
        # positives: other correct rollouts
        pos_idx = correct_idx[correct_idx != i]
        if len(pos_idx) == 0:
            continue
        pos_sim = (anchor @ norm[pos_idx].T) / tau   # [n_pos]
        neg_sim = (anchor @ norm[wrong_idx].T) / tau  # [n_neg]

        # InfoNCE: -log(mean_exp_pos / (mean_exp_pos + sum_exp_neg))
        log_pos = torch.logsumexp(pos_sim, dim=0) - math.log(len(pos_idx))
        log_denom = torch.logsumexp(torch.cat([pos_sim, neg_sim]), dim=0)
        loss = -(log_pos - log_denom)
        rewards[i] = max(float(-loss), clamp_min)

    return rewards


# ── WKV-loop reward (no <think> tokens) ───────────────────────────────────────

def compute_wkv_loop_rewards(
    rollouts: List["WKVLoopRollout"],
    rubric: dict,
    *,
    beta: float = 0.02,
    gamma: float = 0.1,
    delta: float = 0.0,
    stability_threshold: float = 1.5,
) -> tuple:
    """Per-rollout reward for WKV-loop trajectories.

    r = r_correct − β·M − γ·Σ_t ReLU(H_t − H_{t-1}) [+ δ·stability_bonus]

    Args:
        rollouts: list of WKVLoopRollout (one per sample in the GRPO group).
        rubric: dict with "type" and "value" keys (same format as _score_correct).
        beta: step-count penalty coefficient.
        gamma: entropy-increase penalty coefficient.
        delta: WKV-stability bonus coefficient (0 = disabled).
        stability_threshold: mean wkv_stability below this → stable bonus.

    Returns:
        rewards: float tensor [G]
        diag: dict with per-component tensors for logging:
              "r_correct", "r_effort", "r_entropy_penalty", "r_stability",
              "M", "exit_reason"
    """
    G = len(rollouts)
    r_correct_t        = torch.zeros(G)
    r_effort_t         = torch.zeros(G)
    r_entropy_penalty_t = torch.zeros(G)
    r_stability_t      = torch.zeros(G)
    M_t                = torch.zeros(G, dtype=torch.long)
    exit_reasons: List[str] = []

    for i, r in enumerate(rollouts):
        r_correct_t[i] = _score_correct(r.text, rubric)

        r_effort_t[i] = -beta * r.M
        M_t[i] = r.M

        # sum of entropy increases: Σ_t max(0, H_t - H_{t-1})
        traj = r.entropy_trajectory
        entropy_penalty = 0.0
        for t in range(1, len(traj)):
            entropy_penalty += max(0.0, traj[t] - traj[t - 1])
        r_entropy_penalty_t[i] = -gamma * entropy_penalty

        if delta > 0.0 and len(r.wkv_stability) > 1:
            mean_stab = sum(r.wkv_stability[1:]) / len(r.wkv_stability[1:])
            if mean_stab < stability_threshold:
                r_stability_t[i] = delta

        exit_reasons.append(r.exit_reason)

    rewards = r_correct_t + r_effort_t + r_entropy_penalty_t + r_stability_t
    diag = {
        "r_correct":         r_correct_t,
        "r_effort":          r_effort_t,
        "r_entropy_penalty": r_entropy_penalty_t,
        "r_stability":       r_stability_t,
        "M":                 M_t,
        "exit_reason":       exit_reasons,
    }
    return rewards, diag
