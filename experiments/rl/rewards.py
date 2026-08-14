#!/usr/bin/env python3
"""rewards.py — reward computation for GRPO word-search RL.

Three reward terms:
    r_correct  — ±1 binary: regex match on task rubric
    r_clipo    — InfoNCE contrastive over WKV state at </think> (2603.10101)
    r_entropy  — GTPO-style: entropy reduction inside think span (2508.04349)

Usage:
    from experiments.rl.rewards import compute_rewards
    rewards = compute_rewards(group, clipo_head)  # shape [G]
"""
from __future__ import annotations

import math
import re
from typing import List, Optional

import torch
import torch.nn.functional as F

from experiments.rl.rollout import RolloutGroup


# ── r_correct ─────────────────────────────────────────────────────────────────

def _score_correct(text: str, rubric: dict) -> float:
    if rubric.get("type") == "regex":
        pattern = rubric.get("value", "")
        return 1.0 if re.search(pattern, text, re.IGNORECASE) else -1.0
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


# ── r_entropy ─────────────────────────────────────────────────────────────────

THINK_OPEN_IDS  = {61, 35762, 63}   # <think>  (set for fast membership)
THINK_CLOSE_IDS = {754, 35762, 63}  # </think>

def _entropy_reward(log_probs: List[float], output_ids: List[int],
                    alpha: float = 0.1) -> float:
    """Entropy reduction inside think span.

    Positive reward when entropy (= −log_prob) decreases from think entry
    to think exit, i.e. model becomes more confident during state accumulation.
    """
    if not log_probs or not output_ids:
        return 0.0

    # Find first <think> start and </think> end in output
    ids = output_ids
    think_start = think_end = None
    for i in range(len(ids) - 2):
        if ids[i] == 61 and ids[i+1] == 35762 and ids[i+2] == 63:
            think_start = i + 3
        if ids[i] == 754 and ids[i+1] == 35762 and ids[i+2] == 63:
            think_end = i
            break

    if think_start is None or think_end is None or think_end <= think_start:
        return 0.0

    span = log_probs[think_start:think_end]
    if len(span) < 2:
        return 0.0

    mid = len(span) // 2
    h_entry = -sum(span[:mid]) / mid          # mean entropy first half
    h_exit  = -sum(span[mid:]) / (len(span) - mid)  # mean entropy second half
    return alpha * (h_entry - h_exit)         # positive if entropy drops


# ── combined ──────────────────────────────────────────────────────────────────

def compute_rewards(
    group: RolloutGroup,
    clipo_head: Optional[torch.nn.Module] = None,
    clipo_weight: float = 1.0,
    entropy_weight: float = 1.0,
    tau: float = 0.05,
) -> torch.Tensor:
    """Compute total reward for each rollout in the group.

    Returns shape [G] tensor.
    """
    G = len(group.rollouts)
    r_correct = torch.zeros(G)
    r_entropy = torch.zeros(G)

    for i, r in enumerate(group.rollouts):
        r_correct[i] = _score_correct(r.text, group.rubric)
        r_entropy[i] = _entropy_reward(r.log_probs, r.output_ids)

    correct_mask = r_correct > 0

    # CLIPO contrastive reward
    r_clipo = torch.zeros(G)
    if clipo_head is not None:
        states = []
        valid = []
        for i, r in enumerate(group.rollouts):
            if r.wkv_state_think is not None:
                flat = r.wkv_state_think.flatten().unsqueeze(0)
                states.append(flat)
                valid.append(i)
        if len(states) >= 2:
            with torch.no_grad():
                proj = clipo_head(torch.cat(states, dim=0))  # [n_valid, D]
            valid_mask = correct_mask[valid]
            sub_rewards = _infonce_reward(proj, valid_mask, tau=tau)
            for j, idx in enumerate(valid):
                r_clipo[idx] = sub_rewards[j]

    return r_correct + clipo_weight * r_clipo + entropy_weight * r_entropy
