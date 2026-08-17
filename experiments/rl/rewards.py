#!/usr/bin/env python3
"""rewards.py — reward computation for GRPO / WKV-loop RL.

Two APIs:

  WKV-loop (current, preferred):
    from experiments.rl.rewards import compute_wkv_loop_rewards
    rewards, diag = compute_wkv_loop_rewards(rollouts, rubric)

    r = r_correct − β·M − γ·Σ_t ReLU(ΔH_t) [+ δ·stability_bonus]

  Legacy RolloutGroup (kept for compatibility while rollout.py exists):
    from experiments.rl.rewards import compute_rewards
    rewards = compute_rewards(group, clipo_head)
"""
from __future__ import annotations

import math
import re
from typing import List, Optional

import torch
import torch.nn.functional as F

from experiments.rl.rollout import RolloutGroup
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


# ── r_entropy ─────────────────────────────────────────────────────────────────

# WorldTokenizer sequences
_THINK_OPEN_WORLD  = [61, 35762, 63]    # <think>
_THINK_CLOSE_WORLD = [754, 35762, 63]   # </think>
# Byte-mode UTF-8 sequences
_THINK_OPEN_BYTES  = list("<think>".encode())   # [60,116,104,105,110,107,62]
_THINK_CLOSE_BYTES = list("</think>".encode())  # [60,47,116,104,105,110,107,62]


def _find_think_span(ids: List[int], open_seq: List[int], close_seq: List[int]):
    """Return (think_start, think_end) indices or (None, None)."""
    n, lo, lc = len(ids), len(open_seq), len(close_seq)
    think_start = think_end = None
    for i in range(n - lo + 1):
        if ids[i:i + lo] == open_seq:
            think_start = i + lo
    for i in range(n - lc + 1):
        if ids[i:i + lc] == close_seq:
            think_end = i
            break
    return think_start, think_end


def _entropy_reward(log_probs: List[float], output_ids: List[int],
                    alpha: float = 0.1, byte_mode: bool = False) -> float:
    """Entropy reduction inside think span.

    Positive reward when entropy (= −log_prob) decreases from think entry
    to think exit, i.e. model becomes more confident during state accumulation.
    """
    if not log_probs or not output_ids:
        return 0.0

    open_seq  = _THINK_OPEN_BYTES  if byte_mode else _THINK_OPEN_WORLD
    close_seq = _THINK_CLOSE_BYTES if byte_mode else _THINK_CLOSE_WORLD
    think_start, think_end = _find_think_span(output_ids, open_seq, close_seq)

    if think_start is None or think_end is None or think_end <= think_start:
        return 0.0

    span = log_probs[think_start:think_end]
    if len(span) < 2:
        return 0.0

    mid = len(span) // 2
    h_entry = -sum(span[:mid]) / mid
    h_exit  = -sum(span[mid:]) / (len(span) - mid)
    return alpha * (h_entry - h_exit)


# ── combined ──────────────────────────────────────────────────────────────────

def compute_rewards(
    group: RolloutGroup,
    clipo_head: Optional[torch.nn.Module] = None,
    clipo_weight: float = 1.0,
    entropy_weight: float = 1.0,
    tau: float = 0.05,
    byte_mode: bool = False,
) -> torch.Tensor:
    """Compute total reward for each rollout in the group.

    Returns shape [G] tensor.
    """
    G = len(group.rollouts)
    r_correct = torch.zeros(G)
    r_entropy = torch.zeros(G)

    for i, r in enumerate(group.rollouts):
        r_correct[i] = _score_correct(r.text, group.rubric)
        r_entropy[i] = _entropy_reward(r.log_probs, r.output_ids, byte_mode=byte_mode)

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
