#!/usr/bin/env python3
"""grpo.py — GRPO advantage computation and policy update for RWKV-7.

GRPO (Group Relative Policy Optimisation):
    advantage_i = (reward_i - mean(rewards)) / std(rewards)
    ratio_i     = exp(log_pi_theta(a_i) - log_pi_old(a_i))
    loss_i      = -min(ratio_i * adv_i, clip(ratio_i, 1±ε) * adv_i)

For RWKV-7 the policy ratio is defined only over visible token positions.
Latent positions (future Phase 4 with <latent> tokens) contribute nothing.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from experiments.rl.rollout import RolloutGroup


def compute_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalise rewards within a group to get advantages. Shape [G]."""
    mean = rewards.mean()
    std  = rewards.std() + eps
    return (rewards - mean) / std


def recompute_log_probs(
    model,
    prompt_ids: List[int],
    output_ids: List[int],
) -> torch.Tensor:
    """Recompute per-token log π_θ(a_t | context) with gradients.

    Runs a single forward pass over prompt + output, returns log-probs
    for output positions only. Shape [len(output_ids)].

    model: RWKV-PEFT model with gradient support.
    """
    all_ids = prompt_ids + output_ids
    # Forward over full sequence; logits[t] predicts token at t+1
    logits, _ = model.forward(all_ids, None)  # [T, vocab] or [vocab] for last
    if logits.dim() == 1:
        # model returns only last logit — need sequential forward
        return _sequential_log_probs(model, prompt_ids, output_ids)

    # logits[i] → distribution for position i+1
    # output starts at position len(prompt_ids)
    offset = len(prompt_ids) - 1
    out_logits = logits[offset: offset + len(output_ids)]  # [G_out, vocab]
    out_ids_t  = torch.tensor(output_ids, dtype=torch.long)
    log_probs  = F.log_softmax(out_logits, dim=-1)
    return log_probs[torch.arange(len(output_ids)), out_ids_t]


def _sequential_log_probs(
    model, prompt_ids: List[int], output_ids: List[int]
) -> torch.Tensor:
    """Fallback: token-by-token forward to get log-probs with gradients."""
    logits, state = model.forward(prompt_ids, None)
    lps = []
    for tok_id in output_ids:
        lp = F.log_softmax(logits if logits.dim() == 1 else logits[-1], dim=-1)
        lps.append(lp[tok_id])
        logits, state = model.forward([tok_id], state)
    return torch.stack(lps)


def grpo_loss(
    model,
    groups: List[RolloutGroup],
    rewards_per_group: List[torch.Tensor],
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
    h12bi_weight: float = 0.0,
) -> torch.Tensor:
    """Compute GRPO loss over a batch of rollout groups.

    rewards_per_group: list of [G] tensors, one per group.
    h12bi_weight: H12b.i LoRA rank-entropy aux loss weight (0 = disabled).
                  No-op if model has no lora_A/lora_B parameters.
    Returns scalar loss.
    """
    total_loss = torch.tensor(0.0, requires_grad=True)
    n_tokens = 0

    for group, rewards in zip(groups, rewards_per_group):
        advantages = compute_advantages(rewards)  # [G]

        for i, (rollout, adv) in enumerate(
            zip(group.rollouts, advantages.tolist())
        ):
            if not rollout.output_ids:
                continue

            log_pi_theta = recompute_log_probs(
                model, rollout.prompt_ids, rollout.output_ids
            )  # [T_out]
            log_pi_old = torch.tensor(
                rollout.log_probs[:len(rollout.output_ids)],
                dtype=torch.float32,
            )

            ratio = torch.exp(log_pi_theta - log_pi_old)
            adv_t = torch.full_like(ratio, adv)

            unclipped = ratio * adv_t
            clipped   = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
            surrogate = -torch.min(unclipped, clipped).mean()

            # KL penalty (approximate: ratio - 1 - log_ratio)
            log_ratio = log_pi_theta - log_pi_old
            kl = (ratio - 1 - log_ratio).mean()

            total_loss = total_loss + surrogate + kl_coef * kl
            n_tokens += len(rollout.output_ids)

    policy_loss = total_loss / max(n_tokens, 1)

    # H12b.i: LoRA rank-entropy regulariser (prevents rank collapse during RL).
    # Requires LoRA parameters named *lora_A / *lora_B in the training model.
    if h12bi_weight > 0.0:
        import sys
        from pathlib import Path
        _root = Path(__file__).parents[2]
        if str(_root / "training") not in sys.path:
            sys.path.insert(0, str(_root / "training"))
        try:
            from state_reg import compute_h12bi_aux
            lora_A = {n[:-len("lora_A")]: p
                      for n, p in model.named_parameters()
                      if n.endswith("lora_A")}
            lora_B = {n[:-len("lora_B")]: p
                      for n, p in model.named_parameters()
                      if n.endswith("lora_B")}
            lora_pairs = [(a, lora_B[k]) for k, a in lora_A.items() if k in lora_B]
            if lora_pairs:
                policy_loss = policy_loss + h12bi_weight * compute_h12bi_aux(lora_pairs)
        except ImportError:
            pass  # state_reg not on path — skip silently

    return policy_loss
