#!/usr/bin/env python3
"""rollout.py — GRPO rollout generation for RWKV-7.

Generates G completions per prompt using the rwkv inference model (no grad).
Captures WKV state at the first </think> boundary for CLIPO reward.

Token constants (WorldTokenizer):
    THINK_OPEN  = [61, 35762, 63]   # <think>
    THINK_CLOSE = [754, 35762, 63]  # </think>

Usage:
    from experiments.rl.rollout import generate_rollouts, RolloutGroup
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).parents[2] / "experiments/A0_state_probe"))

THINK_CLOSE = [754, 35762, 63]  # </think>
EOS_ID = 0


@dataclass
class Rollout:
    prompt_ids: List[int]
    output_ids: List[int]              # generated tokens only
    log_probs: List[float]             # per-token log π_old
    wkv_state_think: Optional[torch.Tensor] = None  # state at </think>, shape [n_layers, H, h, h]
    text: str = ""


@dataclass
class RolloutGroup:
    task_id: str
    prompt: str
    answer: str
    rubric: dict
    rollouts: List[Rollout] = field(default_factory=list)


def _detect_think_close(ids: List[int], pattern: List[int] = THINK_CLOSE) -> bool:
    if len(ids) < len(pattern):
        return False
    return ids[-len(pattern):] == pattern


def generate_rollouts(
    model,
    tokenizer,
    tasks: List[dict],
    G: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: str = "cpu",
) -> List[RolloutGroup]:
    """Generate G rollouts per task. Returns one RolloutGroup per task.

    model: rwkv inference model (model.forward(ids, state) → logits, state)
    tokenizer: has .encode(str) → List[int] and .decode(List[int]) → str
    tasks: list of task dicts with keys: id, prompt, answer, rubric
    """
    groups: List[RolloutGroup] = []

    for task in tasks:
        prompt_ids = tokenizer.encode(task["prompt"])
        group = RolloutGroup(
            task_id=task["id"],
            prompt=task["prompt"],
            answer=task.get("answer", ""),
            rubric=task.get("rubric", {}),
        )

        for _ in range(G):
            rollout = _single_rollout(
                model, tokenizer, prompt_ids,
                max_new_tokens, temperature, top_p, device,
            )
            group.rollouts.append(rollout)

        groups.append(group)

    return groups


def _single_rollout(
    model,
    tokenizer,
    prompt_ids: List[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> Rollout:
    """One forward rollout. Captures WKV state at first </think> boundary."""
    with torch.no_grad():
        # Prefill
        logits, state = model.forward(prompt_ids, None)

        output_ids: List[int] = []
        log_probs: List[float] = []
        wkv_state_think: Optional[torch.Tensor] = None
        close_buf: List[int] = []

        for _ in range(max_new_tokens):
            next_id = _sample(logits, temperature, top_p)
            lp = float(torch.log_softmax(logits if logits.dim() == 1
                                         else logits[-1], dim=-1)[next_id])
            output_ids.append(next_id)
            log_probs.append(lp)
            close_buf.append(next_id)
            if len(close_buf) > len(THINK_CLOSE):
                close_buf.pop(0)

            if wkv_state_think is None and _detect_think_close(close_buf):
                wkv_state_think = _extract_wkv(state)

            if next_id == EOS_ID:
                break

            logits, state = model.forward([next_id], state)

        text = tokenizer.decode(output_ids)
        return Rollout(
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            log_probs=log_probs,
            wkv_state_think=wkv_state_think,
            text=text,
        )


def _sample(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if logits.dim() > 1:
        logits = logits[-1]
    if temperature == 0.0:
        return int(logits.argmax().item())
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumprobs = torch.cumsum(sorted_probs, dim=0)
    cutoff = (cumprobs - sorted_probs > top_p).nonzero()
    if cutoff.numel():
        sorted_probs[cutoff[0].item():] = 0.0
    sorted_probs /= sorted_probs.sum()
    chosen = int(torch.multinomial(sorted_probs, 1).item())
    return int(sorted_ids[chosen].item())


def _extract_wkv(state) -> Optional[torch.Tensor]:
    """Extract WKV matrices from RWKV state list.

    State layout: [shift_x, wkv_state, shift_ffn] per layer × n_layers.
    WKV state is at index 3*L+1, shape [n_head, head_size, head_size].
    Returns stacked tensor [n_layers, n_head, head_size, head_size] or None.
    """
    if state is None:
        return None
    try:
        wkv_layers = []
        for i in range(1, len(state), 3):
            s = state[i]
            if s is not None and isinstance(s, torch.Tensor):
                wkv_layers.append(s.detach().cpu())
        if wkv_layers:
            return torch.stack(wkv_layers, dim=0)
    except Exception:
        pass
    return None
