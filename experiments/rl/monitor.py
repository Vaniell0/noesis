"""monitor.py — training health monitors for WKV-loop RL.

Four emergency-stop conditions, checked after each GRPO batch:

    SHORTCUT   model exits commit in ≤1 step on >80% of rollouts
               (collapsed to memorized token, not reasoning)
    HACKING    mean_reward↑ but accuracy flat or declining for N batches
    STATE_COL  mean(wkv_stability) < eps_state for entire batch
               (WKV state not updating → silent freeze)
    MODE_COL   fraction of unique decoded texts < diversity_threshold
               (all rollouts produce same output)

Usage:
    from experiments.rl.monitor import TrainingMonitor
    mon = TrainingMonitor()
    stop, flags = mon.step(rollouts, rewards, diag)
    if stop:
        raise RuntimeError(f"Emergency stop: {flags}")
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import torch

from experiments.rl.wkv_loop import WKVLoopRollout


class TrainingMonitor:
    """Stateful monitor that tracks per-batch health across training steps.

    All thresholds are keyword-configurable so ablations can widen/tighten them
    without subclassing.
    """

    def __init__(
        self,
        *,
        shortcut_commit1_frac: float = 0.80,   # SHORTCUT: fraction threshold
        hack_window: int = 10,                  # HACKING: look-back batches
        hack_reward_delta: float = 0.05,        # mean reward must rise by this
        hack_acc_delta: float = -0.02,          # accuracy may not drop more than this
        state_col_eps: float = 0.01,            # STATE_COL: mean stability floor
        diversity_threshold: float = 0.20,      # MODE_COL: unique text fraction
    ):
        self.shortcut_commit1_frac = shortcut_commit1_frac
        self.hack_window = hack_window
        self.hack_reward_delta = hack_reward_delta
        self.hack_acc_delta = hack_acc_delta
        self.state_col_eps = state_col_eps
        self.diversity_threshold = diversity_threshold

        self._reward_history: deque = deque(maxlen=hack_window)
        self._acc_history: deque = deque(maxlen=hack_window)

    # ------------------------------------------------------------------

    def step(
        self,
        rollouts: List[WKVLoopRollout],
        rewards: torch.Tensor,
        diag: Dict,
    ) -> Tuple[bool, List[str]]:
        """Check health for one batch.

        Args:
            rollouts: list of WKVLoopRollout from this batch.
            rewards:  float tensor [G] (total reward per rollout).
            diag:     diagnostics dict from compute_wkv_loop_rewards.

        Returns:
            (stop, flags) — stop=True means training should halt;
            flags is a list of triggered condition names.
        """
        flags: List[str] = []
        G = len(rollouts)

        # ── SHORTCUT ──────────────────────────────────────────────────
        commit1 = sum(
            1 for r in rollouts
            if r.exit_reason == "commit" and r.M <= 1
        )
        if commit1 / G >= self.shortcut_commit1_frac:
            flags.append("SHORTCUT")

        # ── STATE_COL ────────────────────────────────────────────────
        stab_means = []
        for r in rollouts:
            stab = r.wkv_stability
            if len(stab) > 1:
                stab_means.append(sum(stab[1:]) / len(stab[1:]))
        if stab_means and (sum(stab_means) / len(stab_means)) < self.state_col_eps:
            flags.append("STATE_COL")

        # ── MODE_COL ─────────────────────────────────────────────────
        texts = [r.text for r in rollouts]
        unique_frac = len(set(texts)) / G
        if unique_frac < self.diversity_threshold:
            flags.append("MODE_COL")

        # ── HACKING ──────────────────────────────────────────────────
        mean_reward = float(rewards.mean().item())
        r_correct = diag.get("r_correct")
        accuracy = float((r_correct > 0).float().mean().item()) if r_correct is not None else None

        self._reward_history.append(mean_reward)
        if accuracy is not None:
            self._acc_history.append(accuracy)

        if (
            len(self._reward_history) == self.hack_window
            and len(self._acc_history) == self.hack_window
        ):
            half = self.hack_window // 2
            rh = list(self._reward_history)
            ah = list(self._acc_history)
            reward_rose = (sum(rh[half:]) / half) - (sum(rh[:half]) / half) > self.hack_reward_delta
            acc_dropped = (sum(ah[half:]) / half) - (sum(ah[:half]) / half) < self.hack_acc_delta
            if reward_rose and acc_dropped:
                flags.append("HACKING")

        stop = bool(flags)
        return stop, flags

    def diagnostics(self) -> Dict:
        """Latest rolling statistics (for logging)."""
        rh = list(self._reward_history)
        ah = list(self._acc_history)
        return {
            "mean_reward_rolling": sum(rh) / len(rh) if rh else float("nan"),
            "accuracy_rolling":    sum(ah) / len(ah) if ah else float("nan"),
            "history_len":         len(rh),
        }
