"""monitor.py — training health monitors for WKV-loop RL.

Six emergency-stop conditions, checked after each GRPO batch:

    SHORTCUT   model exits commit in ≤1 step on >80% of rollouts
               (collapsed to memorized token, not reasoning)
    NO_COMMIT  model never exits early — >90% of rollouts hit M_max
               (mirror of SHORTCUT: the loop stopped trusting its own
               commit/plateau criteria at all, not that it's using them
               too eagerly. Found 2026-08-19: g1i_real_run6 collapsed
               from 100% commit-at-M=2 to 100% M_max between step20 and
               step50, invisible to every other flag — HACKING needs
               reward to rise, which it won't while beta*M is climbing;
               MODE_COL needs near-identical text, but the degenerate
               output here is boilerplate that varies rollout to
               rollout, just never engages with the actual task.)
    ECHO       answer text substantially overlaps with the prompt on
               >echo_frac of rollouts (copying the input instead of
               answering it). Found 2026-08-19: g1i_real_run8 step106 —
               a rubric regex bug let echoed-matrix rollouts score
               correct=True (fixed separately in rewards.py::
               _score_correct), but the underlying behavior — the model
               copying its input instead of computing an answer — is a
               real failure mode rubric fixes alone don't catch for
               every rubric family (e.g. word-lookahead rubrics, where
               a genuine correct answer can *also* legitimately overlap
               the prompt). This flag watches the behavior directly,
               independent of whether any particular rubric happens to
               be exploitable by it.
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


def _is_prompt_echo(text: str, prompt: str, min_overlap: int = 15) -> bool:
    """True if `text` contains a run of `min_overlap`+ characters that also
    appears verbatim in `prompt` — the answer is (at least partly) a copy
    of the input, not a real response. Whitespace-normalized so line
    breaks/padding differences don't hide a real echo.

    A genuine terse answer basically never shares a 15+ character run
    with a multi-line prompt by chance; a copied matrix row or repeated
    prompt sentence does, by construction.
    """
    text_n = " ".join(text.split())
    if len(text_n) < min_overlap:
        return False
    prompt_n = " ".join(prompt.split())
    for i in range(len(text_n) - min_overlap + 1):
        if text_n[i:i + min_overlap] in prompt_n:
            return True
    return False


class TrainingMonitor:
    """Stateful monitor that tracks per-batch health across training steps.

    All thresholds are keyword-configurable so ablations can widen/tighten them
    without subclassing.
    """

    def __init__(
        self,
        *,
        shortcut_commit1_frac: float = 0.80,   # SHORTCUT: fraction threshold
        no_commit_frac: float = 0.90,           # NO_COMMIT: fraction hitting M_max
        echo_frac: float = 0.30,                # ECHO: fraction overlapping prompt
        echo_min_overlap: int = 15,              # ECHO: min shared character run
        hack_window: int = 10,                  # HACKING: look-back batches
        hack_reward_delta: float = 0.05,        # mean reward must rise by this
        hack_acc_delta: float = -0.02,          # accuracy may not drop more than this
        state_col_eps: float = 0.01,            # STATE_COL: mean stability floor
        diversity_threshold: float = 0.20,      # MODE_COL: unique text fraction
    ):
        self.shortcut_commit1_frac = shortcut_commit1_frac
        self.no_commit_frac = no_commit_frac
        self.echo_frac = echo_frac
        self.echo_min_overlap = echo_min_overlap
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

        # ── NO_COMMIT ────────────────────────────────────────────────
        exit_reasons = diag.get("exit_reason")
        if exit_reasons:
            m_max_frac = sum(1 for e in exit_reasons if e == "M_max") / len(exit_reasons)
            if m_max_frac >= self.no_commit_frac:
                flags.append("NO_COMMIT")

        # ── ECHO ─────────────────────────────────────────────────────
        echo_count = sum(
            1 for r in rollouts
            if r.prompt_text and _is_prompt_echo(
                r.text, r.prompt_text, self.echo_min_overlap)
        )
        if echo_count / G >= self.echo_frac:
            flags.append("ECHO")

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

    # ------------------------------------------------------------------
    # Checkpoint round-trip

    def state_dict(self) -> dict:
        return {
            "reward_history": list(self._reward_history),
            "acc_history": list(self._acc_history),
        }

    def load_state_dict(self, state: dict) -> None:
        # Found 2026-08-19: without this, every resume started HACKING's
        # 10-batch window from empty — two separate resumes from the same
        # ckpt_step000020 both tripped HACKING within ~17-18 steps of
        # restarting (a freshly-filling small-sample window, not
        # necessarily a real reward/accuracy divergence). Restoring the
        # history means the window reflects real training history across
        # a resume, same reasoning as CorpusScheduler's state_dict.
        self._reward_history = deque(state["reward_history"], maxlen=self.hack_window)
        self._acc_history = deque(state["acc_history"], maxlen=self.hack_window)
