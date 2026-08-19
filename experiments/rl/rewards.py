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
_NUMBER_TOKEN = re.compile(r"(?<!\d)\d+(?!\d)")
# Matches the exact family of digit rubrics in corpus_open/matrix_tasks.jsonl
# ("(?<!\d)N(?!\d)" for a target number N) — used to decide when the
# first-number anchor below applies, without touching word-lookahead
# rubrics ("(?=.*\bWORD\b)...") which have no single anchor position.
_IS_BARE_NUMBER_RUBRIC = re.compile(r"^\(\?<!\\d\)\d+\(\?!\\d\)$")


def _score_correct(text: str, rubric: dict) -> float:
    rtype = rubric.get("type")
    if rtype == "regex":
        pattern = rubric.get("value", "")
        m = re.search(pattern, text, re.IGNORECASE)
        if m and _IS_BARE_NUMBER_RUBRIC.match(pattern):
            # Found 2026-08-19: this rubric matches an isolated digit
            # ANYWHERE in the text — a rollout that echoes the input
            # matrix verbatim (itself a sequence of isolated 0/1 digits)
            # gets scored correct whenever the target digit happens to
            # appear in the echo, independent of whether the model
            # computed anything. Confirmed live: 45/153 correct=True
            # rollouts in g1i_real_run8 showed input-echo formatting.
            # Anchor to the FIRST number-like token in the whole text —
            # a genuine terse answer ("1", "= 1", "the answer is 1") has
            # its one real number there; an echoed multi-number matrix
            # row does not, since the target digit is buried after
            # earlier (wrong) numbers from the copied input.
            first_num = _NUMBER_TOKEN.search(text)
            if first_num is None or first_num.start() != m.start():
                m = None
        if m:
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
    beta: float = 0.005,
    gamma: float = 0.02,
    delta: float = 0.0,
    zeta: float = 0.0,
    stability_threshold: float = 1.5,
    density_eps: float = 0.05,
    gate_on_correct: bool = True,
) -> tuple:
    """Per-rollout reward for WKV-loop trajectories.

    r = r_correct − β·M − γ·Σ_t ReLU(H_t − H_{t-1}) [+ δ·stability_bonus]

    `beta`/`gamma` reduced from the original 0.02/0.1 on 2026-08-18 after a
    real `HACKING` monitor stop at step 20 of a real G1i training run:
    reward rose while accuracy fell over the trailing window. Two
    contributing mechanisms, both addressed by shrinking these weights:
    (1) genuine over-optimization of the effort/entropy terms at
    correctness's expense is plausible with the old weights large enough
    to matter; (2) a metric-divergence artifact — `TrainingMonitor`
    computes its accuracy from raw `r_correct>0` (ignoring the penalty
    terms), while `train_wkv_loop.py::main()`'s logged accuracy uses
    `reward>0` (penalty-inclusive) — with `beta·M + gamma·entropy_penalty`
    large enough to occasionally flip a correct answer's *reward* negative
    without touching `r_correct`, the two accuracy figures can diverge
    even with no real hacking behavior. Shrinking beta/gamma narrows that
    gap either way without removing the shaping terms entirely (they still
    encourage concise, stable reasoning, just no longer dominate over
    correctness).

    `gate_on_correct` (added 2026-08-19, default on): shaping only applies
    to rollouts that already got the task right. Found via a real
    bisected collapse (g1i_real_run6: 100% commit-at-M=2 at step20 → 100%
    M_max-saturated boilerplate by step50) that an *unconditional* shaping
    term is the likely cause — when raw accuracy is near zero, GRPO's
    within-group advantage is `(r - mean)/std`; if every rollout in a
    group is equally wrong (r_correct=-1 for all), the entire group's
    reward variance — and thus 100% of the gradient signal — comes from
    beta*M/gamma*entropy, i.e. the model is only ever taught to minimize
    effort, never to be more correct, during exactly the phase where it
    most needs correctness signal. Gating on r_correct>0 means a
    wrong-but-"efficient" rollout can never outscore a right-but-verbose
    one — shaping can only rank *among* already-correct answers, the same
    way a human gets "you could have done that more efficiently" feedback
    only after already succeeding, not instead of being told they're
    wrong. Confirmed no collapse through step49 of a beta=gamma=0 ablation
    (g1i_real_run7) resumed from the same step20 checkpoint; this flag is
    the next step — reintroduce shaping, but only where it can't
    substitute for correctness signal.

    `zeta` (added 2026-08-19, default 0.0 — grafted, not activated):
    information-density reward, user-proposed —
    ΔMI(S_t;Y) / (M · Σ‖ΔS_m‖). Real mutual information isn't computable
    per-step from a single rollout without an estimator (the codebase has
    one, dormant: `_infonce_reward` + `clipo_head.py`); this uses the
    cheap proxy already available — entropy *drop* across the M-loop
    (mirror of the gamma-penalized entropy *rise*) — divided by total WKV
    state motion (`r.wkv_stability`, `+ density_eps` so a near-frozen
    state doesn't send the ratio to infinity — that state is exactly what
    STATE_COL exists to catch, this must not reward it). Same
    `gate_on_correct` gate as beta/gamma/delta, for the identical reason:
    an ungated efficiency ratio could dominate GRPO's within-group
    gradient whenever a whole group is equally wrong. See
    `project_noesis_info_density_reward` memory for the full design
    discussion and why zeta defaults to 0.0 — meant to be swept in
    experiments/rl/reward_sweep.py before ever being turned on in a real
    training run, not guessed at.

    Args:
        rollouts: list of WKVLoopRollout (one per sample in the GRPO group).
        rubric: dict with "type" and "value" keys (same format as _score_correct).
        beta: step-count penalty coefficient.
        gamma: entropy-increase penalty coefficient.
        delta: WKV-stability bonus coefficient (0 = disabled).
        zeta: information-density reward coefficient (0 = disabled, see above).
        stability_threshold: mean wkv_stability below this → stable bonus.
        density_eps: denominator floor for the zeta term (avoids /~0 blowup).
        gate_on_correct: if True, beta/gamma/delta/zeta terms only apply to
            rollouts with r_correct > 0 (see above). If False, shaping
            applies unconditionally (pre-2026-08-19 behavior).

    Returns:
        rewards: float tensor [G]
        diag: dict with per-component tensors for logging:
              "r_correct", "r_effort", "r_entropy_penalty", "r_stability",
              "r_density", "M", "exit_reason"
    """
    G = len(rollouts)
    r_correct_t        = torch.zeros(G)
    r_effort_t         = torch.zeros(G)
    r_entropy_penalty_t = torch.zeros(G)
    r_stability_t      = torch.zeros(G)
    r_density_t        = torch.zeros(G)
    M_t                = torch.zeros(G, dtype=torch.long)
    exit_reasons: List[str] = []

    for i, r in enumerate(rollouts):
        r_correct_t[i] = _score_correct(r.text, rubric)
        M_t[i] = r.M

        # Shaping only ranks among already-correct rollouts (see
        # gate_on_correct docstring above) — a wrong rollout never gets
        # an effort/entropy/stability/density adjustment, so it can never
        # outscore a correct one on shaping alone.
        apply_shaping = (not gate_on_correct) or r_correct_t[i] > 0

        if apply_shaping:
            r_effort_t[i] = -beta * r.M

            # sum of entropy increases: Σ_t max(0, H_t - H_{t-1})
            traj = r.entropy_trajectory
            entropy_penalty = 0.0
            entropy_drop = 0.0
            for t in range(1, len(traj)):
                d = traj[t] - traj[t - 1]
                entropy_penalty += max(0.0, d)
                entropy_drop += max(0.0, -d)
            r_entropy_penalty_t[i] = -gamma * entropy_penalty

            if delta > 0.0 and len(r.wkv_stability) > 1:
                mean_stab = sum(r.wkv_stability[1:]) / len(r.wkv_stability[1:])
                if mean_stab < stability_threshold:
                    r_stability_t[i] = delta

            if zeta > 0.0 and r.M > 0:
                motion = sum(r.wkv_stability[1:]) if len(r.wkv_stability) > 1 else 0.0
                r_density_t[i] = zeta * entropy_drop / (r.M * motion + density_eps)

        exit_reasons.append(r.exit_reason)

    rewards = r_correct_t + r_effort_t + r_entropy_penalty_t + r_stability_t + r_density_t
    diag = {
        "r_correct":         r_correct_t,
        "r_effort":          r_effort_t,
        "r_entropy_penalty": r_entropy_penalty_t,
        "r_stability":       r_stability_t,
        "r_density":         r_density_t,
        "M":                 M_t,
        "exit_reason":       exit_reasons,
    }
    return rewards, diag
