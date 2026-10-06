#!/usr/bin/env python3
"""reward_sweep.py — sweep the shaping coefficients of
`rewards.py::compute_wkv_loop_rewards` over real rollout groups and report
what each one actually does to the GRPO gradient.

The file `rewards.py`'s own `zeta` docstring has promised this tool since
2026-08-19 ("meant to be swept in experiments/rl/reward_sweep.py before ever
being turned on in a real training run, not guessed at"). It was never
written, and `zeta` has never been non-zero in any run.

What this measures, and why it is not just "print the reward":

GRPO does not consume the reward. It consumes the *within-group advantage*,
`(r - mean(r)) / (std(r) + eps)` (`grpo.py::compute_advantages`). A reward
term that is CONSTANT across a group therefore contributes nothing beyond
rescaling the contrast that is already there — and under
`feed_mode="discrete"` (and `"expected"`/`"residual"` alike) the whole WKV
loop is a deterministic function of the prompt: prefill is deterministic,
the fed token is an argmax, and the exit criteria are thresholds. Sampling
only starts at the answer decode. So every rollout in a group walks the
SAME loop, and `M` / `entropy_trajectory` / `wkv_stability` — the only
inputs beta, gamma, delta and zeta have — are identical within the group by
construction.

This script makes that concrete rather than argued: it reports the
within-group spread of each shaping component, and the maximum change in
the advantage vector against the `zeta=0` baseline. `--synthetic` runs the
argument alone in a second, with no model; the default path generates real
rollouts so the magnitudes are real too.

The cross-prompt question is separate and is also reported: even if density
cannot separate rollouts WITHIN a group, does it separate prompts the model
gets right from prompts it gets wrong? That is the question of whether
information density is a marker of a good trajectory at all, and it is the
one that decides whether this reward is worth redesigning or dropping.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rl.grpo import compute_advantages
from experiments.rl.rewards import compute_wkv_loop_rewards
from experiments.rl.wkv_loop import WKVLoopRollout

DEFAULT_ZETAS = (0.0, 0.01, 0.05, 0.1, 0.5, 1.0)


def _spread(xs: Sequence[float]) -> float:
    return max(xs) - min(xs) if len(xs) > 1 else 0.0


def sweep_group(rollouts: List[WKVLoopRollout], rubric: dict, *,
                zetas: Sequence[float], beta: float, gamma: float,
                gate_on_correct: bool) -> dict:
    """One prompt's group across the zeta sweep. Returns a plain dict."""
    base_adv = None
    rows = []
    for z in zetas:
        rewards, diag = compute_wkv_loop_rewards(
            rollouts, rubric, beta=beta, gamma=gamma, zeta=z,
            gate_on_correct=gate_on_correct)
        adv = compute_advantages(rewards)
        if base_adv is None:
            base_adv = adv
        rows.append({
            "zeta": z,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "r_density_mean": float(diag["r_density"].mean()),
            # the number the whole argument turns on: how much the density
            # term varies BETWEEN rollouts that GRPO is asked to rank
            "r_density_within_group_spread": _spread(
                [float(x) for x in diag["r_density"]]),
            "advantage_max_abs_delta_vs_zeta0": float(
                (adv - base_adv).abs().max()),
        })
    return {
        "n_rollouts": len(rollouts),
        "n_correct": sum(1 for r in rollouts
                          if compute_wkv_loop_rewards([r], rubric, beta=0.0,
                                                      gamma=0.0)[0][0].item() > 0),
        "M_values": [r.M for r in rollouts],
        "M_within_group_spread": _spread([r.M for r in rollouts]),
        "motion_values": [round(sum(r.wkv_stability[1:]), 4) for r in rollouts],
        "entropy_drop_values": [
            round(sum(max(0.0, r.entropy_trajectory[t - 1] - r.entropy_trajectory[t])
                      for t in range(1, len(r.entropy_trajectory))), 4)
            for r in rollouts],
        "exit_reasons": [r.exit_reason for r in rollouts],
        "sweep": rows,
    }


def synthetic_groups(n_groups: int = 3, group_size: int = 4) -> list:
    """Groups that differ ONLY in answer text, sharing one loop trace.

    This is not a stand-in for real data — it is the exact structure the
    real generator produces (one deterministic loop per prompt, G sampled
    answers off its final state), reduced to the part the reward sees. It
    exists so the invariance claim can be checked without a model.
    """
    out = []
    for g in range(n_groups):
        traj = [4.0, 3.2, 2.9, 2.85]           # a plausible entropy decay
        stab = [0.0, 1.8, 0.9, 0.5]            # ||dWKV|| per loop step
        rolls = []
        for i in range(group_size):
            rolls.append(WKVLoopRollout(
                prompt_ids=[1, 2, 3], answer_ids=[9],
                M=len(traj), entropy_trajectory=list(traj),
                wkv_stability=list(stab), exit_reason="plateau",
                answer_log_probs=[-0.1],
                # half the group right, half wrong — the contrast GRPO ranks
                text=("42" if i < group_size // 2 else "not it"),
                prompt_text=f"synthetic-{g}"))
        out.append((f"synthetic-{g}", rolls, {"type": "exact", "value": "42"}))
    return out


def generate_groups(model_path: str, tasks_path: Path, *, n_tasks: int,
                    group_size: int, device: str, m_max: int,
                    max_answer_tokens: int, temperature: float) -> list:
    from experiments.rl.loader import load_rwkv7
    from experiments.rl.wkv_loop import generate_rollout

    tasks = []
    with open(tasks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    tasks = tasks[:n_tasks]

    print(f"[reward_sweep] loading {model_path} on {device} (blink)", flush=True)
    loaded = load_rwkv7(model_path, device=device, backend="blink")

    out = []
    for t in tasks:
        rolls = []
        for i in range(group_size):
            r = generate_rollout(loaded, t["prompt"], feed_mode="discrete",
                                 M_max=m_max, max_answer_tokens=max_answer_tokens,
                                 answer_temperature=temperature)
            rolls.append(r)
            print(f"  [{t['id']}] rollout {i+1}/{group_size} "
                  f"M={r.M} exit={r.exit_reason} text={r.text[:40]!r}", flush=True)
        out.append((t["id"], rolls, t["rubric"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None,
                     help="Checkpoint to generate rollouts from. Omit with "
                          "--synthetic.")
    ap.add_argument("--tasks", type=Path,
                     default=Path("experiments/A0_eval/tasks.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--synthetic", action="store_true",
                     help="Skip generation; use structurally-identical stub "
                          "groups. Checks the invariance claim only.")
    ap.add_argument("--n-tasks", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--m-max", type=int, default=16)
    ap.add_argument("--max-answer-tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--beta", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--gate-on-correct", default="on", choices=("on", "off"))
    ap.add_argument("--zetas", default=",".join(str(z) for z in DEFAULT_ZETAS))
    args = ap.parse_args()

    zetas = tuple(float(z) for z in args.zetas.split(","))
    gate = args.gate_on_correct == "on"

    if args.synthetic:
        groups = synthetic_groups()
        model_name = "synthetic"
    else:
        if args.model is None:
            ap.error("--model is required unless --synthetic")
        groups = generate_groups(args.model, args.tasks, n_tasks=args.n_tasks,
                                  group_size=args.group_size, device=args.device,
                                  m_max=args.m_max,
                                  max_answer_tokens=args.max_answer_tokens,
                                  temperature=args.temperature)
        model_name = args.model

    per_group = {}
    for name, rolls, rubric in groups:
        per_group[name] = sweep_group(rolls, rubric, zetas=zetas,
                                       beta=args.beta, gamma=args.gamma,
                                       gate_on_correct=gate)

    max_adv_delta = max(
        row["advantage_max_abs_delta_vs_zeta0"]
        for g in per_group.values() for row in g["sweep"])
    max_density_spread = max(
        row["r_density_within_group_spread"]
        for g in per_group.values() for row in g["sweep"])
    max_M_spread = max(g["M_within_group_spread"] for g in per_group.values())

    payload = {
        "model": model_name,
        "gate_on_correct": gate,
        "beta": args.beta, "gamma": args.gamma, "zetas": list(zetas),
        "groups": per_group,
        "_summary": {
            "max_advantage_delta_vs_zeta0": f"{max_adv_delta:.3e}",
            "max_within_group_density_spread": f"{max_density_spread:.3e}",
            "max_within_group_M_spread": str(max_M_spread),
        },
    }

    from experiments._common.results import save_result
    save_result(args.out, payload, experiment="reward_sweep",
                hypothesis=["H25"], model=model_name,
                script="experiments/rl/reward_sweep.py")

    print("\n=== summary ===")
    print(f"within-group M spread (max over groups):        {max_M_spread}")
    print(f"within-group r_density spread (max):            {max_density_spread:.3e}")
    print(f"advantage change vs zeta=0 (max over sweep):    {max_adv_delta:.3e}")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
