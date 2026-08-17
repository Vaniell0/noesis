#!/usr/bin/env python3
"""sweep_alpha.py — micro-sweep α ∈ {0, 0.3, 0.7, 1.0} on word-search L1-L3.

Tests whether the residual-delta feed_mode outperforms the baseline
expected-embedding mode as α increases. Task #10 in the RL rewrite plan.

Feed modes by α:
    α = 0.0  → feed_mode="expected"   (pure expected embedding, differentiable)
    α > 0.0  → feed_mode="residual"   (expected + α·MLP_delta, differentiable)

Both modes require peft backend (GPU). Discrete baseline (CPU-compatible) is
included as α = None for reference.

Sweep matrix:
    α ∈ {None, 0.0, 0.3, 0.7, 1.0}  ×  N_PROMPTS tasks from L1–L3

Metrics per cell:
    accuracy        — fraction of rollouts with r_correct > 0
    mean_M          — mean WKV loop steps
    mean_H_delta    — mean entropy increase penalty (Σ ReLU(ΔH_t))
    exit_reason     — distribution (commit / plateau / M_max)
    sr_L4           — mean stable_rank at layer 4 (reasoning probe)

Results saved to experiments/rl/runs/alpha_sweep/results.jsonl

Usage (GPU required for α ≥ 0):
    python3 experiments/rl/sweep_alpha.py \
        --model ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
        --n-prompts 32 --G 4 --out experiments/rl/runs/alpha_sweep
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rl.corpus import load_corpus
from experiments.rl.loader import load_rwkv7
from experiments.rl.wkv_loop import generate_rollout, WKVLoopRollout
from experiments.rl.rewards import compute_wkv_loop_rewards
from experiments.rl.probes import stable_rank_probe


ALPHA_VALUES = [None, 0.0, 0.3, 0.7, 1.0]   # None = discrete (CPU baseline)

CORPUS_PATH = ROOT / "training/corpus_open/matrix_tasks.jsonl"


# ------------------------------------------------------------------
# Small MLP_delta for residual mode

class _MLPDelta(nn.Module):
    """Two-layer MLP that outputs a residual in embedding space.

    Input/output dim = model embedding dim D. Hidden dim = D // 4.
    Initialized near zero so α=0 starts from the same embedding as "expected".
    """
    def __init__(self, D: int):
        super().__init__()
        H = max(64, D // 4)
        self.net = nn.Sequential(
            nn.Linear(D, H),
            nn.GELU(),
            nn.Linear(H, D),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------------

def _rollout_cell(
    loaded,
    prompts: List[dict],
    G: int,
    alpha: Optional[float],
    mlp_delta: Optional[nn.Module],
    M_max: int,
    max_answer_tokens: int,
) -> List[List[WKVLoopRollout]]:
    """G rollouts for each prompt. Returns list[prompt] of list[rollout]."""
    if alpha is None:
        feed_mode = "discrete"
    elif alpha == 0.0:
        feed_mode = "expected"
    else:
        feed_mode = "residual"

    results = []
    for task in prompts:
        group = []
        for _ in range(G):
            r = generate_rollout(
                loaded,
                task["prompt"],
                feed_mode=feed_mode,
                M_max=M_max,
                tau_commit=0.90,
                eps_plateau=0.02,
                max_answer_tokens=max_answer_tokens,
                answer_temperature=0.7,
                mlp_delta=mlp_delta,
                alpha=alpha if alpha is not None else 0.0,
                eos_id=0,
            )
            group.append(r)
        results.append(group)
    return results


def _cell_metrics(
    groups: List[List[WKVLoopRollout]],
    rubrics: List[dict],
) -> dict:
    all_rollouts = [r for g in groups for r in g]
    rubric_map = {i: rubrics[i] for i in range(len(rubrics))}

    correct_total = 0
    M_vals = []
    H_delta_vals = []
    exit_counts: dict = {}

    for i, (group, rubric) in enumerate(zip(groups, rubrics)):
        _, diag = compute_wkv_loop_rewards(group, rubric)
        correct_total += int((diag["r_correct"] > 0).sum().item())
        M_vals.extend(diag["M"].tolist())
        for r in group:
            traj = r.entropy_trajectory
            h_delta = sum(max(0.0, traj[t] - traj[t-1]) for t in range(1, len(traj)))
            H_delta_vals.append(h_delta)
        for reason in diag["exit_reason"]:
            exit_counts[reason] = exit_counts.get(reason, 0) + 1

    n = len(all_rollouts)
    return {
        "accuracy":    correct_total / n if n else 0.0,
        "mean_M":      sum(M_vals) / len(M_vals) if M_vals else 0.0,
        "mean_H_delta": sum(H_delta_vals) / len(H_delta_vals) if H_delta_vals else 0.0,
        "exit_reason": exit_counts,
        "n_rollouts":  n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n-prompts", type=int, default=32,
                    help="Tasks sampled from L1-L3 per alpha cell")
    ap.add_argument("--G", type=int, default=4,
                    help="Rollouts per prompt")
    ap.add_argument("--M-max", type=int, default=16)
    ap.add_argument("--max-answer", type=int, default=24)
    ap.add_argument("--levels", type=str, default="1,2,3",
                    help="Comma-separated curriculum levels to sample from")
    ap.add_argument("--alphas", type=str, default="",
                    help="Override alpha list (e.g. '0.0,0.3,0.7,1.0'). "
                         "Empty = use default including discrete baseline.")
    ap.add_argument("--out", type=str,
                    default="experiments/rl/runs/alpha_sweep")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    # Alpha list
    if args.alphas:
        alphas: List[Optional[float]] = []
        for a in args.alphas.split(","):
            a = a.strip()
            alphas.append(None if a == "discrete" else float(a))
    else:
        alphas = ALPHA_VALUES

    # Validate: expected/residual need peft backend
    non_discrete = [a for a in alphas if a is not None]
    if non_discrete and args.device == "cpu":
        print(
            "[sweep] WARNING: expected/residual modes require peft backend (GPU). "
            "On CPU only α=None (discrete) will work. "
            "Run this script on Selectel 4090."
        )

    # Corpus
    levels = [int(l) for l in args.levels.split(",")]
    sched = load_corpus(str(CORPUS_PATH), start_level=max(levels), rng_seed=42)
    # Sample enough for the sweep
    batch = sched.sample_batch(args.n_prompts * 4)
    # Filter to requested levels
    batch = [t for t in batch if t.get("level", 1) in levels][:args.n_prompts]
    if len(batch) < args.n_prompts:
        print(f"[sweep] only {len(batch)} tasks at levels {levels} — using all")

    rubrics = [t["rubric"] for t in batch]

    # Load model
    backend = "peft" if args.device != "cpu" else "blink"
    loaded = load_rwkv7(args.model, device=args.device, backend=backend)

    # MLP_delta (shared across alpha cells for fair comparison)
    D = loaded.embedding_weight.shape[1]
    mlp_delta = _MLPDelta(D).to(args.device) if non_discrete else None

    # Run sweep
    all_results = []
    for alpha in alphas:
        label = "discrete" if alpha is None else f"alpha={alpha:.1f}"
        print(f"\n[sweep] cell {label}  n_prompts={len(batch)}  G={args.G}")

        if alpha is not None and backend == "blink":
            print(f"  [SKIP] {label} requires peft backend — no GPU")
            entry = {"alpha": alpha, "label": label, "skipped": True}
        else:
            groups = _rollout_cell(
                loaded, batch, args.G, alpha, mlp_delta,
                M_max=args.M_max, max_answer_tokens=args.max_answer,
            )
            metrics = _cell_metrics(groups, rubrics)

            # SR probe (cheap, runs on blink too)
            sr = stable_rank_probe(loaded, layers=[4, 16])
            sr_L4_mean = sum(
                v.get(4, float("nan")) for v in sr.values()
            ) / len(sr) if sr else float("nan")

            entry = {
                "alpha":   alpha,
                "label":   label,
                "skipped": False,
                **metrics,
                "sr_L4_mean": sr_L4_mean,
            }

        print(f"  {json.dumps({k: v for k, v in entry.items() if k not in ('exit_reason',)})}")
        all_results.append(entry)
        with open(results_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # Summary table
    print("\n=== ALPHA SWEEP SUMMARY ===")
    print(f"{'label':<20} {'accuracy':>10} {'mean_M':>8} {'mean_H_delta':>13}")
    for e in all_results:
        if e.get("skipped"):
            print(f"{e['label']:<20} {'(skipped)':>10}")
        else:
            print(f"{e['label']:<20} {e['accuracy']:>10.3f} {e['mean_M']:>8.2f} {e['mean_H_delta']:>13.3f}")

    print(f"\nResults → {results_path}")


if __name__ == "__main__":
    main()
