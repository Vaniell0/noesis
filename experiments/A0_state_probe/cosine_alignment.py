#!/usr/bin/env python3
"""cosine_alignment.py — does a training delta point into NEW weight-space
directions (expanding used volume) or REDUNDANT ones (reinforcing existing
dominant modes)? Compares a delta's (trained - base) top singular vectors
against the BASE matrix's own top singular vectors via cosine similarity.

Redesigned 2026-09-04 after the first version's result (mean_best_cos~0.04)
was found uninterpretable on its own: an ad-hoc random-rank-32 control gave
~0.035, statistically indistinguishable from the real trained delta. In a
2560-dim ambient space, even two UNRELATED generic directions have nonzero
best-of-k cosine purely from dimensionality — a raw score has no meaning
without a null distribution matched to the same shape. This version builds
that null in: for each matrix, draws `--n-random` random Gaussian matrices
of the SAME shape as the delta, computes each one's own top-k singular
vectors, and reports the real delta's cosine score as a z-score against
that null's mean/std. z~0 means the observed alignment (or lack of it) is
exactly what high-dimensional geometry predicts for a task-irrelevant
direction — not evidence either way. |z| large means the delta's relation
to base's dominant modes is a real, non-geometric effect.

Complements lora_rank_analysis.py (which reports rank/energy of the delta
alone, no vectors) and the base-weight effective_rank run (which showed
base att_proj/ffn are near-full-rank, i.e. NOT structurally bottlenecked) —
this answers a different question: even with plenty of base-matrix room
available, did training actually USE fresh room, or double down on what was
already dominant — and now, whether that answer is distinguishable from
chance at all.

Usage:
    python cosine_alignment.py \
        --base ~/.libs/models/rwkv7/rwkv7-g1h-2.9b-20260710-ctx10240.pth \
        --trained ~/.libs/models/rwkv7/rwkv-step9b-e1.pth \
        --keys blocks.0.att.key.weight,blocks.0.att.value.weight,blocks.0.att.receptance.weight \
        --k 8 --n-random 30 --seed 0 \
        --out results/cosine_alignment_step9b_e1.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common.results import save_result


def _mean_best_cos(top_a: torch.Tensor, top_b: torch.Tensor) -> float:
    """Mean over top_a's directions of the best-matching |cos| against any
    of top_b's directions (not forced index-to-index pairing)."""
    cos = (top_a.T @ top_b).abs()  # [k, k]
    return cos.max(dim=1).values.mean().item()


def _random_null(shape, base_top_k: torch.Tensor, k: int, n_trials: int,
                  generator: torch.Generator) -> List[float]:
    samples = []
    for _ in range(n_trials):
        r = torch.randn(*shape, generator=generator)
        Ur, _, _ = torch.linalg.svd(r, full_matrices=False)
        samples.append(_mean_best_cos(Ur[:, :k], base_top_k))
    return samples


def cosine_alignment(base_path: str, trained_path: str, keys: List[str], k: int,
                      n_random: int = 30, seed: int = 0) -> Dict:
    print("Loading checkpoints...")
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    trained = torch.load(trained_path, map_location="cpu", weights_only=True)
    generator = torch.Generator().manual_seed(seed)

    rows = []
    for key in keys:
        b = base[key].float()
        delta = trained[key].float() - b
        Ub, Sb, _ = torch.linalg.svd(b, full_matrices=False)
        Ud, Sd, _ = torch.linalg.svd(delta, full_matrices=False)
        top_b = Ub[:, :k]
        top_d = Ud[:, :k]

        real_cos = _mean_best_cos(top_d, top_b)
        null_samples = _random_null(list(b.shape), top_b, k, n_random, generator)
        null_mean = sum(null_samples) / len(null_samples)
        null_var = sum((x - null_mean) ** 2 for x in null_samples) / max(len(null_samples) - 1, 1)
        null_std = null_var ** 0.5
        z = (real_cos - null_mean) / null_std if null_std > 1e-9 else float("nan")

        if z != z:  # nan
            verdict = "undefined (null_std=0)"
        elif abs(z) < 2.0:
            verdict = "indistinguishable from random-direction null"
        elif z > 0:
            verdict = "MORE aligned with base's dominant modes than random (z={:+.2f})".format(z)
        else:
            verdict = "MORE orthogonal to base's dominant modes than random (z={:+.2f})".format(z)

        rows.append({
            "key": key,
            "shape": list(b.shape),
            "real_mean_best_cos": real_cos,
            "null_mean": null_mean,
            "null_std": null_std,
            "n_random": n_random,
            "z_score": z,
            "verdict": verdict,
            "base_sigma_top{}".format(k): Sb[:k].tolist(),
            "delta_sigma_top{}".format(k): Sd[:k].tolist(),
        })
        print(f"  {key}: real={real_cos:.4f}  null={null_mean:.4f}+-{null_std:.4f}  "
              f"z={z:+.2f}  -> {verdict}")

    overall_real = sum(r["real_mean_best_cos"] for r in rows) / len(rows)
    overall_null = sum(r["null_mean"] for r in rows) / len(rows)
    zs = [r["z_score"] for r in rows if r["z_score"] == r["z_score"]]
    overall_z = sum(zs) / len(zs) if zs else float("nan")
    print(f"\nOverall: real={overall_real:.4f}  null={overall_null:.4f}  mean_z={overall_z:+.2f}")
    if len(zs) == len(rows) and all(abs(z) < 2.0 for z in zs):
        print("  -> across ALL matrices, indistinguishable from a random-direction null. "
              "The raw cosine score alone (real or null) is not evidence of fresh vs. "
              "redundant use of weight-space volume at this k/rank/sample size.")

    return {"base": base_path, "trained": trained_path, "k": k, "n_random": n_random,
            "seed": seed, "n_matrices": len(rows), "overall_real_mean_best_cos": overall_real,
            "overall_null_mean": overall_null, "overall_mean_z": overall_z, "matrices": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--keys", required=True, help="Comma-separated state_dict keys.")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-random", type=int, default=30, dest="n_random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    keys = [k.strip() for k in args.keys.split(",")]
    result = cosine_alignment(args.base, args.trained, keys, args.k, args.n_random, args.seed)
    result["_summary"] = {
        "overall_real_mean_best_cos": f"{result['overall_real_mean_best_cos']:.4f}",
        "overall_null_mean": f"{result['overall_null_mean']:.4f}",
        "overall_mean_z": f"{result['overall_mean_z']:+.2f}",
    }

    out_path = save_result(
        args.out, result, experiment="cosine_alignment", hypothesis=["H16"],
        model=args.trained, script=__file__,
    )
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
