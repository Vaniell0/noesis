#!/usr/bin/env python3
"""cosine_alignment.py — does a training delta point into NEW weight-space
directions (expanding used volume) or REDUNDANT ones (reinforcing existing
dominant modes)? The Fisher-Information-adjacent test proposed at the start
of this session's capacity-vs-decode line of reasoning: compare a delta's
(trained - base) top singular vectors against the BASE matrix's own top
singular vectors via cosine similarity. Near-orthogonal -> training used
fresh capacity. Near-aligned -> training just re-weighted an already-
dominant mode, volume didn't grow, it got redistributed within a narrow one.

Complements lora_rank_analysis.py (which reports rank/energy of the delta
alone, no vectors) and today's base-weight effective_rank run (which showed
base att_proj/ffn are near-full-rank, i.e. NOT structurally bottlenecked) —
this answers a different question: even with plenty of base-matrix room
available, did training actually USE fresh room, or double down on what was
already dominant?

Usage:
    python cosine_alignment.py \
        --base ~/.libs/models/rwkv7/rwkv7-g1h-2.9b-20260710-ctx10240.pth \
        --trained ~/.libs/models/rwkv7/rwkv-step9b-e1.pth \
        --keys blocks.0.att.key.weight,blocks.0.att.value.weight,blocks.0.att.receptance.weight \
        --k 8 \
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


def cosine_alignment(base_path: str, trained_path: str, keys: List[str], k: int) -> Dict:
    print("Loading checkpoints...")
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    trained = torch.load(trained_path, map_location="cpu", weights_only=True)

    rows = []
    for key in keys:
        b = base[key].float()
        delta = trained[key].float() - b
        # Full SVD (need vectors this time, not just values).
        Ub, Sb, _ = torch.linalg.svd(b, full_matrices=False)
        Ud, Sd, _ = torch.linalg.svd(delta, full_matrices=False)
        top_b = Ub[:, :k]   # [dim, k] base's top-k left singular directions
        top_d = Ud[:, :k]   # [dim, k] delta's top-k left singular directions

        # For each delta direction, best-matching cosine against ANY of the
        # base's top-k directions (not forced pairing index-to-index —
        # a delta direction could legitimately match base's 3rd mode, not
        # its 1st).
        cos = (top_d.T @ top_b).abs()  # [k, k], |cos| since sign is arbitrary
        best_per_delta_dir = cos.max(dim=1).values  # [k]
        mean_best_cos = best_per_delta_dir.mean().item()

        rows.append({
            "key": key,
            "shape": list(b.shape),
            "mean_best_cos_delta_vs_base_top{}".format(k): mean_best_cos,
            "per_direction_best_cos": best_per_delta_dir.tolist(),
            "base_sigma_top{}".format(k): Sb[:k].tolist(),
            "delta_sigma_top{}".format(k): Sd[:k].tolist(),
        })
        print(f"  {key}: mean best |cos(delta_dir, nearest base_dir)| = {mean_best_cos:.4f} "
              f"(1.0=fully redundant with existing dominant modes, 0.0=fully new)")

    overall = sum(r[f"mean_best_cos_delta_vs_base_top{k}"] for r in rows) / len(rows)
    print(f"\nOverall mean across {len(rows)} matrices: {overall:.4f}")
    if overall > 0.7:
        print("  -> training mostly REINFORCED already-dominant base directions "
              "(redundant, volume didn't grow much)")
    elif overall < 0.3:
        print("  -> training mostly used FRESH directions, near-orthogonal to "
              "base's dominant modes (volume genuinely expanded)")
    else:
        print("  -> mixed: partial overlap, neither clean story")

    return {"base": base_path, "trained": trained_path, "k": k,
            "n_matrices": len(rows), "overall_mean_best_cos": overall,
            "matrices": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--keys", required=True, help="Comma-separated state_dict keys.")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    keys = [k.strip() for k in args.keys.split(",")]
    result = cosine_alignment(args.base, args.trained, keys, args.k)
    result["_summary"] = {"overall_mean_best_cos": f"{result['overall_mean_best_cos']:.4f}"}

    out_path = save_result(
        args.out, result, experiment="cosine_alignment", hypothesis=["H16"],
        model=args.trained, script=__file__,
    )
    print(f"\nSaved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
