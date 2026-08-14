#!/usr/bin/env python3
"""lora_rank_analysis.py — measure effective rank of weight deltas (trained - base).

When LoRA rank=r is merged into full weights, the delta W = B×A has at most
rank r. SVD of the delta reveals the actual rank used and what fraction of
the change energy lives in the top-r subspace.

Usage:
    python lora_rank_analysis.py \
        --base ~/.libs/models/rwkv7/rwkv7-g1h-2.9b-20260710-ctx10240.pth \
        --trained ~/.libs/models/rwkv7/rwkv-step9b-e1.pth \
        --lora-rank 32 \
        --out results/lora_rank_step9b_e1.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import torch


def _effective_rank(sv: torch.Tensor, threshold: float = 0.01) -> int:
    """Number of singular values > threshold × sv[0]."""
    if sv[0] < 1e-10:
        return 0
    return int((sv / sv[0] > threshold).sum().item())


def _rank_r_energy(sv: torch.Tensor, r: int) -> float:
    """Fraction of total Frobenius energy captured by top-r singular values."""
    total = float((sv ** 2).sum())
    if total < 1e-20:
        return 1.0
    top_r = float((sv[:r] ** 2).sum())
    return top_r / total


def analyze(base_path: str, trained_path: str, lora_rank: int) -> Dict:
    print("Loading base checkpoint...")
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    print("Loading trained checkpoint...")
    trained = torch.load(trained_path, map_location="cpu", weights_only=True)

    common = set(base.keys()) & set(trained.keys())
    # Only 2D matrices large enough for meaningful SVD
    mats = [k for k in common
            if len(base[k].shape) == 2 and min(base[k].shape) >= 32]

    print(f"\nAnalysing {len(mats)} weight matrices...")

    results = []
    for key in sorted(mats):
        b = base[key].float()
        t = trained[key].float()
        delta = t - b
        frob = float(delta.norm())
        if frob < 1e-8:
            continue  # unchanged weight

        sv = torch.linalg.svdvals(delta)
        eff_rank = _effective_rank(sv, threshold=0.01)
        energy_r = _rank_r_energy(sv, lora_rank)

        # Classify weight type
        if ".att." in key and any(x in key for x in ["receptance", "key", "value", "output", "gate"]):
            wtype = "att_proj"
        elif ".ffn." in key:
            wtype = "ffn"
        elif ".att." in key:
            wtype = "att_other"
        elif "emb" in key or "head" in key:
            wtype = "emb_head"
        else:
            wtype = "other"

        results.append({
            "key": key,
            "shape": list(base[key].shape),
            "type": wtype,
            "delta_frob": frob,
            "effective_rank": eff_rank,
            f"top{lora_rank}_energy": energy_r,
            "sigma1": float(sv[0]),
        })

    # Summary by type
    print(f"\n{'Type':<12} {'Count':>6} {'Mean eff_rank':>14} "
          f"{'Mean top{r} energy':>{16}} {'Mean Δfrob':>12}".format(r=lora_rank))
    print("-" * 65)

    by_type: Dict[str, List] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    for wtype, rows in sorted(by_type.items()):
        mean_er = sum(r["effective_rank"] for r in rows) / len(rows)
        mean_en = sum(r[f"top{lora_rank}_energy"] for r in rows) / len(rows)
        mean_fr = sum(r["delta_frob"] for r in rows) / len(rows)
        print(f"{wtype:<12} {len(rows):>6} {mean_er:>14.1f} {mean_en:>16.3f} {mean_fr:>12.4f}")

    # Highlight outliers: matrices with eff_rank > lora_rank (unexpected for LoRA)
    outliers = [r for r in results if r["effective_rank"] > lora_rank]
    print(f"\nMatrices with effective_rank > {lora_rank} (unexpected if pure LoRA): {len(outliers)}")
    for r in sorted(outliers, key=lambda x: -x["effective_rank"])[:10]:
        print(f"  {r['key']}: eff_rank={r['effective_rank']}, "
              f"top{lora_rank}_energy={r[f'top{lora_rank}_energy']:.3f}, frob={r['delta_frob']:.4f}")

    # Low-rank confirmation: att_proj matrices with top-r energy
    att = [r for r in results if r["type"] == "att_proj"]
    if att:
        mean_att_en = sum(r[f"top{lora_rank}_energy"] for r in att) / len(att)
        print(f"\nAttention projection matrices (n={len(att)}): "
              f"mean top-{lora_rank} energy = {mean_att_en:.3f}")
        if mean_att_en > 0.95:
            print(f"  → LoRA r={lora_rank} accounts for >95% of delta energy: confirmed low-rank")
        elif mean_att_en > 0.80:
            print(f"  → LoRA r={lora_rank} accounts for >80%: mostly low-rank, some full-FT leak")
        else:
            print(f"  → Top-{lora_rank} energy <80%: delta is NOT well-approximated by rank-{lora_rank}")

    return {
        "base": base_path,
        "trained": trained_path,
        "lora_rank": lora_rank,
        "n_matrices": len(results),
        "by_type": {k: {
            "count": len(v),
            "mean_effective_rank": sum(r["effective_rank"] for r in v) / len(v),
            f"mean_top{lora_rank}_energy": sum(r[f"top{lora_rank}_energy"] for r in v) / len(v),
        } for k, v in by_type.items()},
        "matrices": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--trained", required=True)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result = analyze(args.base, args.trained, args.lora_rank)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nSaved → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
