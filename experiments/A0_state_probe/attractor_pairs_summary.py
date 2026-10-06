#!/usr/bin/env python3
"""attractor_pairs_summary.py — tails and paired comparison for the matched-step pairs.

Merges seeds 0..5 of attractor_depth.partial.jsonl (the four pair arms only) with the
split runs attractor_pairs_s*.partial.jsonl, then reports, per arm over all seeds:
the median old-skill change after 32 blank ticks, how often it collapses (below -0.5 and
below -5), and — per pair, seed by seed on the same base — how often the Adam arm loses
more than the Muon arm, with an exact two-sided sign test and Fisher's exact test on the
collapse counts. The base (no finetuning) row is the noise floor of the same quantity.

The question it settles: at a matched induced step and matched new-skill accuracy, is the
old-skill collapse under repeated blank ticks more frequent for Adam than for Muon, or was
the 3-in-6 tail of the first run chance?
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
ARMS = {("full_adam", 0.003), ("full_muon", 0.005), ("lora_adam", 0.001), ("lora_muon_fw", 0.001),
        ("base", 0.0)}
PAIRS = [(("full_adam", 0.003), ("full_muon", 0.005)),
         (("lora_adam", 0.001), ("lora_muon_fw", 0.001))]


def load() -> dict:
    rows = {}
    files = [RES / "attractor_depth.partial.jsonl", *sorted(RES.glob("attractor_pairs_s*.partial.jsonl"))]
    for f in files:
        if not f.exists():
            continue
        for line in f.open():
            r = json.loads(line)
            if (r["arm"], r["lr"]) in ARMS:
                rows[(r["seed"], r["arm"], r["lr"])] = r
    return rows


def sign_test(wins: int, n: int) -> float:
    """Two-sided exact binomial p for `wins` of `n` at p = 0.5 (ties dropped by the caller)."""
    if n == 0:
        return 1.0
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fisher(a: int, an: int, b: int, bn: int) -> float:
    """Two-sided Fisher exact p for a/an versus b/bn collapses."""
    total, hits = an + bn, a + b

    def pmf(x: int) -> float:
        return math.comb(hits, x) * math.comb(total - hits, an - x) / math.comb(total, an)

    p0 = pmf(a)
    return min(1.0, sum(pmf(x) for x in range(max(0, an - (total - hits)), min(an, hits) + 1)
                        if pmf(x) <= p0 + 1e-12))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    rows = load()
    by = defaultdict(dict)
    for (seed, arm, lr), r in rows.items():
        by[(arm, lr)][seed] = r
    d32 = lambda r: r["curve_old"]["32"]["d_r2"]  # noqa: E731
    print(f"{'arm':14s}{'lr':>7s}{'n':>4s}{'new med':>9s}{'dR2@32 med':>11s}{'<-0.5':>7s}{'<-5':>5s}"
          f"{'  collapse rate [95% Wilson]':>30s}{'top1@0':>8s}{'erank@0':>8s}{'top1@32':>9s}{'ablate med':>11s}")
    coll = {}
    for key in sorted(by):
        R = list(by[key].values())
        dr = [d32(r) for r in R]
        k = sum(x < -0.5 for x in dr)
        lo, hi = wilson(k, len(R))
        coll[key] = (k, len(R))
        print(f"{key[0]:14s}{key[1]:7.4f}{len(R):4d}{st.median(r['new_r2'] for r in R):9.3f}"
              f"{st.median(dr):11.3f}{k:7d}{sum(x < -5 for x in dr):5d}"
              f"{k / len(R):12.2f} [{lo:.2f},{hi:.2f}]{st.median(r['curve_old']['0']['top1'] for r in R):8.3f}"
              f"{st.median(r['curve_old']['0']['erank'] for r in R):8.2f}"
              f"{st.median(r['curve_old']['32']['top1'] for r in R):9.3f}"
              f"{st.median(r['ablation_a_gate_0_r2'] for r in R):11.3f}")
    print()
    for A, M in PAIRS:
        a, m = by[A], by[M]
        S = sorted(set(a) & set(m))
        both = [s for s in S if a[s]["new_r2"] >= 0.98 and m[s]["new_r2"] >= 0.98]
        adam_worse = sum(d32(a[s]) < d32(m[s]) for s in both)
        ties = sum(d32(a[s]) == d32(m[s]) for s in both)
        n = len(both) - ties
        ka, na = coll[A]
        km, nm = coll[M]
        print(f"{A[0]}:{A[1]:g} vs {M[0]}:{M[1]:g}")
        print(f"  seeds with both arms matched (new >= 0.98): {len(both)} of {len(S)}")
        print(f"  Adam loses more old skill on the same base: {adam_worse}/{n}   "
              f"sign test p = {sign_test(adam_worse, n):.3f}")
        print(f"  collapse (< -0.5): Adam {ka}/{na}, Muon {km}/{nm}   Fisher p = {fisher(ka, na, km, nm):.3f}")
        # Right after finetuning (K=0), before any blank tick: does Adam leave the state more
        # concentrated on one direction than Muon at the same step? (the capacity story for LoRA)
        t1 = [(a[s]["curve_old"]["0"]["top1"], m[s]["curve_old"]["0"]["top1"]) for s in both]
        hi_ = sum(x > y for x, y in t1)
        nt = sum(x != y for x, y in t1)
        print(f"  state concentration at K=0 (top-1 energy): Adam higher on {hi_}/{nt} bases, "
              f"sign test p = {sign_test(hi_, nt):.3f}; medians {st.median(x for x, _ in t1):.3f} vs "
              f"{st.median(y for _, y in t1):.3f}")
        print(f"  mean induced step: {st.mean(r['induced_step_mean'] for r in a.values()):.4f} vs "
              f"{st.mean(r['induced_step_mean'] for r in m.values()):.4f}")


if __name__ == "__main__":
    main()
