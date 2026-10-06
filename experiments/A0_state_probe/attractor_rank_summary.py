#!/usr/bin/env python3
"""attractor_rank_summary.py — capacity (rank) vs step vs optimizer, with the step controlled.

Question (user, 2026-10-06): Adam concentrates the state (top-1 energy 0.87) more than
Muon (0.79) after a LoRA fine-tune. Is that LoRA capacity running out (the rank), or the
update shape, or just Adam taking a bigger effective step at small rank?

Why a plain "top-1 vs rank at lr=1e-3" cannot answer it: the induced step
||dW||/||W|| changes with the rank, so concentration, rank and step move together. This
reads the rank sweep (run_rank_matched.sh: an lr grid per optimizer per rank) and puts
both optimizers on the SAME induced step by interpolating each family's per-lr means on
log(step) — then asks, per rank, at equal step:

  top1@K0, erank@K0   concentration right after the fine-tune (shape of the state)
  new R2              what the step bought
  steps_to_new        updates until new R2 >= 0.99 (time; None counted as not reached)
  new R2 / moved      new-skill R2 per unit of total relative weight moved (efficiency)

Reading rule, fixed before the data: if Adam's top-1 rises as r falls at equal step while
Muon's stays flat, the cause is how Adam SPENDS capacity, not how much LoRA has; if both
rise, it is volume; if neither depends on r once the step is matched, the first reading
(capacity) is not supported.

    training/.venv/bin/python experiments/A0_state_probe/attractor_rank_summary.py
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
RANKS = (2, 8, 32)
FAMILIES = {"adam": "lora_adam", "muon_fw": "lora_muon_fw"}
STEPS = (3e-4, 7e-4, 1.8e-3)


def _load(r: int):
    p = R / f"attractor_rank_r{r}.partial.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else []


def _mean(v):
    v = [x for x in v if x is not None]
    return st.mean(v) if v else None


def per_lr(rows, arm):
    """lr -> dict of seed-means for one arm."""
    by = defaultdict(list)
    for x in rows:
        if x["arm"] == arm:
            by[x["lr"]].append(x)
    out = {}
    for lr, rs in sorted(by.items()):
        reached = [x["steps_to_new"] for x in rs if x.get("steps_to_new") is not None]
        k0 = [x["curve_old"]["0"] for x in rs]
        out[lr] = {
            "n": len(rs),
            "step": _mean([x["induced_step_mean"] for x in rs]),
            "top1_K0": _mean([c["top1"] for c in k0]),
            "erank_K0": _mean([c["erank"] for c in k0]),
            "new_r2": _mean([x["new_r2"] for x in rs]),
            "old_r2": _mean([x["old_r2"] for x in rs]),
            "reached": f"{len(reached)}/{len(rs)}",
            "steps_to_new": st.median(reached) if reached else None,
            "per_moved": _mean([x["new_r2"] / x["weight_moved"] for x in rs if x.get("weight_moved")]),
        }
    return out


def at_step(table, target, key):
    """Interpolate `key` at an induced step on log scale; None outside the measured range."""
    pts = sorted((v["step"], v[key]) for v in table.values() if v["step"] and v[key] is not None)
    for (s0, y0), (s1, y1) in zip(pts, pts[1:]):
        if s0 <= target <= s1 and s1 > s0:
            f = (math.log(target) - math.log(s0)) / (math.log(s1) - math.log(s0))
            return y0 + f * (y1 - y0)
    return None


def fmt(x, nd=3):
    return "  n/a" if x is None else f"{x:.{nd}f}"


def main() -> int:
    for r in RANKS:
        rows = _load(r)
        if not rows:
            print(f"== rank {r}: no rows yet")
            continue
        print(f"\n== rank {r}   ({len({x['seed'] for x in rows})} seeds, {len(rows)} rows)")
        tabs = {fam: per_lr(rows, arm) for fam, arm in FAMILIES.items()}
        for fam, tab in tabs.items():
            print(f"  {fam}: per lr")
            for lr, v in tab.items():
                print(f"    lr={lr:<7g} step={v['step']:.2e} new={fmt(v['new_r2'])} old={fmt(v['old_r2'])} "
                      f"top1@K0={fmt(v['top1_K0'], 2)} erank@K0={fmt(v['erank_K0'], 2)} "
                      f"to0.99={v['reached']} med={v['steps_to_new']} new/moved={fmt(v['per_moved'], 1)}")
        print("  at EQUAL induced step (log-interpolated):")
        print("    step      | " + " | ".join(f"{f:>7s} top1  erank  new  steps" for f in tabs))
        for s in STEPS:
            cells = []
            for fam, tab in tabs.items():
                cells.append(f"{fmt(at_step(tab, s, 'top1_K0'), 2):>13s} {fmt(at_step(tab, s, 'erank_K0'), 2):>6s} "
                             f"{fmt(at_step(tab, s, 'new_r2')):>6s} {fmt(at_step(tab, s, 'steps_to_new'), 0):>5s}")
            print(f"    {s:.1e}  | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
