#!/usr/bin/env python3
"""geomgrid_summary.py — Muon vs AdamW on G1d-0.4B across a learning-rate grid, compared on the
MEDIAN first-step ||dW||/||W||, with data-order seeds.

Reads experiments/rl/results/geomgrid/lr{1e-5,3e-5,1e-4}_s{0,1,2}.json written by
experiments/_logs/muon_window_2026-10-06/run_geom_grid.sh (optimizer_geometry_probe.py, full
fine-tune, 30 steps, --shuffle-data on, both arms per file).

Why the median: the maximum first step falls on `blocks.N.ffn.x_k`, a 1024-vector that both
optimizers update identically (it sits in the AdamW group of the Muon hybrid), so the maximum
matches at equal lr and decides nothing. The median is the tensor-bulk step, where the two
optimizers differ by construction.

Reports per optimizer and lr (mean +- std over seeds): median and max step, final train loss,
held-out dCE, d_live; then dCE interpolated on log(median step) so the two optimizers can be
read at one step, and the lr at which each first makes held-out CE worse.

    training/.venv/bin/python experiments/rl/geomgrid_summary.py
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent / "results" / "geomgrid"
LRS = ("1e-5", "3e-5", "1e-4")


def ms(v):
    v = [x for x in v if x is not None]
    if not v:
        return None, None, 0
    return st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0), len(v)


def main() -> int:
    cells = defaultdict(lambda: defaultdict(list))
    for lr in LRS:
        for f in sorted(D.glob(f"lr{lr}_s*.json")):
            d = json.load(open(f))
            for r in d["arms"]:
                c = cells[(r["optimizer"], lr)]
                fs = r["first_step_relative"]
                c["median"].append(fs["median"]); c["max"].append(fs["max"])
                c["loss"].append(r["final_loss"]); c["dce"].append(r["d_ce"]); c["dlive"].append(r["d_live"])
    if not cells:
        print("no runs yet")
        return 0
    print(f"{'opt':6s} {'lr':>6s} {'n':>2s}  {'median step':>12s} {'max step':>10s} {'train loss':>14s} "
          f"{'dCE held-out':>16s} {'d_live':>8s}")
    curve = defaultdict(list)
    for (opt, lr), c in sorted(cells.items(), key=lambda kv: (kv[0][0], float(kv[0][1]))):
        m, _, n = ms(c["median"]); mx, _, _ = ms(c["max"]); l, ls, _ = ms(c["loss"])
        e, es, _ = ms(c["dce"]); dl, _, _ = ms(c["dlive"])
        print(f"{opt:6s} {lr:>6s} {n:>2d}  {m:12.2e} {mx:10.2e} {l:7.3f}+-{ls:5.3f} {e:+9.3f}+-{es:5.3f} {dl:+8.3f}")
        curve[opt].append((m, e, es))
    print("\nheld-out dCE interpolated on log(median step):")
    steps = sorted({m for pts in curve.values() for m, _, _ in pts})
    lo, hi = max(min(m for m, _, _ in pts) for pts in curve.values()), min(max(m for m, _, _ in pts) for pts in curve.values())
    grid = [lo * (hi / lo) ** (i / 4) for i in range(5)] if hi > lo else [lo]
    for g in grid:
        row = []
        for opt, pts in sorted(curve.items()):
            pts = sorted(pts)
            val = None
            for (s0, y0, _), (s1, y1, _) in zip(pts, pts[1:]):
                if s0 <= g <= s1 and s1 > s0:
                    f = (math.log(g) - math.log(s0)) / (math.log(s1) - math.log(s0))
                    val = y0 + f * (y1 - y0)
            row.append(f"{opt} {val:+.3f}" if val is not None else f"{opt}   n/a")
        print(f"  median step {g:.2e}:  " + "   ".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
