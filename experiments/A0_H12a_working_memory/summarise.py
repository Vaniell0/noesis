#!/usr/bin/env python3
"""A0.H12a — summariser and decision-tree verdict.

Reads ``results/N{n}.json`` for the width sweep and ``results/dist-{gap}.json``
for the distance sweep produced by ``run_probe.py``, prints two tables
(accuracy vs N, accuracy vs mean word-gap), and applies the falsification
decision tree from ``README.md``:

- Width falls sharply, distance flat → width bottleneck → H12b worth building.
- Distance falls, width flat → decay bottleneck → H12b not worth building.
- Both fall → inconclusive.
- Neither falls → probe too easy — increase difficulty.

"Falls sharply" is Δ_accuracy ≥ ``--drop-threshold`` (default 0.30)
between the sweep's lowest and highest x-axis point.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from typing import Dict, List, Tuple


def _load(path: pathlib.Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def _collect(results_dir: pathlib.Path) -> Tuple[List[Tuple[int, Dict]], List[Tuple[int, Dict]]]:
    width_rows: List[Tuple[int, Dict]] = []
    dist_rows: List[Tuple[int, Dict]] = []
    for p in sorted(results_dir.glob("N*.json")):
        m = re.match(r"N(\d+)\.json$", p.name)
        if not m:
            continue
        width_rows.append((int(m.group(1)), _load(p)))
    for p in sorted(results_dir.glob("dist-*.json")):
        m = re.match(r"dist-(\d+)\.json$", p.name)
        if not m:
            continue
        dist_rows.append((int(m.group(1)), _load(p)))
    width_rows.sort(key=lambda r: r[0])
    dist_rows.sort(key=lambda r: r[0])
    return width_rows, dist_rows


def _fmt_table(title: str, header: List[str], rows: List[List[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    def line(cells): return "  " + "  ".join(cells[i].rjust(widths[i]) for i in range(len(cells)))
    out = [title, line(header), line(["-" * w for w in widths])]
    for r in rows:
        out.append(line(r))
    return "\n".join(out)


def _series_drop(vals: List[float]) -> float:
    if len(vals) < 2:
        return 0.0
    return max(vals) - min(vals)


def main() -> int:
    ap = argparse.ArgumentParser(description="A0.H12a summariser.")
    ap.add_argument("--results-dir", type=pathlib.Path, default=None,
                    help="Directory containing N*.json and dist-*.json.")
    ap.add_argument("--drop-threshold", type=float, default=0.30,
                    help="Δ_accuracy that counts as 'falls sharply' (default: 0.30).")
    args = ap.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    results_dir = args.results_dir or (here / "results")
    if not results_dir.exists():
        print(f"No results directory at {results_dir}")
        return 1

    width_rows, dist_rows = _collect(results_dir)
    print(f"=== A0.H12a summary — {results_dir} ===\n")

    if width_rows:
        rows = [
            [str(n),
             f"{r['aggregate']['n']}",
             f"{r['aggregate']['accuracy_exact']:.2f}",
             f"{r['aggregate']['mean_f1']:.2f}",
             f"{r['aggregate']['mean_precision']:.2f}",
             f"{r['aggregate']['mean_recall']:.2f}",
             f"{r['aggregate']['mean_word_gap']:.0f}"]
            for n, r in width_rows
        ]
        print(_fmt_table(
            "Width sweep (fixed n_pairs, varying N):",
            ["N", "tasks", "acc_exact", "F1", "prec", "recall", "gap_w"],
            rows,
        ))
        print()
    if dist_rows:
        rows = [
            [str(g),
             f"{r['aggregate']['n']}",
             f"{r['aggregate']['accuracy_exact']:.2f}",
             f"{r['aggregate']['mean_f1']:.2f}",
             f"{r['aggregate']['mean_precision']:.2f}",
             f"{r['aggregate']['mean_recall']:.2f}",
             f"{r['aggregate']['mean_word_gap']:.0f}"]
            for g, r in dist_rows
        ]
        print(_fmt_table(
            "Distance sweep (fixed N=16, varying target gap):",
            ["gap_target", "tasks", "acc_exact", "F1", "prec", "recall", "gap_measured"],
            rows,
        ))
        print()

    width_accs = [r["aggregate"]["accuracy_exact"] for _, r in width_rows]
    dist_accs = [r["aggregate"]["accuracy_exact"] for _, r in dist_rows]
    width_drop = _series_drop(width_accs) if width_accs else None
    dist_drop = _series_drop(dist_accs) if dist_accs else None

    print("=== Decision tree verdict ===")
    thr = args.drop_threshold
    print(f"drop threshold = {thr:.2f}")
    if width_drop is not None:
        print(f"width  Δ_accuracy (max−min): {width_drop:.2f}"
              f"  → {'FALLS' if width_drop >= thr else 'flat'}")
    if dist_drop is not None:
        print(f"dist   Δ_accuracy (max−min): {dist_drop:.2f}"
              f"  → {'FALLS' if dist_drop >= thr else 'flat'}")
    print()

    verdict = None
    if width_drop is not None and dist_drop is not None:
        w_falls = width_drop >= thr
        d_falls = dist_drop >= thr
        if w_falls and not d_falls:
            verdict = "width-bottleneck → H12b (multi-slot LoRA) IS worth running."
        elif d_falls and not w_falls:
            verdict = "decay-bottleneck → H12b NOT worth running. Fix via retrieval / longer effective context / different decay schedule."
        elif w_falls and d_falls:
            verdict = "inconclusive → both axes fell; run finer-grained probes (mid-N distance sweep and mid-gap width sweep) to disentangle."
        else:
            # Neither falls — probe is too easy OR both baselines are already at floor.
            top = max([max(width_accs) if width_accs else 0.0,
                       max(dist_accs) if dist_accs else 0.0])
            if top >= 0.8:
                verdict = "probe too easy → increase difficulty (more n_pairs, closer-property distractors) and rerun."
            else:
                verdict = "flat but at low accuracy → floor effect; task may be too hard even at N=4 / gap=50. Simplify or check tokeniser handling."
    print(f"VERDICT: {verdict if verdict else 'insufficient data — need both sweeps.'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
