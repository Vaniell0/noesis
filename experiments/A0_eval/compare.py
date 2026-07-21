#!/usr/bin/env python3
"""Compare two A0.2 eval runs (baseline vs A1 fine-tune).

Prints Markdown table: overall + per-category, with delta and win/loss/
regression tally. Also lists per-task flips so we see which specific
tasks A1 gained or lost.

Usage:
    python3 compare.py baseline.json a1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple


def load(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _delta(a: float, b: float) -> str:
    d = (b - a) * 100
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}pp"


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two A0.2 runs.")
    ap.add_argument("baseline", help="Path to baseline eval JSON.")
    ap.add_argument("a1", help="Path to A1 eval JSON.")
    args = ap.parse_args()

    base = load(args.baseline)
    a1 = load(args.a1)

    b_agg = base["aggregate"]
    a_agg = a1["aggregate"]

    print(f"# A0.2 eval — {a1['model']} vs {base['model']}")
    print()
    print(f"Baseline: **{_pct(b_agg['overall_accuracy'])}** "
          f"({b_agg['n_correct']}/{b_agg['n_total']})")
    print(f"A1:       **{_pct(a_agg['overall_accuracy'])}** "
          f"({a_agg['n_correct']}/{a_agg['n_total']}) "
          f"— **Δ {_delta(b_agg['overall_accuracy'], a_agg['overall_accuracy'])}**")
    print()

    print("| category | baseline | A1 | Δ |")
    print("|---|---:|---:|---:|")
    cats = sorted(set(list(b_agg["per_category"]) + list(a_agg["per_category"])))
    for cat in cats:
        b = b_agg["per_category"].get(cat, {"accuracy": 0.0, "n": 0})
        a = a_agg["per_category"].get(cat, {"accuracy": 0.0, "n": 0})
        print(f"| {cat} | {_pct(b['accuracy'])} ({b['n']}) | "
              f"{_pct(a['accuracy'])} ({a['n']}) | "
              f"{_delta(b['accuracy'], a['accuracy'])} |")
    print()

    # Per-task flips
    b_by_id = {r["id"]: r for r in base["results"]}
    a_by_id = {r["id"]: r for r in a1["results"]}
    gains = [tid for tid in b_by_id
             if tid in a_by_id and not b_by_id[tid]["correct"] and a_by_id[tid]["correct"]]
    losses = [tid for tid in b_by_id
              if tid in a_by_id and b_by_id[tid]["correct"] and not a_by_id[tid]["correct"]]

    print(f"## Per-task flips")
    print(f"- gains (A1 fixed): {len(gains)} — {gains}")
    print(f"- losses (A1 broke): {len(losses)} — {losses}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
