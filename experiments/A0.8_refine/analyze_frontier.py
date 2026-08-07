#!/usr/bin/env python3
"""H10 Pareto frontier analysis.

Reads all result JSONs from a matrix sweep directory and finds:
  1. The Pareto frontier on (accuracy, -compute_cost).
  2. Whether any non-default cell beats (N=1, K=large, prompt_cot).
  3. Whether state_readout > silent at same N (readout tokens carry signal).

Compute cost proxy: N * (readout_k + K) tokens processed per task.
"""
import argparse
import json
import os
import re
from pathlib import Path


def load_cell(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    meta = {
        "file": path.name,
        "n_passes": d.get("n_passes", 1),
        "num_predict": d.get("num_predict", 256),
        "readout_mode": d.get("readout_mode", "prompt_cot"),
        "readout_k": d.get("readout_k", 0),
        "accuracy": d.get("aggregate", {}).get("overall_accuracy", 0.0),
        "n_correct": d.get("aggregate", {}).get("n_correct", 0),
        "n_total": d.get("aggregate", {}).get("n_total", 0),
    }
    # compute cost: N passes × prompt + think tokens
    think = meta["readout_k"] if meta["readout_mode"] == "state_readout" else meta["num_predict"]
    meta["cost"] = meta["n_passes"] * think
    return meta


def is_dominated(cell: dict, others: list) -> bool:
    for o in others:
        if o is cell:
            continue
        if o["accuracy"] >= cell["accuracy"] and o["cost"] <= cell["cost"]:
            if o["accuracy"] > cell["accuracy"] or o["cost"] < cell["cost"]:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory with result JSONs.")
    args = ap.parse_args()

    cells = [load_cell(p) for p in sorted(Path(args.dir).glob("*.json"))]
    if not cells:
        print("No result files found.")
        return

    # Sort by accuracy desc
    cells.sort(key=lambda c: -c["accuracy"])

    print(f"{'File':<40} {'N':>2} {'K':>5} {'mode':<14} {'rk':>4} {'acc':>6} {'cost':>6}")
    print("-" * 85)
    for c in cells:
        marker = " *" if not is_dominated(c, cells) else ""
        print(f"{c['file']:<40} {c['n_passes']:>2} {c['num_predict']:>5} "
              f"{c['readout_mode']:<14} {c['readout_k']:>4} "
              f"{c['accuracy']:>6.3f} {c['cost']:>6}{marker}")

    frontier = [c for c in cells if not is_dominated(c, cells)]
    print(f"\nPareto frontier: {len(frontier)} cell(s) marked *")

    # Default cell: N=1, largest K, prompt_cot
    cot_cells = [c for c in cells if c["readout_mode"] == "prompt_cot" and c["n_passes"] == 1]
    default = max(cot_cells, key=lambda c: c["num_predict"]) if cot_cells else None
    if default:
        print(f"\nDefault cell (N=1, K={default['num_predict']}, prompt_cot): "
              f"acc={default['accuracy']:.3f}")
        better = [c for c in frontier if c is not default and
                  c["accuracy"] > default["accuracy"] and c["cost"] <= default["cost"] * 1.05]
        if better:
            print(f"NON-DEFAULT CELLS BEAT DEFAULT at ≤1.05× cost:")
            for c in better:
                print(f"  {c['file']}: acc={c['accuracy']:.3f} cost={c['cost']}")
            print("→ H10 NON-DEGENERATE FRONTIER: SUPPORTED")
        else:
            print("→ Default cell is Pareto-dominant: H10 frontier claim REFUTED "
                  "(knobs collapse to Transformer conventions)")

    # state_readout vs silent at same N
    print("\nstate_readout vs silent (same N):")
    for n in sorted({c["n_passes"] for c in cells}):
        silent = next((c for c in cells if c["n_passes"] == n and c["readout_mode"] == "silent"), None)
        readouts = [c for c in cells if c["n_passes"] == n and c["readout_mode"] == "state_readout"]
        if not silent or not readouts:
            continue
        best_ro = max(readouts, key=lambda c: c["accuracy"])
        delta = best_ro["accuracy"] - silent["accuracy"]
        verdict = "SIGNAL" if delta >= 0.02 else "NO SIGNAL"
        print(f"  N={n}: silent={silent['accuracy']:.3f} "
              f"best_readout={best_ro['accuracy']:.3f} Δ={delta:+.3f} → {verdict}")


if __name__ == "__main__":
    main()
