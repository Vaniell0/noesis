#!/usr/bin/env python3
"""chain_write_budget.py — what a latent tick spends and what it deposits.

Reads a `state_trajectory_probe.py` result JSON and answers one question the
existing summary never asked: per tick, how much does the mechanism ERASE
from the WKV state, and how much CONTENT does it write back?

RWKV-7 already carries the instrument, so nothing new is measured here —
only read. The captured `rkvwag` block holds, per layer per step:

  a          the erase / in-context-learning-rate gate of the delta rule.
             ~0 = write without removing what is there; ~1 = full erase
             along the key direction. This is the model's own mechanism
             for "how much does absorbing this input displace what I hold".
  k          where the write lands.
  v          what gets written — a real token's `v` tracks what the token
             MEANS, so its magnitude moves with content.
  retention  exp(-exp(w)), the per-channel decay multiplier.

Comparing the `chain` branch (phase markers) against `read` (real prompt
tokens) on the same prompts turns those into a budget: a tick that erases
like a token but writes a fraction of one is spending state to deposit
less than it removed.

The across-prompt spread of `v` is reported separately and matters as much
as the level. A marker is one fixed vector, so its `v` can only vary
through token-shift and the state it lands on. If that spread is near zero
while a real token's is large, the write is content-INDEPENDENT — which
bounds what the mechanism can be: an operator applied to whatever is in
state, never a note whose content depends on what happened.

Usage:
    chain_write_budget.py --result <state_trajectory json> [--out <json>]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BANDS = {"L0-7": range(0, 8), "L8-15": range(8, 16),
         "L16-23": range(16, 24), "L24-31": range(24, 32)}

# `expected` is the content-DEPENDENT latent feed — the marker's control.
# Absent from older result files, so every branch is opt-in on presence.
BRANCHES = ("read", "loop", "chain", "expected")
REFERENCE = "read"


def _vals(entries, field, stat, layers) -> list[float]:
    return [lr[field][stat] for e in entries
            for lr in e["rkvwag"] if lr["layer"] in layers]


def _branch(result: dict, name: str) -> list:
    """Chain drops its first entry — the shared entry cue is not a phase tick."""
    return result["chain"][1:] if name == "chain" else result.get(name, [])


def analyse(data: dict) -> dict:
    results = data["results"]
    n_layer = max(lr["layer"] for r in results.values()
                  for lr in r["read"][0]["rkvwag"]) + 1
    bands = {k: set(v) for k, v in BANDS.items() if max(v) < n_layer}
    present = [br for br in BRANCHES
               if any(_branch(r, br) for r in results.values())]

    by_band = {}
    for bname, ls in bands.items():
        acc = {br: {"v": [], "a": [], "k": [], "retention": []} for br in present}
        for r in results.values():
            for br in present:
                ent = _branch(r, br)
                if not ent:
                    continue
                acc[br]["v"] += _vals(ent, "v", "norm", ls)
                acc[br]["k"] += _vals(ent, "k", "norm", ls)
                acc[br]["a"] += _vals(ent, "a", "mean", ls)
                acc[br]["retention"] += _vals(ent, "retention", "mean", ls)
        row = {}
        for br, d in acc.items():
            if d["v"]:
                row[br] = {f: round(st.mean(xs), 4) for f, xs in d.items()}
        if REFERENCE in row:
            ratios = {br: {"write": round(row[br]["v"] / row[REFERENCE]["v"], 4),
                           "erase": round(row[br]["a"] / row[REFERENCE]["a"], 4)}
                      for br in row if br != REFERENCE}
            if ratios:
                row["ratios"] = ratios
        by_band[bname] = row

    # content dependence: does the write magnitude move with the prompt?
    spread = {}
    for bname, ls in bands.items():
        spread[bname] = {}
        for br in present:
            per_prompt = []
            for r in results.values():
                ent = _branch(r, br)
                if ent:
                    per_prompt.append(st.mean(_vals(ent, "v", "norm", ls)))
            if len(per_prompt) > 1:
                spread[bname][br] = {
                    "per_prompt": [round(x, 3) for x in per_prompt],
                    "spread": round(max(per_prompt) - min(per_prompt), 4),
                    "relative": round((max(per_prompt) - min(per_prompt))
                                      / st.mean(per_prompt), 4),
                }
    return {"n_layer": n_layer, "branches": present, "by_band": by_band,
            "v_spread_across_prompts": spread}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True,
                     help="A state_trajectory_probe.py output JSON.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    data = json.load(open(args.result))
    out = analyse(data)

    others = [b for b in out["branches"] if b != REFERENCE]
    print(f"source: {args.result}  ({out['n_layer']} layers)")
    print(f"branches: {', '.join(out['branches'])}   reference: {REFERENCE}\n")

    print("write = v.norm vs a real token's,  erase = a.mean vs a real token's")
    head = f"{'band':<9}{'v ' + REFERENCE:>10}"
    for b in others:
        head += f"{b + ' write':>14}{b + ' erase':>14}"
    print(head)
    for bname, row in out["by_band"].items():
        if REFERENCE not in row:
            continue
        line = f"{bname:<9}{row[REFERENCE]['v']:>10.2f}"
        for b in others:
            rt = row.get("ratios", {}).get(b)
            line += (f"{rt['write']:>14.2f}{rt['erase']:>14.2f}" if rt
                     else f"{'-':>14}{'-':>14}")
        print(line)

    print("\nv.norm spread across prompts — content dependence "
          "(a note varies with the task, an operator does not)")
    print(f"{'band':<9}" + "".join(f"{b:>12}" for b in out["branches"]))
    for bname, row in out["v_spread_across_prompts"].items():
        if not row:
            continue
        print(f"{bname:<9}" + "".join(
            f"{row[b]['relative']*100:>11.1f}%" if b in row else f"{'-':>12}"
            for b in out["branches"]))

    if args.out:
        from experiments._common.results import save_result
        deep = out["by_band"].get("L24-31", {}).get("ratios", {})
        shallow = out["by_band"].get("L0-7", {}).get("ratios", {})
        save_result(args.out, {"source": str(args.result), **out,
                               "_summary": {
                                   "write_ratio_deep": json.dumps(
                                       {k: v["write"] for k, v in deep.items()}),
                                   "write_ratio_shallow": json.dumps(
                                       {k: v["write"] for k, v in shallow.items()}),
                               }},
                     experiment="chain_write_budget", hypothesis=["H25"],
                     model=data.get("model"),
                     script="experiments/rl/chain_write_budget.py")
        print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
