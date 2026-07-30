#!/usr/bin/env python3
"""Build retrieval-augmented item pairs for H21 sanity test.

For each factual item (val_fact_* or inv_fact_*) whose source claim is
findable in TruthfulQA, emit two versions:

- ``<id>``:      original prompt (no context) — baseline
- ``<id>_ctx``:  augmented with ``Context: <best_answer>. `` prefix

Predicts: if state uses retrieved context, then adding a correct-fact
context to an inv_fact item should lower p_valid (head now sees the
mismatch); adding it to a val_fact item should keep p_valid high.

The LOO comparison base vs ctx will show whether the state actually
reads the retrieved fact.

Usage:
    python build_ctx_items.py --n-inv 20 --n-val 20 --out items_ctx.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_FALSE_RX = re.compile(r"false-claim: '(.+)'\Z")
_TRUE_RX = re.compile(r"true-claim: '(.+)'\Z")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-inv", type=int, default=20)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--items", default=os.path.join(_HERE, "items_v2.jsonl"))
    ap.add_argument("--out", default=os.path.join(_HERE, "items_ctx.jsonl"))
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")

    # Build lookup: claim text (lowercased+stripped) -> best_answer
    incorrect_to_best: Dict[str, str] = {}
    correct_to_best: Dict[str, str] = {}
    for row in ds:
        best = (row.get("best_answer") or "").strip()
        if not best:
            continue
        for c in row.get("incorrect_answers", []) or []:
            k = c.strip().lower()
            if k and k not in incorrect_to_best:
                incorrect_to_best[k] = best
        for c in row.get("correct_answers", []) or []:
            k = c.strip().lower()
            if k and k not in correct_to_best:
                correct_to_best[k] = best

    print(f"[ctx] index: incorrect={len(incorrect_to_best)}  correct={len(correct_to_best)}",
          file=sys.stderr)

    items = [json.loads(l) for l in open(args.items) if l.strip()]
    inv_pairs: List[Dict] = []
    val_pairs: List[Dict] = []
    missed = 0
    for it in items:
        note = it.get("notes", "") or ""
        m_false = _FALSE_RX.search(note)
        m_true = _TRUE_RX.search(note)
        if not (m_false or m_true):
            continue
        claim = (m_false or m_true).group(1).strip().lower()
        lookup = incorrect_to_best if m_false else correct_to_best
        best = lookup.get(claim)
        if not best:
            missed += 1
            continue

        base = dict(it)  # baseline copy
        ctx = dict(it)
        ctx["id"] = it["id"] + "_ctx"
        ctx["prompt"] = f"Context: {best}. {it['prompt']}"
        ctx["notes"] = f"retrieval-augmented; ctx_best={best!r}; source: {it.get('notes','')}"

        if m_false and len(inv_pairs) < args.n_inv * 2:
            inv_pairs.extend([base, ctx])
        elif m_true and len(val_pairs) < args.n_val * 2:
            val_pairs.extend([base, ctx])
        if len(inv_pairs) >= args.n_inv * 2 and len(val_pairs) >= args.n_val * 2:
            break

    out = inv_pairs + val_pairs
    print(f"[ctx] emitting: inv={len(inv_pairs)//2} pairs  val={len(val_pairs)//2} pairs  "
          f"missed={missed}", file=sys.stderr)

    with open(args.out, "w") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[ctx] wrote {args.out}  total_items={len(out)}", file=sys.stderr)

    # Show 2 examples
    print("\n--- example inv (base vs ctx) ---", file=sys.stderr)
    if inv_pairs:
        for it in inv_pairs[:2]:
            print(f"  {it['id']}: {it['prompt'][:140]}", file=sys.stderr)
    print("--- example val (base vs ctx) ---", file=sys.stderr)
    if val_pairs:
        for it in val_pairs[:2]:
            print(f"  {it['id']}: {it['prompt'][:140]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
