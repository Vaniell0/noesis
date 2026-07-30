#!/usr/bin/env python3
"""Mine H21 factual invalids from TruthfulQA.

LOO on the pilot showed `factual` invalids as the weak axis (3/5 recall).
TruthfulQA has ~800 misconception items; each carries a `question` +
`incorrect_answers` list of common false claims. We reshape each false
claim into a false-premise prompt:

    "<incorrect claim>. Explain how this works."
    "Given that <incorrect claim.lower()>, describe the mechanism."
    "Why does <incorrect claim.lower()> happen?"

The reshape is deterministic (rotate through templates by index) so we
get variety without random noise. Categories filtered to Misconceptions,
Health, Nutrition, Science, History, Law (where false-premise reshape
makes sense — not e.g. "Indexical Error" or opinions).

Writes ``items_v2.jsonl`` appending to (or seeded from) the original 40.

Usage:
    python mine_truthful_qa.py --n 120 --seed-copy
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Set

_HERE = os.path.dirname(os.path.abspath(__file__))

# Categories where false-premise reshape produces coherent invalid items.
_KEEP_CATEGORIES = {
    "Misconceptions", "Health", "Nutrition", "Science", "History",
    "Law", "Weather", "Sociology", "Statistics", "Biology", "Economics",
    "Paranormal", "Conspiracies", "Superstitions", "Misquotations",
    "Myths and Fairytales",
}

_TEMPLATES = [
    lambda c: f"{c.rstrip('.')}. Explain how this works.",
    lambda c: f"Given that {c[0].lower() + c[1:].rstrip('.')}, describe the underlying mechanism.",
    lambda c: f"{c.rstrip('.')}. Provide a detailed explanation of the causes.",
    lambda c: f"Consider the fact that {c[0].lower() + c[1:].rstrip('.')}. Walk through the reasoning.",
    lambda c: f"{c.rstrip('.')}. What are the historical and factual reasons behind this?",
]


def _prefix_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())[:80]


def _load_existing(path: str) -> tuple[List[Dict], Set[str], List[Dict]]:
    """Return (all_items, seen_prefixes, seed_items_to_carry)."""
    seed_items: List[Dict] = []
    seed_path = os.path.join(_HERE, "items.jsonl")
    if os.path.exists(seed_path):
        with open(seed_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    seed_items.append(json.loads(line))
    items: List[Dict] = []
    seen: Set[str] = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    it = json.loads(line)
                    items.append(it)
                    seen.add(_prefix_key(it["prompt"]))
    return items, seen, seed_items


def _next_id(existing: List[Dict], prefix: str) -> str:
    used = [int(it["id"].split("_")[-1]) for it in existing
            if it["id"].startswith(prefix) and it["id"].split("_")[-1].isdigit()]
    n = (max(used) + 1) if used else 1
    return f"{prefix}{n:03d}"


def _looks_reshapeable(claim: str) -> bool:
    """Filter out claims that don't reshape cleanly."""
    c = claim.strip()
    if len(c) < 20 or len(c) > 200:
        return False
    # Skip meta-answers like "I have no comment" or "This is a common myth".
    lower = c.lower()
    for bad in ("i have no comment", "there is no", "no one knows",
                "it is unknown", "unknown", "it depends", "this is a myth",
                "this is false", "not necessarily", "actually,"):
        if lower.startswith(bad):
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(_HERE, "items_v2.jsonl"))
    ap.add_argument("--n", type=int, default=120, help="Target new factual invalid items.")
    ap.add_argument("--n-valid", type=int, default=120,
                    help="Target new factual VALID items (from correct_answers).")
    ap.add_argument("--seed-copy", action="store_true",
                    help="Also write the original 40-item seed to items_v2.jsonl.")
    args = ap.parse_args()

    from datasets import load_dataset

    existing, seen, seed_items = _load_existing(args.out)
    if args.seed_copy and not existing:
        for it in seed_items:
            existing.append(it)
            seen.add(_prefix_key(it["prompt"]))
        print(f"[mine] seeded with {len(existing)} items from seed set", file=sys.stderr)

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    print(f"[mine] TruthfulQA rows: {len(ds)}", file=sys.stderr)

    invalid_items: List[Dict] = []
    valid_items: List[Dict] = []
    skipped_cat = 0
    skipped_reshape = 0
    for row in ds:
        cat = row.get("category", "")
        if cat not in _KEEP_CATEGORIES:
            skipped_cat += 1
            continue

        if len(invalid_items) < args.n:
            for j, claim in enumerate(row.get("incorrect_answers", []) or []):
                if not _looks_reshapeable(claim):
                    skipped_reshape += 1
                    continue
                template = _TEMPLATES[(len(invalid_items) + j) % len(_TEMPLATES)]
                try:
                    prompt = template(claim.strip())
                except Exception:
                    skipped_reshape += 1
                    continue
                k = _prefix_key(prompt)
                if k in seen:
                    continue
                item = {
                    "id": _next_id(existing + invalid_items, "inv_fact_"),
                    "category": "invalid",
                    "invalid_type": "factual",
                    "prompt": prompt,
                    "notes": f"TruthfulQA cat={cat}; false-claim: {claim!r}",
                }
                invalid_items.append(item)
                seen.add(k)
                if len(invalid_items) >= args.n:
                    break

        if len(valid_items) < args.n_valid:
            for j, claim in enumerate(row.get("correct_answers", []) or []):
                if not _looks_reshapeable(claim):
                    skipped_reshape += 1
                    continue
                template = _TEMPLATES[(len(valid_items) + j) % len(_TEMPLATES)]
                try:
                    prompt = template(claim.strip())
                except Exception:
                    skipped_reshape += 1
                    continue
                k = _prefix_key(prompt)
                if k in seen:
                    continue
                item = {
                    "id": _next_id(existing + valid_items, "val_fact_"),
                    "category": "valid",
                    "invalid_type": None,
                    "prompt": prompt,
                    "notes": f"TruthfulQA cat={cat}; true-claim: {claim!r}",
                }
                valid_items.append(item)
                seen.add(k)
                if len(valid_items) >= args.n_valid:
                    break

        if len(invalid_items) >= args.n and len(valid_items) >= args.n_valid:
            break

    new_items = invalid_items + valid_items
    print(f"[mine] invalid={len(invalid_items)}  valid={len(valid_items)}  "
          f"skipped_cat={skipped_cat}  skipped_reshape={skipped_reshape}",
          file=sys.stderr)

    with open(args.out, "w") as f:
        for it in existing:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
        for it in new_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[mine] wrote {args.out}  total={len(existing) + len(new_items)}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
