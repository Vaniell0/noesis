#!/usr/bin/env python3
"""Mine H22 items from streaming C4:en.

Filters sentences by rhetorical-marker regex into two buckets:

- ``unattributed``: appeals to a collective without source
  ("it is generally accepted", "most people believe", "usually X",
  "one might argue", "the scientific community agrees", …)
- ``attributable``: names a source or scopes to a specific referent
  ("According to X (2020)", "Y et al. showed", "in my last three
  sessions", "based on the logs at /var/log/…", …)

Streams C4:en; splits paragraphs into sentences; keeps sentences of
length in [40, 220] chars that match exactly one bucket's pattern set.
Deduplicates on prefix.

Writes ``items_v2.jsonl`` (append-safe) with the same schema as the
seed set: ``{id, category, prompt, notes}``.

Usage:
    python mine_c4.py --n-per-class 120 --out items_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, Iterable, List, Tuple

# --- Regex patterns ---------------------------------------------------------

# Unattributed collective claims.
UNATTR_PATTERNS = [
    r"\bit is (?:generally|widely|commonly) (?:accepted|believed|held|thought|assumed|understood|acknowledged)\b",
    r"\bit has (?:been )?(?:generally|widely|commonly)? ?(?:shown|argued|noted|reported|claimed|said)\b",
    r"\bmost (?:people|scientists|experts|researchers|developers|users|professionals) (?:believe|think|say|agree|argue|find|prefer)\b",
    r"\busually (?:developers|users|people|professionals|scientists|researchers)\b",
    r"\bthe (?:scientific |research )?community (?:agrees|believes|holds|thinks|argues)\b",
    r"\bmany (?:argue|believe|suggest|say|think|find)\b",
    r"\bone might (?:argue|say|think|believe|suggest)\b",
    r"\bit is common(?:ly)? (?:accepted|held|believed|assumed|thought)\b",
    r"\bin general,? (?:users|people|developers|professionals|scientists|programmers)\b",
    r"\bsome (?:argue|say|claim|believe|think|suggest) that\b",
    r"\bit is often (?:said|thought|assumed|argued|claimed|believed)\b",
    r"\bexperts (?:agree|say|believe|argue|claim|suggest)\b",
    r"\bpeople (?:generally|often|usually|commonly) (?:believe|think|say|find|prefer|agree)\b",
]

# Attributable claims.
ATTR_PATTERNS = [
    r"\baccording to [A-Z][a-zA-Z]+(?: (?:and|et al\.?) [A-Z][a-zA-Z]+)?(?:,? \(?\d{4}\)?)?\b",
    r"\b[A-Z][a-zA-Z]+ et al\.?,? \(?\d{4}\)?\b",
    r"\b[A-Z][a-zA-Z]+ and [A-Z][a-zA-Z]+ \(?\d{4}\)?\b",
    r"\bcited in [A-Z][a-zA-Z]+\b",
    r"\bin (?:my|his|her|our|their) (?:last|previous|recent|earlier) (?:conversation|session|study|paper|memo|report|analysis|experiment)\b",
    r"\bbased on (?:the|our|my|his|her|their) (?:logs?|data|memo|report|paper|study|records?|measurements?|observations?) (?:at|in|from|for)\b",
    r"\bin the (?:paper|manual|README|documentation|book|memo|study|report) by [A-Z][a-zA-Z]+\b",
    r"\bthe (?:paper|study|report|manual) (?:titled|entitled|by) [\"'A-Z]",
    r"\b(?:as )?(?:shown|reported|demonstrated|argued|noted) (?:in|by) [A-Z][a-zA-Z]+(?:'s)? (?:paper|study|book|work|report|memo|analysis)\b",
    r"\bin (?:19|20)\d{2}, [A-Z][a-zA-Z]+ (?:showed|argued|reported|demonstrated|found|noted|claimed)\b",
    r"\bthe (?:19|20)\d{2} (?:paper|study|report|memo|analysis) by [A-Z]",
    r"\b(?:from|per) (?:the )?[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)? (?:19|20)\d{2}\b",
]

_UNATTR_RX = [re.compile(p, re.IGNORECASE) for p in UNATTR_PATTERNS]
_ATTR_RX = [re.compile(p) for p in ATTR_PATTERNS]  # attribution needs case (proper nouns)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _sentences(text: str) -> Iterable[str]:
    """Simple sentence split. Good enough for regex mining."""
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for s in _SENT_SPLIT.split(para):
            s = s.strip()
            if 40 <= len(s) <= 220:
                yield s


def _classify(sentence: str) -> Tuple[str, str]:
    """Return (category, matched_pattern) or ('none', '')."""
    for rx in _UNATTR_RX:
        m = rx.search(sentence)
        if m:
            return "unattributed", m.group(0)
    for rx in _ATTR_RX:
        m = rx.search(sentence)
        if m:
            return "attributable", m.group(0)
    return "none", ""


def _prefix_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())[:80]


def _load_existing(path: str) -> Tuple[List[Dict], set]:
    """Load existing items (if any) to prevent duplicates."""
    items: List[Dict] = []
    seen = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                it = json.loads(line)
                items.append(it)
                seen.add(_prefix_key(it["prompt"]))
    return items, seen


def _next_id(existing: List[Dict], category: str) -> str:
    prefix = "attr_" if category == "attributable" else "unattr_"
    used = [int(it["id"].split("_")[-1]) for it in existing
            if it["id"].startswith(prefix) and it["id"].split("_")[-1].isdigit()]
    n = (max(used) + 1) if used else 1
    return f"{prefix}{n:03d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "items_v2.jsonl"))
    ap.add_argument("--n-per-class", type=int, default=120,
                    help="Target items per class (attributable, unattributed).")
    ap.add_argument("--max-docs", type=int, default=200_000,
                    help="Cap on streamed C4 documents (safety net).")
    ap.add_argument("--seed-copy", action="store_true",
                    help="Also copy the 19-item seed set from items.jsonl into items_v2.jsonl.")
    args = ap.parse_args()

    from datasets import load_dataset

    existing, seen = _load_existing(args.out)
    if args.seed_copy and not existing:
        seed_path = os.path.join(os.path.dirname(args.out), "items.jsonl")
        if os.path.exists(seed_path):
            with open(seed_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        it = json.loads(line)
                        existing.append(it)
                        seen.add(_prefix_key(it["prompt"]))
            print(f"[mine] seeded with {len(existing)} items from {seed_path}",
                  file=sys.stderr)

    counts = {"attributable": 0, "unattributed": 0}
    for it in existing:
        if it["category"] in counts:
            counts[it["category"]] += 1
    target_attr = max(0, args.n_per_class - counts["attributable"])
    target_unattr = max(0, args.n_per_class - counts["unattributed"])
    print(f"[mine] need {target_attr} attributable + {target_unattr} unattributed "
          f"(existing attr={counts['attributable']} unattr={counts['unattributed']})",
          file=sys.stderr)

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    new_items: List[Dict] = []
    t0 = time.time()
    n_docs = 0
    for row in ds:
        n_docs += 1
        if n_docs > args.max_docs:
            break
        if target_attr <= 0 and target_unattr <= 0:
            break
        text = row.get("text", "")
        for sent in _sentences(text):
            k = _prefix_key(sent)
            if k in seen:
                continue
            cat, matched = _classify(sent)
            if cat == "none":
                continue
            if cat == "attributable" and target_attr <= 0:
                continue
            if cat == "unattributed" and target_unattr <= 0:
                continue
            item = {
                "id": _next_id(existing + new_items, cat),
                "category": cat,
                "prompt": sent,
                "notes": f"c4-mined; matched: {matched!r}",
            }
            new_items.append(item)
            seen.add(k)
            if cat == "attributable":
                target_attr -= 1
            else:
                target_unattr -= 1
            if len(new_items) % 20 == 0:
                print(f"[mine] {len(new_items)} new items "
                      f"(docs seen={n_docs} wall={time.time()-t0:.1f}s)",
                      file=sys.stderr, flush=True)

    print(f"[mine] wrote {len(new_items)} new items; total will be {len(existing) + len(new_items)}",
          file=sys.stderr)

    with open(args.out, "w") as f:
        for it in existing:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
        for it in new_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[mine] {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
