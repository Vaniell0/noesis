#!/usr/bin/env python3
"""Minimal-pair items for H21 and H22 whose classes differ in CONTENT, not wording
(check C-H21-H22-redesign, hypotheses/checks-h20-h26.yaml; owner 2026-10-20).

The v2 sets were mined by the very phrases the head is asked to detect, and a bag-of-words
classifier beat the WKV-state head on H22 (0.975 vs 0.947). Here every word pool is shared
by both classes, so word presence alone cannot separate them:

  h22  every prompt = a memo stating two facts + a claim "according to the memo above".
       Attributable (y=1): the claim restates one fact. Unattributed (y=0): the claim binds one
       fact's entity to the other fact's relation and value - all its words are in the memo, the
       memo never says it. Same word set in both classes; only the binding differs.
  h21  questions predicate(entity). Valid (y=1): the predicate fits the entity's category.
       Invalid (y=0): a predicate of another category. Every entity and every predicate
       appears in both classes across the set.

Output rows: {"id", "category", "prompt", "y"} - category "attributable"/"unattributed" or
"valid"/"invalid", as the existing run.py / loo.py expect. Gate before any state head is
trained on a set: `bow_control.py --gate <items>` must give accuracy <= 0.55.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

H22_ENTITIES = ["the Halden bridge", "the north warehouse", "the Mirov account", "the spring audit",
                "the Ostrava plant", "the Kell project", "the blue fleet", "the Tarn office",
                "the Lusa contract", "the west server room", "the Brim ferry", "the Onda branch"]
H22_FACTS = [("closes at", ["six", "seven", "eight", "nine"]),
             ("was moved to", ["Monday", "Tuesday", "Thursday", "Friday"]),
             ("is managed by", ["Petra", "Anil", "Sofia", "Marek"]),
             ("has a budget of", ["forty thousand", "sixty thousand", "ninety thousand", "two hundred thousand"])]

H21_CATS = {
    "colour": (["the sky", "fresh grass", "a ripe lemon", "coal", "snow", "a ripe tomato"],
               ["What colour is {e}?", "Is {e} usually bright or dark in colour?"]),
    "weekday": (["Tuesday", "Friday", "Sunday", "Monday", "Wednesday", "Saturday"],
                ["Which day comes right after {e}?", "Is {e} part of the working week?"]),
    "number": (["seven", "twelve", "forty", "three", "ninety", "eleven"],
               ["Is {e} an even number?", "What is {e} multiplied by two?"]),
    "animal": (["a red panda", "a goat", "an owl", "a salmon", "a horse", "a frog"],
               ["What does {e} usually eat?", "Where does {e} usually live?"]),
}


def gen_h22(n_pairs: int, rng: random.Random) -> list:
    """The memo states two facts. Attributable: the claim restates one of them. Unattributed: the
    claim binds the entity of one fact to the relation and value of the other - every word of it is
    in the memo, but the memo never says it. Same memo, same word set in both classes; only the
    binding differs (a word-presence classifier sees identical inputs up to word order)."""
    rows = []
    for i in range(n_pairs):
        e1, e2 = rng.sample(H22_ENTITIES, 2)
        (r1, vals1), (r2, vals2) = rng.sample(H22_FACTS, 2)
        v1, v2 = rng.choice(vals1), rng.choice(vals2)
        memo = f"Memo from the operations desk: {e1} {r1} {v1}; {e2} {r2} {v2}."
        if rng.random() < 0.5:
            claim_ok, claim_bad = f"{e1} {r1} {v1}", f"{e2} {r1} {v1}"
        else:
            claim_ok, claim_bad = f"{e2} {r2} {v2}", f"{e1} {r2} {v2}"
        rows.append({"id": f"mp22_{i:03d}a", "category": "attributable", "y": 1,
                     "prompt": f"{memo}\nAccording to the memo above, {claim_ok}."})
        rows.append({"id": f"mp22_{i:03d}u", "category": "unattributed", "y": 0,
                     "prompt": f"{memo}\nAccording to the memo above, {claim_bad}."})
    return rows


def gen_h21(n_pairs: int, rng: random.Random) -> list:
    rows, cats = [], list(H21_CATS)
    for i in range(n_pairs):
        c1, c2 = rng.sample(cats, 2)
        e = rng.choice(H21_CATS[c1][0])
        tmpl_ok = rng.choice(H21_CATS[c1][1])
        tmpl_bad = rng.choice(H21_CATS[c2][1])
        rows.append({"id": f"mp21_{i:03d}v", "category": "valid", "y": 1, "prompt": tmpl_ok.format(e=e)})
        rows.append({"id": f"mp21_{i:03d}i", "category": "invalid", "invalid_type": "category", "y": 0,
                     "prompt": tmpl_bad.format(e=e)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["h21", "h22"], required=True)
    ap.add_argument("--pairs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = gen_h22(args.pairs, rng) if args.which == "h22" else gen_h21(args.pairs, rng)
    args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
