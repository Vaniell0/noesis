#!/usr/bin/env python3
"""spatium rewrite derivations as text items for train_latent_s1.py (laptop precursor of the World vs G1d round).

    python experiments/rl/deriv_items.py ~/tmp/noesis-export/items.jsonl OUT.jsonl --train 800 --held 200

Each item (true derivations only): eight rules #1..#8 and a question t0 = t; the answer is the local id of the rule
used at the last hop ("#k"). Symbols are renamed per item within their class (binary operations, letters,
constants), so no law is recognisable and the rules must be read. Held-out items come from spatium's val_laws split
and are placed at every --heldout-every-th id, which is what train_latent_s1.py holds out; depths are balanced
1..8 in both halves. Each row keeps depth and split for the per-depth read.
"""
import argparse
import json
import random

CLASSES = [["log", "exp", "d", "scale", "vadd", "add", "mul"], ["p", "q", "r", "u", "w"], ["0", "1/2", "2", "-1"]]


def rename(tokens, rng):
    m = {}
    for cls in CLASSES:
        perm = cls[:]
        rng.shuffle(perm)
        m.update(zip(cls, perm))
    return [m.get(t, t) for t in tokens]


def render(r, rng):
    toks = rename(r["tokens"], rng)
    rules, cur, q = [], None, None
    for t in toks:
        if t.startswith("#") and t[1:].isdigit():
            cur = [t]; rules.append(cur)
        elif t == "?":
            q = []
        elif q is not None:
            q.append(t)
        else:
            cur.append(t)
    lines = [" ".join(x) for x in rules]
    prompt = ("Rewrite rules (Polish notation, either direction, at any subterm):\n" + "\n".join(lines) +
              "\nThe left side of the question rewrites into the right side.\nQuestion: " + " ".join(q) +
              "\nWhich rule is applied at the last step? Answer with its tag.")
    return prompt, f"#{r['last_rule']}"


def pick(rows, n, rng):
    by = {}
    for r in rows:
        by.setdefault(r["depth"], []).append(r)
    out = []
    for d in sorted(by):
        rng.shuffle(by[d])
        out += by[d][: n // len(by)]
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("items")
    ap.add_argument("out")
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--held", type=int, default=200)
    ap.add_argument("--heldout-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [json.loads(line) for line in open(args.items)]
    rows = [r for r in rows if r["answer"] == "true"]
    train = pick([r for r in rows if r["split"] == "train"], args.train, rng)
    held = pick([r for r in rows if r["split"] == "val_laws"], args.held, rng)
    k = args.heldout_every
    n = 0
    with open(args.out, "w") as f:
        while train or held:
            src = held if (n % k == 0 and held) or not train else train
            r = src.pop()
            prompt, answer = render(r, rng)
            f.write(json.dumps({"id": n, "prompt": prompt, "answer": answer, "depth": r["depth"],
                                "split": "held" if src is held else "train", "spatium_id": r["id"]}) + "\n")
            n += 1
    print(f"{n} items -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
