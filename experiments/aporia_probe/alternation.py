#!/usr/bin/env python3
"""H20 alternation probe (check C-H20-alternation, hypotheses/checks-h20-h26.yaml).

Question: when the model continues past an ambiguity site, does the state keep
both alternatives readable by ALTERNATING between them (as Hesse, ..., Tsao 2026
report for seen vs suppressed stimuli), or does it collapse to one?

For each item: prefill the prompt, then greedy-decode T tokens. After every
step t (t = 0 .. T) a clone of the state is read and both alternatives are scored by teacher forcing.
Readout ``bridge`` (default since 2026-10-20) feeds only a short constant phrase; ``reask`` feeds the
whole question again, which re-reads the context (an N=2 effect) and was the first run's readout:

    d_t = log p(X | state_t, re-ask) - log p(Y | state_t, re-ask)

Two trajectories per item:
  own      - the model's own greedy continuation;
  shuffled - another item's greedy tokens fed into this item's state
             (control: any text moves the state; does the own path flip more?)

Prediction written before the run (2026-10-20):
  collapse    - after the first few steps sign(d_t) is fixed and |d_t| grows;
                decisive flips (sign change between steps with |d| >= MARGIN)
                in own <= shuffled.
  alternation - decisive flips in own clearly above shuffled (paired by item).
Kill for the alternation reading: own decisive flips not above shuffled.

Usage:
  python alternation.py --items items_100.jsonl --steps 16 --out alternation_g1d04b.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

import torch  # noqa: E402
from _common.model import load_model  # noqa: E402

DEFAULT_MODEL = os.path.expanduser("~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth")
MARGIN = 0.5  # nats; a step counts as decisive only beyond this gap


def clone(state):
    return [t.clone() for t in state]


def flat(logits):
    return logits.reshape(-1).float() if logits.dim() > 1 else logits.float()


def seq_logprob(model, state, logits, ids):
    """log p(ids) continuing from (state, logits); state is consumed."""
    total = 0.0
    for tok in ids:
        total += float(torch.log_softmax(flat(logits), -1)[tok])
        logits, state = model.forward([tok], state)
    return total


READOUT = "reask"  # set from --readout in main()
BRIDGE = " ... In short, it was"


def readout(model, tok, state, prompt, alts):
    """reask: feed the whole question again (N=2 confound: re-reading strengthens the prompt prior);
    bridge: a short constant phrase, no history re-fed - reads the state as it is."""
    st = clone(state)
    text = ("\n\n" + prompt) if READOUT == "reask" else BRIDGE
    logits, st = model.forward(tok(text)["input_ids"], st)
    scores = []
    for alt in alts:
        scores.append(seq_logprob(model, clone(st), logits.clone(), tok(" " + alt)["input_ids"]))
    return scores[0] - scores[1]


def trajectory(model, tok, prompt, alts, feed_ids):
    """d_t for t = 0..len(feed_ids); feed_ids None means greedy decode."""
    logits, state = model.forward(tok(prompt)["input_ids"], None)
    d = [readout(model, tok, state, prompt, alts)]
    emitted = []
    n = len(feed_ids)
    for t in range(n):
        nxt = feed_ids[t] if feed_ids[t] is not None else int(torch.argmax(flat(logits)))
        emitted.append(nxt)
        logits, state = model.forward([nxt], state)
        d.append(readout(model, tok, state, prompt, alts))
    return d, emitted


def decisive_flips(d):
    signs = [math.copysign(1, x) for x in d if abs(x) >= MARGIN]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def raw_flips(d):
    return sum(1 for a, b in zip(d, d[1:]) if (a > 0) != (b > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--items", default=os.path.join(_HERE, "items_100.jsonl"))
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(_HERE, "alternation_g1d04b.json"))
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--readout", default="bridge", choices=["bridge", "reask"])
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.items) if l.strip()]
    if args.limit:
        items = items[: args.limit]
    global READOUT
    READOUT = args.readout
    torch.set_num_threads(args.threads)
    model, tok = load_model(args.model, device="cpu", dtype=args.dtype)
    t0 = time.time()

    own = []
    for it in items:
        d, emitted = trajectory(model, tok, it["prompt"], it["alternatives"], [None] * args.steps)
        own.append({"id": it["id"], "category": it["category"], "d": d, "emitted": emitted,
                    "text": tok.decode(emitted)})
    # shuffled control: item i receives item (i+1)'s greedy tokens
    for i, it in enumerate(items):
        donor = own[(i + 1) % len(items)]["emitted"]
        d, _ = trajectory(model, tok, it["prompt"], it["alternatives"], donor)
        own[i]["d_shuffled"] = d
        own[i]["donor"] = own[(i + 1) % len(items)]["id"]

    def summ(rows):
        df = [decisive_flips(r["d"]) for r in rows]
        ds = [decisive_flips(r["d_shuffled"]) for r in rows]
        diff = [a - b for a, b in zip(df, ds)]
        return {
            "n": len(rows),
            "decisive_flips_own_mean": statistics.mean(df),
            "decisive_flips_shuffled_mean": statistics.mean(ds),
            "paired_diff_mean": statistics.mean(diff),
            "paired_diff_sd": statistics.pstdev(diff),
            "items_own_gt_shuffled": sum(1 for x in diff if x > 0),
            "items_own_lt_shuffled": sum(1 for x in diff if x < 0),
            "raw_flips_own_mean": statistics.mean(raw_flips(r["d"]) for r in rows),
            "raw_flips_shuffled_mean": statistics.mean(raw_flips(r["d_shuffled"]) for r in rows),
            "abs_d_first_mean": statistics.mean(abs(r["d"][0]) for r in rows),
            "abs_d_last_own_mean": statistics.mean(abs(r["d"][-1]) for r in rows),
            "abs_d_last_shuffled_mean": statistics.mean(abs(r["d_shuffled"][-1]) for r in rows),
        }

    cats = sorted({r["category"] for r in own})
    out = {
        "model": os.path.basename(args.model), "items": os.path.basename(args.items), "dtype": args.dtype,
        "steps": args.steps, "margin_nats": MARGIN, "readout": args.readout, "bridge": BRIDGE if args.readout == "bridge" else None,
        "wall_s": round(time.time() - t0, 1),
        "all": summ(own), "by_category": {c: summ([r for r in own if r["category"] == c]) for c in cats},
        "rows": own,
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("model", "steps", "wall_s", "all", "by_category")}, indent=1))


if __name__ == "__main__":
    main()
