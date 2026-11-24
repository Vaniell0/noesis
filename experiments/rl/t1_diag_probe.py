#!/usr/bin/env python3
"""t1_diag_probe.py - what did the 1t distillation change: reading the table, or learning the Caesar shift?

    python experiments/rl/t1_diag_probe.py --model G1D --weights CK.pth --ticks 0,1,2,4,8 --out OUT.json

Inverse-key ladder rows (table plain -> cipher), teacher-forced on the gold trace; the score is the share of the
letters after '=' predicted right. Three reads per tick count:
  own      the row's own table (what training saw, on held rows);
  swapped  the same message with another row's table of the same cipher family: reading the table must fall to the
           letter prior, a learned shift or prior would not;
  family   the score per cipher family: the 1t pilot trained on Caesar rows only, so katakana and Cyrillic
           substitutions are transfer.
The base model is read at M = 0 only (no ticks on a model not trained for them). Inference only (plan 1t).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiments.rl.loader import load_rwkv7, load_weights_into
from experiments.rl.train_latent_s1 import tick_logits_all


def letters_score(lg: torch.Tensor, a: list, tok) -> float:
    """Share of the letters after '=' whose teacher-forced argmax is the gold token."""
    pred = lg.argmax(-1).tolist()
    pos = [i for i in range(1, len(a)) if tok.decode(a[:i]).rstrip().endswith("=")]
    return sum(pred[i] == a[i] for i in pos) / max(1, len(pos))


def swap_table(r: dict, other: dict) -> dict:
    t = re.search(r"Table: [^\n]*", other["prompt"]).group(0)
    return dict(r, prompt=re.sub(r"Table: [^\n]*", t, r["prompt"]))


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--items", default="training/corpus_open/ciphers_ladder_indexed_inv_eval.jsonl")
    ap.add_argument("--per-family", type=int, default=24)
    ap.add_argument("--ticks", default="0,1,2,4,8")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rows = [json.loads(l) for l in open(a.items)]
    fam = {}
    for r in rows:
        fam.setdefault(r["answer_format"].split("/")[0], []).append(r)
    rng = random.Random(0)
    items = []
    for f, rs in sorted(fam.items()):
        rs = rs[:a.per_family]
        for i, r in enumerate(rs):
            items.append((f, r, swap_table(r, rs[(i + 1) % len(rs)])))
    loaded = load_rwkv7(a.model, device="cpu", dtype=torch.float32, backend="peft", ctx_len=2048)
    if a.weights:
        load_weights_into(loaded, a.weights)
    tok = loaded.tokenizer
    end = tok.encode("\n\n")
    ticks = [int(x) for x in a.ticks.split(",")]
    if not a.weights:
        ticks = [0]
    res, t0 = {}, time.time()
    for f, r, rs in items:
        for kind, row in (("own", r), ("swapped", rs)):
            row = dict(row); row["_ans"] = tok.encode(row["trace"]) + end
            ans = row["_ans"]
            if 0 in ticks:
                from experiments.rl.train_latent_s1 import seq_logits
                lg, _ = seq_logits(loaded, tok, row, 0)
                res.setdefault((f, kind, 0), []).append(letters_score(lg, ans, tok))
            mx = max(ticks)
            if mx:
                per = tick_logits_all(loaded, tok, row, mx)
                for m in ticks:
                    if m:
                        res.setdefault((f, kind, m), []).append(letters_score(per[m - 1], ans, tok))
        print(f"[t1diag] {f} done {time.time() - t0:.0f}s", flush=True)
    out = {f"{f}_{k}_M{m}": round(sum(v) / len(v), 4) for (f, k, m), v in sorted(res.items())}
    for k, v in out.items():
        print(f"[t1diag] {k}: {v:.3f}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"model": a.model, "weights": str(a.weights) if a.weights else None,
                                 "per_family": a.per_family, "letters_right": out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
