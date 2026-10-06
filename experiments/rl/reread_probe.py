#!/usr/bin/env python3
"""reread_probe.py — does feeding the prompt N times help? (H10's N axis, controlled)

The user's reading of why N=2 worked in H10 (2026-08, N=2 silent 33.3% beat every
CoT cell, N=3 collapsed to 6.3%): the second pass gives the state time to catch
up — the most primitive form of the model spending extra steps on the question.
This tests it directly on staged-recall items with many shared-key pairs, where
a single pass is not enough. No training: same weights, prompt fed N = 1, 2, 3
times, answer read as in train_staged_recall.py (argmax over the 100 value
tokens). Run on the base model and on the P1 checkpoint (--weights).

Reported per cell and N: accuracy; and the change vs N=1 on the SAME items.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rl.loader import load_rwkv7, load_weights_into  # noqa: E402
from experiments.rl.train_staged_recall import _candidate_ids, _load  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--eval", type=Path, default=Path("training/corpus_open/p2_hard_eval.jsonl"))
    ap.add_argument("--n-list", default="1,2,3")
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(10)

    loaded = load_rwkv7(args.model, device="cpu", dtype=torch.float32,
                        backend="peft", lora_r=0, ctx_len=4096)
    if args.weights is not None:
        load_weights_into(loaded, args.weights)
    tok = loaded.tokenizer
    cand = _candidate_ids(tok)
    items = [r for r in _load(args.eval) if r["arm"] == "recall"]
    by = defaultdict(list)
    for r in items:
        by[(r["n_pairs"], r["gap_words"])].append(r)
    chosen = [r for k in sorted(by) for r in by[k][:args.per_cell]]
    ns = [int(x) for x in args.n_list.split(",")]
    hits = defaultdict(lambda: defaultdict(list))        # cell -> N -> [0/1]
    for i, r in enumerate(chosen):
        cell = f"n{r['n_pairs']}|g{r['gap_words']}"
        for n in ns:
            text = "\n".join([r["prompt"]] * n)
            ids = tok.encode(text)
            st = loaded.new_state(batch=1)
            logits, _ = loaded.forward_stateful(torch.tensor([ids]), st)
            pred = int(logits[0, -1, cand.to(logits.device)].float().argmax())
            hits[cell][n].append(int(pred == int(r["answer"])))
        print(f"[reread] {i+1}/{len(chosen)} {cell}", flush=True)
    res = {"model": args.model, "weights": str(args.weights), "cells": {}, "overall": {}}
    for cell, d in sorted(hits.items()):
        res["cells"][cell] = {str(n): round(sum(v) / len(v), 3) for n, v in d.items()}
    for n in ns:
        allv = [x for d in hits.values() for x in d[n]]
        res["overall"][str(n)] = round(sum(allv) / len(allv), 3)
    print(json.dumps(res, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
