#!/usr/bin/env python3
"""WKV-state head on the minimal-pair sets (check C-H21-H22-redesign), evaluated leave-PAIR-out.

Same feature (per-layer per-head mean+std of the WKV state after the prompt, run.py::_pool_state)
and the same 128-64-1 MLP head (loo.py::_train_one) as the H21/H22 pilots, so the number is
comparable; only the data (gen_minimal_pairs.py, classes share their words, BoW gate 0.50-0.53)
and the split (the whole pair is held out) differ. Labels come from each row's "y".

    python experiments/attribution_probe/pair_head.py --items experiments/attribution_probe/items_mp_h22.jsonl \\
        --out experiments/attribution_probe/mp_h22_g1d04b
Prediction written before the run (2026-10-20): if the state carries attribution (H22) / premise validity
(H21) beyond word presence, accuracy > 0.60 on held-out pairs; at or near 0.50 the earlier positives were
lexical.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "A0_state_probe"))
sys.path.insert(0, _HERE)  # last insert wins: this directory's run.py / loo.py, not A0_state_probe/run.py

from probe import _extract_wkv_per_layer, load_model  # noqa: E402
from run import DEFAULT_MODEL, _pool_state  # noqa: E402
from loo import _train_one  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True, help="path stem: writes <out>.features.npz and <out>.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(args.threads)

    rows = [json.loads(l) for l in open(args.items) if l.strip()]
    feat_path = args.out + ".features.npz"
    if os.path.exists(feat_path):
        X = np.load(feat_path)["X"]
    else:
        model, tok = load_model(args.model, device="cpu")
        feats = []
        for i, r in enumerate(rows):
            _, state = model.forward(tok(r["prompt"])["input_ids"], None)
            feats.append(_pool_state(_extract_wkv_per_layer(state)))
            if (i + 1) % 40 == 0:
                print(f"[pair_head] features {i + 1}/{len(rows)}", flush=True)
        X = np.stack(feats)
        np.savez(feat_path, X=X)
    y = np.array([int(r["y"]) for r in rows], dtype=np.float32)
    groups = np.array([r["id"][:-1] for r in rows])

    t0 = time.time()
    probs = np.zeros(len(rows), dtype=np.float32)
    for g in sorted(set(groups)):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        probs[te] = _train_one(X[tr], y[tr], X[te], epochs=args.epochs, lr=args.lr, wd=args.wd,
                               seed=2000 + int(te[0]))
    pred = (probs >= 0.5).astype(int)
    yi = y.astype(int)
    acc = float((pred == yi).mean())
    # per pair: does the head rank the positive member above the negative one?
    pair_rank = []
    for g in sorted(set(groups)):
        idx = np.where(groups == g)[0]
        pos = [i for i in idx if yi[i] == 1]
        neg = [i for i in idx if yi[i] == 0]
        if pos and neg:
            pair_rank.append(float(probs[pos[0]] > probs[neg[0]]))
    out = {"items": os.path.basename(args.items), "model": os.path.basename(args.model), "n": len(rows),
           "split": "leave-pair-out", "accuracy": acc, "pair_rank_accuracy": float(np.mean(pair_rank)),
           "n_pairs": len(pair_rank), "wall_s_head": round(time.time() - t0, 1),
           "rows": [{"id": r["id"], "y": int(r["y"]), "p": float(p)} for r, p in zip(rows, probs)]}
    with open(args.out + ".json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
