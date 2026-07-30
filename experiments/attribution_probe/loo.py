#!/usr/bin/env python3
"""H22 seed leave-one-out from saved features.

The pilot's single 12/4 split yielded F1=1.000 on 4 held-out items — same
overfit risk as H21. LOO over the 16 labelled items (attributable=1 /
unattributed=0) is the honest number. Ambiguous items (y=-1) are held
out from LOO but scored by the full-data model at the end.

Usage:
    python loo.py [--features features.npz] [--epochs 500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def _train_one(X_train, y_train, X_test, epochs, lr, wd, seed):
    torch.set_grad_enabled(True)
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X_train).float()
    yt = torch.from_numpy(y_train).float()
    Xv = torch.from_numpy(X_test).float()
    mu = Xt.mean(0, keepdim=True); sd = Xt.std(0, keepdim=True).clamp_min(1e-6)
    Xt = (Xt - mu) / sd; Xv = (Xv - mu) / sd
    m = _MLPHead(Xt.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        loss = lossf(m(Xt), yt)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return torch.sigmoid(m(Xv)).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=os.path.join(_HERE, "features.npz"))
    ap.add_argument("--items", default=os.path.join(_HERE, "items.jsonl"))
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X, y = d["X"], d["y"]
    items = [json.loads(l) for l in open(args.items) if l.strip()]

    labelled = np.where(y >= 0)[0]
    ambiguous = np.where(y < 0)[0]
    print(f"[LOO] labelled={len(labelled)}  ambiguous_held_out={len(ambiguous)}",
          file=sys.stderr)

    X_lab = X[labelled]; y_lab = y[labelled].astype(np.float32)
    probs = np.zeros(len(labelled), dtype=np.float32)
    for k, i in enumerate(labelled):
        mask = np.arange(len(labelled)) != k
        pv = _train_one(X_lab[mask], y_lab[mask], X_lab[k:k+1],
                        epochs=args.epochs, lr=args.lr, wd=args.wd, seed=2000 + k)
        probs[k] = float(pv[0])
        print(f"[LOO] {k+1}/{len(labelled)} {items[i]['id']} y={int(y_lab[k])} "
              f"p_attr={probs[k]:.3f}", file=sys.stderr, flush=True)

    preds = (probs >= 0.5).astype(int)
    y_int = y_lab.astype(int)
    tp = int(((y_int == 1) & (preds == 1)).sum()); fp = int(((y_int == 0) & (preds == 1)).sum())
    tn = int(((y_int == 0) & (preds == 0)).sum()); fn = int(((y_int == 1) & (preds == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (preds == y_int).mean()

    print("\n=== H22 seed LOO summary ===", file=sys.stderr)
    print(f"n_labelled={len(labelled)}  F1={f1:.3f}  acc={acc:.3f}", file=sys.stderr)
    print(f"conf: TP={tp} FP={fp} TN={tn} FN={fn}", file=sys.stderr)

    # Score ambiguous with a full-data model.
    amb_probs = np.zeros(len(ambiguous), dtype=np.float32)
    if len(ambiguous):
        pv = _train_one(X_lab, y_lab, X[ambiguous],
                        epochs=args.epochs, lr=args.lr, wd=args.wd, seed=9000)
        amb_probs[:] = pv
        print("\nambiguous scores (full-data model):", file=sys.stderr)
        for k, i in enumerate(ambiguous):
            print(f"  {items[i]['id']}  p_attr={amb_probs[k]:.3f}  "
                  f"prompt={items[i]['prompt'][:80]!r}",
                  file=sys.stderr)

    out_path = os.path.join(_HERE, "loo_seed_results.jsonl")
    with open(out_path, "w") as f:
        for k, i in enumerate(labelled):
            it = items[i]
            f.write(json.dumps({
                "id": it["id"], "category": it["category"],
                "y_true": int(y_lab[k]), "p_attr": float(probs[k]),
                "y_pred": int(preds[k]),
            }) + "\n")
        for k, i in enumerate(ambiguous):
            it = items[i]
            f.write(json.dumps({
                "id": it["id"], "category": it["category"],
                "y_true": None, "p_attr": float(amb_probs[k]),
                "y_pred": int(amb_probs[k] >= 0.5),
            }) + "\n")
    print(f"\nwrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
