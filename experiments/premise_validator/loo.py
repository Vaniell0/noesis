#!/usr/bin/env python3
"""H21 leave-one-out re-scoring from saved features.

The pilot's single 32/8 stratified split yielded F1=1.000, but that's
one draw with 8 test items. LOO over all 40 items is a stronger estimate
of what the pooled state actually separates.

Usage:
    python loo.py [--epochs 500] [--lr 1e-3] [--wd 1e-3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

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
    ap.add_argument("--out", default=os.path.join(_HERE, "loo_results.jsonl"))
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    X, y, cats = d["X"], d["y"], list(d["cats"])
    items: List[Dict] = []
    with open(args.items) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    n = len(y)
    probs = np.zeros(n, dtype=np.float32)
    for i in range(n):
        mask = np.arange(n) != i
        pv = _train_one(X[mask], y[mask].astype(np.float32), X[i:i+1],
                        epochs=args.epochs, lr=args.lr, wd=args.wd, seed=1000 + i)
        probs[i] = float(pv[0])
        print(f"[LOO] {i+1}/{n} {items[i]['id']} y={int(y[i])} p_valid={probs[i]:.3f}",
              file=sys.stderr, flush=True)

    preds = (probs >= 0.5).astype(int)
    tp = int(((y == 1) & (preds == 1)).sum()); fp = int(((y == 0) & (preds == 1)).sum())
    tn = int(((y == 0) & (preds == 0)).sum()); fn = int(((y == 1) & (preds == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (preds == y).mean()

    # Per-invalid-type recall (invalid class only).
    per_type: Dict[str, List[int]] = {}
    for i in range(n):
        it = items[i]
        if it["category"] != "invalid":
            continue
        key = it.get("invalid_type") or "?"
        per_type.setdefault(key, []).append(int(preds[i] == y[i]))

    print("\n=== H21 leave-one-out summary ===", file=sys.stderr)
    print(f"n={n}  F1={f1:.3f}  acc={acc:.3f}", file=sys.stderr)
    print(f"conf: TP={tp} FP={fp} TN={tn} FN={fn}", file=sys.stderr)
    print("per-invalid-type recall:", file=sys.stderr)
    for k in sorted(per_type):
        v = per_type[k]
        print(f"  {k}: {sum(v)}/{len(v)}", file=sys.stderr)

    with open(args.out, "w") as f:
        for i in range(n):
            it = items[i]
            f.write(json.dumps({
                "id": it["id"], "category": it["category"],
                "invalid_type": it.get("invalid_type"),
                "y_true": int(y[i]), "p_valid": float(probs[i]),
                "y_pred": int(preds[i]),
            }) + "\n")
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
