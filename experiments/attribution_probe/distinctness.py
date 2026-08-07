#!/usr/bin/env python3
"""H22 distinctness measurement.

Extracts pooled WKV features on `items_overlap.jsonl` (32 items with
both `premise_valid` and `attributable` labels), then trains two
LOO heads on the same features — one predicts p_valid, the other
predicts p_attributable — and reports the Pearson correlation
between the two prediction vectors.

Design target: rho < 0.4 (H21 and H22 heads must fire independently
on overlap items — otherwise one signal subsumes the other).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "A0_state_probe"))

import numpy as np
import torch
import torch.nn as nn
from probe import _extract_wkv_per_layer, load_model


DEFAULT_MODEL = "/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth"


def _pool_state(wkv_per_layer):
    feats = []
    for w in wkv_per_layer:
        mean_per_head = w.mean(dim=(-1, -2))
        std_per_head = w.std(dim=(-1, -2))
        feats.extend(mean_per_head.tolist())
        feats.extend(std_per_head.tolist())
    return np.asarray(feats, dtype=np.float32)


def _extract(model, tokenizer, items):
    X, y_valid, y_attr = [], [], []
    for i, it in enumerate(items):
        enc = tokenizer(it["prompt"], return_tensors="pt")
        prompt_ids = enc["input_ids"][0].tolist()
        t0 = time.time()
        _logits, state = model.forward(prompt_ids, None)
        wkv = _extract_wkv_per_layer(state)
        X.append(_pool_state(wkv))
        y_valid.append(int(it["premise_valid"]))
        y_attr.append(int(it["attributable"]))
        print(f"[dist] feat {i+1}/{len(items)} {it['id']} v={y_valid[-1]} a={y_attr[-1]} wall={time.time()-t0:.2f}s",
              file=sys.stderr, flush=True)
    return np.stack(X), np.asarray(y_valid), np.asarray(y_attr)


class Head(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x)


def _train_and_predict(X_tr, y_tr, X_ho, epochs=500, lr=1e-3, wd=1e-3, seed=13):
    torch.manual_seed(seed)
    d = X_tr.shape[1]
    m = Head(d)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(X_tr); yt = torch.from_numpy(y_tr.astype(np.float32)).unsqueeze(1)
    m.train()
    with torch.enable_grad():
        for _ in range(epochs):
            opt.zero_grad()
            logit = m(Xt)
            loss = loss_fn(logit, yt)
            loss.backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(torch.from_numpy(X_ho))).squeeze(-1).numpy()
    return p


def _loo_predictions(X, y, epochs=500, lr=1e-3, wd=1e-3, seed=13):
    n = len(y)
    preds = np.zeros(n, dtype=np.float32)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        X_tr = X[mask]; y_tr = y[mask]
        X_ho = X[i:i+1]
        p = _train_and_predict(X_tr, y_tr, X_ho, epochs=epochs, lr=lr, wd=wd, seed=seed)
        preds[i] = p[0]
        print(f"[loo] {i+1}/{n} p={preds[i]:.3f} (y={y[i]})", file=sys.stderr, flush=True)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--items", default=os.path.join(_HERE, "items_overlap.jsonl"))
    ap.add_argument("--out", default=os.path.join(_HERE, "distinctness"))
    ap.add_argument("--from-features", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    features_path = os.path.join(args.out, "features.npz")

    items = [json.loads(l) for l in open(args.items) if l.strip()]

    if args.from_features and os.path.exists(features_path):
        print(f"[dist] loading features {features_path}", file=sys.stderr)
        z = np.load(features_path)
        X, y_v, y_a = z["X"], z["y_valid"], z["y_attr"]
    else:
        device = os.environ.get("NOESIS_EVAL_DEVICE", "cpu")
        print(f"[dist] loading model {args.model} on {device}", file=sys.stderr)
        model, tokenizer = load_model(args.model, device=device)
        X, y_v, y_a = _extract(model, tokenizer, items)
        np.savez(features_path, X=X, y_valid=y_v, y_attr=y_a)
        print(f"[dist] features saved {features_path}", file=sys.stderr)

    print(f"[dist] X.shape={X.shape}  y_valid.mean={y_v.mean():.2f}  y_attr.mean={y_a.mean():.2f}",
          file=sys.stderr)

    print("[dist] === LOO p_valid ===", file=sys.stderr)
    p_valid = _loo_predictions(X, y_v)
    print("[dist] === LOO p_attributable ===", file=sys.stderr)
    p_attr = _loo_predictions(X, y_a)

    # Metrics
    def f1(y_true, p):
        pred = (p >= 0.5).astype(int)
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        tn = int(((y_true == 0) & (pred == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (tp + tn) / len(y_true)
        return f, acc, (tp, fp, fn, tn)

    f_v, acc_v, cm_v = f1(y_v, p_valid)
    f_a, acc_a, cm_a = f1(y_a, p_attr)

    rho = float(np.corrcoef(p_valid, p_attr)[0, 1])

    print(f"\n=== distinctness summary ===")
    print(f"H21 p_valid:   F1={f_v:.3f} acc={acc_v:.3f}  cm(TP,FP,FN,TN)={cm_v}")
    print(f"H22 p_attr:    F1={f_a:.3f} acc={acc_a:.3f}  cm(TP,FP,FN,TN)={cm_a}")
    print(f"rho(p_valid, p_attr) = {rho:.3f}   (target < 0.4)")

    # Per-cell breakdown
    print("\nPer-cell mean predictions:")
    for pv in (0, 1):
        for at in (0, 1):
            mask = (y_v == pv) & (y_a == at)
            if mask.any():
                print(f"  premise_valid={pv} attributable={at}: "
                      f"p_valid={p_valid[mask].mean():.3f}  p_attr={p_attr[mask].mean():.3f}  (n={int(mask.sum())})")

    rows = []
    for i, it in enumerate(items):
        rows.append({
            "id": it["id"], "y_valid": int(y_v[i]), "y_attr": int(y_a[i]),
            "p_valid": float(p_valid[i]), "p_attr": float(p_attr[i]),
        })
    with open(os.path.join(args.out, "loo_results.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({
            "n": len(y_v),
            "H21_F1": f_v, "H21_acc": acc_v, "H21_cm": cm_v,
            "H22_F1": f_a, "H22_acc": acc_a, "H22_cm": cm_a,
            "rho": rho, "target_rho_max": 0.4,
        }, f, indent=2)


if __name__ == "__main__":
    main()
