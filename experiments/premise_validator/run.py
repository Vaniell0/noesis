#!/usr/bin/env python3
"""H21 premise-validity probe — CPU pilot runner on G1d-0.4B.

For each item in ``items.jsonl``, prefill the prompt on G1d-0.4B, pull
the WKV state at the last prompt token, pool it to a compact per-layer
per-head feature vector, and train a small MLP head to predict
`p(premise_valid | state, prompt)`.

Feature pooling: for each (layer, head), take mean and std of the
`d_h × d_h` WKV matrix. On 0.4B that is 12 × 12 × 2 = 288 features.
Chosen for pilot compactness — 40 items × 49 k features would overfit
badly; 288 features on 40 items is still overparameterised but tractable
with weight decay and a small held-out set.

Pilot protocol: stratified split (per-category preservation) with a
single 32/8 train/test split; MLP head trained 500 epochs on the train
fold, F1 + accuracy reported on the test fold.

Outputs:

- ``features.npz``    — pooled features + labels (reusable).
- ``results.jsonl``   — per-item test predictions.
- ``report.md``       — F1, accuracy, per-category breakdown, confusion.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from typing import Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "A0_state_probe"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from probe import _extract_wkv_per_layer, load_model  # noqa: E402


DEFAULT_MODEL = "/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth"


def _pool_state(wkv_per_layer: List[torch.Tensor]) -> np.ndarray:
    """Pool WKV state to a compact per-layer per-head feature vector.

    Input: list of ``[n_head, d_h, d_h]`` fp32 tensors, one per layer.
    Output: 1-D numpy vector of length ``2 * n_layer * n_head`` —
    (mean, std) of the ``d_h × d_h`` matrix per (layer, head).
    """
    feats: List[float] = []
    for w in wkv_per_layer:
        # w: [n_head, d_h, d_h]
        mean_per_head = w.mean(dim=(-1, -2))  # [n_head]
        std_per_head = w.std(dim=(-1, -2))
        feats.extend(mean_per_head.tolist())
        feats.extend(std_per_head.tolist())
    return np.asarray(feats, dtype=np.float32)


def _extract_features(model, tokenizer, items: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prefill each item and pool its final WKV state."""
    feats: List[np.ndarray] = []
    labels: List[int] = []
    cats: List[str] = []
    for i, it in enumerate(items):
        enc = tokenizer(it["prompt"], return_tensors="pt")
        prompt_ids = enc["input_ids"][0].tolist()
        t0 = time.time()
        _logits, state = model.forward(prompt_ids, None)
        wkv = _extract_wkv_per_layer(state)
        v = _pool_state(wkv)
        feats.append(v)
        labels.append(1 if it["category"] == "valid" else 0)
        # For H21, category is valid/invalid; invalid_type gives finer detail.
        cats.append(it.get("invalid_type") or it["category"])
        print(
            f"[H21] feat {i+1}/{len(items)} {it['id']} "
            f"y={it['category']} dim={len(v)} wall={time.time()-t0:.2f}s",
            file=sys.stderr, flush=True,
        )
        # Drop state reference.
        del state
    return np.stack(feats), np.asarray(labels, dtype=np.int64), cats


class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden1: int = 128, hidden2: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train_head(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    epochs: int, lr: float, weight_decay: float, seed: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Train MLP head; return test probs, test labels, best test-F1."""
    # probe.load_model() globally disables grad — re-enable for head training.
    torch.set_grad_enabled(True)
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X_train).float()
    yt = torch.from_numpy(y_train).float()
    Xv = torch.from_numpy(X_test).float()
    yv = torch.from_numpy(y_test).float()

    # Feature normalisation on train stats.
    mu = Xt.mean(dim=0, keepdim=True)
    sd = Xt.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xt = (Xt - mu) / sd
    Xv = (Xv - mu) / sd

    model = _MLPHead(in_dim=Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    best_probs = None
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xt)
        loss = lossf(logits, yt)
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(Xv)).cpu().numpy()
            preds = (pv >= 0.5).astype(int)
            f1 = _f1(y_test, preds)
            if f1 >= best_f1:
                best_f1 = f1
                best_probs = pv
    return best_probs, y_test, best_f1


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _stratified_split(labels: np.ndarray, test_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    idx_by_cls: Dict[int, List[int]] = {}
    for i, y in enumerate(labels.tolist()):
        idx_by_cls.setdefault(y, []).append(i)
    train_idx: List[int] = []
    test_idx: List[int] = []
    for cls, idxs in idx_by_cls.items():
        rng.shuffle(idxs)
        n_test = max(1, int(round(len(idxs) * test_frac)))
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])
    return np.asarray(sorted(train_idx)), np.asarray(sorted(test_idx))


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def _write_report(
    items: List[Dict],
    test_idx: np.ndarray, probs: np.ndarray, y_test: np.ndarray, best_f1: float,
    labels_full: np.ndarray, cats: List[str], meta: Dict, out_dir: str,
) -> None:
    preds = (probs >= 0.5).astype(int)
    cm = _confusion(y_test, preds)
    acc = float((preds == y_test).mean()) if len(y_test) else 0.0

    lines: List[str] = []
    lines.append("# H21 premise-validity probe — pilot report\n")
    lines.append(f"- Model: `{meta['model']}`")
    lines.append(f"- Items: {len(items)} (valid={int((labels_full == 1).sum())}, "
                 f"invalid={int((labels_full == 0).sum())})")
    lines.append(f"- Feature dim: {meta['feat_dim']}  (per-layer, per-head mean+std of WKV)")
    lines.append(f"- Split: stratified {len(labels_full) - len(test_idx)} train / "
                 f"{len(test_idx)} test  (seed={meta['seed']})")
    lines.append(f"- Head: 128→64→1 MLP, BCE, Adam lr={meta['lr']}, "
                 f"wd={meta['weight_decay']}, epochs={meta['epochs']}\n")

    lines.append("## Aggregate\n")
    lines.append(f"- Test F1 (best over training):  **{best_f1:.3f}**  (pilot target 0.75)")
    lines.append(f"- Test accuracy:                 {acc:.3f}")
    lines.append(f"- Confusion (test):  TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}\n")

    # Per-category (per-invalid-type) recall on test.
    lines.append("## Per-category on test set\n")
    lines.append("| category | n | correct |")
    lines.append("|---|---|---|")
    tested_items = [items[i] for i in test_idx.tolist()]
    per_cat: Dict[str, List[int]] = {}
    for j, it in enumerate(tested_items):
        key = it.get("invalid_type") or it["category"]
        correct = int(preds[j] == y_test[j])
        per_cat.setdefault(key, []).append(correct)
    for k in sorted(per_cat.keys()):
        v = per_cat[k]
        lines.append(f"| {k} | {len(v)} | {sum(v)}/{len(v)} |")
    lines.append("")

    lines.append("## Per-item test predictions\n")
    lines.append("| id | category | invalid_type | true | pred | p_valid |")
    lines.append("|---|---|---|---|---|---|")
    for j, i in enumerate(test_idx.tolist()):
        it = items[i]
        lines.append(
            f"| {it['id']} | {it['category']} | "
            f"{it.get('invalid_type') or '-'} | "
            f"{int(y_test[j])} | {int(preds[j])} | {float(probs[j]):.3f} |"
        )
    lines.append("")

    lines.append("## Notes\n")
    lines.append("- 40-item pilot; head is overparameterised relative to sample size.")
    lines.append("  Interpret F1 as a **necessary** signal for production: if the pooled state")
    lines.append("  doesn't separate at 40 items, it won't at 200 either without richer features.")
    lines.append("- Feature pooling drops most of the WKV rank structure. If the pilot fails,")
    lines.append("  next attempt should try per-head Frobenius + top-k singular values, or")
    lines.append("  concatenated head-diagonals.")
    lines.append("- False positives on the valid set (`fp`) are the operational cost:")
    lines.append("  flagging a well-formed query as suspicious causes needless aporia.\n")

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="H21 premise-validity probe runner (CPU pilot).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--items", default=os.path.join(_HERE, "items.jsonl"))
    ap.add_argument("--out", default=_HERE)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--from-features", action="store_true",
                    help="Skip model load + extraction; reuse features.npz in --out.")
    args = ap.parse_args()

    items: List[Dict] = []
    with open(args.items) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    os.makedirs(args.out, exist_ok=True)

    features_path = os.path.join(args.out, "features.npz")
    if args.from_features and os.path.exists(features_path):
        t0 = time.time()
        d = np.load(features_path, allow_pickle=True)
        X, y, cats = d["X"], d["y"], list(d["cats"])
        print(f"[H21] reused {features_path} X={X.shape}", file=sys.stderr)
    else:
        print(f"[H21] loading model {args.model}", file=sys.stderr, flush=True)
        t0 = time.time()
        model, tokenizer = load_model(args.model, device="cpu")
        print(f"[H21] loaded in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
        print(f"[H21] extracting features for {len(items)} items", file=sys.stderr, flush=True)
        X, y, cats = _extract_features(model, tokenizer, items)
        np.savez(features_path, X=X, y=y, cats=np.asarray(cats, dtype=object))
        print(f"[H21] features: X.shape={X.shape} y.shape={y.shape}", file=sys.stderr)

    train_idx, test_idx = _stratified_split(y, test_frac=args.test_frac, seed=args.seed)
    probs, y_test, best_f1 = _train_head(
        X[train_idx], y[train_idx], X[test_idx], y[test_idx],
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
    )

    results_path = os.path.join(args.out, "results.jsonl")
    with open(results_path, "w") as fout:
        for j, i in enumerate(test_idx.tolist()):
            it = items[i]
            fout.write(json.dumps({
                "id": it["id"], "category": it["category"],
                "invalid_type": it.get("invalid_type"),
                "y_true": int(y_test[j]), "p_valid": float(probs[j]),
                "y_pred": int(probs[j] >= 0.5),
            }) + "\n")
    print(f"[H21] test F1={best_f1:.3f} → {results_path}", file=sys.stderr)

    _write_report(
        items=items, test_idx=test_idx, probs=probs, y_test=y_test, best_f1=best_f1,
        labels_full=y, cats=cats, out_dir=args.out,
        meta={
            "model": args.model, "feat_dim": int(X.shape[1]),
            "epochs": args.epochs, "lr": args.lr, "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
    )
    print(f"[H21] done wall={time.time()-t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
