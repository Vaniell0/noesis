#!/usr/bin/env python3
"""H22 attribution probe — CPU pilot runner on G1d-0.4B.

For each item in ``items.jsonl``, prefill the prompt on G1d-0.4B, pool
the WKV state at the final prompt token, and train a binary head to
distinguish `attributable` (label=1) from `unattributed` (label=0).
Ambiguous items are held out from training/testing (their scores are
reported as diagnostic only).

19 items is too few for a proper train/test split; the pilot uses
leave-one-out cross-validation over the 16 non-ambiguous items and
reports out-of-fold F1.

Distinctness check vs H21: the current 19-item seed has no direct H21
overlap. This runner therefore *also* scores the H21 pilot items with
the trained H22 head so we can report a proxy ρ(H21-features, H22-preds)
using the two heads' features side-by-side. Real ρ needs cross-labelled
items and is deferred to A1.

Outputs:

- ``features.npz``   — pooled features + labels (16 non-ambiguous + 3 amb)
- ``results.jsonl``  — per-item LOO predictions + ambiguous scores
- ``report.md``      — F1, per-item table, ambiguous-item scores, note
                       on cross-head distinctness follow-up.
"""

from __future__ import annotations

import argparse
import json
import os
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
    feats: List[float] = []
    for w in wkv_per_layer:
        mean_per_head = w.mean(dim=(-1, -2))
        std_per_head = w.std(dim=(-1, -2))
        feats.extend(mean_per_head.tolist())
        feats.extend(std_per_head.tolist())
    return np.asarray(feats, dtype=np.float32)


def _extract_features(model, tokenizer, items: List[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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
        # Label: 1 = attributable, 0 = unattributed, -1 = ambiguous (held out).
        cat = it["category"]
        if cat == "attributable":
            labels.append(1)
        elif cat == "unattributed":
            labels.append(0)
        else:
            labels.append(-1)
        cats.append(cat)
        print(
            f"[H22] feat {i+1}/{len(items)} {it['id']} y={cat} "
            f"wall={time.time()-t0:.2f}s",
            file=sys.stderr, flush=True,
        )
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


def _train_and_predict_one(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,
    epochs: int, lr: float, weight_decay: float, seed: int,
) -> np.ndarray:
    """Train MLP head on train set; return probs on test set."""
    # probe.load_model() globally disables grad — re-enable for head training.
    torch.set_grad_enabled(True)
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X_train).float()
    yt = torch.from_numpy(y_train).float()
    Xv = torch.from_numpy(X_test).float()

    mu = Xt.mean(dim=0, keepdim=True)
    sd = Xt.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xt = (Xt - mu) / sd
    Xv = (Xv - mu) / sd

    m = _MLPHead(in_dim=Xt.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.BCEWithLogitsLoss()

    for _ep in range(epochs):
        m.train()
        opt.zero_grad()
        logits = m(Xt)
        loss = lossf(logits, yt)
        loss.backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        pv = torch.sigmoid(m(Xv)).cpu().numpy()
    return pv


def _f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def _leave_one_out(
    X: np.ndarray, y: np.ndarray,
    epochs: int, lr: float, weight_decay: float, seed: int,
) -> np.ndarray:
    """LOO on the labelled subset. Returns per-example held-out probs."""
    n = len(y)
    probs = np.zeros(n, dtype=np.float32)
    for i in range(n):
        mask = np.arange(n) != i
        pv = _train_and_predict_one(
            X[mask], y[mask], X[i:i+1],
            epochs=epochs, lr=lr, weight_decay=weight_decay, seed=seed + i,
        )
        probs[i] = float(pv[0])
        print(f"[H22] LOO {i+1}/{n} held={i} p_attr={probs[i]:.3f} y={int(y[i])}",
              file=sys.stderr, flush=True)
    return probs


def _write_report(
    items: List[Dict], labels: np.ndarray, probs_labeled: np.ndarray,
    ambiguous_scores: List[Tuple[str, float]], meta: Dict, out_dir: str,
) -> None:
    labelled_idx = [i for i, y in enumerate(labels.tolist()) if y in (0, 1)]
    y_true = labels[labelled_idx]
    preds = (probs_labeled >= 0.5).astype(int)
    f1 = _f1(y_true, preds)
    acc = float((preds == y_true).mean()) if len(y_true) else 0.0
    cm = _confusion(y_true, preds)

    lines: List[str] = []
    lines.append("# H22 attribution probe — pilot report\n")
    lines.append(f"- Model: `{meta['model']}`")
    lines.append(f"- Items: {len(items)}  "
                 f"(attributable={int((labels == 1).sum())}, "
                 f"unattributed={int((labels == 0).sum())}, "
                 f"ambiguous={int((labels == -1).sum())})")
    lines.append(f"- Feature dim: {meta['feat_dim']}  (per-layer, per-head mean+std of WKV)")
    lines.append(f"- Protocol: leave-one-out over 16 labelled items; ambiguous held out\n")

    lines.append("## Aggregate (labelled subset, LOO)\n")
    lines.append(f"- LOO F1:     **{f1:.3f}**  (pilot target 0.75)")
    lines.append(f"- LOO acc:    {acc:.3f}")
    lines.append(f"- Confusion:  TP={cm['tp']}  FP={cm['fp']}  TN={cm['tn']}  FN={cm['fn']}\n")

    lines.append("## Per-item (labelled)\n")
    lines.append("| id | category | y_true | y_pred | p_attributable |")
    lines.append("|---|---|---|---|---|")
    for j, i in enumerate(labelled_idx):
        it = items[i]
        lines.append(
            f"| {it['id']} | {it['category']} | {int(y_true[j])} | "
            f"{int(preds[j])} | {float(probs_labeled[j]):.3f} |"
        )
    lines.append("")

    if ambiguous_scores:
        lines.append("## Ambiguous items (held out; diagnostic only)\n")
        lines.append("Trained on all 16 labelled items, scored on ambiguous set:\n")
        lines.append("| id | p_attributable |")
        lines.append("|---|---|")
        for name, s in ambiguous_scores:
            lines.append(f"| {name} | {s:.3f} |")
        lines.append("")
        lines.append("A well-calibrated head would place ambiguous scores near 0.5; hard")
        lines.append("commitments in either direction suggest the head is latching onto")
        lines.append("surface features (e.g. presence of `I` / `my`) rather than genuine")
        lines.append("attribution structure.\n")

    lines.append("## Distinctness vs H21 (deferred)\n")
    lines.append("H22 must remain distinct from H21 (premise-validity). The design metric")
    lines.append("is ρ < 0.4 between head decisions on **overlap items** (invalid-premise +")
    lines.append("unattributed / valid-premise + attributable). The current 19-item pilot")
    lines.append("has no cross-labelled overlap; deferring to A1 dataset expansion where")
    lines.append("cross-labels get authored explicitly. As a weaker proxy, compare feature")
    lines.append("distributions across the two probes (both share the same pooling scheme)")
    lines.append("in ``features.npz`` here and in ``../premise_validator/features.npz``.\n")

    lines.append("## Notes\n")
    lines.append("- 19-item pilot: F1 is noisy, use as *directional* signal, not verdict.")
    lines.append("- Ambiguous scoring is the interesting output — where the model 'thinks'")
    lines.append("  the borderline cases sit tells us where the classifier decision surface")
    lines.append("  actually cuts.\n")

    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="H22 attribution probe runner (CPU pilot).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--items", default=os.path.join(_HERE, "items.jsonl"))
    ap.add_argument("--out", default=_HERE)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=17)
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
        print(f"[H22] reused {features_path} X={X.shape}", file=sys.stderr)
    else:
        print(f"[H22] loading model {args.model}", file=sys.stderr, flush=True)
        t0 = time.time()
        model, tokenizer = load_model(args.model, device="cpu")
        print(f"[H22] loaded in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
        print(f"[H22] extracting features for {len(items)} items", file=sys.stderr, flush=True)
        X, y, cats = _extract_features(model, tokenizer, items)
        np.savez(features_path, X=X, y=y, cats=np.asarray(cats, dtype=object))
        print(f"[H22] features: X.shape={X.shape}", file=sys.stderr)

    labelled_idx = np.array([i for i, yy in enumerate(y.tolist()) if yy in (0, 1)])
    amb_idx = np.array([i for i, yy in enumerate(y.tolist()) if yy == -1])

    probs_labeled = _leave_one_out(
        X[labelled_idx], y[labelled_idx],
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
    )

    ambiguous_scores: List[Tuple[str, float]] = []
    if len(amb_idx) > 0:
        # Train on all 16 labelled items, score ambiguous.
        pv = _train_and_predict_one(
            X[labelled_idx], y[labelled_idx].astype(np.float32),
            X[amb_idx], epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, seed=args.seed,
        )
        for j, i in enumerate(amb_idx.tolist()):
            ambiguous_scores.append((items[i]["id"], float(pv[j])))

    results_path = os.path.join(args.out, "results.jsonl")
    with open(results_path, "w") as fout:
        for j, i in enumerate(labelled_idx.tolist()):
            it = items[i]
            fout.write(json.dumps({
                "id": it["id"], "category": it["category"],
                "y_true": int(y[i]),
                "p_attributable": float(probs_labeled[j]),
                "y_pred": int(probs_labeled[j] >= 0.5),
                "fold": "loo",
            }) + "\n")
        for name, s in ambiguous_scores:
            fout.write(json.dumps({
                "id": name, "category": "ambiguous",
                "y_true": None, "p_attributable": s,
                "y_pred": int(s >= 0.5), "fold": "amb_scoring",
            }) + "\n")
    print(f"[H22] wrote {results_path}", file=sys.stderr)

    _write_report(
        items=items, labels=y, probs_labeled=probs_labeled,
        ambiguous_scores=ambiguous_scores, out_dir=args.out,
        meta={"model": args.model, "feat_dim": int(X.shape[1])},
    )
    print(f"[H22] done wall={time.time()-t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
