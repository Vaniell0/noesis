"""Lexical control for H21/H22 (H22 contradicts_if #2): leave-one-out bag-of-words logistic regression
on the same items, compared with the recorded WKV-state MLP head predictions.

Usage: python bow_control.py <items.jsonl> <results.jsonl> [<items.jsonl> <results.jsonl> ...]
       python bow_control.py --gate <items.jsonl>   # labels from each row's "y"; a new item set passes
                                                      # only if BoW accuracy <= 0.55 (no lexical shortcut)
"""
import json, re, sys
import numpy as np

def load(p): return [json.loads(l) for l in open(p) if l.strip()]

def f1(y, p):
    y, p = np.array(y), np.array(p)
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)

def train_lr(X, y, l2=1.0, it=300, lr=0.5):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(it):
        z = X @ w + b; p = 1 / (1 + np.exp(-z)); g = p - y
        w -= lr * (X.T @ g / len(y) + l2 * w / len(y)); b -= lr * g.mean()
    return w, b

def loo(texts, y, groups=None):
    """Leave-one-out; with `groups`, leave the whole group out (paired items share words, and
    leaving only one member out lets the model copy its partner's opposite label)."""
    preds = []
    for i in range(len(texts)):
        tr = [j for j in range(len(texts)) if j != i and (groups is None or groups[j] != groups[i])]
        vocab = sorted(set().union(*[set(re.findall(r"[a-z0-9]+", texts[j].lower())) for j in tr]))
        idx = {v: k for k, v in enumerate(vocab)}
        def vec(t):
            x = np.zeros(len(vocab))
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                if tok in idx: x[idx[tok]] = 1
            return x
        X = np.stack([vec(texts[j]) for j in tr]); yy = np.array([y[j] for j in tr], float)
        w, b = train_lr(X, yy)
        preds.append(int(vec(texts[i]) @ w + b > 0))
    return preds

def run(items_path, results_path):
    items = {r["id"]: r for r in load(items_path)}
    res = [r for r in load(results_path) if r["id"] in items and r.get("y_true") in (0, 1)]
    texts = [items[r["id"]]["prompt"] for r in res]; y = [r["y_true"] for r in res]
    bow = loo(texts, y)
    head = [r["y_pred"] for r in res]
    acc = lambda p: np.mean(np.array(p) == np.array(y))
    print(f"{items_path}: n={len(y)} positives(y_true=1)={sum(y)}")
    inv=lambda p:[1-v for v in p]
    print(f"  WKV-state MLP head (recorded): F1(pos)={f1(y, head):.3f} F1(neg)={f1(inv(y), inv(head)):.3f} acc={acc(head):.3f}")
    print(f"  BoW logistic, LOO           : F1(pos)={f1(y, bow):.3f} F1(neg)={f1(inv(y), inv(bow)):.3f} acc={acc(bow):.3f}")
    return res, bow, y

def gate(items_path, limit=0.55):
    rows = load(items_path)
    texts, y = [r["prompt"] for r in rows], [int(r["y"]) for r in rows]
    groups = [r["id"][:-1] for r in rows]  # pair key: id minus the class suffix letter
    bow = loo(texts, y, groups)
    acc = float(np.mean(np.array(bow) == np.array(y)))
    print(f"{items_path}: n={len(y)} BoW leave-pair-out acc={acc:.3f} F1={f1(y, bow):.3f} -> "
          f"{'PASS (no lexical shortcut)' if acc <= limit else 'FAIL (words alone separate the classes)'}")
    return acc <= limit


if __name__ == "__main__":
    if sys.argv[1] == "--gate":
        sys.exit(0 if all([gate(p) for p in sys.argv[2:]]) else 1)
    for it, rp in zip(sys.argv[1::2], sys.argv[2::2]):
        run(it, rp)
