#!/usr/bin/env python3
"""IB probe — WKV state reconstruction and downstream quality curves.

Implements two complementary measurements from docs/state-and-reasoning.md §4:

1. RECONSTRUCTION PROBE (upper-bounds I(X;Z)):
   At each token position t, trains a linear decoder from the pooled WKV
   state Z_t to predict token X_{t-k} for lags k ∈ {1,2,4,8,16,32,64}.
   Reports cross-entropy loss vs. unigram baseline per lag.
   Metric: fraction_explained = 1 - CE_probe / CE_unigram
   Decay curve: how many past tokens the state materially preserves.

2. DOWNSTREAM PROBE (lower-bounds I(Z;Y)):
   Uses the WKV state at the END of a labelled prompt to predict a
   task label (same protocol as H21/H22). Accepts --labels jsonl.
   Skipped if --labels not provided.

Ratio downstream_quality / reconstruction_quality = state bit-efficiency
for the given task (how much of what's stored is task-relevant).

Both probes use the SAME pooled WKV features as H21/H22 so numbers
are directly comparable across experiments.

Usage:
    python experiments/ib_probe/run.py \\
        --model ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \\
        --corpus experiments/ib_probe/corpus.txt \\
        --out experiments/ib_probe/results_g1d_base \\
        --lags 1,2,4,8,16,32,64
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
DEFAULT_CORPUS = os.path.join(_HERE, "corpus.txt")


# ---------------------------------------------------------------------------
# Feature pooling (identical to H21/H22)
# ---------------------------------------------------------------------------

def _pool_state(wkv_per_layer) -> np.ndarray:
    feats = []
    for w in wkv_per_layer:
        feats.extend(w.mean(dim=(-1, -2)).tolist())
        feats.extend(w.std(dim=(-1, -2)).tolist())
    return np.asarray(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Per-token state collection
# ---------------------------------------------------------------------------

def collect_token_states(model, tokenizer, text: str, max_tokens: int = 512):
    """Run forward token-by-token; return (token_ids, pooled_states)."""
    enc = tokenizer(text, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()[:max_tokens]

    state = None
    token_ids_out: List[int] = []
    states_out: List[np.ndarray] = []

    for tok in ids:
        logits, state = model.forward([tok], state)
        wkv = _extract_wkv_per_layer(state)
        states_out.append(_pool_state(wkv))
        token_ids_out.append(tok)

    return token_ids_out, states_out


# ---------------------------------------------------------------------------
# Reconstruction probe
# ---------------------------------------------------------------------------

def _unigram_ce(token_ids: List[int], vocab_size: int) -> float:
    """CE of predicting by unigram frequency (baseline)."""
    from collections import Counter
    counts = Counter(token_ids)
    total = len(token_ids)
    probs = {t: c / total for t, c in counts.items()}
    ce = -sum(probs[t] * np.log(probs[t] + 1e-12) for t in probs)
    return float(ce)


class LinearDecoder(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def probe_lag(
    states: np.ndarray,
    token_ids: List[int],
    lag: int,
    n_classes: int,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
) -> float:
    """Train linear decoder from state[t] → token[t-lag]. Return probe CE."""
    if len(states) <= lag:
        return float("nan")

    torch.manual_seed(seed)
    X = torch.from_numpy(states[lag:]).float()
    y = torch.tensor(token_ids[:len(states) - lag], dtype=torch.long)

    # Restrict to known classes (tokens that actually appear)
    unique = sorted(set(token_ids))
    tok2cls = {t: i for i, t in enumerate(unique)}
    y_cls = torch.tensor([tok2cls[t.item()] for t in y], dtype=torch.long)
    n_cls = len(unique)

    # Normalise features
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True).clamp_min(1e-6)
    X = (X - mu) / sd

    m = LinearDecoder(X.shape[1], n_cls)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    torch.set_grad_enabled(True)
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = m(X)
        loss = loss_fn(logits, y_cls)
        loss.backward()
        opt.step()

    m.eval()
    with torch.no_grad():
        ce = loss_fn(m(X), y_cls).item()
    torch.set_grad_enabled(False)
    return ce


def reconstruction_curve(
    all_states: List[np.ndarray],
    all_token_ids: List[int],
    lags: List[int],
    epochs: int = 200,
) -> Dict[int, Dict[str, float]]:
    """Compute reconstruction CE at each lag across all collected sequences."""
    states = np.stack(all_states)
    n_classes = len(set(all_token_ids))

    # Unigram baseline CE (log of vocab size in the corpus)
    baseline_ce = _unigram_ce(all_token_ids, n_classes)

    results: Dict[int, Dict[str, float]] = {}
    for lag in lags:
        t0 = time.time()
        probe_ce = probe_lag(states, all_token_ids, lag=lag,
                              n_classes=n_classes, epochs=epochs)
        frac_explained = max(0.0, 1.0 - probe_ce / baseline_ce) if not np.isnan(probe_ce) else float("nan")
        results[lag] = {
            "lag": lag,
            "probe_ce": probe_ce,
            "baseline_ce": baseline_ce,
            "frac_explained": frac_explained,
            "wall_s": time.time() - t0,
        }
        print(f"  lag={lag:3d}: CE={probe_ce:.3f} baseline={baseline_ce:.3f} "
              f"explained={frac_explained:.3f} ({time.time()-t0:.1f}s)",
              file=sys.stderr, flush=True)
    return results


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus_texts(corpus_path: str, max_chars_per_chunk: int = 2000) -> List[str]:
    """Split a text file or .jsonl into chunks suitable for per-token probing."""
    texts: List[str] = []
    if corpus_path.endswith(".jsonl"):
        with open(corpus_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                # Accept "prompt", "text", or "content" fields
                text = d.get("prompt") or d.get("text") or d.get("content") or ""
                if text:
                    texts.append(str(text)[:max_chars_per_chunk])
    else:
        with open(corpus_path, encoding="utf-8") as f:
            raw = f.read()
        # Split into paragraphs of ~max_chars_per_chunk
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        chunk = []
        chunk_len = 0
        for p in paragraphs:
            if chunk_len + len(p) > max_chars_per_chunk and chunk:
                texts.append("\n\n".join(chunk))
                chunk = []
                chunk_len = 0
            chunk.append(p)
            chunk_len += len(p)
        if chunk:
            texts.append("\n\n".join(chunk))
    return texts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                    help="text file or .jsonl with prompt/text/content fields")
    ap.add_argument("--out", default=os.path.join(_HERE, "results"))
    ap.add_argument("--lags", default="1,2,4,8,16,32,64",
                    help="comma-separated reconstruction lags")
    ap.add_argument("--max-tokens-per-chunk", type=int, default=400,
                    help="max tokens to process per corpus chunk")
    ap.add_argument("--epochs", type=int, default=200,
                    help="linear probe training epochs per lag")
    ap.add_argument("--labels", default=None,
                    help="jsonl with {prompt, label} for downstream probe")
    args = ap.parse_args()

    lags = [int(x) for x in args.lags.split(",")]
    os.makedirs(args.out, exist_ok=True)

    device = os.environ.get("NOESIS_EVAL_DEVICE", "cpu")
    print(f"[ib] loading model {args.model} on {device}", file=sys.stderr)
    model, tokenizer = load_model(args.model, device=device)

    # --- Reconstruction probe ---
    print(f"[ib] loading corpus {args.corpus}", file=sys.stderr)
    texts = load_corpus_texts(args.corpus)
    print(f"[ib] {len(texts)} corpus chunks", file=sys.stderr)

    all_states: List[np.ndarray] = []
    all_token_ids: List[int] = []
    t0 = time.time()
    for i, text in enumerate(texts):
        tok_ids, states = collect_token_states(model, tokenizer, text,
                                               max_tokens=args.max_tokens_per_chunk)
        all_states.extend(states)
        all_token_ids.extend(tok_ids)
        print(f"[ib] chunk {i+1}/{len(texts)} tokens={len(tok_ids)} "
              f"total={len(all_token_ids)} wall={time.time()-t0:.1f}s",
              file=sys.stderr, flush=True)

    print(f"\n[ib] reconstruction probe: {len(all_token_ids)} tokens, "
          f"feature_dim={len(all_states[0])}", file=sys.stderr)
    rec_results = reconstruction_curve(all_states, all_token_ids,
                                        lags=lags, epochs=args.epochs)

    # --- Summary output ---
    print(f"\n=== IB reconstruction curve ===")
    print(f"model: {os.path.basename(args.model)}")
    print(f"corpus: {os.path.basename(args.corpus)}  ({len(all_token_ids)} tokens)")
    print(f"feature_dim: {len(all_states[0])}")
    print(f"\n{'lag':>6}  {'probe_CE':>9}  {'baseline_CE':>11}  {'explained':>10}")
    print("-" * 45)
    for lag in lags:
        r = rec_results[lag]
        print(f"{lag:>6}  {r['probe_ce']:>9.4f}  {r['baseline_ce']:>11.4f}  {r['frac_explained']:>10.4f}")

    # Save results
    out_rec = os.path.join(args.out, "reconstruction.json")
    with open(out_rec, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "corpus": args.corpus,
            "n_tokens": len(all_token_ids),
            "feature_dim": len(all_states[0]),
            "lags": rec_results,
        }, f, indent=2)
    print(f"\n[ib] saved {out_rec}", file=sys.stderr)

    # --- Downstream probe (optional) ---
    if args.labels:
        print(f"\n[ib] downstream probe from {args.labels}", file=sys.stderr)
        label_items = [json.loads(l) for l in open(args.labels, encoding="utf-8") if l.strip()]
        X_ds, y_ds = [], []
        for item in label_items:
            prompt = item.get("prompt", "")
            label = item.get("label", item.get("premise_valid", item.get("attributable")))
            if label is None:
                continue
            enc = tokenizer(prompt, return_tensors="pt")
            ids = enc["input_ids"][0].tolist()
            _, state = model.forward(ids, None)
            wkv = _extract_wkv_per_layer(state)
            X_ds.append(_pool_state(wkv))
            y_ds.append(int(label))

        if len(set(y_ds)) >= 2:
            from sklearn.linear_model import LogisticRegression  # type: ignore
            from sklearn.model_selection import cross_val_score  # type: ignore
            X_arr = np.stack(X_ds)
            mu = X_arr.mean(0); sd = X_arr.std(0) + 1e-6
            X_arr = (X_arr - mu) / sd
            clf = LogisticRegression(max_iter=500, C=1.0)
            cv_scores = cross_val_score(clf, X_arr, y_ds, cv=5, scoring="f1")
            print(f"\n=== IB downstream probe ===")
            print(f"n={len(y_ds)}  5-fold F1: mean={cv_scores.mean():.3f} std={cv_scores.std():.3f}")
            with open(os.path.join(args.out, "downstream.json"), "w") as f:
                json.dump({"f1_mean": float(cv_scores.mean()), "f1_std": float(cv_scores.std()),
                           "n": len(y_ds)}, f)

    return 0


if __name__ == "__main__":
    sys.exit(main())
