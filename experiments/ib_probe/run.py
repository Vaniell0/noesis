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
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn as nn

from experiments._common import registry
from experiments._common.model import load_model
from experiments._common.results import save_result
from experiments.A0_state_probe.probe import _extract_wkv_per_layer


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

def _unigram_ce_heldout(train_token_ids: List[int], test_token_ids: List[int]) -> float:
    """CE of predicting held-out tokens using unigram frequencies fit on train only.

    Mirrors the probe's own train/test discipline (see `probe_lag`) — the
    baseline must be fit-then-scored the same way the probe is, otherwise
    "fraction explained" compares a held-out number against an in-sample
    one and the ratio is not meaningful.
    """
    from collections import Counter
    counts = Counter(train_token_ids)
    vocab = set(train_token_ids) | set(test_token_ids)
    smoothed_total = len(train_token_ids) + len(vocab)  # +1 Laplace smoothing
    log_probs = [np.log((counts.get(t, 0) + 1) / smoothed_total) for t in test_token_ids]
    return float(-np.mean(log_probs)) if log_probs else float("nan")


class LinearDecoder(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class MLPDecoder(nn.Module):
    """2-layer classifier — the nonlinear counterpart to LinearDecoder.

    Added 2026-08-18, same motivation as mlp_probe.py's nonlinear IPC:
    on G1d, linear IPC found ~0 held-out signal at short trajectories
    while an MLP probe (same held-out discipline) found real, consistent
    structure (up to R^2~=0.86) — the state carries information nonlinearly,
    invisible to a linear decoder. `probe_lag` below tests whether the
    same gap exists for IB's reconstruction task (predict token[t-lag]
    from state[t]) — linear IB came back ~=0 beyond lag=2 (see FAILED.md/
    HYPOTHESES.md note 2026-08-18); this is the direct check for whether
    that's a real absence of structure or the same linear-decoder blindness.
    """
    def __init__(self, in_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def probe_lag(
    states: np.ndarray,
    token_ids: List[int],
    lag: int,
    n_classes: int,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
    train_frac: float = 0.8,
    nonlinear: bool = False,
    hidden: int = 256,
) -> Dict[str, float]:
    """Train a decoder from state[t] -> token[t-lag]. Held-out CE (probe and
    matching unigram baseline, both scored on the same test slice — see
    `_unigram_ce_heldout`). Contiguous split (first train_frac = train, tail =
    test), same convention as `ridge_r2` in ipc_analysis.py.

    `nonlinear=True` swaps LinearDecoder for a 2-layer MLPDecoder — see its
    docstring for why.

    Previously trained and evaluated on the same data (no split) — that made
    the "78-96% explained variance" numbers in-sample overfitting artifacts,
    the same class of bug fixed for linear IPC in 9b28b7f. Fixed 2026-08-17.
    """
    if len(states) <= lag:
        return {"probe_ce": float("nan"), "baseline_ce": float("nan"), "n_train": 0, "n_test": 0}

    torch.manual_seed(seed)
    X = torch.from_numpy(states[lag:]).float()
    tok_targets = token_ids[:len(states) - lag]
    y = torch.tensor(tok_targets, dtype=torch.long)

    # Restrict to known classes (tokens that actually appear)
    unique = sorted(set(token_ids))
    tok2cls = {t: i for i, t in enumerate(unique)}
    y_cls = torch.tensor([tok2cls[t.item()] for t in y], dtype=torch.long)
    n_cls = len(unique)

    T = X.shape[0]
    n_train = max(1, min(T - 1, int(T * train_frac)))
    if n_train < 1 or n_train >= T:
        return {"probe_ce": float("nan"), "baseline_ce": float("nan"), "n_train": n_train, "n_test": T - n_train}

    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y_cls[:n_train], y_cls[n_train:]

    # Normalise using train-split stats only — fitting mu/sd on data that
    # includes the test slice would leak test information into the features.
    mu = X_tr.mean(0, keepdim=True)
    sd = X_tr.std(0, keepdim=True).clamp_min(1e-6)
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd

    m = MLPDecoder(X_tr.shape[1], hidden, n_cls) if nonlinear else LinearDecoder(X_tr.shape[1], n_cls)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    torch.set_grad_enabled(True)
    m.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = m(X_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        opt.step()

    m.eval()
    with torch.no_grad():
        probe_ce = loss_fn(m(X_te), y_te).item()
    torch.set_grad_enabled(False)

    baseline_ce = _unigram_ce_heldout(tok_targets[:n_train], tok_targets[n_train:])
    return {"probe_ce": probe_ce, "baseline_ce": baseline_ce, "n_train": n_train, "n_test": T - n_train}


def reconstruction_curve(
    all_states: List[np.ndarray],
    all_token_ids: List[int],
    lags: List[int],
    epochs: int = 200,
    nonlinear: bool = False,
    hidden: int = 256,
) -> Dict[int, Dict[str, float]]:
    """Compute held-out reconstruction CE at each lag across all collected sequences."""
    states = np.stack(all_states)
    n_classes = len(set(all_token_ids))

    results: Dict[int, Dict[str, float]] = {}
    for lag in lags:
        t0 = time.time()
        r = probe_lag(states, all_token_ids, lag=lag, n_classes=n_classes, epochs=epochs,
                       nonlinear=nonlinear, hidden=hidden)
        probe_ce, baseline_ce = r["probe_ce"], r["baseline_ce"]
        frac_explained = (max(0.0, 1.0 - probe_ce / baseline_ce)
                           if not (np.isnan(probe_ce) or np.isnan(baseline_ce) or baseline_ce == 0)
                           else float("nan"))
        results[lag] = {
            "lag": lag,
            "probe_ce": probe_ce,
            "baseline_ce": baseline_ce,
            "frac_explained": frac_explained,
            "n_train": r["n_train"],
            "n_test": r["n_test"],
            "wall_s": time.time() - t0,
        }
        print(f"  lag={lag:3d}: CE={probe_ce:.3f} baseline={baseline_ce:.3f} "
              f"explained={frac_explained:.3f} (held-out n={r['n_test']}, {time.time()-t0:.1f}s)",
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
# Core probe (loading-free — usable both standalone and via the registry)
# ---------------------------------------------------------------------------

def _run_ib_probe(model, tokenizer, corpus: str, lags: List[int],
                   max_tokens_per_chunk: int, epochs: int,
                   labels: Optional[str] = None,
                   nonlinear: bool = False, hidden: int = 256) -> Dict:
    print(f"[ib] loading corpus {corpus}", file=sys.stderr)
    texts = load_corpus_texts(corpus)
    print(f"[ib] {len(texts)} corpus chunks", file=sys.stderr)

    all_states: List[np.ndarray] = []
    all_token_ids: List[int] = []
    t0 = time.time()
    for i, text in enumerate(texts):
        tok_ids, states = collect_token_states(model, tokenizer, text,
                                               max_tokens=max_tokens_per_chunk)
        all_states.extend(states)
        all_token_ids.extend(tok_ids)
        print(f"[ib] chunk {i+1}/{len(texts)} tokens={len(tok_ids)} "
              f"total={len(all_token_ids)} wall={time.time()-t0:.1f}s",
              file=sys.stderr, flush=True)

    print(f"\n[ib] reconstruction probe: {len(all_token_ids)} tokens, "
          f"feature_dim={len(all_states[0])}", file=sys.stderr)
    rec_results = reconstruction_curve(all_states, all_token_ids, lags=lags, epochs=epochs,
                                        nonlinear=nonlinear, hidden=hidden)

    print(f"\n=== IB reconstruction curve (held-out, {'nonlinear MLP' if nonlinear else 'linear'} decoder) ===")
    print(f"corpus: {os.path.basename(corpus)}  ({len(all_token_ids)} tokens)")
    print(f"feature_dim: {len(all_states[0])}")
    print(f"\n{'lag':>6}  {'probe_CE':>9}  {'baseline_CE':>11}  {'explained':>10}  {'n_test':>7}")
    print("-" * 55)
    for lag in lags:
        r = rec_results[lag]
        print(f"{lag:>6}  {r['probe_ce']:>9.4f}  {r['baseline_ce']:>11.4f}  "
              f"{r['frac_explained']:>10.4f}  {r['n_test']:>7d}")

    decoder_tag = "MLP" if nonlinear else "linear"
    result: Dict = {
        "corpus": corpus,
        "n_tokens": len(all_token_ids),
        "feature_dim": len(all_states[0]),
        "decoder": decoder_tag,
        "lags": rec_results,
        "_summary": {f"lag={lag} frac_explained (held-out, {decoder_tag})": f"{rec_results[lag]['frac_explained']:.3f}"
                     for lag in lags},
    }

    if labels:
        print(f"\n[ib] downstream probe from {labels}", file=sys.stderr)
        label_items = [json.loads(l) for l in open(labels, encoding="utf-8") if l.strip()]
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
            result["downstream"] = {"f1_mean": float(cv_scores.mean()), "f1_std": float(cv_scores.std()),
                                     "n": len(y_ds)}
            result["_summary"]["downstream F1 (5-fold)"] = f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}"

    return result


def _add_ib_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                    help="text file or .jsonl with prompt/text/content fields")
    ap.add_argument("--lags", default="1,2,4,8,16,32,64",
                    help="comma-separated reconstruction lags")
    ap.add_argument("--max-tokens-per-chunk", type=int, default=400,
                    help="max tokens to process per corpus chunk")
    ap.add_argument("--epochs", type=int, default=200,
                    help="linear probe training epochs per lag")
    ap.add_argument("--labels", default=None,
                    help="jsonl with {prompt, label} for downstream probe")
    ap.add_argument("--nonlinear", action="store_true",
                    help="Use a 2-layer MLP decoder instead of linear — the "
                         "nonlinear-IB counterpart to mlp_ipc's nonlinear IPC. "
                         "Linear IB came back ~=0 held-out beyond lag=2 "
                         "(2026-08-18); this checks whether that's a real "
                         "absence of structure or linear-decoder blindness.")
    ap.add_argument("--hidden", type=int, default=256,
                    help="MLP hidden width, only used with --nonlinear")


@registry.probe(
    "ib_probe", hypothesis=["H8"],
    description="Held-out WKV state reconstruction (predict token[t-lag] from state[t]) + optional downstream label probe. "
                "--nonlinear swaps in a 2-layer MLP decoder (linear came back ~=0 beyond lag=2). "
                "SLOW (~10h/model on CPU, full corpus) and does NOT respect --n-tokens (scale is --corpus + "
                "--max-tokens-per-chunk instead) — the only probe in the registry where 'small --n-tokens for a "
                "quick check' silently doesn't apply. Pass --corpus pointing at a small file to actually go fast.",
    add_args=_add_ib_args,
)
def run(model, tokenizer, args) -> Dict:
    lags = [int(x) for x in args.lags.split(",")]
    result = _run_ib_probe(model, tokenizer, args.corpus, lags,
                            args.max_tokens_per_chunk, args.epochs, args.labels,
                            nonlinear=args.nonlinear, hidden=args.hidden)
    return {"model": args.model, **result}


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
    ap.add_argument("--nonlinear", action="store_true",
                    help="Use a 2-layer MLP decoder instead of linear")
    ap.add_argument("--hidden", type=int, default=256,
                    help="MLP hidden width, only used with --nonlinear")
    args = ap.parse_args()

    lags = [int(x) for x in args.lags.split(",")]
    os.makedirs(args.out, exist_ok=True)

    device = os.environ.get("NOESIS_EVAL_DEVICE", "cpu")
    print(f"[ib] loading model {args.model} on {device}", file=sys.stderr)
    model, tokenizer = load_model(args.model, device=device)

    result = _run_ib_probe(model, tokenizer, args.corpus, lags,
                            args.max_tokens_per_chunk, args.epochs, args.labels,
                            nonlinear=args.nonlinear, hidden=args.hidden)
    result["model"] = args.model

    out_path = save_result(
        os.path.join(args.out, "reconstruction.json"), result,
        experiment="ib_probe", hypothesis=["H8"], model=args.model, script=__file__,
    )
    print(f"\n[ib] saved {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
