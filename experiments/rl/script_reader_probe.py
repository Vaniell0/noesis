#!/usr/bin/env python3
"""script_reader_probe.py - is the state script-invariant, so that only the output interface differs?

    python experiments/rl/script_reader_probe.py --model MODEL [--weights W.pth] --out OUT.json

A linear (kernel ridge) reader maps the deep-layer WKV state after a text prefix to the next English word (one of
the 256 most frequent words of the c1d train texts). It is trained on prefixes read in plain English and tested on
held-out prefixes read in plain, in the fixed Cyrillic table and in the logographic script of
script_alignment_probe.py; a reader trained and tested within each script is the ceiling. Model frozen, inference
only (plan noesis-closed-loop-toy, section 1q).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiments.rl.loader import load_rwkv7, load_weights_into
from experiments.rl.script_alignment_probe import WORD, logo_table, states, texts, to_cyr, to_logo


def items(T: list[str], vocab: dict[str, int], rng: random.Random) -> list[tuple[str, int]]:
    """One prefix per text, cut before a word in the vocabulary, with at least three words before it."""
    out = []
    for t in T:
        ms = [m for m in WORD.finditer(t)][3:]
        ms = [m for m in ms if m.group(0).lower() in vocab]
        if ms:
            m = rng.choice(ms)
            out.append((t[:m.start()].rstrip(), vocab[m.group(0).lower()]))
    return out


def ridge_fit_predict(Xtr, ytr, Xte, n_cls: int, lam: float) -> torch.Tensor:
    mu = Xtr.mean(0)
    A, B = (Xtr - mu).double(), (Xte - mu).double()
    K = A @ A.T
    Y = torch.nn.functional.one_hot(ytr, n_cls).double()
    alpha = torch.linalg.solve(K + lam * K.diagonal().mean() * torch.eye(len(K), dtype=K.dtype), Y)
    return (B @ A.T @ alpha).argmax(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--train", default="/home/vaniello/tmp/c1d/train.jsonl")
    ap.add_argument("--eval", default="/home/vaniello/tmp/c1d/eval.jsonl")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--ticks", type=int, default=4)
    ap.add_argument("--features", type=int, default=4096, help="state coordinates sampled per layer")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    table = logo_table(texts(a.train))
    vocab = {w: i for i, w in enumerate(table)}
    rng = random.Random(0)
    ev = texts(a.eval)
    tr_items = items(ev[:a.n_train], vocab, rng)
    te_items = items(ev[a.n_train:a.n_train + a.n_test * 2], vocab, rng)[:a.n_test]
    loaded = load_rwkv7(a.model, device="cpu", dtype=torch.float32, backend="peft", ctx_len=2048)
    if a.weights:
        load_weights_into(loaded, a.weights)
    L = loaded.n_layer
    deep = list(range(L - L // 3, L))
    D = loaded.wkv_stack(loaded.new_state(batch=1)).reshape(L, -1).shape[1]
    idx = torch.randperm(D, generator=torch.Generator().manual_seed(0))[:a.features]
    render = {"plain": lambda t: t, "cyr": to_cyr, "logo": lambda t: to_logo(t, table)}
    t0 = time.time()

    def feats(prefixes, script):
        rows = [states(loaded, render[script](p), a.ticks, idx) for p in prefixes]
        return {M: torch.stack([r[k][deep].reshape(-1) for r in rows]) for k, M in ((0, 0), (1, a.ticks))}

    ytr = torch.tensor([y for _, y in tr_items]); yte = torch.tensor([y for _, y in te_items])
    F = {("train", s): feats([p for p, _ in tr_items], s) for s in ("plain", "cyr", "logo")}
    print(f"[reader] train states {time.time() - t0:.0f}s", flush=True)
    F.update({("test", s): feats([p for p, _ in te_items], s) for s in ("plain", "cyr", "logo")})
    print(f"[reader] test states {time.time() - t0:.0f}s", flush=True)
    floor = float((yte == torch.bincount(ytr).argmax()).float().mean())
    out = {"model": a.model, "weights": str(a.weights) if a.weights else None, "n_train": len(tr_items),
           "n_test": len(te_items), "deep_layers": deep, "features": f"{len(idx)} of {D} per layer",
           "majority_floor": round(floor, 4), "acc": {}}
    n_val = len(tr_items) // 5
    for M in (0, a.ticks):
        best = None
        for lam in (1e-3, 1e-2, 1e-1, 1.0):                                    # lambda by a held-out fifth of train
            Xa = F[("train", "plain")][M]
            acc = float((ridge_fit_predict(Xa[n_val:], ytr[n_val:], Xa[:n_val], len(vocab), lam) == ytr[:n_val]).float().mean())
            best = max(best or (acc, lam), (acc, lam))
        lam = best[1]
        for tr_s in ("plain", "cyr", "logo"):
            for te_s in ("plain", "cyr", "logo"):
                if tr_s != "plain" and tr_s != te_s:
                    continue
                pred = ridge_fit_predict(F[("train", tr_s)][M], ytr, F[("test", te_s)][M], len(vocab), lam)
                acc = round(float((pred == yte).float().mean()), 4)
                out["acc"][f"M{M}_train_{tr_s}_test_{te_s}"] = acc
                print(f"[reader] M{M} reader on {tr_s} -> {te_s}: {acc:.3f} (floor {floor:.3f}, lambda {lam})", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
