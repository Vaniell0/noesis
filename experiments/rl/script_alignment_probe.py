#!/usr/bin/env python3
"""script_alignment_probe.py - does a pretrained RWKV-7 hold the same text the same way in three scripts?

    python experiments/rl/script_alignment_probe.py --model models/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
        [--weights experiments/rl/results/s1_pilot_g1d04b/kl_T2_k4.pth] --out OUT.json

Unseen texts (the c1d eval plaintexts) are read in plain English, in one fixed Cyrillic letter table, and in a
logographic script (the 256 most frequent words of the c1d train texts become one CJK ideograph each, every other
word is spelled in a fixed katakana letter table). The representation is the WKV state of every layer after the text,
at M = 0 and after `--ticks` own-latent ticks. Linear CKA in Gram form between the plain reading and each other
script, per layer, against a shuffled pairing. Inference only (plan noesis-closed-loop-toy, section 1p).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiments.rl.loader import load_rwkv7, load_weights_into

CYR = dict(zip("abcdefghijklmnopqrstuvwxyz", "абцдефгхийклмнопярстувшжыз"))
CYR.update({k.upper(): v.upper() for k, v in CYR.items()})
KANA = dict(zip("abcdefghijklmnopqrstuvwxyz", "アイウエオカキクケコサシスセソタチツテトナニヌネノハ"))
WORD = re.compile(r"[A-Za-z]+")


def texts(path: str) -> list[str]:
    seen, out = set(), []
    for r in map(json.loads, open(path)):
        if r["answer"] not in seen:
            seen.add(r["answer"]); out.append(r["answer"])
    return out


def logo_table(train: list[str], n: int = 256) -> dict[str, str]:
    c = collections.Counter(w.lower() for t in train for w in WORD.findall(t))
    return {w: chr(0x4E00 + i) for i, (w, _) in enumerate(c.most_common(n))}


def to_logo(t: str, table: dict[str, str]) -> str:
    return WORD.sub(lambda m: table.get(m.group(0).lower()) or "".join(KANA[ch.lower()] for ch in m.group(0)), t)


def to_cyr(t: str) -> str:
    return "".join(CYR.get(c, c) for c in t)


@torch.no_grad()
def states(loaded, text: str, ticks: int, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-layer WKV state after the text (M = 0) and after `ticks` own-latent ticks, at the sampled coordinates
    `idx` (a fixed random subset per run: the Gram matrices over it estimate the full ones without bias)."""
    ids = loaded.tokenizer.encode(text)
    E = loaded.embedding_weight
    _, st, h = loaded.forward_stateful_embeds(E[torch.tensor([ids])], loaded.new_state(batch=1), return_hidden=True)
    s0 = loaded.wkv_stack(st).float().reshape(loaded.n_layer, -1)[:, idx].clone()
    h = h[:, -1:]
    for _ in range(ticks):
        _, st, h = loaded.forward_stateful_embeds(h, st, return_hidden=True)
    return s0, loaded.wkv_stack(st).float().reshape(loaded.n_layer, -1)[:, idx].clone()


def cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between (n, d1) and (n, d2) through centred Gram matrices."""
    def gram(Z):
        Z = Z.double() - Z.double().mean(0)
        return Z @ Z.T
    K, L = gram(X), gram(Y)
    return float((K * L).sum() / (K.norm() * L.norm() + 1e-30))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--train", default="/home/vaniello/tmp/c1d/train.jsonl")
    ap.add_argument("--eval", default="/home/vaniello/tmp/c1d/eval.jsonl")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--max-chars", type=int, default=300)
    ap.add_argument("--ticks", type=int, default=4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--features", type=int, default=8192, help="state coordinates sampled per layer")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    table = logo_table(texts(a.train))
    T = [t[:a.max_chars] for t in texts(a.eval)[:a.n]]
    views = {"plain": T, "cyr": [to_cyr(t) for t in T], "logo": [to_logo(t, table) for t in T]}
    loaded = load_rwkv7(a.model, device="cpu", dtype=torch.float32, backend="peft", ctx_len=2048)
    if a.weights:
        load_weights_into(loaded, a.weights)
    D = loaded.wkv_stack(loaded.new_state(batch=1)).reshape(loaded.n_layer, -1).shape[1]
    idx = torch.randperm(D, generator=torch.Generator().manual_seed(0))[:a.features]
    t0 = time.time()
    S = {}
    for v, xs in views.items():
        rows = [states(loaded, x, a.ticks, idx) for x in xs]
        S[(v, 0)] = torch.stack([r[0] for r in rows], 1)                       # (layers, n, features)
        S[(v, a.ticks)] = torch.stack([r[1] for r in rows], 1)
        print(f"[align] {v}: {len(xs)} texts, mean tokens "
              f"{sum(len(loaded.tokenizer.encode(x)) for x in xs) / len(xs):.0f}, {time.time() - t0:.0f}s", flush=True)
    perm = torch.roll(torch.arange(len(T)), 1)
    out = {"features": f"{len(idx)} of {D} per layer", "model": a.model, "weights": str(a.weights) if a.weights else None, "n": len(T), "ticks": a.ticks,
           "logo_words": len(table), "example": {v: xs[0][:120] for v, xs in views.items()}, "cka": {}}
    for M in (0, a.ticks):
        for other in ("cyr", "logo"):
            same = [cka(S[("plain", M)][L], S[(other, M)][L]) for L in range(loaded.n_layer)]
            shuf = [cka(S[("plain", M)][L], S[(other, M)][L][perm]) for L in range(loaded.n_layer)]
            out["cka"][f"M{M}_plain_{other}"] = [round(x, 4) for x in same]
            out["cka"][f"M{M}_plain_{other}_shuffled"] = [round(x, 4) for x in shuf]
            k = loaded.n_layer // 3
            print(f"[align] M{M} plain-{other}: deepest third {sum(same[-k:]) / k:.3f} "
                  f"(shuffled {sum(shuf[-k:]) / k:.3f}); first third {sum(same[:k]) / k:.3f}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
