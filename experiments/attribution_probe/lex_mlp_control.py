#!/usr/bin/env python3
"""Lexical control with the SAME 128-64-1 MLP head as the state probe (loo.py::_train_one), on word
n-gram presence (n<=1 and n<=2), leave-pair-out. The linear unigram gate in bow_control.py cannot see
word conjunctions (entity category x predicate category) or word order (which entity sits next to which
relation); this control can. A state result counts only if it beats this control.

    python experiments/attribution_probe/lex_mlp_control.py experiments/attribution_probe/items_mp_h2{1,2}.jsonl
"""
import json, re, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loo import _train_one
def toks(t): return re.findall(r"[a-z0-9]+", t.lower())
def feats(texts, n):
    grams = lambda t: set(tuple(toks(t)[i:i+k]) for k in range(1, n+1) for i in range(len(toks(t))-k+1))
    G = [grams(t) for t in texts]; vocab = {g: i for i, g in enumerate(sorted(set().union(*G)))}
    X = np.zeros((len(texts), len(vocab)), np.float32)
    for r, g in enumerate(G):
        for x in g: X[r, vocab[x]] = 1
    return X
for p in sys.argv[1:]:
    rows = [json.loads(l) for l in open(p)]
    texts = [r["prompt"] for r in rows]; y = np.array([r["y"] for r in rows], np.float32)
    groups = np.array([r["id"][:-1] for r in rows])
    for n in (1, 2):
        X = feats(texts, n); probs = np.zeros(len(y))
        for g in sorted(set(groups)):
            te = np.where(groups == g)[0]; tr = np.where(groups != g)[0]
            probs[te] = _train_one(X[tr], y[tr], X[te], epochs=500, lr=1e-3, wd=1e-3, seed=2000 + int(te[0]))
        acc = ((probs >= .5) == y).mean()
        pr = np.mean([probs[np.where((groups == g) & (y == 1))[0][0]] > probs[np.where((groups == g) & (y == 0))[0][0]] for g in sorted(set(groups))])
        print(f"{os.path.basename(p)} n-gram<= {n} + MLP head, leave-pair-out: acc={acc:.3f} pair_rank={pr:.3f}", flush=True)
