#!/usr/bin/env python3
"""closed_loop_toy.py — a small RWKV-7 trained from scratch with its own latent loop (H25 §8, HZLK thesis 14).

Plan: ~/.claude/plans/noesis-closed-loop-toy-2026-11-15.md. Every toy before this one is open-loop
(`micro_wkv.py`: the controls never see the state, so silent steps add no computable function, H25 Corollary 2).
Here the controls of every layer are computed from the previous layer's output, and a tick feeds the last
layer's output back as the next input vector, so the loop is closed.

Model. RWKV-7 time mix as in the kernel: token shift, data-dependent decay exp(-exp(w)) with
w = -softplus(-(w0 + lora)) - 0.5 (retention floor 0.5452), the delta rule
S <- S (diag(decay) - kk^T (a * kk)) + v k^T, bonus (r * k * r_k) v, output gate, v residual from layer 0;
channel mix relu(.)^2. ln0 on the input, ln_out before the head. A tick input is ln_out of the last layer's
output at the previous step; it goes through ln0 like an embedding.

Task. Pointer chasing in context. A random single cycle over n of the node symbols is listed as pairs "x y" in
random order, then the query "Q s H_k"; the answer is f^k(s), one symbol. The query comes last, so the hops have
to be done after it is read: a one-token pass can do about as many lookups as it has layers; ticks add steps.
Intermediate nodes f^t(s) are known, which gives every tick its own target in the JEPA arms.

Arms (stage 1 of the plan): --ticks M, --jepa (cosine of a predictor read of tick t's latent to the frozen
embedding of f^t(s)), --sigreg (Epps-Pulley on random 1-D projections of the tick latents across the batch,
after LeJEPA). Stage 0 is --ticks 0 alone.
--distill T.pt (stage 1g): the student learns only the final answer distribution of a trained toy (KL at a
temperature), no per-tick target; with --ticks 0 it asks whether one pass can mimic the loop.
Tasks: --task chase (the table is given whole), affine (stage 1b: m runs of E(x) = a x + b mod 16 in context and a
query symbol not among them, so the rule has to be inferred and applied k times), shift (the same with
E(x) = x + b mod 13: odd a mod 16 has order <= 4, so affine powers shortcut to the identity), mix (half each),
store (stage 1c: K fixed tables in the weights, named by two id tokens; multi-hop tested on held-out starts;
capacity = bits carried by one-hop over every stored edge), deriv (stage 1e: spatium's rewrite derivations, 8
premises in context, depth 1..8 as k; --jepa there pulls tick t toward the model's own latent of hop t's term),
cs (half chase with the curriculum, half one-hop queries over K stored tables: does the step learned on context
apply to facts in the weights; read store_M8 k >= 2, never trained).

    python experiments/A0_state_probe/closed_loop_toy.py --ticks 0 --seed 0 --out results/closed_loop_toy/s0_M0_seed0.json
"""
from __future__ import annotations

import argparse
import collections
import json
import copy
import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

N_NODES = 16                      # node symbols 0..15
Q, PAD = N_NODES, N_NODES + 1     # query marker, unused pad
HOP0 = N_NODES + 2                # hop-count tokens H_1..H_KMAX
KMAX = 8
ID_BASE = 64                      # store task: a table is named by two id tokens, up to 64 x 64 tables
SHIFT_MOD = 13                    # the shift cipher lives on symbols 0..12 (prime: k b != 0 for k <= KMAX)
VOCAB = HOP0 + KMAX


# ---------------------------------------------------------------- task

def make_batch(B: int, n: int, ks: list[int], g: torch.Generator) -> dict:
    """Pairs of a random n-cycle in random order, then Q s H_k. Returns tokens, answer, path f^1..f^k(s)."""
    toks = torch.empty(B, 2 * n + 3, dtype=torch.long)
    ans = torch.empty(B, dtype=torch.long)
    k = torch.tensor([ks[i] for i in torch.randint(len(ks), (B,), generator=g).tolist()])
    path = torch.full((B, KMAX), -1, dtype=torch.long)
    for b in range(B):
        nodes = torch.randperm(N_NODES, generator=g)[:n]
        nxt = {nodes[i].item(): nodes[(i + 1) % n].item() for i in range(n)}       # one cycle: f^t(s) != s for t < n
        order = torch.randperm(n, generator=g).tolist()
        for j, i in enumerate(order):
            x = nodes[i].item()
            toks[b, 2 * j], toks[b, 2 * j + 1] = x, nxt[x]
        s = nodes[torch.randint(n, (1,), generator=g)].item()
        toks[b, 2 * n], toks[b, 2 * n + 1], toks[b, 2 * n + 2] = Q, s, HOP0 + k[b].item() - 1
        cur = s
        for t in range(k[b].item()):
            cur = nxt[cur]
            path[b, t] = cur
        ans[b] = cur
    return {"tokens": toks, "answer": ans, "k": k, "path": path}


def make_affine(B: int, m: int, ks: list[int], g: torch.Generator, length: int = 0, shift: bool = False) -> dict:
    """m runs x -> E(x) of an affine cipher E(x) = a x + b mod N_NODES (a odd), then Q s H_k with s not among the
    runs, so the answer E^k(s) needs the rule, not a lookup. Left-padded with PAD to `length` when given.
    shift: E(x) = x + b mod SHIFT_MOD (prime, b != 0), so E^k is no shortcut for any k <= KMAX."""
    T = max(length, 2 * m + 3)
    toks = torch.full((B, T), PAD, dtype=torch.long)
    ans = torch.empty(B, dtype=torch.long)
    k = torch.tensor([ks[i] for i in torch.randint(len(ks), (B,), generator=g).tolist()])
    path = torch.full((B, KMAX), -1, dtype=torch.long)
    o = T - (2 * m + 3)
    mod = SHIFT_MOD if shift else N_NODES
    for b in range(B):
        if shift:
            a, c = 1, torch.randint(1, mod, (1,), generator=g).item()
            xs = torch.randperm(mod, generator=g)[:m + 1].tolist()
        else:
            a = 2 * torch.randint(N_NODES // 2, (1,), generator=g).item() + 1
            c = torch.randint(N_NODES, (1,), generator=g).item()
            while True:                                                         # an odd difference fixes a
                xs = torch.randperm(N_NODES, generator=g)[:m + 1].tolist()
                if len({x % 2 for x in xs[1:]}) == 2:
                    break
        for j, x in enumerate(xs[1:]):
            toks[b, o + 2 * j], toks[b, o + 2 * j + 1] = x, (a * x + c) % mod
        s = xs[0]
        toks[b, o + 2 * m], toks[b, o + 2 * m + 1], toks[b, o + 2 * m + 2] = Q, s, HOP0 + k[b].item() - 1
        cur = s
        for t in range(k[b].item()):
            cur = (a * cur + c) % mod
            path[b, t] = cur
        ans[b] = cur
    return {"tokens": toks, "answer": ans, "k": k, "path": path}


DEV = torch.device("cpu")          # --device: batches are built on the CPU and moved here


def dev(b: dict) -> dict:
    return {k: v.to(DEV) for k, v in b.items()}


STORE: dict = {}                  # store task: "next" (K, N_NODES) fixed cycles, "held" (K, N_NODES) held-out starts


def set_store(K: int, seed: int) -> None:
    """K fixed random N_NODES-cycles, the same for every arm with this seed; half the starts of each table are held
    out of the multi-hop training queries (one-hop queries see every edge)."""
    g = torch.Generator().manual_seed(777_000 + seed)
    nxt = torch.empty(K, N_NODES, dtype=torch.long)
    held = torch.zeros(K, N_NODES, dtype=torch.bool)
    for i in range(K):
        c = torch.randperm(N_NODES, generator=g)
        nxt[i, c] = torch.roll(c, -1)
        held[i, torch.randperm(N_NODES, generator=g)[:N_NODES // 2]] = True
    STORE.update(next=nxt, held=held, K=K)


def make_store(B: int, ks: list[int], g: torch.Generator, held: bool = False) -> dict:
    """Queries over the stored tables: ID_hi ID_lo Q s H_k -> f_table^k(s); multi-hop starts from the training half,
    or from the held-out half when held (one-hop starts are any node)."""
    K, nxt, hm = STORE["K"], STORE["next"], STORE["held"]
    k = torch.tensor([ks[i] for i in torch.randint(len(ks), (B,), generator=g).tolist()])
    tab = torch.randint(K, (B,), generator=g)
    s = torch.empty(B, dtype=torch.long)
    for b in range(B):
        pool = torch.arange(N_NODES) if k[b] == 1 else (hm[tab[b]] if held else ~hm[tab[b]]).nonzero().flatten()
        s[b] = pool[torch.randint(len(pool), (1,), generator=g)]
    toks = torch.stack([VOCAB + tab // ID_BASE, VOCAB + tab % ID_BASE, torch.full((B,), Q), s, HOP0 + k - 1], 1)
    path = torch.full((B, KMAX), -1, dtype=torch.long)
    cur = s.clone()
    for t in range(KMAX):
        cur = nxt[tab, cur]
        path[:, t] = torch.where(t < k, cur, torch.full_like(cur, -1))
    ans = path[torch.arange(B), k - 1]
    return {"tokens": toks, "answer": ans, "k": k, "path": path}


NEW_ID = ID_BASE * ID_BASE - 1    # doubt task: the id of a table that is not stored (context is the only source)


def make_doubt(B: int, g: torch.Generator, kind: str = "") -> dict:
    """ID_hi ID_lo, 8 pairs of the table in random order, Q s H_1 -> the TRUE next of s. Kinds: known_clean (a stored
    table, context true), known_lie (a stored table, the queried pair altered in context: the weights must overrule
    it), new (a fresh cycle under NEW_ID, context true and the only source). Training draws 1/4, 1/4, 1/2."""
    K, nxt_all = STORE["K"], STORE["next"]
    toks = torch.empty(B, 2 + 16 + 3, dtype=torch.long)
    ans = torch.empty(B, dtype=torch.long)
    kinds = ([kind] * B if kind else
             [("known_clean", "known_lie", "new", "new")[i] for i in torch.randint(4, (B,), generator=g).tolist()])
    for b, kd in enumerate(kinds):
        if kd == "new":
            c = torch.randperm(N_NODES, generator=g)
            nxt = torch.empty(N_NODES, dtype=torch.long)
            nxt[c] = torch.roll(c, -1)
            tab = NEW_ID
        else:
            tab = torch.randint(K, (1,), generator=g).item()
            nxt = nxt_all[tab]
        src = torch.randperm(N_NODES, generator=g)[:8]
        s = src[torch.randint(8, (1,), generator=g)].item()
        toks[b, 0], toks[b, 1] = VOCAB + tab // ID_BASE, VOCAB + tab % ID_BASE
        for j, x in enumerate(src.tolist()):
            y = nxt[x].item()
            if kd == "known_lie" and x == s:                                    # a wrong target, never s itself
                y0 = y
                while y in (y0, s):
                    y = torch.randint(N_NODES, (1,), generator=g).item()
            toks[b, 2 + 2 * j], toks[b, 3 + 2 * j] = x, y
        toks[b, 18], toks[b, 19], toks[b, 20] = Q, s, HOP0
        ans[b] = nxt[s]
    path = torch.full((B, KMAX), -1, dtype=torch.long)
    path[:, 0] = ans
    return {"tokens": toks, "answer": ans, "k": torch.ones(B, dtype=torch.long), "path": path}


@torch.no_grad()
def store_capacity(model, M: int) -> dict:
    """One-hop accuracy over every stored edge and the bits it carries (Fano bound, 15 alternatives per edge)."""
    K, nxt = STORE["K"], STORE["next"]
    tab = torch.arange(K).repeat_interleave(N_NODES)
    s = torch.arange(N_NODES).repeat(K)
    toks = torch.stack([VOCAB + tab // ID_BASE, VOCAB + tab % ID_BASE, torch.full_like(s, Q), s,
                        torch.full_like(s, HOP0)], 1)
    hit = torch.cat([model(toks[i:i + 4096].to(DEV), M)[0].argmax(-1).cpu() == nxt[tab[i:i + 4096], s[i:i + 4096]]
                     for i in range(0, len(s), 4096)])
    p = hit.float().mean().item()
    h = 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    bits = len(s) * max(0.0, math.log2(N_NODES - 1) - h - (1 - p) * math.log2(N_NODES - 2))
    n_params = sum(q.numel() for q in model.parameters())
    return {"edge_acc": round(p, 4), "bits": round(bits), "bits_per_param": round(bits / n_params, 4)}


DERIV_SYMS = (["log", "exp", "d", "g", "scale", "neg", "vadd", "add", "mul", "p", "q", "r", "u", "w", "0", "1/2", "2",
               "-1", "=", "?", "true", "false"] + [f"#{i}" for i in range(1, 9)])
DERIV_ID = {t: VOCAB + i for i, t in enumerate(DERIV_SYMS)}       # deriv symbols after the toy's own tokens
DERIV: dict = {}                  # deriv task: items by split, loaded once from the spatium export


def load_deriv(path: str, answer: str, target: str = "hops") -> None:
    """Rewrite-derivation items (spatium gen_rewrite.py): 8 premises #1..#8, `?`, t0 = t; answer true/false, or
    the local id of the last hop's rule (answer = "last_rule", true items only). Hop terms kept for the per-tick
    latent target (Polish order is the S-expression without parentheses)."""
    rows = {}
    for line in open(path):
        r = json.loads(line)
        if answer == "last_rule" and r["answer"] != "true":
            continue
        ans = f"#{r['last_rule']}" if answer == "last_rule" else r["answer"]
        if target in ("goal", "both"):                                          # no labelled steps: every tick aims at
            q = r["tokens"][len(r["tokens"]) - 1 - r["tokens"][::-1].index("="):]  # the question's right side
            terms = [[DERIV_ID[x] for x in q[1:]]] * KMAX
        else:
            terms = [[DERIV_ID[x] for x in h["term"].replace("(", " ").replace(")", " ").split()] for h in r["hops"]]
        rows.setdefault(r["split"], []).append(([DERIV_ID[x] for x in r["tokens"]], DERIV_ID[ans], r["depth"], terms))
    for split, R in rows.items():
        by = {}
        for i, it in enumerate(R):
            by.setdefault(it[2], []).append(i)
        DERIV[split] = (R, by)


ANON_CLASSES = [["log", "exp", "d", "scale", "vadd", "add", "mul"], ["p", "q", "r", "u", "w"], ["0", "1/2", "2", "-1"]]
ANON = {"on": False}              # --anon: symbols renamed per item within their arity class


def anon_map(g: torch.Generator) -> torch.Tensor:
    """A random renaming of the deriv symbols, each within its class (binary ops, letters, constants), so the
    Polish structure stays readable but no law can be recognised by its symbols: the rules must be read."""
    m = torch.arange(VOCAB + len(DERIV_SYMS))
    for cls in ANON_CLASSES:
        ids = torch.tensor([DERIV_ID[x] for x in cls])
        m[ids] = ids[torch.randperm(len(ids), generator=g)]
    return m


def make_deriv(B: int, ks: list[int], g: torch.Generator, split: str = "train") -> dict:
    """A batch of items at depths drawn from ks, left-padded with PAD; hop terms padded for the tick targets."""
    R, by = DERIV[split]
    ds = [d for d in ks if d in by]
    pick = [by[ds[i]][torch.randint(len(by[ds[i]]), (1,), generator=g).item()]
            for i in torch.randint(len(ds), (B,), generator=g).tolist()]
    items = [R[i] for i in pick]
    T = max(len(it[0]) for it in items)
    toks = torch.full((B, T), PAD, dtype=torch.long)
    TL = max(len(t) for it in items for t in it[3])
    terms = torch.full((B, KMAX, TL), PAD, dtype=torch.long)
    has = torch.zeros(B, KMAX, dtype=torch.bool)
    for b, (tk, ans, d, tm) in enumerate(items):
        mp = anon_map(g) if ANON["on"] else None
        toks[b, T - len(tk):] = mp[torch.tensor(tk)] if mp is not None else torch.tensor(tk)
        for t, term in enumerate(tm[:KMAX]):
            tt = torch.tensor(term)
            terms[b, t, TL - len(term):] = mp[tt] if mp is not None else tt; has[b, t] = True
    return {"tokens": toks, "answer": torch.tensor([it[1] for it in items]), "k": torch.tensor([it[2] for it in items]),
            "terms": terms, "has": has}


CIPHER: dict = {}                 # cipher task: char vocabulary and rows by split


def _cipher_row(r: dict) -> tuple[list[str], list[str]]:
    """Compact form of a gen_ciphers row: cipher and mode tokens, key or table, the message; the answer is the
    plaintext. The English instructions are dropped; every character is one token."""
    lines = r["prompt"].split("\n")
    key = [l.split(":", 1)[1].strip() for l in lines if l.startswith(("Key:", "Table:", "Keyword:", "Shift:"))]
    msg = r["prompt"].split("Message:\n", 1)[1].rsplit("\nAnswer", 1)[0]
    prompt = [f"<{r['category']}>", f"<{r.get('mode', 'known')}>"] + list(" | ".join(key)) + ["<sep>"] + list(msg) + ["<sep>"]
    return prompt, list(r["answer"]) + ["<end>"]


def load_cipher(train_path: str, eval_path: str, max_len: int) -> int:
    """Rows of both splits as token ids over a vocabulary built from them; returns the vocabulary size."""
    raw = {sp: [json.loads(l) for l in open(p)] for sp, p in (("train", train_path), ("eval", eval_path))}
    rows = {sp: [x for x in map(_cipher_row, R) if len(x[0]) + len(x[1]) <= max_len] for sp, R in raw.items()}
    fam = {sp: [r["category"] for r in R if sum(map(len, _cipher_row(r))) <= max_len] for sp, R in raw.items()}
    vocab = sorted({t for R in rows.values() for p_, a in R for t in p_ + a} | {"<pad>"})
    tid = {t: VOCAB + i for i, t in enumerate(vocab)}
    for sp, R in rows.items():
        CIPHER[sp] = [([tid[t] for t in p_], [tid[t] for t in a]) for p_, a in R]
        CIPHER[sp + "_fam"] = fam[sp]
    CIPHER["pad"] = tid["<pad>"]
    return len(vocab)


def make_cipher(B: int, g: torch.Generator, split: str = "train", idx: list | None = None) -> dict:
    """Prompts left-padded, answers right-padded (mask), and the answer repeated per tick as the goal term."""
    R = CIPHER[split]
    idx = idx if idx is not None else torch.randint(len(R), (B,), generator=g).tolist()
    items = [R[i] for i in idx]
    if CIPHER.get("control"):                                                   # control: another row's prompt, so the
        gc = torch.Generator().manual_seed(int(sum(idx)) % (1 << 31))          # answer is guessable from the language
        items = [(R[j][0], a) for (_, a), j in                                  # alone (teacher-forced prefix)
                 zip(items, torch.randint(len(R), (len(items),), generator=gc).tolist())]
    pad = CIPHER["pad"]
    T, A = max(len(p_) for p_, _ in items), max(len(a) for _, a in items)
    toks = torch.full((len(items), T), pad, dtype=torch.long)
    ans = torch.full((len(items), A), pad, dtype=torch.long)
    mask = torch.zeros(len(items), A, dtype=torch.bool)
    for b, (p_, a) in enumerate(items):
        toks[b, T - len(p_):] = torch.tensor(p_)
        ans[b, :len(a)] = torch.tensor(a); mask[b, :len(a)] = True
    terms = torch.full((len(items), KMAX, A), pad, dtype=torch.long)
    for b, (_, a) in enumerate(items):
        terms[b, :, A - len(a):] = torch.tensor(a)
    return {"tokens": toks, "ans": ans, "amask": mask, "terms": terms,
            "has": torch.ones(len(items), KMAX, dtype=torch.bool), "idx": torch.tensor(idx)}


CLM: dict = {}                    # cipherlm task: (plain ids, cipher ids, family) rows by split, char vocabulary
CLM_FAMS = ("caesar", "atbash", "vigenere")   # letter-preserving; plus every "script_*" substitution


def load_cipherlm(train_path: str, eval_path: str, plain_share: float) -> int:
    """Stage 1j: plain and enciphered texts of the c1d rows (letter-preserving families, each row its own key, the key
    and the cipher name never shown); returns the vocabulary size."""
    rows = {}
    for sp, p in (("train", train_path), ("eval", eval_path)):
        R = []
        for r in map(json.loads, open(p)):
            fam = r["category"].replace("cipher_", "")
            if fam in CLM_FAMS or fam.startswith("script_"):
                msg = r["prompt"].split("Message:\n", 1)[1].rsplit("\nAnswer", 1)[0]
                R.append((r["answer"], msg, fam))
        rows[sp] = R
    vocab = sorted({c for R in rows.values() for a, m, _ in R for c in a + m} | {"<pad>", "<text>", "<end>"})
    tid = {t: VOCAB + i for i, t in enumerate(vocab)}
    for sp, R in rows.items():
        CLM[sp] = [([tid[c] for c in a], [tid[c] for c in m], f) for a, m, f in R]
    CLM.update(pad=tid["<pad>"], text=tid["<text>"], end=tid["<end>"], plain=plain_share)
    return len(vocab)


def make_cipherlm(B: int, g: torch.Generator | None, split: str = "train", idx: list | None = None,
                  kind: str = "") -> dict:
    """The first half of a text (at most 120 characters) after <text>, then the next characters (at most 100) and
    <end> as the answer. kind: 'plain', 'cipher', or '' = plain with probability CLM['plain'] (training)."""
    R = CLM[split]
    idx = idx if idx is not None else torch.randint(len(R), (B,), generator=g).tolist()
    seqs = []
    for i in idx:
        a, m, _ = R[i]
        k = kind or ("plain" if torch.rand(1, generator=g).item() < CLM["plain"] else "cipher")
        s = a if k == "plain" else m
        P = min(len(s) // 2, 120)
        seqs.append(([CLM["text"]] + s[:P], s[P:P + 100] + [CLM["end"]]))
    pad = CLM["pad"]
    T, A = max(len(p_) for p_, _ in seqs), max(len(x) for _, x in seqs)
    toks = torch.full((len(seqs), T), pad, dtype=torch.long)
    ans = torch.full((len(seqs), A), pad, dtype=torch.long)
    mask = torch.zeros(len(seqs), A, dtype=torch.bool)
    for b, (p_, x) in enumerate(seqs):
        toks[b, T - len(p_):] = torch.tensor(p_)
        ans[b, :len(x)] = torch.tensor(x); mask[b, :len(x)] = True
    return {"tokens": toks, "ans": ans, "amask": mask}


@torch.no_grad()
def evaluate_cipherlm(model, ticks: int, n: int) -> dict:
    """Teacher-forced next-character accuracy on held-out contents: enciphered (overall and by family) and plain."""
    out = {}
    R = CLM["eval"]
    idx = list(range(min(n, len(R))))
    for M in sorted({0, ticks}):
        for kind in ("cipher", "plain"):
            hit, tot, fam = 0, 0, {}
            for i in range(0, len(idx), 128):
                b = dev(make_cipherlm(0, None, "eval", idx[i:i + 128], kind))
                lg, _ = model.forward_seq(b["tokens"], M, b["ans"])
                ok = (lg.argmax(-1) == b["ans"]) & b["amask"]
                hit += int(ok.sum()); tot += int(b["amask"].sum())
                for j, k in enumerate(idx[i:i + 128]):
                    f = "script" if R[k][2].startswith("script_") else R[k][2]
                    c = fam.setdefault(f, [0, 0]); c[0] += int(ok[j].sum()); c[1] += int(b["amask"][j].sum())
            out[f"M{M}_{kind}"] = round(hit / max(1, tot), 4)
            if kind == "cipher":
                out[f"M{M}_cipher_by_family"] = {f: round(a / max(1, c), 3) for f, (a, c) in sorted(fam.items())}
    return out


@torch.no_grad()
def evaluate_cipher(model, ticks: int, n: int) -> dict:
    """Teacher-forced character accuracy and exact match on held-out contents, overall and by cipher family."""
    out = {}
    R = CIPHER["eval"]
    idx = list(range(min(n, len(R))))
    for M in sorted({0, ticks}):
        hit, tot, ex, fam = 0, 0, 0, {}
        for i in range(0, len(idx), 128):
            b = dev(make_cipher(0, None, "eval", idx[i:i + 128]))
            lg, _ = model.forward_seq(b["tokens"], M, b["ans"])
            right = (lg.argmax(-1) == b["ans"]) | ~b["amask"]
            hit += int(((lg.argmax(-1) == b["ans"]) & b["amask"]).sum()); tot += int(b["amask"].sum())
            for j, k in enumerate(idx[i:i + 128]):
                f = CIPHER["eval_fam"][k].replace("cipher_", "")
                c = fam.setdefault(f, [0, 0]); c[0] += bool(right[j].all()); c[1] += 1
            ex += int(right.all(1).sum())
        out[f"M{M}"] = {"char_acc": round(hit / max(1, tot), 4), "exact": round(ex / max(1, len(idx)), 4),
                        "exact_by_family": {f: round(a / c, 3) for f, (a, c) in sorted(fam.items())}}
    return out


TL: dict = {}                     # translit task (stage 1n): plain and fixed-Cyrillic ids by content split
CYR = dict(zip("abcdefghijklmnopqrstuvwxyz", "абцдефгхийклмнопярстувшжыз"))
CYR.update({k.upper(): v.upper() for k, v in CYR.items()})


def learn_bpe(texts: list[str], merges: int) -> list[tuple[str, str]]:
    """Byte-pair merges learned on `texts`, inside space-led word chunks as subword tokenizers do."""
    words = collections.Counter(w for t in texts for w in re.findall(r" ?[^ ]+| +", t))
    segs = {w: list(w) for w in words}
    out = []
    for _ in range(merges):
        pairs = collections.Counter()
        for w, c in words.items():
            sg = segs[w]
            for a, b in zip(sg, sg[1:]):
                pairs[a, b] += c
        if not pairs:
            break
        (a, b), _ = pairs.most_common(1)[0]
        out.append((a, b))
        for w, sg in segs.items():
            i, new = 0, []
            while i < len(sg):
                if i + 1 < len(sg) and sg[i] == a and sg[i + 1] == b:
                    new.append(a + b); i += 2
                else:
                    new.append(sg[i]); i += 1
            segs[w] = new
    return out


def bpe_encode(t: str, merges: list[tuple[str, str]], cache: dict) -> list[str]:
    toks = []
    for w in re.findall(r" ?[^ ]+| +", t):
        if w not in cache:
            sg = list(w)
            for a, b in merges:
                i, new = 0, []
                while i < len(sg):
                    if i + 1 < len(sg) and sg[i] == a and sg[i + 1] == b:
                        new.append(a + b); i += 2
                    else:
                        new.append(sg[i]); i += 1
                sg = new
            cache[w] = sg
        toks += cache[w]
    return toks


def load_translit(train_path: str, eval_path: str, mode: str, tok: str = "char", merges: int = 128) -> int:
    """Stage 1n: the plaintexts of the c1d rows, each also in one fixed Cyrillic letter table (never shown). Contents
    split with no overlap: A (even train rows) seen only in plain, B (odd train rows) only in Cyrillic, C (eval)
    never. mode 'plain' / 'cyr': every text in that script (the controls). tok (stage 1s): 'char', or subword merges
    learned on the train texts in plain English only ('bpe_en', as real tokenizers favour one script) or in both
    scripts ('bpe_both'); the prefix/answer cut stays at the same characters. Returns the vocabulary size."""
    texts = {}
    for sp, p in (("train", train_path), ("eval", eval_path)):
        seen, R = set(), []
        for r in map(json.loads, open(p)):
            if r["answer"] not in seen:
                seen.add(r["answer"]); R.append(r["answer"])
        texts[sp] = R
    chars = {c for R in texts.values() for t in R for c in t}
    cy = lambda t: "".join(CYR.get(c, c) for c in t)
    mg = []
    if tok != "char":
        src = texts["train"] + ([cy(t) for t in texts["train"]] if tok == "bpe_both" else [])
        mg = learn_bpe(src, merges)
    vocab = sorted(chars | {CYR.get(c, c) for c in chars} | {a + b for a, b in mg} | {"<pad>", "<text>", "<end>"})
    tid = {t: VOCAB + i for i, t in enumerate(vocab)}
    cache: dict = {}

    def one(t):                                     # (prefix ids, answer ids, answer characters) at a fixed cut
        P = min(len(t) // 2, 120)
        cut = (lambda x: [tid[c] for c in x]) if not mg else (lambda x: [tid[p] for p in bpe_encode(x, mg, cache)])
        return cut(t[:P]), cut(t[P:P + 100]), len(t[P:P + 100])
    enc = lambda t: (one(t), one(cy(t)))
    TL.update(A=[enc(t) for t in texts["train"][0::2]], B=[enc(t) for t in texts["train"][1::2]],
              C=[enc(t) for t in texts["eval"]], pad=tid["<pad>"], text=tid["<text>"], end=tid["<end>"],
              mode=mode, q=0.0, tok=tok, pairs=[(tid[c], tid[CYR[c]]) for c in sorted(CYR) if c in tid and CYR[c] in tid])
    return len(vocab)


def make_translit(B: int, g: torch.Generator | None, part: str = "", idx: list | None = None, kind: str = "") -> dict:
    """Training (part ''): an A row in plain, or with probability TL['q'] a B row in Cyrillic (in plain under the
    plain control). Eval: rows `idx` of `part` in script `kind`. Prefix of at most 120 characters after <text>,
    then up to 100 characters and <end> as the answer, as in 1j. nchar = answer characters + <end> (bpc unit)."""
    seqs, nchar = [], []
    for j in range(B if idx is None else len(idx)):
        if idx is None:
            useB = torch.rand(1, generator=g).item() < TL["q"]
            R = TL["B" if useB else "A"]
            pl, cy = R[torch.randint(len(R), (1,), generator=g).item()]
            s = cy if TL["mode"] == "cyr" or (useB and TL["mode"] != "plain") else pl
        else:
            pl, cy = TL[part][idx[j]]
            s = pl if kind == "plain" else cy
        seqs.append(([TL["text"]] + s[0], s[1] + [TL["end"]]))
        nchar.append(s[2] + 1)
    pad = TL["pad"]
    T, A = max(len(p_) for p_, _ in seqs), max(len(x) for _, x in seqs)
    toks = torch.full((len(seqs), T), pad, dtype=torch.long)
    ans = torch.full((len(seqs), A), pad, dtype=torch.long)
    mask = torch.zeros(len(seqs), A, dtype=torch.bool)
    for b, (p_, x) in enumerate(seqs):
        toks[b, T - len(p_):] = torch.tensor(p_)
        ans[b, :len(x)] = torch.tensor(x); mask[b, :len(x)] = True
    return {"tokens": toks, "ans": ans, "amask": mask, "nchar": torch.tensor(nchar)}


def _cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between two (n, d) representations of the same n items."""
    X, Y = X - X.mean(0), Y - Y.mean(0)
    return float((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm() + 1e-12))


@torch.no_grad()
def evaluate_translit(model, ticks: int, n: int) -> dict:
    """bpc on every content split in both scripts, the two transfer ratios, per-layer CKA between the plain and the
    Cyrillic reading of the same held-out texts (control: another text's Cyrillic), and embedding alignment."""
    out = {}
    for M in sorted({0, ticks}):
        bpc = {}
        for part in ("A", "B", "C"):
            for kind in ("plain", "cyr"):
                idx = list(range(min(n, len(TL[part]))))
                nll, tot = 0.0, 0
                for i in range(0, len(idx), 128):
                    b = dev(make_translit(0, None, part, idx[i:i + 128], kind))
                    lg, _ = model.forward_seq(b["tokens"], M, b["ans"])
                    nll += F.cross_entropy(lg[b["amask"]], b["ans"][b["amask"]], reduction="sum").item()
                    tot += int(b["nchar"].sum())
                bpc[f"{part}_{kind}"] = round(nll / max(1, tot) / math.log(2), 4)
        out[f"M{M}_bpc"] = bpc
        d_pc = bpc["C_plain"] - bpc["A_plain"]
        d_cp = bpc["C_cyr"] - bpc["B_cyr"]
        out[f"M{M}_transfer_plain_to_cyr"] = round((bpc["C_cyr"] - bpc["A_cyr"]) / d_pc, 3) if abs(d_pc) > 1e-3 else None
        out[f"M{M}_transfer_cyr_to_plain"] = round((bpc["C_plain"] - bpc["B_plain"]) / d_cp, 3) if abs(d_cp) > 1e-3 else None
    idx = list(range(min(256, len(TL["C"]))))
    bp, bc = make_translit(0, None, "C", idx, "plain"), make_translit(0, None, "C", idx, "cyr")
    rot = idx[1:] + idx[:1]
    bo = make_translit(0, None, "C", rot, "cyr")
    rp, rc, ro = (model.trace(dev(x)["tokens"]) for x in (bp, bc, bo))
    if TL.get("tok", "char") == "char":                                         # positions align only per character
        m = (bp["tokens"] != TL["pad"]) & (bo["tokens"] != TL["pad"])             # positions real in both texts
        out["cka_same_text"] = [round(_cka(a[m], c[m]), 3) for a, c in zip(rp, rc)]
        out["cka_other_text"] = [round(_cka(a[m], o[m]), 3) for a, o in zip(rp, ro)]
    out["cka_end_same"] = [round(_cka(a[:, -1], c[:, -1]), 3) for a, c in zip(rp, rc)]    # last prefix position:
    out["cka_end_other"] = [round(_cka(a[:, -1], o[:, -1]), 3) for a, o in zip(rp, ro)]   # any tokenization
    E = model.emb.weight
    lat = torch.tensor([a for a, _ in TL["pairs"]], device=E.device)
    cyr = torch.tensor([c for _, c in TL["pairs"]], device=E.device)
    sim = F.normalize(E[cyr], dim=-1) @ F.normalize(E[lat], dim=-1).T
    out["emb_align"] = round((sim.argmax(1) == torch.arange(len(lat), device=E.device)).float().mean().item(), 4)
    return out


def sample(task: str, B: int, n: int, m: int, ks: list[int], g: torch.Generator) -> dict:
    """chase (tables given whole), affine (rule seen through runs) or mix (half each, one length)."""
    if task == "chase":
        return make_batch(B, n, ks, g)
    if task == "store":
        return make_store(B, ks, g)
    if task == "deriv":
        return make_deriv(B, ks, g)
    if task == "cipher":
        return make_cipher(B, g)
    if task == "cipherlm":
        return make_cipherlm(B, g)
    if task == "translit":
        return make_translit(B, g)
    if task == "doubt":
        return make_doubt(B, g)
    if task == "cs":                                                           # the step from context, facts by one hop
        x, y = make_batch(B // 2, n, ks, g), make_store(B - B // 2, [1], g)
        T = x["tokens"].shape[1]
        y["tokens"] = torch.cat([torch.full((len(y["tokens"]), T - y["tokens"].shape[1]), PAD), y["tokens"]], 1)
        return {key: torch.cat([x[key], y[key]]) for key in ("tokens", "answer", "k", "path")}
    if task in ("affine", "shift"):
        return make_affine(B, m, ks, g, shift=task == "shift")
    x, y = make_batch(B // 2, n, ks, g), make_affine(B - B // 2, m, ks, g, length=2 * n + 3)
    return {key: torch.cat([x[key], y[key]]) for key in x}


# ---------------------------------------------------------------- model

class TimeMix(nn.Module):
    def __init__(self, C: int, H: int, layer: int, n_layer: int, lora: int = 8):
        super().__init__()
        self.C, self.H, self.N, self.layer = C, H, C // H, layer
        ratio = layer / max(1, n_layer - 1)
        self.mu = nn.Parameter(torch.full((6, C), 0.5))                        # r w k v a g
        self.W_r, self.W_k = nn.Linear(C, C, bias=False), nn.Linear(C, C, bias=False)
        self.W_v, self.W_o = nn.Linear(C, C, bias=False), nn.Linear(C, C, bias=False)
        nn.init.zeros_(self.W_o.weight)
        decay = torch.tensor([-6 + 5 * (i / (C - 1)) ** (0.85 + ratio ** 0.5) for i in range(C)])
        self.w0 = nn.Parameter(decay + 0.5)
        self.w1, self.w2 = nn.Parameter(torch.zeros(C, lora)), nn.Parameter(torch.randn(lora, C) * 0.1)
        self.a0 = nn.Parameter(torch.zeros(C))
        self.a1, self.a2 = nn.Parameter(torch.zeros(C, lora)), nn.Parameter(torch.randn(lora, C) * 0.1)
        self.v0 = nn.Parameter(torch.ones(C))
        self.v1, self.v2 = nn.Parameter(torch.zeros(C, lora)), nn.Parameter(torch.randn(lora, C) * 0.1)
        self.g1, self.g2 = nn.Parameter(torch.zeros(C, lora)), nn.Parameter(torch.randn(lora, C) * 0.1)
        self.k_k, self.k_a = nn.Parameter(torch.full((C,), 0.85)), nn.Parameter(torch.ones(C))
        self.r_k = nn.Parameter(torch.zeros(self.H, self.N))
        self.ln_x = nn.GroupNorm(self.H, C, eps=64e-5)

    def forward(self, x, prev, S, v_first):
        """One token step. x, prev: (B, C); S: (B, H, N, N) as value x key. Returns out, S, v_first."""
        B, H, N = x.shape[0], self.H, self.N
        xx = prev - x
        xr, xw, xk, xv, xa, xg = (x + xx * self.mu[i] for i in range(6))
        r, k, v = self.W_r(xr), self.W_k(xk), self.W_v(xv)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        decay = torch.exp(-torch.exp(w))
        if self.layer == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = F.normalize((k * self.k_k).view(B, H, N), dim=-1)
        k = k * (1 + (a - 1) * self.k_a)
        r_, k_, v_, w_, a_ = (t.view(B, H, N) for t in (r, k, v, decay, a))
        S = (S * w_.unsqueeze(-2)
             - (S @ kk.unsqueeze(-1)) @ (kk * a_).unsqueeze(-2)
             + v_.unsqueeze(-1) @ k_.unsqueeze(-2))
        out = (S @ r_.unsqueeze(-1)).squeeze(-1)
        out = self.ln_x(out.reshape(B, -1))
        out = out + ((r_ * k_ * self.r_k).sum(-1, keepdim=True) * v_).reshape(B, -1)
        return self.W_o(out * g), S, v_first


class ChannelMix(nn.Module):
    def __init__(self, C: int, mult: int = 2):
        super().__init__()
        self.mu = nn.Parameter(torch.full((C,), 0.5))
        self.W_k, self.W_v = nn.Linear(C, mult * C, bias=False), nn.Linear(mult * C, C, bias=False)
        nn.init.zeros_(self.W_v.weight)

    def forward(self, x, prev):
        return self.W_v(torch.relu(self.W_k(x + (prev - x) * self.mu)) ** 2)


class ToyRWKV7(nn.Module):
    def __init__(self, C: int = 48, H: int = 3, L: int = 2, vocab: int = VOCAB):
        super().__init__()
        self.C, self.H, self.L = C, H, L
        self.emb = nn.Embedding(vocab, C)
        nn.init.uniform_(self.emb.weight, -1e-4 * 100, 1e-4 * 100)
        self.ln0 = nn.LayerNorm(C)
        self.ln1 = nn.ModuleList(nn.LayerNorm(C) for _ in range(L))
        self.ln2 = nn.ModuleList(nn.LayerNorm(C) for _ in range(L))
        self.att = nn.ModuleList(TimeMix(C, H, i, L) for i in range(L))
        self.ffn = nn.ModuleList(ChannelMix(C) for _ in range(L))
        self.ln_out = nn.LayerNorm(C)
        self.head = nn.Linear(C, vocab, bias=False)
        self.pred = nn.Linear(C, C, bias=False)                                 # JEPA predictor read of a tick

    def init_state(self, B: int):
        d = self.emb.weight.device
        z = torch.zeros(B, self.C, device=d)
        return {"S": [torch.zeros(B, self.H, self.C // self.H, self.C // self.H, device=d) for _ in range(self.L)],
                "att": [z] * self.L, "ffn": [z] * self.L}

    def step(self, x_in, st):
        """x_in: (B, C) input vector before ln0. Returns ln_out(h) and the new state."""
        x = self.ln0(x_in)
        v_first = None
        S, att, ffn = list(st["S"]), list(st["att"]), list(st["ffn"])
        for i in range(self.L):
            xa = self.ln1[i](x)
            dx, S[i], v_first = self.att[i](xa, att[i], S[i], v_first)
            att[i] = xa
            x = x + dx
            xf = self.ln2[i](x)
            x = x + self.ffn[i](xf, ffn[i])
            ffn[i] = xf
        return self.ln_out(x), {"S": S, "att": att, "ffn": ffn}

    def encode(self, tokens):
        """ln_out after reading `tokens` from a fresh state: the latent of a term (deriv per-tick target)."""
        st = self.init_state(tokens.shape[0])
        e = self.emb(tokens)
        for t in range(tokens.shape[1]):
            h, st = self.step(e[:, t], st)
        return h

    def forward_seq(self, tokens, ticks: int, ans):
        """Read the prompt, `ticks` latent steps, then the answer teacher-forced: logits for every answer token."""
        st = self.init_state(tokens.shape[0])
        e = self.emb(tokens)
        for t in range(tokens.shape[1]):
            h, st = self.step(e[:, t], st)
        lat = []
        for _ in range(ticks):
            h, st = self.step(h, st)
            lat.append(h)
        outs, ea = [self.head(h)], self.emb(ans)
        for i in range(ans.shape[1] - 1):
            h, st = self.step(ea[:, i], st)
            outs.append(self.head(h))
        return torch.stack(outs, 1), lat

    def forward(self, tokens, ticks: int, noise: float = 0.0, perm: torch.Tensor | None = None):
        """Read the tokens, then `ticks` latent steps. Returns logits at the answer and the tick latents.
        perm (ablation): each tick is fed another example's latent, the state stays the example's own."""
        B, T = tokens.shape
        st = self.init_state(B)
        e = self.emb(tokens)
        for t in range(T):
            h, st = self.step(e[:, t], st)
        lat = []
        for _ in range(ticks):
            z = h + noise * torch.randn_like(h) if noise else h
            if perm is not None:
                z = z[perm]
            h, st = self.step(z, st)
            lat.append(h)
        return self.head(h), lat, st

    def rollout(self, tokens, ticks: int, sigma: float):
        """A Gaussian policy over the tick latents (stage 1k): each tick's input is z = h + sigma * eps, an action.
        Returns the answer logits and log N(z; h, sigma^2) summed over ticks (0 when sigma = 0)."""
        st = self.init_state(tokens.shape[0])
        e = self.emb(tokens)
        for t in range(tokens.shape[1]):
            h, st = self.step(e[:, t], st)
        logp = torch.zeros(tokens.shape[0], device=h.device)
        for _ in range(ticks):
            if sigma:
                z = (h + sigma * torch.randn_like(h)).detach()
                logp = logp - ((z - h) ** 2).sum(-1) / (2 * sigma ** 2)
            else:
                z = h
            h, st = self.step(z, st)
        return self.head(h), logp

    def sample_seq(self, tokens, ticks: int, A: int, sigma: float):
        """Stage 1m: read the prompt, `ticks` noisy latent steps (Gaussian policy), then sample A answer tokens from the
        model's own distribution. Returns the tokens, the latent log-probability and per-token log-probabilities."""
        st = self.init_state(tokens.shape[0])
        e = self.emb(tokens)
        for t in range(tokens.shape[1]):
            h, st = self.step(e[:, t], st)
        lpl = torch.zeros(tokens.shape[0], device=h.device)
        for _ in range(ticks):
            if sigma:
                z = (h + sigma * torch.randn_like(h)).detach()
                lpl = lpl - ((z - h) ** 2).sum(-1) / (2 * sigma ** 2)
            else:
                z = h
            h, st = self.step(z, st)
        out, lpt = [], []
        for i in range(A):
            lp = F.log_softmax(self.head(h), -1)
            a = torch.multinomial(lp.exp(), 1).squeeze(1)
            out.append(a); lpt.append(lp.gather(1, a[:, None]).squeeze(1))
            if i < A - 1:
                h, st = self.step(self.emb(a), st)
        return torch.stack(out, 1), lpl, torch.stack(lpt, 1)

    def trace(self, tokens):
        """Each layer's output (before ln_out) at every prompt position, plus ln0's input embedding: a list of
        (B, T, C), embedding first (stage 1n's CKA)."""
        st = self.init_state(tokens.shape[0])
        e = self.emb(tokens)
        reps = [[] for _ in range(self.L + 1)]
        for t in range(tokens.shape[1]):
            x = self.ln0(e[:, t])
            reps[0].append(x)
            S, att, ffn = list(st["S"]), list(st["att"]), list(st["ffn"])
            v_first = None
            for i in range(self.L):
                xa = self.ln1[i](x)
                dx, S[i], v_first = self.att[i](xa, att[i], S[i], v_first)
                att[i] = xa
                x = x + dx
                xf = self.ln2[i](x)
                x = x + self.ffn[i](xf, ffn[i])
                ffn[i] = xf
                reps[i + 1].append(x)
            st = {"S": S, "att": att, "ffn": ffn}
        return [torch.stack(r, 1) for r in reps]

    def forward_seq_noisy(self, tokens, ticks: int, ans, sigma: float):
        """forward_seq with a Gaussian policy on each tick latent: logits for every answer token and log N(z; h)."""
        st = self.init_state(tokens.shape[0])
        e = self.emb(tokens)
        for t in range(tokens.shape[1]):
            h, st = self.step(e[:, t], st)
        lpl = torch.zeros(tokens.shape[0], device=h.device)
        for _ in range(ticks):
            z = (h + sigma * torch.randn_like(h)).detach()
            lpl = lpl - ((z - h) ** 2).sum(-1) / (2 * sigma ** 2)
            h, st = self.step(z, st)
        outs, ea = [self.head(h)], self.emb(ans)
        for i in range(ans.shape[1] - 1):
            h, st = self.step(ea[:, i], st)
            outs.append(self.head(h))
        return torch.stack(outs, 1), lpl

    def tick_logits(self, tokens, ticks: int):
        """The answer head read after every tick, no noise: (B, ticks, V)."""
        st = self.init_state(tokens.shape[0])
        e = self.emb(tokens)
        for t in range(tokens.shape[1]):
            h, st = self.step(e[:, t], st)
        out = []
        for _ in range(ticks):
            h, st = self.step(h, st)
            out.append(self.head(h))
        return torch.stack(out, 1)

    def rollout_stop(self, tokens, ticks: int, sigma: float, tau: float, stop: torch.Tensor | None = None,
                     sample: bool = False, explore: float = 0.0):
        """Stage 1l: no stop head. The answer head is read after every tick and the loop stops at the first tick whose
        answer is confident (max p >= tau), else at `ticks`; `stop` (1-based) forces the tick instead (control);
        `sample` (training) stops at each tick with probability = that confidence.
        Returns the logits at the stop, the stop tick and the latent log-probability of the ticks taken."""
        B = tokens.shape[0]
        st = self.init_state(B)
        e = self.emb(tokens)
        for t in range(tokens.shape[1]):
            h, st = self.step(e[:, t], st)
        logits, lps, lp = [], [], torch.zeros(B, device=h.device)
        for _ in range(ticks):
            if sigma:
                z = (h + sigma * torch.randn_like(h)).detach()
                lp = lp - ((z - h) ** 2).sum(-1) / (2 * sigma ** 2)
            else:
                z = h
            h, st = self.step(z, st)
            logits.append(self.head(h))
            lps.append(lp)
        logits, lps = torch.stack(logits, 1), torch.stack(lps, 1)                # (B, M, V), (B, M)
        if stop is None and sample:                                             # stop at tick t with probability = the
            c = logits.softmax(-1).amax(-1).clamp(1e-4, 1 - 1e-4)              # answer's own confidence; the choices
            c = c * (1 - explore)                                               # explore: some rollouts go on
            go = torch.bernoulli(c.detach()).bool()                             # enter log pi
            go[:, -1] = True
            stop = go.float().argmax(1) + 1
            t = torch.arange(ticks, device=h.device)[None]
            dec = torch.where(t < (stop - 1)[:, None], torch.log1p(-c), torch.zeros_like(c))
            dec = dec + torch.where(t == (stop - 1)[:, None], torch.log(c), torch.zeros_like(c)) * (stop[:, None] < ticks)
            lps = lps + dec.cumsum(1)
        if stop is None:
            conf = logits.softmax(-1).amax(-1) >= tau
            conf[:, -1] = True
            stop = conf.float().argmax(1) + 1                                   # first confident tick
        i = (stop - 1).clamp(0, ticks - 1)
        ar = torch.arange(B, device=h.device)
        return logits[ar, i], stop, lps[ar, i]


# ---------------------------------------------------------------- losses

def sigreg(z: torch.Tensor, n_slices: int = 64, g: torch.Generator | None = None) -> torch.Tensor:
    """Epps-Pulley statistic of random 1-D projections of z (B, C) against N(0, 1), averaged (LeJEPA, Alg. 1)."""
    A = torch.randn(z.shape[1], n_slices, generator=g).to(z.device)
    A = A / A.norm(dim=0)
    t = torch.linspace(-5, 5, 17)
    phi = torch.exp(-0.5 * t ** 2)
    x = (z @ A).unsqueeze(-1) * t                                               # (B, M, T)
    ecf = torch.complex(torch.cos(x), torch.sin(x)).mean(0)
    err = (ecf - phi).abs().square() * phi
    return torch.trapezoid(err, t, dim=-1).mean() * z.shape[0]


# ---------------------------------------------------------------- train / eval

@torch.no_grad()
def evaluate(model, n: int, ticks: int, per_k: int, seed: int, shuffle: bool = False,
             task: str = "chase", m: int = 4) -> dict:
    """Accuracy by k at M = 0 and M = ticks (and with tick latents shuffled); per family, prefixed in a mix."""
    g = torch.Generator().manual_seed(10_000 + seed)
    out = {}
    if task == "deriv":                                                         # accuracy by depth per held-out split
        for split in ("val", "val_laws"):
            if split not in DERIV:
                continue
            runs = [(f"{split}_M{M}", M, False) for M in sorted({0, ticks})]
            if shuffle and ticks:
                runs.append((f"{split}_M{ticks}_shuffled", ticks, True))
            for key, M, shuf in runs:
                acc = {}
                for k in range(1, KMAX + 1):
                    b = dev(make_deriv(min(per_k, 256), [k], g, split))
                    perm = torch.randperm(len(b["answer"]), generator=g).to(DEV) if shuf else None
                    acc[k] = round((model(b["tokens"], M, perm=perm)[0].argmax(-1) == b["answer"]).float().mean().item(), 4)
                out[key] = acc
        return out
    if task == "cs":                                                            # transfer: k-hop over stored facts never
        out.update({f"chase_{k_}": v for k_, v in evaluate(model, n, ticks, per_k, seed, task="chase").items()})
        out.update({f"store_{k_}": v for k_, v in evaluate(model, n, ticks, per_k, seed, task="store").items()})
        return out                                                              # trained beyond one hop
    if task == "doubt":                                                         # truth by kind; on lies, how often the
        for M in sorted({0, ticks}):                                            # model repeats the context instead
            for kd in ("known_clean", "known_lie", "new"):
                b = dev(make_doubt(per_k * 2, g, kd))
                pred = model(b["tokens"], M)[0].argmax(-1)
                out[f"M{M}_{kd}"] = round((pred == b["answer"]).float().mean().item(), 4)
                if kd == "known_lie":
                    told = b["tokens"][:, 3:18:2][b["tokens"][:, 2:18:2] == b["tokens"][:, 19:20]]
                    out[f"M{M}_lie_repeated"] = round((pred == told).float().mean().item(), 4)
            out[f"capacity_M{M}"] = store_capacity(model, M)
        return out
    if task == "store":                                                         # held-out multi-hop starts; capacity
        for M in sorted({0, ticks}):
            for key, held in ((f"M{M}", True), (f"M{M}_seen", False)):
                acc = {}
                for k in range(1, KMAX + 1):
                    b = dev(make_store(per_k, [k], g, held=held))
                    acc[k] = round((model(b["tokens"], M)[0].argmax(-1) == b["answer"]).float().mean().item(), 4)
                out[key] = acc
            out[f"capacity_M{M}"] = store_capacity(model, M)
        return out
    for fam in (["chase", "affine"] if task == "mix" else [task]):
        pre = f"{fam}_" if task == "mix" else ""
        runs = [(f"{pre}M{M}", M, False) for M in sorted({0, ticks})]
        if shuffle and ticks:                                                   # tick latents shuffled across examples
            runs.append((f"{pre}M{ticks}_shuffled", ticks, True))
        for key, M, shuf in runs:
            acc = {}
            for k in range(1, KMAX + 1):
                b = dev(sample(fam, per_k, n, m, [k], g))
                perm = torch.randperm(per_k, generator=g).to(DEV) if shuf else None
                logits, _, _ = model(b["tokens"], M, perm=perm)
                acc[k] = round((logits.argmax(-1) == b["answer"]).float().mean().item(), 4)
            out[key] = acc
    return out


@torch.no_grad()
def evaluate_stop(model, n: int, ticks: int, tau: float, per_k: int, seed: int) -> dict:
    """Stage 1l, chase: accuracy and ticks used by k under the model's own stop (first confident tick), against the
    fixed M = ticks and a random stop with the same ticks permuted across all items (equal mean M, ZLK.5)."""
    g = torch.Generator().manual_seed(20_000 + seed)
    bs = [dev(make_batch(per_k, n, [k], g)) for k in range(1, KMAX + 1)]
    toks, gold = torch.cat([b["tokens"] for b in bs]), torch.cat([b["answer"] for b in bs])
    lg, used, _ = model.rollout_stop(toks, ticks, 0.0, tau)
    lf = model(toks, ticks)[0]
    lr_, _, _ = model.rollout_stop(toks, ticks, 0.0, tau, stop=used[torch.randperm(len(used), generator=g).to(DEV)])
    out = {"own": {}, "ticks": {}, "fixed": {}, "random_stop": {}}
    for k in range(1, KMAX + 1):
        s = slice((k - 1) * per_k, k * per_k)
        out["own"][k] = round((lg[s].argmax(-1) == gold[s]).float().mean().item(), 4)
        out["ticks"][k] = round(used[s].float().mean().item(), 3)
        out["fixed"][k] = round((lf[s].argmax(-1) == gold[s]).float().mean().item(), 4)
        out["random_stop"][k] = round((lr_[s].argmax(-1) == gold[s]).float().mean().item(), 4)
    out["tick_share"] = round(used.float().mean().item() / ticks, 4)
    for key in ("own", "fixed", "random_stop"):
        out[f"{key}_mean"] = round(sum(out[key].values()) / KMAX, 4)
    return out


def train(args) -> dict:
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    if args.task in ("store", "cs", "doubt"):
        set_store(args.K, args.fact_seed)
    if args.task == "deriv":
        load_deriv(args.deriv_file, args.deriv_answer, args.jepa_target)
        ANON["on"] = args.anon
    extra = {"store": ID_BASE, "cs": ID_BASE, "doubt": ID_BASE, "deriv": len(DERIV_SYMS)}.get(args.task, 0)
    if args.task == "cipher":
        extra = load_cipher(args.cipher_train, args.cipher_eval, args.cipher_max_len)
        CIPHER["control"] = args.cipher_control
    if args.task == "cipherlm":
        extra = load_cipherlm(args.cipher_train, args.cipher_eval, args.clm_plain)
    if args.task == "translit":
        extra = load_translit(args.cipher_train, args.cipher_eval, args.tl_mode, args.tl_tok, args.tl_merges)
    model = ToyRWKV7(args.width, args.heads, args.layers, VOCAB + extra)
    if args.jepa_target == "both":                                              # its own head for the self target
        model.pred_self = nn.Linear(args.width, args.width, bias=False)
    global DEV
    DEV = torch.device(args.device)
    model.to(DEV)
    ref = None
    if args.init:                                                               # continue a trained toy (stage 1k)
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
        if (args.grpo and args.grpo_kl) or (args.s3_self and args.s3_target == "ref"):              # the start, frozen: KL anchor and
            ref = copy.deepcopy(model).requires_grad_(False).eval()             # the reference the ticks imitate
    teacher = None
    if args.distill:                                                            # a trained toy, its own size and ticks
        targs = json.loads(Path(args.distill).with_suffix(".json").read_text())["args"]
        teacher = ToyRWKV7(targs["width"], targs["heads"], targs["layers"], VOCAB + extra)
        teacher.load_state_dict(torch.load(args.distill, map_location="cpu"))
        teacher.to(DEV).requires_grad_(False).eval()
        teacher_ticks = targs["ticks"]
    frozen_emb = model.emb.weight.detach().clone()                              # fixed teacher for the JEPA target
    ema = (copy.deepcopy(model).requires_grad_(False)
           if args.jepa and (args.task in ("deriv", "cipher") or args.jepa_target == "self") else None)
    n_params = sum(p.numel() for p in model.parameters())
    if args.eval_only:                                                          # re-read a saved toy with new reads
        assert args.task == "translit" and args.init, "--eval-only: translit, with --init"
        args.save_model = False
        return {"args": vars(args), "n_params": n_params, "history": [], "sec": 0.0,
                "final": evaluate_translit(model, args.ticks, args.eval_final)}
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    ks = [int(x) for x in args.train_ks.split(',')] if args.train_ks else list(range(1, KMAX + 1))
    hist, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        if args.curriculum:                                                     # k_max grows 1 -> KMAX over 3/4 of the steps
            cs = max(0, step - 1 - args.warm)                                   # after the one-hop warm phase
            kmax = min(KMAX, 1 + cs * KMAX // max(1, int(0.75 * (args.steps - args.warm))))
            ks = list(range(1, kmax + 1)) if step > args.warm else [1]
        if args.task == "translit":                                             # plain only for the first quarter,
            TL["q"] = (0.5 if args.init else                                    # a fine-tune keeps the final mix
                       0.5 * min(1.0, max(0.0, (step / args.steps - 0.25) / 0.5)))  # then B (Cyrillic) up to half
        b = dev(sample(args.task, args.batch, args.n, args.m, ks, g))
        if args.task in ("cipher", "cipherlm", "translit"):                                 # CE on every answer character
            lg, lat = model.forward_seq(b["tokens"], args.ticks, b["ans"])
            ce = F.cross_entropy(lg[b["amask"]], b["ans"][b["amask"]])
            if args.grpo and args.task == "cipher":                             # loss and RL at once: the model writes
                G = args.grpo                                                   # its own decryption, reward = share of
                toks, gold = b["tokens"].repeat_interleave(G, 0), b["ans"].repeat_interleave(G, 0)  # right characters
                msk = b["amask"].repeat_interleave(G, 0)
                a, lpl, lpt = model.sample_seq(toks, args.ticks, gold.shape[1], args.grpo_sigma)
                n_ = msk.sum(1).clamp(min=1)
                r = (((a == gold) & msk).sum(1) / n_).view(-1, G)
                adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
                pg = -(adv * (lpl + (lpt * msk).sum(1)) / n_).mean()            # per answer character
                ce, rl_parts = ce + args.grpo_w * pg, {"pg": pg.item(), "reward": r.mean().item()}
            if args.grpo and args.task == "translit":                           # RL on the ticks only: G noisy tick
                G = args.grpo                                                   # trajectories per text, reward = minus
                toks, gold = b["tokens"].repeat_interleave(G, 0), b["ans"].repeat_interleave(G, 0)  # the continuation's
                msk = b["amask"].repeat_interleave(G, 0)                        # CE; the answer is never sampled
                lgn, lpl = model.forward_seq_noisy(toks, args.ticks, gold, args.grpo_sigma)
                nll = F.cross_entropy(lgn.transpose(1, 2), gold, reduction="none")
                r = (-(nll * msk).sum(1) / msk.sum(1).clamp(min=1)).detach().view(-1, G)
                adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
                pg = -(adv * lpl).mean() / max(1, args.ticks * args.width)      # per latent coordinate
                ce, rl_parts = ce + args.grpo_w * pg, {"pg": pg.item(), "reward": r.mean().item()}
        elif args.s3_self and not args.grpo:                                    # imitation alone: every tick toward the
            tl = F.log_softmax(model.tick_logits(b["tokens"], args.ticks), -1)  # frozen start's answer, no label, no
            with torch.no_grad():                                               # reward (isolates the shared-head cost)
                tgt = F.softmax(ref(b["tokens"], args.ticks)[0], -1)[:, None]
            w = (torch.arange(1, args.ticks + 1, device=tl.device) / args.ticks) ** args.s3_ramp  # tick t weighs (t/M)^p:
            ce = -((tgt * tl).sum(-1) * w).sum(1).mean() / w.sum()              # the last tick holds the reference,
            lat, logits = [], None                                              # early ticks are pulled less
        elif args.grpo:                                                         # G rollouts per prompt, reward = exact
            G = args.grpo                                                       # answer, advantage within the group
            toks, gold = b["tokens"].repeat_interleave(G, 0), b["answer"].repeat_interleave(G, 0)
            if args.s3_tau:                                                     # own stop: correct first, then fewer
                logits, used, logp_lat = model.rollout_stop(toks, args.ticks, args.grpo_sigma, args.s3_tau, sample=True,
                                                            explore=args.s3_explore)
            else:
                logits, logp_lat = model.rollout(toks, args.ticks, args.grpo_sigma)
            logp_all = F.log_softmax(logits, -1)
            a = torch.multinomial(logp_all.exp(), 1).squeeze(1)
            r = (a == gold).float()
            if args.s3_tau:                                                     # price < 1: order stays lexicographic
                r = r - args.s3_price * used.float() / args.ticks
            r = r.view(-1, G)
            adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
            ce = -(adv * (logp_all.gather(1, a[:, None]).squeeze(1) + logp_lat)).mean()
            lat, logits = [], None
            parts_extra = {"reward": r.mean().item(), "groups_live": (r.std(1) > 0).float().mean().item()}
            if args.s3_tau:
                parts_extra["ticks_used"] = used.float().mean().item()
            if args.s3_self:                                                    # every tick toward the same model's
                tl = F.log_softmax(model.tick_logits(b["tokens"], args.ticks), -1)  # answer at the last tick: the
                tgt = (F.softmax(ref(b["tokens"], args.ticks)[0], -1)[:, None] if args.s3_target == "ref"
                       else tl[:, -1:].detach().exp())                          # head learns to doubt where the
                ls = -(tgt * (tl if args.s3_target == "ref" else tl[:, :-1])).sum(-1).mean()  # answer is not yet there
                ce, parts_extra["self_ticks"] = ce + args.s3_self * ls, ls.item()
            if ref is not None:                                                 # KL(model || start) on the clean answer
                lc = F.log_softmax(model(b["tokens"], args.ticks)[0], -1)
                with torch.no_grad():
                    lr_ = F.log_softmax(ref(b["tokens"], args.ticks)[0], -1)
                kl_ref = (lc.exp() * (lc - lr_)).sum(-1).mean()
                ce, parts_extra["kl_ref"] = ce + args.grpo_kl * kl_ref, kl_ref.item()
        else:
            logits, lat, _ = model(b["tokens"], args.ticks)
            ce = F.cross_entropy(logits, b["answer"])
        loss, parts = ce, {("grpo" if args.grpo else "ce"): ce.item()}
        if args.grpo and args.task in ("cipher", "translit"):
            parts.update(rl_parts)
        elif args.grpo:
            parts.update(parts_extra)
        if teacher is not None:                                                # soft answers of the teacher (Hinton
            with torch.no_grad():                                               # 2015): only the final distribution,
                tl = teacher(b["tokens"], teacher_ticks)[0]                     # no per-tick target
            T_ = args.distill_temp
            kl = F.kl_div(F.log_softmax(logits / T_, -1), F.log_softmax(tl / T_, -1),
                          log_target=True, reduction="batchmean") * T_ * T_
            loss, parts["kl"] = args.distill_ce * ce + kl, kl.item()
        if args.jepa and lat and args.jepa_target in ("self", "both"):         # tick t -> EMA's own tick t+1: no
            with torch.no_grad():                                               # external target, the loop learns to
                _, elat, _ = ema(b["tokens"], args.ticks)                       # reach its later state sooner
            head = model.pred_self if args.jepa_target == "both" else model.pred
            pred = head(torch.stack(lat[:-1], 1).flatten(0, 1))
            enc = torch.stack(elat[1:], 1).flatten(0, 1)
            jl = (1 - F.cosine_similarity(pred, enc, dim=-1)).mean()
            loss, parts["jepa_self"] = loss + args.jepa * jl, jl.item()
            parts["tgt_std_self"] = enc.std(0).mean().item()
        if args.jepa and lat and args.jepa_target == "self":
            pass
        elif args.jepa and lat and args.task in ("deriv", "cipher"):           # tick t -> Enc(term / goal)
            Bt, K_, TLn = b["terms"].shape
            sel = b["has"][:, :len(lat)]
            with torch.no_grad():                                               # EMA target encoder (I-JEPA): the
                enc = ema.encode(b["terms"][:, :len(lat)][sel])                 # model's own encoder would collapse
            pred = model.pred(torch.stack(lat, 1)[sel])
            jl = (1 - F.cosine_similarity(pred, enc, dim=-1)).mean()
            loss, parts["jepa"] = loss + args.jepa * jl, jl.item()
            parts["tgt_std"] = enc.std(0).mean().item()                        # collapse shows as this falling to 0
        elif args.jepa and lat:
            terms = []
            for t, h in enumerate(lat):
                m = b["path"][:, t] >= 0                                       # tick t+1 aims at f^(t+1)(s) while t < k
                if m.any():
                    terms.append(1 - F.cosine_similarity(model.pred(h[m]), frozen_emb[b["path"][m, t]], dim=-1).mean())
            if terms:
                jl = torch.stack(terms).mean()
                loss, parts["jepa"] = loss + args.jepa * jl, jl.item()
        if args.sigreg and lat:
            sl = torch.stack([sigreg(h, g=g) for h in lat]).mean()
            loss, parts["sigreg"] = loss + args.sigreg * sl, sl.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ema is not None:
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), model.parameters()):
                    pe.mul_(args.ema).add_(p, alpha=1 - args.ema)
        if step % args.eval_every == 0 or step == args.steps:
            ev = (evaluate_stop(model, args.n, args.ticks, args.s3_tau, args.eval_per_k, args.seed) if args.s3_tau else
                  evaluate_cipher(model, args.ticks, args.eval_per_k) if args.task == "cipher" else
                  evaluate_cipherlm(model, args.ticks, args.eval_per_k) if args.task == "cipherlm" else
                  evaluate_translit(model, args.ticks, args.eval_per_k) if args.task == "translit" else
                  evaluate(model, args.n, args.ticks, args.eval_per_k, args.seed, task=args.task, m=args.m))
            hist.append({"step": step, **{k2: round(v, 4) for k2, v in parts.items()}, "eval": ev,
                         "sec": round(time.time() - t0, 1)})
            print(json.dumps(hist[-1]), flush=True)
    final = (evaluate_stop(model, args.n, args.ticks, args.s3_tau, args.eval_final, args.seed + 1) if args.s3_tau else
             evaluate_cipher(model, args.ticks, args.eval_final) if args.task == "cipher" else
             evaluate_cipherlm(model, args.ticks, args.eval_final) if args.task == "cipherlm" else
             evaluate_translit(model, args.ticks, args.eval_final) if args.task == "translit" else
             evaluate(model, args.n, args.ticks, args.eval_final, args.seed + 1, shuffle=True,
                      task=args.task, m=args.m))
    if args.save_model:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(args.out).with_suffix(".pt"))
    return {"args": vars(args), "n_params": n_params, "history": hist, "final": final,
            "sec": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ticks", type=int, default=0)
    ap.add_argument("--jepa", type=float, default=0.0, help="weight of the per-tick latent target (0 = off)")
    ap.add_argument("--sigreg", type=float, default=0.0, help="weight of SIGReg on tick latents (0 = off)")
    ap.add_argument("--n", type=int, default=10, help="nodes per cycle")
    ap.add_argument("--task", choices=["chase", "affine", "shift", "mix", "store", "deriv", "cs", "cipher", "doubt",
                                       "cipherlm", "translit"],
                    default="chase",
                    help="chase: the table is given; affine: a rule seen through m runs; mix: half each")
    ap.add_argument("--deriv-file", default="", help="deriv: items.jsonl from spatium gen_rewrite.py")
    ap.add_argument("--ema", type=float, default=0.99, help="deriv --jepa: decay of the EMA target encoder")
    ap.add_argument("--jepa-target", choices=["hops", "goal", "self", "both"], default="hops",
                    help="--jepa target: hop t's term (deriv, labelled), the goal term (deriv, no labels), or the EMA "
                         "copy's own next tick (any task, no external target), or both = goal + self (deriv)")
    ap.add_argument("--cipher-train", default="", help="cipher: gen_ciphers rows (train split)")
    ap.add_argument("--cipher-eval", default="", help="cipher: gen_ciphers rows (eval split, other contents)")
    ap.add_argument("--cipher-max-len", type=int, default=320, help="cipher: prompt + answer tokens at most")
    ap.add_argument("--tl-mode", choices=["joint", "plain", "cyr"], default="joint",
                    help="translit: B contents in Cyrillic (joint), or every text in plain / in Cyrillic (the controls)")
    ap.add_argument("--tl-tok", choices=["char", "bpe_en", "bpe_both"], default="char",
                    help="translit: characters, or subword merges learned on plain English only / on both scripts")
    ap.add_argument("--tl-merges", type=int, default=128, help="--tl-tok bpe_*: number of merges")
    ap.add_argument("--eval-only", action="store_true", help="with --init: no training, the final evaluation only")
    ap.add_argument("--clm-plain", type=float, default=0.5, help="cipherlm: share of plain texts in training (0 = cipher only)")
    ap.add_argument("--cipher-control", action="store_true",
                    help="cipher: each answer paired with another row's prompt (train and eval): the language prior alone")
    ap.add_argument("--anon", action="store_true", help="deriv: rename symbols per item within their arity class")
    ap.add_argument("--deriv-answer", choices=["truth", "last_rule"], default="truth")
    ap.add_argument("--K", type=int, default=256, help="store: number of fixed tables held in the weights")
    ap.add_argument("--fact-seed", type=int, default=0, help="store: seed of the tables (shared by the arms)")
    ap.add_argument("--m", type=int, default=4, help="runs of the affine cipher in context")
    ap.add_argument("--curriculum", action="store_true", help="hop counts 1..k_max, k_max rising to KMAX by 3/4 of the steps")
    ap.add_argument("--warm", type=int, default=0, help="first steps on one hop only (store first), then the curriculum")
    ap.add_argument("--train-ks", default="", help="hop counts to train on, e.g. 1,2 (default 1..KMAX)")
    ap.add_argument("--distill", default="", help="teacher .pt (its .json beside it gives size and ticks): the student "
                                                       "learns the teacher's final answer distribution (chase-like tasks)")
    ap.add_argument("--distill-temp", type=float, default=2.0, help="--distill: softmax temperature")
    ap.add_argument("--distill-ce", type=float, default=0.0, help="--distill: weight of the hard-label CE beside the KL")
    ap.add_argument("--init", default="", help="start from this trained toy's weights (.pt; same size flags)")
    ap.add_argument("--grpo", type=int, default=0, help="stage 1k: group size G of rollouts per prompt (0 = off); "
                                                        "reward = exact answer, no labels inside the loop")
    ap.add_argument("--grpo-sigma", type=float, default=0.0,
                    help="--grpo: std of the Gaussian policy on each tick latent (0 = only the answer is sampled)")
    ap.add_argument("--grpo-w", type=float, default=1.0, help="--grpo on cipher: weight of the policy term beside CE")
    ap.add_argument("--grpo-kl", type=float, default=0.0, help="--grpo: weight of KL to the --init model")
    ap.add_argument("--s3-tau", type=float, default=0.0,
                    help="stage 1l (chase, with --grpo): stop at the first tick whose answer has max p >= tau, no stop "
                         "head; reward = correct, then fewer ticks (0 = off)")
    ap.add_argument("--s3-explore", type=float, default=0.0,
                    help="--s3-tau: training stop probability = (1 - explore) * confidence")
    ap.add_argument("--s3-self", type=float, default=0.0,
                    help="--s3-tau: weight of CE from each tick's answer to the same model's last-tick answer")
    ap.add_argument("--s3-target", choices=["last", "ref"], default="last",
                    help="--s3-self target: the model's own last tick (drifts), or the frozen --init model at M = ticks")
    ap.add_argument("--s3-ramp", type=float, default=0.0,
                    help="--s3-self without --grpo: tick t's imitation weight (t / ticks)^ramp (0 = flat)")
    ap.add_argument("--s3-price", type=float, default=0.1, help="--s3-tau: price of all ticks used, < 1 (lexicographic)")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-per-k", type=int, default=256)
    ap.add_argument("--eval-final", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu or cuda; batches are built on the CPU and moved")
    ap.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    res = train(args)
    from experiments._common.results import save_result
    fin = res["final"]
    save_result(args.out, res, experiment="closed_loop_toy", hypothesis=["H25", "HZLK"],
                summary=({k_: f"char {v['char_acc']:.3f} exact {v['exact']:.3f}" for k_, v in fin.items()}
                         if args.task == "cipher" else
                         {key: (json.dumps(v) if isinstance(v, (dict, list)) or v is None else f"{v:.3f}") for key, v in fin.items()}
                         if args.task in ("doubt", "cipherlm", "translit") or args.s3_tau else
                         {(key if "capacity" in key else f"acc {key} k=1..{KMAX}"):
                          (json.dumps(fin[key]) if "capacity" in key
                           else " ".join(f"{fin[key][k]:.2f}" for k in range(1, KMAX + 1))) for key in fin}),
                script="experiments/A0_state_probe/closed_loop_toy.py")
    print(json.dumps({"final": fin, "n_params": res["n_params"], "sec": res["sec"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
