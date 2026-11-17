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
import json
import copy
import math
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


@torch.no_grad()
def store_capacity(model, M: int) -> dict:
    """One-hop accuracy over every stored edge and the bits it carries (Fano bound, 15 alternatives per edge)."""
    K, nxt = STORE["K"], STORE["next"]
    tab = torch.arange(K).repeat_interleave(N_NODES)
    s = torch.arange(N_NODES).repeat(K)
    toks = torch.stack([VOCAB + tab // ID_BASE, VOCAB + tab % ID_BASE, torch.full_like(s, Q), s,
                        torch.full_like(s, HOP0)], 1)
    hit = torch.cat([model(toks[i:i + 4096], M)[0].argmax(-1) == nxt[tab[i:i + 4096], s[i:i + 4096]]
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
        toks[b, T - len(tk):] = torch.tensor(tk)
        for t, term in enumerate(tm[:KMAX]):
            terms[b, t, TL - len(term):] = torch.tensor(term); has[b, t] = True
    return {"tokens": toks, "answer": torch.tensor([it[1] for it in items]), "k": torch.tensor([it[2] for it in items]),
            "terms": terms, "has": has}


def sample(task: str, B: int, n: int, m: int, ks: list[int], g: torch.Generator) -> dict:
    """chase (tables given whole), affine (rule seen through runs) or mix (half each, one length)."""
    if task == "chase":
        return make_batch(B, n, ks, g)
    if task == "store":
        return make_store(B, ks, g)
    if task == "deriv":
        return make_deriv(B, ks, g)
    if task == "cs":                                                            # the step from context, facts by one hop
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
        z = torch.zeros(B, self.C)
        return {"S": [torch.zeros(B, self.H, self.C // self.H, self.C // self.H) for _ in range(self.L)],
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


# ---------------------------------------------------------------- losses

def sigreg(z: torch.Tensor, n_slices: int = 64, g: torch.Generator | None = None) -> torch.Tensor:
    """Epps-Pulley statistic of random 1-D projections of z (B, C) against N(0, 1), averaged (LeJEPA, Alg. 1)."""
    A = torch.randn(z.shape[1], n_slices, generator=g)
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
                    b = make_deriv(min(per_k, 256), [k], g, split)
                    perm = torch.randperm(len(b["answer"]), generator=g) if shuf else None
                    acc[k] = round((model(b["tokens"], M, perm=perm)[0].argmax(-1) == b["answer"]).float().mean().item(), 4)
                out[key] = acc
        return out
    if task == "cs":                                                            # transfer: k-hop over stored facts never
        out.update({f"chase_{k_}": v for k_, v in evaluate(model, n, ticks, per_k, seed, task="chase").items()})
        out.update({f"store_{k_}": v for k_, v in evaluate(model, n, ticks, per_k, seed, task="store").items()})
        return out                                                              # trained beyond one hop
    if task == "store":                                                         # held-out multi-hop starts; capacity
        for M in sorted({0, ticks}):
            for key, held in ((f"M{M}", True), (f"M{M}_seen", False)):
                acc = {}
                for k in range(1, KMAX + 1):
                    b = make_store(per_k, [k], g, held=held)
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
                b = sample(fam, per_k, n, m, [k], g)
                perm = torch.randperm(per_k, generator=g) if shuf else None
                logits, _, _ = model(b["tokens"], M, perm=perm)
                acc[k] = round((logits.argmax(-1) == b["answer"]).float().mean().item(), 4)
            out[key] = acc
    return out


def train(args) -> dict:
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    if args.task in ("store", "cs"):
        set_store(args.K, args.fact_seed)
    if args.task == "deriv":
        load_deriv(args.deriv_file, args.deriv_answer, args.jepa_target)
    extra = {"store": ID_BASE, "cs": ID_BASE, "deriv": len(DERIV_SYMS)}.get(args.task, 0)
    model = ToyRWKV7(args.width, args.heads, args.layers, VOCAB + extra)
    frozen_emb = model.emb.weight.detach().clone()                              # fixed teacher for the JEPA target
    if args.jepa_target == "both":                                              # its own head for the self target
        model.pred_self = nn.Linear(args.width, args.width, bias=False)
    ema = (copy.deepcopy(model).requires_grad_(False)
           if args.jepa and (args.task == "deriv" or args.jepa_target == "self") else None)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    ks = [int(x) for x in args.train_ks.split(',')] if args.train_ks else list(range(1, KMAX + 1))
    hist, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        if args.curriculum:                                                     # k_max grows 1 -> KMAX over 3/4 of the steps
            cs = max(0, step - 1 - args.warm)                                   # after the one-hop warm phase
            kmax = min(KMAX, 1 + cs * KMAX // max(1, int(0.75 * (args.steps - args.warm))))
            ks = list(range(1, kmax + 1)) if step > args.warm else [1]
        b = sample(args.task, args.batch, args.n, args.m, ks, g)
        logits, lat, _ = model(b["tokens"], args.ticks)
        ce = F.cross_entropy(logits, b["answer"])
        loss, parts = ce, {"ce": ce.item()}
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
        elif args.jepa and lat and args.task == "deriv":                       # tick t -> Enc(term after hop t)
            Bt, K_, TL = b["terms"].shape
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
            ev = evaluate(model, args.n, args.ticks, args.eval_per_k, args.seed, task=args.task, m=args.m)
            hist.append({"step": step, **{k2: round(v, 4) for k2, v in parts.items()}, "eval": ev,
                         "sec": round(time.time() - t0, 1)})
            print(json.dumps(hist[-1]), flush=True)
    final = evaluate(model, args.n, args.ticks, args.eval_final, args.seed + 1, shuffle=True,
                     task=args.task, m=args.m)
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
    ap.add_argument("--task", choices=["chase", "affine", "shift", "mix", "store", "deriv", "cs"], default="chase",
                    help="chase: the table is given; affine: a rule seen through m runs; mix: half each")
    ap.add_argument("--deriv-file", default="", help="deriv: items.jsonl from spatium gen_rewrite.py")
    ap.add_argument("--ema", type=float, default=0.99, help="deriv --jepa: decay of the EMA target encoder")
    ap.add_argument("--jepa-target", choices=["hops", "goal", "self", "both"], default="hops",
                    help="--jepa target: hop t's term (deriv, labelled), the goal term (deriv, no labels), or the EMA "
                         "copy's own next tick (any task, no external target), or both = goal + self (deriv)")
    ap.add_argument("--deriv-answer", choices=["truth", "last_rule"], default="truth")
    ap.add_argument("--K", type=int, default=256, help="store: number of fixed tables held in the weights")
    ap.add_argument("--fact-seed", type=int, default=0, help="store: seed of the tables (shared by the arms)")
    ap.add_argument("--m", type=int, default=4, help="runs of the affine cipher in context")
    ap.add_argument("--curriculum", action="store_true", help="hop counts 1..k_max, k_max rising to KMAX by 3/4 of the steps")
    ap.add_argument("--warm", type=int, default=0, help="first steps on one hop only (store first), then the curriculum")
    ap.add_argument("--train-ks", default="", help="hop counts to train on, e.g. 1,2 (default 1..KMAX)")
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
    ap.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    res = train(args)
    from experiments._common.results import save_result
    fin = res["final"]
    save_result(args.out, res, experiment="closed_loop_toy", hypothesis=["H25", "HZLK"],
                summary={(key if "capacity" in key else f"acc {key} k=1..{KMAX}"):
                         (json.dumps(fin[key]) if "capacity" in key
                          else " ".join(f"{fin[key][k]:.2f}" for k in range(1, KMAX + 1))) for key in fin},
                script="experiments/A0_state_probe/closed_loop_toy.py")
    print(json.dumps({"final": fin, "n_params": res["n_params"], "sec": res["sec"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
