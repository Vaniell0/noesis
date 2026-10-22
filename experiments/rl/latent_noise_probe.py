#!/usr/bin/env python3
"""latent_noise_probe.py — is there a noise window where RL on the own latent can start?

The old GRPO had log pi over answer tokens only and a deterministic loop, so the loop was never
part of the policy (owner 2026-10-22: that was the problem). With the own latent the loop's
stochasticity is a noisy feed z = h + noise; its log pi is a Gaussian over the noise
coordinates. Two things can kill that before training starts:

  - the attractor swallows the noise (untrained loop: per-prompt fixed point by tick 3), so all
    rollouts of a prompt give the same answer -> group advantage 0 again;
  - the noise wrecks the answer before rollouts start to differ.

So, untrained, per (M, sigma, k): R rollouts per item, each fed latent perturbed by
sigma * |h| * (eps @ B) / sqrt(k), B = k orthonormal directions (top principal directions of
the latents of every prompt in the file; k <= that count; k = D means isotropic). A small k also keeps the log-ratio of a
Gaussian policy from summing D terms per tick. Read out the answer margin
log p(YES) - log p(NO) at the next position after the last tick.

    acc        sign of the margin agrees with the label (all rollouts pooled)
    flip_frac  items whose rollouts disagree in sign - the room GRPO needs (> 0)
    margin_sd  spread of the margin across rollouts, mean over items

The window = (sigma, k) with flip_frac > 0 and acc not far below sigma = 0.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._common import registry  # noqa: E402
from experiments._common.runtime import limit_threads, progress  # noqa: E402
from experiments.rl.latent_geometry_probe import prefill_latent  # noqa: E402


def _floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _add_args(ap):
    ap.add_argument("--ln-items", type=Path, help="jsonl with prompt, conts {YES, NO}, y (head_ablation_items.py)")
    ap.add_argument("--ln-kind", default="binding", help="keep only items of this 'kind' ('' = all)")
    ap.add_argument("--ln-limit", type=int, default=20)
    ap.add_argument("--ln-m", default="1,2")
    ap.add_argument("--ln-sigma", default="0,0.02,0.05,0.1,0.2,0.5")
    ap.add_argument("--ln-k", default="16,64,0", help="0 = isotropic (k = D)")
    ap.add_argument("--ln-rollouts", type=int, default=8)


@registry.probe("latent_noise", hypothesis=["HZLK"], backend="peft", add_args=_add_args,
                description="noisy own-latent feed: answer accuracy vs rollout disagreement per (M, sigma, k) - the RL start window")
def run(loaded, tok, args) -> dict:
    g = torch.Generator().manual_seed(0)
    rows = [json.loads(l) for l in open(args.ln_items) if l.strip()]
    basis_prompts = [r["prompt"] for r in rows]           # basis from every row, not just the scored ones
    if args.ln_kind:
        rows = [r for r in rows if r.get("kind") == args.ln_kind]
    rows = rows[:args.ln_limit]
    yes = tok.encode(rows[0]["conts"]["YES"])[0]
    no = tok.encode(rows[0]["conts"]["NO"])[0]
    R = args.ln_rollouts
    ms, sigmas, ks = _ints(args.ln_m), _floats(args.ln_sigma), _ints(args.ln_k)

    def margin(lg):                                            # lg [B, V]
        lp = torch.log_softmax(lg.float(), dim=-1)
        return lp[:, yes] - lp[:, no]

    with torch.no_grad():
        pre = []
        for r in rows:
            lg, st, h = prefill_latent(loaded, tok.encode(r["prompt"]))
            pre.append((lg, st, h))
        H = torch.cat([prefill_latent(loaded, tok.encode(p))[2][0] for p in basis_prompts]).float()  # [n_basis, D]
        D = H.shape[1]
        _, _, Vt = torch.linalg.svd(H - H.mean(0), full_matrices=False)
        bases = {k: (None if k == 0 else Vt[:k]) for k in ks}   # [k, D] or isotropic
        y = torch.tensor([r["y"] for r in rows])
        m0 = torch.stack([margin(lg)[0] for lg, _, _ in pre])
        out = {"M0": {"acc": round(float(((m0 > 0).long() == y).float().mean()), 3)}}
        cells = []
        t0, done, total = time.time(), 0, len(ms) * len(sigmas) * len(ks)
        for M in ms:
            for k in ks:
                B = bases[k]
                kk = D if B is None else B.shape[0]
                for sig in sigmas:
                    if sig == 0 and k != ks[0]:
                        continue                               # sigma 0 is the same for every k
                    marg = torch.zeros(len(rows), R)
                    for i, (lg, st, h) in enumerate(pre):
                        s = type(st)(st.shift.repeat(1, 1, R, 1), st.wkv.repeat(1, R, 1, 1, 1))
                        hh = h.repeat(R, 1, 1)
                        for _ in range(M):
                            eps = torch.randn(R, kk, generator=g)
                            noise = eps if B is None else eps @ B
                            z = hh + sig * hh.norm(dim=-1, keepdim=True) * noise[:, None, :] / kk ** 0.5
                            lgt, s, hh = loaded.forward_stateful_embeds(z.to(hh.dtype), s, return_hidden=True)
                        marg[i] = margin(lgt[:, -1])
                    sign = (marg > 0)
                    cells.append({"M": M, "k": kk, "sigma": sig,
                                  "acc": round(float((sign.long() == y[:, None]).float().mean()), 3),
                                  "flip_frac": round(float((sign.any(1) & ~sign.all(1)).float().mean()), 3),
                                  "margin_sd": round(float(marg.std(1).mean()), 3),
                                  "margin_mean": round(float(marg.mean()), 3)})
                    done += 1
                    progress(f"[latent_noise] M={M} k={kk} sigma={sig}: {cells[-1]} ({time.time() - t0:.0f}s)")
    window = [c for c in cells if c["flip_frac"] > 0 and c["acc"] >= out["M0"]["acc"] - 0.1]
    out.update({"model": args.model, "items": str(args.ln_items), "n": len(rows), "rollouts": R,
                "cells": cells, "window": window,
                "_summary": {"M0 acc": str(out["M0"]["acc"]),
                             "window cells (flip>0, acc>=M0-0.1)": str(len(window))}})
    return out


def main() -> int:
    from experiments._common.results import save_result
    from experiments.rl.loader import load_rwkv7
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    _add_args(ap)
    args = ap.parse_args()
    limit_threads(args.threads)
    loaded = load_rwkv7(args.model, device="cpu", dtype=torch.float32, backend="peft", ctx_len=2048)
    for p in loaded.model.parameters():
        p.requires_grad_(False)
    out = run(loaded, loaded.tokenizer, args)
    save_result(args.out, out, experiment="latent_noise", hypothesis=["HZLK"], model=args.model,
                script="experiments/rl/latent_noise_probe.py")
    print(json.dumps({k: v for k, v in out.items() if k != "_summary"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
