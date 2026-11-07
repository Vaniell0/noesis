#!/usr/bin/env python3
"""latent_geometry_probe.py — where does the model's own latent lie, and when does it settle?

C1 feeds the last block's output (before ln_out/head) back as the next input, Coconut-style
(owner 2026-10-21: no vocabulary rows in the loop). Block 0 applies ln0 to its input, so the
norm of a fed vector never reaches the model — only its direction after ln0. Per tick of the
loop, against the token embeddings:

    nn_cos      cosine of ln0(h) to its nearest ln0(E) row (a real token gives 1.0)
    emb_frac    share of ln0(h)'s energy inside the top-K principal subspace of ln0(E)
                (baselines printed for token rows and for random directions)
    argmax_cos  cosine to ln0(E[argmax logits]) - is it the word the model would say?
    tick_cos    cosine to the previous tick's ln0(h) - the owner's attractor: 1 = fixed point
    cross_cos   mean pairwise cosine across prompts at the same tick - 1 = one constant for
                every task (the marker failure). UNCENTRED: a large component shared by all prompts
                inflates it, so it is reported next to the centred measures below (added 2026-10-22
                after an outside review: "0.99 without mean-centring comes for free"):
    (a centred pairwise cosine is NOT reported: centred vectors sum to zero, so its mean is ~-1/(n-1)
    by construction whatever the geometry)
    pc1_share   share of the across-prompt variance in the first principal component
    eff_rank    participation ratio of the across-prompt spectrum (1 = all prompts on one line)
    step_c      per-prompt tick-to-tick displacement of the centred latent, relative to the spread
                across prompts (0 = fixed point relative to how different the prompts are)

Baseline G1i 2.9B untrained (2026-10-22): per-prompt fixed point by tick 3 (tick_cos 0.98),
cross_cos 0.88 -> ~0 by tick 4. A checkpoint that learned to take apart what it knows should
settle later, with ticks that differ. Also checks that prefill through the embeds path gives
the same logits as forward_stateful.

    python experiments/run.py --backend peft --model M --tests latent_geometry --lg-items F ...
    python experiments/rl/latent_geometry_probe.py --model M --lg-items F --out R.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._common import registry  # noqa: E402
from experiments._common.runtime import limit_threads, progress  # noqa: E402


def load_prompts(files, per_file: int) -> list:
    """Every per_file-th row's 'prompt' from each jsonl, strided so families stay mixed."""
    prompts = []
    for f in files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        step = max(1, len(rows) // per_file)
        prompts += [r["prompt"] for r in rows[::step][:per_file]]
    return prompts


def prefill_latent(loaded, ids: list):
    """Prompt through the embeds path. Returns (logits at the last position [1, V], state,
    own latent h at the last position [1, 1, D])."""
    E = loaded.embedding_weight
    logits, st, h = loaded.forward_stateful_embeds(E[torch.tensor([ids])], loaded.new_state(batch=1),
                                                   return_hidden=True)
    return logits[:, -1], st, h[:, -1:]


def _add_args(ap):
    ap.add_argument("--lg-items", type=Path, action="append",
                    help="jsonl with a 'prompt' field; repeatable")
    ap.add_argument("--lg-per-file", type=int, default=12)
    ap.add_argument("--lg-ticks", type=int, default=8, help="H25: useful ticks ~4-8, little past ~4")
    ap.add_argument("--lg-pca-k", type=int, default=256)


@registry.probe("latent_geometry", hypothesis=["HZLK"], backend="peft", add_args=_add_args,
                description="own-latent loop: per-tick direction vs embeddings, fixed point (attractor), cross-prompt sameness")
def run(loaded, tok, args) -> dict:
    torch.manual_seed(0)
    prompts = load_prompts(args.lg_items, args.lg_per_file)
    model = loaded.model
    E = loaded.embedding_weight.float()
    ln0 = lambda x: loaded.feed_view(x).float()                # RWKV: ln0; Qwen: identity (what block 0 sees)
    T = args.lg_ticks

    with torch.no_grad():
        En = F.normalize(ln0(E), dim=-1)                       # [V, D]
        mu = En.mean(0)
        _, _, Vt = torch.linalg.svd((En - mu)[::4], full_matrices=False)
        P = Vt[:args.lg_pca_k]                                 # [K, D]

        def emb_frac(u):                                       # u [N, D], unit rows
            c = u - mu
            return ((c @ P.T).pow(2).sum(-1) / c.pow(2).sum(-1).clamp_min(1e-12))

        rnd = F.normalize(torch.randn(512, E.shape[1], device=E.device, dtype=E.dtype), dim=-1)
        tok_rows = En[torch.randint(0, E.shape[0], (512,))]
        baseline = {"emb_frac_token_rows": float(emb_frac(tok_rows).mean()),
                    "emb_frac_random": float(emb_frac(rnd).mean()),
                    "nn_cos_random": float((rnd @ En.T).max(-1).values.mean())}

        per_tick = [[] for _ in range(T + 1)]                  # unit ln0(h) per prompt
        stats = [{"nn_cos": [], "emb_frac": [], "argmax_cos": [], "tick_cos": []} for _ in range(T + 1)]
        max_dlogit, norms = 0.0, []
        t0 = time.time()
        for i, prompt in enumerate(prompts):
            ids = tok.encode(loaded.tick_frame(prompt)[0] if hasattr(loaded, "tick_frame") else prompt)
            ref, _ = loaded.forward_stateful(torch.tensor([ids]), loaded.new_state(batch=1))
            lg, st, h = prefill_latent(loaded, ids)
            max_dlogit = max(max_dlogit, float((ref[0, -1] - lg[0]).abs().max()))
            prev = None
            for t in range(T + 1):
                u = F.normalize(ln0(h[0, 0]), dim=-1)
                norms.append(float(h.norm() / E[ids].norm(dim=-1).median()))
                per_tick[t].append(u)
                s = stats[t]
                s["nn_cos"].append(float((En @ u).max()))
                s["emb_frac"].append(float(emb_frac(u[None])[0]))
                s["argmax_cos"].append(float(En[int(lg[0].argmax())] @ u))
                s["tick_cos"].append(float(prev @ u) if prev is not None else float("nan"))
                prev = u
                if t < T:
                    lg, st, h = loaded.forward_stateful_embeds(h, st, return_hidden=True)
                    lg = lg[:, -1]
            progress(f"[latent_geometry] {i + 1}/{len(prompts)} {time.time() - t0:.0f}s")

        rows = []
        for t in range(T + 1):
            U = torch.stack(per_tick[t])
            n = len(U)
            r = {"tick": t, "cross_cos": round(float(((U @ U.T).sum() - n) / (n * (n - 1))), 3)}
            for k, v in stats[t].items():
                vv = torch.tensor(v)
                vv = vv[~vv.isnan()]
                r[k] = round(float(vv.mean()), 3) if len(vv) else None
            C = U - U.mean(0, keepdim=True)
            ev = torch.linalg.svdvals(C).pow(2)
            r["pc1_share"] = round(float(ev[0] / ev.sum().clamp_min(1e-12)), 3)
            r["eff_rank"] = round(float(ev.sum() ** 2 / ev.pow(2).sum().clamp_min(1e-12)), 2)
            if t > 0:
                P = torch.stack(per_tick[t - 1])
                Pc = P - P.mean(0, keepdim=True)
                r["step_c"] = round(float((C - Pc).norm(dim=-1).mean() / C.norm(dim=-1).mean().clamp_min(1e-12)), 3)
            rows.append(r)
    settle = next((r["tick"] for r in rows if r["tick_cos"] is not None and r["tick_cos"] >= 0.98), None)
    baseline["norm_hidden_vs_emb"] = round(sum(norms) / len(norms), 2)
    return {"model": args.model, "n_prompts": len(prompts), "items": [str(f) for f in args.lg_items],
            "embeds_path_max_abs_dlogit": round(max_dlogit, 5), "pca_k": args.lg_pca_k,
            "baseline": {k: round(v, 3) for k, v in baseline.items()},
            "settle_tick_cos098": settle, "ticks": rows,
            "note": "tick 0 = last prompt position; ticks >= 1 = hidden fed back",
            "_summary": {"settle tick (tick_cos>=0.98)": str(settle),
                         "cross_cos last": str(rows[-1]["cross_cos"]), "eff_rank last": str(rows[-1]["eff_rank"]),
                         "nn_cos tick0/last": f"{rows[0]['nn_cos']} / {rows[-1]['nn_cos']}"}}


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
    save_result(args.out, out, experiment="latent_geometry", hypothesis=["HZLK"], model=args.model,
                script="experiments/rl/latent_geometry_probe.py")
    print(json.dumps({k: v for k, v in out.items() if k != "_summary"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
