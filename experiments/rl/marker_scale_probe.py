#!/usr/bin/env python3
"""marker_scale_probe.py — is a ThinkChain phase marker a vector this model
has ever seen?

The M-loop content decoder (2026-08-19, `_diag_think_content.py`, deleted after
use) found that a clean G1i checkpoint's invisible loop tokens were only ever
chat-template scaffolding, never task content, and read the cause as "there is
no ground truth for what an M-loop token should be". That is true, and it is
not the only thing wrong. This probe checks the input side instead: what the
marker looks like NEXT TO the embedding distribution it is injected into.

`ThinkChain.__init__` draws `randn(n_phases + 1, n_embd) * 0.02`. At n_embd
2560 that is a norm of 0.02 * sqrt(2560) = 1.012 by construction. G1i's token
embeddings have median norm 0.376 and max 0.642 across the whole 65536-token
vocabulary. So the marker is 2.7x the median token, 1.6x the largest, and NO
token in the vocabulary is as large as it. The constant 0.02 is a generic
small-init std; nothing about it refers to the distribution the vector has to
live in.

Two things this probe reports, separately, because they have separate fixes:

  scale      — where the marker sits in the embedding norm distribution.
  training   — whether training moved it. Per-coordinate std against the init
               constant is the tell: a marker still at 0.0200 after N steps is
               a vector the gradient barely touched.

Neither is an argument against markers. "Markers do not carry enough
information" is the opposite of what a near-orthogonal pair of vectors does.
The argument is that the signal is delivered in the wrong units, and that a
state written by an out-of-distribution input decodes to whatever the model
falls back on — which is what the content decoder saw.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from experiments._common.results import save_result


def expected_mode_norms(emb: torch.Tensor) -> list[dict]:
    """How loud `feed_mode="expected"` is, as a function of confidence.

    `softmax(logits) @ emb.weight` is a CONVEX COMBINATION of embeddings, so it
    can never leave their hull -- its norm is bounded above by the largest
    token (0.642 on G1i) and collapses toward the mean embedding (0.033, 11x
    smaller than a median token) as the distribution flattens.

    That makes the feed strength a function of the model's confidence, which is
    not obviously what a think-loop wants: the early steps are exactly when the
    model is least sure, so the vector it feeds itself is weakest precisely
    where the loop is supposed to do the most work. The marker has the opposite
    failure -- a fixed 2.7x-median vector that no confidence level produces.
    """
    q = emb.float().norm(dim=1)
    mean_e = emb.float().mean(0)
    i = 0
    rows = []
    for p1 in (0.99, 0.9, 0.5, 0.2, 0.05, 0.01):
        v = p1 * emb[i].float() + (1 - p1) * mean_e
        rows.append({"top1_prob": p1, "norm": v.norm().item(),
                     "over_median_token": (v.norm() / q.median()).item()})
    return rows


def analyse(chain: torch.Tensor, emb: torch.Tensor, init_std: float) -> dict:
    c = chain.float()
    n_phase, n_embd = c.shape
    q = emb.float().norm(dim=1)
    expected_norm = init_std * n_embd ** 0.5
    phases = []
    for i in range(n_phase):
        v = c[i]
        sims = F.cosine_similarity(v.unsqueeze(0), emb.float(), dim=1)
        phases.append({
            "phase": i,
            "norm": v.norm().item(),
            "per_coord_std": v.std().item(),
            "norm_over_median_token": (v.norm() / q.median()).item(),
            "norm_over_max_token": (v.norm() / q.max()).item(),
            "frac_tokens_larger": (q > v.norm()).float().mean().item(),
            "max_cos_to_any_token": sims.max().item(),
        })
    pair_cos = [[F.cosine_similarity(c[i], c[j], dim=0).item()
                 for j in range(n_phase)] for i in range(n_phase)]
    return {
        "n_phase": n_phase, "n_embd": n_embd,
        "init_std": init_std, "expected_init_norm": expected_norm,
        "expected_random_pair_cos": n_embd ** -0.5,
        "token_norm": {"p1": q.quantile(0.01).item(), "median": q.median().item(),
                       "p99": q.quantile(0.99).item(), "max": q.max().item()},
        "phases": phases, "pair_cosine": pair_cos,
        "mean_embedding_norm": emb.float().mean(0).norm().item(),
        "expected_mode_by_confidence": expected_mode_norms(emb),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="a checkpoint dir containing meta.pt with mlp_delta.chain")
    ap.add_argument("--model", type=Path, required=True,
                    help="the .pth the embeddings come from")
    ap.add_argument("--init-std", type=float, default=0.02,
                    help="ThinkChain's init std, for the 'did it train' read")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    meta = torch.load(args.checkpoint / "meta.pt", map_location="cpu")
    chain = meta["mlp_delta"]["chain"]
    emb = torch.load(args.model, map_location="cpu", mmap=True)["emb.weight"]
    d = analyse(chain, emb, args.init_std)

    t = d["token_norm"]
    print(f"token embedding norm: p1 {t['p1']:.4f}  median {t['median']:.4f}  "
          f"p99 {t['p99']:.4f}  max {t['max']:.4f}")
    print(f"init would give norm {d['expected_init_norm']:.4f}, per-coord std "
          f"{d['init_std']:.4f}, random pair |cos| ~ {d['expected_random_pair_cos']:.4f}")
    for p in d["phases"]:
        print(f"  phase{p['phase']}: norm {p['norm']:.4f} "
              f"({p['norm_over_median_token']:.2f}x median token, "
              f"{p['norm_over_max_token']:.2f}x the largest); "
              f"per-coord std {p['per_coord_std']:.5f}; "
              f"tokens larger: {p['frac_tokens_larger']:.6f}; "
              f"max cos to any token {p['max_cos_to_any_token']:.4f}")
    print(f"expected-mode feed (softmax(logits) @ emb), mean-embedding norm "
          f"{d['mean_embedding_norm']:.4f}:")
    for r in d["expected_mode_by_confidence"]:
        print(f"  top-1 prob {r['top1_prob']:5.2f} -> norm {r['norm']:.4f} "
              f"({r['over_median_token']:.2f}x median token)")
    print("pairwise cosine between phase markers:")
    for row in d["pair_cosine"]:
        print("   " + " ".join(f"{v:+.4f}" for v in row))

    if args.out is not None:
        p0 = d["phases"][0]
        save_result(
            args.out, d, experiment="marker_scale", hypothesis=["H25"],
            model=str(args.model),
            summary={
                "marker norm vs token embeddings":
                    f"{p0['norm']:.4f} = {p0['norm_over_median_token']:.2f}x median, "
                    f"{p0['norm_over_max_token']:.2f}x max; "
                    f"{p0['frac_tokens_larger']:.6f} of vocab larger",
                "did training move it":
                    f"per-coord std {p0['per_coord_std']:.5f} vs init {d['init_std']:.4f}",
                "expected-mode feed strength tracks confidence":
                    "; ".join(f"p1={r['top1_prob']}->{r['over_median_token']:.2f}x"
                              for r in d["expected_mode_by_confidence"]),
                "phase markers distinguishable":
                    f"pairwise cos {d['pair_cosine'][0][1]:+.4f} vs "
                    f"{d['expected_random_pair_cos']:.4f} expected for random draws",
            },
            script=str(Path(__file__).relative_to(_REPO_ROOT)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
