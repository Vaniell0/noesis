"""Combine step 9 corpus: RFC QA + 10% action-chain anti-forgetting mix.

Usage:
    training/.venv/bin/python training/scripts/combine_step9_corpus.py \\
        --rfc     training/tokenised/step9_rfc_train.pt \\
        --action  training/tokenised/action_chains_dsl_step8_train.pt \\
        --action-fraction 0.10 \\
        --out     training/tokenised/step9_combined_train.pt \\
        --seed    42

Concatenates rollout-level (not token-level), shuffles, saves.
Supports optional --selfcot argument for future self-CoT layer.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch


def _load_rollouts(pt_path: str) -> list[dict]:
    """Split a .pt blob into per-rollout dicts."""
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    ids   = blob["ids"]
    lm    = blob["loss_mask"]
    sm    = blob.get("state_mask", torch.zeros_like(lm))
    starts = blob["starts"].tolist()
    rollouts = []
    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i + 1]
        rollouts.append({
            "ids":        ids[s:e],
            "loss_mask":  lm[s:e],
            "state_mask": sm[s:e],
        })
    return rollouts


def _pack_rollouts(rollouts: list[dict]) -> dict:
    all_ids, all_lm, all_sm, starts = [], [], [], [0]
    for r in rollouts:
        n = r["ids"].numel()
        all_ids.append(r["ids"])
        all_lm.append(r["loss_mask"])
        all_sm.append(r["state_mask"])
        starts.append(starts[-1] + n)
    return {
        "ids":        torch.cat(all_ids),
        "loss_mask":  torch.cat(all_lm),
        "state_mask": torch.cat(all_sm),
        "starts":     torch.tensor(starts, dtype=torch.long),
        "vocab":      "rwkv_vocab_v20230424",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfc",              required=True)
    ap.add_argument("--action",           required=True)
    ap.add_argument("--selfcot",          default=None, help="Optional self-CoT .pt")
    ap.add_argument("--action-fraction",  type=float, default=0.10)
    ap.add_argument("--selfcot-fraction", type=float, default=0.0)
    ap.add_argument("--out",              required=True)
    ap.add_argument("--seed",             type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    rfc_rolls    = _load_rollouts(args.rfc)
    action_rolls = _load_rollouts(args.action)

    rfc_tokens = sum(r["ids"].numel() for r in rfc_rolls)
    # token-budget approach: sample action rollouts until we hit target fraction
    # If individual rollouts exceed the budget, truncate them.
    token_budget = int(rfc_tokens * args.action_fraction / (1.0 - args.action_fraction))
    rng.shuffle(action_rolls)
    sampled_action, used_tokens = [], 0
    for r in action_rolls:
        n = r["ids"].numel()
        remaining = token_budget - used_tokens
        if remaining <= 0:
            break
        if n > remaining:
            # Truncate rollout to remaining budget
            r = {k: v[:remaining] for k, v in r.items()}
            n = remaining
        sampled_action.append(r)
        used_tokens += n

    if not sampled_action:
        print("[combine] WARNING: action budget too small for any rollout — RFC-only corpus")

    combined = rfc_rolls + sampled_action

    if args.selfcot:
        selfcot_rolls = _load_rollouts(args.selfcot)
        n_sc = max(1, int(n_rfc * args.selfcot_fraction / max(1e-6, 1.0 - args.selfcot_fraction)))
        n_sc = min(n_sc, len(selfcot_rolls))
        combined += rng.sample(selfcot_rolls, n_sc)
        print(f"[combine] selfcot: {n_sc}/{len(selfcot_rolls)}")

    rng.shuffle(combined)

    print(f"[combine] rfc={len(rfc_rolls)} ({rfc_tokens} tok)  "
          f"action={len(sampled_action)}/{len(action_rolls)} ({used_tokens} tok)  "
          f"total={len(combined)}")

    blob = _pack_rollouts(combined)
    n_tok = blob["ids"].numel()
    n_sup = int(blob["loss_mask"].sum())
    n_st  = int(blob["state_mask"].sum())
    print(f"[combine] {n_tok} tokens, {n_sup} CE-supervised ({100*n_sup//max(n_tok,1)}%), "
          f"{n_st} state-supervised ({100*n_st//max(n_tok,1)}%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    print(f"[combine] → {args.out}")


if __name__ == "__main__":
    main()
