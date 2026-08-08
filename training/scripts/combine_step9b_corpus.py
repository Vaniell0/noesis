"""Combine step 9b corpus: RFC + self-CoT + action-chains + hh-rlhf + react + step6 sample.

Sources and default token fractions:
    rfc     25%  — RFC binary-protocol QA (anchor)
    hhrlhf  20%  — Anthropic hh-rlhf Constitutional AI
    react   15%  — Glaive v2 + Toolbench in ReAct <think> format
    action  15%  — Action chains DSL (anti-forgetting)
    selfcot 15%  — Self-generated CoT (needs step9 merged checkpoint)
    base    10%  — Step6 mixed (general anti-forgetting)

Usage:
    training/.venv/bin/python training/scripts/combine_step9b_corpus.py \\
        --rfc      training/tokenised/step9_rfc_train.pt \\
        --selfcot  training/tokenised/selfcot_train.pt \\
        --action   training/tokenised/action_chains_dsl_step8_train.pt \\
        --hhrlhf   training/tokenised/hh_rlhf_train.pt \\
        --react    training/tokenised/react_train.pt \\
        --base     training/tokenised/step6_mixed_train.pt \\
        --fractions rfc=0.25,hhrlhf=0.20,react=0.15,action=0.15,selfcot=0.15,base=0.10 \\
        --out      training/tokenised/step9b_combined_train.pt \\
        --seed     42

Token-budget approach: rfc is the anchor (100%), others are sampled to hit
their target fraction of total tokens. Rollouts that exceed remaining budget
are truncated.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch


def _load_rollouts(pt_path: str) -> list[dict]:
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    ids  = blob["ids"]
    lm   = blob["loss_mask"]
    sm   = blob.get("state_mask", torch.zeros_like(lm))
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


def _sample_to_budget(rollouts: list[dict], budget: int, rng: random.Random) -> list[dict]:
    """Shuffle rollouts and sample until token budget is hit; truncate last."""
    rng.shuffle(rollouts)
    sampled, used = [], 0
    for r in rollouts:
        remaining = budget - used
        if remaining <= 0:
            break
        n = r["ids"].numel()
        if n > remaining:
            r = {k: v[:remaining] for k, v in r.items()}
            n = remaining
        sampled.append(r)
        used += n
    return sampled


def _parse_fractions(s: str) -> dict[str, float]:
    result = {}
    for part in s.split(","):
        k, v = part.strip().split("=")
        result[k.strip()] = float(v.strip())
    total = sum(result.values())
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Fractions sum to {total:.3f}, expected 1.0")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfc",      required=True,  help="RFC QA .pt (anchor corpus)")
    ap.add_argument("--selfcot",  default=None,   help="Self-CoT .pt")
    ap.add_argument("--action",   default=None,   help="Action chains .pt")
    ap.add_argument("--hhrlhf",   default=None,   help="hh-rlhf Constitutional AI .pt")
    ap.add_argument("--react",    default=None,   help="ReAct tool-use .pt (glaive + toolbench)")
    ap.add_argument("--base",     default=None,   help="Base/step6 mixed .pt")
    ap.add_argument("--fractions",
                    default="rfc=0.25,hhrlhf=0.20,react=0.15,action=0.15,selfcot=0.15,base=0.10",
                    help="Token fraction per corpus (must sum to 1.0)")
    ap.add_argument("--out",      required=True)
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fracs = _parse_fractions(args.fractions)

    # Load RFC anchor
    print(f"[combine9b] loading rfc {args.rfc} …")
    rfc_rolls = _load_rollouts(args.rfc)
    rfc_tokens = sum(r["ids"].numel() for r in rfc_rolls)
    rfc_frac = fracs.get("rfc", 0.25)
    total_target = int(rfc_tokens / rfc_frac)
    print(f"[combine9b] rfc={len(rfc_rolls)} rollouts ({rfc_tokens} tok, {rfc_frac*100:.0f}%)")
    print(f"[combine9b] total target tokens: {total_target}")

    combined = list(rfc_rolls)

    def _add(name: str, path: str | None):
        if path is None or name not in fracs:
            return
        frac = fracs[name]
        budget = int(total_target * frac)
        print(f"[combine9b] loading {name} {path} …")
        rolls = _load_rollouts(path)
        sampled = _sample_to_budget(rolls, budget, rng)
        used = sum(r["ids"].numel() for r in sampled)
        print(f"[combine9b] {name}: {len(sampled)}/{len(rolls)} rollouts, {used} tok ({frac*100:.0f}%)")
        combined.extend(sampled)

    _add("selfcot", args.selfcot)
    _add("action",  args.action)
    _add("hhrlhf",  args.hhrlhf)
    _add("react",   args.react)
    _add("base",    args.base)

    rng.shuffle(combined)

    blob = _pack_rollouts(combined)
    n_tok = blob["ids"].numel()
    n_sup = int(blob["loss_mask"].sum())
    n_st  = int(blob["state_mask"].sum())
    print(f"\n[combine9b] total: {len(combined)} rollouts, {n_tok} tokens")
    print(f"[combine9b] CE-supervised: {n_sup} ({100*n_sup//max(n_tok,1)}%)")
    print(f"[combine9b] state-supervised: {n_st} ({100*n_st//max(n_tok,1)}%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    print(f"[combine9b] → {args.out}")


if __name__ == "__main__":
    main()
