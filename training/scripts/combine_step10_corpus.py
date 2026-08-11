"""Combine step 10 corpus: RFC + bitsub + self-CoT + hh-rlhf + aporia + premise + attrib.

Sources and default token fractions:
    rfc     30%  — RFC binary-protocol QA (anchor, state-motion signal carrier)
    bitsub  15%  — Bit-substitution lookup tasks (N-stabilization)
    selfcot 15%  — Self-generated CoT from step9 model (bootstrapped reasoning)
    hhrlhf  15%  — Anthropic hh-rlhf Constitutional AI (reduced from 20%)
    aporia  10%  — H20 symmetry feed-back: ambiguity-holding examples
    premise 10%  — H21 symmetry feed-back: invalid-premise refusals
    attrib   5%  — H22 symmetry feed-back: attribution-labeled examples

Usage:
    training/.venv/bin/python training/scripts/combine_step10_corpus.py \\
        --rfc       training/tokenised/step9_rfc_train.pt \\
        --bitsub    training/tokenised/bitsub_train.pt \\
        --selfcot   training/tokenised/selfcot_train.pt \\
        --hhrlhf    training/tokenised/hh_rlhf_train.pt \\
        --aporia    training/tokenised/aporia_train.pt \\
        --premise   training/tokenised/premise_refusal_train.pt \\
        --attrib    training/tokenised/attrib_train.pt \\
        --fractions rfc=0.30,bitsub=0.15,selfcot=0.15,hhrlhf=0.15,aporia=0.10,premise=0.10,attrib=0.05 \\
        --out       training/tokenised/step10_combined_train.pt \\
        --seed      42

Token-budget approach: rfc is the anchor (100%), others sampled to hit
their target fraction of total tokens. Missing optional sources are skipped
and their fraction is redistributed proportionally among present sources.
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


def _redistribute(fracs: dict[str, float], missing: set[str]) -> dict[str, float]:
    """Drop missing keys and scale remaining fractions to sum to 1.0."""
    present = {k: v for k, v in fracs.items() if k not in missing}
    total = sum(present.values())
    return {k: v / total for k, v in present.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rfc",     required=True,  help="RFC QA .pt (anchor corpus)")
    ap.add_argument("--bitsub",  default=None,   help="Bit-substitution tasks .pt")
    ap.add_argument("--selfcot", default=None,   help="Self-CoT .pt")
    ap.add_argument("--hhrlhf",  default=None,   help="hh-rlhf Constitutional AI .pt")
    ap.add_argument("--aporia",  default=None,   help="Aporia (H20) .pt")
    ap.add_argument("--premise", default=None,   help="Premise-refusal (H21) .pt")
    ap.add_argument("--attrib",  default=None,   help="Attribution (H22) .pt")
    ap.add_argument(
        "--fractions",
        default="rfc=0.30,bitsub=0.15,selfcot=0.15,hhrlhf=0.15,aporia=0.10,premise=0.10,attrib=0.05",
        help="Token fraction per corpus (must sum to 1.0)",
    )
    ap.add_argument("--out",  required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    fracs = _parse_fractions(args.fractions)

    source_paths = {
        "bitsub":  args.bitsub,
        "selfcot": args.selfcot,
        "hhrlhf":  args.hhrlhf,
        "aporia":  args.aporia,
        "premise": args.premise,
        "attrib":  args.attrib,
    }

    # Drop fractions for sources that were not provided
    missing = {name for name, path in source_paths.items() if path is None and name in fracs}
    if missing:
        print(f"[combine10] skipping missing sources: {sorted(missing)}")
        fracs = _redistribute(fracs, missing)

    # Load RFC anchor
    print(f"[combine10] loading rfc {args.rfc} …")
    rfc_rolls = _load_rollouts(args.rfc)
    rfc_tokens = sum(r["ids"].numel() for r in rfc_rolls)
    rfc_frac = fracs.get("rfc", 0.30)
    total_target = int(rfc_tokens / rfc_frac)
    print(f"[combine10] rfc={len(rfc_rolls)} rollouts ({rfc_tokens} tok, {rfc_frac*100:.1f}%)")
    print(f"[combine10] total target tokens: {total_target}")

    combined = list(rfc_rolls)

    def _add(name: str, path: str | None):
        if path is None or name not in fracs:
            return
        frac = fracs[name]
        budget = int(total_target * frac)
        print(f"[combine10] loading {name} {path} …")
        rolls = _load_rollouts(path)
        sampled = _sample_to_budget(rolls, budget, rng)
        used = sum(r["ids"].numel() for r in sampled)
        print(f"[combine10] {name}: {len(sampled)}/{len(rolls)} rollouts, {used} tok ({frac*100:.1f}%)")
        combined.extend(sampled)

    _add("bitsub",  args.bitsub)
    _add("selfcot", args.selfcot)
    _add("hhrlhf",  args.hhrlhf)
    _add("aporia",  args.aporia)
    _add("premise", args.premise)
    _add("attrib",  args.attrib)

    rng.shuffle(combined)

    blob = _pack_rollouts(combined)
    n_tok = blob["ids"].numel()
    n_sup = int(blob["loss_mask"].sum())
    n_st  = int(blob["state_mask"].sum())
    print(f"\n[combine10] total: {len(combined)} rollouts, {n_tok} tokens")
    print(f"[combine10] CE-supervised:    {n_sup} ({100*n_sup//max(n_tok,1)}%)")
    print(f"[combine10] state-supervised: {n_st} ({100*n_st//max(n_tok,1)}%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, args.out)
    print(f"[combine10] → {args.out}")


if __name__ == "__main__":
    main()
