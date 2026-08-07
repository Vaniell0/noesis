#!/usr/bin/env python3
"""H20 aporia probe — CPU runner on G1d-0.4B.

For each item in ``items.jsonl``, prefill the prompt, then measure:

1. First-decode logit gap and normalised split between the two alternative
   first-tokens (`alt_x_first`, `alt_y_first`). A well-formed aporia item
   should have a small gap and a mass-share close to 0.5.
2. Modal-collapse rate over sampled continuations: draw K=20 continuations
   at T=1.0, classify each as X-branch / Y-branch / neither by keyword
   scoring against the two alternatives; a strongly-collapsed item has
   one branch capturing near-100 %.
3. Per-category aggregates (contested_facts / bounded_ambiguity /
   underdetermined_inference).

Outputs:

- ``results.jsonl``  — one JSON line per item with raw metrics.
- ``report.md``      — aggregate summary + per-category breakdown.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from typing import Dict, List, Tuple

# Reuse the probe module sitting next to A0_state_probe/probe.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "A0_state_probe"))

import torch  # noqa: E402
from probe import load_model  # noqa: E402


DEFAULT_MODEL = "/home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth"


def _first_tokens_for_continuation(tokenizer, alt: str) -> List[int]:
    """Encode the alternative *as a continuation* and return its ids.

    We prepend a space because the World tokenizer's continuation-form of
    a word carries the leading space. Falls back to the leading-space-less
    form if the space-prefixed form is a single unusual token.
    """
    ids_sp = tokenizer(" " + alt, return_tensors=None)["input_ids"]
    ids_nosp = tokenizer(alt, return_tensors=None)["input_ids"]
    # Prefer the space-prefixed form; return both first-tokens as candidates.
    out = []
    if ids_sp:
        out.append(ids_sp[0])
    if ids_nosp and (not out or ids_nosp[0] != out[0]):
        out.append(ids_nosp[0])
    return out


def _softmax_row(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.to(torch.float32), dim=-1)


def _keyword_tokens(alt: str) -> List[str]:
    """Extract lowercase keyword tokens from an alternative for classification."""
    stop = {"the", "a", "an", "of", "to", "in", "on", "at", "by", "for",
            "with", "and", "or", "as", "is", "was", "were", "be", "been",
            "it", "its", "that", "this", "from", "into", "than", "then"}
    words = re.findall(r"[a-z0-9]+", alt.lower())
    return [w for w in words if len(w) >= 3 and w not in stop]


def _classify_continuation(text: str, kw_x: List[str], kw_y: List[str]) -> str:
    """Classify a continuation as x / y / neither by keyword hits."""
    low = text.lower()
    hits_x = sum(1 for w in kw_x if w in low)
    hits_y = sum(1 for w in kw_y if w in low)
    if hits_x > hits_y:
        return "x"
    if hits_y > hits_x:
        return "y"
    return "neither"


def _clone_state(state) -> list:
    """Deep-copy a rwkv state list so per-seed decodes don't clobber it."""
    return [t.clone() for t in state]


def _sample_continuation(
    model, tokenizer, prefill_state, logits_after_prefill: torch.Tensor,
    max_new_tokens: int, seed: int, temperature: float, top_p: float,
) -> str:
    """Sample a short continuation from a cloned prefill state.

    Clones the shared prefill state per-call so seeds don't interfere.
    A single clone of the 0.4B state is ~10 MB — cheap vs a prefill.
    """
    torch.manual_seed(seed)
    st = _clone_state(prefill_state)
    logits = logits_after_prefill.clone()
    if logits.dim() > 1:
        logits = logits.reshape(-1)
    out_ids: List[int] = []
    for _ in range(max_new_tokens):
        probs = _softmax_row(logits / max(temperature, 1e-6))
        # Nucleus filter.
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        cut = (cum > top_p).nonzero(as_tuple=False)
        if cut.numel() > 0:
            k = int(cut[0].item()) + 1
        else:
            k = sp.numel()
        kp = sp[:k] / sp[:k].sum()
        idx = int(torch.multinomial(kp, num_samples=1).item())
        tok = int(si[idx].item())
        out_ids.append(tok)
        logits, st = model.forward([tok], st)
        if logits.dim() > 1:
            logits = logits.reshape(-1)
        # Break on sentence-final punctuation once we've emitted some text.
        if tok in _SENTENCE_END_TOKENS_CACHE.get(id(tokenizer), []):
            if len(out_ids) >= 6:
                break
    return tokenizer.decode(out_ids)


_SENTENCE_END_TOKENS_CACHE: Dict[int, List[int]] = {}


def _init_sentence_end_tokens(tokenizer) -> None:
    if id(tokenizer) in _SENTENCE_END_TOKENS_CACHE:
        return
    ids = []
    for punct in [".", "!", "?", "\n"]:
        try:
            enc = tokenizer(punct, return_tensors=None)["input_ids"]
            if enc:
                ids.append(enc[-1])
        except Exception:
            pass
    _SENTENCE_END_TOKENS_CACHE[id(tokenizer)] = ids


def _run_item(
    model, tokenizer, item: Dict, n_samples: int, max_new_tokens: int,
    temperature: float, top_p: float,
) -> Dict:
    prompt = item["prompt"]
    alt_x, alt_y = item["alternatives"]

    x_first_ids = _first_tokens_for_continuation(tokenizer, alt_x)
    y_first_ids = _first_tokens_for_continuation(tokenizer, alt_y)

    enc = tokenizer(prompt, return_tensors="pt")
    prompt_ids = enc["input_ids"][0].tolist()

    t0 = time.time()
    logits, prefill_state = model.forward(prompt_ids, None)
    if logits.dim() > 1:
        logits = logits.reshape(-1)
    probs = _softmax_row(logits)

    # For each alternative, take the max probability over its candidate
    # first-tokens (the space-prefixed form vs the raw form).
    p_x = float(max(probs[i].item() for i in x_first_ids))
    p_y = float(max(probs[i].item() for i in y_first_ids))
    l_x = float(max(logits[i].item() for i in x_first_ids))
    l_y = float(max(logits[i].item() for i in y_first_ids))
    logit_gap = abs(l_x - l_y)
    share_x = p_x / (p_x + p_y) if (p_x + p_y) > 0 else 0.5
    # collapse: distance from 50/50.
    collapse_first = abs(share_x - 0.5) * 2.0  # 0 = balanced, 1 = collapsed

    # Continuation sampling.
    kw_x, kw_y = _keyword_tokens(alt_x), _keyword_tokens(alt_y)
    classes: List[str] = []
    for k in range(n_samples):
        text = _sample_continuation(
            model, tokenizer, prefill_state, logits,
            max_new_tokens=max_new_tokens, seed=1000 + k,
            temperature=temperature, top_p=top_p,
        )
        classes.append(_classify_continuation(text, kw_x, kw_y))

    n_x = sum(1 for c in classes if c == "x")
    n_y = sum(1 for c in classes if c == "y")
    n_neither = sum(1 for c in classes if c == "neither")
    committed = n_x + n_y
    branch_share_x = (n_x / committed) if committed > 0 else 0.5
    collapse_cont = abs(branch_share_x - 0.5) * 2.0 if committed > 0 else 0.0

    wall = time.time() - t0
    return {
        "id": item["id"],
        "category": item["category"],
        "prompt": prompt,
        "alt_x": alt_x,
        "alt_y": alt_y,
        "p_x_first": p_x,
        "p_y_first": p_y,
        "logit_x_first": l_x,
        "logit_y_first": l_y,
        "logit_gap": logit_gap,
        "share_x_first": share_x,
        "collapse_first": collapse_first,
        "n_samples": n_samples,
        "n_x_branch": n_x,
        "n_y_branch": n_y,
        "n_neither": n_neither,
        "branch_share_x": branch_share_x,
        "collapse_cont": collapse_cont,
        "wall_s": wall,
    }


def _summ(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"n": 0, "mean": 0.0, "std": 0.0}
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "median": statistics.median(vals),
    }


def _write_report(rows: List[Dict], out_dir: str, meta: Dict) -> None:
    lines: List[str] = []
    lines.append("# H20 aporia probe — pilot report\n")
    lines.append(f"- Model: `{meta['model']}`")
    lines.append(f"- Items: {len(rows)}")
    lines.append(f"- Samples per item: {meta['n_samples']}, "
                 f"max_new_tokens={meta['max_new_tokens']}, "
                 f"T={meta['temperature']}, top_p={meta['top_p']}")
    lines.append(f"- Wall total: {meta['wall_total_s']:.1f} s\n")

    cats = sorted({r["category"] for r in rows})
    lines.append("## Aggregate\n")
    for cat in ["all"] + cats:
        sub = rows if cat == "all" else [r for r in rows if r["category"] == cat]
        if not sub:
            continue
        collapse_first = [r["collapse_first"] for r in sub]
        collapse_cont = [r["collapse_cont"] for r in sub]
        logit_gap = [r["logit_gap"] for r in sub]
        neither = [r["n_neither"] / r["n_samples"] for r in sub]
        lines.append(f"### {cat} (n={len(sub)})\n")
        lines.append(f"- collapse_first (0=balanced, 1=collapsed): {_summ(collapse_first)}")
        lines.append(f"- collapse_cont  (0=balanced, 1=collapsed): {_summ(collapse_cont)}")
        lines.append(f"- logit_gap:  {_summ(logit_gap)}")
        lines.append(f"- p(neither branch): {_summ(neither)}\n")

    lines.append("## Per-item table\n")
    lines.append("| id | cat | logit_gap | share_x_first | branch_x/y/none | collapse_cont |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['category']} | "
            f"{r['logit_gap']:.3f} | {r['share_x_first']:.2f} | "
            f"{r['n_x_branch']}/{r['n_y_branch']}/{r['n_neither']} | "
            f"{r['collapse_cont']:.2f} |"
        )
    lines.append("")

    lines.append("## Notes on interpretation\n")
    lines.append("- Well-formed aporia items should show:")
    lines.append("  - `collapse_first` near 0 (first-token mass roughly balanced)")
    lines.append("  - `collapse_cont` well below 1 (continuations don't uniformly commit to one branch)")
    lines.append("  - non-zero `p(neither branch)` allowed (hedging / no-commit answers)")
    lines.append("- High `logit_gap` with low `collapse_cont` = model expressed indecision downstream even though")
    lines.append("  it had a favourite first token — the state carries the disagreement.")
    lines.append("- Category-level pattern: contested_facts is expected to collapse more than")
    lines.append("  bounded_ambiguity (semantic ambiguity has no pretraining preference), which in")
    lines.append("  turn collapses more than underdetermined_inference (task ambiguity is deeper).\n")

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="H20 aporia probe runner (CPU).")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--items", default=os.path.join(_HERE, "items.jsonl"))
    ap.add_argument("--out", default=_HERE)
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="Debug: only run first N items (0=all).")
    ap.add_argument("--start", type=int, default=0, help="Shard start index (inclusive).")
    ap.add_argument("--end", type=int, default=0, help="Shard end index (exclusive). 0 = all.")
    ap.add_argument("--no-report", action="store_true", help="Skip report.md write (for shards).")
    args = ap.parse_args()

    device = os.environ.get("NOESIS_EVAL_DEVICE", "cpu")
    print(f"[H20] loading model {args.model} on {device}", file=sys.stderr, flush=True)
    t0 = time.time()
    model, tokenizer = load_model(args.model, device=device)
    print(f"[H20] loaded in {time.time()-t0:.1f}s", file=sys.stderr, flush=True)
    _init_sentence_end_tokens(tokenizer)

    items: List[Dict] = []
    with open(args.items) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if args.limit > 0:
        items = items[: args.limit]
    if args.end > 0 or args.start > 0:
        end = args.end if args.end > 0 else len(items)
        items = items[args.start:end]
        print(f"[H20] shard start={args.start} end={end}", file=sys.stderr, flush=True)
    print(f"[H20] running {len(items)} items", file=sys.stderr, flush=True)

    os.makedirs(args.out, exist_ok=True)
    rows: List[Dict] = []
    results_path = os.path.join(args.out, "results.jsonl")
    with open(results_path, "w") as fout:
        for i, item in enumerate(items):
            r = _run_item(
                model, tokenizer, item,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            rows.append(r)
            fout.write(json.dumps(r) + "\n")
            fout.flush()
            print(
                f"[H20] {i+1}/{len(items)} {r['id']} "
                f"gap={r['logit_gap']:.3f} share_x={r['share_x_first']:.2f} "
                f"branch={r['n_x_branch']}/{r['n_y_branch']}/{r['n_neither']} "
                f"wall={r['wall_s']:.1f}s",
                file=sys.stderr, flush=True,
            )

    wall_total = time.time() - t0
    if args.no_report:
        print(f"[H20] shard done wall={wall_total:.1f}s (no report)", file=sys.stderr)
        return 0
    _write_report(rows, args.out, meta={
        "model": args.model,
        "n_samples": args.n_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "wall_total_s": wall_total,
    })
    print(f"[H20] done wall={wall_total:.1f}s → {results_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
