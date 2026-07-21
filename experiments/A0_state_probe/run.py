#!/usr/bin/env python3
"""A0.4 state-utilisation probe — CLI runner.

Iterates over seeds, streams StateSteps from ``probe.generate_with_state_hooks``
into the three online metric functions in ``metrics``, and writes:

- ``<out>/seed_<i>.json`` — per-step metric vectors, one file per seed.
- ``<out>/summary.json``  — cell-level aggregate (mean/std across seeds
  and across steps) for the pre-registered thresholds.

Usage:

    python3 run.py \
        --model 'BlinkDL/rwkv-7-world:RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth' \
        --prompt medium --seeds 3 --pilot \
        --out results/pilot/

    python3 run.py \
        --model 'BlinkDL/rwkv-7-world:RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth' \
        --prompt medium --seeds 10 \
        --out results/world_medium/
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List

# Local imports (probe.py, metrics.py, prompts.py sit next to run.py).
import prompts as PROMPT_BANK
from probe import generate_with_state_hooks, load_model
from metrics import curvature, delta_norm, stable_rank


def _summ(values: List[float]) -> Dict[str, float]:
    """Compact summary stats used across all metrics."""
    if not values:
        return {"n": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": min(values),
        "max": max(values),
    }


def _run_one_seed(
    model,
    tokenizer,
    prompt_text: str,
    seed: int,
    max_new_tokens: int,
    device: str,
    verbose: bool,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    """Generate one seed's tokens; compute all three metrics online."""
    delta_pooled: List[float] = []
    curv_pooled: List[float] = []
    sr_mean_per_step: List[float] = []  # mean over layers × heads
    sr_std_per_step: List[float] = []
    token_ids: List[int] = []

    prev_prev = None
    prev = None
    t0 = time.time()

    for step in generate_with_state_hooks(
        model,
        tokenizer,
        prompt_text,
        max_new_tokens=max_new_tokens,
        seed=seed,
        device=device,
        verbose=verbose and seed == 0,
        temperature=temperature,
        top_p=top_p,
    ):
        curr = step.wkv_per_layer
        token_ids.append(step.token_id)

        if prev is not None:
            d_pool, _ = delta_norm(prev, curr)
            delta_pooled.append(d_pool)
        if prev_prev is not None and prev is not None:
            c_pool, _ = curvature(prev_prev, prev, curr)
            curv_pooled.append(c_pool)

        sr = stable_rank(curr)
        flat = [x for layer in sr for x in layer]
        sr_mean_per_step.append(statistics.fmean(flat))
        sr_std_per_step.append(statistics.pstdev(flat) if len(flat) > 1 else 0.0)

        prev_prev = prev
        prev = curr
        # explicit drop; each state is ~10 MB fp32 across 32 layers.

    wall = time.time() - t0

    return {
        "seed": seed,
        "n_tokens": len(token_ids),
        "wall_s": wall,
        "token_ids": token_ids,
        "delta_pooled": delta_pooled,      # length n-1
        "curvature_pooled": curv_pooled,   # length n-2
        "sr_step_mean": sr_mean_per_step,  # length n
        "sr_step_std": sr_std_per_step,
    }


def _aggregate(seeds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Roll seeds into cell-level summary used for threshold checks."""
    per_seed_delta = [statistics.fmean(s["delta_pooled"]) for s in seeds if s["delta_pooled"]]
    per_seed_curv = [statistics.fmean(s["curvature_pooled"]) for s in seeds if s["curvature_pooled"]]
    per_seed_sr_std = [statistics.fmean(s["sr_step_std"]) for s in seeds if s["sr_step_std"]]

    return {
        "n_seeds": len(seeds),
        "wall_s_total": sum(s["wall_s"] for s in seeds),
        "delta_pooled_seed_mean": _summ(per_seed_delta),
        "curvature_pooled_seed_mean": _summ(per_seed_curv),
        "stable_rank_variance_seed_mean": _summ(per_seed_sr_std),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="RWKV-7 state-utilisation probe (A0.4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--model",
        required=True,
        help=(
            "RWKV-7 .pth checkpoint. Accepted forms: "
            "'owner/repo:filename.pth' (HF-hosted, e.g. "
            "'BlinkDL/rwkv-7-world:RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth') "
            "or a local .pth path. GGUF / Ollama tags are not accepted — "
            "quantised weights distort state dynamics."
        ),
    )
    ap.add_argument(
        "--prompt",
        required=True,
        choices=list(PROMPT_BANK.ALL.keys()),
        help=(
            "Prompt label from the shared A0 bank. For H8/H9 the "
            "planned cells use `medium` (reasoning) and `narrative` "
            "(non-reasoning control of matched length)."
        ),
    )
    ap.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of seeds (independent runs) to sample per cell.",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Tokens generated per seed (matches A0.1 bench).",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="torch device — CPU is the safe default.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Set to 0 for greedy (deterministic).",
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="Nucleus (top-p) sampling threshold.",
    )
    ap.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "Pilot mode — verbose plumbing print on first seed, no "
            "change to output format. Use for the noise-floor run."
        ),
    )
    ap.add_argument(
        "--out",
        required=True,
        help=(
            "Output directory for per-seed JSON and cell summary. "
            "Created if missing."
        ),
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    prompt_text = PROMPT_BANK.ALL[args.prompt]

    print(
        f"[run] model={args.model} prompt={args.prompt} seeds={args.seeds} "
        f"tokens={args.max_new_tokens} device={args.device} out={args.out}",
        file=sys.stderr,
    )

    print(f"[run] loading model ...", file=sys.stderr)
    t0 = time.time()
    model, tokenizer = load_model(args.model, device=args.device)
    print(f"[run] model loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    all_seeds: List[Dict[str, Any]] = []
    for seed in range(args.seeds):
        print(f"[run] seed {seed} ...", file=sys.stderr, flush=True)
        result = _run_one_seed(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            seed=seed,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            verbose=args.pilot,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        all_seeds.append(result)
        seed_path = os.path.join(args.out, f"seed_{seed}.json")
        with open(seed_path, "w") as f:
            json.dump(result, f, indent=None, separators=(",", ":"))
        print(
            f"[run] seed {seed} wall={result['wall_s']:.1f}s "
            f"delta_mean={statistics.fmean(result['delta_pooled']):.3g} "
            f"curv_mean={statistics.fmean(result['curvature_pooled']):.3g} "
            f"sr_step_std_mean={statistics.fmean(result['sr_step_std']):.3g}",
            file=sys.stderr,
        )

    summary = {
        "model": args.model,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "aggregate": _aggregate(all_seeds),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[run] wrote {args.out}/summary.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
