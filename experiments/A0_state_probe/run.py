#!/usr/bin/env python3
"""A0.4 state-utilisation probe — CLI wrapper (skeleton).

The CLI shape is fixed in this commit so the execution session starts
from a known argument surface. Actual probing (weight loading, hooks,
metric computation, output writing) is unimplemented and lives in
`probe.py` and `metrics.py`. Running this script now prints a notice
and exits 0.

Planned usage (once implemented):

    python3 run.py --model RWKV/rwkv-7-world \\
                   --prompt medium --seeds 10 --out results/world_medium/

See ../../docs/state-and-reasoning.md for the underlying design
motivation and ./README.md for the experiment plan.
"""

from __future__ import annotations

import argparse
import sys

import prompts as PROMPT_BANK


def main() -> int:
    ap = argparse.ArgumentParser(
        description="RWKV-7 state-utilisation probe (A0.4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--model",
        required=True,
        help=(
            "HuggingFace repo id of a native-bf16 RWKV-7 checkpoint "
            "(e.g. RWKV/rwkv-7-world). GGUF paths and Ollama tags are "
            "not accepted — quantised weights distort state dynamics "
            "at the level H8/H9 want to measure."
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
        "--out",
        required=True,
        help=(
            "Output directory for the metric time-series. Written as "
            "one parquet per (seed, metric) plus a summary CSV. "
            "Directory is created if missing."
        ),
    )
    args = ap.parse_args()

    print(
        "A0.4 state-utilisation probe: not yet implemented.\n"
        f"  planned run: model={args.model} prompt={args.prompt} "
        f"seeds={args.seeds} tokens={args.max_new_tokens} device={args.device}\n"
        f"  planned out: {args.out}\n"
        "See ./README.md and ../../HYPOTHESES.md (H8, H9). "
        "Execution session will implement probe.py + metrics.py "
        "and remove this notice.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
