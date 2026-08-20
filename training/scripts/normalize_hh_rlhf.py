"""Normalize Anthropic hh-rlhf chosen responses to noesis plain-CoT JSONL.

Takes the 'chosen' field (multi-turn dialogue) and emits one item per
assistant turn. No <think> spans — hh-rlhf is quality-filtered dialogue,
not structured reasoning. state_mask will be all-zero → only ε_out L_state.

Output JSONL fields: id, system, user, think (empty), answer
Compatible with training/tokenize_plain_cot.py.

Usage:
    training/.venv/bin/python training/scripts/normalize_hh_rlhf.py \\
        --out training/corpus_open/hh_rlhf.jsonl \\
        --max-items 30000 \\
        --min-answer-len 20 \\
        --max-answer-len 512
"""
from __future__ import annotations

import argparse
import json
import re
import hashlib
from pathlib import Path

# Constitutional AI filter — reject responses containing these patterns
# These are low-quality or harmful signals from the rejected side that
# sometimes leak into chosen examples
_CAI_REJECT = re.compile(
    r"\b(I cannot and will not|I'm not able to help|As an AI language model"
    r"|I don't have personal opinions|I cannot provide|I'm just an AI)\b",
    re.IGNORECASE,
)

# Minimum quality heuristics
_MIN_WORDS = 8
_MAX_WORDS = 400


def parse_dialogue(text: str) -> list[tuple[str, str]]:
    """Split hh-rlhf chosen text into (human, assistant) turn pairs."""
    # Format: \n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...
    parts = re.split(r"\n\nHuman: |\n\nAssistant: ", text.strip())
    # First element is empty (text starts with \n\nHuman:)
    parts = [p.strip() for p in parts if p.strip()]

    turns: list[tuple[str, str]] = []
    i = 0
    while i + 1 < len(parts):
        human = parts[i]
        assistant = parts[i + 1]
        turns.append((human, assistant))
        i += 2
    return turns


def passes_filter(answer: str, min_len: int, max_len: int) -> bool:
    words = answer.split()
    if len(words) < min_len or len(words) > max_len:
        return False
    if _CAI_REJECT.search(answer):
        return False
    # Reject responses that are mostly lists of profanity (low signal)
    if answer.count(",") > 30 and len(words) < 60:
        return False
    return True


def make_context(prior_turns: list[tuple[str, str]]) -> str:
    """Build condensed context from prior turns (last 2 max)."""
    if not prior_turns:
        return ""
    recent = prior_turns[-2:]
    parts = []
    for h, a in recent:
        parts.append(f"Human: {h}")
        parts.append(f"Assistant: {a}")
    return "\n".join(parts) + "\n\n"


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",            required=True)
    ap.add_argument("--max-items",      type=int, default=30000)
    ap.add_argument("--min-answer-len", type=int, default=_MIN_WORDS)
    ap.add_argument("--max-answer-len", type=int, default=_MAX_WORDS)
    ap.add_argument("--last-turn-only", action="store_true",
                    help="Only emit the final assistant turn per dialogue")
    ap.add_argument("--seed",           type=int, default=42)
    return ap


def run(args: argparse.Namespace) -> dict:
    """Registered entry point (see @registry.stage below) — same body
    `main()` always had, just parameterized on `args` instead of calling
    `ArgumentParser.parse_args()` itself, so training/build_corpus.py can
    invoke this directly with a constructed Namespace. Returns a small
    summary dict for the caller (provenance.py stamps the rest)."""
    from datasets import load_dataset
    import random
    rng = random.Random(args.seed)

    print("[hh_rlhf] loading dataset …")
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with open(out_path, "w") as fout:
        for idx in indices:
            if written >= args.max_items:
                break
            chosen = ds[idx]["chosen"]
            turns = parse_dialogue(chosen)
            if not turns:
                skipped += 1
                continue

            emit_turns = [turns[-1]] if args.last_turn_only else turns

            for turn_i, (human, assistant) in enumerate(emit_turns):
                if written >= args.max_items:
                    break
                if not passes_filter(assistant, args.min_answer_len, args.max_answer_len):
                    skipped += 1
                    continue

                # Context = prior turns (up to 2) before this turn
                prior = turns[:turns.index((human, assistant))] if not args.last_turn_only else turns[:-1]
                context = make_context(prior)
                user_text = context + human if context else human

                uid = hashlib.md5(f"{idx}:{turn_i}:{human[:20]}".encode()).hexdigest()[:8]
                item = {
                    "id":     f"hh_{idx}_{uid}",
                    "system": "You are a helpful, harmless, and honest assistant.",
                    "user":   user_text,
                    "think":  "",       # no think span — plain SFT
                    "answer": assistant,
                    "source": "hh_rlhf_chosen",
                }
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1

    print(f"[hh_rlhf] written={written} skipped={skipped} → {args.out}")
    if written < args.max_items:
        print(f"[hh_rlhf] WARNING: only {written}/{args.max_items} items passed filter")

    return {"out_path": str(out_path), "n_rows": written, "n_skipped": skipped}


try:
    from training._common import registry as _registry
    _registry.stage(
        "hh_rlhf", kind="normalize", provenance="external-hf",
        origin="Anthropic/hh-rlhf",
        out_default="training/corpus_open/hh_rlhf.jsonl",
        description="Anthropic hh-rlhf Constitutional AI chosen-responses -> plain-CoT (no think span).",
    )(run)
except ImportError:
    pass  # standalone invocation (python normalize_hh_rlhf.py) doesn't need the registry


if __name__ == "__main__":
    main_args = _build_argparser().parse_args()
    run(main_args)
