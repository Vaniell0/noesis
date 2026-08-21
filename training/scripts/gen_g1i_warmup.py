"""Build a G1i-native <think> warm-up corpus from matrix_tasks.jsonl.

No foreign model involved — think content is procedurally derived from
each task's own ground-truth answer + category, short and varied by
category (not one universal template, to avoid re-creating the same
collapse-to-one-phrase pattern found in the RFC-heavy corpus). Goal is
teaching the <think></think> TAG STRUCTURE, not a reasoning style — G1i's
own natural output is already the most concise (DE table, hypotheses/H24)
of anything tried, terse-by-construction templates preserve that.

Moved into training/scripts/ + the registry (2026-08-21) from a job-tmp
script that produced both g1i_warmup.jsonl (--per-bucket 15, 585 rows)
and g1i_warmup_v2.jsonl (--per-bucket 50, 1950 rows) by hand-editing one
constant between runs — --per-bucket is now a real CLI arg instead.
matrix_tasks.jsonl has 39 (category, level) buckets, min bucket size 314
(checked 2026-08-21) — --per-bucket up to ~300 draws from real task
diversity with no bucket needing to repeat itself.

Usage:
    training/.venv/bin/python training/scripts/gen_g1i_warmup.py \\
        --tasks training/corpus_open/matrix_tasks.jsonl \\
        --out training/corpus_open/g1i_warmup_v3.jsonl \\
        --per-bucket 300
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random

TEMPLATES = {
    "arithmetic_matrix": [
        "The pattern is arithmetic; applying it gives {answer}.",
        "Following the arithmetic rule directly yields {answer}.",
        "Computing the arithmetic step gives {answer}.",
    ],
    "bits_matrix": [
        "Computing the bitwise operation gives {answer}.",
        "Applying the bit rule directly yields {answer}.",
        "The bitwise result is {answer}.",
    ],
    "pattern_matrix": [
        "Following the matrix's pattern, the missing value is {answer}.",
        "The pattern continues with {answer}.",
        "Matching the rule gives {answer}.",
    ],
    "matrix_wordsearch": [
        "Scanning the grid locates the word: {answer}.",
        "Checking each direction finds {answer}.",
        "The hidden word is {answer}.",
    ],
    "matrix_wordsearch_name": [
        "Scanning the grid locates the name: {answer}.",
        "Checking each direction finds {answer}.",
        "The hidden name is {answer}.",
    ],
    "crossword_enum": [
        "Filling the crossword slots gives {answer}.",
        "The words that fit are {answer}.",
    ],
    "crossword_fill": [
        "Using the crossing letters, the word is {answer}.",
        "The crossing letters spell {answer}.",
    ],
}


def make_think(rng: random.Random, category: str, answer: str) -> str:
    tpl = rng.choice(TEMPLATES.get(category, ["The answer is {answer}."]))
    return tpl.format(answer=answer)


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="training/corpus_open/matrix_tasks.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bucket", type=int, default=50,
                    help="Examples drawn per (category, level) bucket. "
                         "39 buckets, min real bucket size 314 as of "
                         "2026-08-21 — stay under that to avoid repeats.")
    ap.add_argument("--seed", type=int, default=7)
    return ap


def run(args: argparse.Namespace) -> dict:
    rng = random.Random(args.seed)
    buckets: dict[tuple, list] = {}
    with open(args.tasks) as f:
        for line in f:
            d = json.loads(line)
            k = (d["category"], d["level"])
            buckets.setdefault(k, []).append(d)

    out_tasks = []
    for k, items in sorted(buckets.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        out_tasks.extend(shuffled[: args.per_bucket])
    rng.shuffle(out_tasks)

    items = []
    for d in out_tasks:
        answer = str(d["answer"]).replace("\n", " ")
        think = make_think(rng, d["category"], answer)
        uid = hashlib.md5(f"{d['id']}:{think[:40]}".encode()).hexdigest()[:10]
        items.append({
            "id": f"warmup_{d['id']}_{uid}",
            "system": "You are a precise reasoning assistant. Work step by step.",
            "user": d["prompt"],
            "think": think,
            "answer": answer,
            "source": "g1i_native_warmup",
            "base_task": d["id"],
        })

    with open(args.out, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[gen_g1i_warmup] {len(items)} examples ({len(buckets)} buckets x up to "
          f"{args.per_bucket}) -> {args.out}")
    return {"out_path": args.out, "n_rows": len(items), "n_buckets": len(buckets)}


try:
    from training._common import registry as _registry
    _registry.stage(
        "g1i_warmup", kind="normalize", provenance="generated",
        origin="training/corpus_open/matrix_tasks.jsonl (procedural, no foreign model)",
        out_default="training/corpus_open/g1i_warmup_v3.jsonl",
        description="Procedural G1i-native <think> warm-up corpus, drawn from matrix_tasks.jsonl.",
    )(run)
except ImportError:
    pass  # standalone invocation doesn't need the registry


if __name__ == "__main__":
    main_args = _build_argparser().parse_args()
    run(main_args)
