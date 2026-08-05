#!/usr/bin/env python3
"""A0.H12a v2 — width sweep with FIXED gap.

v1 confound: as N grows, the mean distance from any triple to the question
also grows (more triples = more tokens before question). This confounds
width with decay.

v2 fix: after the last item line, insert tail-filler until the total
word-distance from the LAST item to the question equals TARGET_TAIL_WORDS.
This holds the decay axis constant across all N, so any accuracy drop
with N is attributable to width alone.

Sweep: N ∈ {4, 8, 16, 32, 64} at fixed tail-gap ≈ TARGET_TAIL_WORDS.
Items are compact (no inter-item filler) so gap from early triples to
the question still grows with N — this is intentional and matches the
real working-memory challenge (holding item-0 while reading item-N-1).
The controlled quantity is the distance from the LAST item to the question.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from dataclasses import dataclass
from typing import List, Tuple

# Reuse vocabulary from v1.
COLOURS = [
    "red", "blue", "green", "yellow", "purple", "orange", "pink",
    "cyan", "magenta", "lime", "brown", "black", "white", "silver",
    "gold", "teal", "navy", "coral", "olive", "maroon",
    "amber", "azure", "beige", "bronze", "chartreuse", "crimson",
    "emerald", "fuchsia", "indigo", "ivory", "khaki", "lavender",
    "mauve", "mint", "ochre", "peach", "periwinkle", "plum",
    "ruby", "saffron", "salmon", "sapphire", "scarlet", "sienna",
    "slate", "tangerine", "topaz", "turquoise", "vermilion", "violet",
    "wheat", "wine", "aqua", "aquamarine", "auburn", "buff",
    "cerulean", "cinnabar", "cobalt", "copper", "eggshell", "fern",
    "forest", "gainsboro", "ginger", "glaucous", "goldenrod",
    "honey", "iris", "jade", "lilac",
]

ITEM_PREFIXES = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
    "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron",
    "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi",
    "omega", "north", "south", "east", "west", "solar", "lunar",
    "polar", "arctic", "boreal", "austral", "tropic", "coastal",
    "inland", "desert", "forest2", "prairie", "highland", "lowland",
    "urban", "rural", "arid", "humid", "temperate", "boreal2",
    "frost", "ember", "iron", "quartz", "onyx", "jade2", "amber2",
    "coral2", "pearl", "opal", "flint", "granite", "basalt", "chalk",
]

TARGET_TAIL_WORDS = 50   # filler words after last item, before question
N_VALUES = [4, 8, 16, 32, 64]
N_PAIRS = 2              # planted colour pairs per task (same as v1)
SEEDS = list(range(10))  # 10 seeds per N


def _item_name(idx: int) -> str:
    return f"item-{ITEM_PREFIXES[idx % len(ITEM_PREFIXES)]}-{idx:02d}"


def _filler_line(rng: random.Random) -> str:
    kind = rng.choice(["weather", "instrument", "checksum"])
    if kind == "weather":
        return (f"(sensor {rng.randint(1,999)}: temp {rng.uniform(-5,35):.1f}C "
                f"humidity {rng.randint(20,90)}% pressure {rng.randint(980,1030)}hPa)")
    if kind == "instrument":
        return (f"(log {rng.randint(1,999)}: val {rng.uniform(0,100):.2f} "
                f"tol {rng.uniform(0.1,5):.2f} status ok)")
    return (f"(chk {rng.randint(1,999)}: parity even "
            f"hash {rng.randint(0,65535):04x} ts {rng.randint(1000,9999)})")


def _word_count(s: str) -> int:
    return len(s.split())


def _colour_assignment(rng: random.Random, n: int) -> Tuple[List[str], List[Tuple[int, int]]]:
    pool = list(COLOURS)
    rng.shuffle(pool)
    positions = list(range(n))
    rng.shuffle(positions)
    colour_of = [None] * n
    planted: List[Tuple[int, int]] = []
    for k in range(N_PAIRS):
        c = pool.pop()
        i, j = positions[2 * k], positions[2 * k + 1]
        colour_of[i] = c
        colour_of[j] = c
        planted.append((min(i, j), max(i, j)))
    for idx in positions[2 * N_PAIRS:]:
        colour_of[idx] = pool.pop()
    return colour_of, planted


def _build_prompt(colour_of: List[str], rng: random.Random) -> Tuple[str, int]:
    """Return (prompt, tail_filler_words_inserted)."""
    lines = [
        "You are given a list of items and their colours.",
        "Some items share a colour with exactly one other item; the rest are unique.",
        "List every pair of items that share a colour, one pair per line, "
        'in the form "item-X, item-Y". Output only the pairs.',
        "",
        "Items:",
    ]
    for i, c in enumerate(colour_of):
        lines.append(f"- {_item_name(i)} has colour {c}.")

    # Tail filler: pad until TARGET_TAIL_WORDS words added after last item.
    lines.append("")
    words_added = 0
    while words_added < TARGET_TAIL_WORDS:
        fl = _filler_line(rng)
        lines.append(fl)
        words_added += _word_count(fl)

    lines += ["", "Pairs:"]
    return "\n".join(lines), words_added


def generate(out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in N_VALUES:
        tasks = []
        for seed in SEEDS:
            rng = random.Random(seed * 1000 + n)
            colour_of, planted = _colour_assignment(rng, n)
            prompt, tail_words = _build_prompt(colour_of, rng)
            tasks.append({
                "id": f"v2-N{n:02d}-s{seed:02d}",
                "prompt": prompt,
                "expected_pairs": [list(p) for p in planted],
                "n": n,
                "tail_filler_words": tail_words,
                "seed": seed,
                "variant": "width_v2_fixed_tail",
            })
        path = out_dir / f"tasks-v2-N{n:02d}.jsonl"
        with open(path, "w") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")
        print(f"  {path.name}: {len(tasks)} tasks, N={n}, tail≈{TARGET_TAIL_WORDS}w")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="experiments/A0_H12a_working_memory/tasks_v2",
                   help="Output directory for v2 task JSONL files")
    args = p.parse_args()
    print(f"Generating H12a v2 tasks (fixed tail={TARGET_TAIL_WORDS}w) ...")
    generate(pathlib.Path(args.out))
    print("Done.")
