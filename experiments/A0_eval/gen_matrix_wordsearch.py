#!/usr/bin/env python3
"""Generate matrix_wordsearch tasks for A0.2.

Task shape (per user 2026-07-28, memory `project_noesis_matrix_wordsearch_eval`):
  - N×N letter matrix, rendered row-per-line with letters space-separated
    (one letter = one WORLD token). Space-separation is the intentional
    input-side simplification for the first cut — the model does not have
    to also solve subword segmentation while it scans the grid.
  - **Horizontal placement only** for this first cut. Diagonal / vertical /
    reverse orientations are follow-ups once even horizontal moves off zero.
  - **One target word per task.** Prompt asks model to output row and
    column (0-indexed) as `row=R col=C`. Regex rubric matches; per-word
    accuracy across the file gives the aggregate signal.
  - The word is placed at a uniformly-random horizontal slot; remaining
    cells are filled with random uppercase letters that do not spell the
    target word by accident.

Deterministic: takes ``--seed`` so the exact same file can be regenerated.

Usage:
  python3 gen_matrix_wordsearch.py --seed 42 \\
    --out tasks_matrix_wordsearch.jsonl \\
    --tiers 5x5:3:4,6x6:4:5,7x7:5:6

Where ``--tiers`` is comma-separated tier specs ``NxN:min_len:max_len``.
Per tier, ``--per-tier`` tasks are emitted (default 4).
"""

from __future__ import annotations

import argparse
import json
import random
import string
from typing import List, Tuple

# Curated real English word bank — short, common, mixed lengths.
# Not pulled from an online dictionary so tasks stay reproducible without
# an external corpus dependency. All uppercase; no repeated letters is not
# required (repetition just makes the accidental-collision guard trip more
# often, which is fine — the guard handles it).
WORD_BANK = [
    # 3
    "CAT", "DOG", "SUN", "MAP", "KEY", "OWL", "FOX", "PEN", "ICE", "OAK",
    # 4
    "MOON", "TREE", "BIRD", "FIRE", "STAR", "LEAF", "WOLF", "SHIP", "GATE", "ROSE",
    # 5
    "RIVER", "STONE", "CLOUD", "STORM", "BEACH", "PIANO", "GLASS", "HORSE",
    "SNAKE", "FROST", "SMILE", "WATER",
    # 6
    "GARDEN", "TIMBER", "HOLLOW", "SILVER", "GOLDEN", "BREEZE", "FOREST",
    "MEADOW", "BRIDGE", "CANDLE", "FALCON", "PLANET",
    # 7
    "THUNDER", "HARVEST", "GRANITE", "MEADOWS", "COMPASS", "JUPITER",
    "MYSTERY", "SILENCE",
]


def _pick_word(rng: random.Random, min_len: int, max_len: int) -> str:
    pool = [w for w in WORD_BANK if min_len <= len(w) <= max_len]
    if not pool:
        raise ValueError(f"no words with {min_len} <= len <= {max_len}")
    return rng.choice(pool)


def _fill_grid(rng: random.Random, n: int, target: str,
               row: int, col: int) -> List[List[str]]:
    """Fill an N×N grid with random uppercase, place ``target`` horizontally
    at (row, col), and rewrite any accidental left-to-right occurrence of
    ``target`` elsewhere so the placement is unique."""
    grid = [[rng.choice(string.ascii_uppercase) for _ in range(n)] for _ in range(n)]
    for i, ch in enumerate(target):
        grid[row][col + i] = ch
    _break_accidental_matches(rng, grid, target, row, col)
    return grid


def _break_accidental_matches(rng: random.Random, grid: List[List[str]],
                              target: str, keep_row: int, keep_col: int) -> None:
    n = len(grid)
    L = len(target)
    max_passes = 8  # bounded — random restarts are cheap
    for _ in range(max_passes):
        found = _find_horizontal(grid, target)
        # Drop the intended placement.
        found = [(r, c) for (r, c) in found if not (r == keep_row and c == keep_col)]
        if not found:
            return
        # Break one at a time by mutating a random cell in the run.
        for (r, c) in found:
            j = rng.randrange(L)
            cur = grid[r][c + j]
            # Pick a replacement that also breaks the run
            candidates = [ch for ch in string.ascii_uppercase if ch != target[j] and ch != cur]
            grid[r][c + j] = rng.choice(candidates)
    # If after max_passes there is still a collision, drop the accidental
    # ones by force. The rare case; still keeps rubric well-defined.
    for _ in range(max_passes):
        found = [rc for rc in _find_horizontal(grid, target)
                 if rc != (keep_row, keep_col)]
        if not found:
            return
        r, c = found[0]
        grid[r][c] = "X" if target[0] != "X" else "Y"


def _find_horizontal(grid: List[List[str]], word: str) -> List[Tuple[int, int]]:
    n = len(grid)
    L = len(word)
    hits: List[Tuple[int, int]] = []
    for r in range(n):
        for c in range(n - L + 1):
            if "".join(grid[r][c:c + L]) == word:
                hits.append((r, c))
    return hits


def _render(grid: List[List[str]]) -> str:
    return "\n".join(" ".join(row) for row in grid)


def make_prompt(grid_str: str, word: str, n: int) -> str:
    return (
        f"Below is a {n}x{n} letter matrix. Rows are separated by newlines; "
        f"letters within a row are separated by single spaces. Rows and "
        f"columns are 0-indexed (top-left is row=0 col=0).\n\n"
        f"{grid_str}\n\n"
        f"The word {word} appears exactly once, placed horizontally "
        f"(left-to-right). Find it. Output only the position in the form "
        f"'row=R col=C' where R is the row index and C is the column index "
        f"of the word's first letter."
    )


def make_task(rng: random.Random, tier_idx: int, task_idx: int,
              n: int, min_len: int, max_len: int) -> dict:
    word = _pick_word(rng, min_len, max_len)
    row = rng.randrange(n)
    col = rng.randrange(n - len(word) + 1)
    grid = _fill_grid(rng, n, word, row, col)
    grid_str = _render(grid)
    prompt = make_prompt(grid_str, word, n)
    # Regex allows optional whitespace and either quote style; case insensitive
    # is applied by the scorer.
    rubric_value = rf"row\s*=\s*{row}\b[^0-9]*col\s*=\s*{col}\b"
    return {
        "id": f"wsearch_h_{n}x{n}_{tier_idx:02d}_{task_idx:02d}",
        "category": "matrix_wordsearch",
        "prompt": prompt,
        "answer": f"row={row} col={col}",
        "rubric": {"type": "regex", "value": rubric_value},
        "notes": (
            f"Horizontal-only, n={n}, word='{word}' (len {len(word)}), "
            f"placed at (row={row}, col={col}). "
            f"Ceiling test per memory `project_noesis_matrix_wordsearch_eval`."
        ),
    }


def parse_tier(spec: str) -> Tuple[int, int, int]:
    n_part, min_part, max_part = spec.split(":")
    n = int(n_part.lower().split("x")[0])
    return n, int(min_part), int(max_part)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate matrix_wordsearch tasks.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", default="5x5:3:4,6x6:4:5,7x7:5:6",
                    help="Comma-separated NxN:min_len:max_len tuples.")
    ap.add_argument("--per-tier", type=int, default=4)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tiers = [parse_tier(s) for s in args.tiers.split(",")]

    tasks: List[dict] = []
    for ti, (n, mn, mx) in enumerate(tiers):
        for tj in range(args.per_tier):
            tasks.append(make_task(rng, ti, tj, n, mn, mx))

    with open(args.out, "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"[gen] wrote {len(tasks)} tasks to {args.out}")
    for t in tasks[:3]:
        print(f"[gen] sample id={t['id']} answer={t['answer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
