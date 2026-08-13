#!/usr/bin/env python3
"""Generate crossword-style tasks for the RL curriculum (levels 8–11).

Builds on gen_matrix_wordsearch.py. Crosswords add constraint satisfaction
on top of spatial scanning: words must intersect at shared cells, so the
model must reason about both the target word AND its crossing neighbours.

## Task types

### Type A — find-in-crossing (levels 8–9)
Multiple words placed in the grid (2–4 words, some intersecting).
Task: find where a *specific* word is. The crossing neighbours create
partial-match decoys that don't fool naive left-to-right scanning.

### Type B — fill-the-blank (levels 10–11)
A partially filled grid: intersecting cells are shown, gaps are blanks.
Task: output the complete target word given only its crossing letters.
Tests constraint propagation — the model must infer the full word from
partial pattern + spatial position.

## Difficulty relative to word-search

| Dimension | Word-search | Crossword type A | Crossword type B |
|-----------|-------------|-----------------|-----------------|
| Grid size | 5–12 | 7–12 | 8–14 |
| Word count | 1 | 2–4 | 2–4 |
| Decoys | random noise | other real words | hidden gaps |
| Output | row=R col=C | row=R col=C | the full word |
| Constraint | none | find among N | fill from crossings |

## H13b connection (spatial text perception)
Crossword grids test whether the model perceives ASCII text as a 2D
structure. A model that reads left-to-right only will fail at intersecting
diagonal/vertical words and blank-fill tasks. Success requires treating
the grid as a spatial object, not a sequence of characters.
"""

from __future__ import annotations

import argparse
import json
import random
import string
from typing import Dict, List, Optional, Tuple

from gen_matrix_wordsearch import (
    ORIENTATIONS, WORD_BANK,
    _can_place, _place_word, _find_word_any_dir,
    _render, _orient_human,
)


# ── Crossword builder ─────────────────────────────────────────────────────────

def _pick_words_with_crossing(
    rng: random.Random, n: int, n_words: int,
    min_len: int, max_len: int, allowed_orients: List[str],
) -> Optional[List[Tuple[str, int, int, str]]]:
    """Try to place n_words in an n×n grid with at least one intersection.

    Returns list of (word, row, col, orient) or None if placement fails.
    """
    pool = [w for w in WORD_BANK if min_len <= len(w) <= max_len]
    if len(pool) < n_words:
        return None

    for _ in range(200):
        words_used = rng.sample(pool, n_words)
        grid = [[" "] * n for _ in range(n)]
        placements: List[Tuple[str, int, int, str]] = []
        ok = True

        for word in words_used:
            orient = rng.choice(allowed_orients)
            dr, dc = ORIENTATIONS[orient]
            L = len(word)
            valid = [(r, c) for r in range(n) for c in range(n)
                     if _can_place(n, word, r, c, dr, dc)]
            if not valid:
                ok = False
                break

            # Try positions that create intersections with existing words
            crossing = [
                (r, c) for (r, c) in valid
                if any(
                    grid[r + dr * i][c + dc * i] == word[i]
                    for i in range(L)
                    if grid[r + dr * i][c + dc * i] != " "
                )
            ]
            candidates = crossing if crossing else valid
            row, col = rng.choice(candidates)

            # Check no conflicts (different letter at same cell)
            conflict = any(
                grid[row + dr * i][col + dc * i] not in (" ", word[i])
                for i in range(L)
            )
            if conflict:
                ok = False
                break

            _place_word(grid, word, row, col, dr, dc)
            placements.append((word, row, col, orient))

        if not ok:
            continue

        # Verify at least one true intersection (shared cell between ≥2 words)
        if len(placements) >= 2:
            cells: Dict[Tuple[int, int], int] = {}
            for (w, r, c, o) in placements:
                dr2, dc2 = ORIENTATIONS[o]
                for i in range(len(w)):
                    key = (r + dr2 * i, c + dc2 * i)
                    cells[key] = cells.get(key, 0) + 1
            if any(v >= 2 for v in cells.values()):
                return placements

    return None


def _fill_noise(rng: random.Random, grid: List[List[str]]) -> None:
    """Fill remaining spaces with random letters."""
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == " ":
                grid[r][c] = rng.choice(string.ascii_uppercase)


def _grid_from_placements(n: int, placements: List[Tuple[str, int, int, str]],
                           rng: random.Random) -> List[List[str]]:
    grid = [[" "] * n for _ in range(n)]
    for (w, r, c, o) in placements:
        _place_word(grid, w, r, c, *ORIENTATIONS[o])
    _fill_noise(rng, grid)
    return grid


def _blank_grid(n: int, placements: List[Tuple[str, int, int, str]],
                target_word: str) -> List[List[str]]:
    """For type B: show only intersection cells; blank out the target word's cells."""
    grid = [[" "] * n for _ in range(n)]
    # Place all words
    for (w, r, c, o) in placements:
        _place_word(grid, w, r, c, *ORIENTATIONS[o])
    # Find intersection cells (shared by ≥2 words)
    cells: Dict[Tuple[int, int], int] = {}
    for (w, r, c, o) in placements:
        dr, dc = ORIENTATIONS[o]
        for i in range(len(w)):
            key = (r + dr * i, c + dc * i)
            cells[key] = cells.get(key, 0) + 1
    crossing = {k for k, v in cells.items() if v >= 2}
    # Blank out target word cells that are NOT intersections
    tw, tr, tc, to = next(p for p in placements if p[0] == target_word)
    dr, dc = ORIENTATIONS[to]
    L = len(tw)
    for i in range(L):
        r2, c2 = tr + dr * i, tc + dc * i
        if (r2, c2) not in crossing:
            grid[r2][c2] = "_"
    # Fill remaining spaces with noise
    for r in range(n):
        for c in range(n):
            if grid[r][c] == " ":
                grid[r][c] = random.choice(string.ascii_uppercase)
    return grid


def _render_blank(grid: List[List[str]]) -> str:
    return "\n".join(" ".join(cell for cell in row) for row in grid)


# ── Task makers ───────────────────────────────────────────────────────────────

def make_type_a_task(rng: random.Random, level: int, task_idx: int,
                     n: int, n_words: int, min_len: int, max_len: int,
                     allowed_orients: List[str]) -> Optional[dict]:
    """Type A: enumerate all hidden words in the crossword grid."""
    result = _pick_words_with_crossing(rng, n, n_words, min_len, max_len, allowed_orients)
    if result is None:
        return None

    words = sorted(w for w, *_ in result)
    grid = _grid_from_placements(n, result, rng)
    grid_str = _render(grid)

    prompt = (
        f"Below is a {n}×{n} letter grid. Rows are separated by newlines; "
        f"letters within a row are separated by single spaces.\n\n"
        f"{grid_str}\n\n"
        f"Exactly {n_words} words are hidden in this grid. They may run horizontally, "
        f"vertically, or diagonally, forwards or backwards, and some share cells where "
        f"they intersect. List all {n_words} words, one per line, in uppercase."
    )
    # Rubric: all words must appear in the response (order-independent)
    rubric_parts = [rf"\b{w}\b" for w in words]
    rubric_value = "(?=.*" + ")(?=.*".join(rubric_parts) + ")"

    return {
        "id": f"xword_A_L{level}_{n}x{n}_enum_{task_idx:02d}",
        "category": "crossword_enum",
        "level": level,
        "type": "enumerate",
        "n_words": n_words,
        "prompt": prompt,
        "answer": "\n".join(words),
        "rubric": {"type": "regex", "value": rubric_value},
        "notes": (
            f"Level {level} crossword type A, n={n}, {n_words} words: {', '.join(words)}."
        ),
    }


def make_type_b_task(rng: random.Random, level: int, task_idx: int,
                     n: int, n_words: int, min_len: int, max_len: int,
                     allowed_orients: List[str]) -> Optional[dict]:
    """Type B: fill-in-the-blank crossword (output the target word)."""
    result = _pick_words_with_crossing(rng, n, n_words, min_len, max_len, allowed_orients)
    if result is None:
        return None

    target = rng.choice(result)
    tw, tr, tc, to = target
    grid = _blank_grid(n, result, tw)
    grid_str = _render_blank(grid)

    prompt = (
        f"Below is a {n}×{n} crossword-style grid. Rows are separated by newlines; "
        f"letters within a row are separated by spaces. Rows and columns are 0-indexed. "
        f"'_' marks a hidden cell.\n\n"
        f"{grid_str}\n\n"
        f"A word starting at row={tr} col={tc} is placed {_orient_human(to)}. "
        f"Some of its letters are revealed where it crosses other words; the rest are hidden. "
        f"Output the complete word (uppercase, no spaces)."
    )

    return {
        "id": f"xword_B_L{level}_{n}x{n}_{to}_{task_idx:02d}",
        "category": "crossword_fill",
        "level": level,
        "type": "fill",
        "orientation": to,
        "n_words": n_words,
        "prompt": prompt,
        "answer": tw,
        "rubric": {"type": "exact", "value": tw},
        "notes": (
            f"Level {level} crossword type B, n={n}, {n_words} words, "
            f"target='{tw}' {to} at ({tr},{tc}). Fill blanks from crossing letters."
        ),
    }


# ── Level specs ───────────────────────────────────────────────────────────────

XWORD_LEVEL_SPECS = {
    #  level: (n, n_words, min_len, max_len, orients, type)
    8:  ( 7, 2, 4, 5, ["H_LR", "V_TD"],                        "A"),
    9:  ( 8, 3, 4, 6, ["H_LR", "V_TD", "H_RL"],                "A"),
    10: ( 9, 3, 4, 6, ["H_LR", "H_RL", "V_TD", "V_BU"],        "B"),
    11: (10, 4, 5, 7, ["H_LR", "H_RL", "V_TD", "V_BU", "D_DR", "D_DL"], "B"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate crossword tasks (levels 8–11).")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", required=True)

    group = ap.add_mutually_exclusive_group()
    group.add_argument("--level", type=int, choices=list(XWORD_LEVEL_SPECS))
    group.add_argument("--all-levels", action="store_true")

    ap.add_argument("--per-level", type=int, default=6)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    levels = list(XWORD_LEVEL_SPECS) if args.all_levels else [args.level or 8]

    tasks: List[dict] = []
    for level in levels:
        n, n_words, mn, mx, orients, typ = XWORD_LEVEL_SPECS[level]
        maker = make_type_a_task if typ == "A" else make_type_b_task
        attempts = 0
        generated = 0
        while generated < args.per_level and attempts < args.per_level * 20:
            attempts += 1
            t = maker(rng, level, generated, n, n_words, mn, mx, orients)
            if t is not None:
                tasks.append(t)
                generated += 1
        if generated < args.per_level:
            print(f"[gen] WARNING: L{level} only got {generated}/{args.per_level} tasks")

    with open(args.out, "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"[gen] wrote {len(tasks)} crossword tasks → {args.out}")
    by_level: Dict[int, int] = {}
    for t in tasks:
        by_level[t["level"]] = by_level.get(t["level"], 0) + 1
    for lv in sorted(by_level):
        spec = XWORD_LEVEL_SPECS[lv]
        print(f"  L{lv} (type-{spec[5]}, {spec[0]}×{spec[0]}, {spec[1]} words): {by_level[lv]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
