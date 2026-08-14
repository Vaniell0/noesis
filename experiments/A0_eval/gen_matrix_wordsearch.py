#!/usr/bin/env python3
"""Generate matrix_wordsearch tasks for A0.2 / RL curriculum.

Difficulty levels (--level N, or mixed via --tiers):

  Level 1  5×5  H_LR only               3–4 letters   (easiest)
  Level 2  6×6  H_LR + V_TD             4–5 letters
  Level 3  7×7  H_LR + V_TD + H_RL      4–5 letters
  Level 4  8×8  H + V + H_RL + V_BU     5–6 letters
  Level 5  9×9  all 4 axes ± reverse     5–6 letters
  Level 6 10×10 all 4 axes + diagonals   5–7 letters
  Level 7 12×12 all 8 directions         6–8 letters   (hardest)

Within each level multiple orientations are drawn uniformly at random
so the eval/RL set mixes types. Use --seed for reproducibility.

Usage:
  # Single level
  python3 gen_matrix_wordsearch.py --level 3 --per-level 12 \\
      --out tasks_matrix_wordsearch.jsonl

  # All levels, 8 tasks each
  python3 gen_matrix_wordsearch.py --all-levels --per-level 8 \\
      --out tasks_matrix_wordsearch.jsonl

  # Custom tier spec (backward-compat)
  python3 gen_matrix_wordsearch.py \\
      --tiers 5x5:3:4:H_LR,7x7:4:6:H_LR+V_TD+H_RL \\
      --per-tier 6 --out tasks_matrix_wordsearch.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import string
from typing import List, Optional, Tuple

# ── Word bank ─────────────────────────────────────────────────────────────────
WORD_BANK = [
    # 3
    "CAT", "DOG", "SUN", "MAP", "KEY", "OWL", "FOX", "PEN", "ICE", "OAK",
    "ACE", "ARC", "BUD", "DEN", "ELK", "FIG", "GEM", "HUE", "JAW", "LAG",
    # 4
    "MOON", "TREE", "BIRD", "FIRE", "STAR", "LEAF", "WOLF", "SHIP", "GATE",
    "ROSE", "CAVE", "DAWN", "EDGE", "FARM", "GIFT", "HERB", "IRIS", "JADE",
    "KITE", "LIME", "MAST", "NEST", "OPAL", "PINE", "QUIZ", "REEF", "SAGE",
    "TUSK", "URGE", "VEIL", "WREN", "YOKE", "ZERO",
    # 5
    "RIVER", "STONE", "CLOUD", "STORM", "BEACH", "PIANO", "GLASS", "HORSE",
    "SNAKE", "FROST", "SMILE", "WATER", "ANGLE", "BLADE", "CEDAR", "DELTA",
    "EMBER", "FLUTE", "GRAZE", "HAVEN", "IVORY", "JOUST", "KNEEL", "LANCE",
    "MAIZE", "NOBLE", "ORBIT", "PLUME", "QUILL", "RADAR", "SPIRE", "TORCH",
    "UMBRA", "VISOR", "WHIRL", "XENON", "YACHT", "ZONAL",
    # 6
    "GARDEN", "TIMBER", "HOLLOW", "SILVER", "GOLDEN", "BREEZE", "FOREST",
    "MEADOW", "BRIDGE", "CANDLE", "FALCON", "PLANET", "ABLAZE", "BELFRY",
    "COBALT", "DEBRIS", "ENIGMA", "FALLOW", "GRAVEL", "HERALD", "INSECT",
    "JACKAL", "KELVIN", "LAGOON", "MAGNET", "NAPALM", "OBLONG", "PARDON",
    "QUARRY", "RADIAL", "SCHEMA", "TURRET", "UPLIFT", "VORTEX", "WALNUT",
    # 7
    "THUNDER", "HARVEST", "GRANITE", "COMPASS", "JUPITER", "MYSTERY",
    "SILENCE", "ALCHEMY", "BRIGADE", "CAPTAIN", "DORMANT", "ECLIPSE",
    "FEATHER", "GOBELIN", "HABITAT", "ICEBERG", "JOURNEY", "KINETIC",
    "LANTERN", "MONARCH", "NEUTRAL", "OBSCURE", "PATTERN", "QUANTUM",
    "RECRUIT", "SERPENT", "TABLEAU", "UNIFORM", "VAGRANT", "WARRANT",
    # 8
    "ABSOLUTE", "BALANCE", "CALENDAR", "DOMINANT", "ENTRANCE", "FRAGRANT",
    "GRADIENT", "HERITAGE", "INDUSTRY", "JUBILANT", "LABYRINTH"[:8],
    "MARITIME", "NAVIGATE", "OBSOLETE", "PARALLEL", "QUADRANT", "RELATIVE",
    "SPECTRUM", "TRIANGLE", "ULTIMATE", "VARIABLE", "WANDERER",
]

# ── Orientation definitions ────────────────────────────────────────────────────
# Each orientation: (name, dr, dc)
#   dr = row delta per letter, dc = col delta per letter
ORIENTATIONS = {
    "H_LR":  ( 0,  1),   # horizontal left→right
    "H_RL":  ( 0, -1),   # horizontal right→left  (reverse)
    "V_TD":  ( 1,  0),   # vertical top→down
    "V_BU":  (-1,  0),   # vertical bottom→up     (reverse)
    "D_DR":  ( 1,  1),   # diagonal down-right
    "D_DL":  ( 1, -1),   # diagonal down-left
    "D_UR":  (-1,  1),   # diagonal up-right
    "D_UL":  (-1, -1),   # diagonal up-left
}

LEVEL_SPECS = {
    # level: (grid_n, min_word, max_word, allowed_orientations)
    1: ( 5, 3, 4, ["H_LR"]),
    2: ( 6, 4, 5, ["H_LR", "V_TD"]),
    3: ( 7, 4, 5, ["H_LR", "V_TD", "H_RL"]),
    4: ( 8, 5, 6, ["H_LR", "H_RL", "V_TD", "V_BU"]),
    5: ( 9, 5, 6, ["H_LR", "H_RL", "V_TD", "V_BU", "D_DR", "D_DL"]),
    6: (10, 5, 7, ["H_LR", "H_RL", "V_TD", "V_BU", "D_DR", "D_DL", "D_UR", "D_UL"]),
    7: (12, 6, 8, ["H_LR", "H_RL", "V_TD", "V_BU", "D_DR", "D_DL", "D_UR", "D_UL"]),
}


def _pick_word(rng: random.Random, min_len: int, max_len: int) -> str:
    pool = [w for w in WORD_BANK if min_len <= len(w) <= max_len]
    if not pool:
        raise ValueError(f"no words with {min_len} <= len <= {max_len}")
    return rng.choice(pool)


def _can_place(n: int, word: str, row: int, col: int, dr: int, dc: int) -> bool:
    L = len(word)
    r_end = row + dr * (L - 1)
    c_end = col + dc * (L - 1)
    return 0 <= r_end < n and 0 <= c_end < n


def _place_word(grid: List[List[str]], word: str, row: int, col: int,
                dr: int, dc: int) -> None:
    for i, ch in enumerate(word):
        grid[row + dr * i][col + dc * i] = ch


def _find_word(grid: List[List[str]], word: str, dr: int, dc: int
               ) -> List[Tuple[int, int]]:
    n = len(grid)
    L = len(word)
    hits: List[Tuple[int, int]] = []
    for r in range(n):
        for c in range(n):
            if not _can_place(n, word, r, c, dr, dc):
                continue
            if all(grid[r + dr * i][c + dc * i] == word[i] for i in range(L)):
                hits.append((r, c))
    return hits


def _find_word_any_dir(grid: List[List[str]], word: str) -> List[Tuple[int, int, str]]:
    """Find all occurrences of word in any direction; return (r, c, orient)."""
    hits = []
    for oname, (dr, dc) in ORIENTATIONS.items():
        for (r, c) in _find_word(grid, word, dr, dc):
            hits.append((r, c, oname))
    return hits


def _fill_grid(rng: random.Random, n: int, word: str, row: int, col: int,
               dr: int, dc: int, orient_name: str) -> List[List[str]]:
    grid = [[rng.choice(string.ascii_uppercase) for _ in range(n)] for _ in range(n)]
    _place_word(grid, word, row, col, dr, dc)
    # Remove accidental collisions in any direction.
    for _ in range(12):
        extra = [(r2, c2, on) for (r2, c2, on) in _find_word_any_dir(grid, word)
                 if not (r2 == row and c2 == col and on == orient_name)]
        if not extra:
            break
        for (r2, c2, on2) in extra:
            dr2, dc2 = ORIENTATIONS[on2]
            i = rng.randrange(len(word))
            gr, gc = r2 + dr2 * i, c2 + dc2 * i
            if gr == row + dr * i and gc == col + dc * i:
                continue  # don't clobber the intended placement
            cands = [ch for ch in string.ascii_uppercase if ch != grid[gr][gc]]
            grid[gr][gc] = rng.choice(cands)
    return grid


def _render(grid: List[List[str]]) -> str:
    return "\n".join(" ".join(row) for row in grid)


def _orient_human(orient: str) -> str:
    return {
        "H_LR": "horizontally (left-to-right)",
        "H_RL": "horizontally (right-to-left)",
        "V_TD": "vertically (top-to-bottom)",
        "V_BU": "vertically (bottom-to-top)",
        "D_DR": "diagonally (down-right)",
        "D_DL": "diagonally (down-left)",
        "D_UR": "diagonally (up-right)",
        "D_UL": "diagonally (up-left)",
    }[orient]


def make_prompt(grid_str: str, word: str, n: int, orient: str) -> str:
    return (
        f"Below is a {n}x{n} letter matrix. Rows are separated by newlines; "
        f"letters within a row are separated by single spaces. Rows and columns "
        f"are 0-indexed (top-left is row=0 col=0).\n\n"
        f"{grid_str}\n\n"
        f"The word {word} appears exactly once, placed {_orient_human(orient)}. "
        f"Find it. Output only the position of the word's first letter in the "
        f"form 'row=R col=C'."
    )


def make_prompt_name(grid_str: str, n: int) -> str:
    return (
        f"Below is a {n}x{n} letter matrix. Rows are separated by newlines; "
        f"letters within a row are separated by single spaces.\n\n"
        f"{grid_str}\n\n"
        f"Exactly one English word is hidden in this grid — it may run in any "
        f"direction (horizontal, vertical, or diagonal, forwards or backwards). "
        f"Find it and output only the word in uppercase. Output nothing else."
    )


def make_task(rng: random.Random, level: int, tier_idx: int, task_idx: int,
              n: int, min_len: int, max_len: int,
              allowed_orientations: List[str],
              mode: str = "position") -> dict:
    orient = rng.choice(allowed_orientations)
    dr, dc = ORIENTATIONS[orient]
    word = _pick_word(rng, min_len, max_len)

    # Find valid anchor positions for this orientation and word length.
    L = len(word)
    valid = [(r, c) for r in range(n) for c in range(n)
             if _can_place(n, word, r, c, dr, dc)]
    if not valid:
        raise RuntimeError(f"no valid position for {word!r} orient={orient} n={n}")
    row, col = rng.choice(valid)

    grid = _fill_grid(rng, n, word, row, col, dr, dc, orient)
    grid_str = _render(grid)

    if mode == "name":
        prompt = make_prompt_name(grid_str, n)
        rubric_value = rf"(?i)\b{word}\b"
        return {
            "id": f"wsearch_name_L{level}_{n}x{n}_{orient}_{tier_idx:02d}_{task_idx:02d}",
            "category": "matrix_wordsearch_name",
            "level": level,
            "orientation": orient,
            "prompt": prompt,
            "answer": word,
            "rubric": {"type": "regex", "value": rubric_value},
            "notes": (
                f"Level {level}, n={n}, orient={orient}, "
                f"word='{word}' (len {len(word)}), anchor=(row={row}, col={col})."
            ),
        }

    prompt = make_prompt(grid_str, word, n, orient)
    rubric_value = rf"row\s*=\s*{row}\b[^0-9]*col\s*=\s*{col}\b"
    return {
        "id": f"wsearch_L{level}_{n}x{n}_{orient}_{tier_idx:02d}_{task_idx:02d}",
        "category": "matrix_wordsearch",
        "level": level,
        "orientation": orient,
        "prompt": prompt,
        "answer": f"row={row} col={col}",
        "rubric": {"type": "regex", "value": rubric_value},
        "notes": (
            f"Level {level}, n={n}, orient={orient}, "
            f"word='{word}' (len {len(word)}), anchor=(row={row}, col={col})."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate matrix_wordsearch tasks.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)

    group = ap.add_mutually_exclusive_group()
    group.add_argument("--level", type=int, choices=list(LEVEL_SPECS), metavar="N",
                       help="Generate tasks for a single level (1–7).")
    group.add_argument("--all-levels", action="store_true",
                       help="Generate tasks for all levels 1–7.")
    group.add_argument("--tiers",
                       help=(
                           "Custom tiers: comma-separated NxN:min:max:O1+O2+… "
                           "e.g. 7x7:4:6:H_LR+V_TD+H_RL"
                       ))

    ap.add_argument("--per-level", type=int, default=8,
                    help="Tasks per level (--level / --all-levels).")
    ap.add_argument("--per-tier", type=int, default=4,
                    help="Tasks per custom tier (--tiers).")
    ap.add_argument("--mode", choices=["position", "name"], default="position",
                    help="'position': output row=R col=C (default). "
                         "'name': model names the hidden word (no position hint).")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tasks: List[dict] = []

    if args.tiers:
        for ti, spec in enumerate(args.tiers.split(",")):
            parts = spec.split(":")
            n = int(parts[0].lower().split("x")[0])
            mn, mx = int(parts[1]), int(parts[2])
            orients = parts[3].split("+") if len(parts) > 3 else ["H_LR"]
            for tj in range(args.per_tier):
                tasks.append(make_task(rng, 0, ti, tj, n, mn, mx, orients, mode=args.mode))
    else:
        levels = list(LEVEL_SPECS) if args.all_levels else [args.level or 1]
        for level in levels:
            n, mn, mx, orients = LEVEL_SPECS[level]
            for tj in range(args.per_level):
                tasks.append(make_task(rng, level, level, tj, n, mn, mx, orients, mode=args.mode))

    with open(args.out, "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"[gen] wrote {len(tasks)} tasks → {args.out}")
    orient_counts: dict = {}
    for t in tasks:
        orient_counts[t.get("orientation", "?")] = orient_counts.get(t.get("orientation", "?"), 0) + 1
    for o, cnt in sorted(orient_counts.items()):
        print(f"  {o}: {cnt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
