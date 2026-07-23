#!/usr/bin/env python3
"""A0.H12a — cross-linking triple task generator.

Emits JSONL files for two disjoint sweeps:

- **Width sweep** — ``tasks-N{n}.jsonl`` for ``n ∈ {4, 8, 16, 32, 64}``.
  Each task lists ``n`` items with a colour; some items share a colour
  and the model must enumerate all pairs that share.
- **Distance sweep** — ``tasks-dist-{gap}.jsonl`` for
  ``gap ∈ {50, 200, 500, 1000}`` at fixed ``N=16``. Same structure as
  the width sweep at N=16, but filler prose is inserted between item
  lines so the *mean word-distance* between the two members of a shared-
  colour pair is approximately ``gap`` words.

Word-distance is used as a coarse token-distance proxy. Report the
mean measured distance in the summary; the intent is monotone
increase across gap values, not exact token counts.

Determinism is by seed: for a given
``(seed, n, planted_pairs, gap)`` the emitted task set is bit-exact.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from dataclasses import dataclass
from typing import List, Tuple


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
    "inland", "desert", "forest", "prairie", "highland", "lowland",
    "urban", "rural", "arid", "humid", "temperate", "boreal2",
    "frost", "ember", "iron", "quartz", "onyx", "jade", "amber",
    "coral2", "pearl", "opal", "flint", "granite", "basalt", "chalk",
    "sandstone", "shale",
]


def _item_name(idx: int) -> str:
    return f"item-{ITEM_PREFIXES[idx % len(ITEM_PREFIXES)]}-{idx:02d}"


@dataclass
class Task:
    id: str
    prompt: str
    expected_pairs: List[Tuple[str, str]]
    n: int
    mean_word_gap: float
    seed: int
    variant: str  # "width" or "dist"

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "expected_pairs": [list(p) for p in self.expected_pairs],
            "n": self.n,
            "mean_word_gap": self.mean_word_gap,
            "seed": self.seed,
            "variant": self.variant,
        }


def _planted_colour_assignment(rng: random.Random, n: int, n_pairs: int) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Return (colour_per_item, planted_pair_indices).

    Plants exactly ``n_pairs`` colour-pairs (each a two-item share) among
    ``n`` items. All other items receive distinct singleton colours. This
    keeps the number of *expected* pairs deterministic across the sweep,
    which is required for the falsification decision tree — otherwise a
    sweep at larger N would also be a sweep over the number of correct
    answers.
    """
    if 2 * n_pairs > n:
        raise ValueError(f"n_pairs={n_pairs} too large for n={n}; need 2*n_pairs <= n")
    # Distinct colours needed: n_pairs (shared) + (n - 2*n_pairs) singletons.
    distinct_needed = n_pairs + (n - 2 * n_pairs)
    if distinct_needed > len(COLOURS):
        raise ValueError(
            f"colour vocabulary too small (have {len(COLOURS)}, need {distinct_needed})"
        )
    positions = list(range(n))
    rng.shuffle(positions)
    colour_of = [None] * n
    used_colours: List[str] = []
    pool = list(COLOURS)
    rng.shuffle(pool)
    for k in range(n_pairs):
        c = pool.pop()
        used_colours.append(c)
        i, j = positions[2 * k], positions[2 * k + 1]
        colour_of[i] = c
        colour_of[j] = c
    for idx in positions[2 * n_pairs:]:
        colour_of[idx] = pool.pop()
    planted_pairs: List[Tuple[int, int]] = []
    for k in range(n_pairs):
        i, j = positions[2 * k], positions[2 * k + 1]
        lo, hi = min(i, j), max(i, j)
        planted_pairs.append((lo, hi))
    return colour_of, planted_pairs


def _distractor_line(rng: random.Random) -> str:
    kind = rng.choice(["weather", "reading", "checksum"])
    if kind == "weather":
        t = rng.uniform(-5, 35)
        h = rng.randint(20, 90)
        p = rng.randint(980, 1030)
        return f"(reading {rng.randint(1, 999)}: temperature {t:.1f}C, humidity {h}%, pressure {p} hPa)"
    if kind == "reading":
        return f"(instrument log {rng.randint(1, 999)}: value {rng.uniform(0, 100):.2f}, tolerance {rng.uniform(0.1, 5):.2f})"
    return f"(checksum {rng.randint(1, 999)}: parity even, hash-suffix {rng.randint(0, 65535):04x})"


def _word_count(s: str) -> int:
    return len(s.split())


def _render_task(colour_of: List[str], distractor_words_between_items: int,
                 planted_pairs: List[Tuple[int, int]], rng: random.Random) -> Tuple[str, float]:
    """Render the prompt; return (prompt, mean_word_gap_for_planted_pairs).

    ``distractor_words_between_items`` is the target number of filler
    words inserted between consecutive item lines. When the gap is 0 the
    prompt is compact (width-sweep baseline).
    """
    n = len(colour_of)
    lines = [
        "You are given a list of items and their colours.",
        "Some items share a colour with another item; the rest have unique colours.",
        "Your task: list every pair of items that share a colour.",
        "",
        "Items:",
    ]
    # Track *word* positions of each item line's item-name so we can
    # compute the mean gap between planted pairs after the fact.
    positions: List[int] = []
    for i, c in enumerate(colour_of):
        # Insert distractor block between items (not before the first).
        if i > 0 and distractor_words_between_items > 0:
            words_so_far = 0
            while words_so_far < distractor_words_between_items:
                d = _distractor_line(rng)
                lines.append(d)
                words_so_far += _word_count(d)
        item_line = f"- {_item_name(i)} has colour {c}."
        prefix_word_count = sum(_word_count(l) for l in lines)
        positions.append(prefix_word_count)
        lines.append(item_line)
    lines += [
        "",
        "Question: list every pair of items that share the same colour, one pair per line, "
        'in the form "item-X, item-Y". Output only the pairs, nothing else.',
    ]
    prompt = "\n".join(lines)
    gaps = [abs(positions[j] - positions[i]) for (i, j) in planted_pairs]
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    return prompt, mean_gap


def _pair_names(pairs: List[Tuple[int, int]]) -> List[Tuple[str, str]]:
    return [(_item_name(i), _item_name(j)) for (i, j) in pairs]


def gen_width_tasks(n: int, n_pairs: int, seeds: List[int]) -> List[Task]:
    out: List[Task] = []
    for s in seeds:
        rng = random.Random(s * 1_000 + n)
        colour_of, planted = _planted_colour_assignment(rng, n=n, n_pairs=n_pairs)
        prompt, gap = _render_task(colour_of, distractor_words_between_items=0,
                                   planted_pairs=planted, rng=rng)
        out.append(Task(
            id=f"h12a-w-n{n}-s{s}",
            prompt=prompt,
            expected_pairs=_pair_names(planted),
            n=n,
            mean_word_gap=gap,
            seed=s,
            variant="width",
        ))
    return out


def gen_distance_tasks(n: int, n_pairs: int, gap_words: int, seeds: List[int]) -> List[Task]:
    out: List[Task] = []
    # We want mean word-gap between paired items ≈ gap_words. Since
    # planted pairs are at random positions after shuffle, distractor
    # blocks between consecutive items of size d give a paired-item gap
    # of approximately (avg pair position distance) * (item_line_words + d).
    # Rather than solve analytically, we pick d such that d ≈ gap_words / avg_pair_step
    # and let the render pass measure the actual mean gap.
    for s in seeds:
        rng = random.Random(s * 1_000 + n + gap_words * 10_000)
        colour_of, planted = _planted_colour_assignment(rng, n=n, n_pairs=n_pairs)
        # avg pair "step" = expected |j - i| for two random distinct indices in [0, n) is ~(n+1)/3.
        step = max(1, (n + 1) / 3)
        d = max(0, int(round(gap_words / step)) - 6)  # 6 = crude item-line words
        prompt, measured_gap = _render_task(
            colour_of, distractor_words_between_items=d,
            planted_pairs=planted, rng=rng,
        )
        out.append(Task(
            id=f"h12a-d-g{gap_words}-s{s}",
            prompt=prompt,
            expected_pairs=_pair_names(planted),
            n=n,
            mean_word_gap=measured_gap,
            seed=s,
            variant="dist",
        ))
    return out


def _write_tasks(tasks: List[Task], path: pathlib.Path) -> None:
    with path.open("w") as f:
        for t in tasks:
            f.write(json.dumps(t.to_json()) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="A0.H12a triple task generator.")
    ap.add_argument("--out-dir", type=pathlib.Path, default=None,
                    help="Output directory (default: alongside this script).")
    ap.add_argument("--seeds", type=int, default=5,
                    help="Number of seeds per task cell.")
    ap.add_argument("--n-pairs", type=int, default=2,
                    help="Planted colour-pairs per task (default: 2). "
                         "Kept constant across the sweep so answer complexity is not confounded.")
    ap.add_argument("--width-ns", type=int, nargs="+", default=[4, 8, 16, 32, 64],
                    help="N values for the width sweep.")
    ap.add_argument("--dist-gaps", type=int, nargs="+", default=[50, 200, 500, 1000],
                    help="Target word-gap values for the distance sweep.")
    ap.add_argument("--dist-n", type=int, default=16,
                    help="Fixed N for the distance sweep. Default 16 matches "
                         "the original design; drop to 8 if width sweep shows "
                         "the model floors before N=16.")
    args = ap.parse_args()

    here = pathlib.Path(__file__).resolve().parent
    out_dir = args.out_dir or (here / "tasks")
    out_dir.mkdir(exist_ok=True)

    seeds = list(range(args.seeds))
    for n in args.width_ns:
        tasks = gen_width_tasks(n=n, n_pairs=args.n_pairs, seeds=seeds)
        _write_tasks(tasks, out_dir / f"tasks-N{n}.jsonl")
        gaps = [t.mean_word_gap for t in tasks]
        print(f"  width  N={n:>3}  {len(tasks)} tasks   mean-gap={sum(gaps)/len(gaps):.1f}w")

    for g in args.dist_gaps:
        tasks = gen_distance_tasks(n=args.dist_n, n_pairs=args.n_pairs, gap_words=g, seeds=seeds)
        _write_tasks(tasks, out_dir / f"tasks-dist-{g}.jsonl")
        gaps = [t.mean_word_gap for t in tasks]
        print(f"  dist   g={g:>4}  {len(tasks)} tasks   mean-gap={sum(gaps)/len(gaps):.1f}w (target ~{g})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
