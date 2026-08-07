#!/usr/bin/env python3
"""H12b — Multi-slot working memory probe dataset generator.

Tests whether RWKV-7 WKV state can hold K independent information slots
simultaneously, without slot cross-contamination.

Protocol:
  - Generate K independent "tracks" (e.g., name→colour associations).
  - Interleave all K tracks in a single prompt sequence (round-robin).
  - At probe point, ask about a specific (track, position) — e.g.
    "What colour does Alice have?" where Alice appeared in track 2.
  - Correct = model retrieves the right value without confabulating from
    another track.

H12a measures depth (how far back in one track); H12b measures width
(how many parallel tracks before cross-contamination).

Output: JSONL file, one probe per line:
  {"id": "ms_k2_p3_t1", "prompt": "...", "answer": "blue",
   "n_slots": 2, "n_pairs": 3, "probe_slot": 1}
"""
import argparse
import json
import random
import sys

NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hal",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Ned", "Ora", "Pete",
    "Quinn", "Rosa", "Sam", "Tina", "Uri", "Vera", "Walt", "Xena",
    "Yara", "Zane", "Amy", "Ben", "Cleo", "Dan", "Ella", "Finn",
    "Gina", "Hugo", "Ivy", "Joel", "Kim", "Lars", "Mona", "Nick",
]

COLOURS = [
    "red", "blue", "green", "yellow", "purple", "orange", "pink",
    "brown", "black", "white", "cyan", "magenta", "teal", "lime",
    "navy", "maroon", "gold", "silver", "indigo", "violet",
    "coral", "crimson", "amber", "jade", "ruby", "cobalt", "sage",
    "lilac", "ivory", "ebony", "turquoise", "scarlet", "olive", "beige",
    "peach", "lavender", "ochre", "slate", "fuchsia", "azure",
]

ITEMS = [
    "hat", "scarf", "gloves", "bag", "jacket", "shoes", "ring",
    "watch", "belt", "tie", "socks", "coat", "vest", "cap", "mask",
]


def gen_probe(n_slots: int, n_pairs: int, probe_slot: int,
              rng: random.Random, idx: int) -> dict:
    """Generate one interleaved multi-slot probe."""
    assert 1 <= probe_slot <= n_slots

    # Assign unique names and colours to each (slot, pair)
    all_names = rng.sample(NAMES, n_slots * n_pairs)
    all_colours = rng.sample(COLOURS, n_slots * n_pairs)

    slots: list[list[tuple]] = []
    for s in range(n_slots):
        pairs = []
        for p in range(n_pairs):
            idx2 = s * n_pairs + p
            pairs.append((all_names[idx2], all_colours[idx2]))
        slots.append(pairs)

    # Build interleaved sequence (round-robin across slots)
    lines = []
    for p in range(n_pairs):
        for s in range(n_slots):
            name, colour = slots[s][p]
            lines.append(f"{name}'s favourite colour is {colour}.")

    # Probe: cloze completion — works for base models without instruction-following.
    # The model must fill in the blank after "is " by continuing the pattern.
    target_name, target_colour = slots[probe_slot - 1][-1]
    prompt = (
        "\n".join(lines)
        + f"\n{target_name}'s favourite colour is"
    )

    return {
        "id": f"ms_k{n_slots}_p{n_pairs}_s{probe_slot}_{idx:04d}",
        "prompt": prompt,
        "answer": target_colour,
        "n_slots": n_slots,
        "n_pairs": n_pairs,
        "probe_slot": probe_slot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--k-values", nargs="+", type=int, default=[2, 4, 8],
                    help="Slot counts to generate (H12b width axis).")
    ap.add_argument("--p-values", nargs="+", type=int, default=[1, 2, 4],
                    help="Pairs per slot (sequence depth per track).")
    ap.add_argument("--n-per-cell", type=int, default=20,
                    help="Probes per (K, P, probe_slot) cell.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    probes = []

    for k in args.k_values:
        for p in args.p_values:
            for probe_slot in range(1, k + 1):
                for i in range(args.n_per_cell):
                    probes.append(gen_probe(k, p, probe_slot, rng, i))

    rng.shuffle(probes)

    with open(args.out, "w") as f:
        for probe in probes:
            f.write(json.dumps(probe) + "\n")

    print(f"Generated {len(probes)} probes → {args.out}", file=sys.stderr)
    print(f"  K={args.k_values}, P={args.p_values}, "
          f"{args.n_per_cell} per cell", file=sys.stderr)


if __name__ == "__main__":
    main()
