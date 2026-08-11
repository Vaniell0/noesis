"""Generate bit-substitution lookup tasks for N-stabilization.

Tasks that require N=2 re-read to solve: a substitution table is given
in the prompt, followed by a sequence to decode. At N=1 the model must
hold the full table and decode simultaneously; at N=2 it reads the table
on the first pass (accumulating it into WKV state) and decodes on the second.

Difficulty scales with table length (4, 6, 8, 12, 16 symbols) and sequence
length (2–8 symbols to decode). Tasks are designed so N=1 is at the edge
of WKV working memory capacity, making N=2 reliably better.

No <think> spans — these are direct decode tasks. state_mask is all-zero.
L_state fires at ε_out = 0.05α level (outside think spans).

Output JSONL schema (tokenize_plain_cot.py compatible):
  {"id": str, "system": str, "user": str, "think": str, "answer": str}

Usage:
    training/.venv/bin/python training/scripts/gen_bitsub_tasks.py \\
        --out training/corpus_open/bitsub_train.jsonl \\
        --n 100 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path

SYSTEM = "Decode the sequence using the substitution table. Output only the decoded result."

# Symbol sets for lookup tables
GREEK_LOWER = list("αβγδεζηθικλμνξοπρστυφχψω")
SYMBOLS_ASCII = list("@#$%&*!?+=~^<>{}[]|")
PHONETIC_NATO = [
    "Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot",
    "Golf", "Hotel", "India", "Juliet", "Kilo", "Lima",
    "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo",
    "Sierra", "Tango", "Uniform", "Victor", "Whiskey", "Xray",
    "Yankee", "Zulu",
]
EMOJI_CODES = list("☀☁☂☃☄★☆♠♣♥♦♪♫♬♭♮♯⚡⚀⚁⚂⚃⚄⚅")

SYMBOL_SETS = [
    ("greek", GREEK_LOWER),
    ("ascii", SYMBOLS_ASCII),
    ("nato", PHONETIC_NATO),
    ("emoji", EMOJI_CODES),
]

ALPHABET = list(string.ascii_lowercase)


def _build_table(symbols: list, n: int, rng: random.Random) -> dict:
    """Return symbol→letter mapping for n symbols."""
    chosen_syms = rng.sample(symbols, min(n, len(symbols)))
    chosen_letters = rng.sample(ALPHABET, len(chosen_syms))
    return dict(zip(chosen_syms, chosen_letters))


def _format_table(table: dict, sym_type: str) -> str:
    """Format table as 'sym=letter, sym=letter, ...'."""
    entries = [f"{s}={c}" for s, c in table.items()]
    return ", ".join(entries)


def _encode(table: dict, word: str) -> list[str]:
    """Encode word as list of symbols using reverse table."""
    rev = {c: s for s, c in table.items()}
    return [rev[c] for c in word if c in rev]


def _build_item(
    sym_type: str,
    symbols: list,
    table_size: int,
    seq_len: int,
    rng: random.Random,
    idx: int,
) -> dict | None:
    """Build one decode task. Returns None if table is too small."""
    if table_size > len(symbols):
        return None

    table = _build_table(symbols, table_size, rng)
    # Pick a word using only letters in the table
    available = list(table.values())
    if len(available) < seq_len:
        return None

    seq_letters = [rng.choice(available) for _ in range(seq_len)]
    target_word = "".join(seq_letters)

    # Encode
    encoded = _encode(table, target_word)
    if not encoded or len(encoded) != seq_len:
        return None

    # Format prompt
    if sym_type == "nato":
        encoded_str = "-".join(encoded)
        table_str = _format_table(table, sym_type)
        prompt = (
            f"Given the substitution table {table_str}, "
            f"decode the following: {encoded_str}. "
            f"Output only the decoded word."
        )
    else:
        encoded_str = "".join(encoded)
        table_str = _format_table(table, sym_type)
        prompt = (
            f"Given the substitution table {table_str}, "
            f"decode the following: {encoded_str}. "
            f"Output only the decoded word."
        )

    return {
        "id": f"bitsub_{sym_type}_t{table_size}_s{seq_len}_{idx:04d}",
        "system": SYSTEM,
        "user": prompt,
        "think": "",
        "answer": target_word,
        "_meta": {
            "table_size": table_size,
            "seq_len": seq_len,
            "sym_type": sym_type,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100, help="Total items to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-table", type=int, default=4)
    ap.add_argument("--max-table", type=int, default=16)
    ap.add_argument("--min-seq", type=int, default=2)
    ap.add_argument("--max-seq", type=int, default=6)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    items = []
    attempts = 0
    table_sizes = list(range(args.min_table, args.max_table + 1, 2))  # 4,6,8,10,12,14,16
    seq_lens = list(range(args.min_seq, args.max_seq + 1))

    while len(items) < args.n and attempts < args.n * 20:
        attempts += 1
        sym_type, symbols = rng.choice(SYMBOL_SETS)
        table_size = rng.choice(table_sizes)
        seq_len = rng.choice(seq_lens)
        item = _build_item(sym_type, symbols, table_size, seq_len, rng, len(items))
        if item is not None:
            items.append(item)

    rng.shuffle(items)

    written = 0
    with out.open("w") as f:
        for item in items:
            out_item = {k: v for k, v in item.items() if not k.startswith("_")}
            f.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            written += 1

    # Stats
    by_size = {}
    for it in items:
        ts = it["_meta"]["table_size"]
        by_size[ts] = by_size.get(ts, 0) + 1

    print(f"Written {written} bitsub items → {out}")
    print("Table size distribution:", dict(sorted(by_size.items())))


if __name__ == "__main__":
    main()
