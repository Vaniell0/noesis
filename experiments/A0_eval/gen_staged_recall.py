#!/usr/bin/env python3
"""gen_staged_recall.py — training/eval data where the answer can ONLY come from state.

The first training candidate of the redefined RL track (memory:
project_noesis_rl_attachment_plan). Key->value pairs are written in a window,
a neutral filler gap follows, and only then is one key queried. The answer
token is ordinary next-token CE — the attachment point is the output — but the
task is built so that the output cannot be produced without having kept the
pair separable in the WKV state across the gap. It is the "design the task, not
the objective" rule (memory: project_noesis_carry_ambiguity_task) applied to
the quantity measured on 2026-10-01: the state is used but mixed, with only 1-2
facts linearly separable (memory: project_noesis_usable_capacity).

Three arms, identical in everything except where the pairs sit:

  recall    [pairs][gap][query]   the measured condition
  inwindow  [gap][pairs][query]   ceiling: same tokens, pairs right before the
                                  query, so nothing has to survive a gap
  blank     [filler][gap][query]  floor: pairs replaced by filler of the same
                                  word count, the answer is not in the input

The gap and the query are identical across arms for a given seed, so a
difference between arms is the pairs' position and nothing else. Values are
uniform 0..99 and keys carry no information about them, so the query cannot be
answered from the query itself (the frozen-readout rule from H25).

Output rows: {"id", "arm", "n_pairs", "gap_words", "key", "prompt", "answer",
"text"}. `text` = prompt + answer for plain CE; `prompt`/`answer` are separate
for masked loss and for candidate-ranking eval.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Same 32 keys as experiments/rl/state_capacity_probe.py, so results from the
# capacity probe and from this data speak about the same tokens.
KEYS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
        "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
        "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
        "victor", "whiskey", "xray", "yankee", "zulu", "anchor", "beacon",
        "cipher", "domino", "ember", "falcon"]

# Filler vocabulary: no digits, no key words, nothing that resembles a binding.
_SUBJ = ["the river", "a quiet street", "the old library", "the morning train",
         "a small garden", "the harbour", "an empty field", "the kitchen",
         "the northern road", "a wooden bridge", "the market", "the hillside"]
_VERB = ["looks", "seems", "stays", "feels", "grows", "turns", "becomes",
         "remains", "appears"]
_ADJ = ["calm", "grey", "busy", "warm", "silent", "bright", "narrow", "cold",
        "green", "distant", "still", "crowded", "pale", "soft"]
_TAIL = ["after the rain", "in the evening", "during the week", "at dawn",
         "under the clouds", "near the coast", "by the end of summer",
         "for a while", "as usual", "on most days"]


def _filler(rng: random.Random, n_words: int) -> str:
    out: list[str] = []
    while len(out) < n_words:
        s = (f"{rng.choice(_SUBJ).capitalize()} {rng.choice(_VERB)} "
             f"{rng.choice(_ADJ)} {rng.choice(_TAIL)}.")
        out.extend(s.split())
    return " ".join(out[:n_words])


# Compositional keys for the "shared" style. Each key is an (adjective, noun)
# pair drawn from a SMALL vocabulary, so every word appears in several keys
# bound to different values. A reader that looks a key up by one word gets the
# wrong value; only binding the combination works.
#
# Why this exists (measured 2026-10-02, g1d-0.4b): with distinct single-word
# keys the base model already scores recall 0.989 / inwindow 0.967 / blank
# 0.011 — exact-key associative recall is what RWKV-7's delta rule is built
# for (the state update is an SGD step fitting S·k ≈ v), so that task has no
# headroom and does not force interference. Shared components put several
# bindings on the same key directions — the user's "paths" — which is the
# interference the readout lens measured.
_KEY_ADJ = ["red", "blue", "green", "black", "white", "brown", "grey", "pink"]
_KEY_NOUN = ["fox", "owl", "cat", "bee", "elk", "ram", "yak", "eel"]


def _keys(rng: random.Random, n_pairs: int, style: str, vocab: int) -> list:
    if style == "distinct":
        return rng.sample(KEYS, n_pairs)
    adj, noun = _KEY_ADJ[:vocab], _KEY_NOUN[:vocab]
    combos = [f"{a} {b}" for a in adj for b in noun]
    if n_pairs > len(combos):
        raise ValueError(f"{n_pairs} pairs need vocab >= {int(n_pairs ** 0.5) + 1}")
    return rng.sample(combos, n_pairs)


# Filler words that match one pair in World tokens, per key style (measured).
_WORDS_PER_PAIR = {"distinct": 3.5, "shared": 3.75}


def make_item(rng: random.Random, idx: int, n_pairs: int, gap_words: int,
              arm: str, key_style: str = "distinct", vocab: int = 4) -> dict:
    keys = _keys(rng, n_pairs, key_style, vocab)
    vals = [rng.randint(0, 99) for _ in keys]
    order = list(range(n_pairs))
    rng.shuffle(order)
    pairs = " ".join(f"{keys[i]}={vals[i]}" for i in order)
    write = f"Record the values.\n{pairs}\nEnd of record."
    gap = _filler(rng, gap_words) if gap_words else ""
    q = rng.randrange(n_pairs)
    query = f"Recall: {keys[q]}="
    answer = str(vals[q])

    if arm == "recall":
        parts = [write, gap, query]
    elif arm == "inwindow":
        parts = [gap, write, query]
    elif arm == "blank":
        # Length-matched to the write block in TOKENS, not words: one pair like
        # `victor=96` is ~4 World tokens and a filler word ~1.1, so 2 words per
        # pair left this arm 4-13 tokens short (measured). 3.5 brings it level.
        blank = "Record the values.\n" \
                + _filler(rng, int(n_pairs * _WORDS_PER_PAIR[key_style])) \
                + "\nEnd of record."
        parts = [blank, gap, query]
    else:
        raise ValueError(arm)
    prompt = "\n".join(p for p in parts if p)
    return {"id": f"sr_{key_style}{vocab if key_style == 'shared' else ''}_{arm}"
                  f"_n{n_pairs}_g{gap_words}_{idx:06d}",
            "arm": arm, "key_style": key_style, "vocab": vocab,
            "n_pairs": n_pairs, "gap_words": gap_words, "key": keys[q],
            "prompt": prompt, "answer": answer, "text": prompt + answer}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("training/corpus_open"))
    ap.add_argument("--name", default="staged_recall")
    ap.add_argument("--n-pairs", default="2,4,8")
    ap.add_argument("--gap-words", default="0,50,200")
    ap.add_argument("--arms", default="recall,inwindow,blank")
    ap.add_argument("--train-per-cell", type=int, default=400,
                    help="Rows per (arm, n_pairs, gap) cell in the TRAIN split. "
                         "Train is recall-only by default — see --train-arms.")
    ap.add_argument("--eval-per-cell", type=int, default=40)
    ap.add_argument("--train-arms", default="recall",
                    help="Controls are for evaluation; training on them would "
                         "teach the floor arm to guess.")
    ap.add_argument("--key-style", default="distinct", choices=("distinct", "shared"),
                    help="'shared' = compositional (adjective noun) keys from a "
                         "small vocabulary, so words recur across bindings and "
                         "only the combination identifies a value.")
    ap.add_argument("--vocab", type=int, default=4,
                    help="Words per slot for --key-style shared (combos = vocab^2). "
                         "Smaller = more sharing = more interference.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_list = [int(x) for x in args.n_pairs.split(",")]
    g_list = [int(x) for x in args.gap_words.split(",")]
    arms = [a.strip() for a in args.arms.split(",")]
    train_arms = [a.strip() for a in args.train_arms.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split, per_cell, use_arms, seed in (
            ("train", args.train_per_cell, train_arms, args.seed),
            ("eval", args.eval_per_cell, arms, args.seed + 10_000)):
        rows, idx = [], 0
        for arm in use_arms:
            for n in n_list:
                for g in g_list:
                    # one RNG per cell, seeded identically across arms, so the
                    # same (n, g, k) produces the same gap and query in every arm
                    for k in range(per_cell):
                        rng = random.Random(f"{seed}-{n}-{g}-{k}")
                        rows.append(make_item(rng, idx, n, g, arm,
                                              args.key_style, args.vocab))
                        idx += 1
        random.Random(seed).shuffle(rows)
        path = args.out_dir / f"{args.name}_{split}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[gen] {split}: {len(rows)} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
