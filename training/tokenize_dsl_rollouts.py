#!/usr/bin/env python3
"""Step 8 tokenizer: DSL-in-think rollouts with dual loss masks.

Reads rollouts produced by ``corpus/convert_anthropic_to_dsl.py`` (turns
with ``think_dsl`` fields) and emits packed .pt files with two masks:

- ``loss_mask``  (L_CE_SFT): 1 on the full <think>…</think> block
  (including tags, so the model learns when to open/close it).
- ``state_mask`` (L_state): 1 on the same <think>…</think> span.
  Used in step 8 training to restrict state-regularisation to think spans.

Both masks are identical for step 8. They are kept separate so that future
steps can decouple CE and state supervision independently.

## Input format

Each line in --input JSONL files is a rollout::

    {
        "id": str,
        "source": str,
        "turns": [
            {"role": "user",      "content": "..."},
            {"role": "assistant", "think_dsl": "tool_call ... \\ntool_result ...",
                                  "content": "optional prose"},
            ...
        ]
    }

## Output

    training/tokenised/<name>_train.pt
    training/tokenised/<name>_val.pt

Each is ``torch.save`` of::

    {
        "ids":        LongTensor [N],
        "loss_mask":  LongTensor [N],   # 1 on <think>…</think> spans
        "state_mask": LongTensor [N],   # identical for step 8
        "starts":     LongTensor [n_rollouts],
        "vocab":      "rwkv_vocab_v20230424",
    }

## Rendering

Each turn is rendered as a flat string then tokenized:

    user    →  <user>{content}                            (both masks=0)
    assistant with think_dsl:
              <assistant>                                  (masks=0)
              <think>{dsl}</think>                         (masks=1)
              {optional prose}                             (masks=0)
    assistant without think_dsl:
              <assistant>{content}                         (masks=0)

## Usage

    training/.venv/bin/python training/tokenize_dsl_rollouts.py \\
        --input training/corpus_open/action_chains_dsl.jsonl \\
        --name  action_chains_dsl_step8
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
OUT_DIR = _HERE / "tokenised"

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "0")

sys.path.insert(0, str(_HERE))
from tokenize_fixture import _VocabOnlyPipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

# Segment: (text, ce_mask, state_mask)
_Seg = tuple[str, int, int]


def _render_dsl_turns(turns: list[dict]) -> list[_Seg]:
    segs: list[_Seg] = []
    for t in turns:
        role = t.get("role", "")
        if role == "user":
            content = t.get("content", "")
            segs.append((f"<user>{content}", 0, 0))
        elif role == "assistant":
            think_dsl = t.get("think_dsl", "")
            content = t.get("content", "")
            segs.append(("<assistant>", 0, 0))
            if think_dsl:
                segs.append((f"<think>{think_dsl}</think>", 1, 1))
            if content:
                segs.append((content, 0, 0))
        # tool_result turns (legacy) — unsupervised context
        elif role == "tool_result":
            content = t.get("content", "")
            segs.append((f"<tool_result>{content}", 0, 0))
    return segs


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _val_bucket(rollout_id: str, val_pct: int) -> bool:
    h = hashlib.blake2b(rollout_id.encode("utf-8"), digest_size=4).hexdigest()
    return (int(h, 16) % 100) < val_pct


def _iter_rollouts(patterns: list[str]):
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)


def _tokenize_split(rollouts: list[dict], tok) -> dict:
    all_ids: list[int] = []
    all_ce: list[int] = []
    all_state: list[int] = []
    starts: list[int] = []
    n_sup = 0

    for rollout in rollouts:
        turns = rollout.get("turns", [])
        segs = _render_dsl_turns(turns)
        starts.append(len(all_ids))
        for text, ce, state in segs:
            ids = tok.encode(text)
            all_ids.extend(ids)
            all_ce.extend([ce] * len(ids))
            all_state.extend([state] * len(ids))
            if ce:
                n_sup += len(ids)

    return {
        "ids":        torch.tensor(all_ids,   dtype=torch.long),
        "loss_mask":  torch.tensor(all_ce,    dtype=torch.long),
        "state_mask": torch.tensor(all_state, dtype=torch.long),
        "starts":     torch.tensor(starts,    dtype=torch.long),
        "vocab":      "rwkv_vocab_v20230424",
        "_n_supervised": n_sup,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", action="append", required=True,
                    help="rollouts JSONL path or glob (repeatable)")
    ap.add_argument("--name", required=True,
                    help="output basename (no extension)")
    ap.add_argument("--val-pct", type=int, default=2,
                    help="percent of rollouts held for validation (default 2)")
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help="output directory")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rollout count (debug)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = _VocabOnlyPipeline()
    train_rollouts: list[dict] = []
    val_rollouts: list[dict] = []
    total = 0

    for rollout in _iter_rollouts(args.input):
        total += 1
        rid = str(rollout.get("id", f"row_{total}"))
        if _val_bucket(rid, args.val_pct):
            val_rollouts.append(rollout)
        else:
            train_rollouts.append(rollout)
        if args.limit and total >= args.limit:
            break

    if not train_rollouts:
        print("no rollouts", file=sys.stderr)
        return 1

    train_pack = _tokenize_split(train_rollouts, tok)
    val_pack = _tokenize_split(val_rollouts, tok) if val_rollouts else None

    _SAVE_KEYS = ("ids", "loss_mask", "state_mask", "starts", "vocab")

    train_out = out_dir / f"{args.name}_train.pt"
    torch.save({k: train_pack[k] for k in _SAVE_KEYS}, train_out)

    n_ids = len(train_pack["ids"])
    n_sup = train_pack["_n_supervised"]
    print(f"[step8-tok] train: {len(train_rollouts)} rollouts | "
          f"{n_ids} tokens | {n_sup} supervised ({100*n_sup/max(1,n_ids):.1f}%)")
    print(f"[step8-tok] → {train_out}")

    if val_pack:
        val_out = out_dir / f"{args.name}_val.pt"
        torch.save({k: val_pack[k] for k in _SAVE_KEYS}, val_out)
        print(f"[step8-tok] val:   {len(val_rollouts)} rollouts → {val_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
