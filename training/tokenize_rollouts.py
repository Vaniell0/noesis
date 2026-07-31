"""Tokenize the full A1 rollout corpus (Variant C hybrid primary).

Sibling of ``tokenize_fixture.py`` — same tokenizer, same rendering,
same loss-mask contract. This one takes multiple JSONL files (globbable),
does a deterministic train/val split, and emits chunked .pt files
suitable for the A1 pilot / full run.

## Inputs

One or more rollout JSONL files matching the schema of
``training/fixtures/tool_call_open.jsonl``::

    {"id": "...", "source": "...", "turns": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "tool_use": {"name": "...", "input": {...}}},
        {"role": "tool_result", "content": "..."},
        ...
    ]}

Typical source: ``training/corpus_open/xlam_60k.jsonl`` produced by
``scripts/normalize_xlam.py``.

## Outputs

    training/tokenised/<name>_train.pt
    training/tokenised/<name>_val.pt

Both are torch.save of a dict with the same shape as
``tokenize_fixture.py``::

    {
      "ids":       LongTensor [N_total],
      "loss_mask": LongTensor [N_total],
      "starts":    LongTensor [n_rollouts],
      "vocab":     "rwkv_vocab_v20230424",
    }

## Split

Deterministic hash of rollout ``id`` → val if hash % 100 < val_pct.
Default val_pct=2 (600 held out from 30k, sufficient for CE curve).

## Loss mask (unchanged from tokenize_fixture.py)

Only ``<tool_use>`` regions get ``loss_mask=1``. Everything else
(user prompt, tool_result, assistant prose) stays at 0. This
implements the Variant C policy: "behavior-cloning on *what to do
next*, not *how to sound while thinking*."

## Run (from repo root)

    training/.venv/bin/python training/tokenize_rollouts.py \\
        --input training/corpus_open/xlam_60k.jsonl \\
        --name xlam_60k
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

# Reuse the pipeline + rendering from the fixture tokenizer so the two
# entry points cannot drift on rendering shape.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tokenize_fixture import _VocabOnlyPipeline, _render_turns  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "tokenised"


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
    all_mask: list[int] = []
    starts: list[int] = []
    n_supervised_tokens = 0

    for rollout in rollouts:
        segs, sup = _render_turns(rollout["turns"])
        starts.append(len(all_ids))
        for seg, is_sup in zip(segs, sup):
            ids = tok.encode(seg)
            all_ids.extend(ids)
            all_mask.extend([1 if is_sup else 0] * len(ids))
            if is_sup:
                n_supervised_tokens += len(ids)

    return {
        "ids": torch.tensor(all_ids, dtype=torch.long),
        "loss_mask": torch.tensor(all_mask, dtype=torch.long),
        "starts": torch.tensor(starts, dtype=torch.long),
        "vocab": "rwkv_vocab_v20230424",
        "_n_supervised": n_supervised_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="rollouts JSONL path or glob (repeatable)",
    )
    parser.add_argument(
        "--name", required=True, help="output basename (no extension)"
    )
    parser.add_argument(
        "--val-pct",
        type=int,
        default=2,
        help="percent of rollouts held out for validation (default 2)",
    )
    parser.add_argument(
        "--out-dir", default=str(OUT_DIR), help="output directory"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="cap rollout count (debug)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = _VocabOnlyPipeline()

    train_rollouts: list[dict] = []
    val_rollouts: list[dict] = []
    total_seen = 0
    for rollout in _iter_rollouts(args.input):
        total_seen += 1
        rid = rollout.get("id", f"row_{total_seen}")
        if _val_bucket(str(rid), args.val_pct):
            val_rollouts.append(rollout)
        else:
            train_rollouts.append(rollout)
        if args.limit and total_seen >= args.limit:
            break

    if not train_rollouts:
        print("no rollouts to tokenize", file=sys.stderr)
        return 1

    train_pack = _tokenize_split(train_rollouts, tok)
    val_pack = _tokenize_split(val_rollouts, tok) if val_rollouts else None

    train_out = out_dir / f"{args.name}_train.pt"
    torch.save({k: v for k, v in train_pack.items() if not k.startswith("_")}, train_out)
    print(f"[tokenize_rollouts] wrote {train_out}")
    print(f"[tokenize_rollouts]   rollouts:          {len(train_rollouts)}")
    print(f"[tokenize_rollouts]   total tokens:      {len(train_pack['ids'])}")
    print(
        f"[tokenize_rollouts]   supervised tokens: {train_pack['_n_supervised']} "
        f"({100 * train_pack['_n_supervised'] / max(1, len(train_pack['ids'])):.1f}%)"
    )

    if val_pack is not None:
        val_out = out_dir / f"{args.name}_val.pt"
        torch.save({k: v for k, v in val_pack.items() if not k.startswith("_")}, val_out)
        print(f"[tokenize_rollouts] wrote {val_out}")
        print(f"[tokenize_rollouts]   rollouts:          {len(val_rollouts)}")
        print(f"[tokenize_rollouts]   total tokens:      {len(val_pack['ids'])}")
        print(
            f"[tokenize_rollouts]   supervised tokens: {val_pack['_n_supervised']} "
            f"({100 * val_pack['_n_supervised'] / max(1, len(val_pack['ids'])):.1f}%)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
