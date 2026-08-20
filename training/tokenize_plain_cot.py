"""Tokenize plain-CoT QA corpus (RFC binary-protocol tasks, step 9).

Input JSONL schema (from restructure_rfc.py):
    {
        "id":       str,
        "system":   str,
        "user":     str,
        "think":    str,   -- step-by-step reasoning
        "answer":   str,   -- final answer
    }

Rendered as:
    System: {system}

    {user}

    <think>
    {think}
    </think>
    {answer}

Loss masks:
    loss_mask  (L_CE_SFT) : 1 on <think>…</think> + answer tokens
    state_mask (L_state)  : 1 on <think>…</think> tokens only

Usage:
    training/.venv/bin/python training/tokenize_plain_cot.py \\
        --input training/corpus_open/rfc_qa.jsonl \\
        --out-train training/tokenised/step9_rfc_train.pt \\
        --out-val   training/tokenised/step9_rfc_val.pt \\
        --val-pct 10
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

HERE = Path(__file__).resolve().parent
os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "0")

sys.path.insert(0, str(HERE))
from tokenize_fixture import _VocabOnlyPipeline  # noqa: E402

# rwkv_vocab_v20230424's EOS id — confirmed 2026-08-18 (`tok.decode([0])`
# decodes to the EOS char) and again 2026-08-20: no rollout in
# g1i_warmup_v2_train.pt (this renderer's actual output) ever ended in
# token 0, meaning no example ever taught the model to stop after the
# answer — root cause of the answer degenerating into a non-terminating
# repeat loop once generation runs past the corpus's trained-answer
# length. `_render` returns a `None` text sentinel for this segment
# since it can't be produced by `tok.encode(str)` like the others.
EOS_ID = 0


def _render(item: dict) -> list[tuple[str | None, int, int]]:
    """Return list of (text, ce_mask, state_mask) segments.
    `text=None` is the EOS sentinel — emits EOS_ID directly, not via
    tok.encode()."""
    system = item.get("system", "").strip()
    user   = item.get("user", "").strip()
    think  = item.get("think", "").strip()
    answer = item.get("answer", "").strip()

    segs: list[tuple[str | None, int, int]] = []
    if system:
        segs.append((f"System: {system}\n\n", 0, 0))
    segs.append((f"{user}\n\n", 0, 0))
    segs.append(("<think>\n", 1, 1))   # opening tag — supervised (CE + state)
    segs.append((f"{think}\n", 1, 1))
    segs.append(("</think>\n", 1, 1))  # closing tag
    segs.append((answer, 1, 0))        # answer: CE-supervised, not state-supervised
    segs.append((None, 1, 0))          # EOS: CE-supervised (must learn to predict it)
    return segs


def _val_bucket(item_id: str, val_pct: int) -> bool:
    h = hashlib.blake2b(item_id.encode("utf-8"), digest_size=4).hexdigest()
    return (int(h, 16) % 100) < val_pct


def _tokenize_split(items: list[dict], tok) -> dict:
    all_ids:   list[int] = []
    all_ce:    list[int] = []
    all_state: list[int] = []
    starts:    list[int] = []
    n_sup = 0

    for item in items:
        starts.append(len(all_ids))
        segs = _render(item)
        for text, ce, st in segs:
            toks = [EOS_ID] if text is None else tok.encode(text)
            all_ids.extend(toks)
            all_ce.extend([ce] * len(toks))
            all_state.extend([st] * len(toks))
        n_sup += sum(c for _, c, _ in segs if c)

    starts.append(len(all_ids))
    return {
        "ids":        torch.tensor(all_ids,   dtype=torch.long),
        "loss_mask":  torch.tensor(all_ce,    dtype=torch.long),
        "state_mask": torch.tensor(all_state, dtype=torch.long),
        "starts":     torch.tensor(starts,    dtype=torch.long),
        "vocab":      "rwkv_vocab_v20230424",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",     required=True, help="JSONL file(s), glob ok")
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val",   required=True)
    ap.add_argument("--val-pct",   type=int, default=10,
                    help="Percent of items held out for val (default 10)")
    args = ap.parse_args()

    tok = _VocabOnlyPipeline()

    items: list[dict] = []
    for path in sorted(glob.glob(args.input)):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    print(f"[tokenize_plain_cot] {len(items)} items from {args.input}")

    train_items = [x for x in items if not _val_bucket(x["id"], args.val_pct)]
    val_items   = [x for x in items if     _val_bucket(x["id"], args.val_pct)]
    print(f"  train={len(train_items)} val={len(val_items)}")

    for split_items, out_path in [(train_items, args.out_train),
                                   (val_items,   args.out_val)]:
        blob = _tokenize_split(split_items, tok)
        n_tok = blob["ids"].numel()
        n_sup = int(blob["loss_mask"].sum().item())
        n_st  = int(blob["state_mask"].sum().item())
        print(f"  {out_path}: {n_tok} tokens, {n_sup} CE-supervised, {n_st} state-supervised")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torch.save(blob, out_path)
        print(f"  → saved {out_path}")


if __name__ == "__main__":
    main()
