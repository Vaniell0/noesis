"""Tokenize the open tool-call fixture for the A1 pilot smoke run.

Reads ``training/fixtures/tool_call_open.jsonl`` (turn-structured, open
sources per §7 corpus policy) and emits a single .pt file with token
ids and a loss mask.

## Format

Each rollout is flattened into one token stream tagged inline:

    <user>{content}<tool_use>{json}<tool_result>{content}<assistant>{content}

Tokens outside ``<tool_use>...</tool_use>`` regions get ``loss_mask=0``
(supervision only on the tool-call structure and JSON payload — the
model learns *when* and *how* to call a tool, not the surrounding
prose). Assistant plain-text content is also unmasked (=0). Refusal /
clarify turns without any tool_use contribute no supervision positions
and are dropped from the training index (kept in the raw stream so the
model still sees them at forward-time, but they don't drive loss).

## Output

``training/fixtures/tool_call_open.pt`` — torch.save of a dict::

    {
      "ids":       LongTensor [N_total],
      "loss_mask": LongTensor [N_total],  # 1 where supervised, 0 elsewhere
      "starts":    LongTensor [n_rollouts],  # rollout start offsets in ids
      "vocab":     "rwkv_vocab_v20230424",
    }

## Not what this is

- Not the production ``tokenize_rollouts.py`` from ``training/README.md``
  Step 5. This is a smoke-run fixture tokenizer: fewer edge cases, no
  chunking, no shuffling, no train/val split.
- Not aware of ``<REDACTED:*>`` markers (the personal-corpus sanitiser
  emits them; the open fixture doesn't contain them by construction).

Run:
    .venv/bin/python training/tokenize_fixture.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "tool_call_open.jsonl"
OUT = HERE / "fixtures" / "tool_call_open.pt"

# The rwkv package's PIPELINE is a thin wrapper around the vocab table
# and exposes .encode / .decode. Same tokeniser the training loop uses.
os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "0")

from rwkv.utils import PIPELINE  # noqa: E402


class _VocabOnlyPipeline:
    """PIPELINE requires a model instance for its full interface, but
    tokenisation only needs the vocab table. Duck-type the two methods
    we call: encode and decode."""

    def __init__(self):
        # PIPELINE(None, ...) works — the model is only used by generate,
        # not by encode/decode. Confirmed by reading rwkv/utils.py.
        self._p = PIPELINE.__new__(PIPELINE)
        # Copy the same vocab init PIPELINE.__init__ does when passed
        # "rwkv_vocab_v20230424" as the second argument.
        from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
        vocab_path = os.path.join(
            os.path.dirname(__import__("rwkv").__file__),
            "rwkv_vocab_v20230424.txt",
        )
        self._p.tokenizer = TRIE_TOKENIZER(vocab_path)

    def encode(self, text: str) -> list[int]:
        return self._p.tokenizer.encode(text)


def _render_turns(turns: list[dict]) -> tuple[list[str], list[bool]]:
    """Flatten one rollout to (segments, supervised_flags).

    Each segment is a substring; supervised_flags[i] says whether the
    tokens for segment i contribute to the loss.
    """
    segs: list[str] = []
    sup: list[bool] = []
    for t in turns:
        role = t["role"]
        if role == "user":
            segs.append(f"<user>{t['content']}")
            sup.append(False)
        elif role == "tool_result":
            segs.append(f"<tool_result>{t['content']}")
            sup.append(False)
        elif role == "assistant":
            # Assistant turn may have BOTH content and tool_use (honesty
            # pattern: "I don't know, let me check <tool_use>..."). In
            # that case, emit prose as unsupervised and tool_use as
            # supervised.
            if "content" in t and t["content"]:
                segs.append(f"<assistant>{t['content']}")
                sup.append(False)
            if "tool_use" in t:
                tu = json.dumps(t["tool_use"], separators=(",", ":"))
                segs.append(f"<tool_use>{tu}")
                sup.append(True)
        else:
            raise ValueError(f"unknown role: {role!r}")
    return segs, sup


def main() -> int:
    if not FIXTURE.exists():
        print(f"missing fixture: {FIXTURE}", file=sys.stderr)
        return 1

    tok = _VocabOnlyPipeline()

    all_ids: list[int] = []
    all_mask: list[int] = []
    starts: list[int] = []
    n_rollouts = 0
    n_supervised_tokens = 0

    with FIXTURE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rollout = json.loads(line)
            segs, sup = _render_turns(rollout["turns"])
            starts.append(len(all_ids))
            for seg, is_sup in zip(segs, sup):
                ids = tok.encode(seg)
                all_ids.extend(ids)
                all_mask.extend([1 if is_sup else 0] * len(ids))
                if is_sup:
                    n_supervised_tokens += len(ids)
            n_rollouts += 1

    out = {
        "ids": torch.tensor(all_ids, dtype=torch.long),
        "loss_mask": torch.tensor(all_mask, dtype=torch.long),
        "starts": torch.tensor(starts, dtype=torch.long),
        "vocab": "rwkv_vocab_v20230424",
    }
    torch.save(out, OUT)

    print(f"[tokenize_fixture] wrote {OUT}")
    print(f"[tokenize_fixture]   rollouts:            {n_rollouts}")
    print(f"[tokenize_fixture]   total tokens:        {len(all_ids)}")
    print(f"[tokenize_fixture]   supervised tokens:   {n_supervised_tokens}"
          f" ({100 * n_supervised_tokens / max(1, len(all_ids)):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
