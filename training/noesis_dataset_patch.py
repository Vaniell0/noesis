"""Monkey-patch RWKV-PEFT's MyDataset to consume our .pt corpus schema.

Our tokenized corpus lives in a torch-serialized dict:
    {
        "ids":       LongTensor[N_tokens],
        "loss_mask": LongTensor[N_tokens],  # 1 where supervised, 0 elsewhere
        "starts":    LongTensor[N_rollouts + 1],
        "vocab":     str,
    }

The vendored ``rwkvt.dataset.dataset.MyDataset`` for ``data_type='sft'`` expects
``self.data`` to be a 3-tuple ``(inputs, labels, attn_mask)`` of shape
``[N_rollouts, ctx_len]``. Its ``__getitem__`` then does
``labels = torch.roll(labels, shifts=-1, dims=-1)`` — so we prepare pre-roll
labels equal to token IDs at supervised positions and -100 elsewhere. After the
roll, position ``t`` predicts token ``t+1``: if the model output at ``t`` is not
supervised in our schema, the corresponding pre-roll label at ``t+1`` is -100.

Apply from ``train_pilot.py`` before ``runpy.run_path`` invokes ``train.py``.
"""

from __future__ import annotations

import os
import torch


def _load_noesis_pt(path: str, ctx_len: int):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    ids = blob["ids"].long()
    loss_mask = blob["loss_mask"].long()
    starts = blob["starts"].long()

    n = starts.numel() - 1
    inputs = torch.zeros((n, ctx_len), dtype=torch.long)
    labels = torch.full((n, ctx_len), -100, dtype=torch.long)
    attn_mask = torch.zeros((n, ctx_len), dtype=torch.long)

    truncated = 0
    supervised_tokens = 0
    total_tokens = 0
    for i in range(n):
        s, e = starts[i].item(), starts[i + 1].item()
        span_ids = ids[s:e]
        span_mask = loss_mask[s:e]
        L = span_ids.numel()
        if L > ctx_len:
            span_ids = span_ids[:ctx_len]
            span_mask = span_mask[:ctx_len]
            L = ctx_len
            truncated += 1
        inputs[i, :L] = span_ids
        labels[i, :L] = torch.where(
            span_mask == 1, span_ids, torch.full_like(span_ids, -100)
        )
        attn_mask[i, :L] = 1
        supervised_tokens += int(span_mask.sum().item())
        total_tokens += L

    if truncated:
        print(
            f"[noesis_dataset_patch] {truncated}/{n} rollouts truncated to ctx_len={ctx_len}"
        )

    skip = int(os.environ.get("NOESIS_SKIP_ROLLOUTS", "0"))
    if skip > 0:
        inputs = inputs[skip:]
        labels = labels[skip:]
        attn_mask = attn_mask[skip:]
        print(f"[noesis_dataset_patch] skipping first {skip} rollouts (resume)")

    remaining = inputs.shape[0]
    print(
        f"[noesis_dataset_patch] loaded {n} rollouts "
        f"({remaining} after skip), "
        f"{total_tokens} tokens, {supervised_tokens} supervised "
        f"({100.0 * supervised_tokens / max(total_tokens, 1):.1f}%)"
    )
    return inputs, labels, attn_mask


def apply() -> str:
    from rwkvt.dataset import dataset as ds

    _orig_init = ds.MyDataset.__init__

    def _patched_init(self, args, processor=None):
        data_file = getattr(args, "data_file", None)
        if (
            data_file
            and str(data_file).endswith(".pt")
            and getattr(args, "data_type", None) == "sft"
        ):
            self.args = args
            self.processor = processor
            self.data_type = args.data_type
            self.data = _load_noesis_pt(data_file, args.ctx_len)
            return
        _orig_init(self, args, processor)

    ds.MyDataset.__init__ = _patched_init
    return "noesis_dataset_patch: MyDataset.__init__ patched for .pt sft"
