#!/usr/bin/env python3
"""Equivalence test for the CPU path of the differentiable (peft) backend.

`loader.py::_enable_peft_on_cpu` swaps `rwkvop.py`'s triton `chunk_rwkv7`
for `rwkvfla`'s pure-torch `naive_recurrent_rwkv7` and adapts the head
layout around it. That swap is the whole reason `feed_mode="expected"` and
every gradient check can now run without a GPU — which makes it exactly the
kind of change that must not be trusted on inspection. A transposed head
axis or a mis-ordered einsum gives numbers that look entirely plausible.

The check: run the same prompt through the peft backend and through the
independent `blink` backend (BlinkDL's own `rwkv` package, a completely
separate implementation) and require them to agree.

Tolerance is measured, not guessed. blink runs bf16 and peft here runs
fp32, so exact equality is not the bar. Running the SAME peft
implementation twice, changing only the dtype, already costs
`max|dlogp| = 0.0778` over the top-50 on g1d-0.4b — that is the precision
floor. peft-fp32 against blink-bf16 comes in at 0.0879, and two bf16 runs
of the two different implementations at 0.1392, all with correlation
>= 0.99988 and the same argmax. So the default bar is 0.15: above the
floor, far below anything a layout bug could hide under — a transposed
head axis scrambles the ranking outright, it does not shave hundredths off
a log-probability.

Run:
    training/.venv/bin/python experiments/rl/test_peft_cpu_equivalence.py \\
        --model /home/vaniello/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rl.loader import load_rwkv7

PROMPTS = [
    "The capital of France is",
    "You are a precise reasoning assistant. Work step by step.\n\n"
    "What is the next number in this sequence: 2, 4, 6, 8, ?\n\n<think>\n",
]


def _logits(loaded, prompt: str) -> torch.Tensor:
    state = loaded.new_state(batch=1)
    ids = loaded.tokenizer.encode(prompt)
    inp = torch.tensor([ids]) if loaded.backend == "peft" else ids
    logits, _ = loaded.forward_stateful(inp, state)
    return logits.reshape(-1)[-loaded.vocab_size:] if logits.dim() == 1 \
        else logits[0, -1].float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-logprob-delta", type=float, default=0.15,
                     help="See the module docstring — calibrated against the "
                          "measured bf16/fp32 precision floor, not guessed.")
    args = ap.parse_args()

    print("loading blink (reference, independent implementation) ...", flush=True)
    blink = load_rwkv7(args.model, device="cpu", backend="blink")
    print("loading peft on cpu (the path under test) ...", flush=True)
    peft = load_rwkv7(args.model, device="cpu", dtype=torch.float32,
                      backend="peft", ctx_len=2048)

    failures = 0
    for prompt in PROMPTS:
        lb = _logits(blink, prompt).float()
        lp = _logits(peft, prompt).float()
        pb = F.log_softmax(lb, dim=-1)
        pp = F.log_softmax(lp, dim=-1)

        tb = torch.topk(pb, args.top_k).indices.tolist()
        tp = torch.topk(pp, args.top_k).indices.tolist()
        delta = float((pp[tb] - pb[tb]).abs().max())
        same_argmax = tb[0] == tp[0]
        overlap = len(set(tb) & set(tp))

        ok = same_argmax and delta <= args.max_logprob_delta
        failures += 0 if ok else 1
        print(f"\nprompt: {prompt[:48]!r}")
        print(f"  blink top-{args.top_k}: "
              f"{[blink.tokenizer.decode([i]) for i in tb]}")
        print(f"  peft  top-{args.top_k}: "
              f"{[blink.tokenizer.decode([i]) for i in tp]}")
        print(f"  same argmax: {same_argmax}   top-k overlap: "
              f"{overlap}/{args.top_k}   max |dlogp| on blink's top-k: {delta:.4f}")
        print(f"  {'PASS' if ok else 'FAIL'}")

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
