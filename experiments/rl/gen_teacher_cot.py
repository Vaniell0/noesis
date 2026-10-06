#!/usr/bin/env python3
"""gen_teacher_cot.py — a REAL reasoning teacher for M>1 phase training.

Why it exists (measured 2026-10-05, memory project_noesis_warmup_think_is_answer):
the default data of `train_think_distill.py` (`g1i_warmup_v3_eos_train.pt`) has a
one-sentence templated think on all 10 529 examples, and every one contains the
answer verbatim ("Computing the arithmetic step gives 4736."). There is no
reasoning in it to split into phases, so M>1 on it is meaningless.

This generates the teacher from the model itself (same-model rule): matrix items
from the project's generator, the answer format taken from the question (content x
format), the model thinks in its native `<think>` mode, then answers. An attempt is
KEPT only if the verifier passes both content (first integer == answer) and format
(strict check of the requested form). Up to --tries attempts per item: greedy first,
then sampled.

Outputs
  <out>.jsonl   one record per kept attempt (text, think length, metadata) —
                resumable, appended as it goes
  <out>.pt      the tokenised blob `train_think_distill.load_examples` reads:
                ids / state_mask (think span) / loss_mask (think + answer) / starts

Meant for the GPU (CPU: a long think is minutes per item on 2.9B); the CPU path
exists for smoke tests on the 0.4B.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rl.loader import load_rwkv7  # noqa: E402
from experiments.rl.matrix_n_probe import FORMATS, make_items, with_format  # noqa: E402


@torch.no_grad()
def _sample(logits, temperature: float, gen) -> int:
    v = logits[0, -1].float()
    if temperature <= 0:
        return int(v.argmax())
    return int(torch.multinomial(torch.softmax(v / temperature, -1), 1, generator=gen))


@torch.no_grad()
def attempt(loaded, q: str, max_think: int, max_answer: int, temperature: float, gen):
    """One think+answer attempt. Returns (think_text, answer_text, closed)."""
    tok = loaded.tokenizer
    ids = tok.encode(f"User: {q}\n\nAssistant: <think>")
    logits, st = loaded.forward_stateful(torch.tensor([ids], device=loaded.device),
                                         loaded.new_state(batch=1))
    think, closed = [], False
    for _ in range(max_think):
        nxt = _sample(logits, temperature, gen)
        if nxt == 0:
            break
        think.append(nxt)
        logits, st = loaded.forward_stateful(torch.tensor([[nxt]], device=loaded.device), st)
        if "</think>" in tok.decode(think[-8:]):
            closed = True
            break
    if not closed:
        return tok.decode(think), "", False
    ans = []
    for _ in range(max_answer):
        nxt = _sample(logits, 0.0, gen)          # the answer itself is always greedy
        if nxt == 0:
            break
        ans.append(nxt)
        piece = tok.decode(ans)
        if "\n\n" in piece or "User:" in piece:
            break
        logits, st = loaded.forward_stateful(torch.tensor([[nxt]], device=loaded.device), st)
    text = tok.decode(ans).split("\n\n")[0].split("User:")[0].strip()
    return tok.decode(think), text, True


def verify(answer_text: str, gold: str, fmt: str) -> tuple:
    m = re.search(r"-?\d+", answer_text)
    content = bool(m and m.group(0) == gold)
    first_line = next((l for l in answer_text.split("\n") if l.strip()), "")
    return content, bool(FORMATS[fmt][1](first_line))


def tokenise(tok, rows: list) -> dict:
    """Same layout as the existing tokenised blobs: prompt (masks 0), think
    (state 1, loss 1), answer + EOS (loss 1)."""
    ids, sm, lm, starts = [], [], [], [0]
    for r in rows:
        p = tok.encode(f"User: {r['question']}\n\nAssistant:")
        t = tok.encode(" <think>" + r["think"].split("</think>")[0] + "</think>\n")
        a = tok.encode(r["answer_text"]) + [0]
        ids += p + t + a
        sm += [0] * len(p) + [1] * len(t) + [0] * len(a)
        lm += [0] * len(p) + [1] * len(t) + [1] * len(a)
        starts.append(len(ids))
    return {"ids": torch.tensor(ids), "state_mask": torch.tensor(sm),
            "loss_mask": torch.tensor(lm), "starts": torch.tensor(starts),
            "vocab": "rwkv_world"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True, help="Path stem; writes .jsonl and .pt")
    ap.add_argument("--per-cell", type=int, default=50)
    ap.add_argument("--levels", default="1,2,3,4")
    ap.add_argument("--formats", default="num,eq,json,sent,json2,sent2")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="For attempts after the first (the first is greedy).")
    ap.add_argument("--max-think", type=int, default=2048)
    ap.add_argument("--max-answer", type=int, default=48)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    if args.device == "cpu" and dtype == torch.bfloat16:
        print("[teacher] bf16 matmul on this CPU is pathologically slow — using float32")
        dtype = torch.float32

    loaded = load_rwkv7(args.model, device=args.device, dtype=dtype,
                        backend="peft", lora_r=0, ctx_len=8192)
    tok = loaded.tokenizer
    items = make_items(args.per_cell, [int(x) for x in args.levels.split(",")], args.seed)
    fmts = args.formats.split(",")
    jl = args.out.with_suffix(".jsonl")
    jl.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["item"], r["format"]) for r in map(json.loads, open(jl))} if jl.exists() else set()
    tried = jl.with_suffix(".tried")
    tried_set = set(tuple(json.loads(l)) for l in open(tried)) if tried.exists() else set()
    print(f"[teacher] {len(items)} items x {len(fmts)} formats, {len(done)} kept, "
          f"{len(tried_set)} cells already tried", flush=True)

    for i, it in enumerate(items):
        for f in fmts:
            if (i, f) in done or (i, f) in tried_set:
                continue
            q = with_format(it["prompt"], FORMATS[f][0])
            kept = None
            for t in range(args.tries):
                think, ans, closed = attempt(loaded, q, args.max_think, args.max_answer,
                                             0.0 if t == 0 else args.temperature, gen)
                content, form = verify(ans, it["answer"], f) if closed else (False, False)
                if content and form:
                    kept = {"item": i, "family": it["family"], "level": it["level"],
                            "format": f, "question": q, "gold": it["answer"],
                            "think": think, "answer_text": ans, "attempt": t,
                            "think_tokens": len(tok.encode(think))}
                    break
            if kept:
                with open(jl, "a") as fh:
                    fh.write(json.dumps(kept) + "\n")
            with open(tried, "a") as fh:
                fh.write(json.dumps([i, f]) + "\n")
            print(f"[teacher] item {i+1}/{len(items)} {it['family']} L{it['level']} {f}: "
                  f"{'KEPT t=' + str(kept['attempt']) + ' think=' + str(kept['think_tokens']) if kept else 'none'}",
                  flush=True)

    rows = [json.loads(l) for l in open(jl)] if jl.exists() else []
    blob = tokenise(tok, rows)
    torch.save(blob, args.out.with_suffix(".pt"))
    print(f"[teacher] kept {len(rows)} -> {args.out.with_suffix('.pt')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
