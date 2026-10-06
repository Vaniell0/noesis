#!/usr/bin/env python3
"""cot_vs_routes_probe.py — a better answer than the CoT teacher, for less compute?

The goal the user set (2026-10-05): beat the teacher's CoT on accuracy at lower
compute. On the 10 finished items of phase_probe.py (G1i, pattern, JSON) the silent
answer was right 6/10 with zero thinking tokens, the CoT 5/10 at 306 tokens on
average — short CoTs (~100 tokens) were always right, CoTs that ran to the budget
never — and where the silent routes agreed the silent answer was right 3/3. An
offline policy "answer silently when routes agree, else think" gave 7/10 at ~194
tokens. Ten items is a hint. This measures it on more items, untrained model:

  silent routes (each from a clone of the one post-question state, think closed):
    r0      answer straight away
    lat2    2 ticks of the model's own expected embedding, then answer
    lat8    8 ticks of the same
    const8  8 ticks of a constant vector (mean embedding), then answer
    reread  the question fed a second time (H10's N=2), then answer
  teacher: greedy CoT in native <think> mode up to --max-think tokens; at every
    checkpoint K the state is cloned, "</think>" forced and the answer read — so the
    teacher's own compute/accuracy curve comes out of one generation, and a CoT that
    runs away can be checked for an answer it already had and then lost.

Everything per item is written (answers, token counts, the decoder's confidence on
r0), so any routing policy can be scored offline: accuracy vs tokens. No training.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rl.loader import load_rwkv7, _PeftState  # noqa: E402
from experiments.rl.matrix_n_probe import FORMATS, make_items, with_format  # noqa: E402
from experiments.rl.fork_agreement_probe import _gen, _first_int, _first_number_token_prob  # noqa: E402


def _clone(st):
    return _PeftState(st.shift.clone(), st.wkv.clone())


@torch.no_grad()
def _ticks(loaded, logits, st, n, mode, feed_norm, mean_emb):
    E = loaded.embedding_weight.float()
    for _ in range(n):
        if mode == "expected":
            feed = F.softmax(logits[0, -1].float(), -1) @ E
        else:
            feed = mean_emb.clone()
        feed = feed * (feed_norm / feed.norm().clamp_min(1e-8))
        logits, st = loaded.forward_stateful_embeds(feed.to(loaded.dtype).view(1, 1, -1), st)
    return logits, st


@torch.no_grad()
def _close_and_answer(loaded, logits, st, closer):
    logits, st = loaded.forward_stateful(torch.tensor([closer]), st)
    text, ids, lps = _gen(loaded, logits, st)
    return text, ids, lps


@torch.no_grad()
def one_item(loaded, q, max_think, checkpoints, feed_norm, mean_emb):
    tok = loaded.tokenizer
    closer = tok.encode("\n</think>\n")
    rec = {"routes": {}, "cot": {}}
    head = tok.encode(f"User: {q}\n\nAssistant: <think>")
    logits0, st0 = loaded.forward_stateful(torch.tensor([head]), loaded.new_state(batch=1))

    # silent routes, all from clones of the same post-question state
    text, ids, lps = _close_and_answer(loaded, logits0, _clone(st0), closer)
    rec["routes"]["r0"] = _first_int(text)
    rec["r0_first_num_prob"] = _first_number_token_prob(tok, ids, lps)
    rec["r0_mean_logprob"] = sum(lps) / max(1, len(lps))
    for name, n, mode in (("lat2", 2, "expected"), ("lat8", 8, "expected"), ("const8", 8, "const")):
        lg, st = _ticks(loaded, logits0, _clone(st0), n, mode, feed_norm, mean_emb)
        rec["routes"][name] = _first_int(_close_and_answer(loaded, lg, st, closer)[0])
    rr = tok.encode(f"User: {q}\n\nUser: {q}\n\nAssistant: <think>")
    lg, st = loaded.forward_stateful(torch.tensor([rr]), loaded.new_state(batch=1))
    rec["routes"]["reread"] = _first_int(_close_and_answer(loaded, lg, st, closer)[0])
    rec["reread_extra_tokens"] = len(rr) - len(head)

    # teacher CoT with answer read at checkpoints
    logits, st = logits0, _clone(st0)
    out, closed = [], False
    cps = sorted(set(checkpoints))
    for step in range(1, max_think + 1):
        nxt = int(logits[0, -1].float().argmax())
        if nxt == 0:
            break
        out.append(nxt)
        logits, st = loaded.forward_stateful(torch.tensor([[nxt]]), st)
        if "</think>" in tok.decode(out[-8:]):
            closed = True
            break
        if step in cps:
            rec["cot"][str(step)] = _first_int(_close_and_answer(loaded, logits, _clone(st), closer)[0])
    if closed:
        text, _, _ = _gen(loaded, *loaded.forward_stateful(torch.tensor([tok.encode("\n")]), st))
        rec["cot_final"] = _first_int(text)
    else:
        rec["cot_final"] = _first_int(_close_and_answer(loaded, logits, st, closer)[0])
    rec["cot_tokens"] = len(out)
    rec["cot_closed"] = closed
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--levels", default="1,2,3,4")
    ap.add_argument("--families", default="pattern,arith")
    ap.add_argument("--format", default="json")
    ap.add_argument("--max-think", type=int, default=768)
    ap.add_argument("--checkpoints", default=",".join(str(k) for k in range(16, 769, 16)))
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    loaded = load_rwkv7(args.model, device="cpu", dtype=torch.float32,
                        backend="peft", lora_r=0, ctx_len=8192)
    E = loaded.embedding_weight.float()
    feed_norm = float(E.norm(dim=-1).median())
    mean_emb = E.mean(0)
    fams = set(args.families.split(","))
    items = [it for it in make_items(args.per_cell, [int(x) for x in args.levels.split(",")], args.seed)
             if it["family"] in fams]
    cps = [int(x) for x in args.checkpoints.split(",")]
    part = args.out.with_suffix(".partial.jsonl")
    done = [json.loads(l) for l in open(part)] if part.exists() else []
    print(f"[cvr] {len(items)} items, {len(done)} done, max_think {args.max_think}", flush=True)
    for i, it in enumerate(items):
        if i < len(done):
            continue
        q = with_format(it["prompt"], FORMATS[args.format][0])
        rec = one_item(loaded, q, args.max_think, cps, feed_norm, mean_emb)
        rec.update({"item": i, "family": it["family"], "level": it["level"], "gold": it["answer"]})
        with open(part, "a") as f:
            f.write(json.dumps(rec) + "\n")
        done.append(rec)
        print(f"[cvr] {i+1}/{len(items)} {it['family']} L{it['level']} gold={it['answer']} "
              f"routes={rec['routes']} cot={rec['cot_final']}@{rec['cot_tokens']}"
              f"{'' if rec['cot_closed'] else '(open)'}", flush=True)

    # offline policies
    def acc(preds):
        return round(sum(p == r["gold"] for p, r in zip(preds, done)) / len(done), 3)
    summ = {"n": len(done)}
    for rname in ("r0", "lat2", "lat8", "const8", "reread"):
        summ[f"silent:{rname}"] = acc([r["routes"][rname] for r in done])
    summ["cot"] = {"acc": acc([r["cot_final"] for r in done]),
                   "mean_tokens": round(sum(r["cot_tokens"] for r in done) / len(done), 1)}
    for k in cps:
        summ[f"cot@{k}"] = acc([r["cot"].get(str(k), r["cot_final"]) for r in done])
    for need in (5, 4, 3):
        preds, toks = [], []
        for r in done:
            vals = [v for v in r["routes"].values() if v is not None]
            top, cnt = (Counter(vals).most_common(1)[0] if vals else (None, 0))
            if cnt >= need:
                preds.append(top); toks.append(0)
            else:
                preds.append(r["cot_final"]); toks.append(r["cot_tokens"])
        summ[f"adaptive(agree>={need}/5)"] = {"acc": acc(preds),
                                              "mean_tokens": round(sum(toks) / len(toks), 1),
                                              "silent_share": round(toks.count(0) / len(toks), 2)}
    print(json.dumps(summ, indent=1))
    args.out.write_text(json.dumps({"model": args.model, "summary": summ, "rows": done}, indent=1))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
