#!/usr/bin/env python3
"""m0_pass_probe.py — what does the model answer with no thinking at all (M=0), per family and level?

Stage S0's gate (HZLK §6): every family needs items the model sometimes solves, otherwise there is
nothing for S1/S2 to start from. The same pass gives the S1 self-distillation set (items right at
M=0: the latent tick must learn not to lose what the model already knows) and the H19 split of the
cipher tasks (question/mixed rows: is the fact in the weights).

Prompt is the project's silent route: "User: {q}\\n\\nAssistant: <think>\\n</think>\\n", greedy answer,
cut at a blank line. Checks: cipher knowledge rows by training/scripts/gen_ciphers.py::check_answer
(content and form apart). Matrix rows: no model keeps "Output only X", they reason in the answer, so content and form are
apart as for ciphers: the answer runs to the end (no blank-line cut, --m0-max-new-matrix), content =
the row's rubric (experiments/A0_eval/eval.py::score_task) on the model's conclusion: the first
\\boxed{...} (generation stops there), else the last non-empty line of a finished answer, form = the whole answer is the gold answer. acc_lenient = the rubric anywhere: on 2026-10-22 it
was the only score and matched stray digits in truncated reasoning ("The plaintext is: 0 1...").
Every response is saved.

    python experiments/run.py --backend peft --model M --tests m0_pass --m0-items F [--m0-items G] ...
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._common import registry  # noqa: E402
from experiments._common.runtime import progress  # noqa: E402


def _cell(r: dict) -> tuple:
    fam = r.get("task") and f"cipher_{r['task']}" or r.get("category", "?").split("_")[0]
    return fam, r.get("level"), r.get("answer_format") or r.get("rubric", {}).get("format", "")


def _sample(rows: list, per_cell: int, seed: int) -> list:
    if not per_cell:
        return rows
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[_cell(r)[:2]].append(r)
    out = []
    for k in sorted(by, key=str):
        out += rng.sample(by[k], min(per_cell, len(by[k])))
    return out


@torch.no_grad()
def answer(loaded, tok, q: str, max_new: int, cut_blank: bool = True, ticks: int = 0, noise=None,
           temp: float = 0.0, patch: tuple | None = None) -> tuple:
    """(response, finished). cut_blank=False lets a model that reasons in the answer reach its conclusion.
    ticks > 0: after the prompt, feed the model's own latent back `ticks` times (the S1 loop, as
    train_latent_s1.seq_logits does), then answer from the state after the last tick. noise: h -> h, applied
    to the latent before each tick is fed (latent_noise --ln-free). temp > 0 samples each token from
    softmax(logits / temp) instead of the greedy argmax (chat --cc-temps: what the distribution holds, not only its top).
    patch = (src, dst[, src_ids]): the prompt runs through loaded.forward_patched (back-patching, M=0 only)."""
    head, close = loaded.tick_frame(q) if ticks and hasattr(loaded, "tick_frame") else (loaded.silent_prompt(q), "")
    ids = tok.encode(head)
    if ticks:
        E = loaded.embedding_weight
        logits, st, h = loaded.forward_stateful_embeds(E[torch.tensor([ids])], loaded.new_state(batch=1),
                                                       return_hidden=True)
        h = h[:, -1:]
        for _ in range(ticks):
            if noise is not None:
                h = noise(h)
            logits, st, h = loaded.forward_stateful_embeds(h, st, return_hidden=True)
        if close:
            logits, st = loaded.forward_stateful_embeds(E[torch.tensor([tok.encode(close)])], st)
    elif patch:
        logits, st = loaded.forward_patched(torch.tensor([ids]), loaded.new_state(batch=1), *patch)
    else:
        logits, st = loaded.forward_stateful(torch.tensor([ids]), loaded.new_state(batch=1))
    out = []
    for _ in range(max_new):
        lg = logits[0, -1].float()
        nxt = int(torch.multinomial(torch.softmax(lg / temp, dim=-1), 1)) if temp > 0 else int(lg.argmax())
        if nxt in loaded.eos_ids:
            break
        out.append(nxt)
        tail = tok.decode(out[-4:])
        if (cut_blank and "\n\n" in tail) or "User:" in tail:
            break
        if not cut_blank and "}" in tail and _BOXED.search(tok.decode(out[-64:])):
            break
        if ticks:
            logits, st = loaded.forward_stateful_embeds(loaded.embedding_weight[torch.tensor([[nxt]])], st)
        else:
            logits, st = loaded.forward_stateful(torch.tensor([[nxt]]), st)
    else:
        text = tok.decode(out)
        return (text.split("\n\n")[0] if cut_blank else text).split("User:")[0].strip(), False
    text = tok.decode(out)
    return (text.split("\n\n")[0] if cut_blank else text).split("User:")[0].strip(), True


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def _conclusion(resp: str, finished: bool):
    """The model's own final answer: the first \\boxed{...}, else the last line of a finished answer."""
    m = _BOXED.search(resp)
    if m:
        return m.group(1)
    return next((l for l in reversed(resp.splitlines()) if l.strip()), "") if finished else None


def _strict(r: dict, resp: str) -> bool:
    norm = lambda t: re.sub(r"[\s.\"'`]+", " ", t).strip().lower()
    return norm(resp) == norm(str(r["answer"]))


@torch.no_grad()
def _rank_capitals(loaded, tok, q: str, r: dict) -> dict | None:
    """Teacher-forced log p (sum over the answer tokens, M=0) of every capital written in the gold's own format
    (plain / quoted / json / line), and the gold's rank among them. None if the gold is not a capital."""
    from training.scripts.gen_ciphers import CAPITALS
    gold = next((c for c in CAPITALS.values() if c in r["answer"]), None)
    if gold is None:
        return None
    ids = tok.encode(loaded.silent_prompt(q))
    lp = {}
    for c in sorted(set(CAPITALS.values())):
        a = tok.encode(r["answer"].replace(gold, c))
        logits, _ = loaded.forward_stateful(torch.tensor([ids + a]), loaded.new_state(batch=1))
        logp = torch.log_softmax(logits[0, len(ids) - 1:-1].float(), dim=-1)
        lp[c] = round(float(logp.gather(1, torch.tensor(a)[:, None]).sum()), 3)
    order = sorted(lp, key=lp.get, reverse=True)
    return {"gold": gold, "rank": order.index(gold) + 1, "top": order[0], "logp_gold": lp[gold], "logp_top": lp[order[0]]}


def _add_args(ap):
    ap.add_argument("--m0-items", type=Path, action="append", help="task jsonl with prompt + rubric; repeatable")
    ap.add_argument("--m0-per-cell", type=int, default=6, help="items per (family, level); 0 = all")
    ap.add_argument("--m0-max-new", type=int, default=48, help="cipher rows (answer cut at a blank line)")
    ap.add_argument("--m0-max-new-matrix", type=int, default=256, help="matrix rows: room to reason to a conclusion")
    ap.add_argument("--m0-seed", type=int, default=0)
    ap.add_argument("--m0-hint", type=int, default=0,
                    help="plain-rung rows: insert hint level k (gen_ciphers.hint_ladder) before the question; 0 = none")
    ap.add_argument("--m0-ticks", type=int, default=0, help="own-latent ticks before the answer (0 = the M=0 pass)")
    ap.add_argument("--m0-rank", action="store_true",
                    help="rows whose gold is a capital: teacher-forced log p of every capital in the gold's format at M=0, "
                         "rank of the gold - is the fact in the weights when the greedy answer misses it (owner 2026-11-04: "
                         "lost in pretraining like bad sectors, or present and not reached)")
    ap.add_argument("--m0-samples", type=int, default=0, help="also sample this many answers at --m0-temp, count gold hits")
    ap.add_argument("--m0-temp", type=float, default=1.0)


@registry.probe("m0_pass", hypothesis=["HZLK", "H19"], backend="peft", add_args=_add_args,
                description="silent (M=0) answers per family/level/format: S0 gate, S1 self-distillation set, H19 split")
def run(loaded, tok, args) -> dict:
    from experiments.A0_eval.eval import score_task
    from training.scripts.gen_ciphers import check_answer, with_hint
    rows = []
    for f in args.m0_items:
        rows += _sample([json.loads(l) for l in open(f) if l.strip()], args.m0_per_cell, args.m0_seed)
    cells = collections.defaultdict(lambda: [0, 0, 0, 0])         # n, content, form, lenient
    right, t0, examples, responses, n_fin = [], time.time(), {}, {}, 0
    ranks, sampled = {}, {}
    for i, r in enumerate(rows):
        if "task" in r:
            resp, finished = answer(loaded, tok, with_hint(r, args.m0_hint), args.m0_max_new, ticks=args.m0_ticks)
            content, form = check_answer(r, resp)
            lenient = content
        else:
            resp, finished = answer(loaded, tok, r["prompt"], args.m0_max_new_matrix, cut_blank=False,
                                    ticks=args.m0_ticks)
            concl = _conclusion(resp, finished)
            content = concl is not None and score_task(r, concl)["correct"]
            finished = finished or concl is not None
            form = _strict(r, resp)
            lenient = score_task(r, resp)["correct"]
        n_fin += finished
        responses[r["id"]] = resp
        c = cells[_cell(r)]
        examples.setdefault(str(_cell(r)), {"id": r["id"], "gold": r["answer"], "response": resp[:200],
                                            "content": bool(content), "form": bool(form)})
        c[0] += 1; c[1] += content; c[2] += form; c[3] += lenient
        if content:
            right.append(r["id"])
        if args.m0_rank and "task" in r:
            rk = _rank_capitals(loaded, tok, with_hint(r, args.m0_hint), r)
            if rk:
                ranks[r["id"]] = rk
        if args.m0_samples and "task" in r:
            hits = 0
            for k in range(args.m0_samples):
                torch.manual_seed(1000 * i + k)
                sr, _ = answer(loaded, tok, with_hint(r, args.m0_hint), args.m0_max_new, ticks=args.m0_ticks,
                               temp=args.m0_temp)
                hits += bool(check_answer(r, sr)[0])
            sampled[r["id"]] = hits
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            progress(f"[m0_pass] {i + 1}/{len(rows)} right {len(right)} ({time.time() - t0:.0f}s)")
    table = [{"family": k[0], "level": k[1], "format": k[2], "n": v[0],
              "acc": round(v[1] / v[0], 3), "form": round(v[2] / v[0], 3),
              "acc_lenient": round(v[3] / v[0], 3)} for k, v in sorted(cells.items(), key=str)]
    fam = collections.defaultdict(lambda: [0, 0])
    for t in table:
        fam[t["family"]][0] += t["n"]; fam[t["family"]][1] += round(t["acc"] * t["n"])
    return {"model": args.model, "files": [str(f) for f in args.m0_items], "n": len(rows),
            "cells": table, "right_ids": right, "examples": examples, "responses": responses,
            "finished": n_fin, "ticks": args.m0_ticks, "hint": args.m0_hint, "ranks": ranks,
            "sampled": sampled, "sample_temp": args.m0_temp if args.m0_samples else None,
            "scoring": "matrix: acc = rubric on the last line of a finished answer; form = whole answer == gold; "
                       "acc_lenient = rubric anywhere (the old score)",
            "families": {k: {"n": v[0], "acc": round(v[1] / v[0], 3)} for k, v in fam.items()},
            "_summary": {f"{k} acc": f"{v[1]}/{v[0]}" for k, v in fam.items()}}
