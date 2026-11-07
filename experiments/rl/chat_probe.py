#!/usr/bin/env python3
"""chat_probe.py — does the model still talk, in English and Russian, and what is in its latent ticks?

Task accuracy hid what S1 did to the model (2026-10-27, 0.4B pilot): CE scored best on the plain
cipher rung while its M=0 path looped on "**Answer:**" and its tick path answered "I would like to
have a car."; KL T=2 kept a fluent tick path. So every checkpoint gets a few everyday questions,
answers saved in full, at several tick counts.

Per question and tick count M: the free greedy answer after M own-latent ticks (m0_pass.answer, the
same route the S1 trainer uses). For M > 0 also what each tick holds, read two ways:
    says     top tokens of the logits the tick produces (what the latent would say if read as text)
    nearest  the vocabulary row closest to ln0(h) - the direction the next block actually receives

    python experiments/run.py --backend peft --model M [--weights W] --tests chat --cc-ticks 0,1,4
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._common import registry  # noqa: E402
from experiments._common.runtime import progress  # noqa: E402

QUESTIONS = [
    "What is your favourite colour, and why?",
    "What car would you like to have, and why?",
    "Name three animals that live in the ocean.",
    "Какой твой любимый цвет и почему?",
    "Какую машину ты хотел бы иметь и почему?",
    "Назови трёх животных, которые живут в океане.",
]


def _add_args(ap):
    ap.add_argument("--cc-ticks", default="0,1", help="tick counts to answer at")
    ap.add_argument("--cc-max-new", type=int, default=256)
    ap.add_argument("--cc-top", type=int, default=5)
    ap.add_argument("--cc-temps", default="", help="also sample at these temperatures (comma list), --cc-samples each")
    ap.add_argument("--cc-samples", type=int, default=3)


@torch.no_grad()
def decode_ticks(loaded, tok, q: str, ticks: int, top: int) -> list:
    E = loaded.embedding_weight
    ln0 = loaded.feed_view
    En = F.normalize(ln0(E.float()), dim=-1)
    ids = tok.encode(loaded.silent_prompt(q))
    _, st, h = loaded.forward_stateful_embeds(E[torch.tensor([ids])], loaded.new_state(batch=1), return_hidden=True)
    h = h[:, -1:]
    out = []
    for t in range(1, ticks + 1):
        u = F.normalize(ln0(h[0, 0].float()), dim=-1)
        near = (En @ u).topk(top)
        lg, st, h = loaded.forward_stateful_embeds(h, st, return_hidden=True)
        p = torch.softmax(lg[0, -1].float(), dim=-1).topk(top)
        out.append({"tick": t,
                    "says": [[tok.decode([int(i)]), round(float(v), 3)] for v, i in zip(p.values, p.indices)],
                    "nearest": [[tok.decode([int(i)]), round(float(v), 3)] for v, i in zip(near.values, near.indices)]})
    return out


@registry.probe("chat", hypothesis=["HZLK"], backend="peft", add_args=_add_args,
                description="everyday questions EN/RU, full answers at several tick counts + what each latent tick holds")
def run(loaded, tok, args) -> dict:
    from experiments.rl.m0_pass_probe import answer
    ticks = [int(x) for x in args.cc_ticks.split(",") if x.strip()]
    rows = []
    for qi, q in enumerate(QUESTIONS):
        for M in ticks:
            resp, finished = answer(loaded, tok, q, args.cc_max_new, cut_blank=False, ticks=M)
            row = {"q": q, "M": M, "answer": resp, "finished": finished}
            if M:
                row["ticks"] = decode_ticks(loaded, tok, q, M, args.cc_top)
            rows.append(row)
            progress(f"[chat] q{qi + 1}/{len(QUESTIONS)} M={M}: {resp[:80]!r}")
            for T in [float(x) for x in args.cc_temps.split(",") if x.strip()]:
                for k in range(args.cc_samples):
                    torch.manual_seed(1000 * qi + 100 * M + k)
                    resp, finished = answer(loaded, tok, q, args.cc_max_new, cut_blank=False, ticks=M, temp=T)
                    rows.append({"q": q, "M": M, "T": T, "sample": k, "answer": resp, "finished": finished})
    return {"model": args.model, "weights": getattr(args, "weights", None), "ticks": ticks, "rows": rows,
            "_summary": {f"{r['q'][:24]} M={r['M']}": r["answer"][:60] for r in rows if "T" not in r}}
