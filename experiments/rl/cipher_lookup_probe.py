#!/usr/bin/env python3
"""cipher_lookup_probe.py — below the solve threshold: does the model put mass on the right plaintext letter at all?

The cipher ladder (gen_ciphers.py --task ladder) is 0/576 on G1d 0.4B at M=0: greedy answers copy the
ciphertext or the shown prefix, no hidden letter is ever right. A solve rate of 0 says nothing about how far
the model is from solving, so this probe scores the gold answer teacher-forced, one forward per row and
variant, and reads the hidden letters only:

- table "orig": the row's own prompt (script ciphers: plain->cipher table A=ク, the encoding direction;
  Caesar: the shift). Decoding then asks an inverse lookup (find what stands before ク).
- table "dec": the same key written cipher->plain (ク=A; Caesar as its full shifted table), so decoding is a
  forward lookup, the direction an attention copy (induction) mechanism reads.
- format "plain": the answer is the plaintext; per-token argmax and log p over the answer tokens.
- format "trace": the answer is "ク=A ス=B / ..." (cipher symbol, '=', its letter; '/' between words); each
  letter is its own segment, so argmax and log p are per letter, and the position index says how many
  decoded pairs precede it (in-context examples of the mapping).

- order "msg_first": the message stands before the key, so the key is read knowing which symbols are asked;
  "twice": the key line is repeated after the message (JRT-Prompt, arXiv 2407.05483).
- format "row": the row's own prompt and its own pair trace (ladder styles trace / indexed, either table direction:
  gen_ciphers --ladder-table); the table axis is then the row's, so --cl-tables is not crossed with it.
- --cl-ticks M: M own-latent ticks between the prompt and the answer (the S1 loop, as m0_pass --m0-ticks).

Chance for a letter is 1/26 (log p = -3.26). Rows: the ladder file's revealed=0 rung. The headline number is
first_p50: over trace letters whose plain letter has not appeared earlier in the answer (so it cannot be copied
from the model's own earlier pair) and that are not the first pair, the share with p(gold) > 0.5.

    python experiments/run.py --backend peft --model M --tests cipher_lookup --cl-items F [--cl-ticks 4]
"""
from __future__ import annotations

import collections
import json
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

ABC = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _key(r: dict) -> dict:
    """cipher symbol -> plain letter, parsed from the row's prompt."""
    p = r["prompt"]
    m = re.search(r"Shift: (\d+)", p)
    if m:
        s = int(m.group(1))
        return {ABC[(i + s) % 26]: ABC[i] for i in range(26)}
    m = re.search(r"Table: (.*)\n", p)
    return {c: a for a, c in (x.split("=", 1) for x in m.group(1).split())}


def _prompt(r: dict, table: str, fmt: str, order: str = "table_first") -> str:
    p = r["prompt"]
    if fmt == "row":
        return p
    if order == "msg_first":                      # the message before the key: the table is read knowing the symbols
        m = re.search(r"Message:\n.*\n", p)
        p = p.replace(m.group(0), "").replace("Cipher:", m.group(0) + "Cipher:", 1)
    elif order == "twice":                        # the key read again after the message (JRT-Prompt, arXiv 2407.05483)
        m = re.search(r"Message:\n.*\n", p)
        p = p.replace(m.group(0), m.group(0) + "{TABLE_AGAIN}\n")
    if table == "dec":
        key = _key(r)
        line = "Table: " + " ".join(f"{c}={a}" for c, a in sorted(key.items(), key=lambda kv: kv[1]))
        if "Table: " in p:
            p = re.sub(r"Table: .*", lambda _: line, p)
            p = p.replace("according to the table", "according to the table (symbol=letter)")
        else:
            p = re.sub(r"Shift: \d+", lambda _: line, p)
            p = p.replace("Rule: each letter moves forward in the alphabet by the shift",
                          "Rule: each cipher letter stands for the letter given in the table")
    if "{TABLE_AGAIN}" in p:
        t = re.search(r"(Table: .*|Shift: \d+)\n", p).group(1)
        p = p.replace("{TABLE_AGAIN}", t)
    if fmt == "trace":
        p = p.replace("Answer with the whole plaintext only, in capital letters.",
                      "Answer symbol by symbol: each cipher symbol, '=', its plain letter; '/' between words.")
    return p


def _segments(r: dict, fmt: str) -> list:
    """(text, is_letter, position) answer segments; position = index of the letter in the plaintext."""
    if fmt == "plain":
        return [(r["answer"], False, -1)]
    if fmt == "row":                              # "1:ク=T 2:ス=H / 3:..." -> "1:ク=" + "T", " 2:ス=" + "H", " / 3:ク=" + ...
        seg, k = [], 0
        for m in re.finditer(r"(\s*(?:/\s*)?\S+?=)([A-Z])", r["trace"]):
            seg += [(m.group(1), False, -1), (m.group(2), True, k)]
            k += 1
        assert "".join(t for t, _, _ in seg) == r["trace"], r["id"]
        return seg
    key = {a: c for c, a in _key(r).items()}
    seg, k = [], 0
    for wi, w in enumerate(r["answer"].split()):
        if wi:
            seg.append((" / ", False, -1))
        for li, a in enumerate(w):
            seg += [((" " if li else "") + key[a] + "=", False, -1), (a, True, k)]
            k += 1
    return seg


@torch.no_grad()
def _score(loaded, tok, q: str, segs: list, ticks: int = 0) -> tuple:
    """ticks > 0: the prompt, then `ticks` own-latent feeds (as m0_pass_probe.answer), then the answer
    teacher-forced from the state after the last tick; the first answer token is read off the last tick."""
    head = tok.encode(loaded.silent_prompt(q))
    ids, marks = list(head), []
    for text, is_letter, pos in segs:
        t = tok.encode(text)
        marks += [(len(ids) + j, is_letter, pos) for j in range(len(t))]
        ids += t
    # one path for every M (M = 0 too): the embeds route differs from forward_stateful on token ids by up to
    # 0.0125 in log p (kernel numerics, checked 11-13), so M = 0 vs M = k is compared on the same route
    E, n = loaded.embedding_weight, len(head)
    lg, st, h = loaded.forward_stateful_embeds(E[torch.tensor([head])], loaded.new_state(batch=1),
                                               return_hidden=True)
    h = h[:, -1:]
    for _ in range(ticks):
        lg, st, h = loaded.forward_stateful_embeds(h, st, return_hidden=True)
    rest, _ = loaded.forward_stateful_embeds(E[torch.tensor([ids[n:]])], st)
    # row j of `logits` predicts ids[j + 1]: prompt rows are not read, row n - 1 is the last tick's
    logits = torch.cat([torch.zeros_like(rest[:, :1]).expand(-1, n - 1, -1), lg[:, -1:], rest], dim=1)
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    out = []
    for i, is_letter, pos in marks:
        lp = float(logp[i - 1, ids[i]])
        out.append({"pos": pos, "letter": is_letter, "hit": int(logp[i - 1].argmax()) == ids[i], "logp": lp})
    return out


def _add_args(ap):
    ap.add_argument("--cl-items", type=Path, help="ladder jsonl (gen_ciphers.py --task ladder)")
    ap.add_argument("--cl-revealed", type=float, default=0.0, help="ladder rung to read (revealed fraction)")
    ap.add_argument("--cl-tables", default="orig,dec")
    ap.add_argument("--cl-formats", default="plain,trace")
    ap.add_argument("--cl-orders", default="table_first", help="table_first, msg_first, twice")
    ap.add_argument("--cl-ticks", type=int, default=0, help="own-latent ticks between the prompt and the answer")


@registry.probe("cipher_lookup", hypothesis=["HZLK"], backend="peft", add_args=_add_args,
                description="teacher-forced per-letter accuracy and log p on cipher decoding, table direction x format")
def run(loaded, tok, args) -> dict:
    rows = [json.loads(l) for l in open(args.cl_items) if l.strip()]
    rows = [r for r in rows if r.get("task") == "ladder" and r.get("revealed", r.get("pairs_shown")) == args.cl_revealed]
    fmts = args.cl_formats.split(",")
    variants = [(t, f, o) for o in args.cl_orders.split(",") for t in args.cl_tables.split(",")
                for f in fmts if f != "row"] + [("row", "row", o) for o in args.cl_orders.split(",") if "row" in fmts]
    cells = collections.defaultdict(lambda: collections.Counter())
    by_pos = collections.defaultdict(lambda: collections.Counter())
    t0, per_row = time.time(), {}
    for i, r in enumerate(rows):
        for table, fmt, order in variants:
            sc = _score(loaded, tok, _prompt(r, table, fmt, order), _segments(r, fmt), ticks=args.cl_ticks)
            read = [s for s in sc if s["letter"]] if fmt in ("trace", "row") else sc
            table = r.get("table", table) if fmt == "row" else table
            n_letters = sum(ch.isalpha() for ch in r["answer"])
            c = cells[(r["view"], r["payload_kind"], table, fmt, order)]
            c["rows"] += 1; c["units"] += len(read); c["hit"] += sum(s["hit"] for s in read)
            c["logp_x1e3"] += round(1e3 * sum(s["logp"] for s in read)); c["letters"] += n_letters
            if fmt in ("trace", "row"):
                # a letter met before in the answer can be copied from the model's own earlier pair; only a
                # first occurrence after the first pair is a lookup in the key
                seen = set()
                for s, a in zip(read, (ch for ch in r["answer"] if ch.isalpha())):
                    if a not in seen and s["pos"] > 0:
                        c["first_n"] += 1; c["first_p50"] += s["logp"] > -0.6931
                    seen.add(a)
                    b = by_pos[(r["view"], table, order, min(s["pos"], 16))]
                    b["n"] += 1; b["hit"] += s["hit"]
            per_row[f"{r['id']}|{table}|{fmt}|{order}"] = [round(s["logp"], 3) for s in read]
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            progress(f"[cipher_lookup] {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)")
    table_out = [{"view": k[0], "payload": k[1], "table": k[2], "format": k[3], "order": k[4], "rows": c["rows"],
                  "unit": "letter" if k[3] in ("trace", "row") else "token", "acc": round(c["hit"] / max(c["units"], 1), 3),
                  "logp_per_letter": round(c["logp_x1e3"] / 1e3 / max(c["letters"], 1), 3),
                  "first_p50": round(c["first_p50"] / c["first_n"], 3) if c["first_n"] else None}
                 for k, c in sorted(cells.items())]
    pos_out = [{"view": k[0], "table": k[1], "order": k[2], "pos": k[3], "n": c["n"], "acc": round(c["hit"] / c["n"], 3)}
               for k, c in sorted(by_pos.items())]
    return {"model": args.model, "file": str(args.cl_items), "revealed": args.cl_revealed, "n_rows": len(rows),
            "ticks": args.cl_ticks, "chance": {"acc": round(1 / 26, 3), "logp_per_letter": -3.258},
            "first_p50": "share of first occurrences (not the first pair) with p(gold letter) > 0.5",
            "cells": table_out, "by_position": pos_out, "per_row_logp": per_row,
            "_summary": {f"{t['view']}/{t['payload']}/{t['table']}/{t['order']} first_p50": t["first_p50"]
                         for t in table_out if t["format"] in ("trace", "row")}}
