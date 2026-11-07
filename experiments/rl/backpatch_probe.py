#!/usr/bin/env python3
"""backpatch_probe.py — does a two-hop answer fail because the bridge is resolved too late, and where would a tick go?

Biran et al. 2024 ("Hopping Too Late"): on two-hop queries the bridge entity is resolved in late layers, too late for
the second hop; patching the residual of a later layer into an earlier one at the last position fixes many of them,
without training. Our tick reads the last layer and writes the input embedding (src = all layers, dst = 0). Qwen 0.8B
mixed rows: 0 of 120 at M=0, while its own CoT states the capital and the answer still copies the country (2026-11-04).

Per row: the plain greedy answer, then the answer after back-patching for every (src, dst) pair with dst < src.
Control for the best pairs: the same patch taken from ANOTHER row's prompt - a fix that survives it is not the row's
own bridge but a generic push.

    python experiments/run.py --backend peft --model QWEN --tests backpatch --bp-items mixed_eval.jsonl --bp-n 40
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments._common import registry  # noqa: E402
from experiments._common.runtime import progress  # noqa: E402


def _add_args(ap):
    ap.add_argument("--bp-items", type=Path, action="append", help="task jsonl (gen_ciphers rows with check_answer)")
    ap.add_argument("--bp-n", type=int, default=40, help="first n rows")
    ap.add_argument("--bp-src", default="8,12,16,20", help="layers read from (residual after this many layers)")
    ap.add_argument("--bp-dst", default="0,2,4,6,8,10,12,14,16", help="layers written into (residual entering it)")
    ap.add_argument("--bp-control-top", type=int, default=3, help="best pairs re-run with another row's source")
    ap.add_argument("--bp-max-new", type=int, default=24)


@registry.probe("backpatch", hypothesis=["HZLK"], backend="peft", add_args=_add_args,
                description="two-hop answers after patching a later layer's residual into an earlier layer (HF models)")
def run(loaded, tok, args) -> dict:
    from experiments.rl.m0_pass_probe import answer
    from training.scripts.gen_ciphers import check_answer
    rows = [json.loads(l) for f in args.bp_items for l in open(f) if l.strip()][:args.bp_n]
    pairs = [(s, d) for s in map(int, args.bp_src.split(",")) for d in map(int, args.bp_dst.split(",")) if d < s]
    base, fixed, resp, t0 = {}, {p: [] for p in pairs}, {}, time.time()
    for i, r in enumerate(rows):
        a, _ = answer(loaded, tok, r["prompt"], args.bp_max_new)
        base[r["id"]] = bool(check_answer(r, a)[0])
        resp[r["id"]] = {"base": a}
        for s, d in pairs:
            a, _ = answer(loaded, tok, r["prompt"], args.bp_max_new, patch=(s, d))
            resp[r["id"]][f"{s}>{d}"] = a
            if check_answer(r, a)[0] and not base[r["id"]]:
                fixed[(s, d)].append(r["id"])
        progress(f"[backpatch] {i + 1}/{len(rows)} base {sum(base.values())} "
                 f"best {max(len(v) for v in fixed.values())} ({time.time() - t0:.0f}s)")
    top = sorted(pairs, key=lambda p: -len(fixed[p]))[:args.bp_control_top]
    control = {}
    for s, d in top:
        n = 0
        for i, r in enumerate(rows):
            other = rows[(i + 1) % len(rows)]
            src_ids = torch.tensor([tok.encode(loaded.silent_prompt(other["prompt"]))])
            a, _ = answer(loaded, tok, r["prompt"], args.bp_max_new, patch=(s, d, src_ids))
            n += bool(check_answer(r, a)[0]) and not base[r["id"]]
        control[f"{s}>{d}"] = n
    oracle = set().union(*fixed.values())
    return {"model": args.model, "n": len(rows), "base_right": sum(base.values()),
            "fixed": {f"{s}>{d}": len(v) for (s, d), v in fixed.items()},
            "fixed_ids": {f"{s}>{d}": v for (s, d), v in fixed.items() if v},
            "oracle_any_pair": len(oracle), "control_other_row": control, "responses": resp,
            "_summary": {"base": f"{sum(base.values())}/{len(rows)}", "oracle": len(oracle),
                         **{f"fix {s}>{d}": f"{len(fixed[(s, d)])} (ctrl {control[f'{s}>{d}']})" for s, d in top}}}
