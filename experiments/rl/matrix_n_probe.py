#!/usr/bin/env python3
"""matrix_n_probe.py — what does extra read-time buy, content or format? (H10's N axis, on matrices)

H10's N data (N=2 silent 33.3% beat every CoT cell, N=3 collapsed) came from tasks
very unlike matrices, and from a LoRA-trained model, so it cannot separate
CONTENT from FORMAT. A matrix can carry the same encoded content in many answer
formats, so here the content (the grid and its answer) is held fixed per item and
only the instruction suffix changes (4 formats), while the arm changes how much
read-time the model gets:

  n1     the wrapped prompt once                               (baseline)
  n2     the wrapped prompt fed twice                          (re-read, H10's N=2)
  n3     the wrapped prompt fed three times                    (H10's N=3)
  pad2   the prompt once + neutral filler of the same length   (time WITHOUT the
         as the prompt, placed after it                         content again)
  cot    the model's own <think> chain (up to --cot-k tokens),   (the expensive way
         then </think> forced and the answer decoded             to move the state)

G1-series chat models open <think> by default (measured 2026-10-02: a 24-token
budget never leaves the first sentence of the reasoning). The silent arms therefore
close the think block themselves (`<think>\n</think>\n`) and the answer is decoded
straight away; only `cot` lets the model think in tokens. Per generation we also
record how many tokens were generated before the answer, so accuracy per token can
be read off.

pad2 is the control the old N experiments never had: if n2 helps but pad2 does not,
the gain is re-exposure to the content, not "more steps"; if both help, it is time.

Two scores per generation, kept apart:
  content  the first integer in the response equals the answer (lenient)
  format   the first line matches the requested format exactly (strict)

No training; same weights, greedy decode. Paired by item across arms.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "A0_eval"))
import gen_tasks as G  # noqa: E402
from experiments.rl.loader import load_rwkv7, load_weights_into  # noqa: E402

FORMATS = {
    "num":  ("Output only the number.",
             lambda l: re.fullmatch(r"-?\d+", l.strip()) is not None),
    "eq":   ('Reply with one line of the form "answer = " followed by the number.',
             lambda l: re.fullmatch(r"answer\s*=\s*-?\d+", l.strip()) is not None),
    "json": ('Reply with JSON only: {"answer": <number>}',
             lambda l: _is_json_answer(l)),
    # Paraphrases of json/sent: same output type, different wording. They give
    # the noise floor for "the format changed the content" (G1i, 2026-10-02:
    # json and sent disagreed on 32/48 items, low accuracy, no floor yet).
    "json2": ('Return only a JSON object with the key "result" holding the number.',
              lambda l: _is_json_key(l, "result")),
    "sent2": ("Answer with a single sentence that states the number.",
              lambda l: re.fullmatch(r"[A-Z].*\d+.*\.", l.strip()) is not None),
    "sent": ('Reply with one sentence: "The answer is" followed by the number and a period.',
             lambda l: re.fullmatch(r"The answer is -?\d+\.?", l.strip()) is not None),
}
FILLER = ("The river looks calm in the evening. A quiet street stays grey after the "
          "rain. The old library feels warm during the week. ")


def _is_json_answer(line: str) -> bool:
    try:
        d = json.loads(line.strip())
        return isinstance(d, dict) and "answer" in d
    except Exception:
        return False


def _is_json_key(line: str, key: str) -> bool:
    try:
        d = json.loads(line.strip())
        return isinstance(d, dict) and key in d
    except Exception:
        return False


def make_items(per_cell: int, levels: list, seed: int) -> list:
    """Items from the project's own generators, rejection-sampled by family/level.
    Answers are numbers only, so one grid supports every format."""
    rng = random.Random(seed)
    items = []
    for fam, gen, need in (("pattern", G._pattern_gen, None),
                           ("arith", G._arith_gen, "find_sum")):
        for lvl in levels:
            got, tries = [], 0
            while len(got) < per_cell and tries < 4000:
                tries += 1
                t = gen(rng, tries)
                if t is None or t.get("level") != lvl:
                    continue
                if need and need not in t["id"]:
                    continue
                if not re.fullmatch(r"-?\d+", str(t["answer"])):
                    continue
                got.append(t)
            for t in got:
                items.append({"family": fam, "level": lvl, "prompt": t["prompt"],
                              "answer": str(t["answer"])})
    return items


def with_format(prompt: str, instr: str) -> str:
    out = re.sub(r"Output only[^\n]*$", instr, prompt.strip())
    return out if out != prompt.strip() else prompt.strip() + "\n" + instr


@torch.no_grad()
def _decode(loaded, logits, st, max_new, stop):
    tok = loaded.tokenizer
    out = []
    for _ in range(max_new):
        nxt = int(logits[0, -1].float().argmax())
        if nxt == 0:
            # EOS. decode() of a list that contains id 0 returns a single U+FFFD
            # for the WHOLE list — the first smoke scored correct JSON answers as
            # garbage because of exactly this (found 2026-10-02).
            break
        out.append(nxt)
        if stop(tok.decode(out)):
            break
        logits, st = loaded.forward_stateful(torch.tensor([[nxt]]), st)
    return out, logits, st


def _answer_stop(piece: str) -> bool:
    return "\n\n" in piece or "User:" in piece


@torch.no_grad()
def run_arm(loaded, arm: str, q: str, n_tok_q: int, cot_k: int, max_new: int = 24):
    """Returns (response_text, tokens_generated_before_answer, tokens_read)."""
    tok = loaded.tokenizer
    wrapped = f"User: {q}\n\n"
    close = "Assistant: <think>\n</think>\n"
    think_tokens = 0
    if arm == "cot":
        text = wrapped + "Assistant: <think>"
    elif arm == "n1":
        text = wrapped + close
    elif arm == "n2":
        text = wrapped * 2 + close
    elif arm == "n3":
        text = wrapped * 3 + close
    elif arm == "pad2":
        filler = ""
        while len(tok.encode(filler)) < n_tok_q:
            filler += FILLER
        text = wrapped + filler.strip() + "\n\n" + close
    else:
        raise ValueError(arm)
    ids = tok.encode(text)
    st = loaded.new_state(batch=1)
    logits, st = loaded.forward_stateful(torch.tensor([ids]), st)
    if arm == "cot":
        think, logits, st = _decode(loaded, logits, st, cot_k, lambda p: "</think>" in p)
        think_tokens = len(think)
        if "</think>" not in tok.decode(think):
            closer = tok.encode("\n</think>\n")
            logits, st = loaded.forward_stateful(torch.tensor([closer]), st)
        else:
            logits, st = loaded.forward_stateful(torch.tensor([tok.encode("\n")]), st)
    out, _, _ = _decode(loaded, logits, st, max_new, _answer_stop)
    s = tok.decode(out)
    return s.split("\n\n")[0].split("User:")[0], think_tokens, len(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--per-cell", type=int, default=8)
    ap.add_argument("--levels", default="2,4")
    ap.add_argument("--formats", default="num,eq,json,sent")
    ap.add_argument("--arms", default="n1,n2,n3,pad2,cot")
    ap.add_argument("--cot-k", type=int, default=192)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    loaded = load_rwkv7(args.model, device="cpu", dtype=torch.float32,
                        backend="peft", lora_r=0, ctx_len=4096)
    if args.weights is not None:
        load_weights_into(loaded, args.weights)
    tok = loaded.tokenizer
    items = make_items(args.per_cell, [int(x) for x in args.levels.split(",")], args.seed)
    fmts = args.formats.split(","); arms = args.arms.split(",")
    print(f"[n-probe] {len(items)} items x {len(fmts)} formats x {len(arms)} arms", flush=True)

    rows = []
    for i, it in enumerate(items):
        for f in fmts:
            q = with_format(it["prompt"], FORMATS[f][0])
            n_tok_q = len(tok.encode(f"User: {q}\n\n"))
            for arm in arms:
                resp, n_think, n_read = run_arm(loaded, arm, q, n_tok_q, args.cot_k)
                nums = re.findall(r"-?\d+", resp)
                first_line = next((l for l in resp.split("\n") if l.strip()), "")
                rows.append({"family": it["family"], "level": it["level"], "item": i,
                             "format": f, "arm": arm, "response": resp[:120], "think_tokens": n_think, "read_tokens": n_read,
                             "content": bool(nums and nums[0] == it["answer"]),
                             "format_ok": FORMATS[f][1](first_line)})
        print(f"[n-probe] item {i+1}/{len(items)}", flush=True)

    def rate(sel, key):
        sel = list(sel)
        return round(sum(r[key] for r in sel) / max(1, len(sel)), 3)

    summary = {"by_arm": {}, "by_arm_format": {}, "by_family_level_arm": {}}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        summary["by_arm"][arm] = {"content": rate(rs, "content"), "format": rate(rs, "format_ok"), "n": len(rs),
                                  "mean_think_tokens": round(sum(r["think_tokens"] for r in rs) / max(1, len(rs)), 1),
                                  "mean_read_tokens": round(sum(r["read_tokens"] for r in rs) / max(1, len(rs)), 1)}
        for f in fmts:
            rf = [r for r in rs if r["format"] == f]
            summary["by_arm_format"][f"{arm}|{f}"] = {"content": rate(rf, "content"), "format": rate(rf, "format_ok")}
        for fam in ("pattern", "arith"):
            for lvl in sorted({r["level"] for r in rs}):
                rc = [r for r in rs if r["family"] == fam and r["level"] == lvl]
                summary["by_family_level_arm"][f"{fam}|L{lvl}|{arm}"] = {"content": rate(rc, "content"), "format": rate(rc, "format_ok")}
    print(json.dumps(summary["by_arm"], indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model, "weights": str(args.weights),
                                    "summary": summary, "rows": rows}, indent=1))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
