#!/usr/bin/env python3
"""fork_agreement_probe.py — is there a veto that does not live in the decoder?

Proposal 2026-10-05 (memory: project_noesis_track_spine_2026_10_02): think in the
state, fork the state into several paths, judge the paths, answer only if they
converge. Before building any of it, check the premise on the untrained model:

  1. SAMPLING AGREEMENT — a CONTROL, not the signal. Prefill once, clone the
     state k times, sample one answer per clone from the SAME logits. The only
     source of variation is multinomial sampling, so agreement measures the
     sharpness of one decoder distribution (classic self-consistency, Wang et al.
     2022). It is expected to track the decoder's own confidence (mean log-prob,
     first number-token probability) almost by construction — an outside review
     caught that the first version of this docstring sold it as "paths without
     tokens". It is not: the paths never diverge here.
     The path signals are elsewhere: (a) the paraphrase answers below are
     different PROMPT routes into the state (agreement across them is computed
     post hoc), (b) `phase_probe.py` gives different LATENT routes (T ticks of the
     expected feed, argmax feed, constant feed). A path signal counts as new only
     if it predicts correctness BEYOND both this control and the decoder's
     confidence.
  2. FORMAT PARAPHRASE FLOOR. Same grid answered greedily under json / json2 /
     sent / sent2. If json vs json2 (same output type, different wording) disagree
     as often as json vs sent, the 2026-10-02 "format changes the content" result
     is general prompt sensitivity, not format.

No training. Items: pattern L1-L4 from the project's own generator (arithmetic
is 0/24 in silent mode, nothing to rank there).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rl.loader import load_rwkv7, _PeftState  # noqa: E402
from experiments.rl.matrix_n_probe import FORMATS, make_items, with_format  # noqa: E402

CLOSE = "Assistant: <think>\n</think>\n"


def _clone(st):
    return _PeftState(st.shift.clone(), st.wkv.clone())


def _first_int(s: str):
    m = re.search(r"-?\d+", s)
    return m.group(0) if m else None


@torch.no_grad()
def _gen(loaded, logits, st, max_new=24, temperature=0.0, gen=None):
    """Decode from (logits, state). Returns text and per-token log-probs of the
    chosen tokens; stops at EOS (id 0), a blank line, or 'User:'."""
    tok = loaded.tokenizer
    out, lps = [], []
    for _ in range(max_new):
        lp = torch.log_softmax(logits[0, -1].float(), -1)
        if temperature > 0:
            nxt = int(torch.multinomial(torch.softmax(lp / temperature, -1), 1, generator=gen))
        else:
            nxt = int(lp.argmax())
        if nxt == 0:
            break
        out.append(nxt); lps.append(float(lp[nxt]))
        piece = tok.decode(out)
        if "\n\n" in piece or "User:" in piece:
            break
        logits, st = loaded.forward_stateful(torch.tensor([[nxt]]), st)
    text = tok.decode(out).split("\n\n")[0].split("User:")[0]
    return text, out, lps


def _first_number_token_prob(tok, ids, lps):
    """Probability the decoder gave to the first token that contains a digit."""
    for i, t in enumerate(ids):
        if re.search(r"\d", tok.decode([t])):
            return float(torch.tensor(lps[i]).exp())
    return 0.0


def auroc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-cell", type=int, default=20)
    ap.add_argument("--levels", default="1,2,3,4")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--paraphrase", default="json,json2,sent,sent2")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)

    loaded = load_rwkv7(args.model, device="cpu", dtype=torch.float32,
                        backend="peft", lora_r=0, ctx_len=4096)
    tok = loaded.tokenizer
    items = [it for it in make_items(args.per_cell, [int(x) for x in args.levels.split(",")],
                                     args.seed) if it["family"] == "pattern"]
    print(f"[fork] {len(items)} items, k={args.k}, T={args.temperature}", flush=True)

    # Resumable: one JSON line per finished item. The laptop restarted twice under
    # long CPU runs on 2026-10-02/05 and took hours of results with it.
    part = args.out.with_suffix(".partial.jsonl")
    rows = [json.loads(l) for l in open(part)] if part.exists() else []
    if rows:
        print(f"[fork] resuming after {len(rows)} finished items", flush=True)
    for i, it in enumerate(items):
        if i < len(rows):
            continue
        rec = {"level": it["level"], "answer": it["answer"], "para": {}}
        # 1. fork agreement, json format
        q = with_format(it["prompt"], FORMATS["json"][0])
        ids = tok.encode(f"User: {q}\n\n" + CLOSE)
        logits, st = loaded.forward_stateful(torch.tensor([ids]), loaded.new_state(batch=1))
        g_text, g_ids, g_lps = _gen(loaded, logits, _clone(st))
        rec["greedy"] = _first_int(g_text)
        rec["greedy_correct"] = rec["greedy"] == it["answer"]
        rec["mean_logprob"] = sum(g_lps) / max(1, len(g_lps))
        rec["first_num_prob"] = _first_number_token_prob(tok, g_ids, g_lps)
        samples = []
        for _ in range(args.k):
            s_text, _, _ = _gen(loaded, logits, _clone(st), temperature=args.temperature, gen=gen)
            samples.append(_first_int(s_text))
        rec["samples"] = samples
        vals = [s for s in samples if s is not None]
        mode, cnt = (Counter(vals).most_common(1)[0] if vals else (None, 0))
        rec["agreement"] = cnt / args.k
        rec["majority"] = mode
        rec["majority_correct"] = mode == it["answer"]
        # 2. paraphrase floor
        for f in args.paraphrase.split(","):
            qf = with_format(it["prompt"], FORMATS[f][0])
            idf = tok.encode(f"User: {qf}\n\n" + CLOSE)
            lf, sf = loaded.forward_stateful(torch.tensor([idf]), loaded.new_state(batch=1))
            t, _, _ = _gen(loaded, lf, sf)
            first_line = next((l for l in t.split("\n") if l.strip()), "")
            rec["para"][f] = {"num": _first_int(t), "format_ok": FORMATS[f][1](first_line),
                              "text": t[:80]}
        rows.append(rec)
        part.parent.mkdir(parents=True, exist_ok=True)
        with open(part, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[fork] {i+1}/{len(items)} L{it['level']} ans={it['answer']} greedy={rec['greedy']} "
              f"samples={samples}", flush=True)

    labels = [r["greedy_correct"] for r in rows]
    summary = {
        "n": len(rows),
        "greedy_acc": round(sum(labels) / len(rows), 3),
        "majority_acc": round(sum(r["majority_correct"] for r in rows) / len(rows), 3),
        "auroc_agreement": auroc([r["agreement"] for r in rows], labels),
        "auroc_mean_logprob": auroc([r["mean_logprob"] for r in rows], labels),
        "auroc_first_num_prob": auroc([r["first_num_prob"] for r in rows], labels),
    }
    fs = args.paraphrase.split(",")

    def disagree(a, b):
        pairs = [(r["para"][a]["num"], r["para"][b]["num"]) for r in rows
                 if r["para"][a]["num"] is not None and r["para"][b]["num"] is not None]
        return {"n": len(pairs), "rate": round(sum(x != y for x, y in pairs) / max(1, len(pairs)), 3)}
    summary["disagreement"] = {f"{a}|{b}": disagree(a, b)
                               for ai, a in enumerate(fs) for b in fs[ai + 1:]}
    summary["para_acc"] = {f: round(sum(r["para"][f]["num"] == r["answer"] for r in rows) / len(rows), 3)
                           for f in fs}
    summary["para_format_ok"] = {f: round(sum(r["para"][f]["format_ok"] for r in rows) / len(rows), 3)
                                 for f in fs}
    print(json.dumps(summary, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model, "args": {k: str(v) for k, v in vars(args).items()},
                                    "summary": summary, "rows": rows}, indent=1))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
