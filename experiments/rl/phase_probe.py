#!/usr/bin/env python3
"""phase_probe.py — what does an UNTRAINED latent phase do, against the CoT teacher?

The target the user named (2026-10-05): make M>1 work and make the first phase more
efficient than the teacher's CoT. Before training anything, the baseline on the
untrained model, on matrix items with headroom (pattern, JSON answer, silent mode):

  latent T   after "<think>", T ticks whose input is the model's own expected
             embedding softmax(logits) @ E, rescaled to the median token norm
             (the `expected` feed of wkv_loop.py with feed_norm) — then "</think>"
             is forced and the answer decoded greedily
  const T    same T ticks, but the fed vector is a constant (mean embedding at
             the same norm) — the failed constant-marker family, as a control
  argmax T   same T ticks feeding the model's own argmax TOKEN — H10's old
             self-feed loop, the discrete control
  cot K      the model thinks in tokens (up to K), then the answer — the teacher

Reported per arm: accuracy, tokens/ticks spent before the answer, and the number of
distinct tokens the latent ticks decode to (does the state move or sit at a point).
Same items in every arm. No training. The arms double as different LATENT ROUTES to
one answer: per item, agreement across routes vs the decoder's confidence (recorded
per arm) is the "paths without tokens" test that fork_agreement_probe.py does not do.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rl.loader import load_rwkv7  # noqa: E402
from experiments.rl.matrix_n_probe import FORMATS, make_items, with_format  # noqa: E402
from experiments.rl.fork_agreement_probe import _gen, _first_int, _first_number_token_prob  # noqa: E402


@torch.no_grad()
def run(loaded, q: str, arm: str, T: int, feed_norm: float, mean_emb: torch.Tensor):
    tok = loaded.tokenizer
    E = loaded.embedding_weight
    ids = tok.encode(f"User: {q}\n\nAssistant: <think>")
    logits, st = loaded.forward_stateful(torch.tensor([ids]), loaded.new_state(batch=1))
    spent, tick_tokens = 0, []
    closed = False
    if arm == "cot":
        out = []
        for _ in range(T):
            nxt = int(logits[0, -1].float().argmax())
            if nxt == 0:
                break
            out.append(nxt)
            logits, st = loaded.forward_stateful(torch.tensor([[nxt]]), st)
            if "</think>" in tok.decode(out[-8:]):
                closed = True
                break
        spent = len(out)
    else:
        for _ in range(T):
            v = logits[0, -1].float()
            tick_tokens.append(int(v.argmax()))
            if arm == "argmax":
                logits, st = loaded.forward_stateful(torch.tensor([[int(v.argmax())]]), st)
            else:
                if arm == "latent":
                    feed = F.softmax(v, -1) @ E.float()
                elif arm == "const":
                    feed = mean_emb.clone()
                else:
                    raise ValueError(arm)
                feed = feed * (feed_norm / feed.norm().clamp_min(1e-8))
                logits, st = loaded.forward_stateful_embeds(
                    feed.to(loaded.dtype).view(1, 1, -1), st)
            spent += 1
    closer = tok.encode("\n" if closed else "\n</think>\n")
    logits, st = loaded.forward_stateful(torch.tensor([closer]), st)
    text, ids_out, lps = _gen(loaded, logits, st)
    conf = {"mean_logprob": sum(lps) / max(1, len(lps)),
            "first_num_prob": _first_number_token_prob(tok, ids_out, lps)}
    return text, spent, len(set(tick_tokens)), conf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--levels", default="1,3")
    ap.add_argument("--arms", default="latent:0,latent:2,latent:8,latent:32,const:8,argmax:8,cot:512")
    ap.add_argument("--format", default="json")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    loaded = load_rwkv7(args.model, device="cpu", dtype=torch.float32,
                        backend="peft", lora_r=0, ctx_len=8192)
    E = loaded.embedding_weight.float()
    norms = E.norm(dim=-1)
    feed_norm = float(norms.median())
    mean_emb = E.mean(0)
    items = [it for it in make_items(args.per_cell, [int(x) for x in args.levels.split(",")],
                                     args.seed) if it["family"] == "pattern"]
    arms = [(a.split(":")[0], int(a.split(":")[1])) for a in args.arms.split(",")]
    print(f"[phase] {len(items)} items, arms {arms}, feed_norm {feed_norm:.4f}", flush=True)

    # Resumable, one JSON line per finished item (list of that item's arm rows).
    part = args.out.with_suffix(".partial.jsonl")
    done = [json.loads(l) for l in open(part)] if part.exists() else []
    rows = [r for item_rows in done for r in item_rows]
    if done:
        print(f"[phase] resuming after {len(done)} finished items", flush=True)
    for i, it in enumerate(items):
        if i < len(done):
            continue
        q = with_format(it["prompt"], FORMATS[args.format][0])
        for arm, T in arms:
            text, spent, distinct, conf = run(loaded, q, arm, T, feed_norm, mean_emb)
            num = _first_int(text)
            rows.append({"item": i, "level": it["level"], "arm": f"{arm}:{T}", "answer": it["answer"],
                         "pred": num, "correct": num == it["answer"], "spent": spent,
                         "distinct_tick_tokens": distinct, "text": text[:80], **conf})
        part.parent.mkdir(parents=True, exist_ok=True)
        with open(part, "a") as f:
            f.write(json.dumps(rows[-len(arms):]) + "\n")
        line = " ".join(f"{r['arm']}={r['pred']}" for r in rows[-len(arms):])
        print(f"[phase] {i+1}/{len(items)} ans={it['answer']} {line}", flush=True)

    summary = {}
    for arm, T in arms:
        rs = [r for r in rows if r["arm"] == f"{arm}:{T}"]
        summary[f"{arm}:{T}"] = {
            "acc": round(sum(r["correct"] for r in rs) / len(rs), 3),
            "mean_spent": round(sum(r["spent"] for r in rs) / len(rs), 1),
            "mean_distinct_tick_tokens": round(sum(r["distinct_tick_tokens"] for r in rs) / len(rs), 2),
            "n": len(rs)}
    print(json.dumps(summary, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model, "feed_norm": feed_norm,
                                    "summary": summary, "rows": rows}, indent=1))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
