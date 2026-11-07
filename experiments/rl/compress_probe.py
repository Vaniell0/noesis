#!/usr/bin/env python3
"""compress_probe.py — does the latent loop make the model compress fresh text better? (bits per byte at M ticks vs M=0)

Owner 2026-11-05: the Kolmogorov criterion is what everything here is built on; BlinkDL ranks models by compression of
text published after their training (UncheatableEval). Here the same number measures the loop: one model, the same
fresh documents, bits per byte with M own-latent ticks fed at every window boundary versus M=0. No answers, formats or
teachers are involved, so a parasite (tags, placeholder copies) cannot show up as a gain. The difference is the
loop's dMI on text (КПД, HZLK §2).

Per document: the first --cp-tokens tokens, read in windows of --cp-window. After each window (except the last) M
ticks of the loader's feed (NOESIS_HF_FEED / RWKV latent) are fed into the state; the window's first token is then
predicted by the last tick's logits. M=0 is the plain forward. Reported per category and overall:
bits/byte, and the bits of the first --cp-near tokens after a boundary separately (where a tick acts first).

    python experiments/run.py --backend peft --model M [--weights W] --tests compress \\
        --cp-data ~/.libs/datasets/uncheatable/2026-07/test.parquet --cp-ticks 0,4
"""
from __future__ import annotations

import math
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
    ap.add_argument("--cp-data", type=Path, help="UncheatableEval parquet (content, category)")
    ap.add_argument("--cp-categories", default="wikipedia_english,arxiv_cs,bbc_news,github_python")
    ap.add_argument("--cp-docs", type=int, default=8, help="documents per category (first n)")
    ap.add_argument("--cp-tokens", type=int, default=512)
    ap.add_argument("--cp-window", type=int, default=64)
    ap.add_argument("--cp-near", type=int, default=4, help="tokens after a boundary counted separately")
    ap.add_argument("--cp-ticks", default="0,4")
    ap.add_argument("--cp-jsonl", type=Path, help="instead of --cp-data: jsonl rows {title, text} (one category 'jsonl'); "
                         "text the model read in pretraining (wiki vital articles) for the bad-sector map")
    ap.add_argument("--cp-save-tokens", action="store_true",
                    help="keep per-token bits and the rank of the real token (1 = the model's top choice): the map of "
                         "expensive tokens the model nearly reads (owner 2026-11-05: find what was written badly)")


@torch.no_grad()
def doc_bits(loaded, ids: list, ticks: int, window: int, near: int, per_token: list | None = None) -> tuple:
    """(total bits, bits of the first `near` tokens after each boundary, count of those tokens) for ids[1:].
    per_token: if a list, gets [bits, rank] of every predicted token in order."""
    E = loaded.embedding_weight
    st = loaded.new_state(batch=1)
    total, near_bits, near_n, prev = 0.0, 0.0, 0, None
    for s in range(0, len(ids), window):
        chunk = ids[s:s + window]
        if prev is not None:                                   # predict the chunk's first token from the state
            row = prev[0, -1].float()
            lp = torch.log_softmax(row, -1)[chunk[0]]
            total -= float(lp) / math.log(2)
            if per_token is not None:
                per_token.append([round(-float(lp) / math.log(2), 3), int((row > row[chunk[0]]).sum()) + 1])
            near_bits -= float(lp) / math.log(2); near_n += 1
        lg, st, h = loaded.forward_stateful_embeds(E[torch.tensor([chunk])], st, return_hidden=True)
        lp = torch.log_softmax(lg[0, :-1].float(), -1).gather(1, torch.tensor(chunk[1:])[:, None])[:, 0]
        b = -lp / math.log(2)
        total += float(b.sum())
        if per_token is not None:
            g = lg[0, :-1].float()
            rk = (g > g.gather(1, torch.tensor(chunk[1:])[:, None])).sum(1) + 1
            per_token += [[round(float(x), 3), int(k)] for x, k in zip(b, rk)]
        if s:                                                  # near a boundary only after the first window
            near_bits += float(b[:near - 1].sum()); near_n += min(near - 1, len(b))
        prev, h = lg[:, -1:], h[:, -1:]
        if s + window < len(ids):
            for _ in range(ticks):
                prev, st, h = loaded.forward_stateful_embeds(h, st, return_hidden=True)
                prev, h = prev[:, -1:], h[:, -1:]
    return total, near_bits, near_n


@registry.probe("compress", hypothesis=["HZLK"], backend="peft", add_args=_add_args,
                description="bits per byte of fresh text with M latent ticks at every window boundary vs M=0")
def run(loaded, tok, args) -> dict:
    import json
    if args.cp_jsonl:
        rows = [json.loads(l) for l in open(Path(args.cp_jsonl).expanduser()) if l.strip()][:args.cp_docs]
        cats, docs, titles = ["jsonl"], {"jsonl": [r["text"] for r in rows]}, [r.get("title", "") for r in rows]
    else:
        import pyarrow.parquet as pq
        t = pq.read_table(str(Path(args.cp_data).expanduser()), columns=["category", "content"]).to_pydict()
        cats, titles = args.cp_categories.split(","), []
        docs = {c: [] for c in cats}
        for c, x in zip(t["category"], t["content"]):
            if c in docs and len(docs[c]) < args.cp_docs:
                docs[c].append(x)
    tokens = {}
    ticks = [int(x) for x in args.cp_ticks.split(",")]
    res, t0 = {}, time.time()
    for c, texts in docs.items():
        for M in ticks:
            tot = nb = nn = nbytes = 0
            for j, x in enumerate(texts):
                ids = tok.encode(x)[:args.cp_tokens]
                pt = [] if args.cp_save_tokens else None
                b, n_b, n_n = doc_bits(loaded, ids, M, args.cp_window, args.cp_near, pt)
                if pt is not None:
                    tokens.setdefault(f"{c}/{j}", {"title": titles[j] if titles else "", "ids": ids})[f"M{M}"] = pt
                tot += b; nb += n_b; nn += n_n
                nbytes += len(tok.decode(ids[1:]).encode("utf-8"))
            res.setdefault(c, {})[f"M{M}"] = {"bits_per_byte": round(tot / nbytes, 4),
                                             "near_bits_per_token": round(nb / max(nn, 1), 3)}
            progress(f"[compress] {c} M={M}: {res[c][f'M{M}']} ({time.time() - t0:.0f}s)")
    overall = {f"M{M}": round(sum(res[c][f"M{M}"]["bits_per_byte"] for c in cats) / len(cats), 4) for M in ticks}
    return {"model": args.model, "weights": getattr(args, "weights", None), "per_category": res, "mean_bpb": overall,
            "window": args.cp_window, "tokens": args.cp_tokens, "docs_per_category": args.cp_docs, "per_token": tokens,
            "_summary": {f"mean bpb M={M}": overall[f"M{M}"] for M in ticks}}
