#!/usr/bin/env python3
"""head_ablation_probe.py — which WKV heads does each continuation depend on?

Rework of H20-H22 (owner 2026-10-21): a classifier head trained on the state cannot tell
"stored in the state" from "present in the words" (C-H21-H22-redesign: the same MLP on word
n-grams matched it). This asks the model instead: after the prompt, zero ONE head's WKV state
(layer L, head h) and re-score each named continuation from that state. The drop in log p is
how much that continuation depends on what that head holds. No probe is trained.

Items (jsonl, from head_ablation_items.py): {"id", "prompt", "conts": {name: text}, ...}.
The continuation is scored token by token after the prompt; only the continuation's own tokens
count. Output: <out>.npz with
    base  [items, conts]          log p of each continuation, intact state
    delta [items, heads, conts]   log p with head ablated - base   (heads = layer * H + h)
and <out>.json with the item ids, continuation names and shapes. Analysis per hypothesis lives
in head_ablation_report.py.

    --gate-only   score the intact state only (behavioural gate: does the model separate the
                  classes at all? if not, there is no head to look for)

Inference-grade on CPU via the peft backend, like state_readout_lens.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.rl.loader import load_rwkv7, _PeftState  # noqa: E402
from experiments._common import registry  # noqa: E402
from experiments._common.runtime import limit_threads, progress  # noqa: E402


def cont_logp(loaded, shift, wkv, last_id: int, cont_ids: list) -> float:
    """Sum log p(cont_ids | state), feeding the prompt's last token first."""
    inp = torch.tensor([[last_id] + cont_ids[:-1]])
    with torch.no_grad():
        logits, _ = loaded.forward_stateful(inp, _PeftState(shift.clone(), wkv.clone()))
        lsm = F.log_softmax(logits[0].float(), dim=-1)
        return float(lsm.gather(1, torch.tensor(cont_ids)[:, None]).sum())


def cont_logp_ablated(loaded, shift, wkv, last_id: int, cont_ids: list, chunk: list) -> np.ndarray:
    """Same as cont_logp, once per head in `chunk` [(layer, head), ...], batched: row j of the
    batch is the state with head chunk[j] zeroed. wkv [N, 1, H, d, d], shift [N, 2, 1, C]."""
    B = len(chunk)
    w = wkv.repeat(1, B, 1, 1, 1)
    for j, (L, h) in enumerate(chunk):
        w[L, j, h] = 0
    inp = torch.tensor([[last_id] + cont_ids[:-1]] * B)
    with torch.no_grad():
        logits, _ = loaded.forward_stateful(inp, _PeftState(shift.repeat(1, 1, B, 1), w))
        lsm = F.log_softmax(logits.float(), dim=-1)                       # [B, T, V]
        tgt = torch.tensor(cont_ids)[None, :, None].expand(B, -1, 1)
        return lsm.gather(2, tgt)[..., 0].sum(1).numpy()


def ablate(loaded, tok, rows: list, heads: list, batch: int = 64, gate_only: bool = False, log=print):
    """Core: base [items, conts] log p, and delta [items, heads, conts] with each head zeroed."""
    names = list(rows[0]["conts"])
    base = np.zeros((len(rows), len(names)), np.float32)
    delta = None if gate_only else np.zeros((len(rows), len(heads), len(names)), np.float32)
    t0 = time.time()
    for i, r in enumerate(rows):
        ids = tok.encode(r["prompt"])
        st = loaded.new_state(batch=1)
        with torch.no_grad():
            _, st = loaded.forward_stateful(torch.tensor([ids[:-1]]), st)
        shift, wkv = st.shift.detach().clone(), st.wkv.detach().clone()
        conts = [tok.encode(r["conts"][n]) for n in names]
        for c, cid in enumerate(conts):
            base[i, c] = cont_logp(loaded, shift, wkv, ids[-1], cid)
        if delta is not None:
            for s in range(0, len(heads), batch):
                chunk = heads[s:s + batch]
                for c, cid in enumerate(conts):
                    delta[i, s:s + len(chunk), c] = (
                        cont_logp_ablated(loaded, shift, wkv, ids[-1], cid, chunk) - base[i, c])
        el = time.time() - t0
        log(f"[head_abl] {i + 1}/{len(rows)} {el:.0f}s, ~{el / (i + 1) * (len(rows) - i - 1):.0f}s left")
    return names, base, delta


def _parse_heads(spec: str) -> list:
    return [tuple(int(v) for v in x.split(":")) for x in spec.split(",") if x.strip()]


# Heads found on G1i 2.9B, 2026-10-22 (C-H20-H22-heads): L14h26 is the top head for H22 binding
# and presence and H21 valid; L19h30, L19h34, L20h6, L25h32 carry H21 invalid rejection.
KNOWN_HEADS_G1I29 = "14:26,11:37,21:5,19:30,19:34,20:6,25:32,3:17,17:21"


def _add_args(ap):
    ap.add_argument("--kh-items", type=Path, action="append",
                    help="items jsonl (head_ablation_items.py), with 'kind' and 'y'; repeatable")
    ap.add_argument("--kh-heads", default=KNOWN_HEADS_G1I29, help="layer:head,... to ablate")
    ap.add_argument("--kh-limit", type=int, default=0)


@registry.probe("known_heads", hypothesis=["H20", "H21", "H22", "HZLK"], backend="peft", add_args=_add_args,
                description="zero a few known WKV heads after the prompt; per kind: how much each is needed for the right Yes/No")
def run(loaded, tok, args) -> dict:
    heads = _parse_heads(args.kh_heads)
    out = {"model": args.model, "heads": [f"L{L}h{h}" for L, h in heads], "files": {}}
    summ = {}
    for f in args.kh_items:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        if args.kh_limit:
            rows = rows[:args.kh_limit]
        names, base, delta = ablate(loaded, tok, rows, heads, batch=len(heads), log=progress)
        yi, ni = names.index("YES"), names.index("NO")
        m = base[:, yi] - base[:, ni]
        y = np.array([r.get("y", 1) for r in rows])
        sgn = np.where(y == 1, 1.0, -1.0)
        harm = -((delta[:, :, yi] - delta[:, :, ni]) * sgn[:, None])   # >0: head needed for the right answer
        kinds = np.array([r.get("kind", "") for r in rows])
        res = {}
        for kind in sorted(set(kinds)):
            sel = kinds == kind
            res[kind] = {"n": int(sel.sum()),
                         "acc": round(float(((m[sel] > 0) == (y[sel] == 1)).mean()), 3),
                         "harm": {f"L{L}h{h}": round(float(harm[sel, j].mean()), 3) for j, (L, h) in enumerate(heads)}}
            summ[f"{Path(f).stem}/{kind} acc"] = str(res[kind]["acc"])
            summ[f"{Path(f).stem}/{kind} L14h26 harm"] = str(res[kind]["harm"].get("L14h26"))
        out["files"][str(f)] = res
    out["_summary"] = summ
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="path stem")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--layers", default="", help="subset of layers to ablate (default all)")
    ap.add_argument("--heads", default="", help="explicit layer:head,... (overrides --layers)")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64, help="ablated heads per batched forward")
    args = ap.parse_args()
    limit_threads(args.threads)

    rows = [json.loads(l) for l in open(args.items) if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    names = list(rows[0]["conts"])
    loaded = load_rwkv7(args.model, device="cpu", dtype=torch.float32, backend="peft", ctx_len=2048)
    for p in loaded.model.parameters():
        p.requires_grad_(False)
    tok = loaded.tokenizer
    st0 = loaded.new_state(batch=1)
    n_layer, _, H = st0.wkv.shape[0], st0.wkv.shape[1], st0.wkv.shape[2]
    layers = [int(x) for x in args.layers.split(",") if x.strip()] or list(range(n_layer))
    heads = _parse_heads(args.heads) if args.heads else [(L, h) for L in layers for h in range(H)]

    names, base, delta = ablate(loaded, tok, rows, heads, batch=args.batch, gate_only=args.gate_only, log=progress)
    np.savez(str(args.out) + ".npz", base=base, **({} if delta is None else {"delta": delta}))
    meta = {"model": args.model, "items": str(args.items), "n": len(rows), "conts": names,
            "ids": [r["id"] for r in rows], "labels": [r.get("y") for r in rows],
            "heads": heads, "n_heads_per_layer": H, "gate_only": args.gate_only,
            "ablation": "zero one head's WKV state after the prompt"}
    Path(str(args.out) + ".json").write_text(json.dumps(meta, indent=1))
    print(f"written {args.out}.npz / .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
