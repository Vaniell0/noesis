#!/usr/bin/env python3
"""state_step_influence_probe.py — how much one step moves the WKV state, by kind of step.

The question (user, 2026-10-06): the state updates on every token; what differs is how *useful* the
updates are. Teacher CoT spends hundreds of tokens, a silent re-read (N=2) spends the question again, a
latent tick spends none. What does each kind of step do to the state, and how much of the answer is open
(entropy) while it does it? Previously we only had loop ticks and marker-chain steps (state_trajectory_probe,
4 prompts) — no CoT tokens and no read tokens with deltas. This fills that in, all from ONE post-question
state so the branches are comparable:

  read1   the question tokens, fed one by one (the first pass)
  read2   the same tokens fed again (the N=2 re-read) — continues from the end of read1
  cot     greedy tokens after "<think>" (the teacher's own thinking)
  tick_e  latent ticks fed with the model's own expected embedding (renormalised to a median token norm)
  tick_c  latent ticks fed with a constant vector (the mean embedding)

per step, per layer: relative displacement ||S_t - S_{t-1}||_F / ||S_{t-1}||_F, the cosine of this
displacement with the previous one (same layer; positive = pushing one way, negative = undoing the last
step), and the entropy of the next-token distribution (the answer side). No training.

    training/.venv/bin/python experiments/rl/state_step_influence_probe.py \\
        --model ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth --items 6 \\
        --out experiments/rl/results/state_step_influence_g1d04b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rl.loader import load_rwkv7, _PeftState  # noqa: E402
from experiments.rl.matrix_n_probe import FORMATS, make_items, with_format  # noqa: E402


def _clone(st):
    return _PeftState(st.shift.clone(), st.wkv.clone())


def _layer_norms(x: torch.Tensor) -> torch.Tensor:
    return x.float().flatten(1).norm(dim=1)  # [n_layer]


class Recorder:
    """Accumulates per-step stats for one branch."""

    def __init__(self):
        self.rel, self.cos, self.ent = [], [], []
        self._prev_delta = None

    def step(self, wkv_before, wkv_after, logits):
        d = (wkv_after.float() - wkv_before.float()).flatten(1)          # [L, ...]
        self.rel.append((d.norm(dim=1) / _layer_norms(wkv_before).clamp_min(1e-12)).tolist())
        if self._prev_delta is not None:
            c = F.cosine_similarity(d, self._prev_delta, dim=1)
            self.cos.append(c.tolist())
        self._prev_delta = d
        p = F.softmax(logits[0, -1].float(), -1)
        self.ent.append(float(-(p * torch.log(p.clamp_min(1e-12))).sum()))


@torch.no_grad()
def feed_tokens(loaded, st, ids, rec):
    logits = None
    for t in ids:
        before = st.wkv.clone()
        logits, st = loaded.forward_stateful(torch.tensor([[t]]), st)
        rec.step(before, st.wkv, logits)
    return logits, st


@torch.no_grad()
def feed_ticks(loaded, st, logits, n, mode, feed_norm, mean_emb, rec):
    E = loaded.embedding_weight.float()
    for _ in range(n):
        feed = (F.softmax(logits[0, -1].float(), -1) @ E) if mode == "expected" else mean_emb.clone()
        feed = feed * (feed_norm / feed.norm().clamp_min(1e-8))
        before = st.wkv.clone()
        logits, st = loaded.forward_stateful_embeds(feed.to(loaded.dtype).view(1, 1, -1), st)
        rec.step(before, st.wkv, logits)
    return logits, st


@torch.no_grad()
def one_item(loaded, q, n_cot, n_ticks, feed_norm, mean_emb):
    tok = loaded.tokenizer
    head = tok.encode(f"User: {q}\n\nAssistant: <think>")
    recs = {k: Recorder() for k in ("read1", "read2", "cot", "tick_e", "tick_c")}
    logits, st_read = feed_tokens(loaded, loaded.new_state(batch=1), head, recs["read1"])
    # branches from the post-question state (copies, so they are comparable)
    feed_tokens(loaded, _clone(st_read), head, recs["read2"])
    lg, st = logits, _clone(st_read)
    for _ in range(n_cot):
        nxt = int(lg[0, -1].float().argmax())
        if nxt == 0:
            break
        before = st.wkv.clone()
        lg, st = loaded.forward_stateful(torch.tensor([[nxt]]), st)
        recs["cot"].step(before, st.wkv, lg)
    feed_ticks(loaded, _clone(st_read), logits, n_ticks, "expected", feed_norm, mean_emb, recs["tick_e"])
    feed_ticks(loaded, _clone(st_read), logits, n_ticks, "const", feed_norm, mean_emb, recs["tick_c"])
    return {k: {"rel": r.rel, "cos": r.cos, "ent": r.ent} for k, r in recs.items()}, len(head)


def summarise(items: list[dict]) -> dict:
    out = {}
    for seg in ("read1", "read2", "cot", "tick_e", "tick_c"):
        rel_all, cos_all, ent_all, rel_first = [], [], [], []
        per_layer = None
        for it in items:
            r = it[seg]["rel"]
            if not r:
                continue
            t = torch.tensor(r)                          # [steps, L]
            rel_all.append(t.mean())
            rel_first.append(t[:10].mean())
            per_layer = t.mean(0) if per_layer is None else per_layer + t.mean(0)
            if it[seg]["cos"]:
                cos_all.append(torch.tensor(it[seg]["cos"]).mean())
            ent_all.append(torch.tensor(it[seg]["ent"]).mean())
        n = len(rel_all)
        if not n:
            continue
        out[seg] = {"n_items": n,
                    "rel_step_mean": float(torch.stack(rel_all).mean()),
                    "rel_step_first10": float(torch.stack(rel_first).mean()),
                    "cos_prev_mean": float(torch.stack(cos_all).mean()) if cos_all else None,
                    "entropy_mean": float(torch.stack(ent_all).mean()),
                    "steps_mean": sum(len(it[seg]["rel"]) for it in items) / n,
                    "rel_step_by_layer_group": {
                        "L0-7": float((per_layer / n)[:8].mean()), "L8-15": float((per_layer / n)[8:16].mean()),
                        "L16-23": float((per_layer / n)[16:].mean())}}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", type=int, default=6)
    ap.add_argument("--levels", default="1,2")
    ap.add_argument("--format", default="json")
    ap.add_argument("--cot", type=int, default=96)
    ap.add_argument("--ticks", type=int, default=32)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    loaded = load_rwkv7(args.model, device="cpu", dtype=torch.float32, backend="peft", lora_r=0, ctx_len=8192)
    E = loaded.embedding_weight.float()
    feed_norm, mean_emb = float(E.norm(dim=-1).median()), E.mean(0)
    its = [it for it in make_items(args.items, [int(x) for x in args.levels.split(",")], args.seed)
           if it["family"] == "pattern"][:args.items]
    items = []
    for i, it in enumerate(its):
        q = with_format(it["prompt"], FORMATS[args.format][0])
        rec, n_head = one_item(loaded, q, args.cot, args.ticks, feed_norm, mean_emb)
        items.append(rec)
        print(f"[influence] {i + 1}/{len(its)} question tokens {n_head}", flush=True)
    summ = summarise(items)
    print(json.dumps(summ, indent=1))
    args.out.write_text(json.dumps({"model": args.model, "n_items": len(items), "summary": summ,
                                    "items": items}, indent=1))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
