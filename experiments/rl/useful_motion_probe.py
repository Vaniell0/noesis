#!/usr/bin/env python3
"""useful_motion_probe.py — does a step move the WKV state TOWARD the answer, or just move it?

The second half of T-influence (HZLK section 7). state_step_influence_probe.py measured how FAR one step
moves the state (|dS|/|S|: read 0.24, re-read 0.20, CoT 0.21, expected-feed tick 0.21, constant tick 0.04)
and said itself that magnitude is not usefulness. This measures the useful part, for the same branches
from ONE post-question state:

  read2   the question fed again (N=2)
  cot     the model's own greedy thinking tokens
  tick_e  latent ticks of the model's own expected embedding (median-token norm)
  tick_c  latent ticks of a constant vector (mean embedding) - control, known fixed point

At every state along a branch the answer is read by forcing "\\n</think>\\n" + '{"answer": N}' and scoring
only the number tokens (same closing as cot_vs_routes_probe.py). Per step:

  d_lp      lp(S_t) - lp(S_{t-1})              actual change in log p(correct answer)    (causal)
  d_lp_w    the same for a WRONG answer (another item's)  -> specific = d_lp - d_lp_w
  pred      <G_{t-1}, S_t - S_{t-1}>           first-order prediction, G = d lp / d S (readout lens)
  cos       cos(G_{t-1}, S_t - S_{t-1})         how much of the motion points along the answer gradient
  rel       |S_t - S_{t-1}| / |S_{t-1}|         magnitude, for comparison with the earlier probe

Prediction written before the run (2026-10-20): per unit of motion, tick_e is no more useful than cot on
an untrained model (ZLK.2: the plain model cannot use extra time); tick_c is ~0 by every measure; the
specific part (correct minus wrong answer) of cot is positive but small per token. If tick_e already moves
lp(correct) as much per step as cot, the channel exists before training and C1 only has to shape it.

    python experiments/rl/useful_motion_probe.py \\
        --model ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth --items 6 \\
        --out experiments/rl/results/useful_motion_g1d04b.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.rl.loader import load_rwkv7, _PeftState  # noqa: E402
from experiments.rl.matrix_n_probe import FORMATS, make_items, with_format  # noqa: E402

BRANCHES = ("read2", "cot", "tick_e", "tick_c")


def _clone(st):
    return _PeftState(st.shift.clone(), st.wkv.clone())


def answer_cont(tok, answer: str):
    closer = tok.encode("\n</think>\n")
    pre = tok.encode('{"answer": ')
    num = tok.encode(answer)
    return closer + pre + num, len(closer) + len(pre)


def answer_lp(loaded, st, cont, target_from, grad: bool):
    """log p(number tokens) after forcing the closer; returns (lp, dlp/dwkv or None). st untouched."""
    shift = st.shift.detach().clone()
    wkv = st.wkv.detach().clone()
    if grad:
        wkv.requires_grad_(True)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        # the last state token is already in st; feed the continuation and score cont[t] from position t-1
        logits, _ = loaded.forward_stateful(torch.tensor([cont]), _PeftState(shift, wkv))
        lsm = F.log_softmax(logits[0].float(), dim=-1)
        tgt = torch.tensor(cont[target_from:])
        lp = lsm[target_from - 1:len(cont) - 1].gather(1, tgt[:, None]).sum()
        if grad:
            lp.backward()
            return float(lp.detach()), wkv.grad.detach().float()
    return float(lp), None


class Branch:
    def __init__(self):
        self.d_lp, self.d_lp_w, self.pred, self.cos, self.rel = [], [], [], [], []
        self.lp = []

    def summary(self):
        if not self.d_lp:
            return None
        n = len(self.d_lp)
        spec = [a - b for a, b in zip(self.d_lp, self.d_lp_w)]
        motion = sum(self.rel)
        return {
            "steps": n,
            "lp_start": self.lp[0], "lp_end": self.lp[-1],
            "d_lp_total": self.lp[-1] - self.lp[0],
            "specific_total": sum(spec),
            "d_lp_per_step": sum(self.d_lp) / n,
            "specific_per_step": sum(spec) / n,
            "specific_per_unit_motion": sum(spec) / motion if motion > 0 else 0.0,
            "cos_mean": statistics.mean(self.cos),
            "abs_cos_mean": statistics.mean(abs(c) for c in self.cos),
            "rel_mean": statistics.mean(self.rel),
            "pred_vs_actual_corr": _corr(self.pred, self.d_lp),
        }


def _corr(a, b):
    if len(a) < 3:
        return float("nan")
    ta, tb = torch.tensor(a), torch.tensor(b)
    ta, tb = ta - ta.mean(), tb - tb.mean()
    d = float(ta.norm() * tb.norm())
    return float(ta @ tb) / d if d > 0 else float("nan")


def run_branch(loaded, st0, logits0, kind, n_steps, head_ids, cont, tf, cont_w, tf_w, feed_norm, mean_emb):
    E = loaded.embedding_weight.float()
    br = Branch()
    st, logits = _clone(st0), logits0
    lp_prev, G = answer_lp(loaded, st, cont, tf, grad=True)
    lpw_prev, _ = answer_lp(loaded, st, cont_w, tf_w, grad=False)
    br.lp.append(lp_prev)
    feed_ids = head_ids if kind == "read2" else None
    steps = len(head_ids) if kind == "read2" else n_steps
    for t in range(steps):
        before = st.wkv.detach().float().clone()
        with torch.no_grad():
            if kind == "read2":
                logits, st = loaded.forward_stateful(torch.tensor([[feed_ids[t]]]), st)
            elif kind == "cot":
                nxt = int(logits[0, -1].float().argmax())
                if nxt == 0:
                    break
                logits, st = loaded.forward_stateful(torch.tensor([[nxt]]), st)
            else:
                feed = (F.softmax(logits[0, -1].float(), -1) @ E) if kind == "tick_e" else mean_emb.clone()
                feed = feed * (feed_norm / feed.norm().clamp_min(1e-8))
                logits, st = loaded.forward_stateful_embeds(feed.to(loaded.dtype).view(1, 1, -1), st)
        d = st.wkv.detach().float() - before
        lp_now, G_now = answer_lp(loaded, st, cont, tf, grad=True)
        lpw_now, _ = answer_lp(loaded, st, cont_w, tf_w, grad=False)
        br.d_lp.append(lp_now - lp_prev)
        br.d_lp_w.append(lpw_now - lpw_prev)
        br.pred.append(float((G * d).sum()))
        br.cos.append(float(F.cosine_similarity(G.flatten(), d.flatten(), dim=0)))
        br.rel.append(float(d.norm() / before.norm().clamp_min(1e-12)))
        br.lp.append(lp_now)
        lp_prev, lpw_prev, G = lp_now, lpw_now, G_now
    return br


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", type=int, default=6)
    ap.add_argument("--family", default="pattern")
    ap.add_argument("--levels", default="1,2")
    ap.add_argument("--format", default="json")
    ap.add_argument("--cot", type=int, default=64)
    ap.add_argument("--ticks", type=int, default=32)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()

    loaded = load_rwkv7(args.model, device="cpu", dtype=torch.float32, backend="peft", lora_r=0, ctx_len=8192)
    for p in loaded.model.parameters() if hasattr(loaded, "model") else []:
        p.requires_grad_(False)
    tok = loaded.tokenizer
    E = loaded.embedding_weight.float()
    feed_norm, mean_emb = float(E.norm(dim=-1).median()), E.mean(0)
    its = [it for it in make_items(args.items * 4, [int(x) for x in args.levels.split(",")], args.seed)
           if it["family"] == args.family][:args.items]
    if len(its) < 2:
        raise SystemExit(f"need >= 2 items of family {args.family}, got {len(its)}")

    rows = []
    for i, it in enumerate(its):
        wrong = next(o["answer"] for o in its[i + 1:] + its[:i] if o["answer"] != it["answer"])
        q = with_format(it["prompt"], FORMATS[args.format][0])
        head = tok.encode(f"User: {q}\n\nAssistant: <think>")
        cont, tf = answer_cont(tok, it["answer"])
        cont_w, tf_w = answer_cont(tok, wrong)
        with torch.no_grad():
            logits0, st0 = loaded.forward_stateful(torch.tensor([head]), loaded.new_state(batch=1))
        rec = {"answer": it["answer"], "wrong": wrong, "branches": {}}
        for kind in BRANCHES:
            br = run_branch(loaded, st0, logits0, kind, args.cot if kind == "cot" else args.ticks,
                            head, cont, tf, cont_w, tf_w, feed_norm, mean_emb)
            rec["branches"][kind] = {"summary": br.summary(), "d_lp": br.d_lp, "d_lp_w": br.d_lp_w,
                                     "cos": br.cos, "rel": br.rel, "lp": br.lp}
        rows.append(rec)
        print(f"[useful] {i + 1}/{len(its)} " + " ".join(
            f"{k}:{rec['branches'][k]['summary']['specific_total']:+.2f}" for k in BRANCHES
            if rec["branches"][k]["summary"]), flush=True)

    agg = {}
    for k in BRANCHES:
        ss = [r["branches"][k]["summary"] for r in rows if r["branches"][k]["summary"]]
        agg[k] = {m: statistics.mean(s[m] for s in ss) for m in ss[0]} if ss else None
        if ss:
            agg[k]["n_items"] = len(ss)
    print(json.dumps(agg, indent=1))
    args.out.write_text(json.dumps({"model": args.model, "family": args.family, "n_items": len(rows),
                                    "wall_s": round(time.time() - t0, 1), "summary": agg, "items": rows},
                                   indent=1))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
