#!/usr/bin/env python3
"""Report for head_ablation_probe.py output (H20-H22 reworked as "which heads hold what").

    python experiments/rl/head_ablation_report.py --which h20 experiments/rl/results/head_ablation/abl_h20

Predictions written before any ablation run (2026-10-21):
  gate (h21/h22): the intact model's Yes-No margin must separate the classes, AUC >= 0.65; below
      that there is no behaviour to localise and the ablation is not run.
  h20: the neck (1-2 separable facts per head, keys clustered; state_readout_lens) predicts that
      the two alternatives depend on the SAME heads and compete there: more heads whose removal
      raises one alternative and lowers the other (opposite-sign pairs) for X/Y than for X/OFF.
      Overlap of the top heads alone is confounded by topic (X and Y share the prompt's topic,
      OFF does not), so it is reported but not the criterion.
  h22: if binding is held separately from word presence, the heads that carry the binding
      margin differ from those that carry the presence margin (top-10 Jaccard < 0.3).
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def topk(v, k):
    return set(np.argsort(v)[-k:].tolist())


def jacc(a, b):
    return len(a & b) / max(len(a | b), 1)


def load(stem):
    meta = json.load(open(stem + ".json"))
    z = np.load(stem + ".npz")
    return meta, z["base"], (z["delta"] if "delta" in z.files else None)


def gate(meta, base, items):
    c = meta["conts"]
    m = base[:, c.index("YES")] - base[:, c.index("NO")]
    kinds = np.array([r.get("kind", "") for r in items])
    y = np.array([r.get("y", -1) for r in items])
    out = {}
    for kind in sorted(set(kinds)):
        sel = kinds == kind
        if len(set(y[sel])) > 1:
            out[kind] = {"n": int(sel.sum()), "auc_margin": round(auc(m[sel], y[sel]), 3),
                         "acc_sign": round(float(((m[sel] > 0) == (y[sel] == 1)).mean()), 3)}
        else:
            out[kind] = {"n": int(sel.sum()), "frac_yes": round(float((m[sel] > 0).mean()), 3)}
    if len(set(y[y >= 0])) > 1:
        sel = y >= 0
        out["all_labelled"] = {"auc_margin": round(auc(m[sel], y[sel]), 3)}
    return m, out


def h20(meta, delta, k=10, thr=0.1):
    c = meta["conts"]
    X, Y, O = (delta[:, :, c.index(n)] for n in ("X", "Y", "OFF"))
    res = {"jaccard_top%d" % k: {}, "opposite_sign_heads": {}}
    for name, A, B in (("X_Y", X, Y), ("X_OFF", X, O), ("Y_OFF", Y, O)):
        res["jaccard_top%d" % k][name] = round(float(np.mean(
            [jacc(topk(-A[i], k), topk(-B[i], k)) for i in range(len(A))])), 3)
        opp = ((np.sign(A) != np.sign(B)) & (np.abs(A) > thr) & (np.abs(B) > thr)).sum(1)
        res["opposite_sign_heads"][name] = round(float(opp.mean()), 2)
    bal = X - Y
    res["max_abs_balance_shift_per_item_mean"] = round(float(np.abs(bal).max(1).mean()), 3)
    res["heads_flipping_preference_per_item_mean"] = None  # filled by caller (needs base)
    return res


def judge(meta, base, delta, items, k=10):
    c = meta["conts"]
    dm = delta[:, :, c.index("YES")] - delta[:, :, c.index("NO")]   # change of Yes-No margin
    y = np.array([r.get("y", 1) for r in items])
    sgn = np.where(y == 1, 1.0, -1.0)[:, None]
    harm = -(dm * sgn)                                                 # >0: head needed for the right answer
    kinds = np.array([r.get("kind", "") for r in items])
    per_kind = {kind: harm[kinds == kind].mean(0) for kind in sorted(set(kinds))}
    out = {kind: {"top_heads": [[meta["heads"][j], round(float(v[j]), 3)] for j in np.argsort(v)[-k:][::-1]]}
           for kind, v in per_kind.items()}
    ks = list(per_kind)
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            out[f"jaccard_top{k}_{ks[a]}_{ks[b]}"] = round(jacc(topk(per_kind[ks[a]], k), topk(per_kind[ks[b]], k)), 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["h20", "h21", "h22"], required=True)
    ap.add_argument("stem")
    args = ap.parse_args()
    meta, base, delta = load(args.stem)
    items = [json.loads(l) for l in open(meta["items"]) if l.strip()][:meta["n"]][:len(base)]
    rep = {"stem": args.stem, "n": len(base)}
    if args.which == "h20":
        if delta is not None:
            rep.update(h20(meta, delta))
            c = meta["conts"]
            b = base[:, c.index("X")] - base[:, c.index("Y")]
            new = b[:, None] + delta[:, :, c.index("X")] - delta[:, :, c.index("Y")]
            rep["heads_flipping_preference_per_item_mean"] = round(float((np.sign(new) != np.sign(b)[:, None]).sum(1).mean()), 2)
    else:
        _, rep["gate"] = gate(meta, base, items)
        if delta is not None:
            rep.update(judge(meta, base, delta, items))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
