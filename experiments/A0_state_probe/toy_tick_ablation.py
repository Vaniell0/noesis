#!/usr/bin/env python3
"""Does each tick use its own heads ("MoE in time") or one procedure repeated? Per-tick head ablation on the toy.

    python experiments/A0_state_probe/toy_tick_ablation.py results/closed_loop_toy/s1e_lr_ce_seed1.pt [...]

For a trained closed-loop toy (its .json beside the .pt gives task and size), each head's read-out (its slice of the
time-mix output before W_o) is zeroed at one tick only; the state write is kept. Accuracy drop per (layer, head,
tick) on held-out items: chase at hops 4-8, derivations on val_laws at depth 4-8. Tick profiles are compared by
cosine (plan noesis-closed-loop-toy, section 1r).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import closed_loop_toy as T                                                     # noqa: E402

ABL = {"layer": -1, "head": -1, "tick": -1, "now": -2}
def run(model, tokens, ticks):
    st = model.init_state(tokens.shape[0])
    e = model.emb(tokens)
    ABL["now"] = -2
    for t in range(tokens.shape[1]):
        h, st = model.step(e[:, t], st)
    for t in range(ticks):
        ABL["now"] = t
        h, st = model.step(h, st)
    ABL["now"] = -2
    return model.head(h)


def head_mask_hook(model):
    """Zero one head's slice of (o * g) at the ablated tick via a forward pre-hook on W_o."""
    def pre(mod, inp):
        if ABL["now"] == ABL["tick"] and mod._layer == ABL["layer"]:
            x = inp[0].clone()
            N = mod._N
            x[:, ABL["head"] * N:(ABL["head"] + 1) * N] = 0
            return (x,)
    for att in model.att:
        att.W_o._layer, att.W_o._N = att.layer, att.N
        att.W_o.register_forward_pre_hook(pre)


@torch.no_grad()
def main() -> int:
    out = {}
    for path in sys.argv[1:]:
        path = Path(path)
        a = json.loads(path.with_suffix(".json").read_text())["args"]
        task = a.get("task") or "chase"
        extra = 0
        if task == "deriv":
            T.load_deriv(a["deriv_file"], a.get("deriv_answer", "truth"), a.get("jepa_target", "hops"))
            T.ANON["on"] = a.get("anon", False)
            extra = len(T.DERIV_SYMS)
        model = T.ToyRWKV7(a["width"], a["heads"], a["layers"], T.VOCAB + extra)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        T.DEV = torch.device("cpu")
        head_mask_hook(model)
        M = a["ticks"]
        g = torch.Generator().manual_seed(123)
        batches = []
        for k in range(4, 9):
            b = T.make_deriv(256, [k], g, "val_laws") if task == "deriv" else T.make_batch(256, a["n"], [k], g)
            batches.append(b)
        def acc():
            return sum((run(model, b["tokens"], M).argmax(-1) == b["answer"]).float().mean().item()
                       for b in batches) / len(batches)
        base = acc()
        drop = torch.zeros(a["layers"], a["heads"], M)
        for L in range(a["layers"]):
            for hd in range(a["heads"]):
                for t in range(M):
                    ABL.update(layer=L, head=hd, tick=t)
                    drop[L, hd, t] = base - acc()
        ABL.update(layer=-1, head=-1, tick=-1)
        prof = drop.reshape(-1, M).T                                           # (ticks, heads)
        nz = prof.norm(dim=1) > 1e-3
        P = torch.nn.functional.normalize(prof[nz], dim=1)
        C = P @ P.T
        iu = torch.triu_indices(len(P), len(P), 1)
        mean_cos = float(C[iu[0], iu[1]].mean()) if len(P) > 1 else float("nan")
        out[path.stem] = {"task": task, "acc": round(base, 4), "mean_tick_cosine": round(mean_cos, 3),
                          "ticks_with_effect": int(nz.sum()),
                          "drop": [[[round(float(drop[L, hd, t]), 3) for t in range(M)] for hd in range(a["heads"])]
                                   for L in range(a["layers"])]}
        print(f"{path.stem:24s} {task:6s} acc {base:.3f}  mean cosine between tick profiles {mean_cos:.3f} "
              f"({int(nz.sum())} ticks with effect)", flush=True)
        for L in range(a["layers"]):
            for hd in range(a["heads"]):
                print(f"    L{L} h{hd}: " + " ".join(f"{float(drop[L, hd, t]):+.2f}" for t in range(M)))
    res = Path(__file__).resolve().parent / "results" / "closed_loop_toy" / "tick_ablation.json"
    prev = json.loads(res.read_text()) if res.exists() else {}
    prev.update(out)
    res.write_text(json.dumps(prev, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
