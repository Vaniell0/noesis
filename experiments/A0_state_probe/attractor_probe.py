#!/usr/bin/env python3
"""attractor_probe.py - what program does the trained loop run: the tick latent read tick by tick (exploration).

    python experiments/A0_state_probe/attractor_probe.py --models results/closed_loop_toy/s1_C_M8_seed{4,5,6}.pt \
        --ticks 8 --n 4096 --out results/closed_loop_toy/attractor_C_M8.json

On fresh chase examples (k = 1..8, the model's own M ticks), for every tick t:
  node_t   a linear probe (fit on one half, scored on the other) reads the walk's node f^t(s) from h_t; for t > k the
           walk has gone past the asked hop;
  answer_t the same probe family reads the answer f^k(s) from h_t; after tick k, answer_t > node_t means the
           program stopped at the answer instead of stepping on;
  move_t   |h_t - h_{t-1}| / |h_{t-1}|, split into ticks before and after k: falling after k is an attractor;
  stop     (--stop-tau) the own stop tick against the tick after which the motion falls most, by k, and the
           answer read from the latent at the stop;
  centroid the answer read from the last latent by its nearest answer centroid (centroids from the fit half): one
           point per answer shared across tables, against a separate point per table.
Reads only models trained with ticks (rule: no ticks on a model not trained for them).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from closed_loop_toy import KMAX, N_NODES, ToyRWKV7, make_batch  # noqa: E402


def ridge_acc(Xf, yf, Xs, ys, lam=1e-2):
    """One-vs-all ridge on standardised features; accuracy on the scored half."""
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    A, B = (Xf - mu) / sd, (Xs - mu) / sd
    A, B = torch.cat([A, torch.ones(len(A), 1)], 1), torch.cat([B, torch.ones(len(B), 1)], 1)
    Y = torch.nn.functional.one_hot(yf, N_NODES).float()
    W = torch.linalg.solve(A.T @ A + lam * len(A) * torch.eye(A.shape[1]), A.T @ Y)
    return ((B @ W).argmax(1) == ys).float().mean().item()


@torch.no_grad()
def probe(model, M, n, seed, stop_tau=0.0):
    g = torch.Generator().manual_seed(seed)
    b = make_batch(n, 10, list(range(1, KMAX + 1)), g)
    logits, lat, _ = model(b["tokens"], M)
    H = torch.stack(lat, 1)                                                     # (n, M, C)
    k, ans, path = b["k"], b["answer"], b["path"]
    acc = (logits.argmax(-1) == ans).float()
    half = torch.arange(n) % 2 == 0
    out = {"acc_by_k": {int(kk): round(acc[k == kk].mean().item(), 3) for kk in range(1, KMAX + 1)}}
    # node f^t(s) on the walk continued past k: extend path along the cycle by re-walking the table
    toks = b["tokens"]
    walk = torch.empty(n, M, dtype=torch.long)
    for i in range(n):
        nxt = {toks[i, 2 * j].item(): toks[i, 2 * j + 1].item() for j in range(10)}
        cur = toks[i, 21].item()
        for t in range(M):
            cur = nxt[cur]
            walk[i, t] = cur
    rows = []
    for t in range(M):
        X = H[:, t]
        node = ridge_acc(X[half], walk[half, t], X[~half], walk[~half, t])
        answ = ridge_acc(X[half], ans[half], X[~half], ans[~half])
        before, after = (k > t + 1), (k <= t)                                  # tick t+1 is past hop k when k <= t
        rows.append({"tick": t + 1, "node": round(node, 3), "answer": round(answ, 3),
                     "answer_if_past_k": round(ridge_acc(X[half & after], ans[half & after],
                                                         X[~half & after], ans[~half & after]), 3)
                     if after.sum() > 64 else None,
                     "node_if_past_k": round(ridge_acc(X[half & after], walk[half & after, t],
                                                       X[~half & after], walk[~half & after, t]), 3)
                     if after.sum() > 64 else None})
    mv = (H[:, 1:] - H[:, :-1]).norm(dim=-1) / H[:, :-1].norm(dim=-1)           # (n, M-1): move into tick t+2
    pre, post = [], []
    for t in range(M - 1):
        pre.append(mv[k > t + 1, t].mean().item() if (k > t + 1).any() else float("nan"))
        post.append(mv[k <= t + 1, t].mean().item() if (k <= t + 1).any() else float("nan"))
    last = H[:, -1]
    cent = torch.stack([last[half & (ans == a)].mean(0) if (half & (ans == a)).any() else torch.full_like(last[0], 1e9)
                        for a in range(N_NODES)])
    cen_acc = ((torch.cdist(last[~half], cent).argmin(1) == ans[~half]).float().mean().item())
    within = torch.stack([(last[ans == a] - last[ans == a].mean(0)).norm(dim=-1).mean() for a in ans.unique()]).mean()
    between = torch.pdist(torch.stack([last[ans == a].mean(0) for a in ans.unique()])).mean()
    if stop_tau:                                                                # own stop: first confident tick
        P = torch.stack([model.head(H[:, t]).softmax(-1).amax(-1) for t in range(M)], 1)   # (n, M)
        conf = P >= stop_tau
        tau = torch.where(conf.any(1), conf.float().argmax(1), torch.full((n,), M - 1))      # 0-based tick index
        drop = torch.cat([torch.zeros(n, 1), mv[:, :-1] - mv[:, 1:]], 1)        # fall of motion after tick t+1
        dtk = drop.argmax(1)                                                    # 0-based tick after which motion falls most
        Xs = H[torch.arange(n), tau]
        out["stop"] = {"ticks_used_by_k": {int(kk): round((tau[k == kk] + 1).float().mean().item(), 2)
                                           for kk in range(1, KMAX + 1)},
                       "drop_tick_eq_stop": round((dtk == tau).float().mean().item(), 3),
                       "drop_tick_within1": round(((dtk - tau).abs() <= 1).float().mean().item(), 3),
                       "drop_tick_by_k": {int(kk): round((dtk[k == kk] + 1).float().mean().item(), 2)
                                          for kk in range(1, KMAX + 1)},
                       "answer_read_at_stop": round(ridge_acc(Xs[half], ans[half], Xs[~half], ans[~half]), 3)}
    out.update({"per_tick": rows, "move_before_k": [round(x, 4) for x in pre],
                "move_after_k": [round(x, 4) for x in post], "centroid_acc_last": round(cen_acc, 3),
                "spread_within_over_between": round((within / between).item(), 3)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--ticks", type=int, default=8)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--stop-tau", type=float, default=0.0, help="also read the own stop (first tick with max p >= tau)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    res = {}
    for mp in a.models:
        model = ToyRWKV7()
        model.load_state_dict(torch.load(mp, map_location="cpu"))
        model.eval()
        r = probe(model, a.ticks, a.n, a.seed, a.stop_tau)
        res[Path(mp).stem] = r
        print(Path(mp).stem, json.dumps({x: r[x] for x in ("acc_by_k", "centroid_acc_last",
                                                         "spread_within_over_between")}), flush=True)
        if "stop" in r:
            print("   stop", json.dumps(r["stop"]), flush=True)
        for row in r["per_tick"]:
            print("  ", row, flush=True)
        print("   move before k", r["move_before_k"], "\n   move after k ", r["move_after_k"], flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
