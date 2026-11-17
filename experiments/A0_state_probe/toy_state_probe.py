"""What the closed-loop toy writes: decode every pair x -> next(x) from the WKV state after the context is read.

The state after the 2n pair tokens (all layers and heads, flattened) is the feature; for each node x one linear
softmax probe predicts next(x) over the node symbols, trained on examples whose cycle holds x, scored held out.
A delta-rule memory answers a fixed key linearly, so a linear probe per key reads what the state holds whether or
not the model's own reader uses it. An untrained model of the same shape is the floor (--random). Also the
effective rank of each WKV head (16 x 16, 10 pairs written) after the pairs, the prompt and 8 ticks (M = 0 runs
are read with 8 ticks too): does the state hold the pairs as separate directions, and do ticks collapse it.

Reading, written before the run: probe accuracy within 0.05 between M = 0 and the tick arms means the loop taught
reading, not writing; tick arms >= +0.10 over M = 0 means the loop also changed what the state holds.

    python experiments/A0_state_probe/toy_state_probe.py results/closed_loop_toy/s1abl_A_M0_seed0.pt ... --random
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.A0_state_probe.closed_loop_toy import N_NODES, ToyRWKV7, make_batch  # noqa: E402


@torch.no_grad()
def states(model, n: int, count: int, seed: int):
    """WKV state after the pairs, flattened (count, L*H*d*d), and next(x) per node (-1 when x is absent)."""
    b = make_batch(count, n, [1], torch.Generator().manual_seed(seed))
    toks = b["tokens"]
    st = model.init_state(count)
    e = model.emb(toks)
    for t in range(2 * n):
        _, st = model.step(e[:, t], st)
    feats = torch.cat([S.flatten(1) for S in st["S"]], dim=1)
    nxt = torch.full((count, N_NODES), -1, dtype=torch.long)
    rows = torch.arange(count)
    for j in range(n):
        nxt[rows, toks[:, 2 * j]] = toks[:, 2 * j + 1]
    return feats, nxt


def erank(S: torch.Tensor) -> torch.Tensor:
    """Effective rank exp(H(sigma / sum sigma)) of each (d, d) matrix in S (..., d, d)."""
    sv = torch.linalg.svdvals(S)
    p = sv / sv.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.exp(-(p * p.clamp_min(1e-12).log()).sum(-1))


@torch.no_grad()
def ranks(model, n: int, ticks: int, count: int, seed: int) -> dict:
    """Mean effective rank per layer of the WKV heads after the pairs, after the prompt, after the ticks."""
    b = make_batch(count, n, [1], torch.Generator().manual_seed(seed))
    st = model.init_state(count)
    e = model.emb(b["tokens"])
    out = {}
    for t in range(b["tokens"].shape[1]):
        h, st = model.step(e[:, t], st)
        if t == 2 * n - 1:
            out["after_pairs"] = [round(erank(S).mean().item(), 2) for S in st["S"]]
    out["after_prompt"] = [round(erank(S).mean().item(), 2) for S in st["S"]]
    for _ in range(ticks):
        h, st = model.step(h, st)
    out["after_ticks"] = [round(erank(S).mean().item(), 2) for S in st["S"]]
    return out


def probe(ftr, ytr, fte, yte, steps: int) -> float:
    mu, sd = ftr.mean(0), ftr.std(0).clamp_min(1e-6)
    ftr, fte = (ftr - mu) / sd, (fte - mu) / sd
    accs = []
    for x in range(N_NODES):
        mtr, mte = ytr[:, x] >= 0, yte[:, x] >= 0
        W = torch.zeros(ftr.shape[1], N_NODES, requires_grad=True)
        c = torch.zeros(N_NODES, requires_grad=True)
        opt = torch.optim.Adam([W, c], lr=1e-2, weight_decay=1e-4)
        for _ in range(steps):
            loss = F.cross_entropy(ftr[mtr] @ W + c, ytr[mtr, x])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            accs.append(((fte[mte] @ W + c).argmax(-1) == yte[mte, x]).float().mean().item())
    return sum(accs) / len(accs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="+", help="<run>.pt files saved by closed_loop_toy.py (args read from <run>.json)")
    ap.add_argument("--random", action="store_true", help="also probe an untrained model of the first run's shape")
    ap.add_argument("--train", type=int, default=20000)
    ap.add_argument("--test", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    jobs = [(Path(mp).name, json.loads(Path(mp).with_suffix(".json").read_text())["args"], mp) for mp in args.models]
    if args.random:
        jobs.append(("untrained", jobs[0][1], None))
    rows = []
    for name, run, mp in jobs:
        torch.manual_seed(run["seed"])
        model = ToyRWKV7(run["width"], run["heads"], run["layers"])
        if mp:
            model.load_state_dict(torch.load(mp, map_location="cpu"))
        ftr, ytr = states(model, run["n"], args.train, 50_000)
        fte, yte = states(model, run["n"], args.test, 60_000)
        rows.append({"model": name, "ticks": run["ticks"], "jepa": run["jepa"],
                     "probe_acc": round(probe(ftr, ytr, fte, yte, args.steps), 4),
                     "erank": ranks(model, run["n"], run["ticks"] or 8, 2000, 70_000)})
        print(json.dumps(rows[-1]), flush=True)
    from experiments._common.results import save_result
    save_result(args.out, {"args": vars(args), "rows": rows}, experiment="toy_state_probe", hypothesis=["H25", "HZLK"],
                summary={r["model"]: f"pair decode {r['probe_acc']:.3f}; erank {r['erank']}" for r in rows},
                script="experiments/A0_state_probe/toy_state_probe.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
