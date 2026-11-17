"""Bits per parameter a trained closed-loop toy needs: quantize its weights and read accuracy by hop count.

Every weight matrix is rounded to b bits per parameter, symmetric uniform with a per-row absmax scale (vectors
and LayerNorm parameters per tensor); accuracy by k is read at the run's own tick count. The smallest b that keeps
accuracy within --tol of full precision at every k the model solves is the procedure's bits per parameter.
Compares programs: a loop that applies one step per tick should survive coarser weights than an unrolled one.

    python experiments/A0_state_probe/quant_sweep.py experiments/A0_state_probe/results/closed_loop_toy/s1abl_C_M8_seed1.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.A0_state_probe.closed_loop_toy import KMAX, ToyRWKV7, evaluate  # noqa: E402


def quantize_(model: torch.nn.Module, bits: int) -> None:
    levels = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for p in model.parameters():
            w = p.view(p.shape[0], -1) if p.dim() >= 2 else p.view(1, -1)
            scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / levels
            w.copy_((w / scale).round().clamp(-levels, levels) * scale)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("models", nargs="+", help="<run>.pt files saved by closed_loop_toy.py (args read from <run>.json)")
    ap.add_argument("--bits", default="16,8,6,5,4,3,2")
    ap.add_argument("--per-k", type=int, default=1000)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--solved", type=float, default=0.5, help="a k counts as solved at full precision above this")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rows = []
    for mp in args.models:
        run = json.loads(Path(mp).with_suffix(".json").read_text())["args"]
        state = torch.load(mp, map_location="cpu")
        acc = {}
        for b in [32] + [int(x) for x in args.bits.split(",")]:
            model = ToyRWKV7(run["width"], run["heads"], run["layers"])
            model.load_state_dict(state)
            if b < 32:
                quantize_(model, b)
            acc[b] = evaluate(model, run["n"], run["ticks"], args.per_k, run["seed"] + 2)[f"M{run['ticks']}"]
        solved = [k for k in range(1, KMAX + 1) if acc[32][k] > args.solved]
        ok = [b for b in acc if b < 32 and solved and all(acc[32][k] - acc[b][k] <= args.tol for k in solved)]
        rows.append({"model": Path(mp).name, "solved_k": solved, "min_bits": min(ok) if ok else None,
                     "acc": {str(b): acc[b] for b in acc}})
        print(json.dumps({k: rows[-1][k] for k in ("model", "solved_k", "min_bits")}), flush=True)
    from experiments._common.results import save_result
    save_result(args.out, {"args": vars(args), "rows": rows}, experiment="quant_sweep", hypothesis=["H25", "HZLK"],
                summary={r["model"]: f"min bits {r['min_bits']} over k {r['solved_k']}" for r in rows},
                script="experiments/A0_state_probe/quant_sweep.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
