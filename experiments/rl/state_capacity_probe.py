#!/usr/bin/env python3
"""state_capacity_probe.py — how many bindings can be read back at once.

Everything this project has measured about the WKV state counts how many
directions are OCCUPIED. That is not the same question as how many are
USABLE, and the difference is the whole point: occupancy says energy is
spread over k directions, it says nothing about whether distinct content can
be retrieved from distinct directions.

This measures the second. Feed N independent key→value bindings, read the
state once at the end, and fit one linear probe per binding. The curve of
"how many of the N are still readable above the permutation floor" against N
is the usable capacity.

Why this shape rather than a reasoning task: `carry_separability_probe.py`
asked a similar question on column addition and the answer was confounded by
the model simply not being able to do the arithmetic. Here nothing has to be
computed — only held and handed back — so a failure can only mean the state
ran out of room or the readout cannot separate. The task is deliberately
trivial; the capacity is the measurement.

Ceilings to compare against, both structural:
  head_size          64 per head — the rank ceiling of one head's state
  min(T, head_size)  the reachable rank after T tokens

And the open claim it addresses, `hypotheses/H25.md:330-333`: the recurrence
supplies a linear/bilinear substrate of 32 layers x 40 heads of independent
operators, "whether training actually organizes it that way is untested, not
yet claimed".

Runs on the peft backend on CPU — inference only, no gradient step anywhere.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from experiments.rl.loader import load_rwkv7
from experiments.rl.wkv_linear_probe import held_out_linear_probe

KEYS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
        "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
        "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform",
        "victor", "whiskey", "xray", "yankee", "zulu", "anchor", "beacon",
        "cipher", "domino", "ember", "falcon"]


def make_sample(rng: random.Random, n: int, vmax: int) -> tuple[str, list[int]]:
    """One record. Key ORDER is shuffled per sample, deliberately.

    Two reasons, and the first is a measurement bug caught on the first run.
    With a fixed template and only the values changing, the states come out
    nearly identical across samples: the training design measured effective
    rank **4.9**, so a probe over 67 rows could fit at most ~5 independent
    directions and a null at N=32 would have said something about the design,
    not about the state. `held_out_linear_probe`'s own docstring names that
    number as exactly what decides between the two readings.

    Shuffling the order raises the design's rank, and it also sharpens the
    question: the target for slot i is the value bound to KEY i wherever that
    key happened to land, so a readout that only tracks position cannot score.
    """
    vals = [rng.randint(0, vmax) for _ in range(n)]
    order = list(range(n))
    rng.shuffle(order)
    body = " ".join(f"{KEYS[i]}={vals[i]}" for i in order)
    return f"Record the values.\n{body}\nEnd of record.", vals


@torch.no_grad()
def state_row(loaded, prompt: str, layers) -> torch.Tensor:
    state = loaded.new_state(batch=1)
    ids = loaded.tokenizer.encode(prompt)
    inp = torch.tensor([ids]) if loaded.backend == "peft" else ids
    _, state = loaded.forward_stateful(inp, state)
    wkv = loaded.wkv_stack(state)
    feats = []
    for L in layers:
        s = wkv[L].float()
        s = s[0] if s.dim() == 4 else s
        feats.append(s.flatten())
    return torch.cat(feats)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", required=True,
                     help="From THIS checkpoint's own measured profile.")
    ap.add_argument("--n-list", default="1,2,4,8,16,32")
    ap.add_argument("--samples", type=int, default=96)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--vmax", type=int, default=99,
                     help="Wider values give the design more variation, which "
                          "is what the probe's power depends on here.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    n_list = [int(x) for x in args.n_list.split(",")]
    print(f"[capacity] loading {args.model} ({args.device}, peft fp32)", flush=True)
    loaded = load_rwkv7(args.model, device=args.device, dtype=torch.float32,
                        backend="peft", ctx_len=2048)

    results = {}
    for n in n_list:
        if n > len(KEYS):
            continue
        rng = random.Random(args.seed + n)
        t0 = time.time()
        rows, targets = [], []
        for _ in range(args.samples):
            prompt, vals = make_sample(rng, n, args.vmax)
            rows.append(state_row(loaded, prompt, layers))
            targets.append(vals)
        X = torch.stack(rows)
        n_train = int(len(rows) * args.train_frac)
        slots = []
        for i in range(n):
            y = torch.tensor([float(t[i]) for t in targets])
            p = held_out_linear_probe(X, y, n_train)
            slots.append({"slot": i, "key": KEYS[i],
                          "r2": p["held_out_r2"],
                          "floor": p["shuffled_held_out_mean"],
                          "margin_sd": p["margin_sd"],
                          "above_max": p["above_shuffled_max"]})
        readable = sum(1 for s in slots if s["above_max"])
        results[n] = {"n": n, "readable": readable,
                      "fraction": readable / n,
                      "mean_r2": sum(s["r2"] for s in slots) / n,
                      "eff_rank": held_out_linear_probe(
                          X, torch.tensor([float(t[0]) for t in targets]),
                          n_train)["design_effective_rank"],
                      "slots": slots,
                      "seconds": round(time.time() - t0, 1)}
        print(f"[capacity] N={n:>2}: readable {readable}/{n} "
              f"(mean R2 {results[n]['mean_r2']:+.3f}, "
              f"design eff-rank {results[n]['eff_rank']:.1f}, "
              f"{results[n]['seconds']:.0f}s)", flush=True)

    print("\n=== usable capacity ===")
    print(f"{'N':>4}{'readable':>10}{'fraction':>10}{'mean R2':>10}")
    for n, r in results.items():
        print(f"{n:>4}{r['readable']:>10}{r['fraction']:>10.2f}{r['mean_r2']:>10.3f}")

    from experiments._common.results import save_result
    save_result(args.out, {"model": args.model, "layers": layers,
                           "samples": args.samples, "vmax": args.vmax,
                           "by_n": results,
                           "_summary": {
                               f"N={n}": f"{r['readable']}/{n} readable"
                               for n, r in results.items()}},
                experiment="state_capacity", hypothesis=["H25"],
                model=args.model, script="experiments/rl/state_capacity_probe.py")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
