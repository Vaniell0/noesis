#!/usr/bin/env python3
"""halting_chain.py — state-dependent stopping on the toy WKV controller (H25).

fleeb83's reported property (H25.md, "Fourth finding"): the system stops by what
its STATE holds, not by a counter. `micro_wkv.py` built the chain but not the
stopping; this is that test, on the same `micro_wkv_step` physics.

Task. One operand (uniform 0..1) is revealed per round and never shown again.
The controller must stop at the first round where the running sum reaches
THRESHOLD, and output that sum. The running sum exists only in the WKV state, so
the round at which to stop is a function of state content. A counter cannot
solve it: the stop round depends on the data (3..8 in the default setting).

Training is teacher-forced up to the stop: BCE on a halt head reading the
read-out `out = r @ S` at every round up to t*, MSE on the sum at t*. Test is
free-running: stop at the first round where the halt head exceeds 0.5.

Controls (every number is reported with its own, per the discriminability rule):
  memoryless  same net, state reset to zero every round — cannot integrate, so
              stop-round accuracy must fall to what the current operand allows.
  counter     stop round predicted from the round index alone (the best constant
              per round, computed from the data, no model) — the floor a
              counter-only policy gets.
  zero-state  trained model, state zeroed before round 2 — if stopping is read
              from state this must break it.
  swap-state  trained model, states permuted across examples before round 2. The
              stop must follow the DONOR's accumulated sum (the state), not the
              recipient's own prefix. Scored against both oracles on the
              examples where the two disagree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.A0_state_probe.micro_wkv import micro_wkv_step  # noqa: E402

THRESHOLD = 2.0


class HaltingController(nn.Module):
    def __init__(self, head_size: int, n_rounds: int, hidden: int = 64,
                 memoryless: bool = False):
        super().__init__()
        self.head_size, self.n_rounds, self.memoryless = head_size, n_rounds, memoryless
        self.round_embed = nn.Embedding(n_rounds, 8)
        self.net = nn.Sequential(
            nn.Linear(1 + 8, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 5 * head_size),
        )
        self.readout = nn.Linear(head_size, 1)
        self.halt = nn.Linear(head_size, 1)

    def controls(self, operand: torch.Tensor, t: int) -> dict:
        re = self.round_embed(torch.full((operand.shape[0],), t, dtype=torch.long))
        raw = self.net(torch.cat([operand.unsqueeze(-1), re], dim=-1))
        r, k, v, a_logit, w_raw = raw.split(self.head_size, dim=-1)
        return {"r": r, "k": k, "v": v, "w": -F.softplus(-w_raw) - 0.5,
                "a_gate": torch.sigmoid(a_logit)}

    def run(self, operands: list, intervene=None):
        """Returns per-round (halt_logit [B], y [B]). `intervene(t, state)` may
        replace the state before round t (zero / swap controls)."""
        B = operands[0].shape[0]
        state = torch.zeros(B, self.head_size, self.head_size)
        halts, ys = [], []
        for t in range(self.n_rounds):
            if self.memoryless:
                state = torch.zeros_like(state)
            if intervene is not None:
                state = intervene(t, state)
            c = self.controls(operands[t], t)
            out, state = micro_wkv_step(state, c["r"], c["k"], c["v"], c["w"], c["a_gate"])
            halts.append(self.halt(out).squeeze(-1))
            ys.append(self.readout(out).squeeze(-1))
        return torch.stack(halts, 1), torch.stack(ys, 1)


def sample(n: int, n_rounds: int, lo: float = 0.0, hi: float = 1.0):
    operands = [torch.empty(n).uniform_(lo, hi) for _ in range(n_rounds)]
    cs = torch.stack(operands, 1).cumsum(1)                      # [n, T]
    reached = cs >= THRESHOLD
    stop = torch.where(reached.any(1), reached.float().argmax(1),
                       torch.full((n,), n_rounds - 1))
    target = cs.gather(1, stop[:, None]).squeeze(1)
    return operands, stop, target


def first_halt(halt_logits: torch.Tensor) -> torch.Tensor:
    fired = halt_logits > 0
    last = torch.full((halt_logits.shape[0],), halt_logits.shape[1] - 1)
    return torch.where(fired.any(1), fired.float().argmax(1), last)


def train(model: HaltingController, steps: int, batch: int, lr: float, seed: int):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    T = model.n_rounds
    for step in range(steps):
        operands, stop, target = sample(batch, T)
        halts, ys = model.run(operands)
        upto = torch.arange(T)[None, :] <= stop[:, None]          # rounds 0..t*
        halt_lab = (torch.arange(T)[None, :] == stop[:, None]).float()
        bce = (F.binary_cross_entropy_with_logits(halts, halt_lab, reduction="none")
               * upto).sum() / upto.sum()
        y_at = ys.gather(1, stop[:, None]).squeeze(1)
        loss = bce + F.mse_loss(y_at, target)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0 or step == steps - 1:
            print(f"  [{'memoryless' if model.memoryless else 'state'}] step {step} "
                  f"bce {bce.item():.4f} loss {loss.item():.4f}", flush=True)


@torch.no_grad()
def evaluate(model: HaltingController, n: int, lo: float = 0.0, hi: float = 1.0,
             seed: int = 1234) -> dict:
    torch.manual_seed(seed)
    T = model.n_rounds
    operands, stop, target = sample(n, T, lo, hi)
    halts, ys = model.run(operands)
    pred = first_halt(halts)
    y_at = ys.gather(1, pred[:, None]).squeeze(1)
    r2 = 1 - F.mse_loss(y_at, target).item() / target.var().item()
    return {"stop_acc": (pred == stop).float().mean().item(),
            "within1": ((pred - stop).abs() <= 1).float().mean().item(),
            "y_r2_at_pred_stop": r2}


def counter_floor(n_rounds: int, n: int = 20000, seed: int = 99) -> float:
    """Best a counter-only policy can do: the most common stop round."""
    torch.manual_seed(seed)
    _, stop, _ = sample(n, n_rounds)
    return torch.bincount(stop, minlength=n_rounds).max().item() / n


@torch.no_grad()
def causal_controls(model: HaltingController, n: int = 4000, at: int = 2,
                    seed: int = 4321) -> dict:
    torch.manual_seed(seed)
    T = model.n_rounds
    operands, stop, _ = sample(n, T)
    base = first_halt(model.run(operands)[0])

    def zero(t, s):
        return torch.zeros_like(s) if t == at else s
    zeroed = first_halt(model.run(operands, intervene=zero)[0])

    perm = torch.randperm(n)

    def swap(t, s):
        return s[perm] if t == at else s
    swapped = first_halt(model.run(operands, intervene=swap)[0])

    # oracle after the swap: donor's prefix (rounds < at) + recipient's own rest
    ops = torch.stack(operands, 1)
    mixed = torch.cat([ops[perm][:, :at], ops[:, at:]], 1).cumsum(1)
    reached = mixed >= THRESHOLD
    swap_oracle = torch.where(reached.any(1), reached.float().argmax(1),
                              torch.full((n,), T - 1))
    differs = swap_oracle != stop
    return {
        "zero_state": {"stop_acc_vs_true": (zeroed == stop).float().mean().item(),
                       "unchanged_vs_unablated": (zeroed == base).float().mean().item()},
        "swap_state": {"n_disagreeing": int(differs.sum()),
                       "follows_donor_state": (swapped == swap_oracle)[differs].float().mean().item(),
                       "follows_own_prefix": (swapped == stop)[differs].float().mean().item()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--head-size", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("experiments/A0_state_probe/results/halting_chain.json"))
    args = ap.parse_args()
    torch.set_num_threads(1)   # runs beside the P1 training on the laptop

    res = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "counter_floor_stop_acc": counter_floor(args.rounds)}
    print(f"counter-only floor (most common stop round): {res['counter_floor_stop_acc']:.3f}")
    for name, memless in (("state", False), ("memoryless", True)):
        torch.manual_seed(args.seed)
        m = HaltingController(args.head_size, args.rounds, memoryless=memless)
        train(m, args.steps, args.batch, args.lr, args.seed)
        res[name] = {"in_dist": evaluate(m, 4000),
                     "ood_scale_0.5_1.5": evaluate(m, 4000, 0.5, 1.5)}
        if not memless:
            res[name]["causal"] = causal_controls(m)
        print(name, json.dumps(res[name], indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
