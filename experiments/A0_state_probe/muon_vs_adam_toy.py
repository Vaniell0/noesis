#!/usr/bin/env python3
"""muon_vs_adam_toy.py — cheap, CPU-only check of Muon vs. Int8AdamW-style
Adam on the H25 toy before spending real engineering time hand-integrating
Muon into the actual G1i full-FT stack.

Context (docs/rl-track.md, "Adam vs. Muon" note, 2026-09-02): Muon's real
draw for Phase 1.5 full-FT is that it carries only a momentum buffer, no
second-moment state — a direct answer to the fixed-cost VRAM wall in
docs/rl-track.md §Known risks #9. The real blocker there is integration
cost: trainable weights in the RWKV-PEFT/FORGE stack live in the
`.z`/state-dict path `experiments/rl/loader.py::Int8AdamW` was hand-built
against, not as ordinary `nn.Parameter`s a stock Muon implementation
expects.

This script does NOT touch that integration. It answers a narrower,
cheaper question first: on an architecture from the SAME family (a small
controller driving `micro_wkv_step`'s exact RWKV-7 delta-rule recurrence,
`micro_wkv.py`), does Muon's orthogonalized-momentum update even train at
all, and does it reach comparable held-out generalization to the existing
Adam baseline (`train_task(..., freeze_final_readout=True)`, R²=0.9998,
hypotheses/H25.md)? If Muon can't cleanly solve the SAME multiplication
task this toy already validated Adam on, hand-integrating it into the real
2.9B stack isn't worth attempting yet.

Honest scope limit: this is a held-out-R² check only. The VRAM-savings
half of the original comparison (this note's other stated purpose) cannot
be measured here — the toy model is a few hundred KB regardless of
optimizer, nowhere near where second-moment state would matter, and this
machine has no GPU to profile against. That half still needs the real
model and a GPU session; this script only clears (or fails to clear) the
cheaper, GPU-free precondition.

Muon reference implementation (`SingleDeviceMuon`, `zeropower_via_
newtonschulz5`) copied near-verbatim from github.com/KellerJordan/Muon
(muon.py, MIT-style research code, no license header in the source file)
— re-derived from the published algorithm would risk a subtly wrong
Newton-Schulz coefficient set, worse than reusing the canonical one
directly. Only trimmed: the distributed (`Muon`/`MuonWithAuxAdam`)
variants, since this is single-process CPU.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from experiments.A0_state_probe.micro_wkv import FrozenFinalReadoutController, train_task
from experiments._common.results import save_result


# --------------------------------------------------------------------------- #
# Muon, copied from github.com/KellerJordan/Muon (muon.py), single-device
# variant only. Not modified beyond removing the dist.* distributed path.
# --------------------------------------------------------------------------- #

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta: float = 0.95,
                 ns_steps: int = 5, nesterov: bool = True) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


class SingleDeviceMuon(torch.optim.Optimizer):
    """Muon for hidden 2D weight matrices only — see class docstring in the
    upstream source for why 1D params/embeddings/output heads don't belong
    here. This project pairs it with plain AdamW for everything else
    (`MuonWithAuxAdam`'s split, done by hand below instead of importing
    that class, since this toy has few enough params to split explicitly
    and explicitly is easier to audit)."""

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.0,
                 momentum: float = 0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])
        return loss


# --------------------------------------------------------------------------- #
# Muon+AdamW hybrid training, mirroring micro_wkv.py::train_task's exact
# protocol (same seed, same sampling, same eval) so the ONLY variable is
# the optimizer.
# --------------------------------------------------------------------------- #

def _split_muon_adam_params(model: torch.nn.Module):
    """Muon's own usage guidance (upstream docstring): hidden weight
    matrices only. Everything else — the step embedding (input-side),
    the readout (output-side), biases, and FrozenFinalReadoutController's
    two extra 1D vectors — goes to AdamW, exactly the MuonWithAuxAdam
    split, done explicitly here."""
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if name.startswith("net.") and name.endswith(".weight") and p.ndim == 2:
            muon_params.append(p)
        else:
            adam_params.append(p)
    return muon_params, adam_params


def train_task_muon(task: str, n_steps: int, head_size: int, n_train_steps: int = 4000,
                     batch_size: int = 64, muon_lr: float = 0.02, adam_lr: float = 3e-3,
                     seed: int = 0) -> dict:
    """Same protocol as micro_wkv.py::train_task(freeze_final_readout=True),
    optimizer swapped for Muon (hidden net.*.weight) + AdamW (everything
    else). muon_lr default (0.02) is Muon's own published default — this
    project has not tuned it; adam_lr matches train_task's own default."""
    torch.manual_seed(seed)
    target_fn = {"add": lambda a, b: a + b, "multiply": lambda a, b: a * b}[task]
    model = FrozenFinalReadoutController(head_size, n_steps)

    muon_params, adam_params = _split_muon_adam_params(model)
    muon_opt = SingleDeviceMuon(muon_params, lr=muon_lr)
    adam_opt = torch.optim.AdamW(adam_params, lr=adam_lr)

    def sample_batch(lo: float, hi: float, n: int):
        return torch.empty(n).uniform_(lo, hi), torch.empty(n).uniform_(lo, hi)

    t0 = time.perf_counter()
    for step in range(n_train_steps):
        a, b = sample_batch(-3.0, 3.0, batch_size)
        y_hat, _, _ = model(a, b)
        loss = F.mse_loss(y_hat, target_fn(a, b))
        muon_opt.zero_grad()
        adam_opt.zero_grad()
        loss.backward()
        muon_opt.step()
        adam_opt.step()
        if step % 1000 == 0 or step == n_train_steps - 1:
            print(f"  [muon:{task}] step {step}: train_mse={loss.item():.5f}")
    train_seconds = time.perf_counter() - t0

    with torch.no_grad():
        a_id, b_id = sample_batch(-3.0, 3.0, 1000)
        y_id, _, _ = model(a_id, b_id)
        id_r2 = 1.0 - F.mse_loss(y_id, target_fn(a_id, b_id)).item() / target_fn(a_id, b_id).var().item()

        sign_a = torch.randint(0, 2, (1000,)) * 2 - 1
        a_ood = sign_a.float() * torch.empty(1000).uniform_(3.0, 5.0)
        sign_b = torch.randint(0, 2, (1000,)) * 2 - 1
        b_ood = sign_b.float() * torch.empty(1000).uniform_(3.0, 5.0)
        y_ood, _, _ = model(a_ood, b_ood)
        ood_target = target_fn(a_ood, b_ood)
        ood_r2 = 1.0 - F.mse_loss(y_ood, ood_target).item() / ood_target.var().item()

        y_gate0, _, _ = model(a_id, b_id, force_a_gate=0.0)
        r2_gate0 = 1.0 - F.mse_loss(y_gate0, target_fn(a_id, b_id)).item() / target_fn(a_id, b_id).var().item()

    result = {
        "optimizer": "muon+adamw", "task": task, "n_steps": n_steps, "head_size": head_size,
        "muon_lr": muon_lr, "id_r2": id_r2, "ood_r2": ood_r2, "ablation_a_gate_0_r2": r2_gate0,
        "train_seconds": train_seconds, "n_muon_params": len(muon_params),
        "n_adam_params": len(adam_params),
    }
    print(f"  [muon:{task}] in-distribution R²={id_r2:.4f}  held-out(OOD) R²={ood_r2:.4f}  "
          f"a_gate=0 ablation R²={r2_gate0:.4f}  train_time={train_seconds:.1f}s")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["add", "multiply"], default="multiply")
    ap.add_argument("--steps", type=int, default=4, help="Recurrence steps (T), matches train_task's --steps default.")
    ap.add_argument("--head-size", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--lr-sweep", type=str, default=None,
                     help="Comma-separated muon_lr values, e.g. '0.005,0.01,0.02,0.04,0.08'. "
                          "Runs the Muon side only (Adam baseline computed once) at each, "
                          "skips the single-run --muon-lr path.")
    ap.add_argument("--n-seeds", type=int, default=None,
                     help="Run Adam and Muon (at --muon-lr) across seeds 0..n-1, report "
                          "mean/std per metric instead of a single seed's point estimate. "
                          "Added after a single-seed comparison (seed=0) turned out to be "
                          "misleadingly favorable to Adam on ood_r2 - it happened to be one "
                          "of Adam's better seeds. Skips --lr-sweep and the single-run path.")
    ap.add_argument("--out", type=Path, default=None,
                     help="If set, write results as JSON via save_result.")
    args = ap.parse_args()

    if args.n_seeds is not None:
        adam_runs, muon_runs = [], []
        for seed in range(args.n_seeds):
            a = train_task(args.task, args.steps, args.head_size,
                            n_train_steps=args.train_steps, seed=seed,
                            freeze_final_readout=True)
            a["optimizer"] = "adam"
            adam_runs.append(a)
            m = train_task_muon(args.task, args.steps, args.head_size,
                                 n_train_steps=args.train_steps, seed=seed,
                                 muon_lr=args.muon_lr)
            muon_runs.append(m)
            print(f"seed={seed}: adam(id_r2={a['id_r2']:.4f} ood_r2={a['ood_r2']:.4f} "
                  f"ablation={a['ablation_a_gate_0_r2']:+.4f})  "
                  f"muon(id_r2={m['id_r2']:.4f} ood_r2={m['ood_r2']:.4f} "
                  f"ablation={m['ablation_a_gate_0_r2']:+.4f})")

        def _stats(runs, key):
            vals = [r[key] for r in runs]
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            return mean, var ** 0.5, min(vals), max(vals)

        print(f"\n=== {args.n_seeds}-seed summary (muon_lr={args.muon_lr}) ===")
        for name, runs in (("adam", adam_runs), ("muon", muon_runs)):
            for key in ("id_r2", "ood_r2", "ablation_a_gate_0_r2"):
                mean, std, lo, hi = _stats(runs, key)
                print(f"  {name:5s} {key:24s} mean={mean:+.4f} std={std:.4f} "
                      f"range=[{lo:+.4f}, {hi:+.4f}]")

        if args.out is not None:
            save_result(
                args.out, {"adam": adam_runs, "muon": muon_runs,
                           "muon_lr": args.muon_lr, "n_seeds": args.n_seeds},
                experiment="muon_vs_adam_toy_multiseed", hypothesis=["H25"],
                summary={"purpose": "is the Adam-vs-Muon quality/mechanism difference "
                                     "real across seeds, or an artifact of seed=0? Also "
                                     "checks whether the a_gate=0 ablation effect (H25's "
                                     "'delta-rule necessary' claim) is optimizer-dependent"},
                script=str(Path(__file__).relative_to(_REPO_ROOT)),
            )
        return 0

    if args.lr_sweep is not None:
        print(f"=== baseline: Adam, freeze_final_readout=True (existing micro_wkv.py protocol) ===")
        t0 = time.perf_counter()
        adam_result = train_task(args.task, args.steps, args.head_size,
                                  n_train_steps=args.train_steps, seed=args.seed,
                                  freeze_final_readout=True)
        adam_result["train_seconds"] = time.perf_counter() - t0
        adam_result["optimizer"] = "adam"

        sweep = []
        for lr in [float(x) for x in args.lr_sweep.split(",")]:
            print(f"\n=== Muon lr={lr} ===")
            r = train_task_muon(args.task, args.steps, args.head_size,
                                 n_train_steps=args.train_steps, seed=args.seed, muon_lr=lr)
            sweep.append(r)

        print(f"\n=== sweep summary ===")
        print(f"  Adam (reference): id_r2={adam_result['id_r2']:.4f} ood_r2={adam_result['ood_r2']:.4f} "
              f"ablation={adam_result['ablation_a_gate_0_r2']:.4f}")
        for r in sweep:
            print(f"  Muon lr={r['muon_lr']:<6}: id_r2={r['id_r2']:.4f} ood_r2={r['ood_r2']:.4f} "
                  f"ablation={r['ablation_a_gate_0_r2']:.4f}")

        if args.out is not None:
            save_result(
                args.out, {"adam": adam_result, "muon_sweep": sweep},
                experiment="muon_vs_adam_toy_lr_sweep", hypothesis=["H25"],
                summary={"purpose": "does tuning muon_lr close the gap to Adam seen at "
                                     "Muon's generic default (0.02)? single seed per point, "
                                     "not a statistically robust sweep"},
                script=str(Path(__file__).relative_to(_REPO_ROOT)),
            )
        return 0

    print(f"=== baseline: Adam, freeze_final_readout=True (existing micro_wkv.py protocol) ===")
    t0 = time.perf_counter()
    adam_result = train_task(args.task, args.steps, args.head_size,
                              n_train_steps=args.train_steps, seed=args.seed,
                              freeze_final_readout=True)
    adam_result["train_seconds"] = time.perf_counter() - t0
    adam_result["optimizer"] = "adam"

    print(f"\n=== Muon (hidden net.*.weight) + AdamW (everything else) ===")
    muon_result = train_task_muon(args.task, args.steps, args.head_size,
                                   n_train_steps=args.train_steps, seed=args.seed,
                                   muon_lr=args.muon_lr)

    print(f"\n=== summary ===")
    print(f"  Adam : id_r2={adam_result['id_r2']:.4f} ood_r2={adam_result['ood_r2']:.4f} "
          f"ablation(a_gate=0)={adam_result['ablation_a_gate_0_r2']:.4f} "
          f"time={adam_result['train_seconds']:.1f}s")
    print(f"  Muon : id_r2={muon_result['id_r2']:.4f} ood_r2={muon_result['ood_r2']:.4f} "
          f"ablation(a_gate=0)={muon_result['ablation_a_gate_0_r2']:.4f} "
          f"time={muon_result['train_seconds']:.1f}s")
    print("  NOTE: no VRAM/memory comparison here — CPU-only toy, see module "
          "docstring. This only checks whether Muon trains this architecture "
          "family at all, not whether it saves memory on the real model.")

    if args.out is not None:
        save_result(
            args.out,
            {"adam": adam_result, "muon": muon_result},
            experiment="muon_vs_adam_toy",
            hypothesis=["H25"],
            summary={
                "purpose": "cheap precondition check before hand-integrating Muon "
                           "into the real G1i full-FT stack (docs/rl-track.md "
                           "'Adam vs. Muon' note, 2026-09-02) — held-out R² only, "
                           "no VRAM comparison possible without a GPU",
            },
            script=str(Path(__file__).relative_to(_REPO_ROOT)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
