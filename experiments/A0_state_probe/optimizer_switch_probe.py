#!/usr/bin/env python3
"""optimizer_switch_probe.py — does switching optimizer mid-training destroy a
working solution, and is it the geometry or just the step size?

This is H26's P2 prediction plus its own pre-registered confound control
(hypotheses/H26.md, "Falsification / strong version, confound"):

    Refuted if a mid-training optimizer switch produces no discontinuity in
    mechanism signature beyond what an LR-matched change under the SAME
    optimizer produces -- i.e. the effect is step size, not geometry.

Why this exists: the 2026-09-10 session established that Muon fine-tuning
collapses real generation quality on G1i (LoRA and full-FT, two learning
rates, with and without L_state, and on a clean-lineage checkpoint), while
every training diagnostic stayed healthy. That evidence is un-recheckable
without the VM -- the collapsed checkpoints live there, and marker-driven eval
needs the GPU-only `peft` backend. So the claim is re-tested here in the one
setting that IS repeatable for free: the same toy substrate H25/H26 already
use, where a switch can be run dozens of times and the mechanism signature
(the a_gate=0 ablation) is directly measurable.

Conditions, all on the identical task/seed/protocol:

  adam            pure Adam for the full budget -- the working baseline
  muon            pure Muon for the full budget
  adam_then_muon  Adam to the halfway point, then Muon -- the switch under test
  adam_then_adam_lrXn  Adam throughout, but LR multiplied by n at the halfway
                  point -- the confound control, run at several multipliers
                  rather than one "matched" value, because Adam's and Muon's
                  learning rates are not on a common scale and picking a single
                  equivalence would be the assumption under test. If ANY
                  Adam-LR bump reproduces the switch's damage, the effect is
                  step size; if none does, it is geometry.

Reported per condition: held-out R² (did it still solve the task), the
a_gate=0 ablation (which mechanism the solution relies on), and the loss
immediately before/after the switch point (does the switch visibly disturb
training at all, or does the damage appear only at eval -- the 2.9B signature).
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from experiments.A0_state_probe.micro_wkv import FrozenFinalReadoutController
from experiments.A0_state_probe.muon_vs_adam_toy import (
    SingleDeviceMuon,
    _split_muon_adam_params,
)
from experiments._common.results import save_result


def _eval(model, target_fn, n: int = 1000) -> dict:
    """Held-out R², OOD R², and the a_gate=0 ablation — the same three numbers
    muon_vs_adam_toy.py reports, so conditions here are comparable with the
    existing 10-seed baseline rather than a separate universe."""
    with torch.no_grad():
        a = torch.empty(n).uniform_(-3.0, 3.0)
        b = torch.empty(n).uniform_(-3.0, 3.0)
        y = target_fn(a, b)
        id_r2 = 1.0 - F.mse_loss(model(a, b)[0], y).item() / y.var().item()

        sa = (torch.randint(0, 2, (n,)) * 2 - 1).float()
        sb = (torch.randint(0, 2, (n,)) * 2 - 1).float()
        ao = sa * torch.empty(n).uniform_(3.0, 5.0)
        bo = sb * torch.empty(n).uniform_(3.0, 5.0)
        yo = target_fn(ao, bo)
        ood_r2 = 1.0 - F.mse_loss(model(ao, bo)[0], yo).item() / yo.var().item()

        yg = model(a, b, force_a_gate=0.0)[0]
        abl = 1.0 - F.mse_loss(yg, y).item() / y.var().item()
    return {"id_r2": id_r2, "ood_r2": ood_r2, "ablation_a_gate_0_r2": abl}


def run_condition(condition: str, seed: int, *, n_steps: int, head_size: int,
                  n_train_steps: int, batch_size: int, muon_lr: float,
                  adam_lr: float, lr_mult: float = 1.0) -> dict:
    torch.manual_seed(seed)
    model = FrozenFinalReadoutController(head_size, n_steps)
    target_fn = lambda x, y: x * y  # noqa: E731
    muon_params, adam_params = _split_muon_adam_params(model)
    switch_at = n_train_steps // 2

    def make_main(kind: str, lr_scale: float = 1.0):
        if kind == "muon":
            return SingleDeviceMuon(muon_params, lr=muon_lr)
        return torch.optim.AdamW(muon_params, lr=adam_lr * lr_scale)

    starts_with = "muon" if condition == "muon" else "adam"
    opt_main = make_main(starts_with)
    opt_aux = torch.optim.AdamW(adam_params, lr=adam_lr)

    losses: list[float] = []
    switched = False
    for step in range(n_train_steps):
        if step == switch_at and not switched:
            if condition == "adam_then_muon":
                opt_main = make_main("muon")
                switched = True
            elif condition.startswith("adam_then_adam_lr"):
                opt_main = make_main("adam", lr_scale=lr_mult)
                switched = True

        a = torch.empty(batch_size).uniform_(-3.0, 3.0)
        b = torch.empty(batch_size).uniform_(-3.0, 3.0)
        loss = F.mse_loss(model(a, b)[0], target_fn(a, b))
        opt_main.zero_grad()
        opt_aux.zero_grad()
        loss.backward()
        opt_main.step()
        opt_aux.step()
        losses.append(loss.item())

    def window(lo: int, hi: int) -> float:
        w = [x for x in losses[lo:hi] if x == x]
        return sum(w) / len(w) if w else float("nan")

    out = {
        "condition": condition, "seed": seed, "lr_mult": lr_mult,
        "switch_at": switch_at,
        "loss_before_switch": window(switch_at - 200, switch_at),
        "loss_after_switch": window(switch_at, switch_at + 200),
        "loss_final": window(n_train_steps - 200, n_train_steps),
    }
    out.update(_eval(model, target_fn))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--head-size", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--lr-mults", default="3,10,30",
                     help="Adam-LR multipliers for the step-size control arm.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    conditions = [("adam", 1.0), ("muon", 1.0), ("adam_then_muon", 1.0)]
    for m in (float(x) for x in args.lr_mults.split(",")):
        conditions.append((f"adam_then_adam_lrx{m:g}", m))

    runs = []
    for cond, mult in conditions:
        for seed in range(args.seeds):
            r = run_condition(cond, seed, n_steps=args.steps, head_size=args.head_size,
                              n_train_steps=args.train_steps, batch_size=args.batch_size,
                              muon_lr=args.muon_lr, adam_lr=args.adam_lr, lr_mult=mult)
            runs.append(r)
            print(f"[{cond:24s} seed={seed}] id_r2={r['id_r2']:+.4f} "
                  f"ood={r['ood_r2']:+.4f} abl={r['ablation_a_gate_0_r2']:+.4f} "
                  f"loss {r['loss_before_switch']:.4f}->{r['loss_after_switch']:.4f} "
                  f"(final {r['loss_final']:.4f})")

    print("\n=== summary over seeds ===")
    summary = {}
    for cond, _ in conditions:
        rs = [r for r in runs if r["condition"] == cond]
        line = {}
        for k in ("id_r2", "ood_r2", "ablation_a_gate_0_r2",
                  "loss_after_switch", "loss_final"):
            vals = [r[k] for r in rs if r[k] == r[k]]
            line[k] = {"mean": sum(vals) / len(vals),
                       "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0}
        summary[cond] = line
        print(f"  {cond:24s} id_r2={line['id_r2']['mean']:+.4f}±{line['id_r2']['std']:.4f}  "
              f"abl={line['ablation_a_gate_0_r2']['mean']:+.4f}±{line['ablation_a_gate_0_r2']['std']:.4f}  "
              f"final_loss={line['loss_final']['mean']:.4f}")

    # Pre-registered read, computed not eyeballed: did the switch damage the
    # solution, and did any pure-Adam LR bump reproduce that damage?
    base = summary["adam"]["id_r2"]["mean"]
    switch = summary["adam_then_muon"]["id_r2"]["mean"]
    lr_arms = {c: s["id_r2"]["mean"] for c, s in summary.items()
               if c.startswith("adam_then_adam_lrx")}
    worst_lr_arm = min(lr_arms.values()) if lr_arms else float("nan")
    switch_damage = base - switch
    lr_damage = base - worst_lr_arm

    # H26's P1 is about the MECHANISM SIGNATURE moving, not about accuracy
    # damage — the first smoke run of this file showed exactly that shape
    # (accuracy essentially intact, ablation sitting between the two pure
    # arms), which an accuracy-only verdict would have reported as "nothing
    # happened". Both axes are computed and reported; neither is dropped.
    abl_adam = summary["adam"]["ablation_a_gate_0_r2"]["mean"]
    abl_muon = summary["muon"]["ablation_a_gate_0_r2"]["mean"]
    abl_switch = summary["adam_then_muon"]["ablation_a_gate_0_r2"]["mean"]
    span = abl_muon - abl_adam
    # 0.0 = kept Adam's mechanism, 1.0 = fully adopted Muon's.
    shift_frac = ((abl_switch - abl_adam) / span) if abs(span) > 1e-9 else float("nan")
    lr_shift_fracs = {
        c: ((s["ablation_a_gate_0_r2"]["mean"] - abl_adam) / span) if abs(span) > 1e-9 else float("nan")
        for c, s in summary.items() if c.startswith("adam_then_adam_lrx")
    }
    worst_lr_shift = max((v for v in lr_shift_fracs.values() if v == v), default=float("nan"))

    verdict = (
        f"accuracy: switch damage {switch_damage:+.4f} vs worst LR-bump {lr_damage:+.4f}. "
        f"mechanism: ablation adam {abl_adam:+.3f} / muon {abl_muon:+.3f} / "
        f"switch {abl_switch:+.3f} = {shift_frac:.0%} of the way to Muon's signature; "
        f"best step-size-only arm reaches {worst_lr_shift:.0%}. -> "
        + ("switch moves the mechanism further than any pure-Adam LR change does "
           "(P1 supported, step size does not explain it)"
           if (shift_frac == shift_frac and worst_lr_shift == worst_lr_shift
               and shift_frac > worst_lr_shift + 0.15)
           else "step size accounts for as much mechanism movement as the switch "
                "does, or neither moved it")
    )
    print(f"\n{verdict}")

    if args.out is not None:
        save_result(
            args.out,
            {"runs": runs, "summary": summary, "verdict": verdict,
             "switch_damage_id_r2": switch_damage, "worst_lr_bump_damage_id_r2": lr_damage,
             "mechanism_shift_fraction_switch": shift_frac,
             "mechanism_shift_fraction_lr_arms": lr_shift_fracs,
             "config": vars(args) | {"out": str(args.out)}},
            experiment="optimizer_switch_toy", hypothesis=["H26"],
            summary={
                "a_gate ablation, adam / muon (pure arms)": f"{abl_adam:+.3f} / {abl_muon:+.3f}",
                "a_gate ablation, adam→muon switch": f"{abl_switch:+.3f}",
                "mechanism shift toward Muon, switch": f"{shift_frac:.0%}",
                "mechanism shift, best step-size-only control": f"{worst_lr_shift:.0%}",
                "id_r2 adam / switch": f"{base:+.4f} / {switch:+.4f}",
            },
            script=str(Path(__file__).relative_to(_REPO_ROOT)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
