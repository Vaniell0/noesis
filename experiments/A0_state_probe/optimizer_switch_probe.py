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
from experiments._common.convergence import is_converged
from experiments._common.runtime import limit_threads, progress


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
    ap.add_argument("--threads", type=int, default=None,
                    help="cap torch intra-op threads (default 4, or "
                         "$NOESIS_PROBE_THREADS). Set this when running "
                         "several probes at once: torch otherwise sizes "
                         "its pool from the core count and concurrent "
                         "probes oversubscribe the machine.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    n_threads = limit_threads(args.threads)
    progress(f"[{Path(__file__).stem}] torch threads = {n_threads}")

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
            progress(f"[{cond:24s} seed={seed}] id_r2={r['id_r2']:+.4f} "
                  f"ood={r['ood_r2']:+.4f} abl={r['ablation_a_gate_0_r2']:+.4f} "
                  f"loss {r['loss_before_switch']:.4f}->{r['loss_after_switch']:.4f} "
                  f"(final {r['loss_final']:.4f})")

    print("\n=== summary over seeds (CONVERGED runs only) ===")
    summary = {}
    for cond, _ in conditions:
        rs = [r for r in runs if r["condition"] == cond]
        ok = [r for r in rs if is_converged(r)]
        line = {"converged_frac": len(ok) / len(rs), "n_converged": len(ok),
                "n_seeds": len(rs)}
        for k in ("id_r2", "ood_r2", "ablation_a_gate_0_r2",
                  "loss_after_switch", "loss_final"):
            vals = [r[k] for r in ok if r[k] == r[k]]
            line[k] = {"mean": (sum(vals) / len(vals)) if vals else float("nan"),
                       "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0}
        summary[cond] = line
        if not ok:
            progress(f"  {cond:24s} 0/{len(rs)} converged — arm DESTROYED the model "
                     f"(final loss {sum(r['loss_final'] for r in rs) / len(rs):.3f}); "
                     f"it cannot serve as a control")
            continue
        progress(f"  {cond:24s} id_r2={line['id_r2']['mean']:+.4f}±{line['id_r2']['std']:.4f}  "
                 f"abl={line['ablation_a_gate_0_r2']['mean']:+.4f}±{line['ablation_a_gate_0_r2']['std']:.4f}  "
                 f"final_loss={line['loss_final']['mean']:.4f}  "
                 f"conv={len(ok)}/{len(rs)}")

    # Pre-registered read, computed not eyeballed — and REFUSED wherever the
    # arm it would be computed from did not converge.
    #
    # The first full run of this file made exactly that mistake: the lrx10 and
    # lrx30 control arms destroyed the model on all 5 seeds (final loss ~8.94,
    # i.e. predicting the mean), which left their a_gate ablation at ~-0.002 —
    # a number that LOOKS like "barely depends on the erase-rewrite channel"
    # but only means there is no solution left to ablate. Fed into the
    # mechanism-shift formula those arms produced a confident "best
    # step-size-only arm reaches 153%". H26 pre-registered this degeneracy in
    # its confound criterion; the probe now enforces it instead of restating
    # it. A control that destroyed the model is a FAILED control — it bounds
    # nothing — and is reported as such rather than scored.
    def conv(cond: str) -> bool:
        return summary.get(cond, {}).get("n_converged", 0) > 0

    refs_ok = conv("adam") and conv("muon") and conv("adam_then_muon")
    lr_conds = [c for c, _ in conditions if c.startswith("adam_then_adam_lrx")]
    lr_live = [c for c in lr_conds if conv(c)]
    lr_dead = [c for c in lr_conds if not conv(c)]

    base = summary["adam"]["id_r2"]["mean"]
    switch = summary["adam_then_muon"]["id_r2"]["mean"]
    lr_arms = {c: summary[c]["id_r2"]["mean"] for c in lr_live}
    worst_lr_arm = min(lr_arms.values()) if lr_arms else float("nan")
    switch_damage = base - switch
    lr_damage = base - worst_lr_arm

    abl_adam = summary["adam"]["ablation_a_gate_0_r2"]["mean"]
    abl_muon = summary["muon"]["ablation_a_gate_0_r2"]["mean"]
    abl_switch = summary["adam_then_muon"]["ablation_a_gate_0_r2"]["mean"]
    span = abl_muon - abl_adam
    shift_frac = (((abl_switch - abl_adam) / span)
                  if (refs_ok and abs(span) > 1e-9) else float("nan"))
    lr_shift_fracs = {
        c: (((summary[c]["ablation_a_gate_0_r2"]["mean"] - abl_adam) / span)
            if (refs_ok and abs(span) > 1e-9) else float("nan"))
        for c in lr_live
    }
    # The control has to reproduce the switch's movement TOWARD Muon. An arm
    # that moves the mechanism the other way has not "partly reproduced" it.
    best_lr_shift = max((v for v in lr_shift_fracs.values() if v == v),
                        default=float("nan"))

    if not refs_ok:
        verdict = ("NO VERDICT — one of the reference arms (adam / muon / "
                   "adam_then_muon) has no converged seed, so the mechanism "
                   "span it would be measured against does not exist")
    elif not lr_live:
        verdict = (f"switch moves the mechanism {shift_frac:.0%} toward Muon's "
                   f"signature, but EVERY step-size control destroyed the model "
                   f"({', '.join(lr_dead)}) — the confound is UNTESTED, not "
                   f"excluded. Rerun with smaller multipliers.")
    else:
        verdict = (
            f"accuracy: switch damage {switch_damage:+.4f} vs worst surviving "
            f"LR-bump {lr_damage:+.4f}. mechanism: ablation adam {abl_adam:+.3f} / "
            f"muon {abl_muon:+.3f} / switch {abl_switch:+.3f} = {shift_frac:.0%} "
            f"of the way to Muon's signature; best SURVIVING step-size-only arm "
            f"reaches {best_lr_shift:.0%}"
            + (f" (controls that destroyed the model and were excluded: "
               f"{', '.join(lr_dead)})" if lr_dead else "")
            + ". -> "
            + ("switch moves the mechanism further than any surviving pure-Adam "
               "LR change does (P1 supported, step size does not explain it)"
               if shift_frac > best_lr_shift + 0.15 else
               "step size accounts for as much mechanism movement as the switch "
               "does, or neither moved it")
            + (f"  [!] only {len(lr_live)} of {len(lr_conds)} step-size controls "
               f"survived, so this bound rests on a thin sweep"
               if len(lr_live) < 2 else "")
        )
    progress(f"\n{verdict}")


    if args.out is not None:
        save_result(
            args.out,
            {"runs": runs, "summary": summary, "verdict": verdict,
             "switch_damage_id_r2": switch_damage, "worst_lr_bump_damage_id_r2": lr_damage,
             "mechanism_shift_fraction_switch": shift_frac,
             "mechanism_shift_fraction_lr_arms": lr_shift_fracs,
             "lr_controls_destroyed_model": lr_dead,
             "lr_controls_surviving": lr_live,
             "config": vars(args) | {"out": str(args.out)}},
            experiment="optimizer_switch_toy", hypothesis=["H26"],
            summary={
                "a_gate ablation, adam / muon (pure arms)": f"{abl_adam:+.3f} / {abl_muon:+.3f}",
                "a_gate ablation, adam→muon switch": f"{abl_switch:+.3f}",
                "mechanism shift toward Muon, switch": f"{shift_frac:.0%}",
                "mechanism shift, best SURVIVING step-size-only control": f"{best_lr_shift:.0%}",
                "step-size controls that destroyed the model": ", ".join(lr_dead) or "none",
                "id_r2 adam / switch": f"{base:+.4f} / {switch:+.4f}",
            },
            script=str(Path(__file__).relative_to(_REPO_ROOT)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
