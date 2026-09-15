#!/usr/bin/env python3
"""step_geometry_probe.py — does ANY fixed weight-space norm predict how far
the WKV state trajectory actually moves when the optimizer takes a step?

This is H26's first measurement, and it tests the strong version's *premise*
(hypotheses/H26.md, "Falsification / strong version, premise"), pre-registered
before this file was written:

    Refuted if induced state-trajectory displacement turns out to be well
    predicted by a fixed weight-space norm -- |corr| >= 0.9 between induced
    ||dS|| and one of ||dW||_spec, ||dW||_F, ||dW||_inf across training. If
    some existing fixed norm already tracks state motion, there is nothing
    for a state-induced metric to fix.

The asymmetry this looks for: an optimizer normalises the step in ITS OWN
norm -- Muon makes the update's singular values ~1 by construction (Newton-
Schulz), Adam bounds it per-coordinate -- while the quantity that actually
matters for a recurrent model is how far the STATE trajectory moved, which
the recurrence compounds over T steps. Neither optimizer measures that, and
neither is told about it. So at every measurement point we record both
halves and look at whether they track each other.

Deliberately cheap: CPU, minutes, no GPU / VM / 2.9B checkpoint. It reuses
H25's existing substrate (`micro_wkv.py`'s FrozenFinalReadoutController,
whose `forward` already returns a full `state_trace`) and the already-
validated Muon implementation and param split from `muon_vs_adam_toy.py`
rather than re-deriving either.

What this does NOT test (stated so the result is not over-read): the
mid-training optimizer-switch prediction, the LR-matched control for it, and
the practical "does a state-normalised step rule help" rung. Those are
separate criteria in H26 and separate runs.
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
from experiments._common.convergence import CONVERGED_ID_R2 as _CONVERGED_ID_R2
from experiments._common.results import save_result
from experiments._common.runtime import limit_threads, progress


# Below this many measurement points a Pearson r is not interpretable: any
# 3-5 points are nearly collinear by chance, so |r| comes out ~1 with an
# arbitrary sign. Found by the first smoke run of this file (60 train steps,
# measure_every=20 -> 3 points), which printed a confident "premise REFUTED"
# off |r|=0.996 that was pure small-n artifact. Guarding it here rather than
# relying on the caller to pick sane flags, because that verdict is written
# into a hypothesis record.
MIN_POINTS_FOR_CORR = 30

# Shared with every other toy probe — see experiments/_common/convergence.py
# for why this is one definition in one place and not three local copies.
CONVERGED_ID_R2 = _CONVERGED_ID_R2


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Plain Pearson r. Written out rather than pulled from scipy/numpy —
    this package's probes are stdlib+torch only, and n here is ~200."""
    n = len(xs)
    if n < MIN_POINTS_FOR_CORR:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _cv(vals: list[float]) -> float:
    """Coefficient of variation (std/mean). The headline shape statistic:
    an optimizer whose own norm is well-matched to state motion would keep
    this low for the ratio, high CV means its notion of 'unit step' and the
    state's actual displacement are decoupled."""
    if not vals:
        return float("nan")
    m = sum(vals) / len(vals)
    if m == 0:
        return float("nan")
    return statistics.pstdev(vals) / abs(m)


@torch.no_grad()
def _state_trace(model: FrozenFinalReadoutController, a: torch.Tensor, b: torch.Tensor
                  ) -> list[torch.Tensor]:
    _, _, trace = model(a, b)
    return [s.clone() for s in trace]


@torch.no_grad()
def _trace_displacement(before: list[torch.Tensor], after: list[torch.Tensor]
                         ) -> tuple[float, float]:
    """(absolute, relative) displacement of the state trajectory.

    Absolute: mean over timesteps of ||S_t_after - S_t_before||_F.
    Relative: the same, divided by the mean ||S_t_before||_F — because the
    state's own scale drifts during training, an absolute displacement that
    grows could just be the state getting bigger, not the trajectory being
    disturbed more.
    """
    diffs, mags = [], []
    for s0, s1 in zip(before, after):
        diffs.append(torch.linalg.norm((s1 - s0).flatten(start_dim=1), dim=-1).mean().item())
        mags.append(torch.linalg.norm(s0.flatten(start_dim=1), dim=-1).mean().item())
    abs_d = sum(diffs) / len(diffs)
    mean_mag = sum(mags) / len(mags)
    return abs_d, (abs_d / mean_mag if mean_mag > 0 else float("nan"))


@torch.no_grad()
def _update_norms(before: list[torch.Tensor], after: list[torch.Tensor]) -> dict:
    """The step's size in each candidate FIXED weight-space norm.

    Per-matrix spectral norms are reported as both max and mean because there
    is no single canonical way to aggregate a matrix norm across a parameter
    list; Frobenius has an exact aggregate (the norm of the concatenation) so
    that one is summed properly rather than averaged.
    """
    specs, infs, fro_sq = [], [], 0.0
    for w0, w1 in zip(before, after):
        d = (w1 - w0).float()
        fro_sq += d.pow(2).sum().item()
        infs.append(d.abs().max().item())
        if d.ndim == 2:
            specs.append(torch.linalg.matrix_norm(d, ord=2).item())
    return {
        "dW_fro_total": fro_sq ** 0.5,
        "dW_spec_max": max(specs) if specs else float("nan"),
        "dW_spec_mean": (sum(specs) / len(specs)) if specs else float("nan"),
        "dW_inf_max": max(infs) if infs else float("nan"),
    }


def run_one(optimizer: str, seed: int, n_steps: int, head_size: int,
            n_train_steps: int, batch_size: int, measure_every: int,
            muon_lr: float, adam_lr: float, probe_batch: int,
            grad_clip: float = 0.0) -> dict:
    """One training run, instrumented. Protocol (task, model, batch sampling,
    lr defaults) matches muon_vs_adam_toy.py exactly so these numbers sit
    alongside that script's 10-seed accuracy/ablation results rather than
    being a separate incomparable universe."""
    torch.manual_seed(seed)
    model = FrozenFinalReadoutController(head_size, n_steps)
    target_fn = lambda x, y: x * y  # noqa: E731 — 'multiply', same as the toy

    muon_params, adam_params = _split_muon_adam_params(model)
    if optimizer == "muon":
        opt_main = SingleDeviceMuon(muon_params, lr=muon_lr)
        opt_aux = torch.optim.AdamW(adam_params, lr=adam_lr)
        own_norm_key = "dW_spec_max"   # what Muon itself controls
    elif optimizer == "adam":
        opt_main = torch.optim.AdamW(muon_params, lr=adam_lr)
        opt_aux = torch.optim.AdamW(adam_params, lr=adam_lr)
        own_norm_key = "dW_inf_max"    # what Adam itself (approximately) controls
    else:
        raise ValueError(optimizer)

    # Fixed, frozen probe batch: the SAME inputs at every measurement, so an
    # observed change in the state trace is caused by the weight update and
    # not by different data. Drawn from its own generator so it does not
    # consume the training RNG stream and shift the run being measured.
    g = torch.Generator().manual_seed(10_000 + seed)
    pa = torch.empty(probe_batch).uniform_(-3.0, 3.0, generator=g)
    pb = torch.empty(probe_batch).uniform_(-3.0, 3.0, generator=g)

    rows: list[dict] = []
    for step in range(n_train_steps):
        a = torch.empty(batch_size).uniform_(-3.0, 3.0)
        b = torch.empty(batch_size).uniform_(-3.0, 3.0)
        y_hat, _, _ = model(a, b)
        loss = F.mse_loss(y_hat, target_fn(a, b))
        opt_main.zero_grad()
        opt_aux.zero_grad()
        loss.backward()
        if grad_clip and grad_clip > 0:
            # Off by default, so every earlier result stays reproducible.
            # It exists because at T=32 Adam reaches id_r2 ~0.001 on every seed
            # and at every lr tried (3e-3 / 1e-3 / 3e-4, 2026-09-15), i.e. it
            # learns nothing, while Muon converges on the same task -- and Muon
            # is structurally immune to an exploding gradient through a 32-step
            # recurrence because it orthogonalises the update. Comparing an
            # optimizer that survives that to one that does not is a comparison
            # about clipping, not about geometry.
            torch.nn.utils.clip_grad_norm_(
                [q for q in list(muon_params) + list(adam_params)
                 if q.grad is not None], grad_clip)

        measuring = (step % measure_every == 0)
        if measuring:
            w_before = [p.detach().clone() for p in muon_params]
            s_before = _state_trace(model, pa, pb)

        # Order matters for attribution, not for training: the two param
        # groups are disjoint (split by name) and both gradients are already
        # computed, so stepping main-then-aux is identical in outcome to
        # stepping them together — but measuring BETWEEN them means the
        # observed state displacement is caused only by the weights whose
        # update norm we measure. Measuring after both steps (the first
        # version of this file) contaminated dS with the aux AdamW's update
        # to the step embedding / readout / final_r / final_w_raw, which
        # never appear in dW — weakening exactly the link under test.
        opt_main.step()

        if measuring:
            w_after = [p.detach().clone() for p in muon_params]
            s_after = _state_trace(model, pa, pb)

        opt_aux.step()

        if measuring:
            row = _update_norms(w_before, w_after)
            row["dS_abs"], row["dS_rel"] = _trace_displacement(s_before, s_after)
            row["step"] = step
            row["train_mse"] = loss.item()
            rows.append(row)

    # Held-out accuracy, same protocol as the toy script — so a run that
    # failed to learn can be recognised rather than silently contributing
    # geometry statistics from a broken solution.
    with torch.no_grad():
        ia = torch.empty(1000).uniform_(-3.0, 3.0)
        ib = torch.empty(1000).uniform_(-3.0, 3.0)
        iy, _, _ = model(ia, ib)
        id_r2 = 1.0 - F.mse_loss(iy, target_fn(ia, ib)).item() / target_fn(ia, ib).var().item()

    dS = [r["dS_abs"] for r in rows]
    dS_rel = [r["dS_rel"] for r in rows]
    own = [r[own_norm_key] for r in rows]
    ratio = [d / o if o > 0 else float("nan") for d, o in zip(dS, own)]
    ratio = [r for r in ratio if r == r]  # drop NaNs

    return {
        "optimizer": optimizer,
        "seed": seed,
        "id_r2": id_r2,
        "converged": bool(id_r2 > CONVERGED_ID_R2),
        "own_norm_key": own_norm_key,
        "n_measurements": len(rows),
        # Does the optimizer actually hold its own norm constant? (sanity: for
        # Muon this should be near zero by construction; if it isn't, the
        # measurement itself is wrong and nothing below can be trusted.)
        "cv_own_norm": _cv(own),
        "cv_dS_abs": _cv(dS),
        "cv_dS_rel": _cv(dS_rel),
        # The headline: how much the induced state displacement varies per
        # unit of the optimizer's own idea of step size.
        "cv_ratio_dS_per_own_norm": _cv(ratio),
        # Pre-registered refutation check: does ANY fixed weight-space norm
        # track state displacement? |r| >= 0.9 for any of these refutes the
        # strong version's premise.
        "corr_dS_vs_spec_max": _pearson([r["dW_spec_max"] for r in rows], dS),
        "corr_dS_vs_spec_mean": _pearson([r["dW_spec_mean"] for r in rows], dS),
        "corr_dS_vs_fro_total": _pearson([r["dW_fro_total"] for r in rows], dS),
        "corr_dS_vs_inf_max": _pearson([r["dW_inf_max"] for r in rows], dS),
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4, help="Recurrence steps (T).")
    ap.add_argument("--head-size", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--probe-batch", type=int, default=64)
    ap.add_argument("--measure-every", type=int, default=20)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--grad-clip", type=float, default=0.0,
                     help="global grad-norm clip, 0 = off (the default, so "
                          "earlier results reproduce). Needed to make the "
                          "long-T comparison fair: see the note at the clip "
                          "site.")
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

    all_runs = []
    for opt_name in ("adam", "muon"):
        for seed in range(args.seeds):
            r = run_one(opt_name, seed, args.steps, args.head_size,
                        args.train_steps, args.batch_size, args.measure_every,
                        args.muon_lr, args.adam_lr, args.probe_batch,
                        grad_clip=args.grad_clip)
            all_runs.append(r)
            progress(f"[{opt_name} seed={seed}] id_r2={r['id_r2']:.4f}  "
                  f"CV(own {r['own_norm_key']})={r['cv_own_norm']:.3f}  "
                  f"CV(dS)={r['cv_dS_abs']:.3f}  "
                  f"CV(dS/own)={r['cv_ratio_dS_per_own_norm']:.3f}  "
                  f"corr(dS,spec)={r['corr_dS_vs_spec_max']:+.3f}  "
                  f"corr(dS,fro)={r['corr_dS_vs_fro_total']:+.3f}  "
                  f"corr(dS,inf)={r['corr_dS_vs_inf_max']:+.3f}")

    def agg(opt_name: str, key: str) -> tuple[float, float]:
        vals = [r[key] for r in all_runs
                if r["optimizer"] == opt_name and r["converged"] and r[key] == r[key]]
        if not vals:
            return float("nan"), float("nan")
        return sum(vals) / len(vals), statistics.pstdev(vals)

    print("\n=== summary over seeds (CONVERGED runs only) ===")
    summary_rows = {}
    for opt_name in ("adam", "muon"):
        runs_o = [r for r in all_runs if r["optimizer"] == opt_name]
        n_ok = sum(1 for r in runs_o if r["converged"])
        line = {"converged_frac": (n_ok / len(runs_o)) if runs_o else 0.0,
                "n_converged": n_ok, "n_seeds": len(runs_o)}
        if n_ok == 0:
            print(f"  {opt_name:5s} NO CONVERGED RUN — every statistic for this "
                  f"arm is measured on a model that never learned the task, and "
                  f"is reported as NaN rather than as a number")
        for key in ("id_r2", "cv_own_norm", "cv_dS_abs", "cv_ratio_dS_per_own_norm",
                    "corr_dS_vs_spec_max", "corr_dS_vs_fro_total", "corr_dS_vs_inf_max"):
            m, s = agg(opt_name, key)
            line[key] = {"mean": m, "std": s}
            print(f"  {opt_name:5s} {key:28s} {m:+.4f} +- {s:.4f}"
                  + ("" if n_ok == len(runs_o) else f"   [{n_ok}/{len(runs_o)} seeds]"))
        summary_rows[opt_name] = line

    # Pre-registered verdict, computed rather than eyeballed — and refused
    # outright if the run was too short for the correlations to mean anything.
    n_points = min(r["n_measurements"] for r in all_runs)
    live_opts = [o for o in ("adam", "muon") if summary_rows[o]["n_converged"] > 0]
    dead_opts = [o for o in ("adam", "muon") if summary_rows[o]["n_converged"] == 0]
    corrs = [
        abs(summary_rows[o][k]["mean"])
        for o in live_opts
        for k in ("corr_dS_vs_spec_max", "corr_dS_vs_fro_total", "corr_dS_vs_inf_max")
        if summary_rows[o][k]["mean"] == summary_rows[o][k]["mean"]
    ]
    if n_points < MIN_POINTS_FOR_CORR or not corrs:
        worst_corr = float("nan")
        premise_refuted = None
        verdict = (f"NO VERDICT — {n_points} measurement points per run, "
                   f"need >= {MIN_POINTS_FOR_CORR} for a meaningful correlation")
    else:
        worst_corr = max(corrs)
        premise_refuted = worst_corr >= 0.9
        verdict = (f"strongest |corr(dS, any fixed norm)| = {worst_corr:.3f} "
                   f"-> H26 strong-version premise "
                   f"{'REFUTED' if premise_refuted else 'survives'} "
                   f"(pre-registered threshold 0.9, n={n_points} points/run, "
                   f"over converged arms: {', '.join(live_opts)})")
        if dead_opts:
            verdict += ("\n[!] NOT EVALUABLE for " + ", ".join(dead_opts) +
                        " — no seed of that arm learned the task at this "
                        "configuration, so this T is void for it and must not "
                        "be scored as a pass or a failure")
    print(f"\n{verdict}")

    if args.out is not None:
        save_result(
            args.out,
            {"runs": all_runs, "summary": summary_rows,
             "strongest_abs_corr_dS_vs_fixed_norm": worst_corr,
             "premise_refuted_at_0.9": premise_refuted,
             "verdict": verdict,
             "n_measurement_points_per_run": n_points,
             "config": vars(args) | {"out": str(args.out)}},
            experiment="step_geometry_mismatch",
            hypothesis=["H26"],
            summary={
                "strongest |corr(dS, fixed weight norm)|": f"{worst_corr:.3f}",
                "CV(dS per own-norm step), adam": f"{summary_rows['adam']['cv_ratio_dS_per_own_norm']['mean']:.2f}",
                "CV(dS per own-norm step), muon": f"{summary_rows['muon']['cv_ratio_dS_per_own_norm']['mean']:.2f}",
            },
            script=str(Path(__file__).relative_to(_REPO_ROOT)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
