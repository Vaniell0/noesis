#!/usr/bin/env python3
"""rank_ceiling_calibration.py — what fraction of the REACHABLE rank does
plain training actually use, and is that fraction a property of training or of
the task?

This exists to control one number. `breadth_growth_probe.py` reported that
plain training stops at 41-46% of the structural rank ceiling, and that entry
went into H26 next to the observation that real G1i base sits at a similar
fraction (13-16 live directions of a reachable 32). A fraction that reproduces
across scales would be a finding about training. A fraction that changes with
the task is a finding about the task, and the resemblance is a coincidence.

The ceiling itself is structural, not a hyperparameter: the WKV state takes at
most one rank-1 write per recurrence step, so reachable rank is
min(n_recurrence_steps, head_size) — NOT head_size. A configuration sitting on
that number is saturated and says nothing about whether training would have
gone further.

Two substrates, deliberately different in how much room they leave:

  chain   `ChainedController` — the M-chain task. Its recurrence length IS its
          chain length, so short chains have very low ceilings (3, 4, 6) and
          the task has almost no room to be below them.
  staged  `FrozenFinalReadoutController` — the H25/H26 task, where recurrence
          length is set independently of the task's difficulty and can be run
          far above what the task needs.

If "plain training uses ~45% of reachable rank" is about training, both
substrates should show it at matched ceilings. The first measurement of this
(2026-09-13, scratch script, not saved — this file is its durable replacement)
found the chain substrate essentially AT its ceiling: 2.98/3 and 3.84/4. That
is the task being breadth-saturated, and it means the 45% cannot be read as a
law of plain training.
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

from experiments.A0_state_probe.micro_wkv import (
    ChainedController,
    FrozenFinalReadoutController,
    _sample_chain,
)
from experiments.A0_state_probe.muon_vs_adam_toy import (
    SingleDeviceMuon,
    _split_muon_adam_params,
)
from experiments._common.results import save_result
from experiments._common.runtime import limit_threads, progress

CONVERGED_ID_R2 = 0.5


@torch.no_grad()
def _live(state: torch.Tensor) -> float:
    sv = torch.linalg.svdvals(state.float())
    return (sv > 0.01 * sv[:, :1]).sum(dim=-1).float().mean().item()


def _train(model, sample, opt_name: str, steps: int, batch: int,
           muon_lr: float, adam_lr: float):
    mp, ap = _split_muon_adam_params(model)
    main = (SingleDeviceMuon(mp, lr=muon_lr) if opt_name == "muon"
            else torch.optim.AdamW(mp, lr=adam_lr))
    aux = torch.optim.AdamW(ap, lr=adam_lr)
    for _ in range(steps):
        loss = sample(batch)
        main.zero_grad()
        aux.zero_grad()
        loss.backward()
        main.step()
        aux.step()
    return model


def run_chain(opt_name: str, n_rounds: int, head_size: int, seed: int,
              steps: int, batch: int, muon_lr: float, adam_lr: float) -> dict:
    torch.manual_seed(seed)
    m = ChainedController(head_size, n_rounds)

    def loss_fn(b: int) -> torch.Tensor:
        o, c, t = _sample_chain(b, n_rounds, -2.5, 2.5)
        return F.mse_loss(m(o, c), t)

    _train(m, loss_fn, opt_name, steps, batch, muon_lr, adam_lr)
    with torch.no_grad():
        o, c, t = _sample_chain(2000, n_rounds, -2.5, 2.5)
        id_r2 = 1 - F.mse_loss(m(o, c), t).item() / t.var().item()
        oo, oc, ot = _sample_chain(2000, n_rounds, -4.5, 4.5)
        ood = 1 - F.mse_loss(m(oo, oc), ot).item() / ot.var().item()
        _, st, _ = m(o, c, return_state=True)
        live = _live(st)
    ceiling = min(n_rounds, head_size)
    return {"substrate": "chain", "opt": opt_name, "seed": seed,
            "recurrence_steps": n_rounds, "head_size": head_size,
            "rank_ceiling": ceiling, "id_r2": id_r2, "ood_r2": ood,
            "live_directions": live, "fraction_of_ceiling": live / ceiling,
            "converged": bool(id_r2 > CONVERGED_ID_R2)}


def run_staged(opt_name: str, n_steps: int, head_size: int, seed: int,
               steps: int, batch: int, muon_lr: float, adam_lr: float) -> dict:
    torch.manual_seed(seed)
    m = FrozenFinalReadoutController(head_size, n_steps)
    target = lambda x, y: x * y  # noqa: E731

    def loss_fn(b: int) -> torch.Tensor:
        a = torch.empty(b).uniform_(-3.0, 3.0)
        bb = torch.empty(b).uniform_(-3.0, 3.0)
        return F.mse_loss(m(a, bb)[0], target(a, bb))

    _train(m, loss_fn, opt_name, steps, batch, muon_lr, adam_lr)
    with torch.no_grad():
        a = torch.empty(2000).uniform_(-3.0, 3.0)
        b2 = torch.empty(2000).uniform_(-3.0, 3.0)
        y = target(a, b2)
        y_hat, state, _ = m(a, b2)
        id_r2 = 1 - F.mse_loss(y_hat, y).item() / y.var().item()
        sa = (torch.randint(0, 2, (2000,)) * 2 - 1).float()
        sb = (torch.randint(0, 2, (2000,)) * 2 - 1).float()
        ao = sa * torch.empty(2000).uniform_(3.0, 5.0)
        bo = sb * torch.empty(2000).uniform_(3.0, 5.0)
        yo = target(ao, bo)
        ood = 1 - F.mse_loss(m(ao, bo)[0], yo).item() / yo.var().item()
        live = _live(state)
    ceiling = min(n_steps, head_size)
    return {"substrate": "staged", "opt": opt_name, "seed": seed,
            "recurrence_steps": n_steps, "head_size": head_size,
            "rank_ceiling": ceiling, "id_r2": id_r2, "ood_r2": ood,
            "live_directions": live, "fraction_of_ceiling": live / ceiling,
            "converged": bool(id_r2 > CONVERGED_ID_R2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--head-size", type=int, default=8)
    ap.add_argument("--ceilings", default="3,4,6",
                    help="recurrence lengths to run on BOTH substrates, so the "
                         "comparison is at matched ceilings rather than across "
                         "two different configurations.")
    ap.add_argument("--train-steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--adam-lr", type=float, default=3e-3)
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

    ceils = [int(x) for x in args.ceilings.split(",") if x.strip()]
    runs = []
    for T in ceils:
        for opt in ("adam", "muon"):
            for seed in range(args.seeds):
                for fn, label in ((run_chain, "chain"), (run_staged, "staged")):
                    r = fn(opt, T, args.head_size, seed, args.train_steps,
                           args.batch_size, args.muon_lr, args.adam_lr)
                    runs.append(r)
                    progress(f"[{label:6s} T={T:2d} {opt:4s} seed={seed}] "
                          f"id_r2={r['id_r2']:+.4f} ood={r['ood_r2']:+.4f} "
                          f"live={r['live_directions']:5.2f}/{r['rank_ceiling']} "
                          f"= {r['fraction_of_ceiling']:.0%}"
                          + ("" if r["converged"] else "  [DID NOT CONVERGE]"))

    print("\n=== fraction of reachable rank used, converged seeds only ===")
    summary = {}
    for sub in ("chain", "staged"):
        for T in ceils:
            for opt in ("adam", "muon"):
                rs = [r for r in runs if r["substrate"] == sub
                      and r["recurrence_steps"] == T and r["opt"] == opt
                      and r["converged"]]
                key = f"{sub}_T{T}_{opt}"
                if not rs:
                    summary[key] = None
                    print(f"  {key:22s} no converged seed")
                    continue
                fr = [r["fraction_of_ceiling"] for r in rs]
                summary[key] = {
                    "n_converged": len(rs),
                    "live_mean": sum(r["live_directions"] for r in rs) / len(rs),
                    "ceiling": rs[0]["rank_ceiling"],
                    "fraction_mean": sum(fr) / len(fr),
                    "fraction_std": statistics.pstdev(fr) if len(fr) > 1 else 0.0,
                    "ood_mean": sum(r["ood_r2"] for r in rs) / len(rs),
                }
                s = summary[key]
                print(f"  {key:22s} live={s['live_mean']:5.2f}/{s['ceiling']} "
                      f"= {s['fraction_mean']:.0%}±{s['fraction_std']:.0%} "
                      f"({s['n_converged']}/{args.seeds} seeds, ood={s['ood_mean']:+.4f})")

    def frac(sub: str) -> list[float]:
        return [v["fraction_mean"] for k, v in summary.items()
                if v and k.startswith(sub + "_")]

    ch, st = frac("chain"), frac("staged")
    if ch and st:
        ch_m, st_m = sum(ch) / len(ch), sum(st) / len(st)
        spread = max(ch + st) - min(ch + st)
        verdict = (
            f"fraction of reachable rank used by plain training: chain "
            f"{ch_m:.0%} (range {min(ch):.0%}-{max(ch):.0%}), staged {st_m:.0%} "
            f"(range {min(st):.0%}-{max(st):.0%}); spread across all "
            f"configurations {spread:.0%}. -> "
            + ("the fraction is roughly constant across substrates — it is a "
               "property of plain training, and the resemblance to G1i's "
               "13-16 of 32 is worth pursuing"
               if spread < 0.20 else
               "the fraction is NOT constant across substrates — it is a "
               "property of the configuration, so '~45% of reachable rank' "
               "must not be read as a law of plain training, and its "
               "resemblance to G1i's fraction is a coincidence")
        )
    else:
        verdict = "not enough converged seeds to compare substrates"
    print(f"\n{verdict}")

    if args.out is not None:
        save_result(
            args.out,
            {"runs": runs, "summary": summary, "verdict": verdict,
             "config": vars(args) | {"out": str(args.out)}},
            experiment="rank_ceiling_calibration", hypothesis=["H26", "H8"],
            summary={
                "chain substrate, fraction of ceiling used":
                    f"{(sum(ch)/len(ch)):.0%}" if ch else "n/a",
                "staged substrate, fraction of ceiling used":
                    f"{(sum(st)/len(st)):.0%}" if st else "n/a",
                "spread across configurations":
                    f"{(max(ch+st)-min(ch+st)):.0%}" if (ch and st) else "n/a",
            },
            script=str(Path(__file__).relative_to(_REPO_ROOT)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
