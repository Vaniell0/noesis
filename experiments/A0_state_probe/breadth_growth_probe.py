#!/usr/bin/env python3
"""breadth_growth_probe.py — can training be made to BUILD state breadth,
or does it only ever bake what pretraining handed over?

H26's central criterion after the 2026-09-13 reframing:

    If no training procedure that can be built raises the state's
    live-direction count above the pretrained baseline, then "builds the
    geometry" has no operational content and the distinction is decoration.

The baseline this is aimed at is real, not hypothetical: G1i base carries
13-16 live directions of 64 at the WKV state, and the Phase-1 LoRA checkpoint
that *works* carries the same 13-16 with near-identical per-head distributions
(`results/rank_recheck*/jlens.json`). Ordinary fine-tuning does not move that
number. The question here is whether anything does.

Arms, same task/seed/protocol as the rest of the H25/H26 toy family:

  adam                  the baseline that works
  muon                  the other pure optimizer
  muon_breadth_L<w>     Muon plus an explicit breadth term at weight w
  adam_breadth_L<w>     Adam plus the same term — because if the term is what
                        matters, it should work under either optimizer, and if
                        it only works under one, that is itself the finding

The breadth term maximises the entropy of the state's normalised singular
values — i.e. it directly optimises `effective_rank_entropy`, the same
quantity the probe reports. **This is deliberate Goodhart and is named as
such**: of course the metric moves when you optimise it. The informative part
is the pair of things it does NOT trivially determine — whether task quality
survives, and whether the mechanism signature (the a_gate ablation) changes.
H26's second refutation criterion is exactly this: breadth gained with quality
flat or falling is metric-chasing, not a result.
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


def _entropy_rank(state: torch.Tensor) -> torch.Tensor:
    """exp(H(p)) over normalised singular values, averaged over the batch —
    differentiable, so it can be used as a loss term. Same definition as the
    `effective_rank_entropy` the jlens probe reports, so the training signal
    and the measurement are literally the same quantity."""
    sv = torch.linalg.svdvals(state.float())          # [B, h]
    p = sv / (sv.sum(dim=-1, keepdim=True) + 1e-9)
    ent = -(p * (p + 1e-9).log()).sum(dim=-1)
    return ent.exp().mean()


@torch.no_grad()
def _measure(model, target_fn, n_recurrence_steps: int, n: int = 1000) -> dict:
    a = torch.empty(n).uniform_(-3.0, 3.0)
    b = torch.empty(n).uniform_(-3.0, 3.0)
    y = target_fn(a, b)
    y_hat, final_state, trace = model(a, b)
    id_r2 = 1.0 - F.mse_loss(y_hat, y).item() / y.var().item()

    yg, _, _ = model(a, b, force_a_gate=0.0)
    abl = 1.0 - F.mse_loss(yg, y).item() / y.var().item()

    # Breadth of the state, measured the way the real probe measures it:
    # count of singular values above 1% of the largest, plus entropy rank.
    sv = torch.linalg.svdvals(final_state.float())     # [B, h]
    top = sv[:, :1]
    live = (sv > 0.01 * top).sum(dim=-1).float().mean().item()
    return {
        "id_r2": id_r2,
        "ablation_a_gate_0_r2": abl,
        "live_directions_final": live,
        "entropy_rank_final": float(_entropy_rank(final_state)),
        "head_size": final_state.shape[-1],
        # Reported because the first smoke run of this file saturated it and
        # the verdict misread saturation as "no growth": the state gains at
        # most one rank-1 write per recurrence step, so the reachable rank is
        # min(n_steps, head_size), NOT head_size. Any arm sitting exactly on
        # this number is at the ceiling and the comparison is uninformative.
        "rank_ceiling": min(n_recurrence_steps, final_state.shape[-1]),
    }


def run_arm(arm: str, seed: int, *, n_steps: int, head_size: int,
            n_train_steps: int, batch_size: int, muon_lr: float,
            adam_lr: float, breadth_w: float) -> dict:
    torch.manual_seed(seed)
    model = FrozenFinalReadoutController(head_size, n_steps)
    target_fn = lambda x, y: x * y  # noqa: E731
    muon_params, adam_params = _split_muon_adam_params(model)

    use_muon = arm.startswith("muon")
    opt_main = (SingleDeviceMuon(muon_params, lr=muon_lr) if use_muon
                else torch.optim.AdamW(muon_params, lr=adam_lr))
    opt_aux = torch.optim.AdamW(adam_params, lr=adam_lr)
    w = breadth_w if "breadth" in arm else 0.0

    for _ in range(n_train_steps):
        a = torch.empty(batch_size).uniform_(-3.0, 3.0)
        b = torch.empty(batch_size).uniform_(-3.0, 3.0)
        y_hat, final_state, _ = model(a, b)
        task = F.mse_loss(y_hat, target_fn(a, b))
        loss = task - w * _entropy_rank(final_state) if w > 0 else task
        opt_main.zero_grad()
        opt_aux.zero_grad()
        loss.backward()
        opt_main.step()
        opt_aux.step()

    out = {"arm": arm, "seed": seed, "breadth_weight": w}
    out.update(_measure(model, target_fn, n_recurrence_steps=n_steps))
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
    ap.add_argument("--breadth-weights", default="0.01,0.1")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    arms = [("adam", 0.0), ("muon", 0.0)]
    for w in (float(x) for x in args.breadth_weights.split(",")):
        arms.append((f"muon_breadth_L{w:g}", w))
        arms.append((f"adam_breadth_L{w:g}", w))

    runs = []
    for arm, w in arms:
        for seed in range(args.seeds):
            r = run_arm(arm, seed, n_steps=args.steps, head_size=args.head_size,
                        n_train_steps=args.train_steps, batch_size=args.batch_size,
                        muon_lr=args.muon_lr, adam_lr=args.adam_lr, breadth_w=w)
            runs.append(r)
            print(f"[{arm:22s} seed={seed}] id_r2={r['id_r2']:+.4f} "
                  f"abl={r['ablation_a_gate_0_r2']:+.4f} "
                  f"live={r['live_directions_final']:.2f}/{r['rank_ceiling']} "
                  f"(head {r['head_size']})  eRank={r['entropy_rank_final']:.2f}")

    print("\n=== summary over seeds ===")
    summary = {}
    for arm, _ in arms:
        rs = [r for r in runs if r["arm"] == arm]
        line = {k: {"mean": sum(r[k] for r in rs) / len(rs),
                    "std": statistics.pstdev([r[k] for r in rs]) if len(rs) > 1 else 0.0}
                for k in ("id_r2", "ablation_a_gate_0_r2",
                          "live_directions_final", "entropy_rank_final")}
        summary[arm] = line
        print(f"  {arm:22s} live={line['live_directions_final']['mean']:5.2f}"
              f"±{line['live_directions_final']['std']:.2f}  "
              f"eRank={line['entropy_rank_final']['mean']:5.2f}  "
              f"id_r2={line['id_r2']['mean']:+.4f}±{line['id_r2']['std']:.4f}  "
              f"abl={line['ablation_a_gate_0_r2']['mean']:+.4f}")

    # The two questions, computed rather than eyeballed: did breadth rise
    # above the plain arms, and did task quality survive it?
    base_live = max(summary["adam"]["live_directions_final"]["mean"],
                    summary["muon"]["live_directions_final"]["mean"])
    base_q = max(summary["adam"]["id_r2"]["mean"], summary["muon"]["id_r2"]["mean"])
    best_arm, best_live, best_q = None, -1.0, None
    for arm, line in summary.items():
        if "breadth" not in arm:
            continue
        if line["live_directions_final"]["mean"] > best_live:
            best_arm = arm
            best_live = line["live_directions_final"]["mean"]
            best_q = line["id_r2"]["mean"]
    ceiling = min(args.steps, args.head_size)
    grew = best_live > base_live + 0.5
    kept = best_q is not None and best_q > base_q - 0.02
    at_ceiling = best_live >= ceiling - 0.05
    verdict = (
        f"rank ceiling for this config = min(n_steps={args.steps}, head={args.head_size}) "
        f"= {ceiling}"
        + ("  [!] best arm is AT the ceiling — this config cannot show growth, "
           "rerun with more recurrence steps" if at_ceiling else "")
        + f". best breadth arm: {best_arm} live={best_live:.2f} vs plain best {base_live:.2f}; "
        f"quality {best_q:+.4f} vs {base_q:+.4f}. -> "
        + ("breadth CAN be built and the task survives it"
           if grew and kept else
           "breadth rose but the task did not survive — metric-chasing, per H26's Goodhart criterion"
           if grew and not kept else
           "breadth did not rise above the plain arms — H26's central criterion is not met by this term")
    )
    print(f"\n{verdict}")

    if args.out is not None:
        save_result(
            args.out,
            {"runs": runs, "summary": summary, "verdict": verdict,
             "breadth_grew": grew, "quality_kept": kept,
             "config": vars(args) | {"out": str(args.out)}},
            experiment="breadth_growth_toy", hypothesis=["H26"],
            summary={
                "live directions, plain adam / muon":
                    f"{summary['adam']['live_directions_final']['mean']:.2f} / "
                    f"{summary['muon']['live_directions_final']['mean']:.2f}",
                "live directions, best breadth arm": f"{best_live:.2f} ({best_arm})",
                "id_r2, plain best / best breadth arm": f"{base_q:+.4f} / {best_q:+.4f}",
            },
            script=str(Path(__file__).relative_to(_REPO_ROOT)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
