#!/usr/bin/env python3
"""aspect_dose_probe.py — is the aspect coefficient's pairing-blindness a cause,
or just a thing that is true about the shapes?

Standing as of 2026-09-14: Muon's upstream rescale

    update *= max(1, m / n) ** 0.5                 (muon_opt.py:59)

gives one member of a matched pair a larger step than the other whenever the
two have mismatched orientation. Measured on the real G1i checkpoint:

    lora_A  (32, 2560)      1.000     |  ffn.key   (10240, 2560)   2.000
    lora_B  (2560, 32)      8.944     |  ffn.value (2560, 10240)   1.000

Both are matched pairs — the LoRA factors compose into one update, and
`ffn.key`/`ffn.value` are the two halves of one FFN block. The coefficient is
correct upstream, where it normalises a standalone weight's per-element RMS;
it is blind here because it cannot see that these tensors are paired.

**That is an observation about shapes, not a measured effect.** Nobody has
shown the asymmetry damages anything. This probe asks two questions the
existing evidence cannot answer:

1. **Dose-response.** If the asymmetry is the cause, damage should grow with
   the coefficient. The probe sweeps the adapter's aspect ratio by varying
   hidden width at fixed rank, so `sqrt(hidden/r)` takes several values, and
   asks whether retention loss tracks it. If damage is flat across a 4x range
   of coefficient, the mechanism is dead regardless of how good the story is.

2. **Step size vs pairing — the control H26 pre-registered and the LoRA result
   never had.** Two different interventions, which the earlier `muon_balanced`
   arm conflated:

     noaspect   drop the coefficient (1.0 on both factors) — removes the
                ASYMMETRY, and incidentally lowers B's step
     lr_div     keep the coefficient, divide the shared lr by it — lowers the
                STEP without removing the asymmetry

   Damage removed by `noaspect` but not `lr_div`  -> pairing, as claimed.
   Damage removed by both                         -> it was step size.
   Damage removed by neither                      -> neither; look elsewhere.

The protocol is `lora_muon_probe`'s: Muon-pretrained base (what G1i is), frozen,
LoRA finetuned on a narrow slice, and retention of the pretrained ability is the
number that matters — the 2.9B collapse showed up only on data the finetune
never touched.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.functional as F

from experiments.A0_state_probe.micro_wkv import FrozenFinalReadoutController
from experiments.A0_state_probe.lora_muon_probe import LoRALinear, LoRAMuon, lorafy, _ns
from experiments.A0_state_probe.muon_vs_adam_toy import (
    SingleDeviceMuon,
    _split_muon_adam_params,
)
from experiments._common.convergence import CONVERGED_ID_R2
from experiments._common.results import save_result
from experiments._common.runtime import limit_threads, progress

ARMS = ("adam", "factorwise", "noaspect", "lr_div")
TARGET = lambda x, y: x * y  # noqa: E731


class AspectMuon(LoRAMuon):
    """`LoRAMuon` with the two interventions this probe needs.

    Registers its extra mode with the parent's whitelist rather than widening
    that whitelist in `lora_muon_probe` — the modes there are the ones its own
    result file was produced with, and adding to that tuple would make an older
    result look like it had run against a different arm set.

    `factorwise` is inherited unchanged so the baseline here is literally the
    production update rule, not a re-implementation of it.
    """

    EXTRA_MODES = ("muon_noaspect",)

    def __init__(self, adapters, mode: str, lr: float, **kw):
        if mode in self.EXTRA_MODES:
            super().__init__(adapters, "muon_factorwise", lr, **kw)
            self.mode = mode
        else:
            super().__init__(adapters, mode, lr, **kw)

    @torch.no_grad()
    def directions(self, ad: LoRALinear, gA: torch.Tensor, gB: torch.Tensor):
        if self.mode == "muon_noaspect":
            # Orthogonalize both factors and apply NO shape-derived rescale.
            # This removes the asymmetry; it also lowers B's step, which is
            # exactly why `lr_div` exists as its partner control.
            return _ns(gA), _ns(gB)
        return super().directions(ad, gA, gB)


def _r2(model, lo, hi, n=1000, force_a_gate=None) -> float:
    with torch.no_grad():
        a = torch.empty(n).uniform_(lo, hi)
        b = torch.empty(n).uniform_(lo, hi)
        y = TARGET(a, b)
        return 1.0 - F.mse_loss(model(a, b, force_a_gate=force_a_gate)[0], y).item() / y.var().item()


def pretrain(seed, head_size, n_steps, hidden, steps, batch, muon_lr, adam_lr):
    """Muon-pretrained, because that is what the real base is."""
    torch.manual_seed(seed)
    m = FrozenFinalReadoutController(head_size, n_steps, hidden=hidden)
    mp, ap = _split_muon_adam_params(m)
    main = SingleDeviceMuon(mp, lr=muon_lr)
    aux = torch.optim.AdamW(ap, lr=adam_lr)
    for _ in range(steps):
        a = torch.empty(batch).uniform_(-3.0, 3.0)
        b = torch.empty(batch).uniform_(-3.0, 3.0)
        loss = F.mse_loss(m(a, b)[0], TARGET(a, b))
        main.zero_grad(); aux.zero_grad(); loss.backward(); main.step(); aux.step()
    return m


def run_arm(arm, base, hidden, lora_r, lora_alpha, steps, batch, narrow,
            muon_lr, adam_lr, seed) -> dict:
    torch.manual_seed(seed + 10_000)
    m = copy.deepcopy(base)
    adapters = lorafy(m, lora_r, lora_alpha)
    params = [p for ad in adapters for p in (ad.lora_A, ad.lora_B)]
    aspect = max(1.0, hidden / lora_r) ** 0.5

    if arm == "adam":
        opt = torch.optim.AdamW(params, lr=adam_lr)
    elif arm == "factorwise":
        opt = AspectMuon(adapters, "muon_factorwise", lr=muon_lr)
    elif arm == "noaspect":
        opt = AspectMuon(adapters, "muon_noaspect", lr=muon_lr)
    elif arm == "lr_div":
        # Same update rule as production, smaller shared step. Isolates "the
        # step was too big" from "the two factors' steps were mismatched".
        opt = AspectMuon(adapters, "muon_factorwise", lr=muon_lr / aspect)
    else:
        raise ValueError(arm)

    for _ in range(steps):
        a = torch.empty(batch).uniform_(-narrow, narrow)
        b = torch.empty(batch).uniform_(-narrow, narrow)
        loss = F.mse_loss(m(a, b)[0], TARGET(a, b))
        opt.zero_grad(); loss.backward(); opt.step()

    return {"arm": arm, "seed": seed, "hidden": hidden, "lora_r": lora_r,
            "aspect": aspect,
            "retain_r2": _r2(m, -3.0, 3.0), "adapt_r2": _r2(m, -narrow, narrow),
            "ablation_a_gate_0_r2": _r2(m, -3.0, 3.0, force_a_gate=0.0)}


def summarize(runs, cells):
    """Table + verdict from a run list, so sharded runs can be merged.

    Sharding is by SEED, not by grid cell: cells at the same width share their
    pretrained bases, and splitting cells across processes would re-pretrain
    them once per process.
    """
    progress("\n=== damage vs aspect coefficient (retain_r2, mean over seeds) ===")
    progress(f"{'hidden':>7} {'r':>4} {'aspect':>7} "
             + " ".join(f"{a:>11}" for a in ARMS) + f" {'damage':>8}")
    summary, dose_by_width = {}, {}
    for hidden, lora_r in cells:
        aspect = max(1.0, hidden / lora_r) ** 0.5
        row = {}
        for arm in ARMS:
            vals = [r["retain_r2"] for r in runs
                    if r["hidden"] == hidden and r["lora_r"] == lora_r
                    and r["arm"] == arm]
            row[arm] = {"mean": statistics.mean(vals) if vals else float("nan"),
                        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                        "n": len(vals)}
        # "Damage" is measured against Adam on the SAME base and budget, so it
        # is the optimizer's cost, not the task's difficulty.
        dmg = row["factorwise"]["mean"] - row["adam"]["mean"]
        row["damage_vs_adam"] = dmg
        key = f"h{hidden}_r{lora_r}"
        summary[key] = row | {"aspect": aspect, "hidden": hidden,
                              "lora_r": lora_r}
        dose_by_width.setdefault(hidden, []).append((aspect, dmg, key))
        progress(f"{hidden:>7} {lora_r:>4} {aspect:>7.3f} "
                 + " ".join(f"{row[a]['mean']:>+11.4f}" for a in ARMS)
                 + f" {dmg:>+8.4f}")

    # Does damage track the coefficient, and which intervention removes it?
    #
    # Thresholds are RELATIVE to the measured damage, not absolute. The first
    # version used a flat 0.02 on every comparison; at smoke scale the entire
    # factorwise-vs-Adam gap was 0.018, so no test could fire and the verdict
    # printed "damage does NOT track the coefficient" over a table where it had
    # tripled (-0.0059 -> -0.0179). An absolute threshold chosen before seeing
    # the effect size is as wrong as one chosen after.
    # Read WITHIN one width, for the reason given on --grid: a cross-width
    # dose mixes capacity with the coefficient. The widest family with the most
    # cells is the one the sweep was built around.
    fam = max(dose_by_width, key=lambda h: (len(dose_by_width[h]), h))
    dose = sorted(dose_by_width[fam])
    if len(dose) < 2:
        progress("\nonly one cell in the largest width family — nothing to dose")
        raise SystemExit(1)
    lo, hi = dose[0], dose[-1]
    dmg_lo, dmg_hi = abs(lo[1]), abs(hi[1])
    grows = dmg_hi > 1.5 * dmg_lo and dmg_hi > 1e-4
    last = summary[hi[2]]
    gap = last["adam"]["mean"] - last["factorwise"]["mean"]     # the damage to undo

    def recovered(arm: str) -> float:
        """Fraction of the factorwise-vs-Adam gap this intervention closes."""
        if gap <= 1e-9:
            return float("nan")
        return (last[arm]["mean"] - last["factorwise"]["mean"]) / gap

    rec_noaspect, rec_lrdiv = recovered("noaspect"), recovered("lr_div")
    FIXED = 0.5                      # closes at least half the gap
    fixed_by_noaspect = rec_noaspect == rec_noaspect and rec_noaspect >= FIXED
    fixed_by_lrdiv = rec_lrdiv == rec_lrdiv and rec_lrdiv >= FIXED

    verdict = (
        f"at hidden={fam} (capacity held fixed, coefficient moved by rank) "
        f"damage vs Adam at aspect {lo[0]:.2f}: {lo[1]:+.4f}; at aspect "
        f"{hi[0]:.2f}: {hi[1]:+.4f} ({dmg_hi / max(dmg_lo, 1e-9):.1f}x). -> "
        + (f"damage GROWS with the coefficient" if grows else
           "damage does NOT track the coefficient across this range, so the "
           "asymmetry is not the mechanism no matter how good the story is")
        + f"; at the largest aspect the gap to Adam is {gap:+.4f}, of which "
        f"noaspect recovers {rec_noaspect:.0%} and lr_div {rec_lrdiv:.0%} -> "
        + ("removing the asymmetry fixes it and lowering the step alone does not "
           "— the pairing blindness is the cause"
           if (fixed_by_noaspect and not fixed_by_lrdiv) else
           "BOTH interventions fix it — this probe cannot separate pairing from "
           "step size, and step size is the simpler explanation of the two"
           if (fixed_by_noaspect and fixed_by_lrdiv) else
           "only the step-size control fixes it — it was step size, not pairing"
           if fixed_by_lrdiv else
           "neither intervention recovers half the gap — the damage is neither "
           "the asymmetry nor the step size")
    )
    return summary, dose, fam, verdict


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seeds", type=int, default=6)
    ap_.add_argument("--seed-start", type=int, default=0,
                     help="first seed this shard owns. Shards must take "
                          "disjoint ranges wide enough for skipped seeds.")
    ap_.add_argument("--merge", default=None,
                     help="comma-separated shard result JSONs: re-derive the "
                          "table and verdict over their pooled runs and write "
                          "--out, running nothing.")
    ap_.add_argument("--head-size", type=int, default=16)
    ap_.add_argument("--steps", type=int, default=16)
    ap_.add_argument("--hiddens", default="64,256,1024",
                     help="hidden widths; with --lora-r these set the aspect "
                          "coefficient sqrt(hidden/r) the sweep varies.")
    ap_.add_argument("--lora-r", type=int, default=8)
    ap_.add_argument("--grid", default=None,
                     help="explicit `hidden:rank` cells, e.g. "
                          "'64:32,64:8,64:2'. Moving the coefficient by RANK at "
                          "a fixed width is the clean dose axis: widening the "
                          "net changes its capacity too, so a width sweep "
                          "confounds 'wider nets forget less' with 'a bigger "
                          "coefficient hurts more'. Overrides --hiddens.")
    ap_.add_argument("--lora-alpha", type=float, default=16.0)
    ap_.add_argument("--pretrain-steps", type=int, default=4000)
    ap_.add_argument("--finetune-steps", type=int, default=1500)
    ap_.add_argument("--batch-size", type=int, default=64)
    ap_.add_argument("--muon-lr", type=float, default=0.02)
    ap_.add_argument("--adam-lr", type=float, default=3e-3)
    ap_.add_argument("--narrow", type=float, default=1.0)
    ap_.add_argument("--max-seed-attempts", type=int, default=3)
    ap_.add_argument("--threads", type=int, default=None)
    ap_.add_argument("--out", type=Path, default=None)
    args = ap_.parse_args()
    progress(f"[aspect_dose_probe] torch threads = {limit_threads(args.threads)}")

    if args.merge:
        pooled = []
        for f in args.merge.split(","):
            pooled += json.loads(Path(f.strip()).read_text())["runs"]
        cells = sorted({(r["hidden"], r["lora_r"]) for r in pooled},
                       key=lambda c: max(1.0, c[0] / c[1]) ** 0.5)
        summary, dose, fam, verdict = summarize(pooled, cells)
        progress(f"\n{verdict}")
        last = summary[dose[-1][2]]
        if args.out is not None:
            save_result(
                args.out,
                {"runs": pooled, "summary": summary, "verdict": verdict,
                 "dose": [(a, d) for a, d, _ in dose],
                 "dose_family_hidden": fam,
                 "merged_from": args.merge.split(",")},
                experiment="aspect_dose_toy", hypothesis=["H26"],
                summary={
                    "damage vs Adam, by aspect coefficient (hidden fixed)":
                        "; ".join(f"{a:.2f}->{d:+.4f}" for a, d, _ in dose),
                    "removing the asymmetry (noaspect)":
                        f"{last['noaspect']['mean']:+.4f} vs factorwise "
                        f"{last['factorwise']['mean']:+.4f}",
                    "lowering the step instead (lr_div)":
                        f"{last['lr_div']['mean']:+.4f}",
                },
                script=str(Path(__file__).relative_to(_REPO_ROOT)),
            )
        return 0

    if args.grid:
        cells = [tuple(int(v) for v in tok.split(":"))
                 for tok in args.grid.split(",") if tok.strip()]
    else:
        cells = [(int(x), args.lora_r)
                 for x in args.hiddens.split(",") if x.strip()]

    runs = []
    base_cache: dict[int, dict] = {}
    for hidden, lora_r in cells:
        aspect = max(1.0, hidden / lora_r) ** 0.5
        if hidden not in base_cache:
            # Pretraining is Muon on dense `hidden`x`hidden` weights, and its
            # Newton-Schulz is cubic in that width: 0.05 s/step at 64 becomes
            # 23 s/step at 1024 (measured 2026-09-14), i.e. six days for the
            # width sweep as first launched. Bases are cached per width and
            # shared across every rank that reuses it.
            bases = {}
            seed = args.seed_start
            limit = seed + args.seeds * args.max_seed_attempts
            while len(bases) < args.seeds and seed < limit:
                m = pretrain(seed, args.head_size, args.steps, hidden,
                             args.pretrain_steps, args.batch_size,
                             args.muon_lr, args.adam_lr)
                r2 = _r2(m, -3.0, 3.0)
                if r2 > CONVERGED_ID_R2:
                    bases[seed] = m
                else:
                    progress(f"[hidden={hidden} seed={seed}] base r2={r2:+.4f} "
                             f"— did not learn, seed skipped")
                seed += 1
            base_cache[hidden] = bases
        bases = base_cache[hidden]
        progress(f"\n--- hidden={hidden} lora_r={lora_r}  "
                 f"aspect=sqrt({hidden}/{lora_r})={aspect:.3f}  "
                 f"({len(bases)} usable bases) ---")
        for arm in ARMS:
            for sd, base in bases.items():
                r = run_arm(arm, base, hidden, lora_r, args.lora_alpha,
                            args.finetune_steps, args.batch_size, args.narrow,
                            args.muon_lr, args.adam_lr, sd)
                runs.append(r)
                progress(f"[h={hidden:5d} r={lora_r:<3d} {arm:11s} seed={sd}] "
                         f"retain={r['retain_r2']:+.4f} "
                         f"adapt={r['adapt_r2']:+.4f}")

    summary, dose, fam, verdict = summarize(runs, cells)
    last = summary[dose[-1][2]]
    progress(f"\n{verdict}")

    if args.out is not None:
        save_result(
            args.out, {"runs": runs, "summary": summary, "verdict": verdict,
                       "dose": [(a, d) for a, d, _ in dose],
                       "dose_family_hidden": fam,
                       "config": vars(args) | {"out": str(args.out)}},
            experiment="aspect_dose_toy", hypothesis=["H26"],
            summary={
                "damage vs Adam, by aspect coefficient (hidden fixed)":
                    "; ".join(f"{a:.2f}->{d:+.4f}" for a, d, _ in dose),
                "removing the asymmetry (noaspect)":
                    f"{last['noaspect']['mean']:+.4f} vs factorwise "
                    f"{last['factorwise']['mean']:+.4f}",
                "lowering the step instead (lr_div)":
                    f"{last['lr_div']['mean']:+.4f}",
            },
            script=str(Path(__file__).relative_to(_REPO_ROOT)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
