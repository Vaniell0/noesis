#!/usr/bin/env python3
"""lineage_probe.py — the project's ACTUAL training lineage, staged, with the
one control it has never had: was it the optimizer, or was it going from a
rank-r adapter to the full weight space?

The real sequence, corrected 2026-09-14 (the earlier toy switch arm did not
reproduce it and should not be cited as if it did):

    Muon pretraining (G1i, upstream)
      -> Adam on a LoRA adapter over frozen base   (step500, Phase 1 — WORKED)
      -> merge the adapter into the base            (`--merge 1`, the default;
                                                     docs/phase15-gap-matrix.md:182)
      -> Muon on ALL weights                        (Phase 1.5 — COLLAPSED)

`optimizer_switch_probe.py`'s `adam_then_muon` arm is Adam-then-Muon on ONE
fixed parameter set, starting from an Adam-trained model. It differs from the
above in three ways, and the third was never stated anywhere in this project:

1. the origin optimizer (Adam there, Muon here),
2. the middle stage's parameter set (all weights there, a rank-r adapter here),
3. **the final stage EXPANDS the trainable set** — from the adapter to every
   weight. So "we switched optimizer" is confounded with "we switched which
   subspace is being optimized", and no run so far separates them.

Point 3 has a mechanism attached, which is why it is worth a probe rather than
a caveat. The Phase-1 adapter was fitted against a FROZEN base. Full-FT then
starts moving exactly the base weights the adapter was fitted to — breaking a
coupling that was never trained to survive being broken. Nothing in that story
mentions Muon.

The 2x2 at stage 2 separates the two factors, at matched steps and matched lr:

                      Adam                 Muon
    keep adapter      lora_adam            lora_muon      <- optimizer only
    merge + full      full_adam            full_muon      <- what actually ran

`full_adam` is the control the project has never run and the one that can
settle it. If `full_adam` collapses too, the optimizer is exonerated and the
fault is in Phase 1.5's construction — which changes the plan, not just the
write-up. If only `full_muon` collapses, Muon is implicated in the specific
setting of full-FT after a merged adapter, and `lora_muon` says whether it is
implicated more broadly than that.

What is measured after every stage, not just at the end — because the real
collapse was invisible in training diagnostics and showed up only on data the
finetune never touched:

    pretrain_r2   on [-3,3]   the broad ability the base was pretrained for
    stage1_r2     on [-1,1]   the narrow ability Phase 1 added
    ood_r2        on ±[3,5]   generalisation
    ablation      a_gate=0    which mechanism the solution leans on
    live/eRank                state breadth
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
import torch.nn as nn
import torch.nn.functional as F

from experiments.A0_state_probe.micro_wkv import FrozenFinalReadoutController
from experiments.A0_state_probe.lora_muon_probe import LoRALinear, lorafy
from experiments.A0_state_probe.muon_vs_adam_toy import (
    SingleDeviceMuon,
    _split_muon_adam_params,
)
from experiments._common.convergence import CONVERGED_ID_R2, is_converged
from experiments._common.results import save_result
from experiments._common.runtime import limit_threads, progress

STAGE2_ARMS = ("lora_adam", "lora_muon", "full_adam", "full_muon",
               "direct_full_adam", "direct_full_muon")

# Two skills on disjoint regions of the input, so "did stage 1 add a NEW
# ability" is a real question rather than a tautology.
#
# The first version of this file used one target (a*b) on [-3,3] and a narrow
# slice [-1,1] for stage 1. The base already solved the narrow slice at stage 0
# (r2 = 0.9917), so stage 1 had nothing to learn — and its finding that LoRA
# added zero live directions was therefore empty: nothing was built because
# nothing needed building. To ask whether LoRA CREATES capacity or REDISTRIBUTES
# existing weights, the new skill has to be one the pretrained model genuinely
# cannot do.
#
# Beyond the boundary the rule INVERTS. That cannot be reached by extrapolating
# the old rule — a base trained only on the old region predicts +a*b there and
# scores strongly negative — so learning it is real acquisition, not
# generalisation.
OLD_HI, NEW_LO, NEW_HI = 2.0, 2.0, 3.0


def sample_old(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a = torch.empty(n).uniform_(-OLD_HI, OLD_HI)
    b = torch.empty(n).uniform_(-OLD_HI, OLD_HI)
    return a, b, a * b


def sample_new(n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sgn = (torch.randint(0, 2, (n,)) * 2 - 1).float()
    a = sgn * torch.empty(n).uniform_(NEW_LO, NEW_HI)
    b = torch.empty(n).uniform_(-OLD_HI, OLD_HI)
    return a, b, -(a * b)


def make_finetune_sampler(replay: float):
    """Finetune batches: mostly the new skill, with `replay` of the old mixed in.

    Without any replay every arm — LoRA included — simply forgets the old skill
    outright (measured: old_r2 +0.99 -> -2.94 during the LoRA stage alone), and
    catastrophic forgetting swamps the effect under test. That is also not what
    the real runs looked like: the Phase-1 corpus was not exclusively new-skill
    data, and step500 kept 50% on the held-out eval. So a finetune mixture is
    the faithful setup, and `replay` is exposed as a knob rather than baked in,
    because "how much replay is needed" is itself a question the VM runs care
    about.
    """
    def sampler(n: int):
        k = int(round(n * replay))
        if k <= 0:
            return sample_new(n)
        if k >= n:
            return sample_old(n)
        ao, bo, yo = sample_old(k)
        an, bn, yn = sample_new(n - k)
        return (torch.cat([ao, an]), torch.cat([bo, bn]), torch.cat([yo, yn]))
    return sampler


def _r2(model, sampler, n: int = 1000, force_a_gate: float | None = None) -> float:
    with torch.no_grad():
        a, b, y = sampler(n)
        return 1.0 - F.mse_loss(model(a, b, force_a_gate=force_a_gate)[0], y).item() / y.var().item()


@torch.no_grad()
def _breadth(model, n: int = 512) -> dict:
    # Measured over BOTH regions: the question is how many directions the model
    # uses across everything it is now supposed to know, not within one skill.
    ao, bo, _ = sample_old(n // 2)
    an, bn, _ = sample_new(n - n // 2)
    a, b = torch.cat([ao, an]), torch.cat([bo, bn])
    _, state, _ = model(a, b)
    sv = torch.linalg.svdvals(state.float())
    p = sv / (sv.sum(dim=-1, keepdim=True) + 1e-9)
    return {"live_directions": (sv > 0.01 * sv[:, :1]).sum(dim=-1).float().mean().item(),
            "entropy_rank": (-(p * (p + 1e-9).log()).sum(dim=-1)).exp().mean().item()}


def _snapshot(model) -> dict:
    d = {"old_r2": _r2(model, sample_old),
         "new_r2": _r2(model, sample_new),
         "ablation_a_gate_0_r2": _r2(model, sample_old, force_a_gate=0.0)}
    d.update(_breadth(model))
    return d


def _train(model, params_main, params_aux, opt_name: str, steps: int,
           batch: int, sampler, muon_lr: float, adam_lr: float):
    main = (SingleDeviceMuon(params_main, lr=muon_lr) if opt_name == "muon"
            else torch.optim.AdamW(params_main, lr=adam_lr))
    aux = torch.optim.AdamW(params_aux, lr=adam_lr) if params_aux else None
    for _ in range(steps):
        a, b, y = sampler(batch)
        loss = F.mse_loss(model(a, b)[0], y)
        main.zero_grad()
        if aux:
            aux.zero_grad()
        loss.backward()
        main.step()
        if aux:
            aux.step()
    return model


def merge_adapters(model: nn.Module, adapters: list[LoRALinear]) -> None:
    """Fold each adapter into its frozen base and put the plain Linear back —
    the toy equivalent of the pipeline's `--merge 1` on save. After this the
    model has no adapter at all, which is exactly the state Phase 1.5's
    full-FT actually started from (`...step500-merged.pth`)."""
    for i, mod in enumerate(model.net):
        if isinstance(mod, LoRALinear):
            with torch.no_grad():
                mod.base.weight.add_(mod.delta_w())
            model.net[i] = mod.base
    for p in model.parameters():
        p.requires_grad_(True)


def run_seed(seed: int, *, head_size: int, n_steps: int, pretrain_opt: str,
             pretrain_steps: int, stage1_steps: int, stage2_steps: int,
             batch: int, muon_lr: float, adam_lr: float, lora_r: int,
             lora_alpha: float, replay: float, arms: list[str]) -> dict | None:
    finetune = make_finetune_sampler(replay)
    # --- stage 0: pretrain the OLD skill only, with the optimizer the real
    # base was built by. The new region is never shown here. ---
    torch.manual_seed(seed)
    base = FrozenFinalReadoutController(head_size, n_steps)
    mp, ap = _split_muon_adam_params(base)
    _train(base, mp, ap, pretrain_opt, pretrain_steps, batch, sample_old,
           muon_lr, adam_lr)
    s0 = _snapshot(base)
    if s0["old_r2"] <= CONVERGED_ID_R2:
        progress(f"[seed {seed}] stage-0 base did not learn the old skill "
                 f"(old_r2={s0['old_r2']:+.4f}) — seed skipped")
        return None
    progress(f"[seed {seed} stage0     ] old={s0['old_r2']:+.4f} "
             f"new={s0['new_r2']:+.4f} (must be poor — the skill is unseen)  "
             f"live={s0['live_directions']:.2f}")

    # --- stage 1: Adam on a LoRA adapter over the frozen base (what worked) ---
    torch.manual_seed(seed + 10_000)
    stage1 = copy.deepcopy(base)
    adapters = lorafy(stage1, lora_r, lora_alpha)
    lora_params = [p for ad in adapters for p in (ad.lora_A, ad.lora_B)]
    _train(stage1, lora_params, [], "adam", stage1_steps, batch, finetune,
           muon_lr, adam_lr)
    s1 = _snapshot(stage1)
    s1["d_live_vs_stage0"] = s1["live_directions"] - s0["live_directions"]
    s1["d_old_vs_stage0"] = s1["old_r2"] - s0["old_r2"]
    s1["d_new_vs_stage0"] = s1["new_r2"] - s0["new_r2"]
    progress(f"[seed {seed} stage1 LoRA] old={s1['old_r2']:+.4f} "
             f"(Δ{s1['d_old_vs_stage0']:+.4f})  new={s1['new_r2']:+.4f} "
             f"(Δ{s1['d_new_vs_stage0']:+.4f})  "
             f"live={s1['live_directions']:.2f} (Δ{s1['d_live_vs_stage0']:+.2f})")

    out = {"seed": seed, "stage0": s0, "stage1": s1, "stage2": {}}

    for arm in arms:
        direct = arm.startswith("direct_")
        opt_name = "muon" if arm.endswith("muon") else "adam"
        # `direct_*` skips LoRA entirely: full-FT straight from the pretrained
        # base onto the new skill. This is the arm that answers whether the
        # rank-r constraint is doing anything useful at all, or whether it only
        # forces the new skill to be written by redistributing weights the old
        # skill was using.
        m = copy.deepcopy(base if direct else stage1)
        ref = s0 if direct else s1
        if direct:
            for p in m.parameters():
                p.requires_grad_(True)
            main, aux = _split_muon_adam_params(m)
        else:
            ads = [mod for mod in m.net if isinstance(mod, LoRALinear)]
            if arm.startswith("full"):
                merge_adapters(m, ads)
                main, aux = _split_muon_adam_params(m)
            else:
                main = [p for ad in ads for p in (ad.lora_A, ad.lora_B)]
                aux = []
        torch.manual_seed(seed + 20_000)
        # Matched budget: a direct arm gets stage1+stage2 steps, because it is
        # doing both stages' work in one. Without this the comparison would be
        # "LoRA had twice the training", not "LoRA versus no LoRA".
        steps = (stage1_steps + stage2_steps) if direct else stage2_steps
        _train(m, main, aux, opt_name, steps, batch, finetune, muon_lr, adam_lr)
        s2 = _snapshot(m)
        s2["reference_stage"] = "stage0" if direct else "stage1"
        s2["d_old"] = s2["old_r2"] - ref["old_r2"]
        s2["d_new"] = s2["new_r2"] - ref["new_r2"]
        s2["d_live"] = s2["live_directions"] - ref["live_directions"]
        s2["d_old_vs_stage0"] = s2["old_r2"] - s0["old_r2"]
        s2["d_live_vs_stage0"] = s2["live_directions"] - s0["live_directions"]
        out["stage2"][arm] = s2
        progress(f"[seed {seed} {arm:16s}] old={s2['old_r2']:+.4f} "
                 f"(Δvs0 {s2['d_old_vs_stage0']:+.4f})  new={s2['new_r2']:+.4f}  "
                 f"live={s2['live_directions']:.2f} "
                 f"(Δvs0 {s2['d_live_vs_stage0']:+.2f})")
    return out


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seeds", type=int, default=5)
    ap_.add_argument("--head-size", type=int, default=16)
    ap_.add_argument("--steps", type=int, default=8)
    ap_.add_argument("--pretrain-opt", default="muon", choices=("muon", "adam"),
                     help="optimizer for stage 0. Default muon, because the "
                          "real base (G1i) was pretrained with Muon — the "
                          "earlier toy switch arm started from Adam and so did "
                          "not reproduce this lineage.")
    ap_.add_argument("--pretrain-steps", type=int, default=4000)
    ap_.add_argument("--stage1-steps", type=int, default=1500)
    ap_.add_argument("--stage2-steps", type=int, default=1500)
    ap_.add_argument("--batch-size", type=int, default=64)
    ap_.add_argument("--muon-lr", type=float, default=0.02)
    ap_.add_argument("--adam-lr", type=float, default=3e-3)
    ap_.add_argument("--lora-r", type=int, default=8)
    ap_.add_argument("--lora-alpha", type=float, default=16.0)
    ap_.add_argument("--arms", default=",".join(STAGE2_ARMS))
    ap_.add_argument("--replay", type=float, default=0.25,
                     help="fraction of finetune batches drawn from the OLD "
                          "skill. 0 reproduces pure catastrophic forgetting, "
                          "which swamps every other effect; 0.25 is the "
                          "default as the faithful analogue of a finetune "
                          "corpus that still contains ordinary data.")
    ap_.add_argument("--max-seed-attempts", type=int, default=3)
    ap_.add_argument("--threads", type=int, default=None)
    ap_.add_argument("--out", type=Path, default=None)
    ap_.add_argument("--from-result", type=Path, default=None,
                     help="re-derive the tables and the verdict from a stored "
                          "result's per-seed records instead of training. The "
                          "first full run's verdict contradicted its own table; "
                          "the fix was in the verdict code, so the measurements "
                          "are still good and re-running them would only burn "
                          "CPU to reproduce numbers that are already on disk.")
    args = ap_.parse_args()
    progress(f"[lineage_probe] torch threads = {limit_threads(args.threads)}")

    arms = [a for a in args.arms.split(",") if a]
    if args.from_result is not None:
        prior = json.loads(args.from_result.read_text())
        results = prior["seeds"]
        arms = sorted({a for r in results for a in r["stage2"]})
        progress(f"re-deriving from {args.from_result} — {len(results)} stored "
                 f"seed(s), arms {arms}")
    else:
        results, seed, limit = [], 0, args.seeds * args.max_seed_attempts
        while len(results) < args.seeds and seed < limit:
            r = run_seed(seed, head_size=args.head_size, n_steps=args.steps,
                         pretrain_opt=args.pretrain_opt,
                         pretrain_steps=args.pretrain_steps,
                         stage1_steps=args.stage1_steps,
                         stage2_steps=args.stage2_steps, batch=args.batch_size,
                         muon_lr=args.muon_lr, adam_lr=args.adam_lr,
                         lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                         replay=args.replay, arms=arms)
            if r is not None:
                results.append(r)
            seed += 1
    if not results:
        progress("no seed produced a usable stage-0 base")
        return 1

    progress(f"\n=== over {len(results)} seed(s) with a converged stage-0 base ===")

    def agg(path: list, key: str) -> tuple[float, float]:
        vals = []
        for r in results:
            node = r
            for p in path:
                node = node[p]
            vals.append(node[key])
        return sum(vals) / len(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

    summary = {"stage0": {}, "stage1": {}, "stage2": {}}
    for k in ("old_r2", "new_r2", "live_directions", "entropy_rank"):
        for st_ in ("stage0", "stage1"):
            m_, sd = agg([st_], k)
            summary[st_][k] = {"mean": m_, "std": sd}
    s0o, s0n = summary["stage0"]["old_r2"]["mean"], summary["stage0"]["new_r2"]["mean"]
    s0l = summary["stage0"]["live_directions"]["mean"]
    s1o, s1n = summary["stage1"]["old_r2"]["mean"], summary["stage1"]["new_r2"]["mean"]
    s1l = summary["stage1"]["live_directions"]["mean"]
    progress(f"  stage0 (pretrain {args.pretrain_opt}, old skill only)  "
             f"old={s0o:+.4f}  new={s0n:+.4f}  live={s0l:.2f}")
    progress(f"  stage1 (adam on LoRA, learns the NEW skill)  "
             f"old={s1o:+.4f} (Δ{s1o - s0o:+.4f})  new={s1n:+.4f} "
             f"(Δ{s1n - s0n:+.4f})  live={s1l:.2f} (Δ{s1l - s0l:+.2f})")

    for arm in arms:
        line = {}
        for k in ("old_r2", "new_r2", "live_directions", "entropy_rank",
                  "ablation_a_gate_0_r2", "d_old_vs_stage0", "d_live_vs_stage0"):
            m_, sd = agg(["stage2", arm], k)
            line[k] = {"mean": m_, "std": sd}
        summary["stage2"][arm] = line
        progress(f"  {arm:16s} old={line['old_r2']['mean']:+.4f}"
                 f"±{line['old_r2']['std']:.4f} "
                 f"(Δvs0 {line['d_old_vs_stage0']['mean']:+.4f})  "
                 f"new={line['new_r2']['mean']:+.4f}  "
                 f"live={line['live_directions']['mean']:.2f} "
                 f"(Δvs0 {line['d_live_vs_stage0']['mean']:+.2f})")

    def g(arm: str, k: str) -> float:
        return summary["stage2"][arm][k]["mean"] if arm in summary["stage2"] else float("nan")

    # Question 1 (the one this file was built for): optimizer, or LoRA->full?
    fa, fm = g("full_adam", "d_old_vs_stage0"), g("full_muon", "d_old_vs_stage0")
    lm = g("lora_muon", "d_old_vs_stage0")
    COLLAPSE = -0.10

    # Question 2 (added 2026-09-14): does the rank-r constraint EARN its place?
    # The claim it is being tested against: LoRA is too narrow, so a new skill
    # cannot be written into fresh capacity and gets written by redistributing
    # weights the old skill was using — damaging rather than adding. The
    # signature of redistribution is a new skill acquired with NO growth in live
    # directions and a measurable cost to the old one; the signature of added
    # capacity is the opposite.
    lora_new, lora_old = s1n - s0n, s1o - s0o
    lora_dlive = s1l - s0l
    da = g("direct_full_adam", "new_r2")
    da_old = g("direct_full_adam", "d_old_vs_stage0")
    da_live = g("direct_full_adam", "d_live_vs_stage0")

    # Q1 compares the two FACTORS as groups, not one hand-picked pair.
    #
    # The first full run's verdict compared only full_adam against full_muon and
    # printed "only full_muon collapses -> it is specific to full-FT after a
    # merged adapter" — while lora_muon, sitting in its own table, was WORSE
    # than full_muon (-0.358 vs -0.244). Every Muon arm was worse than every
    # Adam arm, so the factor that separated the table was the optimizer and
    # the printed conclusion contradicted the data it was printed from.
    adam_arms = [a for a in arms if a.endswith("adam")]
    muon_arms = [a for a in arms if a.endswith("muon")]
    lora_arms = [a for a in arms if a.startswith("lora")]
    full_arms = [a for a in arms if "full" in a]

    def worst(group: list) -> float:
        vals = [g(a, "d_old_vs_stage0") for a in group]
        vals = [v for v in vals if v == v]
        return min(vals) if vals else float("nan")

    def best(group: list) -> float:
        vals = [g(a, "d_old_vs_stage0") for a in group]
        vals = [v for v in vals if v == v]
        return max(vals) if vals else float("nan")

    # A factor "separates" the table when its two groups do not overlap at all.
    opt_separates = best(muon_arms) < worst(adam_arms)
    sub_separates = (best(lora_arms) < worst(full_arms)) or (best(full_arms) < worst(lora_arms))

    parts = [
        f"Q1 optimizer vs subspace — old-skill change from the pretrained base, "
        f"worst-to-best: "
        + ", ".join(f"{a} {g(a, 'd_old_vs_stage0'):+.4f}"
                    for a in sorted(arms, key=lambda x: g(x, "d_old_vs_stage0")))
        + f" (collapse threshold {COLLAPSE}). -> "
        + ("the OPTIMIZER separates the table: every Muon arm is worse than "
           f"every Adam arm (best Muon {best(muon_arms):+.4f} < worst Adam "
           f"{worst(adam_arms):+.4f}), across both subspace choices"
           if opt_separates else
           "the SUBSPACE separates the table, not the optimizer"
           if sub_separates else
           "neither factor separates the table cleanly — the arms interleave, "
           "so no single-factor attribution is supported")
    ]
    if da == da:
        parts.append(
            f"Q2 does LoRA earn its constraint — LoRA stage1 learned the new "
            f"skill {lora_new:+.4f} while adding {lora_dlive:+.2f} live "
            f"directions and changing the old skill by {lora_old:+.4f}; "
            f"direct full-FT (no LoRA, matched budget) reached new={da:+.4f} "
            f"with {da_live:+.2f} live directions and old {da_old:+.4f}. -> "
            + ("LoRA NARROWS the state while acquiring the new skill — worse "
               "than redistribution: fewer directions carry more skills"
               if lora_dlive <= -0.3 else
               "LoRA acquires the new skill with NO growth in capacity — the "
               "redistribution reading"
               if abs(lora_dlive) < 0.3 else
               "LoRA adds capacity while learning")
            + (f"; dropping LoRA retains the old skill better "
               f"({da_old:+.4f} vs {lora_old:+.4f}) at comparable new-skill "
               f"quality, so the rank-r constraint is not paying for itself"
               if (da_old > lora_old + 0.05 and da > 0.8) else
               f"; dropping LoRA costs old-skill retention "
               f"({da_old:+.4f} vs {lora_old:+.4f}), so the constraint is "
               f"doing protective work"
               if da_old < lora_old - 0.05 else
               "; dropping LoRA changes old-skill retention little either way")
        )
    verdict = "\n".join(parts)
    progress(f"\n{verdict}")

    if args.out is not None:
        save_result(
            args.out,
            {"seeds": results, "summary": summary, "verdict": verdict,
             "collapse_threshold": COLLAPSE,
             "lora_stage_delta_live": lora_dlive,
             "config": {k: (str(v) if isinstance(v, Path) else v)
                               for k, v in vars(args).items()}},
            experiment="lineage_toy", hypothesis=["H26", "H25"],
            summary={
                "old-skill change from the pretrained base, per arm":
                    "; ".join(f"{a} {g(a, 'd_old_vs_stage0'):+.4f}" for a in arms),
                "new skill reached, per arm":
                    "; ".join(f"{a} {g(a, 'new_r2'):+.4f}" for a in arms),
                "live directions added by the LoRA stage": f"{lora_dlive:+.2f}",
            },
            script=str(Path(__file__).relative_to(_REPO_ROOT)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
