#!/usr/bin/env python3
"""attractor_depth_probe.py — does Muon push a finetuned model into an attractor less than Adam?

The user's reading of the build/bake question (2026-10-06): finetuning can TEACH
something new or DEEPEN an attractor the model already has — sharpen the dominant
direction so answers look better on a narrow eval while paths close. The N=3
collapse on the step-8/9 LoRA models looks like the second. Muon's update is
orthogonalised (flat spectrum), so by construction it cannot preferentially amplify
the dominant direction; if that matters, Muon is the right optimizer for latent
phases specifically, not just because it fits in RAM.

Known before this probe, both ways:
  for     toy, 10 seeds: Adam's solution leans on one channel (a_gate=0 ablation
          -2.62 +- 2.22), Muon's is spread (-0.30 +- 0.56); real model: the step-9b
          Adam-LoRA delta at L12 att.key aligns with the base's dominant directions,
          z = +4.19 (1 of 6 matrices significant)
  against lineage toy: Muon arms lost MORE of the old skill — at muon lr 0.02, the
          pretraining step later measured to wreck finetuning; and H26's confound
          fired: an Adam lr bump moved the mechanism further than switching
          optimizer (91% vs 25%) — so step size, not geometry, may be the cause
  never   attractor depth itself, at a matched step.

Design, after an outside review of the first version (2026-10-06):
  * the STEP is compared as the induced relative change of the effective weights
    per update, mean ||W_after - W_before|| / ||W_before|| over the hidden matrices
    (for LoRA, W = base + B A * scale) — measured on every update, not the nominal
    lr. Each optimizer gets an lr grid; pairs are compared where the induced step
    AND the accuracy match. Critical for LoRA: production Muon steps lora_B up to
    8.9-17.9x harder than lora_A (muon-rwkv7-finetuning.md), so nominal lr means
    nothing there.
  * LoRA arms: lora_adam, lora_muon_fw (production factor-wise Muon) and
    lora_muon_bal (the §2.4 fix: the contributions ||B dA|| and ||dB A|| equalised
    downward, LoRAMuon mode "muon_balanced").
  * base seeds kept only if the old skill is >= 0.95 before finetuning; summaries
    are mean +- std over seeds, and a "matched" view keeps runs with new >= 0.98.
  * curves over extra blank ticks K (the latent-tick / re-read situation — the
    blank step's controls are input-independent, so repeating it is a fixed affine
    map iterated): old and new skill answered after K ticks, |dS|/|S| per tick,
    content spread retained across examples, live directions and the state's
    spectral entropy (effective rank) and top-1 energy share. "Answer entropy" has
    no meaning in this toy (the answer is a scalar regression); it belongs to the
    real-model version.
  * "was it there early and lost later": d_old at K=2 vs K=32 — the analogue of a
    runaway CoT that had the answer and talked past it.
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.A0_state_probe.micro_wkv import FrozenFinalReadoutController, micro_wkv_step  # noqa: E402
from experiments.A0_state_probe.lora_muon_probe import LoRALinear, LoRAMuon, lorafy  # noqa: E402
from experiments.A0_state_probe.muon_vs_adam_toy import SingleDeviceMuon, _split_muon_adam_params  # noqa: E402
from experiments.A0_state_probe.lineage_probe import (  # noqa: E402
    _r2, _snapshot, _train, make_finetune_sampler, sample_old, sample_new)

TICKS = (0, 1, 2, 4, 8, 16, 32)


# ---------------------------------------------------------------- dynamics
def _controls(model, x, t):
    c = model.step_controls(x, t)
    return c["r"], c["k"], c["v"], -F.softplus(-c["w_raw"]) - 0.5, torch.sigmoid(c["a_logit"])


def _spectral(S):
    sv = torch.linalg.svdvals(S.float())                       # [B, N]
    p = sv / sv.sum(-1, keepdim=True).clamp_min(1e-12)
    erank = (-(p * (p + 1e-12).log()).sum(-1)).exp().mean().item()
    live = (sv > 0.01 * sv[:, :1]).sum(-1).float().mean().item()
    e = sv ** 2
    top1 = (e[:, 0] / e.sum(-1).clamp_min(1e-12)).mean().item()
    return erank, live, top1


@torch.no_grad()
def tick_curve(model, sampler, n: int = 512, full: bool = True) -> dict:
    a, b, y = sampler(n)
    n_steps = model.n_steps
    state = torch.zeros(a.shape[0], model.head_size, model.head_size)
    for t in range(n_steps - 1):
        x = a if t == 0 else torch.zeros_like(a)
        _, state = micro_wkv_step(state, *_controls(model, x, t))
    blank = _controls(model, torch.zeros_like(a), n_steps - 2)
    final = _controls(model, b, n_steps - 1)

    def answer(S):
        out, _ = micro_wkv_step(S, *final)
        return 1 - F.mse_loss(model.readout(out).squeeze(-1), y).item() / y.var().item()

    def spread(S):
        return (S - S.mean(0, keepdim=True)).flatten(1).norm(dim=1).mean().item()

    sp0, r0 = spread(state), answer(state)
    curve, prev = {}, state
    for k in range(0, max(TICKS) + 1):
        if k > 0:
            _, state = micro_wkv_step(state, *blank)
        if k in TICKS:
            row = {"r2": answer(state)}
            row["d_r2"] = row["r2"] - r0
            if full:
                er, live, top1 = _spectral(state)
                row.update({
                    "rel_motion": 0.0 if k == 0 else
                    ((state - prev).flatten(1).norm(dim=1) /
                     state.flatten(1).norm(dim=1).clamp_min(1e-12)).mean().item(),
                    "retained": spread(state) / max(sp0, 1e-12),
                    "erank": er, "live": live, "top1": top1})
            curve[str(k)] = row
        prev = state
    return curve


# ---------------------------------------------------------------- training
def _eff_weights(model) -> list:
    ws = []
    for mod in model.net:
        if isinstance(mod, LoRALinear):
            ws.append((mod.base.weight + mod.delta_w()).detach().clone())
        elif isinstance(mod, nn.Linear):
            ws.append(mod.weight.detach().clone())
    return ws


def train_measured(model, arm: str, lr: float, steps: int, batch: int, sampler,
                   aux_lr: float, lora_r: int, lora_alpha: float,
                   new_thr: float = 0.99, eval_every: int = 20) -> dict:
    """Finetune `model` in place; returns the induced relative step statistics and the
    number of updates until the new skill first reaches `new_thr` R2 (None = never).
    The periodic new-skill read uses a saved/restored RNG so the training stream is
    identical to a run without it (seeds stay comparable with earlier rows)."""
    if arm.startswith("lora"):
        adapters = lorafy(model, lora_r, lora_alpha)
        if arm == "lora_adam":
            opt = torch.optim.AdamW([p for ad in adapters for p in (ad.lora_A, ad.lora_B)], lr=lr)
            opts = [opt]
        else:
            mode = {"lora_muon_fw": "muon_factorwise", "lora_muon_bal": "muon_balanced"}[arm]
            opts = [LoRAMuon(adapters, mode=mode, lr=lr)]
    else:
        for p in model.parameters():
            p.requires_grad_(True)
        main, aux = _split_muon_adam_params(model)
        if arm == "full_muon":
            # aux AdamW group on its own lr (memory: project_noesis_muon_decision)
            opts = [SingleDeviceMuon(main, lr=lr), torch.optim.AdamW(aux, lr=aux_lr)]
        else:
            opts = [torch.optim.AdamW(main + aux, lr=lr)]
    rel = []
    steps_to_new = None
    new_curve = []
    for i in range(steps):
        a, b, y = sampler(batch)
        loss = F.mse_loss(model(a, b)[0], y)
        for o in opts:
            o.zero_grad()
        loss.backward()
        before = _eff_weights(model)
        for o in opts:
            o.step()
        after = _eff_weights(model)
        rel.append(statistics.mean(
            ((wa - wb).norm() / wb.norm().clamp_min(1e-12)).item() for wa, wb in zip(after, before)))
        if (i + 1) % eval_every == 0:
            rng = torch.get_rng_state()
            r2n = _r2(model, sample_new)
            torch.set_rng_state(rng)
            if (i + 1) % (eval_every * 10) == 0:
                new_curve.append([i + 1, round(r2n, 4)])
            if steps_to_new is None and r2n >= new_thr:
                steps_to_new = i + 1
    return {"induced_step_mean": statistics.mean(rel),
            "induced_step_median": statistics.median(rel),
            "induced_step_first100": statistics.mean(rel[:100]),
            "induced_step_first1": rel[0],
            "steps_to_new": steps_to_new, "new_thr": new_thr, "new_curve": new_curve,
            "weight_moved": sum(rel)}


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="First seed. Seeds are independent, so a long run can be split "
                         "across processes (own --out each) and the .partial.jsonl files "
                         "concatenated, then summarised with --aggregate-only.")
    ap.add_argument("--head-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8, help="recurrence length of the toy")
    ap.add_argument("--pretrain-steps", type=int, default=4000)
    ap.add_argument("--finetune-steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--replay", type=float, default=0.25)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--aux-lr", type=float, default=3e-4)
    ap.add_argument("--base-min-old", type=float, default=0.95)
    ap.add_argument("--matched-min-new", type=float, default=0.98)
    ap.add_argument("--arms", default=(
        "full_adam:3e-3,full_adam:1e-3,full_adam:3e-4,"
        "full_muon:0.02,full_muon:0.005,full_muon:0.001,full_muon:0.0002,"
        "lora_adam:3e-3,lora_adam:1e-3,"
        "lora_muon_fw:0.02,lora_muon_fw:0.005,lora_muon_fw:0.001,"
        "lora_muon_bal:0.02,lora_muon_bal:0.005,lora_muon_bal:0.001"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip training; rebuild the summary from the .partial.jsonl.")
    ap.add_argument("--out", type=Path,
                    default=Path("experiments/A0_state_probe/results/attractor_depth.json"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    arms = [(x.split(":")[0], float(x.split(":")[1])) for x in args.arms.split(",")]

    part = args.out.with_suffix(".partial.jsonl")
    done = [json.loads(l) for l in open(part)] if part.exists() else []
    done_keys = {(r["seed"], r["arm"], r["lr"]) for r in done}

    def record(rec):
        with open(part, "a") as f:
            f.write(json.dumps(rec) + "\n")
        done.append(rec)

    for seed in ([] if args.aggregate_only
                 else range(args.seed_start, args.seed_start + args.seeds)):
        torch.manual_seed(seed)
        base = FrozenFinalReadoutController(args.head_size, args.steps)
        mp, ap_ = _split_muon_adam_params(base)
        _train(base, mp, ap_, "muon", args.pretrain_steps, args.batch, sample_old,
               muon_lr=0.02, adam_lr=3e-3)       # Muon pretraining, as the real base
        s0 = _snapshot(base)
        if s0["old_r2"] < args.base_min_old:
            print(f"[seed {seed}] base old_r2 {s0['old_r2']:+.3f} < {args.base_min_old} — skipped",
                  flush=True)
            continue
        if (seed, "base", 0.0) not in done_keys:
            record({"seed": seed, "arm": "base", "lr": 0.0, **s0,
                    "curve_old": tick_curve(base, sample_old),
                    "curve_new": tick_curve(base, sample_new, full=False)})
        for arm, lr in arms:
            if (seed, arm, lr) in done_keys:
                continue
            torch.manual_seed(seed + 20_000)
            m = copy.deepcopy(base)
            step = train_measured(m, arm, lr, args.finetune_steps, args.batch,
                                  make_finetune_sampler(args.replay), args.aux_lr,
                                  args.lora_r, args.lora_alpha)
            s = _snapshot(m)
            rec = {"seed": seed, "arm": arm, "lr": lr, "base_old_r2": s0["old_r2"], **s, **step,
                   "curve_old": tick_curve(m, sample_old),
                   "curve_new": tick_curve(m, sample_new, full=False)}
            record(rec)
            co = rec["curve_old"]
            print(f"[seed {seed} {arm:13s} lr={lr:<7g}] step={step['induced_step_mean']:.2e} "
                  f"old={s['old_r2']:+.3f} new={s['new_r2']:+.3f} | old after K=2/8/32: "
                  f"{co['2']['r2']:+.2f}/{co['8']['r2']:+.2f}/{co['32']['r2']:+.2f} "
                  f"retained@32={co['32']['retained']:.2f} top1@32={co['32']['top1']:.2f}",
                  flush=True)

    # aggregate, mean +- std over seeds, per arm/lr
    def ms(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return [round(statistics.mean(vals), 4),
                round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0, len(vals)]

    agg = {}
    for key in sorted({(r["arm"], r["lr"]) for r in done}):
        rs = [r for r in done if (r["arm"], r["lr"]) == key]
        row = {"old_r2": ms([r["old_r2"] for r in rs]), "new_r2": ms([r["new_r2"] for r in rs]),
               "induced_step": ms([r.get("induced_step_mean") for r in rs])}
        for k in ("0", "2", "8", "32"):
            for f_ in ("r2", "d_r2", "rel_motion", "retained", "erank", "live", "top1"):
                row[f"old@{k}_{f_}"] = ms([r["curve_old"][k].get(f_) for r in rs])
            row[f"new@{k}_r2"] = ms([r["curve_new"][k]["r2"] for r in rs])
        # The losing interval, per seed (user, 2026-10-06): old skill at K=8 minus
        # at K=2 — where Adam-on-LoRA already showed -4.9 in the smoke. If the gap
        # between arms holds on every seed, it is a result, not a hint.
        row["old_K2_to_K8"] = ms([r["curve_old"]["8"]["r2"] - r["curve_old"]["2"]["r2"] for r in rs])
        row["old_K8_to_K32"] = ms([r["curve_old"]["32"]["r2"] - r["curve_old"]["8"]["r2"] for r in rs])
        row["old_K2_to_K8_per_seed"] = [round(r["curve_old"]["8"]["r2"] - r["curve_old"]["2"]["r2"], 3)
                                        for r in rs]
        matched = [r for r in rs if r["new_r2"] >= args.matched_min_new]
        row["matched_n"] = len(matched)
        row["matched_old@32_d_r2"] = ms([r["curve_old"]["32"]["d_r2"] for r in matched])
        agg[f"{key[0]}:{key[1]:g}"] = row
    print(json.dumps({k: {kk: v[kk] for kk in ("induced_step", "old_r2", "new_r2",
                                               "old@2_r2", "old_K2_to_K8", "old@32_r2",
                                               "old@32_retained", "matched_n")}
                      for k, v in agg.items()}, indent=1))
    args.out.write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                    "aggregate": agg, "rows": done}, indent=1))
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
