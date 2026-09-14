#!/usr/bin/env python3
"""lora_muon_probe.py — Muon on LoRA's rectangular factors: is the production
form of it even the algorithm it claims to be, and can the state supply the
metric instead of the weight shape?

Two questions, one substrate, because they turn out to live in the same
thirty lines of `training/rwkv-peft-patches/muon_opt.py`.

────────────────────────────────────────────────────────────────────────────
Question 1 — the rectangular-factor problem.

Production Muon (`_is_muon_param`, muon_opt.py:64) selects any 2D `.weight`
under `.att.`/`.ffn.`, which under PEFT includes the injected `lora_A` and
`lora_B`. Each is then orthogonalized on its own and rescaled by the upstream
aspect correction (muon_opt.py:58):

    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

Measured on this project's real shapes (r=32, d=2560), 2026-09-14:

    lora_A  32x2560   scale 1.000   update RMS 0.0199
    lora_B  2560x32   scale 8.944   update RMS 0.1776

Same adapter, one shared `lr`, and one factor steps ~9x faster than the other
purely because it is tall rather than wide. That coefficient is not wrong
upstream — it normalises a standalone layer weight's per-element RMS to
Adam-like scale. It is blind here because a LoRA factor's per-element scale is
not what reaches the model: the model only ever sees the product, and
`ΔW = s(B δA + δB A)` weights the two factors by `‖B‖` and `‖A‖`, not by their
shapes. PEFT initialises `lora_B` to zeros (peft/tuners/lora/layer.py:235) and
`lora_A` kaiming, so at step 0 the entire update flows through `δB·A` — i.e.
entirely through the factor that is also being stepped 9x too fast.

The second half of the same finding: **factor-wise Muon is itself a breadth
term.** Newton-Schulz drives every singular value of each factor to 1, and the
induced product inherits it — measured at init, `ΔW` came out rank 32 with
σ₃₂/σ₁ = 0.733 and entropy-rank 31.88 of 32, i.e. an almost perfectly flat
full-rank-r update, injected every step by construction. Adam in the same slot
produces a spiky one. That is the same axis `breadth_growth_probe.py` is
testing, arrived at from the opposite direction, and it is a mechanism-level
candidate for the 2.9B fine-tuning collapse: a model whose WKV state carries
13-16 live directions of 64 (`results/rank_recheck*/jlens.json`) being handed a
flat width-32 update on every step.

Arms `muon_factorwise` (production, as-is), `muon_balanced` (aspect scale
replaced by one that equalises the two factors' *induced* contribution) and
`muon_product` (orthogonalise the induced ΔW itself, then backprop that through
the factorisation) separate "Muon is wrong for finetuning" from "this
particular application of Muon to a factorised weight is wrong".

────────────────────────────────────────────────────────────────────────────
Question 2 — can the state define the metric?

Muon is steepest descent under the spectral norm on W: a geometry fixed in
advance by the weight's shape. The standing question in this project is
whether a recurrent-state model should instead take its metric from the state
it actually maintains — move along the manifold rather than straight through
it.

`muon_state` is the cheapest honest version of that: a **per-tensor diagonal
approximation** of the state-induced metric, and it is labelled as such rather
than as a Riemannian anything. Every K steps the probe applies each parameter's
own orthogonalized update at small ε, measures the resulting displacement of
the final WKV state on a fixed batch, reverts, and rescales that parameter's
step so all parameters move the STATE by an equal amount — instead of moving
in WEIGHT space by an equal amount, which is what Muon's fixed norm does.

Note this subsumes Question 1 for free: the shape-derived 8.944 is replaced by
a measured quantity, so the A/B asymmetry either survives measurement or it
does not. That is the point — it is the difference between a geometry asserted
from the tensor's dimensions and one read off the model's behaviour.

────────────────────────────────────────────────────────────────────────────
Protocol — the collapse analogue.

Pretrain the base with Adam on `a*b` over [-3,3] (the toy's stand-in for an
Adam-pretrained checkpoint, which is what G1i is), freeze it, attach LoRA,
then fine-tune on a NARROW slice [-1,1]. Three numbers come out:

    adapt_r2   on [-1,1]   did it learn what it was fine-tuned on
    retain_r2  on [-3,3]   did the pretrained ability survive   <- the collapse
    ood_r2     on ±[3,5]   did anything generalise

`retain_r2` is the one that matters. Every training diagnostic stayed healthy
through the real 2.9B collapse; what died was general ability, visible only on
data the fine-tune never mentioned. This is the cheap, repeatable version of
that measurement, and unlike the real one it does not need the shelved VM.
"""
from __future__ import annotations

import argparse
import math
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
from experiments.A0_state_probe.muon_vs_adam_toy import zeropower_via_newtonschulz5
from experiments._common.convergence import CONVERGED_ID_R2 as _CONVERGED_ID_R2
from experiments._common.results import save_result
from experiments._common.runtime import limit_threads, progress

ARMS = ("adam", "muon_factorwise", "muon_balanced", "muon_product", "muon_state")

# Shared with every other toy probe — see experiments/_common/convergence.py.
# Here it gates the PRETRAINED BASE: "did the finetune destroy what it knew" is
# not a question that can be asked of a base that never knew it.
CONVERGED_ID_R2 = _CONVERGED_ID_R2


class LoRALinear(nn.Module):
    """Frozen base `nn.Linear` plus a rank-r adapter, matching PEFT's
    conventions exactly where they matter to this probe: `lora_A` is
    [r, in_features] and kaiming-initialised, `lora_B` is [out_features, r]
    and initialised to ZERO (peft/tuners/lora/layer.py:235/269), and the
    adapter output is scaled by alpha/r. The zero-init is load-bearing here,
    not a detail — it is why the first steps flow entirely through `lora_B`."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling

    def delta_w(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling


def lorafy(model: nn.Module, r: int, alpha: float) -> list[LoRALinear]:
    """Attach adapters to exactly the tensors production Muon would claim —
    the hidden linear weights of `net`, the toy's analogue of `.att.`/`.ffn.`
    projections. Everything else (step_embed, readout, the frozen final
    r/w vectors) stays frozen: this is a LoRA finetune, not a full one."""
    for p in model.parameters():
        p.requires_grad_(False)
    adapters = []
    for i, mod in enumerate(model.net):
        if isinstance(mod, nn.Linear):
            wrapped = LoRALinear(mod, r, alpha)
            model.net[i] = wrapped
            adapters.append(wrapped)
    return adapters


def _ns(x: torch.Tensor) -> torch.Tensor:
    return zeropower_via_newtonschulz5(x, steps=5).to(x.dtype)


class LoRAMuon:
    """One optimizer, five modes, so every arm differs from the production
    one by a single named change rather than by a reimplementation.

    `adam` defers entirely to AdamW. The rest keep Muon's one momentum buffer
    per tensor and differ only in what they do with the orthogonalized
    direction:

      muon_factorwise  upstream aspect scale, per factor — production today
      muon_balanced    aspect scale replaced by a per-factor scale that
                       equalises the two factors' contribution to ΔW
      muon_product     orthogonalise the INDUCED ΔW, push it back through the
                       factorisation, then set the step size in W-space
      muon_state       muon_factorwise, with the shape-derived scale replaced
                       by a measured state-displacement scale (see module doc)
    """

    def __init__(self, adapters: list[LoRALinear], mode: str, lr: float,
                 momentum: float = 0.95, max_scale_ratio: float = 16.0):
        assert mode in ARMS
        self.adapters = adapters
        self.mode = mode
        self.lr = lr
        self.momentum = momentum
        self.bufs: dict[int, torch.Tensor] = {}
        # per-tensor multiplier used by muon_state; 1.0 until first measured
        self.state_scale: dict[int, float] = {}
        # Bound on how far the measured metric may move a tensor's step away
        # from the median. Needed because at LoRA init `lora_B` is zero, so
        # `lora_A`'s gradient is exactly zero and its measured state
        # sensitivity is ~0 — an unbounded `median / sensitivity` sent the
        # first smoke run to 1e27. The default is chosen to be WIDER than the
        # 8.944 the shape-based rule asserts for these shapes, so the
        # measurement can confirm or contradict that number rather than be
        # clipped into agreeing with it.
        self.max_scale_ratio = max_scale_ratio

    def params(self) -> list[torch.Tensor]:
        out = []
        for ad in self.adapters:
            out += [ad.lora_A, ad.lora_B]
        return out

    def zero_grad(self) -> None:
        for p in self.params():
            p.grad = None

    def _momentum(self, p: torch.Tensor) -> torch.Tensor:
        buf = self.bufs.get(id(p))
        if buf is None:
            buf = torch.zeros_like(p)
            self.bufs[id(p)] = buf
        buf.lerp_(p.grad, 1 - self.momentum)
        return p.grad.lerp(buf, self.momentum)      # nesterov, as upstream

    @torch.no_grad()
    def directions(self, ad: LoRALinear, gA: torch.Tensor, gB: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """The per-mode update directions for one adapter, before `lr`.

        Split out from `step()` so the invariants each mode is supposed to
        satisfy are testable directly (test_lora_muon.py) rather than only
        observable through a training run: `muon_balanced` must equalise the
        two factors' contributions, `muon_product` must hand back a step whose
        INDUCED ΔW has unit spectral norm, `muon_factorwise` must reproduce
        the production asymmetry rather than quietly fix it."""
        if self.mode == "muon_product":
            # Direction this gradient pair induces in W-space, then
            # orthogonalized THERE — which is where Muon's spectral-norm
            # claim actually lives — and pushed back through the product
            # by the same chain rule backprop would use. At LoRA init
            # B=0 gives dA=0, which is correct rather than a bug: with
            # B=0, A cannot affect the output at all.
            D = _ns(ad.scaling * (ad.lora_B @ gA + gB @ ad.lora_A))
            dA = ad.lora_B.T @ D
            dB = D @ ad.lora_A.T
            # Step size fixed in W-space, not per factor — the whole
            # point of asking the question in W-space to begin with.
            induced = ad.scaling * (ad.lora_B @ dA + dB @ ad.lora_A)
            nrm = torch.linalg.matrix_norm(induced.float(), ord=2).clamp_min(1e-8)
            dA, dB = dA / nrm, dB / nrm
        else:
            dA, dB = _ns(gA), _ns(gB)
            if self.mode == "muon_factorwise":
                dA *= max(1, dA.size(-2) / dA.size(-1)) ** 0.5
                dB *= max(1, dB.size(-2) / dB.size(-1)) ** 0.5
            elif self.mode == "muon_balanced":
                # Equalise what each factor CONTRIBUTES to ΔW instead of
                # what each factor looks like. ‖B δA‖_F and ‖δB A‖_F are
                # the two halves of the induced update; scale each to the
                # geometric mean of the pair so neither dominates by
                # accident of orientation.
                cA = (ad.lora_B @ dA).float().norm().clamp_min(1e-8)
                cB = (dB @ ad.lora_A).float().norm().clamp_min(1e-8)
                tgt = (cA * cB).sqrt()
                dA, dB = dA * (tgt / cA), dB * (tgt / cB)
            elif self.mode == "muon_state":
                dA *= self.state_scale.get(id(ad.lora_A), 1.0)
                dB *= self.state_scale.get(id(ad.lora_B), 1.0)
        return dA, dB

    @torch.no_grad()
    def step(self) -> None:
        for ad in self.adapters:
            if ad.lora_A.grad is None or ad.lora_B.grad is None:
                continue
            gA, gB = self._momentum(ad.lora_A), self._momentum(ad.lora_B)
            dA, dB = self.directions(ad, gA, gB)
            ad.lora_A.add_(dA.to(ad.lora_A.dtype), alpha=-self.lr)
            ad.lora_B.add_(dB.to(ad.lora_B.dtype), alpha=-self.lr)

    @torch.no_grad()
    def recalibrate_state_metric(self, model, probe_batch, eps: float = 1e-3) -> dict:
        """Measure, per tensor, how far a unit-norm step along that tensor's
        own orthogonalized direction moves the FINAL WKV state, then set the
        per-tensor scale so every tensor moves the state equally.

        This is the diagonal approximation named in the module docstring: it
        gives each tensor its own step size from measured state-sensitivity,
        and says nothing about directions WITHIN a tensor. The full metric
        (whitening the update by the state's second moment) is the version
        this is a first, cheap rung below — worth building only if this one
        shows the axis matters at all."""
        a0, b0 = probe_batch
        _, s_ref, _ = model(a0, b0)
        sens: dict[int, float] = {}
        for ad in self.adapters:
            for p in (ad.lora_A, ad.lora_B):
                # A tensor with no gradient signal yet (notably `lora_A` while
                # `lora_B` is still at its zero init) has no direction to probe
                # along; leave whatever scale it already has rather than
                # measuring noise and dividing by it.
                if p.grad is None or float(p.grad.float().norm()) < 1e-12:
                    continue
                d = _ns(p.grad)
                dn = float(d.float().norm())
                if dn < 1e-8:
                    continue
                d = d / dn
                p.add_(d.to(p.dtype), alpha=eps)
                _, s_pert, _ = model(a0, b0)
                p.sub_(d.to(p.dtype), alpha=eps)
                sens[id(p)] = float((s_pert - s_ref).norm() / eps)
        if not sens:
            return {}
        # Equalise onto the median so the overall step size stays in the same
        # range as the other arms — this arm is about the RELATIVE allocation
        # across tensors, and rescaling everything would confound it with lr.
        med = statistics.median(sens.values())
        lo, hi = 1.0 / self.max_scale_ratio, self.max_scale_ratio
        for k, v in sens.items():
            self.state_scale[k] = min(hi, max(lo, med / max(v, 1e-12)))
        return sens


def _r2(model, target_fn, lo: float, hi: float, n: int, signed_ood: bool = False,
        force_a_gate: float | None = None) -> float:
    with torch.no_grad():
        if signed_ood:
            s1 = (torch.randint(0, 2, (n,)) * 2 - 1).float()
            s2 = (torch.randint(0, 2, (n,)) * 2 - 1).float()
            a = s1 * torch.empty(n).uniform_(lo, hi)
            b = s2 * torch.empty(n).uniform_(lo, hi)
        else:
            a = torch.empty(n).uniform_(lo, hi)
            b = torch.empty(n).uniform_(lo, hi)
        y = target_fn(a, b)
        y_hat = model(a, b, force_a_gate=force_a_gate)[0]
        return 1.0 - F.mse_loss(y_hat, y).item() / y.var().item()


@torch.no_grad()
def _state_breadth(model, n: int = 512) -> dict:
    a = torch.empty(n).uniform_(-3.0, 3.0)
    b = torch.empty(n).uniform_(-3.0, 3.0)
    _, state, _ = model(a, b)
    sv = torch.linalg.svdvals(state.float())
    live = (sv > 0.01 * sv[:, :1]).sum(dim=-1).float().mean().item()
    p = sv / (sv.sum(dim=-1, keepdim=True) + 1e-9)
    ent = (-(p * (p + 1e-9).log()).sum(dim=-1)).exp().mean().item()
    return {"live_directions": live, "entropy_rank": ent}


@torch.no_grad()
def _adapter_spectrum(adapters: list[LoRALinear]) -> dict:
    """What the adapter actually injects into the frozen weights. The claim
    under test is that factor-wise Newton-Schulz forces this flat by
    construction, regardless of what the task wanted."""
    ranks, flats = [], []
    for ad in adapters:
        dw = ad.delta_w().float()
        sv = torch.linalg.svdvals(dw)
        if float(sv[0]) < 1e-12:
            continue
        p = sv / sv.sum()
        ranks.append(float((-(p * (p + 1e-12).log()).sum()).exp()))
        flats.append(float(sv[min(ad.r, len(sv)) - 1] / sv[0]))
    return {"delta_w_entropy_rank": (sum(ranks) / len(ranks)) if ranks else 0.0,
            "delta_w_sigma_r_over_1": (sum(flats) / len(flats)) if flats else 0.0}


def pretrain_base(seed: int, head_size: int, n_steps: int, n_train_steps: int,
                  batch_size: int, adam_lr: float):
    """Adam-pretrained base, because that is what the real checkpoint is."""
    torch.manual_seed(seed)
    model = FrozenFinalReadoutController(head_size, n_steps)
    target = lambda x, y: x * y  # noqa: E731
    opt = torch.optim.AdamW(model.parameters(), lr=adam_lr)
    for _ in range(n_train_steps):
        a = torch.empty(batch_size).uniform_(-3.0, 3.0)
        b = torch.empty(batch_size).uniform_(-3.0, 3.0)
        loss = F.mse_loss(model(a, b)[0], target(a, b))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, target


def run_arm(arm: str, seed: int, *, head_size: int, n_steps: int,
            pretrain_steps: int, finetune_steps: int, batch_size: int,
            adam_lr: float, muon_lr: float, lora_r: int, lora_alpha: float,
            narrow: float, recal_every: int, base_state: dict | None = None) -> dict:
    if base_state is None:
        model, target = pretrain_base(seed, head_size, n_steps, pretrain_steps,
                                      batch_size, adam_lr)
    else:
        # Every arm must start from the BYTE-IDENTICAL pretrained base, not
        # from an independently re-run pretrain that happens to share a seed.
        # Re-running was also 5x the compute for a guarantee that rested on
        # RNG determinism holding across arms.
        torch.manual_seed(seed)
        model = FrozenFinalReadoutController(head_size, n_steps)
        model.load_state_dict(base_state)
        target = lambda x, y: x * y  # noqa: E731
    pre = {"pre_retain_r2": _r2(model, target, -3.0, 3.0, 1000),
           "pre_ood_r2": _r2(model, target, 3.0, 5.0, 1000, signed_ood=True)}
    pre.update({f"pre_{k}": v for k, v in _state_breadth(model).items()})

    torch.manual_seed(seed + 10_000)
    adapters = lorafy(model, lora_r, lora_alpha)

    if arm == "adam":
        params = [p for ad in adapters for p in (ad.lora_A, ad.lora_B)]
        opt = torch.optim.AdamW(params, lr=adam_lr)
        muon = None
    else:
        muon = LoRAMuon(adapters, arm, lr=muon_lr)
        opt = None

    probe_batch = (torch.empty(256).uniform_(-3.0, 3.0),
                   torch.empty(256).uniform_(-3.0, 3.0))
    sens_log = None
    for step in range(finetune_steps):
        a = torch.empty(batch_size).uniform_(-narrow, narrow)
        b = torch.empty(batch_size).uniform_(-narrow, narrow)
        loss = F.mse_loss(model(a, b)[0], target(a, b))
        (opt or muon).zero_grad()
        loss.backward()
        if arm == "muon_state" and step % recal_every == 0:
            s = muon.recalibrate_state_metric(model, probe_batch)
            if sens_log is None:
                sens_log = {"first_measured_sensitivities": sorted(s.values())}
        (opt or muon).step()

    out = {
        "arm": arm, "seed": seed, **pre,
        "adapt_r2": _r2(model, target, -narrow, narrow, 1000),
        "retain_r2": _r2(model, target, -3.0, 3.0, 1000),
        "ood_r2": _r2(model, target, 3.0, 5.0, 1000, signed_ood=True),
        "ablation_a_gate_0_r2": _r2(model, target, -3.0, 3.0, 1000, force_a_gate=0.0),
    }
    out.update(_state_breadth(model))
    out.update(_adapter_spectrum(adapters))
    out["retain_drop"] = out["retain_r2"] - pre["pre_retain_r2"]
    if sens_log:
        out.update(sens_log)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--head-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--pretrain-steps", type=int, default=4000)
    ap.add_argument("--finetune-steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--adam-lr", type=float, default=3e-3)
    ap.add_argument("--muon-lr", type=float, default=0.02)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--narrow", type=float, default=1.0,
                    help="fine-tune operand range [-narrow, narrow]; the "
                         "pretrain range is [-3,3], so this is the 'narrow "
                         "domain corpus' the collapse analogue needs.")
    ap.add_argument("--recal-every", type=int, default=50)
    ap.add_argument("--max-seed-attempts", type=int, default=3,
                    help="how many seeds to try per requested seed before "
                         "giving up on finding a base that pretrains.")
    ap.add_argument("--arms", default=",".join(ARMS))
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

    arms = [a for a in args.arms.split(",") if a]

    # Keep pretraining until `--seeds` bases actually LEARNED the pretrain
    # task. This is not defensive padding: the pretrain here is Adam at the
    # same recurrence length where breadth_growth_probe.py found plain Adam
    # failing outright on some seeds, and the first full run of this file
    # duly produced 2 dead bases out of 5 (pre_retain_r2 = 0.0000, 0.0007).
    # Every arm shares the cached base per seed, so a dead base silently
    # dragged all five arm means to ~0.60 and the table had to be recomputed
    # by hand. A base that never learned the task cannot show whether a
    # finetune destroyed what it knew.
    bases: dict[int, dict] = {}
    skipped: list[tuple[int, float]] = []
    seed = 0
    limit = args.seeds * args.max_seed_attempts
    while len(bases) < args.seeds and seed < limit:
        m, _ = pretrain_base(seed, args.head_size, args.steps,
                             args.pretrain_steps, args.batch_size, args.adam_lr)
        r2 = _r2(m, lambda x, y: x * y, -3.0, 3.0, 1000)
        if r2 > CONVERGED_ID_R2:
            bases[seed] = {k: v.clone() for k, v in m.state_dict().items()}
            progress(f"[pretrain seed={seed}] retain_r2={r2:+.4f}")
        else:
            skipped.append((seed, r2))
            progress(f"[pretrain seed={seed}] retain_r2={r2:+.4f}  "
                  f"[BASE DID NOT LEARN — seed skipped]")
        seed += 1
    if len(bases) < args.seeds:
        print(f"\n[!] only {len(bases)}/{args.seeds} bases converged within "
              f"{limit} attempts — results below rest on fewer seeds than asked")
    if skipped:
        progress(f"[!] {len(skipped)} seed(s) skipped for a non-converged base: "
              + ", ".join(f"{s}({v:+.4f})" for s, v in skipped))

    runs = []
    for arm in arms:
        for seed in sorted(bases):
            r = run_arm(arm, seed, head_size=args.head_size, n_steps=args.steps,
                        pretrain_steps=args.pretrain_steps,
                        finetune_steps=args.finetune_steps,
                        batch_size=args.batch_size, adam_lr=args.adam_lr,
                        muon_lr=args.muon_lr, lora_r=args.lora_r,
                        lora_alpha=args.lora_alpha, narrow=args.narrow,
                        recal_every=args.recal_every, base_state=bases[seed])
            runs.append(r)
            progress(f"[{arm:16s} seed={seed}] adapt={r['adapt_r2']:+.4f} "
                  f"retain={r['retain_r2']:+.4f} (pre {r['pre_retain_r2']:+.4f}, "
                  f"drop {r['retain_drop']:+.4f}) ood={r['ood_r2']:+.4f} "
                  f"live={r['live_directions']:.2f} "
                  f"dW_eRank={r['delta_w_entropy_rank']:.2f}/{args.lora_r}")

    print("\n=== summary over seeds ===")
    keys = ("adapt_r2", "retain_r2", "retain_drop", "ood_r2",
            "ablation_a_gate_0_r2", "live_directions", "entropy_rank",
            "delta_w_entropy_rank", "delta_w_sigma_r_over_1")
    summary = {}
    for arm in arms:
        rs = [r for r in runs if r["arm"] == arm]
        summary[arm] = {k: {"mean": sum(r[k] for r in rs) / len(rs),
                            "std": statistics.pstdev([r[k] for r in rs]) if len(rs) > 1 else 0.0}
                        for k in keys}
        l = summary[arm]
        print(f"  {arm:16s} adapt={l['adapt_r2']['mean']:+.4f}  "
              f"retain={l['retain_r2']['mean']:+.4f}±{l['retain_drop']['std']:.4f} "
              f"(drop {l['retain_drop']['mean']:+.4f})  "
              f"ood={l['ood_r2']['mean']:+.4f}  "
              f"live={l['live_directions']['mean']:5.2f}  "
              f"dW_eRank={l['delta_w_entropy_rank']['mean']:.2f}/{args.lora_r} "
              f"(σr/σ1={l['delta_w_sigma_r_over_1']['mean']:.3f})")

    def g(arm: str, k: str) -> float:
        return summary[arm][k]["mean"] if arm in summary else float("nan")

    lines = []
    if "muon_factorwise" in summary:
        lines.append(
            f"production Muon on LoRA factors: retain {g('muon_factorwise','retain_r2'):+.4f} "
            f"(drop {g('muon_factorwise','retain_drop'):+.4f}), adapter ΔW entropy-rank "
            f"{g('muon_factorwise','delta_w_entropy_rank'):.2f}/{args.lora_r} "
            f"— vs Adam retain {g('adam','retain_r2'):+.4f} "
            f"(drop {g('adam','retain_drop'):+.4f}), ΔW entropy-rank "
            f"{g('adam','delta_w_entropy_rank'):.2f}/{args.lora_r}")
    for alt, label in (("muon_balanced", "equalising the two factors' contribution"),
                       ("muon_product", "orthogonalising the induced ΔW instead"),
                       ("muon_state", "letting the state set the per-tensor metric")):
        if alt in summary and "muon_factorwise" in summary:
            d = g(alt, "retain_r2") - g("muon_factorwise", "retain_r2")
            a = g(alt, "adapt_r2") - g("muon_factorwise", "adapt_r2")
            lines.append(f"{label}: retain {d:+.4f}, adapt {a:+.4f} vs production Muon"
                         + ("  <- recovers what factor-wise Muon loses" if d > 0.05 else ""))
    verdict = "\n".join(lines)
    print(f"\n{verdict}")

    if args.out is not None:
        save_result(
            args.out,
            {"runs": runs, "summary": summary, "verdict": verdict,
             "pretrain_seeds_used": sorted(bases),
             "pretrain_seeds_skipped": [{"seed": s_, "retain_r2": v} for s_, v in skipped],
             "config": vars(args) | {"out": str(args.out)}},
            experiment="lora_muon_toy", hypothesis=["H25", "H26"],
            summary={
                "retain_r2 (pretrained ability after narrow finetune)":
                    "; ".join(f"{a} {g(a,'retain_r2'):+.4f}" for a in arms),
                "adapt_r2 (the finetune's own range)":
                    "; ".join(f"{a} {g(a,'adapt_r2'):+.4f}" for a in arms),
                "adapter ΔW entropy-rank (of r=%d)" % args.lora_r:
                    "; ".join(f"{a} {g(a,'delta_w_entropy_rank'):.2f}" for a in arms),
            },
            script=str(Path(__file__).relative_to(_REPO_ROOT)),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
