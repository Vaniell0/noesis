"""MuonWithAuxAdam — single `torch.optim.Optimizer` combining Muon
(orthogonalized-momentum, hidden 2D matrices) with AdamW (everything
else), for the Lightning-driven RWKV-PEFT trainer.

Why this file exists: `light_rwkv.py::configure_optimizers` already has
an `args.optimizer == 'muon'` branch that calls `MuonWithAuxAdam(...)`,
but that class was never vendored anywhere in this tree — selecting
`--optimizer muon` here has always raised `NameError`. This is the
missing piece, not a new design: same Newton-Schulz update, same
param-selection rule, and the same validated defaults as
`experiments/rl/loader.py::MuonHybrid` (used all day 2026-09-10 on the
newer `train_think_distill.py` pipeline) — re-derived here as one
`torch.optim.Optimizer` subclass (rather than MuonHybrid's own
hand-rolled step()/zero_grad()) specifically because Lightning's
`configure_optimizers` contract expects a real Optimizer: `.param_groups`
for LR-scheduler hookup, checkpointing via `.state_dict()`, and
`.step(closure)` under whatever precision plugin is active.

Muon param selection matches MuonHybrid exactly: `ndim==2`, name ends in
`.weight`, and contains `.att.` or `.ffn.` — this also matches LoRA's
injected `lora_A`/`lora_B` weights when the base model is frozen (PEFT
names them e.g. `...att.key.lora_A.default.weight`), so Muon applies to
the LoRA adapter matrices themselves in a LoRA run, exactly as it did in
today's `train_think_distill.py --lora-r --muon` runs.
"""
from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Copied near-verbatim from github.com/KellerJordan/Muon (muon.py,
    MIT-style research code, no license header in the source file) —
    same copy already used in `experiments/rl/loader.py`, kept
    byte-identical so both pipelines share one validated implementation.
    """
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


def _muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta: float,
                  ns_steps: int, nesterov: bool = True,
                  aspect: bool = True) -> torch.Tensor:
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    if aspect:
        update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def _balanced_pair_scale(lora_A, lora_B, dA, dB):
    """Equalise the two LoRA factors' contributions to the update, downward.

    Same function as `experiments/rl/loader.py::_balanced_pair_scale`, and it
    is duplicated rather than imported on purpose: this file is vendored into
    `rwkvt/` on the training host, where `experiments/` does not exist.

    The aspect rescale above is right for a standalone weight and meaningless
    for one member of a factored pair -- at r=32/d=2560 it gives `lora_A`
    1.000 and `lora_B` 8.944 (17.889 for `ffn.key`), so at `--muon-lr 0.002`
    the B factors are stepped at 0.0179 and 0.0358, at and above the 0.02 on
    record as collapsing this project's 2.9B model.

    Equalising is done to the SMALLER contribution, never to the geometric
    mean. Measured 2026-09-15
    (`experiments/A0_state_probe/results/aspect_dose_toy.json`, arm `lr_mul`):
    putting both factors on the LARGE step, symmetrically, is catastrophic --
    retain_r2 -747 at coefficient 4.0, -9892 at 5.66, far worse than the
    asymmetry it removes. Raising any factor's step is what does the harm.

    Both norms are computed in r x r space via
    ||B dA||_F^2 = <B^T B, dA dA^T> and ||dB A||_F^2 = <dB^T dB, A A^T>, so
    the out x in product is never formed. PEFT zeroes `lora_B`, making
    ||B dA|| exactly 0 on step 0; balancing to zero would freeze the adapter,
    so a vanishing contribution falls back to no rescale.
    """
    a32, b32 = lora_A.float(), lora_B.float()
    dA32, dB32 = dA.float(), dB.float()
    cA = ((b32.T @ b32) * (dA32 @ dA32.T)).sum().clamp_min(0).sqrt()
    cB = ((dB32.T @ dB32) * (a32 @ a32.T)).sum().clamp_min(0).sqrt()
    if not (cA > 1e-12 and cB > 1e-12):
        return 1.0, 1.0
    tgt = torch.minimum(cA, cB)
    return (tgt / cA).item(), (tgt / cB).item()


def _find_lora_pairs(named_params):
    """id(param) -> (partner, role) for PEFT's `lora_A`/`lora_B` factors."""
    by_base = {}
    for name, p in named_params:
        if not _is_muon_param(name, p):
            continue
        for role in ("A", "B"):
            tag = ".lora_%s." % role
            if tag in name:
                head, tail = name.split(tag, 1)
                by_base.setdefault(head + "|" + tail, {})[role] = p
    partner, which = {}, {}
    for pair in by_base.values():
        if "A" in pair and "B" in pair:
            a, b = pair["A"], pair["B"]
            partner[id(a)], which[id(a)] = b, "A"
            partner[id(b)], which[id(b)] = a, "B"
    return partner, which


def _is_muon_param(name: str, p: torch.nn.Parameter) -> bool:
    return p.ndim == 2 and name.endswith(".weight") and (".att." in name or ".ffn." in name)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Two internally-tagged param groups (`use_muon: True/False`), one
    `step()` dispatching Newton-Schulz-orthogonalized momentum to the
    muon group and standard AdamW to the rest — mirrors upstream
    `KellerJordan/Muon`'s `MuonWithAuxAdam` shape (group-tagged single
    optimizer), not `MuonHybrid`'s two-separate-optimizers shape, since
    Lightning expects exactly one `torch.optim.Optimizer` back from
    `configure_optimizers`.

    Constructed from `named_parameters()` directly (not the pre-built
    `optim_groups` in `light_rwkv.py`, which already discards param
    names by the time it groups by `layerwise_lr`/`weight_decay` —
    losing exactly the `.att./.ffn.` information this split needs).
    Momentum warmup (0.85 -> `momentum` over `momentum_warmup_steps`)
    matches MuonHybrid's own tuning choice, carried over unchanged.
    """

    def __init__(self, named_params, lr: float = 0.02, adam_lr: float | None = None,
                 lr_init: float | None = None,
                 momentum: float = 0.95, momentum_start: float = 0.85,
                 momentum_warmup_steps: int = 500, weight_decay: float = 0.0,
                 adam_betas: tuple[float, float] = (0.9, 0.999), adam_eps: float = 1e-8,
                 ns_steps: int = 5):
        named_params = list(named_params)
        muon_params = [p for n, p in named_params if p.requires_grad and _is_muon_param(n, p)]
        self._lora_partner, self._lora_role = _find_lora_pairs(
            [(n, p) for n, p in named_params if p.requires_grad])
        adam_params = [p for n, p in named_params if p.requires_grad and not _is_muon_param(n, p)]
        _adam_lr = adam_lr if adam_lr is not None else lr
        # `lr_init` is train.py's --lr_init (the schedule's base value the
        # callback actually multiplies `my_lr_scale` against) — defaults to
        # `_adam_lr` since callers normally pass adam_lr=args.lr_init.
        lr_init = lr_init if lr_init is not None else _adam_lr
        # rwkvt/lightning_train/trainer.py's train_callback overwrites
        # every param_group["lr"] on EVERY batch via
        # `lr_init/lr_final schedule * param_group["my_lr_scale"]` — not
        # optional, not something a custom Optimizer can opt out of. Without
        # this key the callback KeyErrors; without matching its scale
        # convention the muon/adam LR split would get silently clobbered
        # back to a single shared value each step. `my_lr_scale` here is
        # relative to `lr_init` (the scheduled base), so the actual
        # applied LR stays `lr` / `_adam_lr` as long as `lr_init` doesn't
        # drift from its start (true whenever lr_final==lr_init, i.e. no
        # decay — the common case for a short diagnostic run).
        groups = [
            dict(params=muon_params, use_muon=True, lr=lr, weight_decay=weight_decay,
                 momentum=momentum, my_lr_scale=lr / lr_init if lr_init else 1.0),
            dict(params=adam_params, use_muon=False, lr=_adam_lr,
                 weight_decay=weight_decay, betas=adam_betas, eps=adam_eps,
                 my_lr_scale=_adam_lr / lr_init if lr_init else 1.0),
        ]
        super().__init__(groups, defaults=dict())
        self.momentum_start = momentum_start
        self.momentum_final = momentum
        self.momentum_warmup_steps = momentum_warmup_steps
        self.ns_steps = ns_steps
        self._step_count = 0
        print(f"[muon_opt] MuonWithAuxAdam: {len(muon_params)} hidden matrices on Muon "
              f"(lr={lr}), {len(adam_params)} params on AdamW (lr={groups[1]['lr']})")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1
        frac = min(self._step_count / self.momentum_warmup_steps, 1.0) \
            if self.momentum_warmup_steps > 0 else 1.0
        cur_momentum = (1 - frac) * self.momentum_start + frac * self.momentum_final

        for group in self.param_groups:
            if group["use_muon"]:
                done = set()

                def raw(q, aspect):
                    st = self.state[q]
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = torch.zeros_like(q)
                    return _muon_update(q.grad, st["momentum_buffer"],
                                        beta=cur_momentum, ns_steps=self.ns_steps,
                                        aspect=aspect)

                def apply(q, upd, scale):
                    q.mul_(1 - group["lr"] * group["weight_decay"])
                    q.add_(upd.reshape(q.shape).to(q.dtype),
                           alpha=-group["lr"] * scale)

                for p in group["params"]:
                    if id(p) in done or p.grad is None:
                        continue
                    mate = self._lora_partner.get(id(p))
                    if mate is not None and mate.grad is not None:
                        a, b = ((p, mate) if self._lora_role[id(p)] == "A"
                                else (mate, p))
                        dA, dB = raw(a, False), raw(b, False)
                        sA, sB = _balanced_pair_scale(a.data, b.data, dA, dB)
                        apply(a, dA, sA)
                        apply(b, dB, sB)
                        done.add(id(a)); done.add(id(b))
                        continue
                    apply(p, raw(p, True), 1.0)
            else:
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.lerp_(p.grad, 1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                    bias1 = 1 - beta1 ** state["step"]
                    bias2 = 1 - beta2 ** state["step"]
                    if group["weight_decay"] != 0:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    denom = (exp_avg_sq / bias2).sqrt().add_(group["eps"])
                    p.addcdiv_(exp_avg, denom, value=-group["lr"] / bias1)
        return loss
