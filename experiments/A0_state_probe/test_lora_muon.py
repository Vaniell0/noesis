#!/usr/bin/env python3
"""Invariants for lora_muon_probe.py.

The probe's whole content is a set of claims about what an optimizer does to a
factorised weight. Each of those claims is a property of the code that can be
checked without training anything — so it is checked here, rather than being
inferred from a run's numbers, where an indentation slip or a transposed shape
would show up only as "this arm is worse" and get written into a hypothesis.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn

from experiments.A0_state_probe.lora_muon_probe import (
    LoRALinear,
    LoRAMuon,
    lorafy,
    _adapter_spectrum,
)
from experiments.A0_state_probe.micro_wkv import FrozenFinalReadoutController

# The real adapter shapes this project runs, so the numbers the module
# docstring quotes are the numbers these tests check.
REAL_D, REAL_R = 2560, 32


def _adapter(out_f: int = REAL_D, in_f: int = REAL_D, r: int = REAL_R,
             alpha: float = 64.0) -> LoRALinear:
    return LoRALinear(nn.Linear(in_f, out_f, bias=False), r, alpha)


def test_lora_init_matches_peft_convention():
    """lora_B zero, lora_A not, adapter contributes nothing at init — the
    zero-init is what makes lora_A's first gradient exactly zero, which every
    other claim in the module depends on."""
    ad = _adapter()
    assert ad.lora_A.shape == (REAL_R, REAL_D)
    assert ad.lora_B.shape == (REAL_D, REAL_R)
    assert torch.count_nonzero(ad.lora_B) == 0
    assert torch.count_nonzero(ad.lora_A) > 0
    assert torch.allclose(ad.delta_w(), torch.zeros_like(ad.delta_w()))


def test_lora_A_has_zero_gradient_at_init():
    """The load-bearing fact behind "the first steps flow entirely through
    lora_B": with B=0, A has no path to the output."""
    ad = _adapter(out_f=8, in_f=8, r=4, alpha=8.0)
    x = torch.randn(16, 8)
    ad(x).pow(2).mean().backward()
    assert ad.lora_A.grad is not None and float(ad.lora_A.grad.abs().max()) == 0.0
    assert float(ad.lora_B.grad.abs().max()) > 0.0


def test_factorwise_reproduces_the_production_asymmetry():
    """The finding this probe exists to test must actually be present in the
    arm that claims to reproduce production, not silently fixed by a rewrite.
    Expected ~8.944 = sqrt(2560/32) between the two factors' step RMS."""
    torch.manual_seed(0)
    ad = _adapter()
    opt = LoRAMuon([ad], "muon_factorwise", lr=0.02)
    gA, gB = torch.randn_like(ad.lora_A), torch.randn_like(ad.lora_B)
    dA, dB = opt.directions(ad, gA, gB)
    rms = lambda t: float(t.float().pow(2).mean().sqrt())  # noqa: E731
    ratio = rms(dB) / rms(dA)
    assert 8.0 < ratio < 10.0, ratio


def test_balanced_equalises_the_two_contributions():
    """muon_balanced's definition: ‖B δA‖_F == ‖δB A‖_F after scaling. Uses a
    nonzero B, since at the zero init the A half is identically zero and the
    property is vacuous."""
    torch.manual_seed(0)
    ad = _adapter(out_f=64, in_f=64, r=8, alpha=16.0)
    with torch.no_grad():
        ad.lora_B.normal_(0, 0.1)
    opt = LoRAMuon([ad], "muon_balanced", lr=0.02)
    dA, dB = opt.directions(ad, torch.randn_like(ad.lora_A), torch.randn_like(ad.lora_B))
    with torch.no_grad():
        cA = float((ad.lora_B @ dA).float().norm())
        cB = float((dB @ ad.lora_A).float().norm())
    assert abs(cA - cB) / max(cA, cB) < 1e-4, (cA, cB)


def test_product_step_has_unit_spectral_norm_in_w_space():
    """muon_product's definition: the step size is set on the INDUCED ΔW, so
    that induced update must come back with spectral norm 1 before lr."""
    torch.manual_seed(0)
    ad = _adapter(out_f=64, in_f=64, r=8, alpha=16.0)
    with torch.no_grad():
        ad.lora_B.normal_(0, 0.1)
    opt = LoRAMuon([ad], "muon_product", lr=0.02)
    dA, dB = opt.directions(ad, torch.randn_like(ad.lora_A), torch.randn_like(ad.lora_B))
    with torch.no_grad():
        induced = ad.scaling * (ad.lora_B @ dA + dB @ ad.lora_A)
        sigma1 = float(torch.linalg.matrix_norm(induced.float(), ord=2))
    assert abs(sigma1 - 1.0) < 1e-3, sigma1


def test_product_leaves_A_alone_while_B_is_zero():
    """Not a bug but a property worth pinning: with B=0 the adapter's output
    does not depend on A, so a W-space method must hand A a zero step."""
    torch.manual_seed(0)
    ad = _adapter(out_f=64, in_f=64, r=8, alpha=16.0)
    opt = LoRAMuon([ad], "muon_product", lr=0.02)
    dA, _ = opt.directions(ad, torch.randn_like(ad.lora_A), torch.randn_like(ad.lora_B))
    assert float(dA.abs().max()) == 0.0


def test_factorwise_forces_a_flat_full_rank_adapter():
    """The second half of the finding: Newton-Schulz on the factors makes the
    injected ΔW near-flat at full adapter rank, by construction rather than
    because the task asked for it."""
    torch.manual_seed(0)
    r = 16
    ad = _adapter(out_f=256, in_f=256, r=r, alpha=32.0)
    opt = LoRAMuon([ad], "muon_factorwise", lr=0.05)
    for _ in range(20):
        ad.lora_A.grad = torch.randn_like(ad.lora_A)
        ad.lora_B.grad = torch.randn_like(ad.lora_B)
        opt.step()
    spec = _adapter_spectrum([ad])
    assert spec["delta_w_entropy_rank"] > 0.85 * r, spec
    assert spec["delta_w_sigma_r_over_1"] > 0.3, spec


def test_state_metric_is_bounded_and_skips_dead_tensors():
    """The fix for the first smoke run's 1e27 divergence: a tensor with no
    gradient signal is skipped rather than measured, and no measured scale may
    exceed max_scale_ratio in either direction."""
    torch.manual_seed(0)
    model = FrozenFinalReadoutController(8, 6)
    adapters = lorafy(model, r=4, alpha=8.0)
    opt = LoRAMuon(adapters, "muon_state", lr=0.02, max_scale_ratio=16.0)

    a = torch.empty(32).uniform_(-1, 1)
    b = torch.empty(32).uniform_(-1, 1)
    model(a, b)[0].pow(2).mean().backward()

    sens = opt.recalibrate_state_metric(model, (a, b))
    # lora_A is dead at init (B=0) and must not appear among the measured.
    assert all(id(ad.lora_A) not in sens for ad in adapters)
    assert any(id(ad.lora_B) in sens for ad in adapters)
    for v in opt.state_scale.values():
        assert 1 / 16.0 - 1e-9 <= v <= 16.0 + 1e-9, v
    assert all(v == v and abs(v) != float("inf") for v in opt.state_scale.values())


def test_state_metric_can_contradict_the_shape_rule():
    """The bound must be wide enough that the measurement is allowed to
    disagree with the 8.944 the shape-based rule asserts — otherwise the arm
    could only ever confirm it."""
    opt = LoRAMuon([], "muon_state", lr=0.02)
    assert opt.max_scale_ratio > 8.944


def test_lorafy_freezes_the_base_and_claims_only_net_weights():
    torch.manual_seed(0)
    model = FrozenFinalReadoutController(8, 6)
    adapters = lorafy(model, r=4, alpha=8.0)
    assert len(adapters) == 3, len(adapters)
    for ad in adapters:
        assert not ad.base.weight.requires_grad
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable), trainable


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
