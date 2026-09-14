#!/usr/bin/env python3
"""MuonHybrid must treat a LoRA A/B pair as ONE update, not two weights.

The production bug this pins down: Muon's aspect rescale is per tensor, and a
LoRA pair is two tensors of transposed shape, so the whole coefficient lands on
`lora_B` — 8.944 at r=32/n_embd=2560, 17.889 for `ffn.key`. At the lr recorded
as stable (0.002) that steps B at 0.0179 and ffn.key's B at 0.0358, at and
above the 0.02 recorded as collapsing the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn

from experiments.rl.loader import MuonHybrid, _balanced_pair_scale, _muon_update


def _named(r=8, n_embd=64, dim_ffn=256):
    """PEFT's naming, on the modules MuonHybrid's predicate admits."""
    ps = {
        "blocks.0.att.key.weight": nn.Parameter(torch.randn(n_embd, n_embd)),
        "blocks.0.att.key.lora_A.default.weight": nn.Parameter(torch.randn(r, n_embd)),
        "blocks.0.att.key.lora_B.default.weight": nn.Parameter(torch.zeros(n_embd, r)),
        "blocks.0.ffn.key.lora_A.default.weight": nn.Parameter(torch.randn(r, n_embd)),
        "blocks.0.ffn.key.lora_B.default.weight": nn.Parameter(torch.zeros(dim_ffn, r)),
        "blocks.0.ln1.weight": nn.Parameter(torch.randn(n_embd)),
    }
    return list(ps.items()), ps


def test_pairs_are_found():
    named, ps = _named()
    opt = MuonHybrid(named, lr=0.002)
    assert opt.n_lora_pairs == 2, opt.n_lora_pairs
    a = ps["blocks.0.att.key.lora_A.default.weight"]
    b = ps["blocks.0.att.key.lora_B.default.weight"]
    assert opt._lora_partner[id(a)] is b and opt._lora_partner[id(b)] is a
    # a plain weight is still a muon param, and has no partner
    w = ps["blocks.0.att.key.weight"]
    assert w in opt.muon_params and id(w) not in opt._lora_partner
    # a 1D param is not Muon's business at all
    assert ps["blocks.0.ln1.weight"] in opt.other_params


def test_aspect_still_applies_to_standalone_weights():
    g = torch.randn(256, 64)
    buf = torch.zeros_like(g)
    with_aspect = _muon_update(g.clone(), buf.clone(), 0.9, 5, aspect=True)
    without = _muon_update(g.clone(), buf.clone(), 0.9, 5, aspect=False)
    ratio = (with_aspect.norm() / without.norm()).item()
    assert abs(ratio - 2.0) < 1e-3, ratio          # sqrt(256/64) = 2


def test_balanced_scale_equalises_contributions():
    r, n_in, n_out = 8, 64, 256
    A = torch.randn(r, n_in)
    B = torch.randn(n_out, r)                      # past init: B is nonzero
    dA, dB = torch.randn(r, n_in), torch.randn(n_out, r)
    sA, sB = _balanced_pair_scale(A, B, dA, dB)
    cA = (B @ (dA * sA)).norm().item()
    cB = ((dB * sB) @ A).norm().item()
    assert abs(cA - cB) / max(cA, cB) < 1e-4, (cA, cB)


def test_zero_init_B_does_not_freeze_the_adapter():
    """PEFT zeroes lora_B, so ||B dA|| is exactly 0 on step 0. Balancing to a
    geometric mean of zero would scale BOTH factors to zero and the adapter
    would never move."""
    r, n_in, n_out = 8, 64, 256
    A, B = torch.randn(r, n_in), torch.zeros(n_out, r)
    sA, sB = _balanced_pair_scale(A, B, torch.randn(r, n_in), torch.randn(n_out, r))
    assert (sA, sB) == (1.0, 1.0), (sA, sB)


def test_step_moves_B_at_the_same_scale_as_A():
    """The regression itself: with the production rescale B's step is
    sqrt(n_out/r) times A's. After the fix the two contributions match."""
    named, ps = _named(r=8, n_embd=64)
    a = ps["blocks.0.att.key.lora_A.default.weight"]
    b = ps["blocks.0.att.key.lora_B.default.weight"]
    b.data = torch.randn_like(b)                   # past init
    opt = MuonHybrid(named, lr=0.01, momentum=0.0, momentum_start=0.0)
    a0, b0 = a.data.clone(), b.data.clone()
    for p in opt.muon_params:
        p.grad = torch.randn_like(p)
    opt.step()
    cA = (b0 @ (a.data - a0)).norm().item()
    cB = ((b.data - b0) @ a0).norm().item()
    assert abs(cA - cB) / max(cA, cB) < 1e-3, (cA, cB)


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                fails += 1
                print("FAIL", name, e)
    raise SystemExit(1 if fails else 0)
