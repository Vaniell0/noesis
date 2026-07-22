"""Smoke test for state_reg hookup in lora_train.

Verifies three invariants required before merging any state_reg loss
changes:

(a) Two forward + backward steps produce finite (non-NaN, non-inf)
    gradients on every trainable parameter.
(b) ``loss.item()`` is materially different between ``alpha=0`` and
    ``alpha=alpha_pos`` — i.e. the state_reg term actually contributes
    to the total loss, not silently zero'd by the hookup.
(c) ``compute_state_reg`` returns exactly 0 for sequences too short to
    admit even a single curvature triple (T < 2 in the docstring's
    per-token indexing; here we test T = 1 and T = 2, both must return
    0 without raising).

Runs on CPU with a mock micro-model that mimics the RWKV attention
module's state-writing contract (``self._captured_wkv``) — no CUDA
kernels, no vendored RWKV-PEFT, no real checkpoint. This is the same
interface ``StateCapture`` uses against the real model, so passing here
is a necessary (not sufficient) precondition for real training.

Run:
    /home/vaniello/Desktop/projects/noesis/training/.venv/bin/python \\
        training/tests/test_state_reg_hookup.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

from state_reg import StateRegConfig, compute_state_reg  # noqa: E402
from lora_train import StateCapture, train_step          # noqa: E402


# --------------------------------------------------------------------------- #
# Mock micro-model
# --------------------------------------------------------------------------- #

class MockTmix(nn.Module):
    """Stand-in for RWKV-7 T-mix module.

    Contract exercised by StateCapture (protocol A): after ``forward``
    the module exposes ``self._captured_wkv`` as a tensor of shape
    ``[B, T, H, h, h]``. Has ``layer_id`` so StateCapture's discovery
    walk picks it up.

    Class name intentionally contains ``Tmix`` — StateCapture's
    discovery filters modules by that substring.
    """

    def __init__(self, layer_id: int, hidden: int, n_head: int, head_size: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_head = n_head
        self.head_size = head_size
        state_dim = n_head * head_size * head_size
        self.state_proj = nn.Linear(hidden, state_dim, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)
        self._captured_wkv: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, hidden]
        B, T, _ = x.shape
        raw = self.state_proj(x)  # [B, T, n_head * head_size^2]
        state = raw.reshape(B, T, self.n_head, self.head_size, self.head_size)
        self._captured_wkv = state
        return self.out_proj(x)


class MockRWKV(nn.Module):
    """Stack of MockTmix blocks with an embedding + LM head.

    Emits per-token logits and, as a side effect of each block's
    forward, populates ``block._captured_wkv``. That is exactly what
    StateCapture (protocol A) expects to find.
    """

    def __init__(
        self,
        vocab: int = 32,
        hidden: int = 16,
        n_layer: int = 4,
        n_head: int = 2,
        head_size: int = 4,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, hidden)
        self.blocks = nn.ModuleList(
            [MockTmix(L, hidden, n_head, head_size) for L in range(n_layer)]
        )
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(input_ids)
        for blk in self.blocks:
            x = x + blk(x)
        return self.head(x)


# --------------------------------------------------------------------------- #
# Test primitives
# --------------------------------------------------------------------------- #

def _ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
    )


def _fresh_model_and_batch(seed: int, T: int = 6):
    torch.manual_seed(seed)
    model = MockRWKV()
    input_ids = torch.randint(0, 32, (1, T))
    labels = torch.randint(0, 32, (1, T))
    return model, input_ids, labels


def _default_cfg(mode: str = "trajectory_reg", alpha: float = 0.1) -> StateRegConfig:
    return StateRegConfig(
        mode=mode,
        alpha=alpha,
        lambda_delta=1.0,
        lambda_curvature=1.0,
        work_layers=(0, 1, 2, 3),
        layer_weights={0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
    )


# --------------------------------------------------------------------------- #
# Test (a) — two steps, gradients finite
# --------------------------------------------------------------------------- #

def test_two_steps_grad_finite() -> None:
    model, input_ids, labels = _fresh_model_and_batch(seed=0)
    cfg = _default_cfg(alpha=0.1)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)

    losses = []
    for step in range(2):
        optim.zero_grad()
        out = train_step(model, input_ids, labels, _ce, cfg)
        assert torch.isfinite(out.total_loss).item(), (
            f"step {step}: total_loss = {out.total_loss.item()} (non-finite)"
        )
        out.total_loss.backward()

        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            finite = torch.isfinite(p.grad).all().item()
            assert finite, f"step {step}: param {name} has non-finite grad"

        optim.step()
        losses.append(out.total_loss.item())

    # Sanity: loss should evolve (not stuck at initial value bit-for-bit).
    assert losses[0] != losses[1], (
        f"loss unchanged across steps: {losses} — optim.step is a no-op?"
    )
    print(f"(a) two_steps_grad_finite: losses = {losses}")


# --------------------------------------------------------------------------- #
# Test (b) — alpha != 0 changes total loss
# --------------------------------------------------------------------------- #

def test_alpha_changes_total_loss() -> None:
    # Same weights, same input — only alpha differs. The CE piece is
    # identical between the two runs; total_loss must differ iff the
    # state_reg piece is actually plumbed in.
    model_a, input_ids, labels = _fresh_model_and_batch(seed=42)
    model_b, _, _ = _fresh_model_and_batch(seed=42)
    # Verify weight parity (sanity, no drift after seeding).
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb), "seed 42 did not produce identical models"

    out_a = train_step(model_a, input_ids, labels, _ce,
                       _default_cfg(mode="trajectory_reg", alpha=0.0))
    out_b = train_step(model_b, input_ids, labels, _ce,
                       _default_cfg(mode="trajectory_reg", alpha=0.1))

    ce_a = out_a.ce_loss.item()
    ce_b = out_b.ce_loss.item()
    total_a = out_a.total_loss.item()
    total_b = out_b.total_loss.item()

    assert abs(ce_a - ce_b) < 1e-6, (
        f"CE loss diverged despite identical seeds: {ce_a} vs {ce_b}"
    )
    delta = total_b - total_a
    assert abs(delta) > 1e-6, (
        f"total_loss unchanged when alpha 0 -> 0.1 "
        f"(alpha=0: {total_a}, alpha=0.1: {total_b}). "
        "state_reg either not computed or not added to CE."
    )
    print(f"(b) alpha_changes_total_loss: "
          f"CE={ce_a:.6f}, α=0→{total_a:.6f}, α=0.1→{total_b:.6f}, Δ={delta:+.6f}")


# --------------------------------------------------------------------------- #
# Test (c) — short sequences return zero
# --------------------------------------------------------------------------- #

def test_short_sequence_returns_zero() -> None:
    cfg = _default_cfg()

    def _make_states(T: int) -> list[torch.Tensor]:
        # Random states, requires_grad so we'd notice if a fake path
        # accidentally returned a detached zero from a live tensor.
        B, H, hs = 1, 2, 4
        return [
            torch.randn(B, T, H, hs, hs, requires_grad=True)
            for _ in range(len(cfg.work_layers))
        ]

    for T in (0, 1, 2):
        states = _make_states(T)
        # Wrap into a dict-like lookup that matches work_layers indices.
        lookup = {L: states[i] for i, L in enumerate(cfg.work_layers)}
        loss = compute_state_reg(lookup, cfg)
        assert loss.item() == 0.0, (
            f"T={T}: expected loss=0, got {loss.item()}"
        )
    print("(c) short_sequence_returns_zero: T in {0,1,2} all return 0.0")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def main() -> int:
    tests = [
        test_two_steps_grad_finite,
        test_alpha_changes_total_loss,
        test_short_sequence_returns_zero,
    ]
    failed: list[tuple[str, BaseException]] = []
    for t in tests:
        try:
            t()
        except BaseException as e:  # noqa: BLE001
            failed.append((t.__name__, e))
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{len(failed)}/{len(tests)} tests FAILED")
        return 1
    print(f"\n{len(tests)}/{len(tests)} tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
