"""State-regularization loss — A1 pilot implementation.

Formulation locked after A0.5 verdict (see
`experiments/A0_state_probe/results.md` §A0.5 and
`experiments/A0_state_probe/results/a05_ext/verdict.md`).

## The signal A0.5 validated

Three sub-tests, all four cells:
  * A: σ-response superlinear (log-log slope 1.56–2.10)
  * B: layer localisation — L16 dominates, mid-late layers carry load
  * C: cross-prompt vs norm-matched noise ratio 21–99×

The training signal here reifies (B) into a per-layer weighted penalty
and (A)-ish behaviour by rewarding *state motion* + *trajectory
curvature* on the layers A0.5 found load-bearing.

## Formula

For each token t >= 2 in a training sequence:

    L_state(t) = sum_L w_L * (
                    -λ_δ * ‖s_L(t) − s_L(t-1)‖_2
                    -λ_κ * ‖(s_L(t) − s_L(t-1)) − (s_L(t-1) − s_L(t-2))‖_2
                 )

where:
  * s_L(t) is the WKV state at layer L, timestep t — shape
    [n_head, head_size, head_size] per sample.
  * w_L is the layer weight — from A0.5 zero_layer KL profile,
    normalized so sum(w_L) = 1 over the work layers.
  * λ_δ, λ_κ are subterm coefficients (default 1.0, 1.0).
  * The signs are chosen so *minimising* L_state *increases* motion +
    curvature — i.e. the gradient rewards state doing more per token.

Sequence-level loss = mean over t of L_state(t). Zero for t < 2.

Combined training objective (in lora_train.py):

    L_total = L_CE_SFT + config.alpha * L_state_mean

## Layer weights (A0.5-derived)

From `verdict.md` §Cell world_medium §zero_layer profile (mean KL):
  L0=0.004, L4=0.014, L8=0.029, L12=0.101, L16=0.200, L20=0.094.

For a 24-layer model, work_layers default to the sampled set
{12, 16, 20}. Between-sample layers are set to 0 by default; a
config toggle exposes linear interpolation for future ablations.

Normalized weights over {12, 16, 20}:
    w_12 = 0.101 / 0.395 = 0.256
    w_16 = 0.200 / 0.395 = 0.506
    w_20 = 0.094 / 0.395 = 0.238

## What's deferred

  * **stable_rank** term (per Appendix J). Full SVD per step per layer
    is O(n_head · head_size³). At 32 layers × 40 heads × 64 head_size
    that's ~5×10⁷ FLOPs per token just for the SR loss — comparable to
    the whole forward pass. Included as `mode='trajectory_reg_with_sr'`
    only for future benchmarking; A1 pilot uses `'trajectory_reg'`.
  * **Perturbation-consistency term** (H8-causal-C mimic). Would
    require paired forward passes with corrupted state, doubling
    training compute. Not in A1 pilot; potential Phase-2 augmentation.

## Hookup

This module only supplies the *loss function*. The training loop
(`lora_train.py`, to be written / patched in vendored RWKV-PEFT) is
responsible for:
  1. Extracting WKV state per layer per timestep via forward hooks
     — same access pattern as
     `experiments/A0_state_probe/probe.py:_extract_wkv_per_layer`.
  2. Assembling into the tensor shape `compute_state_reg` expects.
  3. Calling `compute_state_reg(state_seq, config)` and adding
     `config.alpha * result` to the CE loss before backward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


# ------------------------------------------------------------------------- #
# Config
# ------------------------------------------------------------------------- #

VALID_MODES: tuple[str, ...] = (
    "off",
    "trajectory_reg",           # delta + curvature (A1 pilot default)
    "trajectory_reg_with_sr",   # + stable_rank (deferred, expensive)
)

# A0.5 zero_layer KL profile, world_medium cell.
# Copied verbatim from verdict.md §Cell world_medium §H8-causal-B.
_A05_ZERO_LAYER_KL: dict[int, float] = {
    0: 0.004,
    4: 0.014,
    8: 0.029,
    12: 0.101,
    16: 0.200,
    20: 0.094,
}

# Default work-layer set — the A0.5-sampled layers whose KL is above
# the "input-layer floor" of ~0.01. This is the load-bearing subset;
# other layers get weight 0 in the pilot.
DEFAULT_WORK_LAYERS: tuple[int, ...] = (12, 16, 20)


def default_layer_weights(work_layers: Sequence[int] = DEFAULT_WORK_LAYERS
                          ) -> dict[int, float]:
    """A0.5-derived layer weights, normalized to sum=1 over work_layers."""
    raw = {L: _A05_ZERO_LAYER_KL.get(L, 0.0) for L in work_layers}
    total = sum(raw.values())
    if total <= 0:
        n = len(work_layers)
        return {L: 1.0 / n for L in work_layers}
    return {L: v / total for L, v in raw.items()}


@dataclass
class StateRegConfig:
    """Config for L_state; instantiated from `training/config/pilot.yaml`."""
    mode: str = "off"
    alpha: float = 0.0                      # outer coefficient in L_total
    lambda_delta: float = 1.0               # δ-norm subterm weight
    lambda_curvature: float = 1.0           # κ subterm weight
    lambda_stable_rank: float = 0.0         # SR subterm (deferred; expensive)
    work_layers: tuple[int, ...] = DEFAULT_WORK_LAYERS
    layer_weights: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"state_reg_mode={self.mode!r} not in {VALID_MODES}"
            )
        if not self.layer_weights:
            self.layer_weights = default_layer_weights(self.work_layers)
        missing = set(self.work_layers) - set(self.layer_weights)
        if missing:
            raise ValueError(
                f"work_layers {sorted(missing)} missing from layer_weights"
            )


# ------------------------------------------------------------------------- #
# Differentiable subterms
# ------------------------------------------------------------------------- #

def _layer_delta(s_prev: torch.Tensor, s_curr: torch.Tensor) -> torch.Tensor:
    """‖s_curr − s_prev‖_2 (scalar tensor with grad).

    Input shape: [batch, seq_or_scalar_over_pair, n_head, h, h]. The
    L2 norm is over the last three dims (per-sample, per-timestep),
    then averaged across batch × seq to give a scalar.
    """
    diff = (s_curr - s_prev).flatten(start_dim=-3)
    per_step_norm = torch.linalg.vector_norm(diff, dim=-1)
    return per_step_norm.mean()


def _layer_curvature(s_pp: torch.Tensor,
                     s_p: torch.Tensor,
                     s_c: torch.Tensor) -> torch.Tensor:
    """‖(s_c − s_p) − (s_p − s_pp)‖_2 (scalar tensor with grad)."""
    d1 = (s_c - s_p).flatten(start_dim=-3)
    d2 = (s_p - s_pp).flatten(start_dim=-3)
    per_step = torch.linalg.vector_norm(d1 - d2, dim=-1)
    return per_step.mean()


def _layer_stable_rank_std(s_seq: torch.Tensor) -> torch.Tensor:
    """Per-head SR standard deviation across timesteps.

    s_seq shape: [batch, T, n_head, h, h]. Returns scalar tensor.
    Expensive — do NOT enable in A1 pilot unless benchmarking.

    SR = ‖A‖_F² / ‖A‖_2². Uses full SVD; power iteration would be
    cheaper but is left for a later optimisation pass.
    """
    B, T, H, hs, _ = s_seq.shape
    A = s_seq.reshape(B * T * H, hs, hs)
    frob_sq = A.pow(2).sum(dim=(-2, -1))
    svals = torch.linalg.svdvals(A)
    top_sq = svals[..., 0].pow(2).clamp(min=1e-30)
    sr = frob_sq / top_sq
    sr = sr.reshape(B, T, H)
    return sr.std(dim=1).mean()


# ------------------------------------------------------------------------- #
# Main loss entry point
# ------------------------------------------------------------------------- #

def compute_state_reg(
    wkv_per_layer,
    config: StateRegConfig,
) -> torch.Tensor:
    """Compute L_state.

    Args:
      wkv_per_layer: indexable of per-layer state tensors, each shape
        [batch, T, n_head, head_size, head_size]. T is the sequence
        length (state at each timestep). ``__getitem__(L)`` must
        resolve every ``L in config.work_layers``. For T < 3 the
        summation range (t in [2, T-1]) is empty and the returned
        loss is zero — matches the "Zero for t < 2" contract in the
        formula section of this module's docstring.
      config: StateRegConfig.

    Returns:
      Scalar tensor with autograd. Multiply by config.alpha and add
      to CE loss in the training loop.
    """
    if config.mode == "off":
        return torch.zeros((), dtype=torch.float32)

    # Materialise the first work-layer state so we can (a) probe device/dtype
    # and (b) short-circuit degenerate sequence lengths without doing further
    # indexing (matters when the caller's __getitem__ has side effects).
    first_layer = config.work_layers[0]
    first_state = wkv_per_layer[first_layer]
    device = first_state.device
    dtype = first_state.dtype
    if first_state.shape[1] < 3:
        # Empty summation range → 0. Returning float32 loses grad edge
        # cases; but T<3 means no state trajectory exists to
        # differentiate, so a detached zero is correct.
        return torch.zeros((), device=device, dtype=dtype)

    total = torch.zeros((), device=device, dtype=dtype)

    for L in config.work_layers:
        w_L = config.layer_weights[L]
        s = wkv_per_layer[L]                    # [B, T, H, h, h]
        s_pp = s[:, :-2]
        s_p = s[:, 1:-1]
        s_c = s[:, 2:]

        layer_loss = torch.zeros((), device=device, dtype=dtype)
        if config.lambda_delta != 0.0:
            # NB: only the last-1 vs last-2 slice for delta (T-1 pairs
            # exist, but for shape parity with curvature we use T-2).
            delta = _layer_delta(s_p, s_c)
            layer_loss = layer_loss - config.lambda_delta * delta
        if config.lambda_curvature != 0.0:
            kappa = _layer_curvature(s_pp, s_p, s_c)
            layer_loss = layer_loss - config.lambda_curvature * kappa
        if (config.mode == "trajectory_reg_with_sr"
                and config.lambda_stable_rank != 0.0):
            sr_std = _layer_stable_rank_std(s)
            layer_loss = layer_loss - config.lambda_stable_rank * sr_std

        total = total + w_L * layer_loss

    return total


__all__ = [
    "VALID_MODES",
    "DEFAULT_WORK_LAYERS",
    "default_layer_weights",
    "StateRegConfig",
    "compute_state_reg",
]
