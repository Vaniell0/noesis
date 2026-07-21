"""State-dynamics metrics for A0.4.

Three disjoint measures. All accept per-layer WKV state as either a
single tensor of shape ``[n_head, head_size, head_size]`` or a list of
such tensors (one per RWKV block). Inputs are expected in bf16 (or fp32
already); accumulation happens in fp32 regardless.

The three metrics are intended to answer distinct questions:

- ``delta_norm``      — "how far did the state move?"
- ``curvature``       — "is the trajectory straight (memory) or bent (compute)?"
- ``stable_rank``     — "how many effective directions does the state use?" (per head)

Definitions are matched to the paper (Appendix J stable-rank) plus the
plan's own delta / second-difference framing. See
``../../docs/state-and-reasoning.md`` §1 (Appendix J) and
``../../HYPOTHESES.md`` §H8 for the underlying rationale.
"""

from __future__ import annotations

from typing import Sequence

import torch


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _as_layer_list(state) -> "list[torch.Tensor]":
    """Normalise to a list-of-per-layer-tensors."""
    if isinstance(state, torch.Tensor):
        return [state]
    return list(state)


def _flat_fp32(state) -> torch.Tensor:
    """Concatenate all per-layer tensors into a single fp32 vector."""
    layers = _as_layer_list(state)
    return torch.cat([t.reshape(-1).to(torch.float32) for t in layers])


# --------------------------------------------------------------------------- #
# 1. delta norm
# --------------------------------------------------------------------------- #

def delta_norm(prev, curr) -> "tuple[float, list[float]]":
    """L2 norm of the state's change between consecutive tokens.

    Returns ``(pooled, per_layer)``:

    - ``pooled``     — ``‖curr − prev‖_2`` over the concatenation of all
                       per-layer tensors flattened; single scalar.
    - ``per_layer``  — one non-negative float per layer.

    Interpretation: high ``pooled`` means the state moved a lot this
    step. On its own, that is compatible with both memory-update and
    compute readings; use in conjunction with ``curvature``.

    Inputs may be a single tensor or a list of per-layer tensors of
    equal length. Accumulation is fp32.
    """
    prev_list = _as_layer_list(prev)
    curr_list = _as_layer_list(curr)
    if len(prev_list) != len(curr_list):
        raise ValueError(
            f"delta_norm: layer count mismatch prev={len(prev_list)} "
            f"curr={len(curr_list)}"
        )
    per_layer = [
        torch.linalg.vector_norm(
            (c.to(torch.float32) - p.to(torch.float32)).reshape(-1)
        ).item()
        for p, c in zip(prev_list, curr_list)
    ]
    pooled = float(torch.tensor(per_layer, dtype=torch.float64).pow(2).sum().sqrt())
    return pooled, per_layer


# --------------------------------------------------------------------------- #
# 2. curvature
# --------------------------------------------------------------------------- #

def curvature(prev_prev, prev, curr) -> "tuple[float, list[float]]":
    """L2 norm of the second difference of the state trajectory.

    Returns ``(pooled, per_layer)`` where the underlying quantity is:

        ``‖(curr − prev) − (prev − prev_prev)‖_2``

    over the flattened concatenation of per-layer tensors.

    Interpretation: low curvature = trajectory is locally linear
    (memory update — the state moves in a consistent direction). High
    curvature = the trajectory bends, which is the compute-style
    signature under the paper §2 SGD-step framing.

    The metric is defined for token ``t >= 2`` in the generation loop.
    Caller responsible for skipping the first two tokens.
    """
    pp_list = _as_layer_list(prev_prev)
    p_list = _as_layer_list(prev)
    c_list = _as_layer_list(curr)
    n = len(pp_list)
    if not (len(p_list) == n == len(c_list)):
        raise ValueError("curvature: layer count mismatch across three states")
    per_layer = []
    for pp, p, c in zip(pp_list, p_list, c_list):
        pp_f = pp.to(torch.float32)
        p_f = p.to(torch.float32)
        c_f = c.to(torch.float32)
        d1 = (c_f - p_f).reshape(-1)
        d2 = (p_f - pp_f).reshape(-1)
        per_layer.append(float(torch.linalg.vector_norm(d1 - d2)))
    pooled = float(torch.tensor(per_layer, dtype=torch.float64).pow(2).sum().sqrt())
    return pooled, per_layer


# --------------------------------------------------------------------------- #
# 3. stable rank (paper Appendix J)
# --------------------------------------------------------------------------- #

def stable_rank(state) -> "list[list[float]]":
    """Effective rank of every WKV head, per RWKV-7 paper Appendix J.

    For each per-layer tensor of shape ``[n_head, head_size, head_size]``
    and each head-matrix ``A`` inside it, returns:

        ``SR(A) = (‖A‖_F² / ‖A‖_2²)``

    where ``‖A‖_F`` is the Frobenius norm and ``‖A‖_2`` is the spectral
    norm (largest singular value).

    Layout: returns a nested list ``sr[layer_idx][head_idx]``. Callers
    that want a flat list can `sum(sr, [])`.

    Interpretation: SR near 1 = state concentrated in one dominant
    direction; SR approaching ``head_size`` = state spread across all
    directions. Reasoning-vs-narrative variance over the trajectory
    is what H8 pre-registration checks.
    """
    layers = _as_layer_list(state)
    per_layer: "list[list[float]]" = []
    for t in layers:
        if t.dim() == 2:
            t = t.unsqueeze(0)
        if t.dim() != 3:
            raise ValueError(
                f"stable_rank: expected [n_head, h, h] or [h, h], got {tuple(t.shape)}"
            )
        t_f = t.to(torch.float32)
        heads: "list[float]" = []
        for h in range(t_f.shape[0]):
            A = t_f[h]
            frob_sq = float(torch.linalg.matrix_norm(A, ord="fro").pow(2))
            svals = torch.linalg.svdvals(A)
            top_sq = float(svals[0].pow(2))
            if top_sq < 1e-30:
                heads.append(0.0)
            else:
                heads.append(frob_sq / top_sq)
        per_layer.append(heads)
    return per_layer


__all__ = ["delta_norm", "curvature", "stable_rank"]
