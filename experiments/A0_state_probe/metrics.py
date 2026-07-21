"""State-dynamics metrics for A0.4 — skeleton.

Three disjoint measures, one function each. Bodies raise
NotImplementedError; docstrings pin the math so the execution session
starts from the definition, not from a re-derivation.

All functions accept per-layer WKV state tensors of shape
``[n_head, head_size, head_size]``. Accumulation is expected in fp32
regardless of the input dtype — inputs are bf16 in the current design
to bound memory.
"""

from __future__ import annotations


def delta_norm(prev, curr) -> float:
    """L2 norm of the state's change between consecutive tokens.

    Computes ``||curr - prev||_2`` over the fully flattened state
    tensor (concatenation of all layers' WKV states). Returns a
    single scalar per (model, prompt, seed, step) — the pooled
    magnitude of state movement.

    A per-layer variant is trivial: apply the same formula per layer
    tensor and return a ``[n_layer]`` vector. The execution session
    should emit both — the pooled scalar for coarse comparison and
    the per-layer vector for the layer-wise activity story.

    Interpretation: high ``delta_norm`` = state changed a lot this
    step. It is not on its own evidence of computation (memory
    updates also cause large deltas); interpretation requires the
    curvature companion (see below).

    Args:
        prev: WKV state at token ``t-1``, either a single tensor or
            a list of per-layer tensors.
        curr: WKV state at token ``t``, same shape as ``prev``.

    Returns:
        Non-negative float in fp32.

    Raises:
        NotImplementedError: skeleton.
    """
    raise NotImplementedError("A0.4 skeleton — implement in execution session")


def curvature(prev_prev, prev, curr) -> float:
    """L2 norm of the second difference of the state trajectory.

    Computes ``||(curr - prev) - (prev - prev_prev)||_2`` over the
    flattened state. Requires three consecutive states, so the metric
    is defined for token ``t >= 2`` in the generation loop.

    Interpretation: low curvature = the trajectory is (locally)
    linear, which corresponds to a pure memory update — the state
    moves in a consistent direction. High curvature = the trajectory
    bends, which is what a computation-style update would look like
    when the "answer" the state is being trained toward shifts across
    tokens (see paper §2 SGD-step framing in
    ../../docs/state-and-reasoning.md).

    Reasoning-vs-narrative curvature contrast is the core evidence
    for H8 — a memory-only reading of the state predicts they should
    look alike after prompt-content controls; a compute-reading
    predicts systematic curvature differences during a `<think>` or
    reasoning-flavoured continuation.

    Args:
        prev_prev: WKV state at token ``t-2``.
        prev: WKV state at token ``t-1``.
        curr: WKV state at token ``t``.

    Returns:
        Non-negative float in fp32.

    Raises:
        NotImplementedError: skeleton.
    """
    raise NotImplementedError("A0.4 skeleton — implement in execution session")


def stable_rank(state) -> "list[float]":
    """Effective rank of each WKV head, per RWKV-7 paper Appendix J.

    For each head's ``[head_size, head_size]`` matrix ``A``, computes:

        ``SR(A) = (||A||_F / ||A||_2)^2``

    where ``||A||_F`` is the Frobenius norm and ``||A||_2`` is the
    spectral norm (largest singular value). Returns a list of length
    ``n_head * n_layer`` (or a nested list `[n_layer][n_head]` — pick
    a consistent layout in the execution session and stick to it).

    Interpretation: SR is a smoothed rank measure. SR near 1 = state
    concentrated in a single dominant direction; SR approaching
    ``head_size`` = state uses all directions roughly equally. Under
    the SGD-at-test-time framing, more diverse state use during
    reasoning would show as elevated SR variance over time.

    This metric is chosen specifically because the RWKV-7 paper
    reports it (Appendix J, p. 50), giving A0.4 results direct
    comparability to the authors' own probing.

    Args:
        state: A single per-layer WKV tensor
            (``[n_head, head_size, head_size]``) or a list of them.

    Returns:
        List of stable-rank values, one per head. fp32.

    Raises:
        NotImplementedError: skeleton.
    """
    raise NotImplementedError("A0.4 skeleton — implement in execution session")


__all__ = ["delta_norm", "curvature", "stable_rank"]
