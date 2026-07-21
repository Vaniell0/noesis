"""Causal intervention primitives for A0.5.

Given a WKV state list (``3 * n_layer`` flat, per ``probe.py`` conventions
— attn-shift at ``3*i``, WKV at ``3*i+1``, ffn-shift at ``3*i+2``),
apply various corruptions and single-forward divergence metrics. See
``A05_intervention_plan.md`` §Design for the taxonomy.

**Invariant.** Every ``corrupt_*`` returns a state list whose tensors
are *fully independent* of the input tensors — the BlinkDL rwkv package
mutates state in-place during ``forward()``, so a corrupt state that
shared shift-buffer references with the clean state would silently
corrupt the clean side on the very next ``model.forward`` call. Each
function therefore starts with ``snapshot(state)`` (~8 MB fp32 alloc
on a 32-layer 0.4B checkpoint) and modifies the fresh copy.
"""

from __future__ import annotations

from typing import List, Optional

import torch


def _wkv_positions(state: List[torch.Tensor]) -> List[int]:
    return [3 * i + 1 for i in range(len(state) // 3)]


def snapshot(state: List[torch.Tensor]) -> List[torch.Tensor]:
    """Deep-copy every state tensor. Needed because the rwkv package's
    ``forward(idx, state)`` may reuse the list's tensors in-place."""
    return [t.detach().clone() for t in state]


# --------------------------------------------------------------------------- #
# Corruptions
# --------------------------------------------------------------------------- #

def corrupt_gauss(
    state: List[torch.Tensor],
    sigma: float,
    generator: Optional[torch.Generator] = None,
) -> List[torch.Tensor]:
    """Additive Gaussian scaled to per-layer state Frobenius norm.

    Per layer: ``s' = s + sigma * ||s||_F / sqrt(dim) * N(0, I)``.
    Isotropic perturbation, sigma is the fractional norm shift.
    """
    out = snapshot(state)
    for pos in _wkv_positions(out):
        t = out[pos]
        norm = float(t.to(torch.float32).norm().item())
        dim = t.numel()
        noise = torch.randn(t.shape, generator=generator, dtype=torch.float32)
        out[pos] = (t.to(torch.float32) + (sigma * norm / (dim ** 0.5)) * noise).to(t.dtype)
    return out


def corrupt_zero_layer(state: List[torch.Tensor], layer_idx: int) -> List[torch.Tensor]:
    """Zero the WKV state at a single layer. Localises where state work
    happens: layers whose zeroing barely moves the output are not doing
    computational work at this step."""
    out = snapshot(state)
    pos = 3 * layer_idx + 1
    out[pos] = torch.zeros_like(out[pos])
    return out


def corrupt_zero_head(
    state: List[torch.Tensor], layer_idx: int, head_idx: int
) -> List[torch.Tensor]:
    """Zero one head (row 0 of the [n_head, hd, hd] tensor) at one layer."""
    out = snapshot(state)
    pos = 3 * layer_idx + 1
    out[pos][head_idx].zero_()
    return out


def corrupt_shuffle_heads(
    state: List[torch.Tensor],
    layer_idx: int,
    generator: Optional[torch.Generator] = None,
) -> List[torch.Tensor]:
    """Permute the head dimension within one layer. Preserves the Frobenius
    norm and stable rank; destroys inter-head structure. Tests whether
    the *arrangement* of state matters vs merely its bulk statistics."""
    out = snapshot(state)
    pos = 3 * layer_idx + 1
    t = out[pos]
    n_head = t.shape[0]
    perm = torch.randperm(n_head, generator=generator)
    out[pos] = t[perm].contiguous()
    return out


def corrupt_freeze(
    state: List[torch.Tensor], earlier_snapshot: List[torch.Tensor]
) -> List[torch.Tensor]:
    """Replace WKV positions with those from an earlier snapshot.
    Non-WKV entries (shift buffers) keep their *current* values (fresh
    snapshot, not references), so the difference isolates recurrent-state
    motion between the two decode steps."""
    out = snapshot(state)
    for pos in _wkv_positions(out):
        out[pos] = earlier_snapshot[pos].detach().clone()
    return out


# --------------------------------------------------------------------------- #
# Divergence metrics
# --------------------------------------------------------------------------- #

def corrupt_scale(
    state: List[torch.Tensor], alpha: float
) -> List[torch.Tensor]:
    """Multiplicative rescale of all WKV positions by scalar ``alpha``.
    Tests whether the *magnitude* of state matters (vs direction alone).
    α<1 shrinks state, α>1 amplifies. α=1.0 = identity (no-op)."""
    out = snapshot(state)
    for pos in _wkv_positions(out):
        out[pos] = (out[pos].to(torch.float32) * alpha).to(out[pos].dtype)
    return out


def corrupt_cross(
    state: List[torch.Tensor], donor_snapshot: List[torch.Tensor]
) -> List[torch.Tensor]:
    """Replace WKV positions with those from a DIFFERENT-prompt decode's
    state at the same step index. Cleanest H8-causal-C test: if the state
    carries prompt-conditional computation, swapping in another prompt's
    state should shift the output much more than swapping in a matched
    baseline. Requires ``donor_snapshot`` to have the same layer count."""
    out = snapshot(state)
    for pos in _wkv_positions(out):
        out[pos] = donor_snapshot[pos].detach().clone()
    return out


# --------------------------------------------------------------------------- #
# Divergence metrics
# --------------------------------------------------------------------------- #

def _softmax_fp32(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.log_softmax(
        logits.reshape(-1).to(torch.float32) / temperature, dim=-1
    )


def kl_next(
    logits_a: torch.Tensor, logits_b: torch.Tensor, temperature: float = 1.0
) -> float:
    """KL(softmax(a) || softmax(b)) over a single-token logits vector.

    Computed in fp32; bf16 noise floor is well below meaningful signals
    at the expected scale but not zero, so we upcast before softmax.
    """
    log_p = _softmax_fp32(logits_a, temperature)
    log_q = _softmax_fp32(logits_b, temperature)
    p = torch.exp(log_p)
    return float((p * (log_p - log_q)).sum().item())


def argmax_flip(logits_a: torch.Tensor, logits_b: torch.Tensor) -> int:
    """1 iff the greedy top-1 token differs between the two logit vectors."""
    return int(
        torch.argmax(logits_a.reshape(-1)).item()
        != torch.argmax(logits_b.reshape(-1)).item()
    )


def entropy_change(
    logits_clean: torch.Tensor,
    logits_corrupt: torch.Tensor,
    temperature: float = 1.0,
) -> float:
    """``H(corrupt) - H(clean)`` in nats. Signed:

    - > 0 — the corruption made the distribution *more uniform* (state
      injury reduced the model's confidence).
    - < 0 — the corruption made it *sharper* (state injury pushed mass
      onto a specific alternative). Rare but diagnostic.

    Together with KL_next, entropy_change discriminates "the model
    became confused" from "the model became confidently wrong about a
    different token". KL is order-agnostic; entropy change carries
    directionality on the confidence axis."""
    log_p = _softmax_fp32(logits_clean, temperature)
    log_q = _softmax_fp32(logits_corrupt, temperature)
    H_clean = float(-(torch.exp(log_p) * log_p).sum().item())
    H_corr = float(-(torch.exp(log_q) * log_q).sum().item())
    return H_corr - H_clean


def rank_shift(logits_clean: torch.Tensor, logits_corrupt: torch.Tensor) -> int:
    """0-indexed rank of the clean top-1 token within the corrupt
    distribution's descending sort. 0 = both top-1 tokens agree (== not
    argmax_flip); larger = the corruption demoted clean's preferred
    token deep into the tail. More graded than argmax_flip."""
    clean_top = int(torch.argmax(logits_clean.reshape(-1)).item())
    corrupt = logits_corrupt.reshape(-1)
    sorted_idx = torch.argsort(corrupt, descending=True)
    rank = int((sorted_idx == clean_top).nonzero(as_tuple=False).flatten()[0].item())
    return rank


# --------------------------------------------------------------------------- #
# Trajectory-level metrics — need model.forward continuation
# --------------------------------------------------------------------------- #

def greedy_continue(
    model,
    state: List[torch.Tensor],
    initial_logits: torch.Tensor,
    n_steps: int,
) -> tuple:
    """From ``(state, initial_logits)``, greedy-decode ``n_steps`` tokens.

    Returns ``(token_ids, all_logits)`` where ``all_logits`` has length
    ``n_steps + 1`` — the initial logits plus one per generated token.

    Greedy (not sampled) so the divergence between two continuations
    comes purely from the input state, not from sampling variance.
    """
    tokens: List[int] = []
    all_logits = [initial_logits]
    cur_logits = initial_logits
    cur_state = state  # already an independent copy by caller
    for _ in range(n_steps):
        next_id = int(torch.argmax(cur_logits.reshape(-1)).item())
        tokens.append(next_id)
        cur_logits, cur_state = model.forward([next_id], cur_state)
        all_logits.append(cur_logits)
    return tokens, all_logits


def trajectory_metrics(
    tokens_clean: List[int],
    tokens_corr: List[int],
    logits_clean: list,
    logits_corr: list,
    temperature: float = 1.0,
) -> dict:
    """Given two greedy continuations from clean vs corrupt states,
    compute the trajectory-level divergences:

    - ``token_overlap_N`` — fraction of positions where the two greedy
      continuations picked the same token.
    - ``cum_KL_N`` — sum of per-step KL(softmax(clean) || softmax(corr))
      over the continuation window. Captures decay vs accumulation of
      divergence.
    """
    n = min(len(tokens_clean), len(tokens_corr))
    overlap = sum(1 for a, b in zip(tokens_clean, tokens_corr) if a == b) / max(1, n)
    cum_kl = 0.0
    for i in range(1, min(len(logits_clean), len(logits_corr))):
        cum_kl += kl_next(logits_clean[i], logits_corr[i], temperature=temperature)
    return {"token_overlap_N": overlap, "cum_KL_N": cum_kl, "n_steps": n}


__all__ = [
    "snapshot",
    "corrupt_gauss",
    "corrupt_zero_layer",
    "corrupt_zero_head",
    "corrupt_shuffle_heads",
    "corrupt_freeze",
    "corrupt_scale",
    "corrupt_cross",
    "kl_next",
    "argmax_flip",
    "entropy_change",
    "rank_shift",
    "greedy_continue",
    "trajectory_metrics",
]
