"""State-utilisation probe for RWKV-7.

CPU-only path via BlinkDL's ``rwkv`` PyPI package (native bf16/fp32,
pure PyTorch, no triton dependency). The HuggingFace + ``fla`` path is
avoided on purpose: ``flash-linear-attention`` requires triton, which
is CUDA-only in practice — on a laptop CPU it fails at import.

The rwkv package exposes the WKV state directly as an element of the
Python ``state`` list, which makes state extraction a no-hook read:

- ``state[3*i + 0]`` — attention shift buffer, ``[n_embd]``
- ``state[3*i + 1]`` — **WKV recurrent state**, ``[n_head, head_size, head_size]``
- ``state[3*i + 2]`` — FFN shift buffer, ``[n_embd]``

For the World3 2.9B checkpoint that is ``[40, 64, 64]`` per layer,
~640 kB fp32 per layer, ~20 MB across all 32 layers in fp32. Do not
retain full state history; ``run.py`` feeds ``StateStep`` into online
metric functions and drops the reference.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

import torch

# This module is conventionally run as a bare script (`python
# experiments/A0_state_probe/some_probe.py`), which puts only this
# directory on sys.path — not the repo root. `experiments._common` needs
# the repo root importable, so ensure it's there before reaching for it.
# (`experiments/rl/*.py` scripts are invoked via `-m` from repo root and
# don't need this; this bootstrap is a no-op — path already present —
# when run that way too.)
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments._common.model import load_model  # noqa: E402,F401 — re-exported
# for backward compat: `ipc_analysis.py` and others import `load_model` from
# here. New code should import `experiments._common.model` directly.


# --------------------------------------------------------------------------- #
# StateStep
# --------------------------------------------------------------------------- #

@dataclass
class StateStep:
    """One generated token's worth of extracted state.

    Attributes:
        step_idx: 0-indexed position of this token in the generated
            sequence (prefill is not emitted).
        token_id: The generated token id.
        wkv_per_layer: List of tensors, one per RWKV block, each shape
            ``[n_head, head_size, head_size]``. fp32 on CPU.
    """

    step_idx: int
    token_id: int
    wkv_per_layer: List[torch.Tensor]


# --------------------------------------------------------------------------- #
# State extraction
# --------------------------------------------------------------------------- #

def _extract_wkv_per_layer(state: List[torch.Tensor]) -> List[torch.Tensor]:
    """Pull the WKV tensor for every RWKV-7 block from the flat state list.

    The rwkv package lays state out as a flat Python list of length
    ``3 * n_layer``. WKV lives at indices ``3*i + 1``, shape
    ``[n_head, head_size, head_size]``. We cast to fp32 CPU and detach
    so the caller can freely retain / free the tensor.
    """
    n_layer = len(state) // 3
    out: List[torch.Tensor] = []
    for i in range(n_layer):
        t = state[3 * i + 1]
        out.append(t.detach().to(torch.float32).cpu())
    return out


# --------------------------------------------------------------------------- #
# Generation loop with state emission
# --------------------------------------------------------------------------- #

def _sample_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Nucleus sampling on a single-token logits vector.

    ``logits`` is a 1-D fp32 tensor over the vocabulary. Returns the
    sampled token id. Uses ``torch.multinomial`` so the currently
    seeded ``torch.random`` state governs choice — that is what makes
    per-seed trajectories genuinely different.
    """
    logits = logits.to(torch.float32)
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / temperature, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    cutoff = (cum > top_p).nonzero(as_tuple=False)
    if cutoff.numel() > 0:
        k = int(cutoff[0].item()) + 1
        keep_probs = sorted_probs[:k]
        keep_idx = sorted_idx[:k]
    else:
        keep_probs = sorted_probs
        keep_idx = sorted_idx
    keep_probs = keep_probs / keep_probs.sum()
    choice = int(torch.multinomial(keep_probs, num_samples=1).item())
    return int(keep_idx[choice].item())


def generate_with_state_hooks(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.85,
) -> Iterator[StateStep]:
    """Prefill the prompt, then nucleus-sample ``max_new_tokens`` tokens.

    Yields ``StateStep`` per generated token. Sampled decoding with a
    fixed ``torch.manual_seed(seed)`` — different seeds walk different
    trajectories through the state space, which is what makes the
    "between-seed variance" of the state metrics a real noise-floor
    signal rather than 0 (greedy decode + deterministic model + no
    stochastic ops = identical trajectories, per-seed variance ≡ 0).

    ``temperature`` and ``top_p`` set the nucleus. Defaults match the
    interactive setting used by the noesis runtime; ``temperature=0``
    falls back to argmax if a caller needs deterministic output.

    The rwkv package's ``forward(idx, state)`` takes either a single
    token id (int) or a list of token ids (the prefill). It returns
    ``(logits, new_state)`` — we thread ``new_state`` through the loop
    and drop the reference between yields. Memory: WKV tensors are
    materialised in fp32 on CPU per token but the caller must
    consume-and-drop; nothing is retained here.
    """
    torch.manual_seed(seed)

    enc = tokenizer(prompt, return_tensors="pt")
    prompt_ids = enc["input_ids"][0].tolist()

    if verbose:
        print(
            f"[probe] prompt tokens = {len(prompt_ids)} "
            f"sampling={{T={temperature}, top_p={top_p}, seed={seed}}}"
        )

    # Prefill: feed the whole prompt at once, get initial state.
    logits, state = model.forward(prompt_ids, None)

    if verbose:
        wkv_probe = _extract_wkv_per_layer(state)
        print(
            f"[probe] state length = {len(state)}; "
            f"n_layer = {len(wkv_probe)}; "
            f"layer0.shape = {tuple(wkv_probe[0].shape) if wkv_probe else 'EMPTY'}"
        )

    for step_idx in range(max_new_tokens):
        if logits.dim() > 1:
            logits = logits.reshape(-1)
        next_id = _sample_top_p(logits, temperature=temperature, top_p=top_p)
        logits, state = model.forward([next_id], state)

        wkv_per_layer = _extract_wkv_per_layer(state)
        yield StateStep(
            step_idx=step_idx,
            token_id=next_id,
            wkv_per_layer=wkv_per_layer,
        )


__all__ = ["StateStep", "load_model", "generate_with_state_hooks"]
