"""State-utilisation probe for RWKV-7 — skeleton.

All function bodies raise NotImplementedError. The purpose of this file
in the current commit is to fix the interface so the execution session
starts from a known API surface, not a blank page.

Design constraints (see ../../docs/state-and-reasoning.md):

- WKV state per layer is `[n_head, head_size, head_size]` — for the
  2.9B World3 checkpoint that is `[40, 64, 64]`, ~320 kB bf16 per
  layer, ~10.5 MB per token across all 32 layers.
- Do not retain full state history for a full sweep; compute metrics
  online and store only the metric time-series.
- Weights must be bf16-native; Q4 GGUF distorts state dynamics enough
  to confound H8/H9.
- WKV kernel should run in fp32 accumulator (paper §8) to match how
  the model was trained.

Reference for state layout: `RWKV-v7/rwkv_v7_demo_rnn.py` in
BlinkDL/RWKV-LM at commit 846b08c1 — lines 92–102 (forward pass access
to `state[3*i + 1]`) and 284–288 (init).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

# torch and transformers are import-only in the execution session;
# leaving them out here keeps the skeleton importable without the
# heavy stack installed.


@dataclass
class StateStep:
    """One token's worth of extracted state, with minimal metadata.

    Attributes:
        step_idx: 0-indexed position of this token in the generated
            sequence (does not include the prefix — prefix state is
            emitted separately as `step_idx = -1` if requested).
        token_id: The generated token at this step.
        wkv_per_layer: List of tensors, one per RWKV block, each shape
            `[n_head, head_size, head_size]`. Held in bf16 to bound
            memory; caller is responsible for casting before metric
            math if higher precision is required (metrics generally
            accumulate in fp32).
    """

    step_idx: int
    token_id: int
    wkv_per_layer: "list"  # list[torch.Tensor] — quoted to avoid import


def load_model(name: str, device: str = "cpu"):
    """Load a native-bf16 RWKV-7 model and its tokenizer from HF hub.

    Args:
        name: HF repo id, e.g. ``RWKV/rwkv-7-world`` (native bf16).
            GGUF paths / Ollama tags are explicitly not supported —
            quantised weights distort state dynamics.
        device: ``"cpu"`` (default) or ``"cuda"``.

    Returns:
        Tuple of ``(model, tokenizer)``. Model is loaded with
        ``torch_dtype=torch.bfloat16, trust_remote_code=True`` and set
        to eval mode.

    Raises:
        NotImplementedError: skeleton.
    """
    raise NotImplementedError("A0.4 skeleton — implement in execution session")


def generate_with_state_hooks(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    seed: int = 0,
) -> Iterator[StateStep]:
    """Generate ``max_new_tokens`` tokens; yield a StateStep per token.

    Behaviour intended for the execution session:

    1. Encode ``prompt`` with ``tokenizer``.
    2. Register forward hooks on every RWKV-7 block that capture the
       WKV state tensor after the block's forward call.
    3. Run prefill (feed the prompt) — do not emit StateSteps here;
       the prefill state is the starting point of the trajectory
       being probed.
    4. Autoregressive loop with greedy decoding under the given
       ``seed`` for reproducibility (torch.manual_seed) — even for
       greedy, sampling paths inside layers may depend on RNG for
       dropout etc.
    5. For each generated token, yield a StateStep with the
       per-layer WKV state (bf16, detached, on CPU).

    Memory hygiene: yield-and-forget. Caller is expected to feed
    StateSteps into metric functions that consume the tensor and drop
    the reference. Do not accumulate a list of StateSteps unless
    strictly required — that path is the 2.7 GB/sequence trap.

    Raises:
        NotImplementedError: skeleton.
    """
    raise NotImplementedError("A0.4 skeleton — implement in execution session")


__all__ = ["StateStep", "load_model", "generate_with_state_hooks"]
