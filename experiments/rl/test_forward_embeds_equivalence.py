"""Numerical equivalence test: `_peft_forward_embeds` vs `RWKV7.forward_infctx`.

Written 2026-08-17 in response to code review flagging this as the most
dangerous unverified assumption in the WKV-loop stack: `_peft_forward_embeds`
(loader.py) hand-reimplements `forward_infctx`'s block loop to accept
pre-computed embeddings (needed for `feed_mode="expected"`/`"residual"` — see
`wkv_loop.py::generate_rollout`). If it diverges from the reference, gradient
flows through the wrong computation and training looks like it's working
while learning something else. Inspection alone isn't enough — this is the
actual check.

Requires GPU (peft backend — CUDA + rwkvfla). Not runnable in this session
(CPU-only, ~3 more days per project budget as of 2026-08-17) — written so the
next GPU session can run it immediately rather than starting from a blank
page. If this fails, do not proceed with expected/residual feed_mode training
until the divergence is understood.

Two sub-tests:
  1. Single-token step equivalence — matches wkv_loop.py's actual usage
     (always T=1 per generate_rollout iteration).
  2. Multi-token (prefill-shaped) sequence equivalence — broader coverage,
     T=5, closer to how forward_infctx is used for the prompt prefill itself.

Known gap this test does NOT cover: `attention_mask` is always None here
(matches current WKV-loop usage — unpadded, batch=1 rollouts). If the loop
is ever batched with padding, `_peft_forward_embeds`'s hardcoded
`attention_mask=None` needs a real value and this test would need extending.

Run (GPU):
    /home/vaniello/Desktop/projects/noesis/training/.venv/bin/python \\
        experiments/rl/test_forward_embeds_equivalence.py \\
        --model models/rwkv7-g1i-2.9b-20260805-ctx16384.pth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from experiments.rl.loader import load_rwkv7


def _assert_close(a: torch.Tensor, b: torch.Tensor, name: str, atol: float, rtol: float) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs()
        raise AssertionError(
            f"{name} MISMATCH: max_abs_diff={diff.max().item():.6g} "
            f"mean_abs_diff={diff.mean().item():.6g} "
            f"(atol={atol}, rtol={rtol}) shape={tuple(a.shape)}"
        )
    print(f"[test]   {name}: OK (max_abs_diff={((a - b).abs().max().item()):.6g})")


def test_single_token_step(loaded, prompt_ids: list[int], atol: float, rtol: float) -> None:
    """One decode step, matching generate_rollout's actual per-step call shape."""
    print("[test] --- single-token step ---")

    # Reference path: token id -> forward_stateful (uses emb() internally).
    state_a = loaded.new_state(batch=1)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=loaded.device)
    logits_a, state_a = loaded.forward_stateful(input_ids, state_a)

    next_id = int(logits_a[0, -1].argmax().item())
    step_ids_a = torch.tensor([[next_id]], dtype=torch.long, device=loaded.device)
    logits_a2, state_a2 = loaded.forward_stateful(step_ids_a, state_a)

    # Embeds path: same next_id, looked up manually, fed via forward_stateful_embeds.
    state_b = loaded.new_state(batch=1)
    logits_b, state_b = loaded.forward_stateful(input_ids, state_b)
    assert int(logits_b[0, -1].argmax().item()) == next_id, "prefill diverged before the step under test"

    emb_w = loaded.embedding_weight
    step_embed = emb_w[next_id].unsqueeze(0).unsqueeze(0).to(loaded.dtype)  # [1, 1, D]
    logits_b2, state_b2 = loaded.forward_stateful_embeds(step_embed, state_b)

    _assert_close(logits_a2, logits_b2, "logits (single step)", atol, rtol)
    _assert_close(state_a2.shift, state_b2.shift, "shift_state (single step)", atol, rtol)
    _assert_close(state_a2.wkv, state_b2.wkv, "wkv_state (single step)", atol, rtol)


def test_multi_token_sequence(loaded, token_ids: list[int], atol: float, rtol: float) -> None:
    """T>1 in one call — closer to how forward_infctx is used for prefill."""
    print(f"[test] --- multi-token sequence (T={len(token_ids)}) ---")

    state_a = loaded.new_state(batch=1)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=loaded.device)
    logits_a, state_a = loaded.forward_stateful(input_ids, state_a)

    state_b = loaded.new_state(batch=1)
    emb_w = loaded.embedding_weight
    embeds = emb_w[torch.tensor(token_ids, device=loaded.device)].unsqueeze(0).to(loaded.dtype)  # [1, T, D]
    logits_b, state_b = loaded.forward_stateful_embeds(embeds, state_b)

    _assert_close(logits_a, logits_b, "logits (multi-token)", atol, rtol)
    _assert_close(state_a.shift, state_b.shift, "shift_state (multi-token)", atol, rtol)
    _assert_close(state_a.wkv, state_b.wkv, "wkv_state (multi-token)", atol, rtol)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--atol", type=float, default=1e-3, help="bf16 rounding needs looser tolerance than fp32")
    ap.add_argument("--rtol", type=float, default=1e-3)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This test requires the peft backend (CUDA + rwkvfla) — "
            "forward_stateful_embeds raises NotImplementedError on blink/CPU."
        )

    print(f"[test] loading {args.model} (peft backend)")
    loaded = load_rwkv7(args.model, backend="peft", device="cuda")

    tok = loaded.tokenizer
    prompt_ids = tok.encode(args.prompt)
    if len(prompt_ids) < 5:
        prompt_ids = prompt_ids * 2  # ensure enough tokens for the multi-token sub-test

    test_single_token_step(loaded, prompt_ids[:3], args.atol, args.rtol)
    test_multi_token_sequence(loaded, prompt_ids[:5], args.atol, args.rtol)

    print("[test] ALL PASS — _peft_forward_embeds matches forward_infctx within tolerance")


if __name__ == "__main__":
    main()
