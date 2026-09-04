"""wkv_loop.py — rollout as WKV internal loop (no <think> tokens).

Flow:
    prompt_ids → prefill → WKV state
                           ↓
                       [ internal loop × M ]:
                         logits → feed → forward_stateful → new state
                           ↓  (exit criterion satisfied)
                       decode answer tokens (single commit)

Feed modes (parametrised, matches the α-slider discussion):
    "discrete":  argmax(logits) → id → forward_stateful (works on blink,
                 non-differentiable)
    "expected":  softmax(logits) @ emb.weight → forward_stateful_embeds
                 (peft-only, differentiable)
    "residual":  expected_emb + α · MLP_delta(expected_emb)
                 (peft-only, α ∈ [0, 1] — the slider from the design
                 discussion. α=0 collapses to "expected".)

Exit criterion (built-in, zero-parameter):
    exit if  |H_t − H_{t-1}| < eps_plateau  OR  max(softmax) > tau_commit
             OR  M reaches M_max

Later this can be replaced by a learned MLP gate over WKV state (H16
gate head) — the loop already exposes wkv_state at each step for that.

`generate_rollout_latent_chain` (below, separate from the above) is a
different design, not a variant of this one: model-decided marker
presence instead of tau_commit/eps_plateau — the model itself, via its
own logits, decides whether to keep generating privately or answer.
See its own docstring for the reasoning and requirements
(`loader.py::extend_vocab_for_marker` must run first).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.rl.loader import LoadedModel


@dataclass
class WKVLoopRollout:
    prompt_ids: List[int]
    answer_ids: List[int]                    # single commit
    M: int                                    # internal loops actually taken
    entropy_trajectory: List[float]           # H at each loop step (pre-exit)
    wkv_stability: List[float]                # ||W_t − W_{t-1}||_F per step
    exit_reason: str                          # "plateau" | "commit" | "M_max"
    answer_log_probs: List[float] = None      # log π_old(a_t) for GRPO ratio
    text: str = ""
    prompt_text: str = ""


def _entropy_of_logits(logits: torch.Tensor) -> float:
    """Shannon entropy of softmax(logits) — scalar (over vocab)."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    return float(-(probs * log_probs).sum().item())


def _last_vec(logits: torch.Tensor) -> torch.Tensor:
    """Reduce [B,T,V] | [T,V] | [V] to [V]."""
    if logits.dim() == 1:
        return logits
    if logits.dim() == 2:
        return logits[-1]
    return logits[0, -1]


def _wkv_delta_norm(cur: torch.Tensor, prev: Optional[torch.Tensor]) -> float:
    """Frobenius norm of (W_t − W_{t-1}) averaged over layers."""
    if prev is None:
        return 0.0
    d = (cur - prev).float()
    return float(d.norm(dim=(-2, -1)).mean().item())


def _sample_token(logits: torch.Tensor, temperature: float,
                  top_p: float = 0.9) -> int:
    """Nucleus sampling. temperature=0 → argmax."""
    v = _last_vec(logits)
    if temperature == 0.0:
        return int(v.argmax().item())
    v = v.float() / temperature
    probs = F.softmax(v, dim=-1)
    sorted_p, sorted_ids = torch.sort(probs, descending=True)
    cum = sorted_p.cumsum(0)
    cutoff = (cum - sorted_p > top_p).nonzero()
    if cutoff.numel():
        sorted_p[cutoff[0].item():] = 0.0
    sorted_p /= sorted_p.sum()
    pick = int(torch.multinomial(sorted_p, 1).item())
    return int(sorted_ids[pick].item())


def generate_rollout(
    loaded: LoadedModel,
    prompt: str,
    *,
    feed_mode: str = "discrete",             # "discrete" | "expected" | "residual"
    M_max: int = 32,
    tau_commit: float = 0.9,
    eps_plateau: float = 0.05,
    max_answer_tokens: int = 32,
    answer_temperature: float = 0.7,
    mlp_delta: Optional[nn.Module] = None,   # required for "residual"
    alpha: float = 0.0,                       # residual weight
    eos_id: int = 0,
) -> WKVLoopRollout:
    """One rollout: prefill → WKV loop → decode answer.

    All internal state (logits, embeds, wkv) is kept on the model's
    device. Numeric outputs (entropy, stability) are floats.

    Runs entirely under `torch.no_grad()`: only token ids and Python
    floats (via `.item()`) leave this function (see `WKVLoopRollout` —
    no tensors), and GRPO recomputes log π_θ separately, with gradients,
    in `train_wkv_loop.py::_recompute_wkv_log_probs`. Before this was
    added (found 2026-08-18 while chasing real-scale OOMs), every
    rollout built a full, live BPTT autograd graph across the whole
    prompt+M-loop+answer chain purely to be thrown away — the same cost
    as the later gradient recompute, paid twice per rollout for no
    reason. This was the actual dominant memory cost the whole session's
    checkpointing/gc.collect() investigation never touched, since both
    only targeted `_recompute_wkv_log_probs`.
    """
    if feed_mode not in {"discrete", "expected", "residual"}:
        raise ValueError(f"unknown feed_mode {feed_mode!r}")
    if feed_mode in {"expected", "residual"} and loaded.backend != "peft":
        raise RuntimeError(
            f"feed_mode={feed_mode!r} needs peft backend (GPU + differentiable)"
        )
    if feed_mode == "residual" and mlp_delta is None:
        raise ValueError("feed_mode='residual' requires mlp_delta module")

    tok = loaded.tokenizer
    prompt_ids = tok.encode(prompt)

    with torch.no_grad():
        # --- prefill ----------------------------------------------------
        state = loaded.new_state(batch=1)
        if loaded.backend == "peft":
            input_ids = torch.tensor([prompt_ids], dtype=torch.long,
                                     device=loaded.device)
        else:
            input_ids = prompt_ids
        logits, state = loaded.forward_stateful(input_ids, state)

        # --- internal WKV loop -------------------------------------------
        entropy_traj: List[float] = []
        stability_traj: List[float] = []
        prev_H: Optional[float] = None
        prev_wkv: Optional[torch.Tensor] = None
        exit_reason = "M_max"
        M_used = 0

        for step in range(M_max):
            # Metrics on the current logits (before feeding back)
            v = _last_vec(logits)
            H_t = _entropy_of_logits(v)
            entropy_traj.append(H_t)
            cur_wkv = loaded.wkv_stack(state)
            stability_traj.append(_wkv_delta_norm(cur_wkv, prev_wkv))
            prev_wkv = cur_wkv

            # Exit criteria
            max_p = float(F.softmax(v.float(), dim=-1).max().item())
            if max_p > tau_commit:
                exit_reason = "commit"
                break
            if prev_H is not None and abs(H_t - prev_H) < eps_plateau:
                exit_reason = "plateau"
                break
            prev_H = H_t

            # Feed next input into WKV
            if feed_mode == "discrete":
                next_id = int(v.argmax().item())
                if loaded.backend == "peft":
                    step_input = torch.tensor([[next_id]], dtype=torch.long,
                                              device=loaded.device)
                else:
                    step_input = [next_id]
                logits, state = loaded.forward_stateful(step_input, state)
            else:
                # expected or residual (peft only, differentiable)
                emb_w = loaded.embedding_weight       # [V, D]
                probs = F.softmax(v.float(), dim=-1)  # [V]
                expected = (probs.unsqueeze(0) @ emb_w.float()).to(loaded.dtype)  # [1, D]
                if feed_mode == "residual":
                    delta = mlp_delta(expected)                # [1, D]
                    feed = expected + alpha * delta
                else:
                    feed = expected
                feed = feed.unsqueeze(1)                        # [1, 1, D]
                logits, state = loaded.forward_stateful_embeds(feed, state)

            M_used = step + 1

        # --- decode answer (single commit) --------------------------------
        answer_ids: List[int] = []
        answer_log_probs: List[float] = []
        for _ in range(max_answer_tokens):
            v = _last_vec(logits)
            lp_vec = F.log_softmax(v.float(), dim=-1)
            next_id = _sample_token(v, answer_temperature)
            answer_ids.append(next_id)
            answer_log_probs.append(float(lp_vec[next_id].item()))
            if next_id == eos_id:
                break
            if loaded.backend == "peft":
                step_input = torch.tensor([[next_id]], dtype=torch.long,
                                          device=loaded.device)
            else:
                step_input = [next_id]
            logits, state = loaded.forward_stateful(step_input, state)

    assert len(answer_log_probs) == len(answer_ids), (
        f"answer_log_probs/answer_ids length mismatch: "
        f"{len(answer_log_probs)} != {len(answer_ids)}"
    )
    # `answer_ids` keeps the trailing eos_id (needed downstream —
    # _recompute_wkv_log_probs replays every id in this list, including
    # eos, to score the model's decision to stop). `text` must NOT
    # include it: found 2026-08-18 that `tok.decode([eos_id])` alone is
    # `'�'` (not a real vocabulary token, just a stop marker) — decoding
    # it as part of the string corrupted `.text` for every rollout that
    # ended in eos right after a short correct answer (exactly what
    # "Output only 0 or 1"-style prompts ask for), which in turn (a)
    # broke TrainingMonitor's MODE_COL diversity check (many distinct
    # short answers all collapsed to the same '�' string — the false
    # trigger behind two same-day emergency stops) and (b) broke
    # rewards.py's regex rubric matching against `.text` directly,
    # silently zeroing reward for correct-but-terse answers.
    text_ids = answer_ids[:-1] if answer_ids and answer_ids[-1] == eos_id else answer_ids
    # Found 2026-08-19 (answers_log.jsonl, once it started capturing real
    # text): the model sometimes spends its small max_answer_tokens
    # budget on chat-template preamble ("istant: <think>Okay, I need to
    # find...") instead of the answer — pretrained IT-corpus habit
    # leaking through, unrelated to the M-loop itself (which stays
    # latent/untokenized by design). Truncating `.text` at the leak
    # marker is the safe fix (same pattern as the eos_id strip above):
    # doesn't change what gets sampled or what answer_ids/log_probs
    # record for GRPO — just stops the leaked chat text from polluting
    # what gets scored/logged/diversity-checked. Does NOT recover an
    # answer the model never actually generated; a generation-time
    # token mask would, but that's a bigger, riskier change deliberately
    # not made here without discussing it first.
    raw_text = tok.decode(text_ids)
    think_idx = raw_text.find("<think")
    text = raw_text[:think_idx] if think_idx != -1 else raw_text
    return WKVLoopRollout(
        prompt_ids=prompt_ids,
        prompt_text=prompt,
        answer_ids=answer_ids,
        answer_log_probs=answer_log_probs,
        M=M_used,
        entropy_trajectory=entropy_traj,
        wkv_stability=stability_traj,
        exit_reason=exit_reason,
        text=text,
    )


def _feed_token(loaded: LoadedModel, token_id: int, state):
    """One ordinary token, either backend — same dispatch generate_rollout
    inlines at each of its own three call sites; factored out here since
    generate_rollout_latent_chain needs it twice (marker, chain tokens)
    plus the answer-decode loop below, and duplicating the peft/blink
    branch three times in one function invites the two copies drifting
    apart silently."""
    if loaded.backend == "peft":
        step_input = torch.tensor([[token_id]], dtype=torch.long,
                                  device=loaded.device)
    else:
        step_input = [token_id]
    return loaded.forward_stateful(step_input, state)


def generate_rollout_latent_chain(
    loaded: LoadedModel,
    prompt: str,
    *,
    marker_id: int,
    M_max: int = 8,
    chain_len: int = 16,
    chain_temperature: float = 0.8,
    max_answer_tokens: int = 32,
    answer_temperature: float = 0.7,
    eos_id: int = 0,
) -> WKVLoopRollout:
    """Marker-gated self-feed rollout: prefill -> [round loop] -> answer.

    Flow (design settled 2026-09-04, after an earlier version of this
    function's first draft imported machinery — Coconut-style continuous
    feed, tight per-round step caps — that nobody had actually asked
    for; see the two docstring notes below for what changed and why):

        prompt -> prefill -> state
                              |
        [round loop, up to M_max rounds]:
            argmax(logits) == marker_id ?
                no  -> break, fall through to answer decode
                yes -> feed marker_id (ordinary token, ordinary embedding)
                       -> generate a "thought": up to chain_len ordinary
                          SAMPLED tokens (temperature, not argmax), fed
                          back one at a time exactly like any other
                          generated text — no special mechanism, same
                          code path prompt tokens go through
                       -> back to the top of the loop
                              |
        decode answer (unchanged pattern: sampled, temperature,
        stops on eos_id)

    The model runs exactly as it always does; the only addition is that
    sometimes, after the marker, its own output goes back to it as input
    instead of out to the user — nothing about *how* it generates
    changes, and nothing bypasses the ordinary embedding lookup or the
    ordinary ffm/attention-mix weights at any step, marker or chain
    token alike.

    Structurally separate from `generate_rollout` above, not a
    `feed_mode` variant of it — this changes the EXIT logic (marker
    presence decided by the model's own logits, not
    tau_commit/eps_plateau). Kept as its own function so the four
    existing callers of `generate_rollout` (train_wkv_loop.py, eval.py,
    sweep_alpha.py, this module's own smoke test) are untouched.

    Why a real vocab token for the marker, not an injected embedding
    (ThinkChain's approach, train_think_distill.py:84-121): the model
    has to be able to natively CHOOSE to re-emit it through ordinary
    sampling for the presence-check above to mean anything — the LM
    head only assigns probability to real vocab ids. Same native
    mechanism this module already relies on for `eos_id` in the answer
    decode below. `marker_id` must come from
    `loader.py::extend_vocab_for_marker(loaded)`, called once by the
    caller before any rollout — this function does not validate that
    `marker_id` is collision-free, it trusts the caller.

    **Changed from this function's first draft, same day: dropped the
    Coconut-style continuous ("expected embedding") sub-loop and the
    K/M_max hard caps that were sized off `hypotheses/H25.md`'s toy
    divergence proof.** That proof is about a LITERALLY CONSTANT input
    fed to the recurrence every step (x_t identical, so r/k/v/w/a/g
    become constant too, giving a literal fixed affine map
    `x_{t+1}=Ax_t+b`) — a narrow precondition. Ordinary sampled
    generation doesn't meet it: each chain token's embedding differs
    (real sampling diversity), so there is no constant input for the
    argument to apply to. What CAN still degenerate a sampled loop is
    the separate, unrelated, well-documented failure of GREEDY/
    deterministic decoding collapsing into repetition — which is why
    chain tokens use `chain_temperature`-sampling here, not argmax, and
    is a solved problem (temperature/nucleus sampling), not one that
    needs a horizon-matched hard cap. `M_max`/`chain_len` remain as
    ordinary loop-hygiene bounds (don't run forever on a pathological
    input), not safety-critical values copied from an unrelated proof.

    NOT YET RUN against a real checkpoint — new code, no GPU available
    in this session to smoke-test, though (unlike the first draft) this
    version no longer requires the peft backend specifically, since it
    only feeds ordinary discrete tokens — `generate_rollout`'s
    `_smoke()` pattern (CPU/blink) should exercise this directly once
    `marker_id` exists on a real loaded checkpoint. Mirrors
    `generate_rollout`'s helpers (`_last_vec`, `_entropy_of_logits`,
    `_wkv_delta_norm`, `_sample_token`) so it reuses the same
    conventions.
    """
    tok = loaded.tokenizer
    prompt_ids = tok.encode(prompt)

    with torch.no_grad():
        state = loaded.new_state(batch=1)
        if loaded.backend == "peft":
            input_ids = torch.tensor([prompt_ids], dtype=torch.long,
                                     device=loaded.device)
        else:
            input_ids = prompt_ids
        logits, state = loaded.forward_stateful(input_ids, state)

        entropy_traj: List[float] = []
        stability_traj: List[float] = []
        prev_wkv: Optional[torch.Tensor] = None
        exit_reason = "M_max"
        rounds_used = 0

        for round_idx in range(M_max):
            v = _last_vec(logits)
            entropy_traj.append(_entropy_of_logits(v))
            cur_wkv = loaded.wkv_stack(state)
            stability_traj.append(_wkv_delta_norm(cur_wkv, prev_wkv))
            prev_wkv = cur_wkv

            if int(v.argmax().item()) != marker_id:
                exit_reason = "no_marker"
                break

            logits, state = _feed_token(loaded, marker_id, state)

            for _ in range(chain_len):
                next_id = _sample_token(logits, chain_temperature)
                logits, state = _feed_token(loaded, next_id, state)

            rounds_used = round_idx + 1
        else:
            exit_reason = "M_max"

        # --- decode answer (same pattern as generate_rollout) -------------
        answer_ids: List[int] = []
        answer_log_probs: List[float] = []
        for _ in range(max_answer_tokens):
            v = _last_vec(logits)
            lp_vec = F.log_softmax(v.float(), dim=-1)
            next_id = _sample_token(v, answer_temperature)
            answer_ids.append(next_id)
            answer_log_probs.append(float(lp_vec[next_id].item()))
            if next_id == eos_id:
                break
            logits, state = _feed_token(loaded, next_id, state)

    text_ids = answer_ids[:-1] if answer_ids and answer_ids[-1] == eos_id else answer_ids
    text = tok.decode(text_ids)

    return WKVLoopRollout(
        prompt_ids=prompt_ids,
        prompt_text=prompt,
        answer_ids=answer_ids,
        answer_log_probs=answer_log_probs,
        M=rounds_used,
        entropy_trajectory=entropy_traj,
        wkv_stability=stability_traj,
        exit_reason=exit_reason,
        text=text,
    )


# --------------------------------------------------------------------- #
# Smoke test — CPU (blink, feed_mode='discrete')
# --------------------------------------------------------------------- #

def _smoke(model_path: str,
           prompt: str = "The capital of France is",
           M_max: int = 8) -> None:
    """Run one discrete-mode rollout on CPU. Verifies loop mechanics.

    Continuous / residual modes require peft backend (GPU) — not
    exercised here. See generate_rollout docstring for full mode list.
    """
    from experiments.rl.loader import load_rwkv7
    loaded = load_rwkv7(model_path, device="cpu", backend="blink")
    r = generate_rollout(
        loaded, prompt,
        feed_mode="discrete",
        M_max=M_max,
        tau_commit=0.9,
        eps_plateau=0.02,
        max_answer_tokens=8,
        answer_temperature=0.0,
    )
    print(f"[smoke] prompt={prompt!r}")
    print(f"[smoke] M={r.M} exit={r.exit_reason}")
    print(f"[smoke] entropy_trajectory={['%.2f' % h for h in r.entropy_trajectory]}")
    print(f"[smoke] wkv_stability={['%.3f' % s for s in r.wkv_stability]}")
    print(f"[smoke] answer_ids={r.answer_ids}")
    print(f"[smoke] answer_text={r.text!r}")
