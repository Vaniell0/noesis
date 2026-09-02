#!/usr/bin/env python3
"""train_wkv_loop.py — WKV-loop RL training (replaces train_wordsearch.py).

Stack:
    loader.py        — LoadedModel (peft for GPU, blink for CPU smoke)
    wkv_loop.py      — generate_rollout (no <think> tokens)
    rewards.py       — compute_wkv_loop_rewards
    grpo.py          — compute_advantages + PPO-clip surrogate
    corpus.py        — CorpusScheduler (curriculum advance/drop)
    checkpoint.py    — save_checkpoint/load_checkpoint (directory format)
    monitor.py       — TrainingMonitor (emergency stops)
    vm_watchdog.py   — WatchdogHook (24h Selectel VM deadline)
    probes.py        — run_inline_probes (stable_rank + effort frontier)

Usage (GPU, peft backend):
    python3 experiments/rl/train_wkv_loop.py \\
        --model ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \\
        --out experiments/rl/runs/wkv_loop_01 \\
        --feed-mode discrete --G 8 --M-max 16 --lr 1e-5 --steps 2000

CPU smoke (discrete only, no grad update):
    python3 experiments/rl/train_wkv_loop.py \\
        --model ~/.libs/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \\
        --out experiments/rl/runs/smoke_cpu \\
        --feed-mode discrete --G 2 --M-max 4 --steps 2 --no-update
"""
from __future__ import annotations

import argparse
import gc
import json
import signal
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from experiments.rl.corpus import load_corpus
from experiments.rl.loader import load_rwkv7, LoadedModel
from experiments.rl.wkv_loop import generate_rollout, WKVLoopRollout, _last_vec
from experiments.rl.rewards import compute_wkv_loop_rewards
from experiments.rl.grpo import compute_advantages
from experiments.rl.monitor import TrainingMonitor
from experiments.rl.vm_watchdog import VMWatchdog, WatchdogHook
from experiments.rl.probes import run_inline_probes
from experiments.rl.checkpoint import save_checkpoint, load_checkpoint
from experiments._common.results import save_result
from experiments._common.heartbeat import write_heartbeat


CORPUS_PATH = ROOT / "training/corpus_open/matrix_tasks.jsonl"


# ------------------------------------------------------------------
# Log-prob recompute (WKV-aware)

def _recompute_wkv_log_probs(
    loaded: LoadedModel,
    rollout: WKVLoopRollout,
    feed_mode: str,
    mlp_delta: Optional[torch.nn.Module],
    alpha: float,
    l_state_delta_weight: float = 0.0,
    l_state_kappa_weight: float = 0.0,
    l_state_answer_eps: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Recompute log π_θ(answer | prompt + WKV-loop) with gradients.

    Strategy: replay prefill + WKV loop (deterministic), then compute
    log-probs for answer tokens with gradients flowing through
    the peft model's parameters.

    Gradient flows through answer-token forward passes. Loop steps are
    replayed deterministically (same argmax choices); their WKV state
    influence propagates through the answer-phase forward passes.

    Per-timestep `torch.utils.checkpoint` (`use_checkpoint`, peft backend
    only): each `forward_stateful`/`forward_stateful_embeds` call
    recomputes its own forward during backward instead of keeping every
    timestep's activations alive for the whole BPTT chain.

    First tried 2026-08-18 and reverted the same day — measured WORSE
    peak memory than no checkpointing at a config that "used to succeed
    at 13.66GB." That comparison was misleading: at the time, Int8AdamW's
    optimizer state (~5.8GB) plus full-FT grad buffers (~5.9GB) plus
    weights (~5.9GB) already summed to ~17.4GB — over the 16GB T4's
    capacity regardless of activation-memory tuning, so checkpointing's
    real per-step benefit was invisible under an already-blown fixed
    budget (and the "13.66GB success" was luck: a curriculum-sampled
    rollout short enough to land under the ceiling by chance, not a
    stable operating point). `Int8AdamW(offload_state=True)` (loader.py)
    fixed the fixed-cost side the same day (weights+grad alone now
    ~11.8GB, confirmed via direct measurement), which is what makes
    checkpointing's transient per-backward-call memory (measured
    separately at ~1.8-3GB per micro-step, task-length-dependent — the
    actual proximate OOM trigger once the fixed cost was already fixed)
    worth cutting now. Re-added the same day once the fixed-cost fix
    made the transient cost the binding constraint.

    `l_state_delta_weight`/`l_state_kappa_weight` (added 2026-08-19,
    default 0.0 — grafted, not activated): revives
    `reference_noesis_loss_design`'s L_state mechanism from the old
    A1-pilot SFT stack (inverted-SFA — *reward* WKV state motion and
    curvature instead of penalizing it) for the current RL stack, as a
    direct differentiable auxiliary loss (not GRPO reward-shaping, unlike
    beta/gamma/delta/zeta — L_state doesn't need to know the eventual
    answer, so it belongs on the loss side, added straight into
    `surrogate` before `.backward()` in `wkv_grpo_loss`).

    Real trap found and avoided: `loader.py::wkv_stack()` calls
    `.detach()` on the peft path — using it here would silently produce
    a term that computes a real number but contributes ZERO gradient
    (backward through a detached tensor does nothing upstream). This
    reads `state.wkv` directly instead (peft backend only, same
    differentiable path `_step()`'s checkpointed calls already use for
    the log-probs this function returns) to keep the gradient live.

    Formula (mean over layers, not the old per-layer w_L profile — that
    was fit on the SFT-era A1 pilot model, not verified for G1i/this RL
    setup, so using it here would be false precision):
        loss -= delta_weight · ‖s(t)−s(t−1)‖ + kappa_weight · ‖s(t)−2s(t−1)+s(t−2)‖
    averaged over the M-loop + answer-decode timesteps. Only computed
    when either weight is > 0 (peft backend only) — zero when both are
    0.0, so this must not change behavior or add overhead for any
    current run. Gradient-flow through the checkpointed `state.wkv` path
    has NOT yet been verified on real GPU — do that before trusting this
    for anything beyond "present but off." See
    `project_noesis_info_density_reward` memory (renamed to cover both
    queued M-loop ideas) for the full design discussion.

    **Same tension flagged in that memory, restated as code, not just
    prose**: the existing (also-0-by-default) `delta` stability bonus in
    `rewards.py::compute_wkv_loop_rewards` rewards LOW motion; this
    rewards HIGH motion. Turning both on together without resolving the
    direction conflict would fight itself — pick one, don't stack them
    blindly.
    """
    use_checkpoint = loaded.backend == "peft"
    track_l_state = (
        loaded.backend == "peft"
        and (l_state_delta_weight > 0.0 or l_state_kappa_weight > 0.0)
    )

    def _step(fwd_fn, inp, state):
        if use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                fwd_fn, inp, state, use_reentrant=False
            )
        return fwd_fn(inp, state)

    # Reads state.wkv directly, NOT loader.py::wkv_stack() — wkv_stack
    # calls .detach() on the peft path, which would silently zero the
    # gradient through this whole term (see docstring above).
    l_state_terms: List[torch.Tensor] = []
    prev_wkv: Optional[torch.Tensor] = None
    prev_prev_wkv: Optional[torch.Tensor] = None

    def _track_l_state(state, phase_weight: float = 1.0) -> None:
        nonlocal prev_wkv, prev_prev_wkv
        if not track_l_state:
            return
        # Full [n_layer, ...] tensor, NOT reduced yet — diff first, norm
        # after, so opposite-signed changes in different layers/entries
        # don't cancel out the way they would if reduced (e.g. .mean())
        # before differencing. Per-layer w_L weighting from the old A1
        # pilot isn't used (unverified for this model), so this is a
        # single norm over the whole stacked state, not a per-layer sum.
        #
        # `phase_weight` — the old A1-pilot SFT stack's ε-mask
        # (`reference_noesis_loss_design` memory: α_eff = α·(think_frac·
        # (1−ε_out) + ε_out), ε_out=0.05) applied L_state at ~full weight
        # inside <think> spans and ~5% outside, via a state_mask tensor —
        # proven in step9 (best A1-pilot checkpoint) to stop the model
        # learning to produce WKV-useless tokens outside think spans.
        # The current M-loop design has no visible <think> spans to mask
        # (that framing is retired), but the *same* phase distinction
        # exists structurally in the code: M-loop steps vs. answer-decode
        # are already separate loops, no mask tensor needed — just weight
        # the term differently depending on which loop called this.
        wkv = state.wkv
        if prev_wkv is not None:
            delta_term = (wkv - prev_wkv).norm()
            term = -l_state_delta_weight * delta_term
            if prev_prev_wkv is not None:
                kappa_term = (wkv - 2 * prev_wkv + prev_prev_wkv).norm()
                term = term - l_state_kappa_weight * kappa_term
            l_state_terms.append(phase_weight * term)
        prev_prev_wkv, prev_wkv = prev_wkv, wkv

    state = loaded.new_state(batch=1)

    # Prefill
    if loaded.backend == "peft":
        inp = torch.tensor([rollout.prompt_ids], dtype=torch.long,
                           device=loaded.device)
    else:
        inp = rollout.prompt_ids
    logits, state = _step(loaded.forward_stateful, inp, state)
    _track_l_state(state)

    # WKV loop replay (deterministic, same choices as rollout)
    emb_w = loaded.embedding_weight if feed_mode != "discrete" else None
    for _ in range(rollout.M):
        v = _last_vec(logits)
        if feed_mode == "discrete":
            next_id = int(v.argmax().item())
            if loaded.backend == "peft":
                step_inp = torch.tensor([[next_id]], dtype=torch.long,
                                        device=loaded.device)
            else:
                step_inp = [next_id]
            logits, state = _step(loaded.forward_stateful, step_inp, state)
        else:
            probs = F.softmax(v.float(), dim=-1)
            expected = (probs.unsqueeze(0) @ emb_w.float()).to(loaded.dtype)
            if feed_mode == "residual" and mlp_delta is not None:
                expected = expected + alpha * mlp_delta(expected)
            logits, state = _step(
                loaded.forward_stateful_embeds, expected.unsqueeze(1), state
            )
        _track_l_state(state)

    # Answer tokens: compute log-probs with grad
    log_probs: List[torch.Tensor] = []
    for tok_id in rollout.answer_ids:
        v = _last_vec(logits)
        lp = F.log_softmax(v.float(), dim=-1)
        log_probs.append(lp[tok_id])
        if loaded.backend == "peft":
            step_inp = torch.tensor([[tok_id]], dtype=torch.long,
                                    device=loaded.device)
        else:
            step_inp = [tok_id]
        logits, state = _step(loaded.forward_stateful, step_inp, state)
        _track_l_state(state, phase_weight=l_state_answer_eps)

    l_state_loss = (
        torch.stack(l_state_terms).mean() if l_state_terms
        else torch.tensor(0.0, device=loaded.device)
    )

    if not log_probs:
        return torch.tensor(0.0, requires_grad=True), l_state_loss
    return torch.stack(log_probs), l_state_loss  # [T_answer], has grad


# ------------------------------------------------------------------
# GRPO loss over a batch of (rollout, reward) pairs

def wkv_grpo_loss(
    loaded: LoadedModel,
    batch_rollouts: List[List[WKVLoopRollout]],   # [n_prompts][G]
    batch_rewards: List[torch.Tensor],             # [n_prompts] each [G]
    feed_mode: str,
    mlp_delta: Optional[torch.nn.Module],
    alpha: float,
    clip_eps: float = 0.2,
    kl_coef: float = 0.01,
    forge_manager=None,
    lr: Optional[float] = None,
    l_state_delta_weight: float = 0.0,
    l_state_kappa_weight: float = 0.0,
    l_state_answer_eps: float = 0.05,
) -> float:
    """PPO-clip GRPO loss over batch. Calls `.backward()` once per (prompt,
    rollout) pair internally and returns a plain float (the mean loss, for
    logging) instead of a tensor — callers must NOT call `.backward()` again.

    Previously built one shared graph across the whole batch (every rollout's
    forward calls accumulated into one `total_loss` tensor) and called
    `.backward()` once, outside this function. That broke `--forge`: FORGE's
    FusedLinear applies its optimizer update *inside* backward(), on the
    assumption that a given layer's weight is touched by backward at most
    once per training step. Every FusedLinear layer here (att/ffn
    projections, formerly `head` too) is invoked once per generated token,
    across every rollout in the batch — with one shared backward() call,
    each of those per-token invocations independently triggers another
    premature in-place weight update before autograd reaches an earlier
    invocation's saved-for-backward tensor: "modified by an inplace
    operation ... version N; expected version 1" (confirmed 2026-08-18 on a
    real GPU — moved from `head` to an `ffn` layer after excluding `head`
    alone, i.e. not a single-layer bug).

    Fix: call `.backward()` once per rollout instead, right after computing
    its surrogate loss (scaled to match the original mean-over-all-answer-
    tokens normalisation), then let that rollout's graph be freed before the
    next one is built. This keeps every layer's per-backward-call invocation
    count at the same "once per token position in *this* rollout" scale
    `forward_infctx` itself already has — not "once per rollout, all rollouts
    sharing one call." `forge_manager.pre_step(is_accumulating=...)` is
    called once per rollout too, using FORGE's own native gradient-
    accumulation support (see `FusedOptimizerManager.pre_step`) so the
    fused optimizer only actually writes the update on the last rollout in
    the batch, not after every one.

    This is correct for the non-FORGE path too, not a FORGE-only hack:
    standard autograd accumulates `.grad` correctly across multiple
    `.backward()` calls with no `zero_grad()` between them (that's the
    ordinary gradient-accumulation pattern). It's also lower peak memory
    either way, since only one rollout's graph is ever alive at a time
    instead of the whole batch's.
    """
    micro_steps: List[Tuple["WKVLoopRollout", float]] = []
    for rollouts, rewards in zip(batch_rollouts, batch_rewards):
        advantages = compute_advantages(rewards)  # [G]
        for rollout, adv in zip(rollouts, advantages.tolist()):
            if rollout.answer_ids:
                micro_steps.append((rollout, adv))

    n_tokens_total = sum(len(r.answer_ids) for r, _ in micro_steps)
    if not micro_steps or n_tokens_total == 0:
        return 0.0

    total_loss_val = 0.0
    n_micro = len(micro_steps)
    n_skipped = 0
    for i, (rollout, adv) in enumerate(micro_steps):
        if forge_manager is not None:
            forge_manager.pre_step(lr=lr, is_accumulating=(i < n_micro - 1))

        try:
            log_pi_theta, l_state_loss = _recompute_wkv_log_probs(
                loaded, rollout, feed_mode, mlp_delta, alpha,
                l_state_delta_weight=l_state_delta_weight,
                l_state_kappa_weight=l_state_kappa_weight,
                l_state_answer_eps=l_state_answer_eps,
            )  # [T_ans], grad

            # Old log-probs (stored at rollout time)
            if rollout.answer_log_probs:
                log_pi_old = torch.tensor(
                    rollout.answer_log_probs[:len(rollout.answer_ids)],
                    dtype=torch.float32, device=log_pi_theta.device,
                )
            else:
                # answer_log_probs should always be populated by generate_rollout
                # (wkv_loop.py) for a non-empty answer — an empty/None value here
                # means rollout generation likely dropped log-probs due to a bug,
                # not a deliberate on-policy mode. ratio collapses to 1 (REINFORCE-
                # equivalent: gradient flows only through log_pi_theta), which is a
                # safe fallback but should not happen silently.
                print(f"[train] WARNING: rollout missing answer_log_probs "
                      f"({len(rollout.answer_ids)} answer tokens) — falling back "
                      f"to on-policy ratio=1", file=sys.stderr)
                log_pi_old = log_pi_theta.detach()  # on-policy fallback

            ratio = torch.exp(log_pi_theta - log_pi_old)
            adv_t = torch.full_like(ratio, adv)
            unclipped = ratio * adv_t
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
            surrogate = -torch.min(unclipped, clipped).mean()

            # KL penalty
            kl = (ratio - 1 - (log_pi_theta - log_pi_old)).mean()
            surrogate = surrogate + kl_coef * kl

            # L_state (see _recompute_wkv_log_probs docstring) — direct
            # differentiable auxiliary loss, not GRPO reward-shaping.
            # l_state_loss is exactly 0.0 (a constant, no grad path
            # built for it) when both weights are 0, so this is a true
            # no-op addition in the default/current-run case.
            surrogate = surrogate + l_state_loss

            # Weight to match the original total_loss/n_tokens_total mean —
            # each rollout contributes proportionally to its own answer length.
            weight = len(rollout.answer_ids) / n_tokens_total
            (surrogate * weight).backward()
            total_loss_val += float(surrogate.item()) * weight
        except torch.cuda.OutOfMemoryError:
            # G1i (2.9B) full-FT on a 16GB T4 runs with a thin activation
            # margin (see docs/rl-track.md, added 2026-08-18): the fixed
            # cost (weights+grad+optimizer-state) plus per-timestep
            # checkpointing already claws most of the way there, but an
            # unusually long rollout (curriculum task-length variance:
            # M near M_max together with a long pre-EOS answer) can still
            # occasionally exceed the remaining ~2-4GB budget. Rather than
            # crash an hours-long run over one unlucky rollout, drop this
            # rollout's contribution and continue — GRPO/PPO gradients are
            # already noisy-tolerant, and a skip is a minor sample-count
            # reduction for this micro-batch, not a correctness bug.
            #
            # gc.collect() before empty_cache() is load-bearing here, NOT
            # the dead end it was for the *successful*-backward case
            # earlier this session: a backward() that raises mid-computation
            # abandons `torch.utils.checkpoint`'s pack/unpack hook state and
            # the autograd engine's partially-built graph, which — same
            # root cause as before, reference cycles between checkpoint's
            # hooks and the graph nodes — CPython's refcounting won't
            # collect without a cycle-detection pass. Measured 2026-08-18:
            # without this, one real OOM cascaded into 13/16 rollouts
            # failing in the same training step (each retry saw ~10-30MB
            # free, empty_cache() alone never recovered it).
            n_skipped += 1
            print(f"[train] WARNING: OOM on rollout {i+1}/{n_micro} "
                  f"(M={rollout.M}, answer_len={len(rollout.answer_ids)}) — "
                  f"skipping, {n_skipped} skipped this step", file=sys.stderr)
            gc.collect()
            torch.cuda.empty_cache()

    return total_loss_val


# ------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feed-mode", default="discrete",
                    choices=["discrete", "expected", "residual"])
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--G", type=int, default=8, help="Rollouts per prompt")
    ap.add_argument("--batch", type=int, default=4, help="Prompts per update")
    ap.add_argument("--M-max", type=int, default=16)
    ap.add_argument("--max-answer", type=int, default=32)
    ap.add_argument("--beta", type=float, default=0.005,
                    help="Reward penalty weight on M (thinking-step count). "
                         "0.0 disables — pure correctness reward.")
    ap.add_argument("--gamma", type=float, default=0.02,
                    help="Reward penalty weight on entropy-increase (ReLU(dH)). "
                         "0.0 disables — pure correctness reward.")
    ap.add_argument("--zeta", type=float, default=0.0,
                    help="Information-density reward weight (entropy-drop "
                         "per unit WKV state motion). 0.0 (default) "
                         "disables — grafted but not activated, see "
                         "rewards.py::compute_wkv_loop_rewards docstring. "
                         "Meant to be swept, not guessed at.")
    ap.add_argument("--l-state-delta-weight", type=float, default=0.0,
                    help="L_state motion term weight (revived from the old "
                         "A1-pilot SFT stack's inverted-SFA loss — rewards "
                         "WKV state motion instead of penalizing it). 0.0 "
                         "(default) disables — grafted but not activated, "
                         "gradient-flow not yet verified on real GPU, see "
                         "_recompute_wkv_log_probs docstring.")
    ap.add_argument("--l-state-kappa-weight", type=float, default=0.0,
                    help="L_state curvature term weight (same mechanism, "
                         "second-difference of WKV state). 0.0 (default) "
                         "disables — same caveats as --l-state-delta-weight.")
    ap.add_argument("--l-state-answer-eps", type=float, default=0.05,
                    help="L_state weight during answer-decode, relative to "
                         "full weight during the M-loop (1.0). Default 0.05 "
                         "matches the old A1-pilot SFT stack's proven "
                         "ε_out — full L_state pressure inside 'think' "
                         "(the M-loop), ~5%% outside (answer-decode). Only "
                         "matters when l-state-*-weight > 0.")
    ap.add_argument("--gate-on-correct", dest="gate_on_correct",
                    action="store_true", default=True,
                    help="beta/gamma shaping only applies to already-correct "
                         "rollouts (default on) — prevents shaping from "
                         "substituting for correctness signal when accuracy "
                         "is near zero.")
    ap.add_argument("--no-gate-on-correct", dest="gate_on_correct",
                    action="store_false",
                    help="Apply beta/gamma shaping unconditionally "
                         "(pre-2026-08-19 behavior).")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--probe-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--grad-cp", action="store_true",
                    help="Gradient checkpointing (peft backend only) — recompute "
                         "block activations during backward instead of storing "
                         "the whole M-loop+answer BPTT chain. Needs deepspeed "
                         "installed (rwkvt/rwkv7/model.py's grad_cp path calls "
                         "deepspeed.checkpointing.checkpoint). Trades compute for "
                         "memory — the real lever if full-FT OOMs even at small "
                         "G/batch/M_max (found 2026-08-18: G1i full-FT was "
                         "hitting the 16GB T4 ceiling even at G=2/batch=2/M_max=4).")
    ap.add_argument("--no-update", action="store_true",
                    help="Rollout + reward only, no gradient update (CPU smoke)")
    ap.add_argument("--vm-lifetime", type=float, default=24.0,
                    help="Selectel VM lifetime in hours (default 24)")
    ap.add_argument("--resume", type=Path, default=None,
                    help="Path to a ckpt_step*/ directory to resume from (peft backend only)")
    ap.add_argument("--forge", action="store_true",
                    help="Use FORGE's (dk4248/FORGE) standalone int8-quantized "
                         "AdamW kernel (loader.py::Int8AdamW) for lower full-FT "
                         "optimizer-state VRAM. NOT FORGE's fused-into-backward "
                         "path — that assumes each layer is touched by "
                         "backward() at most once per step, which BPTT through "
                         "the WKV-loop's recurrent state structurally violates "
                         "(see project_noesis_forge_bptt.md). Int8AdamW instead "
                         "runs ordinary backward() (full BPTT, unmodified) then "
                         "applies FORGE's standalone optimizer_only_adamw_int8state() "
                         "kernel to the resulting .grad tensors — same optimizer-"
                         "state memory win, no backward()-compatibility issue.")
    ap.add_argument("--forge-state-mode", default="int8", choices=["int8", "bf16", "fp8"],
                    help="FORGE optimizer moment precision (only with --forge)")
    ap.add_argument("--forge-offload-state", action="store_true",
                    help="Keep Int8AdamW's moment state (m_q/v_q/scales) in "
                         "CPU RAM, only staging one parameter's state onto GPU "
                         "at a time during step() (frees ~5.8GB VRAM for G1i "
                         "2.9B at the cost of PCIe transfer time per param per "
                         "step). Added 2026-08-18: G1i's fixed cost (weights + "
                         "first-backward grad buffers + this state) measured "
                         "at ~17.4GB on a 16GB T4 — over capacity at ANY batch/"
                         "G/M_max/answer-length, confirmed down to the most "
                         "minimal config tried, so activation-memory tuning "
                         "alone could never close the gap.")
    ap.add_argument("--muon", action="store_true",
                    help="Use Muon (loader.py::MuonHybrid) instead of Int8AdamW "
                         "for the att/ffn hidden weight matrices — same fixed-"
                         "cost VRAM problem --forge targets, no CPU-offload "
                         "state needed (MuonHybrid keeps one momentum buffer "
                         "per param, nothing to offload). Mutually exclusive "
                         "with --forge. Precondition-checked on a CPU toy only "
                         "so far (hypotheses/H25.md 'Second follow-up') — "
                         "finetuning behavior on a trained RWKV-7 is unknown "
                         "even upstream (BlinkDL, 2026-09-02).")
    ap.add_argument("--muon-lr", type=float, default=0.02,
                    help="Muon's own published default — different scale than "
                         "--lr (tuned for Adam/AdamW), not shared on purpose.")
    ap.add_argument("--muon-momentum", type=float, default=0.95)
    ap.add_argument("--muon-momentum-start", type=float, default=0.85,
                    help="Momentum warmup start (BlinkDL's modded-nanogpt-rwkv "
                         "reference: linear ramp 0.85->--muon-momentum over "
                         "--muon-momentum-warmup-steps, not flat from step 1).")
    ap.add_argument("--muon-momentum-warmup-steps", type=int, default=500)
    ap.add_argument("--muon-weight-decay", type=float, default=0.0)
    args = ap.parse_args()
    if args.forge and args.muon:
        raise ValueError("--forge and --muon are alternatives, not both")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    answers_log_path = out_dir / "answers_log.jsonl"

    backend = "peft" if args.device != "cpu" else "blink"
    loaded = load_rwkv7(args.model, device=args.device, backend=backend,
                        grad_cp=1 if args.grad_cp else 0)

    # MLP_delta for residual mode
    mlp_delta: Optional[torch.nn.Module] = None
    if args.feed_mode == "residual":
        D = loaded.embedding_weight.shape[1]
        from experiments.rl.sweep_alpha import _MLPDelta
        mlp_delta = _MLPDelta(D).to(args.device)

    # FORGE — NOT fused into backward (see loader.py::Int8AdamW's docstring
    # for why: FORGE's fused path assumes each layer is touched by
    # backward() at most once per step, BPTT through the WKV-loop's
    # recurrent state violates that no matter how the batch loop is
    # structured). Instead: ordinary backward() (full BPTT, unmodified),
    # then FORGE's standalone int8-quantized-state AdamW kernel applied to
    # the resulting .grad tensors — same optimizer-state memory win,
    # without touching backward() or truncating gradient flow.
    int8_optimizer = None
    muon_optimizer = None
    if args.forge:
        if loaded.backend != "peft":
            raise ValueError("--forge requires --device cuda (backend='peft') — "
                              "FORGE's kernels are CUDA-only")
        print(f"[train] FORGE enabled (int8-optimizer-only path, "
              f"NOT fused-into-backward — see loader.py::Int8AdamW)")
    if args.muon and loaded.backend != "peft":
        raise ValueError("--muon requires --device cuda (backend='peft')")

    # Optimiser (only for peft backend with grad)
    optimizer = None
    if not args.no_update and loaded.backend == "peft":
        params = [p for p in loaded.model.parameters() if p.requires_grad]
        if mlp_delta:
            params += list(mlp_delta.parameters())
        if args.forge:
            from experiments.rl.loader import Int8AdamW
            int8_optimizer = Int8AdamW(params, lr=args.lr, weight_decay=0.01,
                                        offload_state=args.forge_offload_state)
            params = int8_optimizer.other_params
        elif args.muon:
            from experiments.rl.loader import MuonHybrid
            named_params = [(n, p) for n, p in loaded.model.named_parameters() if p.requires_grad]
            if mlp_delta:
                named_params += [(f"mlp_delta.{n}", p) for n, p in mlp_delta.named_parameters()]
            muon_optimizer = MuonHybrid(named_params, lr=args.muon_lr, momentum=args.muon_momentum,
                                         momentum_start=args.muon_momentum_start,
                                         momentum_warmup_steps=args.muon_momentum_warmup_steps,
                                         weight_decay=args.muon_weight_decay)
            params = muon_optimizer.other_params
            print(f"[train] Muon enabled: {len(muon_optimizer.muon_params)} hidden matrices "
                  f"on Muon (lr={args.muon_lr}), {len(params)} params on AdamW")
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    # Corpus + curriculum
    sched = load_corpus(str(CORPUS_PATH), start_level=1, rng_seed=42)

    # Monitor + watchdog
    monitor = TrainingMonitor()
    wd = VMWatchdog(lifetime_hours=args.vm_lifetime)
    wd.print_status()

    def _checkpoint():
        # --no-update means no gradient step ever ran — weights are still
        # exactly what was loaded, nothing new to persist. Was writing a
        # full checkpoint anyway on every run (smoke tests included) —
        # for a 2.9B model that's a ~4.9GB no-op write every time, found
        # 2026-08-18 when a --no-update M-baseline run's checkpoint
        # accidentally got pulled over rsync.
        if args.no_update:
            return
        save_checkpoint(out_dir, loaded, global_step, mlp_delta, sched, monitor)

    hook = WatchdogHook(wd, _checkpoint,
                        force_ckpt_hours=2.0, stop_hours=0.25,
                        check_interval_steps=50)

    global_step = 0
    if args.resume is not None:
        global_step = load_checkpoint(args.resume, loaded, mlp_delta, sched, monitor)
        # train_log.jsonl is append-only, so a run that crashes and gets
        # resumed leaves behind log lines for steps that got redone from
        # the checkpoint — same step number appearing twice with
        # different (stale) values, confusing to read back. Found
        # 2026-08-18 debugging exactly that after two resumes of the
        # same run. Trim anything past the resume point before this
        # run starts appending its own — the checkpoint is the source
        # of truth for "what actually happened," not a log line for
        # a step whose weights were never saved.
        for p in (log_path, answers_log_path):
            if p.exists():
                kept = [
                    line for line in p.read_text().splitlines()
                    if line.strip() and json.loads(line)["step"] <= global_step
                ]
                p.write_text("\n".join(kept) + ("\n" if kept else ""))

    print(f"[train] feed_mode={args.feed_mode} alpha={args.alpha} "
          f"G={args.G} M_max={args.M_max} device={args.device} "
          f"resume_step={global_step}")

    # No signal handler existed here at all before 2026-08-19 — an
    # external kill (manual, or a real Selectel preemption) bypassed
    # VMWatchdog entirely, since that only self-polls from inside the
    # running loop. Everything since the last periodic --ckpt-every save
    # was lost on every such kill this session. A checkpoint on SIGTERM
    # bounds that loss to whatever happened since the signal, not since
    # the last periodic save.
    def _sigterm_handler(signum, frame):
        print(f"[train] SIGTERM received — checkpointing at step "
              f"{global_step} before exit")
        _checkpoint()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    start_step = global_step  # 0, or the resumed checkpoint's step
    for step in range(args.steps):
        global_step = start_step + step + 1

        # Sample batch
        tasks = sched.sample_batch(args.batch)

        # Rollout: G rollouts per prompt
        batch_rollouts: List[List[WKVLoopRollout]] = []
        for task in tasks:
            group = [
                generate_rollout(
                    loaded, task["prompt"],
                    feed_mode=args.feed_mode,
                    M_max=args.M_max,
                    tau_commit=0.90,
                    eps_plateau=0.02,
                    max_answer_tokens=args.max_answer,
                    answer_temperature=0.7,
                    mlp_delta=mlp_delta,
                    alpha=args.alpha,
                    eos_id=0,
                )
                for _ in range(args.G)
            ]
            batch_rollouts.append(group)

        # Rewards
        batch_rewards: List[torch.Tensor] = []
        all_r_correct: List[torch.Tensor] = []
        for rollouts, task in zip(batch_rollouts, tasks):
            rewards, diag = compute_wkv_loop_rewards(
                rollouts, task["rubric"], beta=args.beta, gamma=args.gamma,
                zeta=args.zeta, gate_on_correct=args.gate_on_correct)
            batch_rewards.append(rewards)
            all_r_correct.append(diag["r_correct"])

        # Monitor
        all_rollouts_flat = [r for g in batch_rollouts for r in g]
        all_rewards_flat = torch.cat(batch_rewards)
        combined_diag = {"r_correct": torch.cat(all_r_correct)}
        stop, flags = monitor.step(all_rollouts_flat, all_rewards_flat, combined_diag)
        if flags:
            # Diagnostic sample so a flagged batch is actually inspectable
            # afterward — added 2026-08-18 after a MODE_COL emergency stop
            # whose actual collapsed text was never printed anywhere and
            # the model weights that produced it were lost to an unrelated
            # checkpoint-save OOM-kill, leaving nothing to diagnose with.
            print(f"[train] flagged batch sample (up to 5 of {len(all_rollouts_flat)} rollouts):",
                  file=sys.stderr)
            for r in all_rollouts_flat[:5]:
                print(f"    prompt={r.prompt_text[:80]!r} M={r.M} "
                      f"exit={r.exit_reason} text={r.text[:80]!r}", file=sys.stderr)

        # Gradient update
        loss_val = float("nan")
        if not args.no_update and optimizer is not None:
            optimizer.zero_grad()
            if int8_optimizer is not None:
                int8_optimizer.zero_grad()
            if muon_optimizer is not None:
                muon_optimizer.zero_grad()
            # wkv_grpo_loss calls .backward() internally, once per rollout
            # (not once here) — see its docstring. Ordinary autograd
            # (no forge_manager passed) — --forge no longer fuses anything
            # into backward, see loader.py::Int8AdamW's docstring for why.
            loss_val = wkv_grpo_loss(
                loaded, batch_rollouts, batch_rewards,
                feed_mode=args.feed_mode,
                mlp_delta=mlp_delta,
                alpha=args.alpha,
                l_state_delta_weight=args.l_state_delta_weight,
                l_state_kappa_weight=args.l_state_kappa_weight,
                l_state_answer_eps=args.l_state_answer_eps,
            )
            if loaded.backend == "peft":
                # Clip across every trainable param together (same threshold
                # regardless of which optimizer will consume each .grad) —
                # clipping mutates .grad in place before either optimizer
                # reads it.
                torch.nn.utils.clip_grad_norm_(
                    [p for p in loaded.model.parameters() if p.requires_grad], 1.0
                )
            optimizer.step()
            if int8_optimizer is not None:
                int8_optimizer.step()
            if muon_optimizer is not None:
                muon_optimizer.step()

        # Curriculum update
        accuracy = float((all_rewards_flat > 0).float().mean().item())
        action, cur_level = sched.update_accuracy(sched.current_level, accuracy)

        # Log
        log_entry = {
            "step": global_step,
            "loss": loss_val,
            "accuracy": accuracy,
            "current_level": cur_level,
            "curriculum_action": action,
            "mean_M": sum(r.M for r in all_rollouts_flat) / max(len(all_rollouts_flat), 1),
            "flags": flags,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        # Full rollout texts, every step — not just on a monitor flag.
        # Aggregate accuracy alone doesn't say what the model is actually
        # producing; added 2026-08-18 after wanting to see a specific
        # high-accuracy step's real answers and finding nothing was
        # logged for it (only flagged batches got a text sample before).
        with open(answers_log_path, "a") as f:
            f.write(json.dumps({
                "step": global_step,
                "rollouts": [
                    {"prompt": r.prompt_text, "text": r.text, "M": r.M,
                     "exit_reason": r.exit_reason,
                     "correct": bool(c > 0)}
                    for r, c in zip(all_rollouts_flat, combined_diag["r_correct"].tolist())
                ],
            }) + "\n")
        write_heartbeat(
            out_dir / "status.json",
            progress=(global_step, start_step + args.steps),
            step=global_step, total_planned=start_step + args.steps,
            loss=loss_val, accuracy=accuracy, current_level=cur_level,
            mean_M=log_entry["mean_M"], flags=flags,
            message=f"step {global_step}: loss={loss_val:.4f} acc={accuracy:.2%} level={cur_level}",
        )

        if global_step % 10 == 0 or flags:
            print(f"  step {global_step:5d}: loss={loss_val:.4f} acc={accuracy:.2%} "
                  f"level={cur_level} {('!!! ' + str(flags)) if flags else ''}")

        # Checkpoint — moved ahead of inline probes 2026-08-18: a probe
        # crash (real incident: stable_rank shape mismatch on the first
        # real peft-backend inline-probe run) used to take down the whole
        # process *before* reaching the checkpoint check below, silently
        # discarding every step since the last save even though training
        # itself was fine. Saving first means a probe bug costs at most
        # that one probe, never real training progress.
        if global_step % args.ckpt_every == 0:
            _checkpoint()

        # Inline probes — wrapped 2026-08-18 for the same reason: this is
        # a diagnostic side-channel, not part of the training loop proper,
        # and shouldn't be able to crash a real run (checkpoint above is
        # now safe either way, but no reason to lose the rest of the run
        # over a probe-only bug).
        if global_step % args.probe_every == 0:
            try:
                probe_result = run_inline_probes(
                    loaded, all_rollouts_flat, label=f"step{global_step}"
                )
                probe_path = out_dir / f"probes_step{global_step:06d}.json"
                save_result(
                    probe_path, probe_result,
                    experiment="rl_inline_probes", hypothesis=["H8", "H10"],
                    model=args.model, script=__file__,
                    summary={
                        "shortcut_score": f"{probe_result['shortcut_score']:.2f}",
                        "M_mean": f"{probe_result['M_mean']:.1f}",
                    },
                )
                print(f"  [probe] sr_reasoning_L4={probe_result.get('sr_reasoning_L4', float('nan')):.3f}  "
                      f"shortcut={probe_result['shortcut_score']:.2f}  "
                      f"M_mean={probe_result['M_mean']:.1f}")
            except Exception as e:
                print(f"[train] WARNING: inline probe failed at step {global_step}, "
                      f"skipping this probe only: {type(e).__name__}: {e}",
                      file=sys.stderr)

        # VM watchdog
        if hook.tick():
            print("[train] VM deadline — stopping.")
            break

        # Emergency stop
        if stop:
            print(f"[train] Emergency stop: {flags}")
            _checkpoint()
            break

    _checkpoint()
    print(f"\n[train] done. {global_step} steps → {out_dir}")


if __name__ == "__main__":
    main()
