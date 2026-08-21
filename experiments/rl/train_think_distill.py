#!/usr/bin/env python3
"""train_think_distill.py — state-distillation stabilization pass for the M-loop.

Not RL (no GRPO, no reward, no sampling). Plain supervised training: for
each step9-corpus example, a *teacher* pass sees the real explicit
`<think>...</think>` text and reaches a real WKV state; a *student* pass
sees only the prompt, then takes ONE self-feed step (M=1, mirroring the
M-loop's own mechanism in wkv_loop.py::generate_rollout) and is pulled
toward the teacher's state before decoding the real answer.

Why this exists (2026-08-19): the M-loop content decoder
(_diag_think_content.py, deleted after use) showed that even a "clean"
G1i checkpoint's M-loop was only ever producing chat-template scaffolding
("\n\nAnswer") — never real task content — and this degraded further
toward contest-boilerplate under RL. There is no ground truth for what an
invisible M-loop token *should* be, so RL alone has nothing to select
for beyond "whatever the reward shapes" — that's the actual cause,
not a symptom to patch with tighter M_max. This script gives the loop
a real target: the state a genuine explicit-think pass reaches for the
same prompt. RL (β/γ/ζ) stays off until this stabilizes — see
docs/rl-track.md and project_noesis_info_density_reward memory.

Loss:  L = -log_prob(answer | student_state_after_M1)
           + l_state_weight * state_distillation_loss

state_distillation_loss = weighted L2 between student's post-M1 WKV state
and the teacher's post-think WKV state (teacher detached — gradient only
flows through the student path). Layer selection and weights are the
already-validated A0.5 set from training/state_reg.py (L12/L16/L20,
KL-profile-derived) — not re-guessed here.

M is fixed at 1 for this phase (not commit/plateau-adaptive) — a single
self-feed step is small enough that the model can plausibly learn to
treat it as "keep thinking," not "the user asked a new question,"
which is the failure mode a larger M risks at this untrained stage.

Data: training/tokenised/step9_combined_train.pt — already-built step9
SFT corpus (real <think> spans, real answers), reused as-is. Each
rollout carries `state_mask` (1 = inside <think>) and `loss_mask`
(1 = inside the assistant turn, think+answer together) — this script
splits on those exactly as combine_step9_corpus.py wrote them, no new
data generation.

Text-format drift (the model settling on *some* internal token pattern
during M's self-feed step, whatever it turns out to be) is an accepted
side effect of this training, not a supervised target — only the WKV
state is supervised, never the specific token chosen at the M-loop step.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.rl.loader import load_rwkv7
from experiments.rl.checkpoint import save_checkpoint, load_checkpoint
from experiments.rl.wkv_loop import _last_vec, _entropy_of_logits
from experiments.rl.kalman_convergence import load_kalman_watch_config
from training.state_reg import DEFAULT_WORK_LAYERS, default_layer_weights


class ThinkMarker(nn.Module):
    """A single trainable embedding-space vector marking "entering the
    self-feed phase," fed via loader.py's forward_stateful_embeds (the
    same continuous-embedding path RL's feed_mode=expected/residual
    already uses, verified end-to-end 2026-08-18 — see docs/rl-track.md).

    Added 2026-08-19 after five straight distillation runs (RFC corpus,
    G1i-native/M=1, M=2 chunked, fixed-8-token-budget full-FT, same
    budget under LoRA) all eventually hit the same failure signature —
    state_loss pinned exactly at the per-layer clamp ceiling, later with
    each fix but never prevented. Since LoRA (which cannot drift the
    frozen base at all) still failed, the cause isn't full-FT weight
    drift. Working hypothesis instead: the self-fed span has no
    structural input-side signal distinguishing "this is my own
    continued thought" from anything else a token stream could be — the
    old N mechanism (re-feed the same prompt) never had this ambiguity,
    it re-reads known content; M's self-generated content has nothing
    marking what it IS. This vector is fed once, right before the
    self-feed loop starts, so the model has an explicit, dedicated
    signal instead of relying on the state-distillation target alone to
    teach the distinction implicitly. Real vocabulary tokens were
    avoided deliberately: RWKV's tokenizer is fixed (65536 pretrained
    ids), and reusing existing text tokens (e.g. literal `<think>` as
    step9 did) risks colliding with genuine occurrences of that text in
    a real prompt — a dedicated embedding has no such collision.
    """
    def __init__(self, n_embd: int):
        super().__init__()
        self.embed = nn.Parameter(torch.randn(n_embd) * 0.02)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_examples(path: Path) -> List[Dict[str, List[int]]]:
    """Split a step9-style combined_train.pt into per-example
    prompt/think/answer token-id lists, using its own state_mask/loss_mask
    to find the boundaries (no re-derivation from text).
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    ids, sm, lm = blob["ids"], blob["state_mask"], blob["loss_mask"]
    starts = blob["starts"].tolist()

    examples = []
    for i in range(len(starts) - 1):
        s, e = starts[i], starts[i + 1]
        seq, smask, lmask = ids[s:e], sm[s:e], lm[s:e]
        think_idx = (smask == 1).nonzero(as_tuple=True)[0]
        loss_idx = (lmask == 1).nonzero(as_tuple=True)[0]
        if think_idx.numel() == 0 or loss_idx.numel() == 0:
            continue  # no think span or no supervised tokens — not usable here
        prompt_end = int(think_idx[0].item())
        think_end = int(think_idx[-1].item())
        after_think = loss_idx[loss_idx > think_end]
        if after_think.numel() == 0:
            continue  # think fills the whole loss span — no answer to teacher-force
        answer_start = int(after_think[0].item())
        examples.append({
            "prompt_ids": seq[:prompt_end].tolist(),
            "think_ids": seq[prompt_end:think_end + 1].tolist(),
            "answer_ids": seq[answer_start:].tolist(),
        })
    return examples


# --------------------------------------------------------------------------- #
# Per-example step
# --------------------------------------------------------------------------- #

def distill_step(
    loaded,
    ex: Dict[str, List[int]],
    layers: Tuple[int, ...],
    layer_weights: Dict[int, float],
    state_loss_clamp: float = 100.0,
    M: int = 1,
    max_phase_tokens: int = 4,
    think_marker: Optional["ThinkMarker"] = None,
    dynamic_phase_stop: bool = False,
    tau_commit: float = 0.9,
    eps_plateau: float = 0.05,
    norm_anchor_threshold: float = 300.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """One teacher+student forward pair. Returns (answer_ce, state_loss,
    norm_penalty, cos_sim, n_answer_tokens, n_phase_tokens_used) — caller
    combines/weights/backwards (norm_penalty is unscaled here, weighted
    by --norm-anchor-weight at the call site, same convention as
    state_loss/--l-state-weight; cos_sim is diagnostic-only, logged but
    never added to the loss — see the per-layer computation below).

    M>1 (added 2026-08-19, "latent overshooting", see Dreaming arXiv
    2007.14535's J^k_KL): the teacher's think span is sliced into M
    chunks; the student's self-feed step m is pulled toward the
    teacher's state after chunk m, not the single far-away endpoint.
    First cut used ONE self-fed token per chunk regardless of chunk
    size — still diverged (state_loss clamp-pinned by step ~321 vs
    ~287 for the M=1/single-endpoint version — delayed, not fixed).
    Root cause: 1 token is a much smaller state-update than the several
    real tokens each teacher chunk represents — the student was asked
    to close the same distance in far fewer updates. Fixed by matching
    budgets: student now self-feeds the SAME token count as the
    corresponding teacher chunk before each comparison, not a fixed 1.
    See project_noesis_think_distill_experiments memory. M=1 with a
    1-token think span reduces to the exact original behavior.

    dynamic_phase_stop (added 2026-08-21): reuses wkv_loop.py's
    generate_rollout commit/plateau exit criterion (same
    tau_commit/eps_plateau semantics, same check-before-feed order) for
    the inner self-feed loop instead of always spending the full
    max_phase_tokens — max_phase_tokens becomes a ceiling the model can
    stop short of, not a fixed count. Same principle as the EOS fix
    (let the model decide when a unit of content is done), applied one
    level up: to a *phase*, not just the final answer. Off by default
    (fixed-count behavior unchanged) — the LoRA-1950 run that produced
    the first clean (post-EOS-fix) checkpoint used the fixed-count path,
    so this needs its own isolated test before being trusted as a
    default, same "one variable at a time" discipline as everywhere
    else in this file's history.

    norm_anchor_threshold: paired with --norm-anchor-weight (main()) —
    a soft per-layer penalty on the student's *absolute* WKV state norm
    exceeding this threshold, unlike state_loss_clamp (a hard cap on the
    *distance-to-teacher*, doesn't stop the student's own magnitude from
    growing). Added after `experiments/rl/state_trajectory_probe.py`
    found the eos_full LoRA checkpoint's end-of-read state norm running
    3-4x the base model's (L20: ~135-190 base vs. 340-650 trained) —
    the default here (300.0) is an empirical starting point in that gap
    (comfortably above base, below the observed blowup range), same
    "set from measurement, not derived" convention as
    state_loss_clamp's own 100.0 and training/state_reg.py's clamps.
    """
    device = loaded.device
    prompt = torch.tensor([ex["prompt_ids"]], dtype=torch.long, device=device)
    think_ids = ex["think_ids"]
    n = len(think_ids)
    M_eff = max(1, min(M, n)) if n > 0 else 1

    # Teacher: real explicit think text, teacher-forced, incrementally —
    # capture the state after each of M_eff roughly-equal chunks. No
    # grad — these are fixed targets, not a trainable path.
    bounds = [round(i * n / M_eff) for i in range(M_eff + 1)]
    teacher_states = []
    with torch.no_grad():
        state_t = loaded.new_state(batch=1)
        _, state_t = loaded.forward_stateful(prompt, state_t)
        if think_marker is not None:
            marker = think_marker.embed.detach().to(dtype=loaded.embedding_weight.dtype).view(1, 1, -1)
            _, state_t = loaded.forward_stateful_embeds(marker, state_t)
        pos = 0
        for i in range(M_eff):
            end = min(n, max(bounds[i + 1], pos + 1))  # each chunk gets >=1 token, never past n
            chunk = think_ids[pos:end]
            pos = end
            chunk_t = torch.tensor([chunk], dtype=torch.long, device=device)
            _, state_t = loaded.forward_stateful(chunk_t, state_t)
            teacher_states.append(state_t.wkv)

    # Student: prompt, then M_eff self-feed PHASES — each phase feeds
    # back up to max_phase_tokens self-generated tokens (argmax → feed
    # back, repeated; early exit if dynamic_phase_stop, see docstring),
    # not tied to the teacher chunk's own (variable, per-example)
    # length. Deliberately generous
    # (default 8) rather than squeezed — compression/efficiency is RL's
    # job (β·M penalty + entropy-plateau exit already exist for that);
    # this stabilization phase should "unfold the chain" and reduce
    # pressure, not fight for a tight budget on top of everything else.
    # The teacher target is still the real chunk-end state; the student
    # approaches it within this budget, not necessarily reaches it
    # exactly. Same generate_rollout mechanism (argmax → feed back), no
    # sampling/reward here.
    state_s = loaded.new_state(batch=1)
    logits, state_s = loaded.forward_stateful(prompt, state_s)
    if think_marker is not None:
        marker = think_marker.embed.to(dtype=loaded.embedding_weight.dtype).view(1, 1, -1)
        logits, state_s = loaded.forward_stateful_embeds(marker, state_s)
    state_loss = torch.zeros((), device=device, dtype=torch.float32)
    norm_penalty = torch.zeros((), device=device, dtype=torch.float32)
    cos_sim_sum = torch.zeros((), device=device, dtype=torch.float32)
    n_phase_tokens_used = 0
    for i in range(M_eff):
        prev_H: Optional[float] = None
        for _ in range(max_phase_tokens):
            v = _last_vec(logits)
            if dynamic_phase_stop:
                # Same check-before-feed order as generate_rollout: a
                # token that triggers commit/plateau is never fed back —
                # the phase is judged "done" on the logits it already
                # has, not on one more (redundant) self-feed step.
                H_t = _entropy_of_logits(v)
                max_p = float(F.softmax(v.float(), dim=-1).max().item())
                if max_p > tau_commit or (prev_H is not None and abs(H_t - prev_H) < eps_plateau):
                    break
                prev_H = H_t
            next_id = int(v.argmax().item())
            step_inp = torch.tensor([[next_id]], dtype=torch.long, device=device)
            logits, state_s = loaded.forward_stateful(step_inp, state_s)
            n_phase_tokens_used += 1
        student_wkv = state_s.wkv
        teacher_wkv = teacher_states[i]
        for L in layers:
            s_flat = student_wkv[L].float().flatten()
            t_flat = teacher_wkv[L].float().detach().flatten()
            # Diagnostic only (not part of the loss) — separates *direction*
            # from *magnitude* in the state_loss L2 distance above. Added
            # 2026-08-21: after finding the student's absolute norm running
            # 3-4x the base model's, the open question was whether direction
            # also drifts or it's purely a scale effect (which norm_penalty
            # already targets). High cos_sim despite large state_loss would
            # mean "same direction, wrong scale"; low cos_sim would mean a
            # real directional divergence norm_penalty doesn't address.
            cos_sim = F.cosine_similarity(s_flat.unsqueeze(0), t_flat.unsqueeze(0)).squeeze()
            cos_sim_sum = cos_sim_sum + layer_weights[L] * cos_sim
            d = s_flat - t_flat
            # Clamp, same convention as training/state_reg.py's per-layer
            # cap (there: -10.0, "≈2× typical pretrained baseline").
            # Found 2026-08-19: an unclamped run's state_loss ran away
            # (25 → 4493 over ~10 steps, answer_ce rising in lockstep) —
            # a few outlier examples with naturally large gaps produced
            # gradient strong enough to push weights toward states that
            # diverge *further* next step. Cap set empirically from the
            # pre-blowup stable range (weighted-sum ~25-40).
            dist = torch.linalg.vector_norm(d.flatten()).clamp(max=state_loss_clamp)
            state_loss = state_loss + layer_weights[L] * dist
            # Soft anchor on the student's OWN absolute norm — see
            # docstring. Unlike `dist` above (distance to teacher, hard
            # clamped), this is unclamped-below (relu(...) is 0 while
            # under threshold, a true no-op there) and only grows past
            # the threshold, so it doesn't fight state_loss's own
            # gradient while the student is within a normal range.
            student_norm = torch.linalg.vector_norm(s_flat)
            norm_penalty = norm_penalty + layer_weights[L] * F.relu(student_norm - norm_anchor_threshold) ** 2
    state_loss = state_loss / M_eff  # keep scale comparable across M
    norm_penalty = norm_penalty / M_eff
    cos_sim_mean = cos_sim_sum / M_eff  # diagnostic only, not in the loss

    answer_ids = ex["answer_ids"]
    logits_last = logits
    if len(answer_ids) == 1:
        all_logits = logits_last
    else:
        answer_t = torch.tensor([answer_ids], dtype=torch.long, device=device)
        logits_rest, _ = loaded.forward_stateful(answer_t[:, :-1], state_s)
        all_logits = torch.cat([logits_last, logits_rest], dim=1)

    log_probs = F.log_softmax(all_logits.float(), dim=-1)
    target = torch.tensor(answer_ids, dtype=torch.long, device=device)
    token_lp = log_probs[0, torch.arange(len(answer_ids), device=device), target]
    answer_ce = -token_lp.mean()

    return answer_ce, state_loss, norm_penalty, cos_sim_mean, len(answer_ids), n_phase_tokens_used


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data", default=str(_REPO_ROOT / "training/tokenised/step9_combined_train.pt"))
    ap.add_argument("--l-state-weight", type=float, default=0.01,
                     help="Weight on the state-distillation term (λ_state). "
                          "Not swept yet — start small, watch state_loss trend "
                          "before raising (same lesson as l_state_delta_weight "
                          "in train_wkv_loop.py: 0.01 turned out too large there "
                          "for the unconditional-motion version; this is a "
                          "distillation target instead, may tolerate more, but "
                          "don't assume it without checking).")
    ap.add_argument("--work-layers", default=",".join(str(x) for x in DEFAULT_WORK_LAYERS))
    ap.add_argument("--batch", type=int, default=4, help="Examples per optimizer step (grad accum)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--grad-cp", action="store_true",
                     help="Gradient checkpointing (peft backend only) — needed for "
                          "full-FT G1i on a 16GB T4, same lesson as train_wkv_loop.py "
                          "(full-FT hit the ceiling even at G=2/batch=2/M_max=4 there).")
    ap.add_argument("--forge", action="store_true")
    ap.add_argument("--forge-offload-state", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                     help="Max gradient norm (torch.nn.utils.clip_grad_norm_) "
                          "before the optimizer step. 0 disables clipping. "
                          "Added 2026-08-20 after a full-FT run's answer_ce "
                          "spiked 2.5->18 within one step on a single hard "
                          "wordsearch example (long prompt, multi-token "
                          "answer, a task category the model has ~0%% "
                          "baseline on) — with LoRA the adapter bottleneck "
                          "(1.58%% of params) implicitly capped how far any "
                          "one example's gradient could move the model; "
                          "full-FT has no such bottleneck, so an explicit "
                          "clip is needed instead. 1.0 is a standard default, "
                          "not tuned for this run specifically.")
    ap.add_argument("--state-loss-clamp", type=float, default=100.0,
                     help="Per-layer cap on the distillation distance before "
                          "weighting — see distill_step docstring comment "
                          "(found 2026-08-19: unbounded state_loss ran away, "
                          "25→4493 over ~10 steps, dragging answer_ce up with it).")
    ap.add_argument("--M", type=int, default=1,
                     help="Number of self-feed steps, each scored against its "
                          "own slice of the teacher's think trajectory (latent "
                          "overshooting) instead of one far endpoint. M=1 is "
                          "the original single-step behavior.")
    ap.add_argument("--max-phase-tokens", type=int, default=8,
                     help="Self-fed token count per M phase — a ceiling if "
                          "--dynamic-phase-stop is on (model can stop earlier), "
                          "otherwise a fixed count. Deliberately generous "
                          "rather than squeezed either way.")
    ap.add_argument("--dynamic-phase-stop", action="store_true",
                     help="Let each M phase exit early via wkv_loop.py's own "
                          "commit/plateau criterion (see distill_step docstring) "
                          "instead of always spending the full --max-phase-tokens. "
                          "Off by default — the first clean post-EOS-fix "
                          "checkpoint used the fixed-count path; needs its own "
                          "isolated test, added 2026-08-21.")
    ap.add_argument("--tau-commit", type=float, default=0.9,
                     help="--dynamic-phase-stop only: exit a phase early if "
                          "max(softmax) exceeds this (model already confident "
                          "about the next token). Same default as generate_rollout.")
    ap.add_argument("--eps-plateau", type=float, default=0.05,
                     help="--dynamic-phase-stop only: exit a phase early if "
                          "entropy stops moving by more than this between "
                          "steps. Same default as generate_rollout.")
    ap.add_argument("--norm-anchor-weight", type=float, default=0.0,
                     help="Weight on a soft penalty for the student's own WKV "
                          "state norm exceeding --norm-anchor-threshold, added "
                          "per-layer alongside state_loss. 0 (default) = true "
                          "no-op, same convention as --l-state-weight. Added "
                          "2026-08-21 after state_trajectory_probe.py found the "
                          "eos_full checkpoint's state norm running 3-4x the "
                          "base model's (see distill_step docstring).")
    ap.add_argument("--norm-anchor-threshold", type=float, default=300.0,
                     help="--norm-anchor-weight only: per-layer norm above "
                          "which the penalty activates. Empirical starting "
                          "point, not derived — see distill_step docstring.")
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=0,
                     help="LoRA rank. 0 (default) = full-FT, all params "
                          "trainable. >0 = LoRA on receptance/key/value/output "
                          "(same target_modules as pilot_step9.yaml), base "
                          "weights frozen — added 2026-08-19 after repeated "
                          "full-FT divergence (all 3 tracked layers blowing "
                          "the state-loss clamp simultaneously by ~step 150-300 "
                          "regardless of M/phase-budget tuning — see "
                          "project_noesis_think_distill_experiments memory). "
                          "LoRA can't drift the base model at all, only its "
                          "own tiny adapter — tests whether full-FT itself "
                          "was the destabilizing factor.")
    ap.add_argument("--lora-alpha", type=int, default=0,
                     help="LoRA alpha. 0 (default) = 2*lora_r.")
    ap.add_argument("--keep-last-n", type=int, default=3,
                     help="Delete all but the last N checkpoint directories "
                          "on every save — unattended runs must not be able "
                          "to fill the disk regardless of --ckpt-every/--steps.")
    ap.add_argument("--think-marker", action="store_true",
                     help="Feed a dedicated trainable embedding (not a "
                          "vocabulary token) before the self-feed phase "
                          "starts — an explicit input-side signal for "
                          "'this is my own continued thought,' instead of "
                          "relying on the state-distillation target alone "
                          "to teach that distinction implicitly. See "
                          "ThinkMarker docstring.")
    ap.add_argument("--kalman-config", default="training/config/kalman_watch.yaml",
                     help="Path to a multi-track Kalman auto-stop config (see "
                          "training/config/kalman_watch.yaml and "
                          "experiments/rl/kalman_convergence.py's KalmanTrack "
                          "docstring) — watches state_loss/answer_ce/cos_sim (or "
                          "whatever the file lists) simultaneously, each with "
                          "its own bad-direction and consecutive-bad-streak "
                          "threshold, auto-stopping (same graceful checkpoint-"
                          "then-exit path as SIGTERM) the moment any one track "
                          "hits its streak. Deliberately a checked-in file, not "
                          "CLI flags — this is a safety mechanism for a run "
                          "that matters, not something a launch command should "
                          "be able to silently omit or mistype. Empty string "
                          "disables monitoring entirely. Added 2026-08-21, "
                          "replacing an earlier state_loss-only, "
                          "--kalman-check-every/--kalman-max-rising CLI-only "
                          "version after two runs this session ran hundreds of "
                          "steps past a real reversal before a human caught it "
                          "after the fact.")
    args = ap.parse_args()

    layers = tuple(int(x) for x in args.work_layers.split(","))
    layer_weights = default_layer_weights(layers)

    loaded = load_rwkv7(args.model, device=args.device, backend="peft",
                         grad_cp=1 if args.grad_cp else 0,
                         lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    examples = load_examples(Path(args.data))
    print(f"[distill] {len(examples)} usable examples from {args.data}")
    rng = random.Random(args.seed)

    think_marker = ThinkMarker(loaded.n_embd).to(args.device) if args.think_marker else None
    if think_marker is not None:
        print(f"[distill] think-marker enabled ({loaded.n_embd}-dim trainable embedding)")

    int8_optimizer = None
    params = [p for p in loaded.model.parameters() if p.requires_grad]
    if think_marker is not None:
        params += list(think_marker.parameters())
    all_trainable_params = list(params)  # kept separately: `params` gets
    # reassigned to int8_optimizer.other_params below when --forge is on
    # (a subset — the rest is managed inside int8_optimizer), but grad
    # clipping needs the full set regardless of which optimizer owns each.
    if args.forge:
        from experiments.rl.loader import Int8AdamW
        int8_optimizer = Int8AdamW(params, lr=args.lr, weight_decay=0.01,
                                    offload_state=args.forge_offload_state)
        params = int8_optimizer.other_params
        print("[distill] FORGE enabled (int8-optimizer-only path)")
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "distill_log.jsonl"

    global_step = 0
    if args.resume is not None:
        global_step = load_checkpoint(args.resume, loaded, mlp_delta=think_marker)
        if log_path.exists():
            kept = [l for l in log_path.read_text().splitlines()
                    if l.strip() and json.loads(l)["step"] <= global_step]
            log_path.write_text("\n".join(kept) + ("\n" if kept else ""))

    stop = {"flag": False}

    def _checkpoint():
        save_checkpoint(args.out, loaded, global_step, mlp_delta=think_marker)
        # Retention: save_checkpoint (checkpoint.py) never deletes old
        # directories on its own — every call is a brand-new ckpt_step*/
        # dir. Fine for a short interactive run someone is watching, not
        # for unattended overnight operation: an earlier full-FT run's
        # unbounded accumulation (many 5.5GB checkpoints across several
        # runs) filled the VM disk and corrupted an in-progress
        # checkpoint write (see project_noesis_forge_bptt memory).
        # LoRA+marker checkpoints are far smaller (~180MB), but "small
        # enough this time" isn't a fix — keep only the last N so this
        # class of failure can't recur regardless of run length.
        ckpts = sorted(args.out.glob("ckpt_step*"),
                        key=lambda p: p.stat().st_mtime)
        for old in ckpts[:-args.keep_last_n]:
            shutil.rmtree(old, ignore_errors=True)

    def _sigterm_handler(signum, frame):
        print(f"[distill] SIGTERM received — checkpointing at step {global_step} before exit")
        _checkpoint()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    kalman_watch = None
    if args.kalman_config:
        kalman_watch = load_kalman_watch_config(args.kalman_config)
        track_desc = ", ".join(f"{t.field}({t.bad_direction},streak{t.max_bad_streak})"
                                for t in kalman_watch["tracks"])
        print(f"[distill] kalman watch: {args.kalman_config} "
              f"check_every={kalman_watch['check_every']} tracks=[{track_desc}]")

    print(f"[distill] work_layers={layers} l_state_weight={args.l_state_weight} "
          f"batch={args.batch} device={args.device} resume_step={global_step} "
          f"dynamic_phase_stop={args.dynamic_phase_stop} norm_anchor_weight={args.norm_anchor_weight}")

    order = list(range(len(examples)))
    idx = 0
    step = global_step
    while step < args.steps:
        step += 1
        global_step = step
        optimizer.zero_grad()
        if args.forge and int8_optimizer is not None:
            int8_optimizer.zero_grad()

        ce_sum, state_sum, norm_penalty_sum, cos_sim_sum_batch, n_tok_sum, n_phase_tok_sum = 0.0, 0.0, 0.0, 0.0, 0, 0
        for _ in range(args.batch):
            if idx >= len(order):
                rng.shuffle(order)
                idx = 0
            ex = examples[order[idx]]
            idx += 1
            ce, state_loss, norm_penalty, cos_sim, n_tok, n_phase_tok = distill_step(
                loaded, ex, layers, layer_weights,
                state_loss_clamp=args.state_loss_clamp,
                M=args.M,
                max_phase_tokens=args.max_phase_tokens,
                think_marker=think_marker,
                dynamic_phase_stop=args.dynamic_phase_stop,
                tau_commit=args.tau_commit,
                eps_plateau=args.eps_plateau,
                norm_anchor_threshold=args.norm_anchor_threshold,
            )
            total = (ce + args.l_state_weight * state_loss
                     + args.norm_anchor_weight * norm_penalty) / args.batch
            total.backward()
            ce_sum += float(ce.item())
            state_sum += float(state_loss.item())
            norm_penalty_sum += float(norm_penalty.item())
            cos_sim_sum_batch += float(cos_sim.item())
            n_tok_sum += n_tok
            n_phase_tok_sum += n_phase_tok

        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable_params, args.grad_clip)
        else:
            grad_norm = None

        if args.forge and int8_optimizer is not None:
            int8_optimizer.step()
        optimizer.step()

        mean_ce = ce_sum / args.batch
        mean_state = state_sum / args.batch
        mean_norm_penalty = norm_penalty_sum / args.batch
        mean_cos_sim = cos_sim_sum_batch / args.batch
        grad_norm_str = f" grad_norm={float(grad_norm):.4f}" if grad_norm is not None else ""
        print(f"[distill] step {step}: answer_ce={mean_ce:.4f} state_loss={mean_state:.4f} "
              f"norm_penalty={mean_norm_penalty:.4f} cos_sim={mean_cos_sim:.4f} "
              f"n_answer_tok={n_tok_sum} n_phase_tok={n_phase_tok_sum}{grad_norm_str}")
        with open(log_path, "a") as f:
            f.write(json.dumps({"step": step, "answer_ce": mean_ce,
                                 "state_loss": mean_state,
                                 "norm_penalty": mean_norm_penalty,
                                 "cos_sim": mean_cos_sim,
                                 "n_phase_tok": n_phase_tok_sum,
                                 "grad_norm": float(grad_norm) if grad_norm is not None else None}) + "\n")

        if kalman_watch is not None:
            field_values = {"answer_ce": mean_ce, "state_loss": mean_state,
                             "norm_penalty": mean_norm_penalty, "cos_sim": mean_cos_sim}
            for t in kalman_watch["tracks"]:
                if t.field in field_values:
                    t.update(field_values[t.field])
            if step % kalman_watch["check_every"] == 0:
                critical = [t for t in kalman_watch["tracks"] if t.critical]
                for t in kalman_watch["tracks"]:
                    r = t.filter
                    if r is None:
                        continue
                    trend = "BAD (real)" if t.streak > 0 else "ok"
                    print(f"[distill] [kalman] step {step}: {t.field} level={r.x[0]:.4f} "
                          f"slope={r.x[1]:+.5f}/step (±{r.P[1][1] ** 0.5:.5f}) — {trend} "
                          f"(streak={t.streak}/{t.max_bad_streak})")
                if critical:
                    names = ", ".join(t.field for t in critical)
                    print(f"[distill] [kalman] CRITICAL TREND on [{names}] — stopping "
                          f"(same checkpoint-then-exit path as SIGTERM). Resume from an "
                          f"earlier checkpoint if this run's later ones aren't useful.")
                    _checkpoint()
                    return 0

        if step % args.ckpt_every == 0:
            _checkpoint()

    _checkpoint()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
