#!/usr/bin/env python3
"""train_think_distill.py — state-distillation stabilization pass for the M-loop.

Not RL (no GRPO, no reward, no sampling). Plain supervised training: for
each step9-corpus example, a *teacher* pass sees the real explicit
`<think>...</think>` text and reaches a real WKV state; a *student* pass
sees only the prompt, then runs M ThinkChain phases (--think-marker,
2026-08-21 — M_eff explicitly-distinct learned embeddings, each fed
straight into WKV, budget-matched to its teacher chunk's real token
count; see ThinkChain/distill_step docstrings) and is pulled toward the
teacher's per-phase states before decoding the real answer. No self-feed
loop — this no longer mirrors wkv_loop.py::generate_rollout's mechanism
(that script's discrete/expected/residual feed modes are all still the
old homogeneous self-feed loop; porting ThinkChain there is separate,
not-yet-done work, see project_noesis_rl_track memory).

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

Loss:  L = -log_prob(answer | student_state_after_M_phases)
           + l_state_weight * state_distillation_loss

state_distillation_loss = weighted L2 between student's post-phase WKV
states and the teacher's post-chunk WKV states, summed over phases
(teacher detached — gradient only flows through the student path).
Layer selection and weights are the already-validated A0.5 set from
training/state_reg.py (L12/L16/L20, KL-profile-derived) — not
re-guessed here.

M is a launch choice (--M), not fixed at 1 — ThinkChain's explicitly-
distinct per-phase markers were built specifically so a larger M no
longer risks the "self-feed loop can't tell phase i from i+1" collapse
that motivated keeping M=1 under the old self-feed mechanism (see
ThinkChain docstring below).

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
import gc
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
from experiments.rl.kalman_convergence import load_kalman_watch_config
from training.state_reg import DEFAULT_WORK_LAYERS, default_layer_weights


class ThinkChain(nn.Module):
    """N+1 distinct trainable embedding-space vectors: step(0) is the
    shared "entering think-mode" cue (fed once to both teacher and
    student, same role the old single ThinkMarker played); step(1..M)
    are per-PHASE markers fed to the student only, one per phase,
    directly via loader.py's forward_stateful_embeds — no self-feed
    token loop.

    Replaces ThinkMarker + the argmax self-feed loop (2026-08-21,
    corrected after review: the original design generated the student's
    per-phase signal by argmax-sampling its own logits and feeding that
    token back, repeated up to `max_phase_tokens` times per phase — a
    homogeneous LOOP applying one transformation to itself repeatedly.
    Two problems: (1) discrete argmax collapses the full logit
    distribution to one token id every step, throwing away information
    at each iteration; (2) as the model's self-generated tokens grew
    more confident/similar, the R/K/V/decay computed from them converged
    toward near-identical transformations step to step — a loop with no
    per-step differentiating signal drifts toward a fixed point rather
    than doing N genuinely different units of work (matches what was
    observed empirically: dynamic-phase-stop's entropy-plateau exit
    fired increasingly early, state norm needed an explicit anchor to
    keep from drifting unboundedly — both are loop-collapse symptoms,
    not independent bugs). The teacher side never had this problem: it
    reads M_eff real, distinct text chunks, each naturally different
    content. ThinkChain gives the student the same property the teacher
    gets for free — M_eff EXPLICITLY DISTINCT, directly-learned signals,
    one per phase, instead of one signal reused in a self-similar loop.

    Original ThinkMarker rationale (why an embedding, not a vocab
    token) still applies unchanged: RWKV's tokenizer is fixed (65536
    pretrained ids), and reusing existing text tokens risks colliding
    with genuine occurrences of that text in a real prompt — a
    dedicated embedding has no such collision.
    """
    def __init__(self, n_embd: int, n_phases: int):
        super().__init__()
        self.chain = nn.Parameter(torch.randn(n_phases + 1, n_embd) * 0.02)

    def step(self, i: int) -> torch.Tensor:
        return self.chain[i]


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


# Best-effort task-category tags from prompt text, checked in order (first
# match wins — more specific keywords before generic ones). Not exhaustive
# (g1i_warmup_v3's full category list was never enumerated, only sampled) —
# good enough for batch-diversity purposes, where an imperfect split still
# beats no split. Added 2026-08-22 after finding step9_combined_train.pt
# was 85% one answer-template (see --data default's help text) and a batch
# could silently be all-one-category even under a full corpus shuffle.
_CATEGORY_KEYWORDS = [
    ("crossword", "xword"),
    ("Find the word", "wordsearch"),
    ("bitwise XOR", "xor"),
    ("column addition", "arithmetic"),
    ("next number in this sequence", "sequence"),
    ("letter matrix", "matrix"),
    ("Decimal", "bits"),
    ("Binary", "bits"),
]


def _infer_category(prompt_text: str) -> str:
    for keyword, cat in _CATEGORY_KEYWORDS:
        if keyword in prompt_text:
            return cat
    return "other"


class _CategoryBatcher:
    """Cycles through categories in shuffled round-robin order so a batch
    draws from as many DISTINCT categories as possible, instead of a flat
    shuffle-and-walk that can (and did — see _infer_category above) land
    a whole batch in one category on a skewed corpus. Within a category,
    also cycles its own shuffled order (no example repeats until that
    category's pool is exhausted). If batch size > number of categories,
    later picks in a batch necessarily repeat a category — unavoidable,
    not a bug."""
    def __init__(self, examples: List[Dict], categories: List[str], rng: random.Random):
        self.examples = examples
        self.rng = rng
        self.by_cat: Dict[str, List[int]] = {}
        for i, c in enumerate(categories):
            self.by_cat.setdefault(c, []).append(i)
        self.cat_order: List[str] = list(self.by_cat)
        self.rng.shuffle(self.cat_order)
        self.cat_pos = 0
        self.within_pos: Dict[str, int] = {c: 0 for c in self.by_cat}
        for idxs in self.by_cat.values():
            self.rng.shuffle(idxs)

    def next(self) -> Dict:
        if self.cat_pos >= len(self.cat_order):
            self.cat_pos = 0
            self.rng.shuffle(self.cat_order)
        cat = self.cat_order[self.cat_pos]
        self.cat_pos += 1
        idxs = self.by_cat[cat]
        pos = self.within_pos[cat]
        if pos >= len(idxs):
            self.rng.shuffle(idxs)
            pos = 0
        self.within_pos[cat] = pos + 1
        return self.examples[idxs[pos]]


# --------------------------------------------------------------------------- #
# CLIPO-style in-batch contrastive term
# --------------------------------------------------------------------------- #

def _clipo_contrastive_loss(student_reprs: List[torch.Tensor],
                             teacher_reprs: List[torch.Tensor],
                             tau: float = 0.05) -> torch.Tensor:
    """In-batch InfoNCE: each example's student state should be closer to
    ITS OWN teacher target than to any OTHER example's target in the same
    batch. Same tau-scaled cosine-similarity InfoNCE machinery as
    rewards.py's `_infonce_reward`, but a genuinely different contrast
    axis — that one contrasts correct-vs-incorrect ROLLOUTS (needs GRPO
    sampling, doesn't exist in this supervised-distillation script, every
    training example here is a "correct" teacher demonstration by
    construction); this one contrasts DIFFERENT EXAMPLES within a batch,
    the standard batch-negatives InfoNCE used in CLIP/SimCLR-style
    self-supervised training.

    Why this addresses the overfitting-to-narrow-answer-style concern
    raised 2026-08-21 (see docs/rl-track.md): a plain per-example L2
    target (state_loss) is satisfied by a shortcut that reproduces THIS
    example's target closely, with no pressure for the representation to
    be discriminative — nothing stops student states from different
    examples collapsing toward a similar region if that happens to lower
    the average L2 distance. Forcing student_i to be identifiably closer
    to teacher_i than to teacher_j (all j != i in the batch) requires a
    representation that actually encodes what's specific to each example,
    not just "close enough on average" — needs batch >= 2 to have any
    negatives at all; a batch of 1 has nothing to contrast against.

    Symmetric InfoNCE via cross-entropy (the CLIP loss trick): similarity
    matrix's diagonal (student_i vs teacher_i) should be the row-wise
    argmax after softmax — equivalent to the explicit logsumexp form
    `_infonce_reward` uses, cheaper to write batched.
    """
    S = F.normalize(torch.stack(student_reprs), dim=-1)          # [B, D], grad-tracked
    T = F.normalize(torch.stack([t.detach() for t in teacher_reprs]), dim=-1)  # [B, D], fixed targets
    sim = (S @ T.T) / tau                                        # [B, B]
    targets = torch.arange(sim.shape[0], device=sim.device)
    return F.cross_entropy(sim, targets)


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
    think_marker: Optional["ThinkChain"] = None,
    norm_anchor_threshold: float = 300.0,
    dynamic_phase_stop: bool = False,
    eps_plateau: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """One teacher+student forward pair. Returns (answer_ce, state_loss,
    norm_penalty, cos_sim, student_repr, teacher_repr, n_answer_tokens,
    n_phase_tokens_used) — caller combines/weights/backwards (norm_penalty is unscaled here,
    weighted by --norm-anchor-weight at the call site, same convention
    as state_loss/--l-state-weight; cos_sim is diagnostic-only, logged
    but never added to the loss).

    student_repr/teacher_repr (added 2026-08-21, for the CLIPO-style
    in-batch contrastive term — see _clipo_contrastive_loss): the final
    phase's student_wkv / teacher_wkv, flattened and concatenated across
    `layers`, gradient-tracked for student (detached for teacher, same
    as state_loss's target). NOT combined into a loss here — a
    contrastive term needs multiple examples' representations in the
    same computational context (this example's target vs. OTHER
    examples in the batch as negatives), which a single distill_step
    call can't see; the caller collects these across a whole batch
    before computing it.

    M>1 ("latent overshooting", see Dreaming arXiv 2007.14535's J^k_KL):
    the teacher's think span is sliced into M chunks; the student's
    phase i is pulled toward the teacher's state after chunk i, not the
    single far-away endpoint.

    2026-08-21 rewrite: the student's per-phase signal used to come
    from an argmax self-feed LOOP (generate a token from its own
    logits, feed it back, repeat up to a token budget) — a single
    transformation applied to itself repeatedly, with no signal telling
    the model which phase (of M) it was in. That converges toward a
    fixed point rather than doing M genuinely different units of work:
    as the model's self-generated tokens grow more confident, the
    R/K/V/decay computed from them become more similar step to step,
    so the loop drifts toward self-similarity instead of tracking M
    distinct targets. (Symptoms this produced, in hindsight: the
    dynamic-phase-stop entropy-plateau exit fired increasingly early,
    and the student's absolute state norm needed an explicit anchor to
    keep from drifting unboundedly — both are loop-collapse signatures.)
    Replaced with ThinkChain: M_eff EXPLICITLY DISTINCT, directly
    learned per-phase markers, fed straight into WKV, no self-feed loop
    at all. Mirrors what the teacher already has for free — M_eff real,
    naturally-distinct text chunks — instead of asking the student to
    manufacture that distinctness from a homogeneous loop.

    2026-08-22 fix (the 08-21 rewrite was incomplete, not wrong): each
    phase now feeds its marker REPEATED chunk_lens[i] times (budget-
    matched to the teacher chunk it's scored against) instead of a
    single fixed step — the 08-21 version silently dropped budget-
    matching, a fix this exact file's history already needed once
    before for the old self-feed loop (see the M=2/1-token-budget
    divergence in `project_noesis_think_distill_experiments` memory).
    Repeating a CONSTANT embedding isn't the loop-collapse failure mode
    above: R/K/V/decay/a/g derive from `x` and `time_shift(x)-x`, and
    the latter is exactly zero from the second repeat onward (identical
    consecutive inputs) — a fixed transformation applied T times to the
    evolving state, not a self-referential one that drifts as its own
    input grows more homogeneous. chunk_lens[i] is a ceiling on that
    repeat count, not mandatory — --dynamic-phase-stop exits early once
    the WKV state itself stops moving (relative delta vs. --eps-plateau),
    not once the phase's readout is confident (an earlier same-day version
    borrowed generate_rollout's confidence check verbatim and found it
    fires almost immediately on g1i_warmup_v3's templated answer format —
    see --dynamic-phase-stop's help text).

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
    chunk_lens: List[int] = []  # real per-chunk token count — reused below
    # so the student's phase gets the SAME step budget as the teacher chunk
    # it's compared against, not a fixed 1. Restores the budget-matching
    # fix from the pre-ThinkChain self-feed loop (see M>1 docstring section
    # above) that the 2026-08-21 rewrite dropped: each phase there was a
    # single forward_stateful_embeds call (T=1) regardless of chunk length,
    # asking one learned vector's one-step WKV update to land wherever the
    # teacher's chunk_lens[i] real tokens landed it.
    with torch.no_grad():
        state_t = loaded.new_state(batch=1)
        _, state_t = loaded.forward_stateful(prompt, state_t)
        if think_marker is not None:
            marker = think_marker.step(0).detach().to(dtype=loaded.embedding_weight.dtype).view(1, 1, -1)
            _, state_t = loaded.forward_stateful_embeds(marker, state_t)
        pos = 0
        for i in range(M_eff):
            end = min(n, max(bounds[i + 1], pos + 1))  # each chunk gets >=1 token, never past n
            chunk = think_ids[pos:end]
            pos = end
            chunk_lens.append(len(chunk))
            chunk_t = torch.tensor([chunk], dtype=torch.long, device=device)
            _, state_t = loaded.forward_stateful(chunk_t, state_t)
            teacher_states.append(state_t.wkv)

    # Student: prompt, then M_eff PHASES — each phase feeds its own
    # distinct, directly-learned marker (think_marker.step(i+1)) via
    # forward_stateful_embeds, REPEATED up to chunk_lens[i] times
    # (budget-matched to the teacher chunk it's scored against; a
    # ceiling, not mandatory — see --dynamic-phase-stop below). One
    # call per repeat (not a single batched [B,T,D] call) specifically
    # so an early-exit check can run between repeats. Repeating a
    # CONSTANT embedding is not the old loop-collapse failure mode:
    # R/K/V/decay/a/g are derived from `x` and `time_shift(x) - x`, and
    # the latter is exactly zero from the second repeat onward
    # (identical consecutive inputs) — so this is a fixed, well-defined
    # transformation applied T times to the evolving WKV state (like a
    # constant decay rate applied for T steps), not a self-referential
    # loop whose own transformation drifts as its input grows more
    # confident/homogeneous (see distill_step's M>1 docstring section
    # for why THAT was the actual collapse mechanism). Mirrors the
    # teacher's M_eff real, naturally-distinct chunks with M_eff
    # explicitly-distinct learned signals, each given the same real budget.
    state_s = loaded.new_state(batch=1)
    logits, state_s = loaded.forward_stateful(prompt, state_s)
    if think_marker is not None:
        marker = think_marker.step(0).to(dtype=loaded.embedding_weight.dtype).view(1, 1, -1)
        logits, state_s = loaded.forward_stateful_embeds(marker, state_s)
    state_loss = torch.zeros((), device=device, dtype=torch.float32)
    norm_penalty = torch.zeros((), device=device, dtype=torch.float32)
    cos_sim_sum = torch.zeros((), device=device, dtype=torch.float32)
    last_student_parts: List[torch.Tensor] = []
    last_teacher_parts: List[torch.Tensor] = []
    n_phase_tokens_used = 0
    for i in range(M_eff):
        if think_marker is not None:
            phase_marker = think_marker.step(i + 1).to(dtype=loaded.embedding_weight.dtype).view(1, 1, -1)
            # chunk_lens[i] is a CEILING on this phase's repeat budget, not
            # a mandatory count. No intermediate teacher target is needed
            # for an early exit: the loss below still compares whatever
            # state the student reaches against the teacher's end-of-chunk
            # target regardless of how many repeats it took to get there.
            #
            # 2026-08-22, second correction: the first version of this
            # criterion (re-added same day) borrowed wkv_loop.py::
            # generate_rollout's readout-confidence check (max_p/entropy on
            # the phase's logits) verbatim. Real run diagnosis found it
            # fires almost always after exactly 1 repeat (157/~200 logged
            # steps had n_phase_tok == batch size) — traced to the actual
            # cause: g1i_warmup_v3's answer format is templated (e.g.
            # "Decimal: ..."), so the readout is >99.9% confident about the
            # first ANSWER TOKEN almost immediately, regardless of whether
            # the phase has done any real work — generate_rollout's
            # confidence check is valid THERE because it inspects the
            # actual next token about to be generated in a real answer
            # stream; here the phase readout is a side artifact with no
            # such meaning. Replaced with a criterion that checks the
            # THING the phase is actually supposed to be doing: has the
            # WKV state (not the incidental readout) stopped moving,
            # relative to its own scale. `eps_plateau` is now a relative
            # threshold (fraction of current state norm), not an entropy
            # difference — `tau_commit` had no analogous meaning to
            # preserve and was removed.
            prev_phase_wkv: Optional[Dict[int, torch.Tensor]] = None
            for _ in range(chunk_lens[i]):
                logits, state_s = loaded.forward_stateful_embeds(phase_marker, state_s)
                n_phase_tokens_used += 1
                if dynamic_phase_stop:
                    cur_phase_wkv = {L: state_s.wkv[L] for L in layers}
                    if prev_phase_wkv is not None:
                        delta_norm = sum(
                            torch.linalg.vector_norm((cur_phase_wkv[L].float() - prev_phase_wkv[L].float()).flatten()).item()
                            for L in layers)
                        ref_norm = sum(
                            torch.linalg.vector_norm(cur_phase_wkv[L].float().flatten()).item()
                            for L in layers)
                        if delta_norm < eps_plateau * max(ref_norm, 1e-6):
                            break
                    prev_phase_wkv = cur_phase_wkv
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
            if i == M_eff - 1:
                last_student_parts.append(s_flat)
                last_teacher_parts.append(t_flat)
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
    student_repr = torch.cat(last_student_parts)
    teacher_repr = torch.cat(last_teacher_parts)

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

    return (answer_ce, state_loss, norm_penalty, cos_sim_mean, student_repr, teacher_repr,
            len(answer_ids), n_phase_tokens_used)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data", default=str(_REPO_ROOT / "training/tokenised/g1i_warmup_v3_eos_train.pt"),
                     help="Tokenised corpus (see load_examples). Default switched 2026-08-22 "
                          "from step9_combined_train.pt (268 examples, 85%% share one answer-"
                          "template first token — real, verified skew, not a sampling artifact) "
                          "to g1i_warmup_v3_eos_train.pt (10529 examples, 1409 unique first "
                          "words, matrix_tasks.jsonl-derived, already has the EOS fix).")
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
                     help="Number of ThinkChain phases (needs --think-marker to "
                          "mean anything — without it there's no per-phase "
                          "signal at all). Each phase is scored against its own "
                          "slice of the teacher's think trajectory (latent "
                          "overshooting), budget-matched to that slice's real "
                          "token count (a ceiling — see --dynamic-phase-stop). "
                          "M=1 collapses to a single phase.")
    ap.add_argument("--dynamic-phase-stop", action="store_true",
                     help="Let each phase exit its repeat budget early once the "
                          "WKV state itself stops moving (see --eps-plateau), "
                          "instead of always spending the full budget-matched "
                          "chunk_lens[i] repeats. 2026-08-22, second version: "
                          "the first version checked readout-logit confidence "
                          "(borrowed from generate_rollout) — found to fire "
                          "almost always after exactly 1 repeat on "
                          "g1i_warmup_v3, because that corpus's templated "
                          "answer format (e.g. 'Decimal: ...') makes the FIRST "
                          "answer token >99.9%% predictable regardless of "
                          "whether the phase did any real work. Readout "
                          "confidence answers a question that only makes sense "
                          "for a real generated token stream (generate_rollout's "
                          "context); this phase's readout is a side artifact. "
                          "State-motion is the thing the phase is actually "
                          "supposed to be doing.")
    ap.add_argument("--eps-plateau", type=float, default=0.05,
                     help="--dynamic-phase-stop only: exit a phase early once "
                          "the WKV state's move this repeat is smaller than "
                          "this fraction of its own current norm (relative, "
                          "not absolute — so it scales with however large the "
                          "state has grown, e.g. from --norm-anchor-weight "
                          "drift).")
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
    ap.add_argument("--clipo-weight", type=float, default=0.0,
                     help="Weight on the CLIPO-style in-batch contrastive term "
                          "(_clipo_contrastive_loss) — pulls each example's "
                          "student state toward ITS OWN teacher target and "
                          "away from other examples' targets in the same "
                          "batch. 0 (default) = true no-op, same convention "
                          "as --l-state-weight/--norm-anchor-weight. Needs "
                          "--batch >= 2 to have any negatives to contrast "
                          "against — silently skipped (a 0.0 term) on smaller "
                          "batches regardless of this weight. Added "
                          "2026-08-21 — see _clipo_contrastive_loss docstring "
                          "for why this needed batch>1 and a restructured "
                          "loop (every example's forward pass now stays in "
                          "the graph until one combined backward, instead of "
                          "each example calling .backward() immediately).")
    ap.add_argument("--clipo-tau", type=float, default=0.05,
                     help="--clipo-weight only: InfoNCE temperature. Same "
                          "default as rewards.py's _infonce_reward.")
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
                     help="Feed M+1 dedicated trainable embeddings (not "
                          "vocabulary tokens): one shared 'entering think "
                          "mode' cue, then one distinct per-phase marker for "
                          "each of the M phases, fed straight into WKV — no "
                          "self-feed token loop. Required for --M > 1 to be "
                          "meaningful (without it the student has no signal "
                          "distinguishing phase i from phase i+1 at all). "
                          "See ThinkChain docstring.")
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

    categories = [_infer_category(loaded.tokenizer.decode(ex["prompt_ids"])) for ex in examples]
    cat_counts = {c: categories.count(c) for c in sorted(set(categories))}
    print(f"[distill] category split (best-effort, see _infer_category): {cat_counts}")
    batcher = _CategoryBatcher(examples, categories, rng)

    think_marker = ThinkChain(loaded.n_embd, args.M).to(args.device) if args.think_marker else None
    if think_marker is not None:
        print(f"[distill] think-chain enabled ({loaded.n_embd}-dim, {args.M + 1} distinct markers: 1 entry + {args.M} phase)")

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
          f"M={args.M} norm_anchor_weight={args.norm_anchor_weight} "
          f"dynamic_phase_stop={args.dynamic_phase_stop}")

    step = global_step
    while step < args.steps:
        step += 1
        global_step = step
        optimizer.zero_grad()
        if args.forge and int8_optimizer is not None:
            int8_optimizer.zero_grad()

        # Collect the whole batch's forward passes BEFORE any backward —
        # required for _clipo_contrastive_loss, which needs every
        # example's student_repr/teacher_repr alive in the same
        # computational context (each example used to call .backward()
        # immediately and free its graph, which a cross-example
        # contrastive term can't work with). Real cost: peak memory now
        # holds args.batch forward graphs simultaneously instead of one
        # at a time — watch VRAM if raising --batch on this VM.
        ce_list, state_list, norm_penalty_list, cos_sim_list = [], [], [], []
        student_repr_list, teacher_repr_list = [], []
        n_tok_sum = n_phase_tok_sum = n_skipped = 0
        for _ in range(args.batch):
            ex = batcher.next()
            try:
                ce, state_loss, norm_penalty, cos_sim, student_repr, teacher_repr, n_tok, n_phase_tok = distill_step(
                    loaded, ex, layers, layer_weights,
                    state_loss_clamp=args.state_loss_clamp,
                    M=args.M,
                    think_marker=think_marker,
                    norm_anchor_threshold=args.norm_anchor_threshold,
                    dynamic_phase_stop=args.dynamic_phase_stop,
                    eps_plateau=args.eps_plateau,
                )
            except torch.cuda.OutOfMemoryError:
                # Same pattern as train_wkv_loop.py's per-rollout OOM
                # catch-and-skip (added 2026-08-18 there after one OOM
                # cascaded into 13/16 rollouts failing the same step) —
                # this loop holds args.batch forward graphs simultaneously
                # (see comment above, for CLIPO), so an unusually long
                # example (a long chunk_lens[i] repeat budget under
                # --dynamic-phase-stop, or a long think-span) can push an
                # otherwise-fine batch over the edge. Drop this example,
                # keep the step going with whatever the rest of the batch
                # produces, rather than crash an hours-long run over one
                # example. gc.collect() before empty_cache() — same
                # ordering, same reference-cycle reasoning as
                # train_wkv_loop.py (no torch.utils.checkpoint hooks here,
                # but harmless and consistent to keep the order anyway).
                n_skipped += 1
                print(f"[distill] WARNING: OOM on an example this batch — skipping, "
                      f"{n_skipped} skipped this step", file=sys.stderr)
                gc.collect()
                torch.cuda.empty_cache()
                continue
            ce_list.append(ce)
            state_list.append(state_loss)
            norm_penalty_list.append(norm_penalty)
            cos_sim_list.append(cos_sim)
            student_repr_list.append(student_repr)
            teacher_repr_list.append(teacher_repr)
            n_tok_sum += n_tok
            n_phase_tok_sum += n_phase_tok

        if not ce_list:
            print(f"[distill] step {step}: entire batch OOM'd, skipping this "
                  f"optimizer step entirely", file=sys.stderr)
            continue

        ce_t = torch.stack(ce_list).mean()
        state_t = torch.stack(state_list).mean()
        norm_penalty_t = torch.stack(norm_penalty_list).mean()
        cos_sim_t = torch.stack(cos_sim_list).mean()

        clipo_loss = torch.zeros((), device=loaded.device)
        if args.clipo_weight > 0 and len(student_repr_list) >= 2:
            clipo_loss = _clipo_contrastive_loss(student_repr_list, teacher_repr_list, tau=args.clipo_tau)

        total = (ce_t + args.l_state_weight * state_t
                 + args.norm_anchor_weight * norm_penalty_t
                 + args.clipo_weight * clipo_loss)
        total.backward()

        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable_params, args.grad_clip)
        else:
            grad_norm = None

        if args.forge and int8_optimizer is not None:
            int8_optimizer.step()
        optimizer.step()

        mean_ce = float(ce_t.item())
        mean_state = float(state_t.item())
        mean_norm_penalty = float(norm_penalty_t.item())
        mean_cos_sim = float(cos_sim_t.item())
        mean_clipo_loss = float(clipo_loss.item())
        grad_norm_str = f" grad_norm={float(grad_norm):.4f}" if grad_norm is not None else ""
        print(f"[distill] step {step}: answer_ce={mean_ce:.4f} state_loss={mean_state:.4f} "
              f"norm_penalty={mean_norm_penalty:.4f} cos_sim={mean_cos_sim:.4f} "
              f"clipo_loss={mean_clipo_loss:.4f} n_answer_tok={n_tok_sum} "
              f"n_phase_tok={n_phase_tok_sum}{grad_norm_str}")
        with open(log_path, "a") as f:
            f.write(json.dumps({"step": step, "answer_ce": mean_ce,
                                 "state_loss": mean_state,
                                 "norm_penalty": mean_norm_penalty,
                                 "cos_sim": mean_cos_sim,
                                 "clipo_loss": mean_clipo_loss,
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
