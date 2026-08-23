# Effort frontier — noesis test-time compute knob

## What this is, as of 2026-08-17 (full rewrite — the old 3-axis framing is retired, not patched)

**There is one knob: M.** Not three (N, K, readout_mode). That was the design
before WKV-loop existed; WKV-loop made two of the three axes moot, and this
document kept describing all three anyway. Fixed here, in one pass:

| Old axis | Verdict | Why |
|---|---|---|
| **N** (re-feed the same prompt N times) | **Dead. No analog in `wkv_loop.py` at all.** | `generate_rollout` prefills the prompt exactly once. Nothing re-reads it. There is no code path this axis could even refer to anymore. |
| **K** (decode K invisible tokens from state, feed back) | **Absorbed into M, not a separate axis.** | `feed_mode="discrete"` in `generate_rollout` does exactly this — sample, feed back, repeat — the identical operation K used to name, now under a dynamic exit (plateau/commit) instead of a fixed budget. Calling it "K" alongside "M" would be double-counting the same mechanism. |
| **readout_mode** (`silent`/`prompt_cot`/`state_readout`) | **Dead**, for the same reason as N — it was about *where K's tokens came from relative to the prompt*, and there is no prompt-continuation decode step in the loop at all. | |
| **M** (WKV-loop internal steps) | **This is the live mechanism.** `feed_mode` (`discrete`/`expected`/`residual`) is M's only remaining sub-parameter — an implementation detail of *what gets fed back*, not a separate sweep axis. | |

Everything below this point is rewritten around M as the one axis. Historical
N-sweep/K-sweep numbers are kept where they exist (they're real data), but
relabeled for what they actually inform: K-sweep data is a *direct* prior for
M in `discrete` mode (same mechanism); N-sweep data is *not* informative
about M at all (no shared mechanism) and is kept only as a historical record,
not as a frontier data point.

## Status

**M-sweep: first real data landed 2026-08-18 — prediction confirmed.**
`experiments/A0_eval/eval.py --axis m` (new — routes through
`experiments.rl.wkv_loop.generate_rollout` instead of the retired H10
N/K/readout_mode axes) run on **G1i base, pre-RL**, discrete feed_mode,
M_max=16: **12.5% (6/48)**, vs. the `state_readout` baseline's 33.3%
(16/48) on the identical task set. Both `bit_decoding` and `extraction`
dropped to 0% under M. Single run, no seed variation — not a controlled
sweep yet — but directionally exactly the predicted pattern (see below):
flat and low pre-RL, because nothing has trained the model to write
anything useful during the M-loop steps yet. `mean_M` measured separately
on G1d/G1i via `--no-update` M-baseline runs (G1d: M=3 deterministic;
G1i: 7.0→2.0→2.0 across 3 steps, real GPU data, no controlled sweep over
seeds/prompts yet either).

**Prediction, stated plainly (confirmed directionally 2026-08-18):**
discrete-mode M on G1i base (pre-RL) should look like the old K-sweep —
flat, low accuracy — because K-sweep was flat for a reason (`step4_merged`,
2026-08-05: K=0→0/48, K=128/512/2048→4/48 flat) that has nothing to do with
which checkpoint or axis-name is in use: nothing had trained the model to
write anything useful into those intermediate steps. GRPO's
`−β·M − γ·Σ ReLU(ΔH_t)` reward is exactly the "write something useful"
objective that was missing. A real divergence from flat *after* RL
training is the actual signal to watch for — not the pre-RL number itself,
which is now measured (12.5%) rather than merely predicted.

**eval.py mode bug: FIXED 2026-08-12.** `state_readout` shared a decode path
with `prompt_cot` before that date; results from before are invalid — moot
now that `readout_mode` itself is retired, kept here only because `eval.py`
(the A0.2 harness, unrelated to `wkv_loop.py`) still has the flag and old
results referencing it exist in hypotheses/README.md §H10.

**Post-RL M-sweep: blocked behind a prerequisite, not just "not run yet"
(2026-08-19).** The "real divergence from flat after RL training" signal
this section calls for can't be collected — RL itself is currently
paused. A content decoder found the M-loop's internal step was never
real task content on any checkpoint tested, RL-trained or not, and
`mean_M` was frozen at a constant across an entire run with zero
variance — i.e. there was nothing for M to correlate against, on either
side of the flat-vs-divergent question this section poses. Think-loop
state distillation (`experiments/rl/train_think_distill.py`) is the
prerequisite now being worked — see `docs/rl-track.md`'s "Track status"
section (renamed 2026-08-23, was "RL status") for the current
experimental history. The 12.5%/33.3% numbers above remain the valid
pre-RL baseline; nothing here is retroactively wrong, the *next* data
point just isn't a post-RL M-sweep yet.

**"M" now names two different mechanisms — not yet reconciled
(2026-08-23).** This whole document's "one knob: M" framing (2026-08-17
rewrite, above) describes `wkv_loop.py::generate_rollout`'s self-feed
loop: M identical self-referential steps, same transformation applied
each time, exit on plateau/commit/M_max. ThinkChain
(`experiments/rl/train_think_distill.py`, 2026-08-21+, `docs/rl-track.md`
§Track status) uses the same letter for something structurally
different: M *distinct*, separately-trained phase markers, each
internally repeated up to a per-example `chunk_lens[i]` budget — the
step-count-vs-quality tradeoff this document analyzes is about the
*inner* repeat count of one phase, not about M itself, which is closer
in spirit to the old retired K (a fixed content-bearing unit) than to
this document's self-feed M. **Consequence for the effort registry
below:** `effort=fast/normal/deep → M_max` was designed for one scalar
that trades linearly against quality. ThinkChain-M is not that scalar —
adding a phase changes *what kind* of computation happens (a new,
distinct marker), not just *how much*. This gets sharper, not simpler,
with the 2026-08-23 Phase 2 revision (`docs/rl-track.md`): if phase
sequences interleave explore markers with a rewind/retreat marker
(`[phase_A, rewind, phase_B, rewind, ...]`), a ThinkChain M-count also
stops being uniformly "more compute → answer-relevant work" — retreat
ticks exist to *settle* state, not to write new task content, so
pricing them the same as explore ticks in a future `−β·M`-style cost
term is not obviously right. Not resolved here — flagged so whoever
designs a ThinkChain-native effort registry doesn't inherit this
document's single-scalar assumption by default. This document's own
M-sweep/registry design (below) still describes `wkv_loop.py`'s
mechanism accurately; it just isn't the right frame for ThinkChain
without this caveat.

**Current baselines (2026-08-14/17), for reference, not M data:**
- G1i chatwrap: 41.7% (20/48) — best single-pass baseline
- G1i base, `state_readout` (eval.py, post-fix, 2026-08-17): 33.3% (16/48) — see hypotheses/README.md §H10
- step9b-e1: 39.6% — regression from step9 e0 (43.8%)
- G1h base: 7.1% (format mismatch; not a useful baseline)
- Word-search nsp: G1i 0%/3.6% (baseline/np=256) — target for RL

## Problem

Foreign LLM APIs (Claude, GPT, etc.) expose an "effort" or "thinking" dial —
usually `fast / normal / thinking`, translating internally into a CoT-token
budget. `fast` = short CoT, `thinking` = long CoT. Prompt-conditioned CoT
tokens are the *only* test-time compute mechanism they have.

RWKV-7 has a genuinely different one: **M internal WKV-refinement steps,
with no tokens emitted or read from a prompt at all.** Each step feeds the
model's own current output back into the recurrence (`feed_mode="discrete"`:
the sampled token id; `"expected"`/`"residual"`: a continuous embedding,
peft/GPU only, differentiable) and updates state. The loop exits on
`plateau` (entropy stopped moving), `commit` (confident enough already), or
`M_max` (budget exhausted) — see `docs/rl-track.md` §RL design for the exact
mechanism and reward.

Copying the Transformer-industry convention (a token-budget dial) would be
wrong here even before WKV-loop existed: it constrains intermediate
computation to look like human-readable text when the only thing that
matters is the effect on WKV state. WKV-loop's M-step design is what
actually deletes that constraint, rather than working around it.

## CoT-as-WKV-input (why this axis is architecturally different)

**In a Transformer**, CoT tokens are part of the attention context, read by
subsequent attention heads alongside all other tokens. Human-readable
reasoning works because it provides structured content attention can index
over — the tokens are *output* that doubles as computation.

**In RWKV-7**, there is no attention over intermediate steps. Each one
passes through the WKV recurrence and updates state
`s(t+1) = f(s(t), x(t))`. Its value is entirely in how it shifts state — not
in surface form. There is no mechanism by which human-readable content would
help, and in the M-step design there isn't even surface form to begin with
in `expected`/`residual` mode (no discrete token exists at all).

**Consequence for training:** training on human CoT traces would be a wasted
degree of freedom for this architecture — optimizing invisible computation
for human readability is strictly wrong. **The actual objective**, and the
one WKV-loop's reward implements, is state quality: `r_correct` rewards the
answer that results, `−β·M` rewards getting there in fewer steps, and
`−γ·Σ ReLU(ΔH_t)` penalizes steps that make the model *less* confident. None
of these reward surface form.

## Framing — the frontier is now one-dimensional

Quality vs. M is the frontier; `feed_mode` picks which of three ways M steps
happen, not a second axis to cross with M. Approximate compute cost: prefill
`L * hidden * n_layer` once, plus `M * hidden * n_layer` for the loop, plus
answer decode. Total ≈ `hidden * n_layer * (L + M)` — M is directly
comparable to the old N's cost model in shape (both linear in step count),
which is part of why they got conflated; the mechanism is what differs, not
the cost shape.

The frontier is where an effort registry would get defined:
- `fast` = smallest M that meets a quality floor (e.g. ≥ 90% of best rubric).
- `normal` = knee of the M-vs-quality curve.
- `deep` = largest M that still adds ≥ +0.05 rubric over `normal`.

If the post-RL curve is flat (single M dominates), the registry has one
useful setting and M is runtime clutter, not a dial. That's a real possible
outcome, not assumed away.

## Sweep design

**Axis: M ∈ {0, 1, 2, 4, 8, 16, 32}.** (0 = `M_max` set to 0, i.e. answer
immediately after prefill — the single-pass baseline `generate_rollout`
doesn't currently expose directly but is the natural floor.) `feed_mode`
fixed to `discrete` for the sweep — `expected`/`residual` are training-time
mechanisms (need `mlp_delta`, differentiability), not eval-time dial
settings in the same sense.

**Task set.** A0.2 held-out rubric set, or the matrix-task curriculum
(`training/corpus_open/matrix_tasks.jsonl`) for RL-relevant categories.

**Two measurements per checkpoint:**
1. **Pre-RL, G1i base.** M forced via a modified `M_max` per run (bypass the
   plateau/commit exit to get a genuine M-response curve, not just whatever
   the untrained model's exit criteria happen to fire at). Expected: flat,
   per the K-sweep prior above.
2. **Post-RL checkpoint(s).** M's *natural* distribution (`exit_reason`
   histogram from `probes.py::effort_frontier`, already wired into
   `train_wkv_loop.py`'s periodic probe logging) plus accuracy at that
   natural M. This is the actual frontier data point — not a forced sweep,
   the trained model's own choice of M vs. the accuracy it gets.

**Verdict rule:**
- Non-degenerate: accuracy at high forced-M exceeds accuracy at low
  forced-M by ≥ +0.05 rubric on the pre-RL sweep → the loop can carry signal
  once trained, even though pre-RL it won't use it well.
- Post-RL confirmation: natural M distribution shifts away from `M_max`
  (more `plateau`/`commit` exits) while accuracy holds or rises → GRPO
  taught the model to use the loop, not just to stop.

## Effort registry (deliverable if the sweep confirms non-degenerate)

A runtime module mapping a task-time `effort` argument to an `M_max`
override:

```
effort=fast    → M_max=?   # smallest M meeting quality floor
effort=normal  → M_max=?   # natural post-RL M_max (no override — trust the model's own exit)
effort=deep    → M_max=?   # largest M with +0.05 over normal
```

Single value, not a tuple — `feed_mode` is a training-time choice, not a
runtime dial. Not a fine-tuning signal (model doesn't know its `effort`
setting), not a training-time objective (M_max is a runtime cap on an
already-trained exit policy, not something to bake in differently per
setting).

## Dependency chain

1. **A1.5 RL checkpoint** — the sweep needs a WKV-loop-trained model to be
   informative at all; the pre-RL number is a floor, not the answer.
2. **A0.2 / matrix-task eval infrastructure** — exists (`experiments/A0_eval/`).
3. ~~A0.6/A0.7 verdicts (state survives re-feed)~~ — **dropped.** That
   dependency existed because N needed the state-survives-re-feed property
   to be *defined at all*. M doesn't re-feed anything; the dependency is
   gone along with N.

## Open questions

- **Answer-decode temperature.** `generate_rollout` uses `answer_temperature`
  (default 0.7) for the final answer, greedy (`argmax`) inside the M-loop
  itself (`feed_mode="discrete"`). Is greedy-inside-the-loop the right
  choice, or would sampling inside the loop expose more of what the state
  "would say" at each step? Untested either way.
- **M_max ceiling choice.** Current default 16 (`train_wkv_loop.py` CLI),
  `generate_rollout`'s own default is 32. No data yet on whether either is
  near a real plateau or just an arbitrary budget.
- **Cross-task M shape.** Does wordsearch (spatial scanning, plausibly
  benefits from many refinement steps) show a different M-response curve
  than arithmetic (plausibly benefits from few, information-dense steps,
  per H25's step-function prediction below)? Untested.

## Cross-task and cross-model plan (once GPU + RL checkpoint exist)

**Cross-task.** Run the M-sweep per category separately
(`matrix_wordsearch`, `bits_matrix`, `arithmetic_matrix`, `pattern_matrix`,
`crossword_enum`/`crossword_fill`). Wordsearch is hypothesised to benefit
most from M (multi-pass grid coverage); arithmetic may show H25's predicted
step-function shape (flat below `M = rank(system)`, then a step, then flat
again) rather than a smooth curve — if observed, that's independent
confirmation that M-steps act as a write-budget for the arithmetic state,
not generic "extra thinking time."

**Cross-model.** Compare G1d-0.4B base → G1h-2.9B base → G1i-2.9B base →
G1i post-RL. Prediction (H2-adjacent, though H2 itself is withdrawn): post-RL
G1i should show a different M-response than any base checkpoint — a real
step-count/accuracy dependency where the bases show none. External anchor:
fleeb83's G1h 7.2B result (48/48 state-dependent stopping via silent
recurrent ticks, frozen backbone) bounds what G1i 2.9B should approach
post-RL — same architecture, 2.5× smaller, and via a different mechanism
(frozen-backbone ticks, not GRPO-trained M-exit), so not a direct
apples-to-apples target, but the right order of magnitude to compare against.

---

## H25 connection — state as compute, not just memory

H25 (WKV state as computational substrate for approximate algebra) predicts
what the M-frontier should look like for math tasks specifically: if the
model learns to write equation coefficients into WKV state during M-steps
and read them back at the answer, M is not "more time to think" — it is
"more write operations into numerical state." Bounds:

- **Capacity floor:** M ≥ rank(equation system) for all rows to be written
  before decay erases earlier ones (a 3×3 system needs M ≥ 3).
- **Precision ceiling:** bf16 limits to ~3 significant digits regardless of M.
- **Model size:** larger head_size × n_head → less rank-1 interference →
  higher effective precision per write.

Predicts a step-function M-response for math tasks (flat below
`M = rank(problem)`, step up, flat again) — qualitatively different from a
smooth curve, and testable in the cross-task plan above.

ROSA (RWKV-8) would eliminate the precision ceiling and decay floor
entirely (exact writes, no forgetting on the ROSA channel) — deferred
indefinitely as of 2026-08-16 (BlinkDL updating datasets), architecture
analysis still valid, timeline unknown.

Credit: fleeb83, proof-of-mechanism for state-based computation (symbolic
domain, 2026-08-16); H25 formalised from that result.
