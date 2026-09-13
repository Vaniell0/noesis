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

## Why M should help at all — measured, 2026-09-13

Until now M was justified by cost shape and by the K-sweep prior ("same
mechanism, so K's numbers transfer"). Neither says *why* extra internal steps
would recover anything. This does, and it is the first mechanistic argument
for the axis rather than an empirical hope.

`jlens_probe.py` was rewritten this day after the realisation that
`stable_rank` — the number every previous state-rank reading in this project
rested on — is an **energy-concentration** ratio (‖A‖²_F/σ₁²), not a count of
live directions. A matrix with one dominant singular value plus thirty-one
real ones scores ~1.03 on it (`test_jlens_spectrum.py` pins this down). The
project's measured 1.1-1.3 sits exactly in that regime, so "the state is
nearly rank-1" was never established by that statistic. The probe now also
reports entropy-effective rank, participation ratio, and a plain count of
directions above 1% of σ₁ — and keeps the per-head distribution, which every
earlier run computed and then averaged away.

Re-measured on **G1i base**, 32-token prompt, per head, 64 dimensions
available (`experiments/A0_state_probe/results/rank_recheck/jlens.json`):

| L | stable_rank | eff. rank (entropy) | directions > 1% of σ₁ | per-head min/med/max |
|---|---|---|---|---|
| 0 | 1.291 | 6.50 | 13.2 | 3 / 12 / 26 |
| 4 | 1.211 | 6.44 | 15.2 | 5 / 16 / 28 |
| 8 | 1.173 | 5.86 | 13.7 | 6 / 13 / 26 |
| 12 | 1.181 | 5.88 | 13.7 | 5 / 13 / 22 |
| 16 | 1.229 | 6.36 | 14.7 | 7 / 15 / 22 |
| 20 | 1.247 | 6.98 | **15.8** | 8 / 16 / 26 |
| 24 | 1.095 | 3.60 | **8.4** | 4 / 8 / 20 |
| 28 | 1.148 | 4.50 | 11.0 | 5 / 11 / 16 |

**The state is not collapsed — it is superposed.** 13-16 live directions of
64, up to 28 in individual heads, with the energy concentrated in roughly one
of them. The content is present; a single-pass readout weighted by magnitude
extracts a fraction of it.

**M is the mechanism for traversing that breadth — and M is a chain length,
not a repeat count.** Stating this explicitly because this project has already
caught the "M names two different mechanisms" confusion once (`docs/rl-track.md`)
and it is easy to re-import. M is the number of *distinct* phases, each with
its own learned marker `chain[i]`; repeating one marker is a different knob
(`--phase-repeat-ticks`), on a different axis. The distinction is measured, not
assumed (`docs/phase15-gap-matrix.md`, repeat-count vs. M-count isolation):
repeating the same marker inside a phase converges within about four ticks
(`delta_cos_prev` → 0.997), while crossing into a new phase resets the
direction near-orthogonally (`delta_cos_prev` = 0.097). Phases do different
directional work; repeats do the same work harder and stop paying almost
immediately.

So the mechanistic claim is not "read the same thing again until it comes out
cleaner" — it is that **a chain of M near-orthogonal phases can engage
different parts of a state that holds 13-16 live directions, where one readout
pass engages approximately one.** This is the same phenomenon the
readout-corrector scaling curve shows from the outside (+7 / +6 / +2 rubric at
1.5B / 2.9B / 7.2B — the smaller the model, the more an external
read-correction recovers), and the same one looped architectures exploit: the
win is in extraction, not in storage.

**Quantitative prediction that follows, and can be wrong:** useful M should be
bounded by the available breadth rather than unbounded. With 13-16 live
directions mid-stack and 8.4 at L24, there is nothing left to traverse much
past that, so the M-vs-quality curve should saturate in that neighbourhood
rather than continuing to pay. If a future M-sweep keeps gaining well beyond
~16, this reading of what M is doing is wrong.

Two structural details that fall out of the per-head distribution, neither
visible in any earlier artifact:

- **The narrowing is late.** Breadth builds through the middle (L4-L20:
  13-16) and drops sharply at L24 (8.4, eff. rank 3.60) and L28 (11.0) —
  immediately before readout. The bottleneck is at the exit, not spread
  through depth.
- **Width and strength are close to independent across heads.**
  corr(directions, σ₁) = +0.30 at L4: the widest head there (28 directions)
  has σ₁ = 6.06, below that layer's mean of 9.35, while a 5-direction head
  carries σ₁ = 11.77. Quiet-and-wide heads exist.
  *First reading of this was wrong and is corrected here rather than deleted:*
  it looked like an intervention point ("a magnitude-weighted readout would
  miss them"), but RWKV-7 normalises per head already — `self.ln_x =
  nn.GroupNorm(H, C)` with `num_groups = n_head`, so each head's channels are
  their own normalisation group and every head reaches the mixer at unit
  variance regardless of its σ₁. Quiet heads are not down-weighted; they are
  equalised. Whether equalising a genuinely low-signal head is good (it also
  amplifies its noise) is a separate and untested question — but the
  "readout ignores them" version of the concern does not survive reading the
  architecture.

### What this predicts about `feed_mode` — ordinary vs. latent tokens

This turns `feed_mode` from an implementation detail into a testable
consequence, and the prediction is sharp enough to be wrong:

- **`discrete`** samples a real vocabulary token and feeds it back. That
  projects the state through the vocabulary bottleneck — whatever the
  magnitude-weighted readout produced, quantised to one token. If the state
  holds 13-16 directions and the readout sees mostly one, a discrete token
  can carry back approximately that one. **Prediction: `discrete` recovers
  little of the measured breadth, and its M-curve should flatten early.**
- **`expected`** feeds back `softmax(logits) @ emb.weight` — a mixture, no
  vocabulary projection, differentiable. It can carry more than one
  direction's worth across a phase boundary. **Prediction: `expected` should
  carry more between phases than `discrete`, and the gap should widen with
  chain length.**

If the two feed modes give the same M-curve, this mechanistic story is wrong
and M's benefit (where it exists) comes from something else. Worth stating
plainly because the first real M-sweep data (2026-08-18, G1i base, discrete,
M_max=16: 12.5% vs. 33.3% for the retired `state_readout` baseline) was run
in `discrete` mode only — i.e. in exactly the mode this section predicts is
the weaker of the two, which is a confound in the only M data currently held.

### Connection to DE

`DE = accuracy / mean_output_words_for_correct_answer × 100` (H24). If M works
the way this section argues — more of the already-present state content
extracted per pass — then M should raise accuracy **without** raising emitted
token count, since the extra work happens in the loop and not in the visible
output. That is precisely what DE measures, and it is why DE, not raw
accuracy, is the right frontier metric for this axis: raw accuracy would also
go up if the model simply talked more, and DE would not.

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

## Sweep design, Effort registry, Dependency chain — STALE (2026-09-03)

**Everything from here to the next `---` describes the self-feed
`wkv_loop.py` M mechanism (`train_wkv_loop.py`, A1.5 RL checkpoint).
That track is paused by explicit decision (`docs/rl-track.md`'s
appendix: an RL-trained checkpoint's internal M-step was never real
task content, just chat-template scaffolding — RL had nothing to
select for). The live mechanism is ThinkChain
(`experiments/rl/train_think_distill.py`), and per this file's own
2026-08-23 note above ("M now names two different mechanisms"),
ThinkChain-M is not the single linear scalar this section assumes —
adding a phase changes *what kind* of computation happens, not just
how much, and rewind ticks (Phase 2, designed but not built) settle
state rather than doing answer-relevant work, so pricing them like
explore ticks isn't obviously right either. Kept below as a historical
record of the self-feed-M design, not a live plan. A ThinkChain-native
effort registry needs its own design — not started, and not
buildable yet: it needs a real multi-phase (M>1) trained checkpoint,
which needs Phase 2 (rewind marker) built and Phase 1.5 (full-FT)
done first. Don't resume this section's plan as written once those
land — redesign it around distinct phase-kind + rewind-budget, not a
single `M_max`.

**New grounding for the redesign, 2026-09-04 — the loop channel is not
a generic "more compute" knob, there's now a small citable demonstration
of that.** `hypotheses/H25.md`'s parity replica of arXiv 2505.21024
(pause tokens strictly increase constant-depth Transformer expressivity)
found the WKV-native analog: held-out accuracy 0.882 (`n_extra=0`) ->
0.980 (1) -> 1.000 (2) -> plateau — a real, structured jump from adding
loop steps, not a smooth "more M, somewhat better" curve. Consistent
with this file's own step-function prediction for math tasks (§H25
connection below) rather than the old single-scalar `effort=fast/
normal/deep` framing: whatever a ThinkChain-native registry ends up
being, "more phases = more of the same kind of gain" is the wrong
mental model to redesign it around — expect thresholds tied to what a
task actually needs written/held, not a dial.

**Third self-feed mechanism, 2026-09-05 — do not conflate with either
mechanism above.** `experiments/rl/wkv_loop.py::generate_rollout_latent_chain`
(+ `experiments/rl/loader.py::extend_vocab_for_marker`), built 2026-09-04,
is neither this section's paused GRPO-M nor ThinkChain-M. A single real
vocab-id marker is fed once per round (not repeated), followed by
ordinarily-sampled tokens each round — genuinely distinct content per
tick, not a constant repeated, so it doesn't inherit this section's
repeated-operator caveat either. Currently diagnostic-only: no training
signal yet (marker embedding is cold-start/untrained, sampled-token
rollout isn't differentiable — no GRPO-style recompute pass built).
Doesn't belong in this file's registry/sweep design above — it has no
effort/`M_max` notion at all yet, being untrained. Full design reasoning
and status: `docs/rl-track.md`'s Track status section and memory
`project_noesis_rl_track`.

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
