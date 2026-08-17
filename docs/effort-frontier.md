# Effort frontier — noesis-specific test-time compute knobs

## Status (updated 2026-08-17)

**eval.py mode bug — FIXED (2026-08-12), not "known, not yet patched" as this
doc previously said.** `state_readout` shared its decode path with `prompt_cot`
prior to 2026-08-12 (see `eval.py` lines ~220-221 for the code's own note); all
`state_readout` results from before that date are invalid. The H10 rerun
launched 2026-08-16 (`experiments/A0_eval/results/h10_state_readout_g1i_base.json`)
is the first run on the fixed code — see `HYPOTHESES.md` §H10 for the result
once it lands.

**WKV-loop note.** `experiments/rl/wkv_loop.py` implements the
`(N=M, K=0, mode=silent)` corner of the sweep space by construction — M
internal state-refinement steps, no token emission, entropy-plateau exit.
M-axis measurements on the post-RL WKV-loop checkpoint ARE effort-frontier
data points. The (N, K, mode) sweep below remains the full characterisation
framework; WKV-loop collapses it to one specific path and measures how far M
can push accuracy before the exit criterion fires.

**N-sweep: NOT YET RUN.** Target model: G1i base or post-RL checkpoint.
External validation of the N-axis mechanism: fleeb83 (2026-08-16) demonstrated
48/48 state-dependent stopping and multi-step symbolic composition on G1h
7.2B using a frozen backbone + silent recurrent ticks (no output tokens) +
causal swap/replace/zero controls — an empirical realisation of N > 1, state
accumulating usable computation across passes. Open question: does G1i base
(vs. fine-tuned) show the same behaviour on the task rubric.

**K-sweep** (2026-08-05, step4_merged @45% epoch, A1 plan — superseded):
K ∈ {0, 128, 512, 2048}, N=1, mode=prompt_cot, 48 tasks. Raw: K=0 → 0/48 (no
output), K=128/512/2048 → 4/48 flat. Interpretation carries over even though
the A1 plan and step4 checkpoint don't: K-axis flat because the model was
never trained with a state-quality objective for intermediate tokens (see
§CoT-as-WKV-input). Training base is now G1i 2.9B.

**Current baselines (2026-08-14/16):**
- G1i chatwrap (N=1, K=256): 41.7% (20/48) — best single-pass baseline
- step9b-e1 (N=1, K=256): 39.6% — regression from step9 e0 (43.8%)
- G1h base: 7.1% (format mismatch; not a useful baseline)
- Word-search nsp: G1i 0%/3.6% (baseline/np=256) — target for RL

**First design-time data point (2026-07-23).** Adaptive per-N budget variant
of H12a on G1d-0.4B: at N ≤ 8, budget → recall is monotone-positive; at
N ≥ 32, +952 tokens above 2048 did not move recall from floor. Still needs
the full (N, K, mode) matrix to conclude anything general.

## Problem

Foreign LLM APIs (Claude, GPT, etc.) expose "effort" or "thinking"
dials — usually a single scalar like `fast / normal / thinking` that
translates internally into a CoT-token budget. The convention is:
`fast` = short CoT, `thinking` = long CoT. All rely on
prompt-conditioned CoT tokens as the sole test-time compute
mechanism.

RWKV-7 exposes at least three orthogonal dials, and it's not obvious
which combination Pareto-dominates on which task type:

- **N** — WKV state cycling passes. The same prompt token sequence
  is fed through the backbone N times; each pass runs the WKV
  recurrence *on top of the accumulated state from the previous pass*.
  Not a reset — state compounds. Each additional pass refines the
  WKV representation of the same input without emitting any tokens.
  This is an architectural property testable without task-specific
  training: does the WKV recurrence converge to a better state
  representation when given N looks at the same input?
- **K** — intermediate token budget. K tokens are decoded from the
  current state, fed back through WKV one by one, updating state
  at each step, then the answer is decoded. These tokens are
  *invisible* — not part of the output, not human-facing. Their
  value is entirely in how they update the WKV state. See
  §CoT-as-WKV-input for why this is architecturally different from
  Transformer CoT.
- **readout_mode** — source of the K intermediate tokens:
  - `silent` — K=0, state after N passes is decoded directly.
  - `prompt_cot` — K tokens decoded as continuation of the prompt;
    their structure is constrained by the prompt's linguistic context.
  - `state_readout` — K tokens decoded freely from the post-N state,
    no prompt scaffold; the state drives its own intermediate
    computation. Highest potential for WKV-useful structure.

Copying the Transformer-industry convention (single K dial,
prompt_cot mode implicit) is wrong for this architecture: it
constrains intermediate tokens to be human-readable text continuations
when what matters is their effect on the WKV state update.

## CoT-as-WKV-input

This is the design constraint that distinguishes noesis effort from
Transformer-style thinking:

**In a Transformer**, CoT tokens are part of the attention context.
They are read by subsequent attention heads alongside all other tokens.
Human-readable reasoning ("Step 1: ..., Step 2: ...") works because
it provides structured content that attention can index over. The
tokens are *output* that doubles as computation.

**In RWKV-7**, there is no attention over the CoT tokens. Each
intermediate token passes through the WKV recurrence and updates the
state `s(t+1) = f(s(t), x(t))`. The token's value to the model is
entirely in how it shifts the WKV state — not in its surface form.
A human-readable "Step 1" is no better than any other token sequence
unless "Step 1" happens to produce a better WKV state update than
alternatives. There is no mechanism by which readability helps.

**Consequence for training:** Training on human CoT traces (SFT on
chain-of-thought datasets, RLHF on reasoning paths) teaches the model
to produce tokens that look like human reasoning. For RWKV this is a
wasted degree of freedom — the model is constrained to a subspace of
token sequences that are optimised for human reading rather than for
WKV state update quality. The intermediate tokens are invisible;
optimising them for visibility is strictly wrong.

**The right objective** for K-token training is to reward the *state
quality after K tokens*, not the surface form of the K tokens
themselves. This requires a state-quality proxy (e.g. downstream task
accuracy, IB-style `I(Z;Y)` estimate, or state-reg style dynamics
reward). It is a separate training track from A1 and belongs in a
future corpus design — not in any existing reasoning-trace dataset.

**Practical implication for the sweep:** Until a model is trained with
a state-quality objective for intermediate tokens, the K-axis and
readout_mode comparison will be weak signal. The N-axis (WKV cycling
with no token emission) is the cleanest test because it requires no
specialised training — it tests the raw WKV accumulation property.

## Framing

The three knobs live on a **Pareto frontier**: quality vs compute
cost. Each cell in the `(N, K, mode)` grid is one point on that
frontier; the shape of the frontier is what we don't know.

Compute cost model (approximate, for one query with prompt length L
and vocab-independent inference):

- Prefill: `L * hidden * n_layer` FLOPs per pass, times N passes.
- CoT decode: `K * hidden * n_layer` FLOPs per token, plus state
  update per token.
- Total compute ≈ `hidden * n_layer * (N * L + K)`.

Ratio: `(N * L) / (N * L + K)` is the "state work fraction". For
`N=1, L=200, K=512`: state work is ~28 %. For `N=3, L=200, K=0`:
state work is 100 %. Same *total* compute budget might land at very
different quality depending on how it splits.

The frontier — if non-trivial — is where noesis's effort registry
gets defined:

- `fast` = smallest cell that meets a "reasonable" quality floor
  (say, ≥ 90 % of best rubric).
- `normal` = knee of the frontier — best quality per unit compute.
- `deep` = largest cell that adds ≥ +0.05 rubric over `normal`.

If the frontier is trivial (single point Pareto-dominates
everything), the registry has only one useful setting and the extra
knobs are runtime clutter. Falsifier for that case is in the H10
prediction.

## Sweep design

**Axes.**

| axis | values | rationale |
|------|--------|-----------|
| N | {0, 1, 2, 3, 5} | 0 = no refinement (single-pass baseline); 5 = well beyond diminishing-returns knee if there is one |
| K | {0, 32, 128, 512} | 0 = silent; 32/128/512 = short/medium/long CoT budgets, straddle common effort-dial ranges |
| readout_mode | {silent, prompt_cot, state_readout} | K=0 forces silent regardless; K>0 requires prompt_cot or state_readout |

**Cell count.** 5 × 4 × 3 = 60, minus invalid combinations:
`K=0 × mode ∈ {prompt_cot, state_readout}` is degenerate (both
collapse to silent), so remove those 5 * 2 = 10 cells → **50 cells
per task**.

**Task set.** A0.2 held-out rubric set (≥ 30 tasks). Cell-per-task
count = 50 × 30 = 1500 evaluations per model. On i5-1235U with the
0.4B backbone at ~1 tok/s, per-eval budget:

- Prefill: N passes over prompt ≈ L * N seconds (L ≈ 100–300 tokens).
- CoT: K tokens ≈ K seconds.
- Answer decode: budget varies per task type but roughly 64 tokens.

Worst cell: N=5, K=512, L=300 → 5*300 + 512 + 64 = 2076 tokens ≈
35 min per task. 1500 evaluations at 35 min each = infeasible on CPU
alone. Two options:

1. **Truncate.** Cap N × L at 500 tokens of state work, cap K at 128
   for the CPU pilot; run the full 3D sweep only on G1d-0.4B, on 30
   tasks. Total ≈ 3 h wall on the Windows-box GPU (post A1 training).
2. **Two-stage.** CPU pilot on a 6-task subset with reduced axes
   `{N ∈ {1, 2, 3}, K ∈ {0, 64}, mode ∈ {silent, prompt_cot,
   state_readout}}` → 15 cells × 6 tasks = 90 evals for a rough
   Pareto shape. If PASS, do the full sweep on GPU.

Recommendation: (2). CPU pilot informs whether the frontier is
non-trivial before we spend GPU hours.

**Metric.** Rubric score from A0.2 (LLM-as-judge, spot-checked).
Secondary: wall-time per cell (for the effort registry's cost model).

**Verdict rule.** See H10 falsifier. Two-line summary:

- Non-degenerate: some `(N > 1 OR mode ≠ prompt_cot)` cell ≥ +0.05
  rubric at ≤ 1.0× default compute → the matrix has real content.
- Readout-load-bearing: `state_readout` beats `silent` at same N by
  ≥ +0.02 → the readout mode is worth keeping in the registry.

## Effort registry (deliverable if H10 PASSes)

A runtime module (Rust, tentative path `runtime/noesis-effort/`)
that maps a task-time `effort` argument to a `(N, K, mode)` tuple:

```
effort=fast    → (N=?, K=?, mode=?)  # smallest cell meeting quality floor
effort=normal  → (N=?, K=?, mode=?)  # frontier knee
effort=deep    → (N=?, K=?, mode=?)  # largest cell with +0.05 over normal
```

The `?`s are filled by the A0.8 verdict. The presets are not baked
into the model — they're runtime knobs the supervisor sets per query
based on task-scheduler policy.

Non-goals for the registry:

- **Not** a fine-tuning signal. Model doesn't know its current
  effort setting.
- **Not** a training-time objective. Refinement passes are runtime
  choices, not something to bake into weights.
- **Not** a substitute for CoT training. If H7 (in-context reasoning)
  needs a certain CoT style baked in, that's a separate corpus track;
  effort registry only picks how much of that baked capability to
  invoke per query.

## Dependency chain

1. Requires: A0.6/A0.7 verdicts → tells us if state survives re-feed
   (necessary for N > 1 to be defined) and if state carries content
   across LoRA bumps (necessary for readout to be interpretable
   across model swaps).
2. Requires: A1 checkpoint → so the model being probed is the actual
   noesis backbone, not a proxy.
3. Requires: A0.2 held-out eval set → already exists per ROADMAP.

## Open questions

- **Readout decoding params.** Greedy for the readout tokens too, or
  low-temperature sampling? Greedy is simpler and matches A0.6/A0.7
  discipline; sampling might expose more of the state's content but
  adds variance.
- **State-refinement warm-up.** Does N > 1 need the prompt re-fed
  from scratch each pass, or is it enough to keep the state and
  re-feed the prompt over-and-over on top? The latter would compound
  state; the former resets each pass. First iteration should try
  both and let the frontier speak.
- **Interaction with A0.7 verdict.** If A0.7 tier-1 PASSes, the
  readout tokens are transferable across model swaps (useful for
  runtime hot-swap). If it FAILs, readout is only useful within one
  model — still fine for the effort registry, but the memory-lens
  handoff protocol falls back to text-only.

## Extended sweep plan (post-Step-5)

The K-pilot revealed that K-axis alone, at N=1 with prompt_cot, is
flat. The informative next cuts:

### N-axis (highest priority)

Run the K=0 baseline at N ∈ {0, 1, 2, 3, 5} on the Step5 model.
N=0 (no re-feed) is the single-pass baseline; N>1 re-feeds the prompt
through the backbone multiple times before decoding, updating WKV without
emitting tokens. If state is doing real computation (H8 SUPPORTED), N
should matter. Cost per extra pass: `L × hidden × n_layer` FLOPs —
constant-time relative to the prompt, no token decode.

Command (after Step5 eval passes):
```bash
NOESIS_EVAL_DEVICE=cuda bash experiments/A0_eval/run_effort_sweep.sh \
  --model /tmp/step5_merged.pth --n-values 0,1,2,3,5 --k-values 0 \
  --out /tmp/effort_n_sweep_step5
```

### state_readout vs prompt_cot cross-test

Once `state_readout` mode is implemented (reads CoT tokens directly from
the state, no scaffold prompt), compare:
- `(N=1, K=128, mode=prompt_cot)` — traditional CoT
- `(N=1, K=128, mode=state_readout)` — state self-report, then answer
- `(N=3, K=0,   mode=silent)` — same compute budget, no tokens

The claim (H10, readout-load-bearing sub-hypothesis): state_readout ≥ silent
at same N by ≥ +0.02 rubric. Failure collapses the mode axis — only
N and K remain.

### Cross-task comparison

Run the full (N, K) grid on each task *category* separately.
Current matrix_tasks.jsonl categories: `matrix_wordsearch`, `bits_matrix`,
`arithmetic_matrix`, `pattern_matrix`, `crossword_enum`/`crossword_fill`.
(Old names `bit_decoding`, `arithmetic_chain`, `scheduling` are superseded.)
Different cognitive loads should produce different frontier shapes:
- Wordsearch (spatial scanning) is hypothesised to benefit most from N
  (state re-feed allows multi-pass grid coverage) rather than K.
- Arithmetic may benefit from K (step-by-step CoT) but not from N —
  however H25 predicts a step-function shape: flat for K < rank(system),
  then step up, then flat again. This is qualitatively different from
  the smooth curves expected for extraction/symbolic tasks. If observed,
  it confirms that K is acting as write-budget for matrix rows in WKV
  state, not as reasoning-trace length.
- If both are flat, the frontier is degenerate for all current tasks —
  that tells us the task set needs harder examples for H10 to be
  falsifiable.

### Cross-model comparison

Compare across training stages (A1 plan superseded; current ladder):
1. `G1d-0.4B base` — smallest scale, world knowledge only
2. `G1h-2.9B base` — 7× scale, same training distribution
3. `G1i-2.9B base` — same size as G1h, updated pretraining (ctx16384, think tokens in base output)
4. `step9b-e1` — G1h + LoRA step9b (RFC QA + ε-mask); 39.6% accuracy, DE=0.15
5. `G1i post-RL` — target (word-search GRPO, GPU-blocked)

Prediction (H2, reasoning-first): post-RL G1i should show a *different*
N-response than base — higher N-plateau because RL installs state-write
behaviour that N passes can compound. G1i base already emits `<|im_start|>thought`
tokens without fine-tuning, suggesting G1 pretraining installed latent think-routing
that N > 1 may activate.

External anchor: fleeb83's G1h 7.2B result (48/48 state-dependent stopping,
silent ticks) bounds what G1i 2.9B should approach post-RL — same architecture,
2.5× smaller, partially trained.

### Parameters not yet explored

| Parameter | Current value | Why to vary |
|-----------|---------------|-------------|
| Prompt length L | ~100–300 tokens | N×L is the state-work budget; short L may underestimate N benefit |
| CoT temperature | greedy | Low-temp sampling during readout may expose more state content |
| Number of tasks | A0.2 rubric set (~30) | Need ≥ 50 for reliable Pareto shape, especially for per-category |
| Task difficulty | current held-out set | Ceiling effect: if tasks are easy, K and N both plateau early |

### Verdict hierarchy (H10 sub-claims)

1. **K-axis flat at full epoch?** — K-pilot was at 45% epoch; wait for Step5 verdict.
2. **N-axis non-trivial?** — if yes, frontier has real content; if no, only one dial matters.
3. **state_readout load-bearing?** — distinguishes architecture-specific dial from mode cosmetics.
4. **Cross-task shape variation?** — determines whether effort registry needs per-task presets.

Only after (1) and (2) are settled does the effort registry design have values to fill in.

## Not on the critical path

This document exists to freeze the framing. Update the status block and add
verdict results here rather than re-litigating the framing each session.

---

## H25 connection — state as compute, not just memory

H25 (WKV state as computational substrate for approximate algebra) reframes what
the effort frontier measures. If the model learns to write equation coefficients
into WKV state during think-span tokens and read them back at the answer token,
then K (think-token budget) is not just "more time to reason" — it is "more write
operations into numerical state". The frontier for mathematical tasks is bounded by:

- **Capacity floor**: K ≥ rank(equation system) for all rows to be written before
  decay erases earlier ones (for a 3×3 system, K ≥ 3 think-tokens minimum)
- **Precision ceiling**: bf16 limits to ~3 significant digits regardless of K
- **Model size**: larger head_size × n_head → less rank-1 interference → higher
  effective precision per write

This predicts a qualitatively different K-frontier shape for mathematical prompts
vs. extraction/symbolic prompts: a step function at K = rank(problem) rather than
a smooth curve. Measurable post-RL.

ROSA (RWKV-8) eliminates the precision ceiling and the decay floor: exact writes,
no forgetting on the ROSA channel. The effort-frontier concept still applies, but
the bottleneck shifts from "how many think-tokens to write the matrix" to "how many
read-back passes to refine the solution". A combined WKV+ROSA system's frontier
for exact arithmetic is effectively flat: one pass, exact answer, K=0.

> Note: ROSA 8 deferred indefinitely (2026-08-16) — BlinkDL updating GitHub datasets.
> ROSA architecture analysis remains valid; timeline unknown.

Credit: fleeb83 proof-of-mechanism for state-based computation
(symbolic domain, 2026-08-16); H25 formalised from that result.
