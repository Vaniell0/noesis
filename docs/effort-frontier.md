# Effort frontier — noesis-specific test-time compute knobs

> **Status.** Design draft, 2026-07-22. Not implemented. Blocked on
> A0.6/A0.7 verdicts + A1 checkpoint. Registered here to freeze the
> framing before A0.8 runner design lands. Falsifier + prediction are
> locked as **H10** in `HYPOTHESES.md` (expanded from N-only sweep to
> 3D matrix on 2026-07-22).

## Problem

Foreign LLM APIs (Claude, GPT, etc.) expose "effort" or "thinking"
dials — usually a single scalar like `fast / normal / thinking` that
translates internally into a CoT-token budget. The convention is:
`fast` = short CoT, `thinking` = long CoT. All rely on
prompt-conditioned CoT tokens as the sole test-time compute
mechanism.

RWKV-7 exposes at least three orthogonal dials, and it's not obvious
which combination Pareto-dominates on which task type:

- **N** — state-refinement passes. Feed the prompt through the
  backbone N times before decoding; each pass updates WKV without
  emitting tokens. Cheap (no re-ingest, constant-time per pass), no
  tokens visible.
- **K** — CoT-token budget. Traditional. Decoded think-tokens
  re-ingested via state update per step.
- **readout_mode** — where the CoT tokens come from:
  - `silent` — K=0, no CoT tokens at all.
  - `prompt_cot` — classic. CoT-tokens are the model's continuation
    of the prompt scaffold ("Let me think… step 1…").
  - `state_readout` — CoT-tokens decoded *directly from the refined
    state* (no CoT-prompt scaffold); they are a self-report on the
    state, then the answer decodes from the state-after-readout.
    Related to the scratch-lens design in `docs/memory-lenses.md` —
    the same self-summary idea, but applied at inference time
    instead of handoff time.

Copying the Transformer-industry effort convention (single K dial,
prompt_cot mode implicit) leaves the other two dials untuned.
There's no a-priori reason `(N=1, K=high, prompt_cot)` is optimal for
this architecture on the tasks noesis actually cares about.

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

## Not on the critical path

This document exists to freeze the framing. Real testing runs after
A1 + A0.6/A0.7. Failure of any of those preconditions narrows or
opens the design space; register the impact in a status update here
rather than re-litigating the framing every time.
