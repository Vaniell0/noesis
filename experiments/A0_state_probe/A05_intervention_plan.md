# A0.5 — Causal state intervention (H8 sharper test)

> **Status.** Draft plan. Motivated by the observation that A0.4 measures
> *how* state moves (`Δ`, `κ`, `SR`), which is *description*, not a
> *causal test* of whether state is doing computation. This experiment
> is the load-bearing test for H8.

## What A0.4 does not answer

A0.4 measures state trajectory statistics — pooled Δ-norm, curvature,
stable rank — on reasoning vs non-reasoning prompts. Even a passing
result (large Cohen d on ≥ 2/3 metrics) shows only that state
*trajectories differ* between prompt types. It does not show that state
*carries computation* in the sense H8 claims — state could be a rich
rolling summary whose motion correlates with content novelty, and still
be non-computational.

The canonical causal question: **does perturbing the state change the
model's output distribution in a way commensurate with "state is doing
work"?**

## Design

Standard interventional recipe applied to RWKV-7 WKV state.

### Loop (one intervention experiment)

1. Encode prompt `p` (medium or narrative). Prefill → state `s_0`.
2. Generate `T` tokens normally, saving `state[t]` at each checkpoint
   `t ∈ {t_1, t_2, ..., t_k}` (say 5 evenly-spaced points inside decode).
3. At each checkpoint `t`, clone `state[t]` → `s`. Apply a corruption
   `C` (see below) → `s'`.
4. Two single-step forwards from the **same** input token but different
   state:
   - `logits_clean = forward(x_t, s)`
   - `logits_corr  = forward(x_t, s')`
5. Compute the divergence
   `D_t = KL(softmax(logits_clean) || softmax(logits_corr))`.
6. Optionally continue autoregressive decode from both states for
   `N` further tokens and measure trajectory divergence
   (token-level agreement, cumulative KL). This is the *sequence-level*
   consequence of the perturbation.

### Corruption types `C`

Each is a distinct claim about *what part of state matters*:

- **`gauss(σ)`** — additive Gaussian scaled to state norm:
  `s' = s + σ · ‖s‖_F · N(0, I) / √(dim(s))`.
  Continuous, isotropic. Cleanest control on "any perturbation".
- **`zero_layer(i)`** — zero WKV state of layer `i` only. Locates
  *where* state work happens.
- **`zero_head(i, h)`** — zero one head. Finer localisation.
- **`shuffle_heads(i)`** — permute the head dimension within layer `i`.
  Preserves norm and stable rank; destroys structure. Tests whether
  it's the *distribution* or the *structure* that matters.
- **`freeze(t → t)`** — replace `s_{t}` with `s_{t-Δ}` from an earlier
  checkpoint. Tests whether *recent* state work matters more than
  *cumulative* state.

### Baselines

- **Uncorrupted repeat** — two identical forwards from `s`. Any
  non-zero KL is bf16 numeric noise; that is the *irreducible noise
  floor*.
- **Same-model, resampled seed** — how much does a different sampling
  seed alone shift the *next-token* distribution? Sets a "sampling
  noise" baseline. If `gauss(σ=small)` moves KL by less than this,
  the state was not carrying that information.

## Metrics

Per checkpoint `t`, per corruption `C`, per seed:

- **`KL_next`** — KL divergence at the next token.
- **`argmax_flip`** — 0/1 whether the greedy top-1 changed.
- **`token_overlap_N`** — after continuing decode for N steps from `s`
  and `s'` separately, fraction of positions where the two greedy
  continuations agree.
- **`cum_KL_N`** — sum of per-token KLs over the following N steps
  (measures decay).

Aggregate across checkpoints × seeds to get mean ± SD per
`(model, prompt, corruption_type, corruption_strength)`.

## Hypotheses (sharper form of H8)

Let `D_gauss(σ)` = mean KL under Gaussian perturbation of scale σ,
minus the numeric noise floor.

**H8-causal-A.** `D_gauss(σ)` grows super-linearly in σ within a
regime where state norm is preserved to ~10%. Interpretation: state
carries *localised, non-redundant* information; small perturbations
knock out specific features rather than degrading gracefully.

**H8-causal-B.** `D_layer(i)` shows *specific layers* with `KL_next`
orders of magnitude above the mean-layer effect. Interpretation:
state work is *localised* to a subset of layers, consistent with a
computational division of labour rather than a distributed memory
buffer.

**H8-causal-C.** On reasoning prompts (`medium`) vs non-reasoning
(`narrative`), `D_gauss(σ_matched)` is measurably larger on the former
at matched σ. Interpretation: state work is *task-conditional*, not a
constant background process — this is the H8 claim most directly.

**Support requires ≥ 2 of {A, B, C} at Cohen d ≥ 1.0.**

## Falsification

**Sustained failure across (A, B, C).**

If Gaussian noise scaled to 10 % of state norm shifts next-token KL
by less than the seed-resampling baseline; if no layer is more than
2× the median layer; and if narrative and medium show statistically
indistinguishable `D_gauss` — the state is, causally, not doing much
work. In H8's language, "state-as-computation" is metaphor rather
than mechanism at this scale.

Consequence: same staged flow as A0.4 (verify implementation, repeat,
declare). A refuted H8-causal is a **stronger** signal than a refuted
A0.4 because it's causal — it should carry more weight in the
backbone-decision reopening (see ROADMAP Gate 1).

## Cost model

A0.4 = 128 autoregressive decode steps × N seeds per cell.
A0.5 = k checkpoints × M corruption types × N seeds × **1 forward
pass** (plus optional N-step continuation for trajectory metrics).

Concretely on 0.4B, at ~1.5 tok/s:

- k = 5 checkpoints, M = 6 corruption types (`gauss(σ) × 3 scales` +
  `zero_layer` sampled + `shuffle` + baseline), N = 3 seeds, 2 models.
- Base cost: 5 × 6 × 3 × 2 × (1 forward) = 180 forwards + prefill
  overhead ≈ ~4 min for the single-step KL numbers.
- With N=16 step continuation for trajectory: 180 × 16 = 2880 additional
  forwards ≈ ~30 min.

Order of magnitude cheaper than the A0.4 sweep. Fits comfortably in
one session.

## Order of execution

1. Finish A0.4 full sweep (already in flight; results are useful as
   *description* even if not causal).
2. Sanity-check the causal apparatus on a synthetic tiny model
   (identity map perturbed by known noise) — verifies KL, softmax,
   and forward routing are wired correctly.
3. Run A0.5 on `World-0.4B × medium` first (single cell, k = 5, N = 3,
   all M corruptions). Inspect. Adjust σ scales so `gauss` sweeps
   through the interesting regime.
4. Full 2 × 2 grid: `{World-0.4B, G1d-0.4B} × {medium, narrative}`.
5. Write into `results.md` alongside A0.4 verdict.

## Open decisions (to resolve before running)

- **Where to hook the intervention.** `probe.py` runs the rwkv package's
  `model.forward(idx, state)` per token. State is fully exposed as a
  Python list. **Intervention is a direct write on `state[3*i + 1]`
  before the next `forward` call** — no C-side patching, no hook
  registration. This is one of the reasons we went with the rwkv
  package over HF+FLA.
- **Softmax temperature for KL.** Compute KL on raw logits →
  `softmax(logits, T=1.0)`. Same temperature for both distributions
  eliminates one confound. Sampling temperature is separate.
- **Numerical precision.** bf16 → fp32 before softmax + KL. Otherwise
  the bf16 noise floor is at the same order as small perturbation
  effects.
- **Sequence-level metric ambiguity.** Two divergent continuations may
  eventually re-converge in language space (both reach a reasonable
  answer through different words). Report `token_overlap_N` and
  `cum_KL_N` separately; don't collapse.

## Why this before A1

If H8-causal fails cleanly, the "train against state trajectory" branch
of A1 is closed — same conclusion as A0.4 stage-3 refutation, but with
causal evidence, not correlational. That would tighten the A1 loss
formulation to standard SFT and close a design branch that would
otherwise linger.

If H8-causal passes, we have the strongest possible in-scope evidence
that RWKV-7 state is doing computational work, which materially
strengthens the backbone wager (H4b) even without matched-arch
comparisons.

Either outcome saves weeks of A1 uncertainty. This is the definition
of a P9 (falsify-before-you-build) experiment.

## Related

- H8, H9 (`HYPOTHESES.md`).
- P8, P9, P12 (`docs/principles.md`).
- ROADMAP §Gate 1.
- A0.4 (`results.md`) — provides descriptive baseline, doesn't resolve
  H8 by itself.
