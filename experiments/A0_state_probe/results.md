# A0.4 — state-utilisation probe results

> **Status.** Sweep complete (2026-07-21). Verdict: H8 REFUTED-on-direction at 0.4B scale on both World and G1d; H9 BELOW-THRESHOLD (only `sr_std` supports, 1/3 of the rule). A0.5 (causal intervention) is now the load-bearing H8 test. See §Interpretation and `A05_intervention_plan.md`.

## Setup

| host        | hardware                                     | dtype | framework                             |
|-------------|----------------------------------------------|-------|---------------------------------------|
| `127.0.0.1` | i5-1235U, 32 GB RAM, Intel Iris Xe (no CUDA) | bf16  | BlinkDL `rwkv` 0.8.32 + torch 2.11 CPU |

Models (native bf16 `.pth`, no GGUF, no quantisation):

- **World-0.4B** — `BlinkDL/rwkv-7-world :
  RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth`. RWKV-7 "Goose" base,
  24 layers × 16 heads × 64-dim.
- **G1d-0.4B**   — `BlinkDL/rwkv7-g1 :
  rwkv7-g1d-0.4b-20260210-ctx8192.pth`. Reasoning-line 0.4B ("G1d"
  variant — the 0.4B slot in the G1 line; G1h exists only at 2.9B).
  See `HYPOTHESES.md` H9 for framing.

**Scale caveat.** Original plan targeted 2.9B pairs (World3 + G1h).
CPU-only bf16 throughput on i5-1235U measured at ~0.07 tok/s for 2.9B —
`medium` (880 tokens prefill + 256 decode) = ~4.5 h per seed, 4 cells ×
3 seeds = ~54 h. Not feasible for a laptop session. Fallback: run the
same experiment on the 0.4B pair (~1.5 tok/s, 4×3 seeds ≈ 2.3 h). H8/H9
verdicts here therefore describe the **small-model regime**; a 2.9B
re-run on GPU is a `ROADMAP` follow-up.

Both loaded via BlinkDL's `rwkv` PyPI package (CPU path). The HuggingFace
+ FLA path was rejected because `flash-linear-attention` requires triton,
which is CUDA-only in practice.

Prompts (`prompts.py` re-exports from `../A0_baseline/prompts.py`):

- **`medium`** — event-stream digest, reasoning-flavoured, ~280 words /
  ~890 tokens. Used as the reasoning cell for H8 / H9.
- **`narrative`** — descriptive prose of Lower Ashcombe village, matched
  length, no task / no reasoning demand. Used as the non-reasoning
  control for H8.

Metrics (`metrics.py`):

- **`delta_norm`** — L2 norm of state change between consecutive
  tokens, pooled across all 32 layers × 40 heads. "How much did the
  state move?"
- **`curvature`** — L2 norm of the second difference of the state
  trajectory. "Straight line (memory update) or bending
  (computation)?"
- **`stable_rank`** — per-head `(‖A‖_F / ‖A‖_2)^2` per RWKV-7 paper
  Appendix J. Distribution collapsed to mean / std over layers × heads
  per token. "How many effective directions is the state using?"

All metrics accumulate in fp32; input state tensors are bf16 from the
rwkv package's forward pass.

## Pilot — noise floor

3 seeds × 128 tokens on `World-0.4B × medium`. Between-seed SD is what
the pre-registered `Δ_min = 3 × SD` threshold locks against.

Wall time: 2061 s total (3 × ~690 s / seed at ~1.5 tok/s).

| metric                          | seed-mean of per-step mean | between-seed SD | pilot noise floor 3·SD |
|---------------------------------|----------------------------|-----------------|------------------------|
| `delta_pooled`                  | 39.91                      | 2.97            | **8.90**               |
| `curvature_pooled`              | 62.52                      | 4.50            | **13.49**              |
| `stable_rank` (per-step std)    | 0.571                      | 0.163           | **0.49**               |

Between-seed SD on `stable_rank` per-step std is ~29% of the mean —
much wider relative noise than on `delta` (7%) or `curvature` (7%).
This means the `stable_rank` metric will need a larger absolute effect
to clear Cohen's d ≥ 1.0 in the full sweep. Delta and curvature are the
metrics that will most likely carry H8/H9.

## Full sweep — 4 cells

`{World3, G1h} × {medium, narrative}`, N seeds/cell, 256 tokens.

*Filled after sweep.*

### `delta_norm` (pooled, mean across steps × seeds)

| model | medium (mean ± sd) | narrative (mean ± sd) | reasoning − narrative | Cohen d | direction |
|-------|--------------------|-----------------------|-----------------------|---------|-----------|
| World-0.4B | 37.00 ± 3.70 | 46.87 ± 0.46 | **−9.87** | 3.74 | **wrong** (narr > med) |
| G1d-0.4B   | 34.98 ± 0.98 | 39.16 ± 0.99 | **−4.17** | 4.22 | **wrong** (narr > med) |

### `curvature` (pooled, mean across steps × seeds)

| model | medium (mean ± sd) | narrative (mean ± sd) | reasoning − narrative | Cohen d | direction |
|-------|--------------------|-----------------------|-----------------------|---------|-----------|
| World-0.4B | 57.85 ± 5.95 | 73.71 ± 0.87 | **−15.86** | 3.73 | **wrong** (narr > med) |
| G1d-0.4B   | 55.34 ± 1.70 | 62.45 ± 2.25 | **−7.11**  | 3.57 | **wrong** (narr > med) |

### `stable_rank` (per-step std, mean across steps × seeds)

| model | medium (mean ± sd) | narrative (mean ± sd) | reasoning − narrative | Cohen d | direction |
|-------|--------------------|-----------------------|-----------------------|---------|-----------|
| World-0.4B | 0.569 ± 0.147 | 0.681 ± 0.013 | **−0.112** | 1.07 | **wrong** (narr > med) |
| G1d-0.4B   | 0.775 ± 0.033 | 0.663 ± 0.010 | **+0.112** | 4.59 | **right** (med > narr) |

### H9 contrast — G1d vs World-0.4B on the reasoning prompt

(Original plan targeted G1h; at 0.4B slot the G1 line is `G1d`, so contrast is same-scale.)

| metric              | World-0.4B medium | G1d-0.4B medium | Δ (G1d − World) | Cohen d | direction |
|---------------------|-------------------|-----------------|-----------------|---------|-----------|
| `delta_pooled`      | 37.00 ± 3.70      | 34.98 ± 0.98    | −2.02           | 0.75    | ~tied     |
| `curvature_pooled`  | 57.85 ± 5.95      | 55.34 ± 1.70    | −2.52           | 0.58    | ~tied     |
| `stable_rank` std   | 0.569 ± 0.147     | 0.775 ± 0.033   | **+0.206**      | **1.93**| **right** (G1 > World) |

## Pre-registered thresholds (locked from pilot)

Per H8 / H9 threshold-lock policy (see `../../HYPOTHESES.md`):

- **`Δ_min` per metric** = 3 × between-seed SD from the pilot (this
  file's pilot section).
- **Effect size** — Cohen's `d = |mean_reasoning − mean_narrative| /
  pooled_sd`; support requires `d ≥ 1.0` on ≥ 2 of the 3 metrics with
  Welch's t-test `p < 0.05 / 3` (Bonferroni).
- **H9** — same criterion applied to the `G1h medium` vs `World3
  medium` contrast.

## Pass/fail

| hypothesis | cell | rule | metrics passing `\|d\| ≥ 1.0` | in H-predicted direction | verdict |
|------------|------|------|------------------------------|--------------------------|---------|
| H8 | World-0.4B med vs narr | ≥ 2/3 with `d ≥ 1.0` in reasoning > narr direction | 3/3 (delta 3.74, curv 3.73, sr 1.07) | **0/3** — all show narr > med | **REFUTED-on-direction** |
| H8 | G1d-0.4B med vs narr   | ≥ 2/3 with `d ≥ 1.0` in reasoning > narr direction | 3/3 (delta 4.22, curv 3.57, sr 4.59) | **1/3** — only sr_std shows med > narr | **REFUTED-on-direction** |
| H9 | G1d vs World on medium | ≥ 2/3 with `d ≥ 1.0` in G1 > World direction        | 1/3 (sr_std 1.93; delta/curv d≈0.6–0.8) | 1/1 of the qualifying metric is in the right direction | **BELOW THRESHOLD** (1/3, need 2/3) |

**Note on Bonferroni.** With n=3 seeds per cell, Welch's t-test df is 2–4 and Bonferroni p<0.0167 is at the edge of what 3 seeds can resolve. For H8 (World, delta) the two-sided Welch t ≈ 4.58, df ≈ 2.0, p ≈ 0.045 — passes uncorrected, fails Bonferroni. The verdict above is based on Cohen's d + direction; the descriptive result is unambiguous regardless of whether the p-test bar is cleared, because the direction disagrees with the hypothesis.

## Interpretation

The 4-cell sweep does **not** support H8 by its own pre-registered rule. Direction is wrong on the two energy-of-motion metrics (delta, curvature) for both models — narrative elicits *more* state motion than medium — and only one metric on one model shows the H8-predicted "reasoning uses state more" pattern (G1d sr_std).

**Most parsimonious reading of the direction reversal.** The narrative prompt is a dense village-description with novel proper nouns and adjectives per line, i.e. a high *content novelty* stream. The medium prompt is a repetitive event-stream digest with structural sameness. At 0.4B, δ and κ appear to be dominated by content-tracking rather than by reasoning-computation, so a more-novel prompt drives more state motion regardless of reasoning demand. This is consistent with the state acting as a rolling summary at this scale, which is what H8 was meant to falsify.

**H9 is not resolved either.** The `sr_std` signal (G1d 0.775 vs World 0.569, d=1.93) is genuine and consistent with G1 post-training pushing state to use a wider effective-rank envelope. But delta and curvature are effectively tied between the two models on the reasoning prompt, so the 2-of-3 rule fails. Call H9 **weak / preliminary support on sr_std only**.

**What this hands to A0.5.** A0.4 measures how state *moves*; H8 asks whether state does *work*. The direction reversal is exactly the ambiguity a descriptive probe can't resolve: state might move less on reasoning because it's doing *targeted* computation instead of *bulk* content-tracking, or because the model simply isn't reasoning at 0.4B and both prompts look the same to it. The paired-perturbation KL design in A0.5 is set up to discriminate: if reasoning-time state carries computation, perturbing it should shift outputs more on medium than on narrative — regardless of how much state moves in either case. **A0.5 is now the load-bearing test for H8, not a supplement.**

**Scale caveat reinforced.** All four cells are 0.4B on CPU. At this parameter count, "medium" reasoning is probably not eliciting real chained inference — both models likely fall back to fluent completion. A re-run at 2.9B on GPU (ROADMAP Gate-2) is the proper descriptive H8 test. The current numbers describe the small-model regime only.

**Small consistency caveat.** `world_medium` was executed this session with `--max-new-tokens 256`; the other three cells (from the prior session) used 128. All three metrics are per-step means/stds, which are stationary in expectation, so the mean estimates remain comparable — the longer run gives a slightly tighter per-seed SD (which pushes Cohen's d up on that cell only). If future re-analysis wants strict token-count parity, re-run the three 128-token cells at 256.

## Notes on measurement fidelity

- **State layout.** BlinkDL `rwkv` package stores state as a flat list
  of `3 · n_layer` tensors; WKV lives at `state[3·i + 1]`, shape
  `[n_head, head_size, head_size]`. For 2.9B World3 that is
  `[40, 64, 64]` per layer, 32 layers → ~20 MB fp32 per token. The
  probe reads and detaches these tensors on every step; nothing is
  retained between metric calls.
- **Precision.** Weights + state carried in bf16 through the forward
  pass (paper §8 setting); all three metrics cast to fp32 before
  accumulation. Pooled norms further accumulate in fp64 across layers
  to avoid catastrophic cancellation at large layer counts.
- **Nucleus sampling (`temperature=1.0`, `top_p=0.85`).** Greedy decode
  on a deterministic model gives per-seed variance ≡ 0 — the seeds
  would be measuring bf16 reduction-order noise rather than any real
  decode stochasticity. The probe therefore samples with a fixed
  seed-per-run, top-p nucleus. This is the same setting the noesis
  runtime uses, so the between-seed variance we report is the real
  distribution of state trajectories the runtime will actually walk.

## Follow-ups

- **Sampled decode for real between-seed variance.** See fidelity note
  above.
- **G1 vs G1h distinction.** The plan tests G1h. If the checkpoint URL
  falls back to G1 base, note it in this file and adjust the H9 claim
  scope.
- **Larger sweep with 13.3B.** Out of budget for this session; noted in
  ROADMAP as a Gate-2 optional follow-up.
- **Additional Appendix-J metrics.** RMS of `A - A.T` (a-symmetry) and
  Frobenius norm of the state, both quoted in the paper's own probe
  section. Considered for a follow-up if the current three metrics do
  not discriminate.
