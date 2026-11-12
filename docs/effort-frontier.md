# Effort frontier: the design record of the tick axis

The claim lives in `hypotheses/H10.md`: after training for it, accuracy rises with the number of ticks
`M` on task families one pass cannot solve, and is flat elsewhere. This document keeps how the axis was
designed, what the design predicted, and which of those predictions are settled. Notation is the frame's
(`hypotheses/README.md` §2). Rewritten 2026-11-11 from the versions of 2026-08-17 and 09-15.

## 1. One axis

Three knobs were planned first (2026-07-22): `N` re-reads of the prompt, `K` emitted reasoning tokens,
and a readout mode. On 2026-08-17 all but one were retired:

| old knob | verdict | reason |
|---|---|---|
| `N`, re-feed the prompt | retired | the loop prefills once; re-reading is the known recall effect of repeated context (arXiv 2407.05483), not computation |
| `K`, decoded reasoning tokens fed back | absorbed | sampling a token and feeding it back is a tick with a discrete feed |
| readout mode | retired | it named where `K`'s tokens came from relative to the prompt |
| `M`, ticks | **the axis** | each tick feeds the model's own output back without emitting a token |

What a tick feeds back changed twice. `discrete` (a sampled token) and `expected` (`softmax(logits) @ E`)
were the first two feeds. On 2026-10-21 the channel became the model's own last-layer vector, with no
vocabulary rows: tokens appear only where the model answers. This also closes the August confusion of
two mechanisms under one letter (`wkv_loop.py`'s self-feed steps against ThinkChain's distinct trained
markers): `M` is the number of ticks.

## 2. Why ticks could help: the state is wider than one read

Re-measured 2026-09-13 on G1i 2.9B base, 32-token prompt, per head, 64 dimensions
(`experiments/A0_state_probe/results/rank_recheck/jlens.json`):

| L | stable rank | entropy rank | directions > 1% of σ₁ | per head min / median / max |
|---|---|---|---|---|
| 0 | 1.291 | 6.50 | 13.2 | 3 / 12 / 26 |
| 4 | 1.211 | 6.44 | 15.2 | 5 / 16 / 28 |
| 8 | 1.173 | 5.86 | 13.7 | 6 / 13 / 26 |
| 12 | 1.181 | 5.88 | 13.7 | 5 / 13 / 22 |
| 16 | 1.229 | 6.36 | 14.7 | 7 / 15 / 22 |
| 20 | 1.247 | 6.98 | 15.8 | 8 / 16 / 26 |
| 24 | 1.095 | 3.60 | 8.4 | 4 / 8 / 20 |
| 28 | 1.148 | 4.50 | 11.0 | 5 / 11 / 16 |

Stable rank is an energy-concentration ratio, not a count: one dominant singular value with thirty-one
real ones scores about 1.03 (`experiments/A0_state_probe/test_jlens_spectrum.py`). The state is
superposed, not collapsed: 13-16 live directions of 64 mid-stack, narrowing at L24-L28 just before the
readout. Width and strength are close to independent across heads (corr 0.30 at L4); each head is
normalised separately (`ln_x` is a GroupNorm with one group per head), so quiet heads are equalised, not
ignored.

The reading that followed: a chain of distinct steps can reach directions one read does not. Measured the
same month (`docs/phase15-gap-matrix.md`): repeating one constant marker converges within about four ticks
(`delta_cos_prev` → 0.997), while switching to a new marker turns the direction almost orthogonally
(0.097). The tick lemma (`docs/state-and-reasoning.md` §3.1) explains the first: frozen controls settle at
rate `ρ(A)`. The open-loop lemma (§3.2) makes the second the only source of gain: ticks help only when the
fed vector keeps changing the controls.

## 3. Predictions and where they stand

| prediction (date registered) | state |
|---|---|
| Untrained, the curve is flat and low (2026-08-17) | holds: G1i base self-feed with `M_max` 16 gives 12.5% (6/48) against 33.3% for one read with a 32-token readout; `K` = 0 gives 0/48 and `K` = 128, 512, 2048 give 4/48 each on `step4_merged`; the plain model re-reading scores 0.469 against 0.531 at once, ticks of its own expectation 0.344 (`HZLK.md` evidence) |
| Trained, the curve rises (2026-08-17) | not seen: every trained checkpoint is a pass-through on families one pass solves after distillation (`H10.md` §3) |
| Useful `M` saturates near the number of live directions, about 16 (2026-09-13) | untested |
| A discrete feed recovers less breadth than a continuous one, and the gap widens with `M` (2026-09-13) | moot: both vocabulary feeds were dropped for the model's own latent |
| Renormalising the fed vector into the token-norm range changes what the loop decodes to (2026-09-15) | moot for the same reason. Measured then: a token embedding has median norm 0.3757 and maximum 0.6421 over 65 536 tokens; the trained phase marker 1.0138; the expected embedding falls from 0.93× a median token at top-1 probability 0.99 to 0.09× at 0.01 (`experiments/rl/marker_scale_probe.py`, `experiments/A0_state_probe/results/marker_scale_step500.json`) |
| On linear systems the frontier is a step at the rank of the system (2026-08-17, from H25) | untested; restated in `H25.md` §6 |
| Silent steps buy computation, shown by the parity toy (2026-09-04) | withdrawn 2026-11-11: open-loop cell, one seed (`H25.md` §2) |

## 4. Cost and the metric to report

A tick costs one pass through the layer stack: total compute is about `hidden × n_layer × (L + M)` for a
prompt of `L` tokens, linear in `M` like the old `N`. Accuracy alone would also rise if the model simply
spoke more, so the frontier is reported with the emitted tokens: DE = accuracy / mean words of a correct
answer × 100 (H24), or tokens at equal accuracy (`HZLK.md` §2).

Why ticks are not supervised on human reasoning text: in RWKV-7 there is no attention over earlier steps;
a step's only effect is how it moves the state, so optimising intermediate steps for readability spends a
degree of freedom the architecture does not use.

## 5. Retired: the self-feed loop trained by reward

From 2026-08-16 to 08-19 the loop was trained by GRPO with reward `r_correct − β·M − γ·Σ ReLU(ΔH_t)` and an
exit on entropy plateau or confidence, with a planned effort registry (`fast`, `normal`, `deep` mapped to
`M_max`) and a sweep `M ∈ {0, 1, 2, 4, 8, 16, 32}` per task family. Paused 2026-08-19: a content decoder
found that the loop's steps were never task content on any checkpoint (chat-template scaffolding), and the
chosen `M` was constant with zero variance, so reward had nothing to select. Distillation of the answer
through ticks (`HZLK.md` §6, S1) replaced it. An effort registry needs a trained checkpoint whose `acc(M)`
is not flat; none exists.

## 6. History

- 2026-07-22 three knobs; 08-07 the step-8 sweep; 08-12 the readout evaluator bug found and fixed.
- 2026-08-17 one axis; 08-18 first loop number on G1i base; 08-19 reward training paused.
- 2026-09-13 the breadth re-measurement; 09-15 the marker norm measurement.
- 2026-10-21 the loop channel becomes the model's own latent.
- 2026-10-27..29 trained ticks on G1d 0.4B and Qwen3.5 0.8B: pass-through (`H10.md`).
- 2026-11-11 rewritten as the design record; the claim moved to `H10.md`.
