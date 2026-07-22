# A0.6 — Intra-model state-portability verdict

> Runs: 2026-07-22. World-0.4B and G1d-0.4B, 3 prompt pairs × 2 directions
> × 3 depths × 2 modes − skips = 30 cells per model. Full per-cell dumps
> in `results/a06/`. Raw aggregator output in `results_a06_summary.md`.

## TL;DR

Both models **FAIL** the model-level verdict — 0 of 3 pairs meet the
2-pair quorum on core cells (mode=full × depth ∈ {mid_B, after_B}).

But the picture is more textured than a binary FAIL. State carries
**real lexical content** across the swap (Δhit_donor consistently
positive in before_B and mid_B cells) — the state is not noise. It
just doesn't dominate the 64-step continuation trajectory strongly
enough to pull alignment past the −0.30 bar in most cells. So the
falsifier for H8 tightens: state *is* portable content, but *thin* —
enough to bias next-token distributions, not enough to override a
fresh prompt's structural pull.

| model | pass pairs | verdict | strongest cell |
|-------|-----------|---------|----------------|
| world-0.4b | 0 / 3 | FAIL (2 CAVEAT pairs, 1 FAIL) | code_prose AB/after_B/full: alignment=**−0.946**, Δhit=+0.259 |
| g1d-0.4b   | 0 / 3 | FAIL (2 CAVEAT pairs, 1 FAIL) | code_prose AB/after_B/full: alignment=**−0.327**, Δhit=+0.444 |

## Per-pair verdict

### world-0.4b

| pair | core | pass | fail | caveat | verdict |
|------|------|------|------|--------|---------|
| code_prose | 4 | 1 | 2 | 1 | CAVEAT |
| math_narr | 4 | 0 | 4 | 0 | FAIL |
| tasklist_refl | 4 | 0 | 3 | 1 | FAIL |

### g1d-0.4b

| pair | core | pass | fail | caveat | verdict |
|------|------|------|------|--------|---------|
| code_prose | 4 | 1 | 2 | 1 | CAVEAT |
| math_narr | 4 | 0 | 2 | 2 | CAVEAT |
| tasklist_refl | 4 | 0 | 3 | 1 | FAIL |

Full per-cell tables in `results_a06_summary.md`.

## Cross-cutting observations

1. **The one PASS pair is the same in both models.** `code_prose`,
   direction `AB` (donor = code state, recipient = prose prompt), depth
   `after_B` (swap right before first decoded token), mode `full`. This
   is the maximally-loaded cell: donor state is the most recently
   installed thing, and the whole state (not just hotspot layers) is
   replaced. When state gets the last word, portability is loud.

2. **Direction asymmetry: AB > BA everywhere.** In both models, AB
   yields the strong PASS on code_prose; BA never PASSes. Interpretation:
   code prompts install a state with sharper structure than prose (small
   vocabulary, high syntactic constraint) → prose recipients are more
   easily pulled into it. Prose→code is the harder direction because
   prose state is under-structured.

3. **Hotspot mode is almost universally FAIL.** 11 of 12 hotspot core
   cells across both models fail. This partially refutes the H8-causal-C
   inference from A0.5: the load-bearing layers found by zeroing loss
   are not sufficient carriers when the swap is *content* rather than
   ablation. State portability at the layer level is diffuse; can't
   compress to 3 layers.

4. **Δhit_donor is often positive even where alignment doesn't pass.**
   e.g. world-0.4b `tasklist_refl AB before_B full`: alignment=−0.020
   (not below −0.30), but Δhit_donor=+0.273. Meaning: the donor's task
   lexicon *does* leak into the recipient continuation, but not so
   strongly that cumulative-KL trajectory diverges from clean-recipient
   more than from clean-donor. State reweights the token distribution
   at margin without controlling the long-range plan.

5. **G1d ≈ World on portability magnitude.** G1d has more CAVEATs
   (4 vs 2), fewer straight FAILs on math_narr, and a stronger Δhit on
   its one PASS (+0.444 vs +0.259). Weakly supports the user's
   "world stores noise, g1d structures" hypothesis: G1d state is more
   *legible* per unit alignment, but neither model clears the quorum
   bar. Direct test of that hypothesis is A0.7 tier-1 (cross-checkpoint):
   if g1d→world transfers better than world→g1d, structuring wins.

## What this reframes

**H_portability (informal claim being probed).** "WKV state carries
portable semantic content that continues to steer decoding after prompt
handoff." — Data supports the *carries content* half (Δhit_donor
positive, not just noise) and refutes the *dominates decoding* half
(alignment rarely below −0.30 outside cell 1).

**Consequence for the noesis runtime.**

- State is **not** a semantic override switch. Injecting state won't
  make the model act as if it had run a different prompt.
- State **is** a compressed context bias. Useful as a warm-up
  conditioning source (e.g. handing off task-lexicon priors), not as a
  full context replacement.
- Runtime implication: rely on the **H10 stack** (test-time compute:
  state refinement + CoT budget) for behaviour steering within one
  model; rely on **H11** (zone-typed lens via structured text) for
  cross-model / cross-checkpoint handoff. State-transplant alone is
  not enough.

## Verdict-rule reflection

The verdict rule is calibrated correctly for the question it asks. The
CAVEAT zone did what it should — flagged non-trivial signal even where
alignment falls short. Nothing to relax: the failure mode here is
substantive, not a threshold artefact.

## Follow-ups

- **A0.7 tier-1** (in flight) — does the same thin-portability picture
  survive a checkpoint swap? If world→g1d and g1d→world both stay
  weakly-below-a-06 baseline, cross-checkpoint transfer is not a runtime
  substrate. If asymmetric, one model's WKV format is closer to the
  other's than reverse, and the runtime can be built around
  "canonicalise state through the more-structured model".
- **Effort registry (H10, `docs/effort-frontier.md`)** — the fact that
  state carries lexical bias but doesn't override plan gives extra
  weight to the `state_readout` mode: instead of trying to *inject* a
  state and let it fight the prompt, decode its content directly and
  put it back through prompt_cot with a clean prompt scaffold. Test
  cell in H10 sweep will directly compare.
- **Layer-level portability** — hotspot mode's failure suggests a
  per-layer alignment/lexicon breakdown would clarify whether *any*
  layer subset is a load-bearing carrier under content swap. Deferred
  to A0.7 CAVEAT-zone follow-up per README.

## Files

- `results/a06/*.json` — 60 per-cell dumps (30 per model).
- `results_a06_summary.md` — full raw verdict tables.
- `verdict_a06.py` — regenerate summary from JSON dumps.
