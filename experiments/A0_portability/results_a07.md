# A0.7 tier-1 — cross-checkpoint state-transfer verdict

> Runs: 2026-07-22. World-0.4B ↔ G1d-0.4B, WKV-only transfer via
> `load_wkv_into_state`. 3 prompt pairs × 2 prompt directions × 3 depths
> × 2 modes = 60 cells across 2 checkpoint directions. Full per-cell
> dumps in `results/a07_tier1/`. Raw aggregator output in
> `results_a07_summary.md`.

## TL;DR

**FAIL — 0 of 6 (donor→recipient, pair) groups pass the ratio bar.** WKV
state does not survive a checkpoint swap in either direction.

| donor → recipient | pair | a07 core PASS | a06 baseline PASS | ratio | verdict |
|-------------------|------|---------------|-------------------|-------|---------|
| world-0.4b → g1d-0.4b | code_prose | 0/4 | 1 | 0.00 | FAIL |
| world-0.4b → g1d-0.4b | math_narr | 0/4 | 0 | n/a | FAIL |
| world-0.4b → g1d-0.4b | tasklist_refl | 0/4 | 0 | n/a | FAIL |
| g1d-0.4b → world-0.4b | code_prose | 0/4 | 1 | 0.00 | FAIL |
| g1d-0.4b → world-0.4b | math_narr | 0/4 | 0 | n/a | FAIL |
| g1d-0.4b → world-0.4b | tasklist_refl | 0/4 | 0 | n/a | FAIL |

The `n/a` ratios are pairs where the A0.6 same-checkpoint baseline was
already 0 — the transfer can't degrade what didn't exist. Since the
cross-checkpoint run also produced 0 PASSes on those pairs, the default
FAIL verdict applies.

## Symmetry — the "world stores noise vs g1d structures" test

The user's working hypothesis (2026-07-22) was that world-0.4b might
"store noise in WKV" while g1d-0.4b uses the state coherently. If true,
we'd expect **asymmetric** transfer: state from the structuring model
would land coherently in the noise-model's recipient position, but not
vice-versa (or vice-versa).

**Observed: symmetric FAIL.** Both directions score the same 0/12 on
core-cells. The signal is:

- Best alignment cell world→g1d: `code_prose / BA / after_B / full`,
  alignment = **−0.231** (Δhit_donor = +0.028). Doesn't clear −0.30.
- Best alignment cell g1d→world: `math_narr / BA / after_B / full`,
  alignment = **−0.135** (Δhit_donor = +0.077). Also short.
- Interesting outlier: g1d→world `code_prose / AB / after_B / full`
  scores only −0.044 on alignment but **Δhit_donor = +0.333** — the
  donor's lexicon leaks strongly, but the trajectory doesn't. Same
  pattern as A0.6's "state carries content, not plan" finding, now
  reproduced under a checkpoint swap.

If either model had a "readable" internal WKV format, one direction
should have produced at least one clear PASS. Neither did. Reframed:
**WKV state is checkpoint-private**, not model-family-generic. The
representation is entangled with the exact weight values, not just
the architecture.

## What this closes

- **Tier-2 (learned projector) is now the only cross-checkpoint path.**
  A0.7 tier-1's null result means a naive `load_wkv_into_state` won't
  work; if we want migration across model bumps, we need a trained
  projection between per-checkpoint WKV manifolds. Tier-2 was already
  Phase-2 deferred per README; A0.7 tier-1 confirms it's the only
  option, not a nice-to-have.
- **H11 becomes load-bearing for cross-checkpoint handoff.** The zone-
  typed lens via structured text is now the *only* viable channel between
  checkpoints, not one of several options.

## What this does *not* close

- **Within a single checkpoint** (A0.6), state carries thin but real
  content — that finding stands. H10 (test-time compute frontier via
  state refinement + CoT + readout mode) is still on the table because
  N > 1 refinement happens on the same weights.
- **Same-architecture tier-2 remains open.** A0.7 tier-1 showed
  bare-metal state doesn't transfer, but a learned projector between
  world-0.4b and g1d-0.4b WKV manifolds might. Not investigated here.

## Consequences for the runtime

- Do not design any migration protocol that assumes WKV state survives
  a checkpoint swap. `noesis-runtime` must either (a) freeze the
  checkpoint for the lifetime of a session, or (b) route migration
  through text-bottleneck handoff.
- The `readout_mode = state_readout` option in H10 (decode CoT-tokens
  from refined state) is now the natural bridge to (b): decode state
  content to text, feed text to the new checkpoint, restart. That path
  no longer needs cross-checkpoint state transferability.
- Effort-registry preset design (`docs/effort-frontier.md`) is not
  affected — H10 runs entirely within one checkpoint.

## Follow-ups (not on critical path)

- **Per-layer alignment breakdown** on the 4 marginal cells (both
  −0.10 to −0.20 alignment, non-zero Δhit_donor) — do they cluster on
  the same layer subset that A0.5's hotspot mode used? If yes, tier-2
  should target those layers first.
- **1.5B pair** (World-1.5B ↔ G1H-1.5B). README budgeted this as
  optional; A0.7 tier-1 landed FAIL not CAVEAT, so the extra evidence
  wouldn't shift the verdict. Skip unless something else motivates it.

## Files

- `results/a07_tier1/*.json` — 60 per-cell dumps.
- `results_a07_summary.md` — full aggregator output.
- `verdict_a07.py` — regenerate summary; ratio verdict rule.
