# H12a v2 — probe design fix for the N ↔ gap confound

## The v1 confound (registered 2026-07-23)

The v1 width sweep grows both `N` (number of triples) and the mean
gap-to-question distance together — each triple adds tokens, so N=64
lands with gap ≈ 107 while N=4 lands with gap ≈ 8. A fall in
accuracy across N cannot be attributed to width vs decay.

The v1 distance sweep (fixed N=8, varying filler between pairs) is
uncontaminated and did resolve **decay-axis: PROVEN** (recall
0.40 → 0.02 across gap 14 → 229).

**Width axis remains unresolved.**

## v2 design candidates

Three candidates, evaluated on cost and cleanliness:

### Candidate A — fixed farthest-pair gap, vary density

Pack N triples into a fixed-length window `W` at the end of the
context. Pre-window filled with distractor tokens (same vocabulary,
distinct non-planted colours). Vary N by density within `W`.

- Gap of the *farthest* pair to the question: constant at `W`.
- Gap of the *nearest* pair to the question: also constant (small).
- Mean gap: constant (approximately `W/2`).
- Confound: density itself may cause interference — collision of
  entity representations in nearby positions can be a distinct
  failure mode from "too many things to hold active".

Verdict: cleaner than v1 but introduces a **density confound**.

### Candidate B — same-context, vary asked-subset (recommended)

Build one fixed context with `N_max = 64` triples at fixed
positions. For each N ∈ {2, 4, 8, 16, 32, 64}, ask a question that
requires holding only `N` of those triples simultaneously — e.g.,
"among items {alpha, beta, gamma, delta}, which pairs share a
colour?" for N=4, up to "among all 64 items..." for N=64.

- Same tokens, same layout, same decay per position across all N.
- Only the *required active-count* varies.
- No density confound — layout is identical.
- Confound: model has to *decode which N items are in scope* from
  the question. This is a small linguistic step, but it does add
  a "filter the mention list" subtask. Mitigate by making the
  filter trivial (contiguous ranges: "items 1 through N") or by
  giving the model an explicit list of item names in the question.

Verdict: **cleanest of the three.** Filter subtask is cheap and
symmetric across N (a controllable, measurable overhead), unlike
the v1 gap-coupling which grew monotonically with N.

### Candidate C — fixed-context length, vary N by trimming distractors

For each N, build a context with `N` planted triples + `K − N`
distractor triples, total `K` triples throughout. Question asks
about the planted set only.

- Total context length constant.
- Mean gap for planted items roughly constant (~C/2).
- Confound: at low N, most of the context is distractors, so the
  "signal-to-noise ratio" in what the state has to hold shifts.
  This is a distraction-tolerance confound, distinct from
  active-width.

Verdict: harder to interpret than B.

## Recommended v2 = Candidate B

**Implementation delta from v1**:

1. Add a mode to `gen_triples.py` that emits a *single fixed*
   context with `N_max = 64` triples, plus a set of `question`
   files each targeting an ascending subset of items.
2. Each `tasks-N{n}.jsonl` entry: same `context` field
   (identical across the file family), different `question` field
   selecting the N-item subset.
3. Ground-truth pair-list is precomputed per N by the generator
   from the shared context.
4. Runner (`run_probe.py`) already handles per-item eval; only the
   task-loading path needs to be updated to share context across
   N conditions (currently loads one context per task).

**Metric.**

- **Per-item recall.** For each planted pair `(x, y)` in the asked
  subset, fraction of runs where the model emits both x and y.
  Predicted flat in N if width is not the bottleneck; falls with
  N if width is.
- **Enumeration completeness.** Fraction of runs where the model
  emits *all* planted pairs in the asked subset. Falls faster than
  per-item recall as N grows if width is the bottleneck (all-N
  simultaneous holding is harder than any-one-of-N).
- **Position-conditioned recall.** Break down per-item recall by
  where in the context the pair sits (front vs middle vs back
  of the context). Position-independence is the width signal;
  position-dependence is residual decay leakage.

**Predicted signature (H12a width hypothesis).**

| condition                     | expected shape        |
|-------------------------------|-----------------------|
| Per-item recall vs N          | falls monotonically   |
| Position-conditioned recall   | roughly flat within N |
| Enumeration completeness vs N | falls faster than per-item recall |

**Predicted signature (null H12a — width not the bottleneck).**

| condition                     | expected shape        |
|-------------------------------|-----------------------|
| Per-item recall vs N          | roughly flat          |
| Position-conditioned recall   | falls with position   |
| Enumeration completeness vs N | tracks per-item recall (no super-linear compound) |

**Budget.** v1 sweep (N=30, G1d 0.4B, CPU) took ~4 h wall on
i5-1235U. v2 shares the same runtime; the added shared-context
mode does not increase compute per task, only the tooling. Wall
estimate: ~4 h for the full sweep, same as v1.

## What this unlocks (and does not)

- **Unlocks:** clean attribution of width-vs-decay bottleneck
  for the H12 family; feeds the H12a v2 verdict into the H12b
  interpretation loop.
- **Does not block:** H12b itself is now decoupled from H12a per
  the 2026-07-30 reframe. H12b's LoRA-expansion treatment can be
  built and measured independently; H12a v2 tells us *why* it
  worked or didn't, not *whether* to try.

## Related

- hypotheses/README.md § H12 — parent hypothesis, reframed 2026-07-30 to
  decouple H12b from H12a v2.
- `results-g1d-n30/REPORT.md` — v1 sweep results with the
  confound clearly visible.
- `results-g1d-n30-adaptive/REPORT.md` — v1 adaptive follow-up,
  same confound.
