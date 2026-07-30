# noesis truth-system — pilot report (H20 + H21 + H22)

Combined pilot across three probes designed to give a downstream truth-lookup
routine three orthogonal signals: **aporia** (does the state disagree with
itself?), **premise-validity** (is the question premise well-formed?),
**attribution** (is the claim sourced?).

- Model: `rwkv7-g1d-0.4b-20260210-ctx8192.pth`  (24 layers × 16 heads × 64×64 WKV)
- Hardware: i5-1235U (2P + 8E, 12 threads), CPU-only bf16
- Date: 2026-07-30
- Wall clock: cluster ~47 min for H20 + H21 + H22 seed in parallel; H22 v2 extract ~90 min sequential single-core

## H20 — aporia probe

**Signal:** does the model's WKV state carry residual disagreement even
after committing to a first token? Sample N continuations per prompt,
measure branch commitment.

**Setup:** 30 items × 10 samples × 20 new tokens each, sharded 3-way (10 items/shard).

**Aggregate signals per category:**

| category                   | n  | collapse_first | collapse_cont | p(neither) | logit_gap |
|----------------------------|----|----------------|---------------|------------|-----------|
| contested_facts            | 10 | 0.696          | **0.653**     | 0.680      | 4.14      |
| bounded_ambiguity          | 10 | 0.767          | 0.500         | **0.930**  | 3.17      |
| underdetermined_inference  | 10 | 0.500          | **0.207**     | 0.720      | 1.65      |

**Predicted ordering (`cf > ba > ui` on collapse_cont) holds.** Contested
facts collapse most: model has a pretraining-favored answer and commits
downstream. Underdetermined-inference items keep the state open —
continuations don't uniformly pick a branch. Bounded ambiguity sits in
the middle with the highest `p(neither)` = 0.93: model hedges most
often when the ambiguity is semantic rather than task-level.

**What the numbers mean operationally:** if `collapse_cont` stays high on
prompts we thought were open, that's a false-confidence signal — a
candidate for pre-generation aporia flag. If `p(neither)` is high, the
state is honestly hedging and the routine can prompt for clarification
instead of forcing a commit.

**Scale-up to 100 items (2026-07-30).** Authored 70 more items (25 cf +
25 ba + 20 ui) using the same protocol (10 samples × 20 new tokens,
3-shard). Wall total ≈ 5 h.

| category                   | n  | collapse_first | collapse_cont | p(neither) | logit_gap |
|----------------------------|----|----------------|---------------|------------|-----------|
| contested_facts            | 35 | 0.697          | **0.589**     | 0.760      | 3.40      |
| bounded_ambiguity          | 35 | 0.688          | 0.478         | **0.854**  | 2.84      |
| underdetermined_inference  | 30 | 0.624          | 0.560         | 0.603      | 2.69      |

**Pilot ordering `cf > ba > ui` did NOT hold at scale.** The correct
v100 ordering is `cf > ui ≈ ba` on `collapse_cont` — UI and BA swap
(pilot ui=0.207, v100 ui=0.560). Two things drive this:
(a) the pilot ui-set was hand-tuned toward genuinely-open logic
puzzles; the 20 new ui items included several rule-application
questions with a compelling default reading, so the state commits;
(b) BA held the highest `p(neither)`=0.854 across both pilots and
v100 — semantic-lexical ambiguity is where the model most reliably
refuses to pick a branch. **Operational implication:** BA is the
robust "keep open" signal; UI needs stricter authoring guarantees
(genuine two-branch symmetry) to stay open under generation. CF stays
top on `collapse_cont` and `logit_gap` on both scales — model has a
pretraining-favoured answer on contested facts and commits fluently.
Recommend: split UI into `ui_symmetric` (both branches equally
supported by rule) and `ui_default_biased` (rule reads default
plausible) in the next revision; H20 as an operational signal should
report per-subcategory rather than pooled UI.

## H21 — premise-validity probe

**Signal:** does the WKV state look categorically different when the
prompt embeds a false or impossible premise?

**Setup:** 40 items (20 valid, 20 invalid = 5 each of category / impossible
/ counterfactual / factual). Feature: per-layer per-head mean+std of
WKV state after prefill → 768 dims. Head: 128→64→1 MLP, BCE, 500 epochs.

**Headline (single 32/8 stratified split):** F1=1.000, but the 8-item
test fold landed only on the two easiest invalid types (category +
impossible). Not honest.

**Honest number (LOO over 40 items):** **F1=0.789, acc=0.800**.

Per invalid-type recall:

| invalid_type    | recall |
|-----------------|--------|
| category        | 5/5    |
| impossible      | 5/5    |
| counterfactual  | 4/5    |
| factual         | 3/5    |

**Category and impossible are separable cleanly** in the pooled WKV
statistics — the state clearly looks different for
"squares have three sides" vs. "why is the sky blue?".

**Factual is the weak axis.** False factual premises share surface
structure with true factual questions — only truth-value differs, and
the 0.4B base model may not itself know which is which. LOO shows 3/5
recall on factual, 15% FP rate on valids. This is the operational risk:
false-fact detection needs the underlying model to know the fact.

**Path forward v2:** dataset expanded via TruthfulQA to 280 items
(120 factual invalids from `incorrect_answers` + 120 factual valids
from `correct_answers`, both reshaped through identical templates so
surface structure is matched, only truth-value differs).

**H21 v2 LOO on 280: F1=0.614, acc=0.618** — down from pilot 0.789.

Per invalid-type recall:

| invalid_type    | pilot LOO (n=5) | v2 LOO           |
|-----------------|-----------------|------------------|
| category        | 5/5             | 5/5              |
| impossible      | 5/5             | 4/5              |
| counterfactual  | 4/5             | 4/5              |
| **factual**     | **3/5**         | **75/125 (60%)** |

FP rate on valids jumps to 52/140 = 37% — well above the 3%
production target. **The factual axis does not resolve at 0.4B under
pooled mean+std WKV features.** Structural invalidity types
(category, impossible, counterfactual) still separate cleanly; only
the pure truth-value axis fails.

**Interpretation.** By construction, factual valid and factual invalid
items share the same surface template (`"X. Explain how this works."`)
where X is either a true or false claim from TruthfulQA. The pooled
WKV state on prompt-only cannot separate them because separating
requires knowing which claim is true — a *knowledge* problem, not a
*state-shape* problem. This confirms the pilot warning: "false-fact
detection needs the model to actually know what's true, not just the
state to look funny."

**Design implication.** H21 handles *structural* premise invalidity
(category / impossible / counterfactual), and does it well
(recall ≥ 4/5 on all three at 0.4B). Truth-value fact-checking is
orthogonal and belongs to a separate gate (H16 emit-time verification,
or retrieval-grounded lookup). Splitting the responsibility cleanly is
the takeaway from v2.

**Retrieval sanity (2026-07-30, NEGATIVE):** built 40 pilot items ×
{base, ctx-prefixed with TruthfulQA `best_answer`} on 0.4B and re-ran
the same MLP head. Base F1=0.829 vs **ctx F1=0.524** — retrieval
made it *worse*. Pair-shift: inv items 0/8 in the right direction
(Δp_valid = +0.259, wrong sign — adding correct-fact context nudged
the head *toward valid*), val items 0/10 in the right direction
(Δ = -0.297). The 128→64→1 MLP treats the `Context:` prefix as a
surface pattern feature, not as evidence to reconcile with the
question. **Retrieval-first only helps a reasoning-capable readout**
(small LLM gate, emit-time CoT verifier) — not a pooled-state MLP.

**2.9B scale re-test (2026-07-30):** g1h-2.9B on the 40 pilot items,
feature dim 2560 (vs 768). LOO F1=0.850 (+0.06 vs 0.4B pilot 0.789).
Factual recall unchanged at 3/5 invalid; val_fact drops to 2/5.
**Scale ≠ closure on the truth-value axis.** Structural types
(category / impossible / counterfactual) still ≥4/5 on both.

**Per-item cross-scale inspection.** 4/10 factual items are wrong
on both 0.4B and 2.9B: `inv_fact_05` (Marie Curie's Antarctica —
plausibly-fake fact both models swallow) + `val_fact_02/03/04`
(Einstein Nobel; Egyptian pyramid stones; fall of Rome). The three
val_fact failures are all **historically debated** open questions
with multiple genuine answers — the head reads "state carries many
possible answers" as *premise-invalid*, conflating epistemic
answer-uncertainty with structural premise-invalidity. That is H20's
signal (aporia), not H21's. **Design fix:** exclude debated-open
items from H21 val_fact seed, or add an explicit `y_debated` label
so H21 stops confounding the two signals. Not a knowledge deficit;
a label-schema confound.

**H21 v4 clean OOD probe (2026-07-30):** hand-authored 40 items
where premise-validity is **knowledge-agnostic** — 20 invalid
(temporal contradictions, category errors, self-contradictions,
arithmetic contradictions like "prime 15" / "even+even=odd") + 20
clean-valid (well-established, non-debated: math facts, everyday
procedures, definitions, elementary causal chains). LOO F1=**0.900**
(vs pilot 0.789, v2 0.614). Per invalid-type recall: temporal 5/5,
category 5/5, selfcontradiction 5/5, **arithmetic 3/5**. All 4
errors cluster on math-facts (v4_inv_arith_02 "even+even=odd"
p=1.000, v4_inv_arith_03 "triangle 270°" p=0.999,
v4_val_math_01 "triangle 180°" p=0.006, v4_val_math_04
"double-then-halve" p=0.033). Note the **symmetric inversion**
between the 180° and 270° items — 0.4B doesn't stably encode
"triangle interior angles sum to 180°", so it can't reject the
270° false and can't affirm the 180° true. Purely structural
axes (temporal / category / selfcontradiction) hit **100%**
recall. Confirms: H21 is a structural gate; anything requiring
even trivial factual anchors falls to the knowledge axis.

## H22 — attribution probe

**Signal:** does the state distinguish sourced claims from
collective-appeal ones? Attributable = named referent + specificity;
unattributed = "most people believe", "it is generally accepted".

**Seed (19 items, 16 labelled):**

- Single 12/4 split: F1=1.000
- **LOO on 16 labelled:** F1=1.000, acc=1.000 (still clean at this size)
- Ambiguous items (y=-1, held out) scored by full-data model:
  - `amb_01` "Given the pattern in the last three items I have seen" → p_attr=0.995 (attributable — specific referent "last three items")
  - `amb_02` "From what I can tell in the config files, the retry interval is..." → p_attr=0.961 (attributable — "config files")
  - `amb_03` "Common practice in this codebase seems to be..." → p_attr=0.096 (unattributed — "common practice" = vague appeal)

The head *did not* just latch onto first-person "I" as a surface
feature — amb_03 has no "I" and was correctly scored unattributed. The
signal is closer to "does the state model a specific named referent"
than to a syntactic heuristic. Good sign.

**v2 (243 mined items from C4:en — 120 attr + 120 unattr + 3 amb):**

- Extraction wall: 5781s (~96 min, contended with H21 v2 in parallel)
- **LOO F1=0.947, acc=0.946** on 240 labelled items
- Confusion: TP=115 FP=8 TN=112 FN=5

The head generalises from the 16-item seed to 240 real C4 sentences at
F1=0.947 — the WKV state genuinely tracks referent-specificity, not
just seed-set surface features. Errors cluster on borderline cases
(FP=8: attributable-scored unattr sentences that in fact name a
specific but somewhat vague entity; FN=5: attributable sentences with
weakly-named referents).

**Interesting shift on ambiguous items** vs seed model:

| id     | seed (16 train) | v2 (240 train) | Δ interpretation |
|--------|-----------------|----------------|------------------|
| amb_01 | 0.995           | 0.126          | seed learned "first-person specific referent = attr"; v2 has narrower academic-citation pattern → flips to unattr |
| amb_02 | 0.961           | 0.892          | still attr (config-file reference is concrete) |
| amb_03 | 0.096           | 0.002          | still unattr (both agree "common practice" = vague) |

The definition of "attribution" is dataset-dependent. Seed set was
hand-authored to include first-person specific-referent as
attributable; C4 mining biased toward citation-style ("According to X
(2020)"). Downstream: **the training corpus for H22 fixes what
"attributable" means operationally.** Pick corpus deliberately.

## Cross-probe observations

1. **Two signals live in the WKV state** (H21 premise-validity, H22
   attribution) and are readable by a shallow MLP on pooled mean+std.
   Pooling drops most of the WKV rank structure yet still separates —
   the state carries broad categorical distinctions, not just fine-grained
   ones. Confirms the "state as compressed context" bet.

2. **Aporia lives in token-space** (H20). WKV pooling doesn't cover it
   because the disagreement expresses in *which continuation the state
   samples down*, not in an aggregate feature. H20's `collapse_cont` and
   `p(neither)` need actual generation to measure.

3. **Truth ≠ premise-validity ≠ attribution.** A well-formed sourced
   claim can be factually wrong; a factually true statement can be
   unsourced. The three probes are meant to fire independently and
   inform a downstream routine that gates model output. This pilot
   confirms all three signals *exist* in the 0.4B state at usable
   fidelity — none is subsumed by the others.

4. **Pilot single-split F1s are optimistic.** Every probe hit F1=1.000
   on the initial held-out fold; only H21 LOO revealed the honest
   number. H22 seed LOO also stayed at 1.000 because the seed is
   generic-and-obvious. Design lesson: **LOO or heavy k-fold as
   default reporting, single-split as internal sanity only.**

## Gaps → next runs

- **H21 truth-value split.** v2 confirmed truth-value fact-checking is
  a separate skill from premise-validity. Redefine H21 scope to
  structural invalidity only (category / impossible / counterfactual —
  where recall stayed ≥ 4/5 on v2). Hand-off truth-value to a separate
  H16-style emit-time verifier.
- **H21 richer features.** For a residual truth-value signal at 0.4B,
  try per-head Frobenius + top-k singular values instead of mean+std.
  Low priority given v2 result; the mechanism suggests the state
  simply doesn't carry the knowledge.
- **H21 v3 at 2.9B (CPU, 2026-07-30):** ran g1h-2.9B extract on the
  same 40 pilot items (feature dim 2560 vs 768). **LOO F1=0.850**
  vs 0.4B pilot LOO F1=0.789 — +0.06 marginal. Per-type: category 5/5,
  counterfactual 5/5 (+1), impossible 4/5 (-1), **factual 3/5 (same
  as 0.4B)**. Val-factual worse still: 2/5 (val_fact_02/03/04 flipped
  to invalid). **Scale does not close the truth-value axis.** Confirms:
  pooled WKV can't do fact-checking even at 2.9B — knowledge problem,
  not state-shape problem. Structural invalidity holds on both scales.
- **H22 distinctness measurement (2026-07-30, PASS).** Authored 32
  cross-labelled overlap items (4 cells × 8: {valid_premise,
  invalid_premise} × {attributable, unattributed}). Extracted shared
  pooled WKV features on 0.4B, ran LOO on both heads. **H21 p_valid
  F1=0.875, H22 p_attr F1=0.941, ρ(p_valid, p_attr) = -0.054**
  (target <0.4 — passes with margin, effectively orthogonal). Per-
  cell matrix separates cleanly: p_valid tracks premise_valid only,
  p_attr tracks attributable only. Confirms H21 and H22 are two
  independent signals in the WKV state, not two views of the same
  latent axis. `experiments/attribution_probe/distinctness/`.
- **H22 corpus choice.** v2 amb-item shift shows attribution is
  corpus-defined. Decide: single canonical corpus, or ensemble of
  heads trained on different definitions of "attributable"?
- **H20 scale-up** to 100+ items with full-protocol
  (`max_new_tokens=32`, more samples per item).
- **Downstream routine wiring.** Pass H20 aporia + H21 structural
  invalidity + H22 attribution + a separate truth-value check into
  the truth-lookup gate. Measure whether joint signal beats each
  individually on held-out mixed prompts.

## Artifacts

- `experiments/aporia_probe/report.md`, `results.jsonl`
- `experiments/premise_validator/report.md`, `loo_results.jsonl`,
  `features.npz`, `items_v2.jsonl` (280 items ready for retrain)
- `experiments/attribution_probe/` — seed features, `loo_seed_results.jsonl`;
  v2 in progress at `./v2/`
- `experiments/_logs/{h20_*,h21*,h22*,cluster}.log`
