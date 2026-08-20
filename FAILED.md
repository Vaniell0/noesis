# FAILED

The graveyard of refuted hypotheses, dead experiments, and abandoned
design directions. This file exists because P10 (Report negative
results) is real, not aspirational — and because the hypotheses in
hypotheses/README.md must have a place to *go* when they lose.

Each entry records: what the claim was, what evidence refuted it,
what was learned, and what changed in the project as a result.

Entries are append-only. Do not silently edit past entries — if a
past conclusion is reversed by later evidence, add a new entry
pointing to the reversal, do not overwrite.

## Entry template

    ### YYYY-MM-DD — [short title]
    
    **Was.** [The claim, hypothesis, or design bet — quote the
    original if possible, cite hypotheses/README.md ID or ROADMAP.md phase.]
    
    **Refuted by.** [The evidence — link to the experiment writeup in
    experiments/, numbers, dates. If not a formal experiment, the
    reasoning that made the position untenable.]
    
    **Learned.** [What the failure taught. Often more informative
    than what a success would have taught. Do not skip this section
    even if the lesson feels small.]
    
    **Changed.** [What in the project files or direction was updated
    as a result. Link to commit or diff if applicable.]

## Entries

### 2026-07-22 — WKV state is *not* a semantic override switch (A0.6)

**Was.** The strong reading of H8 / H_portability: WKV state carries
portable semantic content that continues to steer decoding after a
prompt swap. The implicit runtime bet was that noesis could inject
state as a full context substitute — hand off a "mental image" between
sessions and let it drive continuation.

**Refuted by.** `experiments/A0_portability/results_a06.md`, run
2026-07-22 on world-0.4b and g1d-0.4b. 60 core cells per model across
3 prompt pairs × 2 directions × 3 depths × 2 modes. Verdict rule:
alignment ≤ −0.30, Δhit_donor > +0.05, coherence_flag = 1 in ≥ 2 of
3 pairs.

- Both models **FAIL** the model-level quorum (0 of 3 pairs).
- Only 1 PASS cell per model, both in the same corner:
  `code_prose × AB × after_B × full` (donor state = last thing model
  saw, full-state swap). World alignment −0.946 / Δhit +0.259; G1d
  alignment −0.327 / Δhit +0.444.
- Hotspot mode (A0.5 load-bearing layer subset) FAILs 11 of 12 core
  cells across both models — the layers that carry ablation loss are
  not the same layers that carry *content* under transplant.

**Learned.** State portability splits into two claims that A0.5 had
conflated:

1. *State carries content.* Confirmed. Δhit_donor is consistently
   positive in before_B / mid_B cells across both models — the donor's
   task lexicon leaks into the recipient continuation.
2. *State dominates the continuation trajectory.* Refuted. Alignment
   rarely clears −0.30 outside the maximally-loaded cell. The prompt's
   structural pull wins.

Reframing: state is a **compressed context bias**, not a semantic
override. Useful as a warm-up / conditioning source, not a full
context replacement. Direction asymmetry (AB > BA) suggests
better-structured donors (code < prose) transfer more, but not enough
to change the verdict.

**Changed.**
- `hypotheses/README.md` H8 tightened: portability claim split into the two
  sub-claims above; only sub-claim 1 survives.
- Runtime plan (`docs/effort-frontier.md`): H10 `state_readout` mode
  gains weight — decode state content to text and re-inject via
  prompt_cot scaffold, rather than expecting state to fight the prompt.
- H11 zone-typed lens (structured text handoff) confirmed as the
  primary cross-context protocol, not one option of many.

### 2026-07-22 — WKV state does not survive checkpoint swap (A0.7 tier-1)

**Was.** The tier-1 bet: WKV state format is close-enough across two
same-architecture, same-size checkpoints that raw `load_wkv_into_state`
carries a majority of the intra-model portability signal (README PASS
rule: > 50 % of A0.6 same-checkpoint baseline). Sub-hypothesis
2026-07-22 (user): asymmetry — world-0.4b "stores noise", g1d-0.4b
structures state, so g1d → world should transfer better than world → g1d.

**Refuted by.** `experiments/A0_portability/results_a07.md`, same day.
60 cells across World-0.4B ↔ G1d-0.4B, both checkpoint directions.

- **0 of 6 (donor→recipient, pair) groups PASS.** Both directions
  FAIL symmetrically.
- Best alignment cell world→g1d: `code_prose / BA / after_B / full`,
  alignment = −0.231 (short of −0.30).
- Best alignment cell g1d→world: `math_narr / BA / after_B / full`,
  alignment = −0.135. Also short.
- No asymmetry — if either model's WKV had been more "readable" for
  the other, at least one direction should have produced a clear PASS.
  Neither did.

**Learned.**
- WKV state is **checkpoint-private**, not model-family-generic. The
  representation is entangled with the exact weight values, not just
  the architecture.
- The A0.6 finding (state carries content, doesn't dominate trajectory)
  survives the checkpoint swap in weaker form: g1d→world `code_prose /
  AB / after_B / full` shows alignment −0.044 but Δhit_donor +0.333 —
  the donor's lexicon still bleeds in, just even more thinly.
- The `n/a` ratio pairs (baseline A0.6 had 0 PASSes) mean the runtime
  rule "does cross-checkpoint transfer preserve intra-model quality?"
  degenerates when intra-model quality is already 0 — a design gap in
  the verdict rule, harmless here because the actual signal is
  unambiguous FAIL.

**Changed.**
- Tier-2 (learned WKV projector between checkpoints) is now the
  *only* cross-checkpoint substrate option, no longer one of several.
  Still Phase-2 deferred per README non-goals.
- H11 zone-typed lens (structured text handoff) becomes load-bearing
  for any model-version migration, not optional.
- Runtime design constraint (`docs/effort-frontier.md`, runtime CLAUDE.md):
  no protocol may assume WKV state survives a checkpoint swap. Sessions
  are pinned to a single frozen checkpoint or migrate through text.
- 1.5B pair follow-up (World-1.5B ↔ G1H-1.5B) skipped — A0.7 tier-1
  landed a straight FAIL, not CAVEAT, so extra evidence wouldn't shift
  the verdict.

### 2026-07-30 — H20 pilot ordering `cf > ba > ui` did not hold at scale

**Was.** The 2026-07-29 aporia probe pilot report predicted (per
hypotheses/README.md § H20 report notes and `experiments/aporia_probe/README.md`
category shapes): contested_facts should collapse more than
bounded_ambiguity, which in turn collapses more than
underdetermined_inference — a monotone `cf > ba > ui` ordering of
`collapse_cont`. The saas.md draft §2 (pre-2026-07-30) stated this as
"pilot confirmed" and used it as evidence for training the truth-system
detector cluster on this substrate.

**Refuted by.** `experiments/aporia_probe/report.md`, scaled to 100
items (35 cf / 35 ba / 30 ui) × 10 samples, 3 shards, wall ≈18274 s on
G1d-0.4B, run 2026-07-30. Aggregate `collapse_cont`=0.541,
`p(neither)`=0.746. Per-category `collapse_cont`: cf=0.589, ba=0.478,
ui=0.560. Ordering is **cf > ui ≈ ba** — `ba` and `ui` swap places,
`ba` becomes the *lowest*-collapse category, not the middle one.
`p(neither)` shows the same reversal: ba=0.854 > cf=0.760 > ui=0.603.

**Learned.**
- Pilot-sample-size claims about aporia category structure are
  fragile. n=30 was not enough to distinguish `ba` from `ui` at this
  substrate; the pilot's tight `ui` sample happened to land on lower
  branching than the v100 spread.
- `bounded_ambiguity` remains the strongest "keep open" signal by
  `p(neither)` — the model hedges most on semantic ambiguity rather
  than collapsing. But hedging shows up in `p(neither)`, not
  `collapse_cont`, which is why the ordering flipped between metrics.
- `underdetermined_inference` at v100 mixes item subtypes
  (missing-referent, chain-of-inference, etc.) with heterogeneous
  branching. Next iteration should split `ui` by inference-length
  subcategory before comparing.
- H20 as a substrate mechanism is not refuted — the collapse rates
  are middle-of-range, `p(neither)`=0.746 is high, and the aporia
  signal is present. What is refuted is the specific pilot ordering
  narrative that would have shaped training data selection.

**Changed.**
- `hypotheses/README.md` § H20 status updated with the scale-up numbers
  (2026-07-30) and the ordering swap. Kept as an "aporia lives in
  continuation branching, needs decode" finding — pooled-WKV feature
  claim from the earlier pilot survives, category-ordering claim does
  not.
- `noesis-saas.md` §2 H20 line rewritten: "pilot confirmed cf > ba > ui"
  replaced with a retracted-monotone-collapse note and a pointer to
  ROADMAP A4 (targeted corpora for the H19/H20/H21/H22 detector cluster
  as the *product* path — the pilot ordering was never load-bearing for
  A1 corpus selection because H20 sits behind A4, not A1).
- No file-graph changes required for A1 (Variant C hybrid) — the H20
  ordering was informational, not a A1 selection input.

### 2026-07-30 — Variant A A1-corpus scope (reasoning-first, personal-primary) superseded

**Was.** `docs/policies.md § A1 fine-tune corpus scope` locked 2026-07-22
as Variant A: A1 supervision was to be "reasoning-first" — user's local
Claude Code traces named as the primary A1 signal in
`docs/training-data-shortlist.md § 1`, with public function-calling
corpora as supplementary. Design bet: rich reasoning content in
personal traces would teach the RWKV-7 checkpoint the noesis cognitive
runtime better than any open corpus, and privacy was tractable via the
sanitisation pipeline (`sanitize.py`, `audit_sample.py`).

**Refuted by.** Not an experiment — a hard conflict-audit on
2026-07-30 between three canon files:

- `CLAUDE.md` hard constraint: *"no personal corpus in weights"*.
- `docs/policies.md § A1 fine-tune corpus scope` (Variant A):
  *"personal traces are the primary A1 signal"*.
- `docs/training-data-shortlist.md § 1` (pre-2026-07-30):
  *"Local Claude Code traces — primary"*.

The three could not simultaneously be true. Variant A's operational
detail contradicted CLAUDE.md's hard rule. Under the rule
"CLAUDE.md hard constraints override design bets", Variant A had to
go. Independent secondary issues surfaced during the audit:
- Character contamination: Anthropic-style traces would push the
  RWKV-7 output toward "stuttering Claude" rather than a
  distinct noesis voice.
- Legal risk: personal Claude CLI logs are Anthropic ToS-encumbered
  even for local use, so downstream distribution of *any* weight
  trained on them is off the table.
- Verifiability: private data cannot be independently audited, so
  reproducibility of A1 verdicts is impossible without the private
  corpus.

Even ignoring the CLAUDE.md rule, the secondary issues alone would
have collapsed Variant A within one review cycle.

**Learned.**
- A hard constraint in CLAUDE.md silently ignored by a downstream
  design document produces a canon-conflict that only surfaces at
  implementation time. Structural fix (2026-07-30): the reconciled
  `docs/training-data-shortlist.md` opens with a "reconciled with
  policies.md" pointer so future readers cannot pick one file
  without seeing the other.
- The "reasoning-first thesis" (H2) is a *runtime* claim about how
  noesis composes computation from stored representations, not a
  *training-data* claim about which content goes into weights. The
  two got conflated for a week. Fix: H2 stays as a runtime claim
  (reasoning as state-work, per H8/H4b); training-data selection is
  now driven independently by legal + character + verifiability
  filters.
- "Sanitisation can rescue any corpus" is optimistic. Even a
  well-scrubbed personal corpus retains stylistic fingerprints
  (character contamination) that regex passes cannot remove.
  Source-selection > sanitisation. Sanitisation is a safety belt,
  not a substitute.

**Revision note (2026-08-06).** The Variant A exclusion applied to
Claude's *reasoning text* as training targets. Step 5 analysis
established that `extract_traces.py` action chains have a different
structure: `tool_use` decisions as loss targets, personal user queries
as context only. Character contamination cannot occur when loss is
restricted to `tool_use` tokens (no Claude text in the gradient). The
revised position (see `training/corpus/RECLASSIFIED.md §Revision
2026-08-06`) admits action chains under the §2 step-and-tool category.
This does not reopen Variant A — it sharpens what "personal corpus"
meant and excludes the action-chain case from that scope.

**Changed.**
- `docs/policies.md § A1 fine-tune corpus scope` rewritten to
  Variant C hybrid (2026-07-30): action-cloning corpora primary,
  adaptable open reasoning traces secondary (restructured into
  step-and-tool linked form only), personal data excluded from
  weights entirely, Anthropic-derived reasoning distills excluded.
- `docs/training-data-shortlist.md` restructured: personal traces
  demoted from §1 primary to explicit exclusion; public agent
  corpora (Salesforce/xlam-function-calling-60k,
  glaive-ai/glaive-function-calling-v2, thunlp/ToolBench,
  THUDM/AgentInstruct) promoted to §1 primary. §2 "adaptable open
  reasoning traces" added as the escape hatch for open reasoning
  material that can be restructured into tool-linked steps.
  OpenThoughts-114k, Bespoke-Stratos-17k, NuminaMath-CoT
  reclassified from rejected to §2 candidates (per-dataset
  decision at corpus-prep time).
- `training/README.md` corpus-policy section rewritten with
  Variant C primary/secondary/excluded categories.
- `training/corpus/RECLASSIFIED.md` and
  `training/sanitised/RECLASSIFIED.md` retained as-is — their
  2026-07-22 pivot note ("personal traces reclassified to
  retrieval-only") remains valid under Variant C. The 2026-07-22
  reclassification was Variant A's *first-order fix* toward the
  CLAUDE.md constraint; Variant C is the *second-order fix* that
  fills the hole left by dropping personal traces from weights.
- `ROADMAP.md` Cloud training budget split into micro-pilot
  ($5-10 on 4090 spot, Variant C corpus falsifier) vs full-scale
  A1 campaign ($40-50 on A100). H2/H7 corpus-shape now points to
  Variant C hybrid, not Variant A "reasoning traces only".
- Loss target unchanged: standard next-token loss on `tool_use`
  tokens only. `tool_result` tokens stay context; thinking tokens
  stay excluded from the loss mask. Behavior-cloning on *what to
  do next*, not *how to sound while thinking*. This is what makes
  the character-contamination avoidance mechanical rather than
  hopeful.

### 2026-08-06 — glaive-v2 assumed to train direct-answer reasoning

**Was.** A1 pilot Steps 1–5 used glaive-function-calling-v2 as the
primary A1 corpus (Variant C §Primary, "action-cloning corpora"). The
working assumption was: (a) glaive-v2 provides tool-use + reasoning
training data, (b) at ≥ 80% epoch the model would generalise to
direct-answer tasks as well as tool-dispatch tasks, (c) declining eval
scores were attributable to "format bleeding" that would wash out with
more training.

**Refuted by.** `docs/verdicts/2026-08-06-a1-pilot-step5.md` and the
corpus analysis run on 2026-08-06 (63,218 rollouts inspected):

- 100% of rollouts have `assistant: tool_use` as the first turn.
  The eval format (`User: …\nAssistant:`) asks for a first-turn direct
  answer. The model's first-turn tool_call is *correct behaviour per
  its training*, not format bleeding.
- 25,451 rollouts (40%) have a first-turn `content` field — but every
  sampled instance was a pre-tool-call phrase ("Of course, let me
  calculate that for you"), not a reasoned computation.
- glaive-v2 is a pure *reactive* tool-dispatch corpus: 1–2 tool_use
  calls per session, no causal chaining, no multi-step inference.
  It contains zero direct-reasoning examples regardless of epoch count.
- G1h 2.9B base (no SFT) scored 5/48 = 10.4% on A0 eval.
  G1d 0.4B SFT'd on glaive-v2 at Step 4 scored 4/48 = 8.3%;
  at Step 5 (full epoch) scored 3/48 = 6.2%. SFT on glaive-v2
  actively hurts direct-answer capability.

**Learned.**
1. *Reactive corpora cannot be used to test reasoning-first thesis.*
   glaive-v2 trains lookup reflexes, not reasoning state. All Step
   4/5 eval data is confounded and cannot contribute evidence to H2
   or H10.
2. *Format bleeding is the wrong mental model for this failure.* The
   first-turn tool_use is the model's correctly learned behaviour, not
   residual SFT formatting. More epochs deepen the problem, not fix it.
3. *Corpus-architecture fit is a first-class constraint, not a
   parameter.* RWKV's WKV state accumulation advantage is only
   exercised by multi-step reflexive sequences (10+ tool_use calls,
   causal dependencies across many steps). Purely reactive corpora
   (1–2 calls per session) do not engage this mechanism. Training on
   them is architecturally neutral at best, actively harmful at worst
   (overspecialisation to reactive behaviour). Corpus class must be
   chosen before architecture, not after.

**Changed.**
- `hypotheses/README.md` H2: added "Corpus selection constraint" block —
  reactive corpora cannot test H2; corpus must be reflexive-first.
- `hypotheses/README.md` H10: status updated to BLOCKED on corpus fix;
  K-sweep data from Steps 4–5 excluded as confounded.
- `docs/verdicts/2026-08-06-a1-pilot-step5.md` written with root cause,
  structural mismatch analysis, and Step 6 corpus options.
- `docs/verdicts/2026-08-04-a1-pilot-step3-step4.md` updated with
  rescored numbers and cross-link to Step 5 verdict.
- `training/corpus/RECLASSIFIED.md`: revised to admit Claude action
  chains as eligible §2 training data (reflexive-first, tool_use loss
  only — the architectural class that was missing from glaive-v2).

### 2026-08-07 — H10 state_readout axis carries no signal over prompt_cot (step8 epoch0)

**Was.** H10 readout-mode prediction: `state_readout` would score ≥ 0.02 pp
above `prompt_cot` at the same (N, K), because reading the WKV state into
text and re-injecting it yields information beyond what the same tokens
injected via the prompt carry. Rationale: the state representation is
richer than any token-level readout, so decoding it should add signal.

**Refuted by.** H10 sweep 2026-08-07 on G1h-2.9B step8-epoch0 checkpoint
(`experiments/A0.8_refine/results/step8_epoch0/SUMMARY.md`). 20-cell matrix,
N ∈ {1,2,3} × K ∈ {32,128,512} × mode ∈ {silent, prompt_cot, state_readout}.

- `state_readout` == `prompt_cot` at **every single cell** (not just on
  aggregate, but numerically identical across all 6 categories).
- The tie is exact — 0 pp difference on any (N, K, category) pair.
- K=512 cells excluded (file too large to eval), but extraction=0% confirmed
  for those as well.

**Learned.**
- At epoch 0, the model produces the same token distribution from WKV state
  readout as from continuing the same prompt prefix. The state is not yet
  specialised enough to contain information the prompt doesn't have.
- The readout axis is degenerate until A1 training explicitly teaches the
  model to load state-differentiated content into readout tokens — this is a
  *training target problem*, not a mechanism problem.
- H10 readout prediction requires dedicated state-readout supervision corpus
  (examples where the state holds information the prompt cannot reconstruct)
  before the axis becomes measurable. Swept too early.

**Changed.**
- `hypotheses/README.md` §H10: readout-mode prediction marked FALSIFIED at epoch 0.
  Axis dropped from active sweep dimensions. Will be revisited only after A1
  training includes state-readout targets.
- `experiments/A0.8_refine/results/step8_epoch0/SUMMARY.md` §Key Finding 3:
  state_readout ≡ prompt_cot documented with the "axis degenerate" conclusion.

---

### 2026-08-07 — N=3 silent re-feed corrupts rather than refines WKV state (step8)

**Was.** H10 refinement axis: additional silent re-feeds (N>2) continue to
refine the WKV state, accumulating useful signal about the prompt. At minimum,
N=3 should be neutral (≥ N=2 accuracy); the expectation was monotonic
improvement through refinement passes.

**Refuted by.** Same H10 sweep (step8 epoch0, 2026-08-07):

| N | silent   | best K=128 CoT |
|---|----------|----------------|
| 1 | 27.1%    | 16.7%          |
| 2 | **33.3%**| 22.9%          |
| 3 | 6.3%     | 6.3%           |

N=3 silent collapses 27 pp from N=2 — a catastrophic regression, not
diminishing returns. All N=3 CoT cells are also at or below 6.3%.

Proposed mechanism: glaive-v2 DSL training means a silent re-feed triggers an
implicit `<tool_call>` loop in the state. By pass 3 the accumulated
tool-dispatch activation pattern overwrites whatever reasoning signal passes
1-2 built. With no stopping criterion, the third pass turns the state into
DSL noise.

**Learned.**
- The usable refinement axis is N ∈ {1, 2} only at DSL-trained checkpoints.
- "Silent re-feed always helps" is wrong for checkpoints where silent input
  activates a looping pattern (tool_call, role-play turn, etc.). The corpus
  determines the safe N range.
- N=3 is not just uninformative — it actively reverses the N=2 gain. This
  asymmetry is a useful upper bound for test-time compute: never exceed N=2
  unless the checkpoint was specifically trained on N>2 stable re-feed.

**Changed.**
- `hypotheses/README.md` §H10: N=3 result documented; refinement axis bounded to
  N ∈ {1,2} for DSL checkpoints. Mechanism noted as unconfirmed (would need
  N=3 on a non-DSL checkpoint to isolate DSL-loop from general WKV saturation).
- `experiments/A0.8_refine/results/step8_epoch0/SUMMARY.md` §Key Finding 1-2:
  N=3 collapse and N=2 sweet-spot documented.
- Step 9 design: N/K/mode curriculum targets self-selected compute budget
  (H16 gate), deferred to step 10 once the DSL saturation is cleared.

---

### 2026-08-07 — Action-chain training (step7) does not improve multi-slot state retention

**Was.** H12b behavioral baseline probe (quick run 2026-08-07, P=1 only):
G1h-2.9B step7-action at K=8,P=1 scored 51% vs base 53% — interpreted as
"NO CONTAMINATION" and "action-chain training appears to have improved
multi-track state stability independently of the LoRA intervention H12b
predicts" (hypotheses/README.md §H12b, same-day entry).

**Refuted by.** Full-depth probe 2026-08-07, 420 cells: K∈{2,4,8},
P∈{1,2,4}, n=10 per cell, G1d-0.4B + G1h-2.9B base + G1h-2.9B step7.
Results (`experiments/A0_H12b_multislot/results/report_2026-08-07.md`):

| Model          | K=2,P=1 | K=4,P=1 | K=8,P=1 | K=8,P=2 | K=8,P=4 |
|----------------|---------|---------|---------|---------|---------|
| G1h 2.9B base  | 65%     | 67%     | 53%     | 21%     | 5%      |
| G1h 2.9B step7 | 25%     | 45%     | 51%     | 11%     | 8%      |

Step7 is worse on 7 of 9 cells. The "NO CONTAMINATION" result was a P=1
artefact: at P=1 shallow retrieval, step7 ≈ base at K=8; at P=2+ depth,
step7 degrades faster. G1d-0.4B aggregate: K=2: 18%, K=4: 15%, K=8: 7%.

**Learned.**
1. *P=1 probes are insufficient for multi-slot claims.* A single retrieval
   pass only tests whether a slot can be addressed at all; multi-pass (P=2+)
   tests whether retention holds across sequential access. Quick-runs with
   P=1 will always under-report degradation at depth.
2. *Architecture (G1h vs G1d) is the dominant factor, not fine-tuning.*
   G1h 2.9B base at K=8,P=1 = 53%; G1d 0.4B = ~11%. The 2.9B step7 model
   under-performs its own base at K=2 and K=4, meaning action-chain SFT
   actively traded off multi-slot depth retention for shallow-retrieval
   gains that match the one-fact-per-turn structure of its training data.
3. *Step7 specialised toward one-shot slot access.* The corpus taught the
   model "retrieve once from recent context" — exactly the action-chain
   pattern. Multi-round slot cycling (P=2+) is not represented in
   action chains, so it regressed. This is a corpus-architecture fit
   failure, not a weight-capacity failure.
4. *H12b architectural treatment is still unrun and unrefuted.* This
   entry covers only the behavioral baseline (vanilla models, no LoRA
   expansion). The claim "LoRA-expanded multi-slot state with utilisation
   regularizer improves multi-slot retention" is not addressed by this
   probe. H12b + H12b.i remains a live Phase 2 hypothesis.

**Changed.**
- `hypotheses/README.md` §H12b: "NO CONTAMINATION" qualifier added: "P=1 only, K=8";
  the "action-chain training improved stability" sentence struck and replaced
  with the full 9-cell table and "step7 < base on 7/9 cells" verdict.
- H12b status updated: behavioral baseline = MIXED (architecture helps,
  SFT hurt multi-slot depth); architectural treatment pending.
- H12b.i (utilisation regularizer): mandatory confirmed — training did not
  spontaneously produce balanced slot utilisation even at 2.9B.

---

### 2026-08-12 — Reversal: the 2026-08-07 "state_readout axis carries no signal" entry was itself an eval-bug artifact

**Was.** The 2026-08-07 entry above ("H10 state_readout axis carries no signal
over prompt_cot") concluded that `state_readout` == `prompt_cot` at every
(N, K) cell because the WKV state was "not yet specialised enough to contain
information the prompt doesn't have" — i.e. a genuine finding about the
step8-epoch0 checkpoint, with the readout axis marked FALSIFIED at that
checkpoint pending dedicated readout-supervision training.

**Refuted by.** `eval.py` bug found 2026-08-12 (credit: marty1885), fixed
same day: `state_readout` and `prompt_cot` modes shared the same
greedy-decoding code path, so the two modes never actually diverged at
generation time regardless of what the readout mechanism did internally.
The exact tie across all 20 cells was not evidence that the state carries no
extra signal — it was evidence that the eval harness never exercised the
`state_readout` code path differently from `prompt_cot` in the first place.
See `hypotheses/README.md` §H10, step8 sweep, Key Finding 3.

**Learned.**
- An exact numerical tie across every cell of a sweep should have been a red
  flag *at the time* — real experimental results are noisy; a perfect tie is
  a much stronger prior for "these two conditions are secretly the same code
  path" than for "the state genuinely carries zero marginal signal." This
  entry itself did not catch that signal when it was written 2026-08-07.
- The step8-epoch0 checkpoint's `state_readout` capability is **untested**,
  not refuted. The 2026-08-07 entry's causal story (state not specialised
  enough) is unsupported — it may still turn out to be true, but the
  2026-08-07 experiment provides no evidence for or against it.
- Per repo convention, the original entry is left in place uncorrected (see
  `## Entry template` — append-only, do not overwrite); this entry is the
  pointer that supersedes it.

**Changed.**
- `hypotheses/README.md` §H10 step8 sweep: Key Finding 3 rewritten to state the bug
  explicitly and mark all `state_readout` data in that table invalid.
- H10 rerun with the fixed evaluator launched on G1i 2.9B base
  (`experiments/A0_eval/results/h10_state_readout_g1i_base.json`,
  2026-08-16/17) — first valid `state_readout` data point once it completes.
