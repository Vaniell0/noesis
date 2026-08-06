# FAILED

The graveyard of refuted hypotheses, dead experiments, and abandoned
design directions. This file exists because P10 (Report negative
results) is real, not aspirational — and because the hypotheses in
HYPOTHESES.md must have a place to *go* when they lose.

Each entry records: what the claim was, what evidence refuted it,
what was learned, and what changed in the project as a result.

Entries are append-only. Do not silently edit past entries — if a
past conclusion is reversed by later evidence, add a new entry
pointing to the reversal, do not overwrite.

## Entry template

    ### YYYY-MM-DD — [short title]
    
    **Was.** [The claim, hypothesis, or design bet — quote the
    original if possible, cite HYPOTHESES.md ID or ROADMAP.md phase.]
    
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
- `HYPOTHESES.md` H8 tightened: portability claim split into the two
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
HYPOTHESES.md § H20 report notes and `experiments/aporia_probe/README.md`
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
- `HYPOTHESES.md` § H20 status updated with the scale-up numbers
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
- `HYPOTHESES.md` H2: added "Corpus selection constraint" block —
  reactive corpora cannot test H2; corpus must be reflexive-first.
- `HYPOTHESES.md` H10: status updated to BLOCKED on corpus fix;
  K-sweep data from Steps 4–5 excluded as confounded.
- `docs/verdicts/2026-08-06-a1-pilot-step5.md` written with root cause,
  structural mismatch analysis, and Step 6 corpus options.
- `docs/verdicts/2026-08-04-a1-pilot-step3-step4.md` updated with
  rescored numbers and cross-link to Step 5 verdict.
- `training/corpus/RECLASSIFIED.md`: revised to admit Claude action
  chains as eligible §2 training data (reflexive-first, tool_use loss
  only — the architectural class that was missing from glaive-v2).
