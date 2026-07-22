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
