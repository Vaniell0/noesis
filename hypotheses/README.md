# Hypotheses

This directory is the intellectual audit trail of noesis. Every serious design
decision either tests one of these claims or takes one for granted — be
explicit about which.

Not a wishlist. Every claim is falsifiable, and the criterion for rejection
is spelled out. If a claim cannot be stated in a form that could be shown
wrong, it does not belong here.

**2026-08-20 — this directory replaces the old monolithic `HYPOTHESES.md`.**
One file per hypothesis (`H<N>.md`), plus `H0.md` for the overarching
framing. Old content that hasn't been re-reviewed under this system this
week is not treated as settled prior art — see each file's own frontmatter
(`status`, `prior`/`posterior`, `evidence`) for what's actually been
checked, not the prose alone. The pre-split version lives at git tag
`hypotheses-v1-prose-era`.

## Evaluation philosophy

**What counts as evidence.**
- Numbers on the user's real held-out eval set (from A0.2 in ROADMAP.md),
  not on standard benchmarks. Benchmarks are references, not verdicts.
- Sustained-operation metrics (RAM, CPU, wall-clock, quality decay over
  N days) for anything claiming background viability.
- Blind comparison where possible: same task, multiple backbones,
  LLM-as-judge scoring, spot-checked by the user.

**What does not count.**
- Improvement on a benchmark that was in the training corpus.
- One-shot demos on cherry-picked prompts.
- Philosophical elegance.
- Alignment with prior claims made in this file.

**Sources of bias to name and mitigate.**
- *Confirmation bias.* It is emotionally expensive to reject H4 after
  months of work. Pre-commit the refutation criterion before running the
  experiment, not after.
- *Goodhart's Law.* Any single metric will get gamed. Use at least three
  disjoint metrics per hypothesis where the task allows.
- *Sunk cost.* If Gate 1 refutes the RWKV wager on target tasks, honour
  the pre-commitment to re-open the backbone decision. Do not rescue
  with post-hoc reframing.

**What NOT to optimise.**
- Do not optimise for benchmark scores that were not agreed on in
  advance. If you find a metric that noesis happens to win, log it as
  interesting; do not promote it to primary.

**Reporting cadence.**
- Every gate produces a short honest write-up: what was tested, what the
  numbers say, what the interpretation is, what is left unresolved.
  Failure to report a negative result is worse than the negative result
  itself.

**Companion files.**
- `FAILED.md` — refuted hypotheses and abandoned bets, with evidence
  and what changed as a result.
- `PASSED.md` — verified claims and settled sub-questions, so the
  audit trail has a positive side, not only a graveyard.

---

## Navigation

Grouped for orientation — inherited from the old file, not itself
re-verified. Full text is in each `H<N>.md`.

**Foundational.**
- **H0** — Overarching framing (world-model register). Currently a
  stub pending rewrite; see `H0.md`.

**Retracted to operating policy** (was a hypothesis, is now a policy
decision in `docs/policies.md` — the numeric slot is preserved).
- **H1** — Constant-cost background operation. *Retracted 2026-07-25;
  body moved to `docs/policies.md` § CPU / thermal.*

**Reasoning core — architecture and training (Track A).**
- **H2** — Reasoning-first outperforms knowledge-first at this scale.
- **H4a** — RWKV-7-G1 2.9B reaches parity with same-size Transformer.
- **H4b** — State-evolution architectures are viable for reasoning *(wager)*.
- **H7** — Understanding in weights, knowledge in context.
- **H8** — State-as-computation in RWKV-7.
- **H9** — G1-line training amplifies state utilisation.
- **H10** — Test-time compute frontier — state × tokens × readout.
- **H12** — Working-memory bottleneck vs decay-rate bottleneck in WKV.

**Memory system + cross-model handoff.**
- **H3** — Learned memory policy trumps heuristic memory at small scale.
- **H5** — Inter-model state transfer via compact structured summary.
- **H11** — Zone-typed lenses beat monolithic text-bottleneck handoff.

**State + context management (runtime state-work workstream).**
- **H17** — State-substrate absorption substitutes for message-history
  re-injection.
- **H18** — Git-like branch/merge WKV for indefinite structured
  continuation *(new 2026-07-25)*.

**Multimodal substrate.**
- **H13a** — State compresses geometry, not just token distributions *(wager)*.
- **H13b** — Image-in-context beats text-digest for screen-content tasks.
- *`note-multimodal-architecture.md`* — locked architectural decision, not
  a numbered hypothesis.

**Runtime as peer (behaviour, persona, self-initiated speech).**
- **H6** — Cognitive layer on modest hardware.
- **H14** — Domain competence via targeted Phase-2 SFT, not Phase-1
  weights.
- **H15** — Persona-SFT to a dry butler/secretary register beats default
  helpful-assistant tone.
- **H16** — Gated externalisation from a rate-limited silent
  think-stream.

**Truth system / epistemic behaviour** *(added 2026-07-29; H19–H22 make
honesty failures falsifiable in their own right; H2/H7/H8/H10/H16 all
touch pieces implicitly).*
- **H19** — Weight-knowledge contamination detector (empirical arm of
  H7).
- **H20** — State holds contradictory belief pairs without premature
  collapse.
- **H21** — Premise-validity readout — model refuses invalid premises
  before answering.
- **H22** — Unattributed collective claims ("usually X", "most believe Y")
  are a detectable, distinct honesty failure.

**State-work — slot addressability and RL substrate.**
- **H23** — LoRA-added WKV rank is structured-addressable (write-here /
  read-here), not just entropy-spread. *(wager)*
- **H24** — WKV-loop GRPO raises decoding efficiency (DE) without
  accuracy regression. *(wager)*
- **H25** — WKV state is a learnable computational substrate for
  approximate linear algebra without CoT tokens. *(wager, Phase 3)*

<!-- AUTO-GENERATED BELOW: do not hand-edit — regenerate via `python experiments/regenerate_hyp_index.py` -->

| H | Status | Prior | Posterior | File |
|---|--------|-------|-----------|------|
| H0 | CONTESTED | 0.6 | 0.333 | `H0.md` |
| H1 | no structured record | - | - | `H1.md` |
| H2 | no structured record | - | - | `H2.md` |
| H3 | no structured record | - | - | `H3.md` |
| H4a | no structured record | - | - | `H4a.md` |
| H4b | no structured record | - | - | `H4b.md` |
| H5 | no structured record | - | - | `H5.md` |
| H6 | no structured record | - | - | `H6.md` |
| H7 | no structured record | - | - | `H7.md` |
| H8 | SUPPORTED | 0.6 | 0.588 | `H8.md` |
| H9 | SUPPORTED | 0.65 | 0.650 | `H9.md` |
| H10 | PARTIAL | 0.5 | 0.500 | `H10.md` |
| H11 | no structured record | - | - | `H11.md` |
| H12 | no structured record | - | - | `H12.md` |
| H13a | no structured record | - | - | `H13a.md` |
| H13b | no structured record | - | - | `H13b.md` |
| H14 | no structured record | - | - | `H14.md` |
| H15 | no structured record | - | - | `H15.md` |
| H16 | RETRACTED | 0.2 | 0.200 | `H16.md` |
| H17 | no structured record | - | - | `H17.md` |
| H18 | no structured record | - | - | `H18.md` |
| H19 | no structured record | - | - | `H19.md` |
| H20 | no structured record | - | - | `H20.md` |
| H21 | no structured record | - | - | `H21.md` |
| H22 | no structured record | - | - | `H22.md` |
| H23 | no structured record | - | - | `H23.md` |
| H24 | no structured record | - | - | `H24.md` |
| H25 | no structured record | - | - | `H25.md` |
