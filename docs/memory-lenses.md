# Memory lenses — cross-model handoff protocol (design draft)

> **Status.** Design draft, 2026-07-22. Not implemented. Not on the
> critical path for Phase 1. Registered here to freeze the framing
> before the A0.6/A0.7 verdicts land — the outcome of those probes
> narrows or opens the design space for lenses.
>
> **Relation.** Refines H5 (structured-summary handoff to remote Claude)
> into a per-zone protocol that also covers *resident* model swaps
> (Ollama child change, RWKV checkpoint bump, cross-architecture
> fallback). Falsifier and prediction are locked as **H11** in
> `HYPOTHESES.md`.

## Problem

Cross-architecture / cross-model handoff has one MVP shape in noesis
today: text-bottleneck (A0.7 tier-3) — dump the reasoning state as a
natural-language brief and re-prompt the target model. It works, but:

- **Expensive.** One channel carries all four zones (insights, vault,
  events, scratch) as prose. Token cost grows with total memory, not
  with what the target actually needs.
- **Lossy uniformly.** Prose compression loses more from *structured*
  content (facts, timestamps, refs) than from *unstructured* content
  (in-flight reasoning). Same prose channel for both under-serves both.
- **Handoff-quality only.** No path to cheaper protocols where the
  content itself is already portable (a DSL insight record is the same
  compact block whether the target model is noesis's own backbone or
  a foreign one primed for DSL parsing).

## Framing shift

The cognitive-runtime frame from `CLAUDE.md` locks noesis as owner of
the memory *and* the model interface. Models are children of the
runtime, not vice versa. The runtime chooses what format the model
sees; Ollama is a transport for token I/O, nothing more. The DSL
(see `docs/dsl.md`) is noesis's own on-the-wire language for
render/command/return between memory and model — it exists precisely
because a project-owned format lets us compose the model context on
the fly without deferring to any third-party format convention.

The question is therefore not "how does model M_A transfer its state
to model M_B?" It is "how does the runtime present each memory zone
to a fresh model M_B, so M_B can pick up work?" And the answer isn't
constrained by third-party formats — it's constrained by which DSL
blocks (and prose fallbacks) M_B can parse. Two cases:

- **Native models** (noesis's own G1/A1 backbone, any Ollama child
  primed on the DSL). Read DSL blocks directly — no translation cost.
- **Foreign models** (remote Claude on H5-style escalation). Runtime
  transforms DSL blocks into a prose paraphrase at the lens boundary.
  Composer already does DSL ↔ transport JSON at the Ollama edge; a
  DSL ↔ prose lens at the escalation edge is a symmetric add.

Different zones have different portability characteristics. The
protocol should have different lenses per zone, not one universal
lens.

## Zone × lens mapping

| zone | content shape | lens (native path) | foreign fallback | notes |
|------|---------------|---------------------|--------------------|-------|
| `insights` (long-term agent memory) | atomic facts + confidence + refs | DSL `insight` blocks retrieved on demand; runtime composes the block into the model's context | DSL → prose paraphrase at the escalation edge | not "transfer" at all; the zone lives in the runtime, the model reads it |
| `vault` (user preferences / model of user) | prefs, style, facts about the user | DSL `vault` blocks + a short DSL-rendered `style_brief` (≤200 tokens) | DSL → prose paraphrase; style_brief is already close to prose | style-brief carries "tone" for M_B; DSL entries carry the ground truth |
| `events` (input-stream observation zone) | timestamped tuples (`ts`, `kind`, `payload`) | DSL `event` tail composed into the prompt suffix (recent window) | DSL → prose timeline | temporal scoping matters — most handoffs only need the last N minutes/tokens |
| `scratch` (working attention, RWKV state) | current reasoning trajectory | **DSL-rendered scratch summary written by the incumbent model M_A** — a small DSL block that names hypotheses, ruled-out branches, current focus | prose form of the same summary | this is the only zone that actually needs a *transfer* protocol; DSL keeps it cheap on native path; prose fallback for foreign M_B |

## Why "will M_B understand?" is the wrong worry

Three of four zones are read-only retrieval, not transfer. The
runtime owns the model interface — it composes what M_B sees, in
whatever format M_B has been primed on. For native models (any
Ollama child fine-tuned or in-context-primed on the DSL) the answer
is trivially yes: DSL is the language they were assembled to read.
For foreign models (H5-style escalation to remote Claude) the lens
adds one prose-paraphrase step at the escalation edge; the composer
already does DSL ↔ transport JSON at the Ollama edge, so a DSL ↔
prose transform is a symmetric add, not a new architectural cost.

The only lens that involves a *state* translation step is `scratch`,
and there M_A writes a DSL block *about* its state — an act it
already does every time it emits a chain of thought (structured
output is well within any competent LM's range). The runtime then
serves that block verbatim on native handoff or paraphrases it for
foreign handoff.

What is **not** portable, and not attempted by this protocol:

- **Bit-perfect state continuity across architectures.** That is
  `A0.7` tier-2 territory (learned projector between state manifolds),
  deferred to Phase 2 in ROADMAP.
- **Sub-token-level reasoning continuity** (e.g. mid-CoT-token
  interruption resumption). Not a real requirement; handoffs happen at
  natural conversation boundaries.

## Falsifier (H11 test design)

**Setup.** Multi-turn task with a mid-task model swap. Two arms:

- **Raw arm.** M_B receives the full raw log of M_A's session
  (context + M_A's outputs). Cold start on M_B otherwise.
- **Lens arm.** M_B receives the zone-lens bundle: DSL blocks from
  `insights`/`vault`/`events` (scoped) — paraphrased to prose iff M_B
  is foreign — plus M_A's DSL-rendered (or paraphrased) scratch-lens.

**Task set.** ≥30 multi-turn tasks from the same held-out pool that
A0.2 uses, extended to require ≥3 turns and a mid-task handoff.

**Primary metric.** Task-success rate on M_B post-handoff.

**Secondary metric.** Input tokens consumed by M_B for the handoff.

**Pass rule.**
- Task success ≥ 0.9 × raw arm (lens loses less than 10 %)
- Token cost ≤ 0.1 × raw arm (lens uses less than 10 % tokens)

If either bound fails, the specific zone-lens combination is the
culprit; per-zone ablation tells us which lens is under-designed.

## Dependency chain

1. Requires: memory zones exist as structured storage → **Phase B**
   (task #2 today).
2. Requires: runtime supervisor that owns the model child → **Phase D**
   of runtime plan.
3. Requires: A1 checkpoint that can act as M_A convincingly on
   handoff tasks → **A1 training complete** (post-Windows-box GPU
   run).
4. Requires: at least one alternative M_B → any Ollama-servable open
   model (Qwen-2.5-3B, Phi-4-mini already in A0.2 baseline).

## Non-goals

- Not a training-time objective. Lenses are runtime constructs; the
  model is not fine-tuned to emit them.
- Not tied to RWKV specifically. Lens protocol is arch-agnostic; the
  scratch-lens is prose regardless of who wrote it.
- Not a substitute for retrieval. `insights`/`vault` are already
  retrieved on-demand — the lens for these zones is "retrieval works,
  it just happens on the new model instead of the old one".

## Open questions

- **Scratch-lens length budget.** Prose summaries above ~500 tokens
  start eating handoff cost. How aggressive is the incumbent model's
  self-summary? Untested.
- **Scratch-lens format.** Structured (bullet list of hypotheses,
  ruled-out branches, current focus) vs free-form prose. Structured is
  probably cheaper and less lossy but constrains the incumbent.
- **Freshness threshold on `events`.** How much of the recent event
  stream is load-bearing for handoff? Probably answered empirically
  during A0.6/A0.7 (event-tail cutoff sensitivity).
- **Interaction with A0.7 tier-1 result.** If tier-1 PASSES (state
  survives LoRA bump), the scratch-lens is only needed for
  cross-architecture handoffs — same-arch handoffs use raw state
  transfer. If tier-1 FAILS, scratch-lens is the general answer
  regardless of arch.

## Not on the critical path

This document exists to freeze the framing. Real testing runs after A1
+ Phase B/D. If A0.6/A0.7 verdicts land unexpected results (e.g.
scratch state turns out to *not* be summarisable coherently — the
model can't reliably describe its own reasoning), the design opens
back up.
