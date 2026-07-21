# noesis — instructions for Claude sessions

Read this first. Do not re-litigate decisions marked *locked* without an
explicit signal from the user.

## Locked decisions

- **Name.** noesis (Greek νόησις — apprehension of essence). Framed by the
  user as "cognitive memory of the computer, for other computers".
- **Framing.** noesis is a *Persistent Cognitive Runtime*, not an
  assistant, not a memory system, not a chat interface. Its purpose is
  continuous existence alongside the user, effectively as a peer Linux
  user on the same machine. Memory is a byproduct of continuous
  existence; it is not the primary goal.
- **Backbone.** RWKV-7-G1 line (not base). Starting size 2.9B; may scale
  to 13.3B if hardware / cloud permit.
- **Runtime.** Ollama-served, integrated into a Claude-Code-style CLI via
  Ollama's OpenAI-compatible endpoint. No bespoke harness beyond
  integration verification.
- **Single local reasoning model.** noesis is the sole cognitive engine.
  Small task-specific NNs — embedders, classifiers, and *small decision
  policies* (e.g. a ~50M scheduler that routes tasks) — are permitted
  where they earn their keep. The ban is on additional local *reasoning*
  models specifically; not on any NN that lacks reasoning capacity.
  Heuristic: if it emits tokens that participate in a chain of thought,
  it is a reasoning model.
- **Escalation.** Remote Claude is invoked *by the human*, never by
  noesis. When invoked, Claude reads noesis's background-agent state and
  proceeds from there.
- **Absorbs prior daemons.** `local-search` and `key-daemon` are being
  rebuilt as modules inside noesis, not maintained as standalone
  processes. Their standalone code remains valid source material for the
  architecture.

## Hard constraints

- **Open sources only** for training data. No personal corpus in weights.
  No Compilerium contents in weights. Personal data is allowed as a
  runtime retrieval source, never as a fine-tune signal.
- **Cheap by construction.** Assume GTX 1050 for inference and small
  LoRA experiments. Cloud burst is allowed for continued pretraining but
  must be an explicit decision.
- **Logic-only fine-tune for Phase 1.** Domain knowledge is deferred —
  it enters through the runtime context (retrieval, tool observations),
  not through weights. RFCs, CLI docs, personal corpus are neither
  fine-tune sources nor Phase-1 concerns. General reasoning competence
  in weights + fresh knowledge in the context window is the H7 wager.
- **Memory is a separate track** from model training. Do not conflate.

## Non-goals

- Not a Claude replacement.
- Not coupled to Compilerium (retrieval OK, weights not).
- Not a Transformer. Any switch requires an explicit empirical re-open,
  not architectural drift.
- Not a SaaS product.

## Where to find things

- **README.md** — vision and top-level architecture.
- **ROADMAP.md** — three tracks (Cognitive / Memory / Integration) with
  phased milestones and gates.
- **HYPOTHESES.md** — the falsifiable claims noesis is testing, plus
  the evaluation philosophy. Consult before designing any experiment
  or interpreting a result.
- **docs/principles.md** — the twelve architecture principles (P1..P12).
  When a design choice is in doubt, this file is the tiebreaker before
  the roadmap or the plan.
- **docs/policies.md** — operating policies for noesis as a peer Linux
  user (filesystem, execution, autonomy, credentials, user separation).
  Currently a stub with open questions — default is *ask the user*.
- **FAILED.md** — graveyard of refuted hypotheses and dead experiments.
  Append-only.
- **docs/** — other design notes, research summaries. Consult before
  proposing new architectural directions.
- **training/** — corpora prep, curriculum, LoRA configs, eval sets.
- **memory/** — external memory system.
- **runtime/** — agent loop, Ollama integration, tool adapters.
- **experiments/** — throwaway feasibility probes. Not production code.

## Working conventions

- **Punch-list style** when auditing content: propose changes as a list,
  wait for the user to pick, no stealth-fix.
- **Never train** on anything without confirming the source is open and
  the corpus role (weights vs retrieval) is explicit.
- **Any philosophical claim** about RNN vs Transformer must be paired with
  an empirical test proposal. No architectural mysticism.
- **When adding a memory entry** to the external memory system, record
  *why* — the reason will be needed later to judge relevance.

## Interaction style

- The user has thought deeply about the underlying philosophy: state
  evolution, compression-as-intelligence, Bellard-style approaches,
  Anthropic process supervision. Engage that seriously.
- The user values honest push-back over agreement.
- The user is on a constrained hardware budget and cannot afford to spend
  months on architecture without a feedback signal. Prefer cheap
  experiments that produce measurable data over elegant designs without
  runway.
- Russian for conversation. English for code, commits, documentation.
