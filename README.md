# noesis

*Persistent Cognitive Runtime — cognitive memory of the computer, for other computers.*

noesis is **not an assistant that answers questions**. It is a **runtime**:
a continuous cognitive process that lives on the same Linux system as the
user, effectively as another user on the machine. Its job is not to
remember. Its job is to *live*. Memory is a consequence of continuous
existence, not the goal.

Practically: noesis runs as a persistent local process behind Ollama and
is usable through a Claude-Code-style CLI. Heavy tasks are delegated to
remote Claude only when the human explicitly asks for it — noesis does
not route to Claude on its own.

The "peer Linux user" framing raises operational policy questions
(filesystem scope, execution boundaries, credential handling,
autonomy vs. ask). Those are open — see `docs/policies.md`.

## Vision

noesis absorbs and replaces two prior standalone daemons, `local-search` and
`key-daemon`, by pulling their essential functions into the ecosystem of one
small constant-cost RNN-based model. The wager: quality of reasoning and
quality of memory policy matter more than raw parameter count, and O(1)
per-token inference makes constant background operation economically
feasible on modest hardware.

Two long-term research threads live inside noesis:

1. **Reasoning as evolution of internal state.** Whether an RNN-family model
   (RWKV-7-G1), trained heavily on logic and reasoning traces, can serve as
   a persistent background reasoner without paying the quadratic attention
   cost of Transformers.

2. **Inter-model state transfer protocol.** How noesis and other models
   (remote Claude, other agents) exchange compact decodable representations
   of task state so that continuity survives the model handoff — the
   "cognitive memory of the computer, for other computers".

## Architecture at a glance

- **Backbone.** RWKV-7-G1, starting size 2.9B. May scale to 13.3B later if
  hardware and cloud budget permit.
- **Runtime.** Ollama-served. Integration into a Claude-Code-style CLI via
  Ollama's OpenAI-compatible endpoint — to be verified in phase C0.
- **Memory.** External, layered, developed on a *separate* track from the
  model. Not baked into weights.
- **Escalation.** Human decides when to call remote Claude. Once invoked,
  Claude reads noesis's background-agent state and continues from there.
  noesis does not decide on its own to call Claude.
- **Prior daemons.** `local-search` and `key-daemon` are being rebuilt as
  modules inside noesis, not maintained as separate processes. This cuts
  the standing footprint that made them "too demanding".

## Hard constraints

- **Cheap.** Assume GTX 1050 for inference and small LoRA experiments.
  Cloud burst is allowed for continued pretraining but must be budgeted,
  not defaulted to.
- **Open sources only** for training data. No personal corpus in weights.
  Personal data may be a *runtime retrieval* source, never a fine-tune
  signal.
- **Single local reasoning model.** noesis is the sole cognitive engine.
  Small utility NNs (embedders, classifiers, etc.) are permitted where
  they earn their keep — the ban is on additional local *reasoning*
  models, not on any NN.
- **Autonomous.** Standard Ollama + CLI workflow. No bespoke harness.

## Non-goals (explicit)

- **Not a Claude replacement.** Heavy reasoning still goes to remote Claude
  by user's explicit call.
- **Not coupled to Compilerium.** Training corpus is isolated. Compilerium
  may be a retrieval source at runtime, never a fine-tune signal.
- **Not a Transformer.** RWKV is chosen deliberately for constant-cost
  streaming inference. Any switch away from RWKV requires an explicit
  empirical re-open, not architectural drift.
- **Not a SaaS.** noesis is a personal daily bot, not a product.

## Repository layout

```
noesis/
├── README.md          — this file
├── ROADMAP.md         — phased plan across cognitive + memory + integration tracks
├── CLAUDE.md          — locked decisions and constraints for future Claude sessions
├── HYPOTHESES.md      — falsifiable claims and evaluation philosophy
├── docs/              — design notes, research summaries, source-material refs
├── training/          — corpora, curriculum, LoRA configs, eval sets
├── memory/            — external memory system (schema, storage, policy)
├── runtime/           — agent loop, ollama integration, tool adapters
└── experiments/       — throwaway probes, benchmarks, feasibility checks
```
