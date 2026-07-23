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

## Quick Start (Phase-B skeleton, 2026-07-23)

**Status.** Phase-B skeleton — collectors + retention + Ollama-shape HTTP
shim, validated 2026-07-22 (`docs/verdicts/2026-07-22-phase-b-skeleton.md`).
No A1 fine-tune yet; runs against pre-trained RWKV-7 G1d 0.4B World via Nix.

```bash
git clone https://github.com/Vaniell0/noesis.git
cd noesis

# Weights (default variant; also noesis-model-q8_0 / q5_1 / q4_0)
nix build .#noesis-model             # -> ./result/model.bin

# Runtime workspace
cd runtime && cargo build --release  # or: nix build .#noesis-runtime

# Smoke: prompt -> tokens, prints tok/s
cargo run --release --example smoke -p noesis-rwkv -- \
    ../result/model.bin "Hello" 20

# HTTP shim on :11435 (Ollama-compat /api/generate)
cargo run --release -p noesis-runtime
```

**Running today:** collectors, retention sweeps, Ollama-compat HTTP
heartbeat.

**Not running yet:** composer, tool-dispatcher, A1 fine-tuned model,
multi-slot LoRA (H12b — `training/` scaffold only), lens persistence
(state save/load API pending in `noesis-rwkv-sys`), extension surface
(spec in `docs/extensions.md`, Phase-2).

**Interested in ideas rather than code?** `HYPOTHESES.md` lists every
claim noesis is testing (H1..H17). Each has a prediction and a
falsifier. Push-back welcome — see `CONTRIBUTING.md`.

## Vision

noesis absorbs and replaces two prior standalone daemons, `local-search` and
`key-daemon`, by pulling their essential functions into the ecosystem of one
small constant-cost RNN-based model. The wager: quality of reasoning and
quality of memory policy matter more than raw parameter count, and O(1)
per-token inference makes constant background operation economically
feasible on modest hardware.

Three long-term research threads live inside noesis:

1. **Reasoning as evolution of internal state.** Whether an RNN-family model
   (RWKV-7-G1), trained heavily on logic and reasoning traces, can serve as
   a persistent background reasoner without paying the quadratic attention
   cost of Transformers.

2. **Inter-model state transfer protocol.** How noesis and other models
   (remote Claude, other agents) exchange compact decodable representations
   of task state so that continuity survives the model handoff — the
   "cognitive memory of the computer, for other computers".

3. **Unified multimodal substrate (Phase 3+).** Whether the same WKV state
   that absorbs text can also absorb visual patches (framebuffer, video
   frames) and audio mel-frames through the same delta-rule update — one
   backbone for text ⊕ image ⊕ audio, not a split perception/reasoning
   stack. Precedent: VisualRWKV. See `HYPOTHESES.md` §H13a/H13b and the
   locked architectural note on unified vs split backends. A related
   Phase 3+ probe (§H16) asks whether the model can self-initiate output
   from within its own state dynamics — a continuous silent think-stream
   with a gated externalisation head, so noesis "chooses" when to speak
   rather than being polled by the supervisor.

## Architecture at a glance

- **Backbone.** RWKV-7-G1, starting size 2.9B. May scale to 13.3B later if
  hardware and cloud budget permit.
- **Runtime.** In-process rwkv.cpp via `noesis-rwkv-sys` + `noesis-rwkv`
  wrapper crates, driven by the `noesis-runtime` supervisor. An
  Ollama-shape HTTP shim on `:11435` (`/api/generate` NDJSON,
  `/api/tags`, `/api/version`) exposes the model so Claude-Code-style
  clients can talk to noesis without a bespoke harness (C0 verified
  2026-07-23; smoke via `experiments/A0_H12a_working_memory/run_probe.py`).
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
- **Open sources only** for training data. No personal corpus in weights
  for logic-fine-tune (Phase 1) or domain-knowledge fine-tune (Phase 2
  §H14). *Narrow carve-out*: personal chat traces may be used as
  supervision for **persona/style SFT only** (Phase 2 §H15 — teach the
  butler/secretary register), never as a knowledge or reasoning signal.
  Runtime retrieval remains the unrestricted channel.
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
