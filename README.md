# noesis

*Cognitive runtime for the machine.*

The machine already sees everything on it — every keystroke, focus
change, filesystem event, process transition flows through the kernel
whether or not anyone reads it. noesis is the layer that starts to
*mean* those events: absorbing the input stream into structured form,
holding it in a state that survives across sessions, and letting the
machine act on its own understanding. Not surveillance — the OS
already knows. Understanding is the difference.

The runtime is the point. Collectors, structured event store, zones,
composer, tool dispatcher, extension surface, encrypted persistence,
all running under a dedicated Linux user (`noesis` uid, own home under
`/var/lib/noesis`). It runs continuously whether or not anyone is
talking to it. Interfaces on top (an Ollama-compatible CLI over an
HTTP shim, extensions in browser / IDE / game world) are how the owner
engages with it, not the substance.

An RWKV-7 model sits at the centre as the active integrator of this
runtime — the piece that continuously processes incoming events, forms
"what is happening now" from them, and writes that into memory per the
DSL described in `docs/dsl.md`. Reasoning lives in the model's WKV
state, not in generated tokens; the state is where accumulated context
and inference actually happen. Tokens are a rare emit — an
externalisation of what the state already knows.

## Why a specialised model is needed

For the coordinator to keep the runtime coherent — to hold structure
across long-running continuous input, connect events over hours or days,
route the substrate's own tool surface without hallucinating — it has to
reason in its own hidden state, not in generated tokens. Standard LLMs,
of any architecture, degrade under continuous ambient input; Transformers
under long uninterrupted contexts especially so.

noesis bets that an RWKV-7 backbone fine-tuned to keep logical
connections alive in its state (H4b's training arm) is the architecture
that can hold this together where others cannot. Constant-cost per-token
inference is the price side of that bet — it makes the runtime
economically viable to run continuously on the laptop i5-1235U without
cloud dependency for the everyday loop.

This is unproven. `HYPOTHESES.md` records every specific wager — H4b
(state as computation), H7 (understanding in weights, not context),
H12b (multi-slot LoRA as working memory), H16 (self-initiated
externalisation), H17 (state absorption vs history re-injection), and
the rest — with predictions and falsifiers. A cleanly falsified
hypothesis is a useful signal.

## What the runtime does, independent of interaction

- **Absorbs the machine's stream into structured zones.** input-events,
  system-observations, personal-vault (read-only bind mount),
  session-scratch (RAM). Collectors write structured rows; no
  natural-language translation on the write path. See `memory/README.md`.
- **Retains under per-zone policy.** Sizes, importance tiers, encryption.
  See `docs/policies.md`.
- **Composes context on demand.** The composer is the sole component that
  renders structured rows into DSL for the model, and only for the rows
  retrieval selected for a specific turn.
- **Routes tool calls.** The dispatcher exposes the runtime's own
  capabilities (memory ops, retrieval, extension-registered tools) to
  the model.
- **Escalates by the owner's explicit call.** When the owner escalates
  to an external frontier model, the composer emits noesis's state as a
  DSL handoff — not a chat log. The DSL layer is API-agnostic; the
  composer translates to whichever target transport the owner has
  configured. noesis never escalates on its own.
- **Hosts user-authored extensions** as a first-class capability. Same
  substrate, new environments. See `docs/extensions.md` (Phase-2 docs).

The functions that `local-search` (personal-corpus retrieval) and
`key-daemon` (system-context surface) used to provide are becoming zones
inside this runtime rather than standalone processes. Those daemons are
no longer maintained; their code remains source material for the ports.

## Architecture at a glance

- **Backbone.** RWKV-7-G1, 2.9B. May scale to 13.3B if hardware and
  cloud budget permit.
- **Supervisor.** Rust supervisor (`noesis-runtime`) with rwkv-cpp
  in-process inference (`noesis-assistant`). Ollama-compatible HTTP shim on
  `:11435` (`/api/generate` NDJSON, `/api/tags`, `/api/version`) so
  standard clients talk to noesis without a bespoke harness. Ollama
  removed 2026-07-25 — it does not expose WKV state APIs (`rwkv_get_state_len`,
  `rwkv_eval(..., state_out)`, `rwkv_clone_state`) needed by H17/H18/lens persistence.
- **Peer Linux user.** Own uid, own `/var/lib/noesis` home, encrypted
  LUKS+BTRFS store, cannot see the primary user's home. systemd
  hardening locked in `docs/policies.md`.
- **Structured-native pipeline.** collectors → ingest queue → SQLite →
  composer → model. Composer is the only translator to model-facing
  text.

## Status

Phase-B skeleton validated 2026-07-22
(`docs/verdicts/2026-07-22-phase-b-skeleton.md`).

Running:
- Collectors (6: file-events, terminal, clipboard, process, browser-tabs, active-window).
- noesis-store: SQLite insert/get/prune/vacuum for all zones.
- noesis-runtime: orchestrator, zone-permissions, retention scheduler.
- noesis-composer: preamble rendering + keyword retrieval over SQLite (5 tests pass).
- noesis-shim: 8 HTTP endpoints live.
- A1 fine-tune: Step 5 training (16007/33933 steps, 47%) on cloud VM.

Not running yet (stubs):
- Calibration CLI (thermal/RAPL reader is a stub; udev rule exists).
- Lens scratch (design in `docs/memory-lenses.md`, H11 falsifier).
- Utility model lazy-load.
- Tool-call dispatcher (design in `docs/dsl.md`).
- Extension surface (Phase-2 docs, `docs/extensions.md`).
- Lens persistence (pending state save/load API in `noesis-rwkv-sys`).
- Multi-slot LoRA H12b (blocked on H12a v2 verdict).

## Hard constraints

- **Open sources only** for weights. Personal corpus is a runtime
  retrieval channel, never a fine-tune signal. Narrow carve-out:
  persona/style SFT (§H15).
- **Cheap by construction.** Laptop i5-1235U for inference (CPU-only).
  Cloud burst for training is an explicit decision, not a default.
- **Single local reasoning model.** Utility NNs (embedders, routers,
  small policies) are welcome. Additional local *reasoning* models are
  not. Heuristic (`docs/principles.md` P3): *if it emits tokens that
  participate in a chain of thought, it is a reasoning model.*
- **Not a Transformer.** Any switch requires empirical re-open, not
  architectural drift.
- **Not a SaaS.** noesis is a personal daily bot for the owner's own
  machine. Extensions are for the owner to author, giving the runtime
  presence in new environments.

## Who this is for

- **The owner's own machine.** Design target.
- **Community push-back on the wagers.** `HYPOTHESES.md` is the point of
  contact. A clean falsification is more valuable than a green build.
- **Not:** a replacement for frontier reasoning models, a general
  deployable assistant, a product.

## Repository layout

```
noesis/
├── README.md          — this file
├── ROADMAP.md         — phased plan across cognitive + memory + integration
├── HYPOTHESES.md      — falsifiable claims (H1..H17)
├── CONTRIBUTING.md    — how to engage (hypotheses > runtime PRs)
├── FAILED.md          — refuted hypotheses and dead experiments
├── docs/              — policies, principles, DSL, extensions, verdicts
├── training/          — A1 pipeline (blocked on GPU)
├── memory/            — external memory system spec + schema
├── runtime/           — Rust supervisor, collectors, HTTP shim
└── experiments/       — throwaway probes and A0.* feasibility checks
```
