# Extension Surface

**Status.** Phase-2 docs. No implementation yet. This document
freezes the shape so future changes can be evaluated against a
stable target.

**Reading order.**
- `~/.claude/plans/noesis-runtime-extensions-and-frontends.md` §3
  — the reasoning that led here.
- This file — spec.
- `docs/policies.md` §Policy engine — capability enforcement.

## What an extension is

An extension is a user-authored module that extends noesis's
substrate beyond the machine boundary. It can contribute three
things independently:

1. **Event source.** Emits typed events into the unified event
   store (same pipeline as core collectors). Declares an event
   schema at registration.
2. **Tool provider.** Registers callable tools in the DSL
   tool-call dispatcher, subject to policy engine capability
   grants.
3. **Embodiment binding.** Combination of (1) + (2) + optional
   lens template that gives noesis persistent presence in an
   external environment.

Examples: Minecraft-bot (wraps Mineflayer, emits game events,
registers move / chat / place tools, ships `game.minecraft` lens),
browser extension, IDE extension.

## Extension package layout

`/var/lib/noesis/extensions/<name>/`:

```
<name>/
├── manifest.toml          # metadata, capabilities, event schemas
├── lens/                  # optional lens templates this extension ships
│   └── <lens-id>.toml
├── code/                  # in-process callable module (Rust dylib, Python, ...)
├── schemas/               # per-extension event and tool schemas (JSON Schema)
└── README.md
```

## manifest.toml schema

```toml
[extension]
name = "mc-bot"
version = "0.1.0"
description = "Minecraft-bot embodiment via Mineflayer"

[capabilities]
# Capabilities to request. Policy engine (docs/policies.md §Policy
# engine) evaluates each at load time; denied capabilities => the
# extension is rejected and logged.
requires = ["net.socket", "input.synthesize"]
# What extension exposes into the noesis namespace.
provides = ["zone.game.mc", "tool.mc.*"]

[[events]]
# Zone the extension emits into. Declared here so retention and
# composer can size-cap and render events from this extension.
zone = "game.mc"
schema = "schemas/mc-events.json"

[[tools]]
# Tools appear in the model's tool preamble; matched by name +
# semantic-tag lookup, per NVIDIA G-Assist-shape (mirrored form,
# not transport).
name = "mc.move"
description = "Move the Minecraft character to a target position"
tags = ["mc", "movement"]
schema = "schemas/mc-tools.json#/definitions/mcMove"

[[lens]]
# Optional. Lens template registered when this extension loads;
# activated when noesis's lens rules match this extension's context.
id = "game.minecraft"
template = "lens/game.minecraft.toml"
```

Manifest shape mirrors NVIDIA G-Assist's plugin manifest for the
practical reason that self-describing tools enable semantic tool
matching by the model. Fields are simpler than G-Assist's (no
JSON-RPC transport — extensions run in-process).

## Trust model

Phase-2 extensions are user-authored. Runtime uses **in-process
callable dict**: extension registers a function pointer, supervisor
calls it directly. No IPC framing, no marshalling per call.

WASM isolation is reserved for the hypothetical case where noesis
ever accepts untrusted / marketplace extensions. Not Phase-2 scope.

## Registration lifecycle

1. Supervisor scans `/var/lib/noesis/extensions/*/manifest.toml` at
   startup.
2. For each: policy evaluator (`docs/policies.md` §Policy engine)
   checks if requested capabilities are granted for this system.
   Denied capabilities → extension rejected, logged.
3. Approved extensions loaded into runtime:
   - Event schemas registered in composer for rendering.
   - Tool schemas registered in dispatcher for tool-call routing.
   - Lens templates (if any) registered in lens manager.
4. On shutdown: extensions given a chance to flush pending events,
   then unloaded.

## How extension events flow

Same pipeline as core collectors — no separate path.

- Extension writes event to unified store
  (`/var/lib/noesis/store/episodic.db`).
- Retention applies same size + importance-tier rules
  (`docs/policies.md` §Retention).
- Composer renders extension events using the extension's declared
  schema. Schema field names appear verbatim in DSL rendering.
- **More extensions = more event surface = richer context absorbed
  by substrate.** No per-extension cap on emit rate yet (open
  question below).

## How extension tools flow

- Model emits `tool_call` DSL block naming a tool that matches an
  extension-registered tool.
- Dispatcher looks up tool in registry, checks policy (capability
  gate at fire time — some tools are `Allow`, some `Gate`
  per-fire per `docs/policies.md`).
- Extension callable invoked with parsed arguments; result returned
  via `tool_result` DSL block, absorbed into state on the next
  forward pass.

## Example: Minecraft-bot outline (illustrative, not yet built)

**What it wraps.** Mineflayer — Node.js Minecraft protocol library.
Mature, cross-platform, easy tool coverage.

**Events.** `mc.player_join`, `mc.player_leave`, `mc.block_placed`,
`mc.block_broken`, `mc.chat_message`, `mc.entity_hit`,
`mc.player_health`. All typed, schema-declared in
`schemas/mc-events.json`.

**Tools.**
- `mc.move { x, y, z }` — pathfind to coordinates.
- `mc.chat { text }` — send a chat message.
- `mc.place { x, y, z, block }` — place a block.
- `mc.mine { x, y, z }` — mine a block.
- `mc.craft { recipe }` — craft an item from inventory.

**Lens.** `game.minecraft`:
- Elevated `drip.rate_tokens_per_sec` (session-scoped, fans may
  spin — interactive mode per `docs/policies.md` §CPU budget).
- `input.synthesize` gate replaced with `mc.*` gates (safer than
  raw uinput; bounded action space).
- Persona from H15 reused as NPC / companion tone.

**Safety by construction.** Bounded action space (game world
only), no OS reach, no filesystem writes, no network egress beyond
the game server socket (managed by `net.socket` capability with
scope pattern like `mc.myserver.local:25565`).

**Why this matters.** Embodiment-via-extension is the safe
alternative to peer-Linux-user shell access. Even after
peer-Linux-user work matures, embodiment extensions remain the
recommended path for public demos and untrusted experiments — the
OS shell path is for the owner's own machine only.

## Capability taxonomy (from runtime plan §3)

Full table is in the runtime plan (§3 Capability taxonomy to
freeze). Highlights:

- `model.reasoning` → **always denied**. CLAUDE.md
  single-reasoning-model lock. Embodiment extensions use the *core*
  2.9B substrate for reasoning, not their own.
- `zone.personal_vault` → read-only, per-fire user confirm,
  audit-log every read.
- `input.synthesize` → per-fire user confirm (unless replaced by
  more specific bounded gates like `mc.*`).

## Open questions

- **Per-extension emit-rate governance.** Default: none. Reconsider
  when a real extension floods the store.
- **Extension-to-extension communication.** Default: none. All
  cross-extension state flows through the unified event store.
- **Embodiment lens arbitration** when two extensions claim the
  same context (e.g. two chat extensions for the same window
  class). Same open item as core lenses (`docs/memory-lenses.md`):
  priority / modal / merge.
- **Extension versioning and hot reload.** Not designed yet. Static
  load-at-startup is the Phase-2 default.
- **Language bindings.** Rust dylib is the natural first target
  (matches supervisor language). Python via PyO3 or a subprocess
  bridge is a possible Phase-3 add. Not decided.
