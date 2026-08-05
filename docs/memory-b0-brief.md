# B0 — Memory system

B0 is the structured event store and composer pipeline. It is built and running.

## What was built

| Crate | What it does |
|-------|-------------|
| `noesis-store` | SQLite: insert/get/prune/vacuum. Zones (input_events, system_observations, personal_vault, session_scratch, insights) are first-class rows. |
| `noesis-collector` | 6 collectors: file-events, terminal, clipboard, process, browser-tabs, active-window. |
| `noesis-composer` | Renders SQLite rows into DSL blocks for the model. Preamble + keyword retrieval. Wire format: `docs/dsl.md`. |
| `noesis-runtime` | Orchestrator: zone-permissions, retention scheduler, collector supervision. |

Schema: `memory/schema.sql`. Zone ops: `memory/tool_calls.md`.

## What remains

- **B1** — semantic retrieval (HNSW over insight embeddings). Not scaffolded. Reference impl: `local-search/src/semantic/vector_store.{hpp,cpp}` and `hybrid_search.{hpp,cpp}`.
- **B2** — memory-op RL (H3, Memory-R1 approach). Phase 2; blocked on A1 backbone lock.
- **Tool-call dispatcher** — parse + dispatch `tool_call` DSL blocks from model output. Spec in `docs/dsl.md`; not yet wired in `noesis-runtime`.

## Source material (reference implementations, not dependencies)

`key-daemon` and `local-search` at `~/Desktop/projects/` are the reference implementations for the collector and retrieval patterns. They are not imported — read to understand the event shapes and storage layout.

- `key-daemon/src/storage/database.{hpp,cpp}` — episodic table schema origin
- `local-search/src/semantic/vector_store.{hpp,cpp}` — HNSW API for B1
- `local-search/src/semantic/hybrid_search.{hpp,cpp}` — BM25 + vector RRF pattern
