# B0 — Memory schema draft (handoff brief)

This is a session-handoff for the person picking up Track B (external
memory system). B0 runs in parallel with Track A (fine-tune) per
`ROADMAP.md` — it is not blocked by anything currently in flight.

## Scope (from ROADMAP §B0)

Layered memory:

- **working** — short-horizon scratchpad, current session context
- **episodic** — event log (keystroke/window/file/git), timestamped
- **semantic** — extracted facts, distilled from episodic + agent output
- **skill-embedding** — reusable procedures / learned policies

Storage:

- **SQLite** for structured layers (working, episodic, semantic).
  Mirrors the choice `key-daemon` already made.
- **Vector store** for semantic recall + skill-embedding lookup.
  Repurpose the HNSW implementation from `local-search`.

## Source material (absorb, do not depend on)

Both projects are being *rebuilt as modules inside noesis*, not
maintained standalone (see `CLAUDE.md` §Locked decisions).

### key-daemon — `/home/vaniello/Desktop/projects/key-daemon/`

Event collectors — read these to understand the event shape:

- `src/collectors/input_collector.{hpp,cpp}` — libevdev keystroke/mouse
- `src/collectors/window_collector.{hpp,cpp}` — Hyprland IPC (active
  window, workspace transitions)
- `src/collectors/session_collector.{hpp,cpp}` — login/lock/suspend
- `src/collectors/system_collector.{hpp,cpp}` — power state, network,
  CPU/mem gauges
- `src/storage/database.{hpp,cpp}` — SQLite schema for events; the
  columns and table layout are the starting point for the episodic
  layer.

### local-search — `/home/vaniello/Desktop/projects/local-search/`

Retrieval primitives — reference implementations for B1:

- `src/search/engine.{hpp,cpp}` + `src/search/indexer.{hpp,cpp}` — BM25
  path, inverted index over local corpus.
- `src/semantic/vector_store.{hpp,cpp}` — HNSW vector store; take the
  index format and search API as-is.
- `src/semantic/hybrid_search.{hpp,cpp}` — BM25 + vector merge (RRF).
  This is the pattern B1 will replicate.
- `src/semantic/ollama_client.{hpp,cpp}` — embedder call site;
  interesting only for the API surface, not the impl (we already own
  Ollama access).

## B0 concrete deliverables

1. `memory/schema.sql` — DDL for the working + episodic + semantic
   tables. Copy the episodic layout from `key-daemon/src/storage`;
   design working (session-scoped, TTL-marked rows) and semantic
   (fact rows with provenance + confidence).
2. `memory/README.md` — short map of the module layout and how the
   four layers relate. No architecture essay; a one-page charter.
3. `memory/vector_store.md` — decision note on whether to link the
   local-search HNSW C++ code as a library or reimplement in Python.
   Trade-off: C++ is production-grade but adds FFI complexity; Python
   FAISS is simpler but a new dependency. No implementation yet, just
   the decision.
4. `memory/event_ingest.md` — plan for how event collectors feed into
   the episodic table. Reference key-daemon collectors by file:line.

## What NOT to do in B0

- No B1 retrieval implementation (that's a separate milestone).
- No B2 memory-op RL (that's Memory-R1 territory, weeks later).
- No changes to `runtime/` — B0 is schema + design docs only.
- No refactor of `key-daemon` or `local-search` themselves — those
  remain the reference implementations until absorbed.

## In-flight state at handoff

- Track A: `experiments/A0_eval/eval.py` re-run for
  `mollysama/rwkv-7-g1h:2.9b` at `num_predict=2048` is running as
  PID 4184951. Wall estimate 5-10h. Output going to
  `experiments/A0_eval/results/rwkv7_29b_g1h_np2048.{json,log}`. Do
  NOT kill it. Do NOT touch the Ollama daemon.
- `systemd-inhibit` PID 4192727 holds sleep/idle/lid inhibitor for
  12h to protect the run.
- `training/state_reg.py` has the A1 L_state loss implementation,
  ready to be wired into `RWKV-PEFT/lora_train.py` (task #9). This
  is Track A follow-up, not B0's problem.

## Reading order for a fresh session

1. This file.
2. `ROADMAP.md` §Track B.
3. `docs/principles.md` (P1..P12 tiebreakers).
4. `key-daemon/src/storage/database.hpp` (episodic seed).
5. `local-search/src/semantic/vector_store.hpp` (HNSW API).
