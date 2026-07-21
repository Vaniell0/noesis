# noesis/memory — module charter

External durable state for the noesis runtime. Implements P1 ("state is
external, cognition is internal") and is the substrate the composer
reads from and the tool-call surface writes to.

*Rewrite 2026-07-22. Supersedes the four-layer draft. See*
*`~/.claude/projects/-home-vaniello/memory/project_noesis_b0_pivot.md`*
*for the pivot rationale.*

## Zones, not layers

The old draft split memory into working/episodic/semantic/skill by
lifetime. That is not the primary axis. The primary axis is **who owns
the data, what its retention rules are, and what permissions apply**.
Lifetime falls out of the zone.

| Zone                 | Location                                | Content                                | Access                    |
| -------------------- | --------------------------------------- | -------------------------------------- | ------------------------- |
| `input-events`       | SQLite, `ep_*` tables                   | keystrokes / mouse / windows / idle    | append via ingest, read via retrieval |
| `system-observations`| SQLite, `ep_*` tables                   | files / git / session / stats / tool-calls | append via ingest, read via retrieval |
| `insights`           | SQLite `insights` + HNSW                | reflections, facts, skills, preferences | write via tool-call surface, read via retrieval |
| `personal-vault`     | External (Obsidian mount) + `vault_*`   | user's own notes and source content    | read-only via retrieval; content stays external |
| `session-scratch`    | RAM (supervisor process)                | current-session dictionary             | in-process only, never on disk |

Zone-level policies (filesystem permissions, retention windows,
encryption) live in `docs/policies.md`. This module implements the
storage side.

## Structured-native pipeline

The design invariant across every zone: **no natural-language
translation happens on the write path**. Events land in SQLite as
structured rows. The composer is the only component that ever renders
them for the model, and only for the specific rows retrieval selected
for a specific turn.

```
                                        ┌── noesis-composer ── DSL ──▶ model
collectors ─▶ ingest queue ─▶ SQLite ──┤
                                        └── noesis-policies picks rows
                                              ▲
                                              │  tool-call surface (search / add_insight / …)
                                              │
                                            model
```

- **Collectors** emit structured rows. See `event_ingest.md`.
- **Ingest queue** serialises a single writer to reduce WAL contention
  and match the discipline `local-search` learned.
- **SQLite** holds every durable zone. WAL, `synchronous=NORMAL`,
  `foreign_keys=ON`. Full DDL in `schema.sql`.
- **Policies** decide what to fetch for a turn. Zone-scoped —
  personal-vault requires an explicit opt-in per request; input-events
  are timeline-scoped by session or ts window; insights are ranked by
  embedding + importance + recency.
- **Composer** is the sole translator from structured rows to the
  DSL the model reads. Translation happens once, at read time, for
  exactly the rows the model will see. See `docs/composer.md` (TBD).
- **Tool-call surface** is the model's *write* API into insights. See
  `tool_calls.md`.

## Rust crate layout

The runtime is Rust supervising a Rust+C process tree. This module maps
to the following crates (workspace shape; not yet built):

| Crate             | Owns                                                   |
| ----------------- | ------------------------------------------------------ |
| `noesis-schema`   | DSL grammar for what the model reads and writes. Owns the machine-readable spec `schema.sql` mirrors. |
| `noesis-events`   | Collector adapters (evdev, Hyprland, inotify, dbus, git-hook shim, tool-call recorder). |
| `noesis-store`    | SQLite bindings, HNSW binding (via `hnswlib` C++ FFI), migrations, retention. |
| `noesis-policies` | Zone routing, retrieval planners, permission checks, redaction rules. |
| `noesis-composer` | Structured → DSL rendering; the *only* place that talks to the model. |
| `noesis-runtime`  | Supervisor loop, Ollama child process, scheduler, tool dispatch. |

`memory/` in this repo is the design substrate for `noesis-schema`,
`noesis-store`, `noesis-events`, and the parts of `noesis-policies`
that touch retention.

## Files in this module

- `schema.sql` — DDL for every durable zone.
- `tool_calls.md` — the memory-op tool surface the model uses to
  read and write insights.
- `vector_store.md` — decision note for the HNSW backend and the
  on-demand embedding policy.
- `event_ingest.md` — how collectors reach the store.

## What B0 does not decide

- **Retention policy** per zone. Concrete pruning windows land in B1
  once we have measurements. Design invariant is per-zone, not global.
- **Distillation / rollup cadence.** How often the model summarises
  episodic-into-insight is a scheduler question (bounded by the burst
  budget) that B2 owns.
- **DSL grammar.** Draft lives with `noesis-schema` (issue tracker
  task #6). This module is agnostic to the specific token syntax
  as long as it round-trips through the composer.
- **Retrieval ranking** (BM25 + vector merge, RRF, importance decay).
  B1. This module just makes the raw signals queryable.
- **Skill embedding** as a separate namespace. Insight rows with
  `kind='skill'` and `insight_vectors` cover the B0 need. If a
  separate ANN namespace turns out to be worth its cost, it slots in
  under `noesis-store` without a schema break.

## Reading order

If you are picking up B1:

1. This file.
2. `schema.sql` (top-to-bottom; the comments carry the rationale).
3. `tool_calls.md`.
4. `vector_store.md`.
5. `event_ingest.md`.
6. `docs/policies.md` — zone permissions and encryption.
