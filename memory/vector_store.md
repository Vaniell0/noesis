# Vector store — HNSW backend and on-demand embedding

*Rewrite 2026-07-22 under the structured-native pivot. Two questions
merged into one file: (a) which HNSW backend, (b) what gets embedded
and when.*

## What gets embedded (design invariant)

Not everything. The only zones that reach the vector store are:

- **`insights`** — every active row. Retrieval is semantic + filter +
  recency; the vector index is what makes semantic possible.
- **`personal-vault`** — one embedding per file (chunking is a
  B1/B2 refinement; B0 assumes whole-file for the pointer-index
  minimum viable path).

Not embedded:

- **`input-events`** and **`system-observations`** — raw event tables.
  Retrieval over these zones is filter-first (ts window, session, path
  prefix, tool name) plus optional keyword match on the small subset of
  columns that carry text (window titles, git commit summaries, tool
  args). Embedding a keystroke stream is not useful and would drown the
  index in signal.
- **`session-scratch`** — RAM only.

Rationale: on-demand semantic recall is expensive; the burst budget
(H1) means we cannot afford to embed the whole event tail. Insights are
the compressed, model-produced summaries; those are exactly the surface
where semantic search earns its cost.

## When embedding runs

- **On write, deferred.** `add_insight` / `update_insight` mark the row
  as embed-pending in the ingest queue but return immediately. A
  batched embed job runs in the next burst window (see `docs/policies.md`
  §CPU budget) and updates `insight_vectors`.
- **Vault reconciliation** runs on the same burst schedule. It walks
  the read-only vault mount, computes content hashes, inserts new
  `vault_refs`, drops rows whose files disappeared, and queues the
  changed set for embedding.
- **No sync-embed path.** The tool dispatcher never blocks a
  tool-call waiting for an embed. `add_insight.result` returns
  `embedded=false` when the row is still pending; retrieval will
  find it via filters/tags until the vector lands.

## Which HNSW backend

Two candidate paths for the store:

1. **Link `hnswlib` (C++)** into the Rust `noesis-store` crate via FFI
   (`bindgen` over a small C shim, or the existing Rust `hnsw_rs`
   crate).
2. **Reimplement in Rust.** Multiple pure-Rust HNSW crates exist
   (`instant-distance`, `hnsw`, `qdrant`'s embedded index). None are
   as battle-tested as `hnswlib`.

The 2026-07-21 draft recommended reimplementing in Python. That is
obsolete: the runtime is Rust+C now, and the pivot moves us to a
supervisor process that already links C libraries.

### Trade-off (Rust+C context)

| Axis                        | Link `hnswlib` (C++)                             | Pure-Rust crate                               |
| --------------------------- | ------------------------------------------------ | --------------------------------------------- |
| Search / build speed        | Best; the reference implementation               | Same order (all HNSW)                         |
| Build system                | Extra C++ toolchain leg; not free but tractable  | `cargo` only                                  |
| Ops surface                 | Atomic save, mutex, orphan cleanup all done      | Varies per crate; usually needs writing       |
| P7 (absorb prior work)      | Strong; `local-search` proved the params here    | Weak; second implementation of a solved op    |
| P12 (reversibility)         | Moderate; FFI adds swap cost                     | Strong; swap the crate at any time            |
| Failure isolation           | C++ crash bubbles as FFI abort — same process   | Rust panic — same process                     |
| Consistency with `noesis-store` | Fine; we already link C for other reasons  | Fine                                          |

### Recommendation

**Link `hnswlib` via a small C shim** from `noesis-store`. Rationale:

- We already accepted a Rust+C process, so the toolchain leg is paid
  for other reasons. Adding one more C dependency is marginal cost.
- `hnswlib` is the reference; its parameter choices are the ones
  `local-search` validated on this exact hardware. Reproducing them
  pointwise in Rust is work for no clear gain.
- Persistence, mutex discipline, and orphan cleanup are already
  correct in the C++ code path. Rewriting them in Rust is exactly the
  kind of second implementation P7 warns against.
- If C++ FFI turns out to be painful (mostly it will not, `cxx` or a
  hand-written `extern "C"` shim covers this), fallback is
  `instant-distance` — swap-in same interface at the `noesis-store`
  level, no schema change (P12 remains satisfied at the module
  boundary, not the crate boundary).

### Parameters

Same as `local-search`, unchanged:

- Space: cosine, implemented as InnerProductSpace over L2-normalised
  vectors.
- `M = 16`, `ef_construction = 200`, `ef_search = 50`.
- Persistence: atomic-rename (`hnsw.new` → `hnsw`) on flush.
- Reader/writer separation: `RwLock` in Rust wraps the C++ handle.

## Storage layout

- One HNSW index file per embedding namespace on the encrypted volume:
  `/var/lib/noesis/store/hnsw/insights.hnsw` and `.../vault.hnsw`.
- Cross-reference table in SQLite (`insight_vectors`, `vault_vectors`)
  is the single truth for label ↔ row_id. HNSW never carries the row
  body; only the label.
- Content hash lives in the SQLite row, not in the HNSW file. Re-embed
  gate is a hash compare, not an index-scan.

## Failure and recovery

- HNSW index files are content-addressable outputs derived from SQLite
  + the current embedder model. If they are lost or corrupt, the
  supervisor rebuilds from SQLite on next boot — one batched embed
  pass over every active `insights` row and every `vault_refs` row.
  Cost is one burst-window run per ~few thousand rows, bounded and
  predictable. This is why we do not treat the HNSW files as
  irreplaceable state.
- Embedder model change: bump `embedder_model` in `meta`, mark every
  vector row as `embedded=false`, run the same rebuild pass.

## Non-scope for B0

- **Chunking policy** for vault files — B1.
- **Skill-embedding as a separate namespace** — not needed while
  `kind='skill'` insights carry the load; can be added under
  `noesis-store` without schema break.
- **Rerank models.** Retrieval RRF and rerank are B1/B2.
