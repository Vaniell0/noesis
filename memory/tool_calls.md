# Memory-op tool surface

How the model reads and writes the `insights` zone. New in the 2026-07-22
rewrite. Design only; no impl.

*Rationale in short.* We want the model to treat memory the way a
Claude-Code agent treats a filesystem: through a small, uniform tool
surface. This is the MemGPT/Letta pattern — Memory-R1 is an upgrade
path we take if this v1 proves insufficient. See*
*`~/.claude/projects/-home-vaniello/memory/project_noesis_b0_pivot.md`*
*for the paradigm survey.*

## The surface

Five tools. All calls are recorded to `ep_tool_calls` (see
`schema.sql`) — so tool-call history is itself part of the episodic
timeline the model can retrieve later.

| Tool              | Purpose                                            | Writes  |
| ----------------- | -------------------------------------------------- | ------- |
| `search`          | Retrieve insights (and optionally other zones)     | no      |
| `add_insight`     | Persist a new reflection / fact / skill / pref     | yes     |
| `update_insight`  | Replace an existing insight with a corrected one   | yes     |
| `delete_insight`  | Mark an insight as invalidated                     | yes     |
| `mark_important`  | Nudge importance without rewriting the body        | yes     |

Everything else the model needs from memory (raw event lookups,
provenance walks, vault reads) goes through `search` with a different
`zone` argument. Keeping the tool count small matters — every tool
schema costs input tokens on every turn (see the input-tricks backlog).

## Shapes

All shapes are the DSL surface, not the wire format. `noesis-composer`
translates between them and JSON at the Ollama boundary. Fields marked
`optional` may be omitted from the DSL.

### `search`

```
search {
  zone     = "insights" | "input-events" | "system-observations" | "personal-vault"
  query    = "<free-text semantic query>"                # required for embed rank
  filters  = {                                            # optional
    kind        = "reflection" | "fact" | "skill" | "preference"   # insights only
    tags_any    = ["tag1", "tag2"]                        # insights only
    ts_after    = <ms since epoch>
    ts_before   = <ms since epoch>
    session_id  = <int>                                   # timeline zones
    path_prefix = "<absolute path>"                       # personal-vault only
  }
  limit    = <int, default 8>
  min_conf = <float 0..1, default 0.0>                    # insights only
}
```

Result:

```
search.result {
  rows = [
    {
      zone       = "insights"
      id         = <int>
      text_form  = "<opaque body>"
      kind       = "..."
      importance = <int 0..10>
      confidence = <float>
      tags       = [...]
      ts_last_seen = <ms>
      score      = <float>                # merged rank; opaque to the model
    },
    ...
  ]
}
```

Non-`insights` zones return rows shaped by their SQL table (see
`schema.sql`); the composer renders each row as a small DSL block.

### `add_insight`

```
add_insight {
  kind       = "reflection" | "fact" | "skill" | "preference"
  text_form  = "<the body the model wants to remember>"
  tags       = [...]                          # optional
  importance = <int 0..10>                    # model's own judgement
  confidence = <float 0..1>
  provenance = [                              # at least one entry
    { ref_table = "ep_tool_calls", ref_id = <int> },
    { ref_table = "insights",      ref_id = <int> },
    { ref_uri   = "https://…" }
  ]
}
```

Result: `add_insight.result { id = <int>, embedded = <bool> }`.

- `embedded=false` means the embed job was deferred to the next burst
  window (see `vector_store.md`). Row is queryable by filters + tags
  immediately; semantic search catches up on the next embed pass.

### `update_insight`

```
update_insight {
  id            = <int>          # existing insight
  new_text_form = "<corrected body>"     # optional if only fields changed
  new_tags      = [...]                  # optional
  new_importance = <int>                 # optional
  new_confidence = <float>               # optional
  rationale     = "<why the update>"     # required, lands in supersessions
}
```

Semantics: the old row is not mutated in place. It is moved to
`status='superseded'`, a new row is inserted, and an
`insight_supersessions` entry links them. Vector cross-ref moves with
the new row.

Result: `update_insight.result { old_id = <int>, new_id = <int> }`.

### `delete_insight`

```
delete_insight {
  id        = <int>
  rationale = "<why the delete>"
}
```

Semantics: sets `status='deleted'`, drops the vector-index label on the
next embed pass, records a supersessions entry with `new_insight_id=NULL`.
The row is retained (not physically deleted) for audit — P10 reversibility.

Result: `delete_insight.result { id = <int> }`.

### `mark_important`

```
mark_important {
  id         = <int>
  importance = <int 0..10>       # new value
  rationale  = "<why bump/drop>"  # short
}
```

Cheap variant: does not rewrite `text_form`, does not touch vectors,
does not create a supersession row (importance is a soft signal, not
a semantic change).

Result: `mark_important.result { id = <int>, old_importance = <int>, new_importance = <int> }`.

## What the model sees

Every turn's composer output includes a compact tool schema for the
five ops above. Full schema text is expensive; the `noesis-composer`
crate is responsible for compression (see the input-tricks backlog on
Anthropic MyTHOS / Fable). At worst we ship the raw JSON schema in
first turns and switch to compressed form once we measure the cost.

The composer decides — under the retrieval policy — which rows to
render into the turn's prompt. The model does not see the SQL. It sees
a DSL block per row and knows which tool to call by name and shape.

## What the model does not see

- Row ids from timeline zones (input-events / system-observations)
  unless they were pulled by retrieval for the current turn.
- Vector labels, HNSW internals, embedding vectors.
- The supersession log — it is an audit trail for the human, not
  training material for the model.
- Any personal-vault path unless the retrieval policy for this turn
  explicitly opts in and the redaction rule for the path allows it.

## Failure and safety

- Every tool-call is written to `ep_tool_calls` before dispatch, with
  `ok=NULL` (in-flight). On return the row is updated with `ok` and
  `latency_ms`. Crash during dispatch leaves the in-flight row visible
  and the model's next turn sees it as "still open" (P10 reversibility;
  no silent loss of intent).
- `update_insight` / `delete_insight` require `rationale`. Empty
  string is rejected by the tool dispatcher.
- `add_insight` requires at least one provenance edge. `ref_uri` alone
  is enough for external sources. This is P11 (explicit corpus
  lineage) applied to the memory write path.
- The tool dispatcher is the enforcement point for zone permissions.
  A `search` targeting `personal-vault` from a turn without an active
  vault-retrieval grant returns an empty result set with a
  `refused="zone"` flag, not an error. The model can then decide to
  ask the user.

## Non-scope for B0

- **Rate limits** on tool-calls per turn. B2 concern.
- **Batched adds** (`add_insights [...]`). Deferred; single-row is
  simpler and matches how the model already thinks turn-by-turn.
- **Reflection scheduling.** *When* the model should call
  `add_insight` unprompted is a policy question owned by
  `noesis-runtime`, not by this surface. The surface just makes the
  op available.
- **Introspection tools** (`list_recent`, `history_of_insight`). Land
  when B1 retrieval has data to browse.
