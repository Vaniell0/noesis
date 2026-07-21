# noesis DSL — grammar (draft 1)

The surface language the model reads and writes. Owned by the future
`noesis-schema` crate. This file is the reference spec; `noesis-composer`
implements the render side and `noesis-runtime` implements the tool-call
parse side.

*Draft 2026-07-22. Design-only, no impl. Task #6 in the tracker. Written
in chunks — this first chunk covers scope, lexical basis, and event
rendering. Insight/vault rendering, tool-call surface, and result
wrapping land in follow-up chunks.*

---

## Scope

The DSL has three jobs:

1. **Render.** Present SQLite rows and vault content to the model in a
   compact, uniform shape (so the model does not need to know that
   memory is SQL).
2. **Command.** Let the model issue memory-op tool calls
   (`search / add_insight / update_insight / delete_insight /
   mark_important`) without JSON boilerplate on the input side.
3. **Return.** Wrap tool-call results back in the same DSL so the
   read/write loop uses one language end-to-end.

The DSL is **not**:

- A programming language. No control flow, no expressions, no user
  functions. Every construct is either a block or a directive.
- A transport format. Ollama's OpenAI-compatible endpoint speaks JSON;
  the composer translates DSL ↔ JSON at the boundary.
- A stable public API. Draft 1 is expected to churn until A1 pilot
  data confirms what the model actually parses well. See §Open
  questions.

## Design goals

- **Cheap on input tokens.** Every extra character on the input side
  pays H1 tax on every turn. Prefer compact literals, no quoted keys
  where unambiguous, no redundant type tags.
- **Trivially machine-parseable.** LL(1) or LALR(1) with a hand-written
  or PEG parser; no lookahead beyond one block header. The runtime
  cannot afford a slow parser.
- **Trivially model-writeable.** Uniform block shape, minimal escape
  rules, no positional-vs-named-arg mixes. The model should be able to
  emit the DSL from a short exemplar without invoking a mental JSON
  serialiser.
- **Round-trippable through the composer.** Same shape reading and
  writing. If the composer renders an insight as `insight { … }`, the
  model updates it via a tool call whose arguments use the same field
  names.

## Lexical basis

### Comments

```
# line comment to end of line
```

Only line comments. Block comments would let the model hide reasoning
inside `/* … */` on the write path — undesirable for audit (P10).

### Tokens

- **Identifiers:** `[a-zA-Z_][a-zA-Z0-9_]*`. Used for block kinds and
  field keys.
- **Integers:** `-?[0-9]+`. Decimal only. No hex, no underscores in
  literals (simplifies parser; if the user's data needs hex, the
  composer renders a hex-tagged string).
- **Floats:** `-?[0-9]+\.[0-9]+`, optionally with `e[-+]?[0-9]+`.
- **Booleans:** `true`, `false`.
- **Null:** `null`. Rendered by the composer for SQL NULL columns
  when the field is present in the shape but has no value.
- **Timestamps:** integer milliseconds since Unix epoch, unadorned.
  Rationale: same as `schema.sql`. If human-readable dates are needed
  in prompts, that is a composer decision (e.g. adding a
  `ts_human="2026-07-22T00:35"` sibling field), not a DSL type.
- **Strings:** `"…"` with backslash escapes: `\\`, `\"`, `\n`, `\t`.
  No triple-quoted strings; embedded newlines allowed via `\n`. The
  composer prefers unescaped short strings whenever the token
  budget permits — see §Composer normalisation (TBD in later chunk).
- **Arrays:** `[v1, v2, v3]`. Trailing comma allowed; the parser
  ignores it.
- **Nested objects:** `{ key = value, key = value }`. See §Block
  syntax below — a nested object is a value, a block is
  header-prefixed.

### Whitespace and separators

- Whitespace between tokens is optional and insignificant.
- Fields inside a block or object are separated by either whitespace
  or comma. Both are accepted; the composer renders with newlines
  (one field per line) for readability but the parser tolerates
  either.
- No trailing punctuation required at end of block.

## Block syntax

Everything the composer renders is a **block**. A block has three
parts:

```
<kind> [<qualifier>] { field = value  field = value  ... }
```

- `<kind>` — required identifier. Examples: `event`, `insight`,
  `vault_ref`, `tool_call`, `tool_result`.
- `<qualifier>` — optional identifier. For events it is the source
  table without the `ep_` prefix (`keystrokes`, `window_focus`,
  `git_events`, `tool_calls`). For insights it is the `kind` column
  value (`reflection`, `fact`, `skill`, `preference`). For results
  it is the tool name.
- Body is a set of `key = value` bindings inside braces.

Rationale for `<kind> <qualifier>` over `<kind>.<qualifier>`: the dot
introduces an operator the parser now has to distinguish from a
decimal (in `1.5`) and from a path (in `foo.bar` strings). A space
is unambiguous.

## Event rendering

The composer renders `ep_*` rows into `event <table>` blocks. Field
names in the DSL match the SQL column names *minus* the redundant
prefixes (`ts_` → `ts` when unambiguous inside a block, `is_` kept as
a bool marker). `id` and `session_id` are included; the model needs
`id` to reference an event in a follow-up tool call, and
`session_id` grounds the timeline.

### Common fields (all event blocks)

| SQL column        | DSL field     | Notes                                       |
| ----------------- | ------------- | ------------------------------------------- |
| `id`              | `id`          | integer                                     |
| `ts`, `ts_start`  | `ts`          | ms since epoch                              |
| `ts_end`          | `ts_end`      | omitted if NULL (still open)                |
| `session_id`      | `session`     | integer                                     |
| `importance`      | `imp`         | integer 0-10 — short name because it's noisy |

Every event block that carries `importance = 0` at render time has
that field **omitted** (the default). The composer only emits the key
when it carries signal.

### Per-kind shapes

Below, one example per kind. The full field set for each kind mirrors
`schema.sql`; the composer's job is to drop NULL columns and rename
per the table above.

```
event keystrokes { id=134582 ts=1721606712345 session=17
                   keycode=28 key="Return" state=1 window_id=88213
                   modifiers=4 imp=3 }

event mouse_clicks { id=99101 ts=1721606712112 session=17
                     button=1 x=812 y=440 window_id=88213 }

event mouse_movement { id=44120 ts=1721606712500 session=17
                       distance_px=1874 scroll_events=2 }

event window_focus { id=88213 ts=1721606710000 ts_end=1721606712900
                     session=17 window_class="kitty"
                     window_title="claude — noesis"
                     workspace=2 pid=4184001 imp=6 }

event window_events { id=88401 ts=1721606712910 session=17
                      event="close" window_class="kitty"
                      window_title="claude — noesis"
                      workspace=2 }

event idle_periods { id=550 ts=1721603000000 ts_end=1721606100000
                     session=17 }

event session_events { id=12 ts=1721606700000 session=17
                       event="unlock" imp=5 }

event system_stats { id=8801 ts=1721606712000 session=17
                     cpu_percent=4.7 ram_used_mb=6180 ram_total_mb=16384
                     battery_percent=63 battery_charging=false
                     net_rx_bytes=12093 net_tx_bytes=884 }

event file_events { id=7712 ts=1721606712611 session=17
                    op="modify" path="/home/vaniello/Desktop/projects/noesis/memory/schema.sql"
                    size_bytes=15463
                    content_hash="sha256:d1e4…"
                    imp=7 }

event git_events { id=331 ts=1721606750000 session=17
                   repo_path="/home/vaniello/Desktop/projects/noesis"
                   op="commit" rev="a1b2c3d4"
                   summary="B0 rewrite: zones as primary axis"
                   imp=8 }

event tool_calls { id=4402 ts=1721606712100 session=17
                   tool="search"
                   args={ zone="insights" query="B0 pivot" limit=8 }
                   result={ rows_returned=3 }
                   latency_ms=142 ok=true }
```

Notes on the `tool_calls` event block:

- `args` and `result` are rendered as **nested objects**, not string
  blobs. The composer parses the stored JSON and re-emits it in DSL
  so the model reads uniform syntax across the whole prompt.
- `result` is folded to a small summary shape when the full result
  would blow the token budget (e.g. `{ rows_returned=3 }` instead of
  each row). The full result is retrievable via a new `search` call
  by `id`.
- `error` is included only when `ok=false`.

### What the composer does not render

- Aggregate rows the retrieval policy did not pick. A turn typically
  shows 5-20 event blocks total across all kinds, not the full stream.
- Content-hash and size fields on events the model does not need
  them for. `file_events` gets `content_hash` when the model is
  reasoning about content change; it does not get it when the model
  is only reasoning about a modification timeline.
- Any event under the credentials skip-list (see
  `docs/policies.md` §Credentials). Redaction is by construction —
  the row never reaches the composer.

---

## Follow-up chunks (not in this draft)

- **Insight and vault_ref rendering.** Same block shape; insight
  bodies are wrapped strings, vault refs are pointers with optional
  content snippets under retrieval policy.
- **Tool-call surface.** `tool_call <name> { … }` blocks the model
  emits; parse rules and validation on the runtime side.
- **Tool-result wrapping.** `tool_result <name> { … }` the runtime
  writes back for the next turn.
- **Composer normalisation.** Whitespace policy, ordering, truncation
  rules, error blocks.
- **Grammar in EBNF.** Formal spec once the shapes above stabilise.

## Open questions (for later chunks / user)

- Does the DSL need a versioned envelope (e.g. `noesis/v1 { … }`) so
  the runtime can reject prompts fed to the wrong parser? Cheap
  insurance; costs one line per prompt.
- Do we want positional shorthand for very common blocks? E.g.
  `event keystrokes(134582 1721606712345 17 28 "Return" 1)` —
  compresses further but couples DSL to column order in `schema.sql`.
  Prefer named fields for the draft; revisit if the input-tricks
  backlog (Fable/MyTHOS) makes positional cheap enough.
- How does the composer render arrays of events (a search returning
  20 window-focus rows)? One block per row is verbose; a `events
  window_focus [ { … }, { … } ]` array-of-objects shape is compact
  but doubles the parser rules. TBD after we measure the token cost
  on a real prompt.
