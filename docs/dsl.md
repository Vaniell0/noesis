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

## Insight rendering

The composer renders `insights` rows as `insight <kind>` blocks. Body
uses the same key=value shape as events; `text_form` carries the
model-authored body verbatim.

### Fields

| SQL column       | DSL field     | Notes                                                    |
| ---------------- | ------------- | -------------------------------------------------------- |
| `id`             | `id`          | integer; needed for tool-call reference                  |
| `kind`           | *qualifier*   | `reflection` / `fact` / `skill` / `preference`           |
| `text_form`      | `text`        | required; model's own body                                |
| `tags_json`      | `tags`        | array of strings; omitted if empty                       |
| `importance`     | `imp`         | 0-10; omitted when 0                                     |
| `confidence`     | `conf`        | 0.0-1.0                                                  |
| `source`         | `src`         | `agent` / `tool_call` / `user`                           |
| `ts_created`     | `ts`          | ms since epoch                                            |
| `ts_last_seen`   | `ts_seen`     | omitted when equal to `ts_created`                       |
| `ts_last_used`   | `ts_used`     | omitted if never used                                    |
| `status`         | —             | never rendered; `superseded` / `deleted` rows are filtered by policy |

Provenance is rendered inline only when the retrieval policy asked for
it (default: hidden — the model does not need the audit graph on every
turn). When shown, provenance is a nested array of tiny objects:

```
insight fact { id=8812 ts=1721606712340 conf=0.9 src="agent"
               tags=["b0","memory","pivot"] imp=7
               text="B0 rewrite dropped the working-in-SQLite layer; \
                     session-scratch is RAM only."
               provenance=[ { ref="ep_tool_calls" id=4402 },
                            { ref="insights"      id=8790 },
                            { ref="external"      uri="https://…" } ] }

insight skill { id=1201 ts=1721600000000 conf=0.8 src="agent"
                tags=["shell","git"] imp=6
                text="Prefer `git switch` to `git checkout` for branch \
                      moves; `checkout` is overloaded and less safe." }

insight preference { id=15 ts=1721000000000 conf=1.0 src="user" imp=9
                     tags=["style"]
                     text="Russian for conversation, English for code, \
                           commits, docs." }
```

### What the composer does not render

- `status='superseded'` or `status='deleted'` rows. Retrieval filters
  them out. If the model asks for the audit graph explicitly (via a
  future introspection tool), those rows may surface with the status
  visible.
- Vector labels / embedding metadata.
- `confidence` when the retrieval score already dominates the ranking
  (composer's call; keeps prompts terser).

## Vault-ref rendering

Personal-vault content is external. What lives in the DSL is a
pointer plus, under an active retrieval grant, a bounded snippet.

```
vault_ref { id=7712 path="/home/vaniello/Documents/notes/rwkv-notes.md"
            mtime=1721600000000 size_bytes=8412
            snippet="… The G1 line differs from World in that it \
                     also trains on step-by-step reasoning traces \
                     from R1-distill …"
            snippet_range=[1240, 1408]  # byte offsets into the file
            score=0.71 }
```

- `snippet` is present only when the retrieval policy granted
  vault-read scope for this turn (see `docs/policies.md` §Zone
  permissions). Without a grant, the composer emits `vault_ref` with
  path/mtime/size/score and no body — the model knows the file
  matched but not what it says.
- Whole-file rendering is not supported at the DSL layer; snippets
  are always bounded. If the model needs more, it issues another
  `search` with a tighter query and higher `limit`.
- `snippet_range` gives byte offsets so the model can ask for
  adjacent context in a follow-up call. Off-by-one is on the
  composer, not the model.

## Tool-call surface (model → runtime)

The model emits a `tool_call <name>` block per operation. One block
per turn is the norm; multi-call turns are allowed and dispatched in
emission order (the runtime does not reorder).

### Shape

```
tool_call <name> { <arg_key> = <value>  ... }
```

The `<name>` matches one of the five ops in `memory/tool_calls.md`.
Argument keys match that file's shape verbatim — same field names,
same types. The runtime rejects unknown keys with a `tool_result`
error block (see next section); it does not silently drop.

### Examples

```
tool_call search { zone="insights" query="B0 pivot rationale"
                   filters={ kind="reflection" tags_any=["b0","pivot"] }
                   limit=8 min_conf=0.5 }

tool_call add_insight { kind="reflection"
                        text="The composer is the only NL translator; \
                              write-path stays structured."
                        tags=["b0","architecture","composer"]
                        importance=7 confidence=0.85
                        provenance=[ { ref_table="ep_tool_calls" ref_id=4402 } ] }

tool_call update_insight { id=8790
                           new_text="B0 rewrite dropped the working-in-SQLite \
                                     layer; session-scratch is RAM only."
                           rationale="clarified per structured-native pivot" }

tool_call delete_insight { id=1099
                           rationale="superseded by policy — vault snippets \
                                      not stored in insights" }

tool_call mark_important { id=15 importance=10
                           rationale="user-authored preference; never decay" }
```

### Validation on the runtime side

Rules the tool dispatcher applies before writing anything:

- Unknown `<name>` → immediate `tool_result` error, no `ep_tool_calls`
  row (the call never happened).
- Known `<name>` but unknown argument key → error, no
  `ep_tool_calls` row. Prevents typos silently landing wrong fields.
- Missing required argument → error, no `ep_tool_calls` row. Required
  set per op is authoritative in `memory/tool_calls.md`.
- Type mismatch (e.g. `importance="high"` instead of int) → error.
- Empty `rationale` on `update_insight` / `delete_insight` → error.
  Rationale is P10 audit; empty is not acceptable.
- `add_insight` with zero provenance edges → error.
- `search` with `zone="personal-vault"` when the current turn has no
  vault grant → *not* an error — the call is recorded to
  `ep_tool_calls` with `ok=true`, `result_json` carrying an empty
  rows array plus `refused="zone"`. This matches the graceful-refuse
  policy in `memory/tool_calls.md`.

A validated call is recorded to `ep_tool_calls` with `ok=NULL,
result_json=NULL` before dispatch, and updated on return. See
`memory/tool_calls.md` §Failure and safety.

## Tool-result wrapping (runtime → model)

Every dispatched `tool_call` produces a matching `tool_result <name>`
block in the next turn's context. Shape mirrors the `.result` shapes
in `memory/tool_calls.md`.

### Success shape

```
tool_result search { call_id=4402 ok=true latency_ms=142
                     rows=[
                       insight reflection { id=8812 ts=… conf=0.9 src="agent"
                                            tags=["b0","memory","pivot"] imp=7
                                            text="…" score=0.87 },
                       insight fact       { id=8790 ts=… conf=0.85 src="agent"
                                            tags=["b0","pivot"] imp=6
                                            text="…" score=0.71 } ] }

tool_result add_insight    { call_id=4405 ok=true id=8820 embedded=false }
tool_result update_insight { call_id=4406 ok=true old_id=8790 new_id=8821 }
tool_result delete_insight { call_id=4407 ok=true id=1099 }
tool_result mark_important { call_id=4408 ok=true id=15 old_importance=9 new_importance=10 }
```

- `call_id` is the `ep_tool_calls.id` of the originating call. Lets
  the model correlate result to call across turns without positional
  matching.
- `search` results are rendered as inline `insight <kind>` blocks
  (or, for non-insights zones, as inline `event <table>` /
  `vault_ref` blocks). This is the round-trip guarantee — the model
  reads results in the same shape as passively-rendered rows.
- `score` inside a result-row block is the retrieval-merged rank;
  the composer includes it only in tool_result contexts.

### Error shape

```
tool_result <name> { call_id=<int> ok=false
                     error="<short human message>"
                     code="<stable machine code>" }
```

Stable error codes (small enumeration; the model can learn to react):

| Code                    | Meaning                                                    |
| ----------------------- | ---------------------------------------------------------- |
| `unknown_tool`          | `<name>` did not match any op                              |
| `unknown_arg`           | argument key not in the op's schema                        |
| `missing_arg`           | required argument absent                                   |
| `type_mismatch`         | value type wrong for the argument                          |
| `empty_rationale`       | update/delete without rationale                            |
| `no_provenance`         | `add_insight` with empty `provenance` array                |
| `not_found`             | `id` referenced by update/delete/mark_important does not exist |
| `already_superseded`    | target insight is not `status='active'`                    |
| `internal`              | dispatcher / storage failure; details in `error`            |

Errors with codes `unknown_tool`/`unknown_arg`/`missing_arg`/
`type_mismatch`/`empty_rationale`/`no_provenance` are validation
failures and never touch `ep_tool_calls`. Errors with
`not_found`/`already_superseded`/`internal` reflect a call that
reached dispatch — the `ep_tool_calls` row exists with `ok=false`
and the error string.

## Composer normalisation

Rules the composer applies when emitting DSL, so the model sees
predictable shapes and can learn faster:

- **Field order.** Fixed per block kind. For events: `id ts [ts_end]
  session <kind-specific fields> imp`. For insights: `id ts [ts_seen]
  [ts_used] conf src [tags] [imp] text [provenance]`. Order is
  documentation and helps the parser too (one-pass shape check).
- **Omit-when-default.** `imp=0`, empty tag lists, `ts_seen ==
  ts_created`, `ts_used = NULL`, provenance without a grant — all
  omitted. Reduces token load; makes each visible field carry signal.
- **String rendering.** Prefer `"..."` unescaped when the body has no
  `\n`, `"`, `\`. Use `\n`-escaped multi-line for insight `text` when
  the body exceeds ~80 chars. Never truncate insight `text` at the
  composer; if it does not fit the turn budget, the row is dropped
  from the result set entirely (better a missing row than a lying
  one).
- **Timestamps.** Always ms-since-epoch integers in the DSL. If a
  human-readable version is worth the tokens for a specific block,
  the composer adds a sibling `ts_human="2026-07-22T00:35"` string —
  never replaces the integer.
- **Nested objects.** Rendered on one line when small; broken across
  lines with two-space indent when they exceed ~80 chars.
- **Deterministic ordering.** Within a result set of multiple rows,
  ordering is by retrieval score descending, ties broken by
  `ts_last_seen` descending, ties broken by `id` ascending. Stable
  ordering lets the model refer to "the first result" reliably across
  a turn.

## Grammar (EBNF, informal)

```
document       ::= { block } ;
block          ::= kind [ qualifier ] "{" fields "}" ;
kind           ::= identifier ;
qualifier      ::= identifier ;
fields         ::= { field [ separator ] } ;
field          ::= identifier "=" value ;
separator      ::= "," | whitespace ;
value          ::= literal
                 | array
                 | object
                 | block ;                # inline block-as-value in results
literal        ::= integer | float | string | boolean | "null" ;
array          ::= "[" [ value { "," value } [ "," ] ] "]" ;
object         ::= "{" fields "}" ;
identifier     ::= /[a-zA-Z_][a-zA-Z0-9_]*/ ;
integer        ::= /-?[0-9]+/ ;
float          ::= /-?[0-9]+\.[0-9]+([eE][-+]?[0-9]+)?/ ;
string         ::= /"([^"\\]|\\["\\nt])*"/ ;
boolean        ::= "true" | "false" ;
comment        ::= /#[^\n]*/ ;
```

This is intentionally simple: one-pass parseable, no context
sensitivity. Every ambiguity is resolved by "block if it has a header
identifier before `{`, object otherwise."

## Open questions (draft 1)

- **Versioned envelope.** Cheap insurance (`noesis/v1 { … }`) vs one
  more required line per prompt. Lean: skip in draft 1, add if we
  ever ship a v2 that breaks parses.
- **Positional shorthand** for very common blocks. Skipped in draft 1
  (couples DSL to `schema.sql` column order). Revisit when the
  input-tricks backlog acts.
- **Array-of-events wrapping** vs one-block-per-row. Skipped in draft
  1 (per-row is uniform and cheaper to parse). Revisit when we
  measure token cost on real prompts.
- **Escape strategy for embedded DSL in insight `text`.** If a model
  writes an insight whose body is itself DSL (e.g. quoting an
  earlier tool call), the current single-level backslash escape
  works but is ugly. Backtick-delimited raw strings are one option;
  deferred until we see the pattern.
- **Streaming.** If the model emits multi-KB `text` for an insight,
  should the runtime process incrementally? Ollama's SSE stream
  could be parsed as it lands. Deferred; not on the critical path.

---

*Draft 1 complete 2026-07-22. Next revision when A1 pilot data tells
us what shapes the model actually parses well vs badly.*

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
