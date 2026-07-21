# Event ingestion — collectors → structured store

*Rewrite 2026-07-22 under the structured-native pivot. Events stay
structured on the write path; no NL translation happens here.*

How events reach the durable zones defined in `schema.sql`. Design
only; no impl.

## Design invariants

1. **Structured all the way down.** Collectors emit typed rows for
   specific tables. There is no NL rendering, no "digest at ingest".
   Composer is the only translator, and it runs only at read time.
2. **Single writer per DB file.** All collectors funnel into an
   in-process queue owned by the `noesis-events` crate; a single
   writer task in `noesis-store` drains it. Mirrors the discipline
   `local-search` learned when Xapian lock conflicts bit us.
3. **Backpressure over blocking.** Bounded queues, drop-oldest on
   overflow, drop-counter in `meta`. Losing a mouse-move is
   preferable to blocking the runtime — H1 budget dominates.
4. **Cheap importance at ingest.** Every row gets an `importance` 0-10
   from a cheap heuristic (see §importance-scoring). The LLM never
   sees the raw stream; retrieval ranks with `importance` as one
   term. No LLM in the ingest path.

## Sources

Six families of collector. Three lifted from `key-daemon`, one lifted
from `local-search`, one from git, and one is the tool-call recorder
that is internal to `noesis-runtime`.

### 1. Input (keystroke, mouse) — `noesis-events::input`

Reference: `key-daemon/src/collectors/input_collector.{hpp,cpp}`.

- `libevdev` on `/dev/input/event*` (evdev via a Rust binding, e.g.
  `evdev-rs` — chosen over a C FFI shim because there is nothing
  library-shaped to reuse here beyond the wire format).
- Discovers keyboard + mouse devices at startup; returns fds for the
  supervisor's epoll loop.
- Modifiers tracked as a bitmask; mouse movement accumulated inside
  the collector and flushed on a timer as an aggregate row (raw pixel
  deltas are noise).

Writes to: `ep_keystrokes`, `ep_mouse_clicks`, `ep_mouse_movement`.
Direct insert, no supertype join.

### 2. Window (Hyprland IPC) — `noesis-events::window`

Reference: `key-daemon/src/collectors/window_collector.{hpp,cpp}`.

- Connects to Hyprland socket; dispatches on `activewindow` /
  `activewindowv2` / `workspace` / `openwindow` / `closewindow`.
- Emits *intervals* (`ep_window_focus`, close via `ts_end`) and
  *discrete events* (`ep_window_events`).
- Focus-end updates the existing `ep_window_focus` row's `ts_end`,
  matching `Database::update_window_focus_end` in `key-daemon`.

### 3. Session + system — `noesis-events::session`, `noesis-events::system`

References: `session_collector.{hpp,cpp}` and
`system_collector.{hpp,cpp}` under `key-daemon/src/collectors/`.

- Session events: `login` / `lock` / `unlock` / `suspend` / `resume` /
  `logout` via logind dbus. Writes `ep_session_events`.
- System stats: CPU / RAM / battery / net / disk / audio sampled on a
  slow timer (60 s default). Writes `ep_system_stats`. Idle detection
  uses input silence to open/close `ep_idle_periods`.

### 4. File — `noesis-events::files`

Reference: `local-search`'s inotify watcher module.

- `inotify` (Linux native). Path scope = configured working roots
  minus the credentials skip-list in `docs/policies.md`.
- Emits `ep_file_events` rows on `create` / `modify` / `delete` /
  `rename`.
- Content hash (SHA-256) computed only for files under a size
  threshold (proposal: 1 MiB; revisit in B1). Above threshold,
  `content_hash IS NULL`.
- Files under `personal-vault` mount are *not* observed here; that
  zone is read-only and its reconciliation runs on a different
  schedule (see `vector_store.md` §Vault reconciliation).

### 5. Git — `noesis-events::git`

No reference implementation. Two paths:

- **Preferred: hooks.** Post-commit / post-checkout / post-merge
  hooks in user's active repos write a single JSON line to a well-
  known unix socket. Low latency, cheap at rest.
- **Fallback: reflog poll.** For repos the runtime cannot install
  hooks into (permission, immutable checkouts). Higher latency,
  runs on the burst schedule.

Writes `ep_git_events`.

### 6. Tool-call recorder — `noesis-events::tool_calls`

Internal to `noesis-runtime`. Every model-issued memory-op tool-call
lands in `ep_tool_calls` via this recorder — on dispatch (with
`ok=NULL, result_json=NULL`), on return (fills `ok`, `latency_ms`,
`result_json`). See `tool_calls.md`.

## Ingestion path

```
collector (async task)
      │
      ▼
in-proc bounded MPMC queue  ── overflow ──▶ drop-counter in meta
      │
      ▼
noesis-store writer task     ── batched BEGIN/COMMIT ──▶ SQLite (WAL)
```

- Every collector is an async task inside the `noesis-runtime`
  supervisor. No cross-process IPC for durable ingest.
- The queue is bounded per priority band (see below). Overflow drops
  the oldest of the *lowest active priority band*, never blocks.
- The writer batches inside a single transaction. Flush cadence: 1 s
  or on batch size N (default N=256), whichever first. Session-end
  forces a flush.

## Priority bands

Not every event is worth the same. The queue is partitioned:

| Band | Members                                                     | On overflow |
| ---- | ----------------------------------------------------------- | ----------- |
| 0    | `ep_tool_calls`, `ep_session_events`, `ep_git_events`       | never drop  |
| 1    | `ep_window_focus`, `ep_window_events`, `ep_file_events`     | drop last   |
| 2    | `ep_keystrokes`, `ep_mouse_clicks`                          | drop last   |
| 3    | `ep_mouse_movement`, `ep_system_stats`, `ep_idle_periods`   | drop first  |

- Band 0 never drops — losing a tool-call breaks the model's
  self-view (it thinks it wrote, we lost the row). If band 0 saturates
  the queue and the writer is stuck, the supervisor logs and refuses
  new tool-calls until pressure clears. This is a P10 reversibility
  choice: better to fail loud than lose semantic state.
- Coarser high-volume events (band 3) are the first casualty.

## Importance scoring

Cheap per-collector heuristic that produces `importance` 0-10 at
ingest. No LLM. Examples:

- `ep_keystrokes`: 0 for repeats, 1 for typing inside an editor
  window, 3 for shortcuts with modifiers.
- `ep_window_focus`: 5 by default, +2 if window class matches an
  editor/terminal, +1 per minute of focus (bounded at 8).
- `ep_file_events`: 3 default, 5 if under an active project root,
  7 if git-tracked.
- `ep_git_events`: 6 default, 8 for commits with a message ≥ N chars,
  9 for push / merge.
- `ep_tool_calls`: 7 (they carry the model's own intent).
- `ep_system_stats`: 1 default, +1 per gauge that crossed a
  significant threshold since the last row (battery <20 %, CPU pinned,
  RAM near cap).

These are placeholders. `noesis-events` exposes them as tunable
consts; they get calibrated in B1 once we have a real event corpus to
measure against. The LLM never reads or overrides these values on the
ingest path — importance is a retrieval signal, not a semantic tag.

## Crash recovery

- SQLite WAL replay covers atomicity per row. Nothing about ingest is
  transactional across the process boundary.
- The in-proc queue is *lossy by design*. On crash, in-flight batches
  are gone. This is fine for bands 2/3 (already lossy) and fine for
  bands 0/1 only if the collector can re-emit — `ep_tool_calls`
  in-flight rows (`ok=NULL`) survive because they were written before
  dispatch (see `tool_calls.md`); `ep_git_events` is idempotent per
  `rev`; `ep_file_events` catches up on the next inotify batch.

## Non-scope for B0

- Distillation (`ep_*` → `insights`) is a scheduler-plus-model
  concern owned by `noesis-runtime`. Not an ingest question.
- Encryption. Handled at the volume level (`docs/policies.md`).
- Retention windows. B1, per-zone.
- Exact wire format for the git-hook shim. B1.
