-- noesis memory — B0 schema (rewrite 2026-07-22 under structured-native pivot)
--
-- Zones (not layers) are the primary organizing axis. See memory/README.md.
-- Only durable zones live here — `session-scratch` (working) is RAM in the
-- supervisor and never touches disk.
--
-- Design invariants (locked, do not violate without an explicit reopen):
--
--   1. Events stay structured. No natural-language translation happens on
--      the write path. Translation is the composer's job (see
--      memory/README.md §composer), performed only when the model needs to
--      see something.
--   2. Retrieval is on-demand. No automatic embed of every event. The
--      vector store indexes only the zones that need semantic recall:
--      insights (model-produced), personal-vault (external content), and
--      optional rollup digests. See memory/vector_store.md.
--   3. Retention is per-zone. Long-tail raw events prune on a rolling
--      window; insights and cross-refs are kept indefinitely until the
--      supersession log deletes them.
--   4. The whole DB file lives under /var/lib/noesis/store on an
--      encrypted volume. See docs/policies.md §Disk encryption.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
PRAGMA synchronous  = NORMAL;   -- WAL + NORMAL = crash-safe, fast writes

-- ==========================================================================
-- Sessions (bookkeeping)
-- ==========================================================================
--
-- A session is one continuous run of the noesis supervisor. Session rows
-- are the only piece of "working" state that persists — the actual
-- session-scratch dictionary lives in RAM and never lands here.
--
-- Kept because: episodic events carry `session_id` for cheap "what happened
-- during that one supervisor run" queries. If we drop this, that query
-- becomes a range-over-ts join with process starts / exits inferred from
-- ep_session_events, which is possible but ugly.

CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY,
    ts_start   INTEGER NOT NULL,       -- ms since epoch
    ts_end     INTEGER,                -- NULL while open
    process_id INTEGER,                -- OS pid of the supervisor
    note       TEXT                    -- e.g. "restart after crash"
);

-- ==========================================================================
-- Zone: input-events
-- ==========================================================================
--
-- Raw input signals from key-daemon-style collectors. Flat per-kind tables
-- (no supertype). Fields carry ts + session_id inline so no join is needed
-- for the common timeline query. `importance` is a small integer 0-10
-- filled by the collector's cheap heuristic (see memory/event_ingest.md
-- §importance-scoring); it lets retrieval rank events without touching the
-- LLM.

CREATE TABLE IF NOT EXISTS ep_keystrokes (
    id         INTEGER PRIMARY KEY,
    ts         INTEGER NOT NULL,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    keycode    INTEGER NOT NULL,
    key_name   TEXT,
    state      INTEGER NOT NULL,       -- 0=up, 1=down, 2=repeat
    window_id  INTEGER,                -- FK to ep_window_focus.id
    modifiers  INTEGER DEFAULT 0,
    importance INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ep_keystrokes_ts ON ep_keystrokes(ts);

CREATE TABLE IF NOT EXISTS ep_mouse_clicks (
    id         INTEGER PRIMARY KEY,
    ts         INTEGER NOT NULL,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    button     INTEGER,
    x          INTEGER,
    y          INTEGER,
    window_id  INTEGER,
    importance INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ep_mouse_clicks_ts ON ep_mouse_clicks(ts);

CREATE TABLE IF NOT EXISTS ep_mouse_movement (
    id            INTEGER PRIMARY KEY,
    ts            INTEGER NOT NULL,     -- ms; aggregate flush time
    session_id    INTEGER NOT NULL REFERENCES sessions(id),
    distance_px   INTEGER,
    scroll_events INTEGER DEFAULT 0,
    importance    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ep_mouse_movement_ts ON ep_mouse_movement(ts);

CREATE TABLE IF NOT EXISTS ep_window_focus (
    id             INTEGER PRIMARY KEY,
    ts_start       INTEGER NOT NULL,
    ts_end         INTEGER,             -- NULL while focused
    session_id     INTEGER NOT NULL REFERENCES sessions(id),
    window_address TEXT,
    window_class   TEXT,
    window_title   TEXT,
    workspace      INTEGER,
    pid            INTEGER,
    is_fullscreen  INTEGER DEFAULT 0,
    is_floating    INTEGER DEFAULT 0,
    importance     INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ep_window_focus_ts ON ep_window_focus(ts_start);
CREATE INDEX IF NOT EXISTS idx_ep_window_focus_class ON ep_window_focus(window_class);

CREATE TABLE IF NOT EXISTS ep_window_events (
    id             INTEGER PRIMARY KEY,
    ts             INTEGER NOT NULL,
    session_id     INTEGER NOT NULL REFERENCES sessions(id),
    event          TEXT NOT NULL,       -- 'open' | 'close' | 'workspace'
    window_address TEXT,
    window_class   TEXT,
    window_title   TEXT,
    workspace      INTEGER,
    importance     INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ep_window_events_ts ON ep_window_events(ts);

CREATE TABLE IF NOT EXISTS ep_idle_periods (
    id         INTEGER PRIMARY KEY,
    ts_start   INTEGER NOT NULL,
    ts_end     INTEGER,
    session_id INTEGER NOT NULL REFERENCES sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_ep_idle_periods_ts ON ep_idle_periods(ts_start);

-- ==========================================================================
-- Zone: system-observations
-- ==========================================================================
--
-- What the world did around noesis: file changes, git ops, tool results
-- from noesis's own actions, and OS-level lifecycle events. Same
-- flat-per-kind pattern.

CREATE TABLE IF NOT EXISTS ep_session_events (
    id         INTEGER PRIMARY KEY,
    ts         INTEGER NOT NULL,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    event      TEXT NOT NULL,           -- 'login' | 'lock' | 'unlock' | 'suspend' | 'resume' | 'logout'
    importance INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ep_session_events_ts ON ep_session_events(ts);

CREATE TABLE IF NOT EXISTS ep_system_stats (
    id                 INTEGER PRIMARY KEY,
    ts                 INTEGER NOT NULL,
    session_id         INTEGER NOT NULL REFERENCES sessions(id),
    cpu_percent        REAL,
    ram_used_mb        INTEGER,
    ram_total_mb       INTEGER,
    swap_used_mb       INTEGER,
    battery_percent    INTEGER,
    battery_charging   INTEGER,
    brightness_percent INTEGER,
    net_rx_bytes       INTEGER,
    net_tx_bytes       INTEGER,
    disk_read_bytes    INTEGER,
    disk_write_bytes   INTEGER,
    audio_volume       INTEGER,
    audio_muted        INTEGER,
    importance         INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ep_system_stats_ts ON ep_system_stats(ts);

CREATE TABLE IF NOT EXISTS ep_file_events (
    id           INTEGER PRIMARY KEY,
    ts           INTEGER NOT NULL,
    session_id   INTEGER NOT NULL REFERENCES sessions(id),
    op           TEXT NOT NULL,         -- 'create' | 'modify' | 'delete' | 'rename'
    path         TEXT NOT NULL,
    old_path     TEXT,                  -- for 'rename'
    size_bytes   INTEGER,
    content_hash TEXT,                  -- SHA256 for small files, NULL otherwise
    importance   INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ep_file_events_ts ON ep_file_events(ts);
CREATE INDEX IF NOT EXISTS idx_ep_file_events_path ON ep_file_events(path);

CREATE TABLE IF NOT EXISTS ep_git_events (
    id         INTEGER PRIMARY KEY,
    ts         INTEGER NOT NULL,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    repo_path  TEXT NOT NULL,
    op         TEXT NOT NULL,           -- 'commit' | 'checkout' | 'merge' | 'push' | 'pull'
    rev        TEXT,
    summary    TEXT,                    -- first line of commit message
    importance INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ep_git_events_ts ON ep_git_events(ts);
CREATE INDEX IF NOT EXISTS idx_ep_git_events_repo ON ep_git_events(repo_path);

-- Every memory-op tool-call the model issues gets logged here. Rows also
-- flow into the same episodic timeline (query by ts). Kept in a separate
-- table because parameters/results are structured differently from
-- passive observations. See memory/tool_calls.md for the surface.
CREATE TABLE IF NOT EXISTS ep_tool_calls (
    id          INTEGER PRIMARY KEY,
    ts          INTEGER NOT NULL,
    session_id  INTEGER NOT NULL REFERENCES sessions(id),
    tool        TEXT NOT NULL,          -- 'search' | 'add_insight' | 'update_insight' | 'delete_insight' | 'mark_important'
    args_json   TEXT NOT NULL,          -- opaque to storage, structured by tool
    result_json TEXT,                   -- NULL if not yet returned
    latency_ms  INTEGER,
    ok          INTEGER,                -- 1=success 0=error, NULL=in-flight
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_ep_tool_calls_ts ON ep_tool_calls(ts);
CREATE INDEX IF NOT EXISTS idx_ep_tool_calls_tool ON ep_tool_calls(tool, ts);

-- ==========================================================================
-- Zone: insights  (model-produced, not raw events)
-- ==========================================================================
--
-- Anything the model produced that is meant to persist across sessions:
-- reflections, generalizations, learned procedures, and skills. Stored as
-- opaque `text_form` (natural language, whatever the model wrote) plus
-- optional structured tags. Provenance links back to the episodic rows
-- that inspired the insight.
--
-- Rejected earlier draft: subject/predicate/object triples. Reason: no
-- consumer needed them — the model reads `text_form` directly, retrieval
-- ranks on embedding + tags + importance + recency. Triples were storage
-- structure for its own sake.

CREATE TABLE IF NOT EXISTS insights (
    id           INTEGER PRIMARY KEY,
    ts_created   INTEGER NOT NULL,
    ts_last_seen INTEGER NOT NULL,      -- refresh when re-observed / re-derived
    ts_last_used INTEGER,               -- refresh when composer pulled it
    kind         TEXT NOT NULL,         -- 'reflection' | 'fact' | 'skill' | 'preference'
    text_form    TEXT NOT NULL,         -- the natural-language body
    tags_json    TEXT,                  -- JSON array of tag strings, optional
    importance   INTEGER NOT NULL,      -- 0-10; set by LLM at write time (Gen.Agents style)
    confidence   REAL NOT NULL,         -- [0.0, 1.0]
    source       TEXT NOT NULL,         -- 'agent' | 'tool_call' | 'user'
    status       TEXT NOT NULL DEFAULT 'active'  -- 'active' | 'superseded' | 'deleted'
);
CREATE INDEX IF NOT EXISTS idx_insights_kind_status ON insights(kind, status);
CREATE INDEX IF NOT EXISTS idx_insights_importance ON insights(importance);
CREATE INDEX IF NOT EXISTS idx_insights_ts_last_seen ON insights(ts_last_seen);

-- Provenance edges. An insight may be supported by multiple episodic rows
-- (from any ep_* table) or by other insights (summary distilled from
-- sub-insights). Kept as a separate table so a UPDATE from a tool-call can
-- rewrite one edge without touching the insight row.
CREATE TABLE IF NOT EXISTS insight_provenance (
    insight_id  INTEGER NOT NULL REFERENCES insights(id),
    ref_table   TEXT NOT NULL,          -- 'ep_keystrokes' | 'ep_window_focus' | ... | 'insights' | 'external'
    ref_id      INTEGER,                -- id in the referenced table (NULL for 'external')
    ref_uri     TEXT,                   -- for 'external' (URL, file path, ...)
    PRIMARY KEY (insight_id, ref_table, ref_id, ref_uri)
);
CREATE INDEX IF NOT EXISTS idx_insight_provenance_ref ON insight_provenance(ref_table, ref_id);

-- Vector-store cross-reference for insights. Only insights get embedded;
-- raw episodic events do not (see design invariant 2). The `label` is the
-- HNSW label from the semantic index; `content_hash` gates re-embed on
-- text_form change.
CREATE TABLE IF NOT EXISTS insight_vectors (
    insight_id   INTEGER PRIMARY KEY REFERENCES insights(id),
    label        INTEGER NOT NULL,      -- HNSW label
    content_hash TEXT NOT NULL,         -- SHA256 of text_form at embed time
    dims         INTEGER NOT NULL,
    ts_embedded  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_insight_vectors_label ON insight_vectors(label);

-- Supersession log. Tool-call UPDATE or DELETE moves the old insight to
-- status='superseded' and a row lands here linking old→new (or old→NULL
-- for delete). Preserves the audit trail P10 demands.
CREATE TABLE IF NOT EXISTS insight_supersessions (
    id             INTEGER PRIMARY KEY,
    ts             INTEGER NOT NULL,
    old_insight_id INTEGER NOT NULL REFERENCES insights(id),
    new_insight_id INTEGER REFERENCES insights(id),
    op             TEXT NOT NULL,        -- 'UPDATE' | 'DELETE'
    rationale      TEXT NOT NULL,
    tool_call_id   INTEGER REFERENCES ep_tool_calls(id)  -- which call caused this
);
CREATE INDEX IF NOT EXISTS idx_insight_supersessions_old ON insight_supersessions(old_insight_id);

-- ==========================================================================
-- Zone: personal-vault (external, indexed only)
-- ==========================================================================
--
-- Content itself is not stored here. Personal-vault (Obsidian notes,
-- source repos, arbitrary read-only user content) is mounted read-only
-- into noesis's filesystem view (see docs/policies.md §Zone permissions).
-- What we keep here is only:
--   * a pointer to the external file
--   * an embedding label so the vector store can retrieve it
--   * a content hash so we know when to re-embed
-- If the user removes a file from the vault, the pointer row is dropped
-- on the next reconciliation pass. Nothing about vault content lands in
-- ep_* tables.

CREATE TABLE IF NOT EXISTS vault_refs (
    id            INTEGER PRIMARY KEY,
    ts_indexed    INTEGER NOT NULL,
    path          TEXT NOT NULL UNIQUE,   -- absolute path inside the read-only mount
    content_hash  TEXT NOT NULL,          -- SHA256 of file body at index time
    size_bytes    INTEGER,
    mtime         INTEGER
);
CREATE INDEX IF NOT EXISTS idx_vault_refs_path ON vault_refs(path);

CREATE TABLE IF NOT EXISTS vault_vectors (
    vault_ref_id INTEGER PRIMARY KEY REFERENCES vault_refs(id),
    label        INTEGER NOT NULL,        -- HNSW label
    dims         INTEGER NOT NULL,
    ts_embedded  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_vault_vectors_label ON vault_vectors(label);

-- ==========================================================================
-- Metadata
-- ==========================================================================

CREATE TABLE IF NOT EXISTS meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);
-- Runtime seeds (populated by the supervisor on init, not by this DDL):
--   'schema_version'      — semver of this file
--   'embedder_model'      — Ollama model tag for embeddings
--   'embedder_dims'       — must match insight_vectors.dims / vault_vectors.dims
--   'episodic_drops_total'— count of overflow-dropped events; bumped by ingest
--   'last_reconcile_ts'   — timestamp of the last vault reconciliation pass
