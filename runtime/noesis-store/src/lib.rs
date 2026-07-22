//! noesis-store — one SQLite database per zone.
//!
//! Layout under `state_path`:
//!   input_events/db.sqlite
//!   system_obs/db.sqlite
//!   personal_vault/db.sqlite
//!   session_scratch/db.sqlite
//!
//! Rationale for per-zone databases:
//!   * WAL contention is local to a zone (busy tracker doesn't stall vault).
//!   * Retention/backup policy differs per zone (scratch nukeable, vault sacred).
//!   * A corrupted zone doesn't take the runtime down; supervisor can quarantine.
//!
//! Cross-zone references are string-encoded (`"zone:id"`); the store never
//! joins across zones, that's the runtime's job.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use noesis_schema::{Event, EventInput, EventRef, Zone};
use rusqlite::{params, Connection, OptionalExtension};

pub const SCHEMA_VERSION: i32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("schema version {found} unsupported (expected {expected})")]
    UnsupportedSchema { found: i32, expected: i32 },
}

pub type Result<T> = std::result::Result<T, StoreError>;

/// Wraps a `Connection` in a `Mutex` so `ZoneStore` is `Sync` and can be
/// shared across tokio tasks via `Arc`. SQLite disallows concurrent use of
/// one Connection anyway, so a mutex is the honest primitive; per-zone
/// contention stays local because each zone has its own Connection.
pub struct ZoneStore {
    zone: Zone,
    conn: Mutex<Connection>,
}

impl ZoneStore {
    /// Open (or create) a zone store rooted at `state_path/<zone_dir>/db.sqlite`.
    pub fn open(state_path: &Path, zone: Zone) -> Result<Self> {
        let dir = state_path.join(zone.as_dir());
        std::fs::create_dir_all(&dir)?;
        let db_path = dir.join("db.sqlite");
        let conn = Connection::open(&db_path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        let store = Self {
            zone,
            conn: Mutex::new(conn),
        };
        store.migrate()?;
        tracing::info!(zone = zone.as_dir(), path = %db_path.display(), "zone store opened");
        Ok(store)
    }

    pub fn zone(&self) -> Zone {
        self.zone
    }

    fn migrate(&self) -> Result<()> {
        let conn = self.conn.lock().expect("zone store mutex poisoned");
        let user_version: i32 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap_or(0);

        if user_version == 0 {
            conn.execute_batch(
                r#"
                CREATE TABLE IF NOT EXISTS events (
                    id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_us   INTEGER NOT NULL,
                    kind    TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    refs    TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS events_ts_idx  ON events(ts_us);
                CREATE INDEX IF NOT EXISTS events_kind_idx ON events(kind);
                "#,
            )?;
            conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        } else if user_version != SCHEMA_VERSION {
            return Err(StoreError::UnsupportedSchema {
                found: user_version,
                expected: SCHEMA_VERSION,
            });
        }
        Ok(())
    }

    /// Insert an event; returns the assigned id.
    pub fn insert(&self, input: &EventInput) -> Result<i64> {
        let now_us = now_micros();
        let refs_json = serde_json::to_string(&input.refs)?;
        let payload_json = serde_json::to_string(&input.payload)?;
        let conn = self.conn.lock().expect("zone store mutex poisoned");
        conn.execute(
            "INSERT INTO events (ts_us, kind, payload, refs) VALUES (?1, ?2, ?3, ?4)",
            params![now_us, input.kind, payload_json, refs_json],
        )?;
        Ok(conn.last_insert_rowid())
    }

    pub fn get(&self, id: i64) -> Result<Option<Event>> {
        let conn = self.conn.lock().expect("zone store mutex poisoned");
        let row = conn
            .query_row(
                "SELECT id, ts_us, kind, payload, refs FROM events WHERE id = ?1",
                [id],
                Self::row_to_event(self.zone),
            )
            .optional()?;
        Ok(row)
    }

    pub fn recent(&self, limit: usize) -> Result<Vec<Event>> {
        let conn = self.conn.lock().expect("zone store mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT id, ts_us, kind, payload, refs FROM events ORDER BY id DESC LIMIT ?1",
        )?;
        let rows = stmt.query_map([limit as i64], Self::row_to_event(self.zone))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    /// Delete events older than `cutoff_us` (retention policy hook).
    pub fn prune_before(&self, cutoff_us: i64) -> Result<usize> {
        let conn = self.conn.lock().expect("zone store mutex poisoned");
        let n = conn.execute("DELETE FROM events WHERE ts_us < ?1", [cutoff_us])?;
        Ok(n)
    }

    fn row_to_event(zone: Zone) -> impl Fn(&rusqlite::Row<'_>) -> rusqlite::Result<Event> {
        move |row| {
            let payload_str: String = row.get(3)?;
            let refs_str: String = row.get(4)?;
            let payload = serde_json::from_str(&payload_str).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    3,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
            let refs: Vec<EventRef> = serde_json::from_str(&refs_str).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    4,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })?;
            Ok(Event {
                id: row.get(0)?,
                ts_us: row.get(1)?,
                zone,
                kind: row.get(2)?,
                payload,
                refs,
            })
        }
    }
}

/// Bundle of all zone stores. Runtime holds one of these.
pub struct Store {
    pub state_path: PathBuf,
    pub input_events: ZoneStore,
    pub system_obs: ZoneStore,
    pub personal_vault: ZoneStore,
    pub session_scratch: ZoneStore,
}

impl Store {
    pub fn open(state_path: &Path) -> Result<Self> {
        Ok(Self {
            state_path: state_path.to_path_buf(),
            input_events: ZoneStore::open(state_path, Zone::InputEvents)?,
            system_obs: ZoneStore::open(state_path, Zone::SystemObs)?,
            personal_vault: ZoneStore::open(state_path, Zone::PersonalVault)?,
            session_scratch: ZoneStore::open(state_path, Zone::SessionScratch)?,
        })
    }

    pub fn zone(&self, zone: Zone) -> &ZoneStore {
        match zone {
            Zone::InputEvents => &self.input_events,
            Zone::SystemObs => &self.system_obs,
            Zone::PersonalVault => &self.personal_vault,
            Zone::SessionScratch => &self.session_scratch,
        }
    }
}

fn now_micros() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn open_and_insert_roundtrip() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path()).unwrap();
        let id = store
            .input_events
            .insert(&EventInput {
                kind: "keystroke".into(),
                payload: json!({"key": "a"}),
                refs: vec![],
            })
            .unwrap();
        let ev = store.input_events.get(id).unwrap().unwrap();
        assert_eq!(ev.kind, "keystroke");
        assert_eq!(ev.zone, Zone::InputEvents);
        assert_eq!(ev.payload["key"], "a");
    }

    #[test]
    fn recent_orders_desc() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path()).unwrap();
        for i in 0..5 {
            store
                .session_scratch
                .insert(&EventInput {
                    kind: "thought".into(),
                    payload: json!({"idx": i}),
                    refs: vec![],
                })
                .unwrap();
        }
        let recent = store.session_scratch.recent(3).unwrap();
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].payload["idx"], 4);
    }

    #[test]
    fn cross_zone_ref_survives_roundtrip() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path()).unwrap();
        let src_id = store
            .input_events
            .insert(&EventInput {
                kind: "focus".into(),
                payload: json!({"window": "code"}),
                refs: vec![],
            })
            .unwrap();
        let id = store
            .session_scratch
            .insert(&EventInput {
                kind: "note".into(),
                payload: json!({"text": "user is coding"}),
                refs: vec![EventRef {
                    zone: Zone::InputEvents,
                    id: src_id,
                }],
            })
            .unwrap();
        let ev = store.session_scratch.get(id).unwrap().unwrap();
        assert_eq!(ev.refs.len(), 1);
        assert_eq!(ev.refs[0].zone, Zone::InputEvents);
        assert_eq!(ev.refs[0].id, src_id);
    }

    #[test]
    fn prune_before_deletes_old() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path()).unwrap();
        store
            .system_obs
            .insert(&EventInput {
                kind: "load".into(),
                payload: json!({"la1": 0.5}),
                refs: vec![],
            })
            .unwrap();
        // Cutoff far in the future prunes everything.
        let n = store.system_obs.prune_before(i64::MAX).unwrap();
        assert_eq!(n, 1);
        assert!(store.system_obs.recent(10).unwrap().is_empty());
    }
}
