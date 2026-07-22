//! noesis-schema — core types shared by store, runtime, and eventual host tools.
//!
//! Four zones, per B0 pivot:
//!   * `InputEvents`   — user keyboard/mouse/window activity (absorbs key-daemon)
//!   * `SystemObs`     — periodic host observations (procs, load, battery, net)
//!   * `PersonalVault` — long-term personal knowledge (retrieval index, notes)
//!   * `SessionScratch`— ephemeral working memory tied to a supervisor session
//!
//! Zones are addressed uniformly by `EventRef { zone, id }`.

use serde::{Deserialize, Serialize};

pub type EventId = i64;
pub type UnixMicros = i64;

/// Memory zone identifier. Stable numeric encoding (used by the store's
/// on-disk layout), so **never renumber**; append new variants only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum Zone {
    InputEvents = 0,
    SystemObs = 1,
    PersonalVault = 2,
    SessionScratch = 3,
}

impl Zone {
    pub const ALL: [Zone; 4] = [
        Zone::InputEvents,
        Zone::SystemObs,
        Zone::PersonalVault,
        Zone::SessionScratch,
    ];

    pub fn as_dir(self) -> &'static str {
        match self {
            Zone::InputEvents => "input_events",
            Zone::SystemObs => "system_obs",
            Zone::PersonalVault => "personal_vault",
            Zone::SessionScratch => "session_scratch",
        }
    }

    pub fn from_dir(s: &str) -> Option<Self> {
        Zone::ALL.into_iter().find(|z| z.as_dir() == s)
    }
}

/// Cross-zone reference. Runtime uses these to thread context across
/// heterogeneous events without collapsing zones into a single table.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EventRef {
    pub zone: Zone,
    pub id: EventId,
}

/// An event as it appears at the schema boundary — untyped payload, kind tag,
/// microsecond timestamp. The zone-specific interpretation of `payload` lives
/// in the module that owns the zone (input_events collector, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    pub ts_us: UnixMicros,
    pub zone: Zone,
    pub kind: String,
    pub payload: serde_json::Value,
    #[serde(default)]
    pub refs: Vec<EventRef>,
}

/// Insertion payload — id/ts filled in by the store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventInput {
    pub kind: String,
    pub payload: serde_json::Value,
    #[serde(default)]
    pub refs: Vec<EventRef>,
}

#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    #[error("unknown zone dir: {0}")]
    UnknownZone(String),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zone_dir_roundtrip() {
        for z in Zone::ALL {
            assert_eq!(Zone::from_dir(z.as_dir()), Some(z));
        }
    }

    #[test]
    fn zone_repr_is_stable() {
        assert_eq!(Zone::InputEvents as u8, 0);
        assert_eq!(Zone::SystemObs as u8, 1);
        assert_eq!(Zone::PersonalVault as u8, 2);
        assert_eq!(Zone::SessionScratch as u8, 3);
    }
}
