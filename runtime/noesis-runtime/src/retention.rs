//! Retention scheduler.
//!
//! Ticks every N minutes and prunes each zone against two independent
//! bounds — age and byte-size — because either alone is insufficient:
//!
//! - Age-only lets a burst-heavy source (evdev flood, journal spam) blow
//!   the disk within the retention window.
//! - Byte-only lets a quiet zone accumulate rows forever until size finally
//!   catches up; we lose the "recent activity" property.
//!
//! Order per zone: age-prune first, then size-prune on whatever's left.
//! Byte-size uses `db_bytes()` (main + WAL + SHM footprint) and
//! `prune_oldest_until_below()` (incremental-vacuum + terminating VACUUM),
//! so the SQLite file actually shrinks — a bare DELETE would keep the
//! footprint constant.
//!
//! Per-zone defaults:
//!
//! - `input_events`   — age 24h,  size 200 MB. Rich but bulky.
//! - `system_obs`     — age 7d,   size 500 MB. Baseline drift wants a
//!    longer window; retention_stats also lives here so we need headroom.
//! - `personal_vault` — no bounds. The vault is the whole point.
//! - `session_scratch`— age 24h,  size 100 MB. Lens close nukes earlier;
//!    the age+size floor is the garbage-collection net.
//!
//! Emits one `retention_stats` event per tick into `system_obs` recording
//! `{age_pruned, size_pruned, bytes_before, bytes_after}` per zone.

use std::sync::Arc;
use std::time::Duration;

use noesis_schema::{EventInput, Zone};
use noesis_store::Store;
use serde_json::json;
use tracing::{info, warn};

#[derive(Clone, Copy, Debug)]
pub struct ZoneRetention {
    pub max_age: Option<Duration>,
    pub max_bytes: Option<u64>,
}

impl ZoneRetention {
    pub const fn none() -> Self {
        Self { max_age: None, max_bytes: None }
    }
}

pub struct RetentionConfig {
    pub tick: Duration,
    pub input_events: ZoneRetention,
    pub system_obs: ZoneRetention,
    pub personal_vault: ZoneRetention,
    pub session_scratch: ZoneRetention,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            tick: Duration::from_secs(15 * 60),
            input_events: ZoneRetention {
                max_age: Some(Duration::from_secs(24 * 3600)),
                max_bytes: Some(200 * 1024 * 1024),
            },
            system_obs: ZoneRetention {
                max_age: Some(Duration::from_secs(7 * 24 * 3600)),
                max_bytes: Some(500 * 1024 * 1024),
            },
            personal_vault: ZoneRetention::none(),
            session_scratch: ZoneRetention {
                max_age: Some(Duration::from_secs(24 * 3600)),
                max_bytes: Some(100 * 1024 * 1024),
            },
        }
    }
}

pub async fn run(store: Arc<Store>, cfg: RetentionConfig) -> anyhow::Result<()> {
    let mut ticker = tokio::time::interval(cfg.tick);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    // Skip the initial immediate tick — no events to prune at t=0.
    ticker.tick().await;
    loop {
        ticker.tick().await;
        let now_us = now_micros();
        let mut per_zone = serde_json::Map::new();
        for (zone, retention) in [
            (Zone::InputEvents, cfg.input_events),
            (Zone::SystemObs, cfg.system_obs),
            (Zone::PersonalVault, cfg.personal_vault),
            (Zone::SessionScratch, cfg.session_scratch),
        ] {
            per_zone.insert(zone.as_dir().into(), sweep_zone(&store, zone, retention, now_us));
        }
        let input = EventInput {
            kind: "retention_stats".into(),
            payload: json!({ "cutoff_us": now_us, "zones": per_zone }),
            refs: vec![],
        };
        if let Err(e) = store.system_obs.insert(&input) {
            warn!(error = %e, "retention_stats insert failed");
        }
    }
}

fn sweep_zone(
    store: &Store,
    zone: Zone,
    retention: ZoneRetention,
    now_us: i64,
) -> serde_json::Value {
    let z = store.zone(zone);

    let bytes_before = match z.db_bytes() {
        Ok(b) => Some(b),
        Err(e) => {
            warn!(zone = zone.as_dir(), error = %e, "db_bytes failed");
            None
        }
    };

    let age_pruned = match retention.max_age {
        Some(max_age) => {
            let cutoff = now_us.saturating_sub(max_age.as_micros() as i64);
            match z.prune_before(cutoff) {
                Ok(n) => {
                    if n > 0 {
                        info!(zone = zone.as_dir(), pruned = n, mode = "age", "retention prune");
                    }
                    json!(n)
                }
                Err(e) => {
                    warn!(zone = zone.as_dir(), error = %e, "age prune failed");
                    json!("error")
                }
            }
        }
        None => json!(null),
    };

    let size_pruned = match retention.max_bytes {
        Some(max_bytes) => match z.prune_oldest_until_below(max_bytes) {
            Ok(n) => {
                if n > 0 {
                    info!(zone = zone.as_dir(), pruned = n, mode = "size", "retention prune");
                }
                json!(n)
            }
            Err(e) => {
                warn!(zone = zone.as_dir(), error = %e, "size prune failed");
                json!("error")
            }
        },
        None => json!(null),
    };

    let bytes_after = z.db_bytes().ok();

    json!({
        "age_pruned": age_pruned,
        "size_pruned": size_pruned,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "max_age_secs": retention.max_age.map(|d| d.as_secs()),
        "max_bytes": retention.max_bytes,
    })
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
    use tempfile::TempDir;

    #[test]
    fn prune_removes_only_older_than_retention() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path()).unwrap();
        store
            .input_events
            .insert(&EventInput {
                kind: "k".into(),
                payload: json!({}),
                refs: vec![],
            })
            .unwrap();
        let cutoff_past = 0i64;
        let removed = store.input_events.prune_before(cutoff_past).unwrap();
        assert_eq!(removed, 0);
        assert_eq!(store.input_events.recent(10).unwrap().len(), 1);

        let removed = store.input_events.prune_before(i64::MAX).unwrap();
        assert_eq!(removed, 1);
    }

    #[test]
    fn default_config_has_expected_retention() {
        let cfg = RetentionConfig::default();
        assert_eq!(cfg.input_events.max_age.unwrap().as_secs(), 86400);
        assert_eq!(cfg.input_events.max_bytes.unwrap(), 200 * 1024 * 1024);
        assert_eq!(cfg.system_obs.max_age.unwrap().as_secs(), 604800);
        assert_eq!(cfg.system_obs.max_bytes.unwrap(), 500 * 1024 * 1024);
        assert!(cfg.personal_vault.max_age.is_none());
        assert!(cfg.personal_vault.max_bytes.is_none());
        assert_eq!(cfg.session_scratch.max_age.unwrap().as_secs(), 86400);
        assert_eq!(cfg.session_scratch.max_bytes.unwrap(), 100 * 1024 * 1024);
    }

    #[test]
    fn sweep_zone_produces_bytes_and_prune_counts() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path()).unwrap();

        // Populate ~4 KiB rows so size-prune has something to bite.
        let payload = json!({ "blob": "x".repeat(4 * 1024) });
        for _ in 0..64 {
            store
                .session_scratch
                .insert(&EventInput {
                    kind: "k".into(),
                    payload: payload.clone(),
                    refs: vec![],
                })
                .unwrap();
        }
        let before = store.session_scratch.db_bytes().unwrap();
        assert!(before > 0);

        let retention = ZoneRetention {
            max_age: None,
            max_bytes: Some(before / 2),
        };
        let result = sweep_zone(&store, Zone::SessionScratch, retention, i64::MAX);
        let obj = result.as_object().unwrap();
        assert_eq!(obj["age_pruned"], json!(null));
        assert!(obj["size_pruned"].as_u64().unwrap() > 0);
        assert!(obj["bytes_after"].as_u64().unwrap() < obj["bytes_before"].as_u64().unwrap());
    }
}
