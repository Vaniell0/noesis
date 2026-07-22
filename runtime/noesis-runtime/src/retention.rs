//! Retention scheduler.
//!
//! Ticks every N minutes and prunes each zone below its configured age.
//! Rationale: evdev + journal generate hundreds of MB/day. Without a
//! sweeper, the disk fills. Prune policy per zone:
//!
//! - `input_events`  — 24h. Rich but bulky; correlation window is minutes.
//! - `system_obs`    — 7d.  Baseline drift needs a longer window.
//! - `personal_vault`— never. That's the whole point of the vault.
//! - `session_scratch` — 24h. Lens close will nuke earlier when we get to
//!    lifecycle wiring; the 24h floor is just a garbage-collection net.
//!
//! Emits one `retention_stats` event per tick into `system_obs` recording
//! how many rows fell out of each zone, so downstream can plot retention
//! pressure over time.

use std::sync::Arc;
use std::time::Duration;

use noesis_schema::{EventInput, Zone};
use noesis_store::Store;
use serde_json::json;
use tracing::{info, warn};

pub struct RetentionConfig {
    pub tick: Duration,
    pub input_events: Option<Duration>,
    pub system_obs: Option<Duration>,
    pub personal_vault: Option<Duration>,
    pub session_scratch: Option<Duration>,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            tick: Duration::from_secs(15 * 60),
            input_events: Some(Duration::from_secs(24 * 3600)),
            system_obs: Some(Duration::from_secs(7 * 24 * 3600)),
            personal_vault: None,
            session_scratch: Some(Duration::from_secs(24 * 3600)),
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
        let mut removed = serde_json::Map::new();
        for (zone, retention) in [
            (Zone::InputEvents, cfg.input_events),
            (Zone::SystemObs, cfg.system_obs),
            (Zone::PersonalVault, cfg.personal_vault),
            (Zone::SessionScratch, cfg.session_scratch),
        ] {
            let Some(retention) = retention else {
                removed.insert(zone.as_dir().into(), json!(null));
                continue;
            };
            let cutoff = now_us.saturating_sub(retention.as_micros() as i64);
            match store.zone(zone).prune_before(cutoff) {
                Ok(n) => {
                    removed.insert(zone.as_dir().into(), json!(n));
                    if n > 0 {
                        info!(zone = zone.as_dir(), pruned = n, "retention prune");
                    }
                }
                Err(e) => {
                    warn!(zone = zone.as_dir(), error = %e, "retention prune failed");
                    removed.insert(zone.as_dir().into(), json!("error"));
                }
            }
        }
        let input = EventInput {
            kind: "retention_stats".into(),
            payload: json!({ "cutoff_us": now_us, "pruned": removed }),
            refs: vec![],
        };
        if let Err(e) = store.system_obs.insert(&input) {
            warn!(error = %e, "retention_stats insert failed");
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
    use tempfile::TempDir;

    #[test]
    fn prune_removes_only_older_than_retention() {
        let dir = TempDir::new().unwrap();
        let store = Store::open(dir.path()).unwrap();
        // Insert a row, then prune with a zero-retention cutoff — the row is
        // 'now' so it survives; a cutoff of i64::MAX prunes it. This exercises
        // the same code path retention.rs uses without needing a real timer.
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
        assert_eq!(cfg.input_events.unwrap().as_secs(), 86400);
        assert_eq!(cfg.system_obs.unwrap().as_secs(), 604800);
        assert!(cfg.personal_vault.is_none());
        assert_eq!(cfg.session_scratch.unwrap().as_secs(), 86400);
    }
}
