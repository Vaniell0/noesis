//! system_obs collector — periodic host observations from /proc.
//!
//! Emits one event per tick with load, mem, and uptime snapshots. Cheap
//! enough (a few reads of small procfs files) that a 30-second tick is
//! fine even under memory-constrained conditions.
//!
//! Deliberately minimal: no CPU-per-core breakdown, no per-process listing —
//! those are follow-up events that a heavier collector (or a separate
//! `system_obs_deep` module) can add without touching this file.

use std::sync::Arc;
use std::time::Duration;

use noesis_schema::EventInput;
use noesis_store::Store;
use serde_json::json;
use tokio::time::interval;
use tracing::{debug, warn};

pub struct SystemObsConfig {
    pub tick: Duration,
}

impl Default for SystemObsConfig {
    fn default() -> Self {
        Self {
            tick: Duration::from_secs(30),
        }
    }
}

pub async fn run(store: Arc<Store>, cfg: SystemObsConfig) -> anyhow::Result<()> {
    let mut ticker = interval(cfg.tick);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        let snapshot = match tokio::task::spawn_blocking(sample).await {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                warn!(error = %e, "system_obs sample failed");
                continue;
            }
            Err(e) => {
                warn!(error = %e, "system_obs blocking task joined with error");
                continue;
            }
        };
        let input = EventInput {
            kind: "host_snapshot".into(),
            payload: snapshot,
            refs: vec![],
        };
        match store.system_obs.insert(&input) {
            Ok(id) => debug!(id, "system_obs event inserted"),
            Err(e) => warn!(error = %e, "system_obs store insert failed"),
        }
    }
}

fn sample() -> anyhow::Result<serde_json::Value> {
    let loadavg = std::fs::read_to_string("/proc/loadavg")?;
    let uptime = std::fs::read_to_string("/proc/uptime")?;
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;

    let (la1, la5, la15) = parse_loadavg(&loadavg).unwrap_or((0.0, 0.0, 0.0));
    let up_s = uptime
        .split_whitespace()
        .next()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    let (mem_total_kb, mem_avail_kb) = parse_meminfo(&meminfo);

    Ok(json!({
        "loadavg": { "one": la1, "five": la5, "fifteen": la15 },
        "uptime_s": up_s,
        "mem_total_kb": mem_total_kb,
        "mem_available_kb": mem_avail_kb,
    }))
}

fn parse_loadavg(s: &str) -> Option<(f64, f64, f64)> {
    let mut it = s.split_whitespace();
    Some((
        it.next()?.parse().ok()?,
        it.next()?.parse().ok()?,
        it.next()?.parse().ok()?,
    ))
}

fn parse_meminfo(s: &str) -> (u64, u64) {
    let mut total = 0u64;
    let mut avail = 0u64;
    for line in s.lines() {
        let key_val = line.split_once(':');
        let Some((key, val)) = key_val else { continue };
        let num: u64 = val
            .split_whitespace()
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        match key {
            "MemTotal" => total = num,
            "MemAvailable" => avail = num,
            _ => {}
        }
    }
    (total, avail)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loadavg_parses() {
        assert_eq!(
            parse_loadavg("0.42 0.31 0.20 2/512 12345"),
            Some((0.42, 0.31, 0.20))
        );
    }

    #[test]
    fn meminfo_parses() {
        let s = "MemTotal:       16000000 kB\nMemFree:         2000000 kB\nMemAvailable:    8000000 kB\n";
        assert_eq!(parse_meminfo(s), (16_000_000, 8_000_000));
    }
}
