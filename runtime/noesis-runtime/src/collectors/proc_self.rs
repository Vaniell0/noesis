//! /proc/self footprint (L0 collector).
//!
//! Measures the noesis-runtime process itself so A0.3 idle-24h can
//! plot RSS / VmPeak / thread count over wall time without a separate
//! `pidstat` harness. Same collect-raw-interpret-later discipline as
//! the rest of L0: emit the numbers, let downstream compute deltas
//! and identify leaks / drift.

use std::sync::Arc;
use std::time::Duration;

use noesis_schema::EventInput;
use noesis_store::Store;
use serde_json::{json, Map, Value};
use tokio::time::interval;
use tracing::{debug, warn};

pub struct ProcSelfConfig {
    pub tick: Duration,
}

impl Default for ProcSelfConfig {
    fn default() -> Self {
        Self {
            tick: Duration::from_secs(60),
        }
    }
}

pub async fn run(store: Arc<Store>, cfg: ProcSelfConfig) -> anyhow::Result<()> {
    let mut ticker = interval(cfg.tick);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        let payload = match tokio::task::spawn_blocking(sample).await {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                warn!(error = %e, "proc_self sample failed");
                continue;
            }
            Err(e) => {
                warn!(error = %e, "proc_self blocking task joined with error");
                continue;
            }
        };
        let input = EventInput {
            kind: "runtime_footprint".into(),
            payload,
            refs: vec![],
        };
        match store.system_obs.insert(&input) {
            Ok(id) => debug!(id, "proc_self event inserted"),
            Err(e) => warn!(error = %e, "proc_self store insert failed"),
        }
    }
}

fn sample() -> anyhow::Result<Value> {
    let status = std::fs::read_to_string("/proc/self/status")?;
    let statm = std::fs::read_to_string("/proc/self/statm")?;
    Ok(json!({
        "status": parse_status(&status),
        "statm": parse_statm(&statm),
    }))
}

/// /proc/self/status is a `Key:\tvalue` table. We keep the numeric memory
/// / thread lines and drop the noisy string ones; downstream can add more.
const STATUS_KEYS: &[&str] = &[
    "VmPeak",
    "VmSize",
    "VmHWM",
    "VmRSS",
    "RssAnon",
    "RssFile",
    "RssShmem",
    "VmData",
    "VmStk",
    "VmExe",
    "VmLib",
    "VmSwap",
    "Threads",
    "voluntary_ctxt_switches",
    "nonvoluntary_ctxt_switches",
];

fn parse_status(text: &str) -> Value {
    let mut out = Map::new();
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        if !STATUS_KEYS.contains(&key) {
            continue;
        }
        // First whitespace-separated number wins; units (e.g. "kB") ignored.
        let val = rest
            .split_whitespace()
            .next()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);
        out.insert(key.to_string(), json!(val));
    }
    Value::Object(out)
}

/// /proc/self/statm layout (pages): size resident shared text lib data dt
const STATM_FIELDS: &[&str] = &["size", "resident", "shared", "text", "lib", "data", "dt"];

fn parse_statm(text: &str) -> Value {
    let mut out = Map::new();
    let vals: Vec<u64> = text
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    for (i, key) in STATM_FIELDS.iter().enumerate() {
        out.insert((*key).into(), json!(vals.get(i).copied().unwrap_or(0)));
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_status_picks_memory_keys() {
        let text = "\
Name:\tnoesis-runtime
Umask:\t0022
State:\tR (running)
VmPeak:\t   12345 kB
VmSize:\t    9876 kB
VmRSS:\t     420 kB
Threads:\t8
voluntary_ctxt_switches:\t42
";
        let v = parse_status(text);
        assert_eq!(v["VmPeak"], 12345);
        assert_eq!(v["VmSize"], 9876);
        assert_eq!(v["VmRSS"], 420);
        assert_eq!(v["Threads"], 8);
        assert_eq!(v["voluntary_ctxt_switches"], 42);
        assert!(v.get("Name").is_none());
        assert!(v.get("State").is_none());
    }

    #[test]
    fn parse_statm_seven_fields() {
        let v = parse_statm("100 50 10 5 0 40 0\n");
        assert_eq!(v["size"], 100);
        assert_eq!(v["resident"], 50);
        assert_eq!(v["shared"], 10);
        assert_eq!(v["text"], 5);
        assert_eq!(v["data"], 40);
    }

    #[test]
    fn parse_statm_short_defaults_zero() {
        let v = parse_statm("100 50 10");
        assert_eq!(v["size"], 100);
        assert_eq!(v["data"], 0);
        assert_eq!(v["dt"], 0);
    }
}
