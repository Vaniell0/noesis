//! /proc/stat CPU tick counters (L0 collector).
//!
//! Emits one event per tick with the current cumulative tick counters for
//! the aggregate `cpu` line and each per-core `cpuN` line. Raw counters are
//! stored; delta-per-interval is a downstream computation (adjacent rows in
//! the store carry ts_us, so `Δticks / Δt` is trivial from any query side).
//!
//! Rationale for storing raw not deltas: matches the layered-collectors
//! rule — collect wide, interpret later. If we later decide the interpreter
//! wants deltas, we compute them in a query; if we stored deltas, we'd lose
//! the raw counters and couldn't reconstruct.

use std::sync::Arc;
use std::time::Duration;

use noesis_schema::EventInput;
use noesis_store::Store;
use serde_json::{json, Value};
use tokio::time::interval;
use tracing::{debug, warn};

pub struct ProcStatConfig {
    pub tick: Duration,
}

impl Default for ProcStatConfig {
    fn default() -> Self {
        Self {
            tick: Duration::from_secs(10),
        }
    }
}

pub async fn run(store: Arc<Store>, cfg: ProcStatConfig) -> anyhow::Result<()> {
    let mut ticker = interval(cfg.tick);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        let payload = match tokio::task::spawn_blocking(sample).await {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                warn!(error = %e, "proc_stat sample failed");
                continue;
            }
            Err(e) => {
                warn!(error = %e, "proc_stat blocking task joined with error");
                continue;
            }
        };
        let input = EventInput {
            kind: "cpu_ticks".into(),
            payload,
            refs: vec![],
        };
        match store.system_obs.insert(&input) {
            Ok(id) => debug!(id, "proc_stat event inserted"),
            Err(e) => warn!(error = %e, "proc_stat store insert failed"),
        }
    }
}

fn sample() -> anyhow::Result<Value> {
    let text = std::fs::read_to_string("/proc/stat")?;
    let mut aggregate = Value::Null;
    let mut cores = Vec::new();
    for line in text.lines() {
        if !line.starts_with("cpu") {
            break;
        }
        let mut it = line.split_whitespace();
        let head = it.next().unwrap_or("");
        let vals: Vec<u64> = it.filter_map(|s| s.parse().ok()).collect();
        let entry = ticks_to_json(&vals);
        if head == "cpu" {
            aggregate = entry;
        } else {
            cores.push(entry);
        }
    }
    Ok(json!({
        "aggregate": aggregate,
        "per_core": cores,
    }))
}

fn ticks_to_json(vals: &[u64]) -> Value {
    // Field order per Documentation/filesystems/proc.rst; missing tail
    // fields silently degrade to 0.
    let get = |i: usize| vals.get(i).copied().unwrap_or(0);
    json!({
        "user": get(0),
        "nice": get(1),
        "system": get(2),
        "idle": get(3),
        "iowait": get(4),
        "irq": get(5),
        "softirq": get(6),
        "steal": get(7),
        "guest": get(8),
        "guest_nice": get(9),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_to_json_populates_all_fields() {
        let v = ticks_to_json(&[100, 5, 40, 800, 3, 1, 2, 0, 0, 0]);
        assert_eq!(v["user"], 100);
        assert_eq!(v["idle"], 800);
        assert_eq!(v["steal"], 0);
    }

    #[test]
    fn ticks_to_json_handles_short_input() {
        let v = ticks_to_json(&[100, 5, 40, 800]);
        assert_eq!(v["idle"], 800);
        assert_eq!(v["iowait"], 0);
        assert_eq!(v["guest_nice"], 0);
    }
}
