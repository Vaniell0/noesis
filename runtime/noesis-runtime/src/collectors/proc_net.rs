//! /proc/net/dev interface counters (L0 collector).
//!
//! Reads the per-interface byte/packet/error counters that the kernel
//! exposes and emits one snapshot per tick. Same collect-raw-interpret-later
//! discipline as `proc_stat` — deltas are downstream.
//!
//! The `lo` (loopback) interface is included; downstream filtering is
//! cheaper than re-collecting if we later want it.

use std::sync::Arc;
use std::time::Duration;

use noesis_schema::EventInput;
use noesis_store::Store;
use serde_json::{json, Map, Value};
use tokio::time::interval;
use tracing::{debug, warn};

pub struct ProcNetConfig {
    pub tick: Duration,
}

impl Default for ProcNetConfig {
    fn default() -> Self {
        Self {
            tick: Duration::from_secs(15),
        }
    }
}

pub async fn run(store: Arc<Store>, cfg: ProcNetConfig) -> anyhow::Result<()> {
    let mut ticker = interval(cfg.tick);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        ticker.tick().await;
        let payload = match tokio::task::spawn_blocking(sample).await {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                warn!(error = %e, "proc_net sample failed");
                continue;
            }
            Err(e) => {
                warn!(error = %e, "proc_net blocking task joined with error");
                continue;
            }
        };
        let input = EventInput {
            kind: "net_counters".into(),
            payload,
            refs: vec![],
        };
        match store.system_obs.insert(&input) {
            Ok(id) => debug!(id, "proc_net event inserted"),
            Err(e) => warn!(error = %e, "proc_net store insert failed"),
        }
    }
}

fn sample() -> anyhow::Result<Value> {
    let text = std::fs::read_to_string("/proc/net/dev")?;
    Ok(parse(&text))
}

/// Fields per interface, in the order /proc/net/dev exposes them:
/// receive: bytes packets errs drop fifo frame compressed multicast
/// transmit: bytes packets errs drop fifo colls carrier compressed
const RX_FIELDS: &[&str] = &[
    "rx_bytes",
    "rx_packets",
    "rx_errs",
    "rx_drop",
    "rx_fifo",
    "rx_frame",
    "rx_compressed",
    "rx_multicast",
];
const TX_FIELDS: &[&str] = &[
    "tx_bytes",
    "tx_packets",
    "tx_errs",
    "tx_drop",
    "tx_fifo",
    "tx_colls",
    "tx_carrier",
    "tx_compressed",
];

fn parse(text: &str) -> Value {
    let mut ifaces = Map::new();
    for line in text.lines().skip(2) {
        let Some((name, rest)) = line.split_once(':') else {
            continue;
        };
        let name = name.trim();
        let vals: Vec<u64> = rest.split_whitespace().filter_map(|s| s.parse().ok()).collect();
        let mut entry = Map::new();
        for (i, key) in RX_FIELDS.iter().enumerate() {
            entry.insert((*key).into(), json!(vals.get(i).copied().unwrap_or(0)));
        }
        for (i, key) in TX_FIELDS.iter().enumerate() {
            entry.insert(
                (*key).into(),
                json!(vals.get(RX_FIELDS.len() + i).copied().unwrap_or(0)),
            );
        }
        ifaces.insert(name.into(), Value::Object(entry));
    }
    json!({ "interfaces": ifaces })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_two_interfaces() {
        let text = "\
Inter-|   Receive                                                |  Transmit
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed
    lo:  12345      67    0    0    0     0          0        0    12345      67    0    0    0     0       0          0
  eth0: 987654    3210    1    2    3     4          5        6   123456      78    9   10   11    12      13         14
";
        let v = parse(text);
        let ifaces = v["interfaces"].as_object().unwrap();
        assert_eq!(ifaces["lo"]["rx_bytes"], 12345);
        assert_eq!(ifaces["eth0"]["rx_bytes"], 987654);
        assert_eq!(ifaces["eth0"]["tx_bytes"], 123456);
        assert_eq!(ifaces["eth0"]["tx_carrier"], 13);
    }

    #[test]
    fn parse_skips_missing_columns() {
        let text = "\
Inter-|   Receive
 face |bytes
   foo: 42
";
        let v = parse(text);
        assert_eq!(v["interfaces"]["foo"]["rx_bytes"], 42);
        assert_eq!(v["interfaces"]["foo"]["tx_bytes"], 0);
    }
}
