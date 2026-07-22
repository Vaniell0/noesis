//! /dev/input/event* raw kernel input stream (L0 collector).
//!
//! Each input device — keyboard, mouse, touchpad, virtual input, etc.
//! — gets its own blocking task that pumps `input_event` records into
//! the `input_events` zone. We store the raw kernel codes (type/code/
//! value + device name) without translating to KEY_A/BTN_LEFT/etc.:
//! collect-wide-interpret-later. Later a query-side helper can lookup
//! symbolic names from the `evdev` crate's tables.
//!
//! Requires membership in the `input` group (default on modern desktop
//! distros). If a device open fails EACCES, we log once and skip it —
//! we do NOT retry per device, since group membership is a config-time
//! decision. Devices that appear after startup are missed until next
//! restart; hot-plug discovery is a Phase B follow-up.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use noesis_schema::EventInput;
use noesis_store::Store;
use serde_json::json;
use tracing::{debug, info, warn};

pub struct EvdevConfig {
    pub device_glob: String,
    pub rescan_interval: Duration,
}

impl Default for EvdevConfig {
    fn default() -> Self {
        Self {
            device_glob: "/dev/input/event*".into(),
            rescan_interval: Duration::from_secs(300),
        }
    }
}

pub async fn run(store: Arc<Store>, cfg: EvdevConfig) -> anyhow::Result<()> {
    let mut ticker = tokio::time::interval(cfg.rescan_interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut open_devices: Vec<PathBuf> = Vec::new();

    loop {
        let paths = enumerate(&cfg.device_glob);
        for path in paths {
            if open_devices.contains(&path) {
                continue;
            }
            let store = Arc::clone(&store);
            let p = path.clone();
            let spawned = tokio::task::spawn_blocking(move || pump_device(&p, &store));
            open_devices.push(path);
            tokio::spawn(async move {
                match spawned.await {
                    Ok(Ok(())) => info!("evdev device task ended cleanly"),
                    Ok(Err(e)) => warn!(error = %e, "evdev device task errored"),
                    Err(e) => warn!(error = %e, "evdev device task join error"),
                }
            });
        }
        ticker.tick().await;
    }
}

fn enumerate(glob_pat: &str) -> Vec<PathBuf> {
    let Some(dir) = std::path::Path::new(glob_pat).parent() else {
        return vec![];
    };
    let Some(prefix) = std::path::Path::new(glob_pat)
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|s| s.strip_suffix('*'))
    else {
        return vec![];
    };
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_str) = name.to_str() else { continue };
        if name_str.starts_with(prefix) {
            out.push(entry.path());
        }
    }
    out.sort();
    out
}

fn pump_device(path: &std::path::Path, store: &Arc<Store>) -> anyhow::Result<()> {
    let mut dev = match evdev::Device::open(path) {
        Ok(d) => d,
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            warn!(path = %path.display(), "evdev open EACCES — not in input group?");
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };
    let device_name = dev
        .name()
        .map(|s| s.to_string())
        .unwrap_or_else(|| path.display().to_string());
    info!(device = %device_name, path = %path.display(), "evdev device opened");

    loop {
        let events = match dev.fetch_events() {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => continue,
            Err(e) => return Err(e.into()),
        };
        for ev in events {
            let ts = ev.timestamp();
            let elapsed = ts
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default();
            let payload = json!({
                "device": device_name,
                "path": path.display().to_string(),
                "kernel_ts_us": elapsed.as_micros() as u64,
                "type": ev.event_type().0,
                "code": ev.code(),
                "value": ev.value(),
            });
            let input = EventInput {
                kind: "input_event".into(),
                payload,
                refs: vec![],
            };
            if let Err(e) = store.input_events.insert(&input) {
                warn!(error = %e, "evdev store insert failed");
            } else {
                debug!(device = %device_name, ty = ev.event_type().0, code = ev.code(),
                       value = ev.value(), "evdev event inserted");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enumerate_finds_glob_matches() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("event0"), "").unwrap();
        std::fs::write(dir.path().join("event1"), "").unwrap();
        std::fs::write(dir.path().join("mouse0"), "").unwrap();
        let pat = dir.path().join("event*").display().to_string();
        let mut out = enumerate(&pat);
        out.sort();
        assert_eq!(out.len(), 2);
        assert!(out[0].ends_with("event0"));
        assert!(out[1].ends_with("event1"));
    }

    #[test]
    fn enumerate_returns_empty_on_missing_dir() {
        let out = enumerate("/nonexistent-xyz/event*");
        assert!(out.is_empty());
    }
}
