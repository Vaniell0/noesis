//! systemd journal reader (L1 collector).
//!
//! Streams `journalctl -o json --follow --since=now --no-pager` and emits
//! one event per journal line into the `system_obs` zone. We don't
//! interpret fields: layered-collectors rule is collect wide, defer
//! semantic interpretation. Each event carries the entire JSON record
//! from journald so we can re-derive anything later.
//!
//! Failure mode: if `journalctl` exits (crash, socket issue, package
//! missing), we back off exponentially and restart. If the binary is
//! absent, we log once and stop retrying — no point burning CPU on a
//! host without systemd.

use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use noesis_schema::EventInput;
use noesis_store::Store;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tracing::{debug, info, warn};

pub struct JournalConfig {
    pub binary: String,
    pub extra_args: Vec<String>,
    pub max_backoff: Duration,
}

impl Default for JournalConfig {
    fn default() -> Self {
        Self {
            binary: "journalctl".into(),
            extra_args: vec![],
            max_backoff: Duration::from_secs(60),
        }
    }
}

pub async fn run(store: Arc<Store>, cfg: JournalConfig) -> anyhow::Result<()> {
    let mut backoff = Duration::from_secs(1);
    loop {
        match stream_once(&store, &cfg).await {
            Ok(()) => {
                info!("journalctl exited cleanly; restarting");
                backoff = Duration::from_secs(1);
            }
            Err(StreamError::NotFound) => {
                warn!(binary = %cfg.binary, "journalctl not found; disabling journal collector");
                return Ok(());
            }
            Err(StreamError::Other(e)) => {
                warn!(error = %e, backoff_ms = backoff.as_millis() as u64,
                      "journal stream failed; backing off");
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(cfg.max_backoff);
            }
        }
    }
}

enum StreamError {
    NotFound,
    Other(anyhow::Error),
}

async fn stream_once(store: &Arc<Store>, cfg: &JournalConfig) -> Result<(), StreamError> {
    let mut cmd = Command::new(&cfg.binary);
    cmd.arg("-o")
        .arg("json")
        .arg("--follow")
        .arg("--since=now")
        .arg("--no-pager")
        .args(&cfg.extra_args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(StreamError::NotFound),
        Err(e) => return Err(StreamError::Other(e.into())),
    };

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| StreamError::Other(anyhow::anyhow!("journalctl stdout not captured")))?;
    let stderr = child.stderr.take();

    if let Some(stderr) = stderr {
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                warn!(source = "journalctl.stderr", "{line}");
            }
        });
    }

    let mut lines = BufReader::new(stdout).lines();
    let mut count: u64 = 0;
    while let Some(line) = lines
        .next_line()
        .await
        .map_err(|e| StreamError::Other(e.into()))?
    {
        let payload = match parse_line(&line) {
            Some(v) => v,
            None => continue,
        };
        let input = EventInput {
            kind: "journal_line".into(),
            payload,
            refs: vec![],
        };
        match store.system_obs.insert(&input) {
            Ok(id) => debug!(id, "journal event inserted"),
            Err(e) => warn!(error = %e, "journal store insert failed"),
        }
        count += 1;
    }

    let status = child
        .wait()
        .await
        .map_err(|e| StreamError::Other(e.into()))?;
    info!(count, status = ?status, "journalctl stream ended");
    Ok(())
}

fn parse_line(line: &str) -> Option<Value> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    match serde_json::from_str::<Value>(trimmed) {
        Ok(v) => Some(v),
        Err(e) => {
            debug!(error = %e, "skip non-json journal line");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_line_accepts_json_object() {
        let line = r#"{"MESSAGE":"hello","PRIORITY":"6","_SYSTEMD_UNIT":"foo.service"}"#;
        let v = parse_line(line).unwrap();
        assert_eq!(v["MESSAGE"], "hello");
        assert_eq!(v["_SYSTEMD_UNIT"], "foo.service");
    }

    #[test]
    fn parse_line_skips_empty_and_garbage() {
        assert!(parse_line("").is_none());
        assert!(parse_line("   ").is_none());
        assert!(parse_line("not-json-at-all").is_none());
    }
}
