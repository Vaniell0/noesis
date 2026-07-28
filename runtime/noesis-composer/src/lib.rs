//! noesis-composer — builds the substrate prompt prefix from a preamble
//! template and live retrieval against the event store.
//!
//! # Role in the context pipeline
//!
//! ```text
//! HTTP request
//!   └─ Composer::compose(query, store, budget)
//!         ├─ preamble  → TransformConfig::system_preamble  (static for session)
//!         └─ snippet   → RetrievalSlot::Snippet            (per-request)
//!   └─ ContextTransform::build_prompt(turns, snippet)
//!   └─ substrate inference
//! ```
//!
//! # Preamble template DSL
//!
//! A plain string with `{variable}` substitutions. Known variables:
//!   `{date}`     — current date  (YYYY-MM-DD, UTC)
//!   `{time}`     — current time  (HH:MM:SS, UTC)
//!   `{hostname}` — OS hostname
//!
//! Unknown `{…}` patterns are left as-is.
//!
//! # Retrieval
//!
//! Scans the most recent `scan_limit` events from each configured zone,
//! scores each by keyword overlap with the query, takes the top
//! `retrieval_top_k` by score × recency, and formats them as a plain-text
//! block capped at `budget_bytes`.

pub mod preamble;
pub mod retrieval;

use std::sync::Arc;

use noesis_schema::Zone;
use noesis_store::Store;
use serde::Deserialize;

pub use preamble::render_preamble;
pub use retrieval::retrieve;

/// Configuration for [`Composer`]. Deserialises from `[composer]` in
/// `runtime.toml`, or constructed programmatically.
#[derive(Debug, Clone, Deserialize)]
pub struct ComposerConfig {
    /// Preamble template. Variables: `{date}`, `{time}`, `{hostname}`.
    /// Empty string disables the preamble block.
    #[serde(default = "default_preamble")]
    pub preamble: String,

    /// Zones to search for retrieval. Defaults to personal_vault + system_obs.
    #[serde(default = "default_zones")]
    pub retrieval_zones: Vec<String>,

    /// How many recent events to scan per zone.
    #[serde(default = "default_scan_limit")]
    pub scan_limit: usize,

    /// How many top results to include in the snippet.
    #[serde(default = "default_top_k")]
    pub retrieval_top_k: usize,
}

fn default_preamble() -> String {
    "You are noesis, a persistent cognitive runtime.\nDate: {date}  Time: {time}  Host: {hostname}".into()
}

fn default_zones() -> Vec<String> {
    vec!["personal_vault".into(), "system_obs".into()]
}

fn default_scan_limit() -> usize {
    100
}

fn default_top_k() -> usize {
    5
}

impl Default for ComposerConfig {
    fn default() -> Self {
        Self {
            preamble: default_preamble(),
            retrieval_zones: default_zones(),
            scan_limit: default_scan_limit(),
            retrieval_top_k: default_top_k(),
        }
    }
}

/// Stateless composer. Construct once at startup and call [`compose`] per
/// request. Cheap to clone (config is small, store is Arc).
pub struct Composer {
    pub config: ComposerConfig,
    store: Arc<Store>,
}

impl Composer {
    pub fn new(config: ComposerConfig, store: Arc<Store>) -> Self {
        Self { config, store }
    }

    /// Build `(preamble, retrieval_snippet)` for one request.
    ///
    /// - `preamble`  — rendered template; feed into `TransformConfig::system_preamble`
    ///   or prepend in the system block via `ContextTransform`.
    /// - `snippet`   — formatted retrieval results; pass as `RetrievalSlot::Snippet`.
    ///   Empty string when the query is empty or no results found.
    ///
    /// `budget_bytes` caps the snippet (the preamble is not budgeted here;
    /// callers should use `TransformConfig::retrieval_bytes` for the snippet).
    pub fn compose(&self, query: &str, budget_bytes: usize) -> (String, String) {
        let preamble = preamble::render_preamble(&self.config.preamble);

        let zones: Vec<Zone> = self
            .config
            .retrieval_zones
            .iter()
            .filter_map(|s| zone_from_str(s))
            .collect();

        let snippet = if query.is_empty() || zones.is_empty() {
            String::new()
        } else {
            retrieval::retrieve(
                query,
                &self.store,
                &zones,
                self.config.scan_limit,
                self.config.retrieval_top_k,
                budget_bytes,
            )
        };

        (preamble, snippet)
    }
}

fn zone_from_str(s: &str) -> Option<Zone> {
    match s {
        "input_events" => Some(Zone::InputEvents),
        "system_obs"   => Some(Zone::SystemObs),
        "personal_vault" => Some(Zone::PersonalVault),
        "session_scratch" => Some(Zone::SessionScratch),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_roundtrip() {
        let cfg = ComposerConfig::default();
        assert!(!cfg.preamble.is_empty());
        assert!(!cfg.retrieval_zones.is_empty());
        assert!(cfg.scan_limit > 0);
        assert!(cfg.retrieval_top_k > 0);
    }

    #[test]
    fn zone_parse() {
        assert!(zone_from_str("personal_vault").is_some());
        assert!(zone_from_str("unknown_zone").is_none());
    }
}
