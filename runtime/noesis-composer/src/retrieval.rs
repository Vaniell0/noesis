//! Keyword retrieval against the event store.
//!
//! Algorithm:
//!   1. Fetch `scan_limit` recent events per zone.
//!   2. Score each event: (keyword_hits / query_words) × recency_weight.
//!      Recency weight: rank_position / total (newest = 1.0, oldest → 0).
//!   3. Take `top_k` by score (score > 0 only).
//!   4. Format as plain-text block and cap at `budget_bytes`.

use noesis_schema::Zone;
use noesis_store::Store;

/// Run keyword retrieval and return a formatted snippet (may be empty).
pub fn retrieve(
    query: &str,
    store: &Store,
    zones: &[Zone],
    scan_limit: usize,
    top_k: usize,
    budget_bytes: usize,
) -> String {
    let keywords: Vec<String> = tokenize(query);
    if keywords.is_empty() {
        return String::new();
    }

    // Collect candidates from all zones.
    let mut candidates: Vec<(f32, String)> = Vec::new();
    for &zone in zones {
        let events = match store.zone(zone).recent(scan_limit) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let n = events.len();
        for (rank, ev) in events.into_iter().enumerate() {
            // recency_weight: 1.0 for the most recent, linear decay.
            let recency = if n > 1 { (n - rank) as f32 / n as f32 } else { 1.0 };
            let text = format!("{} {}", ev.kind, ev.payload);
            let score = keyword_score(&text, &keywords) * recency;
            if score > 0.0 {
                let line = format_event(&ev.kind, &ev.payload);
                candidates.push((score, line));
            }
        }
    }

    if candidates.is_empty() {
        return String::new();
    }

    // Sort descending by score, take top_k.
    candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    candidates.truncate(top_k);

    // Format and apply byte budget.
    let header = "Context:\n";
    let mut out = String::from(header);
    for (_, line) in &candidates {
        let candidate = format!("- {line}\n");
        if out.len() + candidate.len() > budget_bytes {
            break;
        }
        out.push_str(&candidate);
    }

    if out == header {
        // Nothing fit in budget.
        return String::new();
    }
    out
}

/// Score = fraction of query keywords found in `text` (case-insensitive).
fn keyword_score(text: &str, keywords: &[String]) -> f32 {
    if keywords.is_empty() {
        return 0.0;
    }
    let lower = text.to_lowercase();
    let hits = keywords.iter().filter(|kw| lower.contains(kw.as_str())).count();
    hits as f32 / keywords.len() as f32
}

/// Split query into lowercase words ≥ 3 chars (skip stop words and noise).
fn tokenize(query: &str) -> Vec<String> {
    const STOP: &[&str] = &["the", "and", "for", "are", "was", "has", "you", "that", "with"];
    query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 3)
        .map(|w| w.to_lowercase())
        .filter(|w| !STOP.contains(&w.as_str()))
        .collect()
}

fn format_event(kind: &str, payload: &serde_json::Value) -> String {
    // Try to extract a short summary from the payload.
    let summary = payload_summary(payload);
    if summary.is_empty() {
        kind.to_string()
    } else {
        format!("{kind}: {summary}")
    }
}

fn payload_summary(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => truncate(s, 120),
        serde_json::Value::Object(m) => {
            // Prefer "text", "message", "content", "summary", "body" fields.
            for key in ["text", "message", "content", "summary", "body", "value"] {
                if let Some(serde_json::Value::String(s)) = m.get(key) {
                    return truncate(s, 120);
                }
            }
            // Fall back: first string value in the object.
            for val in m.values() {
                if let serde_json::Value::String(s) = val {
                    return truncate(s, 120);
                }
            }
            String::new()
        }
        _ => String::new(),
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        // Truncate at a char boundary.
        let end = s.char_indices().take_while(|(i, _)| *i < max).last().map(|(i, c)| i + c.len_utf8()).unwrap_or(max);
        format!("{}…", &s[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_drops_short_and_stop_words() {
        let toks = tokenize("the cat sat on a mat");
        assert!(!toks.contains(&"the".into()));
        assert!(!toks.contains(&"on".into()));
        assert!(toks.contains(&"cat".into()));
        assert!(toks.contains(&"sat".into()));
        assert!(toks.contains(&"mat".into()));
    }

    #[test]
    fn keyword_score_full_match() {
        assert_eq!(keyword_score("hello world", &["hello".into(), "world".into()]), 1.0);
    }

    #[test]
    fn keyword_score_partial() {
        let s = keyword_score("hello there", &["hello".into(), "world".into()]);
        assert!((s - 0.5).abs() < 1e-6);
    }

    #[test]
    fn keyword_score_no_match() {
        assert_eq!(keyword_score("nothing here", &["xyz".into()]), 0.0);
    }

    #[test]
    fn payload_summary_prefers_text_key() {
        let v = serde_json::json!({"text": "hello", "other": "skip"});
        assert_eq!(payload_summary(&v), "hello");
    }

    #[test]
    fn truncate_short_passthrough() {
        assert_eq!(truncate("hi", 10), "hi");
    }
}
