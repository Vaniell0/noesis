//! Context transform implementation (plan §10).

use serde::Deserialize;

/// One turn in the client's `messages` array.
#[derive(Debug, Clone)]
pub struct ChatTurn {
    /// "user" | "assistant" | "system"
    pub role: String,
    pub content: String,
}

/// Retrieval snippet from the composer. The composer byte-budgets this to
/// `config.retrieval_bytes` against `insights` + `personal_vault`.
/// Pass [`RetrievalSlot::Empty`] until composer (#16) lands.
pub enum RetrievalSlot<'a> {
    /// Composer produced a rendered snippet — prepend it after the preamble.
    Snippet(&'a str),
    /// No retrieval results (composer absent, or empty result set). Skip slot.
    Empty,
}

/// Configuration for [`ContextTransform`]. Deserialises from the `[context]`
/// TOML section in `runtime.toml` or from a per-lens `lens.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct TransformConfig {
    /// How many completed (user+assistant) turns to keep before the final
    /// user query. Default 4. Set 0 for preamble + retrieval + query only.
    #[serde(default = "default_tail_turns")]
    pub tail_turns: usize,

    /// Byte budget for the retrieval slot. The composer must respect this;
    /// the transform does not enforce it — it trusts the snippet it receives.
    #[serde(default = "default_retrieval_bytes")]
    pub retrieval_bytes: usize,

    /// Static prefix rendered before retrieval and tail. Typically the DSL
    /// header + available tool schemas. Empty until composer (#16) ships the
    /// DSL header template.
    #[serde(default)]
    pub system_preamble: String,
}

fn default_tail_turns() -> usize {
    4
}

fn default_retrieval_bytes() -> usize {
    2048
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            tail_turns: default_tail_turns(),
            retrieval_bytes: default_retrieval_bytes(),
            system_preamble: String::new(),
        }
    }
}

/// Context transform — the boundary between the raw HTTP client request and
/// the substrate prompt. Stateless; construct once and call [`build_prompt`]
/// per request.
///
/// [`build_prompt`]: ContextTransform::build_prompt
pub struct ContextTransform {
    pub config: TransformConfig,
}

impl ContextTransform {
    pub fn new(config: TransformConfig) -> Self {
        Self { config }
    }

    /// Build the substrate prompt from a `messages` array and an optional
    /// retrieval snippet.
    ///
    /// # Prompt shape
    ///
    /// ```text
    /// <|im_start|>system
    /// {effective_system}<|im_end|>
    /// <|im_start|>user
    /// {tail turn 1 user}<|im_end|>
    /// <|im_start|>assistant
    /// {tail turn 1 assistant}<|im_end|>
    /// …
    /// <|im_start|>user
    /// {last_user_query}<|im_end|>
    /// <|im_start|>assistant
    /// ```
    ///
    /// The system block merges (in order): `system_preamble` from config,
    /// any `role: "system"` message at position 0, and the retrieval snippet.
    /// If all three are empty the system block is omitted.
    pub fn build_prompt(&self, messages: &[ChatTurn], retrieval: RetrievalSlot<'_>) -> String {
        // ── 1. Split system from the rest ──────────────────────────────────
        let (system_from_msg, turns) = match messages.first() {
            Some(m) if m.role == "system" => (m.content.as_str(), &messages[1..]),
            _ => ("", messages),
        };

        // ── 2. Find the last user turn ─────────────────────────────────────
        // Walk from the end to find the last "user" message. Everything before
        // it forms the tail pool; everything at-and-after is the final query.
        let last_user_pos = turns.iter().rposition(|m| m.role == "user");
        let (tail_pool, query_and_after) = match last_user_pos {
            Some(i) => turns.split_at(i),
            None => {
                // No user message at all — just emit the preamble.
                return self.system_block(system_from_msg, retrieval);
            }
        };

        // ── 3. Extract tail turns ──────────────────────────────────────────
        // A "completed turn" = one user message + the assistant reply(ies)
        // that immediately follow it. We collect from the end of tail_pool.
        let tail = collect_tail(tail_pool, self.config.tail_turns);

        // ── 4. Assemble ────────────────────────────────────────────────────
        let mut parts: Vec<String> = Vec::new();

        let sys = self.system_block(system_from_msg, retrieval);
        if !sys.is_empty() {
            parts.push(sys);
        }

        for m in tail {
            parts.push(chatml_turn(&m.role, &m.content));
        }

        for m in query_and_after {
            parts.push(chatml_turn(&m.role, &m.content));
        }

        // Cue the assistant turn.
        parts.push("<|im_start|>assistant\n".to_string());

        parts.concat()
    }

    fn system_block(&self, msg_system: &str, retrieval: RetrievalSlot<'_>) -> String {
        let mut parts: Vec<&str> = Vec::new();
        if !self.config.system_preamble.is_empty() {
            parts.push(self.config.system_preamble.as_str());
        }
        if !msg_system.is_empty() {
            parts.push(msg_system);
        }
        if let RetrievalSlot::Snippet(s) = retrieval {
            if !s.is_empty() {
                parts.push(s);
            }
        }
        if parts.is_empty() {
            return String::new();
        }
        chatml_turn("system", &parts.join("\n\n"))
    }
}

/// Collect the last `n` completed (user+follower) turns from the tail pool.
/// Returns them in chronological order.
fn collect_tail(pool: &[ChatTurn], n: usize) -> &[ChatTurn] {
    if n == 0 || pool.is_empty() {
        return &[];
    }
    // Find the boundaries of the last `n` user-headed groups from the end.
    // A group starts at each "user" message.
    let mut group_starts: Vec<usize> = pool
        .iter()
        .enumerate()
        .filter_map(|(i, m)| (m.role == "user").then_some(i))
        .collect();

    // Keep only the last `n` group starts.
    if group_starts.len() > n {
        group_starts.drain(..group_starts.len() - n);
    }

    match group_starts.first() {
        Some(&start) => &pool[start..],
        None => &[],
    }
}

fn chatml_turn(role: &str, content: &str) -> String {
    format!("<|im_start|>{role}\n{content}<|im_end|>\n")
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn turns(pairs: &[(&str, &str)]) -> Vec<ChatTurn> {
        pairs
            .iter()
            .map(|(r, c)| ChatTurn { role: r.to_string(), content: c.to_string() })
            .collect()
    }

    fn tx(tail_turns: usize) -> ContextTransform {
        ContextTransform::new(TransformConfig {
            tail_turns,
            ..Default::default()
        })
    }

    #[test]
    fn single_user_message_no_history() {
        let msgs = turns(&[("user", "Hello")]);
        let prompt = tx(4).build_prompt(&msgs, RetrievalSlot::Empty);
        assert!(prompt.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
        // No system block when preamble + system msg + retrieval all empty.
        assert!(!prompt.contains("<|im_start|>system"));
    }

    #[test]
    fn system_message_rendered_in_system_block() {
        let msgs = turns(&[("system", "Be concise."), ("user", "Hi")]);
        let prompt = tx(4).build_prompt(&msgs, RetrievalSlot::Empty);
        assert!(prompt.starts_with("<|im_start|>system\nBe concise.<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>user\nHi<|im_end|>"));
    }

    #[test]
    fn retrieval_appended_to_system_block() {
        let msgs = turns(&[("user", "What is X?")]);
        let prompt = tx(4).build_prompt(&msgs, RetrievalSlot::Snippet("X is Y."));
        assert!(prompt.contains("<|im_start|>system\nX is Y.<|im_end|>"));
    }

    #[test]
    fn tail_turns_truncation() {
        // 5 turns (user+assistant) + final user query = 6 user messages total.
        let msgs = turns(&[
            ("user", "q1"), ("assistant", "a1"),
            ("user", "q2"), ("assistant", "a2"),
            ("user", "q3"), ("assistant", "a3"),
            ("user", "q4"), ("assistant", "a4"),
            ("user", "q5"), ("assistant", "a5"),
            ("user", "final"),
        ]);
        let prompt = tx(2).build_prompt(&msgs, RetrievalSlot::Empty);
        // Should contain q4, a4, q5, a5, final — but NOT q1..q3.
        assert!(prompt.contains("q4"));
        assert!(prompt.contains("q5"));
        assert!(prompt.contains("final"));
        assert!(!prompt.contains("q1"));
        assert!(!prompt.contains("q3"));
    }

    #[test]
    fn tail_turns_zero_keeps_only_query() {
        let msgs = turns(&[
            ("user", "old"), ("assistant", "old_a"),
            ("user", "query"),
        ]);
        let prompt = tx(0).build_prompt(&msgs, RetrievalSlot::Empty);
        assert!(prompt.contains("query"));
        assert!(!prompt.contains("old"));
    }

    #[test]
    fn incomplete_final_turn_no_assistant_yet() {
        // Typical case: last message is user, no assistant reply yet.
        let msgs = turns(&[("user", "u1"), ("assistant", "a1"), ("user", "u2")]);
        let prompt = tx(4).build_prompt(&msgs, RetrievalSlot::Empty);
        assert!(prompt.contains("u1"));
        assert!(prompt.contains("a1"));
        assert!(prompt.contains("u2"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn preamble_prepended_before_system_msg() {
        let mut cfg = TransformConfig::default();
        cfg.system_preamble = "DSL/1.0".to_string();
        cfg.tail_turns = 4;
        let tx = ContextTransform::new(cfg);
        let msgs = turns(&[("system", "Be brief."), ("user", "Go")]);
        let prompt = tx.build_prompt(&msgs, RetrievalSlot::Empty);
        // system block should have preamble then system msg joined by \n\n
        assert!(prompt.starts_with("<|im_start|>system\nDSL/1.0\n\nBe brief.<|im_end|>"));
    }
}
