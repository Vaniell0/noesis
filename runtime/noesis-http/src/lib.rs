//! Context transform for noesis HTTP endpoints (plan §10).
//!
//! Converts a raw `messages` array from an HTTP client into a prompt
//! the substrate model receives. The full shape (from `project_noesis_context_management`):
//!
//!   [system_preamble] [retrieval] [tail_turns] [last_user_query]
//!
//! All rendered in ChatML format (`<|im_start|>{role}\n{…}<|im_end|>\n`),
//! which is RWKV World v2.9's training format.
//!
//! ## What is NOT this crate's responsibility
//!
//! - **Retrieval**: the composer (#16) runs `search` against `insights` +
//!   `personal_vault` and passes the rendered snippet via [`RetrievalSlot`].
//!   When the composer is absent, pass `RetrievalSlot::Empty`.
//! - **DSL rendering**: events, insights, vault refs. The composer owns that.
//!   This crate renders plain text turns only.
//! - **Utility path**: `X-Noesis-Passthrough: true` routing lives in
//!   `noesis-runtime`'s HTTP layer; this crate only builds the substrate prompt.
//!
//! ## Turn boundary
//!
//! One "turn" = one user message and the assistant reply that follows it
//! (if any). `tail_turns` counts completed turns. The final user message
//! (the one the model will answer) is always included and does not count
//! toward `tail_turns`.

pub mod context;

pub use context::{ChatTurn, ContextTransform, RetrievalSlot, TransformConfig};
