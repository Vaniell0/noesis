//! Zone collectors — background tasks that populate memory zones.
//!
//! Each collector owns one zone (or a slice of one) and drives its own tokio
//! task. Supervisor spawns them at startup and reaps on shutdown.

pub mod evdev;
pub mod journal;
pub mod proc_net;
pub mod proc_self;
pub mod proc_stat;
pub mod system_obs;
