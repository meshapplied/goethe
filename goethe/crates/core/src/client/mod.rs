// goethe-client: the node side of the goethe-server connection for paid routing: live load, routed requests and signed commands (remote policy, pinned passkeys, executor). Telemetry stays on the console heartbeat.

pub mod exec;
pub mod remote;
mod serve;
mod state;

pub use serve::{serve, Body, Gateway};
pub use state::Node;
