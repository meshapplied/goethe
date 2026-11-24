// goethe-client: everything a node says to the outside. Linking, credentials and the console heartbeat (state, usage and errors straight to StateDB, every node), and the paid goethe-server connection (live load, routed requests, signed commands under the node's own policy and pinned passkeys).

pub mod api;
pub mod connect;
pub mod exec;
pub mod heartbeat;
pub mod link;
pub mod login;
pub mod remote;
mod serve;
mod state;

pub use serve::{serve, Body, Gateway};
pub use state::Node;
