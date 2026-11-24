// Remote commands from the console: the node's policy and pinned passkeys, the Ed25519 node key that proves this install, and running a signed command through the same code path as the CLI.

use goethe_proto::{CommandEvent, InstallKey, Remote, SignedCommand};
use std::path::Path;

pub fn node_key(home: &Path) -> Result<InstallKey, String> {
    // The node's Ed25519 key, made once under the platform folder (0600); its public half and hash go up when linking.
    let _ = home;
    Err("not implemented".into())
}

pub fn status(home: &Path) -> Remote {
    // Policy and pinned keys (public data only) for the snapshot's remote block.
    let _ = home;
    Remote { policy: Default::default(), trust: Vec::new() }
}

pub fn handle(home: &Path, signed: &SignedCommand, emit: &mut dyn FnMut(CommandEvent)) {
    // Blocking: verify, check replay, run, and emit accepted or refused, progress, then done.
    let _ = (home, signed, emit);
}

pub fn outcome(home: &Path, cmd_id: &str) -> Option<Vec<CommandEvent>> {
    // The stored events of a command this node has seen (the last 256), for GET /cmd/{id} after a reconnect.
    let _ = (home, cmd_id);
    None
}
