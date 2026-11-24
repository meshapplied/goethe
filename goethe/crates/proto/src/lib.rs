// Wire contract between goethed, the platform and the console: node state, signed commands, install-key proof, sealed secrets.

mod action;
mod command;
mod install;
mod route;
mod seal;
mod state;
mod webauthn;

#[cfg(any(test, feature = "testkit"))]
pub mod testkit;

pub use action::{
    Action, Allow, Empty, Expose, ExposeMode, Gpus, Kind, KeyCreate, KeyRef, KeyRotate, ModelInstall, ModelLoad, ModelRef,
    ModelUnload, Share, TrustAdd, Update,
};
pub use command::{prepare, verify, Batch, BATCH_SCHEMA, COMMAND_SCHEMA, Command, CommandEvent, Expect, Policy, Prepared, Refusal, SignedCommand, TrustEntry, Verified};
pub use install::{install_hash, verify_connect, InstallKey};
pub use route::{Usage, DISPATCH_HEADER, INSTANCE_HEADER, KEY_ID_HEADER, REQUEST_ID_HEADER, USAGE_TRAILER, USER_HEADER};
pub use seal::{seal, Sealed};
pub use state::{Instance, Remote, StateEvent, STATE_SCHEMA};
pub use webauthn::{key_fingerprint, link_challenge, pair_challenge, Assertion, Pairing};

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;

pub const RP_ID: &str = "console.goethe.sh";
pub const CONSOLE_ORIGIN: &str = "https://console.goethe.sh";

pub(crate) fn b64(bytes: &[u8]) -> String {
    URL_SAFE_NO_PAD.encode(bytes)
}

pub(crate) fn unb64(text: &str) -> Option<Vec<u8>> {
    URL_SAFE_NO_PAD.decode(text).ok()
}

pub(crate) fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(ring::digest::digest(&ring::digest::SHA256, bytes).as_ref());
    out
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
