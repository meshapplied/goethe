// GOETHE_HOME/remote.json: which kinds of console change this node accepts (all off until the owner allows one) and the pinned passkeys, with an append-only log of pins.

use goethe_proto::{key_fingerprint, Pairing, Policy, Remote, TrustEntry, CONSOLE_ORIGIN, RP_ID};
pub use goethe_proto::Kind;
use serde::{Deserialize, Serialize};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const FILE: &str = "remote.json";
const MAX_BYTES: u64 = 256 * 1024;
const MAX_LOG: usize = 1000;
const MAX_ID: usize = 256;
pub const LOCAL: &str = "local";

static LOCK: Mutex<()> = Mutex::new(());

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Stored {
    pub policy: Policy,
    pub trust: Vec<TrustEntry>,
    #[serde(default)]
    pub log: Vec<LogEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogEntry {
    pub t_ms: u64,
    pub event: String,
    pub key_id: String,
    pub user_id: String,
    pub cap: Kind,
    pub by: String,
}

pub fn path(home: &Path) -> PathBuf {
    home.join(FILE)
}

pub fn load(home: &Path) -> Result<Stored, String> {
    // The stored policy and keys; a missing file is everything off and nothing pinned, a damaged or loose one is an error, never read.
    let p = path(home);
    let meta = match std::fs::symlink_metadata(&p) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Stored::default()),
        Err(e) => return Err(format!("cannot read {}: {e}", p.display())),
    };
    let uid = unsafe { libc::geteuid() };
    if !meta.file_type().is_file() || meta.uid() != uid || meta.len() > MAX_BYTES {
        return Err(format!("{} is not a regular file of yours under 256 KB; remove it and allow kinds again", p.display()));
    }
    if meta.mode() & 0o077 != 0 {
        return Err(format!("{} can be read or changed by other users; fix it with: chmod 600 {}", p.display(), p.display()));
    }
    let text = std::fs::read_to_string(&p).map_err(|e| format!("cannot read {}: {e}", p.display()))?;
    serde_json::from_str(&text).map_err(|_| format!("{} is damaged; remove it and allow kinds again", p.display()))
}

fn save(home: &Path, s: &Stored) -> Result<(), String> {
    std::fs::create_dir_all(home).map_err(|e| format!("cannot create {}: {e}", home.display()))?;
    let data = serde_json::to_vec_pretty(s).map_err(|e| e.to_string())?;
    crate::paths::write_private(&path(home), &data).map_err(|e| format!("cannot write {}: {e}", path(home).display()))
}

fn change<T>(home: &Path, f: impl FnOnce(&mut Stored) -> Result<T, String>) -> Result<T, String> {
    // Read, change and write back under one lock, so two changes in this process never lose each other.
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut s = load(home)?;
    let out = f(&mut s)?;
    save(home, &s)?;
    Ok(out)
}

pub fn status(home: &Path) -> Remote {
    // Policy and pinned keys (public data only); an unreadable file counts as all off, nothing pinned.
    let s = load(home).unwrap_or_default();
    Remote { policy: s.policy, trust: s.trust }
}

pub fn set_allowed(home: &Path, kind: Kind, on: bool) -> Result<(), String> {
    change(home, |s| {
        let p = s.policy;
        s.policy = match kind {
            Kind::Change => Policy { change: on, ..p },
            Kind::Network => Policy { network: on, ..p },
            Kind::Keys => Policy { keys: on, ..p },
            Kind::Update => Policy { update: on, ..p },
            Kind::Destructive => Policy { destructive: on, ..p },
            Kind::Trust => Policy { trust: on, ..p },
        };
        Ok(())
    })
}

fn logged(s: &mut Stored, event: &str, e: &TrustEntry, by: &str, t_ms: u64) {
    s.log.push(LogEntry { t_ms, event: event.into(), key_id: e.key_id.clone(), user_id: e.user_id.clone(), cap: e.cap, by: by.into() });
    let over = s.log.len().saturating_sub(MAX_LOG);
    s.log.drain(..over);
}

pub fn pin(home: &Path, entry: TrustEntry, by: &str) -> Result<(), String> {
    // Pin a passkey (a second pin of the same key id replaces the first) and log who pinned it; ids are printed by goethe remote status, so no control characters.
    let clean = |s: &str| !s.is_empty() && s.len() <= MAX_ID && !s.chars().any(char::is_control);
    if !clean(&entry.key_id) || !clean(&entry.user_id) || !clean(&entry.added_by) {
        return Err(format!("a pinned key needs a key id, a user and who added it, each up to {MAX_ID} printable characters"));
    }
    change(home, |s| {
        logged(s, "added", &entry, by, entry.added_ms);
        s.trust.retain(|t| t.key_id != entry.key_id);
        s.trust.push(entry);
        Ok(())
    })
}

pub fn unpin(home: &Path, key_id: &str, by: &str, t_ms: u64) -> Result<TrustEntry, String> {
    change(home, |s| {
        let i = s.trust.iter().position(|t| t.key_id == key_id).ok_or_else(|| format!("no pinned key {key_id}; goethe remote trust list shows them"))?;
        let gone = s.trust.remove(i);
        logged(s, "removed", &gone, by, t_ms);
        Ok(gone)
    })
}

pub fn add_pairing(home: &Path, text: &str, now_ms: u64) -> Result<TrustEntry, String> {
    // goethe remote trust add: the console's pairing string, signed by the passkey it names over its key, user and cap.
    let p = Pairing::decode(text).ok_or("that is not a pairing string; copy it again from the console")?;
    if !p.signed(&p.pair_challenge(), RP_ID, CONSOLE_ORIGIN) {
        return Err("the pairing string's signature does not verify; nothing was pinned".into());
    }
    let entry = TrustEntry { key_id: p.key_id, public_key: p.public_key, user_id: p.user_id.clone(), cap: p.cap, added_by: p.user_id, added_ms: now_ms };
    pin(home, entry.clone(), LOCAL)?;
    Ok(entry)
}

pub fn key_line(t: &TrustEntry) -> String {
    format!("{}  user {}  cap {}  fingerprint {}", t.key_id, t.user_id, t.cap.name(), key_fingerprint(&t.public_key))
}

pub fn status_text(home: &Path) -> Result<String, String> {
    // goethe remote status: each kind on or off, then the pinned keys.
    let s = load(home)?;
    let mut out = String::from("Console changes this node accepts (goethe remote allow KIND turns one on):\n");
    for k in Kind::ALL {
        out.push_str(&format!("  {:<12} {}\n", k.name(), if s.policy.allows(k) { "on" } else { "off" }));
    }
    out.push('\n');
    out.push_str(&trust_text(&s.trust));
    Ok(out)
}

pub fn trust_text(trust: &[TrustEntry]) -> String {
    if trust.is_empty() {
        return "No pinned passkeys; goethe remote trust add PAIRING pins one.\n".into();
    }
    let mut out = String::from("Pinned passkeys:\n");
    for t in trust {
        out.push_str(&format!("  {}\n", key_line(t)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use goethe_proto::testkit::Passkey;
    use std::os::unix::fs::PermissionsExt;

    fn pairing(pk: &Passkey, user: &str, cap: Kind) -> Pairing {
        let c = goethe_proto::pair_challenge(&pk.key_id, &pk.public_key_b64(), user, cap);
        Pairing { key_id: pk.key_id.clone(), public_key: pk.public_key_b64(), user_id: user.into(), cap, assertion: pk.assert(&c, RP_ID, CONSOLE_ORIGIN, true) }
    }

    #[test]
    fn every_kind_starts_off_and_each_allow_persists_alone() {
        let h = tempfile::tempdir().unwrap();
        assert_eq!(status(h.path()).policy, Policy::default());
        set_allowed(h.path(), Kind::Network, true).unwrap();
        let p = status(h.path()).policy;
        assert!(p.network && Kind::ALL.iter().filter(|k| p.allows(**k)).count() == 1);
        assert_eq!(std::fs::metadata(path(h.path())).unwrap().mode() & 0o777, 0o600);
        set_allowed(h.path(), Kind::Network, false).unwrap();
        assert_eq!(status(h.path()).policy, Policy::default());
        assert!(status_text(h.path()).unwrap().contains("network      off"));
    }

    #[test]
    fn a_loose_or_damaged_file_fails_closed() {
        let h = tempfile::tempdir().unwrap();
        set_allowed(h.path(), Kind::Change, true).unwrap();
        std::fs::set_permissions(path(h.path()), std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(load(h.path()).unwrap_err().contains("chmod 600"));
        assert_eq!(status(h.path()).policy, Policy::default(), "an exposed file is not trusted");
        assert!(set_allowed(h.path(), Kind::Change, true).is_err());
        std::fs::set_permissions(path(h.path()), std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::write(path(h.path()), r#"{"policy":{"change":true}}"#).unwrap();
        assert!(load(h.path()).unwrap_err().contains("damaged"));
    }

    #[test]
    fn a_pairing_string_pins_only_what_its_key_signed() {
        let h = tempfile::tempdir().unwrap();
        let pk = Passkey::new("k1");
        let good = pairing(&pk, "user1", Kind::Change);
        let t = add_pairing(h.path(), &good.encode(), 5).unwrap();
        assert_eq!((t.cap, t.added_ms, t.user_id.as_str()), (Kind::Change, 5, "user1"));
        let mut raised = good.clone();
        raised.cap = Kind::Trust;
        assert!(add_pairing(h.path(), &raised.encode(), 6).unwrap_err().contains("does not verify"));
        let mut borrowed = pairing(&Passkey::new("k2"), "user1", Kind::Change);
        borrowed.key_id = "k1".into();
        assert!(add_pairing(h.path(), &borrowed.encode(), 6).is_err(), "another key's signature");
        assert!(add_pairing(h.path(), "garbage", 6).is_err());
        let escape = pairing(&Passkey::new("k3"), "user1\u{1b}[2J", Kind::Change);
        assert!(add_pairing(h.path(), &escape.encode(), 6).unwrap_err().contains("printable"), "a terminal escape in an id is refused");
        assert_eq!(status(h.path()).trust, vec![t.clone()]);
        assert!(trust_text(&status(h.path()).trust).contains(&key_fingerprint(&pk.public_key_b64())));
        assert_eq!(unpin(h.path(), "k1", LOCAL, 9).unwrap(), t);
        assert!(unpin(h.path(), "k1", LOCAL, 9).is_err());
        let log = load(h.path()).unwrap().log;
        assert_eq!(log.iter().map(|e| (e.event.as_str(), e.t_ms)).collect::<Vec<_>>(), [("added", 5), ("removed", 9)]);
    }
}
