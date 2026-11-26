// Temporary public gateway (goethe api share): a second nginx listener that exists only while a window is open, for chosen existing keys.

use crate::sys::unix_secs;
use crate::{db, nginx, security};
use serde::{Deserialize, Serialize};
use std::fs;
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const DEFAULT_PORT: u16 = 8444;
pub const MIN_SECS: i64 = 60;
pub const MAX_SECS: i64 = 7 * 86_400;
const SUN_PATH_MAX: usize = 100;

pub fn public_mux_socket(home: &Path) -> PathBuf {
    // Public-gateway traffic reaches goethed only over this socket, which only nginx can open, so the X-Real-IP it sets is the client's.
    nginx::nginx_prefix(home).join("mux-public.sock")
}

pub fn bind_public_mux(home: &Path) -> Result<UnixListener, String> {
    // Owner-only (0600); under root, 0660 for the nobody group nginx workers run as. A stale socket from an earlier goethed is replaced.
    let path = public_mux_socket(home);
    if path.as_os_str().len() > SUN_PATH_MAX {
        return Err(format!("{} is too long for a socket path; use a shorter GOETHE_HOME", path.display()));
    }
    fs::create_dir_all(nginx::nginx_prefix(home)).map_err(|e| e.to_string())?;
    match fs::symlink_metadata(&path) {
        Ok(m) if m.file_type().is_socket() => fs::remove_file(&path).map_err(|e| format!("remove {}: {e}", path.display()))?,
        Ok(_) => return Err(format!("{} exists and is not a socket", path.display())),
        Err(_) => {}
    }
    let listener = UnixListener::bind(&path).map_err(|e| format!("bind {}: {e}", path.display()))?;
    let mode = match nginx::worker_gid() {
        Some(gid) => {
            if let Err(e) = std::os::unix::fs::chown(&path, None, Some(gid)) {
                let _ = fs::remove_file(&path);
                return Err(format!("chown {}: {e}", path.display()));
            }
            0o660
        }
        None => 0o600,
    };
    if let Err(e) = fs::set_permissions(&path, fs::Permissions::from_mode(mode)) {
        let _ = fs::remove_file(&path);
        return Err(format!("chmod {}: {e}", path.display()));
    }
    Ok(listener)
}

const DEFAULT_RATE_PER_IP: u64 = 10;
const DEFAULT_CONN_PER_IP: u64 = 16;

fn share_rate(home: &Path) -> u64 {
    nginx::gateway_limit(home, "share_rate_per_ip", DEFAULT_RATE_PER_IP)
}

fn share_conn(home: &Path) -> u64 {
    nginx::gateway_limit(home, "share_conn_per_ip", DEFAULT_CONN_PER_IP)
}

fn persist_share_limits(home: &Path) {
    // Hidden: GOETHE_SHARE_RATE_PER_IP / GOETHE_SHARE_CONN_PER_IP (1-65535); same shape as the private gateway.
    nginx::persist_named_limits(
        home,
        &[
            ("GOETHE_SHARE_RATE_PER_IP", "share_rate_per_ip", DEFAULT_RATE_PER_IP),
            ("GOETHE_SHARE_CONN_PER_IP", "share_conn_per_ip", DEFAULT_CONN_PER_IP),
        ],
        "goethe api share",
    );
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Share {
    pub open: bool,
    pub until: i64,
    pub port: u16,
    pub keys: Vec<String>,
    #[serde(default)]
    pub allow: Vec<String>,
    #[serde(default)]
    pub grants: Vec<Grant>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Grant {
    pub key: String,
    pub until: i64,
    #[serde(default)]
    pub allow: Vec<String>,
}

impl Share {
    pub fn active_at(&self, now: i64) -> bool {
        self.open && now < self.until
    }

    fn settle(&mut self) {
        // One gateway for every shared key: open until the last key's time ends; nginx admits the union of their allowlists.
        self.keys = self.grants.iter().map(|g| g.key.clone()).collect();
        self.until = self.grants.iter().map(|g| g.until).max().unwrap_or(self.until);
        let mut all: Vec<String> = Vec::new();
        for c in self.grants.iter().flat_map(|g| g.allow.iter()) {
            if !all.contains(c) {
                all.push(c.clone());
            }
        }
        self.allow = all;
    }

    fn grant(&self, key: &str) -> Option<Grant> {
        // A window written before grants existed shares every listed key until its end, from its addresses.
        match self.grants.iter().find(|g| g.key == key) {
            Some(g) => Some(g.clone()),
            None if self.grants.is_empty() && self.keys.iter().any(|k| k == key) => Some(Grant { key: key.to_string(), until: self.until, allow: self.allow.clone() }),
            None => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Access {
    Allowed,
    Closed,
    KeyNotShared,
    KeyExpired,
    NotFromHere,
}

fn cidr_contains(cidr: &str, ip: IpAddr) -> bool {
    // Whether a client address falls in an allowlist entry (a.b.c.d/n or an IPv6 prefix).
    let (addr, bits) = match cidr.split_once('/') {
        Some((a, b)) => (a, b.parse::<u32>().ok()),
        None => (cidr, None),
    };
    match (addr.parse::<IpAddr>(), ip) {
        (Ok(IpAddr::V4(net)), IpAddr::V4(ip)) => {
            let n = bits.unwrap_or(32).min(32);
            let mask = if n == 0 { 0 } else { u32::MAX << (32 - n) };
            u32::from(net) & mask == u32::from(ip) & mask
        }
        (Ok(IpAddr::V6(net)), IpAddr::V6(ip)) => {
            let n = bits.unwrap_or(128).min(128);
            let mask = if n == 0 { 0 } else { u128::MAX << (128 - n) };
            u128::from(net) & mask == u128::from(ip) & mask
        }
        (Ok(IpAddr::V4(net)), IpAddr::V6(ip)) => ip.to_ipv4_mapped().is_some_and(|v4| cidr_contains(&format!("{net}/{}", bits.unwrap_or(32)), IpAddr::V4(v4))),
        _ => false,
    }
}

pub struct SharedKey {
    pub id: String,
    pub name: String,
    pub token: Option<String>,
    pub models: Option<String>,
    pub expires: Option<String>,
}

pub struct Opened {
    pub share: Share,
    pub keys: Vec<SharedKey>,
    pub public_ip: Option<IpAddr>,
    pub cert: PathBuf,
    pub fingerprint: Option<String>,
    pub warnings: Vec<String>,
}


fn path(home: &Path) -> PathBuf {
    nginx::nginx_prefix(home).join("public_share.json")
}

pub fn load(home: &Path) -> Option<Share> {
    fs::read(path(home)).ok().and_then(|b| serde_json::from_slice(&b).ok())
}

fn store(home: &Path, share: &Share) -> Result<(), String> {
    let data = serde_json::to_vec_pretty(share).map_err(|e| e.to_string())?;
    nginx::write_private(&path(home), &data)
}

pub fn active(home: &Path) -> Option<Share> {
    load(home).filter(|s| s.active_at(unix_secs()))
}

pub fn parse_duration(raw: &str) -> Result<i64, String> {
    // 30m, 5h or 1d: a whole number and one unit, from 1 minute to 7 days.
    let s = raw.trim().to_ascii_lowercase();
    let bad = || format!("duration must be minutes, hours or days such as 30m, 5h or 1d, got {raw:?}");
    let (num, unit) = s.split_at(s.char_indices().last().map_or(0, |(i, _)| i));
    let n: i64 = num.parse().map_err(|_| bad())?;
    let secs = match unit {
        "m" => n.checked_mul(60),
        "h" => n.checked_mul(3600),
        "d" => n.checked_mul(86_400),
        _ => None,
    }
    .ok_or_else(bad)?;
    if !(MIN_SECS..=MAX_SECS).contains(&secs) {
        return Err(format!("the public gateway opens for 1m to 7d, got {raw}"));
    }
    Ok(secs)
}

type Stamp = Option<(SystemTime, u64, u64)>;
static CACHE: Mutex<Option<(Stamp, Option<Share>)>> = Mutex::new(None);

fn stamp(home: &Path) -> Stamp {
    use std::os::unix::fs::MetadataExt;
    fs::metadata(path(home)).ok().map(|m| (m.modified().unwrap_or(UNIX_EPOCH), m.len(), m.ino()))
}

pub fn check(home: &Path, key_id: &str, is_admin: bool, client: Option<IpAddr>) -> Access {
    // The router's check on the public listener: an open, unexpired window listing this key (never admin); re-read whenever the file changes.
    let now_stamp = stamp(home);
    let share = {
        let mut slot = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        match slot.as_ref() {
            Some((st, s)) if *st == now_stamp => s.clone(),
            _ => {
                let s = if now_stamp.is_some() { load(home) } else { None };
                slot.replace((now_stamp, s.clone()));
                s
            }
        }
    };
    decide(share.as_ref(), key_id, is_admin, unix_secs(), client)
}

fn decide(share: Option<&Share>, key_id: &str, is_admin: bool, now: i64, client: Option<IpAddr>) -> Access {
    // Each shared key has its own end and addresses; one that ended is refused while the gateway stays open for the others.
    let Some(s) = share.filter(|s| s.active_at(now)) else { return Access::Closed };
    let Some(g) = s.grant(key_id).filter(|_| !is_admin) else { return Access::KeyNotShared };
    if now >= g.until {
        return Access::KeyExpired;
    }
    if g.allow.is_empty() || !client.is_some_and(|ip| g.allow.iter().any(|c| cidr_contains(c, ip))) {
        return Access::NotFromHere;
    }
    Access::Allowed
}

type KeyRow = (String, String, bool, Option<String>, Option<String>);

fn key_rows(home: &Path) -> Result<Vec<KeyRow>, String> {
    let con = db::open(home).map_err(|e| e.to_string())?;
    let mut stmt = con
        .prepare("SELECT key_id, name, is_admin, expires_at, model_scope FROM api_keys ORDER BY created_at")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get::<_, i64>(2)? != 0, r.get(3)?, r.get(4)?)))
        .map_err(|e| e.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string());
    rows
}

fn resolve_keys(home: &Path, wanted: &[String], until: i64, warnings: &mut Vec<String>) -> Result<Vec<SharedKey>, String> {
    // Each name or id must match exactly one existing key that is not an admin key and has not expired.
    let rows = key_rows(home)?;
    let names = || rows.iter().filter(|r| !r.2).map(|r| r.1.clone()).collect::<Vec<_>>().join(", ");
    let until_day = chrono::DateTime::from_timestamp(until, 0).map(|d| d.format("%Y-%m-%d").to_string()).unwrap_or_default();
    let mut out: Vec<SharedKey> = Vec::new();
    for w in wanted {
        let hits: Vec<_> = rows.iter().filter(|r| r.0 == *w || r.1 == *w).collect();
        let row = match hits.as_slice() {
            [one] => *one,
            [] => return Err(format!("no API key named {w:?}; keys: {}", if names().is_empty() { "none (create one with goethe api keys create <name> --models <model>)".into() } else { names() })),
            _ => return Err(format!("several keys are named {w:?}; use the key id from goethe api keys")),
        };
        let (id, name, admin, expires, models) = row.clone();
        if admin {
            return Err(format!("{name} is an admin key; admin keys never work on the public gateway"));
        }
        if security::key_expired(&expires) {
            return Err(format!("{name} expired on {}; pick another key or change its expiry in goethe api keys", expires.unwrap_or_default()));
        }
        if expires.as_deref().is_some_and(|e| e < until_day.as_str()) {
            warnings.push(format!("{name} expires on {}, before the public gateway closes", expires.clone().unwrap_or_default()));
        }
        if models.as_deref().is_none_or(str::is_empty) {
            warnings.push(format!("{name} may use every model; a key limited with --models is safer to share"));
        }
        if !out.iter().any(|k| k.id == id) {
            out.push(SharedKey { id, name, token: None, models, expires });
        }
    }
    Ok(out)
}

fn port_free(port: u16) -> bool {
    TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], port))).is_ok()
}

fn share_allow(cli: &[String], reuse_keys: bool, kept: &[String]) -> Vec<String> {
    // --allow wins; reopening the same keys keeps the addresses from their last share. Never the VPN/API allowlist.
    if !cli.is_empty() {
        return cli.to_vec();
    }
    if reuse_keys && !kept.is_empty() {
        return kept.to_vec();
    }
    Vec::new()
}

fn need_client_allow(home: &Path) -> String {
    // Who may connect is the client's public IP, not this machine's (that one is only the URL).
    let here = crate::client::link::lookup_public_ip(home)
        .ok()
        .map(|ip| format!(" this machine's public address is {ip} (that becomes the URL, not --allow);"))
        .unwrap_or_default();
    format!(
        "name who may connect with --allow <their-ip>/32 (their phone or laptop's public IP, not this machine's).{here} ask them; to open to every address use --dangerously-allow-all-inbound"
    )
}

fn reject_bad_share_cidrs(home: &Path, allow: &[String]) -> Result<(), String> {
    let machine = crate::client::link::lookup_public_ip(home).ok();
    for c in allow {
        if c.ends_with("/0") {
            return Err(
                "the public gateway will not open to every address unless you pass --dangerously-allow-all-inbound; otherwise name the client's public IP with --allow <their-ip>/32"
                    .into(),
            );
        }
        if let Some(ip) = machine {
            let host = format!("{ip}/{}", if ip.is_ipv4() { 32 } else { 128 });
            if *c == host {
                return Err(format!(
                    "{c} is this machine's public address (the URL), not who may connect; use their phone or laptop's public IP"
                ));
            }
        }
    }
    Ok(())
}

fn resolve_share_cidrs(
    home: &Path,
    cli_allow: &[String],
    reuse_keys: bool,
    kept: &[String],
    dangerously_allow_all_inbound: bool,
) -> Result<Vec<String>, String> {
    if dangerously_allow_all_inbound {
        if !cli_allow.is_empty() {
            return Err("--dangerously-allow-all-inbound cannot be combined with --allow".into());
        }
        return Ok(vec!["0.0.0.0/0".into()]);
    }
    let allow = share_allow(cli_allow, reuse_keys, kept);
    if allow.is_empty() {
        return Err(need_client_allow(home));
    }
    reject_bad_share_cidrs(home, &allow)?;
    Ok(allow)
}

fn mux_public_listening(home: &Path) -> bool {
    UnixStream::connect(public_mux_socket(home)).is_ok()
}

fn wait_listening(port: u16, timeout: Duration) -> bool {
    let end = Instant::now() + timeout;
    while Instant::now() < end {
        if TcpStream::connect_timeout(&SocketAddr::from(([127, 0, 0, 1], port)), Duration::from_millis(300)).is_ok() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    false
}

pub struct OpenArgs<'a> {
    pub secs: i64,
    pub keys: &'a [String],
    pub allow: &'a [String],
    pub port: Option<u16>,
    pub dangerously_allow_all_inbound: bool,
}

pub fn open(home: &Path, a: OpenArgs) -> Result<Opened, String> {
    // Validate, then write the window and hot-reload nginx; an open window on the same port restarts its timer from now without closing.
    if !nginx::master_running(home) {
        return Err("the gateway is not running; start goethe with goethe up".to_string());
    }
    if !mux_public_listening(home) {
        return Err("goethed does not serve the public gateway yet; restart it with goethe down then goethe up".to_string());
    }
    let lock = nginx::gateway_lock(home)?;
    let previous = load(home);
    let until = unix_secs() + a.secs;
    let mut warnings = Vec::new();
    let wanted: Vec<String> = if a.keys.is_empty() {
        previous.as_ref().map(|p| p.keys.clone()).filter(|k| !k.is_empty()).ok_or_else(|| {
            "name the API keys to allow with --key <name> (the next time they are remembered); list them with goethe api keys".to_string()
        })?
    } else {
        a.keys.to_vec()
    };
    let reopen = a.keys.is_empty();
    let mut keys = resolve_keys(home, &wanted, until, &mut warnings)?;
    let already: Vec<String> = keys.iter().filter(|k| previous.as_ref().is_some_and(|p| p.active_at(unix_secs()) && p.grant(&k.id).is_some_and(|g| g.until > unix_secs()))).map(|k| k.id.clone()).collect();
    let cli_allow = a.allow.iter().map(|c| security::canonical_cidr(c)).collect::<Result<Vec<_>, _>>()?;
    let port = a.port.or(previous.as_ref().map(|p| p.port)).unwrap_or(DEFAULT_PORT);
    let mut grants: Vec<Grant> = previous
        .as_ref()
        .filter(|p| p.active_at(unix_secs()))
        .map(|p| p.keys.iter().filter_map(|k| p.grant(k)).filter(|g| g.until > unix_secs()).collect())
        .unwrap_or_default();
    for k in &keys {
        let kept = previous.as_ref().and_then(|p| p.grant(&k.id)).map(|g| g.allow).unwrap_or_default();
        let allow = resolve_share_cidrs(home, &cli_allow, reopen, &kept, a.dangerously_allow_all_inbound)?;
        grants.retain(|g| g.key != k.id);
        grants.push(Grant { key: k.id.clone(), until, allow });
    }
    persist_share_limits(home);
    let (rate, conn) = (share_rate(home), share_conn(home));
    if a.dangerously_allow_all_inbound {
        warnings.push(format!(
            "every address may connect; rate limit is {rate} requests/s per client (burst {}), {conn} connections",
            2 * rate
        ));
    }
    if reopen {
        warnings.push("the key secrets are not printed again; goethe api keys rotate <key-id> makes a new one".into());
    }
    let gateway_port = nginx::public_port_u16(home);
    let mux_port = nginx::persisted_mux_port(home)?;
    if port == gateway_port || port == mux_port || port < 1024 {
        return Err(format!("port {port} is not usable for the public gateway; pick a free port above 1023 with --port"));
    }
    let ours = previous.as_ref().is_some_and(|p| p.active_at(unix_secs()) && p.port == port);
    if !ours && !port_free(port) {
        return Err(format!("port {port} is in use; pick another with --port"));
    }
    let public_ip = crate::client::link::lookup_public_ip(home).ok();
    if public_ip.is_none() {
        warnings.push("could not look up this machine's public address; set GOETHE_PUBLIC_IP to name it".to_string());
    }
    let mut share = Share { open: true, until, port, grants, ..Share::default() };
    share.settle();
    ensure_cert(home, public_ip)?;
    store(home, &share)?;
    let applied = nginx::apply_locked(home).and_then(|()| {
        if wait_listening(port, Duration::from_secs(5)) {
            Ok(())
        } else {
            Err(format!("the public listener did not come up on port {port}"))
        }
    });
    if let Err(e) = applied {
        let _ = match &previous {
            Some(p) => store(home, p),
            None => fs::remove_file(path(home)).map_err(|e| e.to_string()),
        };
        let _ = nginx::apply_locked(home);
        return Err(format!("could not open the public gateway: {e}"));
    }
    drop(lock);
    if !reopen {
        renew_secrets(home, &mut keys, &already, &mut warnings);
    }
    let cert = cert_path(home);
    Ok(Opened { fingerprint: fingerprint(&cert), cert, share, keys, public_ip, warnings })
}

fn renew_secrets(home: &Path, keys: &mut [SharedKey], already: &[String], warnings: &mut Vec<String>) {
    // Only hashes are stored, so a newly shared key gets a new secret to send; one already shared keeps the secret its client has.
    for k in keys.iter_mut().filter(|k| !already.contains(&k.id)) {
        match security::rotate_secret(home, &k.id) {
            Ok(token) => k.token = Some(token),
            Err(e) => warnings.push(format!("{}: no new secret ({e}); goethe api keys rotate {} makes one", k.name, k.id)),
        }
    }
    if keys.iter().any(|k| k.token.is_some()) {
        warnings.push("each key above has a new secret; its old one no longer works, here or on the private gateway".into());
    }
}

pub fn stop_key(home: &Path, key: &str) -> Result<bool, String> {
    // Stop sharing one key (name or id); the gateway stays open for the others, and closes when none is left.
    let _lock = nginx::gateway_lock(home)?;
    let Some(mut share) = active(home) else { return Ok(false) };
    let ids: Vec<String> = key_rows(home)?.into_iter().filter(|r| r.0 == key || r.1 == key).map(|r| r.0).collect();
    if share.grants.is_empty() {
        share.grants = share.keys.iter().filter_map(|k| share.grant(k)).collect();
    }
    let before = share.grants.len();
    share.grants.retain(|g| !ids.contains(&g.key) && g.key != key);
    if share.grants.len() == before {
        return Ok(false);
    }
    if share.grants.iter().all(|g| g.until <= unix_secs()) {
        share.open = false;
    } else {
        share.settle();
    }
    store(home, &share)?;
    if nginx::master_running(home) {
        nginx::apply_locked(home)?;
    }
    Ok(true)
}

pub fn shared(home: &Path) -> Vec<(String, Grant)> {
    // The keys the open gateway serves now, by name, with their own end and addresses.
    let Some(s) = active(home) else { return Vec::new() };
    let names = key_names(home, &s.keys);
    s.keys.iter().zip(names).filter_map(|(k, n)| s.grant(k).filter(|g| g.until > unix_secs()).map(|g| (n, g))).collect()
}

pub fn shutdown(home: &Path) -> Result<bool, String> {
    // Close the window now and drop the listener; the chosen keys are remembered for the next opening.
    let _lock = nginx::gateway_lock(home)?;
    let Some(share) = load(home) else { return Ok(false) };
    let was_open = share.active_at(unix_secs());
    if !share.open {
        return Ok(false);
    }
    store(home, &Share { open: false, ..share })?;
    if nginx::master_running(home) {
        nginx::apply_locked(home)?;
    }
    Ok(was_open)
}

static SEEN: Mutex<Option<Stamp>> = Mutex::new(None);

pub fn tick(home: &Path) {
    // goethed, every 2 s: close an expired window and drop its listener; any change to the window sends a heartbeat soon.
    close_expired(home);
    let now_stamp = stamp(home);
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.is_some_and(|s| s != now_stamp) {
        crate::client::link::notify(crate::client::link::CHANGE_STATE);
    }
    seen.replace(now_stamp);
}

fn drop_ended(home: &Path) {
    // While the gateway stays open for some keys, a key whose time ended leaves it, and the addresses only it allowed go with it.
    let Some(share) = active(home) else { return };
    if !share.grants.iter().any(|g| g.until <= unix_secs()) {
        return;
    }
    let Ok(_lock) = nginx::gateway_lock(home) else { return };
    let Some(mut s) = active(home) else { return };
    let before = s.allow.clone();
    s.grants.retain(|g| g.until > unix_secs());
    s.settle();
    if store(home, &s).is_ok() && s.allow != before && nginx::master_running(home) {
        if let Err(e) = nginx::apply_locked(home) {
            nginx::log_gateway_error(home, &format!("public gateway update: {e}"));
        }
    }
}

fn close_expired(home: &Path) {
    // Mark an open window whose time ran out closed, under the gateway lock, and reload nginx without it.
    drop_ended(home);
    let Some(share) = load(home) else { return };
    if !share.open || share.active_at(unix_secs()) {
        return;
    }
    let Ok(_lock) = nginx::gateway_lock(home) else { return };
    match load(home) {
        Some(s) if s.open && !s.active_at(unix_secs()) => {
            if store(home, &Share { open: false, ..s }).is_ok() {
                            nginx::log_gateway(home, "public gateway closed: its time ran out");
                if nginx::master_running(home) {
                    if let Err(e) = nginx::apply_locked(home) {
                        nginx::log_gateway_error(home, &format!("public gateway close: {e}"));
                    }
                }
            }
        }
        _ => {}
    }
}

pub fn cert_path(home: &Path) -> PathBuf {
    nginx::nginx_prefix(home).join("certs/public.crt")
}

fn ensure_cert(home: &Path, ip: Option<IpAddr>) -> Result<(), String> {
    // A self-signed certificate for the public listener only, naming the public address so a client can verify it with --cacert.
    let dir = nginx::nginx_prefix(home).join("certs");
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let crt = dir.join("public.crt");
    let key = dir.join("public.key");
    let mut names = vec!["DNS:localhost".to_string(), "IP:127.0.0.1".to_string()];
    if let Some(ip) = ip {
        names.insert(0, format!("IP:{ip}"));
    }
    let san = format!("subjectAltName={}", names.join(","));
    let stamp = format!("{san}\nleaf");
    let san_file = dir.join("public_sans.txt");
    let current = fs::read_to_string(&san_file).map(|s| s.trim() == stamp).unwrap_or(false);
    if current && crt.is_file() && key.is_file() {
        return Ok(());
    }
    nginx::self_signed(&key, &crt, true, 30, "goethe-public", &[&san, "basicConstraints=critical,CA:FALSE"])
        .map_err(|_| "openssl could not create the public gateway certificate".to_string())?;
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(&crt, fs::Permissions::from_mode(0o644)).map_err(|e| e.to_string())?;
    nginx::write_private(&san_file, stamp.as_bytes())
}

fn fingerprint(crt: &Path) -> Option<String> {
    let out = Command::new("openssl").args(["x509", "-noout", "-fingerprint", "-sha256", "-in"]).arg(crt).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.trim().split_once('=').map(|(_, f)| f.to_string())
}

pub fn nginx_block(home: &Path, modern_http2: bool, max_body_mb: u64) -> String {
    // The public server for nginx while a window is open, else nothing: its own port, certificate, limits and upstream (the goethed public socket).
    let Some(share) = active(home) else { return String::new() };
    let dir = nginx::nginx_prefix(home);
    if !cert_path(home).is_file() && ensure_cert(home, None).is_err() {
        return String::new();
    }
    let mut allow: String = share.allow.iter().filter_map(|c| security::canonical_cidr(c).ok()).map(|c| format!("    allow {c};\n")).collect();
    allow.push_str("    deny all;\n");
    const TMPL: &str = include_str!("../../../nginx/goethe_public.conf.tmpl");
    TMPL.replace("{{PORT}}", &share.port.to_string())
        .replace("{{HTTP2_LISTEN}}", if modern_http2 { "" } else { " http2" })
        .replace("{{HTTP2_ON}}", if modern_http2 { "    http2 on;" } else { "" })
        .replace("{{CERT}}", &nginx::quote(&dir.join("certs/public.crt")))
        .replace("{{KEY}}", &nginx::quote(&dir.join("certs/public.key")))
        .replace("{{ALLOW}}", &allow)
        .replace("{{RATE_PER_IP}}", &share_rate(home).to_string())
        .replace("{{BURST_PER_IP}}", &(2 * share_rate(home)).to_string())
        .replace("{{CONN_PER_IP}}", &share_conn(home).to_string())
        .replace("{{MAX_BODY_MB}}", &max_body_mb.to_string())
        .replace("{{PUBLIC_MUX}}", &nginx::quote(Path::new(&format!("unix:{}", public_mux_socket(home).display()))))
}

pub fn status_line(home: &Path) -> Option<String> {
    // One line for goethe status while a window is open.
    let s = active(home)?;
    Some(format!("Public gateway: open until {} on port {} ({} key{})", local_time(s.until), s.port, s.keys.len(), if s.keys.len() == 1 { "" } else { "s" }))
}

pub fn local_time(ts: i64) -> String {
    chrono::DateTime::from_timestamp(ts, 0)
        .map(|d| d.with_timezone(&chrono::Local).format("%a %d %b %H:%M").to_string())
        .unwrap_or_default()
}

pub fn key_names(home: &Path, ids: &[String]) -> Vec<String> {
    let rows = key_rows(home).unwrap_or_default();
    ids.iter().map(|id| rows.iter().find(|r| r.0 == *id).map(|r| r.1.clone()).unwrap_or_else(|| format!("{id} (deleted)"))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn share(open: bool, until: i64) -> Share {
        Share { open, until, port: 8444, keys: vec!["k1".into()], allow: vec!["203.0.113.0/24".into()], grants: vec![] }
    }

    fn granted(grants: &[(&str, i64, &[&str])]) -> Share {
        let mut s = Share { open: true, port: 8444, ..Share::default() };
        s.grants = grants.iter().map(|(k, u, a)| Grant { key: k.to_string(), until: *u, allow: a.iter().map(|c| c.to_string()).collect() }).collect();
        s.settle();
        s
    }

    #[test]
    fn two_keys_share_one_gateway_and_one_that_ended_is_refused_while_the_other_works() {
        let ip = |s: &str| Some(s.parse::<IpAddr>().unwrap());
        let s = granted(&[("k1", 1000, &["203.0.113.0/24"]), ("k4", 5000, &["203.0.113.0/24"])]);
        assert_eq!((s.until, s.keys.clone()), (5000, vec!["k1".to_string(), "k4".to_string()]), "open until the last key ends");
        assert_eq!(decide(Some(&s), "k1", false, 999, ip("203.0.113.1")), Access::Allowed);
        assert_eq!(decide(Some(&s), "k1", false, 1000, ip("203.0.113.1")), Access::KeyExpired);
        assert_eq!(decide(Some(&s), "k4", false, 1000, ip("203.0.113.1")), Access::Allowed, "the gateway stays open for the other key");
        assert_eq!(decide(Some(&s), "k4", false, 5000, ip("203.0.113.1")), Access::Closed);
    }

    #[test]
    fn each_key_is_limited_to_its_own_addresses() {
        let ip = |s: &str| Some(s.parse::<IpAddr>().unwrap());
        let s = granted(&[("k1", 1000, &["203.0.113.0/24"]), ("k4", 1000, &["198.51.100.7/32"])]);
        assert_eq!(s.allow, vec!["203.0.113.0/24".to_string(), "198.51.100.7/32".to_string()], "nginx admits the union");
        assert_eq!(decide(Some(&s), "k1", false, 0, ip("203.0.113.9")), Access::Allowed);
        assert_eq!(decide(Some(&s), "k1", false, 0, ip("198.51.100.7")), Access::NotFromHere, "not the other key's address");
        assert_eq!(decide(Some(&s), "k1", false, 0, None), Access::NotFromHere, "no address, no entry");
        let mixed = granted(&[("k1", 1000, &["203.0.113.0/24"]), ("k4", 1000, &[])]);
        assert_eq!(mixed.allow, vec!["203.0.113.0/24".to_string()], "empty allowlists do not open nginx to the world");
        assert_eq!(decide(Some(&mixed), "k1", false, 0, ip("8.8.8.8")), Access::NotFromHere);
        assert_eq!(decide(Some(&mixed), "k4", false, 0, ip("8.8.8.8")), Access::NotFromHere);
        assert!(cidr_contains("2001:db8::/32", "2001:db8::1".parse().unwrap()) && !cidr_contains("2001:db8::/32", "2001:db9::1".parse().unwrap()));
    }

    #[test]
    fn durations_are_minutes_hours_or_days_within_bounds() {
        assert_eq!(parse_duration("30m"), Ok(1800));
        assert_eq!(parse_duration("5h"), Ok(18_000));
        assert_eq!(parse_duration("1D"), Ok(86_400));
        assert_eq!(parse_duration("7d"), Ok(MAX_SECS));
        for bad in ["", "h", "0m", "8d", "59s", "1.5h", "-1h", "1w", "10000000000000000d", "5 h", "5é", "é"] {
            assert!(parse_duration(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn only_listed_non_admin_keys_pass_while_the_window_is_open() {
        let ip = |s: &str| Some(s.parse::<IpAddr>().unwrap());
        let s = share(true, 1000);
        assert_eq!(decide(Some(&s), "k1", false, 999, ip("203.0.113.1")), Access::Allowed);
        assert_eq!(decide(Some(&s), "k1", false, 1000, ip("203.0.113.1")), Access::Closed, "closes at the end second");
        assert_eq!(decide(Some(&s), "k2", false, 999, ip("203.0.113.1")), Access::KeyNotShared);
        assert_eq!(decide(Some(&s), "k1", true, 999, ip("203.0.113.1")), Access::KeyNotShared, "admin keys never");
        assert_eq!(decide(Some(&share(false, 5000)), "k1", false, 999, ip("203.0.113.1")), Access::Closed, "shut down");
        assert_eq!(decide(None, "k1", false, 0, ip("203.0.113.1")), Access::Closed);
    }

    fn home_with_keys() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("nginx")).unwrap();
        let con = db::open(dir.path()).unwrap();
        for (id, name, admin, exp, scope) in [
            ("k1", "app", 0, None, Some("org/m")),
            ("k2", "root", 1, None, None),
            ("k3", "old", 0, Some("2020-01-01"), Some("org/m")),
            ("k4", "wide", 0, None, None),
            ("k5", "dup", 0, None, Some("org/m")),
            ("k6", "dup", 0, None, Some("org/m")),
        ] {
            con.execute(
                "INSERT INTO api_keys (key_id, name, token_hash, raw_prefix, is_admin, expires_at, model_scope, token) VALUES (?1, ?2, ?1, 'p', ?3, ?4, ?5, 'tok')",
                rusqlite::params![id, name, admin, exp, scope],
            )
            .unwrap();
        }
        dir
    }

    #[test]
    fn keys_resolve_by_name_or_id_and_unsafe_ones_are_refused() {
        let dir = home_with_keys();
        let h = dir.path();
        let mut w = Vec::new();
        let got = resolve_keys(h, &["app".into(), "k1".into()], unix_secs() + 60, &mut w).unwrap();
        assert_eq!(got.len(), 1, "the same key twice counts once");
        assert!(w.is_empty());
        assert!(got[0].token.is_none(), "no secret is read back");
        assert!(resolve_keys(h, &["root".into()], unix_secs() + 60, &mut w).err().unwrap().contains("admin"));
        assert!(resolve_keys(h, &["old".into()], unix_secs() + 60, &mut w).err().unwrap().contains("expired"));
        assert!(resolve_keys(h, &["dup".into()], unix_secs() + 60, &mut w).err().unwrap().contains("several"));
        assert!(resolve_keys(h, &["nope".into()], unix_secs() + 60, &mut w).err().unwrap().contains("app"));
        resolve_keys(h, &["wide".into()], unix_secs() + 60, &mut w).unwrap();
        assert!(w.iter().any(|x| x.contains("every model")));
    }

    #[test]
    fn a_newly_shared_key_gets_a_new_secret_and_one_already_shared_keeps_its_own() {
        let dir = home_with_keys();
        let h = dir.path();
        let con = db::open(h).unwrap();
        let stored = |id: &str| con.query_row("SELECT token_hash, token FROM api_keys WHERE key_id = ?1", [id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Option<String>>(1)?))).unwrap();
        let mut w = Vec::new();
        let mut keys = resolve_keys(h, &["app".into(), "wide".into()], unix_secs() + 60, &mut w).unwrap();
        renew_secrets(h, &mut keys, &["k4".to_string()], &mut w);
        let token = keys[0].token.clone().expect("app is newly shared");
        assert_eq!(stored("k1"), (security::token_hash(&token), None), "only the new secret's hash is stored");
        assert!(keys[1].token.is_none() && stored("k4").0 == "k4", "wide was already shared, so its client keeps working");
        assert!(w.iter().any(|x| x.contains("old one no longer works")), "{w:?}");
    }

    #[test]
    fn no_block_is_rendered_without_an_open_window() {
        let dir = home_with_keys();
        let h = dir.path();
        assert_eq!(nginx_block(h, true, 64), "");
        store(h, &share(true, unix_secs() - 1)).unwrap();
        assert_eq!(nginx_block(h, true, 64), "", "expired");
        store(h, &share(false, unix_secs() + 600)).unwrap();
        assert_eq!(nginx_block(h, true, 64), "", "shut down");
        store(h, &Share { allow: vec!["203.0.113.7/32".into()], ..share(true, unix_secs() + 600) }).unwrap();
        let block = nginx_block(h, true, 64);
        assert!(block.contains("listen 0.0.0.0:8444 ssl"), "{block}");
        assert!(block.contains(&format!("server \"unix:{}\";", public_mux_socket(h).display())), "{block}");
        assert!(block.contains("allow 203.0.113.7/32;\n    deny all;"));
        assert!(block.contains("public.crt"));
    }

    #[test]
    fn nginx_block_always_denies_all_even_with_an_empty_allowlist() {
        let dir = home_with_keys();
        let h = dir.path();
        store(h, &Share { allow: vec![], ..share(true, unix_secs() + 600) }).unwrap();
        let block = nginx_block(h, true, 64);
        assert!(block.contains("deny all;"), "{block}");
        assert!(!block.contains("allow "), "empty allowlist must not admit the world: {block}");
    }

    #[test]
    fn an_edited_share_file_cannot_write_nginx_directives() {
        let dir = home_with_keys();
        let h = dir.path();
        store(h, &Share { allow: vec!["198.51.100.7/32; include /etc/passwd".into(), "198.51.100.8".into()], ..share(true, unix_secs() + 600) }).unwrap();
        let block = nginx_block(h, true, 64);
        assert!(!block.contains("include") && block.contains("allow 198.51.100.8/32;\n    deny all;"), "{block}");
    }

    #[test]
    fn share_allow_prefers_cli_then_kept_never_gateway() {
        assert_eq!(share_allow(&["203.0.113.9/32".into()], false, &["198.51.100.1/32".into()]), vec!["203.0.113.9/32".to_string()]);
        assert_eq!(share_allow(&[], true, &["198.51.100.1/32".into()]), vec!["198.51.100.1/32".to_string()]);
        assert!(share_allow(&[], false, &[]).is_empty(), "VPN/API allowlist must not open public share");
        assert!(share_allow(&[], true, &[]).is_empty());
    }

    #[test]
    fn need_client_allow_names_the_client_not_this_machine() {
        let dir = tempfile::tempdir().unwrap();
        let msg = need_client_allow(dir.path());
        assert!(msg.contains("their phone or laptop") && msg.contains("--allow"), "{msg}");
        assert!(msg.contains("dangerously-allow-all-inbound"), "{msg}");
        assert!(!msg.contains("goethe api allow"), "{msg}");
    }

    #[test]
    fn share_refuses_everyone_and_this_machines_public_ip_unless_dangerously() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        assert!(reject_bad_share_cidrs(home, &["0.0.0.0/0".into()]).unwrap_err().contains("dangerously-allow-all-inbound"));
        assert!(reject_bad_share_cidrs(home, &["::/0".into()]).unwrap_err().contains("dangerously-allow-all-inbound"));
        let ip: IpAddr = "203.0.113.50".parse().unwrap();
        std::env::set_var("GOETHE_PUBLIC_IP", ip.to_string());
        let err = reject_bad_share_cidrs(home, &[format!("{ip}/32")]).unwrap_err();
        std::env::remove_var("GOETHE_PUBLIC_IP");
        assert!(err.contains("this machine's public address"), "{err}");
        assert!(reject_bad_share_cidrs(home, &["198.51.100.7/32".into()]).is_ok());

        let all = resolve_share_cidrs(home, &[], false, &[], true).unwrap();
        assert_eq!(all, vec!["0.0.0.0/0".to_string()]);
        assert!(resolve_share_cidrs(home, &["198.51.100.7/32".into()], false, &[], true)
            .unwrap_err()
            .contains("cannot be combined"));
        assert!(resolve_share_cidrs(home, &["0.0.0.0/0".into()], false, &[], false)
            .unwrap_err()
            .contains("dangerously-allow-all-inbound"));
        assert_eq!(
            resolve_share_cidrs(home, &["198.51.100.7/32".into()], false, &[], false).unwrap(),
            vec!["198.51.100.7/32".to_string()]
        );
        assert_eq!(
            resolve_share_cidrs(home, &[], true, &["203.0.113.9/32".into()], false).unwrap(),
            vec!["203.0.113.9/32".to_string()]
        );
        assert!(resolve_share_cidrs(home, &[], false, &[], false).unwrap_err().contains("--allow"));
    }

    #[test]
    fn nginx_block_with_everyone_still_ends_in_deny_all() {
        let dir = home_with_keys();
        let h = dir.path();
        store(h, &Share { allow: vec!["0.0.0.0/0".into()], ..share(true, unix_secs() + 600) }).unwrap();
        let block = nginx_block(h, true, 64);
        assert!(block.contains("allow 0.0.0.0/0;\n    deny all;"), "{block}");
        assert!(block.contains(&format!("rate={DEFAULT_RATE_PER_IP}r/s")), "{block}");
        assert!(block.contains(&format!("limit_conn goethe_pub_conn {DEFAULT_CONN_PER_IP}")), "{block}");
    }

    #[test]
    fn share_limits_follow_env_when_persisted() {
        let dir = home_with_keys();
        let h = dir.path();
        std::env::set_var("GOETHE_SHARE_RATE_PER_IP", "25");
        std::env::set_var("GOETHE_SHARE_CONN_PER_IP", "40");
        persist_share_limits(h);
        std::env::remove_var("GOETHE_SHARE_RATE_PER_IP");
        std::env::remove_var("GOETHE_SHARE_CONN_PER_IP");
        assert_eq!(share_rate(h), 25);
        assert_eq!(share_conn(h), 40);
        store(h, &Share { allow: vec!["203.0.113.7/32".into()], ..share(true, unix_secs() + 600) }).unwrap();
        let block = nginx_block(h, true, 64);
        assert!(block.contains("rate=25r/s"), "{block}");
        assert!(block.contains("burst=50"), "{block}");
        assert!(block.contains("limit_conn goethe_pub_conn 40"), "{block}");
    }

    #[test]
    fn the_router_sees_every_change_at_once() {
        let dir = home_with_keys();
        let h = dir.path();
        let ip = Some("203.0.113.1".parse::<IpAddr>().unwrap());
        assert_eq!(check(h, "k1", false, ip), Access::Closed);
        store(h, &share(true, unix_secs() + 600)).unwrap();
        assert_eq!(check(h, "k1", false, ip), Access::Allowed);
        store(h, &Share { keys: vec!["k4".into()], ..share(true, unix_secs() + 600) }).unwrap();
        assert_eq!(check(h, "k1", false, ip), Access::KeyNotShared, "a replaced key stops at the next request");
        store(h, &Share { open: false, ..share(true, unix_secs() + 600) }).unwrap();
        assert_eq!(check(h, "k1", false, ip), Access::Closed);
        fs::remove_file(path(h)).unwrap();
        assert_eq!(check(h, "k1", false, ip), Access::Closed);
    }

    #[test]
    fn tick_closes_an_expired_window_and_keeps_the_keys() {
        let dir = home_with_keys();
        let h = dir.path();
        store(h, &share(true, unix_secs() - 5)).unwrap();
        tick(h);
        let s = load(h).unwrap();
        assert!(!s.open);
        assert_eq!(s.keys, vec!["k1".to_string()]);
        store(h, &share(true, unix_secs() + 600)).unwrap();
        tick(h);
        assert!(load(h).unwrap().open, "an open window stays open");
    }

    #[test]
    fn the_public_certificate_names_the_address_and_works_with_this_openssl() {
        let dir = tempfile::tempdir().unwrap();
        ensure_cert(dir.path(), Some("203.0.113.9".parse().unwrap())).unwrap();
        let out = Command::new("openssl").args(["x509", "-noout", "-text", "-in"]).arg(cert_path(dir.path())).output().unwrap();
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(text.contains("203.0.113.9") && text.contains("CA:FALSE"), "{text}");
    }
}
