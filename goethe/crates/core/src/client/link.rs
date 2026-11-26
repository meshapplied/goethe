// Optional platform link: URL policy, the credentials file, the heartbeat state file and the daemon change signal.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

pub const DEFAULT_PLATFORM_URL: &str = "https://goethe.sh";
pub const URL_SETTING: &str = "GOETHE_PLATFORM_URL";
const MAX_CREDENTIALS_BYTES: u64 = 16 * 1024;
const MAX_STATE_BYTES: u64 = 64 * 1024;
const FINGERPRINT_SALT: &str = "goethe";

#[derive(Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Secret(String);

impl Secret {
    pub fn new(value: impl Into<String>) -> Secret {
        Secret(value.into())
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

impl std::fmt::Display for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Workspace {
    pub id: String,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Credentials {
    pub platform_url: String,
    pub node_id: String,
    pub workspace: Workspace,
    pub token: Secret,
    pub linked_at: String,
}

pub fn platform_url() -> Result<String, String> {
    // The GOETHE_PLATFORM_URL setting, else the default, after the URL policy.
    check_url(&crate::paths::setting(URL_SETTING).unwrap_or_else(|| DEFAULT_PLATFORM_URL.to_string()))
}

pub fn check_url(raw: &str) -> Result<String, String> {
    // HTTPS with a host and no credentials, query or fragment; plain HTTP only for loopback and *.localhost.
    let url = raw.trim().trim_end_matches('/');
    if url.contains(['?', '#']) {
        return Err(format!("console URL {url:?} must not have a query or fragment"));
    }
    check_link(url)
}

pub fn check_link(raw: &str) -> Result<String, String> {
    // A link under the same policy as the platform URL, query allowed: https, or http on loopback only.
    let url = raw.trim();
    if url.is_empty() || url.len() > 1024 || url.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return Err(format!("{raw:?} is not a usable console URL"));
    }
    let (scheme, rest) = url
        .split_once("://")
        .ok_or_else(|| format!("console URL {url:?} needs a scheme, like {DEFAULT_PLATFORM_URL}"))?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.contains('@') {
        return Err(format!("console URL {url:?} must not carry a user name or password"));
    }
    let host = url_host(authority).ok_or_else(|| format!("console URL {url:?} has no valid host"))?;
    match scheme.to_ascii_lowercase().as_str() {
        "https" => {}
        "http" if is_loopback_host(&host) => {}
        "http" => return Err(format!("console URL {url:?} must use https (plain http is only allowed for localhost)")),
        other => return Err(format!("console URL {url:?} has scheme {other:?}; use https")),
    }
    Ok(format!("{}://{rest}", scheme.to_ascii_lowercase()))
}

pub fn check_dev_link(raw: &str) -> Result<String, String> {
    // A link from a local development platform: also plain http to a private, CGNAT (Tailscale) or loopback address.
    if let Ok(ok) = check_link(raw) {
        return Ok(ok);
    }
    let url = raw.trim();
    let rest = url.strip_prefix("http://").ok_or_else(|| format!("{url:?} is not an http or https link"))?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.contains('@') || url.chars().any(|c| c.is_control() || c.is_whitespace()) || url.len() > 1024 {
        return Err(format!("{url:?} is not a usable link"));
    }
    let host = url_host(authority).ok_or_else(|| format!("{url:?} has no valid host"))?;
    let private = match host.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(v4)) => v4.is_private() || v4.is_loopback() || (v4.octets()[0] == 100 && (64..128).contains(&v4.octets()[1])),
        Ok(std::net::IpAddr::V6(v6)) => v6.is_loopback(),
        Err(_) => false,
    };
    if private {
        Ok(url.to_string())
    } else {
        Err(format!("{url:?} must use https"))
    }
}

fn url_host(authority: &str) -> Option<String> {
    // The host of an authority (brackets kept off IPv6), with a numeric port if one is given.
    let (host, port) = if let Some(v6) = authority.strip_prefix('[') {
        let (h, tail) = v6.split_once(']')?;
        h.parse::<std::net::Ipv6Addr>().ok()?;
        (h.to_string(), tail.strip_prefix(':'))
    } else {
        match authority.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), Some(p)),
            None => (authority.to_string(), None),
        }
    };
    if let Some(p) = port {
        p.parse::<u16>().ok().filter(|p| *p > 0)?;
    }
    let valid = !host.is_empty()
        && host.len() <= 253
        && host.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | ':'))
        && !host.starts_with(['.', '-']);
    valid.then(|| host.to_ascii_lowercase())
}

fn is_loopback_host(host: &str) -> bool {
    if host == "localhost" || host.ends_with(".localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

pub fn platform_dir(home: &Path) -> PathBuf {
    home.join("platform")
}

pub fn credentials_path(home: &Path) -> PathBuf {
    platform_dir(home).join("credentials")
}

fn install_key_path(home: &Path) -> PathBuf {
    platform_dir(home).join("install_key")
}

pub fn install_key(home: &Path) -> Result<Secret, String> {
    // This install's own secret, made on the first login and kept through logout and updates, so the console links it again as the same node.
    let path = install_key_path(home);
    if let Some(key) = read_private(&path, 256, true)?.map(|t| t.trim().to_string()).filter(|k| valid_token(k)) {
        return Ok(Secret::new(key));
    }
    ensure_platform_dir(home)?;
    let key = format!("gik_{}", crate::security::random_hex(32));
    write_private(&path, key.as_bytes())?;
    Ok(Secret::new(key))
}

pub fn node_key(home: &Path) -> Result<goethe_proto::InstallKey, String> {
    // This install's Ed25519 key (pkcs8, hex, 0600), made once like the install key: its public half goes up when linking, and it proves the node when it connects to goethe-server.
    let path = platform_dir(home).join("node_key");
    let parse = |t: &str| {
        let bytes: Option<Vec<u8>> = t.len().is_multiple_of(2).then(|| (0..t.len()).step_by(2).map(|i| u8::from_str_radix(&t[i..i + 2], 16).ok()).collect()).flatten();
        bytes.and_then(|b| goethe_proto::InstallKey::from_pkcs8(&b))
    };
    if let Some(key) = read_private(&path, 1024, true)?.and_then(|t| parse(t.trim())) {
        return Ok(key);
    }
    ensure_platform_dir(home)?;
    let pkcs8 = goethe_proto::InstallKey::generate().ok_or("could not make the node key")?;
    write_private(&path, crate::sha::hex(&pkcs8).as_bytes())?;
    goethe_proto::InstallKey::from_pkcs8(&pkcs8).ok_or_else(|| "could not read the node key back".to_string())
}

pub(crate) fn state_path(home: &Path) -> PathBuf {
    platform_dir(home).join("platform_state.json")
}

fn legacy_paths(home: &Path) -> [(PathBuf, PathBuf); 2] {
    [(home.join("credentials"), credentials_path(home)), (home.join("platform_state.json"), state_path(home))]
}

pub fn check_home(home: &Path) -> Result<(), String> {
    // GOETHE_HOME must be ours and not writable by every user without the sticky bit; group-writable is normal with user private groups.
    let m = match fs::metadata(home) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("cannot read {}: {e}", home.display())),
    };
    let uid = unsafe { libc::geteuid() };
    if m.uid() != uid {
        return Err(format!("{} belongs to uid {}, not to you (uid {uid}), so goethe will not keep credentials there", home.display(), m.uid()));
    }
    if m.mode() & 0o002 != 0 && m.mode() & 0o1000 == 0 {
        return Err(format!(
            "{} can be written by every user (mode {:o}), so goethe will not keep credentials there; fix it with: chmod o-w {}",
            home.display(),
            m.mode() & 0o7777,
            home.display()
        ));
    }
    Ok(())
}

fn check_private_dir(dir: &Path, fix: bool) -> Result<bool, String> {
    // The platform directory: a real directory owned by this user and closed to group and others (fixed to 0700 when allowed); false when missing.
    let m = match fs::symlink_metadata(dir) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(format!("cannot read {}: {e}", dir.display())),
    };
    if !m.file_type().is_dir() {
        return Err(format!("{} is not a directory (a symlink or a file); remove it and run goethe login again", dir.display()));
    }
    let uid = unsafe { libc::geteuid() };
    if m.uid() != uid {
        return Err(format!("{} belongs to uid {}, not to you (uid {uid}); remove it, or fix it with: chown {uid} {}", dir.display(), m.uid(), dir.display()));
    }
    if m.mode() & 0o077 != 0 {
        if !fix {
            return Err(format!("{} can be read by other users (mode {:o}); fix it with: chmod 700 {}", dir.display(), m.mode() & 0o777, dir.display()));
        }
        fs::set_permissions(dir, fs::Permissions::from_mode(0o700)).map_err(|e| format!("cannot chmod 700 {}: {e}", dir.display()))?;
    }
    Ok(true)
}

fn ensure_platform_dir(home: &Path) -> Result<PathBuf, String> {
    // GOETHE_HOME/platform at 0700, created or tightened as needed; GOETHE_HOME itself is only checked, never changed.
    check_home(home)?;
    if !home.exists() {
        fs::DirBuilder::new().recursive(true).mode(0o700).create(home).map_err(|e| format!("cannot create {}: {e}", home.display()))?;
    }
    let dir = platform_dir(home);
    if !check_private_dir(&dir, true)? {
        fs::DirBuilder::new().mode(0o700).create(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        check_private_dir(&dir, true)?;
    }
    Ok(dir)
}

fn migrate(home: &Path) {
    // Move credentials and state from GOETHE_HOME (where early builds kept them) into the platform directory, once.
    for (old, new) in legacy_paths(home) {
        let pending = fs::symlink_metadata(&old).is_ok() && fs::symlink_metadata(&new).is_err();
        if pending && read_private(&old, MAX_CREDENTIALS_BYTES.max(MAX_STATE_BYTES), false).is_ok() && ensure_platform_dir(home).is_ok() {
            let _ = fs::rename(&old, &new);
        }
    }
}

fn read_private(path: &Path, cap: u64, in_private_dir: bool) -> Result<Option<String>, String> {
    // A regular file owned by this user and closed to group and others, in a directory that is too, or None when it does not exist.
    if in_private_dir {
        if let Some(dir) = path.parent() {
            if !check_private_dir(dir, false)? {
                return Ok(None);
            }
        }
    }
    let meta = match fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(format!("cannot read {}: {e}", path.display())),
    };
    if !meta.file_type().is_file() {
        return Err(format!("{} is not a regular file; remove it and run goethe login again", path.display()));
    }
    let uid = unsafe { libc::geteuid() };
    if meta.uid() != uid {
        return Err(format!(
            "{} belongs to uid {}, not to you (uid {uid}); remove it, or fix it with: chown {uid} {}",
            path.display(),
            meta.uid(),
            path.display()
        ));
    }
    if meta.mode() & 0o077 != 0 {
        return Err(format!(
            "{} can be read by other users (mode {:o}); fix it with: chmod 600 {}",
            path.display(),
            meta.mode() & 0o777,
            path.display()
        ));
    }
    if meta.len() > cap {
        return Err(format!("{} is larger than {cap} bytes; remove it and run goethe login again", path.display()));
    }
    fs::read_to_string(path).map(Some).map_err(|e| format!("cannot read {}: {e}", path.display()))
}

pub fn write_private(path: &Path, data: &[u8]) -> Result<(), String> {
    // Only into the private platform directory.
    let dir = path.parent().ok_or_else(|| format!("{} has no parent directory", path.display()))?;
    if !check_private_dir(dir, false)? {
        return Err(format!("{} does not exist", dir.display()));
    }
    crate::paths::write_private(path, data).map_err(|e| format!("cannot replace {}: {e}", path.display()))
}

pub fn load_credentials(home: &Path) -> Result<Option<Credentials>, String> {
    // The saved link, or None when this node was never linked; a damaged or exposed file is an error, never silently used.
    migrate(home);
    let (legacy, current) = legacy_paths(home)[0].clone();
    let old_only = fs::symlink_metadata(&current).is_err() && fs::symlink_metadata(&legacy).is_ok();
    let path = if old_only { legacy } else { current };
    let Some(text) = read_private(&path, MAX_CREDENTIALS_BYTES, !old_only)? else {
        return Ok(None);
    };
    let creds: Credentials = serde_json::from_str(&text)
        .map_err(|_| format!("{} is damaged; remove it and run goethe login again", path.display()))?;
    validate_credentials(&creds).map_err(|e| format!("{}: {e}; remove it and run goethe login again", path.display()))?;
    Ok(Some(creds))
}

fn validate_credentials(c: &Credentials) -> Result<(), String> {
    check_url(&c.platform_url)?;
    valid_id(&c.node_id).then_some(()).ok_or("the node id is not valid")?;
    valid_token(c.token.expose()).then_some(()).ok_or("the node token is not valid")?;
    Ok(())
}

pub fn valid_id(s: &str) -> bool {
    !s.is_empty() && s.len() <= 128 && s.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'))
}

pub fn valid_token(s: &str) -> bool {
    s.len() >= 16 && s.len() <= 512 && s.chars().all(|c| c.is_ascii_graphic())
}

pub fn clean_text(s: &str, max: usize) -> String {
    // Display text from the server without control characters, at most max characters.
    s.chars().filter(|c| !c.is_control()).take(max).collect::<String>().trim().to_string()
}

pub fn save_credentials(home: &Path, c: &Credentials) -> Result<(), String> {
    validate_credentials(c)?;
    let data = serde_json::to_vec_pretty(c).map_err(|e| e.to_string())?;
    ensure_platform_dir(home)?;
    write_private(&credentials_path(home), &data)
}

pub fn remove_credentials(home: &Path) -> Result<bool, String> {
    // Delete the credentials and the heartbeat state; true when there was a link to remove.
    let mut removed = false;
    for (old, new) in legacy_paths(home) {
        for path in [new, old] {
            match fs::remove_file(&path) {
                Ok(()) => removed |= path.ends_with("credentials"),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(format!("cannot remove {}: {e}", path.display())),
            }
        }
    }
    let _ = fs::remove_dir(platform_dir(home));
    Ok(removed)
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlatformState {
    #[serde(default)]
    pub link: String,
    #[serde(default)]
    pub revoked_at: Option<String>,
    #[serde(default)]
    pub last_ok_at: Option<String>,
    #[serde(default)]
    pub last_error_at: Option<String>,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub failures: u32,
    #[serde(default)]
    pub public_ip: Option<String>,
    #[serde(default)]
    pub public_ip_at: Option<String>,
}

pub fn link_key(c: &Credentials) -> String {
    // Node id plus a short hash of the token, so state from an earlier login never applies to a new one with the same node id.
    format!("{}:{}", c.node_id, &fingerprint_of(c.token.expose())[..16])
}

pub fn load_state(home: &Path, link: &str) -> PlatformState {
    // The heartbeat state for this link; a state left by an older link, or an unreadable one, counts as fresh.
    read_private(&state_path(home), MAX_STATE_BYTES, true)
        .ok()
        .flatten()
        .and_then(|t| serde_json::from_str::<PlatformState>(&t).ok())
        .filter(|s| s.link == link)
        .unwrap_or_else(|| PlatformState { link: link.to_string(), ..PlatformState::default() })
}

pub fn save_state(home: &Path, state: &PlatformState) -> Result<(), String> {
    let data = serde_json::to_vec_pretty(state).map_err(|e| e.to_string())?;
    ensure_platform_dir(home)?;
    write_private(&state_path(home), &data)
}

pub fn fingerprint() -> String {
    // sha256 of the machine id and a fixed salt, so the raw id never leaves; a container's id kept in its home comes next (it outlives the container), then the hostname.
    let root = crate::host::sysroot();
    let id = ["etc/machine-id", "var/lib/dbus/machine-id"]
        .iter()
        .map(|p| root.join(p))
        .chain([crate::paths::goethe_home().join(HOME_MACHINE_ID)])
        .filter_map(|p| fs::read_to_string(p).ok())
        .map(|s| s.trim().to_string())
        .find(|s| !s.is_empty())
        .unwrap_or_else(|| format!("hostname:{}", hostname()));
    fingerprint_of(&id)
}

const HOME_MACHINE_ID: &str = "machine-id";

pub fn ensure_home_machine_id(home: &Path) -> Result<(), String> {
    // A container has no /etc/machine-id of its own: one is made once in its home, so recreating the container keeps the same console node.
    let path = home.join(HOME_MACHINE_ID);
    if path.is_file() {
        return Ok(());
    }
    write_private(&path, crate::security::random_hex(16).as_bytes())
}

fn fingerprint_of(machine_id: &str) -> String {
    crate::sha::digest_hex(format!("{machine_id}{FINGERPRINT_SALT}").as_bytes())
}

pub fn hostname() -> String {
    // The kernel's hostname (/proc on Linux, gethostname elsewhere), without macOS's .local suffix.
    let mut raw = fs::read_to_string(crate::host::sysroot().join("proc/sys/kernel/hostname")).unwrap_or_default();
    if raw.trim().is_empty() {
        let mut buf = [0u8; 256];
        if unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) } == 0 {
            raw = String::from_utf8_lossy(&buf[..buf.iter().position(|&b| b == 0).unwrap_or(buf.len())]).into_owned();
        }
    }
    let name = host_label(raw.trim().trim_end_matches(".local"));
    if name.is_empty() {
        "unknown".to_string()
    } else {
        name
    }
}

fn host_label(raw: &str) -> String {
    // The platform takes letters, digits, dot, dash and underscore; anything else (a space, an apostrophe) becomes a dash.
    let mapped: String = raw.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '-' }).collect();
    mapped.trim_matches('-').chars().take(253).collect()
}

fn public_ip_url(home: &Path) -> Result<String, String> {
    // GOETHE_PUBLIC_IP_URL, else /api/ip on the linked platform, else on the configured or default one.
    if let Some(u) = crate::paths::setting("GOETHE_PUBLIC_IP_URL") {
        return Ok(u.trim().to_string());
    }
    let base = match load_credentials(home) {
        Ok(Some(c)) => c.platform_url,
        _ => platform_url()?,
    };
    Ok(format!("{base}/api/ip"))
}

fn parse_ip_answer(url: &str, body: &str) -> Result<std::net::IpAddr, String> {
    // {"ip": "..."} from the platform, {"ip": null} when it sees no public address, or a plain-text address.
    let text = body.trim();
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(text) {
        if let Some(obj) = v.as_object() {
            return match obj.get("ip") {
                Some(serde_json::Value::String(s)) => s.trim().parse().map_err(|_| format!("{url} answered {:?}, not an IP address", clean_text(s, 64))),
                Some(serde_json::Value::Null) => Err(format!("{url} sees no public address for this machine (it reaches out from a private network)")),
                _ => Err(format!("{url} answered JSON without an ip field")),
            };
        }
    }
    text.parse().map_err(|_| format!("{url} answered {:?}, not an IP address", clean_text(text, 64)))
}

pub fn lookup_public_ip(home: &Path) -> Result<std::net::IpAddr, String> {
    // GOETHE_PUBLIC_IP when set; else one GET to public_ip_url (4 s, 1 KB), unless GOETHE_PUBLIC_IP_LOOKUP=0 or offline mode.
    use std::io::Read;
    if let Some(ip) = crate::paths::setting("GOETHE_PUBLIC_IP") {
        return ip.trim().parse().map_err(|_| format!("GOETHE_PUBLIC_IP {ip:?} is not an IP address"));
    }
    if crate::paths::setting("GOETHE_PUBLIC_IP_LOOKUP").is_some_and(|v| v.trim() == "0") || crate::hf::offline_mode() {
        return Err("lookup disabled (GOETHE_PUBLIC_IP_LOOKUP=0 or offline mode; set GOETHE_PUBLIC_IP to name it)".to_string());
    }
    let url = public_ip_url(home)?;
    let resp = crate::net::agent()
        .get(&url)
        .set("Accept", "application/json, text/plain")
        .set("User-Agent", &format!("goethe/{}", env!("CARGO_PKG_VERSION")))
        .timeout(std::time::Duration::from_secs(4))
        .call()
        .map_err(|e| match e {
            ureq::Error::Status(code, _) => format!("lookup at {url} failed (HTTP {code})"),
            ureq::Error::Transport(t) => format!("lookup at {url} failed ({}); no internet access, or outbound HTTPS is blocked", t.kind()),
        })?;
    let mut buf = Vec::new();
    resp.into_reader().take(1024).read_to_end(&mut buf).map_err(|e| format!("lookup at {url} failed ({e})"))?;
    parse_ip_answer(&url, &String::from_utf8_lossy(&buf))
}

pub const CHANGE_STATE: u8 = 1;
pub const CHANGE_LINK: u8 = 2;
static PENDING: AtomicU8 = AtomicU8::new(0);
static HOOK: OnceLock<Box<dyn Fn() + Send + Sync>> = OnceLock::new();

pub fn set_change_hook(hook: Box<dyn Fn() + Send + Sync>) {
    // goethed wakes its heartbeat task through this; without it the change signal is a flag nobody reads.
    let _ = HOOK.set(hook);
}

pub fn notify(kind: u8) {
    // Never blocks: set a flag and wake the heartbeat task if goethed installed one.
    PENDING.fetch_or(kind, Ordering::SeqCst);
    if let Some(h) = HOOK.get() {
        h();
    }
}

pub fn state_changed() {
    notify(CHANGE_STATE);
}

pub fn take_pending() -> u8 {
    PENDING.swap(0, Ordering::SeqCst)
}

pub fn poke_daemon() -> bool {
    // Tell a running goethed the link changed, with one short connection; false when it is not running or too old to know.
    use std::io::{BufRead, BufReader};
    use std::time::Duration;
    let Ok(mut stream) = std::os::unix::net::UnixStream::connect(crate::paths::goethe_sock()) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    if stream.write_all(b"{\"method\":\"platform_changed\",\"params\":null}\n").is_err() {
        return false;
    }
    let mut line = String::new();
    let _ = BufReader::new(stream).read_line(&mut line);
    serde_json::from_str::<serde_json::Value>(line.trim()).is_ok_and(|v| v["ok"] == true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_local_dev_platform_may_link_over_http_to_private_addresses_only() {
        assert!(check_dev_link("http://100.64.1.1:3000/device?code=ABCD-1234").is_ok(), "Tailscale address");
        assert!(check_dev_link("http://192.168.1.5:3000/device").is_ok() && check_dev_link("http://127.0.0.1:3000/x").is_ok());
        assert!(check_dev_link("https://goethe.sh/device?code=A").is_ok());
        assert!(check_dev_link("http://evil.example.com/device").is_err(), "a public host still needs https");
        assert!(check_dev_link("http://8.8.8.8/device").is_err());
        assert!(check_dev_link("http://user@10.0.0.1/x").is_err() && check_dev_link("ftp://10.0.0.1/x").is_err());
        assert!(check_link("http://100.64.1.1:3000/device").is_err(), "the production check is unchanged");
    }

    fn creds() -> Credentials {
        Credentials {
            platform_url: "https://platform.example.com".into(),
            node_id: "node-123".into(),
            workspace: Workspace { id: "ws_1".into(), name: "Lab".into() },
            token: Secret::new("mesh_node_0123456789abcdef0123456789abcdef"),
            linked_at: "2026-11-11T10:00:00Z".into(),
        }
    }

    #[test]
    fn url_policy_wants_https_except_on_loopback() {
        assert_eq!(check_url("https://goethe.sh/").unwrap(), "https://goethe.sh");
        assert_eq!(check_url("HTTPS://p.example.com:8443").unwrap(), "https://p.example.com:8443");
        for ok in ["http://localhost:3000", "http://platform.localhost:3000", "http://127.0.0.1:9", "http://[::1]:80", "https://[2001:db8::1]"] {
            assert!(check_url(ok).is_ok(), "{ok}");
        }
        for bad in [
            "http://goethe.sh",
            "http://10.0.0.1",
            "http://localhost.evil.com",
            "ftp://x.example.com",
            "goethe.sh",
            "https://user:pw@x.example.com",
            "https://x.example.com/?a=1",
            "https://x.example.com/#f",
            "https://",
            "https://x.example.com:0",
            "https://x.example.com:99999",
            "https://exa mple.com",
            "https://x.exa\u{7}mple.com",
            "",
        ] {
            assert!(check_url(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_install_key_is_made_once_private_and_outlives_a_logout() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        let first = install_key(&home).unwrap();
        assert!(first.expose().starts_with("gik_") && first.expose().len() == 68);
        assert_eq!(fs::metadata(install_key_path(&home)).unwrap().mode() & 0o777, 0o600);
        save_credentials(&home, &creds()).unwrap();
        remove_credentials(&home).unwrap();
        assert_eq!(install_key(&home).unwrap().expose(), first.expose(), "logout keeps it");
        let other = dir.path().join("other");
        assert_ne!(install_key(&other).unwrap().expose(), first.expose(), "each home is its own node");
    }

    #[test]
    fn the_node_key_is_made_once_private_and_outlives_a_logout() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        let first = node_key(&home).unwrap();
        assert_eq!(fs::metadata(platform_dir(&home).join("node_key")).unwrap().mode() & 0o777, 0o600);
        save_credentials(&home, &creds()).unwrap();
        remove_credentials(&home).unwrap();
        assert_eq!(node_key(&home).unwrap().public_b64(), first.public_b64(), "logout keeps it");
        assert_ne!(node_key(&dir.path().join("other")).unwrap().public_b64(), first.public_b64(), "each home is its own node");
        fs::write(platform_dir(&home).join("node_key"), "zz").unwrap();
        assert_ne!(node_key(&home).unwrap().public_b64(), first.public_b64(), "a damaged key is made again, never half used");
    }

    #[test]
    fn credentials_round_trip_private_and_atomic() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        assert_eq!(load_credentials(&home).unwrap(), None);
        save_credentials(&home, &creds()).unwrap();
        let meta = fs::metadata(credentials_path(&home)).unwrap();
        assert_eq!(meta.mode() & 0o777, 0o600);
        assert_eq!(fs::metadata(platform_dir(&home)).unwrap().mode() & 0o777, 0o700);
        assert_eq!(load_credentials(&home).unwrap(), Some(creds()));
        let leftovers: Vec<_> = fs::read_dir(platform_dir(&home)).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().contains(".tmp.")).collect();
        assert!(leftovers.is_empty());
        let mut next = creds();
        next.node_id = "node-456".into();
        save_credentials(&home, &next).unwrap();
        assert_eq!(load_credentials(&home).unwrap().unwrap().node_id, "node-456");
        assert!(remove_credentials(&home).unwrap());
        assert!(!remove_credentials(&home).unwrap());
    }

    #[test]
    fn exposed_or_odd_credentials_are_refused_with_the_fix() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        fs::set_permissions(home, fs::Permissions::from_mode(0o700)).unwrap();
        save_credentials(home, &creds()).unwrap();
        let path = credentials_path(home);
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        let err = load_credentials(home).unwrap_err();
        assert!(err.contains("chmod 600") && !err.contains("mesh_node_"), "{err}");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        fs::write(&path, "{not json").unwrap();
        let err = load_credentials(home).unwrap_err();
        assert!(err.contains("damaged"), "{err}");
        fs::remove_file(&path).unwrap();
        let target = home.join("elsewhere");
        fs::write(&target, "{}").unwrap();
        std::os::unix::fs::symlink(&target, &path).unwrap();
        assert!(load_credentials(home).unwrap_err().contains("not a regular file"));
        fs::remove_file(&path).unwrap();
        let mut bad = creds();
        bad.platform_url = "http://evil.example.com".into();
        assert!(save_credentials(home, &bad).is_err());
        fs::write(&path, serde_json::to_string(&bad).unwrap()).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        assert!(load_credentials(home).unwrap_err().contains("https"));
    }

    #[test]
    fn a_group_writable_home_works_and_a_world_writable_one_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o775)).unwrap();
        save_credentials(dir.path(), &creds()).unwrap();
        assert_eq!(load_credentials(dir.path()).unwrap(), Some(creds()));
        assert_eq!(fs::metadata(dir.path()).unwrap().mode() & 0o7777, 0o775, "GOETHE_HOME is never changed");
        assert_eq!(fs::metadata(platform_dir(dir.path())).unwrap().mode() & 0o777, 0o700);
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o1777)).unwrap();
        assert!(save_credentials(dir.path(), &creds()).is_ok(), "sticky world-writable is allowed");
        let other = tempfile::tempdir().unwrap();
        fs::set_permissions(other.path(), fs::Permissions::from_mode(0o777)).unwrap();
        let err = save_credentials(other.path(), &creds()).unwrap_err();
        assert!(err.contains("chmod o-w"), "{err}");
        assert!(!platform_dir(other.path()).exists());
    }

    #[test]
    fn the_platform_directory_is_tightened_or_refused() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        fs::create_dir(platform_dir(home)).unwrap();
        fs::set_permissions(platform_dir(home), fs::Permissions::from_mode(0o755)).unwrap();
        save_credentials(home, &creds()).unwrap();
        assert_eq!(fs::metadata(platform_dir(home)).unwrap().mode() & 0o777, 0o700, "a loose directory is chmodded 0700");
        fs::set_permissions(platform_dir(home), fs::Permissions::from_mode(0o750)).unwrap();
        assert!(load_credentials(home).unwrap_err().contains("chmod 700"), "reading never loosens the check");
        fs::set_permissions(platform_dir(home), fs::Permissions::from_mode(0o700)).unwrap();
        assert!(remove_credentials(home).unwrap());
        assert!(!platform_dir(home).exists());
        let target = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(target.path(), platform_dir(home)).unwrap();
        assert!(save_credentials(home, &creds()).unwrap_err().contains("not a directory"));
        assert!(load_credentials(home).unwrap_err().contains("not a directory"));
    }

    #[test]
    fn credentials_from_the_old_place_move_once() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        fs::set_permissions(home, fs::Permissions::from_mode(0o775)).unwrap();
        for (name, body) in [("credentials", serde_json::to_string(&creds()).unwrap()), ("platform_state.json", r#"{"link":"x"}"#.to_string())] {
            fs::write(home.join(name), body).unwrap();
            fs::set_permissions(home.join(name), fs::Permissions::from_mode(0o600)).unwrap();
        }
        assert_eq!(load_credentials(home).unwrap(), Some(creds()));
        assert!(!home.join("credentials").exists() && !home.join("platform_state.json").exists());
        assert!(credentials_path(home).exists() && state_path(home).exists());
        assert_eq!(load_state(home, "x").link, "x");
    }

    #[test]
    fn the_token_never_prints() {
        let c = creds();
        let shown = format!("{c:?} {} {}", c.token, c.token.clone().expose().len());
        assert!(!shown.contains("mesh_node_"), "{shown}");
        assert!(serde_json::to_string(&c).unwrap().contains("mesh_node_"), "the file itself keeps it");
    }

    #[test]
    fn state_belongs_to_one_link() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        fs::set_permissions(home, fs::Permissions::from_mode(0o700)).unwrap();
        let s = PlatformState { link: "a".into(), revoked_at: Some("t".into()), ..Default::default() };
        save_state(home, &s).unwrap();
        assert_eq!(load_state(home, "a"), s);
        assert_eq!(load_state(home, "b"), PlatformState { link: "b".into(), ..Default::default() }, "another link starts fresh");
        let mut again = creds();
        again.token = Secret::new("mesh_node_ffffffffffffffffffffffffffffffff");
        assert_ne!(link_key(&creds()), link_key(&again), "a new token with the same node id is a new link");
        assert!(!link_key(&creds()).contains("mesh_node_"));
    }

    #[test]
    fn ip_answers_in_json_or_plain_text() {
        let u = "https://p.example.com/api/ip";
        assert_eq!(parse_ip_answer(u, r#"{"ip":"203.0.113.5"}"#).unwrap().to_string(), "203.0.113.5");
        assert_eq!(parse_ip_answer(u, "198.51.100.7\n").unwrap().to_string(), "198.51.100.7");
        assert_eq!(parse_ip_answer(u, "2001:db8::1").unwrap().to_string(), "2001:db8::1");
        assert!(parse_ip_answer(u, r#"{"ip":null}"#).unwrap_err().contains("no public address"));
        assert!(parse_ip_answer(u, r#"{"error":"rate_limited"}"#).unwrap_err().contains("without an ip"));
        assert!(parse_ip_answer(u, "<html>").unwrap_err().contains("not an IP address"));
    }

    #[test]
    fn the_ip_lookup_asks_the_linked_platform() {
        let dir = tempfile::tempdir().unwrap();
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o700)).unwrap();
        save_credentials(dir.path(), &creds()).unwrap();
        if crate::paths::setting("GOETHE_PUBLIC_IP_URL").is_none() {
            assert_eq!(public_ip_url(dir.path()).unwrap(), "https://platform.example.com/api/ip");
        }
    }

    #[test]
    fn a_home_machine_id_is_made_once_and_kept() {
        // A container's node id lives in its home, so a recreated container is the same console node.
        let dir = tempfile::tempdir().unwrap();
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o700)).unwrap();
        ensure_home_machine_id(dir.path()).unwrap();
        let first = fs::read_to_string(dir.path().join(HOME_MACHINE_ID)).unwrap();
        assert_eq!(first.len(), 32);
        ensure_home_machine_id(dir.path()).unwrap();
        assert_eq!(fs::read_to_string(dir.path().join(HOME_MACHINE_ID)).unwrap(), first);
    }

    #[test]
    fn fingerprint_hides_the_machine_id() {
        let fp = fingerprint_of("0123456789abcdef0123456789abcdef");
        assert_eq!(fp.len(), 64);
        assert!(!fp.contains("0123456789abcdef"));
        assert_eq!(fp, fingerprint_of("0123456789abcdef0123456789abcdef"));
        assert_ne!(fp, fingerprint_of("0123456789abcdef0123456789abcdee"));
    }

    #[test]
    fn server_text_loses_control_characters() {
        assert_eq!(clean_text("Lab\x1b[31m\n ok", 64), "Lab[31m ok");
        assert_eq!(clean_text("abcdef", 3), "abc");
        assert!(valid_id("node-1_a.b") && !valid_id("") && !valid_id("a/b") && !valid_id(&"a".repeat(129)));
        assert!(!valid_token("short") && !valid_token("mesh_node_ with space") && valid_token("mesh_node_abcdefghijkl"));
    }

    #[test]
    fn a_hostname_the_platform_would_refuse_is_made_safe() {
        assert_eq!(host_label("Alices-MacBook-Pro"), "Alices-MacBook-Pro");
        assert_eq!(host_label("Alice's MacBook Pro"), "Alice-s-MacBook-Pro");
        assert_eq!(host_label("gpu-box.lan"), "gpu-box.lan");
    }
}
