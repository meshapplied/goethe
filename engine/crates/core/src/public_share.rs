// Temporary public gateway (goethe serve --public): a second nginx listener that exists only while a window is open, for chosen existing keys.

use crate::{db, nginx, security};
use serde::{Deserialize, Serialize};
use std::fs;
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub const DEFAULT_PORT: u16 = 8444;
pub const MIN_SECS: i64 = 60;
pub const MAX_SECS: i64 = 7 * 86_400;
pub const PUBLIC_MUX_IP: Ipv4Addr = Ipv4Addr::new(127, 0, 0, 2);
const RATE_PER_IP: u64 = 10;
const CONN_PER_IP: u64 = 16;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Share {
    pub open: bool,
    pub until: i64,
    pub port: u16,
    pub keys: Vec<String>,
    #[serde(default)]
    pub allow: Vec<String>,
}

impl Share {
    pub fn active_at(&self, now: i64) -> bool {
        self.open && now < self.until
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Access {
    Allowed,
    Closed,
    KeyNotShared,
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

pub fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
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
    load(home).filter(|s| s.active_at(now()))
}

pub fn parse_duration(raw: &str) -> Result<i64, String> {
    // 30m, 5h or 1d: a whole number and one unit, from 1 minute to 7 days.
    let s = raw.trim().to_ascii_lowercase();
    let bad = || format!("duration must be minutes, hours or days such as 30m, 5h or 1d, got {raw:?}");
    let (num, unit) = s.split_at(s.len().saturating_sub(1));
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

pub fn check(home: &Path, key_id: &str, is_admin: bool) -> Access {
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
    decide(share.as_ref(), key_id, is_admin, now())
}

fn decide(share: Option<&Share>, key_id: &str, is_admin: bool, now: i64) -> Access {
    match share {
        Some(s) if s.active_at(now) => {
            if !is_admin && s.keys.iter().any(|k| k == key_id) {
                Access::Allowed
            } else {
                Access::KeyNotShared
            }
        }
        _ => Access::Closed,
    }
}

type KeyRow = (String, String, bool, Option<String>, Option<String>, Option<String>);

fn key_rows(home: &Path) -> Result<Vec<KeyRow>, String> {
    let con = db::open(home).map_err(|e| e.to_string())?;
    let mut stmt = con
        .prepare("SELECT key_id, name, is_admin, expires_at, model_scope, token FROM api_keys ORDER BY created_at")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get::<_, i64>(2)? != 0, r.get(3)?, r.get(4)?, r.get(5)?)))
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
            [] => return Err(format!("no API key named {w:?}; keys: {}", if names().is_empty() { "none (create one with goethe api keys --create <name> --models <model>)".into() } else { names() })),
            _ => return Err(format!("several keys are named {w:?}; use the key id from goethe api keys")),
        };
        let (id, name, admin, expires, models, token) = row.clone();
        if admin {
            return Err(format!("{name} is an admin key; admin keys never work on the public gateway"));
        }
        if security::key_expired(&expires) {
            return Err(format!("{name} expired on {}; pick another key or change its expiry in goethe api manage", expires.unwrap_or_default()));
        }
        if expires.as_deref().is_some_and(|e| e < until_day.as_str()) {
            warnings.push(format!("{name} expires on {}, before the public gateway closes", expires.clone().unwrap_or_default()));
        }
        if models.as_deref().is_none_or(str::is_empty) {
            warnings.push(format!("{name} may use every model; a key limited with --models is safer to share"));
        }
        if !out.iter().any(|k| k.id == id) {
            out.push(SharedKey { id, name, token, models, expires });
        }
    }
    Ok(out)
}

fn port_free(port: u16) -> bool {
    TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], port))).is_ok()
}

fn mux_public_listening(home: &Path) -> bool {
    nginx::persisted_mux_port(home)
        .is_ok_and(|p| TcpStream::connect_timeout(&SocketAddr::new(IpAddr::V4(PUBLIC_MUX_IP), p), Duration::from_millis(500)).is_ok())
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
}

pub fn open(home: &Path, a: OpenArgs) -> Result<Opened, String> {
    // Validate, then write the window and hot-reload nginx; an open window on the same port restarts its timer from now without closing.
    if !nginx::master_running(home) {
        return Err("the gateway is not running; start goethe with goethe up".to_string());
    }
    if !mux_public_listening(home) {
        return Err("goethed does not serve the public gateway yet; restart it with goethe down then goethe up".to_string());
    }
    let _lock = nginx::gateway_lock(home)?;
    let previous = load(home);
    let until = now() + a.secs;
    let mut warnings = Vec::new();
    let wanted: Vec<String> = if a.keys.is_empty() {
        previous.as_ref().map(|p| p.keys.clone()).filter(|k| !k.is_empty()).ok_or_else(|| {
            "name the API keys to allow with --key <name> (the next time they are remembered); list them with goethe api keys".to_string()
        })?
    } else {
        a.keys.to_vec()
    };
    let keys = resolve_keys(home, &wanted, until, &mut warnings)?;
    let allow = a.allow.iter().map(|c| security::canonical_cidr(c)).collect::<Result<Vec<_>, _>>()?;
    let port = a.port.or(previous.as_ref().map(|p| p.port)).unwrap_or(DEFAULT_PORT);
    let gateway_port = nginx::public_port_u16(home);
    let mux_port = nginx::persisted_mux_port(home)?;
    if port == gateway_port || port == mux_port || port < 1024 {
        return Err(format!("port {port} is not usable for the public gateway; pick a free port above 1023 with --port"));
    }
    let ours = previous.as_ref().is_some_and(|p| p.active_at(now()) && p.port == port);
    if !ours && !port_free(port) {
        return Err(format!("port {port} is in use; pick another with --port"));
    }
    let public_ip = crate::platform::lookup_public_ip(home).ok();
    if public_ip.is_none() {
        warnings.push("could not look up this machine's public address; set GOETHE_PUBLIC_IP to name it".to_string());
    }
    let share = Share { open: true, until, port, keys: keys.iter().map(|k| k.id.clone()).collect(), allow };
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
    let cert = cert_path(home);
    Ok(Opened { fingerprint: fingerprint(&cert), cert, share, keys, public_ip, warnings })
}

pub fn shutdown(home: &Path) -> Result<bool, String> {
    // Close the window now and drop the listener; the chosen keys are remembered for the next opening.
    let _lock = nginx::gateway_lock(home)?;
    let Some(share) = load(home) else { return Ok(false) };
    let was_open = share.active_at(now());
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
        crate::platform::notify(crate::platform::CHANGE_STATE);
    }
    seen.replace(now_stamp);
}

fn close_expired(home: &Path) {
    // Mark an open window whose time ran out closed, under the gateway lock, and reload nginx without it.
    let Some(share) = load(home) else { return };
    if !share.open || share.active_at(now()) {
        return;
    }
    let Ok(_lock) = nginx::gateway_lock(home) else { return };
    match load(home) {
        Some(s) if s.open && !s.active_at(now()) => {
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
    let san_file = dir.join("public_sans.txt");
    let current = fs::read_to_string(&san_file).map(|s| s.trim() == san).unwrap_or(false);
    if current && crt.is_file() && key.is_file() {
        return Ok(());
    }
    let ok = Command::new("openssl")
        .args(["req", "-x509", "-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:prime256v1", "-keyout"])
        .arg(&key)
        .arg("-out")
        .arg(&crt)
        .args(["-days", "30", "-nodes", "-subj", "/CN=goethe-public"])
        .args(["-addext", &san, "-addext", "basicConstraints=critical,CA:TRUE"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| e.to_string())?
        .success();
    if !ok {
        return Err("openssl could not create the public gateway certificate".to_string());
    }
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(&key, fs::Permissions::from_mode(0o600)).map_err(|e| e.to_string())?;
    fs::set_permissions(&crt, fs::Permissions::from_mode(0o644)).map_err(|e| e.to_string())?;
    nginx::write_private(&san_file, san.as_bytes())
}

fn fingerprint(crt: &Path) -> Option<String> {
    let out = Command::new("openssl").args(["x509", "-noout", "-fingerprint", "-sha256", "-in"]).arg(crt).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.trim().split_once('=').map(|(_, f)| f.to_string())
}

pub fn nginx_block(home: &Path, mux_port: u16, modern_http2: bool, max_body_mb: u64) -> String {
    // The public server for nginx while a window is open, else nothing: its own port, certificate, limits and upstream (127.0.0.2).
    let Some(share) = active(home) else { return String::new() };
    let dir = nginx::nginx_prefix(home);
    if !cert_path(home).is_file() && ensure_cert(home, None).is_err() {
        return String::new();
    }
    let allow = if share.allow.is_empty() {
        String::new()
    } else {
        let mut lines: String = share.allow.iter().map(|c| format!("    allow {c};\n")).collect();
        lines.push_str("    deny all;\n");
        lines
    };
    const TMPL: &str = include_str!("../../../nginx/goethe_public.conf.tmpl");
    TMPL.replace("{{PORT}}", &share.port.to_string())
        .replace("{{HTTP2_LISTEN}}", if modern_http2 { "" } else { " http2" })
        .replace("{{HTTP2_ON}}", if modern_http2 { "    http2 on;" } else { "" })
        .replace("{{CERT}}", &quote(&dir.join("certs/public.crt")))
        .replace("{{KEY}}", &quote(&dir.join("certs/public.key")))
        .replace("{{ALLOW}}", &allow)
        .replace("{{RATE_PER_IP}}", &RATE_PER_IP.to_string())
        .replace("{{BURST_PER_IP}}", &(2 * RATE_PER_IP).to_string())
        .replace("{{CONN_PER_IP}}", &CONN_PER_IP.to_string())
        .replace("{{MAX_BODY_MB}}", &max_body_mb.to_string())
        .replace("{{PUBLIC_MUX}}", &format!("{PUBLIC_MUX_IP}:{mux_port}"))
}

fn quote(path: &Path) -> String {
    let s = path.display().to_string();
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
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
        Share { open, until, port: 8444, keys: vec!["k1".into()], allow: vec![] }
    }

    #[test]
    fn durations_are_minutes_hours_or_days_within_bounds() {
        assert_eq!(parse_duration("30m"), Ok(1800));
        assert_eq!(parse_duration("5h"), Ok(18_000));
        assert_eq!(parse_duration("1D"), Ok(86_400));
        assert_eq!(parse_duration("7d"), Ok(MAX_SECS));
        for bad in ["", "h", "0m", "8d", "59s", "1.5h", "-1h", "1w", "10000000000000000d", "5 h"] {
            assert!(parse_duration(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn only_listed_non_admin_keys_pass_while_the_window_is_open() {
        let s = share(true, 1000);
        assert_eq!(decide(Some(&s), "k1", false, 999), Access::Allowed);
        assert_eq!(decide(Some(&s), "k1", false, 1000), Access::Closed, "closes at the end second");
        assert_eq!(decide(Some(&s), "k2", false, 999), Access::KeyNotShared);
        assert_eq!(decide(Some(&s), "k1", true, 999), Access::KeyNotShared, "admin keys never");
        assert_eq!(decide(Some(&share(false, 5000)), "k1", false, 999), Access::Closed, "shut down");
        assert_eq!(decide(None, "k1", false, 0), Access::Closed);
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
        let got = resolve_keys(h, &["app".into(), "k1".into()], now() + 60, &mut w).unwrap();
        assert_eq!(got.len(), 1, "the same key twice counts once");
        assert!(w.is_empty());
        assert!(resolve_keys(h, &["root".into()], now() + 60, &mut w).err().unwrap().contains("admin"));
        assert!(resolve_keys(h, &["old".into()], now() + 60, &mut w).err().unwrap().contains("expired"));
        assert!(resolve_keys(h, &["dup".into()], now() + 60, &mut w).err().unwrap().contains("several"));
        assert!(resolve_keys(h, &["nope".into()], now() + 60, &mut w).err().unwrap().contains("app"));
        resolve_keys(h, &["wide".into()], now() + 60, &mut w).unwrap();
        assert!(w.iter().any(|x| x.contains("every model")));
    }

    #[test]
    fn no_block_is_rendered_without_an_open_window() {
        let dir = home_with_keys();
        let h = dir.path();
        assert_eq!(nginx_block(h, 30000, true, 64), "");
        store(h, &share(true, now() - 1)).unwrap();
        assert_eq!(nginx_block(h, 30000, true, 64), "", "expired");
        store(h, &share(false, now() + 600)).unwrap();
        assert_eq!(nginx_block(h, 30000, true, 64), "", "shut down");
        store(h, &Share { allow: vec!["203.0.113.7/32".into()], ..share(true, now() + 600) }).unwrap();
        let block = nginx_block(h, 30000, true, 64);
        assert!(block.contains("listen 0.0.0.0:8444 ssl"), "{block}");
        assert!(block.contains("127.0.0.2:30000"));
        assert!(block.contains("allow 203.0.113.7/32;\n    deny all;"));
        assert!(block.contains("public.crt"));
    }

    #[test]
    fn the_router_sees_every_change_at_once() {
        let dir = home_with_keys();
        let h = dir.path();
        assert_eq!(check(h, "k1", false), Access::Closed);
        store(h, &share(true, now() + 600)).unwrap();
        assert_eq!(check(h, "k1", false), Access::Allowed);
        store(h, &Share { keys: vec!["k4".into()], ..share(true, now() + 600) }).unwrap();
        assert_eq!(check(h, "k1", false), Access::KeyNotShared, "a replaced key stops at the next request");
        store(h, &Share { open: false, ..share(true, now() + 600) }).unwrap();
        assert_eq!(check(h, "k1", false), Access::Closed);
        fs::remove_file(path(h)).unwrap();
        assert_eq!(check(h, "k1", false), Access::Closed);
    }

    #[test]
    fn tick_closes_an_expired_window_and_keeps_the_keys() {
        let dir = home_with_keys();
        let h = dir.path();
        store(h, &share(true, now() - 5)).unwrap();
        tick(h);
        let s = load(h).unwrap();
        assert!(!s.open);
        assert_eq!(s.keys, vec!["k1".to_string()]);
        store(h, &share(true, now() + 600)).unwrap();
        tick(h);
        assert!(load(h).unwrap().open, "an open window stays open");
    }
}
