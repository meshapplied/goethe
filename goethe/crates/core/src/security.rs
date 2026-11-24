// API keys and IP allowlist management.

use crate::db;
use crate::nginx;
use crate::paths::goethe_home;
use rusqlite::{Connection, OptionalExtension};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::fs;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::Path;

pub fn canonical_cidr(raw: &str) -> Result<String, String> {
    // Parse an IPv4/IPv6 address with an optional prefix into canonical network/prefix form.
    let s = raw.trim();
    let bad = || {
        format!("invalid CIDR {raw:?}: expected an IP address with an optional /prefix, e.g. 203.0.113.7 or 10.0.0.0/8")
    };
    let (addr_s, prefix_s) = match s.split_once('/') {
        Some((a, p)) => (a, Some(p)),
        None => (s, None),
    };
    let addr: IpAddr = addr_s.parse().map_err(|_| bad())?;
    let max: u32 = if addr.is_ipv4() { 32 } else { 128 };
    let prefix = match prefix_s {
        None => max,
        Some(p) if !p.is_empty() && p.len() <= 3 && p.bytes().all(|b| b.is_ascii_digit()) => {
            p.parse::<u32>().map_err(|_| bad())?
        }
        Some(_) => return Err(bad()),
    };
    if prefix > max {
        return Err(bad());
    }
    let net = match addr {
        IpAddr::V4(a) => IpAddr::V4(Ipv4Addr::from(u32::from(a) & u32::MAX.checked_shl(32 - prefix).unwrap_or(0))),
        IpAddr::V6(a) => IpAddr::V6(Ipv6Addr::from(u128::from(a) & u128::MAX.checked_shl(128 - prefix).unwrap_or(0))),
    };
    Ok(format!("{net}/{prefix}"))
}

pub fn allowlist_add(cidr: &str, description: Option<&str>) -> Result<(), String> {
    // Store one canonical entry and apply it; when nginx rejects the result the previous state is restored.
    let canonical = canonical_cidr(cidr)?;
    let home = goethe_home();
    let _lock = nginx::gateway_lock(&home)?;
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let previous: Option<Option<String>> = con
        .query_row(
            "SELECT description FROM ip_allowlist WHERE cidr = ?1",
            [&canonical],
            |r| r.get(0),
        )
        .optional()
        .map_err(|e| e.to_string())?;
    con.execute(
        "INSERT OR REPLACE INTO ip_allowlist (cidr, description) VALUES (?1, ?2)",
        rusqlite::params![canonical, description.unwrap_or("")],
    )
    .map_err(|e| e.to_string())?;
    apply_allowlist(&con, &home, || match &previous {
        Some(desc) => con.execute(
            "INSERT OR REPLACE INTO ip_allowlist (cidr, description) VALUES (?1, ?2)",
            rusqlite::params![canonical, desc],
        ),
        None => con.execute("DELETE FROM ip_allowlist WHERE cidr = ?1", [&canonical]),
    })
}

pub fn allowlist_remove(cidr: &str) -> Result<(), String> {
    // Remove an entry by canonical or legacy raw spelling and apply; nginx rejecting the result restores it.
    let raw = cidr.trim();
    let target = canonical_cidr(raw).unwrap_or_else(|_| raw.to_string());
    let home = goethe_home();
    let _lock = nginx::gateway_lock(&home)?;
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let removed: Vec<(String, Option<String>)> = {
        let mut stmt = con
            .prepare("SELECT cidr, description FROM ip_allowlist WHERE cidr = ?1 OR cidr = ?2")
            .map_err(|e| e.to_string())?;
        let rows = stmt
            .query_map([&target, &raw.to_string()], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(|e| e.to_string())?;
        rows.filter_map(|r| r.ok()).collect()
    };
    if removed.is_empty() {
        return Err(format!("{raw} is not in the allowlist (goethe api allow shows the entries)"));
    }
    con.execute(
        "DELETE FROM ip_allowlist WHERE cidr = ?1 OR cidr = ?2",
        [&target, &raw.to_string()],
    )
    .map_err(|e| e.to_string())?;
    apply_allowlist(&con, &home, || {
        for (c, d) in &removed {
            con.execute(
                "INSERT OR REPLACE INTO ip_allowlist (cidr, description) VALUES (?1, ?2)",
                rusqlite::params![c, d],
            )?;
        }
        Ok(removed.len())
    })
}

pub fn allowlist_list() -> Result<Vec<String>, String> {
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let mut stmt = con
        .prepare("SELECT cidr FROM ip_allowlist ORDER BY cidr")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| r.get(0))
        .map_err(|e| e.to_string())?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

fn apply_allowlist<F>(con: &Connection, home: &Path, undo: F) -> Result<(), String>
where
    F: FnOnce() -> rusqlite::Result<usize>,
{
    // Rewrite allowlist.conf from the DB, nginx -t it, and reload; a rejected config undoes the DB and file change.
    let path = home.join("nginx/allowlist.conf");
    let old = fs::read(&path).ok();
    nginx::write_private(&path, render_allowlist(con)?.as_bytes())?;
    if let Err(e) = nginx::test_config(home) {
        let _ = undo();
        let _ = match old {
            Some(bytes) => nginx::write_private(&path, &bytes),
            None => fs::remove_file(&path).map_err(|e| e.to_string()),
        };
        return Err(format!("nginx rejected the new allowlist, change rolled back: {e}"));
    }
    nginx::apply_locked(home).map_err(|e| format!("{e}; the allowlist is saved and applies when the gateway starts"))
}

fn render_allowlist(con: &Connection) -> Result<String, String> {
    // nginx allow lines for every valid stored entry; invalid legacy rows are skipped with a warning.
    let mut stmt = con
        .prepare("SELECT cidr FROM ip_allowlist ORDER BY cidr")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .map_err(|e| e.to_string())?;
    let mut out = String::from("# managed by goethe api allow\n");
    for raw in rows.filter_map(|r| r.ok()) {
        match canonical_cidr(&raw) {
            Ok(c) => {
                let _ = writeln!(out, "allow {c};");
            }
            Err(_) => eprintln!("warning: ignoring invalid allowlist entry {raw:?}; remove it with goethe api allow remove"),
        }
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct ResolvedKey {
    pub token_hash: String,
    pub key_id: String,
    pub rpm_limit: i64,
    pub expires_at: Option<String>,
    pub model_scope: Option<String>,
    pub is_admin: bool,
}

pub fn key_expired(expires_at: &Option<String>) -> bool {
    let Some(raw) = expires_at.as_ref().filter(|s| !s.is_empty()) else {
        return false;
    };
    let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
    today.as_str() > raw.as_str()
}

pub fn model_allowed(scope: &Option<String>, model: &str) -> bool {
    let Some(raw) = scope.as_ref().filter(|s| !s.is_empty()) else {
        return true;
    };
    raw.split(',')
        .map(|s| s.trim())
        .any(|m| m == model)
}

pub fn parse_expires(raw: &str) -> Result<String, String> {
    // "30d" (days from now) or a real calendar date YYYY-MM-DD.
    let s = raw.trim();
    if let Some(days) = s.strip_suffix('d') {
        let dt = days
            .parse::<i64>()
            .ok()
            .filter(|n| *n >= 0)
            .and_then(chrono::Duration::try_days)
            .and_then(|d| chrono::Utc::now().checked_add_signed(d))
            .ok_or_else(|| format!("invalid expires: {raw}"))?;
        return Ok(dt.format("%Y-%m-%d").to_string());
    }
    chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d")
        .map(|d| d.format("%Y-%m-%d").to_string())
        .map_err(|_| format!("expires must be 30d or YYYY-MM-DD, got {raw}"))
}

pub fn key_create(
    name: &str,
    raw: bool,
    rpm: Option<i64>,
    expires: Option<&str>,
    models: Option<&str>,
    admin: bool,
) -> Result<String, String> {
    // Create a key with an independent random id; only its SHA-256 and first characters are stored, so the secret is shown once, here.
    let rpm_limit = rpm.unwrap_or(120);
    if rpm_limit < 0 {
        return Err(format!("rpm must be 0 (unlimited) or positive, got {rpm_limit}"));
    }
    key_name_check(name)?;
    let model_scope = models.map(model_scope_list).transpose()?;
    let expires_at = expires.map(parse_expires).transpose()?;
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let token = new_token();
    let hash = token_hash(&token);
    let key_id = random_hex(6);
    let raw_prefix = token[..14].to_string();
    con.execute(
        "INSERT INTO api_keys (key_id, name, token_hash, raw_prefix, rpm_limit, expires_at, model_scope, is_admin)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        rusqlite::params![
            key_id,
            name,
            hash,
            raw_prefix,
            rpm_limit,
            expires_at,
            model_scope,
            i64::from(admin),
        ],
    )
    .map_err(|e| e.to_string())?;
    write_api_keys_best_effort(&home, &con);
    if raw {
        Ok(token)
    } else {
        Ok(format!("created key {key_id}"))
    }
}

pub fn key_name_check(name: &str) -> Result<(), String> {
    // A name people can read in a table: 1 to 64 visible characters, no control characters.
    if name.trim().is_empty() {
        return Err("key name must not be empty".to_string());
    }
    if name.chars().count() > 64 || name.chars().any(char::is_control) {
        return Err("key names are up to 64 letters, digits, spaces or punctuation".to_string());
    }
    Ok(())
}

pub fn model_scope_list(models: &str) -> Result<String, String> {
    // Comma-separated model ids without blanks; an empty scope is refused instead of creating an unscoped key.
    let ids: Vec<&str> = models.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    if ids.is_empty() {
        return Err("--models needs at least one model id (omit it for a key that may use every model)".to_string());
    }
    Ok(ids.join(","))
}

pub fn key_set_models(key_id: &str, models: Option<&str>) -> Result<(), String> {
    // Change which models a key may use (None: every model); the gateway applies it on the next request.
    let scope = models.map(model_scope_list).transpose()?;
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let n = con
        .execute("UPDATE api_keys SET model_scope = ?1 WHERE key_id = ?2", rusqlite::params![scope, key_id])
        .map_err(|e| e.to_string())?;
    if n == 0 {
        return Err(format!("key {key_id} not found"));
    }
    Ok(())
}

pub fn key_set_limits(key_id: &str, rpm: i64, expires: Option<&str>) -> Result<(), String> {
    // Change a key's requests per minute (0: unlimited) and expiry (None: never); the gateway applies it on the next request.
    if rpm < 0 {
        return Err(format!("rpm must be 0 (unlimited) or positive, got {rpm}"));
    }
    let expires_at = expires.map(parse_expires).transpose()?;
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let n = con
        .execute("UPDATE api_keys SET rpm_limit = ?1, expires_at = ?2 WHERE key_id = ?3", rusqlite::params![rpm, expires_at, key_id])
        .map_err(|e| e.to_string())?;
    if n == 0 {
        return Err(format!("key {key_id} not found"));
    }
    Ok(())
}

pub fn key_revoke(key_id: &str) -> Result<(), String> {
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let n = con
        .execute("DELETE FROM api_keys WHERE key_id = ?1", [key_id])
        .map_err(|e| e.to_string())?;
    if n == 0 {
        return Err(format!("key {key_id} not found"));
    }
    write_api_keys_best_effort(&home, &con);
    Ok(())
}

pub fn key_id_of(token: &str) -> Option<String> {
    // The id of a key by its secret (ids are random, so look it up by the hash).
    let con = db::open(&goethe_home()).ok()?;
    con.query_row("SELECT key_id FROM api_keys WHERE token_hash = ?1", [token_hash(token)], |r| r.get(0)).ok()
}

pub fn key_rotate(key_id: &str) -> Result<String, String> {
    rotate_secret(&goethe_home(), key_id)
}

pub fn rotate_secret(home: &Path, key_id: &str) -> Result<String, String> {
    // A new secret for an existing key (same id, name, scope and limits); the old one stops working on the gateway's next request.
    let con = db::open(home).map_err(|e| e.to_string())?;
    let token = new_token();
    let n = con
        .execute(
            "UPDATE api_keys SET token_hash = ?1, raw_prefix = ?2, token = NULL WHERE key_id = ?3",
            rusqlite::params![token_hash(&token), &token[..14], key_id],
        )
        .map_err(|e| e.to_string())?;
    if n == 0 {
        return Err(format!("key {key_id} not found"));
    }
    write_api_keys_best_effort(home, &con);
    Ok(token)
}

pub fn import_key(home: &Path, name: &str, secret: &str) -> Result<bool, String> {
    // A key whose secret the user chose (the container's GOETHE_API_KEY): only its hash is kept; the same secret again changes nothing, a new one replaces the secret of that name's key. True when something changed.
    key_name_check(name)?;
    if !secret.strip_prefix("goethe_sk_").is_some_and(|h| h.len() >= 32 && h.chars().all(|c| c.is_ascii_hexdigit())) {
        return Err("an API key is goethe_sk_ followed by at least 32 hex characters, for example: echo goethe_sk_$(openssl rand -hex 16)".to_string());
    }
    let con = db::open(home).map_err(|e| e.to_string())?;
    let hash = token_hash(secret);
    let have: Option<(String, String)> = con
        .query_row("SELECT key_id, token_hash FROM api_keys WHERE name = ?1 ORDER BY created_at LIMIT 1", [name], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()
        .map_err(|e| e.to_string())?;
    match have {
        Some((_, h)) if h == hash => return Ok(false),
        Some((key_id, _)) => con.execute(
            "UPDATE api_keys SET token_hash = ?1, raw_prefix = ?2, token = NULL WHERE key_id = ?3",
            rusqlite::params![hash, &secret[..14], key_id],
        ),
        None => con.execute(
            "INSERT INTO api_keys (key_id, name, token_hash, raw_prefix, rpm_limit, expires_at, model_scope, is_admin) VALUES (?1, ?2, ?3, ?4, 120, NULL, NULL, 0)",
            rusqlite::params![random_hex(6), name, hash, &secret[..14]],
        ),
    }
    .map_err(|e| e.to_string())?;
    write_api_keys_best_effort(home, &con);
    Ok(true)
}

pub fn key_list_detailed() -> Result<Vec<serde_json::Value>, String> {
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let mut stmt = con
        .prepare(
            "SELECT key_id, name, rpm_limit, expires_at, model_scope, is_admin, raw_prefix FROM api_keys ORDER BY created_at",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, Option<i64>>(2)?.unwrap_or(120),
                r.get::<_, Option<String>>(3)?,
                r.get::<_, Option<String>>(4)?,
                r.get::<_, i64>(5)? != 0,
                r.get::<_, Option<String>>(6)?.unwrap_or_default(),
            ))
        })
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows.filter_map(|r| r.ok()) {
        let (key_id, name, rpm, expires_at, model_scope, is_admin, prefix) = row;
        let (req_t, pt_t, ct_t, req_all, pt_all, ct_all) =
            crate::key_usage::totals_for_key(&con, &key_id)?;
        out.push(serde_json::json!({
            "id": key_id,
            "prefix": prefix,
            "name": name,
            "rpm": rpm,
            "expires": expires_at,
            "models": model_scope,
            "admin": is_admin,
            "requests_today": req_t,
            "tokens_today": pt_t + ct_t,
            "requests_total": req_all,
            "tokens_total": pt_all + ct_all,
        }));
    }
    Ok(out)
}

pub fn sync_api_keys_conf(home: &Path) {
    let Ok(con) = db::open(home) else {
        return;
    };
    write_api_keys_best_effort(home, &con);
}

fn write_api_keys_best_effort(home: &Path, con: &Connection) {
    if let Err(e) = rewrite_api_keys_conf(con, home) {
        eprintln!("warning: api_keys.conf: {e}");
    }
}

fn rewrite_api_keys_conf(con: &Connection, home: &Path) -> Result<(), String> {
    // Hash-only key map, written atomically with 0600 under the gateway lock.
    let _lock = nginx::gateway_lock(home)?;
    let mut stmt = con
        .prepare("SELECT token_hash FROM api_keys ORDER BY created_at")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .map_err(|e| e.to_string())?;
    let mut out = String::new();
    for hash in rows.filter_map(|r| r.ok()) {
        let _ = writeln!(out, "\"{hash}\" 1;");
    }
    nginx::write_private(&home.join("nginx/api_keys.conf"), out.as_bytes())
}

fn new_token() -> String {
    format!("goethe_sk_{}", random_hex(16))
}

pub(crate) fn random_hex(n: usize) -> String {
    // n bytes of kernel randomness as lowercase hex.
    let mut buf = vec![0u8; n];
    crate::sys::random_bytes(&mut buf);
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn token_hash(token: &str) -> String {
    let mut h = Sha256::new();
    h.update(token.as_bytes());
    format!("{:x}", h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_imported_key_keeps_only_its_hash_and_changes_only_with_a_new_secret() {
        // The container's GOETHE_API_KEY: registered once, unchanged on restart, replaced (same id) when the secret changes; junk is refused.
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (format!("goethe_sk_{}", "a1".repeat(16)), format!("goethe_sk_{}", "b2".repeat(16)));
        assert!(import_key(dir.path(), "default", &a).unwrap());
        assert!(!import_key(dir.path(), "default", &a).unwrap(), "the same secret changes nothing");
        let con = db::open(dir.path()).unwrap();
        let row = |con: &Connection| con.query_row("SELECT key_id, token_hash FROM api_keys WHERE name = 'default'", [], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))).unwrap();
        let (id, hash) = row(&con);
        assert_eq!(hash, token_hash(&a));
        assert!(import_key(dir.path(), "default", &b).unwrap());
        assert_eq!(row(&con), (id, token_hash(&b)), "a new secret replaces the old one under the same key");
        let stored: String = con.query_row("SELECT group_concat(coalesce(token, '') || raw_prefix) FROM api_keys", [], |r| r.get(0)).unwrap();
        assert!(!stored.contains(&b[14..]), "the secret itself is not stored");
        for bad in ["goethe_sk_short", "sk_0123456789abcdef0123456789abcdef", &format!("goethe_sk_{}", "z".repeat(32))] {
            assert!(import_key(dir.path(), "default", bad).is_err(), "{bad}");
        }
        assert!(import_key(dir.path(), "", &a).is_err());
    }

    #[test]
    fn key_names_are_readable() {
        assert!(key_name_check("dev laptop").is_ok() && key_name_check("ci-2026.11").is_ok());
        assert!(key_name_check("").is_err() && key_name_check("  ").is_err());
        assert!(key_name_check("\u{1b}[B").is_err(), "an arrow key typed at the prompt");
        assert!(key_name_check(&"x".repeat(65)).is_err());
    }

    #[test]
    fn model_scope_needs_a_model() {
        assert_eq!(model_scope_list(" a, ,b ").unwrap(), "a,b");
        assert!(model_scope_list("").is_err());
        assert!(model_scope_list(" , ").is_err());
    }

    #[test]
    fn cidr_canonicalization() {
        assert_eq!(canonical_cidr("10.0.0.0/8").unwrap(), "10.0.0.0/8");
        assert_eq!(canonical_cidr(" 10.1.2.3/8 ").unwrap(), "10.0.0.0/8");
        assert_eq!(canonical_cidr("127.0.0.1").unwrap(), "127.0.0.1/32");
        assert_eq!(canonical_cidr("0.0.0.0/0").unwrap(), "0.0.0.0/0");
        assert_eq!(canonical_cidr("2001:db8::1/32").unwrap(), "2001:db8::/32");
        assert_eq!(canonical_cidr("2001:DB8:0:0::5").unwrap(), "2001:db8::5/128");
        assert_eq!(canonical_cidr("::/0").unwrap(), "::/0");
    }

    #[test]
    fn cidr_rejects_injection_and_garbage() {
        for bad in [
            "10.0.0.0/8; allow all",
            "10.0.0.0/8;allow all",
            "all",
            "10.0.0.0/8\nallow all",
            "10.0.0.0 /8",
            "10.0.0.0/",
            "10.0.0.0/33",
            "::1/129",
            "10.0.0.0/+8",
            "10.0.0.0/0008",
            "1.2.3",
            "fe80::1%eth0",
            "",
            "10.0.0.0/8/8",
        ] {
            assert!(canonical_cidr(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn rejected_allowlist_is_rolled_back() {
        if std::env::var_os("NGINX_BIN").is_some() {
            return;
        }
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        let sbin = home.join("nginx-root/usr/sbin");
        fs::create_dir_all(&sbin).unwrap();
        fs::write(sbin.join("nginx"), "#!/bin/sh\necho 'emerg: fake rejection' >&2\nexit 1\n").unwrap();
        fs::set_permissions(sbin.join("nginx"), fs::Permissions::from_mode(0o755)).unwrap();
        fs::create_dir_all(home.join("nginx")).unwrap();
        fs::write(home.join("nginx/nginx.conf"), "events {}\n").unwrap();
        fs::write(home.join("nginx/allowlist.conf"), "allow 192.0.2.1/32;\n").unwrap();
        let con = db::open(home).unwrap();
        con.execute("INSERT INTO ip_allowlist (cidr, description) VALUES ('10.0.0.0/8', '')", []).unwrap();
        let err = apply_allowlist(&con, home, || con.execute("DELETE FROM ip_allowlist WHERE cidr = '10.0.0.0/8'", []))
            .unwrap_err();
        assert!(err.contains("rolled back") && err.contains("fake rejection"), "{err}");
        assert_eq!(fs::read_to_string(home.join("nginx/allowlist.conf")).unwrap(), "allow 192.0.2.1/32;\n");
        let n: i64 = con.query_row("SELECT COUNT(*) FROM ip_allowlist", [], |r| r.get(0)).unwrap();
        assert_eq!(n, 0, "the DB change is undone too");
    }

    #[test]
    fn key_ids_are_independent_of_the_token() {
        let token = new_token();
        assert_eq!(token.len(), 42);
        let id = random_hex(6);
        assert_eq!(id.len(), 12);
        assert!(!token.contains(&id));
        assert_eq!(token_hash("x").len(), 64);
        assert!(parse_expires("2026-02-30").is_err());
        for bad in ["-1d", "99999999999999d", "9223372036854775807d"] {
            assert!(parse_expires(bad).is_err(), "{bad} is refused, not a panic or a key that expired already");
        }
        assert_eq!(parse_expires("2030-01-31").unwrap(), "2030-01-31");
    }
}
