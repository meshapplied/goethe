// API keys and IP allowlist management.

use crate::db;
use crate::nginx;
use crate::paths::mesh_home;
use rusqlite::Connection;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::Path;
use std::thread;
use std::time::Duration;

pub fn allowlist_add(cidr: &str, description: Option<&str>) -> Result<(), String> {
    let home = mesh_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    con.execute(
        "INSERT OR REPLACE INTO ip_allowlist (cidr, description) VALUES (?1, ?2)",
        rusqlite::params![cidr, description.unwrap_or("")],
    )
    .map_err(|e| e.to_string())?;
    rewrite_allowlist(&con, &home)?;
    gateway_reload(&home)?;
    thread::sleep(Duration::from_millis(150));
    Ok(())
}

pub fn allowlist_remove(cidr: &str) -> Result<(), String> {
    let home = mesh_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    con.execute("DELETE FROM ip_allowlist WHERE cidr = ?1", [cidr])
        .map_err(|e| e.to_string())?;
    rewrite_allowlist(&con, &home)?;
    gateway_reload(&home)
}

pub fn allowlist_list() -> Result<Vec<String>, String> {
    let home = mesh_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let mut stmt = con
        .prepare("SELECT cidr FROM ip_allowlist ORDER BY cidr")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| r.get(0))
        .map_err(|e| e.to_string())?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub struct ResolvedKey {
    pub key_id: String,
    pub rpm_limit: i64,
    pub expires_at: Option<String>,
    pub model_scope: Option<String>,
    pub is_admin: bool,
}

pub fn resolve_token(token: &str) -> Option<ResolvedKey> {
    let home = mesh_home();
    let con = db::open(&home).ok()?;
    let hash = hex_hash(token);
    con.query_row(
        "SELECT key_id, rpm_limit, expires_at, model_scope, is_admin FROM api_keys WHERE token_hash = ?1",
        [hash],
        |r| {
            Ok(ResolvedKey {
                key_id: r.get(0)?,
                rpm_limit: r.get::<_, Option<i64>>(1)?.unwrap_or(120),
                expires_at: r.get(2)?,
                model_scope: r.get(3)?,
                is_admin: r.get::<_, i64>(4)? != 0,
            })
        },
    )
    .ok()
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
    let s = raw.trim();
    if let Some(days) = s.strip_suffix('d') {
        let n: i64 = days.parse().map_err(|_| format!("invalid expires: {raw}"))?;
        let dt = chrono::Utc::now() + chrono::Duration::days(n);
        return Ok(dt.format("%Y-%m-%d").to_string());
    }
    if s.len() == 10 && s.as_bytes().get(4) == Some(&b'-') {
        return Ok(s.to_string());
    }
    Err(format!("expires must be 30d or YYYY-MM-DD, got {raw}"))
}

pub fn key_create(
    name: &str,
    raw: bool,
    rpm: Option<i64>,
    expires: Option<&str>,
    models: Option<&str>,
    admin: bool,
) -> Result<String, String> {
    let home = mesh_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let token = new_token();
    let hash = hex_hash(&token);
    let key_id = token[8..16].to_string();
    let raw_prefix = token[..20].to_string();
    let rpm_limit = rpm.unwrap_or(120);
    let expires_at = expires.map(parse_expires).transpose()?;
    let model_scope = models.map(|m| m.trim().to_string());
    let is_admin = if admin { 1i64 } else { 0i64 };
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
            is_admin,
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

pub fn key_revoke(key_id: &str) -> Result<(), String> {
    let home = mesh_home();
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

pub fn key_list() -> Result<Vec<(String, String)>, String> {
    let home = mesh_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let mut stmt = con
        .prepare("SELECT key_id, name FROM api_keys ORDER BY created_at")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
        .map_err(|e| e.to_string())?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn key_list_detailed() -> Result<Vec<serde_json::Value>, String> {
    let home = mesh_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let mut stmt = con
        .prepare(
            "SELECT key_id, name, rpm_limit, expires_at, model_scope, is_admin FROM api_keys ORDER BY created_at",
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
            ))
        })
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows.filter_map(|r| r.ok()) {
        let (key_id, name, rpm, expires_at, model_scope, is_admin) = row;
        let (req_t, pt_t, ct_t, req_all, pt_all, ct_all) =
            crate::key_usage::totals_for_key(&con, &key_id)?;
        out.push(serde_json::json!({
            "id": key_id,
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

pub fn gateway_reload(home: &Path) -> Result<(), String> {
    nginx::require_gateway()?;
    nginx::reload(home).map_err(|e| {
        if e.starts_with("gateway is down:") {
            e
        } else {
            format!("gateway is down: {e}")
        }
    })
}

fn rewrite_allowlist(con: &Connection, home: &Path) -> Result<(), String> {
    let mut stmt = con
        .prepare("SELECT cidr FROM ip_allowlist ORDER BY cidr")
        .map_err(|e| e.to_string())?;
    let mut out = String::new();
    let rows = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .map_err(|e| e.to_string())?;
    for cidr in rows.filter_map(|r| r.ok()) {
        out.push_str(&format!("allow {cidr};\n"));
    }
    fs::write(home.join("nginx/allowlist.conf"), out).map_err(|e| e.to_string())
}

fn write_api_keys_best_effort(home: &Path, con: &Connection) {
    if let Err(e) = rewrite_api_keys_conf(con, home) {
        eprintln!("warning: api_keys.conf: {e}");
    }
}

fn rewrite_api_keys_conf(con: &Connection, home: &Path) -> Result<(), String> {
    let path = home.join("nginx/api_keys.conf");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut stmt = con
        .prepare("SELECT token_hash FROM api_keys ORDER BY created_at")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .map_err(|e| e.to_string())?;
    let mut out = String::new();
    for hash in rows.filter_map(|r| r.ok()) {
        out.push_str(&format!("\"{hash}\" 1;\n"));
    }
    fs::write(&path, out).map_err(|e| e.to_string())?;
    fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o600))
        .map_err(|e| e.to_string())
}

fn new_token() -> String {
    let mut buf = [0u8; 16];
    random_bytes(&mut buf);
    format!("mesh_sk_{:032x}", u128::from_be_bytes(buf))
}

fn random_bytes(buf: &mut [u8]) {
    let mut f = fs::File::open("/dev/urandom").expect("urandom");
    f.read_exact(buf).expect("random bytes");
}

fn hex_hash(token: &str) -> String {
    let mut h = Sha256::new();
    h.update(token.as_bytes());
    format!("{:x}", h.finalize())
}
