// SQLite state database: one Goethe schema, created once per process, WAL mode.

use rusqlite::Connection;
use std::collections::HashSet;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

pub const SCHEMA_VERSION: i32 = 1;

const DDL: &str = r"
CREATE TABLE IF NOT EXISTS installed_models (
    model_id TEXT PRIMARY KEY,
    local_path TEXT NOT NULL,
    architecture TEXT NOT NULL,
    static_vram_gb REAL NOT NULL,
    kv_bytes_per_token INTEGER NOT NULL,
    default_ctx INTEGER NOT NULL,
    default_max_conc INTEGER NOT NULL,
    default_tp INTEGER NOT NULL,
    installed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    kv_elems_per_token INTEGER,
    max_ctx INTEGER,
    num_attention_heads INTEGER,
    num_kv_heads INTEGER,
    attention TEXT,
    managed INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS loaded_instances (
    instance_id TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    pid INTEGER NOT NULL,
    pgid INTEGER NOT NULL,
    internal_port INTEGER UNIQUE NOT NULL,
    gpu_indices TEXT NOT NULL,
    allocated_vram_gb_per_gpu REAL NOT NULL,
    mem_fraction_static REAL NOT NULL,
    ctx_len INTEGER NOT NULL,
    max_conc INTEGER NOT NULL,
    status TEXT NOT NULL,
    slot_key TEXT,
    slot_ordinal INTEGER,
    restart_count INTEGER,
    extra_sglang_args TEXT,
    boot_id TEXT,
    start_ticks INTEGER,
    engine_key TEXT
);

CREATE TABLE IF NOT EXISTS desired_loads (
    slot_key TEXT PRIMARY KEY,
    model_id TEXT NOT NULL,
    gpus TEXT,
    ctx INTEGER,
    max_conc INTEGER,
    tp INTEGER,
    extra_sglang_args TEXT NOT NULL DEFAULT '[]',
    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);

CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    token_hash TEXT UNIQUE NOT NULL,
    raw_prefix TEXT NOT NULL,
    rpm_limit INTEGER DEFAULT 120,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at TEXT,
    model_scope TEXT,
    is_admin INTEGER NOT NULL DEFAULT 0,
    token TEXT
);

CREATE TABLE IF NOT EXISTS key_usage (
    key_id TEXT NOT NULL,
    day TEXT NOT NULL,
    requests INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (key_id, day)
);

CREATE TABLE IF NOT EXISTS gpu_policy (
    gpu_index INTEGER PRIMARY KEY,
    denied INTEGER NOT NULL DEFAULT 1,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_aliases (
    alias TEXT PRIMARY KEY,
    model_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ip_allowlist (
    cidr TEXT PRIMARY KEY,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS goethe_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS watchdog_restarts (
    slot_key TEXT PRIMARY KEY,
    window_start INTEGER NOT NULL,
    count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS engine_overhead (
    model_id TEXT NOT NULL,
    gpu_name TEXT NOT NULL,
    tp INTEGER NOT NULL,
    extra_gb REAL NOT NULL,
    PRIMARY KEY (model_id, gpu_name, tp)
);

CREATE TABLE IF NOT EXISTS watchdog_probe_state (
    instance_id TEXT PRIMARY KEY,
    cheap_fail_since INTEGER,
    deep_fail_streak INTEGER NOT NULL DEFAULT 0,
    last_deep_at INTEGER NOT NULL DEFAULT 0,
    restart_in_flight INTEGER NOT NULL DEFAULT 0,
    cheap_ok_once INTEGER NOT NULL DEFAULT 0
);
";

pub fn open(home: &Path) -> rusqlite::Result<Connection> {
    // Open state.db; the schema is checked once per process and never rewritten when current.
    let path = home.join("state.db");
    let created = !path.exists();
    let con = Connection::open(&path)?;
    if created {
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    con.busy_timeout(Duration::from_secs(10))?;
    con.execute_batch("PRAGMA synchronous=NORMAL;")?;
    if !schema_checked(&path) {
        init_schema_retrying(&con)?;
        mark_schema_checked(&path);
    }
    Ok(con)
}

fn checked_paths() -> &'static Mutex<HashSet<PathBuf>> {
    static CHECKED: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    CHECKED.get_or_init(|| Mutex::new(HashSet::new()))
}

fn schema_checked(path: &Path) -> bool {
    checked_paths().lock().map(|s| s.contains(path)).unwrap_or(false)
}

fn mark_schema_checked(path: &Path) {
    if let Ok(mut s) = checked_paths().lock() {
        s.insert(path.to_path_buf());
    }
}

fn init_schema_retrying(con: &Connection) -> rusqlite::Result<()> {
    // First opens racing on a fresh database can see SQLITE_BUSY on the WAL switch; retry for up to ten seconds.
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        match init_schema(con) {
            Err(e) if is_busy(&e) && std::time::Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(25));
            }
            other => return other,
        }
    }
}

fn is_busy(e: &rusqlite::Error) -> bool {
    matches!(
        e,
        rusqlite::Error::SqliteFailure(f, _)
            if matches!(f.code, rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked)
    )
}

fn init_schema(con: &Connection) -> rusqlite::Result<()> {
    // Create the schema in one immediate transaction, or refuse a database from a newer goethe.
    let ver = user_version(con)?;
    if ver == SCHEMA_VERSION {
        return ensure_columns(con);
    }
    if ver > SCHEMA_VERSION {
        return Err(newer_schema(ver));
    }
    con.execute_batch("PRAGMA journal_mode=WAL;")?;
    con.execute_batch("BEGIN IMMEDIATE;")?;
    let result = (|| {
        let ver = user_version(con)?;
        if ver > SCHEMA_VERSION {
            return Err(newer_schema(ver));
        }
        if ver < SCHEMA_VERSION {
            con.execute_batch(DDL)?;
            con.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION};"))?;
        }
        Ok(())
    })();
    match result {
        Ok(()) => con.execute_batch("COMMIT;").and_then(|()| ensure_columns(con)),
        Err(e) => {
            let _ = con.execute_batch("ROLLBACK;");
            Err(e)
        }
    }
}

fn ensure_columns(con: &Connection) -> rusqlite::Result<()> {
    // Columns added after schema version 1, without a version bump so an older goethe still opens the database.
    let has_token = || con.prepare("SELECT token FROM api_keys LIMIT 0").is_ok();
    match has_token() {
        true => Ok(()),
        false => con.execute_batch("ALTER TABLE api_keys ADD COLUMN token TEXT;").or_else(|e| if has_token() { Ok(()) } else { Err(e) }),
    }
}

fn user_version(con: &Connection) -> rusqlite::Result<i32> {
    con.query_row("PRAGMA user_version", [], |r| r.get(0))
}

fn newer_schema(ver: i32) -> rusqlite::Error {
    rusqlite::Error::SqliteFailure(
        rusqlite::ffi::Error::new(rusqlite::ffi::SQLITE_CANTOPEN),
        Some(format!(
            "state.db uses schema {ver}, newer than this goethe (schema {SCHEMA_VERSION}); upgrade goethe"
        )),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables(con: &Connection) -> Vec<String> {
        con.prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect()
    }

    #[test]
    fn fresh_open_creates_schema_private_and_wal() {
        let dir = tempfile::tempdir().unwrap();
        let con = open(dir.path()).unwrap();
        assert_eq!(user_version(&con).unwrap(), SCHEMA_VERSION);
        let names = tables(&con);
        for t in ["api_keys", "desired_loads", "installed_models", "loaded_instances", "watchdog_probe_state"] {
            assert!(names.iter().any(|n| n == t), "missing table {t}");
        }
        let mode: String = con.query_row("PRAGMA journal_mode", [], |r| r.get(0)).unwrap();
        assert_eq!(mode, "wal");
        let perms = std::fs::metadata(dir.path().join("state.db")).unwrap().permissions().mode();
        assert_eq!(perms & 0o777, 0o600);
    }

    #[test]
    fn current_schema_is_not_rewritten() {
        let dir = tempfile::tempdir().unwrap();
        let watcher = open(dir.path()).unwrap();
        let before: i64 = watcher.query_row("PRAGMA data_version", [], |r| r.get(0)).unwrap();
        let other = Connection::open(dir.path().join("state.db")).unwrap();
        init_schema(&other).unwrap();
        let after: i64 = watcher.query_row("PRAGMA data_version", [], |r| r.get(0)).unwrap();
        assert_eq!(before, after);
    }

    #[test]
    fn newer_schema_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        {
            let con = Connection::open(dir.path().join("state.db")).unwrap();
            con.execute_batch("PRAGMA user_version = 99;").unwrap();
        }
        let con = Connection::open(dir.path().join("state.db")).unwrap();
        let err = init_schema(&con).unwrap_err().to_string();
        assert!(err.contains("newer than this goethe"), "{err}");
    }

    #[test]
    fn concurrent_first_opens_succeed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let p = path.join("state.db");
                std::thread::spawn(move || {
                    let con = Connection::open(&p).unwrap();
                    con.busy_timeout(Duration::from_secs(10)).unwrap();
                    init_schema_retrying(&con).unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let con = open(&path).unwrap();
        assert_eq!(user_version(&con).unwrap(), SCHEMA_VERSION);
    }
}
