// SQLite state database with contract DDL and migrations.

use rusqlite::Connection;
use std::path::Path;

const USER_VERSION: i32 = 6;

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
    installed_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_keys (
    key_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    token_hash TEXT UNIQUE NOT NULL,
    raw_prefix TEXT NOT NULL,
    rpm_limit INTEGER DEFAULT 120,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
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
";

pub fn open(home: &Path) -> rusqlite::Result<Connection> {
    let path = home.join("state.db");
    let con = Connection::open(&path)?;
    con.execute_batch("PRAGMA journal_mode=WAL;")?;
    backup_before_migrate(&con, home)?;
    con.execute_batch(DDL)?;
    migrate(&con)?;
    Ok(con)
}

fn backup_before_migrate(con: &Connection, home: &Path) -> rusqlite::Result<()> {
    let ver: i32 = con.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if ver == 0 || ver >= USER_VERSION {
        return Ok(());
    }
    let bak = home.join(format!("state.db.v{ver}.bak"));
    if bak.exists() {
        return Ok(());
    }
    con.execute("VACUUM INTO ?1", [bak.to_string_lossy().as_ref()])?;
    Ok(())
}

fn migrate(con: &Connection) -> rusqlite::Result<()> {
    let mut ver: i32 = con.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if ver < 1 {
        let cols: Vec<String> = con
            .prepare("PRAGMA table_info(installed_models)")?
            .query_map([], |r| r.get::<_, String>(1))?
            .filter_map(|r| r.ok())
            .collect();
        add_col_if_missing(con, "installed_models", &cols, "kv_elems_per_token", "INTEGER")?;
        add_col_if_missing(con, "installed_models", &cols, "max_ctx", "INTEGER")?;
        add_col_if_missing(con, "installed_models", &cols, "num_attention_heads", "INTEGER")?;
        add_col_if_missing(con, "installed_models", &cols, "num_kv_heads", "INTEGER")?;
        add_col_if_missing(con, "installed_models", &cols, "attention", "TEXT")?;
        ver = 1;
        con.execute_batch("PRAGMA user_version = 1;")?;
    }
    if ver < 2 {
        con.execute_batch(
            "CREATE TABLE IF NOT EXISTS desired_loads (
                model_id TEXT PRIMARY KEY,
                gpus TEXT,
                ctx INTEGER,
                max_conc INTEGER,
                tp INTEGER,
                extra_sglang_args TEXT NOT NULL DEFAULT '[]'
            );
            CREATE TABLE IF NOT EXISTS mesh_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS watchdog_restarts (
                model_id TEXT PRIMARY KEY,
                window_start INTEGER NOT NULL,
                count INTEGER NOT NULL
            );",
        )?;
        ver = 2;
        con.execute_batch("PRAGMA user_version = 2;")?;
    }
    if ver < 3 {
        con.execute_batch(
            "CREATE TABLE IF NOT EXISTS key_usage (
                key_id TEXT NOT NULL,
                day TEXT NOT NULL,
                requests INTEGER NOT NULL DEFAULT 0,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (key_id, day)
            );",
        )?;
        let key_cols = table_cols(con, "api_keys")?;
        add_col_if_missing(con, "api_keys", &key_cols, "expires_at", "TEXT")?;
        let key_cols = table_cols(con, "api_keys")?;
        add_col_if_missing(con, "api_keys", &key_cols, "model_scope", "TEXT")?;
        ver = 3;
        con.execute_batch("PRAGMA user_version = 3;")?;
    }
    if ver < 4 {
        con.execute_batch(
            "CREATE TABLE IF NOT EXISTS engine_overhead (
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
                restart_in_flight INTEGER NOT NULL DEFAULT 0
            );",
        )?;
        let inst_cols = table_cols(con, "loaded_instances")?;
        add_col_if_missing(con, "loaded_instances", &inst_cols, "slot_key", "TEXT")?;
        let inst_cols = table_cols(con, "loaded_instances")?;
        add_col_if_missing(con, "loaded_instances", &inst_cols, "slot_ordinal", "INTEGER")?;
        if table_exists(con, "desired_loads")? {
            let des_cols = table_cols(con, "desired_loads")?;
            add_col_if_missing(con, "desired_loads", &des_cols, "slot_key", "TEXT")?;
        }
        backfill_slot_keys(con)?;
        if table_exists(con, "watchdog_restarts")? {
            con.execute_batch(
                "CREATE TABLE IF NOT EXISTS watchdog_restarts_v4 (
                    slot_key TEXT PRIMARY KEY,
                    window_start INTEGER NOT NULL,
                    count INTEGER NOT NULL
                );",
            )?;
            let _ = con.execute_batch(
                "INSERT OR IGNORE INTO watchdog_restarts_v4 (slot_key, window_start, count)
                 SELECT COALESCE(slot_key, model_id), window_start, count FROM watchdog_restarts",
            );
            con.execute_batch("DROP TABLE IF EXISTS watchdog_restarts;")?;
            con.execute_batch("ALTER TABLE watchdog_restarts_v4 RENAME TO watchdog_restarts;")?;
        } else {
            con.execute_batch(
                "CREATE TABLE IF NOT EXISTS watchdog_restarts (
                    slot_key TEXT PRIMARY KEY,
                    window_start INTEGER NOT NULL,
                    count INTEGER NOT NULL
                );",
            )?;
        }
        con.execute_batch("PRAGMA user_version = 4;")?;
    }
    if ver < 5 {
        if table_exists(con, "desired_loads")? {
            con.execute_batch(
                "CREATE TABLE IF NOT EXISTS desired_loads_v5 (
                    slot_key TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    gpus TEXT,
                    ctx INTEGER,
                    max_conc INTEGER,
                    tp INTEGER,
                    extra_sglang_args TEXT NOT NULL DEFAULT '[]'
                );",
            )?;
            let _ = con.execute_batch(
                "INSERT OR IGNORE INTO desired_loads_v5 (slot_key, model_id, gpus, ctx, max_conc, tp, extra_sglang_args)
                 SELECT COALESCE(NULLIF(slot_key,''), model_id || '|' || COALESCE(gpus,'') || '|0'),
                        model_id, gpus, ctx, max_conc, tp, extra_sglang_args FROM desired_loads",
            );
            con.execute_batch("DROP TABLE desired_loads;")?;
            con.execute_batch("ALTER TABLE desired_loads_v5 RENAME TO desired_loads;")?;
        }
        con.execute_batch(&format!("PRAGMA user_version = {USER_VERSION};"))?;
    }
    let probe_cols = table_cols(con, "watchdog_probe_state")?;
    add_col_if_missing(
        con,
        "watchdog_probe_state",
        &probe_cols,
        "cheap_ok_once",
        "INTEGER NOT NULL DEFAULT 0",
    )?;
    let inst_cols = table_cols(con, "loaded_instances")?;
    add_col_if_missing(con, "loaded_instances", &inst_cols, "restart_count", "INTEGER")?;
    let inst_cols = table_cols(con, "loaded_instances")?;
    add_col_if_missing(con, "loaded_instances", &inst_cols, "extra_sglang_args", "TEXT")?;
    let key_cols = table_cols(con, "api_keys")?;
    add_col_if_missing(con, "api_keys", &key_cols, "is_admin", "INTEGER NOT NULL DEFAULT 0")?;
    con.execute_batch(
        "CREATE TABLE IF NOT EXISTS gpu_policy (
            gpu_index INTEGER PRIMARY KEY,
            denied INTEGER NOT NULL DEFAULT 1,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS model_aliases (
            alias TEXT PRIMARY KEY,
            model_id TEXT NOT NULL
        );",
    )?;
    con.execute_batch(&format!("PRAGMA user_version = {USER_VERSION};"))?;
    Ok(())
}

fn table_exists(con: &Connection, table: &str) -> rusqlite::Result<bool> {
    let n: i64 = con.query_row(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
        [table],
        |r| r.get(0),
    )?;
    Ok(n > 0)
}

fn table_cols(con: &Connection, table: &str) -> rusqlite::Result<Vec<String>> {
    let sql = format!("PRAGMA table_info({table})");
    let mut stmt = con.prepare(&sql)?;
    let rows = stmt.query_map([], |r| r.get::<_, String>(1))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

fn backfill_slot_keys(con: &Connection) -> rusqlite::Result<()> {
    let mut stmt = con.prepare(
        "SELECT instance_id, model_id, gpu_indices FROM loaded_instances WHERE slot_key IS NULL",
    )?;
    let rows: Vec<(String, String, String)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .filter_map(|r| r.ok())
        .collect();
    for (id, model, gpus) in rows {
        let ord = 0i64;
        let sk = format!("{model}|{gpus}|{ord}");
        con.execute(
            "UPDATE loaded_instances SET slot_key = ?1, slot_ordinal = ?2 WHERE instance_id = ?3",
            rusqlite::params![sk, ord, id],
        )?;
    }
    if table_exists(con, "desired_loads")? {
        let mut stmt2 = con.prepare(
            "SELECT model_id, gpus FROM desired_loads WHERE slot_key IS NULL",
        )?;
        let rows2: Vec<(String, Option<String>)> = stmt2
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .filter_map(|r| r.ok())
            .collect();
        for (model, gpus) in rows2 {
            let g = gpus.unwrap_or_default();
            let sk = format!("{model}|{g}|0");
            con.execute(
                "UPDATE desired_loads SET slot_key = ?1 WHERE model_id = ?2 AND COALESCE(gpus,'') = ?3",
                rusqlite::params![sk, model, g],
            )?;
        }
    }
    Ok(())
}

fn add_col_if_missing(
    con: &Connection,
    table: &str,
    cols: &[String],
    name: &str,
    sql_type: &str,
) -> rusqlite::Result<()> {
    if cols.iter().any(|c| c == name) {
        return Ok(());
    }
    let sql = format!("ALTER TABLE {table} ADD COLUMN {name} {sql_type};");
    if let Err(e) = con.execute_batch(&sql) {
        let msg = e.to_string();
        if !msg.contains("duplicate column") {
            return Err(e);
        }
    }
    Ok(())
}

#[cfg(test)]
mod migrate_tests {
    use super::*;
    use std::path::Path;

    fn v2_tables() -> &'static str {
        "CREATE TABLE IF NOT EXISTS desired_loads (
            model_id TEXT PRIMARY KEY,
            gpus TEXT,
            ctx INTEGER,
            max_conc INTEGER,
            tp INTEGER,
            extra_sglang_args TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS mesh_settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS watchdog_restarts (
            model_id TEXT PRIMARY KEY,
            window_start INTEGER NOT NULL,
            count INTEGER NOT NULL
        );"
    }

    fn open_migrated(home: &Path, ver: i32, seed: &str) -> Connection {
        let path = home.join("state.db");
        {
            let con = Connection::open(&path).expect("open seed");
            con.execute_batch(DDL).expect("ddl");
            if ver >= 2 {
                con.execute_batch(v2_tables()).expect("v2");
            }
            if ver >= 3 {
                con.execute_batch(
                    "CREATE TABLE IF NOT EXISTS key_usage (
                        key_id TEXT NOT NULL,
                        day TEXT NOT NULL,
                        requests INTEGER NOT NULL DEFAULT 0,
                        prompt_tokens INTEGER NOT NULL DEFAULT 0,
                        completion_tokens INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (key_id, day)
                    );",
                )
                .expect("v3");
            }
            if ver >= 4 {
                con.execute_batch(
                    "CREATE TABLE IF NOT EXISTS engine_overhead (
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
                        restart_in_flight INTEGER NOT NULL DEFAULT 0
                    );",
                )
                .expect("v4");
                let inst_cols = table_cols(&con, "loaded_instances").expect("cols");
                add_col_if_missing(&con, "loaded_instances", &inst_cols, "slot_key", "TEXT")
                    .expect("slot_key");
                let des_cols = table_cols(&con, "desired_loads").expect("dcols");
                add_col_if_missing(&con, "desired_loads", &des_cols, "slot_key", "TEXT")
                    .expect("dslot");
            }
            if (4..5).contains(&ver) {
                con.execute_batch(
                    "CREATE TABLE IF NOT EXISTS watchdog_restarts (
                        slot_key TEXT PRIMARY KEY,
                        window_start INTEGER NOT NULL,
                        count INTEGER NOT NULL
                    );",
                )
                .ok();
            }
            con.execute_batch(seed).expect("seed");
            con.execute_batch(&format!("PRAGMA user_version = {ver};"))
                .expect("ver");
        }
        open(home).expect("migrate")
    }

    #[test]
    fn migrate_v0_installed_model_row() {
        let dir = tempfile::tempdir().expect("tempdir");
        let con = open_migrated(
            dir.path(),
            0,
            "INSERT INTO installed_models (model_id, local_path, architecture, static_vram_gb,
             kv_bytes_per_token, default_ctx, default_max_conc, default_tp)
             VALUES ('m','p','dense',1.0,100,4096,4,1);",
        );
        let ver: i32 = con.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
        assert_eq!(ver, USER_VERSION);
        let arch: String = con
            .query_row(
                "SELECT architecture FROM installed_models WHERE model_id='m'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(arch, "dense");
    }

    #[test]
    fn migrate_v2_desired_and_api_keys() {
        let dir = tempfile::tempdir().expect("tempdir");
        let con = open_migrated(
            dir.path(),
            2,
            "INSERT INTO desired_loads (model_id, gpus, ctx, max_conc, tp, extra_sglang_args)
             VALUES ('m','2',4096,4,1,'[]');
             INSERT INTO api_keys (key_id, name, token_hash, raw_prefix)
             VALUES ('k','n','h','p');",
        );
        let n: i64 = con
            .query_row("SELECT COUNT(*) FROM desired_loads", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
        let sk: String = con
            .query_row("SELECT slot_key FROM desired_loads LIMIT 1", [], |r| r.get(0))
            .unwrap();
        assert!(!sk.is_empty());
    }

    #[test]
    fn migrate_v3_key_usage() {
        let dir = tempfile::tempdir().expect("tempdir");
        let con = open_migrated(
            dir.path(),
            3,
            "INSERT INTO api_keys (key_id, name, token_hash, raw_prefix) VALUES ('k','n','h','p');
             INSERT INTO key_usage (key_id, day, requests, prompt_tokens, completion_tokens)
             VALUES ('k','2020-01-01',1,2,3);",
        );
        let req: i64 = con
            .query_row("SELECT requests FROM key_usage WHERE key_id='k'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(req, 1);
    }

    #[test]
    fn migrate_v4_slot_key_on_desired() {
        let dir = tempfile::tempdir().expect("tempdir");
        let con = open_migrated(
            dir.path(),
            4,
            "INSERT INTO desired_loads (model_id, gpus, slot_key, extra_sglang_args)
             VALUES ('m','2','m|2|0','[]');",
        );
        let sk: String = con
            .query_row("SELECT slot_key FROM desired_loads LIMIT 1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(sk, "m|2|0");
    }

    #[test]
    fn migrate_v6_policy_and_admin() {
        let dir = tempfile::tempdir().expect("tempdir");
        let con = open_migrated(
            dir.path(),
            5,
            "INSERT INTO api_keys (key_id, name, token_hash, raw_prefix, rpm_limit)
             VALUES ('k','n','h','p',120);",
        );
        let ver: i32 = con.query_row("PRAGMA user_version", [], |r| r.get(0)).unwrap();
        assert_eq!(ver, 6);
        let cols = table_cols(&con, "api_keys").expect("cols");
        assert!(cols.iter().any(|c| c == "is_admin"));
        let gp: i64 = con
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='gpu_policy'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(gp, 1);
        let ma: i64 = con
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='model_aliases'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(ma, 1);
        assert!(dir.path().join("state.db.v5.bak").is_file());
    }
}
