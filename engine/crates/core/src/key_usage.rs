// API key usage accounting (per key, per day).

use crate::db;
use crate::paths::goethe_home;
use chrono::Utc;
use rusqlite::Connection;

pub fn today_utc() -> String {
    Utc::now().format("%Y-%m-%d").to_string()
}

pub fn record_request(key_id: &str, prompt_tokens: i64, completion_tokens: i64) -> Result<(), String> {
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    record_request_con(&con, key_id, prompt_tokens, completion_tokens)
}

pub fn record_request_con(
    con: &Connection,
    key_id: &str,
    prompt_tokens: i64,
    completion_tokens: i64,
) -> Result<(), String> {
    let day = today_utc();
    con.execute(
        "INSERT INTO key_usage (key_id, day, requests, prompt_tokens, completion_tokens)
         VALUES (?1, ?2, 1, ?3, ?4)
         ON CONFLICT(key_id, day) DO UPDATE SET
           requests = requests + 1,
           prompt_tokens = prompt_tokens + excluded.prompt_tokens,
           completion_tokens = completion_tokens + excluded.completion_tokens",
        rusqlite::params![key_id, day, prompt_tokens, completion_tokens],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn totals_for_key(con: &Connection, key_id: &str) -> Result<(i64, i64, i64, i64, i64, i64), String> {
    let day = today_utc();
    let today: (i64, i64, i64) = con
        .query_row(
            "SELECT requests, prompt_tokens, completion_tokens FROM key_usage WHERE key_id = ?1 AND day = ?2",
            rusqlite::params![key_id, day],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap_or((0, 0, 0));
    let all: (i64, i64, i64) = con
        .query_row(
            "SELECT COALESCE(SUM(requests),0), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0)
             FROM key_usage WHERE key_id = ?1",
            [key_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .map_err(|e| e.to_string())?;
    Ok((
        today.0, today.1, today.2, all.0, all.1, all.2,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    const OLD_V2_DDL: &str = r"
CREATE TABLE api_keys (
    key_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    token_hash TEXT UNIQUE NOT NULL,
    raw_prefix TEXT NOT NULL,
    rpm_limit INTEGER DEFAULT 120,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
PRAGMA user_version = 2;
";

    #[test]
    fn migrate_v2_to_v3_adds_key_usage() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = PathBuf::from(dir.path()).join("state.db");
        {
            let con = rusqlite::Connection::open(&db_path).expect("open");
            con.execute_batch(OLD_V2_DDL).expect("old ddl");
        }
        let con = db::open(dir.path()).expect("migrate");
        let ver: i32 = con
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .unwrap();
        assert_eq!(ver, 5);
        let cols: Vec<String> = con
            .prepare("PRAGMA table_info(api_keys)")
            .unwrap()
            .query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert!(cols.iter().any(|c| c == "expires_at"));
        assert!(cols.iter().any(|c| c == "model_scope"));
        con.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='key_usage'",
            [],
            |r| r.get::<_, i64>(0),
        )
        .map(|n| assert_eq!(n, 1))
        .unwrap();
    }
}
