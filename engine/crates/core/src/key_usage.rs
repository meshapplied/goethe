// API key usage accounting (per key, per day).

use chrono::Utc;
use rusqlite::Connection;

pub fn today_utc() -> String {
    Utc::now().format("%Y-%m-%d").to_string()
}

pub fn add_usage(
    con: &Connection,
    key_id: &str,
    requests: i64,
    prompt_tokens: i64,
    completion_tokens: i64,
) -> Result<(), String> {
    // Add a batch of requests and tokens to today's row for one key.
    con.execute(
        "INSERT INTO key_usage (key_id, day, requests, prompt_tokens, completion_tokens)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(key_id, day) DO UPDATE SET
           requests = requests + excluded.requests,
           prompt_tokens = prompt_tokens + excluded.prompt_tokens,
           completion_tokens = completion_tokens + excluded.completion_tokens",
        rusqlite::params![key_id, today_utc(), requests, prompt_tokens, completion_tokens],
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
    use crate::db;

    #[test]
    fn usage_accumulates_per_key() {
        let dir = tempfile::tempdir().expect("tempdir");
        let con = db::open(dir.path()).expect("open");
        add_usage(&con, "k1", 1, 10, 5).unwrap();
        add_usage(&con, "k1", 2, 3, 2).unwrap();
        add_usage(&con, "k2", 1, 1, 1).unwrap();
        let (req, prompt, completion, all_req, _, all_completion) = totals_for_key(&con, "k1").unwrap();
        assert_eq!((req, prompt, completion), (3, 13, 7));
        assert_eq!((all_req, all_completion), (3, 7));
        assert_eq!(totals_for_key(&con, "missing").unwrap(), (0, 0, 0, 0, 0, 0));
    }
}
