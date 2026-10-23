// Model alias storage in SQLite.

use crate::db;
use crate::paths::goethe_home;
use rusqlite::Connection;

pub fn set(alias: &str, model_id: &str) -> Result<(), String> {
    // Refuse an empty alias and one that would shadow an installed model id, then store it.
    if alias.trim().is_empty() {
        return Err("alias must not be empty".to_string());
    }
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let shadows: i64 = con
        .query_row("SELECT COUNT(*) FROM installed_models WHERE model_id = ?1", [alias], |r| r.get(0))
        .map_err(|e| e.to_string())?;
    if shadows > 0 {
        return Err(format!("{alias} is an installed model id; pick another alias name"));
    }
    if alias.contains('/') && crate::paths::validate_model_id(alias).is_ok() {
        eprintln!(
            "warning: {alias} looks like a model id; if a model with that id is installed later, it takes the name and this alias is removed"
        );
    }
    let n: i64 = con
        .query_row(
            "SELECT COUNT(*) FROM installed_models WHERE model_id = ?1",
            [model_id],
            |r| r.get(0),
        )
        .map_err(|e| e.to_string())?;
    if n == 0 {
        return Err(format!("model {model_id} is not installed"));
    }
    con.execute(
        "INSERT OR REPLACE INTO model_aliases (alias, model_id) VALUES (?1, ?2)",
        rusqlite::params![alias, model_id],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn remove(alias: &str) -> Result<(), String> {
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let n = con
        .execute("DELETE FROM model_aliases WHERE alias = ?1", [alias])
        .map_err(|e| e.to_string())?;
    if n == 0 {
        return Err(format!("alias {alias} not found"));
    }
    Ok(())
}

pub fn list() -> Result<Vec<(String, String)>, String> {
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let mut stmt = con
        .prepare("SELECT alias, model_id FROM model_aliases ORDER BY alias")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
        .map_err(|e| e.to_string())?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

pub fn resolve(con: &Connection, name: &str) -> Option<String> {
    con.query_row(
        "SELECT model_id FROM model_aliases WHERE alias = ?1",
        [name],
        |r| r.get(0),
    )
    .ok()
}
