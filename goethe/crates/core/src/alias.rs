// Model alias storage in SQLite.

use crate::db;
use crate::paths::goethe_home;
use rusqlite::Connection;

fn name_check(alias: &str) -> Result<(), String> {
    // An alias clients can type in a request: 1 to 64 characters, no spaces or control characters; matched exactly, case included.
    if alias.is_empty() || alias.chars().count() > 64 || alias.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return Err(format!("{alias:?} is not an alias: use 1 to 64 letters, digits or punctuation, no spaces"));
    }
    Ok(())
}

pub fn set(alias: &str, model_id: &str) -> Result<(), String> {
    // Refuse a malformed alias and one that would shadow an installed model id, then store it.
    name_check(alias)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_are_one_word_without_control_characters() {
        assert!(name_check("fast").is_ok() && name_check("org/small-v2").is_ok() && name_check("Fast.1").is_ok());
        for bad in ["", "two words", "tab\there", "esc\x1b[2J", " lead", &"x".repeat(65)] {
            assert!(name_check(bad).is_err(), "{bad:?}");
        }
    }
}
