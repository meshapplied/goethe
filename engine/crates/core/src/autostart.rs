// Desired load set kept across restarts, restored oldest first through the per-GPU load gate.

use crate::db;
use crate::paths::goethe_home;
use crate::supervisor::{self, LoadParams};
use rusqlite::Connection;
use serde_json::Value;
use std::path::Path;
use std::thread;

pub fn mark_desired(con: &Connection, params: &LoadParams) -> rusqlite::Result<()> {
    // Upsert the slot so a re-insert keeps its original created_at, which orders restore.
    let extra = serde_json::to_string(&params.extra_sglang_args).unwrap_or_else(|_| "[]".to_string());
    let slot = params.slot_key.clone().unwrap_or_else(|| params.model_id.clone());
    con.execute(
        "INSERT INTO desired_loads (slot_key, model_id, gpus, ctx, max_conc, tp, extra_sglang_args)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
         ON CONFLICT(slot_key) DO UPDATE SET model_id = excluded.model_id, gpus = excluded.gpus,
           ctx = excluded.ctx, max_conc = excluded.max_conc, tp = excluded.tp,
           extra_sglang_args = excluded.extra_sglang_args",
        rusqlite::params![
            slot,
            params.model_id,
            params.gpus,
            params.ctx.map(|c| c as i64),
            params.max_conc.map(|c| c as i64),
            params.tp.map(|c| c as i64),
            extra,
        ],
    )?;
    Ok(())
}

pub fn desired_exists(con: &Connection, slot_key: &str) -> bool {
    con.query_row("SELECT 1 FROM desired_loads WHERE slot_key = ?1", [slot_key], |_| Ok(()))
        .is_ok()
}

pub fn restore_enabled(home: &Path) -> bool {
    let Ok(con) = db::open(home) else {
        return true;
    };
    let Ok(v) = con.query_row(
        "SELECT value FROM goethe_settings WHERE key = 'restore_desired'",
        [],
        |r| r.get::<_, String>(0),
    ) else {
        return true;
    };
    v != "off"
}

pub fn set_restore_enabled(on: bool) -> Result<(), String> {
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let v = if on { "on" } else { "off" };
    con.execute(
        "INSERT OR REPLACE INTO goethe_settings (key, value) VALUES ('restore_desired', ?1)",
        [v],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn spawn_restore_background() {
    if !restore_enabled(&goethe_home()) {
        return;
    }
    thread::spawn(|| {
        if let Err(e) = restore_desired_sequential() {
            eprintln!("restore desired: {e}");
        }
    });
}

fn restore_desired_sequential() -> Result<(), String> {
    // Relaunch desired slots oldest first; handle_load waits at the per-GPU gate, so a GPU starts one engine at a time.
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    for params in desired_rows(&con)? {
        if crate::shutdown::shutdown_requested() {
            break;
        }
        let slot = params.slot_key.clone().unwrap_or_else(|| params.model_id.clone());
        if supervisor::slot_has_running_instance(&slot) {
            continue;
        }
        if let Err((code, msg, _)) = supervisor::handle_load(params) {
            eprintln!("restore load {slot} failed ({code}): {msg}");
        }
    }
    Ok(())
}

fn desired_rows(con: &Connection) -> Result<Vec<LoadParams>, String> {
    // Desired slots in creation order, as load parameters for restore.
    let mut stmt = con
        .prepare(
            "SELECT model_id, gpus, ctx, max_conc, tp, extra_sglang_args, slot_key FROM desired_loads
             ORDER BY created_at, rowid",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| {
            let extra: String = r.get(5)?;
            let extra: Value = serde_json::from_str(&extra).unwrap_or(Value::Array(vec![]));
            Ok(LoadParams {
                model_id: r.get(0)?,
                gpus: r.get(1)?,
                ctx: r.get::<_, Option<i64>>(2)?.map(|v| v as u64),
                max_conc: r.get::<_, Option<i64>>(3)?.map(|v| v as u64),
                tp: r.get::<_, Option<i64>>(4)?.map(|v| v as u64),
                extra_sglang_args: extra
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect(),
                force: false,
                yes: true,
                restart_count: None,
                slot_key: r.get(6)?,
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(slot: &str, ctx: u64) -> LoadParams {
        LoadParams {
            model_id: slot.split('|').next().unwrap_or(slot).to_string(),
            gpus: Some("3".to_string()),
            extra_sglang_args: Vec::new(),
            ctx: Some(ctx),
            max_conc: None,
            tp: None,
            force: false,
            yes: false,
            restart_count: None,
            slot_key: Some(slot.to_string()),
        }
    }

    #[test]
    fn restore_follows_created_at_and_upsert_keeps_it() {
        let td = tempfile::tempdir().unwrap();
        let con = db::open(td.path()).unwrap();
        for (slot, created) in [("z|3|0", 100), ("a|3|0", 300), ("m|3|0", 200)] {
            mark_desired(&con, &params(slot, 4096)).unwrap();
            con.execute("UPDATE desired_loads SET created_at = ?1 WHERE slot_key = ?2", rusqlite::params![created, slot])
                .unwrap();
        }
        let order: Vec<String> = desired_rows(&con).unwrap().into_iter().filter_map(|p| p.slot_key).collect();
        assert_eq!(order, vec!["z|3|0", "m|3|0", "a|3|0"]);
        mark_desired(&con, &params("z|3|0", 8192)).unwrap();
        let (created, ctx): (i64, i64) = con
            .query_row("SELECT created_at, ctx FROM desired_loads WHERE slot_key = 'z|3|0'", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .unwrap();
        assert_eq!((created, ctx), (100, 8192));
        assert!(desired_exists(&con, "m|3|0") && !desired_exists(&con, "q|3|0"));
    }
}
