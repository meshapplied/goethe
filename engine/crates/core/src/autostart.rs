// Desired load set and restore after goethed restart.

use crate::db;
use crate::paths::goethe_home;
use crate::supervisor::{self, LoadParams};
use serde_json::Value;
use std::path::Path;
use std::thread;

pub fn mark_desired(params: &LoadParams) -> Result<(), String> {
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let extra = serde_json::to_string(&params.extra_sglang_args).map_err(|e| e.to_string())?;
    let slot = params.slot_key.clone().unwrap_or_else(|| {
        params.model_id.clone()
    });
    con.execute(
        "INSERT OR REPLACE INTO desired_loads
        (slot_key, model_id, gpus, ctx, max_conc, tp, extra_sglang_args)
        VALUES (?1,?2,?3,?4,?5,?6,?7)",
        rusqlite::params![
            slot,
            params.model_id,
            params.gpus,
            params.ctx.map(|c| c as i64),
            params.max_conc.map(|c| c as i64),
            params.tp.map(|c| c as i64),
            extra,
        ],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn clear_desired(model_id: &str) -> Result<(), String> {
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    con.execute("DELETE FROM desired_loads WHERE model_id = ?1", [model_id])
        .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn clear_all_desired() -> Result<(), String> {
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    con.execute("DELETE FROM desired_loads", [])
        .map_err(|e| e.to_string())?;
    Ok(())
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
    let home = goethe_home();
    let rows = desired_rows(&home)?;
    for params in rows {
        let slot = params
            .slot_key
            .as_deref()
            .unwrap_or(params.model_id.as_str());
        if supervisor::slot_has_running_instance(slot) {
            continue;
        }
        let mut p = params;
        p.async_finish = true;
        if let Err((code, msg)) = supervisor::handle_load(p) {
            eprintln!("restore load failed ({code}): {msg}");
        }
        thread::sleep(std::time::Duration::from_millis(500));
    }
    Ok(())
}

fn desired_rows(home: &Path) -> Result<Vec<LoadParams>, String> {
    let con = db::open(home).map_err(|e| e.to_string())?;
    let mut stmt = con
        .prepare(
            "SELECT model_id, gpus, ctx, max_conc, tp, extra_sglang_args, slot_key FROM desired_loads ORDER BY slot_key",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| {
            let extra: String = r.get(5)?;
            let extra: Value = serde_json::from_str(&extra).unwrap_or(Value::Array(vec![]));
            let arr = extra
                .as_array()
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
            Ok(LoadParams {
                model_id: r.get(0)?,
                gpus: r.get(1)?,
                ctx: r.get::<_, Option<i64>>(2)?.map(|v| v as u64),
                max_conc: r.get::<_, Option<i64>>(3)?.map(|v| v as u64),
                tp: r.get::<_, Option<i64>>(4)?.map(|v| v as u64),
                extra_sglang_args: arr,
                force: false,
                yes: true,
                is_tty: false,
                restart_count: None,
                slot_key: r.get(6)?,
                async_finish: true,
            })
        })
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row.map_err(|e| e.to_string())?);
    }
    Ok(out)
}

