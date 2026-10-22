// mesh models list and remove.

use crate::calc::vram_per_gpu_gb;
use crate::db;
use crate::hw::HardwareState;
use crate::paths::{mesh_home, model_dir};
use crate::registry::{fetch, registry_url};
use crate::rpc;
use rusqlite::Connection;
use serde::Serialize;
use serde_json::json;
use std::fs;
use std::io::IsTerminal;
use std::path::Path;

#[derive(Debug, Serialize)]
struct ModelRow {
    id: String,
    status: String,
    gb: f64,
    fit: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    custom: Option<bool>,
}

pub fn run_list(json: bool, installed_only: bool, available_only: bool) -> Result<(), String> {
    let home = mesh_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let loaded = loaded_set(&con);
    let installed = installed_rows(&con)?;
    let reg = fetch(&registry_url()).ok();
    let hw = load_hw(&home);
    let mut rows = Vec::new();
    if !available_only {
        for (id, gb) in &installed {
            let status = if loaded.contains(id) {
                "loaded"
            } else {
                "installed"
            };
            let custom = is_custom_model(id, reg.as_ref());
            rows.push(ModelRow {
                id: id.clone(),
                status: status.to_string(),
                gb: *gb,
                fit: fit_label(id, *gb, reg.as_ref(), &hw),
                custom: custom.then_some(true),
            });
        }
    }
    if !installed_only {
        if let Some(r) = &reg {
            for m in &r.models {
                if installed.iter().any(|(id, _)| id == &m.id) {
                    continue;
                }
                rows.push(ModelRow {
                    id: m.id.clone(),
                    status: "available".to_string(),
                    gb: m.download_gb.unwrap_or(m.static_vram_gb),
                    fit: fit_label(&m.id, m.static_vram_gb, reg.as_ref(), &hw),
                    custom: None,
                });
            }
        }
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?);
    } else {
        for r in rows {
            println!("{}\t{}\t{:.2} GB\t{}", r.id, r.status, r.gb, r.fit);
        }
    }
    Ok(())
}

pub fn run_remove(model_id: &str, yes: bool, force: bool) -> Result<(), String> {
    let home = mesh_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    if is_loaded(&con, model_id) {
        if !force {
            return Err(format!("model {model_id} is loaded; use --force to unload and remove"));
        }
        let _ = rpc::call(
            "unload",
            json!({"model_id": model_id, "all": false}),
        )
        .map_err(|e| e.to_string())?;
    }
    if !yes {
        if !std::io::stdin().is_terminal() {
            return Err("refusing to remove without --yes (non-interactive)".to_string());
        }
        print!("remove {model_id} and delete files? [y/N] ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf).map_err(|e| e.to_string())?;
        if !buf.trim().eq_ignore_ascii_case("y") {
            return Err("remove aborted".to_string());
        }
    }
    let local_path: Option<String> = con
        .query_row(
            "SELECT local_path FROM installed_models WHERE model_id = ?1",
            [model_id],
            |r| r.get(0),
        )
        .ok();
    let base_to_check = local_path.clone();
    con.execute("DELETE FROM installed_models WHERE model_id = ?1", [model_id])
        .map_err(|e| e.to_string())?;
    if let Some(rel) = local_path {
        let dir = home.join(rel);
        if dir.is_symlink() {
            fs::remove_file(&dir).map_err(|e| e.to_string())?;
        } else if !model_id.starts_with("local/") && dir.is_dir() {
            fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
        }
    } else {
        let dir = model_dir(&home, model_id);
        if dir.is_dir() {
            fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
        }
    }
    maybe_remove_orphan_base(&home, &con, model_id, base_to_check.as_deref());
    Ok(())
}

fn maybe_remove_orphan_base(home: &Path, con: &Connection, removed_id: &str, _path: Option<&str>) {
    let Ok(reg) = fetch(&registry_url()) else {
        return;
    };
    let Some(entry) = reg.models.iter().find(|m| m.id == removed_id) else {
        return;
    };
    let Some(base) = entry.base_model_id.as_deref() else {
        return;
    };
    if base == removed_id {
        return;
    }
    let mut stmt = con
        .prepare("SELECT model_id FROM installed_models")
        .ok();
    let mut still_used = false;
    if let Some(s) = &mut stmt {
        if let Ok(rows) = s.query_map([], |r| r.get::<_, String>(0)) {
            for id in rows.filter_map(|r| r.ok()) {
                if reg
                    .models
                    .iter()
                    .any(|m| m.id == id && m.base_model_id.as_deref() == Some(base))
                {
                    still_used = true;
                    break;
                }
            }
        }
    }
    if !still_used {
        let dir = model_dir(home, base);
        if dir.is_dir() {
            let _ = fs::remove_dir_all(dir);
        }
    }
}

fn installed_rows(con: &Connection) -> Result<Vec<(String, f64)>, String> {
    let mut stmt = con
        .prepare("SELECT model_id, static_vram_gb FROM installed_models ORDER BY model_id")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?)))
        .map_err(|e| e.to_string())?;
    Ok(rows.filter_map(|r| r.ok()).collect())
}

fn loaded_set(con: &Connection) -> std::collections::HashSet<String> {
    let mut set = std::collections::HashSet::new();
    let Ok(mut stmt) = con.prepare("SELECT DISTINCT model_id FROM loaded_instances WHERE status = 'RUNNING'") else {
        return set;
    };
    if let Ok(rows) = stmt.query_map([], |r| r.get::<_, String>(0)) {
        for id in rows.filter_map(|r| r.ok()) {
            set.insert(id);
        }
    }
    set
}

fn is_loaded(con: &Connection, model_id: &str) -> bool {
    let Ok(n) = con.query_row(
        "SELECT COUNT(*) FROM loaded_instances WHERE model_id = ?1 AND status = 'RUNNING'",
        [model_id],
        |r| r.get::<_, i64>(0),
    ) else {
        return false;
    };
    n > 0
}

fn load_hw(home: &Path) -> Option<HardwareState> {
    let path = home.join("hardware_state.json");
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn is_custom_model(id: &str, reg: Option<&crate::registry::Registry>) -> bool {
    if id.starts_with("local/") {
        return true;
    }
    reg.is_none_or(|r| !r.models.iter().any(|m| m.id == id))
}

fn fit_label(
    model_id: &str,
    static_gb: f64,
    reg: Option<&crate::registry::Registry>,
    hw: &Option<HardwareState>,
) -> String {
    let Some(hw) = hw else {
        return "unknown (run mesh hw profile)".to_string();
    };
    let Some(reg) = reg else {
        return "unknown".to_string();
    };
    let Some(m) = reg.models.iter().find(|x| x.id == model_id) else {
        return "ok".to_string();
    };
    let sm_ok = m
        .supported_sm
        .iter()
        .any(|sm| hw.gpus.iter().any(|g| g.compute_capability.starts_with(sm)));
    if !sm_ok {
        return "unsupported SM".to_string();
    }
    let ctx = m.effective_max_ctx().unwrap_or(4096).min(131072);
    let kv = m.effective_kv_elems().unwrap_or(1);
    let conc = m.default_max_conc.unwrap_or(8);
    let tp = m.default_tp.filter(|t| *t > 0).unwrap_or(1);
    let need = vram_per_gpu_gb(static_gb, ctx, kv, conc, tp);
    let free_gb = hw
        .gpus
        .iter()
        .map(|g| {
            let mut f = g.vram_free_mb as f64 / 1024.0;
            if g.unified_memory {
                f = (f - crate::hw::UNIFIED_OS_RESERVE_GB as f64).max(0.0);
            }
            f
        })
        .fold(0.0_f64, f64::max);
    let um = hw.gpus.iter().any(|g| g.unified_memory);
    let um_note = if um { " unified memory" } else { "" };
    if need <= free_gb {
        format!("ok{um_note}")
    } else {
        format!("needs {:.1} GB more{um_note}", need - free_gb)
    }
}
