// GPU allow/deny policy stored in state.db.

use crate::hw;
use crate::placement;
use rusqlite::Connection;
use std::collections::HashSet;
use std::path::Path;

pub fn denied_set(con: &Connection) -> Result<HashSet<u32>, String> {
    let mut out = HashSet::new();
    let mut stmt = con
        .prepare("SELECT gpu_index FROM gpu_policy WHERE denied = 1")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| r.get::<_, i64>(0))
        .map_err(|e| e.to_string())?;
    for row in rows.filter_map(|r| r.ok()) {
        if row >= 0 {
            out.insert(row as u32);
        }
    }
    Ok(out)
}

pub fn set_denied(con: &Connection, index: u32, denied: bool) -> Result<(), String> {
    con.execute(
        "INSERT OR REPLACE INTO gpu_policy (gpu_index, denied, updated_at) VALUES (?1, ?2, CURRENT_TIMESTAMP)",
        rusqlite::params![index as i64, if denied { 1i64 } else { 0i64 }],
    )
    .map_err(|e| e.to_string())?;
    Ok(())
}

pub fn rows(con: &Connection) -> Result<Vec<(u32, bool)>, String> {
    let mut stmt = con
        .prepare("SELECT gpu_index, denied FROM gpu_policy ORDER BY gpu_index")
        .map_err(|e| e.to_string())?;
    let mapped = stmt
        .query_map([], |r| Ok((r.get::<_, i64>(0)? as u32, r.get::<_, i64>(1)? != 0)))
        .map_err(|e| e.to_string())?;
    mapped.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GpuListRow {
    pub index: u32,
    pub name: String,
    pub compute_capability: String,
    pub vram_total_mb: u64,
    pub managed_mb: u64,
    pub foreign_mb: u64,
    pub status: String,
}

pub fn list_rows(con: &Connection, home: &Path) -> Result<Vec<GpuListRow>, String> {
    let state = hw::load_hardware_state(home)?;
    let managed = placement::managed_pids(con);
    let denied = denied_set(con)?;
    let mut out = Vec::new();
    for gpu in &state.gpus {
        let managed_mb = hw::managed_vram_listed_mb(gpu.index, &managed)?;
        let used_mb = hw::gpu_used_mb_live(gpu.index)?;
        let foreign_mb = used_mb.saturating_sub(managed_mb);
        let is_denied = denied.contains(&gpu.index);
        out.push(GpuListRow {
            index: gpu.index,
            name: gpu.name.clone(),
            compute_capability: gpu.compute_capability.clone(),
            vram_total_mb: gpu.vram_total_mb,
            managed_mb,
            foreign_mb,
            status: if is_denied {
                "DENIED".to_string()
            } else {
                "ALLOWED".to_string()
            },
        });
    }
    out.sort_by_key(|r| r.index);
    Ok(out)
}

pub fn validate_gpu_index(home: &Path, index: u32) -> Result<(), String> {
    let indices = hw::profile_gpu_indices(home)?;
    if indices.contains(&index) {
        Ok(())
    } else {
        Err(format!("GPU {index} is not in the hardware profile; run `goethe hw profile`"))
    }
}

pub fn instances_on_gpu(con: &Connection, index: u32) -> Result<Vec<String>, String> {
    let needle = index.to_string();
    let mut stmt = con
        .prepare("SELECT instance_id, gpu_indices FROM loaded_instances WHERE status != 'STOPPING'")
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for row in rows.filter_map(|r| r.ok()) {
        if row
            .1
            .split(',')
            .any(|s| s.trim() == needle)
        {
            out.push(row.0);
        }
    }
    Ok(out)
}
