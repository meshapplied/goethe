// Automatic GPU selection for mesh load when preferred_gpus is empty.

use crate::hw::{self, GpuPlacementSnapshot};
use rusqlite::Connection;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuPlacementInput {
    pub index: u32,
    pub foreign_process: bool,
    pub free_vram_mb: u64,
}

impl From<GpuPlacementSnapshot> for GpuPlacementInput {
    fn from(s: GpuPlacementSnapshot) -> Self {
        Self {
            index: s.index,
            foreign_process: s.foreign_compute,
            free_vram_mb: s.free_vram_mb,
        }
    }
}

pub fn ensure_eligible_for_tp(
    inputs: &[GpuPlacementInput],
    tp: usize,
    denied: &HashSet<u32>,
) -> Result<(), String> {
    if tp == 0 {
        return Err("tensor parallel size must be at least 1".to_string());
    }
    if inputs.len() >= tp {
        return Ok(());
    }
    if denied.is_empty() {
        return Err(format!(
            "need {tp} GPUs for tp but only {} visible",
            inputs.len()
        ));
    }
    let mut ids: Vec<String> = denied.iter().map(|i| i.to_string()).collect();
    ids.sort();
    Err(format!(
        "need {tp} eligible GPUs but only {} available; denied GPUs: {}; run `mesh gpu allow <id>`",
        inputs.len(),
        ids.join(",")
    ))
}

pub fn rank_gpus_for_placement(inputs: &[GpuPlacementInput], tp: usize) -> Result<Vec<u32>, String> {
    ensure_eligible_for_tp(inputs, tp, &HashSet::new())?;
    let mut sorted: Vec<&GpuPlacementInput> = inputs.iter().collect();
    sorted.sort_by(|a, b| {
        a.foreign_process
            .cmp(&b.foreign_process)
            .then(b.free_vram_mb.cmp(&a.free_vram_mb))
            .then(a.index.cmp(&b.index))
    });
    let mut picked: Vec<u32> = sorted.iter().take(tp).map(|g| g.index).collect();
    picked.sort_unstable();
    Ok(picked)
}

pub fn auto_pick_gpus(con: &Connection, tp: usize) -> Result<Vec<u32>, String> {
    let managed = managed_pids(con);
    let snaps = hw::list_gpus_for_placement(&managed)?;
    let denied = crate::gpu_policy::denied_set(con)?;
    let inputs: Vec<GpuPlacementInput> = snaps
        .into_iter()
        .map(GpuPlacementInput::from)
        .filter(|g| !denied.contains(&g.index))
        .collect();
    ensure_eligible_for_tp(&inputs, tp, &denied)?;
    rank_gpus_for_placement(&inputs, tp)
}

pub fn run_pick_gpus(tp: u64) -> Result<(), String> {
    let home = crate::paths::mesh_home();
    let con = crate::db::open(&home).map_err(|e| e.to_string())?;
    let gpus = auto_pick_gpus(&con, tp.max(1) as usize)?;
    println!("placed on GPU {}", format_gpu_list(&gpus));
    Ok(())
}

pub fn format_gpu_list(gpus: &[u32]) -> String {
    gpus.iter()
        .map(|g| g.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

pub fn managed_pids(con: &Connection) -> HashSet<i32> {
    let mut set = HashSet::new();
    let Ok(mut stmt) = con.prepare("SELECT pid FROM loaded_instances") else {
        return set;
    };
    let rows = stmt.query_map([], |r| r.get::<_, i64>(0));
    if let Ok(rows) = rows {
        for pid in rows.filter_map(|r| r.ok()) {
            set.insert(pid as i32);
            for child in descendant_pids(pid as i32) {
                set.insert(child);
            }
        }
    }
    set
}

fn descendant_pids(root: i32) -> Vec<i32> {
    let mut out = Vec::new();
    let mut stack = vec![root];
    while let Some(p) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir("/proc") {
            for ent in entries.filter_map(|e| e.ok()) {
                let name = ent.file_name().to_string_lossy().to_string();
                if let Ok(child) = name.parse::<i32>() {
                    if child == p {
                        continue;
                    }
                    if proc_ppid(child) == Some(p) {
                        out.push(child);
                        stack.push(child);
                    }
                }
            }
        }
    }
    out
}

fn proc_ppid(pid: i32) -> Option<i32> {
    let stat = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let rparen = stat.rfind(')')?;
    let rest = stat.get(rparen + 2..)?;
    rest.split_whitespace().nth(1)?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inp(idx: u32, foreign: bool, free_mb: u64) -> GpuPlacementInput {
        GpuPlacementInput {
            index: idx,
            foreign_process: foreign,
            free_vram_mb: free_mb,
        }
    }

    #[test]
    fn clean_gpu_beats_foreign() {
        let gpus = rank_gpus_for_placement(
            &[inp(0, true, 90000), inp(1, false, 1000)],
            1,
        )
        .unwrap();
        assert_eq!(gpus, vec![1]);
    }

    #[test]
    fn orders_by_free_vram() {
        let gpus = rank_gpus_for_placement(
            &[inp(0, false, 1000), inp(1, false, 5000), inp(2, false, 3000)],
            1,
        )
        .unwrap();
        assert_eq!(gpus, vec![1]);
    }

    #[test]
    fn tie_breaks_to_lowest_index() {
        let gpus = rank_gpus_for_placement(
            &[inp(2, false, 5000), inp(3, false, 5000), inp(4, false, 5000)],
            1,
        )
        .unwrap();
        assert_eq!(gpus, vec![2]);
    }

    #[test]
    fn tp_two_returns_ascending_pair() {
        let gpus = rank_gpus_for_placement(
            &[
                inp(0, true, 90000),
                inp(1, true, 90000),
                inp(2, false, 8000),
                inp(3, false, 7000),
                inp(4, false, 6000),
            ],
            2,
        )
        .unwrap();
        assert_eq!(gpus, vec![2, 3]);
    }
}
