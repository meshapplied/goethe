// GPU set ranking and the view of which processes belong to live Goethe instances.

use crate::foreign::{self, GpuLive};
use crate::gpu_arch;
use crate::hw::{self, HardwareState};
use crate::proc_ident::{self, Identity};
use rusqlite::Connection;
use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuPlacementInput {
    pub index: u32,
    pub cc: String,
    pub foreign_process: bool,
    pub free_vram_mb: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetRank {
    pub gpus: Vec<u32>,
    pub cc: String,
    pub foreign: usize,
    pub link: u32,
    pub min_free_mb: u64,
}

pub fn ensure_eligible_for_tp(
    inputs: &[GpuPlacementInput],
    tp: usize,
    denied: &HashSet<u32>,
) -> Result<(), String> {
    // Explain up front when too few GPUs are visible or allowed for the requested tensor parallel size.
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
        "need {tp} eligible GPUs but only {} available; denied GPUs: {}; run `goethe gpu --allow gpu<index>`",
        inputs.len(),
        ids.join(",")
    ))
}

fn combos(n: usize, k: usize) -> Vec<Vec<usize>> {
    // Every k-subset of 0..n in lexicographic order.
    if k == 0 || k > n {
        return Vec::new();
    }
    let mut idx: Vec<usize> = (0..k).collect();
    let mut out = vec![idx.clone()];
    while let Some(i) = (0..k).rev().find(|&i| idx[i] != i + n - k) {
        idx[i] += 1;
        for j in i + 1..k {
            idx[j] = idx[j - 1] + 1;
        }
        out.push(idx.clone());
    }
    out
}

pub fn rank_sets(inputs: &[GpuPlacementInput], hw: Option<&HardwareState>, tp: usize) -> Vec<SetRank> {
    // Every tp-sized set of one compute capability (SGLang picks kernels per rank), fewest foreign, most free, best link.
    let mut sorted: Vec<&GpuPlacementInput> = inputs.iter().collect();
    sorted.sort_by_key(|g| g.index);
    let mut out: Vec<SetRank> = combos(sorted.len(), tp)
        .into_iter()
        .filter(|c| c.iter().all(|&i| gpu_arch::same_cc(&sorted[i].cc, &sorted[c[0]].cc)))
        .map(|c| {
            let gpus: Vec<u32> = c.iter().map(|&i| sorted[i].index).collect();
            SetRank {
                cc: sorted[c[0]].cc.clone(),
                foreign: c.iter().filter(|&&i| sorted[i].foreign_process).count(),
                link: hw.map(|h| hw::worst_link_rank(h, &gpus)).unwrap_or(0),
                min_free_mb: c.iter().map(|&i| sorted[i].free_vram_mb).min().unwrap_or(0),
                gpus,
            }
        })
        .collect();
    out.sort_by_key(|s| (s.foreign, Reverse(s.min_free_mb), s.link, s.gpus.clone()));
    out
}

pub fn placement_inputs(
    snap: &[GpuLive],
    managed: &HashSet<i32>,
    live: &HashSet<String>,
    denied: &HashSet<u32>,
    reserved_gb: impl Fn(u32) -> f64,
) -> Vec<GpuPlacementInput> {
    // Placement view of the live GPUs: denied ones dropped, free VRAM net of STARTING reservations.
    snap.iter()
        .filter(|g| !denied.contains(&g.index))
        .map(|g| GpuPlacementInput {
            index: g.index,
            cc: g.compute_capability.clone(),
            foreign_process: g.procs.iter().any(|(p, _)| !foreign::is_ours(*p, managed, live)),
            free_vram_mb: (g.free_mb as f64 - reserved_gb(g.index) * 1024.0).max(0.0) as u64,
        })
        .collect()
}

pub fn load_hw(home: &Path) -> Option<HardwareState> {
    let text = std::fs::read_to_string(home.join("hardware_state.json")).ok()?;
    serde_json::from_str(&text).ok()
}

pub fn ensure_hw_profile(home: &Path) -> Result<bool, String> {
    // Write hardware_state.json when it is missing or older than this boot; true when a new profile was written.
    let booted = std::fs::read_to_string("/proc/stat").ok().as_deref().and_then(boot_epoch).unwrap_or(0);
    if profile_since(home, booted) {
        return Ok(false);
    }
    let state = hw::profile(home)?;
    hw::write_state(home, &state)?;
    Ok(true)
}

fn profile_since(home: &Path, booted: u64) -> bool {
    std::fs::metadata(home.join("hardware_state.json"))
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .is_some_and(|d| d.as_secs() >= booted)
}

fn boot_epoch(stat: &str) -> Option<u64> {
    stat.lines().find_map(|l| l.strip_prefix("btime "))?.trim().parse().ok()
}

pub fn format_gpu_list(gpus: &[u32]) -> String {
    gpus.iter()
        .map(|g| g.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn live_rows(con: &Connection) -> Vec<(String, Option<Identity>)> {
    // Instance ids and identities of every row that is not a FAILED or CANCELLED tombstone.
    let Ok(mut stmt) = con.prepare(
        "SELECT instance_id, pid, boot_id, start_ticks FROM loaded_instances
         WHERE status NOT IN ('FAILED', 'CANCELLED')",
    ) else {
        return Vec::new();
    };
    stmt.query_map([], |r| {
        let id: String = r.get(0)?;
        let ident = Identity::from_row(&id, r.get(1)?, r.get(2)?, r.get(3)?);
        Ok((id, ident))
    })
    .map(|rows| rows.filter_map(|r| r.ok()).collect())
    .unwrap_or_default()
}

pub fn live_instance_ids(con: &Connection) -> HashSet<String> {
    live_rows(con).into_iter().map(|(id, _)| id).collect()
}

pub fn managed_pids(con: &Connection) -> HashSet<i32> {
    // Verified leaders of live instances and all their descendants (TP ranks), from one /proc scan.
    let roots: HashSet<i32> = live_rows(con)
        .into_iter()
        .filter_map(|(_, ident)| ident.filter(proc_ident::leader_matches).map(|i| i.pid))
        .collect();
    if roots.is_empty() {
        return roots;
    }
    let mut children: HashMap<i32, Vec<i32>> = HashMap::new();
    if let Ok(rd) = std::fs::read_dir("/proc") {
        for ent in rd.flatten() {
            let Some(pid) = ent.file_name().to_str().and_then(|s| s.parse::<i32>().ok()) else {
                continue;
            };
            if let Some(pp) = proc_ident::ppid(pid) {
                children.entry(pp).or_default().push(pid);
            }
        }
    }
    let mut out = roots.clone();
    let mut stack: Vec<i32> = roots.into_iter().collect();
    while let Some(p) = stack.pop() {
        for &c in children.get(&p).into_iter().flatten() {
            if out.insert(c) {
                stack.push(c);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hw::LinkState;

    fn inp(idx: u32, foreign: bool, free_mb: u64) -> GpuPlacementInput {
        on(idx, "12.0", foreign, free_mb)
    }

    fn on(idx: u32, cc: &str, foreign: bool, free_mb: u64) -> GpuPlacementInput {
        GpuPlacementInput {
            index: idx,
            cc: cc.to_string(),
            foreign_process: foreign,
            free_vram_mb: free_mb,
        }
    }

    fn host_hw() -> HardwareState {
        let link = |a, b, t: &str| LinkState {
            a,
            b,
            link_type: t.to_string(),
            topo: t.to_string(),
        };
        HardwareState {
            profiled_at: "t".to_string(),
            driver: "t".to_string(),
            gpus: Vec::new(),
            links: vec![
                link(0, 1, "PHB"),
                link(0, 2, "NODE"),
                link(0, 3, "NODE"),
                link(1, 2, "NODE"),
                link(1, 3, "NODE"),
                link(2, 3, "NODE"),
            ],
            unavailable: Vec::new(),
            host: None,
            environment: None,
        }
    }

    #[test]
    fn hardware_profile_is_stale_when_missing_or_older_than_boot() {
        assert_eq!(boot_epoch("cpu  1 2 3\nbtime 1729700000\nprocesses 9\n"), Some(1729700000));
        let td = tempfile::tempdir().unwrap();
        assert!(!profile_since(td.path(), 0), "a missing profile is stale");
        let path = td.path().join("hardware_state.json");
        std::fs::write(&path, "{}").unwrap();
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        assert!(profile_since(td.path(), now - 60));
        let old = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1000);
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(old).unwrap();
        assert!(!profile_since(td.path(), now - 60), "a profile from before this boot is stale");
    }

    #[test]
    fn combos_enumerates_all_subsets() {
        assert_eq!(combos(4, 2).len(), 6);
        assert_eq!(combos(3, 3), vec![vec![0, 1, 2]]);
        assert!(combos(2, 3).is_empty());
    }

    #[test]
    fn clean_gpu_beats_foreign_then_free_then_index() {
        let r = rank_sets(&[inp(0, true, 90000), inp(1, false, 1000)], None, 1);
        assert_eq!(r[0].gpus, vec![1]);
        let r = rank_sets(&[inp(0, false, 1000), inp(1, false, 5000), inp(2, false, 5000)], None, 1);
        assert_eq!(r[0].gpus, vec![1]);
    }

    #[test]
    fn foreign_count_outranks_a_better_link() {
        let gpus = [inp(0, true, 7000), inp(1, true, 7000), inp(2, true, 77000), inp(3, false, 97000)];
        let r = rank_sets(&gpus, Some(&host_hw()), 2);
        assert_eq!(r[0].gpus, vec![2, 3], "one foreign GPU with most free VRAM wins");
        assert_eq!(r.last().map(|s| s.foreign), Some(2));
        assert!(r.iter().position(|s| s.gpus == vec![0, 1]).unwrap() > 2);
    }

    #[test]
    fn free_vram_outranks_link_on_this_hosts_shape() {
        let gpus = [inp(0, true, 4300), inp(1, true, 4300), inp(2, true, 75 * 1024), inp(3, true, 90 * 1024)];
        let r = rank_sets(&gpus, Some(&host_hw()), 2);
        assert_eq!(r[0].gpus, vec![2, 3], "most free VRAM beats the PHB pair");
        assert_eq!(r[1].gpus, vec![0, 1], "among equally full sets the better link breaks the tie");
    }

    #[test]
    fn tensor_parallel_sets_never_mix_architectures() {
        let gpus = [on(0, "12.0", false, 90000), on(1, "8.6", false, 24000), on(2, "12.0", false, 90000), on(3, "8.6", false, 24000)];
        let r = rank_sets(&gpus, None, 2);
        let sets: Vec<(Vec<u32>, &str)> = r.iter().map(|s| (s.gpus.clone(), s.cc.as_str())).collect();
        assert_eq!(sets, vec![(vec![0, 2], "12.0"), (vec![1, 3], "8.6")]);
        assert!(rank_sets(&gpus[..2], None, 2).is_empty(), "one GPU of each kind has no tp=2 set");
        assert_eq!(rank_sets(&gpus, None, 1).len(), 4);
    }

    #[test]
    fn placement_inputs_drop_denied_and_subtract_ledger() {
        let live = |index, procs| GpuLive {
            index,
            name: "g".to_string(),
            compute_capability: "12.0".to_string(),
            free_mb: 20480,
            procs,
        };
        let snap = vec![live(2, vec![(i32::MAX - 7, 100)]), live(3, Vec::new()), live(4, Vec::new())];
        let denied = HashSet::from([4u32]);
        let got = placement_inputs(&snap, &HashSet::new(), &HashSet::new(), &denied, |g| if g == 3 { 5.0 } else { 0.0 });
        assert_eq!(got, vec![inp(2, true, 20480), inp(3, false, 15360)]);
    }
}
