// Identity of spawned engines, on disk for a restarted goethed and in memory for this one.

use crate::proc_ident::{self, Identity};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard, OnceLock};

fn live() -> MutexGuard<'static, HashMap<String, Identity>> {
    static LIVE: OnceLock<Mutex<HashMap<String, Identity>>> = OnceLock::new();
    LIVE.get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

pub fn tracked() -> Vec<Identity> {
    live().values().cloned().collect()
}

fn run_dir(home: &Path) -> PathBuf {
    home.join("run")
}

fn entry_path(home: &Path, instance_id: &str) -> PathBuf {
    run_dir(home).join(format!("{instance_id}.json"))
}

pub fn register(home: &Path, id: &Identity) -> Result<(), String> {
    // Write the entry atomically (a crash never leaves half an identity), then track the engine in memory.
    fs::create_dir_all(run_dir(home)).map_err(|e| e.to_string())?;
    let path = entry_path(home, &id.instance_id);
    let tmp = path.with_extension("tmp");
    let data = serde_json::to_vec(id).map_err(|e| e.to_string())?;
    fs::write(&tmp, data).map_err(|e| e.to_string())?;
    fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
    live().insert(id.instance_id.clone(), id.clone());
    Ok(())
}

pub fn unregister(home: &Path, instance_id: &str) {
    live().remove(instance_id);
    let _ = fs::remove_file(entry_path(home, instance_id));
}

pub fn list_entries(home: &Path) -> Vec<Identity> {
    // Parse every run entry; files without a readable identity can never be verified and are dropped.
    let Ok(rd) = fs::read_dir(run_dir(home)) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for ent in rd.flatten() {
        let path = ent.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        match fs::read(&path).ok().and_then(|d| serde_json::from_slice::<Identity>(&d).ok()) {
            Some(id) => out.push(id),
            None => {
                let _ = fs::remove_file(&path);
            }
        }
    }
    out
}

pub fn reap_orphans(home: &Path, keep: &HashSet<String>) {
    // Stop verified engines no adopted instance owns and drop their entries; unverified pids are never signaled.
    let orphans: Vec<Identity> = list_entries(home)
        .into_iter()
        .filter(|e| !keep.contains(&e.instance_id))
        .collect();
    if orphans.is_empty() {
        return;
    }
    proc_ident::terminate(&orphans);
    for e in &orphans {
        unregister(home, &e.instance_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ident(id: &str, pid: i32, ticks: u64) -> Identity {
        Identity {
            instance_id: id.to_string(),
            pid,
            boot_id: proc_ident::boot_id().to_string(),
            start_ticks: ticks,
        }
    }

    #[test]
    fn register_roundtrip_and_old_entries_dropped() {
        let td = tempfile::tempdir().unwrap();
        let a = ident("inst-a", 4242, 7);
        register(td.path(), &a).unwrap();
        let old = run_dir(td.path()).join("old.json");
        fs::write(&old, br#"{"instance_id":"old","pgid":1,"pid":1,"model_path":"/x"}"#).unwrap();
        assert_eq!(list_entries(td.path()), vec![a.clone()]);
        assert!(!old.exists());
        assert!(tracked().contains(&a));
        unregister(td.path(), "inst-a");
        assert!(list_entries(td.path()).is_empty());
        assert!(!tracked().contains(&a));
    }

    #[test]
    fn reap_never_signals_a_stale_identity() {
        let td = tempfile::tempdir().unwrap();
        let mut child = std::process::Command::new("sleep").arg("30").spawn().unwrap();
        let pid = child.id() as i32;
        let ticks = proc_ident::start_ticks(pid).unwrap();
        register(td.path(), &ident("inst-stale", pid, ticks + 1)).unwrap();
        reap_orphans(td.path(), &HashSet::new());
        assert!(child.try_wait().unwrap().is_none(), "stale run entry must not kill the pid");
        assert!(list_entries(td.path()).is_empty());
        child.kill().unwrap();
        child.wait().unwrap();
    }
}
