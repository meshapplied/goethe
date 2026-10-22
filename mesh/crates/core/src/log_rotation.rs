// Log size caps and engine log retention.

use std::collections::HashSet;
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const ENGINE_LOG_MAX_BYTES: u64 = 256 * 1024 * 1024;
pub const ENGINE_LOG_KEEP_BYTES: u64 = 64 * 1024 * 1024;
pub const DAEMON_LOG_MAX_BYTES: u64 = 100 * 1024 * 1024;
pub const ENGINE_LOG_MAX_AGE_DAYS: i64 = 14;
pub const ENGINE_LOG_KEEP_NEWEST: usize = 30;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrunePlan {
    pub delete_paths: Vec<PathBuf>,
}

pub fn engine_logs_to_prune(
    all_engine_logs: &[(PathBuf, i64, bool)],
    now_secs: i64,
    max_age_days: i64,
    keep_newest: usize,
) -> PrunePlan {
    let mut unloaded: Vec<(PathBuf, i64)> = all_engine_logs
        .iter()
        .filter(|(_, _, is_loaded)| !*is_loaded)
        .map(|(p, m, _)| (p.clone(), *m))
        .collect();
    unloaded.sort_by_key(|a| std::cmp::Reverse(a.1));
    let mut delete = HashSet::new();
    let cutoff = now_secs - max_age_days * 86400;
    for (path, mtime) in &unloaded {
        if *mtime < cutoff {
            delete.insert(path.clone());
        }
    }
    for (path, _) in unloaded.iter().skip(keep_newest) {
        delete.insert(path.clone());
    }
    PrunePlan {
        delete_paths: delete.into_iter().collect(),
    }
}

pub fn rotate_file_copytruncate(path: &Path, max_bytes: u64, keep_tail_bytes: u64) -> Result<bool, String> {
    let meta = fs::metadata(path).map_err(|e| e.to_string())?;
    if meta.len() <= max_bytes {
        return Ok(false);
    }
    let mut f = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    let len = f.metadata().map_err(|e| e.to_string())?.len();
    let start = len.saturating_sub(keep_tail_bytes);
    f.seek(SeekFrom::Start(start)).map_err(|e| e.to_string())?;
    let mut tail = Vec::new();
    f.read_to_end(&mut tail).map_err(|e| e.to_string())?;
    let backup = PathBuf::from(format!("{}.1", path.display()));
    if backup.is_file() {
        let _ = fs::remove_file(&backup);
    }
    fs::write(&backup, &tail).map_err(|e| e.to_string())?;
    f.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    f.set_len(0).map_err(|e| e.to_string())?;
    f.write_all(&tail).map_err(|e| e.to_string())?;
    Ok(true)
}

fn mtime_secs(path: &Path) -> i64 {
    fs::metadata(path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

pub fn list_engine_logs(logs_dir: &Path, loaded_instance_ids: &HashSet<String>) -> Vec<(PathBuf, i64, bool)> {
    let mut out = Vec::new();
    let Ok(rd) = fs::read_dir(logs_dir) else {
        return out;
    };
    for ent in rd.flatten() {
        let path = ent.path();
        if path.extension().and_then(|e| e.to_str()) != Some("log") {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if !stem.starts_with("inst-") {
            continue;
        }
        let loaded = loaded_instance_ids.contains(stem);
        let mtime = mtime_secs(&path);
        out.push((path, mtime, loaded));
    }
    out
}

pub fn run_prune_pass(home: &Path, now_secs: i64) -> Result<usize, String> {
    let logs = home.join("logs");
    if !logs.is_dir() {
        return Ok(0);
    }
    let con = crate::db::open(home).map_err(|e| e.to_string())?;
    let mut loaded = HashSet::new();
    if let Ok(mut stmt) = con.prepare("SELECT instance_id FROM loaded_instances") {
        let rows = stmt.query_map([], |r| r.get::<_, String>(0));
        if let Ok(rows) = rows {
            for row in rows.flatten() {
                loaded.insert(row);
            }
        }
    }
    let entries = list_engine_logs(&logs, &loaded);
    let plan = engine_logs_to_prune(
        &entries,
        now_secs,
        ENGINE_LOG_MAX_AGE_DAYS,
        ENGINE_LOG_KEEP_NEWEST,
    );
    let mut n = 0;
    for p in plan.delete_paths {
        if fs::remove_file(&p).is_ok() {
            n += 1;
        }
    }
    Ok(n)
}

pub fn run_rotation_pass(home: &Path) -> Result<(), String> {
    let logs = home.join("logs");
    if !logs.is_dir() {
        return Ok(());
    }
    let con = crate::db::open(home).map_err(|e| e.to_string())?;
    let mut loaded = HashSet::new();
    if let Ok(mut stmt) = con.prepare("SELECT instance_id FROM loaded_instances") {
        let rows = stmt.query_map([], |r| r.get::<_, String>(0));
        if let Ok(rows) = rows {
            for row in rows.flatten() {
                loaded.insert(row);
            }
        }
    }
    for (path, _, is_loaded) in list_engine_logs(&logs, &loaded) {
        if is_loaded {
            let _ = rotate_file_copytruncate(&path, ENGINE_LOG_MAX_BYTES, ENGINE_LOG_KEEP_BYTES);
        }
    }
    for name in ["meshd.log", "gateway.log", "nginx.log"] {
        let p = logs.join(name);
        if p.is_file() {
            let _ = rotate_file_copytruncate(&p, DAEMON_LOG_MAX_BYTES, DAEMON_LOG_MAX_BYTES / 4);
        }
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_secs() as i64;
    let _ = run_prune_pass(home, now);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;
    use std::io::Write;

    #[test]
    fn prune_keeps_newest_unloaded() {
        let logs: Vec<(PathBuf, i64, bool)> = (0..5)
            .map(|i| (PathBuf::from(format!("/l/inst-{i}.log")), i, false))
            .collect();
        let plan = engine_logs_to_prune(&logs, 1_000_000, 14, 2);
        assert_eq!(plan.delete_paths.len(), 3);
    }

    #[test]
    fn rotate_creates_backup_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("meshd.log");
        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)
            .unwrap();
        let chunk = vec![b'x'; 1024];
        for _ in 0..300 {
            f.write_all(&chunk).unwrap();
        }
        drop(f);
        let max = 200 * 1024;
        let keep = 50 * 1024;
        assert!(rotate_file_copytruncate(&path, max, keep).unwrap());
        assert!(dir.path().join("meshd.log.1").is_file());
        assert!(fs::metadata(&path).unwrap().len() <= keep + 1024);
    }
}
