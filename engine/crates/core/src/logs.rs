// Engine and daemon log tail helpers.

use crate::db;
use crate::paths::{goethe_home, sanitize_model_id};
use serde_json::json;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

pub fn append_spawn_index(
    home: &Path,
    instance_id: &str,
    model_id: &str,
    log_path: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(home.join("logs")).map_err(|e| e.to_string())?;
    let index = home.join("logs/index.jsonl");
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&index)
        .map_err(|e| e.to_string())?;
    let line = json!({
        "instance_id": instance_id,
        "model_id": model_id,
        "log": log_path.to_string_lossy(),
        "ts": chrono::Utc::now().to_rfc3339(),
    });
    writeln!(f, "{line}").map_err(|e| e.to_string())
}

pub fn run_logs(
    target: Option<&str>,
    follow: bool,
    lines: usize,
    daemon: bool,
    gateway: bool,
    prune: bool,
) -> Result<(), String> {
    let home = goethe_home();
    if prune {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let n = crate::log_rotation::run_prune_pass(&home, now)?;
        let _ = crate::log_rotation::run_rotation_pass(&home);
        println!("pruned {n} engine log file(s)");
        return Ok(());
    }
    let path = if daemon {
        home.join("logs/goethed.log")
    } else if gateway {
        home.join("logs/nginx.log")
    } else {
        let Some(t) = target else {
            return Err("usage: goethe logs <model|instance_id> or --daemon or --gateway".to_string());
        };
        resolve_log_path(&home, t)?
    };
    tail_file(&path, lines, follow)
}

fn resolve_log_path(home: &Path, target: &str) -> Result<PathBuf, String> {
    if target.starts_with("inst-") {
        let p = home.join("logs").join(format!("{target}.log"));
        if p.is_file() {
            return Ok(p);
        }
    }
    let con = db::open(home).map_err(|e| e.to_string())?;
    if let Ok(inst) = con.query_row(
        "SELECT instance_id FROM loaded_instances WHERE model_id = ?1 AND status = 'RUNNING' LIMIT 1",
        [target],
        |r| r.get::<_, String>(0),
    ) {
        let p = home.join("logs").join(format!("{inst}.log"));
        if p.is_file() {
            return Ok(p);
        }
    }
    if let Some(p) = newest_index_log(home, target) {
        return Ok(p);
    }
    let legacy = home.join("logs").join(format!("{}.log", sanitize_model_id(target)));
    if legacy.is_file() {
        return Ok(legacy);
    }
    let any = newest_log_for_model(home, target);
    any.ok_or_else(|| format!("no log found for {target}"))
}

fn newest_index_log(home: &Path, model_id: &str) -> Option<PathBuf> {
    let index = home.join("logs/index.jsonl");
    let text = std::fs::read_to_string(&index).ok()?;
    let mut best: Option<(String, PathBuf)> = None;
    for line in text.lines().rev() {
        let v: serde_json::Value = serde_json::from_str(line).ok()?;
        if v.get("model_id").and_then(|x| x.as_str()) != Some(model_id) {
            continue;
        }
        let log = v.get("log").and_then(|x| x.as_str())?;
        let ts = v
            .get("ts")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        let path = PathBuf::from(log);
        if path.is_file() {
            best = Some((ts, path));
            break;
        }
    }
    best.map(|(_, p)| p)
}

fn newest_log_for_model(home: &Path, _model_id: &str) -> Option<PathBuf> {
    let dir = home.join("logs");
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(&dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("log") {
            continue;
        }
        let name = path.file_stem()?.to_str()?;
        if !name.starts_with("inst-") {
            continue;
        }
        let meta = entry.metadata().ok()?;
        let mtime = meta.modified().ok()?;
        if best.as_ref().is_none_or(|(t, _)| mtime > *t) {
            best = Some((mtime, path));
        }
    }
    best.map(|(_, p)| p)
}

fn tail_file(path: &Path, lines: usize, follow: bool) -> Result<(), String> {
    if !path.is_file() {
        return Err(format!("log not found: {}", path.display()));
    }
    let data = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let all: Vec<&str> = data.lines().collect();
    let start = all.len().saturating_sub(lines);
    for line in &all[start..] {
        println!("{line}");
    }
    if !follow {
        return Ok(());
    }
    let f = File::open(path).map_err(|e| e.to_string())?;
    let mut reader = io::BufReader::new(f);
    reader.seek(SeekFrom::End(0)).ok();
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                thread::sleep(Duration::from_millis(200));
                continue;
            }
            Ok(_) => {
                print!("{line}");
                io::stdout().flush().ok();
            }
            Err(e) => return Err(e.to_string()),
        }
    }
}
