// Engine and daemon log tail helpers.

use crate::db;
use crate::paths::{goethe_home, sanitize_model_id};
use serde_json::json;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

const TAIL_CHUNK: u64 = 64 * 1024;
const FOLLOW_READ_CAP: u64 = 1 << 20;

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
    // Prune, or tail a model/instance log, the daemon log, or the gateway's goethe and nginx logs.
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
    let paths = if daemon {
        vec![home.join("logs/goethed.log")]
    } else if gateway {
        let ngx = crate::nginx::nginx_prefix(&home).join("logs");
        vec![home.join("logs/gateway.log"), ngx.join("error.log"), ngx.join("access.log")]
    } else {
        let Some(t) = target else {
            return Err("usage: goethe logs <model|instance_id> or --daemon or --gateway".to_string());
        };
        vec![resolve_log_path(&home, t)?]
    };
    tail_files(&paths, lines, follow)
}

fn resolve_log_path(home: &Path, target: &str) -> Result<PathBuf, String> {
    // inst-<id> directly, else the model's newest instance log, else the legacy per-model file.
    if target.starts_with("inst-") {
        let p = home.join("logs").join(format!("{target}.log"));
        if p.is_file() {
            return Ok(p);
        }
    }
    if let Some(p) = newest_log_for_model(home, target) {
        return Ok(p);
    }
    let legacy = home.join("logs").join(format!("{}.log", sanitize_model_id(target)));
    if legacy.is_file() {
        return Ok(legacy);
    }
    Err(format!("no log found for {target}"))
}

fn newest_log_for_model(home: &Path, model_id: &str) -> Option<PathBuf> {
    // The RUNNING instance's log, else the newest existing log among this model's DB and index.jsonl instances.
    let logs = home.join("logs");
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(con) = db::open(home) {
        if let Ok(inst) = con.query_row(
            "SELECT instance_id FROM loaded_instances WHERE model_id = ?1 AND status = 'RUNNING' LIMIT 1",
            [model_id],
            |r| r.get::<_, String>(0),
        ) {
            let p = logs.join(format!("{inst}.log"));
            if p.is_file() {
                return Some(p);
            }
        }
        if let Ok(mut stmt) = con.prepare("SELECT instance_id FROM loaded_instances WHERE model_id = ?1") {
            if let Ok(rows) = stmt.query_map([model_id], |r| r.get::<_, String>(0)) {
                candidates.extend(rows.flatten().map(|id| logs.join(format!("{id}.log"))));
            }
        }
    }
    let index = fs::read(logs.join("index.jsonl")).unwrap_or_default();
    for line in String::from_utf8_lossy(&index).lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v.get("model_id").and_then(|x| x.as_str()) != Some(model_id) {
            continue;
        }
        if let Some(log) = v.get("log").and_then(|x| x.as_str()) {
            candidates.push(PathBuf::from(log));
        }
    }
    candidates
        .into_iter()
        .filter_map(|p| Some((fs::metadata(&p).ok().filter(|m| m.is_file())?.modified().ok()?, p)))
        .max_by_key(|(mtime, _)| *mtime)
        .map(|(_, p)| p)
}

fn last_lines(path: &Path, n: usize) -> io::Result<(Vec<u8>, u64, u64)> {
    // The last n lines read backwards in chunks (never the whole file), plus the end offset and inode.
    let mut f = File::open(path)?;
    let meta = f.metadata()?;
    let len = meta.len();
    let mut start = len;
    let mut buf: Vec<u8> = Vec::new();
    loop {
        let body = buf.strip_suffix(b"\n").unwrap_or(&buf);
        if start == 0 || body.iter().filter(|&&b| b == b'\n').count() >= n {
            break;
        }
        let step = TAIL_CHUNK.min(start);
        start -= step;
        f.seek(SeekFrom::Start(start))?;
        let mut chunk = vec![0u8; step as usize];
        f.read_exact(&mut chunk)?;
        chunk.extend_from_slice(&buf);
        buf = chunk;
    }
    let body_len = buf.strip_suffix(b"\n").unwrap_or(&buf).len();
    let cut = if n == 0 {
        buf.len()
    } else {
        buf[..body_len]
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, b)| **b == b'\n')
            .nth(n - 1)
            .map_or(0, |(i, _)| i + 1)
    };
    Ok((buf[cut..].to_vec(), len, meta.ino()))
}

struct Follower {
    path: PathBuf,
    pos: u64,
    ino: u64,
}

impl Follower {
    fn poll(&mut self) -> Vec<u8> {
        // New bytes since the last poll; a replaced file is reread from 0, a truncated one resumes at its end.
        let Ok(meta) = fs::metadata(&self.path) else {
            return Vec::new();
        };
        if meta.ino() != self.ino {
            self.ino = meta.ino();
            self.pos = 0;
        } else if meta.len() < self.pos {
            eprintln!("goethe logs: {} truncated", self.path.display());
            self.pos = meta.len();
        }
        if meta.len() == self.pos {
            return Vec::new();
        }
        let mut buf = Vec::new();
        if let Ok(mut file) = File::open(&self.path) {
            if file.seek(SeekFrom::Start(self.pos)).is_ok() {
                let n = file.take(FOLLOW_READ_CAP).read_to_end(&mut buf).unwrap_or(0);
                self.pos += n as u64;
            }
        }
        buf
    }
}

fn tail_files(paths: &[PathBuf], lines: usize, follow: bool) -> Result<(), String> {
    // Print the last lines of each existing file (headers when several), then optionally follow them all.
    let existing: Vec<&PathBuf> = paths.iter().filter(|p| p.is_file()).collect();
    if existing.is_empty() {
        let names: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
        return Err(format!("log not found: {}", names.join(", ")));
    }
    let many = existing.len() > 1;
    let mut out = io::stdout().lock();
    let mut followers = Vec::new();
    for p in &existing {
        if many {
            let _ = writeln!(out, "==> {} <==", p.display());
        }
        let (text, end, ino) = last_lines(p, lines).map_err(|e| format!("{}: {e}", p.display()))?;
        out.write_all(&text).map_err(|e| e.to_string())?;
        followers.push(Follower {
            path: p.to_path_buf(),
            pos: end,
            ino,
        });
    }
    out.flush().ok();
    if !follow {
        return Ok(());
    }
    let mut last_shown: Option<usize> = None;
    loop {
        for (i, f) in followers.iter_mut().enumerate() {
            let chunk = f.poll();
            if chunk.is_empty() {
                continue;
            }
            if many && last_shown != Some(i) {
                let _ = writeln!(out, "\n==> {} <==", f.path.display());
            }
            last_shown = Some(i);
            out.write_all(&chunk).map_err(|e| e.to_string())?;
            out.flush().ok();
        }
        thread::sleep(Duration::from_millis(200));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_reads_last_lines_and_tolerates_bad_utf8() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.log");
        let mut data = Vec::new();
        for i in 0..20_000 {
            data.extend_from_slice(format!("line{i}\n").as_bytes());
        }
        data.extend_from_slice(b"bad \xff\xfe bytes\n");
        fs::write(&p, &data).unwrap();
        let (text, end, _) = last_lines(&p, 2).unwrap();
        assert_eq!(text, b"line19999\nbad \xff\xfe bytes\n");
        assert_eq!(end, data.len() as u64);
        assert_eq!(last_lines(&p, 0).unwrap().0, b"");
        fs::write(&p, b"a\nb").unwrap();
        assert_eq!(last_lines(&p, 5).unwrap().0, b"a\nb");
        assert_eq!(last_lines(&p, 1).unwrap().0, b"b");
    }

    #[test]
    fn follow_notices_truncation_and_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("f.log");
        fs::write(&p, b"one\n").unwrap();
        let (_, end, ino) = last_lines(&p, 10).unwrap();
        let mut f = Follower { path: p.clone(), pos: end, ino };
        OpenOptions::new().append(true).open(&p).unwrap().write_all(b"two\n").unwrap();
        assert_eq!(f.poll(), b"two\n");
        fs::write(&p, b"x\n").unwrap();
        assert_eq!(f.poll(), b"", "a truncated file resumes at its new end");
        OpenOptions::new().append(true).open(&p).unwrap().write_all(b"three\n").unwrap();
        assert_eq!(f.poll(), b"three\n");
        fs::rename(&p, dir.path().join("f.log.1")).unwrap();
        fs::write(&p, b"fresh\n").unwrap();
        assert_eq!(f.poll(), b"fresh\n", "a rotated file is reread from the start");
    }

    #[test]
    fn model_log_lookup_filters_by_model_and_skips_bad_index_lines() {
        let dir = tempfile::tempdir().unwrap();
        let logs = dir.path().join("logs");
        fs::create_dir_all(&logs).unwrap();
        fs::write(logs.join("inst-a.log"), b"a\n").unwrap();
        fs::write(logs.join("inst-b.log"), b"b\n").unwrap();
        let index = format!(
            "{}\nnot json\n{}\n",
            json!({"instance_id": "inst-a", "model_id": "m/a", "log": logs.join("inst-a.log")}),
            json!({"instance_id": "inst-b", "model_id": "m/b", "log": logs.join("inst-b.log")}),
        );
        fs::write(logs.join("index.jsonl"), index).unwrap();
        assert_eq!(newest_log_for_model(dir.path(), "m/a"), Some(logs.join("inst-a.log")));
        assert_eq!(newest_log_for_model(dir.path(), "m/b"), Some(logs.join("inst-b.log")));
        assert_eq!(newest_log_for_model(dir.path(), "m/c"), None, "other models' logs must not match");
    }
}
