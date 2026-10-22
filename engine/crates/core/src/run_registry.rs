// Track engine process groups on disk so orphans can be reaped after a crash.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEntry {
    pub instance_id: String,
    pub pgid: i32,
    pub pid: i32,
    pub model_path: String,
}

fn run_dir(home: &Path) -> PathBuf {
    home.join("run")
}

fn entry_path(home: &Path, instance_id: &str) -> PathBuf {
    run_dir(home).join(format!("{instance_id}.json"))
}

pub fn ensure_run_dir(home: &Path) -> Result<(), String> {
    fs::create_dir_all(run_dir(home)).map_err(|e| e.to_string())
}

pub fn register(home: &Path, entry: &RunEntry) -> Result<(), String> {
    ensure_run_dir(home)?;
    let path = entry_path(home, &entry.instance_id);
    let data = serde_json::to_string(entry).map_err(|e| e.to_string())?;
    fs::write(&path, data).map_err(|e| e.to_string())
}

pub fn unregister(home: &Path, instance_id: &str) {
    let path = entry_path(home, instance_id);
    let _ = fs::remove_file(path);
}

pub fn read_entry(home: &Path, instance_id: &str) -> Option<RunEntry> {
    let path = entry_path(home, instance_id);
    let data = fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

pub fn list_entries(home: &Path) -> Vec<RunEntry> {
    let dir = run_dir(home);
    let Ok(rd) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for ent in rd.flatten() {
        let path = ent.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if let Ok(data) = fs::read_to_string(&path) {
            if let Ok(entry) = serde_json::from_str::<RunEntry>(&data) {
                out.push(entry);
            }
        }
    }
    out
}

fn pgid_alive(pgid: i32) -> bool {
    if pgid <= 0 {
        return false;
    }
    unsafe { libc::kill(-pgid, 0) == 0 }
}

fn pid_alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    unsafe { libc::kill(pid, 0) == 0 }
}

fn reap_pid(pid: i32) {
    if pid <= 0 {
        return;
    }
    let mut status = 0;
    unsafe {
        libc::waitpid(pid, &mut status, libc::WNOHANG);
    }
}

fn engine_gone(pid: i32, pgid: i32) -> bool {
    reap_pid(pid);
    !pid_alive(pid) && !pgid_alive(pgid)
}

fn leader_zombie(pid: i32) -> bool {
    let Ok(data) = fs::read_to_string(format!("/proc/{pid}/stat")) else {
        return false;
    };
    let Some(rparen) = data.rfind(')') else {
        return false;
    };
    data[rparen + 2..].trim_start().starts_with('Z')
}

fn wait_engines_gone(engines: &[(i32, i32)], until: Instant, zombie_ok: bool) -> bool {
    loop {
        let pending = engines.iter().any(|&(pid, pgid)| {
            !engine_gone(pid, resolve_pgid(pid, pgid)) && !(zombie_ok && leader_zombie(pid))
        });
        if !pending {
            return true;
        }
        if Instant::now() >= until {
            return false;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}

fn read_cmdline(pid: i32) -> String {
    let path = format!("/proc/{pid}/cmdline");
    let Ok(mut f) = fs::File::open(path) else {
        return String::new();
    };
    let mut buf = Vec::new();
    if f.read_to_end(&mut buf).is_err() {
        return String::new();
    }
    String::from_utf8_lossy(&buf).replace('\0', " ")
}

fn cmdline_matches_goethe_engine(cmdline: &str, model_path: &str) -> bool {
    if !cmdline.contains(model_path) {
        return false;
    }
    cmdline.contains("sglang.launch_server") || cmdline.contains("stub_engine.py")
}

pub const ENGINE_TERM_GRACE: Duration = Duration::from_secs(3);

fn resolve_pgid(pid: i32, pgid: i32) -> i32 {
    if pgid > 0 {
        pgid
    } else {
        unsafe {
            let p = libc::getpgid(pid);
            if p > 0 {
                p
            } else {
                pid
            }
        }
    }
    .max(1)
}

pub fn kill_process_groups(engines: &[(i32, i32)]) {
    if engines.is_empty() {
        return;
    }
    let mut pgids: Vec<i32> = engines
        .iter()
        .map(|&(pid, pgid)| resolve_pgid(pid, pgid))
        .collect();
    pgids.sort_unstable();
    pgids.dedup();
    let mut pids: Vec<i32> = engines.iter().map(|&(pid, _)| pid).collect();
    pids.sort_unstable();
    pids.dedup();
    for &pgid in &pgids {
        unsafe {
            libc::kill(-pgid, libc::SIGTERM);
        }
    }
    wait_engines_gone(engines, Instant::now() + ENGINE_TERM_GRACE, true);
    for &(pid, pgid) in engines {
        let g = resolve_pgid(pid, pgid);
        if !engine_gone(pid, g) {
            unsafe {
                libc::kill(-g, libc::SIGKILL);
            }
        }
    }
    wait_engines_gone(engines, Instant::now() + Duration::from_secs(5), false);
    for &pid in &pids {
        reap_pid(pid);
    }
}

pub fn kill_process_group(pgid: i32, pid: i32) {
    kill_process_groups(&[(pid, pgid)]);
}

pub fn kill_entry(home: &Path, entry: &RunEntry) {
    kill_process_groups(&[(entry.pid, entry.pgid)]);
    unregister(home, &entry.instance_id);
}

pub fn kill_instance(home: &Path, instance_id: &str, pgid: i32, pid: i32) {
    if let Some(entry) = read_entry(home, instance_id) {
        kill_entry(home, &entry);
    } else {
        kill_process_group(pgid, pid);
        unregister(home, instance_id);
    }
}

pub fn reap_orphans_on_start(home: &Path) -> Result<usize, String> {
    ensure_run_dir(home)?;
    let entries = list_entries(home);
    let home_s = home.to_string_lossy();
    let mut n = 0;
    for entry in entries {
        let alive = pgid_alive(entry.pgid) || pid_alive(entry.pid);
        if !alive {
            unregister(home, &entry.instance_id);
            continue;
        }
        let cmd = read_cmdline(entry.pid);
        if !entry.model_path.contains(home_s.as_ref())
            && !cmd.contains(home_s.as_ref())
        {
            unregister(home, &entry.instance_id);
            continue;
        }
        if cmdline_matches_goethe_engine(&cmd, &entry.model_path) {
            kill_entry(home, &entry);
            n += 1;
        } else {
            unregister(home, &entry.instance_id);
        }
    }
    kill_all_goethe_engines_under(home);
    Ok(n)
}

pub fn kill_all_registered(home: &Path) {
    let entries = list_entries(home);
    let engines: Vec<(i32, i32)> = entries
        .iter()
        .map(|e| (e.pid, e.pgid))
        .collect();
    kill_process_groups(&engines);
    for entry in entries {
        unregister(home, &entry.instance_id);
    }
}

pub fn pids_matching_model_path(model_path: &str) -> Vec<i32> {
    let Ok(rd) = fs::read_dir("/proc") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for ent in rd.flatten() {
        let Ok(pid) = ent.file_name().to_string_lossy().parse::<i32>() else {
            continue;
        };
        let cmd = read_cmdline(pid);
        if cmdline_matches_goethe_engine(&cmd, model_path) {
            out.push(pid);
        }
    }
    out
}

pub fn kill_cmdline_engines(model_path: &str) {
    for pid in pids_matching_model_path(model_path) {
        let pgid = unsafe {
            let p = libc::getpgid(pid);
            if p > 0 {
                p
            } else {
                pid
            }
        };
        kill_process_group(pgid, pid);
    }
}

pub fn kill_all_goethe_engines_under(home: &Path) {
    let home_s = home.to_string_lossy();
    let Ok(rd) = fs::read_dir("/proc") else {
        return;
    };
    let mut engines = Vec::new();
    for ent in rd.flatten() {
        let Ok(pid) = ent.file_name().to_string_lossy().parse::<i32>() else {
            continue;
        };
        let cmd = read_cmdline(pid);
        if !cmd.contains(home_s.as_ref()) {
            continue;
        }
        if cmd.contains("sglang.launch_server") || cmd.contains("stub_engine.py") {
            let pgid = resolve_pgid(pid, 0);
            engines.push((pid, pgid));
        }
    }
    kill_process_groups(&engines);
}
