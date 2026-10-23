// Live NVML view of GPUs (free VRAM, processes) and the notice for processes Goethe did not start.

use crate::hw;
use crate::proc_ident;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::{Device, Nvml};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::os::unix::fs::MetadataExt;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuLive {
    pub index: u32,
    pub name: String,
    pub cc_major: u32,
    pub free_mb: u64,
    pub procs: Vec<(i32, u64)>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Foreign {
    pub gpu: u32,
    pub pid: i32,
    pub name: String,
    pub user: String,
    pub used_mb: u64,
    #[serde(skip)]
    pub same_user: bool,
}

fn device_procs(dev: &Device) -> Vec<(i32, u64)> {
    // Compute and graphics processes on one device, merged by pid with the larger VRAM figure.
    let mut merged: HashMap<i32, u64> = HashMap::new();
    for procs in [dev.running_compute_processes(), dev.running_graphics_processes()]
        .into_iter()
        .flatten()
    {
        for p in procs {
            let mb = match p.used_gpu_memory {
                UsedGpuMemory::Used(bytes) => bytes / (1024 * 1024),
                UsedGpuMemory::Unavailable => 0,
            };
            merged.entry(p.pid as i32).and_modify(|v| *v = (*v).max(mb)).or_insert(mb);
        }
    }
    let mut out: Vec<(i32, u64)> = merged.into_iter().collect();
    out.sort_unstable();
    out
}

pub fn snapshot() -> Result<Vec<GpuLive>, String> {
    // One pass over every GPU: name, compute capability, live free VRAM (unified memory aware) and its processes.
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let count = nvml.device_count().map_err(|e| format!("device count: {e}"))?;
    let mut out = Vec::new();
    for index in 0..count {
        let dev = nvml
            .device_by_index(index)
            .map_err(|e| format!("device {index}: {e}"))?;
        out.push(GpuLive {
            index,
            name: dev.name().unwrap_or_default(),
            cc_major: dev.cuda_compute_capability().map_or(0, |c| c.major.max(0) as u32),
            free_mb: hw::gpu_free_mb_live(index)?,
            procs: device_procs(&dev),
        });
    }
    Ok(out)
}

pub fn is_ours(pid: i32, managed: &HashSet<i32>, live: &HashSet<String>) -> bool {
    managed.contains(&pid) || proc_ident::instance_tag(pid).is_some_and(|t| live.contains(&t))
}

fn user_name(uid: u32) -> String {
    // Resolve a uid with getpwuid_r; fall back to the number.
    let mut pwd: libc::passwd = unsafe { std::mem::zeroed() };
    let mut buf = vec![0 as libc::c_char; 4096];
    let mut res: *mut libc::passwd = std::ptr::null_mut();
    let rc = unsafe { libc::getpwuid_r(uid, &mut pwd, buf.as_mut_ptr(), buf.len(), &mut res) };
    if rc != 0 || res.is_null() || pwd.pw_name.is_null() {
        return uid.to_string();
    }
    unsafe { std::ffi::CStr::from_ptr(pwd.pw_name) }
        .to_string_lossy()
        .into_owned()
}

pub fn process_owner(pid: i32) -> (String, String, Option<u32>) {
    // Command name, owning user name and uid of a process from /proc ("unknown" when it is gone or hidden).
    let argv0 = std::fs::read(format!("/proc/{pid}/cmdline")).ok().and_then(|raw| {
        let first = raw.split(|b| *b == 0).next().map(|a| String::from_utf8_lossy(a).into_owned())?;
        let base = first.split_whitespace().next()?.rsplit('/').next()?.to_string();
        (!base.is_empty()).then(|| base.chars().take(32).collect::<String>())
    });
    let name = argv0
        .or_else(|| std::fs::read_to_string(format!("/proc/{pid}/comm")).ok().map(|s| s.trim().to_string()))
        .unwrap_or_else(|| "unknown".to_string());
    let uid = std::fs::metadata(format!("/proc/{pid}")).map(|m| m.uid()).ok();
    (name, uid.map(user_name).unwrap_or_else(|| "unknown".to_string()), uid)
}

fn describe(gpu: u32, pid: i32, used_mb: u64) -> Foreign {
    // Name, owner and VRAM of one process from /proc and NVML.
    let (name, user, uid) = process_owner(pid);
    Foreign { gpu, pid, name, user, used_mb, same_user: uid == Some(unsafe { libc::geteuid() }) }
}

pub fn foreign_on(
    snap: &[GpuLive],
    gpus: &[u32],
    managed: &HashSet<i32>,
    live: &HashSet<String>,
) -> Vec<Foreign> {
    // Processes on these GPUs that no live Goethe instance owns.
    snap.iter()
        .filter(|g| gpus.contains(&g.index))
        .flat_map(|g| {
            g.procs
                .iter()
                .filter(|(pid, _)| !is_ours(*pid, managed, live))
                .map(move |&(pid, mb)| describe(g.index, pid, mb))
        })
        .collect()
}

pub fn holder_lines(foreign: &[Foreign]) -> Vec<String> {
    // Who holds VRAM and the command the user may run; Goethe never runs it.
    let mut out = Vec::new();
    for f in foreign {
        out.push(format!(
            "GPU {} has unmanaged process PID {} ({}, user {}) using {:.1} GB VRAM",
            f.gpu,
            f.pid,
            f.name,
            f.user,
            f.used_mb as f64 / 1024.0
        ));
        let kill = if f.same_user { "kill" } else { "sudo kill" };
        out.push(format!(
            "Goethe never stops processes it did not start. To stop it yourself: {kill} {}",
            f.pid
        ));
    }
    out
}

pub fn notice(foreign: &[Foreign], free_gb: &HashMap<u32, f64>, need_gb: f64) -> String {
    // Code 3 text for a load that would share GPUs with processes Goethe did not start.
    let mut lines = holder_lines(foreign);
    let mut gpus: Vec<u32> = foreign.iter().map(|f| f.gpu).collect();
    gpus.dedup();
    for g in gpus {
        lines.push(format!(
            "Free VRAM on GPU {g}: {:.1} GB; this load needs {need_gb:.1} GB.",
            free_gb.get(&g).copied().unwrap_or(0.0)
        ));
    }
    let them = if foreign.len() == 1 { "it" } else { "them" };
    lines.push(format!("Re-run with --yes to load alongside {them}."));
    lines.join("\n")
}

pub fn wait_released(gpus: &[u32], pids: &[i32], timeout: Duration) -> bool {
    // Poll NVML until none of these pids is listed on the GPUs, so the VRAM is really back.
    if gpus.is_empty() || pids.is_empty() {
        return true;
    }
    let Ok(nvml) = Nvml::init() else {
        return false;
    };
    let deadline = Instant::now() + timeout;
    loop {
        let held = gpus.iter().any(|&g| {
            nvml.device_by_index(g)
                .map(|d| device_procs(&d).iter().any(|(p, _)| pids.contains(p)))
                .unwrap_or(false)
        });
        if !held {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alien(gpu: u32, pid: i32, mb: u64, same_user: bool) -> Foreign {
        Foreign {
            gpu,
            pid,
            name: "python".to_string(),
            user: "admin".to_string(),
            used_mb: mb,
            same_user,
        }
    }

    #[test]
    fn notice_matches_contract_lines() {
        let f = vec![alien(2, 3127, 10504, true), alien(2, 4000, 512, false)];
        let free = HashMap::from([(2u32, 75.31)]);
        let text = notice(&f, &free, 12.0);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "GPU 2 has unmanaged process PID 3127 (python, user admin) using 10.3 GB VRAM");
        assert_eq!(lines[1], "Goethe never stops processes it did not start. To stop it yourself: kill 3127");
        assert!(lines[3].ends_with("To stop it yourself: sudo kill 4000"));
        assert_eq!(lines[4], "Free VRAM on GPU 2: 75.3 GB; this load needs 12.0 GB.");
        assert_eq!(lines[5], "Re-run with --yes to load alongside them.");
        assert!(notice(&f[..1], &free, 12.0).ends_with("Re-run with --yes to load alongside it."));
        let json = serde_json::to_value(&f[0]).unwrap();
        assert_eq!(json, serde_json::json!({"gpu":2,"pid":3127,"name":"python","user":"admin","used_mb":10504}));
    }

    fn wait_exec(pid: u32) {
        // Until a spawned child has exec'd, /proc shows the parent's environment; wait for it to become sleep.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while std::time::Instant::now() < deadline {
            if std::fs::read_to_string(format!("/proc/{pid}/comm")).is_ok_and(|c| c.trim() == "sleep") {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("child {pid} did not exec sleep within 5 s");
    }

    #[test]
    fn tagged_child_of_a_live_instance_is_ours_and_a_stranger_is_not() {
        let mut tagged = std::process::Command::new("sleep")
            .arg("30")
            .env(proc_ident::INSTANCE_ENV, "inst-live")
            .spawn()
            .unwrap();
        let mut plain = std::process::Command::new("sleep")
            .arg("30")
            .env_remove(proc_ident::INSTANCE_ENV)
            .spawn()
            .unwrap();
        for c in [&tagged, &plain] {
            wait_exec(c.id());
        }
        let live = HashSet::from(["inst-live".to_string()]);
        let none = HashSet::new();
        let pid = tagged.id();
        assert!(
            is_ours(pid as i32, &none, &live),
            "tagged child {pid}: comm={:?} tag={:?} environ_readable={} alive={}",
            std::fs::read_to_string(format!("/proc/{pid}/comm")).ok(),
            proc_ident::instance_tag(pid as i32),
            std::fs::read(format!("/proc/{pid}/environ")).is_ok(),
            std::path::Path::new(&format!("/proc/{pid}")).exists()
        );
        assert!(!is_ours(tagged.id() as i32, &none, &HashSet::from(["inst-dead".to_string()])));
        assert!(!is_ours(plain.id() as i32, &none, &live));
        assert!(is_ours(plain.id() as i32, &HashSet::from([plain.id() as i32]), &live));
        let me = describe(3, plain.id() as i32, 512);
        assert_eq!(me.name, "sleep");
        assert!(me.same_user);
        for c in [&mut tagged, &mut plain] {
            c.kill().unwrap();
            c.wait().unwrap();
        }
    }
}
