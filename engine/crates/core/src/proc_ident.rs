// Engine process identity (boot id, start ticks, instance env tag) and the one gate every signal passes.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

pub const INSTANCE_ENV: &str = "GOETHE_INSTANCE_ID";
pub const TERM_GRACE: Duration = Duration::from_secs(3);
const KILL_WAIT: Duration = Duration::from_secs(2);
const POLL: Duration = Duration::from_millis(20);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identity {
    pub instance_id: String,
    pub pid: i32,
    pub boot_id: String,
    pub start_ticks: u64,
}

impl Identity {
    pub fn from_row(
        instance_id: &str,
        pid: i64,
        boot_id: Option<String>,
        start_ticks: Option<i64>,
    ) -> Option<Identity> {
        // Identity stored with an instance row; None for tombstones and rows without identity columns.
        let pid = i32::try_from(pid).ok().filter(|p| *p > 1)?;
        Some(Identity {
            instance_id: instance_id.to_string(),
            pid,
            boot_id: boot_id.filter(|b| !b.is_empty())?,
            start_ticks: u64::try_from(start_ticks?).ok()?,
        })
    }
}

struct Stat {
    state: char,
    ppid: i32,
    start_ticks: u64,
}

pub fn boot_id() -> &'static str {
    static BOOT: OnceLock<String> = OnceLock::new();
    BOOT.get_or_init(|| {
        std::fs::read_to_string("/proc/sys/kernel/random/boot_id")
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    })
}

fn read_stat(pid: i32) -> Option<Stat> {
    // Parse state, ppid and start time (fields 3, 4 and 22) of /proc/<pid>/stat.
    if pid <= 0 {
        return None;
    }
    let text = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let rest = text.get(text.rfind(')')? + 2..)?;
    let fields: Vec<&str> = rest.split_whitespace().collect();
    Some(Stat {
        state: fields.first()?.chars().next()?,
        ppid: fields.get(1)?.parse().ok()?,
        start_ticks: fields.get(19)?.parse().ok()?,
    })
}

pub fn start_ticks(pid: i32) -> Option<u64> {
    read_stat(pid).map(|s| s.start_ticks)
}

pub fn ppid(pid: i32) -> Option<i32> {
    read_stat(pid).map(|s| s.ppid)
}

pub fn capture(instance_id: &str, pid: i32) -> Option<Identity> {
    // Record the identity of a just-spawned leader; None when /proc cannot vouch for it.
    let start_ticks = start_ticks(pid)?;
    if boot_id().is_empty() {
        return None;
    }
    Some(Identity {
        instance_id: instance_id.to_string(),
        pid,
        boot_id: boot_id().to_string(),
        start_ticks,
    })
}

pub fn leader_matches(id: &Identity) -> bool {
    // The pid still belongs to the process we spawned: same boot and same start time.
    id.pid > 1
        && !id.boot_id.is_empty()
        && id.boot_id == boot_id()
        && start_ticks(id.pid) == Some(id.start_ticks)
}

pub fn instance_tag(pid: i32) -> Option<String> {
    // Value of GOETHE_INSTANCE_ID in /proc/<pid>/environ, when the process carries one.
    let data = std::fs::read(format!("/proc/{pid}/environ")).ok()?;
    let prefix = format!("{INSTANCE_ENV}=");
    data.split(|b| *b == 0)
        .find_map(|kv| kv.strip_prefix(prefix.as_bytes()))
        .map(|v| String::from_utf8_lossy(v).into_owned())
}

pub fn safe_target(pid: i32) -> bool {
    pid > 1 && pid != std::process::id() as i32 && pid != unsafe { libc::getpgrp() }
}

fn pidfd_open(pid: i32) -> Result<OwnedFd, i32> {
    // Pin the process with a pidfd so a recycled pid can never receive the signal.
    let fd = unsafe { libc::syscall(libc::SYS_pidfd_open, pid, 0) };
    if fd < 0 {
        return Err(std::io::Error::last_os_error().raw_os_error().unwrap_or(0));
    }
    Ok(unsafe { OwnedFd::from_raw_fd(fd as i32) })
}

pub fn signal(id: &Identity, pid: i32, sig: i32) -> bool {
    // The only path that signals engines: pid must be the verified leader or carry this instance tag.
    if !safe_target(pid) {
        return false;
    }
    let pinned = match pidfd_open(pid) {
        Ok(fd) => Some(fd),
        Err(libc::ESRCH) => return false,
        Err(_) => None,
    };
    let ours = (pid == id.pid && leader_matches(id))
        || instance_tag(pid).as_deref() == Some(id.instance_id.as_str());
    if !ours {
        return false;
    }
    match pinned {
        Some(fd) => {
            use std::os::fd::AsRawFd;
            let rc = unsafe {
                libc::syscall(
                    libc::SYS_pidfd_send_signal,
                    fd.as_raw_fd(),
                    sig,
                    std::ptr::null::<libc::siginfo_t>(),
                    0,
                )
            };
            rc == 0
        }
        None => unsafe { libc::kill(pid, sig) == 0 },
    }
}

fn running(pid: i32, ticks: u64) -> bool {
    read_stat(pid).is_some_and(|s| s.start_ticks == ticks && s.state != 'Z' && s.state != 'X')
}

fn reap_leader(id: &Identity) {
    // Collect our own zombie leader so it stops counting as alive; only ever waits on the verified pid.
    if read_stat(id.pid).is_some_and(|s| s.state == 'Z') && leader_matches(id) {
        let mut status = 0;
        unsafe {
            libc::waitpid(id.pid, &mut status, libc::WNOHANG);
        }
    }
}

pub fn leader_alive(id: &Identity) -> bool {
    reap_leader(id);
    running(id.pid, id.start_ticks) && leader_matches(id)
}

pub fn tagged(instance_ids: &HashSet<String>) -> HashMap<String, Vec<(i32, u64)>> {
    // One /proc pass collecting every process tagged with one of these instance ids.
    let mut out: HashMap<String, Vec<(i32, u64)>> = HashMap::new();
    let Ok(rd) = std::fs::read_dir("/proc") else {
        return out;
    };
    for ent in rd.flatten() {
        let Some(pid) = ent.file_name().to_str().and_then(|s| s.parse::<i32>().ok()) else {
            continue;
        };
        let Some(tag) = instance_tag(pid).filter(|t| instance_ids.contains(t)) else {
            continue;
        };
        if let Some(ticks) = start_ticks(pid) {
            out.entry(tag).or_default().push((pid, ticks));
        }
    }
    out
}

fn members(ids: &[Identity]) -> Vec<(usize, i32, u64)> {
    // Verified leaders plus tagged processes of each instance, keyed by index into ids.
    let wanted: HashSet<String> = ids.iter().map(|i| i.instance_id.clone()).collect();
    let mut found = tagged(&wanted);
    let mut out = Vec::new();
    for (n, id) in ids.iter().enumerate() {
        let mut list = found.remove(&id.instance_id).unwrap_or_default();
        if leader_matches(id) && !list.iter().any(|(p, _)| *p == id.pid) {
            list.push((id.pid, id.start_ticks));
        }
        out.extend(list.into_iter().map(|(p, t)| (n, p, t)));
    }
    out
}

fn wait_gone(ids: &[Identity], procs: &[(usize, i32, u64)], until: Instant) -> bool {
    loop {
        for id in ids {
            reap_leader(id);
        }
        if !procs.iter().any(|&(_, p, t)| running(p, t)) {
            return true;
        }
        if Instant::now() >= until {
            return false;
        }
        std::thread::sleep(POLL);
    }
}

pub fn terminate(ids: &[Identity]) -> Vec<i32> {
    // TERM verified leaders, wait TERM_GRACE, then KILL verified leftovers; returns every pid that was ours.
    let first = members(ids);
    for (n, id) in ids.iter().enumerate() {
        if leader_alive(id) {
            signal(id, id.pid, libc::SIGTERM);
        } else {
            for &(_, p, _) in first.iter().filter(|m| m.0 == n) {
                signal(id, p, libc::SIGKILL);
            }
        }
    }
    wait_gone(ids, &first, Instant::now() + TERM_GRACE);
    let left: Vec<(usize, i32, u64)> = members(ids)
        .into_iter()
        .filter(|&(_, p, t)| running(p, t))
        .collect();
    for &(n, p, _) in &left {
        signal(&ids[n], p, libc::SIGKILL);
    }
    wait_gone(ids, &left, Instant::now() + KILL_WAIT);
    let mut seen: HashSet<i32> = first.iter().map(|m| m.1).collect();
    seen.extend(left.iter().map(|m| m.1));
    seen.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::CommandExt;
    use std::process::{Child, Command};

    fn setsid_sleep(tag: Option<&str>) -> Child {
        let mut cmd = Command::new("sleep");
        cmd.arg("30").env_remove(INSTANCE_ENV);
        if let Some(t) = tag {
            cmd.env(INSTANCE_ENV, t);
        }
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
        cmd.spawn().expect("spawn sleep")
    }

    fn exited(child: &mut Child) -> bool {
        for _ in 0..100 {
            if child.try_wait().expect("try_wait").is_some() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }

    #[test]
    fn gate_refuses_wrong_start_ticks_and_boot_id_then_accepts_exact_identity() {
        let mut child = setsid_sleep(None);
        let pid = child.id() as i32;
        let good = capture("inst-test-gate", pid).expect("identity");
        let wrong_ticks = Identity {
            start_ticks: good.start_ticks + 1,
            ..good.clone()
        };
        let wrong_boot = Identity {
            boot_id: "00000000-0000-0000-0000-000000000000".to_string(),
            ..good.clone()
        };
        assert!(!leader_matches(&wrong_ticks));
        assert!(!signal(&wrong_ticks, pid, libc::SIGKILL));
        assert!(!leader_matches(&wrong_boot));
        assert!(!signal(&wrong_boot, pid, libc::SIGKILL));
        assert!(child.try_wait().expect("try_wait").is_none(), "mismatched identity must not signal");
        assert!(leader_matches(&good));
        assert!(signal(&good, pid, libc::SIGKILL));
        assert!(exited(&mut child));
    }

    #[test]
    fn gate_accepts_tagged_member_and_refuses_untagged_stranger() {
        let mut member = setsid_sleep(Some("inst-test-tag"));
        let mut stranger = setsid_sleep(None);
        let leader = Identity {
            instance_id: "inst-test-tag".to_string(),
            pid: i32::MAX,
            boot_id: boot_id().to_string(),
            start_ticks: 0,
        };
        let spid = stranger.id() as i32;
        assert!(!signal(&leader, spid, libc::SIGKILL));
        assert!(stranger.try_wait().expect("try_wait").is_none());
        assert!(signal(&leader, member.id() as i32, libc::SIGKILL));
        assert!(exited(&mut member));
        stranger.kill().expect("kill own stranger");
        stranger.wait().expect("wait stranger");
    }

    #[test]
    fn unsafe_targets_are_refused() {
        let own = std::process::id() as i32;
        let pgrp = unsafe { libc::getpgrp() };
        for pid in [-1, 0, 1, own, pgrp] {
            assert!(!safe_target(pid), "pid {pid} must be refused");
            let id = Identity {
                instance_id: "x".to_string(),
                pid,
                boot_id: boot_id().to_string(),
                start_ticks: start_ticks(pid.max(1)).unwrap_or(0),
            };
            assert!(!signal(&id, pid, 0));
        }
    }

    #[test]
    fn terminate_kills_only_the_instance() {
        let mut leader = setsid_sleep(Some("inst-test-term"));
        let mut stranger = setsid_sleep(None);
        let id = capture("inst-test-term", leader.id() as i32).expect("identity");
        let pids = terminate(std::slice::from_ref(&id));
        assert!(pids.contains(&id.pid));
        assert!(!running(id.pid, id.start_ticks), "leader must be gone and reaped");
        let _ = leader.wait();
        assert!(stranger.try_wait().expect("try_wait").is_none());
        stranger.kill().expect("kill own stranger");
        stranger.wait().expect("wait stranger");
    }
}
