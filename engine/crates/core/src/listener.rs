// Who listens on an engine port: LISTEN sockets in /proc/net/tcp and tcp6 matched to process fds.

use crate::proc_ident::{self, Identity};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortOwner {
    Nobody,
    Ours,
    Foreign { pid: Option<i32>, name: String },
}

impl PortOwner {
    pub fn describe(&self, port: u16) -> String {
        // Human text for a port held by someone else.
        match self {
            PortOwner::Foreign { pid: Some(pid), name } => {
                format!("port {port} is held by PID {pid} ({name}), not by this engine")
            }
            _ => format!("port {port} is held by a process Goethe cannot inspect, not by this engine"),
        }
    }
}

const LOOPBACK_V4: &[&str] = &[
    "0100007F",
    "00000000",
    "0000000000000000FFFF00000100007F",
    "0000000000000000FFFF000000000000",
];
const ANY_V6: &str = "00000000000000000000000000000000";

pub fn parse_listen_inodes(table: &str, port: u16) -> Vec<(u64, bool)> {
    // LISTEN (state 0A) sockets on port that a connection to 127.0.0.1 can reach, as (inode, only_if_dual_stack).
    table
        .lines()
        .skip(1)
        .filter_map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            let (addr, bound) = f.get(1)?.split_once(':')?;
            if u16::from_str_radix(bound, 16).ok()? != port || *f.get(3)? != "0A" {
                return None;
            }
            let dual = addr == ANY_V6;
            (dual || LOOPBACK_V4.contains(&addr)).then_some(())?;
            Some((f.get(9)?.parse().ok()?, dual))
        })
        .collect()
}

fn listen_inodes(port: u16) -> Vec<(u64, bool)> {
    ["/proc/net/tcp", "/proc/net/tcp6"]
        .iter()
        .filter_map(|p| std::fs::read_to_string(p).ok())
        .flat_map(|t| parse_listen_inodes(&t, port))
        .collect()
}

fn socket_inodes(pid: i32) -> HashSet<u64> {
    // Socket inodes among the open fds of pid ("socket:[N]" links); empty when the fds are not readable.
    let Ok(rd) = std::fs::read_dir(format!("/proc/{pid}/fd")) else {
        return HashSet::new();
    };
    rd.flatten()
        .filter_map(|e| std::fs::read_link(e.path()).ok())
        .filter_map(|l| {
            let s = l.to_string_lossy().into_owned();
            s.strip_prefix("socket:[")?.strip_suffix(']')?.parse().ok()
        })
        .collect()
}

fn holder(inode: u64) -> Option<i32> {
    // The process holding a socket inode, from every /proc/<pid>/fd this user can read.
    std::fs::read_dir("/proc")
        .ok()?
        .flatten()
        .filter_map(|e| e.file_name().to_str()?.parse::<i32>().ok())
        .find(|pid| socket_inodes(*pid).contains(&inode))
}

pub fn port_owner(id: &Identity, port: u16) -> PortOwner {
    // Every listener that 127.0.0.1:port reaches must be an fd of this instance (verified leader or tagged process).
    let found = listen_inodes(port);
    if found.is_empty() {
        return PortOwner::Nobody;
    }
    let ours: HashSet<u64> = proc_ident::instance_pids(id).into_iter().flat_map(socket_inodes).collect();
    let ours_v4 = found.iter().any(|(i, dual)| !dual && ours.contains(i));
    let foreign = found.iter().find(|(i, dual)| !ours.contains(i) && !(*dual && ours_v4));
    let Some(&(inode, _)) = foreign else {
        return if found.iter().any(|(i, _)| ours.contains(i)) { PortOwner::Ours } else { PortOwner::Nobody };
    };
    let pid = holder(inode);
    let name = pid
        .and_then(|p| std::fs::read_to_string(format!("/proc/{p}/comm")).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    PortOwner::Foreign { pid, name }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::process::Command;

    const TABLE: &str = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:7531 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 4242 1 0 100 0 0 10 0
   1: 0100007F:7531 0100007F:9C40 01 00000000:00000000 00:00000000 00000000  1000        0 4343 1 0 20 4 30 10 -1
   2: 00000000:7532 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 4444 1 0 100 0 0 10 0
   3: 0501A8C0:7533 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 4545 1 0 100 0 0 10 0
";
    const TABLE6: &str = "  sl  local_address                         remote_address                        st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 00000000000000000000000000000000:7531 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 5151 1 0 100 0 0 10 0
   1: 00000000000000000000000001000000:7531 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 5252 1 0 100 0 0 10 0
   2: 0000000000000000FFFF00000100007F:7531 00000000000000000000000000000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 5353 1 0 100 0 0 10 0
";

    #[test]
    fn parses_listeners_reachable_on_loopback_only() {
        assert_eq!(parse_listen_inodes(TABLE, 30001), vec![(4242, false)]);
        assert_eq!(parse_listen_inodes(TABLE, 30002), vec![(4444, false)]);
        assert!(parse_listen_inodes(TABLE, 30003).is_empty(), "192.168.1.5 is not reached by 127.0.0.1");
        assert_eq!(parse_listen_inodes(TABLE6, 30001), vec![(5151, true), (5353, false)]);
    }

    fn wait_exec(pid: u32) {
        // Until a spawned child has exec'd it still holds this process's sockets; wait for it to become sleep.
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
    fn real_listener_is_ours_only_for_its_own_process() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let me = proc_ident::capture("inst-test-listener", std::process::id() as i32).unwrap();
        assert_eq!(port_owner(&me, port), PortOwner::Ours);
        let mut other = Command::new("sleep").arg("30").env_remove(proc_ident::INSTANCE_ENV).spawn().unwrap();
        wait_exec(other.id());
        let them = proc_ident::capture("inst-test-other", other.id() as i32).unwrap();
        match port_owner(&them, port) {
            PortOwner::Foreign { pid, .. } => assert_eq!(pid, Some(std::process::id() as i32)),
            owner => panic!("expected a foreign holder, got {owner:?}"),
        }
        drop(listener);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while port_owner(&them, port) != PortOwner::Nobody && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(port_owner(&them, port), PortOwner::Nobody, "the socket closes once every fork that inherited it has exec'd");
        other.kill().unwrap();
        other.wait().unwrap();
    }
}
