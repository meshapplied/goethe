// Host profile and runtime environment: CPU, memory, disks, OS, containers, cloud, and prompts.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CString;
use std::io::{IsTerminal, Write};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HostInfo {
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub disks: Vec<DiskInfo>,
    pub os: OsInfo,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nvidia_driver: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda_driver: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CpuInfo {
    pub vendor: String,
    pub model: String,
    pub sockets: u32,
    pub physical_cores: u32,
    pub logical_threads: u32,
    pub numa_nodes: Vec<NumaNode>,
    pub flags: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NumaNode {
    pub node: u32,
    pub cpus: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_mb: u64,
    pub available_mb: u64,
    pub swap_total_mb: u64,
    pub swap_free_mb: u64,
    pub hugepages_total: u64,
    pub hugepages_free: u64,
    pub hugepage_size_kb: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiskInfo {
    pub role: String,
    pub path: String,
    pub mount_point: String,
    pub fs_type: String,
    pub total_gb: f64,
    pub free_gb: f64,
    pub network_fs: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OsInfo {
    pub kernel: String,
    pub distro: String,
    pub glibc: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,
    pub systemd_pid1: bool,
    pub cloud: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instance_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persistent_volume: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub public_endpoints: Vec<String>,
}

const NETWORK_FS: &[&str] = &["nfs", "nfs4", "cifs", "smb3", "smbfs", "ceph", "glusterfs", "9p", "lustre", "beegfs", "gpfs", "afs"];

pub fn sysroot() -> PathBuf {
    // Root for /proc, /sys and /etc reads; GOETHE_TEST_SYSROOT redirects it in tests.
    std::env::var_os("GOETHE_TEST_SYSROOT")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

pub fn probe_host(home: &Path, cloud: &str) -> HostInfo {
    // Read CPU, memory, OS and the filesystems behind GOETHE_HOME and models/.
    let root = sysroot();
    let read = |rel: &str| std::fs::read_to_string(root.join(rel)).unwrap_or_default();
    let mut cpu = parse_cpuinfo(&read("proc/cpuinfo"));
    cpu.numa_nodes = numa_nodes(&root);
    let mountinfo = read("proc/self/mountinfo");
    let disks = [("goethe_home", home.to_path_buf()), ("models", home.join("models"))]
        .into_iter()
        .filter_map(|(role, p)| disk_info(role, &p, &mountinfo, cloud))
        .collect();
    HostInfo {
        cpu,
        memory: parse_meminfo(&read("proc/meminfo")),
        disks,
        os: OsInfo {
            kernel: read("proc/sys/kernel/osrelease").trim().to_string(),
            distro: os_release_name(&read("etc/os-release")),
            glibc: glibc_version(),
        },
        nvidia_driver: parse_nvidia_proc_version(&read("proc/driver/nvidia/version")),
        cuda_driver: None,
    }
}

pub fn probe_environment(public_port: u16) -> EnvironmentInfo {
    let env: BTreeMap<String, String> = std::env::vars().collect();
    detect_environment(&sysroot(), &env, public_port)
}

pub fn mem_available_gb() -> Option<f64> {
    // MemAvailable in GB, or None when /proc/meminfo is unreadable.
    let text = std::fs::read_to_string(sysroot().join("proc/meminfo")).ok()?;
    let mem = parse_meminfo(&text);
    (mem.total_mb > 0).then_some(mem.available_mb as f64 / 1024.0)
}

pub fn detect_environment(root: &Path, env: &BTreeMap<String, String>, public_port: u16) -> EnvironmentInfo {
    // Classify container runtime, init system, cloud provider, volumes and public endpoints.
    let container = detect_container(root, env);
    let (cloud, instance_id) = detect_cloud(root, env);
    let workspace = root.join("workspace");
    let persistent_volume = (workspace.is_dir() && crate::paths::is_writable(&workspace)).then(|| "/workspace".to_string());
    let systemd_pid1 = std::fs::read_to_string(root.join("proc/1/comm"))
        .map(|s| s.trim() == "systemd")
        .unwrap_or(false);
    EnvironmentInfo {
        public_endpoints: endpoint_hints(&cloud, env, public_port),
        container,
        systemd_pid1,
        cloud,
        instance_id,
        persistent_volume,
    }
}

pub fn container_kind() -> Option<String> {
    // The container runtime this process runs in (kubernetes, podman, docker, lxc), None on a host or VM.
    detect_container(Path::new("/"), &std::env::vars().collect())
}

fn detect_container(root: &Path, env: &BTreeMap<String, String>) -> Option<String> {
    // Kubernetes, podman, docker, then lxc markers, in that order.
    let cgroup = std::fs::read_to_string(root.join("proc/1/cgroup")).unwrap_or_default();
    let pid1_env = std::fs::read(root.join("proc/1/environ")).unwrap_or_default();
    let pid1_env = String::from_utf8_lossy(&pid1_env).replace('\0', "\n");
    let container_var = env
        .get("container")
        .cloned()
        .or_else(|| pid1_env.lines().find_map(|l| l.strip_prefix("container=").map(String::from)));
    if env.contains_key("KUBERNETES_SERVICE_HOST") || cgroup.contains("kubepods") {
        return Some("kubernetes".to_string());
    }
    if root.join("run/.containerenv").exists() || container_var.as_deref() == Some("podman") {
        return Some("podman".to_string());
    }
    if root.join(".dockerenv").exists() || cgroup.contains("docker") || container_var.as_deref() == Some("docker") {
        return Some("docker".to_string());
    }
    if cgroup.contains("lxc") || container_var.as_deref() == Some("lxc") {
        return Some("lxc".to_string());
    }
    None
}

fn detect_cloud(root: &Path, env: &BTreeMap<String, String>) -> (String, Option<String>) {
    // Provider env markers win over DMI strings because GPU clouds run on other vendors' hardware.
    let has_prefix = |p: &str| env.keys().any(|k| k.starts_with(p));
    if let Some(pod) = env.get("RUNPOD_POD_ID") {
        return ("runpod".into(), Some(pod.clone()));
    }
    if env.contains_key("VAST_CONTAINERLABEL") || has_prefix("VAST_") {
        return ("vast".into(), env.get("VAST_CONTAINERLABEL").cloned());
    }
    if has_prefix("PAPERSPACE_") {
        return ("paperspace".into(), env.get("PAPERSPACE_MACHINE_ID").cloned());
    }
    if env.keys().any(|k| k.contains("COREWEAVE")) {
        return ("coreweave".into(), None);
    }
    let dmi = |f: &str| {
        std::fs::read_to_string(root.join("sys/class/dmi/id").join(f))
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    };
    let (vendor, product, board) = (dmi("sys_vendor"), dmi("product_name"), dmi("board_vendor"));
    let chassis_tag = dmi("chassis_asset_tag");
    let all = format!("{vendor} {product} {board} {chassis_tag}");
    if all.contains("Lambda") {
        return ("lambda".into(), None);
    }
    if vendor == "Amazon EC2" || board.starts_with("Amazon") {
        let tag = dmi("board_asset_tag");
        return ("aws".into(), tag.starts_with("i-").then_some(tag));
    }
    if product == "Google Compute Engine" {
        return ("gcp".into(), None);
    }
    if vendor == "Microsoft Corporation" && product == "Virtual Machine" {
        return ("azure".into(), None);
    }
    if chassis_tag == "OracleCloud.com" {
        return ("oci".into(), None);
    }
    if all.contains("CoreWeave") {
        return ("coreweave".into(), None);
    }
    if all.contains("Paperspace") {
        return ("paperspace".into(), None);
    }
    ("none".into(), None)
}

fn endpoint_hints(cloud: &str, env: &BTreeMap<String, String>, public_port: u16) -> Vec<String> {
    // Turn provider port-mapping env into reachable URLs for the gateway.
    let mut out = Vec::new();
    let (ip_key, port_prefix) = match cloud {
        "runpod" => ("RUNPOD_PUBLIC_IP", "RUNPOD_TCP_PORT_"),
        "vast" => ("PUBLIC_IPADDR", "VAST_TCP_PORT_"),
        _ => return out,
    };
    if cloud == "runpod" {
        if let Some(pod) = env.get("RUNPOD_POD_ID") {
            out.push(format!("https://{pod}-{public_port}.proxy.runpod.net"));
        }
    }
    let ip = env.get(ip_key).cloned().unwrap_or_else(|| "<public-ip>".to_string());
    for (k, v) in env {
        if let Some(internal) = k.strip_prefix(port_prefix) {
            out.push(format!("tcp://{ip}:{v} -> port {internal}"));
        }
    }
    out
}

pub fn parse_cpuinfo(text: &str) -> CpuInfo {
    // Count sockets, physical cores and threads from /proc/cpuinfo blocks.
    let mut info = CpuInfo::default();
    let mut sockets = BTreeSet::new();
    let mut cores = BTreeSet::new();
    let mut flags = String::new();
    for block in text.split("\n\n") {
        let mut physical = None;
        let mut core = None;
        let mut is_cpu = false;
        for line in block.lines() {
            let Some((k, v)) = line.split_once(':') else {
                continue;
            };
            let (k, v) = (k.trim(), v.trim());
            match k {
                "processor" => is_cpu = true,
                "vendor_id" | "CPU implementer" if info.vendor.is_empty() => info.vendor = v.to_string(),
                "model name" | "Model" if info.model.is_empty() => info.model = v.to_string(),
                "physical id" => physical = Some(v.to_string()),
                "core id" => core = Some(v.to_string()),
                "flags" | "Features" if flags.is_empty() => flags = v.to_string(),
                _ => {}
            }
        }
        if !is_cpu {
            continue;
        }
        info.logical_threads += 1;
        let socket = physical.unwrap_or_else(|| "0".to_string());
        sockets.insert(socket.clone());
        cores.insert((socket, core.unwrap_or_else(|| info.logical_threads.to_string())));
    }
    info.sockets = sockets.len() as u32;
    info.physical_cores = cores.len() as u32;
    let set: BTreeSet<&str> = flags.split_whitespace().collect();
    for f in ["avx2", "avx512f"] {
        if set.contains(f) {
            info.flags.push(f.to_string());
        }
    }
    if set.iter().any(|f| f.starts_with("amx")) {
        info.flags.push("amx".to_string());
    }
    info
}

fn numa_nodes(root: &Path) -> Vec<NumaNode> {
    // NUMA nodes and their CPU lists from sysfs.
    let mut out: Vec<NumaNode> = std::fs::read_dir(root.join("sys/devices/system/node"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            let node = name.strip_prefix("node")?.parse().ok()?;
            let cpus = std::fs::read_to_string(e.path().join("cpulist")).ok()?.trim().to_string();
            Some(NumaNode { node, cpus })
        })
        .collect();
    out.sort_by_key(|n| n.node);
    out
}

pub fn parse_meminfo(text: &str) -> MemoryInfo {
    // Memory, swap and hugepage counters from /proc/meminfo.
    let mut m = MemoryInfo::default();
    for line in text.lines() {
        let Some((k, v)) = line.split_once(':') else {
            continue;
        };
        let n: u64 = v.split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(0);
        match k {
            "MemTotal" => m.total_mb = n / 1024,
            "MemAvailable" => m.available_mb = n / 1024,
            "SwapTotal" => m.swap_total_mb = n / 1024,
            "SwapFree" => m.swap_free_mb = n / 1024,
            "HugePages_Total" => m.hugepages_total = n,
            "HugePages_Free" => m.hugepages_free = n,
            "Hugepagesize" => m.hugepage_size_kb = n,
            _ => {}
        }
    }
    m
}

fn disk_info(role: &str, path: &Path, mountinfo: &str, cloud: &str) -> Option<DiskInfo> {
    // statvfs the path and attach the mount it lives on.
    let canonical = std::fs::canonicalize(path).ok()?;
    let c = CString::new(canonical.as_os_str().as_bytes()).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return None;
    }
    let gb = |blocks: u64| (blocks as f64 * st.f_frsize as f64) / 1e9;
    let (mount_point, fs_type) = mount_for(&canonical, mountinfo).unwrap_or_default();
    Some(DiskInfo {
        role: role.to_string(),
        path: canonical.display().to_string(),
        network_fs: is_network_fs(&fs_type, cloud),
        mount_point,
        fs_type,
        total_gb: gb(st.f_blocks as u64),
        free_gb: gb(st.f_bavail as u64),
    })
}

pub fn mount_for(path: &Path, mountinfo: &str) -> Option<(String, String)> {
    // Longest mount point prefix of path from /proc/self/mountinfo.
    let mut best: Option<(String, String)> = None;
    for line in mountinfo.lines() {
        let Some((left, right)) = line.split_once(" - ") else {
            continue;
        };
        let Some(mount) = left.split_whitespace().nth(4).map(unescape_mount) else {
            continue;
        };
        let fs = right.split_whitespace().next().unwrap_or("").to_string();
        if path.starts_with(&mount) && best.as_ref().is_none_or(|(m, _)| mount.len() >= m.len()) {
            best = Some((mount, fs));
        }
    }
    best
}

fn unescape_mount(s: &str) -> String {
    s.replace("\\040", " ").replace("\\011", "\t").replace("\\134", "\\")
}

pub fn is_network_fs(fs_type: &str, cloud: &str) -> bool {
    let base = fs_type.split('.').next().unwrap_or(fs_type);
    NETWORK_FS.contains(&base) || base == "fuse" || base == "fuseblk" || (base == "overlay" && cloud == "runpod")
}

fn os_release_name(text: &str) -> String {
    // PRETTY_NAME from /etc/os-release.
    text.lines()
        .find_map(|l| l.strip_prefix("PRETTY_NAME="))
        .map(|v| v.trim_matches('"').to_string())
        .unwrap_or_default()
}

fn glibc_version() -> String {
    // Runtime glibc version of this process.
    let p = unsafe { libc::gnu_get_libc_version() };
    if p.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

pub fn parse_nvidia_proc_version(text: &str) -> Option<String> {
    // Kernel module version from /proc/driver/nvidia/version.
    let line = text.lines().find(|l| l.contains("NVRM version"))?;
    line.split_whitespace()
        .find(|t| t.contains('.') && t.chars().all(|c| c.is_ascii_digit() || c == '.'))
        .map(String::from)
}

pub fn summary_lines(host: &HostInfo, env: &EnvironmentInfo, bind: &str) -> Vec<String> {
    // Human-readable host and environment lines for goethe hw profile.
    let c = &host.cpu;
    let numa: Vec<String> = c.numa_nodes.iter().map(|n| format!("node{} cpus {}", n.node, n.cpus)).collect();
    let m = &host.memory;
    let gb = |mb: u64| mb as f64 / 1024.0;
    let mut out = vec![
        format!(
            "CPU: {} ({}), {} socket(s), {} cores, {} threads, flags [{}]",
            c.model,
            c.vendor,
            c.sockets,
            c.physical_cores,
            c.logical_threads,
            c.flags.join(" ")
        ),
        format!("NUMA: {}", if numa.is_empty() { "unknown".to_string() } else { numa.join("; ") }),
        format!(
            "RAM: {:.1} GB total, {:.1} GB available; swap {:.1}/{:.1} GB free; hugepages {}/{} free ({} kB)",
            gb(m.total_mb),
            gb(m.available_mb),
            gb(m.swap_free_mb),
            gb(m.swap_total_mb),
            m.hugepages_free,
            m.hugepages_total,
            m.hugepage_size_kb
        ),
    ];
    for d in &host.disks {
        out.push(format!(
            "Disk {}: {} on {} ({}{}) {:.0} GB free of {:.0} GB",
            d.role,
            d.path,
            d.mount_point,
            d.fs_type,
            if d.network_fs { ", network filesystem" } else { "" },
            d.free_gb,
            d.total_gb
        ));
    }
    out.push(format!("OS: {}, kernel {}, glibc {}", host.os.distro, host.os.kernel, host.os.glibc));
    out.push(format!(
        "NVIDIA driver {}, CUDA driver {}",
        host.nvidia_driver.as_deref().unwrap_or("unknown"),
        host.cuda_driver.as_deref().unwrap_or("unknown")
    ));
    out.push(format!(
        "Environment: {}, {}, cloud {}{}{}",
        env.container.as_deref().map(|c| format!("container {c}")).unwrap_or_else(|| "no container".into()),
        if env.systemd_pid1 { "systemd is PID 1" } else { "no systemd" },
        env.cloud,
        env.instance_id.as_deref().map(|i| format!(" ({i})")).unwrap_or_default(),
        env.persistent_volume.as_deref().map(|v| format!(", persistent volume {v}")).unwrap_or_default()
    ));
    for e in &env.public_endpoints {
        out.push(format!("Public endpoint hint: {e}"));
    }
    if !env.public_endpoints.is_empty() {
        out.push(format!("Public endpoint hint: {}", public_access_hint(bind)));
    }
    out
}

pub fn public_access_hint(bind: &str) -> String {
    // What remote clients still need before the gateway lets them in.
    if bind.parse::<std::net::IpAddr>().map_or(true, |ip| ip.is_loopback()) {
        format!(
            "the gateway listens on {bind} only: run `goethe gateway --bind 0.0.0.0`, then allow clients with `goethe allowlist --add <client-ip>/32`"
        )
    } else {
        format!("the gateway listens on {bind}; allow clients with `goethe allowlist --add <client-ip>/32` (all other addresses are refused)")
    }
}

pub fn confirm(question: &str, yes: bool) -> Result<(), String> {
    // Proceed on --yes or a y answer on a terminal; without a terminal exit 3 (confirmation required).
    if yes {
        return Ok(());
    }
    if !(std::io::stdin().is_terminal() && std::io::stderr().is_terminal()) {
        eprintln!("{question}");
        eprintln!("confirmation required: re-run with --yes (no terminal to ask on)");
        std::process::exit(3);
    }
    eprint!("{question} [y/N] ");
    std::io::stderr().flush().ok();
    let mut buf = String::new();
    std::io::stdin().read_line(&mut buf).map_err(|e| e.to_string())?;
    if matches!(buf.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
        Ok(())
    } else {
        Err("aborted".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root_with(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for (rel, body) in files {
            let p = dir.path().join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, body).unwrap();
        }
        dir
    }

    type Case<'a> = (&'a str, Vec<(&'a str, &'a str)>, Vec<(&'a str, &'a str)>, &'a str, Option<&'a str>);

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn cloud_and_container_matrix() {
        let cases: Vec<Case> = vec![
            ("runpod", vec![(".dockerenv", "")], vec![("RUNPOD_POD_ID", "abc123"), ("RUNPOD_TCP_PORT_8443", "40123"), ("RUNPOD_PUBLIC_IP", "1.2.3.4")], "runpod", Some("docker")),
            ("vast", vec![(".dockerenv", "")], vec![("VAST_CONTAINERLABEL", "C.77"), ("PUBLIC_IPADDR", "5.6.7.8"), ("VAST_TCP_PORT_8443", "41000")], "vast", Some("docker")),
            ("lambda", vec![("sys/class/dmi/id/sys_vendor", "Lambda\n"), ("proc/1/comm", "systemd\n")], vec![], "lambda", None),
            ("aws", vec![("sys/class/dmi/id/sys_vendor", "Amazon EC2\n"), ("sys/class/dmi/id/board_asset_tag", "i-0abc\n")], vec![], "aws", None),
            ("gcp", vec![("sys/class/dmi/id/product_name", "Google Compute Engine\n")], vec![], "gcp", None),
            ("azure", vec![("sys/class/dmi/id/sys_vendor", "Microsoft Corporation\n"), ("sys/class/dmi/id/product_name", "Virtual Machine\n")], vec![], "azure", None),
            ("oci", vec![("sys/class/dmi/id/chassis_asset_tag", "OracleCloud.com\n")], vec![], "oci", None),
            ("docker", vec![(".dockerenv", ""), ("proc/1/cgroup", "0::/\n")], vec![], "none", Some("docker")),
            ("k8s", vec![("proc/1/cgroup", "0::/kubepods/besteffort\n")], vec![("KUBERNETES_SERVICE_HOST", "10.0.0.1")], "none", Some("kubernetes")),
            ("coreweave", vec![], vec![("KUBERNETES_SERVICE_HOST", "10.0.0.1"), ("COREWEAVE_REGION", "us-east")], "coreweave", Some("kubernetes")),
            ("paperspace", vec![], vec![("PAPERSPACE_MACHINE_ID", "ps1")], "paperspace", None),
            ("bare", vec![("proc/1/comm", "systemd\n"), ("sys/class/dmi/id/sys_vendor", "Supermicro\n")], vec![], "none", None),
        ];
        for (name, files, vars, cloud, container) in cases {
            let root = root_with(&files);
            let e = detect_environment(root.path(), &env(&vars), 8443);
            assert_eq!(e.cloud, cloud, "{name}");
            assert_eq!(e.container.as_deref(), container, "{name}");
            if name == "runpod" {
                assert_eq!(e.instance_id.as_deref(), Some("abc123"));
                assert!(e.public_endpoints.contains(&"https://abc123-8443.proxy.runpod.net".to_string()));
                assert!(e.public_endpoints.contains(&"tcp://1.2.3.4:40123 -> port 8443".to_string()));
            }
            if name == "vast" {
                assert!(e.public_endpoints.contains(&"tcp://5.6.7.8:41000 -> port 8443".to_string()));
            }
            if name == "aws" {
                assert_eq!(e.instance_id.as_deref(), Some("i-0abc"));
            }
            assert_eq!(e.systemd_pid1, name == "lambda" || name == "bare", "{name}");
        }
    }

    #[test]
    fn workspace_volume_detected() {
        let root = root_with(&[("workspace/.keep", "")]);
        let e = detect_environment(root.path(), &env(&[("RUNPOD_POD_ID", "p")]), 8443);
        assert_eq!(e.persistent_volume.as_deref(), Some("/workspace"));
    }

    #[test]
    fn cpuinfo_counts_sockets_cores_threads() {
        let block = |p: u32, phys: u32, core: u32| {
            format!(
                "processor\t: {p}\nvendor_id\t: GenuineIntel\nmodel name\t: Xeon\nphysical id\t: {phys}\ncore id\t\t: {core}\nflags\t\t: fpu avx2 avx512f amx_tile\n"
            )
        };
        let text = [block(0, 0, 0), block(1, 0, 0), block(2, 1, 0), block(3, 1, 1)].join("\n");
        let c = parse_cpuinfo(&text);
        assert_eq!((c.sockets, c.physical_cores, c.logical_threads), (2, 3, 4));
        assert_eq!(c.flags, vec!["avx2", "avx512f", "amx"]);
        assert_eq!(c.vendor, "GenuineIntel");
    }

    #[test]
    fn meminfo_and_mounts() {
        let m = parse_meminfo("MemTotal: 2097152 kB\nMemAvailable: 1048576 kB\nSwapTotal: 0 kB\nHugePages_Total: 4\nHugepagesize: 2048 kB\n");
        assert_eq!((m.total_mb, m.available_mb, m.hugepages_total, m.hugepage_size_kb), (2048, 1024, 4, 2048));
        let mi = "22 1 8:1 / / rw - ext4 /dev/sda1 rw\n40 22 0:50 / /workspace rw - fuse.mfs mfs rw\n41 22 0:51 / /mnt/my\\040disk rw - nfs4 srv:/x rw\n";
        assert_eq!(mount_for(Path::new("/workspace/.goethe"), mi), Some(("/workspace".into(), "fuse.mfs".into())));
        assert_eq!(mount_for(Path::new("/home/u"), mi), Some(("/".into(), "ext4".into())));
        assert_eq!(mount_for(Path::new("/mnt/my disk/m"), mi).unwrap().1, "nfs4");
        assert!(is_network_fs("fuse.mfs", "none"));
        assert!(is_network_fs("nfs4", "none"));
        assert!(!is_network_fs("ext4", "runpod"));
        assert!(is_network_fs("overlay", "runpod"));
        assert!(!is_network_fs("overlay", "none"));
    }

    #[test]
    fn nvidia_proc_version() {
        let t = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.82.07  Release Build\nGCC version: gcc 13";
        assert_eq!(parse_nvidia_proc_version(t).as_deref(), Some("580.82.07"));
    }
}
