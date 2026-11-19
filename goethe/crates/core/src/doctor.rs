// goethe doctor health checks for local install.

use crate::hf;
use crate::pyenv::{self, Layer};
use crate::paths::{goethe_home, goethe_sock};
use crate::registry::{fetch, registry_url};
use crate::rpc;
use serde::Serialize;
use serde_json::json;
use std::fs;
use std::net::{TcpListener, TcpStream};
use std::process::Stdio;
use std::path::Path;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

const CHECK_TIMEOUT: Duration = Duration::from_secs(3);

#[derive(Debug, Clone, Serialize)]
pub struct Check {
    pub name: String,
    pub status: String,
    pub hint: String,
}

pub fn run(json: bool, probe_public: bool) -> Result<(), String> {
    let home = goethe_home();
    let checks = collect_checks(&home);
    let fail = checks.iter().any(|c| c.status == "FAIL");
    if json {
        println!("{}", serde_json::to_string_pretty(&checks).map_err(|e| e.to_string())?);
    } else {
        let (shown, folded) = fold_ok(&checks, crate::step::verbose());
        let mut table = crate::term::Table::new(&["CHECK", "STATUS", "DETAIL"]);
        for c in &shown {
            table.row(vec![c.name.clone(), crate::term::status(&c.status), c.hint.clone()]);
        }
        table.print();
        if !folded.is_empty() {
            println!("also OK: {} (goethe -v doctor lists them)", folded.join(", "));
        }
        let fixes = crate::fix::pending(&crate::paths::goethe_home());
        if !fixes.is_empty() {
            println!("\ngoethe doctor --fix can fix: {}", fixes.iter().map(|(_, why)| why.as_str()).collect::<Vec<_>>().join("; "));
        }
        if cfg!(target_os = "macos") {
            println!("\nOn a Mac goethe serves GGUF models with llama.cpp on Metal.");
        } else {
            println!("\nGPU architectures goethe supports: {}", crate::gpu_arch::microarchitectures_text());
            println!("Architectures of this machine: {}", crate::packs::local_archs(&home).join(", "));
        }
    }
    if probe_public {
        probe_public_reachability(&home);
    }
    if fail {
        std::process::exit(1);
    }
    Ok(())
}

fn fold_ok(checks: &[Check], all: bool) -> (Vec<&Check>, Vec<&str>) {
    // The rows that need a look (and the hardware row, which says what goethe sees), with the names of the plain OK rows folded away.
    if all {
        return (checks.iter().collect(), Vec::new());
    }
    let (shown, folded): (Vec<&Check>, Vec<&Check>) = checks.iter().partition(|c| c.status != "OK" || c.name == "hardware");
    (shown, folded.iter().map(|c| c.name.as_str()).collect())
}

fn spawn_check<F: FnOnce() -> Check + Send + 'static>(f: F) -> mpsc::Receiver<Check> {
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx
}

fn recv_check(rx: mpsc::Receiver<Check>, name: &str) -> Check {
    rx.recv_timeout(CHECK_TIMEOUT).unwrap_or_else(|_| Check {
        name: name.to_string(),
        status: "WARN".into(),
        hint: "check timed out after 3s".into(),
    })
}

type CheckFn = fn(&Path) -> Check;

const CHECKS: &[(&str, CheckFn)] = &[
    ("hardware", check_hardware),
    ("driver", check_driver),
    ("platform", check_platform),
    ("glibc", |_| check_glibc()),
    ("nginx", check_nginx),
    ("goethe_home", check_goethe_home),
    ("disk", check_disk),
    ("ram", check_ram),
    ("state_db", check_state_db),
    ("environment", check_environment),
    ("cloud", check_cloud),
    ("py-env", check_pyenv),
    ("ports", check_ports),
    ("goethed", |_| check_goethed()),
    ("gateway", check_gateway),
    ("network", check_network),
    ("tls", check_tls),
    ("huggingface", check_hf),
    ("registry", check_registry),
    ("nvidia_driver", |_| check_nvidia_driver()),
    ("compiler", |_| check_compiler()),
];

const MAC_CHECKS: &[(&str, CheckFn)] = &[
    ("hardware", check_mac_hardware),
    ("platform", check_platform),
    ("llama.cpp", check_llama_cpp),
    ("nginx", check_nginx),
    ("goethe_home", check_goethe_home),
    ("disk", check_disk),
    ("state_db", check_state_db),
    ("environment", |_| check("environment", "OK", "macOS: goethe up adds a LaunchAgent, so goethed starts when you log in")),
    ("ports", check_ports),
    ("goethed", |_| check_goethed()),
    ("gateway", check_gateway),
    ("network", check_network),
    ("tls", check_tls),
    ("huggingface", check_hf),
    ("registry", check_registry),
];

pub fn mac_hardware_note(chip: &str, arm64: bool, version: &str, limit_mb: u64, free_mb: u64, total_mb: u64) -> (&'static str, String) {
    // Apple Silicon and macOS 14 or newer, then the memory the GPU may use and how much of it is free now.
    if !arm64 {
        return ("FAIL", format!("{chip}: Intel Macs are not supported; goethe needs Apple Silicon (M1 or newer)"));
    }
    let major = crate::apple::parse_version(version).map_or(0, |v| v.0);
    let gb = |mb: u64| mb as f64 / 1024.0;
    let text = format!(
        "{chip}, macOS {version}\nGPU may use {:.1} of {:.1} GB unified memory; {:.1} GB free now",
        gb(limit_mb),
        gb(total_mb),
        gb(free_mb)
    );
    if major < 14 {
        return ("FAIL", format!("{text}\nmacOS {version} is too old: llama.cpp's Metal build needs macOS 14 or newer"));
    }
    ("OK", text)
}

pub fn mac_memory_warning(free_mb: u64, smallest: Option<(f64, &str)>) -> Option<String> {
    // RAM and GPU memory are one pool on a Mac, so the only memory warning is: even the smallest installed model does not fit now.
    let (weights_gb, id) = smallest?;
    let need = weights_gb + crate::calc::METAL_OVERHEAD_GB;
    let free = free_mb as f64 / 1024.0;
    (free < need).then(|| format!("{id} needs about {need:.1} GB and only {free:.1} GB is free; quit some apps before loading it"))
}

#[cfg(target_os = "macos")]
fn smallest_installed(home: &Path) -> Option<(f64, String)> {
    let con = crate::db::open(home).ok()?;
    crate::models::disk_gb_by_model(&con).into_iter().filter(|(_, gb)| *gb > 0.0).map(|(id, gb)| (gb, id)).min_by(|a, b| a.0.total_cmp(&b.0))
}

#[cfg(target_os = "macos")]
fn check_mac_hardware(home: &Path) -> Check {
    let g = crate::apple::gpu(&[]);
    let total = crate::sys::memory().total_kb / 1024;
    let (status, mut text) = mac_hardware_note(&g.name, crate::apple::apple_silicon(), &crate::apple::macos_version(), g.vram_total_mb, g.vram_free_mb, total);
    let smallest = smallest_installed(home);
    match mac_memory_warning(g.vram_free_mb, smallest.as_ref().map(|(gb, id)| (*gb, id.as_str()))) {
        Some(w) if status == "OK" => {
            text.push('\n');
            text.push_str(&w);
            check("hardware", "WARN", text)
        }
        _ => check("hardware", status, text),
    }
}

#[cfg(not(target_os = "macos"))]
fn check_mac_hardware(home: &Path) -> Check {
    check_hardware(home)
}

fn check_llama_cpp(home: &Path) -> Check {
    // The Metal build of llama-server, installed or fetched on the first model install.
    match crate::engines::llama_cpp::server_bin(home) {
        Ok(p) => check("llama.cpp", "OK", p.display().to_string()),
        Err(_) => match crate::engines::llama_cpp::server_ready(home) {
            Ok(()) => check("llama.cpp", "INFO", format!("llama.cpp {} is downloaded on the first model install", crate::engines::llama_cpp::TAG)),
            Err(why) => check("llama.cpp", "FAIL", why),
        },
    }
}

fn collect_checks(home: &Path) -> Vec<Check> {
    // Run every check in parallel; each result carries the check's fixed name even on timeout.
    let list = if cfg!(target_os = "macos") { MAC_CHECKS } else { CHECKS };
    let pending: Vec<_> = list
        .iter()
        .map(|(name, f)| {
            let h = home.to_path_buf();
            let f = *f;
            (*name, spawn_check(move || f(&h)))
        })
        .collect();
    pending
        .into_iter()
        .map(|(name, rx)| {
            let mut c = recv_check(rx, name);
            c.name = name.to_string();
            c
        })
        .collect()
}

fn driver_note(driver: &str) -> (&'static str, String) {
    // The NVIDIA driver against goethe's floors: 580 and newer run every supported GPU, 535 to 579 only data-center GPUs (CUDA forward compatibility), older ones nothing.
    match driver.split('.').next().and_then(|m| m.parse::<u32>().ok()) {
        Some(580..) => ("OK", format!("NVIDIA driver {driver}: every supported GPU")),
        Some(535..=579) => ("WARN", format!("NVIDIA driver {driver}: data-center GPUs only (GeForce and TITAN need 580 or newer)")),
        Some(_) => ("FAIL", format!("NVIDIA driver {driver} is older than 535 and not supported: goethe install offers to update it")),
        None => ("WARN", "the NVIDIA driver version could not be read".to_string()),
    }
}

fn check_driver(home: &Path) -> Check {
    match crate::hw::profile(home) {
        Ok(hw) => {
            let (status, note) = driver_note(&hw.driver);
            check("driver", status, note)
        }
        Err(_) => check("driver", "WARN", "no driver answered (see the hardware row)"),
    }
}

fn platform_note(link: crate::platform_cli::Link) -> (&'static str, String) {
    // Whether this node is linked to the Goethe platform, which also decides whether optimised kernels install.
    use crate::platform_cli::Link;
    match link {
        Link::NotLinked => ("INFO", "not logged in: optional; goethe login links this node and unlocks optimised kernels".to_string()),
        Link::Unreadable(e) => ("WARN", format!("login not usable ({e}); goethe login again")),
        Link::Linked { revoked: true, workspace, .. } => ("WARN", format!("revoked (workspace {workspace}); goethe login again")),
        Link::Linked { workspace, heartbeat, .. } => ("OK", format!("logged in, workspace {workspace}, heartbeat {heartbeat}")),
    }
}

fn check_platform(home: &Path) -> Check {
    let (status, note) = platform_note(crate::platform_cli::link_state(home, crate::service::goethed_running()));
    check("platform", status, note)
}

fn check(name: &str, status: &str, hint: impl Into<String>) -> Check {
    Check {
        name: name.to_string(),
        status: status.to_string(),
        hint: hint.into(),
    }
}

fn check_nvidia_driver() -> Check {
    // The kernel module version, or the first blocker (no GPU, no driver, too old, no devices, no libraries, GPU not started).
    let probe = crate::nvidia_driver::probe();
    if let Some(why) = crate::nvidia_driver::diagnose(&probe) {
        return check("nvidia_driver", "FAIL", why);
    }
    let mut version = probe.module_version.clone().unwrap_or_else(|| "version unknown".into());
    let home = crate::paths::goethe_home();
    if crate::cuda_compat::lib_dir(&home).is_some() {
        if let Err(why) = crate::cuda_compat::check(&home) {
            return check("nvidia_driver", "FAIL", format!("{version} is older than 580 (CUDA 13) and {why}; upgrade the driver to 580 or newer"));
        }
        version.push_str(" with CUDA 13 forward compatibility");
    }
    match crate::nvidia_driver::unstarted(&probe, &crate::nvidia_driver::started_bus_ids()) {
        Some((true, why)) => check("nvidia_driver", "FAIL", why),
        Some((false, note)) => check("nvidia_driver", "WARN", format!("{version}; {note}")),
        None if probe.module_version.is_some() => check("nvidia_driver", "OK", version),
        None => check("nvidia_driver", "WARN", "could not read the driver version"),
    }
}

fn check_compiler() -> Check {
    // The C++ compiler run-time kernel builds use (nvcc 13.0 takes gcc 9 to 15 or clang up to 20).
    let distro = crate::nvidia_driver::probe().distro;
    match crate::host_compiler::verdict(crate::host_compiler::find().as_ref(), &distro) {
        crate::host_compiler::Verdict::Ok(label) => check("compiler", "OK", label),
        crate::host_compiler::Verdict::TooNew(why) => check("compiler", "WARN", why),
        crate::host_compiler::Verdict::Unusable(why) => check("compiler", "FAIL", why),
    }
}

fn probe_public_reachability(home: &Path) {
    use std::io::Read;
    let bind = crate::nginx::public_bind(home);
    if bind.parse::<std::net::IpAddr>().map_or(true, |ip| ip.is_loopback()) {
        println!("public probe: {}", crate::host::public_access_hint(&bind));
        return;
    }
    let ip = match crate::domain::public_ipv4() {
        Ok(v) => v,
        Err(e) => {
            println!("public probe: {e}");
            print_public_security_advisory();
            return;
        }
    };
    let port = read_port(home, "nginx/public_port").unwrap_or(8443);
    if let Ok(canary) = std::env::var("GOETHE_CANARY_URL") {
        let base = canary.trim();
        if !base.is_empty() {
            let url = format!("{base}?host={ip}&port={port}");
            let resp = crate::net::agent().get(&url).timeout(CHECK_TIMEOUT).call();
            match resp {
                Ok(r) if r.status() == 200 => {
                    let mut body = String::new();
                    let open = r
                        .into_reader()
                        .read_to_string(&mut body)
                        .ok()
                        .map(|_| body.contains("open"))
                        .unwrap_or(false);
                    if open {
                        println!("public probe: OPEN");
                    } else {
                        println!(
                            "public probe: CLOSED (open TCP {port} on your NAT or firewall)"
                        );
                    }
                }
                Ok(r) => println!("public probe: CLOSED (canary HTTP {})", r.status()),
                Err(e) => println!("public probe: CLOSED ({e})"),
            }
        } else {
            println!("reachability unknown: set GOETHE_CANARY_URL to an external TCP probe");
        }
    } else {
        println!("reachability unknown: set GOETHE_CANARY_URL to an external TCP probe");
    }
    print_public_security_advisory();
}

fn print_public_security_advisory() {
    println!(
        "Public mode exposes the gateway to the internet. Keep the allowlist tight (goethe api allow add and remove), use API keys with scopes and rpm limits, and rotate keys."
    );
}

fn check_hardware(home: &Path) -> Check {
    // NVML probe plus per-GPU support and foreign-process counts.
    let hw = match crate::hw::profile(home) {
        Ok(h) => h,
        Err(e) => {
            let why = crate::nvidia_driver::diagnose(&crate::nvidia_driver::probe());
            return check("hardware", "FAIL", why.unwrap_or_else(|| format!("{e}; make sure nvidia-smi works")));
        }
    };
    let managed = crate::db::open(home)
        .ok()
        .map(|con| crate::placement::managed_pids(&con))
        .unwrap_or_default();
    let mut lines = vec![format!("driver {}", hw.driver)];
    let mut unsupported = 0;
    for g in &hw.gpus {
        let why = crate::gpu_arch::unsupported_reason(&g.compute_capability);
        unsupported += usize::from(why.is_some());
        let foreign = g.running_processes.iter().filter(|p| !managed.contains(&(p.pid as i32))).count();
        let plural = if foreign == 1 { "" } else { "es" };
        let um = if g.unified_memory { ", unified memory" } else { "" };
        let sm_note = why.map_or_else(String::new, |w| format!("; {w}"));
        lines.push(format!(
            "gpu{} {} ({}): {:.1} of {:.1} GB free{um}, {foreign} foreign process{plural}{sm_note}",
            g.index,
            g.name,
            crate::gpu_kind::arch_label(&g.compute_capability).unwrap_or_else(|| g.compute_capability.clone()),
            g.vram_free_mb as f64 / 1024.0,
            g.vram_total_mb as f64 / 1024.0
        ));
    }
    let busy: Vec<(u32, usize, u64, u64)> = hw
        .gpus
        .iter()
        .map(|g| (g.index, g.running_processes.iter().filter(|p| !managed.contains(&(p.pid as i32))).count(), g.vram_free_mb, g.vram_total_mb))
        .collect();
    lines.extend(deny_hint(&busy));
    let ccs: Vec<(u32, &str)> = hw.gpus.iter().map(|g| (g.index, g.compute_capability.as_str())).collect();
    lines.extend(mixed_arch_note(&ccs));
    lines.extend(hw.unavailable.iter().map(|u| format!("{u} (not answering; skipped)")));
    if hw.gpus.is_empty() {
        let probe = crate::nvidia_driver::probe();
        let why = crate::nvidia_driver::diagnose(&probe)
            .or_else(|| crate::nvidia_driver::unstarted(&probe, &[]).map(|(_, w)| w))
            .unwrap_or_else(|| "no usable NVIDIA GPU".into());
        lines.push(why);
    }
    let status = if hw.gpus.is_empty() || unsupported == hw.gpus.len() {
        "FAIL"
    } else if unsupported > 0 || !hw.unavailable.is_empty() {
        "WARN"
    } else {
        "OK"
    };
    check("hardware", status, lines.join("\n"))
}

fn deny_hint(gpus: &[(u32, usize, u64, u64)]) -> Option<String> {
    // One line when a GPU has foreign processes or under a fifth of its memory free: how to keep a card for other workloads. (index, foreign processes, free MB, total MB)
    let (index, ..) = gpus.iter().find(|(_, foreign, free, total)| *foreign > 0 || *free * 5 < *total)?;
    Some(format!("goethe gpu deny gpu{index} keeps a card free for other workloads"))
}

fn mixed_arch_note(gpus: &[(u32, &str)]) -> Option<String> {
    // On a host with more than one supported architecture, say how Goethe groups the GPUs.
    let mut groups: Vec<(&str, Vec<u32>)> = Vec::new();
    for (index, cc) in gpus.iter().filter(|(_, cc)| crate::gpu_arch::is_supported(cc)) {
        match groups.iter_mut().find(|(c, _)| crate::gpu_arch::same_cc(c, cc)) {
            Some((_, members)) => members.push(*index),
            None => groups.push((cc, vec![*index])),
        }
    }
    (groups.len() > 1).then(|| {
        let parts: Vec<String> = groups
            .iter()
            .map(|(cc, g)| format!("SM {cc} on GPU {}", crate::placement::format_gpu_list(g)))
            .collect();
        format!("mixed architectures: {}; each model loads on GPUs of one architecture", parts.join(", "))
    })
}

fn check_glibc() -> Check {
    let out = std::process::Command::new("ldd").arg("--version").output();
    match out {
        Ok(o) if o.status.success() => {
            let line = String::from_utf8_lossy(&o.stdout);
            Check {
                name: "glibc".into(),
                status: "OK".into(),
                hint: line.lines().next().unwrap_or("unknown").to_string(),
            }
        }
        _ => Check {
            name: "glibc".into(),
            status: "WARN".into(),
            hint: "could not read glibc version".into(),
        },
    }
}

fn nginx_http2_fixed(v: (u32, u32, u32)) -> bool {
    // Stable 1.30.7+ or mainline 1.31.9+ (CVE-2026-123007 and related). Goethe enables HTTP/2.
    match v {
        (1, 30, p) => p >= 7,
        (1, 31, p) => p >= 9,
        (1, m, _) if m >= 32 => true,
        (maj, _, _) if maj > 1 => true,
        _ => false,
    }
}

fn check_nginx(home: &Path) -> Check {
    // Same lookup the gateway uses: NGINX_BIN, GOETHE_HOME/nginx-root, then PATH.
    let bin = crate::nginx::nginx_bin(home);
    if !bin.is_file() {
        return check(
            "nginx",
            "FAIL",
            "nginx not found: set NGINX_BIN, install nginx, or rerun the goethe installer (rootless nginx under GOETHE_HOME/nginx-root)",
        );
    }
    match crate::nginx::nginx_version(&bin) {
        Some(v) if !nginx_http2_fixed(v) => check(
            "nginx",
            "WARN",
            format!(
                "{} is nginx/{}.{}.{}; need 1.30.7+ or 1.31.9+ (HTTP/2 fixes). Re-run the installer or goethe update once the package ships that build",
                bin.display(),
                v.0,
                v.1,
                v.2
            ),
        ),
        Some(v) => check("nginx", "OK", format!("{} (nginx/{}.{}.{})", bin.display(), v.0, v.1, v.2)),
        None => check("nginx", "OK", bin.display().to_string()),
    }
}

fn check_goethe_home(home: &Path) -> Check {
    if is_writable(home) {
        Check {
            name: "goethe_home".into(),
            status: "OK".into(),
            hint: home.display().to_string(),
        }
    } else {
        Check {
            name: "goethe_home".into(),
            status: "FAIL".into(),
            hint: format!("{} not writable", home.display()),
        }
    }
}

fn check_disk(home: &Path) -> Check {
    // Free space on the models volume against the largest installed model (and a 50 GB floor).
    let models = home.join("models");
    let _ = fs::create_dir_all(&models);
    let gb = match available_gb(&models) {
        Ok(gb) => gb,
        Err(e) => return check("disk", "WARN", e),
    };
    let largest = crate::models::installed_ids(home)
        .into_iter()
        .filter_map(|id| crate::model_facts::weights_gb(&crate::paths::model_dir(home, &id)).ok().map(|s| (s, id)))
        .max_by(|a, b| a.0.total_cmp(&b.0));
    let need = largest.as_ref().map(|l| l.0).unwrap_or(0.0).max(50.0);
    let detail = largest
        .map(|(s, id)| format!("; largest installed model {id} is {s:.0} GB"))
        .unwrap_or_default();
    if gb >= need {
        check("disk", "OK", format!("{gb:.0} GB free for models{detail}"))
    } else {
        check("disk", "WARN", format!("only {gb:.0} GB free under models/{detail}; free space before installing more"))
    }
}

fn check_ram(home: &Path) -> Check {
    // MemAvailable against the largest host-RAM need of installed models (offloaded weights).
    let Some(avail) = crate::host::mem_available_gb() else {
        return check("ram", "WARN", "cannot read /proc/meminfo");
    };
    let worst = crate::models::installed_ids(home)
        .into_iter()
        .filter_map(|id| crate::models::host_ram_need_gb(home, &id).map(|gb| (gb, id)))
        .max_by(|a, b| a.0.total_cmp(&b.0));
    match worst {
        Some((need, id)) if avail < need.max(12.0) + 4.0 => check(
            "ram",
            "WARN",
            format!("{avail:.1} GB available; {id} offloads {need:.1} GB to host RAM and may fail to load"),
        ),
        _ if avail < 16.0 => check("ram", "WARN", format!("only {avail:.1} GB RAM available; engines need host memory while loading")),
        Some((need, id)) => check("ram", "OK", format!("{avail:.1} GB available; {id} needs {need:.1} GB pinned")),
        None => check("ram", "OK", format!("{avail:.1} GB available")),
    }
}

fn check_state_db(home: &Path) -> Check {
    // SQLite WAL needs shared memory that network filesystems do not provide reliably.
    let host = crate::host::probe_host(home, &crate::host::probe_environment(8443).cloud);
    match host.disks.iter().find(|d| d.role == "goethe_home") {
        Some(d) if d.network_fs => check(
            "state_db",
            "WARN",
            format!("state.db is on {} ({}); SQLite WAL is unsafe there, set GOETHE_HOME to a local disk", d.mount_point, d.fs_type),
        ),
        Some(d) => check("state_db", "OK", format!("{} on {}", d.fs_type, d.mount_point)),
        None => check("state_db", "WARN", "cannot inspect the GOETHE_HOME filesystem"),
    }
}

fn check_environment(home: &Path) -> Check {
    // How goethed should be kept running here: systemd units, or nohup/cron in containers.
    let env = crate::host::probe_environment(crate::nginx::public_port_u16(home));
    if crate::host::is_wsl() {
        let boot = if env.systemd_pid1 {
            "systemd is PID 1: goethe up installs a unit"
        } else {
            "no systemd: goethe up uses crontab @reboot; set systemd=true under [boot] in /etc/wsl.conf and run wsl --shutdown"
        };
        return check(
            "environment",
            "INFO",
            format!("WSL2: {boot}; the NVIDIA driver is on Windows (580 or newer), never inside WSL"),
        );
    }
    match (&env.container, env.systemd_pid1) {
        (Some(c), false) => {
            let volume = match &env.persistent_volume {
                Some(v) if home.starts_with(v) => format!("; GOETHE_HOME is on the persistent volume {v}"),
                Some(v) => format!("; GOETHE_HOME is not on the persistent volume {v} and is lost when the container is recreated"),
                None => String::new(),
            };
            check(
                "environment",
                "INFO",
                format!("container {c} without systemd: goethe up starts goethed with nohup; add goethed to the container start command{volume}"),
            )
        }
        (Some(c), true) => check("environment", "INFO", format!("container {c} with systemd: goethe up installs a unit")),
        (None, true) => check("environment", "OK", "systemd is PID 1: goethe up installs a unit"),
        (None, false) => check("environment", "INFO", "no systemd: goethe up uses crontab @reboot"),
    }
}

fn check_cloud(home: &Path) -> Check {
    // Provider-specific reachability hints for the public gateway port.
    let port = crate::nginx::public_port_u16(home);
    let env = crate::host::probe_environment(port);
    let access = crate::host::public_access_hint(&crate::nginx::public_bind(home));
    let hints = if env.public_endpoints.is_empty() {
        format!("; {access}")
    } else {
        format!("; endpoints: {}; {access}", env.public_endpoints.join(", "))
    };
    match env.cloud.as_str() {
        "none" => check("cloud", "INFO", "no cloud provider detected"),
        "runpod" => check(
            "cloud",
            "INFO",
            format!("RunPod: expose TCP port {port} in the pod settings (the HTTPS proxy terminates TLS, the gateway expects it){hints}"),
        ),
        "vast" => check("cloud", "INFO", format!("Vast.ai: map port {port} in the instance template (-p {port}:{port}){hints}")),
        other => check("cloud", "INFO", format!("{other}: open TCP {port} in the provider firewall or security group{hints}")),
    }
}

fn check_pyenv(home: &Path) -> Check {
    // The hermetic py-env's python and base layer, then SGLang and goethe-engine when installed, read from disk.
    let env = home.join("py-env");
    if !env.join("bin/python").is_file() {
        return check("py-env", "INFO", "not set up yet: the first goethe install of a model sets it up (about 30 MB, more for engines)");
    }
    let (python, sglang) = pyenv_versions(&env);
    if !pyenv::has(home, Layer::Base) {
        return check("py-env", "WARN", format!("python {python} but no huggingface_hub (the hf CLI); the next goethe install puts it back"));
    }
    let engine = match pyenv::engine_installed(home) {
        Some(v) if v == pyenv::ENGINE_VERSION => format!(", goethe-engine {v}"),
        Some(v) => format!(", goethe-engine {v} (goethe pins {}; it updates with the next kernel pack install)", pyenv::ENGINE_VERSION),
        None => String::new(),
    };
    match sglang.or_else(|| sglang_from_python(&env.join("bin/python"))) {
        Some(sglang) => check("py-env", "OK", format!("sglang {sglang}{engine}, python {python}")),
        None => check(
            "py-env",
            "OK",
            format!("python {python}{engine}; SGLang not installed yet; it installs with the first model that needs it"),
        ),
    }
}

fn sglang_from_python(py: &Path) -> Option<String> {
    // Ask the interpreter for the sglang distribution (metadata, then a spec lookup) without importing sglang.
    let code = "import importlib.metadata as m, importlib.util as u\ntry:\n    print(m.version('sglang'))\nexcept Exception:\n    print('installed' if u.find_spec('sglang') else '')";
    let out = std::process::Command::new(py)
        .args(["-c", code])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    let v = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (out.status.success() && !v.is_empty()).then_some(v)
}

fn pyenv_versions(env: &Path) -> (String, Option<String>) {
    // Python version from pyvenv.cfg and the sglang version from its dist-info METADATA.
    let cfg = fs::read_to_string(env.join("pyvenv.cfg")).unwrap_or_default();
    let python = cfg
        .lines()
        .filter_map(|l| l.split_once('='))
        .find(|(k, _)| matches!(k.trim(), "version" | "version_info"))
        .map_or_else(|| "unknown".to_string(), |(_, v)| v.trim().to_string());
    let sglang = fs::read_dir(env.join("lib"))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|d| fs::read_dir(d.path().join("site-packages")).ok())
        .flatten()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("sglang-") && e.file_name().to_string_lossy().ends_with(".dist-info"))
        .find_map(|e| {
            let meta = fs::read_to_string(e.path().join("METADATA")).ok()?;
            meta.lines().find_map(|l| l.strip_prefix("Version:")).map(|v| v.trim().to_string())
        });
    (python, sglang)
}

fn check_ports(home: &Path) -> Check {
    let mux = read_port(home, "nginx/mux_port").unwrap_or(30000);
    let public = read_port(home, "nginx/public_port").unwrap_or(8443);
    let hint = format!(
        "mux {mux} ({}), public {public} ({})",
        port_state(mux),
        port_state(public)
    );
    Check {
        name: "ports".into(),
        status: "OK".into(),
        hint,
    }
}

fn port_state(port: u16) -> &'static str {
    if TcpListener::bind(("127.0.0.1", port)).is_err() {
        "listening"
    } else {
        "free"
    }
}

fn check_goethed() -> Check {
    let sock = goethe_sock();
    if !sock.exists() {
        return Check {
            name: "goethed".into(),
            status: "WARN".into(),
            hint: "goethed not running; start goethed under GOETHE_HOME".into(),
        };
    }
    match rpc::call("ps", json!(null)) {
        Ok(r) if r.code == 0 => Check {
            name: "goethed".into(),
            status: "OK".into(),
            hint: sock.display().to_string(),
        },
        _ => Check {
            name: "goethed".into(),
            status: "WARN".into(),
            hint: "stale socket; goethed not answering RPC ps".into(),
        },
    }
}

fn check_gateway(home: &Path) -> Check {
    let port = read_port(home, "nginx/public_port").unwrap_or(8443);
    if TcpStream::connect(("127.0.0.1", port)).is_ok() {
        Check {
            name: "gateway".into(),
            status: "OK".into(),
            hint: format!("127.0.0.1:{port}"),
        }
    } else {
        Check {
            name: "gateway".into(),
            status: "WARN".into(),
            hint: "gateway not listening".into(),
        }
    }
}

fn check_network(home: &Path) -> Check {
    // When the gateway follows an interface: whether it has an address now, and whether the allowlist lets its clients in.
    let Some(name) = crate::nginx::bind_interface(home) else {
        return check("network", "OK", format!("gateway on {}", crate::nginx::public_bind(home)));
    };
    let Some(ip) = crate::nginx::interface_ipv4(&name) else {
        return check("network", "WARN", format!("{name} has no IPv4 address: only this machine reaches the API until it comes back"));
    };
    let routes = crate::netif::routes();
    let ranges = crate::netif::interfaces(&routes)
        .into_iter()
        .find(|i| i.name == name)
        .map(|i| crate::netif::reach(&i, &routes))
        .unwrap_or_default();
    let allowed: Vec<crate::netif::Net> = crate::security::allowlist_list().unwrap_or_default().iter().filter_map(|c| crate::netif::Net::parse(c)).collect();
    let shown = ranges.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", ");
    if !ranges.is_empty() && !ranges.iter().any(|r| allowed.iter().any(|a| a.overlaps(r))) {
        return check("network", "WARN", format!("{name} reaches {shown}, but no allowlist entry covers it, so its clients are refused; run goethe api expose vpn --interface {name}"));
    }
    check("network", "OK", format!("gateway follows {name} at {ip}"))
}

fn check_tls(home: &Path) -> Check {
    let cert = home.join("nginx/certs/server.crt");
    if !cert.is_file() {
        return Check {
            name: "tls".into(),
            status: "WARN".into(),
            hint: "no server.crt".into(),
        };
    }
    let openssl = |extra: &[&str]| {
        std::process::Command::new("openssl")
            .args(["x509", "-noout", "-in"])
            .arg(&cert)
            .args(extra)
            .stderr(Stdio::null())
            .output()
    };
    let end = match openssl(&["-enddate"]) {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => return check("tls", "WARN", "could not read certificate expiry"),
    };
    let valid_for = |secs: &str| openssl(&["-checkend", secs]).map(|o| o.status.success()).unwrap_or(false);
    if !valid_for("0") {
        return check("tls", "FAIL", format!("certificate expired ({end}); renew it or rerun the installer"));
    }
    if !valid_for("604800") {
        return check("tls", "WARN", format!("certificate expires within 7 days ({end})"));
    }
    check("tls", "OK", end)
}

fn check_hf(_home: &Path) -> Check {
    // Anonymous reachability of the Hub; goethe never logs in.
    if hf::offline_mode() {
        return check("huggingface", "INFO", "HF_HUB_OFFLINE=1");
    }
    let url = format!("{}/api/models?limit=1", hf::hf_endpoint());
    match crate::net::agent().get(&url).timeout(Duration::from_secs(3)).call() {
        Ok(_) | Err(ureq::Error::Status(..)) => check("huggingface", "OK", format!("reachable (anonymous; gated repos: {})", hf::GATED_WAYS)),
        Err(e) => check("huggingface", "WARN", format!("unreachable: {e}")),
    }
}

fn registry_warning(url: &str, e: &str, home: &Path) -> String {
    // An unreachable registry, and when the address comes from goethe.env, the line to delete to go back to the default.
    let from_file = ["GOETHE_REGISTRY_URL", "GOETHE_PKG_URL"].into_iter().find(|k| crate::paths::setting_from_file(k));
    match from_file {
        Some(key) => format!("{url} unreachable ({e}); it comes from {key} in {}: delete that line to use pkg.goethe.sh", home.join("goethe.env").display()),
        None => format!("{url} unreachable ({e}); check network access, or set GOETHE_REGISTRY_URL (file:// works)"),
    }
}

fn check_registry(home: &Path) -> Check {
    if hf::offline_mode() {
        let cache = home.join("registry.cache.json");
        if cache.is_file() {
            return Check {
                name: "registry".into(),
                status: "OK".into(),
                hint: "cached (offline)".into(),
            };
        }
    }
    match fetch(&registry_url()) {
        Ok(r) => Check {
            name: "registry".into(),
            status: "OK".into(),
            hint: format!("{} models", r.models.len()),
        },
        Err(_) if home.join("registry.cache.json").is_file() => Check {
            name: "registry".into(),
            status: "OK".into(),
            hint: "using registry.cache.json".into(),
        },
        Err(e) => Check {
            name: "registry".into(),
            status: "WARN".into(),
            hint: registry_warning(&registry_url(), &e, home),
        },
    }
}

fn read_port(home: &Path, rel: &str) -> Option<u16> {
    fs::read_to_string(home.join(rel))
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

fn is_writable(path: &Path) -> bool {
    let probe = path.join(format!(".goethe-probe-{}", std::process::id()));
    match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&probe)
    {
        Ok(f) => {
            drop(f);
            let _ = fs::remove_file(probe);
            true
        }
        Err(_) => false,
    }
}

fn available_gb(path: &Path) -> Result<f64, String> {
    use std::ffi::CString;
    let cpath = CString::new(path.to_str().ok_or("path")?).map_err(|e| e.to_string())?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(cpath.as_ptr(), &mut stat) };
    if rc != 0 {
        return Err("statvfs failed".into());
    }
    Ok((stat.f_bavail as u64 * stat.f_frsize as u64) as f64 / 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_mac_needs_apple_silicon_and_macos_14() {
        let (st, text) = mac_hardware_note("Apple M3 Max (40-core GPU)", true, "15.1", 48 * 1024, 40 * 1024, 64 * 1024);
        assert_eq!(st, "OK");
        assert!(text.contains("Apple M3 Max") && text.contains("macOS 15.1") && text.contains("GPU may use 48.0 of 64.0 GB") && text.contains("40.0 GB free"), "{text}");
        assert_eq!(mac_hardware_note("Intel Core i9", false, "14.0", 0, 0, 0).0, "FAIL");
        let (st, text) = mac_hardware_note("Apple M1", true, "13.6", 10_000, 8_000, 16_000);
        assert!(st == "FAIL" && text.contains("macOS 14 or newer"), "{text}");
    }

    #[test]
    fn a_mac_warns_only_when_no_installed_model_fits_now() {
        assert_eq!(mac_memory_warning(4_000, None), None, "nothing installed, nothing to warn about");
        assert_eq!(mac_memory_warning(4_000, Some((0.3, "small"))), None);
        let w = mac_memory_warning(4_000, Some((8.0, "big"))).unwrap();
        assert!(w.contains("big needs about 8.5 GB") && w.contains("3.9 GB is free"), "{w}");
    }

    #[test]
    fn plain_ok_rows_fold_into_one_line_unless_everything_is_asked_for() {
        let rows = vec![check("hardware", "OK", "x"), check("glibc", "OK", "y"), check("disk", "WARN", "low"), check("ram", "OK", "z")];
        let (shown, folded) = fold_ok(&rows, false);
        assert_eq!((shown.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), folded), (vec!["hardware", "disk"], vec!["glibc", "ram"]));
        let (all, none) = fold_ok(&rows, true);
        assert_eq!((all.len(), none.len()), (4, 0));
    }

    #[test]
    fn the_driver_and_the_platform_get_their_own_rows() {
        assert_eq!(driver_note("580.126.09").0, "OK");
        assert_eq!(driver_note("570.1").0, "WARN");
        assert_eq!(driver_note("525.60").0, "FAIL");
        assert_eq!(driver_note("").0, "WARN");
        use crate::platform_cli::Link;
        assert_eq!(platform_note(Link::NotLinked).0, "INFO");
        assert!(platform_note(Link::NotLinked).1.contains("goethe login"));
        assert_eq!(platform_note(Link::Linked { workspace: "w".into(), revoked: false, heartbeat: "OK".into(), ok: true }).0, "OK");
        assert_eq!(platform_note(Link::Linked { workspace: "w".into(), revoked: true, heartbeat: String::new(), ok: false }).0, "WARN");
        assert!(CHECKS.iter().any(|(n, _)| *n == "driver") && CHECKS.iter().any(|(n, _)| *n == "platform"));
    }

    #[test]
    fn a_busy_gpu_gets_one_hint_how_to_keep_it_free() {
        assert_eq!(deny_hint(&[(0, 0, 90_000, 96_000), (1, 0, 90_000, 96_000)]), None, "idle cards need no hint");
        assert_eq!(deny_hint(&[(0, 0, 90_000, 96_000), (1, 2, 90_000, 96_000)]).unwrap(), "goethe gpu deny gpu1 keeps a card free for other workloads");
        assert_eq!(deny_hint(&[(0, 0, 10_000, 96_000), (3, 1, 1, 96_000)]).unwrap(), "goethe gpu deny gpu0 keeps a card free for other workloads", "little free memory counts, one line");
    }

    #[test]
    fn the_hugging_face_check_uses_the_gated_wording_of_install() {
        assert!(hf::GATED_WAYS.contains("goethe install DIR --name ID"));
        assert!(hf::gated_guidance("org/m").contains(&hf::GATED_WAYS.replace("ID", "org/m")));
    }

    #[test]
    fn mixed_architecture_hosts_get_one_note() {
        let note = mixed_arch_note(&[(0, "12.0"), (1, "8.6"), (2, "12.0"), (3, "7.5")]).unwrap();
        assert_eq!(note, "mixed architectures: SM 12.0 on GPU 0,2, SM 8.6 on GPU 1; each model loads on GPUs of one architecture");
        assert_eq!(mixed_arch_note(&[(0, "9.0"), (1, "9.0"), (2, "7.5")]), None, "unsupported GPUs do not count");
    }

    #[test]
    fn a_missing_py_env_is_told_when_it_comes() {
        let home = tempfile::tempdir().unwrap();
        let c = check_pyenv(home.path());
        assert_eq!(c.status, "INFO", "the installer no longer sets it up");
        assert!(c.hint.contains("the first goethe install of a model sets it up"), "{}", c.hint);
    }

    #[test]
    fn nginx_http2_floor_matches_stable_and_mainline_fixes() {
        assert!(!nginx_http2_fixed((1, 30, 6)));
        assert!(nginx_http2_fixed((1, 30, 7)));
        assert!(!nginx_http2_fixed((1, 31, 8)));
        assert!(nginx_http2_fixed((1, 31, 9)));
        assert!(!nginx_http2_fixed((1, 29, 7)));
    }

    #[test]
    fn pyenv_versions_are_read_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let env = dir.path();
        assert_eq!(pyenv_versions(env), ("unknown".to_string(), None));
        fs::write(env.join("pyvenv.cfg"), "home = /usr/bin\nimplementation = CPython\nversion_info = 3.12.3\n").unwrap();
        let dist = env.join("lib/python3.12/site-packages/sglang-0.5.3.dist-info");
        fs::create_dir_all(&dist).unwrap();
        fs::write(dist.join("METADATA"), "Metadata-Version: 2.1\nName: sglang\nVersion: 0.5.3+goethe\n").unwrap();
        assert_eq!(pyenv_versions(env), ("3.12.3".to_string(), Some("0.5.3+goethe".to_string())));
        assert!(sglang_from_python(Path::new("/bin/echo")).is_some_and(|v| v.starts_with("-c import")));
        assert_eq!(sglang_from_python(Path::new("/bin/false")), None);
    }
}
