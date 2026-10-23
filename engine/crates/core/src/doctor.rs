// goethe doctor health checks for local install.

use crate::hf;
use crate::paths::{goethe_home, goethe_sock};
use crate::registry::{fetch, registry_url};
use crate::rpc;
use serde::Serialize;
use serde_json::json;
use std::fs;
use std::net::{TcpListener, TcpStream};
use std::process::{Command, Stdio};
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
        for c in &checks {
            println!("{}: {} - {}", c.name, c.status, c.hint);
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
    ("tls", check_tls),
    ("huggingface", check_hf),
    ("registry", check_registry),
    ("nvidia_driver", |_| check_nvidia_driver()),
];

fn collect_checks(home: &Path) -> Vec<Check> {
    // Run every check in parallel; each result carries the check's fixed name even on timeout.
    let pending: Vec<_> = CHECKS
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

fn check(name: &str, status: &str, hint: impl Into<String>) -> Check {
    Check {
        name: name.to_string(),
        status: status.to_string(),
        hint: hint.into(),
    }
}

fn check_nvidia_driver() -> Check {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output();
    let Ok(o) = out else {
        return Check {
            name: "nvidia_driver".into(),
            status: "WARN".into(),
            hint: "nvidia-smi not found".into(),
        };
    };
    if !o.status.success() {
        return Check {
            name: "nvidia_driver".into(),
            status: "WARN".into(),
            hint: "nvidia-smi failed".into(),
        };
    }
    let ver = String::from_utf8_lossy(&o.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    if ver.is_empty() {
        return Check {
            name: "nvidia_driver".into(),
            status: "WARN".into(),
            hint: "could not read driver version".into(),
        };
    }
    let major = ver.split('.').next().and_then(|p| p.parse::<u32>().ok());
    match major {
        Some(m) if m >= 550 => Check {
            name: "nvidia_driver".into(),
            status: "OK".into(),
            hint: ver,
        },
        Some(m) => Check {
            name: "nvidia_driver".into(),
            status: "FAIL".into(),
            hint: format!("driver {ver} (major {m}) below minimum 550"),
        },
        None => Check {
            name: "nvidia_driver".into(),
            status: "WARN".into(),
            hint: format!("unparsed driver version {ver}"),
        },
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
        "Public mode exposes the gateway to the internet. Keep the allowlist tight (goethe allowlist add/remove), use API keys with scopes and rpm limits, and rotate keys."
    );
}

fn check_hardware(home: &Path) -> Check {
    // NVML probe plus per-GPU support and foreign-process counts.
    let hw = match crate::hw::profile(home) {
        Ok(h) => h,
        Err(e) => return check("hardware", "FAIL", format!("{e}; install the NVIDIA driver and make sure nvidia-smi works")),
    };
    let managed = crate::db::open(home)
        .ok()
        .map(|con| crate::placement::managed_pids(&con))
        .unwrap_or_default();
    let mut parts = Vec::new();
    for g in &hw.gpus {
        let ok_sm = ["8.6", "12.0", "12.1"]
            .iter()
            .any(|sm| g.compute_capability.starts_with(sm));
        let foreign = g
            .running_processes
            .iter()
            .filter(|p| !managed.contains(&(p.pid as i32)))
            .count();
        let status_tag = if ok_sm { "OK" } else { "WARN" };
        let um = if g.unified_memory { " unified_memory" } else { "" };
        parts.push(format!(
            "gpu{} {status_tag} {} CC {} VRAM free/total {}/{} MB foreign_procs={foreign}{um}",
            g.index, g.name, g.compute_capability, g.vram_free_mb, g.vram_total_mb
        ));
    }
    check("hardware", "OK", format!("driver {} | gpus: {}", hw.driver, parts.join("; ")))
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

fn check_nginx(home: &Path) -> Check {
    // Same lookup the gateway uses: NGINX_BIN, GOETHE_HOME/nginx-root, then PATH.
    let bin = crate::nginx::nginx_bin(home);
    if bin.is_file() {
        return check("nginx", "OK", bin.display().to_string());
    }
    check(
        "nginx",
        "FAIL",
        "nginx not found: set NGINX_BIN, install nginx, or rerun the goethe installer (rootless nginx under GOETHE_HOME/nginx-root)",
    )
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
                format!("container {c} without systemd: goethe service enable starts goethed with nohup; add goethed to the container start command{volume}"),
            )
        }
        (Some(c), true) => check("environment", "INFO", format!("container {c} with systemd: goethe service enable installs a unit")),
        (None, true) => check("environment", "OK", "systemd is PID 1: goethe service enable installs a unit"),
        (None, false) => check("environment", "INFO", "no systemd: goethe service enable uses crontab @reboot"),
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
    // The hermetic py-env exists and imports sglang.
    let py = home.join("py-env/bin/python");
    if !py.is_file() {
        return check("py-env", "FAIL", "run the installer or goethe update --engine");
    }
    let out = std::process::Command::new(&py)
        .args(["-c", "import sglang; print(getattr(sglang, '__version__', 'ok'))"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output();
    match out {
        Ok(o) if o.status.success() => check("py-env", "OK", format!("sglang {}", String::from_utf8_lossy(&o.stdout).trim())),
        _ => check("py-env", "FAIL", "py-env present but sglang import failed; run goethe update --engine"),
    }
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
        Ok(_) | Err(ureq::Error::Status(..)) => check("huggingface", "OK", "reachable (anonymous; gated repos need a manual download)"),
        Err(e) => check("huggingface", "WARN", format!("unreachable: {e}")),
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
            hint: format!("unreachable: {e}"),
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
