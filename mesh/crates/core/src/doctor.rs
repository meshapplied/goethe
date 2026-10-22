// mesh doctor health checks for local install.

use crate::hf;
use crate::paths::{mesh_home, mesh_sock};
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
    let home = mesh_home();
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

fn collect_checks(home: &Path) -> Vec<Check> {
    let home2 = home.to_path_buf();
    let home3 = home.to_path_buf();
    let home4 = home.to_path_buf();
    let home5 = home.to_path_buf();
    let home6 = home.to_path_buf();
    let home7 = home.to_path_buf();
    let home8 = home.to_path_buf();
    let home9 = home.to_path_buf();
    let home10 = home.to_path_buf();
    let r1 = spawn_check(move || check_hardware(&home2));
    let r3 = spawn_check(check_glibc);
    let r4 = spawn_check(move || check_nginx(&home3));
    let r5 = spawn_check(move || check_mesh_home(&home4));
    let r6 = spawn_check(move || check_disk(&home5));
    let r7 = spawn_check(move || check_pyenv(&home6));
    let r8 = spawn_check(move || check_ports(&home7));
    let r9 = spawn_check(check_meshd);
    let r10 = spawn_check(move || check_gateway(&home8));
    let r11 = spawn_check(move || check_tls(&home9));
    let r12 = spawn_check(move || check_hf(&home10));
    let home11 = home.to_path_buf();
    let r13 = spawn_check(move || check_registry(&home11));
    let r14 = spawn_check(check_nvidia_driver);
    vec![
        recv_check(r1, "hardware"),
        recv_check(r3, "glibc"),
        recv_check(r4, "nginx"),
        recv_check(r5, "mesh_home"),
        recv_check(r6, "disk"),
        recv_check(r7, "py-env"),
        recv_check(r8, "ports"),
        recv_check(r9, "meshd"),
        recv_check(r10, "gateway"),
        recv_check(r11, "tls"),
        recv_check(r12, "huggingface"),
        recv_check(r13, "registry"),
        recv_check(r14, "nvidia_driver"),
    ]
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
    if bind == "127.0.0.1" {
        println!("gateway is loopback only");
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
    if let Ok(canary) = std::env::var("MESH_CANARY_URL") {
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
            println!("reachability unknown: set MESH_CANARY_URL to an external TCP probe");
        }
    } else {
        println!("reachability unknown: set MESH_CANARY_URL to an external TCP probe");
    }
    print_public_security_advisory();
}

fn print_public_security_advisory() {
    println!(
        "Public mode exposes the gateway to the internet. Keep the allowlist tight (mesh allowlist add/remove), use API keys with scopes and rpm limits, and rotate keys."
    );
}

fn check_hardware(home: &Path) -> Check {
    let hw = match crate::hw::profile(home) {
        Ok(h) => h,
        Err(e) => {
            return Check {
                name: "nvml".into(),
                status: "FAIL".into(),
                hint: format!("{e}; install NVIDIA driver and ensure nvidia-smi works"),
            };
        }
    };
    let nvml = Check {
        name: "nvml".into(),
        status: "OK".into(),
        hint: format!("driver {}", hw.driver),
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
    let gpu_hint = parts.join("; ");
    Check {
        name: "hardware".into(),
        status: if nvml.status == "FAIL" {
            "FAIL".into()
        } else {
            "OK".into()
        },
        hint: format!("{} | gpus: {}", nvml.hint, gpu_hint),
    }
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
    let bin = home.join("nginx-root/usr/sbin/nginx");
    if bin.is_file() {
        return Check {
            name: "nginx".into(),
            status: "OK".into(),
            hint: bin.display().to_string(),
        };
    }
    Check {
        name: "nginx".into(),
        status: "FAIL".into(),
        hint: "run mesh installer or place nginx under MESH_HOME/nginx-root".into(),
    }
}

fn check_mesh_home(home: &Path) -> Check {
    if is_writable(home) {
        Check {
            name: "mesh_home".into(),
            status: "OK".into(),
            hint: home.display().to_string(),
        }
    } else {
        Check {
            name: "mesh_home".into(),
            status: "FAIL".into(),
            hint: format!("{} not writable", home.display()),
        }
    }
}

fn check_disk(home: &Path) -> Check {
    let models = home.join("models");
    let _ = fs::create_dir_all(&models);
    match available_gb(&models) {
        Ok(gb) if gb >= 50.0 => Check {
            name: "disk".into(),
            status: "OK".into(),
            hint: format!("{gb:.0} GB free for models"),
        },
        Ok(gb) => Check {
            name: "disk".into(),
            status: "WARN".into(),
            hint: format!("only {gb:.0} GB free under models/"),
        },
        Err(e) => Check {
            name: "disk".into(),
            status: "WARN".into(),
            hint: e,
        },
    }
}

fn check_pyenv(home: &Path) -> Check {
    let py = home.join("py-env/bin/python");
    if !py.is_file() {
        return Check {
            name: "py-env".into(),
            status: "FAIL".into(),
            hint: "run installer or mesh update --engine".into(),
        };
    }
    let out = std::process::Command::new(&py)
        .args(["-c", "import sglang; print(getattr(sglang, '__version__', 'ok'))"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output();
    match out {
        Ok(o) if o.status.success() => Check {
            name: "sglang".into(),
            status: "OK".into(),
            hint: format!(
                "sglang {}",
                String::from_utf8_lossy(&o.stdout).trim()
            ),
        },
        _ => Check {
            name: "sglang".into(),
            status: "FAIL".into(),
            hint: "py-env present but sglang import failed; run mesh update --engine".into(),
        },
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

fn check_meshd() -> Check {
    let sock = mesh_sock();
    if !sock.exists() {
        return Check {
            name: "meshd".into(),
            status: "WARN".into(),
            hint: "meshd not running; start meshd under MESH_HOME".into(),
        };
    }
    match rpc::call("ps", json!(null)) {
        Ok(r) if r.code == 0 => Check {
            name: "meshd".into(),
            status: "OK".into(),
            hint: sock.display().to_string(),
        },
        _ => Check {
            name: "meshd".into(),
            status: "WARN".into(),
            hint: "stale socket; meshd not answering RPC ps".into(),
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
    let out = std::process::Command::new("openssl")
        .args(["x509", "-enddate", "-noout", "-in"])
        .arg(&cert)
        .output();
    match out {
        Ok(o) if o.status.success() => Check {
            name: "tls".into(),
            status: "OK".into(),
            hint: String::from_utf8_lossy(&o.stdout).trim().to_string(),
        },
        _ => Check {
            name: "tls".into(),
            status: "WARN".into(),
            hint: "could not read certificate expiry".into(),
        },
    }
}

fn check_hf(home: &Path) -> Check {
    if hf::offline_mode() {
        return Check {
            name: "huggingface".into(),
            status: "INFO".into(),
            hint: "HF_HUB_OFFLINE=1".into(),
        };
    }
    let endpoint = hf::hf_endpoint();
    let url = format!("{endpoint}/api/whoami-v2");
    let token = hf::resolve_token(home);
    let req = crate::net::agent()
        .get(&url)
        .timeout(Duration::from_secs(3));
    let req = if let Some(t) = token.as_ref() {
        req.set("Authorization", &format!("Bearer {t}"))
    } else {
        req
    };
    match req.call() {
        Ok(r) if r.status() == 200 => Check {
            name: "huggingface".into(),
            status: "OK".into(),
            hint: "reachable, logged in".into(),
        },
        Ok(r) if r.status() == 401 && token.is_none() => Check {
            name: "huggingface".into(),
            status: "OK".into(),
            hint: "reachable, not logged in".into(),
        },
        Ok(r) if r.status() == 401 || r.status() == 403 => Check {
            name: "huggingface".into(),
            status: "WARN".into(),
            hint: format!("reachable but token rejected (HTTP {})", r.status()),
        },
        Ok(r) => Check {
            name: "huggingface".into(),
            status: "INFO".into(),
            hint: format!("HTTP {}", r.status()),
        },
        Err(e) => Check {
            name: "huggingface".into(),
            status: "WARN".into(),
            hint: format!("unreachable: {e}"),
        },
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
    let probe = path.join(format!(".mesh-probe-{}", std::process::id()));
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
