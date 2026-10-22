// Nginx gateway config generation and private instance control.

use crate::paths::mesh_home;
use std::io::Write;
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use std::{fs, thread};

static NGINX_MASTER: OnceLock<Mutex<Option<std::process::Child>>> = OnceLock::new();
static GATEWAY_DOWN: OnceLock<Mutex<Option<String>>> = OnceLock::new();

fn down_slot() -> &'static Mutex<Option<String>> {
    GATEWAY_DOWN.get_or_init(|| Mutex::new(None))
}

pub fn set_gateway_down(reason: String) {
    if let Ok(mut g) = down_slot().lock() {
        let _ = g.replace(reason);
    }
}

pub fn clear_gateway_down() {
    if let Ok(mut g) = down_slot().lock() {
        let _ = g.take();
    }
}

pub fn gateway_down_reason() -> Option<String> {
    down_slot().lock().ok().and_then(|g| g.clone())
}

pub fn require_gateway() -> Result<(), String> {
    if let Some(r) = gateway_down_reason() {
        return Err(format!("gateway is down: {r}"));
    }
    let home = mesh_home();
    if !gateway_listening(&home) && !pid_alive(&home) {
        return Err("gateway is down: not listening on public port".to_string());
    }
    Ok(())
}

fn master_slot() -> &'static Mutex<Option<std::process::Child>> {
    NGINX_MASTER.get_or_init(|| Mutex::new(None))
}

pub fn nginx_bin(home: &Path) -> PathBuf {
    if let Ok(v) = std::env::var("NGINX_BIN") {
        let p = PathBuf::from(v);
        if p.is_file() {
            return p;
        }
    }
    let rootless = home.join("nginx-root/usr/sbin/nginx");
    if rootless.is_file() {
        return rootless;
    }
    let path_dirs = std::env::var("PATH").unwrap_or_default();
    let fixed = ["/usr/sbin", "/usr/local/sbin", "/usr/local/nginx/sbin", "/sbin"];
    for dir in path_dirs.split(':').chain(fixed) {
        let candidate = Path::new(dir).join("nginx");
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from("nginx")
}

pub fn write_gateway_configs(home: &Path) -> Result<(), String> {
    let nginx_dir = home.join("nginx");
    fs::create_dir_all(nginx_dir.join("certs")).map_err(|e| e.to_string())?;
    let allow = nginx_dir.join("allowlist.conf");
    if !allow.is_file() {
        fs::write(&allow, "# managed by mesh allowlist\n").map_err(|e| e.to_string())?;
    }
    let keys = nginx_dir.join("api_keys.conf");
    if !keys.is_file() {
        fs::write(&keys, "").map_err(|e| e.to_string())?;
        let _ = fs::set_permissions(&keys, std::os::unix::fs::PermissionsExt::from_mode(0o600));
    }
    ensure_tls(home)?;
    fs::create_dir_all(nginx_dir.join("logs")).map_err(|e| e.to_string())?;
    let mesh_conf = render_gateway(home);
    fs::write(nginx_dir.join("mesh_gateway.conf"), mesh_conf).map_err(|e| e.to_string())?;
    let main_conf = format!(
        "pid nginx.pid;\nerror_log logs/error.log warn;\n\
         events {{ worker_connections 1024; }}\nhttp {{\n\
         client_body_temp_path logs/b;\n\
         proxy_temp_path logs/p;\n\
         fastcgi_temp_path logs/f;\n\
         uwsgi_temp_path logs/u;\n\
         scgi_temp_path logs/s;\n\
         access_log logs/a;\n\
         map_hash_bucket_size 128;\n\
         include {};\n}}\n",
        nginx_dir.join("mesh_gateway.conf").display()
    );
    fs::write(nginx_dir.join("nginx.conf"), main_conf).map_err(|e| e.to_string())?;
    Ok(())
}

fn render_gateway(home: &Path) -> String {
    let nginx_dir = home.join("nginx");
    let cert = nginx_dir.join("certs/server.crt");
    let key = nginx_dir.join("certs/server.key");
    let allow = nginx_dir.join("allowlist.conf");
    let public_port = public_port(home);
    let public_bind = public_bind(home);
    const TMPL: &str = include_str!("../../../nginx/mesh_gateway.conf.tmpl");
    TMPL.replace("{{CERT}}", &cert.display().to_string())
        .replace("{{KEY}}", &key.display().to_string())
        .replace("{{ALLOWLIST}}", &allow.display().to_string())
        .replace("{{PUBLIC_BIND}}", &public_bind)
        .replace("{{PUBLIC_PORT}}", &public_port)
        .replace("{{MUX_PORT}}", &crate::paths::mesh_mux_port().to_string())
}

pub fn ensure_tls(home: &Path) -> Result<(), String> {
    let cert_dir = home.join("nginx/certs");
    let crt = cert_dir.join("server.crt");
    let key = cert_dir.join("server.key");
    if crt.is_file() && key.is_file() {
        return Ok(());
    }
    let status = Command::new("openssl")
        .args([
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-keyout",
            key.to_str().unwrap(),
            "-out",
            crt.to_str().unwrap(),
            "-days",
            "365",
            "-nodes",
            "-subj",
            "/CN=mesh-local",
        ])
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("openssl cert generation failed".to_string());
    }
    Ok(())
}

pub fn nginx_prefix(home: &Path) -> PathBuf {
    home.join("nginx")
}

pub fn public_bind(home: &Path) -> String {
    if let Ok(con) = crate::db::open(home) {
        if let Ok(v) = con.query_row(
            "SELECT value FROM mesh_settings WHERE key = 'public_bind'",
            [],
            |r| r.get::<_, String>(0),
        ) {
            let t = v.trim();
            if !t.is_empty() {
                return t.to_string();
            }
        }
    }
    "127.0.0.1".to_string()
}

fn public_port(home: &Path) -> String {
    if let Ok(v) = std::env::var("MESH_PUBLIC_PORT") {
        if !v.is_empty() {
            let _ = fs::write(home.join("nginx/public_port"), &v);
            return v;
        }
    }
    if let Ok(v) = fs::read_to_string(home.join("nginx/public_port")) {
        let t = v.trim().to_string();
        if !t.is_empty() {
            return t;
        }
    }
    "8443".to_string()
}

pub fn public_port_u16(home: &Path) -> u16 {
    public_port(home).parse().unwrap_or(8443)
}

pub fn format_gateway_line(home: &Path) -> Option<String> {
    if gateway_down_reason().is_some() {
        return None;
    }
    if !gateway_listening(home) {
        return None;
    }
    Some(format!(
        "gateway: https://127.0.0.1:{}",
        public_port_u16(home)
    ))
}

pub fn gateway_listening(home: &Path) -> bool {
    let addr: SocketAddr = format!("127.0.0.1:{}", public_port_u16(home))
        .parse()
        .unwrap_or_else(|_| "127.0.0.1:8443".parse().unwrap());
    TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok()
}

pub fn log_gateway(home: &Path, msg: &str) {
    let _ = fs::create_dir_all(home.join("logs"));
    let path = home.join("logs/gateway.log");
    let ts = chrono::Local::now().format("%Y-%m-%dT%H:%M:%S");
    let line = format!("[{ts}] {msg}\n");
    if let Ok(mut f) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = f.write_all(line.as_bytes());
    }
    eprintln!("mesh gateway: {msg}");
}

pub fn log_gateway_error(home: &Path, msg: &str) {
    log_gateway(home, &format!("ERROR: {msg}"));
    eprintln!("mesh gateway ERROR: {msg}");
}

pub fn start_private(home: &Path) -> Result<(), String> {
    write_gateway_configs(home)?;
    if pid_alive(home) && gateway_listening(home) {
        return reload(home);
    }
    spawn_private(home)
}

fn reap_stale_master(home: &Path) {
    let pid_path = nginx_prefix(home).join("nginx.pid");
    let Ok(text) = fs::read_to_string(&pid_path) else {
        return;
    };
    let Ok(pid) = text.trim().parse::<i32>() else {
        let _ = fs::remove_file(&pid_path);
        return;
    };
    if unsafe { libc::kill(pid, 0) != 0 } {
        let _ = fs::remove_file(&pid_path);
        return;
    }
    log_gateway(home, &format!("reaping stale nginx master pid {pid}"));
    unsafe {
        libc::kill(pid, libc::SIGTERM);
    }
    for _ in 0..40 {
        if unsafe { libc::kill(pid, 0) != 0 } {
            break;
        }
        thread::sleep(Duration::from_millis(50));
    }
    if unsafe { libc::kill(pid, 0) == 0 } {
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
    }
    let _ = fs::remove_file(&pid_path);
}

fn nginx_test(home: &Path, bin: &Path, prefix: &Path, conf: &Path) -> Result<(), String> {
    let out = Command::new(bin)
        .arg("-t")
        .arg("-p")
        .arg(prefix)
        .arg("-c")
        .arg(conf)
        .output()
        .map_err(|e| format!("nginx -t exec failed: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    let detail = String::from_utf8_lossy(&out.stderr);
    log_gateway(home, &format!("nginx -t failed:\n{detail}"));
    Err(format!("nginx -t failed: {detail}"))
}

fn wait_public_port(home: &Path, timeout: Duration) -> Result<(), String> {
    let port = public_port_u16(home);
    let addr: SocketAddr = format!("127.0.0.1:{port}")
        .parse()
        .map_err(|e: std::net::AddrParseError| e.to_string())?;
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if TcpStream::connect_timeout(&addr, Duration::from_millis(100)).is_ok() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(50));
    }
    Err(format!(
        "gateway did not listen on 127.0.0.1:{port} within {}s",
        timeout.as_secs()
    ))
}

fn spawn_private(home: &Path) -> Result<(), String> {
    reap_stale_master(home);
    if let Ok(mut guard) = master_slot().lock() {
        if let Some(child) = guard.as_mut() {
            if child.try_wait().ok().flatten().is_none() {
                return Ok(());
            }
            let _ = guard.take();
        }
    }

    let bin = nginx_bin(home);
    if !bin.is_file() {
        let msg = format!(
            "nginx binary not found at {}; install nginx or set NGINX_BIN",
            bin.display()
        );
        log_gateway(home, &msg);
        return Err(msg);
    }

    let prefix = nginx_prefix(home);
    let conf = prefix.join("nginx.conf");
    let pid_path = prefix.join("nginx.pid");
    let _ = fs::remove_file(&pid_path);

    nginx_test(home, &bin, &prefix, &conf)?;

    let err_log = home.join("logs/nginx-stderr.log");
    let err_file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&err_log)
        .map_err(|e| e.to_string())?;

    let mut cmd = Command::new(&bin);
    cmd.arg("-p")
        .arg(&prefix)
        .arg("-c")
        .arg(&conf)
        .arg("-g")
        .arg("daemon off;")
        .stdout(Stdio::null())
        .stderr(Stdio::from(err_file));

    unsafe {
        cmd.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }

    let child = cmd.spawn().map_err(|e| {
        let msg = format!("nginx spawn failed: {e}");
        log_gateway(home, &msg);
        msg
    })?;

    if let Ok(mut guard) = master_slot().lock() {
        let _ = guard.insert(child);
    }

    for _ in 0..80 {
        if pid_alive(home) {
            break;
        }
        thread::sleep(Duration::from_millis(50));
    }
    if !pid_alive(home) {
        let tail = fs::read_to_string(&err_log).unwrap_or_default();
        let msg = format!("nginx exited before writing pid; stderr tail:\n{tail}");
        log_gateway(home, &msg);
        return Err(msg);
    }

    wait_public_port(home, Duration::from_secs(10)).inspect_err(|e| {
        log_gateway(home, e);
    })?;

    log_gateway(
        home,
        &format!(
            "listening on 127.0.0.1:{} (pid file {})",
            public_port_u16(home),
            pid_path.display()
        ),
    );
    clear_gateway_down();
    Ok(())
}

pub fn reload(home: &Path) -> Result<(), String> {
    require_gateway()?;
    write_gateway_configs(home)?;
    let bin = nginx_bin(home);
    if !bin.is_file() {
        return Err(format!("nginx binary not found at {}", bin.display()));
    }
    let prefix = nginx_prefix(home);
    let conf = prefix.join("nginx.conf");
    nginx_test(home, &bin, &prefix, &conf)?;
    if !pid_alive(home) {
        return spawn_private(home);
    }
    let status = Command::new(&bin)
        .arg("-s")
        .arg("reload")
        .arg("-p")
        .arg(&prefix)
        .arg("-c")
        .arg(&conf)
        .status()
        .map_err(|e| e.to_string())?;
    if status.success() {
        return Ok(());
    }
    log_gateway(home, "nginx reload failed; respawning gateway");
    let _ = fs::remove_file(prefix.join("nginx.pid"));
    if let Ok(mut guard) = master_slot().lock() {
        let _ = guard.take();
    }
    spawn_private(home)
}

pub fn stop(home: &Path) {
    let prefix = nginx_prefix(home);
    let conf = prefix.join("nginx.conf");
    let bin = nginx_bin(home);
    if prefix.join("nginx.pid").is_file() && bin.is_file() {
        let _ = Command::new(&bin)
            .arg("-s")
            .arg("quit")
            .arg("-p")
            .arg(&prefix)
            .arg("-c")
            .arg(&conf)
            .status();
    }
    for _ in 0..40 {
        if !pid_alive(home) {
            break;
        }
        thread::sleep(Duration::from_millis(50));
    }
    if let Ok(mut guard) = master_slot().lock() {
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
    reap_stale_master(home);
}

fn pid_alive(home: &Path) -> bool {
    let pid_path = nginx_prefix(home).join("nginx.pid");
    let Ok(text) = fs::read_to_string(pid_path) else {
        return false;
    };
    let Ok(pid) = text.trim().parse::<i32>() else {
        return false;
    };
    unsafe { libc::kill(pid, 0) == 0 }
}

pub fn try_system_symlink(home: &Path) -> Result<(), String> {
    let etc = Path::new("/etc/nginx/conf.d");
    if !etc.is_dir() || !etc.metadata().map(|m| m.permissions().readonly()).unwrap_or(true) {
        return Ok(());
    }
    let src = home.join("nginx/mesh_gateway.conf");
    let dst = etc.join("mesh_gateway.conf");
    if dst.exists() {
        return Ok(());
    }
    std::os::unix::fs::symlink(&src, &dst).map_err(|e| e.to_string())
}

pub fn ensure_running() -> Result<(), String> {
    let home = mesh_home();
    let _ = try_system_symlink(&home);
    match start_private(&home) {
        Ok(()) => {
            clear_gateway_down();
            Ok(())
        }
        Err(e) => {
            log_gateway_error(&home, &e);
            set_gateway_down(e);
            Ok(())
        }
    }
}
