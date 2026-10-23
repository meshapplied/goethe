// Nginx gateway config generation and private instance control.

use crate::paths::goethe_home;
use std::io::Write;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpListener, TcpStream};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::os::unix::io::AsRawFd;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};
use std::{fs, io, thread};

const DEFAULT_MAX_BODY_MB: u64 = 10;
const RESPAWN_BACKOFF_MIN: Duration = Duration::from_secs(30);
const RESPAWN_BACKOFF_MAX: Duration = Duration::from_secs(600);

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

fn master_slot() -> MutexGuard<'static, Option<Child>> {
    static NGINX_MASTER: OnceLock<Mutex<Option<Child>>> = OnceLock::new();
    NGINX_MASTER
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
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

pub fn nginx_prefix(home: &Path) -> PathBuf {
    home.join("nginx")
}

pub fn write_private(path: &Path, data: &[u8]) -> Result<(), String> {
    // Atomically replace path with data at mode 0600: temp file in the same directory, fsync, rename.
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let dir = path.parent().ok_or_else(|| format!("{} has no parent", path.display()))?;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = dir.join(format!(".{name}.{}.{}.tmp", std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed)));
    let res = (|| -> io::Result<()> {
        let mut f = fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
        fs::rename(&tmp, path)
    })();
    if res.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    res.map_err(|e| format!("write {}: {e}", path.display()))
}

fn private_dir(path: &Path, worker_traversable: bool) -> Result<(), String> {
    // Create or tighten a directory to 0700; 0711 under root so nginx workers can reach their temp dirs.
    fs::create_dir_all(path).map_err(|e| format!("mkdir {}: {e}", path.display()))?;
    let mode = if worker_traversable && unsafe { libc::geteuid() } == 0 { 0o711 } else { 0o700 };
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).map_err(|e| format!("chmod {}: {e}", path.display()))
}

pub struct GatewayLock {
    _file: fs::File,
}

pub fn gateway_lock(home: &Path) -> Result<GatewayLock, String> {
    lock_gateway(home, true)?.ok_or_else(|| "gateway lock busy".to_string())
}

fn lock_gateway(home: &Path, wait: bool) -> Result<Option<GatewayLock>, String> {
    // flock(2) on nginx/.gateway.lock serializes config writes, reloads and spawns across goethe processes.
    let dir = nginx_prefix(home);
    private_dir(&dir, true)?;
    let file = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .open(dir.join(".gateway.lock"))
        .map_err(|e| format!("gateway lock: {e}"))?;
    let op = if wait { libc::LOCK_EX } else { libc::LOCK_EX | libc::LOCK_NB };
    loop {
        if unsafe { libc::flock(file.as_raw_fd(), op) } == 0 {
            return Ok(Some(GatewayLock { _file: file }));
        }
        let err = io::Error::last_os_error();
        match err.raw_os_error() {
            Some(libc::EINTR) => continue,
            Some(libc::EWOULDBLOCK) if !wait => return Ok(None),
            _ => return Err(format!("gateway lock: {err}")),
        }
    }
}

fn persisted_port(home: &Path, name: &str) -> Option<u16> {
    fs::read_to_string(nginx_prefix(home).join(name))
        .ok()?
        .trim()
        .parse()
        .ok()
        .filter(|p| *p != 0)
}

pub fn persisted_mux_port(home: &Path) -> Result<u16, String> {
    persisted_port(home, "mux_port")
        .ok_or_else(|| "mux port is not persisted under nginx/mux_port; start goethed first".to_string())
}

pub fn public_port_u16(home: &Path) -> u16 {
    persisted_port(home, "public_port").unwrap_or(8443)
}

pub fn max_body_mb(home: &Path) -> u64 {
    persisted_port(home, "max_body_mb").map(u64::from).unwrap_or(DEFAULT_MAX_BODY_MB)
}

pub fn persist_max_body_mb(home: &Path) -> u64 {
    // Daemon side: GOETHE_MAX_BODY_MB (1-1024, default 10) becomes the body limit shared by nginx and the mux.
    let mb = std::env::var("GOETHE_MAX_BODY_MB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|v| (1..=1024).contains(v))
        .unwrap_or(DEFAULT_MAX_BODY_MB);
    if let Err(e) = write_private(&nginx_prefix(home).join("max_body_mb"), format!("{mb}\n").as_bytes()) {
        eprintln!("goethe gateway: {e}");
    }
    mb
}

pub fn public_bind(home: &Path) -> String {
    if let Ok(con) = crate::db::open(home) {
        if let Ok(v) = con.query_row(
            "SELECT value FROM goethe_settings WHERE key = 'public_bind'",
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

pub fn set_public_bind(home: &Path, addr: &str) -> Result<String, String> {
    // Persist the listen address and apply it: hot reload when old and new sockets cannot clash, else a restart.
    let ip: IpAddr = addr
        .trim()
        .parse()
        .map_err(|_| format!("invalid address {addr:?}: use 0.0.0.0, 127.0.0.1 or an IP address of this host"))?;
    if !ip.is_unspecified() && !is_local_address(ip) {
        return Err(format!("{ip} is not an address of this host"));
    }
    let lock = gateway_lock(home)?;
    let previous = public_bind(home);
    store_public_bind(home, &ip.to_string())?;
    let Some(master) = verified_master(home) else {
        return Ok(format!("gateway bind set to {ip}; it applies when goethed starts the gateway"));
    };
    let listening = || format!("gateway listens on {ip} port {}", public_port_u16(home));
    let bin = proc_exe(master).filter(|p| p.is_file()).unwrap_or_else(|| nginx_bin(home));
    let prefix = nginx_prefix(home);
    let checked = write_configs(home, &bin).and_then(|_| nginx_test(home, &bin, &prefix, &prefix.join("nginx.conf")));
    if let Err(e) = checked {
        store_public_bind(home, &previous)?;
        let _ = write_configs(home, &bin);
        return Err(format!("{e}; the gateway bind stays {previous}"));
    }
    if !binds_clash(&previous, ip) {
        let before = worker_pids(master);
        apply_locked(home).map_err(|e| format!("{e}; the bind {ip} applies when the gateway restarts"))?;
        if previous == ip.to_string() || worker_pids(master).iter().any(|p| !before.contains(p)) {
            return Ok(listening());
        }
    }
    log_gateway(home, &format!("restarting nginx to move the listener from {previous} to {ip}"));
    stop_master(home);
    drop(lock);
    let deadline = Instant::now() + Duration::from_secs(25);
    while Instant::now() < deadline {
        if verified_master(home).is_some() && gateway_listening(home) {
            return Ok(format!("{} (restarted)", listening()));
        }
        thread::sleep(Duration::from_millis(250));
    }
    Err(format!(
        "the gateway did not come back on {ip} within 25 s; goethed restarts it (check goethe gateway status and goethe logs --gateway)"
    ))
}

fn is_local_address(ip: IpAddr) -> bool {
    // An interface of this host carries the address; a bind test alone passes everything under ip_nonlocal_bind.
    let mut ifap: *mut libc::ifaddrs = std::ptr::null_mut();
    if unsafe { libc::getifaddrs(&mut ifap) } != 0 {
        return TcpListener::bind((ip, 0)).is_ok();
    }
    let mut found = ip.is_loopback();
    let mut cur = ifap;
    while !cur.is_null() {
        let ifa = unsafe { &*cur };
        let addr = match unsafe { ifa.ifa_addr.as_ref() }.map(|a| i32::from(a.sa_family)) {
            Some(libc::AF_INET) => {
                let sin = unsafe { &*(ifa.ifa_addr as *const libc::sockaddr_in) };
                Some(IpAddr::V4(Ipv4Addr::from(u32::from_be(sin.sin_addr.s_addr))))
            }
            Some(libc::AF_INET6) => {
                let sin6 = unsafe { &*(ifa.ifa_addr as *const libc::sockaddr_in6) };
                Some(IpAddr::V6(Ipv6Addr::from(sin6.sin6_addr.s6_addr)))
            }
            _ => None,
        };
        found |= addr == Some(ip);
        cur = ifa.ifa_next;
    }
    unsafe { libc::freeifaddrs(ifap) };
    found
}

fn binds_clash(previous: &str, next: IpAddr) -> bool {
    // A wildcard listener and any other listener on the same port cannot coexist during a hot reload on Linux.
    previous.parse::<IpAddr>().map_or(true, |p| p != next && (p.is_unspecified() || next.is_unspecified()))
}

fn store_public_bind(home: &Path, value: &str) -> Result<(), String> {
    let con = crate::db::open(home).map_err(|e| e.to_string())?;
    con.execute(
        "INSERT OR REPLACE INTO goethe_settings (key, value) VALUES ('public_bind', ?1)",
        [value],
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}

pub fn gateway_connect_addr(home: &Path) -> SocketAddr {
    // Where local clients reach the gateway: loopback for a wildcard bind, else the bound address.
    let ip = match public_bind(home).parse::<IpAddr>() {
        Ok(IpAddr::V4(a)) if a.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        Ok(IpAddr::V6(a)) if a.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        Ok(ip) => ip,
        Err(_) => IpAddr::V4(Ipv4Addr::LOCALHOST),
    };
    SocketAddr::new(ip, public_port_u16(home))
}

fn quote(path: &Path) -> String {
    let s = path.display().to_string();
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn nginx_version(bin: &Path) -> Option<(u32, u32, u32)> {
    let out = Command::new(bin).arg("-v").output().ok()?;
    let text = String::from_utf8_lossy(&out.stderr).to_string();
    let v = text.split("nginx/").nth(1)?.split_whitespace().next()?.to_string();
    let mut it = v.split('.').map(|p| p.parse::<u32>().ok());
    Some((it.next()??, it.next()??, it.next().flatten().unwrap_or(0)))
}

fn render_gateway(home: &Path, bin: &Path) -> Result<String, String> {
    // Fill the template from the daemon's persisted ports only, never from this process's environment.
    let dir = nginx_prefix(home);
    let mux_port = persisted_mux_port(home)?;
    let bind_raw = public_bind(home);
    let bind: IpAddr = bind_raw
        .parse()
        .map_err(|_| format!("invalid public_bind setting {bind_raw:?}"))?;
    let port = public_port_u16(home);
    let listen = match bind {
        IpAddr::V4(a) => format!("{a}:{port}"),
        IpAddr::V6(a) => format!("[{a}]:{port}"),
    };
    let modern_http2 = nginx_version(bin).is_some_and(|v| v >= (1, 25, 1));
    const TMPL: &str = include_str!("../../../nginx/goethe_gateway.conf.tmpl");
    Ok(TMPL
        .replace("{{LISTEN}}", &listen)
        .replace("{{HTTP2_LISTEN}}", if modern_http2 { "" } else { " http2" })
        .replace("{{HTTP2_ON}}", if modern_http2 { "    http2 on;" } else { "" })
        .replace("{{CERT}}", &quote(&dir.join("certs/server.crt")))
        .replace("{{KEY}}", &quote(&dir.join("certs/server.key")))
        .replace("{{ALLOWLIST}}", &quote(&dir.join("allowlist.conf")))
        .replace("{{MAX_BODY_MB}}", &max_body_mb(home).to_string())
        .replace("{{MUX_PORT}}", &mux_port.to_string()))
}

fn write_configs(home: &Path, bin: &Path) -> Result<(), String> {
    // Render nginx.conf and the gateway include for this nginx binary; create the allowlist and key map once.
    let dir = nginx_prefix(home);
    private_dir(&dir, true)?;
    private_dir(&dir.join("logs"), true)?;
    let allow = dir.join("allowlist.conf");
    if !allow.is_file() {
        write_private(&allow, b"# managed by goethe allowlist\n")?;
    }
    let keys = dir.join("api_keys.conf");
    if !keys.is_file() {
        write_private(&keys, b"")?;
    }
    ensure_tls(home)?;
    let gateway = dir.join("goethe_gateway.conf");
    write_private(&gateway, render_gateway(home, bin)?.as_bytes())?;
    let main_conf = format!(
        "pid nginx.pid;\nerror_log logs/error.log warn;\n\
         events {{ worker_connections 4096; }}\nhttp {{\n\
         client_body_temp_path logs/b;\n\
         proxy_temp_path logs/p;\n\
         fastcgi_temp_path logs/f;\n\
         uwsgi_temp_path logs/u;\n\
         scgi_temp_path logs/s;\n\
         access_log logs/access.log;\n\
         map_hash_bucket_size 128;\n\
         include {};\n}}\n",
        quote(&gateway)
    );
    write_private(&dir.join("nginx.conf"), main_conf.as_bytes())?;
    let _ = fs::remove_file(dir.join("logs/a"));
    Ok(())
}

pub fn write_gateway_configs(home: &Path) -> Result<(), String> {
    let _lock = gateway_lock(home)?;
    write_configs(home, &active_bin(home))
}

fn ensure_tls(home: &Path) -> Result<(), String> {
    // Self-signed certificate on first use; the private key is always 0600 in a 0700 directory.
    let cert_dir = nginx_prefix(home).join("certs");
    private_dir(&cert_dir, false)?;
    let crt = cert_dir.join("server.crt");
    let key = cert_dir.join("server.key");
    if !(crt.is_file() && key.is_file()) {
        let status = Command::new("openssl")
            .args(["req", "-x509", "-newkey", "rsa:2048", "-keyout"])
            .arg(&key)
            .arg("-out")
            .arg(&crt)
            .args(["-days", "365", "-nodes", "-subj", "/CN=goethe-local"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| e.to_string())?;
        if !status.success() {
            return Err("openssl cert generation failed".to_string());
        }
    }
    fs::set_permissions(&key, fs::Permissions::from_mode(0o600)).map_err(|e| e.to_string())
}

pub fn format_gateway_line(home: &Path) -> Option<String> {
    // Only our own verified nginx counts; another service on the port must not look like the gateway.
    if gateway_down_reason().is_some() || verified_master(home).is_none() || !gateway_listening(home) {
        return None;
    }
    Some(format!("gateway: https://{}", gateway_connect_addr(home)))
}

pub fn gateway_listening(home: &Path) -> bool {
    TcpStream::connect_timeout(&gateway_connect_addr(home), Duration::from_millis(200)).is_ok()
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
    eprintln!("goethe gateway: {msg}");
}

pub fn log_gateway_error(home: &Path, msg: &str) {
    log_gateway(home, &format!("ERROR: {msg}"));
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

pub fn test_config(home: &Path) -> Result<(), String> {
    // nginx -t (by the binary that will load it) on the files as written; nothing to test before the first render.
    let bin = active_bin(home);
    let prefix = nginx_prefix(home);
    let conf = prefix.join("nginx.conf");
    if !bin.is_file() || !conf.is_file() {
        return Ok(());
    }
    nginx_test(home, &bin, &prefix, &conf)
}

fn wait_public_port(home: &Path, timeout: Duration) -> Result<(), String> {
    let addr = gateway_connect_addr(home);
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if TcpStream::connect_timeout(&addr, Duration::from_millis(100)).is_ok() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(50));
    }
    Err(format!("gateway did not listen on {addr} within {}s", timeout.as_secs()))
}

struct MasterRecord {
    pid: i32,
    start_ticks: u64,
    boot_id: String,
}

fn record_path(home: &Path) -> PathBuf {
    nginx_prefix(home).join("master.json")
}

fn read_record(home: &Path) -> Option<MasterRecord> {
    let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(record_path(home)).ok()?).ok()?;
    Some(MasterRecord {
        pid: i32::try_from(v.get("pid")?.as_i64()?).ok()?,
        start_ticks: v.get("start_ticks")?.as_u64()?,
        boot_id: v.get("boot_id")?.as_str()?.to_string(),
    })
}

fn boot_id() -> String {
    fs::read_to_string("/proc/sys/kernel/random/boot_id")
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn proc_stat(pid: i32) -> Option<(char, i32, u64)> {
    // (state, process group, start ticks) from /proc/<pid>/stat, parsed after the comm field.
    let text = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let rest = text.get(text.rfind(')')? + 2..)?;
    let f: Vec<&str> = rest.split_whitespace().collect();
    Some((f.first()?.chars().next()?, f.get(2)?.parse().ok()?, f.get(19)?.parse().ok()?))
}

fn proc_exe(pid: i32) -> Option<PathBuf> {
    // /proc/<pid>/exe without the " (deleted)" suffix an upgraded binary gets.
    let exe = fs::read_link(format!("/proc/{pid}/exe")).ok()?.to_string_lossy().to_string();
    Some(PathBuf::from(exe.strip_suffix(" (deleted)").unwrap_or(&exe)))
}

fn is_nginx_exe(pid: i32) -> bool {
    proc_exe(pid)
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().starts_with("nginx")))
        .unwrap_or(false)
}

fn active_bin(home: &Path) -> PathBuf {
    verified_master(home)
        .and_then(proc_exe)
        .filter(|p| p.is_file())
        .unwrap_or_else(|| nginx_bin(home))
}

fn cmdline_has_prefix(pid: i32, prefix: &Path) -> bool {
    let Ok(raw) = fs::read(format!("/proc/{pid}/cmdline")) else {
        return false;
    };
    let text = String::from_utf8_lossy(&raw).replace('\0', " ");
    format!("{} ", text.trim_end()).contains(&format!(" -p {} ", prefix.display()))
}

fn verified_master(home: &Path) -> Option<i32> {
    // Our live master: nginx exe, "-p <our prefix>" in its cmdline, and start ticks matching master.json.
    let prefix = nginx_prefix(home);
    let record = read_record(home);
    let from_file = fs::read_to_string(prefix.join("nginx.pid"))
        .ok()
        .and_then(|t| t.trim().parse::<i32>().ok());
    let boot = boot_id();
    [from_file, record.as_ref().map(|r| r.pid)]
        .into_iter()
        .flatten()
        .filter(|p| *p > 1)
        .find(|&pid| {
            let Some((state, _, start)) = proc_stat(pid) else {
                return false;
            };
            state != 'Z'
                && is_nginx_exe(pid)
                && cmdline_has_prefix(pid, &prefix)
                && record
                    .as_ref()
                    .is_none_or(|r| r.pid != pid || (r.start_ticks == start && r.boot_id == boot))
        })
}

fn terminate(pids: &[i32], still_ours: impl Fn(i32) -> bool, grace: Duration) {
    // TERM, wait for exit, then KILL only what is still verifiably ours.
    for &pid in pids {
        unsafe {
            libc::kill(pid, libc::SIGTERM);
        }
    }
    let deadline = Instant::now() + grace;
    while Instant::now() < deadline && pids.iter().any(|&p| still_ours(p)) {
        thread::sleep(Duration::from_millis(50));
    }
    for &pid in pids {
        if still_ours(pid) {
            unsafe {
                libc::kill(pid, libc::SIGKILL);
            }
        }
    }
}

fn orphan_workers(rec: &MasterRecord) -> Vec<i32> {
    // Live nginx processes still in a dead master's process group, started after it during this boot.
    let Ok(rd) = fs::read_dir("/proc") else {
        return Vec::new();
    };
    rd.flatten()
        .filter_map(|e| e.file_name().to_str()?.parse::<i32>().ok())
        .filter(|&pid| {
            proc_stat(pid).is_some_and(|(state, pgrp, start)| {
                state != 'Z' && pgrp == rec.pid && start >= rec.start_ticks
            }) && is_nginx_exe(pid)
        })
        .collect()
}

fn reap_orphans(home: &Path) {
    // Workers survive a SIGKILLed master and keep serving the old config on our port; stop them.
    let Some(rec) = read_record(home) else {
        return;
    };
    if rec.boot_id != boot_id() {
        let _ = fs::remove_file(record_path(home));
        return;
    }
    if verified_master(home) == Some(rec.pid) {
        return;
    }
    let orphans = orphan_workers(&rec);
    if !orphans.is_empty() {
        log_gateway(home, &format!("stopping orphaned nginx workers {orphans:?} of dead master {}", rec.pid));
        terminate(&orphans, |p| orphan_workers(&rec).contains(&p), Duration::from_secs(2));
    }
    if orphan_workers(&rec).is_empty() {
        let _ = fs::remove_file(record_path(home));
    }
}

fn spawn_on_keeper(cmd: Command) -> io::Result<Child> {
    // Spawn from one long-lived thread: PR_SET_PDEATHSIG fires when the spawning thread exits, not the process.
    type Job = (Command, mpsc::Sender<io::Result<Child>>);
    static KEEPER: OnceLock<mpsc::Sender<Job>> = OnceLock::new();
    let keeper = KEEPER.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<Job>();
        let _ = thread::Builder::new()
            .name("goethe-nginx-keeper".into())
            .spawn(move || {
                for (mut cmd, reply) in rx {
                    let _ = reply.send(cmd.spawn());
                }
            });
        tx
    });
    let gone = || io::Error::other("nginx keeper thread is gone");
    let (reply_tx, reply_rx) = mpsc::channel();
    keeper.send((cmd, reply_tx)).map_err(|_| gone())?;
    reply_rx.recv().map_err(|_| gone())?
}

fn child_alive() -> bool {
    master_slot()
        .as_mut()
        .is_some_and(|c| matches!(c.try_wait(), Ok(None)))
}

fn stop_master(home: &Path) {
    // SIGTERM is nginx's fast shutdown (the master stops its own workers); KILL what lingers, then clear orphans.
    let child = master_slot().as_ref().map(|c| c.id() as i32);
    let mut pids: Vec<i32> = verified_master(home).into_iter().collect();
    if let Some(c) = child.filter(|c| !pids.contains(c)) {
        pids.push(c);
    }
    if !pids.is_empty() {
        log_gateway(home, &format!("stopping nginx master {pids:?}"));
        let live = |p: i32| verified_master(home) == Some(p) || (Some(p) == child && child_alive());
        terminate(&pids, live, Duration::from_secs(2));
    }
    if let Some(mut old) = master_slot().take() {
        let _ = old.wait();
    }
    reap_orphans(home);
}

fn spawn_private(home: &Path) -> Result<(), String> {
    // Fresh master from written configs: stop any old master and orphans, nginx -t, spawn, record identity.
    stop_master(home);
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
    let _ = fs::remove_file(prefix.join("nginx.pid"));
    nginx_test(home, &bin, &prefix, &conf)?;
    let _ = fs::create_dir_all(home.join("logs"));
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
            libc::setpgid(0, 0);
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }
    let child = spawn_on_keeper(cmd).map_err(|e| {
        let msg = format!("nginx spawn failed: {e}");
        log_gateway(home, &msg);
        msg
    })?;
    let pid = child.id() as i32;
    let start_ticks = proc_stat(pid).map(|s| s.2).unwrap_or(0);
    master_slot().replace(child);
    let record = serde_json::json!({"pid": pid, "start_ticks": start_ticks, "boot_id": boot_id()});
    write_private(&record_path(home), record.to_string().as_bytes())?;
    let deadline = Instant::now() + Duration::from_secs(4);
    while verified_master(home) != Some(pid) && Instant::now() < deadline && child_alive() {
        thread::sleep(Duration::from_millis(50));
    }
    if verified_master(home) != Some(pid) {
        let bytes = fs::read(&err_log).unwrap_or_default();
        let tail = String::from_utf8_lossy(&bytes[bytes.len().saturating_sub(4096)..]);
        let msg = format!("nginx exited before writing its pid; stderr tail:\n{tail}");
        log_gateway(home, &msg);
        return Err(msg);
    }
    wait_public_port(home, Duration::from_secs(10)).inspect_err(|e| {
        log_gateway(home, e);
    })?;
    log_gateway(
        home,
        &format!("listening on {} (master pid {pid})", gateway_connect_addr(home)),
    );
    clear_gateway_down();
    Ok(())
}

fn worker_pids(master: i32) -> Vec<i32> {
    fs::read_to_string(format!("/proc/{master}/task/{master}/children"))
        .map(|s| s.split_whitespace().filter_map(|p| p.parse().ok()).collect())
        .unwrap_or_default()
}

fn wait_reloaded(master: i32, before: &[i32]) {
    // nginx lets old workers accept for ~100 ms after SIGHUP; wait until each is gone or titled "shutting down".
    if before.is_empty() {
        thread::sleep(Duration::from_millis(250));
        return;
    }
    let retired = |pid: &i32| {
        fs::read(format!("/proc/{pid}/cmdline"))
            .map_or(true, |c| c.is_empty() || String::from_utf8_lossy(&c).contains("shutting down"))
    };
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if worker_pids(master).iter().any(|p| !before.contains(p)) && before.iter().all(retired) {
            return;
        }
        thread::sleep(Duration::from_millis(10));
    }
}

pub fn apply_locked(home: &Path) -> Result<(), String> {
    // Re-render configs and hot-reload our verified master; the caller holds the gateway lock.
    let Some(pid) = verified_master(home) else {
        let why = gateway_down_reason().unwrap_or_else(|| "nginx is not running".to_string());
        return Err(format!("gateway is down: {why}"));
    };
    let bin = proc_exe(pid).filter(|p| p.is_file()).unwrap_or_else(|| nginx_bin(home));
    write_configs(home, &bin)?;
    nginx_test(home, &bin, &nginx_prefix(home), &nginx_prefix(home).join("nginx.conf"))?;
    let before = worker_pids(pid);
    if unsafe { libc::kill(pid, libc::SIGHUP) } != 0 {
        return Err(format!("gateway is down: reload signal failed: {}", io::Error::last_os_error()));
    }
    wait_reloaded(pid, &before);
    Ok(())
}

pub fn reload(home: &Path) -> Result<(), String> {
    let _lock = gateway_lock(home)?;
    apply_locked(home)
}

pub fn reopen_logs(home: &Path) -> bool {
    verified_master(home).is_some_and(|pid| unsafe { libc::kill(pid, libc::SIGUSR1) } == 0)
}

pub fn stop(home: &Path) {
    stop_master(home);
    let _ = fs::remove_file(record_path(home));
}

fn start_locked(home: &Path) -> Result<(), String> {
    // Render configs, then hot-reload our own live master or spawn a fresh one.
    write_configs(home, &nginx_bin(home))?;
    if child_alive() && verified_master(home).is_some() {
        return apply_locked(home);
    }
    spawn_private(home)
}

pub fn ensure_running() -> Result<(), String> {
    // Daemon start: bring the gateway up; a failure marks it down instead of aborting goethed.
    let home = goethe_home();
    match gateway_lock(&home).and_then(|_lock| start_locked(&home)) {
        Ok(()) => clear_gateway_down(),
        Err(e) => {
            log_gateway_error(&home, &e);
            set_gateway_down(e);
        }
    }
    Ok(())
}

pub fn ensure_gateway_running(home: &Path) -> Result<(), String> {
    // Watchdog hook: respawn nginx when our master died, with backoff; skips while another process holds the lock.
    static NEXT_TRY: Mutex<Option<(Instant, Duration)>> = Mutex::new(None);
    if verified_master(home).is_some() {
        return Ok(());
    }
    let down = || gateway_down_reason().unwrap_or_else(|| "nginx is not running".to_string());
    let mut next = NEXT_TRY.lock().unwrap_or_else(|e| e.into_inner());
    if next.is_some_and(|(at, _)| Instant::now() < at) {
        return Err(format!("gateway is down: {}", down()));
    }
    let Some(_lock) = lock_gateway(home, false)? else {
        return Ok(());
    };
    if verified_master(home).is_some() {
        return Ok(());
    }
    log_gateway(home, "nginx master is not running; respawning");
    let result = write_configs(home, &nginx_bin(home)).and_then(|()| spawn_private(home));
    match &result {
        Ok(()) => *next = None,
        Err(e) => {
            log_gateway_error(home, e);
            set_gateway_down(e.clone());
            let delay = next.map_or(RESPAWN_BACKOFF_MIN, |(_, d)| (d * 2).min(RESPAWN_BACKOFF_MAX));
            next.replace((Instant::now() + delay, delay));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_uses_persisted_ports_and_quotes_paths() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        fs::create_dir_all(home.join("nginx")).unwrap();
        let bin = Path::new("/nonexistent/nginx");
        assert!(render_gateway(home, bin).unwrap_err().contains("mux port"), "no persisted mux port, no config");
        fs::write(home.join("nginx/mux_port"), "30123\n").unwrap();
        fs::write(home.join("nginx/public_port"), "18444\n").unwrap();
        let conf = render_gateway(home, bin).unwrap();
        assert!(conf.contains("server 127.0.0.1:30123;"));
        assert!(conf.contains("listen 127.0.0.1:18444 ssl"));
        assert!(conf.contains(&format!("include \"{}\";", home.join("nginx/allowlist.conf").display())));
        assert!(conf.contains("client_max_body_size 10m;"));
        assert!(!conf.contains("{{"), "unfilled placeholder");
        assert!(!conf.contains(":30000"));
    }

    #[test]
    fn private_writes_are_atomic_and_0600() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.conf");
        write_private(&p, b"one").unwrap();
        write_private(&p, b"two").unwrap();
        assert_eq!(fs::read_to_string(&p).unwrap(), "two");
        assert_eq!(fs::metadata(&p).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1, "no temp files left behind");
        private_dir(&dir.path().join("d"), true).unwrap();
        let mode = fs::metadata(dir.path().join("d")).unwrap().permissions().mode() & 0o777;
        assert!(mode == 0o700 || mode == 0o711);
    }

    static REAL_NGINX: Mutex<()> = Mutex::new(());

    #[test]
    fn respawn_reaps_workers_of_a_killed_master() {
        let Some(bin) = std::env::var_os("NGINX_BIN").map(PathBuf::from).filter(|p| p.is_file()) else {
            return;
        };
        let _serial = REAL_NGINX.lock().unwrap_or_else(|e| e.into_inner());
        let free = || std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        fs::create_dir_all(home.join("nginx")).unwrap();
        fs::write(home.join("nginx/mux_port"), format!("{}\n", free())).unwrap();
        fs::write(home.join("nginx/public_port"), format!("{}\n", free())).unwrap();
        write_configs(home, &bin).unwrap();
        spawn_private(home).unwrap();
        let master = verified_master(home).expect("our master is verified");
        let workers = worker_pids(master);
        assert!(!workers.is_empty());
        unsafe {
            libc::kill(master, libc::SIGKILL);
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while verified_master(home).is_some() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
        }
        assert!(workers.iter().all(|&w| proc_stat(w).is_some_and(|s| s.0 != 'Z')), "workers outlive the master");
        ensure_gateway_running(home).unwrap();
        let respawned = verified_master(home).expect("respawned master");
        assert_ne!(respawned, master);
        assert!(workers.iter().all(|&w| proc_stat(w).is_none_or(|s| s.0 == 'Z')), "orphans were stopped");
        assert!(gateway_listening(home));
        stop(home);
        assert_eq!(verified_master(home), None);
    }

    #[test]
    fn bind_is_validated_and_persisted() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        assert!(set_public_bind(home, "nope").unwrap_err().contains("invalid address"));
        assert!(set_public_bind(home, "192.0.2.77").unwrap_err().contains("not an address of this host"));
        assert!(is_local_address(IpAddr::V4(Ipv4Addr::LOCALHOST)) && is_local_address("127.0.0.2".parse().unwrap()));
        assert!(set_public_bind(home, "0.0.0.0").unwrap().contains("applies when goethed starts"));
        assert_eq!(public_bind(home), "0.0.0.0");
        assert_eq!(gateway_connect_addr(home), "127.0.0.1:8443".parse::<SocketAddr>().unwrap());
        set_public_bind(home, "::").unwrap();
        assert_eq!(gateway_connect_addr(home), "[::1]:8443".parse::<SocketAddr>().unwrap());
        let ip = |s: &str| s.parse::<IpAddr>().unwrap();
        assert!(binds_clash("127.0.0.1", ip("0.0.0.0")));
        assert!(binds_clash("0.0.0.0", ip("127.0.0.1")));
        assert!(!binds_clash("0.0.0.0", ip("0.0.0.0")));
        assert!(!binds_clash("127.0.0.1", ip("127.0.0.2")));
    }

    #[test]
    fn bind_change_between_addresses_hot_reloads() {
        let Some(bin) = std::env::var_os("NGINX_BIN").map(PathBuf::from).filter(|p| p.is_file()) else {
            return;
        };
        let _serial = REAL_NGINX.lock().unwrap_or_else(|e| e.into_inner());
        let free = || std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        let port = free();
        fs::create_dir_all(home.join("nginx")).unwrap();
        fs::write(home.join("nginx/mux_port"), format!("{}\n", free())).unwrap();
        fs::write(home.join("nginx/public_port"), format!("{port}\n")).unwrap();
        write_configs(home, &bin).unwrap();
        spawn_private(home).unwrap();
        let master = verified_master(home).expect("our master is verified");
        let msg = set_public_bind(home, "127.0.0.2").unwrap();
        assert!(msg.contains(&format!("listens on 127.0.0.2 port {port}")), "{msg}");
        assert_eq!(verified_master(home), Some(master), "a hot reload keeps the master");
        let conf = fs::read_to_string(home.join("nginx/goethe_gateway.conf")).unwrap();
        assert!(conf.contains(&format!("listen 127.0.0.2:{port}")), "{conf}");
        assert!(TcpStream::connect(("127.0.0.2", port)).is_ok());
        assert!(TcpStream::connect(("127.0.0.1", port)).is_err());
        assert_eq!(format_gateway_line(home), Some(format!("gateway: https://127.0.0.2:{port}")));
        stop(home);
        assert_eq!(verified_master(home), None);
    }

    #[test]
    fn foreign_processes_are_never_verified() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        fs::create_dir_all(home.join("nginx")).unwrap();
        fs::write(home.join("nginx/nginx.pid"), format!("{}\n", std::process::id())).unwrap();
        assert_eq!(verified_master(home), None, "the test runner is not our nginx");
        fs::write(home.join("nginx/nginx.pid"), "1\n").unwrap();
        assert_eq!(verified_master(home), None);
        assert!(!cmdline_has_prefix(std::process::id() as i32, &home.join("nginx")));
        let (state, pgrp, start) = proc_stat(std::process::id() as i32).unwrap();
        assert!(state != 'Z' && pgrp > 0 && start > 0);
    }
}
