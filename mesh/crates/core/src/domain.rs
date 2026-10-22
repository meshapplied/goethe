// Custom domain TLS via ACME and public gateway bind.

use crate::db;
use crate::nginx;
use crate::paths::mesh_home;
use crate::security;
use std::fs;
use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, TcpListener, ToSocketAddrs};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

pub fn add(domain: &str) -> Result<String, String> {
    validate_domain(domain)?;
    let home = mesh_home();
    let _ = crate::paths::ensure_layout(&home);
    let public_ip = public_ipv4()?;
    check_dns(domain, &public_ip)?;
    if Command::new("certbot").arg("--version").output().is_err() {
        return Err("certbot not found; install with: apt-get install certbot".into());
    }
    if TcpListener::bind(("0.0.0.0", 80)).is_err() {
        return Err("port 80 must be free for certbot standalone".into());
    }
    run_certbot(&home, domain)?;
    install_certs(&home, domain)?;
    let con = db::open(&home).map_err(|e| e.to_string())?;
    con.execute(
        "INSERT OR REPLACE INTO mesh_settings (key, value) VALUES ('domain', ?1)",
        [domain],
    )
    .map_err(|e| e.to_string())?;
    con.execute(
        "INSERT OR REPLACE INTO mesh_settings (key, value) VALUES ('public_bind', ?1)",
        ["0.0.0.0"],
    )
    .map_err(|e| e.to_string())?;
    nginx::write_gateway_configs(&home)?;
    security::gateway_reload(&home)?;
    let port = nginx::public_port_u16(&home);
    Ok(format!("https://{domain}:{port}"))
}

pub fn public_ipv4() -> Result<String, String> {
    if let Ok(v) = std::env::var("MESH_PUBLIC_IP") {
        let t = v.trim();
        if !t.is_empty() {
            return parse_public_v4(t);
        }
    }
    let resp = crate::net::agent()
        .get("https://api.ipify.org")
        .timeout(Duration::from_secs(5))
        .call()
        .map_err(|e| format!("public IP lookup failed: {e}"))?;
    if resp.status() != 200 {
        return Err(format!("public IP lookup HTTP {}", resp.status()));
    }
    let mut body = String::new();
    resp.into_reader()
        .read_to_string(&mut body)
        .map_err(|e| e.to_string())?;
    parse_public_v4(body.trim())
}

fn parse_public_v4(s: &str) -> Result<String, String> {
    s.parse::<Ipv4Addr>()
        .map(|ip| ip.to_string())
        .map_err(|_| format!("invalid public IPv4: {s}"))
}

fn validate_domain(domain: &str) -> Result<(), String> {
    let d = domain.trim();
    if d.is_empty() {
        return Err("domain is empty".into());
    }
    if d.parse::<IpAddr>().is_ok() {
        return Err("domain must not be an IP address".into());
    }
    if !d.contains('.') {
        return Err("domain must contain at least one dot".into());
    }
    if d.starts_with('.') || d.ends_with('.') {
        return Err("invalid domain syntax".into());
    }
    for label in d.split('.') {
        if label.is_empty() || label.len() > 63 {
            return Err("invalid domain syntax".into());
        }
        let ok = label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-');
        if !ok || label.starts_with('-') || label.ends_with('-') {
            return Err("invalid domain syntax".into());
        }
    }
    Ok(())
}

fn check_dns(domain: &str, public_ip: &str) -> Result<(), String> {
    let addrs: Vec<IpAddr> = format!("{domain}:443")
        .to_socket_addrs()
        .map_err(|e| format!("DNS lookup failed: {e}"))?
        .map(|a| a.ip())
        .collect();
    let mut v4 = Vec::new();
    for a in addrs {
        if let IpAddr::V4(v) = a {
            v4.push(v.to_string());
        }
    }
    v4.sort();
    v4.dedup();
    if v4.iter().any(|ip| ip == public_ip) {
        return Ok(());
    }
    let ips = if v4.is_empty() {
        "none".to_string()
    } else {
        v4.join(", ")
    };
    Err(format!(
        "DNS for {domain} resolves to {ips}, expected {public_ip}"
    ))
}

fn run_certbot(home: &Path, domain: &str) -> Result<(), String> {
    let config_dir = home.join("letsencrypt/config");
    let work_dir = home.join("letsencrypt/work");
    let logs_dir = home.join("letsencrypt/logs");
    for p in [&config_dir, &work_dir, &logs_dir] {
        fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    let out = Command::new("certbot")
        .args([
            "certonly",
            "--standalone",
            "--non-interactive",
            "--agree-tos",
            "--register-unsafely-without-email",
            "-d",
            domain,
            "--config-dir",
            config_dir.to_str().unwrap(),
            "--work-dir",
            work_dir.to_str().unwrap(),
            "--logs-dir",
            logs_dir.to_str().unwrap(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("certbot exec failed: {e}"))?;
    if out.status.success() {
        return Ok(());
    }
    let err = String::from_utf8_lossy(&out.stderr);
    eprint!("{err}");
    Err(format!("certbot failed: {err}"))
}

fn install_certs(home: &Path, domain: &str) -> Result<(), String> {
    let live = home
        .join("letsencrypt/config/live")
        .join(domain);
    let chain = live.join("fullchain.pem");
    let key = live.join("privkey.pem");
    if !chain.is_file() || !key.is_file() {
        return Err("certbot did not produce certificate files".into());
    }
    let cert_dir = home.join("nginx/certs");
    fs::create_dir_all(&cert_dir).map_err(|e| e.to_string())?;
    let dst_crt = cert_dir.join("server.crt");
    let dst_key = cert_dir.join("server.key");
    fs::copy(&chain, &dst_crt).map_err(|e| e.to_string())?;
    fs::copy(&key, &dst_key).map_err(|e| e.to_string())?;
    fs::set_permissions(&dst_key, std::os::unix::fs::PermissionsExt::from_mode(0o600))
        .map_err(|e| e.to_string())?;
    Ok(())
}
