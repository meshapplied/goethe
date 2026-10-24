// Your own domain for the API: a DNS A record pointing at the gateway's address; goethe explains it and checks it.

use crate::nginx;
use crate::paths::goethe_home;
use std::io::Read;
use std::net::{IpAddr, Ipv4Addr, TcpStream, ToSocketAddrs};
use std::time::Duration;

fn api_ip() -> Result<IpAddr, String> {
    // The address other machines reach the API on: the bound IP, or the public IP when bound to all addresses.
    let bind: IpAddr = nginx::public_bind(&goethe_home()).parse().map_err(|_| "unreadable gateway bind address".to_string())?;
    if bind.is_loopback() {
        return Err("the API only listens on this machine".to_string());
    }
    if bind.is_unspecified() {
        return public_ipv4().and_then(|ip| ip.parse().map_err(|_| format!("invalid public IP {ip}")));
    }
    Ok(bind)
}

pub fn explain() -> Result<String, String> {
    // A DNS A record is all it takes; works for public, VPN and LAN addresses alike.
    let port = nginx::public_port_u16(&goethe_home());
    let ip = match api_ip() {
        Ok(ip) => ip.to_string(),
        Err(why) => {
            return Ok(format!(
                "To use a domain name, other machines must reach the API first: {why}.\n\n{}",
                crate::term::commands("Choose how they reach it:", &["goethe configure api"])
            ))
        }
    };
    Ok(format!(
        "Give the API a name like api.example.com instead of its IP address.\n\n\
         At your DNS provider, add an A record:\n\n\
         \tapi.example.com  A  {ip}\n\n\
         This works for public and private addresses alike; a VPN or LAN address only resolves to clients on that network.\n\
         Clients then use https://api.example.com:{port}/v1, with the same certificate settings as for the IP address.\n\n{}",
        crate::term::commands("Check that it worked:", &["goethe domain --check api.example.com"])
    ))
}

pub fn check(domain: &str) -> Result<String, String> {
    // The name resolves to the API's address and the gateway port answers there; Err when either fails.
    validate_domain(domain)?;
    let port = nginx::public_port_u16(&goethe_home());
    let resolved: Vec<IpAddr> = format!("{domain}:{port}")
        .to_socket_addrs()
        .map(|a| a.map(|s| s.ip()).collect())
        .unwrap_or_default();
    let shown = |ips: &[IpAddr]| ips.iter().map(IpAddr::to_string).collect::<Vec<_>>().join(", ");
    let dns = match (api_ip(), resolved.is_empty()) {
        (_, true) => Err(format!("{domain} does not resolve yet")),
        (Ok(ip), false) if resolved.contains(&ip) => Ok(format!("{domain} points to {ip}")),
        (Ok(ip), false) => Err(format!("{domain} points to {}, the API is on {ip}", shown(&resolved))),
        (Err(why), false) => Err(why),
    };
    let api = match resolved.first() {
        Some(ip) => match TcpStream::connect_timeout(&std::net::SocketAddr::new(*ip, port), Duration::from_secs(3)) {
            Ok(_) => Ok(format!("{domain}:{port} answers")),
            Err(e) => Err(format!("{domain}:{port}: {e}")),
        },
        None => Err("nothing to connect to".to_string()),
    };
    let ok = dns.is_ok() && api.is_ok();
    let mut table = crate::term::Table::new(&["CHECK", "STATUS", "DETAIL"]);
    for (name, result) in [("DNS", dns), ("API", api)] {
        let (status, detail) = match result {
            Ok(d) => ("OK", d),
            Err(d) => ("FAIL", d),
        };
        table.row(vec![name.to_string(), crate::term::status(status), detail]);
    }
    let text = table.render();
    if ok {
        Ok(format!("{text}\nClients can use: https://{domain}:{port}/v1\n"))
    } else {
        Err(format!("{text}\n{}", crate::term::commands("How to set it up:", &["goethe domain"])))
    }
}

pub fn public_ipv4() -> Result<String, String> {
    if let Ok(v) = std::env::var("GOETHE_PUBLIC_IP") {
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
