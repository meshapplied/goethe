// goethe configure api: a guided choice between local, VPN and public access to the gateway.

use crate::nginx::{self, InterfaceAddr};
use crate::paths::goethe_home;
use crate::security;
use std::io::{IsTerminal, Write};
use std::net::{IpAddr, Ipv4Addr};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Local,
    Vpn,
    Public,
}

impl Mode {
    pub fn parse(raw: &str) -> Result<Mode, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "local" => Ok(Mode::Local),
            "vpn" => Ok(Mode::Vpn),
            "public" => Ok(Mode::Public),
            other => Err(format!("unknown mode {other:?}: use local, vpn or public")),
        }
    }
}

pub struct ApiOpts {
    pub mode: Option<Mode>,
    pub vpn_ip: Option<IpAddr>,
    pub allow: Vec<String>,
    pub yes: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VpnAddr {
    pub name: String,
    pub ip: Ipv4Addr,
    pub kind: &'static str,
    pub range: String,
}

const TAILSCALE_RANGE: &str = "100.64.0.0/10";
const PUBLIC_WARNING: &str = "exposing the API on a public network is not recommended.
  - Anyone who can reach this machine can attempt to connect; only the allowlist and API keys stop them.
  - Allow only the client IPs you trust (goethe allowlist --add <ip>/32) and give every client its own key.
  - Open the gateway port in your firewall or router yourself; Goethe does not change them.
  - The certificate is self-signed, so clients must trust it or skip verification.";

fn in_cgnat(ip: Ipv4Addr) -> bool {
    let o = ip.octets();
    o[0] == 100 && (64..128).contains(&o[1])
}

fn network(ip: Ipv4Addr, prefix: u8) -> String {
    // The CIDR of the network an address with this prefix length belongs to.
    let mask = if prefix == 0 { 0 } else { u32::MAX << (32 - u32::from(prefix.min(32))) };
    format!("{}/{}", Ipv4Addr::from(u32::from(ip) & mask), prefix.min(32))
}

pub fn vpn_addrs(addrs: &[InterfaceAddr]) -> Vec<VpnAddr> {
    // IPv4 addresses on Tailscale, WireGuard, ZeroTier and other tunnel interfaces, each with the range to allow.
    let mut out = Vec::new();
    for a in addrs {
        let IpAddr::V4(ip) = a.ip else { continue };
        if ip.is_loopback() {
            continue;
        }
        let n = a.name.as_str();
        let kind = if n.starts_with("tailscale") || in_cgnat(ip) {
            "Tailscale"
        } else if n.starts_with("wg") {
            "WireGuard"
        } else if n.starts_with("zt") {
            "ZeroTier"
        } else if n.starts_with("tun") || n.starts_with("tap") || n.starts_with("nebula") {
            "VPN tunnel"
        } else {
            continue;
        };
        let range = if kind == "Tailscale" {
            TAILSCALE_RANGE.to_string()
        } else if a.prefix >= 31 {
            network(ip, 24)
        } else {
            network(ip, a.prefix)
        };
        out.push(VpnAddr { name: a.name.clone(), ip, kind, range });
    }
    out
}

fn interactive() -> bool {
    std::io::stdin().is_terminal() && std::io::stderr().is_terminal()
}

fn ask(question: &str, default: &str) -> Result<String, String> {
    // One line from the terminal; an empty answer takes the default.
    eprint!("{question} ");
    std::io::stderr().flush().ok();
    let mut buf = String::new();
    std::io::stdin().read_line(&mut buf).map_err(|e| e.to_string())?;
    let answer = buf.trim();
    Ok(if answer.is_empty() { default.to_string() } else { answer.to_string() })
}

fn confirm(question: &str, default_yes: bool, yes: bool) -> Result<bool, String> {
    // A y/n question; --yes answers yes, and without a terminal the default stands.
    if yes {
        return Ok(true);
    }
    if !interactive() {
        return Ok(default_yes);
    }
    let hint = if default_yes { "[Y/n]" } else { "[y/N]" };
    let a = ask(&format!("{question} {hint}"), if default_yes { "y" } else { "n" })?;
    Ok(matches!(a.to_ascii_lowercase().as_str(), "y" | "yes"))
}

fn is_everyone(cidr: &str) -> bool {
    cidr.ends_with("/0")
}

fn current_text() -> Result<String, String> {
    // The bind, port and allowlist as they are before any change.
    let home = goethe_home();
    let list = security::allowlist_list()?;
    let list = if list.is_empty() { "empty (every client refused)".to_string() } else { list.join(", ") };
    Ok(format!(
        "Gateway now: listening on {} port {}; allowlist: {list}",
        nginx::public_bind(&home),
        nginx::public_port_u16(&home)
    ))
}

fn choose_mode(vpns: &[VpnAddr]) -> Result<Mode, String> {
    // The three choices in an arrow-key menu, VPN preselected when one is up; a numbered prompt when raw mode fails.
    let detected = match vpns.first() {
        Some(v) => format!("detected: {} {} on {}", v.kind, v.ip, v.name),
        None => "none detected; set up Tailscale or WireGuard first".to_string(),
    };
    let options = vec![
        "Local only: programs on this machine".to_string(),
        format!("VPN (recommended): devices on your private network ({detected})"),
        "Public network (not recommended): any address that can reach this machine".to_string(),
    ];
    let modes = [Mode::Local, Mode::Vpn, Mode::Public];
    let default = if vpns.is_empty() { 0 } else { 1 };
    match crate::term::select(&options, default) {
        Ok(Some(i)) => return Ok(modes[i]),
        Ok(None) => return Err("cancelled; nothing changed".to_string()),
        Err(_) => {}
    }
    for (i, o) in options.iter().enumerate() {
        eprintln!("  {}) {o}", i + 1);
    }
    loop {
        let a = ask(&format!("Choice [{}]:", default + 1), &(default + 1).to_string())?;
        if let Some(m) = a.parse::<usize>().ok().and_then(|n| modes.get(n.wrapping_sub(1))) {
            return Ok(*m);
        }
        match Mode::parse(&a) {
            Ok(m) => return Ok(m),
            Err(_) => eprintln!("type 1, 2 or 3"),
        }
    }
}

fn allow(cidr: &str, note: &str) -> Result<(), String> {
    // Add one allowlist entry unless an equal one is already there; a stopped gateway still keeps the saved entry.
    let canonical = security::canonical_cidr(cidr)?;
    if security::allowlist_list()?.contains(&canonical) {
        return Ok(());
    }
    let added = security::allowlist_add(&canonical, Some(note));
    let saved = security::allowlist_list()?.contains(&canonical);
    match added {
        Ok(()) => println!("allowed {canonical} ({note})"),
        Err(e) if saved => println!("allowed {canonical} ({note}); {e}"),
        Err(e) => return Err(e),
    }
    Ok(())
}

fn bind(ip: &str) -> Result<(), String> {
    println!("{}", nginx::set_public_bind(&goethe_home(), ip)?);
    Ok(())
}

fn configure_local() -> Result<String, String> {
    // Loopback bind and 127.0.0.1 allowed: only programs on this machine reach the API.
    bind("127.0.0.1")?;
    allow("127.0.0.1/32", "this machine")?;
    Ok("127.0.0.1".to_string())
}

fn pick_vpn(opts: &ApiOpts, vpns: &[VpnAddr]) -> Result<VpnAddr, String> {
    // The --vpn-ip address, the only VPN address, or the user's pick among several.
    if let Some(ip) = opts.vpn_ip {
        let IpAddr::V4(v4) = ip else { return Err("--vpn-ip must be an IPv4 address".to_string()) };
        return Ok(vpns.iter().find(|v| v.ip == v4).cloned().unwrap_or(VpnAddr {
            name: "given".to_string(),
            ip: v4,
            kind: if in_cgnat(v4) { "Tailscale" } else { "VPN" },
            range: if in_cgnat(v4) { TAILSCALE_RANGE.to_string() } else { network(v4, 24) },
        }));
    }
    match vpns.len() {
        0 => Err("no VPN interface found (Tailscale, WireGuard or ZeroTier); bring one up, or pass --vpn-ip <address>".to_string()),
        1 => Ok(vpns[0].clone()),
        _ if !interactive() || opts.yes => Ok(vpns[0].clone()),
        _ => {
            let labels: Vec<String> = vpns.iter().map(|v| format!("{} {} on {}", v.kind, v.ip, v.name)).collect();
            match crate::term::select(&labels, 0) {
                Ok(Some(i)) => return Ok(vpns[i].clone()),
                Ok(None) => return Err("cancelled; nothing changed".to_string()),
                Err(_) => {}
            }
            for (i, l) in labels.iter().enumerate() {
                eprintln!("  {}) {l}", i + 1);
            }
            let a = ask("Which VPN address? [1]:", "1")?;
            a.parse::<usize>()
                .ok()
                .and_then(|n| vpns.get(n.wrapping_sub(1)))
                .cloned()
                .ok_or_else(|| format!("no VPN address number {a}"))
        }
    }
}

fn configure_vpn(opts: &ApiOpts, vpns: &[VpnAddr]) -> Result<String, String> {
    // Bind to the VPN address so other networks never reach the gateway, and allow the VPN's range.
    let v = pick_vpn(opts, vpns)?;
    let ranges = if opts.allow.is_empty() {
        let default = v.range.clone();
        let specific = interactive()
            && !opts.yes
            && confirm(&format!("Allow only specific addresses instead of the whole VPN range {default}?"), false, false)?;
        if specific {
            ask("Addresses or ranges to allow, comma separated (e.g. 100.64.0.7/32):", &default)?
        } else {
            default
        }
    } else {
        opts.allow.join(",")
    };
    let ranges: Vec<String> = ranges.split(',').map(str::trim).filter(|r| !r.is_empty()).map(String::from).collect();
    for r in &ranges {
        security::canonical_cidr(r)?;
    }
    println!("The gateway will listen only on {} ({} on {}) and allow {}.", v.ip, v.kind, v.name, ranges.join(", "));
    println!("Programs on this machine then use {} too; 127.0.0.1 stops answering.", v.ip);
    if !confirm("Apply?", true, opts.yes)? {
        return Err("nothing changed".to_string());
    }
    bind(&v.ip.to_string())?;
    for r in &ranges {
        allow(r, &format!("{} clients", v.kind))?;
    }
    Ok(v.ip.to_string())
}

fn configure_public(opts: &ApiOpts) -> Result<String, String> {
    // Warn, show the public IP and its API URL, confirm, bind every interface and allow only the named clients.
    println!("{}: {PUBLIC_WARNING}", crate::term::paint("WARNING", crate::term::Color::Yellow));
    let port = nginx::public_port_u16(&goethe_home());
    let public = lookup_public_ip();
    let public_host = public.as_ref().ok().map(|ip| ip.to_string());
    for line in public_ip_lines(public, port, &nginx::interface_addrs().unwrap_or_default()) {
        println!("{line}");
    }
    if !confirm("Expose the API on every network interface anyway?", false, opts.yes)? {
        return Err("nothing changed (goethe configure api again to pick local or VPN)".to_string());
    }
    let mut ranges = opts.allow.clone();
    if ranges.is_empty() && interactive() && !opts.yes && confirm("Allow specific client IP addresses now?", true, false)? {
        let a = ask("Client IPs or ranges to allow, comma separated (e.g. 203.0.113.7/32):", "")?;
        ranges = a.split(',').map(str::trim).filter(|r| !r.is_empty()).map(String::from).collect();
    }
    for r in &ranges {
        security::canonical_cidr(r)?;
        if is_everyone(r) {
            println!("{r} lets every address on the internet try your API keys.");
            if !interactive() || !confirm("Allow every address anyway?", false, false)? {
                return Err(format!("refusing to allow {r}; name the client addresses instead"));
            }
        }
    }
    bind("0.0.0.0")?;
    allow("127.0.0.1/32", "this machine")?;
    for r in &ranges {
        allow(r, "public client")?;
    }
    if ranges.is_empty() {
        println!("No client addresses were added: only addresses already on the allowlist can connect (goethe allowlist --add <ip>/32).");
    }
    Ok(public_host.unwrap_or_else(|| "0.0.0.0".to_string()))
}

const PUBLIC_IP_URL: &str = "https://api.ipify.org";

fn lookup_public_ip() -> Result<IpAddr, String> {
    // This machine's address as the internet sees it; GOETHE_PUBLIC_IP overrides, GOETHE_PUBLIC_IP_LOOKUP=0 disables.
    if let Some(ip) = crate::paths::setting("GOETHE_PUBLIC_IP") {
        return ip.trim().parse().map_err(|_| format!("GOETHE_PUBLIC_IP {ip:?} is not an IP address"));
    }
    if crate::paths::setting("GOETHE_PUBLIC_IP_LOOKUP").is_some_and(|v| v.trim() == "0") || crate::hf::offline_mode() {
        return Err("lookup disabled (GOETHE_PUBLIC_IP_LOOKUP=0 or offline mode; set GOETHE_PUBLIC_IP to name it)".to_string());
    }
    let url = crate::paths::setting("GOETHE_PUBLIC_IP_URL").unwrap_or_else(|| PUBLIC_IP_URL.to_string());
    let body = crate::net::agent()
        .get(&url)
        .timeout(std::time::Duration::from_secs(4))
        .call()
        .map_err(|e| format!("lookup at {url} failed ({e}); no internet access, or outbound HTTPS is blocked"))?
        .into_string()
        .map_err(|e| e.to_string())?;
    body.trim().parse().map_err(|_| format!("{url} answered {:?}, not an IP address", body.trim()))
}

fn public_ip_lines(ip: Result<IpAddr, String>, port: u16, local: &[InterfaceAddr]) -> Vec<String> {
    // What the public IP means for clients: the API URL on it, and whether a router must forward the port.
    match ip {
        Ok(ip) => {
            let mut lines = vec![format!("Your public IP: {ip} -> API: https://{ip}:{port}/v1")];
            if !local.iter().any(|a| a.ip == ip) {
                lines.push(format!(
                    "This machine is behind NAT ({ip} is not on its interfaces): forward TCP port {port} on your router or cloud firewall to it."
                ));
            }
            lines
        }
        Err(e) => vec![format!("Your public IP: unknown, {e}")],
    }
}

fn offer_key(yes: bool) -> Result<(), String> {
    // Without any API key no client can call the API, so offer one; goethe api shows it again later.
    if !security::key_list_detailed()?.is_empty() {
        println!("API keys: {} exist (goethe key; goethe key --create <client> for each new client)", security::key_list_detailed()?.len());
        return Ok(());
    }
    if !interactive() && !yes {
        println!("No API keys yet: goethe key --create <client>");
        return Ok(());
    }
    if !confirm("No API keys exist yet. Create one now?", true, yes)? {
        return Ok(());
    }
    let name = if yes { "default".to_string() } else { ask("Key name [default]:", "default")? };
    let token = security::key_create(&name, true, None, None, None, false)?;
    println!("API key ({name}): {token}");
    println!("goethe api shows it again at any time.");
    Ok(())
}

pub fn apply_access(opts: &ApiOpts) -> Result<String, String> {
    // Show the current setup, pick a mode (asked on a terminal), apply bind and allowlist; returns the API base URL.
    println!("{}", current_text()?);
    let vpns = vpn_addrs(&nginx::interface_addrs().unwrap_or_default());
    let mode = match opts.mode {
        Some(m) => m,
        None if interactive() => choose_mode(&vpns)?,
        None => return Err("no terminal to ask on: pass --mode local, --mode vpn or --mode public".to_string()),
    };
    let bound = match mode {
        Mode::Local => configure_local()?,
        Mode::Vpn => configure_vpn(opts, &vpns)?,
        Mode::Public => configure_public(opts)?,
    };
    let port = nginx::public_port_u16(&goethe_home());
    let host = match bound.as_str() {
        "0.0.0.0" => lookup_public_ip().map_or_else(|_| "<this machine's address>".to_string(), |ip| ip.to_string()),
        _ => bound,
    };
    Ok(format!("https://{host}:{port}/v1"))
}

pub fn run_api(opts: ApiOpts) -> Result<(), String> {
    // goethe configure api: who can reach the API, then a key if none exists, then the URL.
    let api = apply_access(&opts)?;
    offer_key(opts.yes)?;
    println!("\nAPI: {api}\n\n{}", crate::term::commands("For the models and keys run:", &["goethe api"]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(name: &str, ip: &str, prefix: u8) -> InterfaceAddr {
        InterfaceAddr { name: name.to_string(), ip: ip.parse().unwrap(), prefix }
    }

    #[test]
    fn vpn_interfaces_are_found_with_the_range_to_allow() {
        let addrs = vec![
            addr("lo", "127.0.0.1", 8),
            addr("eno1", "192.168.1.20", 24),
            addr("tailscale0", "100.70.155.114", 32),
            addr("wg0", "10.8.0.2", 24),
            addr("wg1", "10.9.0.5", 32),
            addr("zt7", "172.22.4.9", 16),
            addr("docker0", "172.17.0.1", 16),
        ];
        let found: Vec<(String, &str, String)> = vpn_addrs(&addrs).into_iter().map(|v| (v.name, v.kind, v.range)).collect();
        assert_eq!(
            found,
            vec![
                ("tailscale0".to_string(), "Tailscale", "100.64.0.0/10".to_string()),
                ("wg0".to_string(), "WireGuard", "10.8.0.0/24".to_string()),
                ("wg1".to_string(), "WireGuard", "10.9.0.0/24".to_string()),
                ("zt7".to_string(), "ZeroTier", "172.22.0.0/16".to_string()),
            ]
        );
    }

    #[test]
    fn public_ip_lines_show_the_api_and_spot_nat() {
        let local = vec![addr("eno1", "192.168.1.20", 24)];
        let nat = public_ip_lines(Ok("203.0.113.5".parse().unwrap()), 8080, &local);
        assert_eq!(nat[0], "Your public IP: 203.0.113.5 -> API: https://203.0.113.5:8080/v1");
        assert!(nat[1].contains("behind NAT") && nat[1].contains("8080"));
        let direct = public_ip_lines(Ok("192.168.1.20".parse().unwrap()), 8080, &local);
        assert_eq!(direct.len(), 1);
        let down = public_ip_lines(Err("lookup disabled".to_string()), 8080, &local);
        assert_eq!(down, vec!["Your public IP: unknown, lookup disabled".to_string()]);
    }

    #[test]
    fn modes_parse_and_everyone_ranges_are_spotted() {
        assert_eq!(Mode::parse("VPN"), Ok(Mode::Vpn));
        assert!(Mode::parse("lan").is_err());
        assert!(is_everyone("0.0.0.0/0") && is_everyone("::/0") && !is_everyone("203.0.113.7/32"));
        assert_eq!(network("10.1.2.3".parse().unwrap(), 16), "10.1.0.0/16");
    }
}
