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
const PUBLIC_WARNING: &str = "WARNING: exposing the API on a public network is not recommended.
  - Anyone who can reach this machine can attempt to connect; only the allowlist and API keys stop them.
  - Allow only the client IPs you trust (goethe allowlist add <ip>/32) and give every client its own key.
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
        if interactive() && !opts.yes {
            ask(&format!("Allow which VPN addresses? (CIDR, comma separated) [{default}]:"), &default)?
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
    // Warn, confirm, bind every interface and allow only the client addresses the user names.
    println!("{PUBLIC_WARNING}");
    if !confirm("Expose the API on every network interface anyway?", false, opts.yes)? {
        return Err("nothing changed (goethe configure api again to pick local or VPN)".to_string());
    }
    let mut ranges = opts.allow.clone();
    if ranges.is_empty() && interactive() && !opts.yes {
        let a = ask("Client IPs or ranges to allow, comma separated (e.g. 203.0.113.7/32) []:", "")?;
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
        println!("No client addresses were added: only addresses already on the allowlist can connect (goethe allowlist add <ip>/32).");
    }
    Ok("0.0.0.0".to_string())
}

fn offer_key(yes: bool) -> Result<(), String> {
    // Without any API key no client can call the API, so offer one; the token is shown once.
    if !security::key_list_detailed()?.is_empty() {
        println!("API keys: {} exist (goethe key list; goethe key create --name <client> for each new client)", security::key_list_detailed()?.len());
        return Ok(());
    }
    if !interactive() && !yes {
        println!("No API keys yet: goethe key create --name <client>");
        return Ok(());
    }
    if !confirm("No API keys exist yet. Create one now?", true, yes)? {
        return Ok(());
    }
    let name = if yes { "default".to_string() } else { ask("Key name [default]:", "default")? };
    let token = security::key_create(&name, true, None, None, None, false)?;
    println!("API key ({name}): {token}");
    println!("Store it now: Goethe keeps only its hash and cannot show it again.");
    Ok(())
}

pub fn run_api(opts: ApiOpts) -> Result<(), String> {
    // Show the current setup, pick a mode (asked on a terminal), apply bind and allowlist, offer a key, print the URL.
    println!("{}", current_text()?);
    let vpns = vpn_addrs(&nginx::interface_addrs().unwrap_or_default());
    let mode = match opts.mode {
        Some(m) => m,
        None if interactive() => choose_mode(&vpns)?,
        None => return Err("no terminal to ask on: pass --mode local, --mode vpn or --mode public".to_string()),
    };
    let bound = match mode {
        Mode::Local => configure_local()?,
        Mode::Vpn => configure_vpn(&opts, &vpns)?,
        Mode::Public => configure_public(&opts)?,
    };
    offer_key(opts.yes)?;
    let port = nginx::public_port_u16(&goethe_home());
    let host = if bound == "0.0.0.0" { "<this machine's address>".to_string() } else { bound };
    println!("API: https://{host}:{port}/v1 (goethe get api shows it with the loaded models)");
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
    fn modes_parse_and_everyone_ranges_are_spotted() {
        assert_eq!(Mode::parse("VPN"), Ok(Mode::Vpn));
        assert!(Mode::parse("lan").is_err());
        assert!(is_everyone("0.0.0.0/0") && is_everyone("::/0") && !is_everyone("203.0.113.7/32"));
        assert_eq!(network("10.1.2.3".parse().unwrap(), 16), "10.1.0.0/16");
    }
}
