// Network interfaces and routes for choosing the gateway's network, with no knowledge of any VPN product.

use std::fmt;
use std::net::Ipv4Addr;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const ARPHRD_NONE: &str = "65534";
const IFF_POINTOPOINT: u32 = 0x10;
const WIDEST_RANGE: u8 = 8;
const PRIVATE_BLOCKS: [(Ipv4Addr, u8); 4] = [
    (Ipv4Addr::new(10, 0, 0, 0), 8),
    (Ipv4Addr::new(172, 16, 0, 0), 12),
    (Ipv4Addr::new(192, 168, 0, 0), 16),
    (Ipv4Addr::new(100, 64, 0, 0), 10),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Net {
    pub addr: Ipv4Addr,
    pub prefix: u8,
}

fn mask(prefix: u8) -> u32 {
    if prefix == 0 {
        0
    } else {
        u32::MAX << (32 - u32::from(prefix.min(32)))
    }
}

impl Net {
    pub fn new(ip: Ipv4Addr, prefix: u8) -> Net {
        let prefix = prefix.min(32);
        Net { addr: Ipv4Addr::from(u32::from(ip) & mask(prefix)), prefix }
    }

    pub fn parse(raw: &str) -> Option<Net> {
        // "a.b.c.d/n", or a bare address as /32.
        let (addr, prefix) = match raw.trim().split_once('/') {
            Some((a, p)) => (a, p.parse::<u8>().ok().filter(|p| *p <= 32)?),
            None => (raw.trim(), 32),
        };
        Some(Net::new(addr.parse().ok()?, prefix))
    }

    pub fn contains(&self, ip: Ipv4Addr) -> bool {
        u32::from(ip) & mask(self.prefix) == u32::from(self.addr)
    }

    pub fn covers(&self, other: &Net) -> bool {
        self.prefix <= other.prefix && self.contains(other.addr)
    }

    pub fn overlaps(&self, other: &Net) -> bool {
        self.covers(other) || other.covers(self)
    }
}

impl fmt::Display for Net {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.addr, self.prefix)
    }
}

pub fn private_block(ip: Ipv4Addr) -> Option<Net> {
    // The RFC 1918 or RFC 6598 block an address belongs to.
    PRIVATE_BLOCKS.iter().map(|&(a, p)| Net::new(a, p)).find(|n| n.contains(ip))
}

pub fn valid_name(name: &str) -> bool {
    // Linux interface names: 1 to 15 characters, no slash, whitespace or other punctuation beyond . _ - :.
    (1..=15).contains(&name.len()) && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | ':'))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Route {
    pub dev: String,
    pub dst: Option<Net>,
}

pub fn parse_ip_json(text: &str) -> Option<Vec<Route>> {
    // `ip -j -4 route show table all`: unicast routes only; "default" has no destination.
    let rows: Vec<serde_json::Value> = serde_json::from_str(text).ok()?;
    Some(
        rows.iter()
            .filter(|r| r.get("type").and_then(|t| t.as_str()).is_none_or(|t| t == "unicast"))
            .filter_map(|r| {
                let dev = r.get("dev")?.as_str()?.to_string();
                let dst = r.get("dst")?.as_str()?;
                let dst = if dst == "default" { None } else { Some(Net::parse(dst)?) };
                Some(Route { dev, dst })
            })
            .collect(),
    )
}

pub fn parse_proc_route(text: &str) -> Vec<Route> {
    // /proc/net/route (main table only): hex destination and mask in host byte order.
    let hex = |s: &str| u32::from_str_radix(s, 16).ok().map(|v| Ipv4Addr::from(u32::from_be(v)));
    text.lines()
        .skip(1)
        .filter_map(|line| {
            let f: Vec<&str> = line.split_whitespace().collect();
            let (dev, dst, mask) = (f.first()?, hex(f.get(1)?)?, hex(f.get(7)?)?);
            let prefix = u32::from(mask).count_ones() as u8;
            let dst = if prefix == 0 { None } else { Some(Net::new(dst, prefix)) };
            Some(Route { dev: dev.to_string(), dst })
        })
        .collect()
}

fn output_within(cmd: &mut Command, limit: Duration) -> Option<String> {
    // A command's stdout when it exits successfully within the limit; it is killed otherwise.
    let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null()).spawn().ok()?;
    let mut pipe = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut out = String::new();
        std::io::Read::read_to_string(&mut pipe, &mut out).ok().map(|_| out)
    });
    let deadline = Instant::now() + limit;
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => return reader.join().ok().flatten(),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

pub fn routes() -> Vec<Route> {
    // Every IPv4 route in every table from iproute2, or the main table from /proc when ip is missing or fails.
    output_within(Command::new("ip").args(["-j", "-4", "route", "show", "table", "all"]), Duration::from_secs(3))
        .and_then(|t| parse_ip_json(&t))
        .unwrap_or_else(|| parse_proc_route(&std::fs::read_to_string("/proc/net/route").unwrap_or_default()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Iface {
    pub name: String,
    pub addr: Ipv4Addr,
    pub prefix: u8,
    pub tunnel: bool,
    pub virtual_bridge: bool,
    pub default_route: bool,
}

fn sys(name: &str, file: &str) -> Option<String> {
    std::fs::read_to_string(Path::new("/sys/class/net").join(name).join(file)).ok().map(|s| s.trim().to_string())
}

fn is_tunnel(name: &str) -> bool {
    // No link layer (WireGuard, tun, most VPN clients) or a point-to-point link.
    let flags = sys(name, "flags").and_then(|f| u32::from_str_radix(f.trim_start_matches("0x"), 16).ok()).unwrap_or(0);
    sys(name, "type").as_deref() == Some(ARPHRD_NONE) || flags & IFF_POINTOPOINT != 0
}

fn is_virtual_bridge(name: &str) -> bool {
    // A bridge none of whose ports is a physical device: container and VM networks, not a network clients sit on.
    let base = Path::new("/sys/class/net").join(name);
    if !base.join("bridge").is_dir() {
        return false;
    }
    let ports = std::fs::read_dir(base.join("brif")).map(|d| d.flatten().map(|e| e.file_name()).collect::<Vec<_>>()).unwrap_or_default();
    !ports.iter().any(|p| Path::new("/sys/class/net").join(p).join("device").exists())
}

fn usable(ip: Ipv4Addr) -> bool {
    !ip.is_loopback() && !ip.is_link_local() && !ip.is_unspecified()
}

pub fn primary_ipv4(addrs: &[crate::nginx::InterfaceAddr], name: &str) -> Option<(Ipv4Addr, u8)> {
    // The first usable IPv4 address the kernel lists for the interface.
    addrs.iter().find_map(|a| match a.ip {
        std::net::IpAddr::V4(v4) if a.name == name && usable(v4) => Some((v4, a.prefix)),
        _ => None,
    })
}

pub fn interfaces(routes: &[Route]) -> Vec<Iface> {
    // Each interface with a usable IPv4 address, described from sysfs and the routing table.
    let addrs = crate::nginx::interface_addrs().unwrap_or_default();
    let mut names: Vec<&str> = addrs.iter().map(|a| a.name.as_str()).collect();
    names.dedup();
    let mut out: Vec<Iface> = Vec::new();
    for name in names {
        if out.iter().any(|i| i.name == name) {
            continue;
        }
        let Some((addr, prefix)) = primary_ipv4(&addrs, name) else { continue };
        out.push(Iface {
            name: name.to_string(),
            addr,
            prefix,
            tunnel: is_tunnel(name),
            virtual_bridge: is_virtual_bridge(name),
            default_route: routes.iter().any(|r| r.dev == name && r.dst.is_none()),
        });
    }
    out
}

pub fn candidates(ifaces: &[Iface]) -> Vec<Iface> {
    // Networks the API could serve: tunnels and interfaces on private addresses, tunnels first; container bridges and public NICs are left out.
    let mut out: Vec<Iface> = ifaces
        .iter()
        .filter(|i| i.name != "lo" && !i.virtual_bridge && (i.tunnel || private_block(i.addr).is_some()))
        .cloned()
        .collect();
    out.sort_by(|a, b| b.tunnel.cmp(&a.tunnel).then_with(|| a.name.cmp(&b.name)));
    out
}

pub fn reach(iface: &Iface, routes: &[Route]) -> Vec<Net> {
    // Its routes and own subnet, never a default or near-default route; with only per-peer host routes, the private block of its address, so later peers are covered.
    let mut nets: Vec<Net> = routes
        .iter()
        .filter(|r| r.dev == iface.name)
        .filter_map(|r| r.dst)
        .filter(|n| n.prefix >= WIDEST_RANGE && !(n.prefix == 32 && n.addr == iface.addr))
        .collect();
    if iface.prefix <= 30 && iface.prefix >= WIDEST_RANGE {
        nets.push(Net::new(iface.addr, iface.prefix));
    }
    if !nets.iter().any(|n| n.prefix <= 30) {
        if let Some(block) = private_block(iface.addr) {
            nets.retain(|n| !block.covers(n));
            nets.push(block);
        }
    }
    nets.sort_by_key(|n| n.prefix);
    let mut kept: Vec<Net> = Vec::new();
    for n in nets {
        if !kept.iter().any(|k| k.covers(&n)) {
            kept.push(n);
        }
    }
    kept.sort_by_key(|n| (u32::from(n.addr), n.prefix));
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iface(name: &str, addr: &str, prefix: u8, tunnel: bool) -> Iface {
        Iface { name: name.into(), addr: addr.parse().unwrap(), prefix, tunnel, virtual_bridge: false, default_route: false }
    }

    fn route(dev: &str, dst: &str) -> Route {
        Route { dev: dev.into(), dst: if dst == "default" { None } else { Net::parse(dst) } }
    }

    fn shown(nets: &[Net]) -> Vec<String> {
        nets.iter().map(Net::to_string).collect()
    }

    #[test]
    fn nets_parse_normalize_and_nest() {
        let n = Net::parse("10.8.0.7/24").unwrap();
        assert_eq!(n.to_string(), "10.8.0.0/24");
        assert_eq!(Net::parse("100.100.100.100").unwrap().prefix, 32);
        assert!(Net::parse("10.0.0.0/33").is_none() && Net::parse("nope").is_none());
        assert!(Net::parse("10.0.0.0/8").unwrap().covers(&n));
        assert!(!n.covers(&Net::parse("10.0.0.0/8").unwrap()));
        assert!(n.overlaps(&Net::parse("10.8.0.9").unwrap()));
        assert_eq!(private_block("100.70.1.2".parse().unwrap()).unwrap().to_string(), "100.64.0.0/10");
        assert_eq!(private_block("172.31.9.9".parse().unwrap()).unwrap().to_string(), "172.16.0.0/12");
        assert!(private_block("185.92.117.108".parse().unwrap()).is_none());
        assert!(private_block("100.128.0.1".parse().unwrap()).is_none(), "just past 100.64.0.0/10");
    }

    #[test]
    fn ip_json_keeps_unicast_routes_from_every_table() {
        let text = r#"[{"dst":"default","gateway":"185.92.117.1","dev":"eno1","protocol":"static","flags":[]},
            {"dst":"100.65.66.52","dev":"tailscale0","table":"52","flags":[]},
            {"dst":"10.8.0.0/24","dev":"wg0","protocol":"kernel","scope":"link","prefsrc":"10.8.0.2","flags":[]},
            {"type":"local","dst":"100.70.155.114","dev":"tailscale0","table":"local","flags":[]},
            {"type":"broadcast","dst":"10.8.0.255","dev":"wg0","table":"local","flags":[]}]"#;
        let r = parse_ip_json(text).unwrap();
        assert_eq!(r, vec![route("eno1", "default"), route("tailscale0", "100.65.66.52"), route("wg0", "10.8.0.0/24")]);
        assert!(parse_ip_json("not json").is_none());
    }

    #[test]
    fn proc_route_reads_destination_and_mask() {
        let text = "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n\
            eno1\t00000000\t0175E5B9\t0003\t0\t0\t100\t00000000\t0\t0\t0\n\
            wg0\t0000080A\t00000000\t0001\t0\t0\t0\t00FFFFFF\t0\t0\t0\n";
        assert_eq!(parse_proc_route(text), vec![route("eno1", "default"), route("wg0", "10.8.0.0/24")]);
    }

    #[test]
    fn reach_follows_routes_and_the_subnet() {
        let wg = iface("wg0", "10.8.0.2", 24, true);
        let routes = vec![route("wg0", "10.8.0.0/24"), route("wg0", "10.8.0.9"), route("wg0", "192.168.50.0/24"), route("eno1", "10.9.0.0/16")];
        assert_eq!(shown(&reach(&wg, &routes)), vec!["10.8.0.0/24", "192.168.50.0/24"], "peer /32 inside the subnet is folded; other devices' routes are ignored");
    }

    #[test]
    fn per_peer_routes_widen_to_the_private_block() {
        let ts = iface("tailscale0", "100.70.155.114", 32, true);
        let routes = vec![route("tailscale0", "100.65.66.52"), route("tailscale0", "100.100.100.100"), route("tailscale0", "100.70.155.114")];
        assert_eq!(shown(&reach(&ts, &routes)), vec!["100.64.0.0/10"]);
        let wg = iface("wg1", "10.9.0.5", 32, true);
        assert_eq!(shown(&reach(&wg, &[route("wg1", "10.9.0.1")])), vec!["10.0.0.0/8"]);
    }

    #[test]
    fn default_and_near_default_routes_never_count() {
        let full = iface("tun0", "10.200.0.6", 32, true);
        let routes = vec![route("tun0", "default"), route("tun0", "0.0.0.0/1"), route("tun0", "128.0.0.0/1"), route("tun0", "10.200.0.1")];
        assert_eq!(shown(&reach(&full, &routes)), vec!["10.0.0.0/8"], "a full-tunnel VPN does not allow the internet");
        let public = iface("tun1", "203.0.113.9", 32, true);
        assert!(reach(&public, &[route("tun1", "default")]).is_empty(), "nothing known: the user is asked");
    }

    #[test]
    fn candidates_are_tunnels_and_private_networks() {
        let mut bridge = iface("docker0", "172.16.0.1", 24, false);
        bridge.virtual_bridge = true;
        let all = vec![
            iface("eno1np0", "185.92.117.108", 24, false),
            iface("eth1", "10.0.0.5", 24, false),
            bridge,
            iface("tailscale0", "100.70.155.114", 32, true),
            iface("wg0", "10.8.0.2", 24, true),
            iface("tun9", "203.0.113.9", 32, true),
        ];
        let names: Vec<String> = candidates(&all).into_iter().map(|i| i.name).collect();
        assert_eq!(names, vec!["tailscale0", "tun9", "wg0", "eth1"], "tunnels first; the public NIC and the container bridge are left out");
    }

    #[test]
    fn interface_names_are_checked() {
        assert!(valid_name("wg0") && valid_name("tailscale0") && valid_name("eth0:1") && valid_name("enp3s0.100"));
        assert!(!valid_name("") && !valid_name("a/b") && !valid_name("0123456789abcdef") && !valid_name("wg 0"));
    }
}
