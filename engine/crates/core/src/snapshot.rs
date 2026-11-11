// StateSnapshot (goethe.snapshot.v1) for the platform heartbeat, built from local state only and free of secrets.

use crate::platform::clean_text;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::Path;
use std::time::{Duration, Instant};

const MAX_INSTALLED: usize = 512;
const MAX_LOADED: usize = 128;
const MAX_ADDRESSES: usize = 64;
const MAX_GPUS: usize = 64;
const DISK_REFRESH: Duration = Duration::from_secs(3600);
const LIVE_STATUSES: &str = "('STARTING', 'RUNNING', 'RESTARTING', 'DRAINING', 'STOPPING')";

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct StateSnapshot {
    pub schema: u32,
    pub sent_at: String,
    pub reason: String,
    pub node: Node,
    pub network: Network,
    pub gpus: Vec<Gpu>,
    pub models: Models,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Node {
    pub hostname: String,
    pub goethe_version: String,
    pub os: String,
    pub kernel: String,
    pub uptime_s: u64,
    pub environment: Environment,
    pub daemon: Daemon,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Environment {
    pub kind: String,
    pub cloud: Option<String>,
    pub container: Option<String>,
    pub virt: Option<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Daemon {
    pub started_at: String,
    pub gateway_running: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Network {
    pub addresses: Vec<Address>,
    pub public_ip: Option<String>,
    pub gateway: Gateway,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Address {
    pub ip: String,
    pub interface: String,
    pub scope: String,
    pub tunnel: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Gateway {
    pub url: String,
    pub bind: String,
    pub interface: Option<String>,
    pub waiting: bool,
    pub port: u16,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Gpu {
    pub index: u32,
    pub uuid: String,
    pub pci_bus_id: String,
    pub name: String,
    pub sm: String,
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    pub vram_goethe_mb: u64,
    pub vram_foreign_mb: u64,
    pub mem_util_pct: Option<u32>,
    pub allowed: bool,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Models {
    pub installed: Vec<Installed>,
    pub loaded: Vec<Loaded>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Installed {
    pub id: String,
    pub disk_gb: f64,
    pub runtime: String,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Loaded {
    pub instance_id: String,
    pub model_id: String,
    pub gpus: Vec<u32>,
    pub tp_size: u32,
    pub ctx: u64,
    pub max_conc: u64,
    pub status: String,
    pub port: u16,
    pub in_flight: u64,
}

pub struct Collector {
    pub daemon_started_at: String,
    disk: HashMap<String, (f64, Instant)>,
}

impl Collector {
    pub fn new(daemon_started_at: String) -> Collector {
        Collector { daemon_started_at, disk: HashMap::new() }
    }

    pub fn build(&mut self, home: &Path, reason: &str, reported_public_ip: Option<&str>) -> StateSnapshot {
        // Every part is best effort: one that cannot be read comes out empty instead of failing the heartbeat.
        let root = crate::host::sysroot();
        let read = |rel: &str| std::fs::read_to_string(root.join(rel)).unwrap_or_default();
        let con = crate::db::open(home).ok();
        let addresses = addresses();
        let public_ip = public_ip(&addresses, reported_public_ip);
        let models = match &con {
            Some(c) => Models { installed: self.installed(c, home), loaded: loaded(c) },
            None => Models { installed: Vec::new(), loaded: Vec::new() },
        };
        StateSnapshot {
            schema: 1,
            sent_at: now_rfc3339(),
            reason: reason.to_string(),
            node: Node {
                hostname: crate::platform::hostname(),
                goethe_version: env!("CARGO_PKG_VERSION").to_string(),
                os: clean_text(&crate::host::os_release_name(&read("etc/os-release")), 128),
                kernel: clean_text(&read("proc/sys/kernel/osrelease"), 128),
                uptime_s: parse_uptime(&read("proc/uptime")),
                environment: environment(&root),
                daemon: Daemon {
                    started_at: self.daemon_started_at.clone(),
                    gateway_running: crate::nginx::format_gateway_line(home).is_some(),
                },
            },
            network: Network { gateway: gateway(home, &addresses), addresses, public_ip },
            gpus: con.as_ref().map(|c| gpus(c, home)).unwrap_or_default(),
            models,
        }
    }

    fn installed(&mut self, con: &rusqlite::Connection, home: &Path) -> Vec<Installed> {
        // Id, size on disk (walked at most hourly per model) and runtime from the model card.
        let rows: Vec<(String, String)> = con
            .prepare("SELECT model_id, local_path FROM installed_models ORDER BY model_id")
            .and_then(|mut s| s.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect())
            .unwrap_or_default();
        let ids: HashSet<&String> = rows.iter().map(|(id, _)| id).collect();
        self.disk.retain(|id, _| ids.contains(id));
        rows.iter()
            .take(MAX_INSTALLED)
            .map(|(id, local)| {
                let dir = home.join(local);
                let disk_gb = match self.disk.get(id) {
                    Some((gb, at)) if at.elapsed() < DISK_REFRESH => *gb,
                    _ => {
                        let target = std::fs::canonicalize(&dir).unwrap_or_else(|_| dir.clone());
                        let gb = (crate::models::dir_usage_gb(&target).0 * 100.0).round() / 100.0;
                        self.disk.insert(id.clone(), (gb, Instant::now()));
                        gb
                    }
                };
                let runtime = std::fs::read_to_string(dir.join("goethe_card.toml"))
                    .ok()
                    .and_then(|t| toml::from_str::<toml::Value>(&t).ok())
                    .and_then(|c| c.get("runtime").and_then(|v| v.as_str()).map(str::to_string))
                    .map(|r| clean_text(&r, 32))
                    .filter(|r| !r.is_empty())
                    .unwrap_or_else(|| "sglang".to_string());
                Installed { id: clean_text(id, 200), disk_gb, runtime }
            })
            .collect()
    }
}

pub fn public_ip(addresses: &[Address], reported: Option<&str>) -> Option<String> {
    // A public IPv4 on an interface first, else the address the platform saw in its last heartbeat reply; never a third party.
    if let Some(a) = addresses.iter().find(|a| a.scope == "public" && a.ip.parse::<Ipv4Addr>().is_ok()) {
        return Some(a.ip.clone());
    }
    reported.and_then(|r| r.parse::<IpAddr>().ok()).filter(|ip| scope_of(*ip, false) == "public").map(|ip| ip.to_string())
}

pub fn now_rfc3339() -> String {
    chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

fn parse_uptime(text: &str) -> u64 {
    text.split_whitespace().next().and_then(|s| s.parse::<f64>().ok()).map(|f| f.max(0.0) as u64).unwrap_or(0)
}

pub fn scope_of(ip: IpAddr, tunnel: bool) -> &'static str {
    // loopback, vpn (tunnel interfaces, CGNAT 100.64/10 and Tailscale's fd7a:115c:a1e0::/48), private, else public.
    if ip.is_loopback() {
        return "loopback";
    }
    match ip {
        IpAddr::V4(v4) => {
            let o = v4.octets();
            if tunnel || (o[0] == 100 && (64..128).contains(&o[1])) {
                "vpn"
            } else if v4.is_private() || v4.is_link_local() {
                "private"
            } else {
                "public"
            }
        }
        IpAddr::V6(v6) => {
            let s = v6.segments();
            if tunnel || (s[0] == 0xfd7a && s[1] == 0x115c && s[2] == 0xa1e0) {
                "vpn"
            } else if (s[0] & 0xfe00) == 0xfc00 || (s[0] & 0xffc0) == 0xfe80 {
                "private"
            } else {
                "public"
            }
        }
    }
}

fn addresses() -> Vec<Address> {
    // Addresses on this host's interfaces without link-local ones and container or VM bridges.
    let routes = crate::netif::routes();
    let ifaces = crate::netif::interfaces(&routes);
    let bridges: HashSet<&str> = ifaces.iter().filter(|i| i.virtual_bridge).map(|i| i.name.as_str()).collect();
    let mut out: Vec<Address> = Vec::new();
    for a in crate::nginx::interface_addrs().unwrap_or_default() {
        let link_local = match a.ip {
            IpAddr::V4(v4) => v4.is_link_local() || v4.is_unspecified(),
            IpAddr::V6(v6) => (v6.segments()[0] & 0xffc0) == 0xfe80 || v6.is_unspecified(),
        };
        if link_local || bridges.contains(a.name.as_str()) || !crate::netif::valid_name(&a.name) {
            continue;
        }
        let tunnel = a.name != "lo" && crate::netif::is_tunnel(&a.name);
        let ip = a.ip.to_string();
        if out.iter().any(|o| o.ip == ip && o.interface == a.name) {
            continue;
        }
        out.push(Address { scope: scope_of(a.ip, tunnel).to_string(), ip, interface: a.name, tunnel });
        if out.len() >= MAX_ADDRESSES {
            break;
        }
    }
    out
}

fn gateway(home: &Path, addresses: &[Address]) -> Gateway {
    // Where clients reach the gateway: the bound address, or for a wildcard bind the best address this host has.
    let bind = crate::nginx::public_bind(home);
    let port = crate::nginx::public_port_u16(home);
    Gateway {
        url: gateway_url(&bind, port, addresses),
        interface: crate::nginx::bind_interface(home),
        waiting: crate::nginx::bind_waiting(home).is_some(),
        bind,
        port,
    }
}

pub fn gateway_url(bind: &str, port: u16, addresses: &[Address]) -> String {
    let rank = |s: &str| ["vpn", "private", "public"].iter().position(|x| *x == s);
    let ip = match bind.parse::<IpAddr>() {
        Ok(ip) if ip.is_unspecified() => addresses
            .iter()
            .filter(|a| a.ip.parse::<Ipv4Addr>().is_ok())
            .filter_map(|a| rank(&a.scope).map(|r| (r, a)))
            .min_by_key(|(r, _)| *r)
            .and_then(|(_, a)| a.ip.parse().ok())
            .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST)),
        Ok(ip) => ip,
        Err(_) => IpAddr::V4(Ipv4Addr::LOCALHOST),
    };
    format!("https://{}/v1", SocketAddr::new(ip, port))
}

pub fn environment(root: &Path) -> Environment {
    let env = crate::host::probe_environment(0);
    let dmi = |f: &str| std::fs::read_to_string(root.join("sys/class/dmi/id").join(f)).map(|s| s.trim().to_string()).unwrap_or_default();
    let hypervisor = std::fs::read_to_string(root.join("proc/cpuinfo"))
        .unwrap_or_default()
        .lines()
        .any(|l| l.starts_with("flags") && l.split_whitespace().any(|f| f == "hypervisor"));
    classify_environment(env.container.as_deref(), &env.cloud, &dmi("sys_vendor"), &dmi("product_name"), hypervisor)
}

pub fn classify_environment(container: Option<&str>, cloud: &str, vendor: &str, product: &str, hypervisor: bool) -> Environment {
    // kind from container, cloud and virtualization markers; container kubernetes is reported as k8s.
    let all = format!("{vendor} {product}").to_ascii_lowercase();
    let virt = if all.contains("kvm") || all.contains("qemu") || vendor == "Amazon EC2" || product == "Google Compute Engine" {
        Some("kvm")
    } else if all.contains("vmware") {
        Some("vmware")
    } else if vendor == "Microsoft Corporation" && product == "Virtual Machine" {
        Some("hyperv")
    } else if all.contains("xen") || all.contains("hvm domu") {
        Some("xen")
    } else if all.contains("virtualbox") {
        Some("virtualbox")
    } else if hypervisor {
        Some("unknown")
    } else {
        None
    };
    let cloud = (!cloud.is_empty() && cloud != "none").then(|| clean_text(cloud, 32));
    let container = container.map(|c| if c == "kubernetes" { "k8s".to_string() } else { clean_text(c, 32) });
    let kind = match (&container, &cloud, virt) {
        (Some(_), _, _) => "container",
        (None, Some(_), Some(_)) => "cloud_vm",
        (None, None, Some(_)) => "vm",
        _ => "bare_metal",
    };
    Environment { kind: kind.to_string(), cloud, container, virt: virt.map(str::to_string) }
}

fn gpus(con: &rusqlite::Connection, home: &Path) -> Vec<Gpu> {
    // gpu_policy::list_rows (live NVML) joined with the profile's UUIDs and NVML PCI bus ids; empty when the driver does not answer.
    let Ok(rows) = crate::gpu_policy::list_rows(con, home) else {
        return Vec::new();
    };
    let profile = crate::hw::load_hardware_state(home).ok();
    let nvml = crate::nvidia_driver::nvml().ok();
    rows.into_iter()
        .take(MAX_GPUS)
        .map(|r| {
            let uuid = profile.as_ref().and_then(|p| p.gpus.iter().find(|g| g.index == r.index)).map(|g| g.uuid.clone()).unwrap_or_default();
            let pci = nvml
                .as_ref()
                .and_then(|n| n.device_by_index(r.index).ok())
                .and_then(|d| d.pci_info().ok())
                .map(|p| short_bus_id(&p.bus_id))
                .unwrap_or_default();
            Gpu {
                index: r.index,
                uuid: clean_text(&uuid, 64),
                pci_bus_id: pci,
                name: clean_text(&r.name, 128),
                sm: clean_text(&r.compute_capability, 16),
                vram_total_mb: r.vram_total_mb,
                vram_used_mb: r.managed_mb + r.foreign_mb,
                vram_goethe_mb: r.managed_mb,
                vram_foreign_mb: r.foreign_mb,
                mem_util_pct: r.memory_util_percent,
                allowed: r.status == "ALLOWED",
            }
        })
        .collect()
}

pub fn short_bus_id(raw: &str) -> String {
    // NVML's 00000000:01:00.0 in the usual 0000:01:00.0 form, lowercase.
    let s = clean_text(raw, 32).to_ascii_lowercase();
    match s.split_once(':') {
        Some((domain, rest)) if domain.len() == 8 && domain.starts_with("0000") => format!("{}:{rest}", &domain[4..]),
        _ => s,
    }
}

fn loaded(con: &rusqlite::Connection) -> Vec<Loaded> {
    // Live instances only (tombstones stay local); engine keys and command lines are never read.
    let sql = format!(
        "SELECT instance_id, model_id, gpu_indices, ctx_len, max_conc, status, internal_port FROM loaded_instances WHERE status IN {LIVE_STATUSES} ORDER BY rowid"
    );
    let rows: Vec<(String, String, String, i64, i64, String, i64)> = con
        .prepare(&sql)
        .and_then(|mut s| s.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?)))?.collect())
        .unwrap_or_default();
    rows.into_iter()
        .take(MAX_LOADED)
        .map(|(id, model, gpus, ctx, conc, status, port)| {
            let gpus: Vec<u32> = gpus.split(',').filter_map(|g| g.trim().parse().ok()).collect();
            Loaded {
                in_flight: crate::inflight::instance_count(&id) as u64,
                instance_id: clean_text(&id, 128),
                model_id: clean_text(&model, 200),
                tp_size: gpus.len().max(1) as u32,
                gpus,
                ctx: ctx.max(0) as u64,
                max_conc: conc.max(0) as u64,
                status: clean_text(&status, 16),
                port: u16::try_from(port).unwrap_or(0),
            }
        })
        .collect()
}

pub fn loaded_signature(s: &StateSnapshot) -> Vec<(String, String)> {
    s.models.loaded.iter().map(|l| (l.instance_id.clone(), l.status.clone())).collect()
}

pub fn encode(s: &StateSnapshot) -> Result<Vec<u8>, String> {
    // JSON within the 256 KB limit; an oversize snapshot drops installed models first, then everything but the node.
    let mut s = s.clone();
    for _ in 0..3 {
        let data = serde_json::to_vec(&s).map_err(|e| e.to_string())?;
        if data.len() <= crate::platform_api::MAX_SNAPSHOT_BYTES {
            return Ok(data);
        }
        if !s.models.installed.is_empty() {
            s.models.installed.clear();
        } else {
            s.network.addresses.clear();
            s.models.loaded.clear();
            s.gpus.clear();
        }
    }
    Err("the snapshot does not fit in 256 KB".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v4(s: &str) -> IpAddr {
        s.parse().unwrap()
    }

    #[test]
    fn scopes_are_classified() {
        assert_eq!(scope_of(v4("127.0.0.1"), false), "loopback");
        assert_eq!(scope_of(v4("::1"), false), "loopback");
        assert_eq!(scope_of(v4("100.70.1.2"), false), "vpn");
        assert_eq!(scope_of(v4("10.8.0.2"), true), "vpn");
        assert_eq!(scope_of(v4("fd7a:115c:a1e0::1"), false), "vpn");
        assert_eq!(scope_of(v4("192.168.1.5"), false), "private");
        assert_eq!(scope_of(v4("172.16.0.1"), false), "private");
        assert_eq!(scope_of(v4("fd00::1"), false), "private");
        assert_eq!(scope_of(v4("203.0.113.5"), false), "public");
        assert_eq!(scope_of(v4("100.128.0.1"), false), "public");
        assert_eq!(scope_of(v4("2001:db8::1"), false), "public");
    }

    #[test]
    fn environments_are_classified() {
        let k = |c: Option<&str>, cloud: &str, v: &str, p: &str, h: bool| classify_environment(c, cloud, v, p, h);
        assert_eq!(k(None, "none", "Supermicro", "X13", false).kind, "bare_metal");
        let e = k(Some("kubernetes"), "runpod", "", "", true);
        assert_eq!((e.kind.as_str(), e.container.as_deref(), e.cloud.as_deref()), ("container", Some("k8s"), Some("runpod")));
        let e = k(None, "aws", "Amazon EC2", "g6.xlarge", true);
        assert_eq!((e.kind.as_str(), e.virt.as_deref()), ("cloud_vm", Some("kvm")));
        let e = k(None, "none", "QEMU", "Standard PC (Q35 + ICH9, 2009)", true);
        assert_eq!((e.kind.as_str(), e.virt.as_deref(), e.cloud), ("vm", Some("kvm"), None));
        assert_eq!(k(None, "lambda", "Lambda", "Vector", false).kind, "bare_metal");
    }

    #[test]
    fn gateway_url_picks_a_reachable_address_for_a_wildcard_bind() {
        let a = |ip: &str, scope: &str| Address { ip: ip.into(), interface: "x".into(), scope: scope.into(), tunnel: false };
        let addrs = vec![a("127.0.0.1", "loopback"), a("203.0.113.5", "public"), a("192.168.1.5", "private"), a("100.70.1.2", "vpn")];
        assert_eq!(gateway_url("0.0.0.0", 8080, &addrs), "https://100.70.1.2:8080/v1");
        assert_eq!(gateway_url("10.0.0.2", 8080, &addrs), "https://10.0.0.2:8080/v1");
        assert_eq!(gateway_url("0.0.0.0", 8080, &[]), "https://127.0.0.1:8080/v1");
        assert_eq!(gateway_url("::1", 8443, &[]), "https://[::1]:8443/v1");
        assert_eq!(short_bus_id("00000000:01:00.0"), "0000:01:00.0");
        assert_eq!(public_ip(&addrs, Some("198.51.100.9")).as_deref(), Some("203.0.113.5"), "an interface address wins");
        let private = vec![a("192.168.1.5", "private")];
        assert_eq!(public_ip(&private, Some("198.51.100.9")).as_deref(), Some("198.51.100.9"));
        assert_eq!(public_ip(&private, None), None, "null until the first reply");
        assert_eq!(public_ip(&private, Some("10.0.0.1")), None);
        assert_eq!(parse_uptime("12345.67 999.0\n"), 12345);
    }

    fn seeded_home() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("secret-home-path");
        std::fs::create_dir_all(home.join("models/org__m")).unwrap();
        std::fs::write(home.join("models/org__m/goethe_card.toml"), "runtime = \"ninfer\"\n").unwrap();
        std::fs::write(home.join("models/org__m/w.safetensors"), vec![0u8; 4096]).unwrap();
        let con = crate::db::open(&home).unwrap();
        con.execute(
            "INSERT INTO installed_models (model_id, local_path, architecture, static_vram_gb, kv_bytes_per_token, default_ctx, default_max_conc, default_tp)
             VALUES ('org/m', 'models/org__m', 'X', 1.0, 1, 4096, 4, 1)",
            [],
        )
        .unwrap();
        con.execute(
            "INSERT INTO loaded_instances (instance_id, model_id, pid, pgid, internal_port, gpu_indices, allocated_vram_gb_per_gpu,
             mem_fraction_static, ctx_len, max_conc, status, slot_key, extra_sglang_args, engine_key)
             VALUES ('inst-1', 'org/m', 4242, 4242, 30001, '3', 9.0, 0.8, 100000, 4, 'RUNNING', 'org/m', '[\"--api-key\",\"sk-engine-SECRET\"]', 'ek-ENGINE-SECRET')",
            [],
        )
        .unwrap();
        con.execute(
            "INSERT INTO loaded_instances (instance_id, model_id, pid, pgid, internal_port, gpu_indices, allocated_vram_gb_per_gpu,
             mem_fraction_static, ctx_len, max_conc, status) VALUES ('inst-0', 'org/m', 0, 0, 30002, '2', 9.0, 0.8, 1, 1, 'FAILED')",
            [],
        )
        .unwrap();
        con.execute(
            "INSERT INTO api_keys (key_id, name, token_hash, raw_prefix, token) VALUES ('k1', 'app', 'HASH-SECRET-abc', 'gk_pre', 'gk_live_TOKEN-SECRET')",
            [],
        )
        .unwrap();
        (dir, home)
    }

    #[test]
    fn snapshot_has_the_contract_shape_and_no_secrets() {
        let (_dir, home) = seeded_home();
        let mut c = Collector::new("2026-11-11T10:00:00Z".into());
        let s = c.build(&home, "start", None);
        let text = String::from_utf8(encode(&s).unwrap()).unwrap();
        for secret in ["SECRET", "sk-engine", "ek-ENGINE", "gk_live", "HASH", "secret-home-path", "4242", "mesh_node_", "--api-key", "safetensors", "goethe_card"] {
            assert!(!text.contains(secret), "{secret} leaked into {text}");
        }
        let home_str = home.to_string_lossy().to_string();
        assert!(!text.contains(&home_str));
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["schema"], 1);
        assert_eq!(v["reason"], "start");
        for key in ["hostname", "goethe_version", "os", "kernel", "uptime_s", "environment", "daemon"] {
            assert!(v["node"].get(key).is_some(), "node.{key}");
        }
        for key in ["kind", "cloud", "container", "virt"] {
            assert!(v["node"]["environment"].get(key).is_some(), "environment.{key}");
        }
        for key in ["url", "bind", "interface", "waiting", "port"] {
            assert!(v["network"]["gateway"].get(key).is_some(), "gateway.{key}");
        }
        assert!(v["network"].get("public_ip").is_some() && v["network"]["addresses"].is_array());
        assert_eq!(v["models"]["installed"], serde_json::json!([{"id": "org/m", "disk_gb": 0.0, "runtime": "ninfer"}]));
        assert_eq!(
            v["models"]["loaded"],
            serde_json::json!([{"instance_id": "inst-1", "model_id": "org/m", "gpus": [3], "tp_size": 1, "ctx": 100000, "max_conc": 4, "status": "RUNNING", "port": 30001, "in_flight": 0}]),
            "tombstones stay local"
        );
        assert!(v["gpus"].is_array());
        let all_keys = |v: &serde_json::Value| -> Vec<String> {
            fn walk(v: &serde_json::Value, out: &mut Vec<String>) {
                match v {
                    serde_json::Value::Object(m) => m.iter().for_each(|(k, x)| {
                        out.push(k.clone());
                        walk(x, out)
                    }),
                    serde_json::Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
                    _ => {}
                }
            }
            let mut out = Vec::new();
            walk(v, &mut out);
            out
        };
        for k in all_keys(&v) {
            assert!(!["token", "key", "api_key", "engine_key", "path", "local_path", "env", "pid", "args"].contains(&k.as_str()), "field {k}");
        }
    }

    #[test]
    fn an_oversize_snapshot_sheds_installed_models_first() {
        let (_dir, home) = seeded_home();
        let mut s = Collector::new(String::new()).build(&home, "interval", None);
        let one = s.models.installed[0].clone();
        s.models.installed = vec![Installed { id: "x".repeat(200), ..one }; 2000];
        let data = encode(&s).unwrap();
        assert!(data.len() <= crate::platform_api::MAX_SNAPSHOT_BYTES);
        let v: serde_json::Value = serde_json::from_slice(&data).unwrap();
        assert_eq!(v["models"]["installed"], serde_json::json!([]));
        assert_eq!(v["models"]["loaded"].as_array().unwrap().len(), 1);
    }
}
