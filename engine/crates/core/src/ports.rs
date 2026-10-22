// Localhost port selection for the mux, the nginx gateway and the engine port range.

use std::fs;
use std::net::{SocketAddr, TcpListener};
use std::path::Path;

pub fn ensure_service_ports(home: &Path) -> Result<(), String> {
    fs::create_dir_all(home.join("nginx")).map_err(|e| e.to_string())?;
    let mux_explicit = env_port("GOETHE_MUX_PORT")?;
    let mux_persisted = read_port_file(home, "nginx/mux_port");
    let (mux, mux_notice) = resolve_mux_port(mux_explicit, mux_persisted, localhost_port_free)?;
    write_port_file(home, "nginx/mux_port", mux)?;
    if mux_notice {
        eprintln!(
            "goethe mux: 127.0.0.1:30000 busy, listening on 127.0.0.1:{mux} (persisted under nginx/mux_port)"
        );
    }

    let pub_explicit = env_port("GOETHE_PUBLIC_PORT")?;
    let pub_persisted = read_port_file(home, "nginx/public_port");
    let (public, pub_notice) =
        resolve_public_port(pub_explicit, pub_persisted, localhost_port_free)?;
    write_port_file(home, "nginx/public_port", public)?;
    if pub_notice {
        eprintln!(
            "goethe gateway: default ports busy, listening on 127.0.0.1:{public} (persisted under nginx/public_port)"
        );
    }
    Ok(())
}

fn env_port(name: &str) -> Result<Option<u16>, String> {
    match std::env::var(name) {
        Ok(v) if v.trim().is_empty() => Ok(None),
        Ok(v) => v
            .trim()
            .parse::<u16>()
            .map(Some)
            .map_err(|_| format!("invalid {name}")),
        Err(_) => Ok(None),
    }
}

fn read_port_file(home: &Path, rel: &str) -> Option<u16> {
    fs::read_to_string(home.join(rel))
        .ok()
        .and_then(|v| v.trim().parse().ok())
}

fn write_port_file(home: &Path, rel: &str, port: u16) -> Result<(), String> {
    fs::write(home.join(rel), format!("{port}\n")).map_err(|e| e.to_string())
}

const ENGINE_PORTS_DEFAULT: (u16, u16) = (30001, 30099);

pub fn parse_engine_ports(raw: Option<&str>) -> Result<(u16, u16), String> {
    // GOETHE_ENGINE_PORTS=<lo>-<hi>; unset or empty means the default range.
    let Some(v) = raw.map(str::trim).filter(|v| !v.is_empty()) else {
        return Ok(ENGINE_PORTS_DEFAULT);
    };
    let bad = || format!("invalid GOETHE_ENGINE_PORTS '{v}': expected <lo>-<hi> with 1 <= lo <= hi <= 65535");
    let (lo, hi) = v.split_once('-').ok_or_else(bad)?;
    let lo: u16 = lo.trim().parse().map_err(|_| bad())?;
    let hi: u16 = hi.trim().parse().map_err(|_| bad())?;
    if lo == 0 || lo > hi {
        return Err(bad());
    }
    Ok((lo, hi))
}

pub fn engine_port_range() -> Result<(u16, u16), String> {
    parse_engine_ports(std::env::var("GOETHE_ENGINE_PORTS").ok().as_deref())
}

pub fn localhost_port_free(port: u16) -> bool {
    let addr: SocketAddr = format!("127.0.0.1:{port}")
        .parse()
        .unwrap_or_else(|_| "127.0.0.1:0".parse().unwrap());
    TcpListener::bind(addr).is_ok()
}

pub fn resolve_mux_port(
    explicit: Option<u16>,
    persisted: Option<u16>,
    mut is_free: impl FnMut(u16) -> bool,
) -> Result<(u16, bool), String> {
    if let Some(p) = explicit {
        if !is_free(p) {
            return Err(format!(
                "GOETHE_MUX_PORT {p} is in use; set GOETHE_MUX_PORT to a free port"
            ));
        }
        return Ok((p, false));
    }
    if let Some(p) = persisted {
        if is_free(p) {
            return Ok((p, false));
        }
    }
    if is_free(30000) {
        return Ok((30000, false));
    }
    for p in 30100..=30199 {
        if is_free(p) {
            return Ok((p, true));
        }
    }
    Err("no free mux port in 127.0.0.1:30100-30199".to_string())
}

pub fn resolve_public_port(
    explicit: Option<u16>,
    persisted: Option<u16>,
    mut is_free: impl FnMut(u16) -> bool,
) -> Result<(u16, bool), String> {
    if let Some(p) = explicit {
        if !is_free(p) {
            return Err(format!(
                "GOETHE_PUBLIC_PORT {p} is in use; set GOETHE_PUBLIC_PORT to a free port"
            ));
        }
        return Ok((p, false));
    }
    let mut candidates = Vec::new();
    if let Some(p) = persisted {
        candidates.push(p);
    }
    candidates.push(8443);
    candidates.push(8080);
    candidates.extend(18443..=18499);
    let mut seen = std::collections::HashSet::new();
    for p in candidates {
        if !seen.insert(p) {
            continue;
        }
        if is_free(p) {
            let notice = p != 8443;
            return Ok((p, notice));
        }
    }
    Err("no free gateway port in 127.0.0.1:8443, 8080, or 18443-18499".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_ports_parse_and_reject_bad_ranges() {
        assert_eq!(parse_engine_ports(None).unwrap(), (30001, 30099));
        assert_eq!(parse_engine_ports(Some(" ")).unwrap(), (30001, 30099));
        assert_eq!(parse_engine_ports(Some("40000-40010")).unwrap(), (40000, 40010));
        for bad in ["40010-40000", "0-10", "abc", "1-70000", "5"] {
            assert!(parse_engine_ports(Some(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn mux_explicit_busy_errors() {
        let busy = std::collections::HashSet::from([30042u16]);
        let err = resolve_mux_port(Some(30042), None, |p| !busy.contains(&p)).unwrap_err();
        assert!(err.contains("GOETHE_MUX_PORT"));
    }

    #[test]
    fn mux_fallback_when_default_busy() {
        let busy = std::collections::HashSet::from([30000u16, 30100, 30101]);
        let (p, notice) = resolve_mux_port(None, None, |port| !busy.contains(&port)).unwrap();
        assert_eq!(p, 30102);
        assert!(notice);
    }

    #[test]
    fn public_prefers_8443_when_free() {
        let (p, notice) = resolve_public_port(None, None, |_| true).unwrap();
        assert_eq!(p, 8443);
        assert!(!notice);
    }

    #[test]
    fn public_tries_8080_then_range() {
        let busy = std::collections::HashSet::from([8443u16, 8080, 18443]);
        let (p, notice) = resolve_public_port(None, None, |port| !busy.contains(&port)).unwrap();
        assert_eq!(p, 18444);
        assert!(notice);
    }
}
