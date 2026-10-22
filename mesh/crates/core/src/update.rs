// Signed package updates from MESH_PKG_URL.

use crate::paths::{ensure_layout, mesh_home, mesh_sock};
use crate::runtime;
use crate::shutdown;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{IsTerminal, Read};
use std::path::Path;
use std::process::{Command, Stdio};

pub struct UpdateOpts {
    pub check_only: bool,
    pub engine: bool,
    pub ternary: bool,
    pub yes: bool,
}

pub fn run(opts: UpdateOpts) -> Result<(), String> {
    let home = mesh_home();
    ensure_layout(&home).map_err(|e| e.to_string())?;
    let pkg = pkg_base_url();
    let was_running = mesh_sock().exists();
    let tmp = home.join(".update-staging");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    fetch_pkg_file(&pkg, "VERSION", &tmp.join("VERSION"))?;
    fetch_pkg_file(&pkg, "SHA256SUMS", &tmp.join("SHA256SUMS"))?;
    fetch_pkg_file(&pkg, "SHA256SUMS.sig", &tmp.join("SHA256SUMS.sig"))?;
    verify_sums_signature(&home, &tmp)?;
    let remote_ver = fs::read_to_string(tmp.join("VERSION"))
        .map_err(|e| e.to_string())?
        .trim()
        .to_string();
    let local_ver = runtime::mesh_version();
    if remote_ver == local_ver && !opts.engine && !opts.ternary {
        println!("mesh {local_ver} is up to date");
        let _ = fs::remove_dir_all(&tmp);
        return Ok(());
    }
    if opts.check_only {
        println!("update available: {local_ver} -> {remote_ver}");
        let _ = fs::remove_dir_all(&tmp);
        return Ok(());
    }
    if !opts.yes && !std::io::stdin().is_terminal() {
        return Err("update requires --yes in non-interactive mode".to_string());
    }
    let arch = normalize_arch();
    for bin in ["mesh", "meshd"] {
        let rel = format!("bin/linux-{arch}/{bin}");
        let dest = home.join("bin").join(bin);
        install_verified(&pkg, &tmp, &rel, &dest)?;
    }
    if opts.engine {
        refresh_engine(&home, &pkg, &tmp)?;
    }
    if opts.ternary {
        fetch_ternary_pack(&home, &pkg, &tmp)?;
    }
    fs::copy(tmp.join("VERSION"), home.join("VERSION")).ok();
    let _ = fs::remove_dir_all(&tmp);
    if was_running {
        let _ = shutdown::run_stop();
        restart_meshd(&home)?;
        println!(
            "meshd restarted; loaded models return only once autostart exists (next package)"
        );
    }
    println!("updated to mesh {remote_ver}");
    Ok(())
}

pub fn pkg_base_url() -> String {
    std::env::var("MESH_PKG_URL")
        .unwrap_or_else(|_| "https://pkg.meshapplied.com".to_string())
        .trim_end_matches('/')
        .to_string()
}

fn normalize_arch() -> String {
    let raw = std::env::var("MESH_ARCH")
        .ok()
        .or_else(|| {
            std::process::Command::new("uname")
                .arg("-m")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "x86_64".to_string());
    match raw.as_str() {
        "arm64" | "aarch64" => "aarch64".to_string(),
        "x86_64" | "amd64" => "x86_64".to_string(),
        other => other.to_string(),
    }
}

fn fetch_pkg_file(base: &str, rel: &str, dest: &Path) -> Result<(), String> {
    let url = format!("{base}/{rel}");
    let mut req = crate::net::agent().get(&url).timeout(std::time::Duration::from_secs(60));
    if let Ok(tok) = std::env::var("MESH_PKG_TOKEN") {
        if !tok.trim().is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", tok.trim()));
        }
    }
    let resp = req.call().map_err(|e| format!("fetch {rel}: {e}"))?;
    if resp.status() >= 400 {
        return Err(format!("fetch {rel}: HTTP {}", resp.status()));
    }
    let mut reader = resp.into_reader();
    let mut buf = Vec::new();
    reader.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::write(dest, buf).map_err(|e| e.to_string())
}

fn verify_sums_signature(home: &Path, tmp: &Path) -> Result<(), String> {
    let pub_pem = home.join("release_pub.pem");
    if !pub_pem.is_file() {
        return Err("release_pub.pem missing; run installer first".to_string());
    }
    let status = Command::new("openssl")
        .args([
            "dgst",
            "-sha256",
            "-verify",
            pub_pem.to_str().unwrap_or("release_pub.pem"),
            "-signature",
        ])
        .arg(tmp.join("SHA256SUMS.sig"))
        .arg(tmp.join("SHA256SUMS"))
        .status()
        .map_err(|e| format!("openssl verify: {e}"))?;
    if !status.success() {
        return Err("SHA256SUMS signature verification failed".to_string());
    }
    Ok(())
}

fn lookup_sum(sums_path: &Path, rel: &str) -> Result<String, String> {
    let text = fs::read_to_string(sums_path).map_err(|e| e.to_string())?;
    for line in text.lines() {
        let mut parts = line.split_whitespace();
        let hash = parts.next().unwrap_or("");
        let path = parts.next().unwrap_or("");
        if path == rel {
            return Ok(hash.to_string());
        }
    }
    Err(format!("SHA256SUMS missing entry for {rel}"))
}

fn install_verified(base: &str, tmp: &Path, rel: &str, dest: &Path) -> Result<(), String> {
    let expected = lookup_sum(&tmp.join("SHA256SUMS"), rel)?;
    let part = tmp.join(rel.replace('/', "__"));
    fetch_pkg_file(base, rel, &part)?;
    let got = sha256_file(&part)?;
    if got != expected {
        return Err(format!("checksum mismatch for {rel}"));
    }
    if dest.exists() {
        let prev = dest.with_extension("prev");
        fs::rename(dest, &prev).ok();
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    fs::rename(&part, dest).map_err(|e| e.to_string())?;
    fs::set_permissions(dest, std::os::unix::fs::PermissionsExt::from_mode(0o755))
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 65536];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn refresh_engine(home: &Path, base: &str, tmp: &Path) -> Result<(), String> {
    fs::create_dir_all(home.join("pyenv")).map_err(|e| e.to_string())?;
    install_verified(base, tmp, "pyenv/bootstrap.sh", &home.join("pyenv/bootstrap.sh"))?;
    let arch = normalize_arch();
    let req_rel = format!("pyenv/requirements.{arch}.lock");
    let req_dest = if lookup_sum(&tmp.join("SHA256SUMS"), &req_rel).is_ok() {
        install_verified(base, tmp, &req_rel, &home.join("pyenv").join(format!("requirements.{arch}.lock")))?;
        home.join("pyenv").join(format!("requirements.{arch}.lock"))
    } else {
        install_verified(base, tmp, "pyenv/requirements.lock", &home.join("pyenv/requirements.lock"))?;
        home.join("pyenv/requirements.lock")
    };
    let _ = req_dest;
    let wheel_name = {
        let url = format!("{base}/wheels/LATEST");
        let mut req = crate::net::agent().get(&url);
        if let Ok(tok) = std::env::var("MESH_PKG_TOKEN") {
            if !tok.trim().is_empty() {
                req = req.set("Authorization", &format!("Bearer {}", tok.trim()));
            }
        }
        req.call()
            .map_err(|e| format!("wheels/LATEST: {e}"))?
            .into_string()
            .map_err(|e| e.to_string())?
            .trim()
            .to_string()
    };
    let rel = format!("wheels/{wheel_name}");
    let wheel_dest = home.join("pyenv").join(&wheel_name);
    install_verified(base, tmp, &rel, &wheel_dest)?;
    let status = Command::new("bash")
        .arg(home.join("pyenv/bootstrap.sh"))
        .env("MESH_HOME", home)
        .env("MESH_SGLANG_WHEEL", &wheel_dest)
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("py-env bootstrap failed".to_string());
    }
    Ok(())
}

fn fetch_ternary_pack(home: &Path, base: &str, tmp: &Path) -> Result<(), String> {
    let rel = "goethe/ternary-kernels.tar.zst";
    if lookup_sum(&tmp.join("SHA256SUMS"), rel).is_err() {
        return Ok(());
    }
    let part = tmp.join("ternary-kernels.tar.zst");
    fetch_pkg_file(base, rel, &part)?;
    let expected = lookup_sum(&tmp.join("SHA256SUMS"), rel)?;
    if sha256_file(&part)? != expected {
        return Err("ternary pack checksum mismatch".to_string());
    }
    let dest_dir = home.join("goethe");
    fs::create_dir_all(&dest_dir).map_err(|e| e.to_string())?;
    let tar = dest_dir.join("ternary-kernels.tar");
    let status = Command::new("zstd")
        .args(["-d", "-f"])
        .arg(&part)
        .arg("-o")
        .arg(&tar)
        .status()
        .map_err(|e| format!("zstd: {e}"))?;
    if !status.success() {
        return Err("zstd decompress failed".to_string());
    }
    let status = Command::new("tar")
        .args(["-xf"])
        .arg(&tar)
        .arg("-C")
        .arg(&dest_dir)
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("tar extract failed".to_string());
    }
    let _ = fs::remove_file(&tar);
    Ok(())
}

fn restart_meshd(home: &Path) -> Result<(), String> {
    let meshd = home.join("bin/meshd");
    if !meshd.is_file() {
        return Ok(());
    }
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(home.join("logs/meshd.log"))
        .map_err(|e| e.to_string())?;
    Command::new(&meshd)
        .env("MESH_HOME", home)
        .stdout(Stdio::from(log.try_clone().map_err(|e| e.to_string())?))
        .stderr(Stdio::from(log))
        .spawn()
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_lookup_and_file() {
        let dir = std::env::temp_dir().join(format!("mesh-upd-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("f"), b"abc").unwrap();
        let got = sha256_file(&dir.join("f")).unwrap();
        assert_eq!(got.len(), 64);
        let sums = format!("{got}  f\n");
        fs::write(dir.join("SHA256SUMS"), sums).unwrap();
        assert_eq!(lookup_sum(&dir.join("SHA256SUMS"), "f").unwrap(), got);
        let _ = fs::remove_dir_all(&dir);
    }
}
