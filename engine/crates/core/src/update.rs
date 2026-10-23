// Signed package updates from GOETHE_PKG_URL: verify everything, then swap both binaries.

use crate::paths::{ensure_layout, goethe_home, goethe_sock};
use crate::shutdown;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub struct UpdateOpts {
    pub check_only: bool,
    pub engine: bool,
    pub ternary: bool,
    pub yes: bool,
    pub allow_downgrade: bool,
}

struct Staged {
    rel: String,
    part: PathBuf,
    dest: PathBuf,
}

pub fn run(opts: UpdateOpts) -> Result<(), String> {
    // Fetch and verify the signed manifest, compare versions, stage files, then swap atomically.
    let home = goethe_home();
    ensure_layout(&home).map_err(|e| e.to_string())?;
    let pkg = pkg_base_url();
    let tmp = home.join(".update-staging");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    let result = update_in(&home, &pkg, &tmp, &opts);
    let _ = fs::remove_dir_all(&tmp);
    result
}

fn update_in(home: &Path, pkg: &str, tmp: &Path, opts: &UpdateOpts) -> Result<(), String> {
    fetch_pkg_file(pkg, "SHA256SUMS", &tmp.join("SHA256SUMS"))?;
    fetch_pkg_file(pkg, "SHA256SUMS.sig", &tmp.join("SHA256SUMS.sig"))?;
    verify_sums_signature(home, tmp)?;
    let sums = tmp.join("SHA256SUMS");
    let version_file = fetch_verified(pkg, &sums, "VERSION", tmp)?;
    let remote_ver = fs::read_to_string(&version_file).map_err(|e| e.to_string())?.trim().to_string();
    let local_ver = local_version();
    let order = compare_versions(&remote_ver, &local_ver);
    if order == Ordering::Less && !opts.allow_downgrade {
        if opts.check_only {
            println!("goethe {local_ver} is newer than the published {remote_ver}");
            return Ok(());
        }
        return Err(format!(
            "published version {remote_ver} is older than goethe {local_ver}; re-run with --allow-downgrade to install it"
        ));
    }
    let engine = if opts.engine { Some(engine_delta(home, pkg, &sums, tmp)?) } else { None };
    let ternary = if opts.ternary { ternary_delta(home, &sums) } else { None };
    let bins = order != Ordering::Equal;
    let parts = [("goethe", Some(Delta::of(&local_ver, &remote_ver, bins))), ("engine", engine), ("ternary pack", ternary)];
    let pending: Vec<(&str, Delta)> = parts.into_iter().filter_map(|(what, d)| d.map(|d| (what, d))).collect();
    for (what, d) in &pending {
        println!("{}", d.line(what));
    }
    if opts.check_only || pending.iter().all(|(_, d)| !d.differs) {
        return Ok(());
    }
    crate::host::confirm(&format!("Update goethe {local_ver} -> {remote_ver}?"), opts.yes)?;
    let arch = normalize_arch();
    let mut staged = Vec::new();
    for bin in ["goethe", "goethed"].into_iter().filter(|_| bins) {
        let rel = format!("bin/linux-{arch}/{bin}");
        staged.push(stage(pkg, &sums, &rel, tmp, &home.join("bin").join(bin))?);
    }
    let differs = |what: &str| pending.iter().any(|(w, d)| *w == what && d.differs);
    let wheel = if differs("engine") { Some(stage_engine(home, pkg, &sums, tmp, &arch, &mut staged)?) } else { None };
    let ternary = if differs("ternary pack") { stage_ternary(pkg, &sums, tmp)? } else { None };
    let was_running = goethe_sock().exists();
    swap_all(&staged)?;
    if let Some(w) = wheel {
        bootstrap_engine(home, &w)?;
        let name = w.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        fs::write(home.join("pyenv/LATEST"), format!("{name}\n")).map_err(|e| e.to_string())?;
    }
    if let Some(t) = ternary {
        unpack_ternary(home, &t)?;
        let sha = lookup_sum(&sums, TERNARY_REL)?;
        fs::write(home.join(TERNARY_RECORD), format!("{sha}\n")).map_err(|e| e.to_string())?;
    }
    fs::copy(&version_file, home.join("VERSION")).map_err(|e| e.to_string())?;
    if was_running && bins {
        restart_goethed(home)?;
    }
    println!("updated to goethe {remote_ver}");
    Ok(())
}

const TERNARY_REL: &str = "goethe/ternary-kernels.tar.zst";
const TERNARY_RECORD: &str = "goethe/ternary-kernels.sha256";

struct Delta {
    installed: Option<String>,
    published: String,
    differs: bool,
}

impl Delta {
    fn of(installed: &str, published: &str, differs: bool) -> Delta {
        Delta { installed: Some(installed.to_string()), published: published.to_string(), differs }
    }

    fn line(&self, what: &str) -> String {
        let installed = self.installed.as_deref().unwrap_or("not recorded");
        if self.differs {
            format!("update available: {what} {installed} -> {}", self.published)
        } else {
            format!("{what} {installed} is up to date")
        }
    }
}

fn engine_delta(home: &Path, base: &str, sums: &Path, tmp: &Path) -> Result<Delta, String> {
    // Published wheel (verified wheels/LATEST and its signed sha) against the wheel recorded in GOETHE_HOME/pyenv.
    let latest = fetch_verified(base, sums, "wheels/LATEST", tmp)?;
    let published = fs::read_to_string(&latest).map_err(|e| e.to_string())?.trim().to_string();
    let published_sha = lookup_sum(sums, &format!("wheels/{published}"))?;
    let pyenv = home.join("pyenv");
    let recorded = fs::read_to_string(pyenv.join("LATEST")).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    let file_sha = sha256_file(&pyenv.join(&published)).ok();
    Ok(engine_compare(recorded, file_sha.as_deref(), &published, &published_sha))
}

fn engine_compare(recorded: Option<String>, file_sha: Option<&str>, published: &str, published_sha: &str) -> Delta {
    // Same wheel name and, when the wheel file is still there, the same sha means no engine update.
    let installed = recorded.or_else(|| file_sha.map(|_| published.to_string()));
    let differs = installed.as_deref() != Some(published) || file_sha.is_some_and(|s| s != published_sha);
    Delta { installed, published: published.to_string(), differs }
}

fn ternary_delta(home: &Path, sums: &Path) -> Option<Delta> {
    // Published pack sha against the sha recorded when the pack was installed; None when no pack is published.
    let published = lookup_sum(sums, TERNARY_REL).ok()?;
    let recorded = fs::read_to_string(home.join(TERNARY_RECORD)).ok().map(|s| s.trim().to_string());
    let short = |s: &str| s.chars().take(12).collect::<String>();
    Some(Delta {
        installed: recorded.as_deref().map(short),
        differs: recorded.as_deref() != Some(published.as_str()),
        published: short(&published),
    })
}

pub fn pkg_base_url() -> String {
    crate::paths::setting("GOETHE_PKG_URL")
        .unwrap_or_else(|| "https://pkg.meshapplied.com".to_string())
        .trim_end_matches('/')
        .to_string()
}

fn local_version() -> String {
    crate::runtime::goethe_version().to_string()
}

pub fn compare_versions(a: &str, b: &str) -> Ordering {
    // Semver order: numeric major.minor.patch, then a release sorts after any prerelease of it.
    let parse = |v: &str| {
        let v = v.trim().trim_start_matches('v');
        let v = v.split('+').next().unwrap_or(v);
        let (core, pre) = match v.split_once('-') {
            Some((c, p)) => (c, Some(p.to_string())),
            None => (v, None),
        };
        let nums: Vec<u64> = core.split('.').map(|x| x.parse().unwrap_or(0)).collect();
        let n = |i: usize| nums.get(i).copied().unwrap_or(0);
        ((n(0), n(1), n(2)), pre)
    };
    let (ca, pa) = parse(a);
    let (cb, pb) = parse(b);
    ca.cmp(&cb).then_with(|| match (pa, pb) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(x), Some(y)) => x.cmp(&y),
    })
}

fn normalize_arch() -> String {
    let raw = std::env::var("GOETHE_ARCH")
        .ok()
        .or_else(|| {
            Command::new("uname")
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
    // GET one CDN file, with GOETHE_PKG_TOKEN as a bearer token when set.
    let url = format!("{base}/{rel}");
    let mut req = crate::net::agent().get(&url).timeout(std::time::Duration::from_secs(60));
    if let Some(tok) = crate::paths::setting("GOETHE_PKG_TOKEN") {
        req = req.set("Authorization", &format!("Bearer {}", tok.trim()));
    }
    let resp = req.call().map_err(|e| format!("fetch {rel}: {e}"))?;
    let mut buf = Vec::new();
    resp.into_reader().read_to_end(&mut buf).map_err(|e| e.to_string())?;
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
        .args(["dgst", "-sha256", "-verify"])
        .arg(&pub_pem)
        .arg("-signature")
        .arg(tmp.join("SHA256SUMS.sig"))
        .arg(tmp.join("SHA256SUMS"))
        .stdout(Stdio::null())
        .status()
        .map_err(|e| format!("openssl verify: {e}"))?;
    if !status.success() {
        return Err("SHA256SUMS signature verification failed".to_string());
    }
    Ok(())
}

fn lookup_sum(sums_path: &Path, rel: &str) -> Result<String, String> {
    let text = fs::read_to_string(sums_path).map_err(|e| e.to_string())?;
    text.lines()
        .find_map(|line| {
            let mut parts = line.split_whitespace();
            let hash = parts.next()?;
            (parts.next()? == rel).then(|| hash.to_string())
        })
        .ok_or_else(|| format!("SHA256SUMS missing entry for {rel}"))
}

fn fetch_verified(base: &str, sums: &Path, rel: &str, tmp: &Path) -> Result<PathBuf, String> {
    // Download rel into staging and require the hash listed in the signed SHA256SUMS.
    let expected = lookup_sum(sums, rel)?;
    let part = tmp.join(rel.replace('/', "__"));
    fetch_pkg_file(base, rel, &part)?;
    if sha256_file(&part)? != expected {
        return Err(format!("checksum mismatch for {rel}"));
    }
    Ok(part)
}

fn stage(base: &str, sums: &Path, rel: &str, tmp: &Path, dest: &Path) -> Result<Staged, String> {
    Ok(Staged {
        rel: rel.to_string(),
        part: fetch_verified(base, sums, rel, tmp)?,
        dest: dest.to_path_buf(),
    })
}

fn swap_all(staged: &[Staged]) -> Result<(), String> {
    // Move every verified file into place, keeping .prev copies; roll back on the first failure.
    let mut done: Vec<&Staged> = Vec::new();
    for s in staged {
        let r = (|| {
            if let Some(parent) = s.dest.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            let staged_dest = s.dest.with_extension("new");
            fs::copy(&s.part, &staged_dest).map_err(|e| format!("{}: {e}", staged_dest.display()))?;
            fs::set_permissions(&staged_dest, std::os::unix::fs::PermissionsExt::from_mode(0o755))
                .map_err(|e| e.to_string())?;
            if s.dest.exists() {
                fs::rename(&s.dest, s.dest.with_extension("prev")).map_err(|e| e.to_string())?;
            }
            fs::rename(&staged_dest, &s.dest).map_err(|e| e.to_string())
        })();
        if let Err(e) = r {
            for d in done.iter().rev() {
                let _ = fs::rename(d.dest.with_extension("prev"), &d.dest);
            }
            return Err(format!("installing {} failed ({e}); previous files restored", s.rel));
        }
        done.push(s);
    }
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

fn stage_engine(home: &Path, base: &str, sums: &Path, tmp: &Path, arch: &str, staged: &mut Vec<Staged>) -> Result<PathBuf, String> {
    // Stage bootstrap.sh, the lock file and the wheel named by the verified wheels/LATEST.
    let pyenv = home.join("pyenv");
    staged.push(stage(base, sums, "pyenv/bootstrap.sh", tmp, &pyenv.join("bootstrap.sh"))?);
    let arch_lock = format!("pyenv/requirements.{arch}.lock");
    let lock = if lookup_sum(sums, &arch_lock).is_ok() { arch_lock } else { "pyenv/requirements.lock".to_string() };
    let lock_name = lock.trim_start_matches("pyenv/").to_string();
    staged.push(stage(base, sums, &lock, tmp, &pyenv.join(lock_name))?);
    let latest = fetch_verified(base, sums, "wheels/LATEST", tmp)?;
    let wheel_name = fs::read_to_string(&latest).map_err(|e| e.to_string())?.trim().to_string();
    if wheel_name.is_empty() || wheel_name.contains('/') {
        return Err(format!("wheels/LATEST names an invalid wheel {wheel_name:?}"));
    }
    let dest = pyenv.join(&wheel_name);
    staged.push(stage(base, sums, &format!("wheels/{wheel_name}"), tmp, &dest)?);
    Ok(dest)
}

fn bootstrap_engine(home: &Path, wheel: &Path) -> Result<(), String> {
    let status = Command::new("bash")
        .arg(home.join("pyenv/bootstrap.sh"))
        .env("GOETHE_HOME", home)
        .env("GOETHE_SGLANG_WHEEL", wheel)
        .status()
        .map_err(|e| e.to_string())?;
    if !status.success() {
        return Err("py-env bootstrap failed".to_string());
    }
    Ok(())
}

fn stage_ternary(base: &str, sums: &Path, tmp: &Path) -> Result<Option<PathBuf>, String> {
    if lookup_sum(sums, TERNARY_REL).is_err() {
        return Ok(None);
    }
    fetch_verified(base, sums, TERNARY_REL, tmp).map(Some)
}

fn unpack_ternary(home: &Path, archive: &Path) -> Result<(), String> {
    // Decompress and extract the ternary runtime under GOETHE_HOME/goethe.
    let dest_dir = home.join("goethe");
    fs::create_dir_all(&dest_dir).map_err(|e| e.to_string())?;
    let tar = dest_dir.join("ternary-kernels.tar");
    let ok = Command::new("zstd")
        .args(["-d", "-f", "-q"])
        .arg(archive)
        .arg("-o")
        .arg(&tar)
        .status()
        .map_err(|e| format!("zstd: {e}"))?
        .success();
    if !ok {
        return Err("zstd decompress failed".to_string());
    }
    let ok = Command::new("tar")
        .arg("-xf")
        .arg(&tar)
        .arg("-C")
        .arg(&dest_dir)
        .status()
        .map_err(|e| e.to_string())?
        .success();
    let _ = fs::remove_file(&tar);
    if !ok {
        return Err("tar extract failed".to_string());
    }
    Ok(())
}

fn restart_goethed(home: &Path) -> Result<(), String> {
    // Restart through the systemd unit when one manages goethed, else stop and respawn it.
    if let Some(args) = crate::service::systemctl_restart_args() {
        let ok = Command::new("systemctl").args(&args).status().map_err(|e| e.to_string())?.success();
        if !ok {
            return Err(format!("systemctl {} failed", args.join(" ")));
        }
        println!("restarted goethed with systemctl {}", args.join(" "));
        return Ok(());
    }
    shutdown::run_stop()?;
    crate::service::spawn_goethed(home)?;
    println!("restarted goethed; desired loads are restored when restore is on (goethe service restore --on)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_lookup_and_file() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("f"), b"abc").unwrap();
        let got = sha256_file(&dir.path().join("f")).unwrap();
        assert_eq!(got.len(), 64);
        fs::write(dir.path().join("SHA256SUMS"), format!("{got}  f\n")).unwrap();
        assert_eq!(lookup_sum(&dir.path().join("SHA256SUMS"), "f").unwrap(), got);
        assert!(lookup_sum(&dir.path().join("SHA256SUMS"), "g").is_err());
    }

    #[test]
    fn engine_update_only_when_the_wheel_differs() {
        let rec = |s: &str| Some(s.to_string());
        assert!(!engine_compare(rec("sglang-1.whl"), Some("aa"), "sglang-1.whl", "aa").differs);
        assert!(!engine_compare(rec("sglang-1.whl"), None, "sglang-1.whl", "aa").differs, "wheel file pruned");
        assert!(!engine_compare(None, Some("aa"), "sglang-1.whl", "aa").differs, "no record, same file");
        assert!(engine_compare(rec("sglang-1.whl"), Some("bb"), "sglang-1.whl", "aa").differs, "republished");
        assert!(engine_compare(rec("sglang-0.whl"), None, "sglang-1.whl", "aa").differs);
        let none = engine_compare(None, None, "sglang-1.whl", "aa");
        assert!(none.differs);
        assert_eq!(none.line("engine"), "update available: engine not recorded -> sglang-1.whl");
        assert_eq!(Delta::of("0.1.0", "0.1.0", false).line("goethe"), "goethe 0.1.0 is up to date");
        let dir = tempfile::tempdir().unwrap();
        let sums = dir.path().join("SHA256SUMS");
        fs::write(&sums, format!("{}  {TERNARY_REL}\n", "c".repeat(64))).unwrap();
        assert!(ternary_delta(dir.path(), &sums).unwrap().differs);
        fs::create_dir_all(dir.path().join("goethe")).unwrap();
        fs::write(dir.path().join(TERNARY_RECORD), format!("{}\n", "c".repeat(64))).unwrap();
        let same = ternary_delta(dir.path(), &sums).unwrap();
        assert!(!same.differs);
        assert_eq!(same.line("ternary pack"), "ternary pack cccccccccccc is up to date");
        fs::write(&sums, "").unwrap();
        assert!(ternary_delta(dir.path(), &sums).is_none());
    }

    #[test]
    fn semver_ordering() {
        assert_eq!(compare_versions("9.9.9-test", "0.1.0"), Ordering::Greater);
        assert_eq!(compare_versions("0.1.0", "0.1.0"), Ordering::Equal);
        assert_eq!(compare_versions("0.10.0", "0.9.9"), Ordering::Greater);
        assert_eq!(compare_versions("0.2.0-rc1", "0.2.0"), Ordering::Less);
        assert_eq!(compare_versions("v0.1.1+abc", "0.1.1"), Ordering::Equal);
        assert_eq!(compare_versions("0.0.9", "0.1.0"), Ordering::Less);
    }
}
