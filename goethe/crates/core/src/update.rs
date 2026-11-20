// Signed package updates from GOETHE_PKG_URL: verify everything, then swap both binaries.

use crate::paths::{ensure_layout, goethe_home};
use crate::pyenv::{self, Layer};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

pub struct UpdateOpts {
    pub check_only: bool,
    pub engine: bool,
    pub ternary: bool,
    pub yes: bool,
    pub allow_downgrade: bool,
}

pub(crate) struct Staged {
    pub(crate) rel: String,
    pub(crate) part: PathBuf,
    pub(crate) dest: PathBuf,
}

pub fn run(opts: UpdateOpts) -> Result<(), String> {
    // Fetch and verify the signed manifest, compare versions, stage files, then swap atomically.
    let home = goethe_home();
    ensure_layout(&home).map_err(|e| e.to_string())?;
    let _ = crate::setup::migrate_retired_pkg_env(&home);
    let pkg = pkg_base_url();
    let tmp = staging_dir(&home, ".update-staging")?;
    let result = update_in(&home, &pkg, &tmp, &opts);
    let _ = fs::remove_dir_all(&tmp);
    result
}

pub(crate) fn staging_dir(home: &Path, prefix: &str) -> Result<PathBuf, String> {
    // A private folder for this run alone, so concurrent runs never touch each other's verified files; folders of runs whose process is gone (and the old shared one) are swept first.
    for e in fs::read_dir(home).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix(prefix) else { continue };
        let pid = rest.strip_prefix('-').and_then(|r| r.split('-').next()).and_then(|p| p.parse::<i32>().ok());
        if rest.is_empty() || pid.is_some_and(|p| !crate::sys::exists(p)) {
            let _ = fs::remove_dir_all(e.path());
        }
    }
    crate::paths::private_dir_in(home, prefix)
}

fn update_in(home: &Path, pkg: &str, tmp: &Path, opts: &UpdateOpts) -> Result<(), String> {
    let sums = fetch_signed_sums(pkg, tmp)?;
    let version_file = fetch_verified(pkg, &sums, "VERSION", tmp)?;
    let remote_ver = fs::read_to_string(&version_file).map_err(|e| e.to_string())?.trim().to_string();
    let local_ver = local_version();
    let order = compare_versions(&remote_ver, &local_ver);
    let arch = normalize_arch();
    let (bins, kept) = binaries_due(order, opts, &local_ver, &remote_ver, || binaries_differ(home, &sums, &arch))?;
    if let Some(note) = &kept {
        println!("{note}");
    }
    let want_engines = (opts.engine || pyenv::engine_installed(home).is_some() || pyenv::has(home, Layer::Sglang)) && !cfg!(target_os = "macos");
    let want_ternary = opts.ternary || (want_engines && ternary_installed(home));
    let (sglang, engine) = if want_engines { pyenv_deltas(home, pkg, &sums, tmp)? } else { (None, None) };
    let ternary = if want_ternary { ternary_delta(home, &sums) } else { None };
    let buildinfo = fetch_verified(pkg, &sums, "BUILDINFO", tmp).ok().and_then(|p| fs::read_to_string(p).ok()).unwrap_or_default();
    let nginx = nginx_delta(home, &sums, &arch, buildinfo_value(&buildinfo, "nginx_version"));
    let published = published_label(&remote_ver, bins && order == Ordering::Equal);
    let installed = if bins { local_ver.clone() } else { published.clone() };
    let parts = [
        ("goethe", kept.is_none().then(|| Delta::of(&installed, &published, bins))),
        ("nginx", nginx),
        ("SGLang", sglang),
        ("goethe-engine", engine),
        ("ternary pack", ternary),
    ];
    let pending: Vec<(&str, Delta)> = parts.into_iter().filter_map(|(what, d)| d.map(|d| (what, d))).collect();
    for (what, d) in &pending {
        println!("{}", d.line(what));
    }
    if opts.check_only || pending.iter().all(|(_, d)| !d.differs) {
        println!("{}", check_summary(&pending));
        return Ok(());
    }
    let what = pending.iter().filter(|(_, d)| d.differs).map(|(w, _)| *w).collect::<Vec<_>>().join(", ");
    let question = if bins { format!("Update goethe {installed} -> {published}?") } else { format!("Update {what}?") };
    crate::host::confirm(&question, opts.yes)?;
    let mut staged = Vec::new();
    for bin in ["goethe", "goethed"].into_iter().filter(|_| bins) {
        let rel = format!("bin/{}/{bin}", platform_dir(&arch));
        staged.push(stage(pkg, &sums, &rel, tmp, &home.join("bin").join(bin))?);
    }
    let differs = |what: &str| pending.iter().any(|(w, d)| *w == what && d.differs);
    if differs("nginx") {
        let rel = format!("bin/{}/nginx", platform_dir(&arch));
        staged.push(stage(pkg, &sums, &rel, tmp, &home.join("nginx-root/usr/sbin/nginx"))?);
    }
    if differs("SGLang") {
        pyenv::sglang_wheel(home, pkg, &sums, tmp)?;
    }
    let ternary = if differs("ternary pack") { stage_ternary(pkg, &sums, tmp)? } else { None };
    let was_running = crate::service::goethed_running();
    swap_all(&staged)?;
    if bins {
        crate::service::refresh_unit(home);
    }
    if differs("SGLang") || differs("goethe-engine") {
        let _lock = pyenv::lock(home)?;
        for (what, layer) in [("SGLang", Layer::Sglang), ("goethe-engine", Layer::Engine)] {
            if differs(what) && !(bins && layer == Layer::Engine) {
                pyenv::install_from(home, pkg, &sums, tmp, layer).map_err(|e| format!("{e}; goethe update engine retries"))?;
            }
        }
    }
    if let Some(t) = ternary {
        unpack_ternary(home, &t)?;
        let sha = lookup_sum(&sums, TERNARY_REL)?;
        fs::write(home.join(TERNARY_RECORD), format!("{sha}\n")).map_err(|e| e.to_string())?;
    }
    let mut engine_ok = !(bins && pyenv::engine_installed(home).is_some() && differs("goethe-engine"));
    if bins && pyenv::engine_installed(home).is_some() && differs("goethe-engine") {
        engine_ok = Command::new(home.join("bin/goethe"))
            .args(["update", "engine", "--yes"])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !engine_ok {
            eprintln!("note: goethe-engine was not brought to the new pin; run goethe update engine");
        }
    }
    if bins && engine_ok {
        fs::write(home.join("VERSION"), format!("{remote_ver}\n")).map_err(|e| e.to_string())?;
    }
    if was_running && bins {
        restart_goethed(home)?;
    }
    if differs("nginx") && crate::nginx::master_running(home) {
        let _ = crate::nginx::reload(home);
    }
    println!("{}", if bins { format!("updated to goethe {published}") } else { format!("updated {what}") });
    Ok(())
}

fn binaries_due(order: Ordering, opts: &UpdateOpts, local: &str, remote: &str, differ: impl FnOnce() -> Result<bool, String>) -> Result<(bool, Option<String>), String> {
    // Whether goethe and goethed are replaced, and the note when they are kept. The version guard is theirs alone: a published goethe older than this build is kept, and only --allow-downgrade without engine/ternary installs it; engines and packs go on whatever the binaries do.
    match order {
        Ordering::Equal => Ok((differ()?, None)),
        Ordering::Greater => Ok((true, None)),
        Ordering::Less if opts.allow_downgrade && !opts.engine && !opts.ternary => Ok((true, None)),
        Ordering::Less => Ok((false, Some(format!("goethe {local} is newer than the published {remote}; keeping it")))),
    }
}

fn binaries_differ(home: &Path, sums: &Path, arch: &str) -> Result<bool, String> {
    // Another build at the same version: the installed bin/goethe or bin/goethed differs from the signed entry.
    for bin in ["goethe", "goethed"] {
        let want = lookup_sum(sums, &format!("bin/{}/{bin}", platform_dir(arch)))?;
        if sha256_file(&home.join("bin").join(bin)).ok().as_deref() != Some(want.as_str()) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn nginx_delta(home: &Path, sums: &Path, arch: &str, published_version: Option<&str>) -> Option<Delta> {
    // Bundled nginx when the package ships one and the on-disk binary differs (or is missing), named by version where both sides know it.
    let rel = format!("bin/{}/nginx", platform_dir(arch));
    let want = lookup_sum(sums, &rel).ok()?;
    let dest = home.join("nginx-root/usr/sbin/nginx");
    let have = sha256_file(&dest).ok();
    let version = crate::nginx::nginx_version(&dest).map(|(a, b, c)| format!("{a}.{b}.{c}"));
    let (installed, published) = nginx_labels(have.as_deref(), version.as_deref(), &want, published_version);
    Some(Delta { installed: Some(installed), published, differs: have.as_deref() != Some(want.as_str()) })
}

fn nginx_labels(have_sha: Option<&str>, have_version: Option<&str>, want_sha: &str, want_version: Option<&str>) -> (String, String) {
    // Versions when both are known ("1.30.6" -> "1.30.7", or "another build" of the same version), else 12-character hashes.
    let short = |s: &str| s.chars().take(12).collect::<String>();
    let Some(have_sha) = have_sha else {
        return ("missing".to_string(), want_version.map_or_else(|| short(want_sha), str::to_string));
    };
    match (have_version, want_version) {
        (Some(h), _) if have_sha == want_sha => (h.to_string(), h.to_string()),
        (Some(h), Some(w)) if h == w => (h.to_string(), format!("{w} (another build)")),
        (Some(h), Some(w)) => (h.to_string(), w.to_string()),
        _ => (short(have_sha), short(want_sha)),
    }
}

fn buildinfo_value<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    text.lines().find_map(|l| l.trim().strip_prefix(key)?.strip_prefix('=').map(str::trim)).filter(|v| !v.is_empty())
}

fn check_summary(pending: &[(&str, Delta)]) -> String {
    // The last line of goethe update --check: what would change, or that nothing would.
    let due: Vec<&str> = pending.iter().filter(|(_, d)| d.differs).map(|(w, _)| *w).collect();
    match due.len() {
        0 => "everything is up to date".to_string(),
        1 => format!("1 update available ({}): run goethe update", due[0]),
        n => format!("{n} updates available ({}): run goethe update", due.join(", ")),
    }
}

fn published_label(version: &str, rebuilt: bool) -> String {
    // The published build's label; another build of the installed version says so.
    if rebuilt { format!("{version} (another build)") } else { version.to_string() }
}

const TERNARY_REL: &str = "goethe/ternary-kernels.tar.zst";
const TERNARY_RECORD: &str = "goethe/ternary-kernels.sha256";

pub fn ternary_installed(home: &Path) -> bool {
    // Whether the ternary runtime was ever installed here, so goethe update engine refreshes it too.
    home.join(TERNARY_RECORD).exists()
}

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
        if self.differs && installed == self.published {
            format!("update available: {what} {installed} -> another build of it")
        } else if self.differs {
            format!("update available: {what} {installed} -> {}", self.published)
        } else {
            format!("{what} {installed} is up to date")
        }
    }
}

fn pyenv_deltas(home: &Path, base: &str, sums: &Path, tmp: &Path) -> Result<(Option<Delta>, Option<Delta>), String> {
    // The py-env layers that are present, against what is published; an absent layer is never installed here.
    if !pyenv::has(home, Layer::Base) {
        println!("no Python environment yet; goethe install sets it up when a model needs it");
        return Ok((None, None));
    }
    let sglang = if pyenv::has(home, Layer::Sglang) { Some(engine_delta(home, base, sums, tmp)?) } else { None };
    let engine = pyenv::engine_installed(home).map(|v| Delta {
        differs: !pyenv::has(home, Layer::Engine),
        installed: Some(v),
        published: pyenv::ENGINE_VERSION.to_string(),
    });
    if sglang.is_none() {
        println!("SGLang is not installed; it installs with the first model that needs it");
    }
    if engine.is_none() {
        println!("goethe-engine is not installed; it installs with the first kernel pack");
    }
    Ok((sglang, engine))
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
        .unwrap_or_else(|| "https://pkg.goethe.sh".to_string())
        .trim_end_matches('/')
        .to_string()
}

pub(crate) fn plain_http_refusal(url: &str) -> Option<String> {
    // Why a plain http package, registry or token URL is refused: only a loopback host, or a test host GOETHE_ALLOW_HTTP=1 names, may skip TLS.
    let rest = url.strip_prefix("http://")?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    let name = match host.strip_prefix('[') {
        Some(v6) => v6.split(']').next().unwrap_or(""),
        None => host.split(':').next().unwrap_or(""),
    };
    let loopback = name.eq_ignore_ascii_case("localhost") || name.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
    (!loopback && std::env::var("GOETHE_ALLOW_HTTP").as_deref() != Ok("1"))
        .then(|| format!("{url}: plain http is refused for packages, the registry and the package token; use https, file:// or a loopback address (GOETHE_ALLOW_HTTP=1 allows a test host)"))
}

pub fn newer_published() -> Option<String> {
    // The published version when it is newer than this goethe; None when equal, older, over plain http or unreachable within 2 s.
    let url = format!("{}/VERSION", pkg_base_url());
    if plain_http_refusal(&url).is_some() {
        return None;
    }
    let mut req = crate::net::agent().get(&url).timeout(std::time::Duration::from_secs(2));
    if let Some(tok) = crate::paths::setting("GOETHE_PKG_TOKEN") {
        req = req.set("Authorization", &format!("Bearer {}", tok.trim()));
    }
    let remote = req.call().ok()?.into_string().ok()?.trim().to_string();
    let valid = !remote.is_empty() && remote.len() < 40 && remote.chars().all(|c| c.is_ascii_alphanumeric() || ".-+".contains(c));
    (valid && compare_versions(&remote, &local_version()) == Ordering::Greater).then_some(remote)
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

pub fn platform_dir(arch: &str) -> String {
    // The package folder of this system's binaries: darwin-aarch64 on a Mac, linux-<arch> elsewhere.
    if cfg!(target_os = "macos") { format!("darwin-{arch}") } else { format!("linux-{arch}") }
}

pub(crate) fn normalize_arch() -> String {
    let raw = std::env::var("GOETHE_ARCH").unwrap_or_else(|_| std::env::consts::ARCH.to_string());
    match raw.as_str() {
        "arm64" | "aarch64" => "aarch64".to_string(),
        "x86_64" | "amd64" => "x86_64".to_string(),
        other => other.to_string(),
    }
}

const MAX_PKG_FILE: u64 = 32 << 30;
const MAX_PKG_TEXT: u64 = 16 << 20;

fn fetch_pkg_file(base: &str, rel: &str, dest: &Path) -> Result<(), String> {
    // Package lists, versions and lock files are small text; only binaries, wheels and archives may be large.
    let text = matches!(rel, "SHA256SUMS" | "SHA256SUMS.sig" | "VERSION" | "BUILDINFO" | "wheels/LATEST") || rel.ends_with(".lock") || rel.ends_with(".sh");
    fetch_pkg_file_max(base, rel, dest, if text { MAX_PKG_TEXT } else { MAX_PKG_FILE }, Duration::from_secs(60))
}

fn fetch_pkg_file_max(base: &str, rel: &str, dest: &Path, max: u64, idle: Duration) -> Result<(), String> {
    // GET one CDN file into dest, streamed (never held in memory) and refused past `max` bytes; it fails when the host goes quiet for `idle`, never for being slow. GOETHE_PKG_TOKEN goes as a bearer token when set; a file:// package is read from disk.
    let url = format!("{base}/{rel}");
    if let Some(path) = url.strip_prefix("file://") {
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let len = fs::metadata(path).map_err(|e| format!("fetch {rel}: {path}: {e}"))?.len();
        if len > max {
            return Err(format!("fetch {rel}: the file is longer than {max} bytes; refused"));
        }
        return fs::copy(path, dest).map(|_| ()).map_err(|e| format!("fetch {rel}: {path}: {e}"));
    }
    if let Some(why) = plain_http_refusal(&url) {
        return Err(format!("fetch {rel}: {why}"));
    }
    let mut req = crate::net::agent_builder().timeout_read(idle).build().get(&url);
    if let Some(tok) = crate::paths::setting("GOETHE_PKG_TOKEN") {
        req = req.set("Authorization", &format!("Bearer {}", tok.trim()));
    }
    let resp = req.call().map_err(|e| format!("fetch {rel}: {e}"))?;
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut f = fs::File::create(dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    let got = std::io::copy(&mut resp.into_reader().take(max + 1), &mut f).map_err(|e| format!("fetch {rel}: {e}"))?;
    if got > max {
        let _ = fs::remove_file(dest);
        return Err(format!("fetch {rel}: the response is longer than {max} bytes; refused"));
    }
    Ok(())
}

const PROPAGATING: &str = "a new release may still be propagating through the package host's cache (up to 10 minutes): run goethe update again in a few minutes, nothing was changed";

pub(crate) fn fetch_signed_sums(pkg: &str, tmp: &Path) -> Result<PathBuf, String> {
    // SHA256SUMS and its signature into tmp, checked against the release key compiled into this goethe.
    fetch_pkg_file(pkg, "SHA256SUMS", &tmp.join("SHA256SUMS"))?;
    fetch_pkg_file(pkg, "SHA256SUMS.sig", &tmp.join("SHA256SUMS.sig"))?;
    let (sums, sig) = (fs::read(tmp.join("SHA256SUMS")).map_err(|e| e.to_string())?, fs::read(tmp.join("SHA256SUMS.sig")).map_err(|e| e.to_string())?);
    crate::release_key::verify("the package list", &sums, &sig).map_err(|e| {
        let fresh = " If a release was published minutes ago, its files may still be reaching the package host's cache (up to 10 minutes): run the command again in a few minutes; nothing was changed.";
        if e.ends_with(crate::release_key::MISMATCH) { format!("{e}.{fresh}") } else { e }
    })?;
    Ok(tmp.join("SHA256SUMS"))
}

pub(crate) fn lookup_sum(sums_path: &Path, rel: &str) -> Result<String, String> {
    let text = fs::read_to_string(sums_path).map_err(|e| e.to_string())?;
    text.lines()
        .find_map(|line| {
            let mut parts = line.split_whitespace();
            let hash = parts.next()?;
            (parts.next()? == rel).then(|| hash.to_string())
        })
        .ok_or_else(|| format!("SHA256SUMS missing entry for {rel}"))
}

pub(crate) fn fetch_verified(base: &str, sums: &Path, rel: &str, tmp: &Path) -> Result<PathBuf, String> {
    // Download rel into staging and require the hash listed in the signed SHA256SUMS.
    let expected = lookup_sum(sums, rel)?;
    let part = tmp.join(rel.replace('/', "__"));
    fetch_pkg_file(base, rel, &part)?;
    if sha256_file(&part)? != expected {
        return Err(format!("checksum mismatch for {rel}; {PROPAGATING}"));
    }
    Ok(part)
}

pub(crate) fn stage(base: &str, sums: &Path, rel: &str, tmp: &Path, dest: &Path) -> Result<Staged, String> {
    Ok(Staged {
        rel: rel.to_string(),
        part: fetch_verified(base, sums, rel, tmp)?,
        dest: dest.to_path_buf(),
    })
}

pub(crate) fn swap_all(staged: &[Staged]) -> Result<(), String> {
    // Move every verified file into place, keeping .prev copies; on the first failure every file this call moved is put back, and one that had no previous file is removed.
    let restore = |s: &Staged, moved: bool| {
        let _ = if moved { fs::rename(s.dest.with_extension("prev"), &s.dest) } else { fs::remove_file(&s.dest) };
    };
    let mut done: Vec<(&Staged, bool)> = Vec::new();
    for s in staged {
        let staged_dest = s.dest.with_extension("new");
        let mut moved = false;
        let r = (|| {
            if let Some(parent) = s.dest.parent() {
                fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
            fs::copy(&s.part, &staged_dest).map_err(|e| format!("{}: {e}", staged_dest.display()))?;
            fs::set_permissions(&staged_dest, std::os::unix::fs::PermissionsExt::from_mode(0o755))
                .map_err(|e| e.to_string())?;
            if s.dest.exists() {
                fs::rename(&s.dest, s.dest.with_extension("prev")).map_err(|e| e.to_string())?;
                moved = true;
            }
            fs::rename(&staged_dest, &s.dest).map_err(|e| e.to_string())
        })();
        if let Err(e) = r {
            let _ = fs::remove_file(&staged_dest);
            if moved {
                restore(s, true);
            }
            for (d, m) in done.iter().rev() {
                restore(d, *m);
            }
            return Err(format!("installing {} failed ({e}); previous files restored", s.rel));
        }
        done.push((s, moved));
    }
    Ok(())
}

pub(crate) fn sha256_file(path: &Path) -> Result<String, String> {
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
    // Engines and the gateway carry over: stop the old goethed with keep_engines, then start the new one.
    let (before, kept) = crate::service::stop_keeping_engines()?;
    start_upgraded(home, before, kept)
}

pub(crate) fn start_upgraded(home: &Path, before: usize, kept: usize) -> Result<(), String> {
    // The new goethed after a keep-engines stop, by systemd when a unit starts it, then the engines it adopts.
    crate::service::label_programs(home);
    match crate::service::systemctl_restart_args() {
        Some(args) => {
            let ok = Command::new("systemctl").args(&args).status().map_err(|e| e.to_string())?.success();
            if !ok {
                return Err(format!("systemctl {} failed; start goethed with goethe up", args.join(" ")));
            }
            println!("started the new goethed with systemctl {}", args.join(" "));
        }
        None => {
            crate::service::respawn_goethed(home)?;
            println!("started the new goethed");
        }
    }
    crate::service::wait_goethed(Duration::from_secs(30))?;
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut running = crate::service::live_engine_pids().len();
    while running < before && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(500));
        running = crate::service::live_engine_pids().len();
    }
    println!("{}", upgrade_summary(before, kept, running));
    Ok(())
}

fn upgrade_summary(before: usize, kept: usize, running: usize) -> String {
    // What happened to the loaded models across the goethed swap.
    if before == 0 {
        "no models were loaded".to_string()
    } else if kept == before && running >= before {
        format!(
            "the upgrade kept all {before} loaded model instance(s) serving (requests during the few seconds without goethed got a 502)"
        )
    } else {
        format!(
            "{} {running} of {before} loaded instance(s) are RUNNING under the new goethed; goethe ps shows the rest as they come back",
            crate::service::kept_text(before, kept)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_names_nginx_by_version_and_ends_with_a_summary() {
        assert_eq!(nginx_labels(Some("aaaa"), Some("1.30.6"), "bbbb", Some("1.30.7")), ("1.30.6".into(), "1.30.7".into()));
        assert_eq!(nginx_labels(Some("aaaa"), Some("1.30.7"), "bbbb", Some("1.30.7")), ("1.30.7".into(), "1.30.7 (another build)".into()));
        assert_eq!(nginx_labels(Some("aaaaaaaaaaaaaaaa"), None, "bbbbbbbbbbbbbbbb", Some("1.30.7")), ("aaaaaaaaaaaa".into(), "bbbbbbbbbbbb".into()), "an older package names hashes");
        assert_eq!(nginx_labels(None, None, "bbbb", Some("1.30.7")), ("missing".into(), "1.30.7".into()));
        assert_eq!(nginx_labels(Some("aaaa"), Some("1.30.7"), "aaaa", None), ("1.30.7".into(), "1.30.7".into()), "the same file is the same version");
        assert_eq!(Delta::of("w.whl", "w.whl", true).line("SGLang"), "update available: SGLang w.whl -> another build of it");
        let up = Delta::of("1", "1", false);
        let due = |w| Delta { installed: Some("a".into()), published: "b".into(), differs: w };
        assert_eq!(check_summary(&[("goethe", up)]), "everything is up to date");
        assert_eq!(check_summary(&[("goethe", due(true)), ("nginx", due(true)), ("SGLang", due(false))]), "2 updates available (goethe, nginx): run goethe update");
        assert_eq!(buildinfo_value("a=1\nnginx_version=1.30.7\n", "nginx_version"), Some("1.30.7"));
    }

    #[test]
    fn a_file_package_is_read_from_disk() {
        let t = tempfile::tempdir().unwrap();
        fs::create_dir_all(t.path().join("cdn/bin")).unwrap();
        fs::write(t.path().join("cdn/bin/x"), "hi").unwrap();
        let base = format!("file://{}", t.path().join("cdn").display());
        fetch_pkg_file(&base, "bin/x", &t.path().join("out/x")).unwrap();
        assert_eq!(fs::read_to_string(t.path().join("out/x")).unwrap(), "hi");
        assert!(fetch_pkg_file(&base, "missing", &t.path().join("out/y")).unwrap_err().contains("fetch missing"));
    }

    #[test]
    fn a_response_past_the_cap_is_refused_and_not_kept() {
        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming().take(2) {
                let mut s = stream.unwrap();
                let _ = s.read(&mut [0u8; 2048]);
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 3000000\r\nConnection: close\r\n\r\n");
                let _ = s.write_all(&vec![b'A'; 3_000_000]);
            }
        });
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("f");
        let err = fetch_pkg_file_max(&url, "f", &dest, 1 << 20, Duration::from_secs(5)).unwrap_err();
        assert!(err.contains("longer than"), "{err}");
        assert!(!dest.exists());
        fetch_pkg_file_max(&url, "f", &dest, 4 << 20, Duration::from_secs(5)).unwrap();
        assert_eq!(fs::metadata(&dest).unwrap().len(), 3_000_000);
    }

    #[test]
    fn a_slow_download_that_keeps_arriving_is_not_cut_off_but_a_silent_host_is() {
        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for (i, stream) in listener.incoming().take(2).enumerate() {
                let mut s = stream.unwrap();
                let _ = s.read(&mut [0u8; 2048]);
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 40\r\nConnection: close\r\n\r\n");
                for _ in 0..4 {
                    let _ = s.write_all(&[b'A'; 10]);
                    let _ = s.flush();
                    std::thread::sleep(Duration::from_millis(if i == 0 { 600 } else { 3000 }));
                }
            }
        });
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("f");
        fetch_pkg_file_max(&url, "f", &dest, 1 << 20, Duration::from_secs(1)).unwrap();
        assert_eq!(fs::metadata(&dest).unwrap().len(), 40, "1.8 s in total with a 1 s idle limit");
        assert!(fetch_pkg_file_max(&url, "f", &dest, 1 << 20, Duration::from_secs(1)).is_err(), "3 s of silence");
    }

    fn opts(engine: bool, ternary: bool, allow_downgrade: bool) -> UpdateOpts {
        UpdateOpts { check_only: false, engine, ternary, yes: true, allow_downgrade }
    }

    #[test]
    fn a_build_ahead_of_the_published_release_keeps_its_binaries_and_still_updates_what_was_asked() {
        let due = |o: &UpdateOpts, order| binaries_due(order, o, "0.3.3", "0.3.2", || panic!("only an equal version compares the binaries")).map(|(bins, note)| (bins, note.is_some()));
        assert_eq!(due(&opts(false, false, false), Ordering::Less), Ok((false, true)), "no flags: the note, nothing replaced, no error");
        assert_eq!(due(&opts(true, false, false), Ordering::Less), Ok((false, true)), "update engine: the binaries stay, the engine layer is still looked at");
        assert_eq!(due(&opts(false, true, false), Ordering::Less), Ok((false, true)), "ternary likewise");
        assert_eq!(due(&opts(true, false, true), Ordering::Less), Ok((false, true)), "--allow-downgrade never turns update engine into a binary downgrade");
        assert_eq!(due(&opts(false, false, true), Ordering::Less), Ok((true, false)), "--allow-downgrade alone installs the older published binaries");
        assert_eq!(due(&opts(true, false, false), Ordering::Greater), Ok((true, false)), "an older build still updates");
        let note = binaries_due(Ordering::Less, &opts(false, false, false), "0.3.3", "0.3.2", || Ok(false)).unwrap().1.unwrap();
        assert_eq!(note, "goethe 0.3.3 is newer than the published 0.3.2; keeping it");
        assert_eq!(binaries_due(Ordering::Equal, &opts(true, false, false), "0.3.3", "0.3.3", || Ok(true)), Ok((true, None)), "another build at the same version");
        assert_eq!(binaries_due(Ordering::Equal, &opts(false, false, false), "0.3.3", "0.3.3", || Ok(false)), Ok((false, None)));
    }

    #[test]
    fn a_failed_swap_puts_back_what_it_moved_and_never_a_stale_prev() {
        let t = tempfile::tempdir().unwrap();
        let p = t.path();
        let staged = |rel: &str, dest: PathBuf| {
            fs::write(p.join(format!("{rel}.part")), format!("new {rel}")).unwrap();
            Staged { rel: rel.to_string(), part: p.join(format!("{rel}.part")), dest }
        };
        fs::write(p.join("goethe"), "old goethe").unwrap();
        fs::write(p.join("nginx.prev"), "stale nginx").unwrap();
        fs::write(p.join("blocker"), "a file where a folder must be").unwrap();
        let all = [staged("goethe", p.join("goethe")), staged("nginx", p.join("nginx")), staged("goethed", p.join("blocker/goethed"))];
        assert!(swap_all(&all).unwrap_err().contains("installing goethed failed"));
        assert_eq!(fs::read_to_string(p.join("goethe")).unwrap(), "old goethe");
        assert!(!p.join("nginx").exists(), "a file that was not there before is removed, not replaced by an older .prev");
        assert_eq!(fs::read_to_string(p.join("nginx.prev")).unwrap(), "stale nginx");
        assert!(!p.join("goethe.new").exists() && !p.join("nginx.new").exists());
        swap_all(&all[..2]).unwrap();
        assert_eq!(fs::read_to_string(p.join("goethe")).unwrap(), "new goethe");
        assert_eq!(fs::read_to_string(p.join("goethe.prev")).unwrap(), "old goethe");
    }

    #[test]
    fn each_run_stages_in_its_own_folder_and_dead_runs_are_swept() {
        use std::os::unix::fs::PermissionsExt;
        let t = tempfile::tempdir().unwrap();
        let home = t.path();
        let dead = home.join(".update-staging-999999999-1-0");
        let live = home.join(format!(".update-staging-{}-1-0", std::process::id()));
        for d in [&home.join(".update-staging"), &dead, &live, &home.join(".setup-staging")] {
            fs::create_dir_all(d.join("bin")).unwrap();
        }
        let (a, b) = (staging_dir(home, ".update-staging").unwrap(), staging_dir(home, ".update-staging").unwrap());
        assert!(a != b && a.is_dir() && b.is_dir(), "two runs never share a folder");
        assert!(!home.join(".update-staging").exists() && !dead.exists(), "the old shared folder and a dead run's folder go");
        assert!(live.exists() && home.join(".setup-staging").exists(), "a running run's folder and other prefixes stay");
        assert_eq!(fs::metadata(&a).unwrap().permissions().mode() & 0o777, 0o700);
    }

    #[test]
    fn plain_http_is_refused_unless_the_host_is_this_machine() {
        let refused = |u: &str| plain_http_refusal(u).is_some();
        for ok in ["https://pkg.goethe.sh", "file:///srv/cdn", "http://127.0.0.1:8999/a", "http://localhost/x", "http://[::1]:80/", "http://127.0.0.2"] {
            assert!(!refused(ok), "{ok}");
        }
        for bad in ["http://pkg.goethe.sh", "http://10.0.2.2:8999/a", "http://127.0.0.1.evil.example/", "http://127.0.0.1@evil.example/", "http://[fe80::1]/"] {
            assert!(refused(bad), "{bad}");
        }
        let err = fetch_pkg_file("http://pkg.example", "SHA256SUMS", Path::new("/nonexistent/x")).unwrap_err();
        assert!(err.contains("plain http is refused"), "{err}");
    }

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
    fn upgrade_summary_says_what_kept_serving() {
        assert_eq!(upgrade_summary(0, 0, 0), "no models were loaded");
        assert!(upgrade_summary(2, 2, 2).starts_with("the upgrade kept all 2 loaded model instance(s) serving"));
        let lost = upgrade_summary(2, 0, 1);
        assert!(lost.contains("0 of 2 engine(s) still run") && lost.contains("1 of 2 loaded instance(s) are RUNNING"), "{lost}");
    }

    #[test]
    fn builds_are_compared_by_binary_and_commit() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        fs::create_dir_all(home.join("bin")).unwrap();
        fs::write(home.join("bin/goethe"), b"cli").unwrap();
        fs::write(home.join("bin/goethed"), b"daemon").unwrap();
        let sha = |b: &[u8]| {
            let p = home.join("probe");
            fs::write(&p, b).unwrap();
            sha256_file(&p).unwrap()
        };
        let sums = home.join("SHA256SUMS");
        let dir = platform_dir("x86_64");
        fs::write(&sums, format!("{}  bin/{dir}/goethe\n{}  bin/{dir}/goethed\n", sha(b"cli"), sha(b"daemon"))).unwrap();
        assert!(!binaries_differ(home, &sums, "x86_64").unwrap());
        fs::write(home.join("bin/goethed"), b"older daemon").unwrap();
        assert!(binaries_differ(home, &sums, "x86_64").unwrap());
        assert!(binaries_differ(home, &sums, "aarch64").is_err(), "no signed entry for that arch");
        assert_eq!(published_label("0.1.0", true), "0.1.0 (another build)");
        assert_eq!(published_label("0.2.0", false), "0.2.0");
        let d = Delta::of("0.1.0", &published_label("0.1.0", true), true);
        assert_eq!(d.line("goethe"), "update available: goethe 0.1.0 -> 0.1.0 (another build)");
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

    #[test]
    fn sha256_file_hex_digest() {
        let t = tempfile::tempdir().unwrap();
        let p = t.path().join("blob");
        fs::write(&p, b"goethe-sha256-equiv").unwrap();
        let a = sha256_file(&p).unwrap();
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(sha256_file(&t.path().join("missing")).is_err());
    }
}
