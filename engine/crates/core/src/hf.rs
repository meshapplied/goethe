// Hugging Face download and verify for goethe install; goethe never logs in or sends credentials.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

const REVISION_FILE: &str = ".goethe_hf_revision";
const VERIFIED_FILE: &str = ".goethe_verified.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HfSibling {
    pub rfilename: String,
    pub size: u64,
    #[serde(default)]
    pub lfs_sha256: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelPreflight {
    pub gated: bool,
    pub siblings: Vec<HfSibling>,
    pub sha: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VerifiedCache {
    revision: String,
    files: HashMap<String, VerifiedEntry>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct VerifiedEntry {
    size: u64,
    mtime_ns: i64,
    sha256: String,
}

pub struct DownloadOpts {
    pub revision: Option<String>,
    pub force: bool,
    pub no_verify: bool,
    pub only: Option<Vec<String>>,
}

pub fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string())
}

pub fn offline_mode() -> bool {
    std::env::var("HF_HUB_OFFLINE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

static TOKEN: Mutex<Option<String>> = Mutex::new(None);

pub fn set_token(token: Option<String>) {
    // A Hugging Face token for this process only (gated or private repos); it is never written to disk.
    let value = token.filter(|t| !t.trim().is_empty()).map(|t| t.trim().to_string());
    TOKEN.lock().unwrap_or_else(|e| e.into_inner()).clone_from(&value);
}

pub fn token_set() -> bool {
    token().is_some()
}

fn token() -> Option<String> {
    TOKEN.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

pub fn whoami(token: &str) -> Result<String, String> {
    // The account a token belongs to, which proves Hugging Face accepts it.
    let url = format!("{}/api/whoami-v2", hf_endpoint());
    let resp = match crate::net::agent().get(&url).set("Authorization", &format!("Bearer {}", token.trim())).timeout(Duration::from_secs(15)).call() {
        Ok(r) => r,
        Err(ureq::Error::Status(401, _)) => return Err("Hugging Face did not accept this token".to_string()),
        Err(e) => return Err(format!("cannot reach Hugging Face: {e}")),
    };
    let text = resp.into_string().map_err(|e| format!("Hugging Face reply: {e}"))?;
    let body: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("Hugging Face reply: {e}"))?;
    Ok(body.get("name").and_then(|v| v.as_str()).unwrap_or("your account").to_string())
}

pub fn preflight_model(model_id: &str, revision: Option<&str>) -> Result<ModelPreflight, String> {
    preflight_model_opts(model_id, revision, false)
}

pub const GATED_WAYS: &str = "accept the license, then goethe install ID --hf-token, or goethe install DIR --name ID for weights you already have";

pub fn gated_guidance(model_id: &str) -> String {
    format!("{model_id} {}: {}", crate::messages::GATED, GATED_WAYS.replace("ID", model_id))
}

fn status_refusal(model_id: &str, status: u16, has_token: bool) -> Option<String> {
    // An unauthenticated 401, 403 or 404 means no such repo or a private one (a gated repo answers 200 with its gated flag); with a token, no access.
    match (status, has_token) {
        (401 | 403 | 404, true) => Some(format!("{model_id}: not found, or this Hugging Face token has no access (accept the license at {}/{model_id})", hf_endpoint())),
        (401 | 403 | 404, false) => Some(format!("{model_id}: {} (check the id; add --hf-token for a private model)", crate::messages::NOT_FOUND)),
        _ => None,
    }
}

pub fn preflight_model_opts(model_id: &str, revision: Option<&str>, allow_gated: bool) -> Result<ModelPreflight, String> {
    // Model API query (at a revision when given), with the session token if one is set: gating, files, sizes, commit.
    if offline_mode() {
        return Ok(ModelPreflight {
            gated: false,
            siblings: Vec::new(),
            sha: None,
        });
    }
    let endpoint = hf_endpoint();
    let url = match revision {
        Some(rev) => format!("{endpoint}/api/models/{model_id}/revision/{rev}?blobs=true"),
        None => format!("{endpoint}/api/models/{model_id}?blobs=true"),
    };
    let resp = api_get(&url)?;
    let allow_gated = allow_gated || token().is_some();
    if let Some(e) = status_refusal(model_id, resp.status, token().is_some()) {
        return Err(e);
    }
    match resp.status {
        s if s >= 500 => return Err(format!("cannot reach {endpoint}: HTTP {s}")),
        s if s >= 400 => return Err(format!("HF API error for {model_id}: HTTP {s}")),
        _ => {}
    }
    let body: serde_json::Value =
        serde_json::from_str(&resp.body).map_err(|e| format!("HF API parse failed: {e}"))?;
    let gated = body
        .get("gated")
        .map(|v| v.as_bool().unwrap_or(false) || v.as_str().is_some_and(|s| !s.is_empty()))
        .unwrap_or(false);
    if gated && !allow_gated {
        return Err(gated_guidance(model_id));
    }
    Ok(ModelPreflight {
        gated,
        siblings: parse_siblings(&body),
        sha: body.get("sha").and_then(|v| v.as_str()).map(String::from),
    })
}

pub fn hf_owner_org(model_id: &str) -> Option<String> {
    model_id.split('/').next().map(String::from)
}

pub fn is_hf_organization(owner: &str) -> Result<bool, String> {
    if offline_mode() {
        return Ok(true);
    }
    let url = format!("{}/api/organizations/{owner}/overview", hf_endpoint());
    let resp = api_get(&url)?;
    Ok(resp.status == 200)
}

pub fn has_weight_files(siblings: &[HfSibling]) -> bool {
    siblings.iter().any(|s| {
        !s.rfilename.starts_with("original/")
            && (s.rfilename.ends_with(".safetensors")
                || (s.rfilename.starts_with("pytorch_model") && s.rfilename.ends_with(".bin")))
    })
}

pub fn download_excludes(siblings: &[HfSibling]) -> Vec<String> {
    // Skip original/ checkpoints, non-safetensors duplicates, consolidated copies of sharded weights, and goethe's own files: a repo never writes the card or the verify cache.
    let names: Vec<&str> = siblings.iter().map(|s| s.rfilename.as_str()).collect();
    let mut ex = vec!["original/*".to_string(), "goethe_card.toml*".to_string(), ".goethe_*".to_string()];
    let has_st = names
        .iter()
        .any(|n| n.ends_with(".safetensors") && !n.starts_with("original/"));
    if has_st {
        for p in ["*.bin", "*.pth", "*.h5", "*.msgpack", "*.gguf", "*.onnx", "*.ckpt"] {
            ex.push(p.to_string());
        }
    }
    let shards = names.iter().any(|n| n.starts_with("model") && n.ends_with(".safetensors"));
    if shards && names.iter().any(|n| n.starts_with("consolidated") && n.ends_with(".safetensors")) {
        ex.push("consolidated*.safetensors".to_string());
    }
    ex
}

pub fn glob_match(pattern: &str, name: &str) -> bool {
    // fnmatch-style match where '*' spans any characters, including '/'.
    let Some((head, rest)) = pattern.split_once('*') else {
        return pattern == name;
    };
    let Some(mut tail) = name.strip_prefix(head) else {
        return false;
    };
    let parts: Vec<&str> = rest.split('*').collect();
    for (i, part) in parts.iter().enumerate() {
        if i == parts.len() - 1 {
            return tail.ends_with(part);
        }
        match tail.find(part) {
            Some(pos) => tail = &tail[pos + part.len()..],
            None => return false,
        }
    }
    true
}

fn wanted<'a>(siblings: &'a [HfSibling], excludes: &[String]) -> Vec<&'a HfSibling> {
    siblings
        .iter()
        .filter(|s| !excludes.iter().any(|p| glob_match(p, &s.rfilename)))
        .collect()
}

pub fn download_bytes(siblings: &[HfSibling]) -> u64 {
    wanted(siblings, &download_excludes(siblings)).iter().map(|s| s.size).sum()
}

pub fn weight_bytes_from_siblings(model_id: &str, revision: &str, siblings: &[HfSibling]) -> Result<u64, String> {
    // Bytes of the files the safetensors index names, else of every downloaded .safetensors file.
    let size_of: HashMap<&str, u64> = siblings.iter().map(|s| (s.rfilename.as_str(), s.size)).collect();
    if size_of.contains_key("model.safetensors.index.json") {
        let index = fetch_repo_json(model_id, revision, "model.safetensors.index.json")?;
        let files: BTreeSet<&str> = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .map(|m| m.values().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        return Ok(files.iter().map(|f| size_of.get(f).copied().unwrap_or(0)).sum());
    }
    Ok(wanted(siblings, &download_excludes(siblings))
        .iter()
        .filter(|s| s.rfilename.ends_with(".safetensors"))
        .map(|s| s.size)
        .sum())
}

pub fn ensure_model(home: &Path, model_id: &str, local_dir: &Path, opts: &DownloadOpts) -> Result<(), String> {
    // Preflight first (nothing is created for a bad repo), then always resume hf download, then verify.
    let revision = opts.revision.clone().unwrap_or_else(|| "main".to_string());
    let only = opts.only.as_deref();
    let present = |dir: &Path| match only {
        Some(files) => files.iter().all(|f| dir.join(f).is_file()),
        None => dir.join("config.json").exists(),
    };
    if !opts.force && present(local_dir) && is_installed_fast(local_dir, &revision, opts.no_verify) {
        return Ok(());
    }
    if offline_mode() {
        if !present(local_dir) {
            return Err(format!(
                "HF_HUB_OFFLINE=1 but {model_id} is not present under {}",
                local_dir.display()
            ));
        }
        return write_revision(local_dir, &revision);
    }
    let pre = preflight_model(model_id, Some(&revision))?;
    let excludes = match only {
        Some(files) => {
            if let Some(f) = files.iter().find(|f| !pre.siblings.iter().any(|s| &s.rfilename == *f)) {
                return Err(format!("{model_id}: {f} is not in the repository at revision {revision}; nothing was installed"));
            }
            only_excludes(&pre.siblings, files)
        }
        None if !has_weight_files(&pre.siblings) => {
            return Err(format!(
                "{model_id}: repository has no model weight files (no .safetensors at revision {revision}); nothing was installed"
            ));
        }
        None => download_excludes(&pre.siblings),
    };
    let files = wanted(&pre.siblings, &excludes);
    disk_preflight(home, bytes_needed(&files, local_dir))?;
    let existed = local_dir.exists();
    fs::create_dir_all(local_dir).map_err(|e| format!("{}: {e}", local_dir.display()))?;
    let fetched = if native_download() {
        native_with_retries(model_id, local_dir, &revision, &files)
    } else {
        download_with_retries(home, model_id, local_dir, &revision, &excludes, bytes_missing(&files, local_dir))
    };
    if let Err(e) = fetched {
        drop_empty_new_dir(local_dir, existed);
        return Err(e);
    }
    check_present(local_dir, &files)?;
    if opts.no_verify {
        return write_revision(local_dir, &revision);
    }
    verify_model(local_dir, &revision, &files, home, model_id, &excludes)
}

pub fn only_excludes(siblings: &[HfSibling], keep: &[String]) -> Vec<String> {
    // Every repo file but the kept weight files and small root metadata (tokenizer, config, chat template, license).
    let meta = |n: &str| {
        !n.contains('/')
            && [".json", ".jinja", ".txt", ".model", ".md"].iter().any(|x| n.ends_with(x))
    };
    siblings
        .iter()
        .map(|s| s.rfilename.as_str())
        .filter(|n| !keep.iter().any(|k| k == n) && !(meta(n) && siblings.iter().any(|s| s.rfilename == *n && s.size < 64 << 20)))
        .map(String::from)
        .collect()
}

pub fn is_installed_fast(local_dir: &Path, revision: &str, no_verify: bool) -> bool {
    if !local_dir.join("config.json").exists() || read_revision(local_dir).as_deref() != Some(revision) {
        return false;
    }
    no_verify || verify_cache_matches(local_dir)
}

fn bytes_missing(files: &[&HfSibling], local_dir: &Path) -> u64 {
    files
        .iter()
        .filter(|s| fs::metadata(local_dir.join(&s.rfilename)).map(|m| m.len() != s.size).unwrap_or(true))
        .map(|s| s.size)
        .sum()
}

fn bytes_needed(files: &[&HfSibling], local_dir: &Path) -> u64 {
    ((bytes_missing(files, local_dir) as f64) * 1.05) as u64
}

fn check_present(local_dir: &Path, files: &[&HfSibling]) -> Result<(), String> {
    // Every expected file, LFS or not, must exist with the size the Hub reports.
    for s in files {
        let p = local_dir.join(&s.rfilename);
        match fs::metadata(&p) {
            Ok(m) if s.size == 0 || m.len() == s.size => {}
            Ok(m) => {
                return Err(format!(
                    "incomplete download: {} has {} bytes, expected {}; re-run goethe install to resume",
                    s.rfilename,
                    m.len(),
                    s.size
                ))
            }
            Err(_) => return Err(format!("incomplete download: {} is missing; re-run goethe install to resume", s.rfilename)),
        }
    }
    Ok(())
}

pub fn disk_preflight(home: &Path, needed: u64) -> Result<(), String> {
    let free = match std::env::var("GOETHE_TEST_FREE_BYTES") {
        Ok(v) => v.parse().map_err(|_| "invalid GOETHE_TEST_FREE_BYTES".to_string())?,
        Err(_) => {
            let models = home.join("models");
            fs::create_dir_all(&models).map_err(|e| e.to_string())?;
            available_bytes(&models)?
        }
    };
    if needed > free {
        return Err(format!(
            "need {:.2} GB free for download but only {:.2} GB available on models volume",
            needed as f64 / 1e9,
            free as f64 / 1e9
        ));
    }
    Ok(())
}

pub fn native_download() -> bool {
    // goethe's own downloader: always on macOS (no Python there), on Linux only when GOETHE_NATIVE_DOWNLOAD=1.
    cfg!(target_os = "macos") || std::env::var("GOETHE_NATIVE_DOWNLOAD").is_ok_and(|v| v == "1")
}

static FETCHED: AtomicU64 = AtomicU64::new(0);

pub fn resolve_url(model_id: &str, revision: &str, file: &str) -> String {
    let path: Vec<String> = file.split('/').map(|p| p.replace('%', "%25").replace(' ', "%20").replace('#', "%23").replace('?', "%3F")).collect();
    format!("{}/{model_id}/resolve/{revision}/{}", hf_endpoint(), path.join("/"))
}

fn native_with_retries(model_id: &str, local_dir: &Path, revision: &str, files: &[&HfSibling]) -> Result<(), String> {
    // Each missing file over HTTPS, resumed where a partial download stopped; three tries with growing waits.
    let missing: Vec<&HfSibling> = files.iter().copied().filter(|f| fs::metadata(local_dir.join(&f.rfilename)).map(|m| m.len()).ok() != Some(f.size)).collect();
    if missing.is_empty() {
        return Ok(());
    }
    let total: u64 = missing.iter().map(|f| f.size).sum();
    let base = FETCHED.load(Ordering::Relaxed);
    let have: u64 = missing.iter().map(|f| fs::metadata(partial(local_dir, &f.rfilename)).map(|m| m.len()).unwrap_or(0)).sum();
    let _ticker = (total >= 64 << 20).then(|| {
        let done = move || have + FETCHED.load(Ordering::Relaxed).saturating_sub(base);
        crate::progress::Ticker::start("downloading", Some(total), crate::progress::every(), done, crate::progress::eprint_line)
    });
    for f in missing {
        let mut last = String::new();
        for (i, delay) in [2u64, 5, 10].iter().enumerate() {
            match fetch_file(model_id, local_dir, revision, f) {
                Ok(()) => {
                    last.clear();
                    break;
                }
                Err(e) if is_auth_error(&e) => return Err(e),
                Err(e) => {
                    last = e;
                    if i < 2 {
                        thread::sleep(Duration::from_secs(*delay));
                    }
                }
            }
        }
        if !last.is_empty() {
            return Err(last);
        }
    }
    Ok(())
}

fn partial(dir: &Path, file: &str) -> std::path::PathBuf {
    let name = Path::new(file).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    dir.join(Path::new(file).parent().unwrap_or(Path::new(""))).join(format!(".{name}.part"))
}

fn fetch_file(model_id: &str, dir: &Path, revision: &str, f: &HfSibling) -> Result<(), String> {
    // One file into a hidden .part beside it, appended from where it stopped (Range), renamed when complete.
    use std::io::Write;
    let dest = dir.join(&f.rfilename);
    let part = partial(dir, &f.rfilename);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let mut start = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if start > f.size {
        let _ = fs::remove_file(&part);
        start = 0;
    }
    if start < f.size || f.size == 0 {
        let url = resolve_url(model_id, revision, &f.rfilename);
        let agent = crate::net::agent_builder().timeout_connect(Duration::from_secs(20)).timeout_read(Duration::from_secs(60)).build();
        let mut req = agent.get(&url);
        if let Some(t) = token() {
            req = req.set("Authorization", &format!("Bearer {t}"));
        }
        if start > 0 {
            req = req.set("Range", &format!("bytes={start}-"));
        }
        let resp = match req.call() {
            Ok(r) => r,
            Err(ureq::Error::Status(401 | 403, _)) => return Err(format!("{model_id}: {}", crate::messages::GATED)),
            Err(ureq::Error::Status(416, _)) if start == f.size => return finish(&part, &dest),
            Err(ureq::Error::Status(code, _)) => return Err(format!("{}: HTTP {code} from Hugging Face", f.rfilename)),
            Err(e) => return Err(format!("cannot reach huggingface.co (network or proxy): {e}")),
        };
        let resumed = resp.status() == 206;
        let mut out = fs::OpenOptions::new().create(true).append(resumed).write(true).truncate(!resumed).open(&part).map_err(|e| format!("{}: {e}", part.display()))?;
        let mut src = crate::progress::Counted(resp.into_reader());
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = std::io::Read::read(&mut src, &mut buf).map_err(|e| format!("{}: download stopped: {e}", f.rfilename))?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n]).map_err(|e| format!("{}: {e}", part.display()))?;
            FETCHED.fetch_add(n as u64, Ordering::Relaxed);
        }
        out.flush().map_err(|e| e.to_string())?;
    }
    let got = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
    if f.size > 0 && got != f.size {
        return Err(format!("{}: got {got} of {} bytes", f.rfilename, f.size));
    }
    finish(&part, &dest)
}

fn finish(part: &Path, dest: &Path) -> Result<(), String> {
    fs::rename(part, dest).map_err(|e| format!("{}: {e}", dest.display()))
}

fn download_with_retries(home: &Path, model_id: &str, local_dir: &Path, revision: &str, excludes: &[String], total: u64) -> Result<(), String> {
    let delays = [2u64, 5, 10];
    let mut last = String::new();
    for (i, delay) in delays.iter().enumerate() {
        match run_hf_download(home, model_id, local_dir, revision, excludes, total) {
            Ok(()) => return Ok(()),
            Err(e) if is_auth_error(&e) => return Err(e),
            Err(e) => {
                last = e;
                if i + 1 < delays.len() {
                    thread::sleep(Duration::from_secs(*delay));
                }
            }
        }
    }
    Err(last)
}

fn is_auth_error(msg: &str) -> bool {
    msg.contains(crate::messages::GATED) || msg.contains("401") || msg.contains("403")
}

pub fn run_hf_download(home: &Path, model_id: &str, local_dir: &Path, revision: &str, excludes: &[String], total: u64) -> Result<(), String> {
    // hf download resumes partial files and is cheap when everything is already present; every 5 s a line says how far it is.
    let program = hf_binary(home)?;
    let mut cmd = match python_of_script(&program) {
        Some(python) => {
            let mut c = Command::new(python);
            c.arg("-c").arg(HF_QUIET).arg(&program);
            c
        }
        None => Command::new(&program),
    };
    cmd.arg("download")
        .arg(model_id)
        .arg("--local-dir")
        .arg(local_dir)
        .arg("--revision")
        .arg(revision);
    for p in excludes {
        cmd.arg("--exclude").arg(p);
    }
    cmd.env("HF_XET_HIGH_PERFORMANCE", "1");
    cmd.env("HF_HUB_DISABLE_TELEMETRY", "1");
    cmd.env("HF_HUB_VERBOSITY", "error");
    cmd.env("HF_HUB_DISABLE_UPDATE_CHECK", "1");
    cmd.env("PYTHONWARNINGS", "ignore::FutureWarning");
    cmd.env_remove("HF_TOKEN").env_remove("HUGGING_FACE_HUB_TOKEN");
    match token() {
        Some(t) => cmd.env("HF_TOKEN", t),
        None => cmd.env("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1"),
    };
    let base = crate::progress::dir_bytes(local_dir);
    let watched = local_dir.to_path_buf();
    let ticker = (total > 0).then(|| {
        crate::progress::Ticker::start(model_id, Some(total), crate::progress::every(), move || crate::progress::dir_bytes(&watched).saturating_sub(base), crate::progress::eprint_line)
    });
    let outcome = run_guarded(&mut cmd);
    drop(ticker);
    let (status, tail) = outcome.map_err(|e| format!("hf download failed to start: {e}"))?;
    if !status.success() {
        let log = crate::step::log_path().map_or(String::new(), |p| format!("\nfull log: {}", p.display()));
        return Err(format!("{}{log}", download_failure(&status, &tail)));
    }
    Ok(())
}

static DOWNLOAD_PGID: AtomicI32 = AtomicI32::new(0);
const INTERRUPTED: &[u8] = b"\ninterrupted; partial files kept, run the same command to resume\n";
const TRACEBACK: &[u8] = b"Traceback (most recent call last)";
const TAIL_BYTES: usize = 64 * 1024;

extern "C" fn on_interrupt(sig: libc::c_int) {
    // Async-signal-safe only: stop the downloader's whole process group, say so, leave with the shell's code for the signal.
    let group = DOWNLOAD_PGID.load(Ordering::SeqCst);
    unsafe {
        if group > 0 {
            libc::kill(-group, libc::SIGTERM);
            libc::write(2, INTERRUPTED.as_ptr().cast(), INTERRUPTED.len());
            libc::_exit(128 + sig);
        }
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }
}

struct Reaper(i32);

impl Drop for Reaper {
    fn drop(&mut self) {
        // Every way out of the download but a finished one (an error, a panic) takes the downloader's group with it.
        if self.0 > 0 {
            unsafe { libc::kill(-self.0, libc::SIGTERM) };
        }
    }
}

fn run_guarded(cmd: &mut Command) -> std::io::Result<(ExitStatus, Vec<u8>)> {
    // The downloader in its own process group, killed with goethe: SIGINT and SIGTERM stop the group, a SIGKILL of goethe stops the child (parent-death signal). Its stderr is shown live up to a Python traceback, which is kept for one line instead.
    cmd.stderr(Stdio::piped()).stdout(Stdio::null()).stdin(Stdio::null());
    cmd.process_group(0);
    #[cfg(target_os = "linux")]
    unsafe {
        cmd.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }
    let handler = on_interrupt as extern "C" fn(libc::c_int) as libc::sighandler_t;
    let (old_int, old_term) = unsafe { (libc::signal(libc::SIGINT, handler), libc::signal(libc::SIGTERM, handler)) };
    let result = cmd.spawn().and_then(|mut child| {
        let mut reaper = Reaper(child.id() as i32);
        DOWNLOAD_PGID.store(reaper.0, Ordering::SeqCst);
        let err = child.stderr.take();
        let shown = thread::spawn(move || err.map(tee_stderr).unwrap_or_default());
        let status = child.wait()?;
        reaper.0 = 0;
        DOWNLOAD_PGID.store(0, Ordering::SeqCst);
        Ok((status, shown.join().unwrap_or_default()))
    });
    unsafe {
        libc::signal(libc::SIGINT, old_int);
        libc::signal(libc::SIGTERM, old_term);
    }
    result
}

fn tee_stderr(mut r: std::process::ChildStderr) -> Vec<u8> {
    // The downloader's stderr (bars, lock notices) goes to the log, and to the screen with -v until a Python traceback starts; return the last bytes of everything.
    let (mut tail, mut buf, mut live) = (Vec::new(), [0u8; 4096], true);
    let mut err = std::io::stderr();
    loop {
        let n = match r.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        let before = tail.len().min(TRACEBACK.len());
        tail.extend_from_slice(&buf[..n]);
        for line in String::from_utf8_lossy(&buf[..n]).split(['\n', '\r']).filter(|l| !l.trim().is_empty()) {
            crate::step::log_line(line.trim_end());
        }
        if live {
            let from = tail.len() - n - before;
            let cut = tail[from..].windows(TRACEBACK.len()).position(|w| w == TRACEBACK).map(|p| (from + p).saturating_sub(tail.len() - n));
            if crate::step::verbose() {
                let _ = err.write_all(&buf[..cut.unwrap_or(n)]);
                let _ = err.flush();
            }
            live = cut.is_none();
        }
        if tail.len() > TAIL_BYTES {
            tail.drain(..tail.len() - TAIL_BYTES);
        }
    }
    tail
}

fn download_failure(status: &ExitStatus, tail: &[u8]) -> String {
    // One line for a failed download: the network case named, then the downloader's own last line.
    let text = String::from_utf8_lossy(tail);
    let drawn = |l: &&str| l.contains(['\u{2502}', '\u{256d}', '\u{2570}', '\u{2500}', '\u{256e}', '\u{256f}']);
    let lines: Vec<&str> = text.split(['\n', '\r']).map(str::trim).filter(|l| !l.is_empty() && !drawn(l)).collect();
    let last = lines.iter().rev().find(|l| l.contains("Error") || l.contains("Exception")).or(lines.last()).copied().unwrap_or("");
    let low = text.to_lowercase();
    let network = ["connecterror", "connectionerror", "proxyerror", "name or service not known", "temporary failure in name resolution", "max retries exceeded", "network is unreachable", "connection refused", "timed out", "readtimeout", "connecttimeout", "offline"];
    if network.iter().any(|n| low.contains(n)) {
        let host = hf_endpoint();
        let host = host.split("://").last().unwrap_or(&host).trim_end_matches('/').to_string();
        return format!("cannot reach {host} (network or proxy): {last}");
    }
    match last {
        "" => format!("hf download exited with {status}"),
        l if l.contains("gated") || l.contains("401") || l.contains("403") => l.to_string(),
        l => format!("hf download exited with {status}: {l}"),
    }
}

fn has_file(dir: &Path) -> bool {
    // True when any regular file lives under dir (a resumable partial download counts).
    fs::read_dir(dir).map(|rd| rd.flatten().any(|e| e.file_type().map(|t| if t.is_dir() { has_file(&e.path()) } else { true }).unwrap_or(true))).unwrap_or(true)
}

fn drop_empty_new_dir(dir: &Path, existed: bool) {
    // A failed first download leaves no empty model folder behind; one that held anything, or that was there before, stays.
    if !existed && !has_file(dir) {
        let _ = fs::remove_dir_all(dir);
    }
}

const HF_QUIET: &str = "import logging, runpy, sys
logging.getLogger('huggingface_hub.utils._http').addFilter(lambda r: 'HF_TOKEN' not in r.getMessage())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')";

fn python_of_script(script: &Path) -> Option<PathBuf> {
    // The interpreter a console script's shebang names, when it is a Python on disk.
    let mut head = Vec::new();
    fs::File::open(script).ok()?.take(256).read_to_end(&mut head).ok()?;
    let first = String::from_utf8_lossy(&head).lines().next()?.to_string();
    let interp = PathBuf::from(first.strip_prefix("#!")?.split_whitespace().next()?);
    let name = interp.file_name()?.to_string_lossy().into_owned();
    (name.starts_with("python") && interp.is_file()).then_some(interp)
}

pub fn hf_binary(home: &Path) -> Result<PathBuf, String> {
    // The py-env hf CLI, then one next to GOETHE_PYTHON, then PATH.
    let hf = home.join("py-env/bin/hf");
    if hf.is_file() {
        return Ok(hf);
    }
    if let Ok(py) = std::env::var("GOETHE_PYTHON") {
        if let Some(side) = PathBuf::from(py).parent().map(|p| p.join("hf")).filter(|p| p.is_file()) {
            return Ok(side);
        }
    }
    if let Ok(paths) = std::env::var("PATH") {
        for dir in std::env::split_paths(&paths) {
            let p = dir.join("hf");
            if p.is_file() {
                return Ok(p);
            }
        }
    }
    Err("hf CLI not found on PATH or in py-env".to_string())
}

fn hash_all(local_dir: &Path, files: &[&HfSibling]) -> Vec<(String, Result<VerifiedEntry, String>)> {
    // Hash LFS files on a few threads; disks and page cache handle parallel reads well.
    let workers = thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 8);
    let chunk = files.len().div_ceil(workers).max(1);
    thread::scope(|scope| {
        let handles: Vec<_> = files
            .chunks(chunk)
            .map(|part| {
                scope.spawn(move || {
                    part.iter()
                        .map(|s| {
                            let expect = s.lfs_sha256.as_deref().unwrap_or("");
                            (s.rfilename.clone(), verify_one_file(&local_dir.join(&s.rfilename), expect))
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap_or_default()).collect()
    })
}

fn verify_model(
    local_dir: &Path,
    revision: &str,
    files: &[&HfSibling],
    home: &Path,
    model_id: &str,
    excludes: &[String],
) -> Result<(), String> {
    // Check sha256 of every LFS file (re-hashing only files whose size or mtime changed); re-download mismatches once.
    let previous = read_verified_cache(local_dir).filter(|c| c.revision == revision);
    let (reused, lfs): (Vec<&HfSibling>, Vec<&HfSibling>) = files.iter().copied().filter(|s| s.lfs_sha256.is_some()).partition(|s| {
        previous.as_ref().and_then(|c| c.files.get(&s.rfilename)).is_some_and(|e| {
            Some(e.sha256.as_str()) == s.lfs_sha256.as_deref() && unchanged(local_dir, &s.rfilename, e)
        })
    });
    let total: u64 = lfs.iter().map(|s| s.size).sum();
    let base = HASHED.load(Ordering::Relaxed);
    let hashing = (total >= 256 << 20).then(|| {
        let done = move || HASHED.load(Ordering::Relaxed).saturating_sub(base);
        crate::progress::Ticker::start("verifying", Some(total), crate::progress::every(), done, crate::progress::eprint_line)
    });
    let mut results = hash_all(local_dir, &lfs);
    drop(hashing);
    if let Some(prev) = &previous {
        for s in &reused {
            if let Some(e) = prev.files.get(&s.rfilename) {
                results.push((s.rfilename.clone(), Ok(e.clone())));
            }
        }
    }
    let bad: Vec<&HfSibling> = lfs
        .iter()
        .copied()
        .filter(|s| results.iter().any(|(n, r)| n == &s.rfilename && r.is_err()))
        .collect();
    if !bad.is_empty() {
        for s in &bad {
            let _ = fs::remove_file(local_dir.join(&s.rfilename));
        }
        download_with_retries(home, model_id, local_dir, revision, excludes, bad.iter().map(|s| s.size).sum())?;
        let again = hash_all(local_dir, &bad);
        if let Some((name, _)) = again.iter().find(|(_, r)| r.is_err()) {
            return Err(format!("checksum mismatch for {name} after re-download"));
        }
        results.retain(|(_, r)| r.is_ok());
        results.extend(again);
    }
    let mut cache = VerifiedCache {
        revision: revision.to_string(),
        files: HashMap::new(),
    };
    let mut bytes = 0u64;
    for (name, r) in results {
        let entry = r?;
        bytes += entry.size;
        cache.files.insert(name, entry);
    }
    let count = cache.files.len();
    write_verified_cache(local_dir, &cache)?;
    write_revision(local_dir, revision)?;
    if count > 0 {
        crate::step::log_line(&format!("verified {count} files ({:.2} GB)", bytes as f64 / 1e9));
    }
    Ok(())
}

fn verify_one_file(path: &Path, expect_sha: &str) -> Result<VerifiedEntry, String> {
    // Hash one file and record the size and mtime it had while being hashed.
    let (size, mtime_ns) = file_identity(path).ok_or_else(|| format!("missing {}", path.display()))?;
    let got = sha256_file(path)?;
    if !expect_sha.is_empty() && got != expect_sha {
        return Err("sha256 mismatch".to_string());
    }
    Ok(VerifiedEntry { size, mtime_ns, sha256: got })
}

static HASHED: AtomicU64 = AtomicU64::new(0);

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        HASHED.fetch_add(n as u64, Ordering::Relaxed);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn file_identity(path: &Path) -> Option<(u64, i64)> {
    let meta = fs::metadata(path).ok().filter(|m| m.is_file())?;
    let mtime = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some((meta.len(), mtime.as_nanos() as i64))
}

fn read_verified_cache(local_dir: &Path) -> Option<VerifiedCache> {
    serde_json::from_str(&fs::read_to_string(local_dir.join(VERIFIED_FILE)).ok()?).ok()
}

fn unchanged(local_dir: &Path, rel: &str, entry: &VerifiedEntry) -> bool {
    file_identity(&local_dir.join(rel)) == Some((entry.size, entry.mtime_ns))
}

fn verify_cache_matches(local_dir: &Path) -> bool {
    // Every file hashed at verification time still has the same size and modification time.
    match read_verified_cache(local_dir) {
        Some(cache) => cache.files.iter().all(|(rel, entry)| unchanged(local_dir, rel, entry)),
        None => local_dir.join(REVISION_FILE).exists() && !local_dir.join(VERIFIED_FILE).exists(),
    }
}

fn write_verified_cache(local_dir: &Path, cache: &VerifiedCache) -> Result<(), String> {
    let text = serde_json::to_string_pretty(cache).map_err(|e| e.to_string())?;
    fs::write(local_dir.join(VERIFIED_FILE), text).map_err(|e| e.to_string())
}

fn write_revision(local_dir: &Path, revision: &str) -> Result<(), String> {
    fs::write(local_dir.join(REVISION_FILE), revision).map_err(|e| e.to_string())
}

fn read_revision(local_dir: &Path) -> Option<String> {
    fs::read_to_string(local_dir.join(REVISION_FILE))
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn parse_siblings(body: &serde_json::Value) -> Vec<HfSibling> {
    let Some(arr) = body.get("siblings").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|item| {
            let name = item.get("rfilename").and_then(|v| v.as_str()).filter(|n| is_repo_path(n))?;
            Some(HfSibling {
                rfilename: name.to_string(),
                size: item.get("size").and_then(|v| v.as_u64()).unwrap_or(0),
                lfs_sha256: item
                    .get("lfs")
                    .and_then(|l| l.get("sha256").or_else(|| l.get("oid")))
                    .and_then(|v| v.as_str())
                    .map(String::from),
            })
        })
        .collect()
}

fn is_repo_path(name: &str) -> bool {
    // A file name inside the model dir: relative, no empty, '.' or '..' part, so a listing can never point outside it.
    !name.contains(['\0', '\\']) && name.split('/').all(|c| !c.is_empty() && c != "." && c != "..")
}

struct ApiResponse {
    status: u16,
    body: String,
}

fn api_get(url: &str) -> Result<ApiResponse, String> {
    // Anonymous GET that retries transient failures (network, 429, 5xx) with short backoff.
    let delays = [0u64, 2, 5, 10];
    let mut last_err = String::new();
    for (i, delay) in delays.iter().enumerate() {
        if *delay > 0 {
            thread::sleep(Duration::from_secs(*delay));
        }
        match api_get_once(url) {
            Ok(r) if r.status == 429 || r.status >= 500 => last_err = format!("HTTP {}", r.status),
            Ok(r) => return Ok(r),
            Err(e) => {
                last_err = e;
                if i + 1 >= delays.len() {
                    break;
                }
            }
        }
    }
    Err(format!(
        "cannot reach {}: {last_err}; check the network or HF_ENDPOINT",
        hf_endpoint()
    ))
}

fn api_get_once(url: &str) -> Result<ApiResponse, String> {
    let mut req = crate::net::agent().get(url).timeout(Duration::from_secs(30));
    if let Some(t) = token() {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        Err(ureq::Error::Status(_code, r)) => r,
        Err(e) => return Err(e.to_string()),
    };
    let status = resp.status();
    let mut text = String::new();
    resp.into_reader()
        .take(512 * 1024 * 1024)
        .read_to_string(&mut text)
        .map_err(|e| format!("read {url}: {e}"))?;
    Ok(ApiResponse { status, body: text })
}

fn fetch_repo_json(model_id: &str, revision: &str, file: &str) -> Result<serde_json::Value, String> {
    let url = format!("{}/{model_id}/resolve/{revision}/{file}", hf_endpoint());
    let resp = api_get(&url)?;
    if resp.status == 404 {
        return Err(format!("{file} not found for {model_id}"));
    }
    if resp.status >= 400 {
        return Err(format!("HF fetch of {file} failed: HTTP {}", resp.status));
    }
    serde_json::from_str(&resp.body).map_err(|e| format!("{file} parse: {e}"))
}

pub fn safetensors_header(model_id: &str, revision: &str, file: &str) -> Result<Vec<u8>, String> {
    // The JSON header of one safetensors file of a repo by two small ranged requests, no weight read; the 8-byte length first.
    let url = format!("{}/{model_id}/resolve/{revision}/{file}", hf_endpoint());
    let get = |range: String, cap: u64| -> Result<(u16, Vec<u8>), String> {
        let mut req = crate::net::agent().get(&url).set("Range", &range).timeout(Duration::from_secs(30));
        if let Some(t) = token() {
            req = req.set("Authorization", &format!("Bearer {t}"));
        }
        let resp = req.call().map_err(|e| format!("{file}: {e}"))?;
        let status = resp.status();
        let mut body = Vec::new();
        resp.into_reader().take(cap).read_to_end(&mut body).map_err(|e| format!("{file}: {e}"))?;
        Ok((status, body))
    };
    let (_, head) = get("bytes=0-7".into(), 8)?;
    let n = u64::from_le_bytes(head.get(..8).and_then(|b| b.try_into().ok()).ok_or_else(|| format!("{file}: no safetensors length"))?);
    if n == 0 || n > 64 << 20 {
        return Err(format!("{file}: safetensors header of {n} bytes"));
    }
    let (status, body) = get(format!("bytes=8-{}", 7 + n), 8 + n)?;
    let header = if status == 206 { body } else { body.get(8..).unwrap_or_default().to_vec() };
    (header.len() as u64 == n).then_some(header).ok_or_else(|| format!("{file}: short safetensors header"))
}

pub fn fetch_config_json(model_id: &str, revision: &str) -> Result<serde_json::Value, String> {
    fetch_repo_json(model_id, revision, "config.json")
}

fn available_bytes(path: &Path) -> Result<u64, String> {
    use std::ffi::CString;
    let cpath = CString::new(path.to_str().ok_or_else(|| "models path is not UTF-8".to_string())?)
        .map_err(|e| e.to_string())?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(cpath.as_ptr(), &mut stat) };
    if rc != 0 {
        return Err("statvfs failed for models directory".to_string());
    }
    Ok(stat.f_bavail as u64 * stat.f_frsize as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_repo_is_not_found_and_never_called_gated() {
        let anon = status_refusal("org/nope", 401, false).unwrap();
        assert!(anon.contains("not found") && !anon.contains("gated") && anon.contains("--hf-token"), "{anon}");
        assert!(status_refusal("org/nope", 404, true).unwrap().contains("no access"));
        assert_eq!(status_refusal("org/m", 200, false), None);
        assert_eq!(status_refusal("org/m", 500, false), None, "a server error is reported as one");
    }

    #[test]
    fn a_repo_cannot_write_goethes_card_and_a_listing_cannot_leave_the_model_dir() {
        let s = vec![sib("model.safetensors", 4), sib("config.json", 2), sib("goethe_card.toml", 9), sib(".goethe_verified.json", 9)];
        let kept: Vec<&str> = wanted(&s, &download_excludes(&s)).iter().map(|f| f.rfilename.as_str()).collect();
        assert_eq!(kept, ["model.safetensors", "config.json"], "the card of a repo is never downloaded");
        let body = serde_json::json!({"siblings": [{"rfilename": "a.json"}, {"rfilename": "../../x"}, {"rfilename": "/etc/x"}, {"rfilename": "d/../../x"}, {"rfilename": "d//x"}, {"rfilename": "d/ok.json"}]});
        let names: Vec<String> = parse_siblings(&body).into_iter().map(|f| f.rfilename).collect();
        assert_eq!(names, ["a.json", "d/ok.json"]);
    }

    fn sh(script: &str) -> Command {
        let mut c = Command::new("sh");
        c.arg("-c").arg(script);
        c
    }

    #[test]
    fn the_downloader_runs_in_its_own_group_and_its_traceback_becomes_one_line() {
        let (status, tail) = run_guarded(&mut sh("ps -o pid=,pgid= -p $$ >&2; echo Traceback \\(most recent call last\\): >&2; echo '  File \"x.py\", line 1' >&2; echo 'httpx.ConnectError: [Errno -3] Temporary failure in name resolution' >&2; exit 1")).unwrap();
        assert!(!status.success());
        let text = String::from_utf8_lossy(&tail).to_string();
        let ids: Vec<i64> = text.lines().next().unwrap().split_whitespace().filter_map(|n| n.parse().ok()).collect();
        assert_eq!(ids[0], ids[1], "the child leads its own process group: {text}");
        assert_ne!(ids[1], unsafe { libc::getpgrp() } as i64);
        let line = download_failure(&status, &tail);
        assert_eq!(line, "cannot reach huggingface.co (network or proxy): httpx.ConnectError: [Errno -3] Temporary failure in name resolution");
        assert!(!line.contains('\n') && !line.contains("Traceback"));
        assert_eq!(download_failure(&status, b"401 Client Error: Unauthorized"), "401 Client Error: Unauthorized");
    }

    #[test]
    fn leaving_the_download_early_stops_its_whole_group() {
        let mut child = sh("sleep 30 & sleep 30").process_group(0).spawn().unwrap();
        let group = child.id() as i32;
        std::thread::sleep(Duration::from_millis(200));
        drop(Reaper(group));
        let status = child.wait().unwrap();
        assert_eq!(std::os::unix::process::ExitStatusExt::signal(&status), Some(libc::SIGTERM));
        std::thread::sleep(Duration::from_millis(200));
        assert_ne!(unsafe { libc::kill(-group, 0) }, 0, "the sleeping members of the group are gone too");
    }

    #[test]
    fn a_failed_first_download_leaves_no_empty_folder_but_keeps_what_it_got() {
        let home = tempfile::tempdir().unwrap();
        let (fresh, kept, theirs) = (home.path().join("fresh"), home.path().join("kept"), home.path().join("theirs"));
        for d in [&fresh, &kept, &theirs] {
            fs::create_dir_all(d.join(".cache/huggingface/download")).unwrap();
        }
        fs::write(kept.join(".cache/huggingface/download/m.incomplete"), b"half").unwrap();
        drop_empty_new_dir(&fresh, false);
        drop_empty_new_dir(&kept, false);
        drop_empty_new_dir(&theirs, true);
        assert!(!fresh.exists(), "an empty folder this run made is removed");
        assert!(kept.exists(), "a partial file is what a rerun resumes");
        assert!(theirs.exists(), "a folder that was already there stays");
    }

    #[test]
    fn a_pack_install_downloads_its_weights_and_the_small_metadata_only() {
        let s = vec![
            sib("m-Q8_0.gguf", 700 << 20),
            sib("m-Q4_K_M.gguf", 400 << 20),
            sib("tokenizer.json", 7 << 20),
            sib("README.md", 5),
            sib("sub/x.json", 5),
            sib("huge.json", 100 << 20),
        ];
        let ex = only_excludes(&s, &["m-Q8_0.gguf".to_string()]);
        assert_eq!(ex, vec!["m-Q4_K_M.gguf", "sub/x.json", "huge.json"]);
        assert_eq!(wanted(&s, &ex).iter().map(|f| f.rfilename.as_str()).collect::<Vec<_>>(), ["m-Q8_0.gguf", "tokenizer.json", "README.md"]);
    }

    fn sib(name: &str, size: u64) -> HfSibling {
        HfSibling {
            rfilename: name.to_string(),
            size,
            lfs_sha256: None,
        }
    }

    #[test]
    fn hf_runs_under_its_own_python_with_the_hint_filter() {
        let dir = tempfile::tempdir().unwrap();
        let python = dir.path().join("python3.11");
        fs::write(&python, b"").unwrap();
        let script = dir.path().join("hf");
        fs::write(&script, format!("#!{}\nimport sys\n", python.display())).unwrap();
        assert_eq!(python_of_script(&script), Some(python.clone()));
        fs::write(&script, "#!/bin/bash\necho fake hf\n").unwrap();
        assert_eq!(python_of_script(&script), None, "a shell stand-in runs directly");
        fs::write(&script, format!("#!{}/missing/python\n", dir.path().display())).unwrap();
        assert_eq!(python_of_script(&script), None);
        assert!(HF_QUIET.contains("huggingface_hub.utils._http") && HF_QUIET.contains("run_name='__main__'"));
    }

    #[test]
    fn excludes_skip_duplicates_and_keep_spec_maps() {
        let s = vec![
            sib("config.json", 1),
            sib("model-00001-of-00002.safetensors", 10),
            sib("model-00002-of-00002.safetensors", 10),
            sib("consolidated.safetensors", 20),
            sib("original/consolidated.00.pth", 20),
            sib("pytorch_model.bin", 20),
            sib("spec/hot_self_32k.pt", 1),
        ];
        let ex = download_excludes(&s);
        let kept: Vec<&str> = wanted(&s, &ex).iter().map(|x| x.rfilename.as_str()).collect();
        assert_eq!(
            kept,
            vec!["config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "spec/hot_self_32k.pt"]
        );
        assert_eq!(download_bytes(&s), 22);
        assert!(has_weight_files(&s));
        assert!(!has_weight_files(&[sib(".gitattributes", 1), sib("README.md", 5)]));
    }

    #[test]
    fn glob_semantics() {
        assert!(glob_match("original/*", "original/a/b.pth"));
        assert!(glob_match("*.bin", "sub/x.bin"));
        assert!(glob_match("consolidated*.safetensors", "consolidated.safetensors"));
        assert!(!glob_match("consolidated*.safetensors", "model.safetensors"));
        assert!(glob_match("a*b*c", "aXbYc"));
        assert!(!glob_match("a*b*c", "aXcYb"));
    }

    #[test]
    fn presence_check_reports_partial_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), b"{}").unwrap();
        let a = sib("config.json", 2);
        let b = sib("model.safetensors", 10);
        assert!(check_present(dir.path(), &[&a]).is_ok());
        assert!(check_present(dir.path(), &[&a, &b]).unwrap_err().contains("missing"));
        std::fs::write(dir.path().join("model.safetensors"), b"12345").unwrap();
        assert!(check_present(dir.path(), &[&b]).unwrap_err().contains("incomplete"));
    }
}
