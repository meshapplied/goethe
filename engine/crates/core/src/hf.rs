// Hugging Face download, verify, and auth for goethe install.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
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
}

pub fn hf_endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string())
}

pub fn resolve_token(home: &Path) -> Option<String> {
    // HF_TOKEN env, goethe login, then the token files huggingface_hub itself reads.
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(v) = std::env::var(key) {
            if !v.trim().is_empty() {
                return Some(v.trim().to_string());
            }
        }
    }
    let mut files = vec![home.join("hf_token")];
    if let Some(p) = std::env::var_os("HF_TOKEN_PATH") {
        files.push(PathBuf::from(p));
    }
    match std::env::var_os("HF_HOME") {
        Some(h) => files.push(PathBuf::from(h).join("token")),
        None => {
            if let Some(h) = std::env::var_os("HOME") {
                files.push(PathBuf::from(h).join(".cache/huggingface/token"));
            }
        }
    }
    files.iter().find_map(|p| {
        fs::read_to_string(p)
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    })
}

pub fn store_token(home: &Path, token: &str) -> Result<(), String> {
    let path = home.join("hf_token");
    fs::write(&path, format!("{token}\n")).map_err(|e| e.to_string())?;
    fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o600)).map_err(|e| e.to_string())
}

pub fn clear_token(home: &Path) -> Result<(), String> {
    let path = home.join("hf_token");
    if path.is_file() {
        fs::remove_file(&path).map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn validate_token(token: &str) -> Result<(), String> {
    let url = format!("{}/api/whoami-v2", hf_endpoint());
    let resp = api_request("GET", &url, Some(token), None)?;
    if resp.status >= 400 {
        return Err("invalid Hugging Face token".to_string());
    }
    Ok(())
}

pub fn login(home: &Path, token: &str) -> Result<(), String> {
    validate_token(token)?;
    store_token(home, token)
}

pub fn read_token_stdin() -> Result<String, String> {
    let mut buf = String::new();
    std::io::stdin().read_line(&mut buf).map_err(|e| e.to_string())?;
    let t = buf.trim().to_string();
    if t.is_empty() {
        return Err("empty token".to_string());
    }
    Ok(t)
}

pub fn read_token_tty() -> Result<String, String> {
    use std::io::{self, IsTerminal, Write as _};
    if !io::stdin().is_terminal() {
        return Err("login requires a tty or --token-stdin".to_string());
    }
    print!("Hugging Face token: ");
    io::stdout().flush().map_err(|e| e.to_string())?;
    read_tty_secret()
}

fn read_tty_secret() -> Result<String, String> {
    // Read one line from the terminal with echo off, restoring the mode afterwards.
    let tty = fs::File::open("/dev/tty").map_err(|e| e.to_string())?;
    let mut termios: libc::termios = unsafe { std::mem::zeroed() };
    if unsafe { libc::tcgetattr(libc::STDIN_FILENO, &mut termios) } != 0 {
        return Err("tcgetattr failed".to_string());
    }
    let saved = termios;
    termios.c_lflag &= !(libc::ECHO);
    if unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &termios) } != 0 {
        return Err("tcsetattr failed".to_string());
    }
    let mut reader = std::io::BufReader::new(tty);
    let mut line = String::new();
    let read = reader.read_line(&mut line);
    let _ = unsafe { libc::tcsetattr(libc::STDIN_FILENO, libc::TCSANOW, &saved) };
    println!();
    read.map_err(|e| e.to_string())?;
    let t = line.trim().to_string();
    if t.is_empty() {
        return Err("empty token".to_string());
    }
    Ok(t)
}

pub fn offline_mode() -> bool {
    std::env::var("HF_HUB_OFFLINE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn preflight_model(model_id: &str, revision: Option<&str>) -> Result<ModelPreflight, String> {
    preflight_model_opts(model_id, revision, false)
}

pub fn preflight_model_opts(
    model_id: &str,
    revision: Option<&str>,
    allow_gated_without_token: bool,
) -> Result<ModelPreflight, String> {
    // Query the model API (at a revision when given) for gating, files, sizes, and the commit sha.
    if offline_mode() {
        return Ok(ModelPreflight {
            gated: false,
            siblings: Vec::new(),
            sha: None,
        });
    }
    let endpoint = hf_endpoint();
    let token = resolve_token(&crate::paths::goethe_home());
    let url = match revision {
        Some(rev) => format!("{endpoint}/api/models/{model_id}/revision/{rev}?blobs=true"),
        None => format!("{endpoint}/api/models/{model_id}?blobs=true"),
    };
    let resp = api_request("GET", &url, token.as_deref(), None)?;
    let has_token = token.as_ref().is_some_and(|t| !t.is_empty());
    match resp.status {
        401 | 403 if !has_token => {
            return Err(format!(
                "model {model_id} not found, or it is private (Hugging Face reports both as 401): check the id, or run `goethe login`"
            ));
        }
        401 | 403 => {
            return Err(format!(
                "model {model_id} is private or access denied: run `goethe login` (or set HF_TOKEN)"
            ));
        }
        404 => return Err(format!("model {model_id} not found on {endpoint} (check the id and revision)")),
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
    if gated && !has_token && !allow_gated_without_token {
        return Err(format!(
            "model {model_id} is gated: accept its license on huggingface.co/{model_id}, then run `goethe login`"
        ));
    }
    if gated && has_token && !allow_gated_without_token {
        let probe = format!("{endpoint}/{model_id}/resolve/{}/config.json", revision.unwrap_or("main"));
        let r = api_request("GET", &probe, token.as_deref(), None)?;
        if r.status == 401 || r.status == 403 {
            return Err(format!(
                "model {model_id} is gated and the Hugging Face token in use has no access: accept the license on huggingface.co/{model_id} with that account, or run `goethe login` with a token that has access"
            ));
        }
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
    let token = resolve_token(&crate::paths::goethe_home());
    let resp = api_request("GET", &url, token.as_deref(), None)?;
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
    // Skip original/ checkpoints, non-safetensors duplicates, and consolidated copies of sharded weights.
    let names: Vec<&str> = siblings.iter().map(|s| s.rfilename.as_str()).collect();
    let mut ex = vec!["original/*".to_string()];
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
    if !opts.force && is_installed_fast(local_dir, &revision, opts.no_verify) {
        return Ok(());
    }
    if offline_mode() {
        if !local_dir.join("config.json").exists() {
            return Err(format!(
                "HF_HUB_OFFLINE=1 but {model_id} is not present under {}",
                local_dir.display()
            ));
        }
        return write_revision(local_dir, &revision);
    }
    let pre = preflight_model(model_id, Some(&revision))?;
    if !has_weight_files(&pre.siblings) {
        return Err(format!(
            "{model_id}: repository has no model weight files (no .safetensors at revision {revision}); nothing was installed"
        ));
    }
    let excludes = download_excludes(&pre.siblings);
    let files = wanted(&pre.siblings, &excludes);
    disk_preflight(home, bytes_needed(&files, local_dir))?;
    fs::create_dir_all(local_dir).map_err(|e| format!("{}: {e}", local_dir.display()))?;
    download_with_retries(home, model_id, local_dir, &revision, &excludes)?;
    check_present(local_dir, &files)?;
    if opts.no_verify {
        return write_revision(local_dir, &revision);
    }
    verify_model(local_dir, &revision, &files, home, model_id, &excludes)
}

pub fn is_installed_fast(local_dir: &Path, revision: &str, no_verify: bool) -> bool {
    if !local_dir.join("config.json").exists() || read_revision(local_dir).as_deref() != Some(revision) {
        return false;
    }
    no_verify || verify_cache_matches(local_dir)
}

fn bytes_needed(files: &[&HfSibling], local_dir: &Path) -> u64 {
    let need: u64 = files
        .iter()
        .filter(|s| fs::metadata(local_dir.join(&s.rfilename)).map(|m| m.len() != s.size).unwrap_or(true))
        .map(|s| s.size)
        .sum();
    ((need as f64) * 1.05) as u64
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

fn download_with_retries(home: &Path, model_id: &str, local_dir: &Path, revision: &str, excludes: &[String]) -> Result<(), String> {
    let delays = [2u64, 5, 10];
    let mut last = String::new();
    for (i, delay) in delays.iter().enumerate() {
        match run_hf_download(home, model_id, local_dir, revision, excludes) {
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
    msg.contains("gated or private") || msg.contains("401") || msg.contains("403")
}

pub fn run_hf_download(home: &Path, model_id: &str, local_dir: &Path, revision: &str, excludes: &[String]) -> Result<(), String> {
    // hf download resumes partial files and is cheap when everything is already present.
    let program = hf_binary(home)?;
    let mut cmd = Command::new(program);
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
    cmd.env("PYTHONWARNINGS", "ignore::FutureWarning");
    if let Some(t) = resolve_token(home) {
        cmd.env("HF_TOKEN", t);
    }
    cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    let status = cmd.status().map_err(|e| format!("hf download failed to start: {e}"))?;
    if !status.success() {
        return Err(format!("hf download exited with {status}"));
    }
    Ok(())
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
    let mut results = hash_all(local_dir, &lfs);
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
        download_with_retries(home, model_id, local_dir, revision, excludes)?;
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
        println!("verified {count} files ({:.2} GB)", bytes as f64 / 1e9);
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
            let name = item.get("rfilename").and_then(|v| v.as_str())?;
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

struct ApiResponse {
    status: u16,
    body: String,
}

fn api_request(method: &str, url: &str, token: Option<&str>, body: Option<&str>) -> Result<ApiResponse, String> {
    // Retry transient failures (network, 429, 5xx) with short backoff.
    let delays = [0u64, 2, 5, 10];
    let mut last_err = String::new();
    for (i, delay) in delays.iter().enumerate() {
        if *delay > 0 {
            thread::sleep(Duration::from_secs(*delay));
        }
        match api_request_once(method, url, token, body) {
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

fn api_request_once(method: &str, url: &str, token: Option<&str>, body: Option<&str>) -> Result<ApiResponse, String> {
    let mut req = match method {
        "GET" => crate::net::agent().get(url),
        "POST" => crate::net::agent().post(url),
        _ => return Err(format!("unsupported HTTP method {method}")),
    };
    req = req.timeout(Duration::from_secs(30));
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let result = match body {
        Some(b) => req.send_string(b),
        None => req.call(),
    };
    let resp = match result {
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
    let token = resolve_token(&crate::paths::goethe_home());
    let url = format!("{}/{model_id}/resolve/{revision}/{file}", hf_endpoint());
    let resp = api_request("GET", &url, token.as_deref(), None)?;
    if resp.status == 404 {
        return Err(format!("{file} not found for {model_id}"));
    }
    if resp.status >= 400 {
        return Err(format!("HF fetch of {file} failed: HTTP {}", resp.status));
    }
    serde_json::from_str(&resp.body).map_err(|e| format!("{file} parse: {e}"))
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

    fn sib(name: &str, size: u64) -> HfSibling {
        HfSibling {
            rfilename: name.to_string(),
            size,
            lfs_sha256: None,
        }
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
