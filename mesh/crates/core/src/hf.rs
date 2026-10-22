// Hugging Face download, verify, and auth for mesh install.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{BufRead, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

const REVISION_FILE: &str = ".mesh_hf_revision";
const VERIFIED_FILE: &str = ".mesh_verified.json";

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
    files: std::collections::HashMap<String, VerifiedEntry>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct VerifiedEntry {
    size: u64,
    mtime: i64,
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
    for key in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Ok(v) = std::env::var(key) {
            if !v.trim().is_empty() {
                return Some(v.trim().to_string());
            }
        }
    }
    let path = home.join("hf_token");
    fs::read_to_string(&path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub fn store_token(home: &Path, token: &str) -> Result<(), String> {
    let path = home.join("hf_token");
    fs::write(&path, format!("{token}\n")).map_err(|e| e.to_string())?;
    fs::set_permissions(&path, std::os::unix::fs::PermissionsExt::from_mode(0o600))
        .map_err(|e| e.to_string())
}

pub fn clear_token(home: &Path) -> Result<(), String> {
    let path = home.join("hf_token");
    if path.is_file() {
        fs::remove_file(&path).map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn validate_token(token: &str) -> Result<(), String> {
    let endpoint = hf_endpoint();
    let url = format!("{endpoint}/api/whoami-v2");
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
    std::io::stdin()
        .read_line(&mut buf)
        .map_err(|e| e.to_string())?;
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
    if offline_mode() {
        return Ok(ModelPreflight {
            gated: false,
            siblings: Vec::new(),
            sha: None,
        });
    }
    let endpoint = hf_endpoint();
    let token = resolve_token(&crate::paths::mesh_home());
    let mut url = format!("{endpoint}/api/models/{model_id}?blobs=true");
    if let Some(rev) = revision {
        url.push_str("&revision=");
        url.push_str(rev);
    }
    let resp = api_request("GET", &url, token.as_deref(), None)?;
    let has_token = token.as_ref().is_some_and(|t| !t.is_empty());
    match resp.status {
        401 | 403 if !has_token => {
            return Err(format!(
                "model {model_id} not found, or it is private (Hugging Face reports both as 401): check the id, or run `mesh login`"
            ));
        }
        401 | 403 => {
            return Err(format!(
                "model {model_id} is private or access denied: run `mesh login` (or set HF_TOKEN)"
            ));
        }
        404 => return Err(format!("model {model_id} not found on {endpoint}")),
        s if s >= 500 => {
            return Err(format!("cannot reach {endpoint}: HTTP {s}"));
        }
        s if s >= 400 => {
            return Err(format!("HF API error for {model_id}: HTTP {s}"));
        }
        _ => {}
    }
    let body: serde_json::Value =
        serde_json::from_str(&resp.body).map_err(|e| format!("HF API parse failed: {e}"))?;
    let gated = body.get("gated").map(|v| {
        v.as_bool().unwrap_or(false) || v.as_str().is_some_and(|s| !s.is_empty())
    }).unwrap_or(false);
    let siblings = parse_siblings(&body);
    if gated && !has_token && !allow_gated_without_token {
        return Err(format!(
            "model {model_id} is gated: accept its license on huggingface.co/{model_id}, then run `mesh login`"
        ));
    }
    let sha = body
        .get("sha")
        .and_then(|v| v.as_str())
        .map(String::from);
    Ok(ModelPreflight {
        gated,
        siblings,
        sha,
    })
}

pub fn hf_owner_org(model_id: &str) -> Option<String> {
    model_id.split('/').next().map(String::from)
}

pub fn is_hf_organization(owner: &str) -> Result<bool, String> {
    if offline_mode() {
        return Ok(true);
    }
    let endpoint = hf_endpoint();
    let url = format!("{endpoint}/api/organizations/{owner}/overview");
    let token = resolve_token(&crate::paths::mesh_home());
    let resp = api_request("GET", &url, token.as_deref(), None)?;
    Ok(resp.status == 200)
}

pub fn ensure_model(
    home: &Path,
    model_id: &str,
    local_dir: &Path,
    opts: &DownloadOpts,
) -> Result<(), String> {
    fs::create_dir_all(local_dir).map_err(|e| e.to_string())?;
    let revision = opts
        .revision
        .clone()
        .unwrap_or_else(|| "main".to_string());
    if !opts.force && is_installed_fast(local_dir, &revision, opts.no_verify) {
        return Ok(());
    }
    let preflight = if offline_mode() {
        if !local_dir.join("config.json").exists() {
            return Err(format!(
                "HF_HUB_OFFLINE=1 but {model_id} is not present under {}",
                local_dir.display()
            ));
        }
        ModelPreflight {
            gated: false,
            siblings: Vec::new(),
            sha: None,
        }
    } else {
        preflight_model(model_id, Some(&revision))?
    };
    if !offline_mode() {
        let needed = bytes_needed(&preflight.siblings, local_dir);
        disk_preflight(home, needed)?;
    }
    if !opts.force && is_installed_fast(local_dir, &revision, opts.no_verify) {
        return Ok(());
    }
    if !offline_mode() && (!local_dir.join("config.json").exists() || opts.force) {
        download_with_retries(home, model_id, local_dir, Some(&revision))?;
    }
    if !opts.no_verify && !offline_mode() && !preflight.siblings.is_empty() {
        verify_model(local_dir, &revision, &preflight.siblings, home, model_id)?;
    } else if local_dir.join("config.json").exists() {
        write_revision(local_dir, &revision)?;
    }
    Ok(())
}

pub fn is_installed_fast(local_dir: &Path, revision: &str, no_verify: bool) -> bool {
    if !local_dir.join("config.json").exists() {
        return false;
    }
    if read_revision(local_dir).as_deref() != Some(revision) {
        return false;
    }
    if no_verify {
        return true;
    }
    verify_cache_matches(local_dir)
}

fn bytes_needed(siblings: &[HfSibling], local_dir: &Path) -> u64 {
    let mut need = 0u64;
    for s in siblings {
        let path = local_dir.join(&s.rfilename);
        if path.is_file() {
            if let Ok(meta) = fs::metadata(&path) {
                if meta.len() == s.size {
                    continue;
                }
            }
        }
        need = need.saturating_add(s.size);
    }
    ((need as f64) * 1.05) as u64
}

pub fn disk_preflight(home: &Path, needed: u64) -> Result<(), String> {
    if let Ok(v) = std::env::var("MESH_TEST_FREE_BYTES") {
        let free: u64 = v.parse().map_err(|_| "invalid MESH_TEST_FREE_BYTES".to_string())?;
        if needed > free {
            return Err(format!(
                "need {:.2} GB free for download but only {:.2} GB available on models volume",
                needed as f64 / 1e9,
                free as f64 / 1e9
            ));
        }
        return Ok(());
    }
    let models = home.join("models");
    fs::create_dir_all(&models).map_err(|e| e.to_string())?;
    let free = available_bytes(&models)?;
    if needed > free {
        return Err(format!(
            "need {:.2} GB free for download but only {:.2} GB available on models volume",
            needed as f64 / 1e9,
            free as f64 / 1e9
        ));
    }
    Ok(())
}

fn download_with_retries(
    home: &Path,
    model_id: &str,
    local_dir: &Path,
    revision: Option<&str>,
) -> Result<(), String> {
    let delays = [2u64, 5, 10];
    let mut last = String::new();
    for (i, delay) in delays.iter().enumerate() {
        match run_hf_download(home, model_id, local_dir, revision) {
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

pub fn run_hf_download(
    home: &Path,
    model_id: &str,
    local_dir: &Path,
    revision: Option<&str>,
) -> Result<(), String> {
    let program = hf_binary(home)?;
    let token = resolve_token(home);
    let mut cmd = Command::new(program);
    cmd.arg("download").arg(model_id).arg("--local-dir").arg(local_dir);
    if let Some(rev) = revision {
        cmd.arg("--revision").arg(rev);
    }
    cmd.env("HF_XET_HIGH_PERFORMANCE", "1");
    cmd.env("HF_HUB_DISABLE_TELEMETRY", "1");
    cmd.env("HF_HUB_ENABLE_HF_TRANSFER", "1");
    cmd.env("PYTHONWARNINGS", "ignore::FutureWarning");
    if let Some(t) = &token {
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
    let hf = home.join("py-env/bin/hf");
    if hf.is_file() {
        return Ok(hf);
    }
    if let Ok(py) = std::env::var("MESH_PYTHON") {
        let side = PathBuf::from(py)
            .parent()
            .map(|p| p.join("hf"))
            .filter(|p| p.is_file());
        if let Some(s) = side {
            return Ok(s);
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

fn verify_model(
    local_dir: &Path,
    revision: &str,
    siblings: &[HfSibling],
    home: &Path,
    model_id: &str,
) -> Result<(), String> {
    let lfs: Vec<&HfSibling> = siblings.iter().filter(|s| s.lfs_sha256.is_some()).collect();
    if lfs.is_empty() {
        write_revision(local_dir, revision)?;
        return Ok(());
    }
    let mut verified = 0u64;
    let mut bytes = 0u64;
    let mut cache = VerifiedCache {
        revision: revision.to_string(),
        files: std::collections::HashMap::new(),
    };
    for s in &lfs {
        let path = local_dir.join(&s.rfilename);
        let expect = s.lfs_sha256.as_deref().unwrap_or("");
        match verify_one_file(&path, expect) {
            Ok(entry) => {
                bytes += entry.size;
                verified += 1;
                cache.files.insert(s.rfilename.clone(), entry);
            }
            Err(_e) => {
                let _ = fs::remove_file(&path);
                download_with_retries(home, model_id, local_dir, Some(revision))?;
                if verify_one_file(&path, expect).is_err() {
                    return Err(format!("checksum mismatch for {} after re-download", s.rfilename));
                }
                let entry = verify_one_file(&path, expect)?;
                bytes += entry.size;
                verified += 1;
                cache.files.insert(s.rfilename.clone(), entry);
            }
        }
    }
    write_verified_cache(local_dir, &cache)?;
    write_revision(local_dir, revision)?;
    let gb = bytes as f64 / 1e9;
    println!("verified {verified} files ({gb:.2} GB)");
    Ok(())
}

fn verify_one_file(path: &Path, expect_sha: &str) -> Result<VerifiedEntry, String> {
    if !path.is_file() {
        return Err(format!("missing {}", path.display()));
    }
    let meta = fs::metadata(path).map_err(|e| e.to_string())?;
    let got = sha256_file(path)?;
    if !expect_sha.is_empty() && got != expect_sha {
        return Err("sha256 mismatch".to_string());
    }
    Ok(VerifiedEntry {
        size: meta.len(),
        mtime: meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs() as i64).unwrap_or(0),
        sha256: got,
    })
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut f = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn verify_cache_matches(local_dir: &Path) -> bool {
    let cache_path = local_dir.join(VERIFIED_FILE);
    let Ok(text) = fs::read_to_string(&cache_path) else {
        return local_dir.join(REVISION_FILE).exists();
    };
    let Ok(cache): Result<VerifiedCache, _> = serde_json::from_str(&text) else {
        return false;
    };
    for (rel, entry) in &cache.files {
        let path = local_dir.join(rel);
        if !path.is_file() {
            return false;
        }
        let Ok(meta) = fs::metadata(&path) else {
            return false;
        };
        if meta.len() != entry.size {
            return false;
        }
    }
    true
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
    let mut out = Vec::new();
    let Some(arr) = body.get("siblings").and_then(|v| v.as_array()) else {
        return out;
    };
    for item in arr {
        let Some(name) = item.get("rfilename").and_then(|v| v.as_str()) else {
            continue;
        };
        let size = item.get("size").and_then(|v| v.as_u64()).unwrap_or(0);
        let lfs_sha256 = item
            .get("lfs")
            .and_then(|l| l.get("sha256").or_else(|| l.get("oid")))
            .and_then(|v| v.as_str())
            .map(String::from);
        out.push(HfSibling {
            rfilename: name.to_string(),
            size,
            lfs_sha256,
        });
    }
    out
}

struct ApiResponse {
    status: u16,
    body: String,
}

fn api_request(method: &str, url: &str, token: Option<&str>, body: Option<&str>) -> Result<ApiResponse, String> {
    let endpoint = hf_endpoint();
    let delays = [0u64, 2, 5, 10];
    let mut last_err = String::new();
    for (i, delay) in delays.iter().enumerate() {
        if *delay > 0 {
            thread::sleep(Duration::from_secs(*delay));
        }
        match api_request_once(method, url, token, body) {
            Ok(r) if r.status == 429 || r.status >= 500 => {
                last_err = format!("HTTP {}", r.status);
                continue;
            }
            Ok(r) => return Ok(r),
            Err(e) => {
                last_err = e;
                if i + 1 >= delays.len() {
                    break;
                }
            }
        }
    }
    Err(format!("cannot reach {endpoint}: {last_err}; check the network or HF_ENDPOINT"))
}

fn api_request_once(method: &str, url: &str, token: Option<&str>, body: Option<&str>) -> Result<ApiResponse, String> {
    let mut req = match method {
        "GET" => crate::net::agent().get(url),
        "POST" => crate::net::agent().post(url),
        "PUT" => crate::net::agent().put(url),
        "DELETE" => crate::net::agent().delete(url),
        _ => return Err(format!("unsupported HTTP method {method}")),
    };
    req = req.timeout(Duration::from_secs(10));
    if let Some(t) = token {
        req = req.set("Authorization", &format!("Bearer {t}"));
    }
    let resp = match if let Some(b) = body {
        req.send_string(b)
    } else {
        req.call()
    } {
        Ok(r) => r,
        Err(ureq::Error::Status(_code, r)) => r,
        Err(e) => return Err(e.to_string()),
    };
    let status = resp.status();
    let text = resp.into_string().unwrap_or_default();
    Ok(ApiResponse {
        status,
        body: text,
    })
}

pub fn fetch_config_json(model_id: &str, revision: &str) -> Result<serde_json::Value, String> {
    let endpoint = hf_endpoint();
    let token = resolve_token(&crate::paths::mesh_home());
    let url = format!("{endpoint}/{model_id}/resolve/{revision}/config.json");
    let resp = api_request("GET", &url, token.as_deref(), None)?;
    if resp.status == 404 {
        return Err(format!("config.json not found for {model_id}"));
    }
    if resp.status >= 400 {
        return Err(format!("HF config fetch HTTP {}", resp.status));
    }
    serde_json::from_str(&resp.body).map_err(|e| format!("config.json parse: {e}"))
}

pub fn safetensors_bytes_from_siblings(siblings: &[HfSibling]) -> u64 {
    siblings
        .iter()
        .filter(|s| s.rfilename.ends_with(".safetensors"))
        .map(|s| s.size)
        .sum()
}

fn available_bytes(path: &Path) -> Result<u64, String> {
    use std::ffi::CString;
    let cpath = CString::new(
        path.to_str()
            .ok_or_else(|| "models path is not UTF-8".to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(cpath.as_ptr(), &mut stat) };
    if rc != 0 {
        return Err("statvfs failed for models directory".to_string());
    }
    Ok(stat.f_bavail as u64 * stat.f_frsize as u64)
}
