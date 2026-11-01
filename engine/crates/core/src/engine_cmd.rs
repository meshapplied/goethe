// SGLang engine command line and clean environment: Goethe-owned flags, card extras, env allowlist, API key.

use crate::proc_ident::INSTANCE_ENV;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

const OWNED_FLAGS: &[&str] = &[
    "--host",
    "--port",
    "--tp-size",
    "--tensor-parallel-size",
    "--mem-fraction-static",
    "--context-length",
    "--max-running-requests",
    "--max-mamba-cache-size",
    "--swa-full-tokens-ratio",
    "--api-key",
    "--served-model-name",
];
const QUIET_PATHS: &[&str] = &["/health", "/v1/models", "/get_model_info"];
const ENV_KEEP: &[&str] = &["PATH", "HOME", "USER", "LOGNAME", "LANG", "TMPDIR", "LD_LIBRARY_PATH"];
const ENV_KEEP_PREFIX: &[&str] = &["LC_", "HF_", "CUDA_", "NCCL_", "NVIDIA_"];

pub struct EngineSpec<'a> {
    pub home: &'a Path,
    pub instance_id: &'a str,
    pub served_name: &'a str,
    pub engine_key: &'a str,
    pub model_path: &'a Path,
    pub card: &'a toml::Value,
    pub port: u16,
    pub ctx: u64,
    pub max_conc: u64,
    pub tp: u64,
    pub mem_frac: f64,
    pub gpus: &'a [u32],
    pub cc: &'a str,
    pub load_extra: &'a [String],
}

pub fn check_extra_args(args: &[String]) -> Result<(), String> {
    // Refuse Goethe owned flags, and tokenizer workers above 1 because that mode cannot take an API key.
    if multi_tokenizer(args) {
        return Err("extra SGLang arg --tokenizer-worker-num greater than 1 is not allowed: that mode cannot take an API key, so Goethe will not listen on TCP".into());
    }
    for arg in args {
        let name = arg.split('=').next().unwrap_or("");
        if !name.starts_with("--") || name.len() < 3 {
            continue;
        }
        if let Some(owned) = OWNED_FLAGS.iter().find(|o| o.starts_with(name)) {
            return Err(format!(
                "extra SGLang arg {arg} is not allowed: Goethe sets {owned} itself (use goethe load/config options instead)"
            ));
        }
    }
    Ok(())
}

pub fn card_extra_args(card: &toml::Value) -> Vec<String> {
    card.get("extra_sglang_args")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default()
}

fn multi_tokenizer(args: &[String]) -> bool {
    // SGLang refuses --api-key with --tokenizer-worker-num > 1, so such engines run keyless on loopback.
    args.iter().enumerate().any(|(i, a)| {
        let value = match a.strip_prefix("--tokenizer-worker-num") {
            Some("") => args.get(i + 1).map(String::as_str),
            Some(rest) => rest.strip_prefix('='),
            None => None,
        };
        value.and_then(|v| v.parse::<u32>().ok()).is_some_and(|n| n > 1)
    })
}

pub fn new_engine_key() -> Result<String, String> {
    // 32 random hex characters from the kernel CSPRNG.
    let mut buf = [0u8; 16];
    let mut filled = 0;
    while filled < buf.len() {
        let n = unsafe {
            libc::getrandom(
                buf[filled..].as_mut_ptr().cast(),
                buf.len() - filled,
                0,
            )
        };
        if n <= 0 {
            return Err(format!("getrandom: {}", std::io::Error::last_os_error()));
        }
        filled += n as usize;
    }
    Ok(buf.iter().map(|b| format!("{b:02x}")).collect())
}

fn engine_path(home: &Path) -> OsString {
    // The py-env's bin first, so run-time kernel builds find its ninja (and python) on machines without them.
    let mut dirs = vec![home.join("py-env/bin")];
    dirs.extend(std::env::split_paths(&std::env::var_os("PATH").unwrap_or_else(|| "/usr/local/bin:/usr/bin:/bin".into())));
    std::env::join_paths(dirs).unwrap_or_default()
}

fn kept_env() -> Vec<(OsString, OsString)> {
    std::env::vars_os()
        .filter(|(k, _)| {
            k.to_str()
                .is_some_and(|k| ENV_KEEP.contains(&k) || ENV_KEEP_PREFIX.iter().any(|p| k.starts_with(p)))
        })
        .collect()
}

fn expand_home(s: &str, home: &Path) -> String {
    s.replace("$GOETHE_HOME", &home.to_string_lossy())
}

fn default_cuda_home(home: &Path, card: &[(String, String)]) -> Vec<(String, String)> {
    // Run-time kernel builds use the bundled toolkit, so no system CUDA is needed, unless goethed's env or the card names one.
    let named = |k: &str| std::env::var_os(k).is_some() || card.iter().any(|(c, _)| c == k);
    if named("CUDA_HOME") || named("CUDA_PATH") {
        return Vec::new();
    }
    crate::cuda_toolkit::bundled_home(home).map(|d| vec![("CUDA_HOME".to_string(), d.to_string_lossy().into_owned())]).unwrap_or_default()
}

fn resolve_python(home: &Path) -> PathBuf {
    let p = home.join("py-env/bin/python");
    if p.is_file() {
        return p;
    }
    std::env::var("GOETHE_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("python3"))
}

fn arch_env(home: &Path, cc: &str) -> Vec<(String, String)> {
    // Run-time kernel builds target the engine's architecture; caches not keyed by architecture get one dir each.
    let Some(arch) = crate::gpu_arch::lookup(cc) else {
        return Vec::new();
    };
    let cache = home.join("cache").join("jit").join(arch.tag());
    let dir = |name: &str| cache.join(name).to_string_lossy().into_owned();
    let mut env = vec![
        ("TORCH_CUDA_ARCH_LIST".to_string(), arch.target.to_string()),
        ("TORCH_EXTENSIONS_DIR".to_string(), dir("torch_extensions")),
        ("TVM_FFI_CACHE_DIR".to_string(), dir("tvm-ffi")),
    ];
    if arch.nvfp4 == crate::gpu_arch::Nvfp4Path::MarlinAll {
        env.push(("SGLANG_DISABLE_SILU_FP4_QUANT_FUSION".to_string(), "1".to_string()));
    }
    env
}

fn memory_args(spec: &EngineSpec, extras: &[String]) -> Vec<String> {
    // The window ratio, state slots and chunk size the budget assumed; a chunk size the user set wins.
    let Some(m) = crate::memory_spec::installed(spec.model_path, spec.card, spec.load_extra) else { return Vec::new() };
    let args = crate::chooser::engine_args(&m, spec.ctx, spec.max_conc);
    let user_chunk = extras.iter().any(|a| a.starts_with("--chunked-prefill-size"));
    args.chunks(2).filter(|p| !(user_chunk && p[0] == "--chunked-prefill-size")).flatten().cloned().collect()
}

fn compiler_env() -> Vec<(String, String)> {
    // The host compiler for run-time kernel builds; CXX and CC from goethed's environment win.
    let inherited = std::env::var("NVCC_APPEND_FLAGS").ok();
    crate::host_compiler::engine_env(crate::host_compiler::find().as_ref(), inherited.as_deref())
}

fn card_env(spec: &EngineSpec) -> Vec<(String, String)> {
    // Card env table for every runtime, top-level TERNARY_* keys only for runtime ternary, card pythonpath always.
    let home = spec.home;
    let mut out = Vec::new();
    if let Some(table) = spec.card.get("env").and_then(|v| v.as_table()) {
        for (k, v) in table {
            if let Some(s) = v.as_str() {
                out.push((k.clone(), expand_home(s, home)));
            }
        }
    }
    if spec.card.get("runtime").and_then(|v| v.as_str()) == Some("ternary") {
        for (k, v) in spec.card.as_table().into_iter().flatten() {
            if let (true, Some(s)) = (k.starts_with("TERNARY_"), v.as_str()) {
                out.push((k.clone(), expand_home(s, home)));
            }
        }
    }
    let paths: Vec<String> = spec
        .card
        .get("pythonpath")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_str().map(|s| expand_home(s, home)))
        .collect();
    if !paths.is_empty() {
        out.push(("PYTHONPATH".to_string(), paths.join(":")));
    }
    out
}

pub const NINFER_ID: &str = "meshapplied/Qwen3.8-27B-NVFP4-NInfer";
pub const NINFER_FILE: &str = "qwen3_8_27b_nvfp4.ninfer";
pub const NINFER_BYTES: u64 = 23_719_715_844;
pub const NINFER_SHA256: &str = "74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82";
pub const NINFER_PATCHED_SHA256: &str = "afcaac9cd6b684e4a2f0316100b3a9e5f888e6b3d8b84b6f76ee4fed29ef34ed";
pub const NINFER_CTX: u64 = 100_000;
pub const NINFER_MAX_CONC: u64 = 4;
pub const NINFER_URL: &str = concat!(
    "https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/resolve/",
    "a107ba1b5b0609d9ca90d5ed61439f5f5cc7d64d/qwen3_8_27b_nvfp4.ninfer"
);

pub enum NinferHead {
    Patch,
    Ready,
}

pub fn ninfer_head(digest: &str) -> Result<NinferHead, String> {
    // Original digest still needs the draft-shortlist patch; the patched digest is ready to serve.
    if digest == NINFER_SHA256 {
        Ok(NinferHead::Patch)
    } else if digest == NINFER_PATCHED_SHA256 {
        Ok(NinferHead::Ready)
    } else {
        Err(ninfer_reject())
    }
}

pub fn ninfer_reject() -> String {
    format!(
        "{NINFER_FILE} must be {NINFER_BYTES} bytes and sha256 {NINFER_SHA256} or {NINFER_PATCHED_SHA256}; run nvfp4-kernels/fetch_model.sh"
    )
}

pub fn accept_ninfer(len: u64, digest: &str) -> Result<(), String> {
    // Size and one of the two pinned digests. The caller supplies the digest so tests never hash 23 GB.
    if len == NINFER_BYTES && (digest == NINFER_SHA256 || digest == NINFER_PATCHED_SHA256) {
        Ok(())
    } else {
        Err(ninfer_reject())
    }
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

pub fn ninfer_in(dir: &Path) -> Option<PathBuf> {
    let p = dir.join(NINFER_FILE);
    p.is_file().then_some(p)
}

pub fn check_ninfer_with(path: &Path, mut hash: impl FnMut(&Path) -> Result<String, String>) -> Result<(), String> {
    let len = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?.len();
    if len != NINFER_BYTES {
        return accept_ninfer(len, "");
    }
    accept_ninfer(len, &hash(path)?)
}

pub fn verify_ninfer_file(path: &Path) -> Result<(), String> {
    check_ninfer_with(path, sha256_file)
}

fn ninfer_cache_file() -> Result<PathBuf, String> {
    let root = if let Some(h) = std::env::var_os("NVFP4_HOME") {
        PathBuf::from(h)
    } else {
        let home = std::env::var_os("HOME").ok_or_else(|| "HOME is not set".to_string())?;
        PathBuf::from(home).join(".cache/goethe/nvfp4")
    };
    Ok(root.join("models").join(NINFER_FILE))
}

fn ninfer_candidates(card_dir: &Path) -> Vec<PathBuf> {
    let mut cands = vec![card_dir.join(NINFER_FILE)];
    if let Some(p) = std::env::var_os("MODEL") {
        cands.push(PathBuf::from(p));
    }
    if let Some(h) = std::env::var_os("NVFP4_HOME") {
        cands.push(PathBuf::from(h).join("models").join(NINFER_FILE));
    }
    if let Some(h) = std::env::var_os("HOME") {
        cands.push(PathBuf::from(h).join(".cache/goethe/nvfp4/models").join(NINFER_FILE));
    }
    cands
}

pub fn find_ninfer(card_dir: &Path) -> Result<PathBuf, String> {
    // The model dir, $MODEL, or the path serve.sh uses. No copy and no Hugging Face.
    for p in ninfer_candidates(card_dir) {
        if p.is_file() {
            verify_ninfer_file(&p)?;
            return Ok(p);
        }
    }
    Err(format!("{NINFER_ID}: {}", ninfer_reject()))
}

pub fn acquire_ninfer(card_dir: &Path) -> Result<PathBuf, String> {
    // Use a pinned file already on disk, or curl the neroued artifact, then patch the draft shortlist.
    let cache = ninfer_cache_file()?;
    for p in ninfer_candidates(card_dir) {
        if !p.is_file() {
            continue;
        }
        let len = std::fs::metadata(&p).map_err(|e| format!("{}: {e}", p.display()))?.len();
        if len != NINFER_BYTES {
            if p == cache {
                continue;
            }
            return Err(ninfer_reject());
        }
        let digest = sha256_file(&p)?;
        accept_ninfer(len, &digest)?;
        return finish_ninfer(p, &digest);
    }
    fetch_ninfer(&cache)
}

fn finish_ninfer(path: PathBuf, digest: &str) -> Result<PathBuf, String> {
    match ninfer_head(digest)? {
        NinferHead::Ready => Ok(path),
        NinferHead::Patch => patch_ninfer_head(&path),
    }
}

fn fetch_ninfer(dest: &Path) -> Result<PathBuf, String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let status = Command::new("curl")
        .arg("-fL")
        .arg("-C")
        .arg("-")
        .arg("-o")
        .arg(dest)
        .arg(NINFER_URL)
        .status()
        .map_err(|e| format!("curl: {e}"))?;
    if !status.success() {
        let _ = std::fs::remove_file(dest);
        return Err(ninfer_reject());
    }
    let len = std::fs::metadata(dest).map_err(|e| format!("{}: {e}", dest.display()))?.len();
    let digest = match sha256_file(dest) {
        Ok(d) => d,
        Err(e) => {
            let _ = std::fs::remove_file(dest);
            return Err(e);
        }
    };
    if accept_ninfer(len, &digest).is_err() {
        let _ = std::fs::remove_file(dest);
        return Err(ninfer_reject());
    }
    finish_ninfer(dest.to_path_buf(), &digest)
}

fn write_patch_kit() -> Result<PathBuf, String> {
    let root = ninfer_cache_file()?.parent().and_then(|p| p.parent()).ok_or("no nvfp4 cache")?.join("patchkit");
    let patches = root.join("patches");
    std::fs::create_dir_all(&patches).map_err(|e| format!("{}: {e}", patches.display()))?;
    std::fs::write(root.join("NINFER_PIN"), include_str!("../../../../nvfp4-kernels/NINFER_PIN")).map_err(|e| e.to_string())?;
    let script = patches.join("apply_head.py");
    std::fs::write(&script, include_str!("../../../../nvfp4-kernels/patches/apply_head.py")).map_err(|e| e.to_string())?;
    std::fs::write(patches.join("head_swap.json"), include_str!("../../../../nvfp4-kernels/patches/head_swap.json"))
        .map_err(|e| e.to_string())?;
    std::fs::write(patches.join("head_rows.dat"), include_bytes!("../../../../nvfp4-kernels/patches/head_rows.dat"))
        .map_err(|e| e.to_string())?;
    Ok(script)
}

fn patch_ninfer_head(path: &Path) -> Result<PathBuf, String> {
    let script = write_patch_kit()?;
    let status = Command::new("python3").arg(&script).arg(path).status().map_err(|e| format!("python3: {e}"))?;
    if !status.success() {
        return Err(format!("apply_head failed on {}", path.display()));
    }
    let digest = sha256_file(path)?;
    if digest != NINFER_PATCHED_SHA256 {
        return Err(ninfer_reject());
    }
    Ok(path.to_path_buf())
}

pub struct RequestDone {
    pub prefix_cache_hit_tokens: u64,
    pub prompt_tokens: u64,
    pub computed_prefill_tokens: u64,
    pub tool_call_count: u64,
}

pub fn last_request_done(path: &Path) -> Option<RequestDone> {
    // Last request_done in the tail of ninfer's jsonl. A short read skips a line cut in half.
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).ok()?;
    let len = file.metadata().ok()?.len();
    let start = len.saturating_sub(256 * 1024);
    file.seek(SeekFrom::Start(start)).ok()?;
    let mut buf = String::new();
    file.read_to_string(&mut buf).ok()?;
    let mut lines = buf.split('\n');
    if start > 0 {
        lines.next();
    }
    let mut last = None;
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v.get("event").and_then(|e| e.as_str()) != Some("request_done") {
            continue;
        }
        let Some(res) = v.get("result") else { continue };
        let Some(prompt) = json_u64(res, "prompt_tokens") else { continue };
        last = Some(RequestDone {
            prefix_cache_hit_tokens: json_u64(res, "prefix_cache_hit_tokens").unwrap_or(0),
            prompt_tokens: prompt,
            computed_prefill_tokens: json_u64(res, "computed_prefill_tokens").unwrap_or(0),
            tool_call_count: json_u64(res, "tool_call_count").unwrap_or(0),
        });
    }
    last
}

fn json_u64(v: &serde_json::Value, key: &str) -> Option<u64> {
    let n = v.get(key)?;
    n.as_u64().or_else(|| n.as_f64().map(|f| f.round() as u64))
}

fn ninfer_bin() -> Result<PathBuf, String> {
    if let Some(p) = std::env::var_os("NINFER_SERVE") {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Ok(p);
        }
    }
    if let Some(h) = std::env::var_os("HOME") {
        let p = PathBuf::from(h).join(".cache/goethe/nvfp4/ninfer/build/apps/ninfer-serve");
        if p.is_file() {
            return Ok(p);
        }
    }
    Err("ninfer-serve not found; set NINFER_SERVE or run nvfp4-kernels/install.sh".into())
}

fn ninfer_command(spec: &EngineSpec) -> Result<Command, String> {
    // ponytail: one argv, the default profile from nvfp4-kernels/serve.sh. Card ctx and concurrency still apply.
    let bin = ninfer_bin()?;
    let model = spec.model_path.join(NINFER_FILE);
    if !model.is_file() {
        return Err(format!("{}: {}", model.display(), ninfer_reject()));
    }
    let log_dir = spec.home.join("logs");
    std::fs::create_dir_all(&log_dir).map_err(|e| format!("{}: {e}", log_dir.display()))?;
    let request_log = log_dir.join(format!("{}.requests.jsonl", spec.instance_id));
    let mut cmd = Command::new(bin);
    cmd.arg(&model)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(spec.port.to_string())
        .arg("--api-key")
        .arg(spec.engine_key)
        .arg("--model-id")
        .arg(spec.served_name)
        .arg("--max-context")
        .arg(spec.ctx.to_string())
        .arg("--kv-capacity")
        .arg("auto")
        .arg("--max-concurrency")
        .arg(spec.max_conc.to_string())
        .arg("--kv-dtype")
        .arg("fp8")
        .arg("--spec")
        .arg("dflash2")
        .arg("--draft-tokens")
        .arg("7")
        .arg("--lm-head-draft")
        .arg("--prefill-chunk")
        .arg("1024")
        .arg("--request-log-jsonl")
        .arg(&request_log);
    let gpus: Vec<String> = spec.gpus.iter().map(|g| g.to_string()).collect();
    cmd.env_clear()
        .envs(kept_env())
        .env("CUDA_VISIBLE_DEVICES", gpus.join(","))
        .env("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        .env(INSTANCE_ENV, spec.instance_id);
    Ok(cmd)
}

pub fn command(spec: &EngineSpec) -> Result<Command, String> {
    if spec.card.get("runtime").and_then(|v| v.as_str()) == Some("ninfer") {
        return ninfer_command(spec);
    }
    Ok(sglang_command(spec))
}

fn sglang_command(spec: &EngineSpec) -> Command {
    // Goethe flags, then checked card and load extras, in a cleared env; probe paths stay out of the access log.
    let mut cmd = Command::new(resolve_python(spec.home));
    cmd.arg("-m")
        .arg("sglang.launch_server")
        .arg("--model-path")
        .arg(spec.model_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(spec.port.to_string())
        .arg("--tp-size")
        .arg(spec.tp.to_string())
        .arg("--context-length")
        .arg(spec.ctx.to_string())
        .arg("--max-running-requests")
        .arg(spec.max_conc.to_string())
        .arg("--mem-fraction-static")
        .arg(format!("{:.6}", spec.mem_frac))
        .arg("--served-model-name")
        .arg(spec.served_name)
        .arg("--uvicorn-access-log-exclude-prefixes")
        .args(QUIET_PATHS);
    let extras: Vec<String> = card_extra_args(spec.card).into_iter().chain(spec.load_extra.iter().cloned()).collect();
    cmd.args(memory_args(spec, &extras));
    cmd.arg("--api-key").arg(spec.engine_key);
    if let Some(dtype) = spec
        .card
        .get("kv_cache_dtype")
        .and_then(|v| v.as_str())
        .filter(|d| !d.is_empty() && !d.eq_ignore_ascii_case("auto"))
    {
        cmd.arg("--kv-cache-dtype").arg(dtype);
    }
    let custom_all_reduce = spec.card.get("custom_all_reduce").and_then(|v| v.as_bool()) == Some(true);
    if spec.tp > 1 && !custom_all_reduce {
        cmd.arg("--disable-custom-all-reduce");
    }
    cmd.args(&extras);
    let gpus: Vec<String> = spec.gpus.iter().map(|g| g.to_string()).collect();
    let card_vars = card_env(spec);
    cmd.env_clear()
        .envs(kept_env())
        .env("PATH", engine_path(spec.home))
        .envs(default_cuda_home(spec.home, &card_vars))
        .envs(arch_env(spec.home, spec.cc))
        .envs(crate::cuda_compat::env(spec.home))
        .envs(compiler_env())
        .envs(card_vars)
        .env("CUDA_VISIBLE_DEVICES", gpus.join(","))
        .env("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        .env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        .env("SPT_NOENV", "1")
        .env(INSTANCE_ENV, spec.instance_id);
    cmd
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn spec_cmd(card: &str, tp: u64) -> Command {
        let card: toml::Value = toml::from_str(card).unwrap();
        let spec = EngineSpec {
            home: Path::new("/h"),
            instance_id: "inst-x",
            served_name: "org/m",
            engine_key: "k".repeat(32).leak(),
            model_path: Path::new("/h/models/m"),
            card: &card,
            port: 30001,
            ctx: 4096,
            max_conc: 4,
            tp,
            mem_frac: 0.5,
            gpus: &[3, 2],
            cc: "9.0",
            load_extra: &[],
        };
        command(&spec).unwrap()
    }

    fn args(cmd: &Command) -> Vec<String> {
        cmd.get_args().map(|a| a.to_string_lossy().into_owned()).collect()
    }

    fn envs(cmd: &Command) -> HashMap<String, Option<String>> {
        cmd.get_envs()
            .map(|(k, v)| (k.to_string_lossy().into_owned(), v.map(|v| v.to_string_lossy().into_owned())))
            .collect()
    }

    #[test]
    fn owned_flags_are_rejected_in_every_spelling() {
        for bad in [
            "--host", "--host=0.0.0.0", "--port", "--tp", "--tp-size", "--tensor-parallel-size", "--mem-fraction-static",
            "--mem-frac", "--context-length", "--context", "--max-running-requests", "--api-key", "--api-key=x",
            "--served-model-name", "--served",
        ] {
            let err = check_extra_args(&strs(&["--trust-remote-code", bad, "1"])).unwrap_err();
            assert!(err.contains("not allowed"), "{bad}: {err}");
        }
        check_extra_args(&strs(&["--chunked-prefill-size", "4096", "--host-ip-check", "--ep-size", "2", "--"])).unwrap();
    }

    #[test]
    fn serves_the_installed_id_and_keeps_probes_out_of_the_access_log() {
        let a = args(&spec_cmd("model_id = \"card-name\"", 1));
        let n = a.iter().position(|x| x == "--served-model-name").unwrap();
        assert_eq!(a[n + 1], "org/m");
        let q = a.iter().position(|x| x == "--uvicorn-access-log-exclude-prefixes").unwrap();
        assert_eq!(&a[q + 1..q + 4], ["/health", "/v1/models", "/get_model_info"]);
        assert!(a[q + 4].starts_with("--"));
    }

    #[test]
    fn kv_dtype_only_when_card_sets_non_auto() {
        assert!(!args(&spec_cmd("", 1)).contains(&"--kv-cache-dtype".to_string()));
        assert!(!args(&spec_cmd("kv_cache_dtype = \"auto\"", 1)).contains(&"--kv-cache-dtype".to_string()));
        let a = args(&spec_cmd("kv_cache_dtype = \"fp8_e4m3\"", 1));
        let i = a.iter().position(|x| x == "--kv-cache-dtype").unwrap();
        assert_eq!(a[i + 1], "fp8_e4m3");
    }

    #[test]
    fn custom_all_reduce_default_off_for_tp_and_card_override() {
        assert!(args(&spec_cmd("", 2)).contains(&"--disable-custom-all-reduce".to_string()));
        assert!(!args(&spec_cmd("", 1)).contains(&"--disable-custom-all-reduce".to_string()));
        assert!(!args(&spec_cmd("custom_all_reduce = true", 2)).contains(&"--disable-custom-all-reduce".to_string()));
    }

    #[test]
    fn env_is_cleared_and_ternary_only_for_ternary() {
        let plain = spec_cmd("pythonpath = [\"$GOETHE_HOME/x\"]\nTERNARY_PACK_DIR = \"/p\"\n[env]\nA = \"1\"", 1);
        let e = envs(&plain);
        assert_eq!(e.get("CUDA_VISIBLE_DEVICES"), Some(&Some("3,2".to_string())));
        assert_eq!(e.get(INSTANCE_ENV), Some(&Some("inst-x".to_string())));
        assert_eq!(e.get("SPT_NOENV"), Some(&Some("1".to_string())));
        assert_eq!(e.get("PYTHONPATH"), Some(&Some("/h/x".to_string())));
        assert_eq!(e.get("A"), Some(&Some("1".to_string())));
        assert!(!e.contains_key("TERNARY_PACK_DIR"));
        assert!(!e.contains_key("CARGO_MANIFEST_DIR"));
        let tern = envs(&spec_cmd("runtime = \"ternary\"\nTERNARY_PACK_DIR = \"$GOETHE_HOME/p\"", 1));
        assert_eq!(tern.get("TERNARY_PACK_DIR"), Some(&Some("/h/p".to_string())));
        let a = args(&plain);
        let k = a.iter().position(|x| x == "--api-key").unwrap();
        assert_eq!(a[k + 1].len(), 32);
    }

    #[test]
    fn engines_get_the_bundled_cuda_toolkit_unless_one_is_named() {
        let td = tempfile::tempdir().unwrap();
        let cu = td.path().join("py-env/lib/python3.11/site-packages/nvidia/cu13");
        std::fs::create_dir_all(cu.join("bin")).unwrap();
        std::fs::write(cu.join("bin/nvcc"), b"").unwrap();
        if std::env::var_os("CUDA_HOME").is_none() && std::env::var_os("CUDA_PATH").is_none() {
            assert_eq!(default_cuda_home(td.path(), &[]), vec![("CUDA_HOME".to_string(), cu.to_string_lossy().into_owned())]);
        }
        let card = vec![("CUDA_HOME".to_string(), "/opt/cuda".to_string())];
        assert!(default_cuda_home(td.path(), &card).is_empty(), "a card CUDA_HOME wins");
        assert!(default_cuda_home(Path::new("/nonexistent"), &[]).is_empty());
    }

    #[test]
    fn kernels_target_the_engine_architecture_with_per_architecture_caches() {
        let e = envs(&spec_cmd("", 1));
        assert_eq!(e.get("TORCH_CUDA_ARCH_LIST"), Some(&Some("9.0a".to_string())));
        assert_eq!(e.get("TORCH_EXTENSIONS_DIR"), Some(&Some("/h/cache/jit/sm_90a/torch_extensions".to_string())));
        assert_eq!(e.get("TVM_FFI_CACHE_DIR"), Some(&Some("/h/cache/jit/sm_90a/tvm-ffi".to_string())));
        let mine = envs(&spec_cmd("[env]\nTORCH_CUDA_ARCH_LIST = \"9.0\"", 1));
        assert_eq!(mine.get("TORCH_CUDA_ARCH_LIST"), Some(&Some("9.0".to_string())), "a card can still override");
        assert!(arch_env(Path::new("/h"), "7.5").is_empty(), "unknown architectures keep library defaults");
    }

    #[test]
    fn ninfer_accepts_only_the_two_pinned_digests() {
        assert!(accept_ninfer(NINFER_BYTES, NINFER_SHA256).is_ok());
        assert!(accept_ninfer(NINFER_BYTES, NINFER_PATCHED_SHA256).is_ok());
        assert!(accept_ninfer(NINFER_BYTES, "nope").unwrap_err().contains(NINFER_FILE));
        assert!(matches!(ninfer_head(NINFER_SHA256).unwrap(), NinferHead::Patch));
        assert!(matches!(ninfer_head(NINFER_PATCHED_SHA256).unwrap(), NinferHead::Ready));
        assert!(ninfer_head("nope").is_err());
        assert_eq!(
            NINFER_URL,
            concat!(
                "https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer/resolve/",
                "a107ba1b5b0609d9ca90d5ed61439f5f5cc7d64d/qwen3_8_27b_nvfp4.ninfer"
            )
        );
        assert!(accept_ninfer(1, NINFER_SHA256).is_err());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(NINFER_FILE);
        std::fs::File::create(&path).unwrap().set_len(NINFER_BYTES).unwrap();
        let mut hashed = false;
        check_ninfer_with(&path, |_| {
            hashed = true;
            Ok(NINFER_PATCHED_SHA256.into())
        })
        .unwrap();
        assert!(hashed);
        let err = check_ninfer_with(&path, |_| Ok("deadbeef".into())).unwrap_err();
        assert!(err.contains(NINFER_SHA256) && !err.contains("safetensors"), "{err}");
    }

    #[test]
    fn ninfer_runtime_spawns_ninfer_serve_with_an_api_key() {
        let bin = tempfile::tempdir().unwrap();
        let exe = bin.path().join("ninfer-serve");
        std::fs::write(&exe, b"#!/bin/sh\n").unwrap();
        let model = tempfile::tempdir().unwrap();
        std::fs::write(model.path().join(NINFER_FILE), b"x").unwrap();
        let card: toml::Value = toml::from_str("runtime = \"ninfer\"").unwrap();
        let home = tempfile::tempdir().unwrap();
        let spec = EngineSpec {
            home: home.path(),
            instance_id: "inst-x",
            served_name: NINFER_ID,
            engine_key: "k".repeat(32).leak(),
            model_path: model.path(),
            card: &card,
            port: 30001,
            ctx: 100_000,
            max_conc: 4,
            tp: 1,
            mem_frac: 0.5,
            gpus: &[0],
            cc: "12.0",
            load_extra: &[],
        };
        let prev = std::env::var_os("NINFER_SERVE");
        std::env::set_var("NINFER_SERVE", &exe);
        let cmd = command(&spec).unwrap();
        match prev {
            Some(v) => std::env::set_var("NINFER_SERVE", v),
            None => std::env::remove_var("NINFER_SERVE"),
        }
        let a = args(&cmd);
        assert!(cmd.get_program().to_string_lossy().contains("ninfer-serve"), "{a:?}");
        assert!(!a.iter().any(|x| x == "sglang.launch_server"), "{a:?}");
        assert!(a.iter().any(|x| x == "--api-key"), "{a:?}");
        assert!(a.iter().any(|x| x == "--host"), "{a:?}");
        let host = a.iter().position(|x| x == "--host").unwrap();
        assert_eq!(a[host + 1], "127.0.0.1");
        assert!(a.windows(2).any(|w| w[0] == "--spec" && w[1] == "dflash2"));
        assert!(a.windows(2).any(|w| w[0] == "--draft-tokens" && w[1] == "7"));
        assert!(a.windows(2).any(|w| w[0] == "--max-context" && w[1] == "100000"));
        assert!(a.iter().any(|x| x == "--lm-head-draft"));
        let log = a.iter().position(|x| x == "--request-log-jsonl").unwrap();
        assert!(a[log + 1].ends_with("inst-x.requests.jsonl"), "{}", a[log + 1]);
        assert!(!a.iter().any(|x| x.contains("disable") && x.contains("cache")), "{a:?}");
        let sglang = args(&spec_cmd("", 1));
        assert!(!sglang.iter().any(|x| x == "--disable-radix-cache"));
    }

    #[test]
    fn the_last_request_done_line_is_the_prefix_hit() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("i.requests.jsonl");
        assert!(last_request_done(&path).is_none());
        let first = concat!(
            r#"{"event":"request_done","result":{"prompt_tokens":10,"#,
            r#""prefix_cache_hit_tokens":1,"computed_prefill_tokens":9,"tool_call_count":0}}"#
        );
        let second = concat!(
            r#"{"event":"request_done","result":{"prompt_tokens":1000,"#,
            r#""prefix_cache_hit_tokens":800,"computed_prefill_tokens":200,"tool_call_count":1}}"#
        );
        std::fs::write(&path, format!("{first}\n{second}\n")).unwrap();
        let done = last_request_done(&path).unwrap();
        assert_eq!(
            (done.prefix_cache_hit_tokens, done.prompt_tokens, done.computed_prefill_tokens, done.tool_call_count),
            (800, 1000, 200, 1)
        );
        std::fs::write(&path, "{\"event\":\"boot\"}\n").unwrap();
        assert!(last_request_done(&path).is_none());
    }

    #[test]
    fn multi_tokenizer_above_one_is_refused() {
        let err = check_extra_args(&strs(&["--tokenizer-worker-num", "4"])).unwrap_err();
        assert!(err.contains("not allowed"), "{err}");
        assert!(check_extra_args(&strs(&["--tokenizer-worker-num=2"])).is_err());
        let single = args(&spec_cmd("extra_sglang_args = [\"--tokenizer-worker-num\", \"1\"]", 1));
        assert!(single.iter().any(|a| a == "--api-key"));
        assert!(single.windows(2).any(|w| w[0] == "--host" && w[1] == "127.0.0.1"));
    }

    #[test]
    fn engine_keys_are_random_hex() {
        let a = new_engine_key().unwrap();
        let b = new_engine_key().unwrap();
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
    }

    #[test]
    fn hybrid_models_get_the_state_slots_goethe_budgeted() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = r#"{"architectures": ["Qwen3_5ForConditionalGeneration"], "num_hidden_layers": 8, "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
            "full_attention_interval": 4, "layer_types": ["linear_attention", "linear_attention", "linear_attention",
            "full_attention", "linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "linear_num_value_heads": 48, "linear_num_key_heads": 16, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4}"#;
        std::fs::write(dir.path().join("config.json"), cfg).unwrap();
        let card: toml::Value = toml::from_str("").unwrap();
        let m = crate::memory_spec::installed(dir.path(), &card, &[]).unwrap();
        assert_eq!(crate::chooser::engine_args(&m, 4096, 8), vec!["--max-mamba-cache-size".to_string(), "40".to_string()]);
        assert!(crate::memory_spec::installed(Path::new("/nonexistent"), &card, &[]).is_none());
        assert!(check_extra_args(&["--max-mamba-cache-size=4".to_string()]).is_err());
    }

    #[test]
    fn engines_find_the_py_env_tools_first() {
        let path = engine_path(Path::new("/h"));
        let first = std::env::split_paths(&path).next().unwrap();
        assert_eq!(first, Path::new("/h/py-env/bin"));
    }

    #[test]
    fn marlin_only_gpus_skip_the_blackwell_fp4_activation_kernel() {
        let has = |cc: &str| arch_env(Path::new("/h"), cc).iter().any(|(k, v)| k == "SGLANG_DISABLE_SILU_FP4_QUANT_FUSION" && v == "1");
        assert!(has("8.0") && has("8.6") && has("8.9") && has("9.0"));
        assert!(!has("10.0") && !has("12.0"));
    }
}