// NInfer: the pinned .ninfer artifact (fetch, verify, head patch) and its ninfer-serve command line.

use super::{engine_key_env, kept_env, placement_env, EngineSpec};
use std::path::{Path, PathBuf};
use std::process::Command;

pub const ENGINE: super::Engine = super::Engine {
    id: "ninfer",
    overhead_gb: crate::calc::METAL_OVERHEAD_GB,
    layer: None,
    params: None,
    command: ninfer_command,
};

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
    std::fs::write(root.join("NINFER_PIN"), include_str!("../../../../packaging/nvfp4-patch/NINFER_PIN")).map_err(|e| e.to_string())?;
    let script = patches.join("apply_head.py");
    std::fs::write(&script, include_str!("../../../../packaging/nvfp4-patch/patches/apply_head.py.txt")).map_err(|e| e.to_string())?;
    std::fs::write(patches.join("head_swap.json"), include_str!("../../../../packaging/nvfp4-patch/patches/head_swap.json"))
        .map_err(|e| e.to_string())?;
    std::fs::write(patches.join("head_rows.dat"), include_bytes!("../../../../packaging/nvfp4-patch/patches/head_rows.dat"))
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
    cmd.env_clear()
        .envs(kept_env())
        .envs(placement_env(spec));
    engine_key_env(&mut cmd, spec.engine_key);
    Ok(cmd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{command, test_spec};

    fn args(cmd: &Command) -> Vec<String> {
        cmd.get_args().map(|a| a.to_string_lossy().into_owned()).collect()
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
        let spec = EngineSpec { home: home.path(), served_name: NINFER_ID, ctx: 100_000, ..test_spec(model.path(), &card, &[0]) };
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
}
