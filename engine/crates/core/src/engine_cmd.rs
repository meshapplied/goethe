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
    // Refuse extra args that would override a Goethe flag, including --flag=value and argparse prefixes.
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
    vec![
        ("TORCH_CUDA_ARCH_LIST".to_string(), arch.target.to_string()),
        ("TORCH_EXTENSIONS_DIR".to_string(), dir("torch_extensions")),
        ("TVM_FFI_CACHE_DIR".to_string(), dir("tvm-ffi")),
    ]
}

fn memory_args(spec: &EngineSpec, extras: &[String]) -> Vec<String> {
    // The window ratio, state slots and chunk size the budget assumed; a chunk size the user set wins.
    let Some(m) = crate::memory_spec::installed(spec.model_path, spec.card) else { return Vec::new() };
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

pub fn command(spec: &EngineSpec) -> Command {
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
    if !multi_tokenizer(&extras) {
        cmd.arg("--api-key").arg(spec.engine_key);
    }
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
        .envs(default_cuda_home(spec.home, &card_vars))
        .envs(arch_env(spec.home, spec.cc))
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
        command(&spec)
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
    fn multi_tokenizer_engines_run_without_api_key() {
        let multi = args(&spec_cmd("extra_sglang_args = [\"--tokenizer-worker-num\", \"4\"]", 1));
        assert!(!multi.iter().any(|a| a == "--api-key"));
        let eq = args(&spec_cmd("extra_sglang_args = [\"--tokenizer-worker-num=2\"]", 1));
        assert!(!eq.iter().any(|a| a == "--api-key"));
        let single = args(&spec_cmd("extra_sglang_args = [\"--tokenizer-worker-num\", \"1\"]", 1));
        assert!(single.iter().any(|a| a == "--api-key"));
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
        let m = crate::memory_spec::installed(dir.path(), &card).unwrap();
        assert_eq!(crate::chooser::engine_args(&m, 4096, 8), vec!["--max-mamba-cache-size".to_string(), "40".to_string()]);
        assert!(crate::memory_spec::installed(Path::new("/nonexistent"), &card).is_none());
        assert!(check_extra_args(&["--max-mamba-cache-size=4".to_string()]).is_err());
    }
}