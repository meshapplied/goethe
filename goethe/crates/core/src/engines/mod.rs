// Engines goethe launches: one descriptor per engine (py-env layer, request params, command), one file each; shared env and key helpers.

pub mod goethe_engine;
pub mod llama_cpp;
pub mod ninfer;
pub mod sglang;

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct Engine {
    pub id: &'static str,
    pub overhead_gb: f64,
    pub layer: Option<crate::pyenv::Layer>,
    pub params: Option<&'static [&'static str]>,
    pub command: fn(&EngineSpec) -> Result<Command, String>,
}

pub const ENGINES: &[&Engine] = &[&goethe_engine::ENGINE, &ninfer::ENGINE, &sglang::SGLANG, &llama_cpp::ENGINE];

const ENVELOPE: &[&str] = &["model", "messages", "prompt", "input", "stream", "stream_options", "user"];

pub fn by_id(id: &str) -> Result<&'static Engine, String> {
    let ids: Vec<&str> = ENGINES.iter().map(|e| e.id).collect();
    ENGINES.iter().copied().find(|e| e.id == id).ok_or_else(|| format!("engine {id} is not one goethe runs ({})", ids.join(", ")))
}

pub fn of_card(card: &toml::Value) -> Result<&'static Engine, String> {
    by_id(card.get("runtime").and_then(|v| v.as_str()).unwrap_or(sglang::SGLANG.id))
}

pub fn check_params(e: &Engine, body: &[u8]) -> Result<(), String> {
    // The first non-null body field the engine does not honour, as an unsupported_parameter refusal.
    let Some(allowed) = e.params else { return Ok(()) };
    let Ok(serde_json::Value::Object(obj)) = serde_json::from_slice(body) else { return Ok(()) };
    match obj.iter().find(|(k, v)| !v.is_null() && !ENVELOPE.contains(&k.as_str()) && !allowed.contains(&k.as_str())) {
        Some((k, _)) => Err(k.clone()),
        None => Ok(()),
    }
}

fn placement_env(spec: &EngineSpec) -> [(&'static str, String); 3] {
    // The instance's GPUs in PCI order and its id, set after the card env so a card cannot move an engine.
    let gpus: Vec<String> = spec.gpus.iter().map(|g| g.to_string()).collect();
    [("CUDA_VISIBLE_DEVICES", gpus.join(",")), ("CUDA_DEVICE_ORDER", "PCI_BUS_ID".into()), (crate::proc_ident::INSTANCE_ENV, spec.instance_id.into())]
}

pub fn command(spec: &EngineSpec) -> Result<Command, String> {
    (of_card(spec.card)?.command)(spec)
}

#[cfg(test)]
fn test_spec<'a>(dir: &'a Path, card: &'a toml::Value, gpus: &'a [u32]) -> EngineSpec<'a> {
    // A load of one model dir (also the home) on gpus, one tp per GPU.
    EngineSpec {
        home: dir,
        instance_id: "inst-x",
        served_name: "org/m",
        engine_key: "kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk",
        model_path: dir,
        card,
        port: 30001,
        ctx: 4096,
        max_conc: 4,
        pool_tokens: crate::chooser::kv_tokens(4096, 4, crate::chooser::floor(&[])),
        tp: gpus.len() as u64,
        mem_frac: 0.5,
        gpus,
        cc: "9.0",
        load_extra: &[],
    }
}

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
    pub pool_tokens: u64,
    pub tp: u64,
    pub mem_frac: f64,
    pub gpus: &'a [u32],
    pub cc: &'a str,
    pub load_extra: &'a [String],
}

pub fn new_engine_key() -> Result<String, String> {
    // 32 random hex characters from the kernel CSPRNG.
    let mut buf = [0u8; 16];
    crate::sys::random_bytes(&mut buf);
    if buf.iter().all(|b| *b == 0) {
        return Err(format!("random bytes: {}", std::io::Error::last_os_error()));
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

pub fn resolve_python(home: &Path) -> PathBuf {
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

fn compiler_env() -> Vec<(String, String)> {
    // The host compiler for run-time kernel builds; CXX and CC from goethed's environment win.
    let inherited = std::env::var("NVCC_APPEND_FLAGS").ok();
    crate::host_compiler::engine_env(crate::host_compiler::find().as_ref(), inherited.as_deref())
}

pub const ENGINE_KEY_ENV: &str = "GOETHE_ENGINE_API_KEY";

fn engine_key_env(cmd: &mut Command, key: &str) {
    // Owner-only in /proc/<pid>/environ; never put the secret on argv (world-readable cmdline).
    cmd.env(ENGINE_KEY_ENV, key).env("LLAMA_API_KEY", key);
}

fn card_env(spec: &EngineSpec) -> Vec<(String, String)> {
    // Card env table and card pythonpath. Loader hooks from a card never enter the process.
    let (out, dropped) = card_env_parts(spec);
    if !dropped.is_empty() {
        eprintln!(
            "goethe: card [env] dropped {}: loader hooks never enter the engine",
            dropped.join(", ")
        );
    }
    out
}

fn card_env_parts(spec: &EngineSpec) -> (Vec<(String, String)>, Vec<String>) {
    let home = spec.home;
    let mut out = Vec::new();
    let mut dropped = Vec::new();
    if let Some(table) = spec.card.get("env").and_then(|v| v.as_table()) {
        for (k, v) in table {
            if k.starts_with("LD_") || k.starts_with("DYLD_") {
                dropped.push(k.clone());
                continue;
            }
            if let Some(s) = v.as_str() {
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
    (out, dropped)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::sglang::check_extra_args;
    use crate::proc_ident::INSTANCE_ENV;
    use std::collections::HashMap;

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    fn spec_cmd(card: &str, tp: u64) -> Command {
        let card: toml::Value = toml::from_str(card).unwrap();
        command(&EngineSpec { model_path: Path::new("/h/models/m"), tp, ..test_spec(Path::new("/h"), &card, &[3, 2]) }).unwrap()
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
        assert!(!a.iter().any(|x| x == "--disable-radix-cache"));
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
    fn env_is_cleared_and_only_the_card_env_is_passed() {
        let plain = spec_cmd("pythonpath = [\"$GOETHE_HOME/x\"]\nSTRAY_KEY = \"/p\"\n[env]\nA = \"1\"", 1);
        let e = envs(&plain);
        assert_eq!(e.get("CUDA_VISIBLE_DEVICES"), Some(&Some("3,2".to_string())));
        assert_eq!(e.get(INSTANCE_ENV), Some(&Some("inst-x".to_string())));
        assert_eq!(e.get("SPT_NOENV"), Some(&Some("1".to_string())));
        assert_eq!(e.get("PYTHONPATH"), Some(&Some("/h/lib:/h/x".to_string())));
        assert_eq!(e.get("A"), Some(&Some("1".to_string())));
        assert!(!e.contains_key("STRAY_KEY"), "only [env] reaches the engine");
        assert!(!e.contains_key("CARGO_MANIFEST_DIR"));
        let a = args(&plain);
        assert!(!a.iter().any(|x| x == "--api-key"), "{a:?}");
        assert_eq!(e.get(ENGINE_KEY_ENV).and_then(|v| v.as_ref().map(|s| s.len())), Some(32));
        let card: toml::Value = toml::from_str("[env]\nLD_PRELOAD = \"/x.so\"\nDYLD_INSERT_LIBRARIES = \"/y.dylib\"\nOK = \"1\"").unwrap();
        let (kept, dropped) = card_env_parts(&EngineSpec {
            model_path: Path::new("/h/models/m"),
            tp: 1,
            ..test_spec(Path::new("/h"), &card, &[3, 2])
        });
        assert!(kept.iter().any(|(k, v)| k == "OK" && v == "1"));
        assert!(!kept.iter().any(|(k, _)| k.starts_with("LD_") || k.starts_with("DYLD_")));
        assert!(dropped.contains(&"LD_PRELOAD".to_string()) && dropped.contains(&"DYLD_INSERT_LIBRARIES".to_string()), "{dropped:?}");
        let blocked = envs(&spec_cmd("[env]\nLD_PRELOAD = \"/x.so\"\nDYLD_INSERT_LIBRARIES = \"/y.dylib\"\nOK = \"1\"", 1));
        assert!(!blocked.contains_key("LD_PRELOAD") && !blocked.contains_key("DYLD_INSERT_LIBRARIES"));
        assert_eq!(blocked.get("OK"), Some(&Some("1".into())));
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
    fn multi_tokenizer_above_one_is_refused() {
        let err = check_extra_args(&strs(&["--tokenizer-worker-num", "4"])).unwrap_err();
        assert!(err.contains("not allowed"), "{err}");
        assert!(check_extra_args(&strs(&["--tokenizer-worker-num=2"])).is_err());
        let single = args(&spec_cmd("extra_sglang_args = [\"--tokenizer-worker-num\", \"1\"]", 1));
        assert!(!single.iter().any(|a| a == "--api-key"));
        assert!(single.windows(2).any(|w| w[0] == "--host" && w[1] == "127.0.0.1"));
        assert!(single.iter().any(|a| a == "goethe_sglang_boot"));
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
    fn light_engines_pay_half_a_gig_on_cuda_sglang_pays_base() {
        assert!((by_id("llama.cpp").unwrap().overhead_gb - crate::calc::METAL_OVERHEAD_GB).abs() < 1e-9);
        assert!((by_id("goethe-engine").unwrap().overhead_gb - crate::calc::METAL_OVERHEAD_GB).abs() < 1e-9);
        assert!((by_id("ninfer").unwrap().overhead_gb - crate::calc::METAL_OVERHEAD_GB).abs() < 1e-9);
        assert!((by_id("sglang").unwrap().overhead_gb - crate::calc::BASE_OVERHEAD_GB).abs() < 1e-9);
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
        assert_eq!(crate::chooser::engine_args(&m, 8, crate::chooser::kv_tokens(4096, 8, crate::chooser::floor(&[]))), vec!["--max-mamba-cache-size".to_string(), "40".to_string()]);
        assert!(crate::memory_spec::installed(Path::new("/nonexistent"), &card, &[]).is_none());
        assert!(check_extra_args(&["--max-mamba-cache-size=4".to_string()]).is_err());
    }

    #[test]
    fn every_runtime_a_card_was_ever_written_with_resolves_and_an_unknown_one_is_refused() {
        assert_eq!(of_card(&toml::from_str("").unwrap()).unwrap().id, "sglang");
        for id in ["sglang", "ninfer", "goethe-engine", "llama.cpp"] {
            assert_eq!(of_card(&toml::from_str(&format!("runtime = \"{id}\"")).unwrap()).unwrap().id, id);
        }
        assert!(args(&spec_cmd("", 1)).iter().any(|a| a == "goethe_sglang_boot"));
        assert!(of_card(&toml::from_str("runtime = \"vllm\"").unwrap()).map(|e| e.id).unwrap_err().contains("not one goethe runs"));
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
