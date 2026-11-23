// SGLang launch: Goethe-owned flags and card extras.

use super::{
    arch_env, card_env, compiler_env, default_cuda_home, engine_key_env, engine_path, kept_env, placement_env,
    resolve_python, EngineSpec,
};
use crate::sglang_infer::RESERVED_FLAGS;
use std::process::Command;

pub const SGLANG: super::Engine = super::Engine {
    id: "sglang",
    overhead_gb: crate::calc::BASE_OVERHEAD_GB,
    layer: Some(crate::pyenv::Layer::Sglang),
    params: None,
    command: |spec| Ok(sglang_command(spec)),
};

const QUIET_PATHS: &[&str] = &["/health", "/v1/models", "/get_model_info"];

pub fn check_extra_args(args: &[String]) -> Result<(), String> {
    if multi_tokenizer(args) {
        return Err("extra SGLang arg --tokenizer-worker-num greater than 1 is not allowed: that mode cannot take an API key, so Goethe will not listen on TCP".into());
    }
    for arg in args {
        let name = arg.split('=').next().unwrap_or("");
        if !name.starts_with("--") || name.len() < 3 {
            continue;
        }
        if let Some((owned, _)) = RESERVED_FLAGS.iter().find(|(f, _)| *f == name || f.starts_with(name)) {
            return Err(format!("extra SGLang arg {arg} is not allowed: Goethe sets {owned} itself (use goethe load options instead)"));
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
    args.iter().enumerate().any(|(i, a)| {
        let value = match a.strip_prefix("--tokenizer-worker-num") {
            Some("") => args.get(i + 1).map(String::as_str),
            Some(rest) => rest.strip_prefix('='),
            None => None,
        };
        value.and_then(|v| v.parse::<u32>().ok()).is_some_and(|n| n > 1)
    })
}

fn memory_args(spec: &EngineSpec, extras: &[String]) -> Vec<String> {
    let Some(m) = crate::memory_spec::installed(spec.model_path, spec.card, spec.load_extra) else { return Vec::new() };
    let args = crate::chooser::engine_args(&m, spec.ctx, spec.max_conc);
    let user_chunk = extras.iter().any(|a| a.starts_with("--chunked-prefill-size"));
    args.chunks(2).filter(|p| !(user_chunk && p[0] == "--chunked-prefill-size")).flatten().cloned().collect()
}

const SGLANG_BOOT: &str = "import os, runpy, sys\n\nif __name__ == '__main__':\n    sys.argv = ['sglang.launch_server', *sys.argv[1:], '--api-key', os.environ['GOETHE_ENGINE_API_KEY']]\n    runpy.run_module('sglang.launch_server', run_name='__main__')\n";

fn graph_args(max_conc: u64, extras: &[String]) -> Vec<String> {
    // Decode graphs up to concurrency. Prefill capture lives outside the static pool; skip it (ponytail: else OOM on tight cards).
    let theirs = extras.iter().any(|a| a.starts_with("--cuda-graph") || a == "--disable-cuda-graph");
    if theirs {
        return Vec::new();
    }
    vec![
        "--cuda-graph-max-bs-decode".into(),
        max_conc.max(1).to_string(),
        "--cuda-graph-backend-prefill".into(),
        "disabled".into(),
    ]
}

fn sglang_command(spec: &EngineSpec) -> Command {
    // The boot script holds no secret (the key comes from the environment), so a plain write in any lib/ mode; rewritten when this goethe's differs.
    let lib = spec.home.join("lib");
    let boot = lib.join("goethe_sglang_boot.py");
    if std::fs::read(&boot).ok().as_deref() != Some(SGLANG_BOOT.as_bytes()) {
        let _ = std::fs::create_dir_all(&lib);
        let _ = std::fs::write(&boot, SGLANG_BOOT);
    }
    let mut cmd = Command::new(resolve_python(spec.home));
    cmd.arg("-m")
        .arg("goethe_sglang_boot")
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
    let card_extras = {
        let raw = card_extra_args(spec.card);
        match crate::model_facts::read_config(spec.model_path) {
            Ok(cfg) => crate::sglang_infer::drop_quantization_when_config_declares(&raw, &cfg).0,
            Err(_) => raw,
        }
    };
    let extras: Vec<String> = card_extras.into_iter().chain(spec.load_extra.iter().cloned()).collect();
    cmd.args(memory_args(spec, &extras));
    cmd.args(graph_args(spec.max_conc, &extras));
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
    let card_vars = card_env(spec);
    let mut py_path = lib.display().to_string();
    if let Some((_, v)) = card_vars.iter().find(|(k, _)| k == "PYTHONPATH") {
        py_path = format!("{py_path}:{v}");
    }
    cmd.env_clear()
        .envs(kept_env())
        .env("PATH", engine_path(spec.home))
        .envs(default_cuda_home(spec.home, &card_vars))
        .envs(arch_env(spec.home, spec.cc))
        .envs(crate::cuda_compat::env(spec.home))
        .envs(compiler_env())
        .envs(card_vars)
        .envs(placement_env(spec))
        .env("PYTHONPATH", py_path)
        .env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        .env("SPT_NOENV", "1");
    engine_key_env(&mut cmd, spec.engine_key);
    cmd
}

#[cfg(test)]
mod tests {
    use super::super::test_spec;
    use super::{check_extra_args, graph_args};

    #[test]
    fn graphs_are_captured_up_to_the_concurrency_unless_the_user_says_otherwise() {
        assert_eq!(
            graph_args(4, &[]),
            ["--cuda-graph-max-bs-decode", "4", "--cuda-graph-backend-prefill", "disabled"]
        );
        assert_eq!(
            graph_args(0, &[]),
            ["--cuda-graph-max-bs-decode", "1", "--cuda-graph-backend-prefill", "disabled"]
        );
        for theirs in ["--cuda-graph-max-bs-decode=16", "--cuda-graph-bs-prefill", "--cuda-graph-config", "--disable-cuda-graph"] {
            assert!(graph_args(4, &[theirs.to_string()]).is_empty(), "{theirs}");
        }
    }

    #[test]
    fn the_boot_script_runs_sglang_only_as_the_main_module() {
        assert!(super::SGLANG_BOOT.contains("if __name__ == '__main__':"), "spawned scheduler processes re-import the main module; unguarded, each starts a second server");
        let guarded = super::SGLANG_BOOT.lines().skip_while(|l| !l.starts_with("if __name__")).skip(1);
        assert!(guarded.clone().all(|l| l.starts_with("    ")) && guarded.count() == 2, "argv and run_module both inside the guard");
    }

    #[test]
    fn a_changed_boot_script_is_written_again() {
        let home = tempfile::tempdir().unwrap();
        let plain: toml::Value = toml::from_str("runtime = \"sglang\"").unwrap();
        let boot = home.path().join("lib/goethe_sglang_boot.py");
        std::fs::create_dir_all(boot.parent().unwrap()).unwrap();
        std::fs::write(&boot, "an older boot script").unwrap();
        std::fs::set_permissions(boot.parent().unwrap(), std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
        super::super::command(&test_spec(home.path(), &plain, &[0])).unwrap();
        assert_eq!(std::fs::read_to_string(&boot).unwrap(), super::SGLANG_BOOT, "a stale copy is replaced, not kept");
    }

    #[test]
    fn reserved_table_rejects_former_owned_and_engine_pack_flags() {
        for bad in [
            "--host", "--port", "--tp-size", "--tensor-parallel-size", "--mem-fraction-static", "--context-length",
            "--max-running-requests", "--max-mamba-cache-size", "--swa-full-tokens-ratio", "--api-key", "--served-model-name",
            "--pack", "--weights", "--tokenizer", "--model-path", "--model",
        ] {
            let err = check_extra_args(&[bad.to_string(), "1".into()]).unwrap_err();
            assert!(err.contains("not allowed"), "{bad}: {err}");
        }
        check_extra_args(&["--chunked-prefill-size".into(), "4096".into(), "--trust-remote-code".into()]).unwrap();
    }
}
