// SGLang: python -m sglang.launch_server with Goethe-owned flags, card and load extras; ternary is SGLang plus its kernel env.

use super::{arch_env, card_env, compiler_env, default_cuda_home, engine_path, expand_home, kept_env, placement_env, resolve_python, EngineSpec};
use std::process::Command;

pub const SGLANG: super::Engine = super::Engine {
    id: "sglang",
    layer: Some(crate::pyenv::Layer::Sglang),
    params: None,
    command: |spec| Ok(sglang_command(spec)),
};

pub const TERNARY: super::Engine = super::Engine { id: "ternary", command: ternary_command, ..SGLANG };

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
    "--pack",
    "--weights",
    "--tokenizer",
];
const QUIET_PATHS: &[&str] = &["/health", "/v1/models", "/get_model_info"];

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
                "extra SGLang arg {arg} is not allowed: Goethe sets {owned} itself (use goethe load options instead)"
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

fn memory_args(spec: &EngineSpec, extras: &[String]) -> Vec<String> {
    // The window ratio, state slots and chunk size the budget assumed; a chunk size the user set wins.
    let Some(m) = crate::memory_spec::installed(spec.model_path, spec.card, spec.load_extra) else { return Vec::new() };
    let args = crate::chooser::engine_args(&m, spec.ctx, spec.max_conc);
    let user_chunk = extras.iter().any(|a| a.starts_with("--chunked-prefill-size"));
    args.chunks(2).filter(|p| !(user_chunk && p[0] == "--chunked-prefill-size")).flatten().cloned().collect()
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
    let card_vars = card_env(spec);
    cmd.env_clear()
        .envs(kept_env())
        .env("PATH", engine_path(spec.home))
        .envs(default_cuda_home(spec.home, &card_vars))
        .envs(arch_env(spec.home, spec.cc))
        .envs(crate::cuda_compat::env(spec.home))
        .envs(compiler_env())
        .envs(card_vars)
        .envs(placement_env(spec))
        .env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        .env("SPT_NOENV", "1");
    cmd
}

fn ternary_command(spec: &EngineSpec) -> Result<Command, String> {
    // SGLang with the card's top-level TERNARY_* keys in its env, once the ternary kernels are installed.
    crate::runtime::require_ternary_runtime(spec.home)?;
    let mut cmd = sglang_command(spec);
    for (k, v) in spec.card.as_table().into_iter().flatten() {
        if let (true, Some(s)) = (k.starts_with("TERNARY_"), v.as_str()) {
            cmd.env(k, expand_home(s, spec.home));
        }
    }
    Ok(cmd)
}

#[cfg(test)]
mod tests {
    use super::super::test_spec;

    #[test]
    fn ternary_gets_its_kernel_env_once_installed() {
        let home = tempfile::tempdir().unwrap();
        let card: toml::Value = toml::from_str("runtime = \"ternary\"\nTERNARY_PACK_DIR = \"$GOETHE_HOME/p\"").unwrap();
        let spec = test_spec(home.path(), &card, &[0]);
        assert!(super::super::command(&spec).unwrap_err().contains("ternary runtime not installed"));
        std::fs::create_dir_all(crate::runtime::ternary_kernels_dir(home.path())).unwrap();
        let cmd = super::super::command(&spec).unwrap();
        let want = format!("{}/p", home.path().display());
        assert!(cmd.get_envs().any(|(k, v)| k == "TERNARY_PACK_DIR" && v.is_some_and(|v| v.to_string_lossy() == want)));
    }
}
