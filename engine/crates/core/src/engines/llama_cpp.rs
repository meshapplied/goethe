// llama.cpp: llama-server on one GGUF, one GPU, OpenAI-compatible; the universal engine for GGUF models.

use super::{card_env, kept_env, placement_env, EngineSpec};
use std::path::{Path, PathBuf};
use std::process::Command;

// The llama.cpp commit goethe runs and is compared against.
pub const PIN: &str = "61dbf49228c8a96b72e40c89d93cb7c003411f8c";

pub const ENGINE: super::Engine = super::Engine {
    id: "llama.cpp",
    layer: None,
    // Body fields llama-server parses at PIN (tools/server/server-schema.cpp, server-common.cpp).
    params: Some(&[
        "temperature", "top_p", "top_k", "min_p", "max_tokens", "max_completion_tokens", "n", "stop", "seed",
        "presence_penalty", "frequency_penalty", "repeat_penalty", "logit_bias", "logprobs", "top_logprobs",
        "response_format", "tools", "tool_choice", "parallel_tool_calls", "ignore_eos", "json_schema", "grammar",
        "n_probs", "chat_template_kwargs", "reasoning_format", "encoding_format", "embd_normalize",
    ]),
    command: llama_command,
};

fn server_bin(home: &Path) -> Result<PathBuf, String> {
    // $LLAMA_SERVER, else the one under GOETHE_HOME; no PATH lookup, so the build is the one installed for goethe.
    let p = std::env::var_os("LLAMA_SERVER").map(PathBuf::from).unwrap_or_else(|| home.join("engines/llama.cpp/bin/llama-server"));
    if p.is_file() {
        Ok(p)
    } else {
        Err(format!("llama-server not found at {}: build llama.cpp {PIN} there or set LLAMA_SERVER", p.display()))
    }
}

pub fn gguf_in(dir: &Path, card: &toml::Value) -> Result<PathBuf, String> {
    // The card's gguf file, else the directory's only model GGUF (first shard of a split, no mmproj projector).
    if let Some(f) = card.get("gguf").and_then(|v| v.as_str()) {
        return Ok(dir.join(f));
    }
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".gguf") && !n.starts_with("mmproj") && (!n.contains("-of-") || n.contains("-00001-of-")))
        .collect();
    names.sort();
    match names.as_slice() {
        [one] => Ok(dir.join(one)),
        [] => Err(format!("{}: no GGUF file for llama.cpp", dir.display())),
        many => Err(format!("{}: several GGUF files ({}); set gguf = \"<file>\" in goethe_card.toml", dir.display(), many.join(", "))),
    }
}

fn llama_command(spec: &EngineSpec) -> Result<Command, String> {
    // One slot per parallel request in one KV pool of the size goethe budgeted, each capped at the context.
    if spec.gpus.len() != 1 || spec.tp != 1 {
        return Err(format!("llama.cpp runs on one GPU in goethe; this load asks for {} (tp {})", spec.gpus.len(), spec.tp));
    }
    let mut cmd = Command::new(server_bin(spec.home)?);
    cmd.arg("-m")
        .arg(gguf_in(spec.model_path, spec.card)?)
        .args(["--host", "127.0.0.1", "--port"])
        .arg(spec.port.to_string())
        .arg("--api-key")
        .arg(spec.engine_key)
        .arg("--alias")
        .arg(spec.served_name)
        .arg("-c")
        .arg(crate::chooser::kv_tokens(spec.ctx, spec.max_conc).to_string())
        .arg("-np")
        .arg(spec.max_conc.to_string())
        .arg("-kvu")
        .arg("--kv-unified-per-slot")
        .arg(spec.ctx.to_string())
        .args(["-ngl", "all", "--no-webui"]);
    if spec.card.get("task").and_then(|v| v.as_str()) == Some("embed") {
        cmd.arg("--embeddings");
    }
    cmd.env_clear()
        .envs(kept_env())
        .envs(card_env(spec))
        .envs(placement_env(spec));
    Ok(cmd)
}

#[cfg(test)]
mod tests {
    use super::super::{check_params, command, test_spec};
    use super::*;
    use serde_json::json;

    #[test]
    fn launches_llama_server_on_the_gguf_with_goethe_budget_and_one_gpu() {
        let d = tempfile::tempdir().unwrap();
        let bin = d.path().join("engines/llama.cpp/bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(bin.join("llama-server"), b"").unwrap();
        for f in ["m-Q4_K_M.gguf", "mmproj-m.gguf", "config.json"] {
            std::fs::write(d.path().join(f), b"x").unwrap();
        }
        let card: toml::Value = toml::from_str("runtime = \"llama.cpp\"").unwrap();
        let c = command(&test_spec(d.path(), &card, &[2])).unwrap();
        let a: Vec<String> = c.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        let at = |f: &str| a[a.iter().position(|x| x == f).unwrap() + 1].clone();
        assert!(at("-m").ends_with("m-Q4_K_M.gguf"));
        assert_eq!((at("--host").as_str(), at("--alias").as_str(), at("-np").as_str()), ("127.0.0.1", "org/m", "4"));
        assert_eq!((at("-c"), at("--kv-unified-per-slot")), (crate::chooser::kv_tokens(4096, 4).to_string(), "4096".to_string()));
        assert!(!a.iter().any(|x| x == "--embeddings"));
        let env = |k: &str| c.get_envs().find(|(n, _)| *n == k).and_then(|(_, v)| v).map(|v| v.to_string_lossy().into_owned());
        assert_eq!(env("CUDA_VISIBLE_DEVICES").as_deref(), Some("2"));
        let emb: toml::Value = toml::from_str("runtime = \"llama.cpp\"\ntask = \"embed\"").unwrap();
        assert!(command(&test_spec(d.path(), &emb, &[2])).unwrap().get_args().any(|x| x == "--embeddings"));
        assert!(command(&test_spec(d.path(), &card, &[2, 3])).unwrap_err().contains("one GPU"));
        std::fs::write(d.path().join("m-Q8_0.gguf"), b"x").unwrap();
        assert!(command(&test_spec(d.path(), &card, &[2])).unwrap_err().contains("several GGUF"));
        let pick: toml::Value = toml::from_str("runtime = \"llama.cpp\"\ngguf = \"m-Q8_0.gguf\"").unwrap();
        assert!(command(&test_spec(d.path(), &pick, &[2])).is_ok());
    }

    #[test]
    fn a_missing_server_names_the_pin() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("m.gguf"), b"x").unwrap();
        let card: toml::Value = toml::from_str("runtime = \"llama.cpp\"").unwrap();
        if std::env::var_os("LLAMA_SERVER").is_none() {
            assert!(command(&test_spec(d.path(), &card, &[0])).unwrap_err().contains(PIN));
        }
    }

    #[test]
    fn a_split_gguf_is_served_from_its_first_shard() {
        let d = tempfile::tempdir().unwrap();
        for f in ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"] {
            std::fs::write(d.path().join(f), b"x").unwrap();
        }
        assert!(gguf_in(d.path(), &toml::Value::Table(Default::default())).unwrap().ends_with("m-00001-of-00002.gguf"));
    }

    #[test]
    fn conformance_the_same_request_is_served_or_refused_by_name() {
        // Request, then the parameter each engine refuses (None: passed on; goethe-engine checks its pack's io.params itself).
        let ge = &super::super::goethe_engine::ENGINE;
        let cases = [
            (json!({"model": "m", "messages": [], "temperature": 0, "max_tokens": 8, "stop": ["x"], "seed": 1, "stream": true}), None),
            (json!({"model": "m", "prompt": "hi", "top_p": 0.9, "top_k": 40, "n": 1, "logprobs": 2}), None),
            (json!({"model": "m", "messages": [], "structured": {"kind": "regex", "spec": "a"}}), Some("structured")),
            (json!({"model": "m", "input": ["a"], "encoding_format": "float"}), None),
            (json!({"model": "m", "input": ["a"], "dimensions": 64}), Some("dimensions")),
            (json!({"model": "m", "input": ["a"], "normalize": null}), None),
        ];
        for (body, refused) in cases {
            assert_eq!(check_params(&ENGINE, body.to_string().as_bytes()).err().as_deref(), refused, "{body}");
            assert!(check_params(ge, body.to_string().as_bytes()).is_ok(), "goethe-engine refuses in the engine");
        }
    }
}
