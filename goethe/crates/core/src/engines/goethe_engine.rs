// goethe-engine: a signed kernel pack plus the weights its verification lists, served by python -m goethe_engine.

use super::{card_env, engine_key_env, kept_env, placement_env, resolve_python, EngineSpec};
use std::path::PathBuf;
use std::process::Command;

pub const ENGINE: super::Engine = super::Engine {
    id: "goethe-engine",
    overhead_gb: crate::calc::METAL_OVERHEAD_GB,
    layer: Some(crate::pyenv::Layer::Engine),
    params: None,
    numerics: true,
    command: goethe_engine_command,
};

pub const PACK_FILE: &str = "pack.gcpack";
pub const PACK_VERIFY: &str = "pack.verify.json";

fn goethe_engine_command(spec: &EngineSpec) -> Result<Command, String> {
    // The pack and the weights its verification listed (in order), both next to the tokenizer in the model dir; GOETHE_ENGINE_ALLOW_UNSIGNED (it disables signature checks) is never passed on.
    let dir = spec.model_path;
    let text = std::fs::read_to_string(dir.join(PACK_VERIFY)).map_err(|e| format!("{}: {e}; run goethe install again", dir.join(PACK_VERIFY).display()))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| format!("{PACK_VERIFY}: {e}"))?;
    let mut weights: Vec<PathBuf> = v["weights"].as_array().into_iter().flatten().filter_map(|w| w.as_str()).map(|f| dir.join(f)).collect();
    if weights.is_empty() || weights.iter().any(|w| !w.is_file()) {
        return Err(format!("{}: the weights the pack names are missing", dir.display()));
    }
    if weights.iter().all(|w| w.extension().is_some_and(|x| x == "safetensors")) {
        weights = vec![dir.to_path_buf()];
    }
    let drafter: Vec<PathBuf> = v["drafter"].as_array().into_iter().flatten().filter_map(|w| w.as_str()).map(|f| dir.join(f)).filter(|f| f.is_file()).collect();
    let shards = !drafter.is_empty() && drafter.iter().all(|w| w.extension().is_some_and(|x| x == "safetensors"));
    weights.extend(if shards { vec![dir.join("drafter")] } else { drafter });
    let mut cmd = Command::new(resolve_python(spec.home));
    cmd.args(["-m", "goethe_engine", "serve", "--host", "127.0.0.1", "--port"])
        .arg(spec.port.to_string())
        .arg("--context-length")
        .arg(spec.ctx.to_string())
        .arg("--max-running-requests")
        .arg(spec.max_conc.to_string())
        .arg("--served-model-name")
        .arg(spec.served_name)
        .arg("--api-key")
        .arg(spec.engine_key)
        .arg("--pack")
        .arg(dir.join(PACK_FILE))
        .arg("--weights")
        .args(&weights)
        .arg("--tokenizer")
        .arg(dir);
    if let Some(n) = spec.numerics {
        cmd.arg("--numerics").arg(n.name());
    }
    for o in crate::pack_update::installed_overlays(dir) {
        cmd.arg("--overlay").arg(dir.join("overlays").join(o));
    }
    cmd.env_clear()
        .envs(kept_env())
        .envs(card_env(spec))
        .envs(placement_env(spec))
        .env_remove("GOETHE_ENGINE_ALLOW_UNSIGNED");
    engine_key_env(&mut cmd, spec.engine_key);
    Ok(cmd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{command, sglang::check_extra_args, test_spec, ENV_KEEP, ENV_KEEP_PREFIX};
    use std::path::Path;

    fn cmd_in(dir: &Path, card: &str) -> Result<Command, String> {
        let card: toml::Value = toml::from_str(card).unwrap();
        let spec = EngineSpec { home: Path::new("/h"), ..test_spec(dir, &card, &[3]) };
        command(&spec)
    }

    fn model_dir(weights: &[&str]) -> tempfile::TempDir {
        let d = tempfile::tempdir().unwrap();
        for w in weights {
            std::fs::write(d.path().join(w), b"w").unwrap();
        }
        std::fs::write(d.path().join(PACK_VERIFY), serde_json::json!({ "weights": weights }).to_string()).unwrap();
        d
    }

    #[test]
    fn a_listed_drafter_follows_the_target_weights() {
        let d = model_dir(&["t-1.safetensors"]);
        std::fs::create_dir(d.path().join("drafter")).unwrap();
        std::fs::write(d.path().join("drafter/d.gguf"), b"w").unwrap();
        let v = serde_json::json!({"weights": ["t-1.safetensors"], "drafter": ["drafter/d.gguf", "drafter/gone.gguf"]});
        std::fs::write(d.path().join(PACK_VERIFY), v.to_string()).unwrap();
        let a: Vec<String> = cmd_in(d.path(), "runtime = \"goethe-engine\"").unwrap().get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        let at = |f: &str| a.iter().position(|x| x == f).unwrap();
        let dir = d.path().to_string_lossy().into_owned();
        assert_eq!(a[at("--weights") + 1..at("--tokenizer")], [dir.clone(), format!("{dir}/drafter/d.gguf")]);
    }

    #[test]
    fn launches_the_engine_with_the_pack_the_listed_weights_and_the_tokenizer_dir() {
        let d = model_dir(&["t.gguf", "d.gguf"]);
        let c = cmd_in(d.path(), "runtime = \"goethe-engine\"").unwrap();
        let a: Vec<String> = c.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(&a[..3], ["-m", "goethe_engine", "serve"]);
        let at = |f: &str| a.iter().position(|x| x == f).unwrap();
        let dir = d.path().to_string_lossy().into_owned();
        assert_eq!(a[at("--pack") + 1], format!("{dir}/{PACK_FILE}"));
        assert_eq!(a[at("--weights") + 1..at("--tokenizer")], [format!("{dir}/t.gguf"), format!("{dir}/d.gguf")]);
        assert_eq!(a[at("--tokenizer") + 1], dir);
        assert_eq!(a[at("--max-running-requests") + 1], "4");
    }

    #[test]
    fn a_pinned_numerics_reaches_the_engine_and_none_leaves_its_default() {
        let d = model_dir(&["t.gguf"]);
        let card: toml::Value = toml::from_str("runtime = \"goethe-engine\"").unwrap();
        let args = |n| command(&EngineSpec { numerics: n, ..test_spec(d.path(), &card, &[3]) }).unwrap().get_args().map(|a| a.to_string_lossy().into_owned()).collect::<Vec<_>>();
        let a = args(Some(goethe_proto::Numerics::Tolerance));
        assert_eq!(a[a.iter().position(|x| x == "--numerics").unwrap() + 1], "tolerance");
        assert!(!args(None).iter().any(|x| x == "--numerics"));
        for plain in ["sglang", "llama.cpp", "ninfer"] {
            let card: toml::Value = toml::from_str(&format!("runtime = \"{plain}\"")).unwrap();
            let e = command(&EngineSpec { numerics: Some(goethe_proto::Numerics::Invariant), ..test_spec(d.path(), &card, &[3]) }).unwrap_err();
            assert!(e.contains("--numerics invariant needs a kernel pack"), "{plain}: {e}");
        }
    }

    #[test]
    fn missing_verification_or_weights_refuse_to_launch() {
        let d = tempfile::tempdir().unwrap();
        assert!(cmd_in(d.path(), "runtime = \"goethe-engine\"").is_err());
        let d = model_dir(&["t.gguf"]);
        std::fs::remove_file(d.path().join("t.gguf")).unwrap();
        assert!(cmd_in(d.path(), "runtime = \"goethe-engine\"").unwrap_err().contains("missing"));
    }

    #[test]
    fn unsigned_override_is_never_passed_on() {
        let d = model_dir(&["t.gguf"]);
        let c = cmd_in(d.path(), "runtime = \"goethe-engine\"\n[env]\nGOETHE_ENGINE_ALLOW_UNSIGNED = \"1\"").unwrap();
        let v = c.get_envs().find(|(k, _)| *k == "GOETHE_ENGINE_ALLOW_UNSIGNED");
        assert!(v.is_none_or(|(_, v)| v.is_none()));
        assert!(!ENV_KEEP.contains(&"GOETHE_ENGINE_ALLOW_UNSIGNED") && !ENV_KEEP_PREFIX.iter().any(|p| "GOETHE_ENGINE_ALLOW_UNSIGNED".starts_with(p)));
    }

    #[test]
    fn pack_weights_and_tokenizer_flags_are_owned() {
        for bad in ["--pack", "--pack=x", "--weights", "--tokenizer", "--tokenizer=/x"] {
            assert!(check_extra_args(&[bad.to_string(), "v".into()]).unwrap_err().contains("not allowed"), "{bad}");
        }
    }
}
