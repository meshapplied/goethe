// SGLang argument handling: flag groups, merging, arity, reserved flags, and config-derived notes.

use serde_json::Value;
use std::path::Path;

const VALUE_FLAGS: &[&str] = &[
    "--api-key",
    "--attention-backend",
    "--chat-template",
    "--chunked-prefill-size",
    "--context-length",
    "--cpu-offload-gb",
    "--cuda-graph-backend-decode",
    "--cuda-graph-backend-prefill",
    "--cuda-graph-max-bs",
    "--cuda-graph-max-bs-decode",
    "--dtype",
    "--ep-size",
    "--fp4-gemm-backend",
    "--host",
    "--json-model-override-args",
    "--kv-cache-dtype",
    "--load-format",
    "--mamba-full-memory-ratio",
    "--max-mamba-cache-size",
    "--max-prefill-tokens",
    "--max-running-requests",
    "--max-total-tokens",
    "--mem-fraction-static",
    "--model-path",
    "--moe-runner-backend",
    "--page-size",
    "--ple-offload-backend",
    "--port",
    "--quantization",
    "--reasoning-parser",
    "--served-model-name",
    "--speculative-algorithm",
    "--speculative-draft-model-path",
    "--speculative-dspark-block-size",
    "--speculative-eagle-topk",
    "--speculative-num-draft-tokens",
    "--speculative-num-steps",
    "--speculative-token-map",
    "--swa-full-tokens-ratio",
    "--tokenizer-path",
    "--tool-call-parser",
    "--tp-size",
];

pub const RESERVED_FLAGS: &[(&str, &str)] = &[
    ("--model-path", "goethe passes the installed model directory"),
    ("--host", "engines listen on 127.0.0.1 behind the mux"),
    ("--port", "goethed assigns engine ports"),
    ("--tp-size", "use --tp"),
    ("--context-length", "use --ctx"),
    ("--max-running-requests", "use --max-conc"),
    ("--mem-fraction-static", "goethed derives it from the VRAM budget"),
    ("--served-model-name", "goethe serves the model under its id"),
    ("--api-key", "goethed sets a per-engine key"),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgGroup {
    pub flag: String,
    pub values: Vec<String>,
}

pub fn parse_groups(args: &[String]) -> Result<Vec<ArgGroup>, String> {
    // Split tokens into a flag plus the values that follow it; --flag=value counts as one group.
    let mut out: Vec<ArgGroup> = Vec::new();
    for tok in args {
        if tok.starts_with("--") {
            let (flag, inline) = match tok.split_once('=') {
                Some((f, v)) => (f.to_string(), Some(v.to_string())),
                None => (tok.clone(), None),
            };
            out.push(ArgGroup {
                flag,
                values: inline.into_iter().collect(),
            });
        } else if let Some(last) = out.last_mut() {
            last.values.push(tok.clone());
        } else {
            return Err(format!("SGLang argument {tok:?} is not preceded by a --flag"));
        }
    }
    Ok(out)
}

pub fn render(groups: &[ArgGroup]) -> Vec<String> {
    groups
        .iter()
        .flat_map(|g| std::iter::once(g.flag.clone()).chain(g.values.iter().cloned()))
        .collect()
}

pub fn takes_value(flag: &str) -> bool {
    VALUE_FLAGS.contains(&flag)
}

pub fn check_arity(groups: &[ArgGroup]) -> Result<(), String> {
    for g in groups {
        if takes_value(&g.flag) && g.values.is_empty() {
            return Err(format!("SGLang flag {} needs a value", g.flag));
        }
    }
    Ok(())
}

fn group_key(flag: &str) -> String {
    match flag.strip_prefix("--no-") {
        Some(rest) => format!("--{rest}"),
        None => flag.to_string(),
    }
}

fn upsert(out: &mut Vec<ArgGroup>, g: ArgGroup) {
    let key = group_key(&g.flag);
    match out.iter_mut().find(|x| group_key(&x.flag) == key) {
        Some(slot) => *slot = g,
        None => out.push(g),
    }
}

pub fn merge_extra_args(base: &[String], append: &[String]) -> Result<Vec<String>, String> {
    // Merge whole (flag, values) groups; a later group replaces an earlier one with the same flag.
    let mut out = Vec::new();
    for g in parse_groups(base)?.into_iter().chain(parse_groups(append)?) {
        upsert(&mut out, g);
    }
    check_arity(&out)?;
    Ok(render(&out))
}

pub fn reserved_flag_error(args: &[String]) -> Option<String> {
    let groups = parse_groups(args).ok()?;
    groups.iter().find_map(|g| {
        RESERVED_FLAGS
            .iter()
            .find(|(f, _)| *f == g.flag)
            .map(|(f, why)| format!("SGLang flag {f} is managed by goethe ({why})"))
    })
}

pub fn take_flag(args: &[String], flag: &str) -> Result<(Vec<String>, Option<String>), String> {
    // Remove a single-value flag from args and return its value.
    let mut value = None;
    let mut rest = Vec::new();
    for g in parse_groups(args)? {
        if g.flag == flag {
            value = Some(
                g.values
                    .first()
                    .cloned()
                    .ok_or_else(|| format!("SGLang flag {flag} needs a value"))?,
            );
        } else {
            rest.push(g);
        }
    }
    Ok((render(&rest), value))
}

pub fn has_flag(args: &[String], flag: &str) -> bool {
    parse_groups(args)
        .map(|gs| gs.iter().any(|g| g.flag == flag))
        .unwrap_or(false)
}

pub fn flag_value(args: &[String], flag: &str) -> Option<String> {
    parse_groups(args)
        .ok()?
        .into_iter()
        .rev()
        .find(|g| g.flag == flag)
        .and_then(|g| g.values.first().cloned())
}

pub fn resolve_model_dir_args(args: &[String], model_dir: &Path) -> Result<(Vec<String>, Vec<String>), String> {
    // Expand $MODEL_DIR; drop groups whose file is missing, and all speculative flags with them.
    let groups = parse_groups(args)?;
    let dir = model_dir.to_string_lossy().to_string();
    let mut notes = Vec::new();
    let mut drop_speculative = false;
    let mut kept = Vec::new();
    for mut g in groups {
        let mut missing = None;
        let mut values = Vec::with_capacity(g.values.len());
        for v in &g.values {
            match v.strip_prefix("$MODEL_DIR/") {
                Some(rel) if model_dir.join(rel).exists() => values.push(format!("{dir}/{rel}")),
                Some(rel) => {
                    missing = Some(rel.to_string());
                    values.push(v.clone());
                }
                None => values.push(v.clone()),
            }
        }
        g.values = values;
        match missing {
            Some(rel) => {
                notes.push(format!("{} skipped: {rel} is not in the model directory", g.flag));
                drop_speculative |= g.flag.starts_with("--speculative-");
            }
            None => kept.push(g),
        }
    }
    if drop_speculative {
        kept.retain(|g| !g.flag.starts_with("--speculative-"));
        notes.push("speculative decoding flags skipped with it".to_string());
    }
    Ok((render(&kept), notes))
}

pub fn wants_remote_code(cfg: &Value) -> bool {
    cfg.get("auto_map").is_some()
}

pub fn quantization_summary(dir: &Path, cfg: &Value) -> Option<String> {
    // Describe the checkpoint quantization SGLang will detect on its own (never forced by goethe).
    if let Some(qc) = cfg.get("quantization_config") {
        let method = qc.get("quant_method").and_then(|v| v.as_str()).unwrap_or("unknown");
        let algo = qc.get("quant_algo").and_then(|v| v.as_str());
        return Some(match algo {
            Some(a) => format!("{method} {a} (config.json)"),
            None => format!("{method} (config.json)"),
        });
    }
    if cfg.get("compression_config").is_some() {
        return Some("compressed-tensors (config.json compression_config)".to_string());
    }
    let text = std::fs::read_to_string(dir.join("hf_quant_config.json")).ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    let algo = v
        .pointer("/quantization/quant_algo")
        .or_else(|| v.get("quant_algo"))
        .and_then(|x| x.as_str())
        .unwrap_or("unknown");
    Some(format!("modelopt {algo} (hf_quant_config.json)"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn merge_keeps_values_and_last_wins() {
        let base = s(&["--reasoning-parser", "qwen3", "--tool-call-parser", "qwen", "--trust-remote-code"]);
        let add = s(&["--tool-call-parser", "qwen3_coder", "--page-size", "64"]);
        let out = merge_extra_args(&base, &add).unwrap();
        assert_eq!(
            out,
            s(&["--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder", "--trust-remote-code", "--page-size", "64"])
        );
        let same_values = s(&["--ep-size", "2", "--chunked-prefill-size", "2"]);
        assert_eq!(merge_extra_args(&same_values, &[]).unwrap(), same_values);
        let multi = s(&["--cuda-graph-bs", "1", "2", "4"]);
        assert_eq!(merge_extra_args(&multi, &s(&["--x"])).unwrap(), s(&["--cuda-graph-bs", "1", "2", "4", "--x"]));
        let neg = merge_extra_args(&s(&["--ple-offload-embedding"]), &s(&["--no-ple-offload-embedding"])).unwrap();
        assert_eq!(neg, s(&["--no-ple-offload-embedding"]));
        let inline = merge_extra_args(&s(&["--page-size=32"]), &s(&["--page-size", "64"])).unwrap();
        assert_eq!(inline, s(&["--page-size", "64"]));
    }

    #[test]
    fn arity_is_validated() {
        let err = merge_extra_args(&s(&["--reasoning-parser", "minimax-m3", "--tool-call-parser"]), &[]).unwrap_err();
        assert!(err.contains("--tool-call-parser needs a value"), "{err}");
        assert!(merge_extra_args(&s(&["--disable-shared-experts-fusion"]), &[]).is_ok());
        assert!(parse_groups(&s(&["stray", "--x"])).is_err());
        assert!(merge_extra_args(&s(&["--chunked-prefill-size", "-1"]), &[]).is_ok());
    }

    #[test]
    fn reserved_and_take_flag() {
        assert!(reserved_flag_error(&s(&["--max-running-requests", "8"])).is_some());
        assert!(reserved_flag_error(&s(&["--page-size", "8"])).is_none());
        let (rest, kv) = take_flag(&s(&["--kv-cache-dtype", "fp8_e4m3", "--page-size", "64"]), "--kv-cache-dtype").unwrap();
        assert_eq!(kv.as_deref(), Some("fp8_e4m3"));
        assert_eq!(rest, s(&["--page-size", "64"]));
    }

    #[test]
    fn model_dir_flags_follow_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let args = s(&[
            "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3",
            "--speculative-token-map", "$MODEL_DIR/spec/hot_self_32k.pt", "--page-size", "64",
        ]);
        let (out, notes) = resolve_model_dir_args(&args, dir.path()).unwrap();
        assert_eq!(out, s(&["--page-size", "64"]));
        assert_eq!(notes.len(), 2);
        std::fs::create_dir_all(dir.path().join("spec")).unwrap();
        std::fs::write(dir.path().join("spec/hot_self_32k.pt"), b"x").unwrap();
        let (out, notes) = resolve_model_dir_args(&args, dir.path()).unwrap();
        assert!(notes.is_empty());
        assert_eq!(out.len(), 8);
        assert_eq!(out[5], format!("{}/spec/hot_self_32k.pt", dir.path().display()));
    }

    #[test]
    fn quantization_is_detected_not_forced() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let ct: Value = serde_json::from_str(&std::fs::read_to_string(fixtures.join("qwen38-27b-nvfp4-ninfer/config.json")).unwrap()).unwrap();
        let summary = quantization_summary(&fixtures.join("qwen38-27b-nvfp4-ninfer"), &ct).unwrap();
        assert!(summary.starts_with("compressed-tensors"), "{summary}");
        let fnx_dir = fixtures.join("qwen38-flash-next-nvfp4-latest");
        let fnx: Value = serde_json::from_str(&std::fs::read_to_string(fnx_dir.join("config.json")).unwrap()).unwrap();
        assert_eq!(quantization_summary(&fnx_dir, &fnx).as_deref(), Some("modelopt MIXED_PRECISION (hf_quant_config.json)"));
        assert!(!wants_remote_code(&ct));
        let registry = s(&["--quantization", "modelopt_fp4"]);
        assert_eq!(merge_extra_args(&registry, &[]).unwrap(), registry);
    }
}
