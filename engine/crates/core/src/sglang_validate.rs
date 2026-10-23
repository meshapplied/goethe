// Validate registry extra_sglang_args against installed SGLang.

use crate::hf;
use crate::paths::goethe_home;
use crate::registry::Registry;
use std::path::Path;

pub fn validate_registry_sglang(reg: &Registry, online: bool) -> Vec<String> {
    // Check flag names, arity and parser names against the installed SGLang's own help and registries.
    let home = goethe_home();
    let py = home.join("py-env/bin/python");
    if !py.is_file() {
        return vec!["note: --check-sglang skipped (py-env not present)".to_string()];
    }
    let flags = launch_server_flags(&py);
    let reasoning = reasoning_parser_keys(&py);
    let tool = tool_call_parser_keys(&py);
    let arch_keys = sglang_model_arch_keys(&py);
    let mut errors = Vec::new();
    for m in &reg.models {
        if m.effective_runtime() == "ternary" {
            continue;
        }
        let groups = match crate::sglang_infer::parse_groups(&m.extra_sglang_args) {
            Ok(g) => g,
            Err(e) => {
                errors.push(format!("{}: {e}", m.id));
                continue;
            }
        };
        for g in &groups {
            let value = g.values.first().map(String::as_str);
            let parsers = match g.flag.as_str() {
                "--reasoning-parser" => Some(&reasoning),
                "--tool-call-parser" => Some(&tool),
                _ => None,
            };
            if let (Some(keys), Some(v)) = (parsers, value) {
                if !keys.is_empty() && v != "auto" && !keys.iter().any(|k| k == v) {
                    errors.push(format!("{}: unknown {} {v}", m.id, g.flag));
                }
            }
            if flags.is_empty() {
                continue;
            }
            match flags.iter().find(|(f, _)| f == &g.flag) {
                None => errors.push(format!("{}: unknown sglang flag {}", m.id, g.flag)),
                Some((_, true)) if g.values.is_empty() => errors.push(format!("{}: {} needs a value", m.id, g.flag)),
                Some((_, false)) if !g.values.is_empty() => {
                    errors.push(format!("{}: {} takes no value (got {:?})", m.id, g.flag, g.values))
                }
                _ => {}
            }
        }
        if !arch_keys.is_empty() && online && !hf::offline_mode() {
            let rev = m.revision.as_deref().unwrap_or("main");
            if let Ok(cfg) = hf::fetch_config_json(&m.id, rev) {
                if let Some(arch) = cfg
                    .get("architectures")
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.first())
                    .and_then(|v| v.as_str())
                {
                    if !arch_keys.iter().any(|k| k == arch) {
                        errors.push(format!("warning: {}: architectures[0] {arch} unknown to installed SGLang", m.id));
                    }
                }
            }
        }
    }
    errors
}

fn launch_server_flags(py: &Path) -> Vec<(String, bool)> {
    let out = std::process::Command::new(py)
        .args(["-m", "sglang.launch_server", "--help"])
        .output();
    match out {
        Ok(o) if o.status.success() => parse_help_flags(&String::from_utf8_lossy(&o.stdout)),
        _ => Vec::new(),
    }
}

fn parse_help_flags(help: &str) -> Vec<(String, bool)> {
    // argparse option lines: "  --flag VALUE, --alias VALUE   description"; true means it takes a value.
    let mut out = Vec::new();
    for line in help.lines().filter(|l| l.starts_with("  -")) {
        let sig = line.trim_start().split("  ").next().unwrap_or("");
        for part in sig.split(", ") {
            let mut toks = part.split_whitespace();
            if let Some(flag) = toks.next().filter(|f| f.starts_with("--")) {
                out.push((flag.to_string(), toks.next().is_some()));
            }
        }
    }
    out
}

fn reasoning_parser_keys(py: &Path) -> Vec<String> {
    run_py_keys(
        py,
        r#"
try:
    from sglang.srt.parser.reasoning_parser import ReasoningParser
except ImportError:
    from sglang.srt.reasoning_parser import ReasoningParser
print("\n".join(ReasoningParser.DetectorMap.keys()))
"#,
    )
}

fn tool_call_parser_keys(py: &Path) -> Vec<String> {
    run_py_keys(
        py,
        r#"
from sglang.srt.function_call.function_call_parser import FunctionCallParser
print("\n".join(FunctionCallParser.ToolCallParserEnum.keys()))
"#,
    )
}

fn sglang_model_arch_keys(py: &Path) -> Vec<String> {
    run_py_keys(
        py,
        r#"
try:
    from sglang.srt.models.registry import ModelRegistry
    print("\n".join(ModelRegistry.models.keys()))
except Exception:
    pass
"#,
    )
}

fn run_py_keys(py: &Path, script: &str) -> Vec<String> {
    let out = std::process::Command::new(py).arg("-c").arg(script).output();
    let Ok(out) = out else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_flags_and_arity() {
        let help = "options:\n  -h, --help            show help\n  --model-path MODEL_PATH, --model MODEL_PATH\n                        The path\n  --trust-remote-code   Whether to trust\n  --ple-offload-embedding, --no-ple-offload-embedding\n  --tool-call-parser {qwen,qwen3_coder}\n                        Parser\n  --cuda-graph-bs-decode CUDA_GRAPH_BS_DECODE [CUDA_GRAPH_BS_DECODE ...]\n";
        let f = parse_help_flags(help);
        let get = |n: &str| f.iter().find(|(x, _)| x == n).map(|(_, v)| *v);
        assert_eq!(get("--model-path"), Some(true));
        assert_eq!(get("--model"), Some(true));
        assert_eq!(get("--trust-remote-code"), Some(false));
        assert_eq!(get("--ple-offload-embedding"), Some(false));
        assert_eq!(get("--no-ple-offload-embedding"), Some(false));
        assert_eq!(get("--tool-call-parser"), Some(true));
        assert_eq!(get("--cuda-graph-bs-decode"), Some(true));
        assert_eq!(get("--help"), Some(false));
    }
}
