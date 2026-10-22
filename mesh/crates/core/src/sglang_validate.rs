// Validate registry extra_sglang_args against installed SGLang.

use crate::hf;
use crate::paths::mesh_home;
use crate::registry::Registry;
use std::path::Path;

pub fn validate_registry_sglang(reg: &Registry, online: bool) -> Vec<String> {
    let home = mesh_home();
    let py = home.join("py-env/bin/python");
    if !py.is_file() {
        return vec!["note: --check-sglang skipped (py-env not present)".to_string()];
    }
    let help_flags = launch_server_flags(&py);
    let reasoning = reasoning_parser_keys(&py);
    let tool = tool_call_parser_keys(&py);
    let arch_keys = sglang_model_arch_keys(&py);
    let mut errors = Vec::new();
    for m in &reg.models {
        if m.effective_runtime() == "ternary" {
            continue;
        }
        let mut i = 0;
        while i < m.extra_sglang_args.len() {
            let arg = &m.extra_sglang_args[i];
            if arg == "--reasoning-parser" {
                if let Some(v) = m.extra_sglang_args.get(i + 1) {
                    if !reasoning.is_empty() && !reasoning.contains(v) {
                        errors.push(format!("{}: unknown --reasoning-parser {v}", m.id));
                    }
                    i += 2;
                    continue;
                }
            }
            if arg == "--tool-call-parser" {
                if let Some(v) = m.extra_sglang_args.get(i + 1) {
                    if !tool.is_empty() && !tool.contains(v) {
                        errors.push(format!("{}: unknown --tool-call-parser {v}", m.id));
                    }
                    i += 2;
                    continue;
                }
            }
            if arg.starts_with("--") && !help_flags.is_empty() && !flag_known(arg, &help_flags) {
                errors.push(format!("{}: unknown sglang flag {arg}", m.id));
            }
            i += 1;
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
                        errors.push(format!(
                            "warning: {}: architectures[0] {arch} unknown to installed SGLang",
                            m.id
                        ));
                    }
                }
            }
        }
    }
    errors
}

fn flag_known(flag: &str, help: &[String]) -> bool {
    help.iter().any(|h| h == flag || h.starts_with(&format!("{flag}=")))
}

fn launch_server_flags(py: &Path) -> Vec<String> {
    let out = std::process::Command::new(py)
        .args(["-m", "sglang.launch_server", "--help"])
        .output();
    let Ok(out) = out else {
        return Vec::new();
    };
    if !out.status.success() {
        return Vec::new();
    }
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines()
        .filter_map(|l| l.split_whitespace().next())
        .filter(|t| t.starts_with("--"))
        .map(String::from)
        .collect()
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
