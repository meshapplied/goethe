// Infer SGLang CLI args from Hugging Face config.json.

use serde_json::Value;

pub fn infer_extra_sglang_args(cfg: &Value) -> (Vec<String>, Vec<String>) {
    let mut warnings = Vec::new();
    let mut extra = infer_quant_args(cfg, &mut warnings);
    if cfg.get("auto_map").is_some() && !extra.iter().any(|a| a == "--trust-remote-code") {
        extra.push("--trust-remote-code".to_string());
    }
    (extra, warnings)
}

fn infer_quant_args(cfg: &Value, warnings: &mut Vec<String>) -> Vec<String> {
    let Some(qc) = cfg.get("quantization_config") else {
        return Vec::new();
    };
    let format = qc
        .get("quant_method")
        .or_else(|| qc.get("format"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let targets = serde_json::to_string(qc).unwrap_or_default().to_ascii_lowercase();
    if targets.contains("nvfp4") || format.contains("modelopt") {
        return vec![
            "--quantization".to_string(),
            "modelopt_fp4".to_string(),
        ];
    }
    if targets.contains("compressed") {
        return vec![
            "--quantization".to_string(),
            "compressed-tensors".to_string(),
        ];
    }
    warnings.push("quantization_config present but extra_sglang_args not inferred".to_string());
    Vec::new()
}

pub fn merge_extra_args(base: &[String], append: &[String]) -> Vec<String> {
    let mut out = base.to_vec();
    for a in append {
        if !out.contains(a) {
            out.push(a.clone());
        }
    }
    out
}
