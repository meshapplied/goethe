// Model shape facts read from config.json and weights on disk.

use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ModelFacts {
    pub architecture: String,
    pub static_vram_gb: f64,
    pub kv_elems_per_token: u64,
    pub max_ctx: u64,
    pub num_attention_heads: u64,
    pub num_kv_heads: u64,
    pub attention: String,
}

const LM_KEYS: &[&str] = &[
    "text_config",
    "llm_config",
    "language_config",
    "language_model",
    "thinker_config",
];

pub fn resolve_language_config(cfg: &Value) -> Result<&Value, String> {
    for key in LM_KEYS {
        if let Some(v) = cfg.get(key) {
            if language_shape_present(v) {
                return Ok(v);
            }
        }
    }
    if language_shape_present(cfg) {
        return Ok(cfg);
    }
    if let Some(map) = cfg.as_object() {
        for v in map.values() {
            if v.is_object() && language_shape_present(v) {
                return Ok(v);
            }
        }
    }
    Err("cannot read language model config".to_string())
}

fn language_shape_present(v: &Value) -> bool {
    v.get("num_hidden_layers").is_some()
        || v.get("n_layer").is_some()
        || v.get("num_attention_heads").is_some()
        || v.get("n_head").is_some()
}

pub fn facts_from_install_dir(dir: &Path, registry_arch: Option<&str>) -> Result<ModelFacts, String> {
    let cfg = read_config(dir)?;
    let model_id = cfg
        .get("_name_or_path")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| {
            dir.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        });
    facts_from_config(&cfg, safetensors_gb(dir)?, registry_arch, &model_id)
}

pub fn facts_from_config(
    cfg: &Value,
    static_gb: f64,
    registry_arch: Option<&str>,
    model_id: &str,
) -> Result<ModelFacts, String> {
    let lm = resolve_language_config(cfg).map_err(|_| format!("{model_id}: cannot read language model config"))?;
    let num_attention_heads = require_u64(lm, cfg, &["num_attention_heads", "n_head"], model_id)?;
    let _layers = require_u64(lm, cfg, &["num_hidden_layers", "n_layer"], model_id)?;
    let mut num_kv_heads = read_u64(lm, cfg, &["num_key_value_heads", "n_head"])
        .unwrap_or(num_attention_heads);
    let max_ctx = read_u64(lm, cfg, &["max_position_embeddings"])
        .or_else(|| cfg.get("max_position_embeddings").and_then(|v| v.as_u64()))
        .unwrap_or(4096);
    let attention = infer_attention(cfg, lm, model_id);
    let (kv_elems, kv_heads_adj) = compute_kv_elems(cfg, lm, &attention, model_id)?;
    if attention == "mla" || kv_heads_adj == 1 {
        num_kv_heads = kv_heads_adj;
    }
    let architecture = registry_arch
        .map(String::from)
        .unwrap_or_else(|| infer_architecture(cfg, lm, model_id));
    Ok(ModelFacts {
        architecture,
        static_vram_gb: static_gb,
        kv_elems_per_token: kv_elems,
        max_ctx,
        num_attention_heads,
        num_kv_heads,
        attention,
    })
}

pub fn kv_elems_from_config(cfg: &Value) -> u64 {
    let model_id = cfg
        .get("_name_or_path")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let Ok(lm) = resolve_language_config(cfg) else {
        return 0;
    };
    let attention = infer_attention(cfg, lm, model_id);
    compute_kv_elems(cfg, lm, &attention, model_id)
        .map(|(k, _)| k)
        .unwrap_or(0)
}

fn compute_kv_elems(
    cfg: &Value,
    lm: &Value,
    attention: &str,
    model_id: &str,
) -> Result<(u64, u64), String> {
    if is_compressed_attention(cfg, lm) {
        eprintln!(
            "warning: {model_id}: compressed attention, kv estimate is an upper bound"
        );
        return Ok((compressed_kv_upper_bound(cfg, lm, model_id)?, 1));
    }
    match attention {
        "mla" => mla_kv(cfg, lm, model_id),
        "hybrid" => hybrid_kv(cfg, lm, model_id),
        _ => gqa_kv(cfg, lm, model_id),
    }
}

fn mla_kv(cfg: &Value, lm: &Value, model_id: &str) -> Result<(u64, u64), String> {
    let rank = read_u64(lm, cfg, &["kv_lora_rank"]).unwrap_or(0);
    let rope = read_u64(lm, cfg, &["qk_rope_head_dim", "qk_nope_head_dim"]).unwrap_or(0);
    if rank == 0 {
        eprintln!("warning: {model_id}: mla attention but kv_lora_rank missing; using gqa estimate");
        return gqa_kv(cfg, lm, model_id);
    }
    let layers = mla_layer_count(cfg, lm, model_id)?;
    Ok(((layers as u64) * (rank + rope), 1))
}

fn hybrid_kv(cfg: &Value, lm: &Value, model_id: &str) -> Result<(u64, u64), String> {
    if lm.get("kv_lora_rank").is_some() || cfg.get("kv_lora_rank").is_some() {
        let mla_layers = count_mla_or_full_layers(cfg, lm, model_id)?;
        let rank = read_u64(lm, cfg, &["kv_lora_rank"]).unwrap_or(0);
        let rope = read_u64(lm, cfg, &["qk_rope_head_dim"]).unwrap_or(64);
        if rank > 0 {
            return Ok(((mla_layers as u64) * (rank + rope), 1));
        }
    }
    let layers = count_full_attention_layers(lm, model_id);
    if layers == 0 {
        eprintln!(
            "warning: {model_id}: hybrid layer marking unclear; using all hidden layers for kv"
        );
        return gqa_kv_all_layers(cfg, lm, model_id);
    }
    gqa_kv_layers(cfg, lm, layers, model_id)
}

fn gqa_kv(cfg: &Value, lm: &Value, model_id: &str) -> Result<(u64, u64), String> {
    let layers = total_hidden_layers(cfg, lm, model_id)? as usize;
    gqa_kv_layers(cfg, lm, layers, model_id)
}

fn gqa_kv_all_layers(cfg: &Value, lm: &Value, model_id: &str) -> Result<(u64, u64), String> {
    let layers = total_hidden_layers(cfg, lm, model_id)? as usize;
    gqa_kv_layers(cfg, lm, layers, model_id)
}

fn gqa_kv_layers(cfg: &Value, lm: &Value, layers: usize, _model_id: &str) -> Result<(u64, u64), String> {
    let kv_heads = read_u64(lm, cfg, &["num_key_value_heads", "n_head"]).unwrap_or(1);
    let hidden = read_u64(lm, cfg, &["hidden_size"]).unwrap_or(64);
    let head_dim = read_u64(lm, cfg, &["head_dim"]).unwrap_or(hidden / kv_heads.max(1));
    Ok((2 * layers as u64 * kv_heads * head_dim, kv_heads))
}

fn compressed_kv_upper_bound(cfg: &Value, lm: &Value, model_id: &str) -> Result<u64, String> {
    let layers = total_hidden_layers(cfg, lm, model_id)?;
    let kv_heads = read_u64(lm, cfg, &["num_key_value_heads"]).unwrap_or(1);
    let head_dim = read_u64(lm, cfg, &["head_dim"]).unwrap_or(512);
    Ok(2 * layers * kv_heads.max(1) * head_dim)
}

fn is_compressed_attention(_cfg: &Value, lm: &Value) -> bool {
    lm.get("compress_ratios").is_some()
        || lm.get("sliding_window").is_some()
            && lm.get("num_key_value_heads").and_then(|v| v.as_u64()) == Some(1)
}

fn count_full_attention_layers(lm: &Value, model_id: &str) -> usize {
    if let Some(ids) = lm.get("full_attention_layer_ids").and_then(|v| v.as_array()) {
        return ids.len();
    }
    if let Some(lt) = lm.get("layer_types").and_then(|v| v.as_array()) {
        return lt
            .iter()
            .filter(|t| {
                t.as_str().is_some_and(|s| {
                    s.contains("full_attention") || s == "attention" || s.contains("mla")
                })
            })
            .count();
    }
    if lm.get("linear_attn_config").is_some() || lm.get("sparse_attention_config").is_some() {
        eprintln!(
            "warning: {model_id}: linear/sparse attention config present but layer ids missing"
        );
    }
    0
}

fn count_mla_or_full_layers(cfg: &Value, lm: &Value, model_id: &str) -> Result<usize, String> {
    let n = count_full_attention_layers(lm, model_id);
    if n > 0 {
        return Ok(n);
    }
    mla_layer_count(cfg, lm, model_id)
}

fn mla_layer_count(cfg: &Value, lm: &Value, model_id: &str) -> Result<usize, String> {
    let n = count_full_attention_layers(lm, model_id);
    if n > 0 {
        return Ok(n);
    }
    Ok(total_hidden_layers(cfg, lm, model_id)? as usize)
}

fn total_hidden_layers(cfg: &Value, lm: &Value, model_id: &str) -> Result<u64, String> {
    require_u64(lm, cfg, &["num_hidden_layers", "n_layer"], model_id)
}

fn infer_attention(cfg: &Value, lm: &Value, _model_id: &str) -> String {
    let arch = cfg
        .get("architectures")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if (arch.contains("KimiLinear") || arch.contains("LinearForCausalLM"))
        && (lm.get("kv_lora_rank").is_some() || cfg.get("kv_lora_rank").is_some())
    {
        return "hybrid".to_string();
    }
    if lm.get("kv_lora_rank").is_some() || cfg.get("kv_lora_rank").is_some() {
        if lm.get("linear_attn_config").is_some()
            || lm.get("layer_types").is_some()
            || lm.get("full_attention_layer_ids").is_some()
        {
            return "hybrid".to_string();
        }
        return "mla".to_string();
    }
    if lm.get("layer_types").and_then(|v| v.as_array()).is_some_and(|a| {
        a.iter().any(|t| t.as_str().is_some_and(|s| s.contains("linear")))
            && a.iter()
                .any(|t| t.as_str().is_some_and(|s| s.contains("full_attention") || s == "attention"))
    }) {
        return "hybrid".to_string();
    }
    if lm.get("linear_attn_config").is_some() && lm.get("full_attention_layer_ids").is_some() {
        return "hybrid".to_string();
    }
    "gqa".to_string()
}

pub fn infer_architecture(cfg: &Value, lm: &Value, model_id: &str) -> String {
    if is_moe(cfg, lm, model_id) {
        "moe".to_string()
    } else {
        "dense".to_string()
    }
}

fn is_moe(cfg: &Value, lm: &Value, _model_id: &str) -> bool {
    let expert_keys = [
        "num_experts",
        "num_local_experts",
        "n_routed_experts",
        "moe_num_experts",
        "num_routed_experts",
    ];
    for key in expert_keys {
        if read_u64(lm, cfg, &[key]).unwrap_or(0) > 1 {
            return true;
        }
    }
    for key in ["num_experts_per_tok", "num_experts_per_token"] {
        if read_u64(lm, cfg, &[key]).unwrap_or(0) > 0 {
            return true;
        }
    }
    if lm.get("moe_intermediate_size").is_some() || cfg.get("moe_intermediate_size").is_some() {
        return true;
    }
    let arch = cfg
        .get("architectures")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if (arch.contains("Moe") || arch.contains("Sparse"))
        && arch.to_ascii_lowercase().contains("expert")
    {
        return true;
    }
    if arch.contains("Moe") || arch.contains("MoE") {
        return true;
    }
    false
}

fn read_u64(lm: &Value, root: &Value, keys: &[&str]) -> Option<u64> {
    for key in keys {
        if let Some(v) = lm.get(key).and_then(|v| v.as_u64()) {
            return Some(v);
        }
        if let Some(v) = root.get(key).and_then(|v| v.as_u64()) {
            return Some(v);
        }
    }
    None
}

fn require_u64(lm: &Value, root: &Value, keys: &[&str], model_id: &str) -> Result<u64, String> {
    read_u64(lm, root, keys).ok_or_else(|| format!("{model_id}: cannot read language model config"))
}

fn read_config(dir: &Path) -> Result<Value, String> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("config.json: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("config.json parse: {e}"))
}

fn safetensors_gb(dir: &Path) -> Result<f64, String> {
    let mut bytes = 0u64;
    for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            bytes += entry.metadata().map_err(|e| e.to_string())?.len();
        }
    }
    Ok(bytes as f64 / (1024.0_f64.powi(3)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn kv(cfg: Value) -> u64 {
        kv_elems_from_config(&cfg)
    }

    #[test]
    fn llama_8b_gqa_65536() {
        let cfg = json!({
            "_name_or_path": "meta-llama/Llama-3.1-8B",
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_size": 4096
        });
        assert_eq!(kv(cfg), 65536);
    }

    #[test]
    fn mistral_small_81920() {
        let cfg = json!({
            "_name_or_path": "mistral/Mistral-Small-3.1-24B",
            "num_hidden_layers": 40,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128
        });
        assert_eq!(kv(cfg), 81920);
    }

    #[test]
    fn qwen3_vl_gqa_96256() {
        let cfg = json!({
            "_name_or_path": "Qwen/Qwen3-VL-235B",
            "text_config": {
                "num_hidden_layers": 94,
                "num_attention_heads": 64,
                "num_key_value_heads": 4,
                "head_dim": 128
            }
        });
        assert_eq!(kv(cfg), 96256);
    }

    #[test]
    fn qwen38_hybrid_32768() {
        let lt: Vec<_> = (0..64)
            .map(|i| {
                if i % 4 == 3 {
                    json!("full_attention")
                } else {
                    json!("linear_attention")
                }
            })
            .collect();
        let cfg = json!({
            "_name_or_path": "Qwen3.8-27B",
            "text_config": {
                "num_hidden_layers": 64,
                "num_attention_heads": 32,
                "head_dim": 256,
                "num_key_value_heads": 4,
                "layer_types": lt
            }
        });
        assert_eq!(kv(cfg), 32768);
    }

    #[test]
    fn nemotron_nested_language_config() {
        let cfg = json!({
            "_name_or_path": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4",
            "architectures": ["NemotronH_NemotronVLForConditionalGeneration"],
            "language_config": {
                "num_hidden_layers": 48,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "moe_intermediate_size": 8192,
                "num_experts": 128
            }
        });
        let facts = facts_from_config(&cfg, 120.0, None, "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4").unwrap();
        assert_eq!(facts.architecture, "moe");
        assert_eq!(facts.num_attention_heads, 64);
        assert!(facts.kv_elems_per_token > 128);
    }

    #[test]
    fn deepseek_v4_pro_moe_compressed() {
        let cfg = json!({
            "_name_or_path": "nvidia/DeepSeek-V4-Pro-NVFP4",
            "n_routed_experts": 384,
            "num_hidden_layers": 60,
            "num_attention_heads": 32,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "compress_ratios": [1,1,1],
            "sliding_window": 128
        });
        let facts = facts_from_config(&cfg, 900.0, None, "nvidia/DeepSeek-V4-Pro-NVFP4").unwrap();
        assert_eq!(facts.architecture, "moe");
        assert!(facts.kv_elems_per_token > 0);
    }

    #[test]
    fn minimax_m3_moe() {
        let cfg = json!({
            "_name_or_path": "MiniMaxAI/MiniMax-M3",
            "num_local_experts": 128,
            "num_hidden_layers": 60,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "sparse_attention_config": {}
        });
        let facts = facts_from_config(&cfg, 100.0, None, "MiniMaxAI/MiniMax-M3").unwrap();
        assert_eq!(facts.architecture, "moe");
    }

    #[test]
    fn kimi_k3_hybrid_mla_kv() {
        let cfg = json!({
            "_name_or_path": "moonshotai/Kimi-K3",
            "architectures": ["KimiLinearForCausalLM"],
            "text_config": {
                "num_attention_heads": 32,
                "kv_lora_rank": 512,
                "qk_rope_head_dim": 64,
                "num_hidden_layers": 93,
                "num_experts": 896,
                "full_attention_layer_ids": (0..93u64).collect::<Vec<_>>()
            }
        });
        let facts = facts_from_config(&cfg, 200.0, None, "moonshotai/Kimi-K3").unwrap();
        assert_eq!(facts.architecture, "moe");
        assert_eq!(facts.attention, "hybrid");
        assert_eq!(facts.kv_elems_per_token, 53568);
        assert_eq!(facts.num_kv_heads, 1);
    }
}
