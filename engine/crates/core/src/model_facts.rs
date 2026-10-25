// Model shape facts from config.json and weights on disk: KV per token, per-request state, sizes.

use serde_json::Value;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Debug, Clone)]
pub struct ModelFacts {
    pub architecture: String,
    pub static_vram_gb: f64,
    pub kv_elems_per_token: u64,
    pub max_ctx: u64,
    pub num_attention_heads: u64,
    pub num_kv_heads: u64,
    pub attention: String,
    pub swa_kv_elems_per_request: u64,
    pub linear_state_bytes_per_request: u64,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct LayerPlan {
    pub full: u64,
    pub swa: u64,
    pub linear: u64,
}

const LM_KEYS: &[&str] = &[
    "text_config",
    "llm_config",
    "language_config",
    "language_model",
    "thinker_config",
];

pub fn resolve_language_config(cfg: &Value) -> Result<&Value, String> {
    // The nested language-model config of multimodal wrappers, or the root config.
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
        .unwrap_or_else(|| dir.file_name().and_then(|s| s.to_str()).unwrap_or("unknown").to_string());
    facts_from_config(&cfg, weights_gb(dir)?, registry_arch, &model_id)
}

pub fn linear_state_bytes_in_dir(dir: &Path) -> u64 {
    // Per-request linear-attention state of the model in dir, 0 for models without such layers or an unreadable config.
    let Ok(cfg) = read_config(dir) else { return 0 };
    let Ok(lm) = resolve_language_config(&cfg) else { return 0 };
    let layers = read_u64(lm, &cfg, &["num_hidden_layers", "n_layer"]).unwrap_or(0);
    linear_state_bytes(lm, layer_plan(lm, layers))
}

pub fn facts_from_config(
    cfg: &Value,
    static_gb: f64,
    registry_arch: Option<&str>,
    model_id: &str,
) -> Result<ModelFacts, String> {
    // Derive KV per token from full-attention layers only, plus per-request window and state sizes.
    let lm = resolve_language_config(cfg).map_err(|_| format!("{model_id}: cannot read language model config"))?;
    let num_attention_heads = require_u64(lm, cfg, &["num_attention_heads", "n_head"], model_id)?;
    let layers = require_u64(lm, cfg, &["num_hidden_layers", "n_layer"], model_id)?;
    let max_ctx = read_u64(lm, cfg, &["max_position_embeddings", "n_positions"]).unwrap_or(4096);
    let plan = layer_plan(lm, layers);
    let kv = attention_kv(cfg, lm, layers, plan, max_ctx, num_attention_heads, model_id)?;
    let architecture = registry_arch
        .map(String::from)
        .unwrap_or_else(|| infer_architecture(cfg, lm, model_id));
    Ok(ModelFacts {
        architecture,
        static_vram_gb: static_gb,
        kv_elems_per_token: kv.elems,
        max_ctx,
        num_attention_heads,
        num_kv_heads: kv.heads,
        attention: kv.attention,
        swa_kv_elems_per_request: kv.swa_elems,
        linear_state_bytes_per_request: linear_state_bytes(lm, plan),
    })
}

pub fn layer_plan(lm: &Value, layers: u64) -> LayerPlan {
    // Classify layers as full attention, sliding window, or linear/state-space.
    let full_ids = lm
        .get("full_attention_layer_ids")
        .or_else(|| lm.pointer("/linear_attn_config/full_attn_layers"))
        .and_then(|v| v.as_array());
    if let Some(ids) = full_ids {
        let full = ids.len() as u64;
        return LayerPlan { full, swa: 0, linear: layers.saturating_sub(full) };
    }
    if let Some(types) = lm.get("layer_types").and_then(|v| v.as_array()) {
        let mut p = LayerPlan::default();
        for t in types.iter().filter_map(|t| t.as_str()) {
            if t.contains("linear") || t.contains("mamba") || t.contains("ssm") {
                p.linear += 1;
            } else if t.contains("sliding") || t.contains("local") || t.contains("chunked") {
                p.swa += 1;
            } else {
                p.full += 1;
            }
        }
        return p;
    }
    if let Some(pattern) = lm.get("hybrid_override_pattern").and_then(|v| v.as_str()) {
        let count = |c: char| pattern.chars().filter(|x| *x == c).count() as u64;
        return LayerPlan { full: count('*'), swa: 0, linear: count('M') };
    }
    if let Some(n) = lm.get("full_attention_interval").and_then(|v| v.as_u64()).filter(|n| *n > 1) {
        return LayerPlan { full: layers / n, swa: 0, linear: layers - layers / n };
    }
    let pattern = lm.get("sliding_window_pattern").and_then(|v| v.as_u64()).filter(|n| *n > 1);
    if let (Some(n), Some(_)) = (pattern, lm.get("sliding_window").and_then(|v| v.as_u64())) {
        return LayerPlan { full: layers / n, swa: layers - layers / n, linear: 0 };
    }
    LayerPlan { full: layers, swa: 0, linear: 0 }
}

struct KvShape {
    elems: u64,
    heads: u64,
    attention: String,
    swa_elems: u64,
}

fn attention_kv(
    cfg: &Value,
    lm: &Value,
    layers: u64,
    plan: LayerPlan,
    max_ctx: u64,
    num_attention_heads: u64,
    model_id: &str,
) -> Result<KvShape, String> {
    // KV elements per token; MLA caches the latent, GQA/MHA caches K and V per KV head.
    let kv_heads = read_u64(lm, cfg, &["num_key_value_heads", "multi_query_group_num"]).unwrap_or(num_attention_heads);
    let hybrid = plan.swa > 0 || plan.linear > 0;
    if let Some(rank) = read_u64(lm, cfg, &["kv_lora_rank"]).filter(|r| *r > 0) {
        let rope = read_u64(lm, cfg, &["qk_rope_head_dim"]).unwrap_or(64);
        let indexer = read_u64(lm, cfg, &["index_head_dim"]).unwrap_or(0);
        let mla_layers = if plan.full > 0 { plan.full } else { layers };
        return Ok(KvShape {
            elems: mla_layers * (rank + rope + indexer),
            heads: 1,
            attention: if hybrid { "hybrid" } else { "mla" }.to_string(),
            swa_elems: 0,
        });
    }
    let head_dim = match read_u64(lm, cfg, &["head_dim"]) {
        Some(d) => d,
        None => {
            let hidden = require_u64(lm, cfg, &["hidden_size", "n_embd", "d_model"], model_id)?;
            hidden / num_attention_heads.max(1)
        }
    };
    if lm.get("compress_ratios").is_some() || (lm.get("sliding_window").is_some() && kv_heads == 1) {
        eprintln!("warning: {model_id}: compressed attention, kv estimate is an upper bound");
        return Ok(KvShape {
            elems: 2 * layers * head_dim,
            heads: 1,
            attention: "gqa".to_string(),
            swa_elems: 0,
        });
    }
    let full_heads = read_u64(lm, cfg, &["num_global_key_value_heads"]).unwrap_or(kv_heads);
    let full_dim = read_u64(lm, cfg, &["global_head_dim"]).unwrap_or(head_dim);
    let window = read_u64(lm, cfg, &["sliding_window"]).unwrap_or(max_ctx).min(max_ctx);
    Ok(KvShape {
        elems: 2 * plan.full * full_heads * full_dim,
        heads: full_heads,
        attention: if hybrid { "hybrid" } else { "gqa" }.to_string(),
        swa_elems: 2 * plan.swa * kv_heads * head_dim * window,
    })
}

pub fn linear_state_bytes(lm: &Value, plan: LayerPlan) -> u64 {
    // Per-request recurrent state of linear-attention (GDN) or Mamba2 layers.
    if plan.linear == 0 {
        return 0;
    }
    let get = |k: &str| lm.get(k).and_then(|v| v.as_u64());
    if let (Some(vh), Some(kd), Some(vd)) = (
        get("linear_num_value_heads"),
        get("linear_key_head_dim"),
        get("linear_value_head_dim"),
    ) {
        let kh = get("linear_num_key_heads").unwrap_or(vh);
        let kernel = get("linear_conv_kernel_dim").unwrap_or(4).max(1);
        let ssm_bytes = match lm.get("mamba_ssm_dtype").and_then(|v| v.as_str()) {
            Some(d) if d.contains("16") => 2,
            _ => 4,
        };
        let per_layer = vh * kd * vd * ssm_bytes + (2 * kh * kd + vh * vd) * (kernel - 1) * 2;
        return per_layer * plan.linear;
    }
    let kda = |k: &str| lm.pointer(&format!("/linear_attn_config/{k}")).and_then(|v| v.as_u64());
    if let (Some(h), Some(d)) = (kda("num_heads"), kda("head_dim")) {
        let kernel = kda("short_conv_kernel_size").unwrap_or(4).max(1);
        return (h * d * d * 4 + 3 * h * d * (kernel - 1) * 2) * plan.linear;
    }
    if let (Some(h), Some(hd), Some(state)) = (get("mamba_num_heads"), get("mamba_head_dim"), get("ssm_state_size")) {
        let groups = get("n_groups").unwrap_or(1);
        let kernel = get("conv_kernel").unwrap_or(4).max(1);
        let per_layer = h * hd * state * 4 + (h * hd + 2 * groups * state) * (kernel - 1) * 2;
        return per_layer * plan.linear;
    }
    0
}

pub fn infer_architecture(cfg: &Value, lm: &Value, model_id: &str) -> String {
    if is_moe(cfg, lm, model_id) {
        "moe".to_string()
    } else {
        "dense".to_string()
    }
}

fn is_moe(cfg: &Value, lm: &Value, _model_id: &str) -> bool {
    // Any expert count, experts-per-token, or MoE intermediate size marks a mixture of experts.
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
    arch.contains("Moe") || arch.contains("MoE")
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
    read_u64(lm, root, keys).ok_or_else(|| format!("{model_id}: config.json has none of {}", keys.join(", ")))
}

pub fn read_config(dir: &Path) -> Result<Value, String> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn weight_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    // Files holding the weights: those named by model.safetensors.index.json, else every *.safetensors.
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        let text = std::fs::read_to_string(&index).map_err(|e| format!("{}: {e}", index.display()))?;
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", index.display()))?;
        let map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| format!("{} has no weight_map", index.display()))?;
        let names: BTreeSet<&str> = map.values().filter_map(|x| x.as_str()).collect();
        let mut out = Vec::new();
        for n in names {
            let p = dir.join(n);
            if !p.is_file() {
                return Err(format!("{} is listed in model.safetensors.index.json but missing", p.display()));
            }
            out.push(p);
        }
        return Ok(out);
    }
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("safetensors") && p.is_file())
        .collect();
    out.sort();
    Ok(out)
}

pub fn weights_gb(dir: &Path) -> Result<f64, String> {
    let mut bytes = 0u64;
    for p in weight_files(dir)? {
        bytes += std::fs::metadata(&p).map_err(|e| format!("{}: {e}", p.display()))?.len();
    }
    Ok(bytes as f64 / GIB)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn kv(cfg: Value) -> u64 {
        facts_from_config(&cfg, 0.0, None, "t").unwrap().kv_elems_per_token
    }

    fn fixture(name: &str) -> Value {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name).join("config.json");
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
    }

    #[test]
    fn llama_8b_gqa_65536() {
        let cfg = json!({"num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 4096});
        assert_eq!(kv(cfg), 65536);
    }

    #[test]
    fn head_dim_falls_back_to_attention_heads() {
        let qwen25_7b = json!({"num_hidden_layers": 28, "num_attention_heads": 28, "num_key_value_heads": 4, "hidden_size": 3584});
        assert_eq!(kv(qwen25_7b), 2 * 28 * 4 * 128);
        let phi4mm = json!({"num_hidden_layers": 32, "num_attention_heads": 24, "num_key_value_heads": 8, "hidden_size": 3072, "sliding_window": 262144, "max_position_embeddings": 131072});
        assert_eq!(kv(phi4mm), 2 * 32 * 8 * 128);
    }

    #[test]
    fn mha_without_kv_heads_uses_attention_heads() {
        let gpt2 = json!({"n_layer": 12, "n_head": 12, "n_embd": 768, "n_positions": 1024});
        let f = facts_from_config(&gpt2, 0.0, None, "gpt2").unwrap();
        assert_eq!(f.kv_elems_per_token, 2 * 12 * 12 * 64);
        assert_eq!(f.num_kv_heads, 12);
        assert_eq!(f.max_ctx, 1024);
    }

    #[test]
    fn mistral_small_81920() {
        let cfg = json!({"num_hidden_layers": 40, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128});
        assert_eq!(kv(cfg), 81920);
    }

    #[test]
    fn qwen38_27b_fixture_counts_full_attention_only() {
        let f = facts_from_config(&fixture("qwen38-27b-nvfp4-ninfer"), 23.0, None, "27b").unwrap();
        assert_eq!(f.kv_elems_per_token, 2 * 16 * 4 * 256);
        assert_eq!((f.num_attention_heads, f.num_kv_heads), (24, 4));
        assert_eq!(f.attention, "hybrid");
        assert_eq!(f.architecture, "dense");
        assert_eq!(f.max_ctx, 262144);
        let per_layer = 48 * 128 * 128 * 4 + (2 * 16 * 128 + 48 * 128) * 3 * 2;
        assert_eq!(f.linear_state_bytes_per_request, 48 * per_layer);
    }

    #[test]
    fn flash_next_fixture_counts_full_attention_only() {
        let f = facts_from_config(&fixture("qwen38-flash-next-nvfp4-latest"), 0.0, None, "fn").unwrap();
        assert_eq!(f.kv_elems_per_token, 2 * 12 * 2 * 256);
        assert_eq!((f.num_attention_heads, f.num_kv_heads), (24, 2));
        assert_eq!(f.attention, "hybrid");
        assert_eq!(f.architecture, "moe");
        assert!(f.linear_state_bytes_per_request > 100 * 1024 * 1024);
    }

    #[test]
    fn interval_and_nemotron_pattern() {
        let next = json!({"num_hidden_layers": 48, "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256, "full_attention_interval": 4});
        assert_eq!(kv(next), 2 * 12 * 2 * 256);
        let nemotron = json!({"num_hidden_layers": 8, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
            "hybrid_override_pattern": "M-M*-M*-", "mamba_num_heads": 8, "mamba_head_dim": 64, "ssm_state_size": 128});
        let f = facts_from_config(&nemotron, 0.0, None, "nh").unwrap();
        assert_eq!(f.kv_elems_per_token, 2 * 2 * 8 * 128);
        assert!(f.linear_state_bytes_per_request > 0);
    }

    #[test]
    fn gemma_sliding_window_layers_are_per_request() {
        let mut types = vec![json!("sliding_attention"); 50];
        types.extend(vec![json!("full_attention"); 10]);
        let gemma4 = json!({"text_config": {"num_hidden_layers": 60, "num_attention_heads": 32, "num_key_value_heads": 16, "head_dim": 256,
            "global_head_dim": 512, "num_global_key_value_heads": 4, "sliding_window": 1024, "max_position_embeddings": 262144, "layer_types": types}});
        let f = facts_from_config(&gemma4, 0.0, None, "g4").unwrap();
        assert_eq!(f.kv_elems_per_token, 2 * 10 * 4 * 512);
        assert_eq!(f.swa_kv_elems_per_request, 2 * 50 * 16 * 256 * 1024);
        assert_eq!(f.num_kv_heads, 4);
        let gemma3 = json!({"num_hidden_layers": 12, "num_attention_heads": 8, "num_key_value_heads": 4, "head_dim": 256, "sliding_window": 512, "sliding_window_pattern": 6});
        assert_eq!(kv(gemma3), 2 * 2 * 4 * 256);
    }

    #[test]
    fn qwen3_vl_gqa_96256() {
        let cfg = json!({"text_config": {"num_hidden_layers": 94, "num_attention_heads": 64, "num_key_value_heads": 4, "head_dim": 128}});
        assert_eq!(kv(cfg), 96256);
    }

    #[test]
    fn nemotron_nested_language_config() {
        let cfg = json!({"architectures": ["NemotronH_NemotronVLForConditionalGeneration"], "language_config": {
            "num_hidden_layers": 48, "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128,
            "moe_intermediate_size": 8192, "num_experts": 128}});
        let facts = facts_from_config(&cfg, 120.0, None, "nemotron").unwrap();
        assert_eq!(facts.architecture, "moe");
        assert_eq!(facts.num_attention_heads, 64);
        assert!(facts.kv_elems_per_token > 128);
    }

    #[test]
    fn deepseek_v4_pro_moe_compressed() {
        let cfg = json!({"n_routed_experts": 384, "num_hidden_layers": 60, "num_attention_heads": 32, "num_key_value_heads": 1,
            "head_dim": 512, "compress_ratios": [1,1,1], "sliding_window": 128});
        let facts = facts_from_config(&cfg, 900.0, None, "dsv4").unwrap();
        assert_eq!(facts.architecture, "moe");
        assert!(facts.kv_elems_per_token > 0);
    }

    #[test]
    fn deepseek_v3_mla() {
        let cfg = json!({"num_hidden_layers": 61, "num_attention_heads": 128, "num_key_value_heads": 128, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "n_routed_experts": 256});
        let f = facts_from_config(&cfg, 0.0, None, "dsv3").unwrap();
        assert_eq!((f.kv_elems_per_token, f.num_kv_heads, f.attention.as_str()), (35136, 1, "mla"));
    }

    #[test]
    fn kimi_k3_hybrid_mla_kv() {
        let cfg = json!({"architectures": ["KimiK3ForConditionalGeneration"], "text_config": {"num_attention_heads": 96,
            "num_key_value_heads": 96, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "num_hidden_layers": 93, "num_experts": 896,
            "linear_attn_config": {"full_attn_layers": (0..24u64).collect::<Vec<_>>(), "kda_layers": (24..93u64).collect::<Vec<_>>(),
                "num_heads": 96, "head_dim": 128, "short_conv_kernel_size": 4}}});
        let facts = facts_from_config(&cfg, 200.0, None, "kimi").unwrap();
        assert_eq!(facts.architecture, "moe");
        assert_eq!(facts.kv_elems_per_token, 24 * 576);
        assert_eq!((facts.num_kv_heads, facts.attention.as_str()), (1, "hybrid"));
        assert_eq!(facts.linear_state_bytes_per_request, 69 * (96 * 128 * 128 * 4 + 3 * 96 * 128 * 3 * 2));
    }

    #[test]
    fn dsa_indexer_and_sparse_layer_names() {
        let mut types = vec![json!("linear_attention"); 34];
        types.extend(vec![json!("deepseek_sparse_attention"); 11]);
        let glm = json!({"num_hidden_layers": 45, "num_attention_heads": 64, "num_key_value_heads": 64, "kv_lora_rank": 512,
            "qk_rope_head_dim": 0, "head_dim": 0, "index_head_dim": 128, "layer_types": types});
        let f = facts_from_config(&glm, 0.0, None, "glm").unwrap();
        assert_eq!(f.kv_elems_per_token, 11 * (512 + 128));
        assert_eq!(f.attention, "hybrid");
    }

    #[test]
    fn index_weight_map_skips_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model-00001-of-00002.safetensors"), vec![0u8; 1000]).unwrap();
        std::fs::write(dir.path().join("model-00002-of-00002.safetensors"), vec![0u8; 500]).unwrap();
        std::fs::write(dir.path().join("consolidated.safetensors"), vec![0u8; 1500]).unwrap();
        let index = json!({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors", "c": "model-00001-of-00002.safetensors"}});
        std::fs::write(dir.path().join("model.safetensors.index.json"), index.to_string()).unwrap();
        assert_eq!(weight_files(dir.path()).unwrap().len(), 2);
        assert!((weights_gb(dir.path()).unwrap() - 1500.0 / GIB).abs() < 1e-12);
        std::fs::remove_file(dir.path().join("model-00002-of-00002.safetensors")).unwrap();
        assert!(weight_files(dir.path()).unwrap_err().contains("missing"));
        std::fs::remove_file(dir.path().join("model.safetensors.index.json")).unwrap();
        assert_eq!(weight_files(dir.path()).unwrap().len(), 2);
    }
}
