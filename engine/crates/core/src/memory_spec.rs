// What SGLang allocates for a model, from its config.json: token pools per GPU generation, window pool, state slots.

use crate::model_facts::{self, read_u64};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const BF16: u64 = 2;
const DSV4_PAGE: u64 = 256;
const SWA_EVICTION_INTERVAL: u64 = 128;
pub const CHUNKED_PREFILL: u64 = 8192;
const LOAD_PEAK_LAYERS: f64 = 7.0;
const EXTRA_BUFFER_ARCHS: &[&str] = &[
    "KimiLinearForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5ForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "InternS2PreviewForConditionalGeneration",
    "MiniCPMV4_6ForConditionalGeneration",
    "BailingMoeV2_5ForCausalLM",
    "BailingMoeV3ForCausalLM",
    "BailingMoeV3VLForConditionalGeneration",
    "FalconH1ForCausalLM",
    "GraniteMoeHybridForCausalLM",
    "Glm5NextForConditionalGeneration",
    "NemotronHForCausalLM",
    "NemotronHPuzzleForCausalLM",
    "NemotronH_Omni_Reasoning_V3",
    "KimiK3ForConditionalGeneration",
    "InklingForConditionalGeneration",
    "InklingForConditionalGenerationMTP",
];

fn is_zero(v: &u64) -> bool {
    *v == 0
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MemorySpec {
    pub kv_bytes_per_token: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_bytes_per_token_sm10: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_bytes_per_token_sm12: Option<u64>,
    #[serde(default)]
    pub kv_heads: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub kv_replicated_bytes_per_token: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub swa_bytes_per_token: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub swa_heads: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub swa_window: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub state_bytes_per_slot: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub state_slots_per_request: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub request_bytes: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub fixed_bytes: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub chunked_prefill: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub static_state_bytes: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub load_peak_bytes: u64,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub spec_state_bytes_per_request: u64,
}

impl MemorySpec {
    pub fn kv_bytes_for(&self, cc: &str) -> u64 {
        // The token pool layout SGLang picks on this GPU generation (DSA models switch to FP8 on SM 10 and up).
        let major = crate::gpu_arch::parse_cc(cc).map_or(0, |(m, _)| m);
        match major {
            12.. => self.kv_bytes_per_token_sm12.or(self.kv_bytes_per_token_sm10).unwrap_or(self.kv_bytes_per_token),
            10 | 11 => self.kv_bytes_per_token_sm10.unwrap_or(self.kv_bytes_per_token),
            _ => self.kv_bytes_per_token,
        }
    }

    pub fn swa_tokens(&self, max_conc: u64) -> u64 {
        // SGLang's own per-request window sizing (window, eviction interval, two pages, four prefix tails) plus two chunks.
        if self.swa_bytes_per_token == 0 {
            return 0;
        }
        let page = 128;
        let w = self.swa_window;
        max_conc.max(1) * (w + SWA_EVICTION_INTERVAL + 2 * page + 4 * (w + page)) + 2 * CHUNKED_PREFILL + page
    }
}

pub fn installed(dir: &std::path::Path, card: &toml::Value, load_extra: &[String]) -> Option<MemorySpec> {
    // An installed model's spec for its card's KV dtype and the engine flags it runs with (card plus this load's extras).
    let cfg = model_facts::read_config(dir).ok()?;
    let kv_dtype = card.get("kv_cache_dtype").and_then(|v| v.as_str()).filter(|d| !d.eq_ignore_ascii_case("auto"));
    let mut spec = from_config(&cfg, kv_dtype).ok()?;
    let facts = model_facts::facts_from_config(&cfg, 0.0, None, "installed").ok()?;
    let elem = crate::install::kv_element_bytes(kv_dtype.unwrap_or("auto"), &cfg);
    spec.static_state_bytes = (facts.swa_kv_elems_per_request * elem + facts.linear_state_bytes_per_request) * 4;
    let mut args = crate::engine_cmd::card_extra_args(card);
    args.extend_from_slice(load_extra);
    apply_engine_args(&mut spec, Some(&cfg), &args);
    Some(spec)
}

fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    // The last value given for flag, as `--flag value` or `--flag=value`.
    let mut found = None;
    for (i, a) in args.iter().enumerate() {
        if a == flag {
            found = args.get(i + 1).map(String::as_str);
        } else if let Some(v) = a.strip_prefix(flag).and_then(|r| r.strip_prefix('=')) {
            found = Some(v);
        }
    }
    found
}

pub fn apply_engine_args(spec: &mut MemorySpec, cfg: Option<&Value>, args: &[String]) {
    // Engine-allocated linear state: --mamba-ssm-dtype sets its dtype, speculative decoding adds one per draft token.
    if spec.state_bytes_per_slot == 0 {
        return;
    }
    let lm = cfg.and_then(|c| model_facts::resolve_language_config(c).ok().map(|lm| (c, lm)));
    if let (Some(d), Some((cfg, lm))) = (flag_value(args, "--mamba-ssm-dtype"), lm) {
        let bytes = if d.contains("16") { 2 } else { 4 };
        let layers = read_u64(lm, cfg, &["num_hidden_layers", "n_layer"]).unwrap_or(0);
        let state = model_facts::linear_state_bytes_as(lm, model_facts::layer_plan(lm, layers), Some(bytes));
        if state > 0 {
            spec.state_bytes_per_slot = state;
        }
    }
    let algo = flag_value(args, "--speculative-algorithm").unwrap_or("");
    if algo.is_empty() || algo.eq_ignore_ascii_case("none") {
        return;
    }
    let draft_tokens = flag_value(args, "--speculative-num-draft-tokens")
        .and_then(|v| v.parse::<u64>().ok())
        .or_else(|| flag_value(args, "--speculative-num-steps").and_then(|v| v.parse::<u64>().ok()).map(|s| s + 1))
        .unwrap_or(4);
    spec.spec_state_bytes_per_request = draft_tokens * spec.state_bytes_per_slot;
}

pub fn from_config(cfg: &Value, kv_dtype: Option<&str>) -> Result<MemorySpec, String> {
    // Dispatch on the attention family SGLang sizes differently: DeepSeek V4, MLA (with DSA), MiniMax sparse, MHA/GQA.
    let lm = model_facts::resolve_language_config(cfg)?;
    let model_type = lm.get("model_type").or_else(|| cfg.get("model_type")).and_then(|v| v.as_str()).unwrap_or("");
    let arch = cfg.pointer("/architectures/0").and_then(|v| v.as_str()).unwrap_or("");
    let block_layers = lm.get("layers_block_type").and_then(|v| v.as_array()).map(|b| b.len() as u64);
    let layers = read_u64(lm, cfg, &["num_hidden_layers", "n_layer"]).or(block_layers).unwrap_or(0);
    let plan = model_facts::layer_plan(lm, layers);
    let mut spec = if model_type.starts_with("deepseek_v4") || arch.starts_with("DeepseekV4") {
        deepseek_v4(lm, layers)?
    } else if read_u64(lm, cfg, &["kv_lora_rank"]).is_some_and(|r| r > 0) {
        mla(lm, cfg, arch, if plan.full > 0 { plan.full } else { layers })
    } else if lm.pointer("/sparse_attention_config/use_sparse_attention").and_then(|v| v.as_bool()) == Some(true) {
        minimax_sparse(lm, cfg, layers)
    } else {
        mha(lm, cfg, plan)
    };
    spec.load_peak_bytes = fp4_expert_load_peak(cfg, lm, model_type);
    let state = model_facts::linear_state_bytes(lm, plan);
    if state > 0 {
        spec.state_bytes_per_slot = state;
        spec.state_slots_per_request = if EXTRA_BUFFER_ARCHS.contains(&arch) { 5 } else { 3 };
    }
    if let Some(d) = kv_dtype.map(str::to_ascii_lowercase).filter(|d| d.contains("fp8") || d.contains("fp4")) {
        let half = |b: u64| if d.contains("fp4") { b.div_ceil(4) } else { b.div_ceil(2) };
        if spec.kv_bytes_per_token_sm10.is_none() && spec.request_bytes == 0 {
            spec.kv_bytes_per_token = half(spec.kv_bytes_per_token);
            spec.swa_bytes_per_token = half(spec.swa_bytes_per_token);
        }
    }
    Ok(spec)
}

fn fp4_expert_load_peak(cfg: &Value, lm: &Value, model_type: &str) -> u64 {
    // Packed FP4 experts are reordered per layer after loading (original, per-expert copies, stack, allocator slack): 7 layers' worth, as measured on a B200.
    let quant = cfg.get("quantization_config").or_else(|| lm.get("quantization_config"));
    let mxfp4_store = quant.and_then(|q| q.get("store_dtype")).and_then(|v| v.as_str()) == Some("mxfp4");
    if !(model_type.starts_with("deepseek_v4") || mxfp4_store) {
        return 0;
    }
    let get = |k: &str| lm.get(k).and_then(|v| v.as_u64()).unwrap_or(0);
    let per_layer = (get("n_routed_experts") * 3 * get("hidden_size") * get("moe_intermediate_size")) as f64 * (0.5 + 1.0 / 32.0);
    (LOAD_PEAK_LAYERS * per_layer) as u64
}

fn deepseek_v4(lm: &Value, layers: u64) -> Result<MemorySpec, String> {
    // V4: c4/c128 compressed pools, a 0.1 window ratio and per-request c128 state; V4.1: shared low-ratio sources, window cap.
    let get = |k: &str| lm.get(k).and_then(|v| v.as_u64());
    let head = get("head_dim").ok_or("DeepSeek V4 config has no head_dim")?;
    let rope = get("qk_rope_head_dim").unwrap_or(64);
    let index = get("index_head_dim").unwrap_or(128);
    let window = get("sliding_window").unwrap_or(128);
    let kv = (head - rope) + 2 * rope + 8;
    let ratios: Vec<u64> = lm
        .get("compress_ratios")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().take(layers as usize).map(|r| r.as_u64().unwrap_or(0)).collect())
        .unwrap_or_default();
    if let Some(sources) = lm.get("kv_source_layer_ids").and_then(|v| v.as_array()) {
        let low_index = index / 2 + index / 32;
        let low: f64 = sources
            .iter()
            .filter_map(|s| ratios.get(s.as_u64()? as usize).copied())
            .filter(|r| *r == 1 || *r == 2)
            .map(|r| (kv + low_index) as f64 / r as f64)
            .sum();
        let swa = kv * layers;
        let per_request_window = window + SWA_EVICTION_INTERVAL + 2 * DSV4_PAGE + 4 * (window + DSV4_PAGE);
        return Ok(MemorySpec {
            kv_bytes_per_token: low.ceil() as u64,
            request_bytes: per_request_window * swa + 3 * 8 * 4096,
            fixed_bytes: (2 * CHUNKED_PREFILL + DSV4_PAGE) * swa,
            chunked_prefill: CHUNKED_PREFILL,
            ..MemorySpec::default()
        });
    }
    let n4 = ratios.iter().filter(|r| **r == 4).count() as u64;
    let n128 = ratios.iter().filter(|r| **r == 128).count() as u64;
    let indexer = index + index / 128 * 4;
    let swa_token = (kv * layers) as f64 + 8.0 / window as f64 * (2.0 * 2.0 * 512.0 * 4.0 + 2.0 * 2.0 * 128.0 * 4.0) * n4 as f64;
    let full = 0.1 * swa_token + (kv * n4) as f64 / 4.0 + (kv * n128) as f64 / 128.0 + (indexer * n4) as f64 / 4.0;
    let page_padding = 43.75;
    Ok(MemorySpec {
        kv_bytes_per_token: (full + page_padding).ceil() as u64,
        request_bytes: 128 * 2 * 512 * 4 * n128,
        ..MemorySpec::default()
    })
}

fn mla(lm: &Value, cfg: &Value, arch: &str, mla_layers: u64) -> MemorySpec {
    // The latent (kv_lora + rope) per MLA layer, replicated across TP; DSA models add 132-byte FP8 index keys and go FP8 on SM 10+.
    let rank = read_u64(lm, cfg, &["kv_lora_rank"]).unwrap_or(0);
    let rope = read_u64(lm, cfg, &["qk_rope_head_dim"]).unwrap_or(64);
    let dsa = lm.get("index_topk").is_some() && lm.get("index_head_dim").is_some();
    let mut spec = MemorySpec { kv_bytes_per_token: (rank + rope) * BF16 * mla_layers, ..MemorySpec::default() };
    if !dsa {
        return spec;
    }
    let index = read_u64(lm, cfg, &["index_head_dim"]).unwrap_or(128);
    let index_layers = lm
        .get("indexer_types")
        .and_then(|v| v.as_array())
        .map(|t| t.iter().filter(|x| x.as_str() == Some("full")).count() as u64)
        .unwrap_or(mla_layers)
        .min(mla_layers);
    let indexer = index_layers * (index + index / 128 * 4);
    spec.kv_bytes_per_token += indexer;
    if lm.get("learnable_sink").and_then(|v| v.as_bool()) != Some(true) {
        spec.kv_bytes_per_token_sm10 = Some((rank + rope) * mla_layers + indexer);
        if arch == "GlmMoeDsaForCausalLM" {
            spec.kv_bytes_per_token_sm12 = Some((rank + 16 + 2 * rope) * mla_layers + indexer);
        }
    }
    spec
}

fn minimax_sparse(lm: &Value, cfg: &Value, layers: u64) -> MemorySpec {
    // K and V on every layer (split by TP) plus a single-head index pool on sparse layers, K only where values are disabled.
    let sparse = lm.get("sparse_attention_config").cloned().unwrap_or_default();
    let flags = |k: &str| -> Vec<u64> {
        sparse.get(k).and_then(|v| v.as_array()).map(|a| a.iter().map(|x| x.as_u64().unwrap_or(0)).collect()).unwrap_or_default()
    };
    let freq = flags("sparse_attention_freq");
    let no_value = flags("sparse_disable_index_value");
    let sparse_ids: Vec<usize> = freq.iter().enumerate().filter(|(_, f)| **f == 1).map(|(i, _)| i).collect();
    let k_only = sparse_ids.iter().filter(|i| no_value.get(**i) == Some(&1)).count() as u64;
    let index_kv = sparse_ids.len() as u64 - k_only;
    let kv_heads = read_u64(lm, cfg, &["num_key_value_heads"]).unwrap_or(1);
    let head = read_u64(lm, cfg, &["head_dim"]).unwrap_or(128);
    let index_dim = sparse.get("sparse_index_dim").and_then(|v| v.as_u64()).unwrap_or(128);
    MemorySpec {
        kv_bytes_per_token: layers * 2 * kv_heads * head * BF16,
        kv_heads,
        kv_replicated_bytes_per_token: (2 * index_kv + k_only) * index_dim * BF16,
        ..MemorySpec::default()
    }
}

fn mha(lm: &Value, cfg: &Value, plan: model_facts::LayerPlan) -> MemorySpec {
    // K and V per KV head on full layers; window layers in their own pool (Gemma 4 full layers use the global shape).
    let heads = read_u64(lm, cfg, &["num_key_value_heads", "multi_query_group_num"])
        .or_else(|| read_u64(lm, cfg, &["num_attention_heads"]))
        .unwrap_or(1);
    let attention_heads = read_u64(lm, cfg, &["num_attention_heads"]).unwrap_or(1).max(1);
    let hidden = read_u64(lm, cfg, &["hidden_size"]).unwrap_or(0);
    let head = read_u64(lm, cfg, &["head_dim"]).filter(|d| *d > 0).unwrap_or(hidden / attention_heads);
    let v_head = read_u64(lm, cfg, &["v_head_dim"]).unwrap_or(head);
    let full_heads = read_u64(lm, cfg, &["num_global_key_value_heads"]).unwrap_or(heads);
    let full_head = read_u64(lm, cfg, &["global_head_dim"]).unwrap_or(head);
    let full_v = if lm.get("global_head_dim").is_some() { full_head } else { v_head };
    let mut spec = MemorySpec {
        kv_bytes_per_token: full_heads * (full_head + full_v) * BF16 * plan.full,
        kv_heads: full_heads,
        ..MemorySpec::default()
    };
    if plan.swa > 0 {
        let swa_heads = read_u64(lm, cfg, &["swa_num_key_value_heads"]).unwrap_or(heads);
        let swa_head = read_u64(lm, cfg, &["swa_head_dim"]).unwrap_or(head);
        let swa_v = read_u64(lm, cfg, &["swa_v_head_dim"]).unwrap_or(if lm.get("swa_head_dim").is_some() { v_head } else { head });
        spec.swa_bytes_per_token = swa_heads * (swa_head + swa_v) * BF16 * plan.swa;
        spec.swa_heads = swa_heads;
        spec.swa_window = read_u64(lm, cfg, &["sliding_window", "sliding_window_size"]).unwrap_or(4096);
        spec.chunked_prefill = CHUNKED_PREFILL;
    }
    spec
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hybrid_window_models_split_full_and_window_pools() {
        let mut pattern = vec![1u64; 48];
        for i in [0, 5, 11, 17, 23, 29, 35, 41, 47] {
            pattern[i] = 0;
        }
        let mimo = json!({"architectures": ["MiMoV2ForCausalLM"], "num_hidden_layers": 48, "num_attention_heads": 64,
            "num_key_value_heads": 4, "head_dim": 192, "v_head_dim": 128, "hybrid_layer_pattern": pattern,
            "sliding_window": 128, "swa_num_key_value_heads": 8, "swa_head_dim": 192, "swa_v_head_dim": 128});
        let s = from_config(&mimo, None).unwrap();
        assert_eq!((s.kv_bytes_per_token, s.swa_bytes_per_token, s.swa_window), (23_040, 199_680, 128));
        let mut types = vec!["sliding_attention"; 30];
        for i in [5, 11, 17, 23, 29] {
            types[i] = "full_attention";
        }
        let gemma = json!({"architectures": ["Gemma4ForConditionalGeneration"], "text_config": {"num_hidden_layers": 30,
            "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 256, "global_head_dim": 512,
            "num_global_key_value_heads": 2, "sliding_window": 1024, "layer_types": types}});
        let g = from_config(&gemma, None).unwrap();
        assert_eq!((g.kv_bytes_per_token, g.swa_bytes_per_token, g.kv_heads, g.swa_heads), (20_480, 204_800, 2, 8));
    }

    #[test]
    fn mla_and_dsa_follow_the_gpu_generation() {
        let mut idx = vec!["shared"; 78];
        for i in (0..78).step_by(3).take(21) {
            idx[i] = "full";
        }
        let glm52 = json!({"architectures": ["GlmMoeDsaForCausalLM"], "num_hidden_layers": 78, "num_attention_heads": 64,
            "kv_lora_rank": 512, "qk_rope_head_dim": 64, "index_head_dim": 128, "index_topk": 2048, "indexer_types": idx});
        let s = from_config(&glm52, None).unwrap();
        assert_eq!((s.kv_bytes_for("9.0"), s.kv_bytes_for("10.0"), s.kv_bytes_for("12.0")), (92_628, 47_700, 53_940));
        let kimi = json!({"architectures": ["KimiK25ForConditionalGeneration"], "text_config": {"num_hidden_layers": 61,
            "num_attention_heads": 64, "kv_lora_rank": 512, "qk_rope_head_dim": 64}});
        let k = from_config(&kimi, None).unwrap();
        assert_eq!((k.kv_bytes_for("8.6"), k.kv_bytes_for("12.0"), k.kv_heads), (70_272, 70_272, 0));
    }

    #[test]
    fn deepseek_v4_and_v41_pools() {
        let mut ratios = vec![0u64, 0];
        for i in 2..43 {
            ratios.push(if i % 2 == 0 { 4 } else { 128 });
        }
        ratios.extend([0, 0, 0]);
        let v4 = json!({"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4", "num_hidden_layers": 43,
            "num_attention_heads": 64, "head_dim": 512, "qk_rope_head_dim": 64, "index_head_dim": 128,
            "sliding_window": 128, "compress_ratios": ratios});
        let s = from_config(&v4, None).unwrap();
        assert_eq!(s.kv_bytes_per_token, (7705.45f64 + 43.75).ceil() as u64);
        assert_eq!(s.request_bytes, 10 * 1024 * 1024);
        let mut r41 = vec![0u64; 43];
        for i in [2usize, 8, 14] {
            r41[i] = 2;
        }
        r41[20] = 1;
        let v41 = json!({"architectures": ["DeepseekV41ForCausalLM"], "text_config": {"model_type": "deepseek_v41_text",
            "num_hidden_layers": 40, "num_attention_heads": 64, "head_dim": 512, "qk_rope_head_dim": 64,
            "index_head_dim": 128, "sliding_window": 128, "compress_ratios": r41, "kv_source_layer_ids": [2, 8, 14, 20]}});
        let t = from_config(&v41, None).unwrap();
        assert_eq!(t.kv_bytes_per_token, 1630);
        assert_eq!(t.request_bytes, 2304 * 23_360 + 3 * 8 * 4096);
        assert_eq!(t.fixed_bytes, (2 * 8192 + 256) * 23_360);
        let moe = json!({"architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4", "num_hidden_layers": 43,
            "num_attention_heads": 64, "head_dim": 512, "hidden_size": 4096, "moe_intermediate_size": 2048,
            "n_routed_experts": 256, "compress_ratios": [0]});
        let peak = from_config(&moe, None).unwrap().load_peak_bytes as f64 / 1024f64.powi(3);
        assert!((peak - 7.0 * 3.1875).abs() < 1e-6, "{peak}");
    }

    #[test]
    fn state_models_get_their_slot_ratio() {
        let ling = json!({"architectures": ["BailingMoeV3ForCausalLM"], "model_type": "bailing_hybrid", "num_hidden_layers": 72,
            "num_attention_heads": 64, "head_dim": 128, "kv_lora_rank": 512, "qk_rope_head_dim": 64, "layer_group_size": 8,
            "short_conv_kernel_size": 4});
        let s = from_config(&ling, None).unwrap();
        assert_eq!((s.kv_bytes_per_token, s.state_bytes_per_slot, s.state_slots_per_request), (10_368, 273_530_880, 5));
        let mut blocks = vec!["moe"; 88];
        for i in [7usize, 16, 25, 36, 47, 58, 69, 78] {
            blocks[i] = "attention";
        }
        let moe_ids: Vec<usize> = (0..88).filter(|i| blocks[*i] == "moe").take(40).collect();
        for i in moe_ids {
            blocks[i] = "mamba";
        }
        let nemo = json!({"architectures": ["NemotronH_Omni_Reasoning_V3"], "llm_config": {"num_attention_heads": 32,
            "num_key_value_heads": 2, "head_dim": 128, "layers_block_type": blocks, "mamba_num_heads": 128,
            "mamba_head_dim": 64, "ssm_state_size": 128, "n_groups": 8, "conv_kernel": 4}});
        let m = from_config(&nemo, None).unwrap();
        assert_eq!((m.kv_bytes_per_token, m.state_bytes_per_slot, m.state_slots_per_request), (8_192, 170_229_760, 5));
        let flags: Vec<u64> = [0, 0, 0].into_iter().chain([1; 57]).collect();
        let minimax = json!({"architectures": ["MiniMaxM3SparseForConditionalGeneration"], "text_config": {"num_hidden_layers": 60,
            "num_attention_heads": 64, "num_key_value_heads": 4, "head_dim": 128, "sparse_attention_config": {
            "use_sparse_attention": true, "sparse_index_dim": 128, "sparse_attention_freq": flags,
            "sparse_disable_index_value": flags}}});
        let x = from_config(&minimax, None).unwrap();
        assert_eq!(x.kv_bytes_per_token + x.kv_replicated_bytes_per_token, 137_472);
    }

    #[test]
    fn engine_flags_set_state_dtype_and_speculative_state() {
        let types: Vec<&str> = (0..48).map(|i| if i % 4 == 3 { "full_attention" } else { "linear_attention" }).collect();
        let cfg = json!({"architectures": ["Qwen4ExpForConditionalGeneration"], "text_config": {"num_hidden_layers": 48,
            "layer_types": types, "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
            "linear_num_value_heads": 48, "linear_num_key_heads": 16, "linear_key_head_dim": 128,
            "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4, "mamba_ssm_dtype": "float32"}});
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let mut plain = from_config(&cfg, None).unwrap();
        apply_engine_args(&mut plain, Some(&cfg), &[]);
        assert_eq!((plain.state_bytes_per_slot, plain.spec_state_bytes_per_request), (115_458_048, 0));
        let mut bf16 = from_config(&cfg, None).unwrap();
        apply_engine_args(&mut bf16, Some(&cfg), &args(&["--mamba-ssm-dtype", "bfloat16"]));
        assert_eq!(bf16.state_bytes_per_slot, 58_834_944);
        let mut spec = from_config(&cfg, None).unwrap();
        let flags = args(&["--mamba-ssm-dtype=bfloat16", "--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3"]);
        apply_engine_args(&mut spec, Some(&cfg), &flags);
        assert_eq!(spec.spec_state_bytes_per_request, 4 * 58_834_944);
        let mut drafts = from_config(&cfg, None).unwrap();
        apply_engine_args(&mut drafts, None, &args(&["--speculative-algorithm", "EAGLE", "--speculative-num-draft-tokens", "6"]));
        assert_eq!(drafts.spec_state_bytes_per_request, 6 * 115_458_048);
        let mut dense = MemorySpec { kv_bytes_per_token: 1, ..MemorySpec::default() };
        apply_engine_args(&mut dense, None, &args(&["--speculative-algorithm", "NEXTN"]));
        assert_eq!(dense.spec_state_bytes_per_request, 0);
    }
}
