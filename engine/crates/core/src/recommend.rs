// Registry compatibility recommendations against hardware_state.json, using the load budget.

use crate::chooser::{self, ChooserInput};
use crate::hw::{HardwareState, UNIFIED_OS_RESERVE_GB};
use crate::paths::goethe_home;
use crate::registry::{fetch, registry_url, Registry, RegistryModel};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct RecommendOutput {
    pub recommended: Vec<RecommendedModel>,
    pub excluded: Vec<ExcludedModel>,
}

#[derive(Debug, Serialize)]
pub struct RecommendedModel {
    pub model_id: String,
    pub tp: u64,
    pub gpus: Vec<u32>,
    pub vram_per_gpu_gb: f64,
}

#[derive(Debug, Serialize)]
pub struct ExcludedModel {
    pub model_id: String,
    pub reason: String,
}

pub struct Fit {
    pub tp: usize,
    pub gpus: Vec<u32>,
    pub need_gb: f64,
}

pub fn run(json_out: bool) -> Result<(), String> {
    let home = goethe_home();
    let hw_path = home.join("hardware_state.json");
    if !hw_path.is_file() {
        return Err("hardware_state.json missing; run goethe hw profile first".to_string());
    }
    let hw_text = std::fs::read_to_string(&hw_path).map_err(|e| e.to_string())?;
    let hw: HardwareState = serde_json::from_str(&hw_text).map_err(|e| e.to_string())?;
    let reg = fetch(&registry_url())?;
    let out = recommend(&hw, &reg);
    if json_out {
        println!("{}", serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?);
    } else {
        for item in &out.recommended {
            println!(
                "recommended {} tp={} gpus={:?} {:.2} GB/gpu",
                item.model_id, item.tp, item.gpus, item.vram_per_gpu_gb
            );
        }
        for item in &out.excluded {
            println!("excluded {}: {}", item.model_id, item.reason);
        }
    }
    Ok(())
}

pub fn recommend(hw: &HardwareState, reg: &Registry) -> RecommendOutput {
    let total_gb: f64 = hw.gpus.iter().map(|g| g.vram_total_mb as f64 / 1024.0).sum();
    let sm_set: Vec<String> = hw.gpus.iter().map(|g| g.compute_capability.clone()).collect();
    let mut recommended = Vec::new();
    let mut excluded = Vec::new();
    for model in &reg.models {
        if let Some(reason) = exclusion_reason(model, &sm_set, total_gb) {
            excluded.push(ExcludedModel {
                model_id: model.id.clone(),
                reason,
            });
            continue;
        }
        let pinned = model.default_tp.filter(|t| *t > 0).map(|t| t as usize);
        match best_fit(&registry_input(model), hw, pinned) {
            Ok(fit) => recommended.push(RecommendedModel {
                model_id: model.id.clone(),
                tp: fit.tp as u64,
                gpus: fit.gpus,
                vram_per_gpu_gb: fit.need_gb,
            }),
            Err(missing) => excluded.push(ExcludedModel {
                model_id: model.id.clone(),
                reason: format!("insufficient free GPU memory: needs {missing:.1} GB more VRAM per GPU"),
            }),
        }
    }
    RecommendOutput {
        recommended,
        excluded,
    }
}

fn exclusion_reason(model: &RegistryModel, sm_set: &[String], total_gb: f64) -> Option<String> {
    if !model.supported_sm.iter().any(|sm| sm_set.iter().any(|g| g == sm)) {
        return Some(format!(
            "unsupported SM; model needs {:?}, GPUs have {:?}",
            model.supported_sm, sm_set
        ));
    }
    if let Some(min) = model.min_total_vram_gb {
        if total_gb + 1e-6 < min {
            return Some(format!("min total VRAM {min:.1} GB exceeds installed {total_gb:.1} GB"));
        }
    }
    None
}

pub fn registry_input(m: &RegistryModel) -> ChooserInput {
    // Budget input for a registry entry, with KV bytes from the entry dtype (auto means 2 bytes).
    let kv_bytes = m.kv_bytes_per_token.unwrap_or_else(|| {
        let elems = m.kv_elems_per_token.unwrap_or(0);
        let width = if m.kv_cache_dtype.as_deref().is_some_and(|d| d.contains("fp8")) { 1 } else { 2 };
        elems * width
    });
    let max_ctx = m.effective_max_ctx().unwrap_or(4096);
    ChooserInput {
        static_vram_gb: m.static_vram_gb,
        marlin_extra_gb: None,
        apply_marlin: false,
        kv_elems_per_token: m.effective_kv_elems().unwrap_or(kv_bytes).max(1),
        num_attention_heads: m.num_attention_heads.unwrap_or(1),
        num_kv_heads: m.num_kv_heads.unwrap_or(1),
        attention: m.attention.clone().unwrap_or_else(|| "gqa".to_string()),
        max_ctx,
        ctx: max_ctx.min(131072),
        max_conc: m.default_max_conc.unwrap_or(8),
        kv_bytes_override: Some(kv_bytes),
    }
}

pub fn best_fit(input: &ChooserInput, hw: &HardwareState, pinned_tp: Option<usize>) -> Result<Fit, f64> {
    // Same budget and 1.5 GB headroom as goethe load, on the GPUs with the most free memory.
    let mut free: Vec<(u32, f64, f64)> = hw
        .gpus
        .iter()
        .map(|g| {
            let mut f = g.vram_free_mb as f64 / 1024.0;
            if g.unified_memory {
                f = (f - UNIFIED_OS_RESERVE_GB as f64).max(0.0);
            }
            (g.index, f, g.compute_capability.parse::<f64>().unwrap_or(12.0))
        })
        .collect();
    free.sort_by(|a, b| b.1.total_cmp(&a.1));
    let tps = match pinned_tp {
        Some(t) => vec![t],
        None => chooser::valid_tp_values(input, free.len()),
    };
    let mut best_missing = f64::INFINITY;
    for tp in tps {
        if tp == 0 || tp > free.len() {
            continue;
        }
        let picked = &free[..tp];
        let caps: Vec<f64> = picked.iter().map(|g| g.2).collect();
        let need = chooser::per_gpu_budget_gb(input, tp, chooser::bytes_per_elem(&caps, None));
        let free_min = picked.iter().map(|g| g.1).fold(f64::INFINITY, f64::min);
        if need <= free_min - 1.5 {
            return Ok(Fit {
                tp,
                gpus: picked.iter().map(|g| g.0).collect(),
                need_gb: need,
            });
        }
        best_missing = best_missing.min(need - (free_min - 1.5));
    }
    Err(if best_missing.is_finite() { best_missing } else { input.static_vram_gb })
}
