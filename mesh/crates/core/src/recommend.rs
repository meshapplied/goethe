// Registry compatibility recommendations against hardware_state.json.

use crate::calc::vram_per_gpu_gb;
use crate::hw::HardwareState;
use crate::paths::mesh_home;
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

pub fn run(json_out: bool) -> Result<(), String> {
    let home = mesh_home();
    let hw_path = home.join("hardware_state.json");
    if !hw_path.is_file() {
        return Err("hardware_state.json missing; run mesh hw profile first".to_string());
    }
    let hw_text = std::fs::read_to_string(&hw_path).map_err(|e| e.to_string())?;
    let hw: HardwareState = serde_json::from_str(&hw_text).map_err(|e| e.to_string())?;
    let url = registry_url();
    let reg = fetch(&url)?;
    let out = recommend(&hw, &reg);
    if json_out {
        println!(
            "{}",
            serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?
        );
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
    let total_gb: f64 = hw
        .gpus
        .iter()
        .map(|g| g.vram_total_mb as f64 / 1024.0)
        .sum();
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
        match place_model(model, hw) {
            Some(rec) => recommended.push(rec),
            None => excluded.push(ExcludedModel {
                model_id: model.id.clone(),
                reason: "insufficient free GPU memory for tensor parallel placement".to_string(),
            }),
        }
    }
    RecommendOutput {
        recommended,
        excluded,
    }
}

fn exclusion_reason(model: &RegistryModel, sm_set: &[String], total_gb: f64) -> Option<String> {
    if !model
        .supported_sm
        .iter()
        .any(|sm| sm_set.iter().any(|g| g == sm))
    {
        return Some(format!(
            "unsupported SM; model needs {:?}, GPUs have {:?}",
            model.supported_sm, sm_set
        ));
    }
    if let Some(min) = model.min_total_vram_gb {
        if total_gb + 1e-6 < min {
            return Some(format!(
                "min total VRAM {:.1} GB exceeds installed {:.1} GB",
                min, total_gb
            ));
        }
    }
    None
}

fn place_model(model: &RegistryModel, hw: &HardwareState) -> Option<RecommendedModel> {
    let n = hw.gpus.len();
    if n == 0 {
        return None;
    }
    let mut tp_candidates: Vec<u64> = Vec::new();
    let mut t = 1u64;
    while t <= n as u64 {
        tp_candidates.push(t);
        t *= 2;
    }
    if let Some(dtp) = model.default_tp {
        if dtp > 0 && !tp_candidates.contains(&dtp) {
            tp_candidates.push(dtp);
            tp_candidates.sort_unstable();
            tp_candidates.dedup();
        }
    }
    let mut by_free: Vec<&crate::hw::GpuState> = hw.gpus.iter().collect();
    by_free.sort_by_key(|g| std::cmp::Reverse(g.vram_free_mb));
    for tp in tp_candidates {
        if tp as usize > n {
            continue;
        }
        let gpus: Vec<u32> = by_free.iter().take(tp as usize).map(|g| g.index).collect();
        let ctx = model.effective_max_ctx().unwrap_or(4096).min(131072);
        let kv = model.effective_kv_elems().unwrap_or(1);
        let conc = model.default_max_conc.unwrap_or(8);
        let need = vram_per_gpu_gb(model.static_vram_gb, ctx, kv, conc, tp);
        let mut ok = true;
        for idx in &gpus {
            let gpu = hw.gpus.iter().find(|g| g.index == *idx)?;
            let mut free_gb = gpu.vram_free_mb as f64 / 1024.0;
            if gpu.unified_memory {
                free_gb = (free_gb - crate::hw::UNIFIED_OS_RESERVE_GB as f64).max(0.0);
            }
            if need > free_gb + 1e-6 {
                ok = false;
                break;
            }
        }
        if ok {
            return Some(RecommendedModel {
                model_id: model.id.clone(),
                tp,
                gpus,
                vram_per_gpu_gb: need,
            });
        }
    }
    None
}
