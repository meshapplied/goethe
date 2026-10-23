// Registry recommendations and model fit, judged with the plan, budget and learned overhead goethe load uses.

use crate::calc::{self, Need};
use crate::chooser::{self, ChooserInput};
use crate::hw::{GpuState, HardwareState, UNIFIED_OS_RESERVE_GB};
use crate::paths::goethe_home;
use crate::placement::{self, format_gpu_list, GpuPlacementInput};
use crate::registry::{fetch, registry_url, Registry, RegistryModel};
use rusqlite::Connection;
use serde::Serialize;
use std::path::Path;

#[derive(Debug, Serialize)]
pub struct RecommendOutput {
    pub recommended: Vec<RecommendedModel>,
    pub manual: Vec<ManualModel>,
    pub excluded: Vec<ExcludedModel>,
}

#[derive(Debug, Serialize)]
pub struct RecommendedModel {
    pub model_id: String,
    pub tp: u64,
    pub gpus: Vec<u32>,
    pub ctx: u64,
    pub max_conc: u64,
    pub vram_per_gpu_gb: f64,
}

#[derive(Debug, Serialize)]
pub struct ManualModel {
    pub model_id: String,
    pub reason: String,
    pub fit: String,
}

#[derive(Debug, Serialize)]
pub struct ExcludedModel {
    pub model_id: String,
    pub reason: String,
}

#[derive(Debug)]
pub struct Fit {
    pub tp: usize,
    pub gpus: Vec<u32>,
    pub ctx: u64,
    pub max_conc: u64,
    pub need: Need,
}

impl Fit {
    pub fn describe(&self, input: &ChooserInput) -> String {
        // tp, GPUs and context of the plan (noting any reduction), then the budget in parts.
        let conc = if self.max_conc < input.max_conc { format!(", {} concurrent", self.max_conc) } else { String::new() };
        let reduced = if self.ctx < input.ctx || !conc.is_empty() { " (reduced to fit)" } else { "" };
        format!(
            "tp={} on GPU {} at ctx {}{conc}{reduced}: {}",
            self.tp,
            format_gpu_list(&self.gpus),
            self.ctx,
            calc::describe(&self.need)
        )
    }
}

pub struct View {
    pub hw: HardwareState,
    pub gpus: Vec<GpuPlacementInput>,
    con: Option<Connection>,
}

impl View {
    pub fn new(home: &Path, hw: HardwareState) -> View {
        // GPUs as goethe load sees them: live free VRAM and processes where NVML matches hardware_state.json.
        let con = home.join("state.db").is_file().then(|| crate::db::open(home).ok()).flatten();
        let (managed, live, denied) = match &con {
            Some(c) => (
                placement::managed_pids(c),
                placement::live_instance_ids(c),
                crate::gpu_policy::denied_set(c).unwrap_or_default(),
            ),
            None => Default::default(),
        };
        let fresh = crate::hw::live_gpus().unwrap_or_default();
        let gpus = hw
            .gpus
            .iter()
            .filter(|g| !denied.contains(&g.index))
            .map(|g| {
                let (free_mb, pids) = match fresh.iter().find(|l| l.index == g.index && l.uuid == g.uuid) {
                    Some(l) => (l.free_mb, l.pids.clone()),
                    None => (file_free_mb(g), g.running_processes.iter().map(|p| p.pid as i32).collect()),
                };
                GpuPlacementInput {
                    index: g.index,
                    foreign_process: pids.iter().any(|p| !crate::foreign::is_ours(*p, &managed, &live)),
                    free_vram_mb: free_mb,
                }
            })
            .collect();
        View { hw, gpus, con }
    }

    pub fn fit(&self, model_id: &str, input: &ChooserInput, pinned_tp: Option<usize>, marlin_gb: Option<f64>) -> Result<Fit, String> {
        // The plan goethe load would pick (shrinking ctx, then concurrency) and its budget with learned figures.
        let plan = chooser::pick_load_plan(input, &self.gpus, Some(&self.hw), pinned_tp, None).map_err(|e| e.message)?;
        let mut trial = input.clone();
        trial.ctx = plan.ctx;
        trial.max_conc = plan.max_conc;
        trial.static_vram_gb = calc::static_weight_gb(input.static_vram_gb, marlin_gb, self.marlin_applies(&plan.gpus));
        let base = chooser::need(&trial, plan.tp);
        let learned = self.con.as_ref().map(|c| {
            crate::engine_overhead::read_learned(c, model_id, &self.gpu_name(plan.gpus[0]), plan.tp as u64)
        });
        let learned = learned.unwrap_or_default();
        let need = Need {
            weights: base.weights + learned.static_gb,
            kv: base.kv,
            overhead: base.overhead + learned.overhead_gb,
        };
        for g in &plan.gpus {
            let free = self.gpus.iter().find(|x| x.index == *g).map_or(0.0, |x| x.free_vram_mb as f64 / 1024.0);
            if let Some(why) = calc::shortfall(&need, *g, free) {
                return Err(why);
            }
        }
        Ok(Fit { tp: plan.tp, gpus: plan.gpus, ctx: plan.ctx, max_conc: plan.max_conc, need })
    }

    fn gpu_name(&self, idx: u32) -> String {
        self.hw.gpus.iter().find(|g| g.index == idx).map_or_else(|| "gpu".to_string(), |g| g.name.clone())
    }

    fn marlin_applies(&self, gpus: &[u32]) -> bool {
        // The Marlin weight copy exists on compute capability 10 and newer unless GOETHE_MARLIN=0.
        std::env::var("GOETHE_MARLIN").as_deref() != Ok("0")
            && gpus.iter().any(|g| {
                self.hw.gpus.iter().find(|x| x.index == *g).is_some_and(|x| {
                    x.compute_capability.split('.').next().and_then(|m| m.parse::<u32>().ok()).unwrap_or(0) >= 10
                })
            })
    }
}

fn file_free_mb(g: &GpuState) -> u64 {
    if g.unified_memory {
        g.vram_free_mb.saturating_sub(UNIFIED_OS_RESERVE_GB * 1024)
    } else {
        g.vram_free_mb
    }
}

pub fn run(json_out: bool) -> Result<(), String> {
    let home = goethe_home();
    let hw = crate::placement::load_hw(&home).ok_or("hardware_state.json missing; run goethe hw profile first")?;
    let reg = fetch(&registry_url())?;
    let out = recommend(&View::new(&home, hw), &reg);
    if json_out {
        println!("{}", serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?);
        return Ok(());
    }
    for item in &out.recommended {
        println!(
            "recommended {}: tp={} on GPU {} at ctx {}, {:.1} GB per GPU",
            item.model_id,
            item.tp,
            format_gpu_list(&item.gpus),
            item.ctx,
            item.vram_per_gpu_gb
        );
    }
    for item in &out.manual {
        println!("manual download {}: {} ({})", item.model_id, item.reason, item.fit);
    }
    for item in &out.excluded {
        println!("excluded {}: {}", item.model_id, item.reason);
    }
    Ok(())
}

pub fn recommend(view: &View, reg: &Registry) -> RecommendOutput {
    // Fitting downloadable entries are recommended; manual downloads are listed apart; the rest say why not.
    let total_gb: f64 = view.hw.gpus.iter().map(|g| g.vram_total_mb as f64 / 1024.0).sum();
    let sm_set: Vec<String> = view.hw.gpus.iter().map(|g| g.compute_capability.clone()).collect();
    let mut out = RecommendOutput { recommended: Vec::new(), manual: Vec::new(), excluded: Vec::new() };
    for model in &reg.models {
        let excluded = |reason: String| ExcludedModel { model_id: model.id.clone(), reason };
        if let Some(reason) = exclusion_reason(model, &sm_set, total_gb) {
            out.excluded.push(excluded(reason));
            continue;
        }
        let input = registry_input(model);
        let pinned = model.default_tp.filter(|t| *t > 0).map(|t| t as usize);
        let fit = view.fit(&model.id, &input, pinned, None);
        if let Some(reason) = model.manual_download_reason() {
            let fit = match &fit {
                Ok(f) => format!("fits: {}", f.describe(&input)),
                Err(why) => format!("does not fit: {why}"),
            };
            out.manual.push(ManualModel { model_id: model.id.clone(), reason: reason.to_string(), fit });
            continue;
        }
        match fit {
            Ok(f) => out.recommended.push(RecommendedModel {
                model_id: model.id.clone(),
                tp: f.tp as u64,
                gpus: f.gpus,
                ctx: f.ctx,
                max_conc: f.max_conc,
                vram_per_gpu_gb: f.need.total(),
            }),
            Err(why) => out.excluded.push(excluded(format!("does not fit in free GPU memory: {why}"))),
        }
    }
    out
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
    // Budget input for a registry entry as goethe install would write its card (KV bytes from the entry dtype).
    let kv_bytes = m.kv_bytes_per_token.unwrap_or_else(|| {
        let elems = m.kv_elems_per_token.unwrap_or(0);
        let width = if m.kv_cache_dtype.as_deref().is_some_and(|d| d.contains("fp8")) { 1 } else { 2 };
        elems * width
    });
    let max_ctx = m.effective_max_ctx().unwrap_or(4096);
    ChooserInput {
        static_vram_gb: m.static_vram_gb,
        kv_bytes_per_token: kv_bytes.max(1),
        num_attention_heads: m.num_attention_heads.unwrap_or(1),
        num_kv_heads: m.num_kv_heads.unwrap_or(1),
        attention: m.attention.clone().unwrap_or_else(|| "gqa".to_string()),
        max_ctx,
        ctx: m.default_ctx.unwrap_or(max_ctx.min(131072)).min(max_ctx),
        max_conc: m.default_max_conc.unwrap_or(8),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hw::RunningProcess;

    fn gpu(index: u32, free_gb: u64, foreign: bool) -> GpuState {
        GpuState {
            index,
            uuid: format!("test-{index}"),
            name: "TEST".into(),
            compute_capability: "12.0".into(),
            vram_total_mb: 96 * 1024,
            vram_used_mb: (96 - free_gb) * 1024,
            vram_free_mb: free_gb * 1024,
            pcie_gen: 4,
            pcie_width: 16,
            pcie_gen_max: 4,
            pcie_width_max: 16,
            running_processes: if foreign { vec![RunningProcess { pid: 9_999_999, used_vram_mb: 1024 }] } else { vec![] },
            unified_memory: false,
            numa_node: None,
        }
    }

    fn view(gpus: Vec<GpuState>) -> View {
        let hw = HardwareState { profiled_at: "t".into(), driver: "t".into(), gpus, links: vec![], host: None, environment: None };
        let dir = tempfile::tempdir().unwrap();
        View::new(dir.path(), hw)
    }

    fn entry(id: &str, extra: serde_json::Value) -> RegistryModel {
        let mut v = serde_json::json!({"id": id, "static_vram_gb": 5.6, "kv_elems_per_token": 131072,
            "max_ctx": 131072, "supported_sm": ["12.0"], "num_attention_heads": 32, "num_kv_heads": 8});
        v.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn fit_shrinks_context_like_load() {
        let v = view(vec![gpu(2, 20, false)]);
        let m = entry("nvidia/Llama-3.1-8B-Instruct-NVFP4", serde_json::json!({}));
        let input = registry_input(&m);
        assert_eq!(input.ctx, 131072);
        let f = v.fit(&m.id, &input, None, None).expect("an 8B model fits a 20 GB GPU at a reduced context");
        assert_eq!((f.tp, f.gpus.clone()), (1, vec![2]));
        assert!(f.ctx < 131072, "ctx {}", f.ctx);
        assert!(f.describe(&input).contains("(reduced to fit)"), "{}", f.describe(&input));
        let full = view(vec![gpu(2, 2, true)]).fit(&m.id, &input, None, None).unwrap_err();
        assert!(full.contains("missing"), "{full}");
    }

    #[test]
    fn manual_downloads_are_listed_apart() {
        let v = view(vec![gpu(2, 90, false)]);
        let reg = Registry {
            version: 1,
            models: vec![
                entry("meshapplied/manual", serde_json::json!({"manual_download": true})),
                entry("meshapplied/gated", serde_json::json!({"gated": true})),
                entry("meshapplied/open", serde_json::json!({})),
                entry("meshapplied/huge", serde_json::json!({"static_vram_gb": 900.0})),
            ],
        };
        let out = recommend(&v, &reg);
        let ids: Vec<&str> = out.recommended.iter().map(|r| r.model_id.as_str()).collect();
        assert_eq!(ids, vec!["meshapplied/open"]);
        assert_eq!(out.manual.len(), 2);
        assert!(out.manual[0].reason.contains("not on Hugging Face"), "{}", out.manual[0].reason);
        assert!(out.manual[0].fit.starts_with("fits: tp=1 on GPU 2"), "{}", out.manual[0].fit);
        assert!(out.manual[1].reason.contains("gated"));
        assert!(out.excluded.iter().any(|e| e.model_id == "meshapplied/huge" && e.reason.contains("missing")));
    }
}
