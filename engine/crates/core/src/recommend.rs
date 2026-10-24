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
                    cc: g.compute_capability.clone(),
                    foreign_process: pids.iter().any(|p| !crate::foreign::is_ours(*p, &managed, &live)),
                    free_vram_mb: free_mb,
                }
            })
            .collect();
        View { hw, gpus, con }
    }

    pub fn capacity(home: &Path, hw: HardwareState) -> View {
        // The allowed GPUs as if empty: their whole memory and no foreign processes, for what this machine could run.
        let mut view = View::new(home, hw);
        for g in &mut view.gpus {
            let state = view.hw.gpus.iter().find(|s| s.index == g.index);
            g.free_vram_mb = state.map_or(g.free_vram_mb, total_mb);
            g.foreign_process = false;
        }
        view
    }

    pub fn free_allowed_gb(&self) -> f64 {
        self.gpus.iter().map(|g| g.free_vram_mb as f64 / 1024.0).sum()
    }

    pub fn free_text(&self) -> String {
        // "VRAM on allowed GPUs: 191.2 GB (gpu2 95.6, gpu3 95.6)" for a capacity view.
        let each: Vec<String> = self.gpus.iter().map(|g| format!("gpu{} {:.1}", g.index, g.free_vram_mb as f64 / 1024.0)).collect();
        format!("VRAM on allowed GPUs: {:.1} GB ({})", self.free_allowed_gb(), each.join(", "))
    }

    pub fn fit(&self, model_id: &str, input: &ChooserInput, pinned_tp: Option<usize>) -> Result<Fit, String> {
        // The plan goethe load would pick (shrinking ctx, then concurrency) and its budget with learned figures.
        let plan = chooser::pick_load_plan(input, &self.gpus, Some(&self.hw), pinned_tp, None).map_err(|e| e.message)?;
        let mut trial = input.clone();
        trial.ctx = plan.ctx;
        trial.max_conc = plan.max_conc;
        let base = chooser::need(&trial, plan.tp, &plan.cc);
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
}

fn total_mb(g: &GpuState) -> u64 {
    if g.unified_memory {
        g.vram_total_mb.saturating_sub(UNIFIED_OS_RESERVE_GB * 1024)
    } else {
        g.vram_total_mb
    }
}

fn file_free_mb(g: &GpuState) -> u64 {
    if g.unified_memory {
        g.vram_free_mb.saturating_sub(UNIFIED_OS_RESERVE_GB * 1024)
    } else {
        g.vram_free_mb
    }
}

const BANDS_GB: &[u32] = &[24, 48, 96, 192, 384, 768, 1536];

pub fn band_label(total_gb: f64) -> String {
    // The smallest size band a model's total VRAM need falls in.
    match BANDS_GB.iter().find(|b| total_gb <= f64::from(**b)) {
        Some(b) => format!("Up to {b} GB"),
        None => format!("Over {} GB", BANDS_GB[BANDS_GB.len() - 1]),
    }
}

pub fn banded(header: &[&str], mut rows: Vec<(f64, Vec<String>)>) -> String {
    // Rows grouped under their size band, smallest first, each band a titled table.
    rows.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = String::new();
    let mut i = 0;
    while i < rows.len() {
        let label = band_label(rows[i].0);
        let mut table = crate::term::Table::new(header);
        while i < rows.len() && band_label(rows[i].0) == label {
            table.row(rows[i].1.clone());
            i += 1;
        }
        out.push_str(&format!("{}\n{}\n", crate::term::paint(&label, crate::term::Color::Bold), table.render()));
    }
    out
}

pub fn run(json_out: bool) -> Result<(), String> {
    // Models the allowed GPUs can run when empty, grouped by the VRAM they take; --json keeps the full plan.
    let home = goethe_home();
    let hw = crate::placement::load_hw(&home).ok_or("hardware_state.json missing; run goethe hw profile first")?;
    let reg = fetch(&registry_url())?;
    let view = View::capacity(&home, hw);
    let out = recommend(&view, &reg);
    if json_out {
        println!("{}", serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?);
        return Ok(());
    }
    println!("{}\n", view.free_text());
    if out.recommended.is_empty() {
        print!("{}", crate::term::commands("No catalog model fits these GPUs. See why with:", &["goethe models"]));
        return Ok(());
    }
    let rows = out
        .recommended
        .iter()
        .map(|item| {
            let total = item.vram_per_gpu_gb * item.tp as f64;
            let download = reg.models.iter().find(|m| m.id == item.model_id).and_then(|m| m.download_gb);
            (total, vec![
                item.model_id.clone(),
                format!("{total:.1} GB"),
                format_gpu_list(&item.gpus),
                download.map_or_else(|| "-".to_string(), |g| format!("{g:.1} GB")),
            ])
        })
        .collect();
    print!("{}", banded(&["MODEL", "VRAM", "GPUS", "DOWNLOAD"], rows));
    Ok(())
}

pub fn recommend(view: &View, reg: &Registry) -> RecommendOutput {
    // Fitting downloadable entries are recommended and the rest say why not; entries goethe cannot download are left out.
    let total_gb: f64 = view.hw.gpus.iter().map(|g| g.vram_total_mb as f64 / 1024.0).sum();
    let sm_set: Vec<String> = view.hw.gpus.iter().map(|g| g.compute_capability.clone()).collect();
    let mut out = RecommendOutput { recommended: Vec::new(), excluded: Vec::new() };
    for model in reg.models.iter().filter(|m| m.manual_download_reason().is_none()) {
        let excluded = |reason: String| ExcludedModel { model_id: model.id.clone(), reason };
        if let Some(reason) = exclusion_reason(model, &sm_set, total_gb) {
            out.excluded.push(excluded(reason));
            continue;
        }
        let input = registry_input(model);
        let pinned = model.default_tp.filter(|t| *t > 0).map(|t| t as usize);
        match view.fit(&model.id, &input, pinned) {
            Ok(f) => out.recommended.push(RecommendedModel {
                model_id: model.id.clone(),
                tp: f.tp as u64,
                gpus: f.gpus,
                ctx: f.ctx,
                max_conc: f.max_conc,
                vram_per_gpu_gb: f.need.total(),
            }),
            Err(why) => out.excluded.push(excluded(format!("does not fit the allowed GPUs: {why}"))),
        }
    }
    out
}

fn exclusion_reason(model: &RegistryModel, sm_set: &[String], total_gb: f64) -> Option<String> {
    if !sm_set.iter().any(|g| crate::gpu_arch::model_runs_on(&model.supported_sm, g)) {
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
        marlin_extra_gb: 0.0,
        supported_sm: m.supported_sm.clone(),
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
        let f = v.fit(&m.id, &input, None).expect("an 8B model fits a 20 GB GPU at a reduced context");
        assert_eq!((f.tp, f.gpus.clone()), (1, vec![2]));
        assert!(f.ctx < 131072, "ctx {}", f.ctx);
        assert!(f.describe(&input).contains("(reduced to fit)"), "{}", f.describe(&input));
        let full = view(vec![gpu(2, 2, true)]).fit(&m.id, &input, None).unwrap_err();
        assert!(full.contains("missing"), "{full}");
    }

    #[test]
    fn entries_goethe_cannot_download_are_left_out() {
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
        assert!(out.excluded.iter().all(|e| !e.model_id.ends_with("manual") && !e.model_id.ends_with("gated")));
        assert!(out.excluded.iter().any(|e| e.model_id == "meshapplied/huge" && e.reason.contains("missing")));
    }

    #[test]
    fn capacity_counts_allowed_gpus_as_empty() {
        let hw = HardwareState { profiled_at: "t".into(), driver: "t".into(), gpus: vec![gpu(2, 2, true)], links: vec![], host: None, environment: None };
        let dir = tempfile::tempdir().unwrap();
        let busy = View::new(dir.path(), hw.clone());
        let whole = View::capacity(dir.path(), hw);
        assert!(busy.free_allowed_gb() < 3.0 && busy.gpus[0].foreign_process);
        assert_eq!(whole.free_allowed_gb(), 96.0);
        assert!(!whole.gpus[0].foreign_process);
        let m = entry("nvidia/Llama-3.1-8B-Instruct-NVFP4", serde_json::json!({}));
        assert!(busy.fit(&m.id, &registry_input(&m), None).is_err() && whole.fit(&m.id, &registry_input(&m), None).is_ok());
        assert_eq!(whole.free_text(), "VRAM on allowed GPUs: 96.0 GB (gpu2 96.0)");
    }

    #[test]
    fn models_are_grouped_by_the_vram_they_take() {
        assert_eq!(band_label(6.0), "Up to 24 GB");
        assert_eq!(band_label(24.0), "Up to 24 GB");
        assert_eq!(band_label(140.9), "Up to 192 GB");
        assert_eq!(band_label(2000.0), "Over 1536 GB");
        let rows = vec![(140.9, vec!["big".to_string()]), (6.0, vec!["small".to_string()]), (20.0, vec!["mid".to_string()])];
        let text = banded(&["MODEL"], rows);
        let order: Vec<&str> = ["Up to 24 GB", "small", "mid", "Up to 192 GB", "big"].to_vec();
        let mut at = 0;
        for word in order {
            at += text[at..].find(word).unwrap_or_else(|| panic!("{word} out of order in {text}"));
        }
    }
}
