// Tensor parallel size and GPU set selection for goethe load.

use crate::calc::missing_gb;
use crate::hw::HardwareState;
use crate::placement::{format_gpu_list, rank_sets, GpuPlacementInput, SetRank};

#[derive(Debug, Clone)]
pub struct ChooserInput {
    pub static_vram_gb: f64,
    pub kv_bytes_per_token: u64,
    pub num_attention_heads: u64,
    pub num_kv_heads: u64,
    pub attention: String,
    pub max_ctx: u64,
    pub ctx: u64,
    pub max_conc: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadPlan {
    pub tp: usize,
    pub gpus: Vec<u32>,
    pub ctx: u64,
    pub max_conc: u64,
    pub message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanError {
    pub message: String,
    pub gpus: Vec<u32>,
}

pub fn valid_tp_values(input: &ChooserInput, gpu_count: usize) -> Vec<usize> {
    // Tensor parallel sizes that divide the attention heads and fit the KV head layout.
    let heads = input.num_attention_heads.max(1) as usize;
    let kv = input.num_kv_heads.max(1) as usize;
    let mla = input.attention == "mla";
    let mut out = Vec::new();
    for t in 1..=heads.min(gpu_count.max(1)) {
        if !heads.is_multiple_of(t) {
            continue;
        }
        if !mla && !kv.is_multiple_of(t) && !t.is_multiple_of(kv) {
            continue;
        }
        out.push(t);
    }
    out
}

pub fn kv_shard(t: usize, num_kv_heads: u64, attention: &str) -> f64 {
    if attention == "mla" {
        1.0
    } else {
        (t as f64).min(num_kv_heads.max(1) as f64)
    }
}

pub fn per_gpu_budget_gb(input: &ChooserInput, tp: usize) -> f64 {
    // Spec formula per GPU: sharded weights plus sharded KV for min(conc, 4) full contexts plus 1.8 GB CUDA context.
    let shard = kv_shard(tp, input.num_kv_heads, &input.attention);
    let tokens = input.ctx as f64 * input.max_conc.min(4) as f64;
    let kv = tokens * input.kv_bytes_per_token as f64 * 1e-9 / shard;
    (input.static_vram_gb / tp as f64) + kv + 1.8
}

fn free_gb(set: &SetRank) -> f64 {
    set.min_free_mb as f64 / 1024.0
}

pub fn pick_load_plan(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    hw: Option<&HardwareState>,
    pinned_tp: Option<usize>,
    pinned_gpus: Option<&[u32]>,
) -> Result<LoadPlan, PlanError> {
    // Pinned GPUs are checked as given; a pinned tp ranks every fitting set; otherwise the smallest fitting tp wins.
    if let Some(pg) = pinned_gpus.filter(|p| !p.is_empty()) {
        let tp = pinned_tp.unwrap_or(pg.len());
        if tp != pg.len() {
            return Err(PlanError {
                message: format!("--tp {tp} disagrees with {} GPUs on --gpus", pg.len()),
                gpus: Vec::new(),
            });
        }
        return fixed_plan(input, gpus, pg);
    }
    if let Some(tp) = pinned_tp {
        return best_fit(input, gpus, hw, tp)
            .map(|(set, need)| plan(input, tp, set, need, ""))
            .ok_or_else(|| closest_miss(input, gpus, hw, tp));
    }
    auto_plan(input, gpus, hw)
}

fn plan(input: &ChooserInput, tp: usize, set: SetRank, need: f64, prefix: &str) -> LoadPlan {
    let message = format!(
        "{prefix}tp={tp} on GPUs {} ({need:.1} GB/gpu, {:.1} GB free on tightest GPU)",
        format_gpu_list(&set.gpus),
        free_gb(&set)
    );
    LoadPlan {
        tp,
        gpus: set.gpus,
        ctx: input.ctx,
        max_conc: input.max_conc,
        message,
    }
}

fn fixed_plan(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    pinned: &[u32],
) -> Result<LoadPlan, PlanError> {
    // Check the exact GPU set the user asked for against the one fit rule.
    let tp = pinned.len();
    let mut free_mb = Vec::new();
    for g in pinned {
        match gpus.iter().find(|x| x.index == *g) {
            Some(x) => free_mb.push(x.free_vram_mb),
            None => {
                return Err(PlanError {
                    message: format!("GPU {g} is not available for placement"),
                    gpus: Vec::new(),
                })
            }
        }
    }
    let set = SetRank {
        gpus: pinned.to_vec(),
        foreign: 0,
        link: 0,
        min_free_mb: free_mb.into_iter().min().unwrap_or(0),
    };
    let need = per_gpu_budget_gb(input, tp);
    let missing = missing_gb(need, free_gb(&set));
    if missing > 0.0 {
        return Err(PlanError {
            message: format!(
                "VRAM budget {need:.1} GB exceeds free {:.1} GB on GPU set {} (missing {missing:.1} GB)",
                free_gb(&set),
                format_gpu_list(&set.gpus)
            ),
            gpus: set.gpus,
        });
    }
    Ok(plan(input, tp, set, need, ""))
}

fn best_fit(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    hw: Option<&HardwareState>,
    tp: usize,
) -> Option<(SetRank, f64)> {
    let need = per_gpu_budget_gb(input, tp);
    rank_sets(gpus, hw, tp)
        .into_iter()
        .find(|s| missing_gb(need, free_gb(s)) <= 0.0)
        .map(|s| (s, need))
}

fn closest_miss(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    hw: Option<&HardwareState>,
    tp: usize,
) -> PlanError {
    // Report the shortfall on the tp-sized set with the most free VRAM.
    let need = per_gpu_budget_gb(input, tp);
    let Some(set) = rank_sets(gpus, hw, tp).into_iter().max_by_key(|s| s.min_free_mb) else {
        return PlanError {
            message: format!("cannot place tp={tp}: only {} eligible GPUs", gpus.len()),
            gpus: Vec::new(),
        };
    };
    PlanError {
        message: format!(
            "no GPU set fits tp={tp}: VRAM budget {need:.1} GB per GPU, best set {} has {:.1} GB free; missing {:.1} GB",
            format_gpu_list(&set.gpus),
            free_gb(&set),
            missing_gb(need, free_gb(&set))
        ),
        gpus: set.gpus,
    }
}

fn auto_plan(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    hw: Option<&HardwareState>,
) -> Result<LoadPlan, PlanError> {
    // Smallest tp with a fitting set (clean GPUs first), shrinking ctx and then concurrency until something fits.
    if gpus.is_empty() {
        return Err(PlanError {
            message: "no eligible GPUs (all denied by policy or none visible)".to_string(),
            gpus: Vec::new(),
        });
    }
    let tps = valid_tp_values(input, gpus.len());
    let mut trial = input.clone();
    trial.ctx = input.ctx.min(input.max_ctx);
    trial.max_conc = input.max_conc.max(1);
    loop {
        for &tp in &tps {
            if let Some((set, need)) = best_fit(&trial, gpus, hw, tp) {
                return Ok(plan(&trial, tp, set, need, "auto: "));
            }
        }
        if trial.ctx > 8192 {
            trial.ctx /= 2;
        } else if trial.max_conc > 1 {
            trial.max_conc -= 1;
            trial.ctx = input.ctx.min(input.max_ctx);
        } else {
            let mut err = closest_miss(input, gpus, hw, tps.last().copied().unwrap_or(1));
            err.message = format!("no placement fits even at reduced ctx/concurrency; {}", err.message);
            return Err(err);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inp() -> ChooserInput {
        ChooserInput {
            static_vram_gb: 20.0,
            kv_bytes_per_token: 4096,
            num_attention_heads: 8,
            num_kv_heads: 2,
            attention: "gqa".to_string(),
            max_ctx: 131072,
            ctx: 8192,
            max_conc: 4,
        }
    }

    fn gpu(idx: u32, free_gb: f64, foreign: bool) -> GpuPlacementInput {
        GpuPlacementInput {
            index: idx,
            foreign_process: foreign,
            free_vram_mb: (free_gb * 1024.0) as u64,
        }
    }

    #[test]
    fn valid_tp_divisors_and_kv_rule() {
        let tps = valid_tp_values(&inp(), 8);
        assert!(tps.contains(&1) && tps.contains(&2) && tps.contains(&4));
        assert!(!tps.contains(&3));
    }

    #[test]
    fn mla_ignores_kv_heads_rule() {
        let mut m = inp();
        m.attention = "mla".to_string();
        m.num_kv_heads = 1;
        assert!(valid_tp_values(&m, 8).contains(&8));
    }

    #[test]
    fn kv_shard_mla_replicated() {
        assert_eq!(kv_shard(4, 2, "mla"), 1.0);
        assert_eq!(kv_shard(4, 8, "gqa"), 4.0);
    }

    #[test]
    fn tp_gpu_conflict_errors() {
        let err = pick_load_plan(&inp(), &[gpu(3, 80.0, false)], None, Some(2), Some(&[3])).unwrap_err();
        assert!(err.message.contains("disagrees"));
    }

    #[test]
    fn auto_picks_smallest_fitting_tp_on_clean_gpu() {
        let gpus = vec![gpu(2, 100.0, true), gpu(3, 90.0, false)];
        let plan = pick_load_plan(&inp(), &gpus, None, None, None).unwrap();
        assert_eq!((plan.tp, plan.gpus.clone()), (1, vec![3]));
        assert!(plan.message.starts_with("auto:"));
    }

    #[test]
    fn auto_uses_foreign_gpus_when_clean_ones_cannot_fit() {
        let mut big = inp();
        big.static_vram_gb = 120.0;
        let gpus = vec![gpu(0, 7.0, true), gpu(1, 7.0, true), gpu(2, 77.0, true), gpu(3, 95.0, false)];
        let plan = pick_load_plan(&big, &gpus, None, None, None).unwrap();
        assert_eq!((plan.tp, plan.gpus), (2, vec![2, 3]));
    }

    #[test]
    fn pinned_tp_skips_production_pair_and_sets_that_do_not_fit() {
        let gpus = vec![gpu(0, 7.0, true), gpu(1, 7.0, true), gpu(2, 77.0, true), gpu(3, 95.0, false)];
        let plan = pick_load_plan(&inp(), &gpus, None, Some(2), None).unwrap();
        assert_eq!(plan.gpus, vec![2, 3]);
        let mut huge = inp();
        huge.static_vram_gb = 400.0;
        let err = pick_load_plan(&huge, &gpus, None, Some(2), None).unwrap_err();
        assert!(err.message.contains("missing"));
        assert_eq!(err.gpus, vec![2, 3]);
    }

    #[test]
    fn pinned_set_reports_missing_gb() {
        let mut huge = inp();
        huge.static_vram_gb = 200.0;
        let err = pick_load_plan(&huge, &[gpu(3, 90.0, false)], None, None, Some(&[3])).unwrap_err();
        assert!(err.message.contains("missing") && err.gpus == vec![3], "{}", err.message);
    }

    #[test]
    fn ctx_shrinks_when_tight() {
        let mut m = inp();
        m.ctx = 131072;
        m.kv_bytes_per_token = 200_000;
        let plan = pick_load_plan(&m, &[gpu(3, 95.0, false)], None, None, None).unwrap();
        assert!(plan.ctx < 131072);
    }
}
