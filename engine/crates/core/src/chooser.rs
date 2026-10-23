// Tensor parallel size and GPU set selection for goethe load.

use crate::calc::{missing_gb, shortfall, Need, BASE_OVERHEAD_GB};
use crate::engine_overhead::Learned;
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

pub fn need(input: &ChooserInput, tp: usize) -> Need {
    // Spec formula per GPU in parts: sharded weights, sharded KV for min(conc, 4) full contexts, 1.8 GB CUDA context.
    let shard = kv_shard(tp, input.num_kv_heads, &input.attention);
    let tokens = input.ctx as f64 * input.max_conc.min(4) as f64;
    Need {
        weights: input.static_vram_gb / tp as f64,
        kv: tokens * input.kv_bytes_per_token as f64 * 1e-9 / shard,
        overhead: BASE_OVERHEAD_GB,
    }
}

pub fn per_gpu_budget_gb(input: &ChooserInput, tp: usize) -> f64 {
    need(input, tp).total()
}

pub fn with_learned(base: Need, learned: Learned) -> Need {
    // Fold what a model taught us on a GPU type and tp into the budget parts: static extra and non-static overhead.
    Need {
        weights: base.weights + learned.static_gb,
        kv: base.kv,
        overhead: base.overhead + learned.overhead_gb,
    }
}

type LearnedAt<'a> = &'a dyn Fn(usize) -> Learned;

fn free_gb(set: &SetRank) -> f64 {
    set.min_free_mb as f64 / 1024.0
}

fn tightest(set: &SetRank, gpus: &[GpuPlacementInput]) -> u32 {
    gpus.iter()
        .filter(|g| set.gpus.contains(&g.index))
        .min_by_key(|g| g.free_vram_mb)
        .map_or(0, |g| g.index)
}

pub fn pick_load_plan(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    hw: Option<&HardwareState>,
    pinned_tp: Option<usize>,
    pinned_gpus: Option<&[u32]>,
) -> Result<LoadPlan, PlanError> {
    plan_with(input, gpus, hw, pinned_tp, pinned_gpus, &|_| Learned::default())
}

pub fn plan_with(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    hw: Option<&HardwareState>,
    pinned_tp: Option<usize>,
    pinned_gpus: Option<&[u32]>,
    learned: LearnedAt,
) -> Result<LoadPlan, PlanError> {
    // Pinned GPUs as given, a pinned tp over every fitting set, else the smallest fitting tp; budgets include learned GB.
    if let Some(pg) = pinned_gpus.filter(|p| !p.is_empty()) {
        let tp = pinned_tp.unwrap_or(pg.len());
        if tp != pg.len() {
            return Err(PlanError {
                message: format!("--tp {tp} disagrees with {} GPUs on --gpus", pg.len()),
                gpus: Vec::new(),
            });
        }
        return fixed_plan(input, gpus, pg, learned);
    }
    if let Some(tp) = pinned_tp {
        return best_fit(input, gpus, hw, tp, learned)
            .map(|set| plan(input, tp, set))
            .ok_or_else(|| closest_miss(input, gpus, hw, tp, learned));
    }
    auto_plan(input, gpus, hw, learned)
}

fn need_at(input: &ChooserInput, tp: usize, learned: LearnedAt) -> Need {
    with_learned(need(input, tp), learned(tp))
}

fn plan(input: &ChooserInput, tp: usize, set: SetRank) -> LoadPlan {
    LoadPlan {
        tp,
        gpus: set.gpus,
        ctx: input.ctx,
        max_conc: input.max_conc,
    }
}

fn fixed_plan(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    pinned: &[u32],
    learned: LearnedAt,
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
    let need = need_at(input, tp, learned);
    if let Some(message) = shortfall(&need, tightest(&set, gpus), free_gb(&set)) {
        return Err(PlanError { message, gpus: set.gpus });
    }
    Ok(plan(input, tp, set))
}

fn best_fit(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    hw: Option<&HardwareState>,
    tp: usize,
    learned: LearnedAt,
) -> Option<SetRank> {
    let need = need_at(input, tp, learned);
    rank_sets(gpus, hw, tp).into_iter().find(|s| missing_gb(&need, free_gb(s)) <= 0.0)
}

fn closest_miss(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    hw: Option<&HardwareState>,
    tp: usize,
    learned: LearnedAt,
) -> PlanError {
    // Report the shortfall on the tp-sized set with the most free VRAM.
    let need = need_at(input, tp, learned);
    let Some(set) = rank_sets(gpus, hw, tp).into_iter().max_by_key(|s| s.min_free_mb) else {
        return PlanError {
            message: format!("cannot place tp={tp}: only {} eligible GPUs", gpus.len()),
            gpus: Vec::new(),
        };
    };
    let why = shortfall(&need, tightest(&set, gpus), free_gb(&set)).unwrap_or_default();
    PlanError {
        message: format!("no GPU set fits tp={tp} (best GPUs {}): {why}", format_gpu_list(&set.gpus)),
        gpus: set.gpus,
    }
}

fn auto_plan(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    hw: Option<&HardwareState>,
    learned: LearnedAt,
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
            if let Some(set) = best_fit(&trial, gpus, hw, tp, learned) {
                return Ok(plan(&trial, tp, set));
            }
        }
        if trial.ctx > 8192 {
            trial.ctx /= 2;
        } else if trial.max_conc > 1 {
            trial.max_conc -= 1;
            trial.ctx = input.ctx.min(input.max_ctx);
        } else {
            let mut err = closest_miss(input, gpus, hw, tps.last().copied().unwrap_or(1), learned);
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
    fn learned_budget_drives_placement_and_failure_text() {
        let gpus = vec![gpu(2, 30.0, false), gpu(3, 26.0, false)];
        let fits = pick_load_plan(&inp(), &gpus, None, Some(1), None).unwrap();
        assert_eq!(fits.gpus, vec![2]);
        let heavy = |_: usize| Learned { static_gb: 0.0, overhead_gb: 7.0 };
        let err = plan_with(&inp(), &gpus, None, Some(1), Some(&[3]), &heavy).unwrap_err();
        assert!(err.message.contains("overhead 8.8"), "{}", err.message);
        assert_eq!(pick_load_plan(&inp(), &gpus, None, None, None).unwrap().tp, 1);
        let auto = plan_with(&inp(), &gpus, None, None, None, &heavy).unwrap();
        assert_eq!((auto.tp, auto.gpus), (2, vec![2, 3]), "the learned tp=1 budget no longer fits one GPU");
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
