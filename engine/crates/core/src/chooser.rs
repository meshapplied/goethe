// Tensor parallel size and GPU set selection for goethe load.

use crate::calc::{missing_gb, shortfall, Need, BASE_OVERHEAD_GB};
use crate::memory_spec::MemorySpec;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
use crate::engine_overhead::Learned;
use crate::gpu_arch;
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
    pub marlin_extra_gb: f64,
    pub supported_sm: Vec<String>,
    pub memory: Option<MemorySpec>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadPlan {
    pub tp: usize,
    pub gpus: Vec<u32>,
    pub cc: String,
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

pub fn kv_tokens(ctx: u64, max_conc: u64) -> u64 {
    // The KV pool holds one request at the full context, or every parallel request at a quarter of it.
    ctx.max((max_conc.max(1) * ctx).div_ceil(4))
}

pub fn state_slots(m: &MemorySpec, max_conc: u64) -> u64 {
    // State slots SGLang needs for max_conc requests with prefix caching (5 per request on extra-buffer models, else 3).
    m.state_slots_per_request * max_conc.max(1)
}

pub fn swa_ratio(m: &MemorySpec, ctx: u64, max_conc: u64) -> f64 {
    // --swa-full-tokens-ratio that gives the window pool exactly the tokens max_conc requests need (at most 1).
    (m.swa_tokens(max_conc) as f64 / kv_tokens(ctx, max_conc).max(1) as f64).clamp(1e-6, 1.0)
}

pub fn engine_args(m: &MemorySpec, ctx: u64, max_conc: u64) -> Vec<String> {
    // The pool sizes the budget assumed, pinned for SGLang.
    let mut out = Vec::new();
    if m.state_bytes_per_slot > 0 {
        out.extend(["--max-mamba-cache-size".to_string(), state_slots(m, max_conc).to_string()]);
    }
    if m.swa_bytes_per_token > 0 {
        out.extend(["--swa-full-tokens-ratio".to_string(), format!("{:.6}", swa_ratio(m, ctx, max_conc))]);
    }
    if m.chunked_prefill > 0 {
        out.extend(["--chunked-prefill-size".to_string(), m.chunked_prefill.to_string()]);
    }
    out
}

pub fn need(input: &ChooserInput, tp: usize, cc: &str) -> Need {
    // Per GPU: sharded weights (plus the Marlin copy where cc keeps one), the KV pools SGLang builds, state, overhead.
    let marlin = if gpu_arch::nvfp4_dual_copy(cc) { input.marlin_extra_gb } else { 0.0 };
    let tokens = kv_tokens(input.ctx, input.max_conc) as f64;
    let conc = input.max_conc.max(1);
    let request_table = ((conc + 1) * (input.ctx + 4) * 4) as f64 / GIB;
    let Some(m) = &input.memory else {
        let shard = kv_shard(tp, input.num_kv_heads, &input.attention);
        return Need {
            weights: (input.static_vram_gb + marlin) / tp as f64,
            kv: tokens * input.kv_bytes_per_token as f64 / GIB / shard,
            overhead: BASE_OVERHEAD_GB + request_table,
            ..Need::default()
        };
    };
    let split = |heads: u64| if heads == 0 { 1.0 } else { (tp as f64).min(heads as f64) };
    let swa_tokens = (m.swa_tokens(conc) as f64).min(tokens);
    let slots = if m.state_bytes_per_slot > 0 { state_slots(m, conc) + 1 } else { 0 };
    Need {
        weights: ((input.static_vram_gb - m.static_state_bytes as f64 / GIB).max(0.0) + marlin) / tp as f64,
        kv: tokens * (m.kv_bytes_for(cc) as f64 / split(m.kv_heads) + m.kv_replicated_bytes_per_token as f64) / GIB,
        window: swa_tokens * m.swa_bytes_per_token as f64 / split(m.swa_heads) / GIB,
        state: (slots as f64 * m.state_bytes_per_slot as f64 / tp as f64
            + ((conc + 1) * m.request_bytes + m.fixed_bytes) as f64)
            / GIB,
        overhead: BASE_OVERHEAD_GB + request_table,
    }
}

pub fn per_gpu_budget_gb(input: &ChooserInput, tp: usize, cc: &str) -> f64 {
    need(input, tp, cc).total()
}

pub fn with_learned(base: Need, learned: Learned) -> Need {
    // Fold what a model taught us on a GPU type and tp into the budget parts: static extra and non-static overhead.
    Need {
        weights: base.weights + learned.static_gb,
        overhead: base.overhead + learned.overhead_gb,
        ..base
    }
}

type LearnedAt<'a> = &'a dyn Fn(usize, u32) -> Learned;

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
    plan_with(input, gpus, hw, pinned_tp, pinned_gpus, &|_, _| Learned::default())
}

pub fn plan_with(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    hw: Option<&HardwareState>,
    pinned_tp: Option<usize>,
    pinned_gpus: Option<&[u32]>,
    learned: LearnedAt,
) -> Result<LoadPlan, PlanError> {
    // Pinned GPUs as given, else a pinned or the smallest fitting tp over GPUs whose architecture runs the model.
    if let Some(pg) = pinned_gpus.filter(|p| !p.is_empty()) {
        let tp = pinned_tp.unwrap_or(pg.len());
        if tp != pg.len() {
            return Err(PlanError {
                message: format!("--tp {tp} disagrees with {} GPUs on --gpu", pg.len()),
                gpus: Vec::new(),
            });
        }
        if !valid_tp_values(input, MAX_TP).contains(&tp) {
            let splits = valid_tp_values(input, gpus.len().max(1));
            return Err(PlanError {
                message: format!(
                    "this model cannot be split over {tp} GPUs: its {} attention heads and {} KV heads allow {}; name that many GPUs with --gpu",
                    input.num_attention_heads,
                    input.num_kv_heads,
                    or_list(&splits)
                ),
                gpus: Vec::new(),
            });
        }
        return fixed_plan(input, gpus, pg, learned).map_err(|mut e| {
            if let Some(hint) = split_hint(input, gpus, hw, tp, learned) {
                e.message = format!("{}\n{hint}", e.message);
            }
            e
        });
    }
    let eligible: Vec<GpuPlacementInput> =
        gpus.iter().filter(|g| gpu_arch::model_runs_on(&input.supported_sm, &g.cc)).cloned().collect();
    if eligible.is_empty() && !gpus.is_empty() {
        return Err(PlanError { message: no_arch_message(input, gpus), gpus: Vec::new() });
    }
    if let Some(tp) = pinned_tp {
        return best_fit(input, &eligible, hw, tp, learned)
            .map(|set| plan(input, tp, set))
            .ok_or_else(|| closest_miss(input, &eligible, hw, tp, learned));
    }
    auto_plan(input, &eligible, hw, learned)
}

const MAX_TP: usize = 64;

fn or_list(v: &[usize]) -> String {
    let s: Vec<String> = v.iter().map(|n| n.to_string()).collect();
    match s.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} or {last}", rest.join(", ")),
        _ => s.join(""),
    }
}

fn split_hint(input: &ChooserInput, gpus: &[GpuPlacementInput], hw: Option<&HardwareState>, tp: usize, learned: LearnedAt) -> Option<String> {
    // When the named GPUs are too few, the smallest larger split that fits, as a --gpu list to copy.
    let eligible: Vec<GpuPlacementInput> =
        gpus.iter().filter(|g| gpu_arch::model_runs_on(&input.supported_sm, &g.cc)).cloned().collect();
    valid_tp_values(input, MAX_TP).into_iter().filter(|t| *t > tp).find_map(|t| {
        let set = best_fit(input, &eligible, hw, t, learned)?;
        let names: Vec<String> = set.gpus.iter().map(|g| format!("gpu{g}")).collect();
        Some(format!("it fits split over {t} GPUs: --gpu {}", names.join(" ")))
    })
}

fn no_arch_message(input: &ChooserInput, gpus: &[GpuPlacementInput]) -> String {
    // Which architectures the model needs against what the visible GPUs are.
    let mut have: Vec<String> = gpus.iter().map(|g| format!("GPU {} is SM {}", g.index, g.cc)).collect();
    have.dedup();
    let wants = match input.supported_sm.is_empty() {
        true => gpu_arch::supported_text(),
        false => input.supported_sm.join(", "),
    };
    format!("no GPU here can run this model: it runs on SM {wants}; {}", have.join(", "))
}

fn need_at(input: &ChooserInput, set: &SetRank, learned: LearnedAt) -> Need {
    with_learned(need(input, set.gpus.len(), &set.cc), learned(set.gpus.len(), set.gpus[0]))
}

fn plan(input: &ChooserInput, tp: usize, set: SetRank) -> LoadPlan {
    LoadPlan {
        tp,
        gpus: set.gpus,
        cc: set.cc,
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
    // The exact GPU set asked for: available, one architecture the model runs on, then the one fit rule.
    let tp = pinned.len();
    let fail = |message: String| Err(PlanError { message, gpus: Vec::new() });
    let mut picked = Vec::new();
    for g in pinned {
        match gpus.iter().find(|x| x.index == *g) {
            Some(x) if !gpu_arch::model_runs_on(&input.supported_sm, &x.cc) => {
                let why = gpu_arch::unsupported_reason(&x.cc).unwrap_or_else(|| no_arch_message(input, gpus));
                return fail(format!("GPU {g} (SM {}) cannot run this model; {why}", x.cc));
            }
            Some(x) => picked.push(x),
            None => return fail(format!("GPU {g} is not available for placement")),
        }
    }
    let cc = picked[0].cc.clone();
    if let Some(other) = picked.iter().find(|x| !gpu_arch::same_cc(&x.cc, &cc)) {
        return fail(format!(
            "GPUs {} mix SM {cc} and SM {}; a tensor-parallel group needs GPUs of one architecture",
            format_gpu_list(pinned),
            other.cc
        ));
    }
    let set = SetRank {
        gpus: pinned.to_vec(),
        cc,
        foreign: 0,
        link: 0,
        min_free_mb: picked.iter().map(|x| x.free_vram_mb).min().unwrap_or(0),
    };
    let need = need_at(input, &set, learned);
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
    rank_sets(gpus, hw, tp)
        .into_iter()
        .find(|s| missing_gb(&need_at(input, s, learned), free_gb(s)) <= 0.0)
}

fn closest_miss(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    hw: Option<&HardwareState>,
    tp: usize,
    learned: LearnedAt,
) -> PlanError {
    // Report the shortfall on the tp-sized set with the most free VRAM.
    let Some(set) = rank_sets(gpus, hw, tp).into_iter().max_by_key(|s| s.min_free_mb) else {
        return PlanError {
            message: format!("cannot place tp={tp}: no {tp} eligible GPUs of one architecture"),
            gpus: Vec::new(),
        };
    };
    let need = need_at(input, &set, learned);
    let why = shortfall(&need, tightest(&set, gpus), free_gb(&set)).unwrap_or_default();
    PlanError {
        message: format!("no GPU set fits tp={tp} (best GPUs {}): {why}", format_gpu_list(&set.gpus)),
        gpus: set.gpus,
    }
}

fn largest_arch_group(gpus: &[GpuPlacementInput]) -> usize {
    // The widest tensor-parallel group possible: the most GPUs sharing one compute capability.
    gpus.iter().map(|g| gpus.iter().filter(|x| gpu_arch::same_cc(&x.cc, &g.cc)).count()).max().unwrap_or(0)
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
    let tps = valid_tp_values(input, largest_arch_group(gpus));
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
            marlin_extra_gb: 0.0,
            supported_sm: Vec::new(),
            memory: None,
        }
    }

    fn gpu(idx: u32, free_gb: f64, foreign: bool) -> GpuPlacementInput {
        GpuPlacementInput { foreign_process: foreign, ..gpu_on(idx, "12.0", free_gb) }
    }

    fn gpu_on(idx: u32, cc: &str, free_gb: f64) -> GpuPlacementInput {
        GpuPlacementInput {
            index: idx,
            cc: cc.to_string(),
            foreign_process: false,
            free_vram_mb: (free_gb * 1024.0) as u64,
        }
    }

    fn mixed_host() -> Vec<GpuPlacementInput> {
        // Two RTX PRO 6000 (12.0), one H100 (9.0) and one RTX 3090 (8.6).
        vec![gpu_on(0, "12.0", 20.0), gpu_on(1, "12.0", 20.0), gpu_on(2, "9.0", 70.0), gpu_on(3, "8.6", 23.0)]
    }

    #[test]
    fn a_heterogeneous_host_places_each_model_on_an_architecture_it_runs_on() {
        let mut m = inp();
        m.supported_sm = vec!["12.0".to_string(), "12.1".to_string()];
        let plan = pick_load_plan(&m, &mixed_host(), None, None, None).unwrap();
        assert_eq!((plan.gpus.clone(), plan.cc.as_str()), (vec![0, 1], "12.0"), "too big for one 20 GB GPU");
        m.supported_sm = vec!["8.6".to_string()];
        let plan = pick_load_plan(&m, &mixed_host(), None, None, None).unwrap();
        assert_eq!((plan.gpus, plan.cc.as_str()), (vec![2], "9.0"), "Hopper is covered by Ampere validation");
        m.supported_sm = vec!["10.0".to_string()];
        let err = pick_load_plan(&m, &mixed_host(), None, None, None).unwrap_err();
        assert!(err.message.contains("no GPU here can run this model") && err.message.contains("GPU 2 is SM 9.0"), "{}", err.message);
    }

    #[test]
    fn pinned_gpus_must_share_an_architecture_the_model_runs_on() {
        let err = pick_load_plan(&inp(), &mixed_host(), None, None, Some(&[1, 2])).unwrap_err();
        assert!(err.message.contains("mix SM 12.0 and SM 9.0"), "{}", err.message);
        let mut m = inp();
        m.supported_sm = vec!["12.0".to_string()];
        let err = pick_load_plan(&m, &mixed_host(), None, None, Some(&[3])).unwrap_err();
        assert!(err.message.starts_with("GPU 3 (SM 8.6) cannot run this model"), "{}", err.message);
        assert_eq!(pick_load_plan(&inp(), &mixed_host(), None, None, Some(&[2])).unwrap().cc, "9.0");
    }

    #[test]
    fn auto_tp_never_spans_architectures() {
        let mut big = inp();
        big.static_vram_gb = 60.0;
        let hosts: Vec<GpuPlacementInput> = mixed_host().into_iter().filter(|g| g.index != 2).collect();
        let err = pick_load_plan(&big, &hosts, None, None, None).unwrap_err();
        assert!(err.message.contains("best GPUs 0,1"), "the 3090 never joins the 12.0 pair: {}", err.message);
    }

    #[test]
    fn the_marlin_copy_is_budgeted_only_where_the_architecture_keeps_one() {
        let mut m = inp();
        m.marlin_extra_gb = 4.0;
        assert_eq!(need(&m, 2, "12.0").weights, 12.0);
        assert_eq!(need(&m, 2, "10.0").weights, 10.0);
        assert_eq!(need(&m, 2, "9.0").weights, 10.0);
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
        let heavy = |_: usize, _: u32| Learned { static_gb: 0.0, overhead_gb: 7.0 };
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
        m.kv_bytes_per_token = 600_000;
        let plan = pick_load_plan(&m, &[gpu(3, 95.0, false)], None, None, None).unwrap();
        assert!(plan.ctx < 131072);
    }

    #[test]
    fn the_kv_pool_holds_one_full_context_or_every_request_at_a_quarter() {
        assert_eq!(kv_tokens(131072, 1), 131072);
        assert_eq!(kv_tokens(131072, 4), 131072);
        assert_eq!(kv_tokens(131072, 8), 262144);
        assert_eq!(kv_tokens(8192, 32), 65536);
        let hybrid = MemorySpec { state_slots_per_request: 5, ..MemorySpec::default() };
        assert_eq!((state_slots(&hybrid, 1), state_slots(&hybrid, 8)), (5, 40));
    }

    #[test]
    fn hybrid_models_budget_their_state_slots_as_sglang_allocates_them() {
        let state: u64 = 48 * (48 * 128 * 128 * 4 + (2 * 16 * 128 + 48 * 128) * 3 * 2);
        let gib = 1024.0_f64.powi(3);
        let spec = MemorySpec {
            kv_bytes_per_token: 65536,
            kv_heads: 4,
            state_bytes_per_slot: state,
            state_slots_per_request: 5,
            static_state_bytes: 4 * state,
            ..MemorySpec::default()
        };
        let m = ChooserInput {
            static_vram_gb: 21.0,
            num_attention_heads: 24,
            num_kv_heads: 4,
            attention: "hybrid".to_string(),
            ctx: 131072,
            max_conc: 8,
            memory: Some(spec.clone()),
            ..inp()
        };
        let n = need(&m, 1, "12.0");
        assert!((n.weights - (21.0 - 4.0 * state as f64 / gib)).abs() < 1e-9, "{n:?}");
        assert!((n.kv - 262144.0 * 65536.0 / gib).abs() < 1e-9, "16 GiB, not 4 full contexts: {n:?}");
        assert!((n.state - 41.0 * state as f64 / gib).abs() < 1e-9, "5 slots per request plus the padding slot: {n:?}");
        assert_eq!(engine_args(&spec, 131072, 8), vec!["--max-mamba-cache-size".to_string(), "40".to_string()]);
        let two = need(&m, 2, "12.0");
        assert!((two.state - n.state / 2.0).abs() < 1e-9 && (two.kv - n.kv / 2.0).abs() < 1e-9);
        assert_eq!(need(&inp(), 1, "9.0").state, 0.0);
    }

    #[test]
    fn window_models_pin_the_window_pool_to_what_requests_need() {
        let spec = MemorySpec {
            kv_bytes_per_token: 20_480,
            kv_heads: 2,
            swa_bytes_per_token: 204_800,
            swa_heads: 8,
            swa_window: 1024,
            chunked_prefill: 8192,
            ..MemorySpec::default()
        };
        let m = ChooserInput { ctx: 262_144, max_conc: 8, memory: Some(spec.clone()), ..inp() };
        let n = need(&m, 1, "12.0");
        let swa_tokens = 8 * (1024 + 128 + 256 + 4 * (1024 + 128)) + 2 * 8192 + 128;
        assert!((n.window - swa_tokens as f64 * 204_800.0 / 1024f64.powi(3)).abs() < 1e-9, "{n:?}");
        let args = engine_args(&spec, 262_144, 8);
        assert_eq!(args[0], "--swa-full-tokens-ratio");
        assert!((args[1].parse::<f64>().unwrap() - swa_tokens as f64 / 524_288.0).abs() < 1e-6, "{args:?}");
        assert_eq!(&args[2..], &["--chunked-prefill-size".to_string(), "8192".to_string()]);
    }

    #[test]
    fn named_gpus_must_split_the_model_and_too_few_get_a_fitting_split() {
        let m = ChooserInput { num_attention_heads: 24, num_kv_heads: 4, ..inp() };
        let gpus = [gpu(0, 40.0, false), gpu(1, 40.0, false), gpu(2, 40.0, false), gpu(3, 40.0, false)];
        let three = pick_load_plan(&m, &gpus, None, None, Some(&[0, 1, 2])).unwrap_err();
        assert!(three.message.contains("cannot be split over 3 GPUs") && three.message.contains("1, 2 or 4"), "{}", three.message);
        let two = pick_load_plan(&m, &gpus, None, None, Some(&[2, 3])).unwrap();
        assert_eq!((two.tp, two.gpus), (2, vec![2, 3]));
        let big = ChooserInput { static_vram_gb: 60.0, ..m };
        let one = pick_load_plan(&big, &gpus, None, None, Some(&[3])).unwrap_err();
        assert!(one.message.contains("missing") && one.message.contains("it fits split over 2 GPUs: --gpu gpu"), "{}", one.message);
    }
}