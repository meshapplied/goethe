// Automatic tensor parallel and GPU selection for load and recommend.

use crate::calc::static_weight_gb;
use crate::hw::HardwareState;
use crate::placement::{GpuPlacementInput, rank_gpus_for_placement};

#[derive(Debug, Clone)]
pub struct ChooserInput {
    pub static_vram_gb: f64,
    pub marlin_extra_gb: Option<f64>,
    pub apply_marlin: bool,
    pub kv_elems_per_token: u64,
    pub num_attention_heads: u64,
    pub num_kv_heads: u64,
    pub attention: String,
    pub max_ctx: u64,
    pub ctx: u64,
    pub max_conc: u64,
    pub kv_bytes_override: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadPlan {
    pub tp: usize,
    pub gpus: Vec<u32>,
    pub ctx: u64,
    pub max_conc: u64,
    pub message: String,
}

pub fn valid_tp_values(input: &ChooserInput, gpu_count: usize) -> Vec<usize> {
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
    out.sort_unstable();
    out
}

pub fn kv_shard(t: usize, num_kv_heads: u64, attention: &str) -> f64 {
    if attention == "mla" {
        1.0
    } else {
        (t as f64).min(num_kv_heads.max(1) as f64)
    }
}

pub fn bytes_per_elem(gpu_caps: &[f64], override_bytes: Option<u64>) -> u64 {
    if let Some(b) = override_bytes {
        return b.max(1);
    }
    if gpu_caps.iter().all(|c| *c >= 8.9) {
        1
    } else {
        2
    }
}

pub fn per_gpu_budget_gb(input: &ChooserInput, tp: usize, bytes_per_elem: u64) -> f64 {
    let w = static_weight_gb(
        input.static_vram_gb,
        input.marlin_extra_gb,
        input.apply_marlin,
    );
    let kv_bytes = input
        .kv_bytes_override
        .unwrap_or(input.kv_elems_per_token * bytes_per_elem);
    let shard = kv_shard(tp, input.num_kv_heads, &input.attention);
    let kv = (input.ctx as f64) * (kv_bytes as f64) * (input.max_conc.min(4) as f64) * 1e-9 / shard;
    (w / tp as f64) + kv + 1.8
}

pub fn pick_load_plan(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    hw: Option<&HardwareState>,
    pinned_tp: Option<usize>,
    pinned_gpus: Option<&[u32]>,
) -> Result<LoadPlan, String> {
    if let Some(pg) = pinned_gpus {
        if !pg.is_empty() {
            let tp = pinned_tp.unwrap_or(pg.len());
            if let Some(pt) = pinned_tp {
                if pt != pg.len() {
                    return Err(format!(
                        "--tp {pt} disagrees with {} GPUs on --gpus",
                        pg.len()
                    ));
                }
            }
            return finish_plan(input, gpus, hw, tp, pg.to_vec(), input.ctx, input.max_conc);
        }
    }
    if let Some(tp) = pinned_tp {
        let refs: Vec<&GpuPlacementInput> = gpus.iter().collect();
        let picked = pick_gpu_set(&refs, hw, tp)
            .ok_or_else(|| format!("cannot place tp={tp} on available GPUs"))?;
        return finish_plan(input, gpus, hw, tp, picked, input.ctx, input.max_conc);
    }
    auto_plan(input, gpus, hw)
}

fn auto_plan(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    hw: Option<&HardwareState>,
) -> Result<LoadPlan, String> {
    let eligible: Vec<&GpuPlacementInput> = gpus.iter().filter(|g| !g.foreign_process).collect();
    if eligible.is_empty() {
        return Err("no eligible GPUs without foreign processes".to_string());
    }
    let tps = valid_tp_values(input, eligible.len());
    let mut ctx = input.ctx.min(input.max_ctx);
    let mut conc = input.max_conc.max(1);
    loop {
        let mut trial = input.clone();
        trial.ctx = ctx;
        trial.max_conc = conc;
        if let Some(plan) = try_tps(&trial, &eligible, hw, tps.clone()) {
            return Ok(plan);
        }
        if ctx > 8192 {
            ctx /= 2;
            continue;
        }
        if conc > 1 {
            conc -= 1;
            ctx = input.ctx.min(input.max_ctx);
            continue;
        }
        return Err(
            "no placement fits; need more VRAM or reduce ctx/concurrency/model size".to_string(),
        );
    }
}

fn try_tps(
    input: &ChooserInput,
    eligible: &[&GpuPlacementInput],
    hw: Option<&HardwareState>,
    tps: Vec<usize>,
) -> Option<LoadPlan> {
    let caps: Vec<f64> = eligible
        .iter()
        .map(|g| gpu_cap_from_hw(hw, g.index))
        .collect();
    let bpe = bytes_per_elem(&caps, input.kv_bytes_override.map(|b| b / input.kv_elems_per_token.max(1)));
    for tp in tps {
        if tp > eligible.len() {
            continue;
        }
        let picked = pick_gpu_set(eligible, hw, tp)?;
        let need = per_gpu_budget_gb(input, tp, bpe);
        let free_min = picked
            .iter()
            .filter_map(|idx| eligible.iter().find(|g| g.index == *idx))
            .map(|g| g.free_vram_mb as f64 / 1024.0)
            .fold(f64::INFINITY, f64::min);
        if need <= free_min - 1.5 {
            let msg = format!(
                "auto: tp={tp} on GPUs {} ({:.1} GB/gpu need {:.1} GB free on tightest GPU)",
                crate::placement::format_gpu_list(&picked),
                need,
                free_min
            );
            return Some(LoadPlan {
                tp,
                gpus: picked,
                ctx: input.ctx,
                max_conc: input.max_conc,
                message: msg,
            });
        }
    }
    None
}

fn pick_gpu_set(
    eligible: &[&GpuPlacementInput],
    hw: Option<&HardwareState>,
    tp: usize,
) -> Option<Vec<u32>> {
    if tp == 1 {
        return rank_gpus_for_placement(
            &eligible.iter().copied().cloned().collect::<Vec<_>>(),
            1,
        )
        .ok();
    }
    if let Some(hw) = hw {
        if let Some(set) = numa_preferred_set(eligible, hw, tp) {
            return Some(set);
        }
    }
    rank_gpus_for_placement(
        &eligible.iter().copied().cloned().collect::<Vec<_>>(),
        tp,
    )
    .ok()
}

fn numa_preferred_set(
    eligible: &[&GpuPlacementInput],
    hw: &HardwareState,
    tp: usize,
) -> Option<Vec<u32>> {
    if eligible.len() < tp {
        return None;
    }
    let mut picks: Vec<usize> = Vec::new();
    let mut best: Option<(u32, u64, Vec<u32>)> = None;
    combo_rec(eligible.len(), tp, 0, &mut picks, eligible, hw, &mut best);
    best.map(|(_, _, idxs)| idxs)
}

fn combo_rec(
    n: usize,
    tp: usize,
    start: usize,
    picks: &mut Vec<usize>,
    eligible: &[&GpuPlacementInput],
    hw: &HardwareState,
    best: &mut Option<(u32, u64, Vec<u32>)>,
) {
    if picks.len() == tp {
        let mut idxs: Vec<u32> = picks.iter().map(|&i| eligible[i].index).collect();
        idxs.sort_unstable();
        let rank = crate::hw::worst_link_rank(hw, &idxs);
        let min_free = picks
            .iter()
            .map(|&i| eligible[i].free_vram_mb)
            .min()
            .unwrap_or(0);
        let replace = match best {
            None => true,
            Some((br, bf, prev)) => {
                rank < *br
                    || (rank == *br && min_free > *bf)
                    || (rank == *br && min_free == *bf && idxs < *prev)
            }
        };
        if replace {
            best.replace((rank, min_free, idxs));
        }
        return;
    }
    let need = tp - picks.len();
    for i in start..=n.saturating_sub(need) {
        picks.push(i);
        combo_rec(n, tp, i + 1, picks, eligible, hw, best);
        picks.pop();
    }
}

fn gpu_cap_from_hw(hw: Option<&HardwareState>, index: u32) -> f64 {
    hw.and_then(|h| {
        h.gpus
            .iter()
            .find(|g| g.index == index)
            .and_then(|g| g.compute_capability.parse::<f64>().ok())
    })
    .unwrap_or(12.0)
}

fn finish_plan(
    input: &ChooserInput,
    gpus: &[GpuPlacementInput],
    hw: Option<&HardwareState>,
    tp: usize,
    picked: Vec<u32>,
    ctx: u64,
    max_conc: u64,
) -> Result<LoadPlan, String> {
    let caps: Vec<f64> = picked
        .iter()
        .map(|idx| gpu_cap_from_hw(hw, *idx))
        .collect();
    let bpe = bytes_per_elem(
        &caps,
        input.kv_bytes_override.map(|b| b / input.kv_elems_per_token.max(1)),
    );
    let mut trial = input.clone();
    trial.ctx = ctx;
    trial.max_conc = max_conc;
    let need = per_gpu_budget_gb(&trial, tp, bpe);
    let free_min = picked
        .iter()
        .filter_map(|idx| gpus.iter().find(|g| g.index == *idx))
        .map(|g| g.free_vram_mb as f64 / 1024.0)
        .fold(f64::INFINITY, f64::min);
    if need > free_min - 1.5 {
        return Err(format!(
            "VRAM budget {:.1} GB exceeds free {:.1} GB on GPU set (missing {:.1} GB)",
            need,
            free_min,
            need - (free_min - 1.5)
        ));
    }
    let gpu_csv = crate::placement::format_gpu_list(&picked);
    Ok(LoadPlan {
        tp,
        gpus: picked,
        ctx,
        max_conc,
        message: format!("tp={tp} on GPUs {gpu_csv} ({need:.1} GB/gpu)"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inp() -> ChooserInput {
        ChooserInput {
            static_vram_gb: 20.0,
            marlin_extra_gb: None,
            apply_marlin: false,
            kv_elems_per_token: 4096,
            num_attention_heads: 8,
            num_kv_heads: 2,
            attention: "gqa".to_string(),
            max_ctx: 131072,
            ctx: 8192,
            max_conc: 4,
            kv_bytes_override: None,
        }
    }

    fn gpu(idx: u32, free_gb: f64) -> GpuPlacementInput {
        GpuPlacementInput {
            index: idx,
            foreign_process: false,
            free_vram_mb: (free_gb * 1024.0) as u64,
        }
    }

    #[test]
    fn valid_tp_divisors_and_kv_rule() {
        let tps = valid_tp_values(&inp(), 8);
        assert!(tps.contains(&1));
        assert!(tps.contains(&2));
        assert!(tps.contains(&4));
        assert!(!tps.contains(&3));
    }

    #[test]
    fn mla_ignores_kv_heads_rule() {
        let mut m = inp();
        m.attention = "mla".to_string();
        m.num_kv_heads = 1;
        let tps = valid_tp_values(&m, 8);
        assert!(tps.contains(&8));
    }

    #[test]
    fn bytes_per_elem_by_capability() {
        assert_eq!(bytes_per_elem(&[8.9, 12.0], None), 1);
        assert_eq!(bytes_per_elem(&[8.6, 12.0], None), 2);
        assert_eq!(bytes_per_elem(&[12.0], Some(1)), 1);
    }

    #[test]
    fn kv_shard_mla_replicated() {
        assert_eq!(kv_shard(4, 2, "mla"), 1.0);
        assert_eq!(kv_shard(4, 8, "gqa"), 4.0);
    }

    #[test]
    fn tp_gpu_conflict_errors() {
        let gpus = vec![gpu(3, 80.0)];
        let err = pick_load_plan(&inp(), &gpus, None, Some(2), Some(&[3]))
            .unwrap_err();
        assert!(err.contains("disagrees"));
    }

    #[test]
    fn auto_picks_smallest_fitting_tp() {
        let gpus = vec![gpu(3, 100.0), gpu(4, 100.0)];
        let plan = pick_load_plan(&inp(), &gpus, None, None, None).unwrap();
        assert_eq!(plan.tp, 1);
        assert!(plan.message.starts_with("auto:"));
    }

    #[test]
    fn ctx_shrink_when_tight() {
        let mut m = inp();
        m.ctx = 131072;
        m.static_vram_gb = 90.0;
        let gpus = vec![gpu(3, 95.0)];
        let err = pick_load_plan(&m, &gpus, None, None, None);
        assert!(err.is_err() || err.unwrap().ctx <= m.max_ctx);
    }
}
