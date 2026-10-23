// VRAM budget calculator from the product spec, with the one fit rule every load path uses.

const SAFETY_MARGIN_GB: f64 = 1.5;
const MAX_MEM_FRACTION: f64 = 0.95;
pub const BASE_OVERHEAD_GB: f64 = 1.8;
const LEARN_SLACK_GB: f64 = 0.25;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Need {
    pub weights: f64,
    pub kv: f64,
    pub overhead: f64,
}

impl Need {
    pub fn total(&self) -> f64 {
        self.weights + self.kv + self.overhead
    }
}

pub fn vram_per_gpu_gb(w_gb: f64, l_ctx: u64, k_bytes: u64, c_max: u64, tp: u64) -> f64 {
    let tp = tp.max(1) as f64;
    let kv = l_ctx as f64 * k_bytes as f64 * (c_max.min(4) as f64) * 1e-9;
    (w_gb + kv) / tp + 1.8
}

pub fn static_weight_gb(base_gb: f64, marlin_extra_gb: Option<f64>, apply_marlin: bool) -> f64 {
    base_gb + if apply_marlin {
        marlin_extra_gb.unwrap_or(0.0)
    } else {
        0.0
    }
}

pub fn missing_gb(need: &Need, free_gb: f64) -> f64 {
    // GB a budget lacks: the whole budget leaves the safety margin and the static pool stays within 95% of free.
    (need.total() - (free_gb - SAFETY_MARGIN_GB))
        .max(need.weights + need.kv - MAX_MEM_FRACTION * free_gb)
        .max(0.0)
}

fn tenth(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

pub fn describe(need: &Need) -> String {
    // The budget in parts, rounded so the parts add up to the shown total.
    let (w, k, o) = (tenth(need.weights), tenth(need.kv), tenth(need.overhead));
    format!("{:.1} GB per GPU (weights {w:.1} + KV {k:.1} + overhead {o:.1})", w + k + o)
}

pub fn shortfall(need: &Need, gpu: u32, free_gb: f64) -> Option<String> {
    // Why a per-GPU budget does not fit, naming the rule that binds, in figures that add up; None when it fits.
    if missing_gb(need, free_gb) <= 0.0 {
        return None;
    }
    let (w, k, o, f) = (tenth(need.weights), tenth(need.kv), tenth(need.overhead), tenth(free_gb));
    let n = w + k + o;
    let head = format!("needs {}; GPU {gpu} has {f:.1} GB free", describe(need));
    let by_margin = n - (f - SAFETY_MARGIN_GB);
    let cap = tenth(MAX_MEM_FRACTION * f);
    let by_cap = w + k - cap;
    if by_cap > by_margin {
        return Some(format!(
            "{head}, and its static pool (weights + KV, {:.1} GB) may use at most 95% of it ({cap:.1} GB): \
             missing {:.1} GB",
            w + k,
            by_cap.max(0.1)
        ));
    }
    Some(format!("{head}, minus a {SAFETY_MARGIN_GB} GB safety margin: missing {:.1} GB", by_margin.max(0.1)))
}

pub fn mem_fraction_from_free(need: &Need, min_free_gb: f64) -> Result<f64, String> {
    // mem_fraction_static for weights + KV only; CUDA graphs, activations and context land on top, in the overhead.
    let missing = missing_gb(need, min_free_gb);
    if min_free_gb <= 0.0 || missing > 0.0 {
        return Err(format!(
            "VRAM budget {:.2} GB exceeds free {min_free_gb:.2} GB; missing {missing:.2} GB",
            need.total()
        ));
    }
    Ok((need.weights + need.kv) / min_free_gb)
}

pub fn learned_overhead(used_gb: f64, need: &Need, learned_gb: f64) -> Option<f64> {
    // New learned overhead when real per-GPU use beats the budget by more than the slack (the excess is non-static).
    let over = used_gb - need.total();
    (over > LEARN_SLACK_GB).then_some(learned_gb + over)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn need(weights: f64, kv: f64, overhead: f64) -> Need {
        Need { weights, kv, overhead }
    }

    #[test]
    fn mem_fraction_covers_weights_and_kv_only() {
        let f = mem_fraction_from_free(&need(30.0, 8.2, 1.8 + 2.0), 94.0).unwrap();
        assert!((f - 38.2 / 94.0).abs() < 1e-9);
    }

    #[test]
    fn mem_fraction_rejects_over_095_with_missing() {
        let err = mem_fraction_from_free(&need(45.0, 3.2, 1.8), 50.0).unwrap_err();
        assert!(err.contains("missing"), "{err}");
    }

    #[test]
    fn learned_overhead_only_beyond_slack() {
        let n = need(30.0, 5.7, 1.8);
        assert_eq!(learned_overhead(37.6, &n, 0.0), None);
        let got = learned_overhead(41.8, &n, 0.0).unwrap();
        assert!((got - 4.3).abs() < 1e-9);
        let again = learned_overhead(42.0, &need(30.0, 5.7, 1.8 + 4.3), 4.3).unwrap_or(4.3);
        assert!((again - 4.3).abs() < 1e-9, "within slack keeps the old value");
    }

    #[test]
    fn shortfall_text_adds_up_with_margin_and_cap() {
        let msg = shortfall(&need(12.0, 6.0, 1.8), 2, 20.0).unwrap();
        assert_eq!(
            msg,
            "needs 19.8 GB per GPU (weights 12.0 + KV 6.0 + overhead 1.8); GPU 2 has 20.0 GB free, \
             minus a 1.5 GB safety margin: missing 1.3 GB"
        );
        let capped = shortfall(&need(70.0, 9.5, 1.8), 2, 75.6).unwrap();
        assert!(
            capped.ends_with("its static pool (weights + KV, 79.5 GB) may use at most 95% of it (71.8 GB): missing 7.7 GB"),
            "{capped}"
        );
        assert_eq!(shortfall(&need(10.0, 1.0, 1.8), 2, 75.6), None);
    }

    #[test]
    fn missing_uses_margin_and_static_cap() {
        assert_eq!(missing_gb(&need(8.0, 0.2, 1.8), 20.0), 0.0);
        assert!((missing_gb(&need(15.0, 2.2, 1.8), 20.0) - 0.5).abs() < 1e-9);
        assert!((missing_gb(&need(88.0, 5.0, 1.8), 96.0) - (93.0 - 0.95 * 96.0)).abs() < 1e-9);
    }
}
