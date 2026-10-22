// VRAM budget calculator from the product spec, with the one fit rule every load path uses.

pub const SAFETY_MARGIN_GB: f64 = 1.5;
pub const MAX_MEM_FRACTION: f64 = 0.95;

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

pub fn missing_gb(need_gb: f64, free_gb: f64) -> f64 {
    // GB a budget lacks on a GPU: it must leave the safety margin and stay within MAX_MEM_FRACTION of free.
    (need_gb - (free_gb - SAFETY_MARGIN_GB))
        .max(need_gb - MAX_MEM_FRACTION * free_gb)
        .max(0.0)
}

pub fn mem_fraction_from_free(b_gb: f64, min_free_gb: f64) -> Result<f64, String> {
    // mem_fraction_static for the fork, which sizes its pools from free VRAM measured before load.
    let missing = missing_gb(b_gb, min_free_gb);
    if min_free_gb <= 0.0 || missing > 0.0 {
        return Err(format!(
            "VRAM budget {b_gb:.2} GB exceeds free {min_free_gb:.2} GB; missing {missing:.2} GB"
        ));
    }
    Ok(b_gb / min_free_gb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mem_fraction_uses_free_not_total() {
        let f = mem_fraction_from_free(40.0, 94.0).unwrap();
        assert!((f - 40.0 / 94.0).abs() < 1e-6);
    }

    #[test]
    fn mem_fraction_rejects_over_095_with_missing() {
        let err = mem_fraction_from_free(50.0, 50.0).unwrap_err();
        assert!(err.contains("missing"), "{err}");
    }

    #[test]
    fn missing_uses_margin_and_fraction_cap() {
        assert_eq!(missing_gb(10.0, 20.0), 0.0);
        assert!((missing_gb(19.0, 20.0) - 0.5).abs() < 1e-9);
        assert!((missing_gb(95.0, 96.0) - (95.0 - 0.95 * 96.0)).abs() < 1e-9);
    }
}
