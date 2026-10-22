// VRAM budget calculator from the product spec.

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

pub fn mem_fraction_from_free(b_gb: f64, min_free_gb: f64) -> Result<f64, String> {
    if min_free_gb <= 0.0 {
        return Err("no free GPU memory available for load".to_string());
    }
    let f = b_gb / min_free_gb;
    if f > 0.95 {
        return Err(format!(
            "VRAM budget {:.2} GB exceeds free {:.2} GB (mem_fraction_static {:.3} > 0.95); reduce ctx, max-conc, or unload other models",
            b_gb, min_free_gb, f
        ));
    }
    Ok(f)
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
    fn mem_fraction_rejects_over_095() {
        assert!(mem_fraction_from_free(50.0, 50.0).is_err());
    }
}
