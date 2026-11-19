// What kind of GPU a device is (CUDA, Metal, ROCm) and what of it is worth showing: the one place CLI views and the heartbeat ask.

use crate::hw::GpuState;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Metal,
    Rocm,
    Unknown,
}

impl Backend {
    pub fn of(cc: &str) -> Backend {
        // NVIDIA reports a capability like 12.0, a Mac "metal", AMD a gfx target like gfx942.
        let cc = cc.trim();
        if crate::gpu_arch::is_apple(cc) {
            Backend::Metal
        } else if crate::gpu_arch::parse_cc(cc).is_some() {
            Backend::Cuda
        } else if cc.starts_with("gfx") && cc.len() > 3 && cc[3..].chars().all(|c| c.is_ascii_alphanumeric()) {
            Backend::Rocm
        } else {
            Backend::Unknown
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
            Backend::Rocm => "rocm",
            Backend::Unknown => "unknown",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Backend::Cuda => "CUDA",
            Backend::Metal => "Metal",
            Backend::Rocm => "ROCm",
            Backend::Unknown => "GPU",
        }
    }
}

pub fn arch_label(cc: &str) -> Option<String> {
    // The line under a GPU's name: "SM 12.0" for NVIDIA, the gfx target for AMD, "Metal" for a Mac.
    match Backend::of(cc) {
        Backend::Cuda => Some(format!("SM {}", cc.trim())),
        Backend::Rocm => Some(cc.trim().to_string()),
        Backend::Metal => Some("Metal".to_string()),
        Backend::Unknown => None,
    }
}

pub fn is_unified(g: &GpuState) -> bool {
    g.unified_memory || Backend::of(&g.compute_capability) == Backend::Metal
}

pub fn has_pcie(g: &GpuState) -> bool {
    g.pcie_gen.max(g.pcie_gen_max) > 0 && g.pcie_width.max(g.pcie_width_max) > 0
}

pub fn memory_word(unified: bool) -> &'static str {
    // "VRAM" for cards with their own memory, "GPU memory" when the GPU shares the system's.
    if unified { "GPU memory" } else { "VRAM" }
}

pub fn host_memory_word(gpus: &[GpuState]) -> &'static str {
    memory_word(gpus.iter().any(is_unified))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(cc: &str, unified: bool, pcie: u32) -> GpuState {
        GpuState {
            index: 0,
            uuid: String::new(),
            name: "x".into(),
            compute_capability: cc.into(),
            vram_total_mb: 1,
            vram_used_mb: 0,
            vram_free_mb: 1,
            pcie_gen: pcie,
            pcie_width: pcie * 4,
            pcie_gen_max: 0,
            pcie_width_max: 0,
            running_processes: Vec::new(),
            unified_memory: unified,
            numa_node: None,
        }
    }

    #[test]
    fn backends_come_from_the_capability() {
        assert_eq!(Backend::of("12.0"), Backend::Cuda);
        assert_eq!(Backend::of("8"), Backend::Cuda);
        assert_eq!(Backend::of("metal"), Backend::Metal);
        assert_eq!(Backend::of("gfx942"), Backend::Rocm);
        assert_eq!(Backend::of("gfx"), Backend::Unknown);
        assert_eq!(Backend::of(""), Backend::Unknown);
    }

    #[test]
    fn each_backend_shows_only_what_it_has() {
        assert_eq!(arch_label("12.0").as_deref(), Some("SM 12.0"));
        assert_eq!(arch_label("gfx942").as_deref(), Some("gfx942"));
        assert_eq!(arch_label("metal").as_deref(), Some("Metal"));
        assert_eq!(arch_label("?"), None);
    }

    #[test]
    fn a_mac_and_a_gb10_share_memory_a_card_does_not() {
        assert!(is_unified(&gpu("metal", false, 0)));
        assert!(is_unified(&gpu("12.1", true, 4)));
        assert!(!is_unified(&gpu("12.0", false, 4)));
        assert_eq!(host_memory_word(&[gpu("12.0", false, 4)]), "VRAM");
        assert_eq!(host_memory_word(&[gpu("metal", false, 0)]), "GPU memory");
        assert!(!has_pcie(&gpu("metal", false, 0)) && has_pcie(&gpu("12.0", false, 4)));
    }
}
