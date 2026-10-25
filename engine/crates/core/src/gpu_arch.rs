// Supported NVIDIA architectures by compute capability: one table that doctor, the catalog, placement and the engine read.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nvfp4Path {
    MarlinAll,
    MarlinSmall,
    Native,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arch {
    pub major: u32,
    pub minor: u32,
    pub family: &'static str,
    pub nvfp4: Nvfp4Path,
    pub target: &'static str,
    pub covered_by: &'static [&'static str],
}

pub const SUPPORTED: &[Arch] = &[
    Arch {
        major: 8,
        minor: 0,
        family: "Ampere data center (A100, A30)",
        nvfp4: Nvfp4Path::MarlinAll,
        target: "8.0",
        covered_by: &["8.6"],
    },
    Arch {
        major: 8,
        minor: 6,
        family: "Ampere (RTX 3090, RTX A6000, A40, A10)",
        nvfp4: Nvfp4Path::MarlinAll,
        target: "8.6",
        covered_by: &[],
    },
    Arch {
        major: 8,
        minor: 9,
        family: "Ada (RTX 4090, RTX 6000 Ada, L40S, L4)",
        nvfp4: Nvfp4Path::MarlinAll,
        target: "8.9",
        covered_by: &["8.6"],
    },
    Arch {
        major: 9,
        minor: 0,
        family: "Hopper (H100, H200, GH200)",
        nvfp4: Nvfp4Path::MarlinAll,
        target: "9.0a",
        covered_by: &["8.6"],
    },
    Arch {
        major: 10,
        minor: 0,
        family: "Blackwell data center (B100, B200, GB200)",
        nvfp4: Nvfp4Path::Native,
        target: "10.0a",
        covered_by: &["12.0", "12.1"],
    },
    Arch {
        major: 10,
        minor: 3,
        family: "Blackwell Ultra data center (B300, GB300)",
        nvfp4: Nvfp4Path::Native,
        target: "10.3a",
        covered_by: &["12.0", "12.1"],
    },
    Arch {
        major: 12,
        minor: 0,
        family: "Blackwell (RTX 5090, RTX PRO 6000)",
        nvfp4: Nvfp4Path::MarlinSmall,
        target: "12.0a",
        covered_by: &[],
    },
    Arch {
        major: 12,
        minor: 1,
        family: "Blackwell (DGX Spark GB10)",
        nvfp4: Nvfp4Path::MarlinSmall,
        target: "12.1a",
        covered_by: &[],
    },
];

pub struct Recognised {
    pub major: u32,
    pub minor: u32,
    pub family: &'static str,
    pub reason: &'static str,
}

pub const RECOGNISED_UNSUPPORTED: &[Recognised] = &[
    Recognised {
        major: 7,
        minor: 0,
        family: "Volta (V100, Titan V)",
        reason: "the CUDA 13 toolchain Goethe ships cannot build for SM 7.0",
    },
    Recognised {
        major: 7,
        minor: 5,
        family: "Turing (T4, RTX 20 series, Quadro RTX)",
        reason: "SGLang's compiled kernels and the 4-bit Marlin path start at Ampere (SM 8.0)",
    },
];

impl Arch {
    pub fn cc(&self) -> String {
        format!("{}.{}", self.major, self.minor)
    }

    pub fn tag(&self) -> String {
        // nvcc target name such as sm_90a; the "a" targets unlock wgmma (9.0) and tcgen05 (10.x, 12.x).
        format!("sm_{}", self.target.replace('.', ""))
    }
}

pub fn parse_cc(cc: &str) -> Option<(u32, u32)> {
    // "12.0" or "12" into (major, minor).
    let mut it = cc.trim().splitn(2, '.');
    let major = it.next()?.trim().parse().ok()?;
    let minor = it.next().map_or(Some(0), |m| m.trim().parse().ok())?;
    Some((major, minor))
}

pub fn lookup(cc: &str) -> Option<Arch> {
    let (major, minor) = parse_cc(cc)?;
    SUPPORTED.iter().copied().find(|a| a.major == major && a.minor == minor)
}

pub fn is_supported(cc: &str) -> bool {
    lookup(cc).is_some()
}

pub fn unsupported_reason(cc: &str) -> Option<String> {
    // Why Goethe will not place models on this GPU, naming the family for GPUs it recognises; None when supported.
    if is_supported(cc) {
        return None;
    }
    let known = parse_cc(cc).and_then(|(m, n)| RECOGNISED_UNSUPPORTED.iter().find(|r| r.major == m && r.minor == n));
    Some(match known {
        Some(r) => format!("{} is not supported: {}", r.family, r.reason),
        None => format!("SM {cc} is not supported (need {})", supported_text()),
    })
}

pub fn same_cc(a: &str, b: &str) -> bool {
    parse_cc(a).is_some_and(|x| parse_cc(b) == Some(x))
}

pub fn supported_list() -> Vec<String> {
    SUPPORTED.iter().map(Arch::cc).collect()
}

pub fn supported_text() -> String {
    // "8.0, 8.6, 8.9, 9.0, 10.0, 10.3, 12.0 or 12.1" for messages.
    let all = supported_list();
    match all.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} or {last}", rest.join(", ")),
        _ => all.join(""),
    }
}

pub fn microarchitectures_text() -> String {
    // "sm_80 (Ampere data center), sm_86 (Ampere), ..." from the table, for doctor.
    let names: Vec<String> = SUPPORTED
        .iter()
        .map(|a| format!("{} ({})", a.tag(), a.family.split(" (").next().unwrap_or(a.family)))
        .collect();
    names.join(", ")
}

pub fn nvfp4_dual_copy(cc: &str) -> bool {
    // Only the small-batch Marlin path keeps a Marlin weight copy next to the native FP4 weights.
    lookup(cc).is_some_and(|a| a.nvfp4 == Nvfp4Path::MarlinSmall)
}

pub fn model_runs_on(model_sm: &[String], gpu_cc: &str) -> bool {
    // Listed for this capability, or for one on the same NVFP4 path with no more shared memory; empty means any.
    let Some(arch) = lookup(gpu_cc) else {
        return false;
    };
    model_sm.is_empty()
        || model_sm.iter().any(|s| same_cc(s, gpu_cc) || arch.covered_by.iter().any(|c| same_cc(s, c)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sm(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_table_answers_every_question_from_one_place() {
        assert_eq!(supported_list(), sm(&["8.0", "8.6", "8.9", "9.0", "10.0", "10.3", "12.0", "12.1"]));
        assert_eq!(supported_text(), "8.0, 8.6, 8.9, 9.0, 10.0, 10.3, 12.0 or 12.1");
        assert!(is_supported("9.0") && is_supported("8.9") && !is_supported("8.7") && !is_supported("7.5"));
        assert_eq!(lookup("10.3").map(|a| a.nvfp4), Some(Nvfp4Path::Native));
        assert_eq!(lookup("9.0").map(|a| a.nvfp4), Some(Nvfp4Path::MarlinAll));
        assert_eq!(parse_cc("12"), Some((12, 0)));
        assert!(nvfp4_dual_copy("12.0") && !nvfp4_dual_copy("10.0") && !nvfp4_dual_copy("9.0") && !nvfp4_dual_copy("8.6"));
    }

    #[test]
    fn doctor_lists_every_microarchitecture_with_its_family() {
        let text = microarchitectures_text();
        assert!(text.starts_with("sm_80 (Ampere data center), sm_86 (Ampere), sm_89 (Ada), sm_90a (Hopper), sm_100a (Blackwell data center)"), "{text}");
        assert!(text.ends_with("sm_121a (Blackwell)"), "{text}");
    }

    #[test]
    fn arch_specific_targets_for_hopper_and_blackwell() {
        let tags: Vec<String> = SUPPORTED.iter().map(Arch::tag).collect();
        assert_eq!(tags, sm(&["sm_80", "sm_86", "sm_89", "sm_90a", "sm_100a", "sm_103a", "sm_120a", "sm_121a"]));
    }

    #[test]
    fn validation_carries_over_to_architectures_on_the_same_path() {
        let consumer = sm(&["12.0", "12.1"]);
        assert!(model_runs_on(&consumer, "10.0") && model_runs_on(&consumer, "10.3"));
        assert!(!model_runs_on(&consumer, "8.6") && !model_runs_on(&consumer, "9.0"));
        let ampere = sm(&["8.6"]);
        assert!(model_runs_on(&ampere, "8.0") && model_runs_on(&ampere, "9.0") && model_runs_on(&ampere, "8.6"));
        assert!(!model_runs_on(&ampere, "12.0"), "consumer Blackwell needs its own validation");
        assert!(!model_runs_on(&sm(&["10.0"]), "12.0"), "coverage is one way");
    }

    #[test]
    fn volta_and_turing_are_named_with_the_reason() {
        assert!(unsupported_reason("7.0").unwrap().starts_with("Volta (V100, Titan V) is not supported: the CUDA 13"));
        assert!(unsupported_reason("7.5").unwrap().contains("start at Ampere (SM 8.0)"));
        assert!(unsupported_reason("6.1").unwrap().starts_with("SM 6.1 is not supported (need 8.0, 8.6"));
        assert_eq!(unsupported_reason("9.0"), None);
        assert!(!model_runs_on(&[], "7.5") && !model_runs_on(&sm(&["7.5"]), "7.5"), "never placed");
    }

    #[test]
    fn an_empty_list_means_every_supported_gpu() {
        assert!(model_runs_on(&[], "9.0") && model_runs_on(&[], "12.1"));
        assert!(!model_runs_on(&[], "7.5") && !model_runs_on(&sm(&["7.5"]), "7.5"), "unsupported GPUs never run");
    }
}
