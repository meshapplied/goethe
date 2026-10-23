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
}

pub const SUPPORTED: &[Arch] = &[
    Arch { major: 8, minor: 6, family: "Ampere (RTX 3090, RTX A6000)", nvfp4: Nvfp4Path::MarlinAll },
    Arch { major: 10, minor: 0, family: "Blackwell data center (B100, B200, GB200)", nvfp4: Nvfp4Path::Native },
    Arch { major: 10, minor: 3, family: "Blackwell Ultra data center (B300, GB300)", nvfp4: Nvfp4Path::Native },
    Arch { major: 12, minor: 0, family: "Blackwell (RTX 5090, RTX PRO 6000)", nvfp4: Nvfp4Path::MarlinSmall },
    Arch { major: 12, minor: 1, family: "Blackwell (DGX Spark GB10)", nvfp4: Nvfp4Path::MarlinSmall },
];

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

pub fn supported_list() -> Vec<String> {
    SUPPORTED.iter().map(|a| format!("{}.{}", a.major, a.minor)).collect()
}

pub fn supported_text() -> String {
    // "8.6, 10.0, 10.3, 12.0 or 12.1" for messages.
    let all = supported_list();
    match all.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} or {last}", rest.join(", ")),
        _ => all.join(""),
    }
}

pub fn nvfp4_dual_copy(cc_major: u32) -> bool {
    // Only the small-batch Marlin path keeps a Marlin weight copy next to the native FP4 weights.
    SUPPORTED.iter().any(|a| a.major == cc_major && a.nvfp4 == Nvfp4Path::MarlinSmall)
}

pub fn model_runs_on(model_sm: &[String], gpu_cc: &str) -> bool {
    // Listed capabilities match exactly; data-center Blackwell (10.x) also runs models validated on consumer Blackwell (12.x).
    let Some((major, minor)) = parse_cc(gpu_cc) else {
        return false;
    };
    let listed = |m: u32, n: u32| model_sm.iter().filter_map(|s| parse_cc(s)).any(|(a, b)| a == m && b == n);
    listed(major, minor) || (major == 10 && is_supported(gpu_cc) && model_sm.iter().filter_map(|s| parse_cc(s)).any(|(a, _)| a == 12))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_answers_every_question_from_one_place() {
        assert_eq!(supported_list(), vec!["8.6", "10.0", "10.3", "12.0", "12.1"]);
        assert_eq!(supported_text(), "8.6, 10.0, 10.3, 12.0 or 12.1");
        assert!(is_supported("10.0") && is_supported("12.1") && !is_supported("9.0") && !is_supported("8.9"));
        assert_eq!(lookup("10.3").map(|a| a.nvfp4), Some(Nvfp4Path::Native));
        assert_eq!(parse_cc("12"), Some((12, 0)));
        assert!(nvfp4_dual_copy(12) && !nvfp4_dual_copy(10) && !nvfp4_dual_copy(8));
    }

    #[test]
    fn data_center_blackwell_runs_models_validated_on_consumer_blackwell() {
        let consumer = vec!["12.0".to_string(), "12.1".to_string()];
        assert!(model_runs_on(&consumer, "12.0"));
        assert!(model_runs_on(&consumer, "10.0") && model_runs_on(&consumer, "10.3"));
        assert!(!model_runs_on(&consumer, "8.6"), "Ampere needs its own validation");
        assert!(!model_runs_on(&consumer, "9.0"), "Hopper is not supported");
        assert!(model_runs_on(&["8.6".to_string()], "8.6"));
    }
}
