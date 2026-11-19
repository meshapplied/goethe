// Learned VRAM per model, GPU and tp: static extra from startup shortfalls, overhead measured at RUNNING.

use rusqlite::Connection;
use std::path::Path;

pub fn parse_shortfall_gb(log_text: &str) -> Option<f64> {
    // GB an engine start lacked, from SGLang's memory errors; messages without a figure get a 4 GB step.
    for line in log_text.lines() {
        if let Some(rest) = line.split("total_rest_memory=").nth(1) {
            let num: String = rest
                .chars()
                .take_while(|c| *c == '-' || *c == '.' || c.is_ascii_digit())
                .collect();
            if let Ok(v) = num.parse::<f64>() {
                if v < 0.0 {
                    return Some((-v) + 1.0);
                }
            }
        }
        if let Some(rest) = line.split("Raise --mem-fraction-static above ").nth(1) {
            let num: String = rest
                .chars()
                .take_while(|c| *c == '.' || c.is_ascii_digit())
                .collect();
            if let Ok(f) = num.parse::<f64>() {
                return Some(f * 0.05 + 1.0);
            }
        }
        if let Some(gap) = torch_oom_gap_gb(line) {
            return Some(gap.max(1.0) + 1.0);
        }
        if line.contains("CUDA out of memory") || line.contains("state cache is too small to serve any requests") {
            return Some(4.0);
        }
    }
    None
}

fn torch_oom_gap_gb(line: &str) -> Option<f64> {
    // "Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 178.34 GiB of which 1.15 GiB is free": what was missing.
    let number_after = |key: &str| -> Option<f64> {
        let rest = line.split(key).nth(1)?;
        rest.split_whitespace().next()?.trim_end_matches(['.', ',']).parse().ok()
    };
    let wanted = number_after("Tried to allocate ")?;
    let free = number_after("of which ").unwrap_or(0.0);
    Some(wanted - free)
}

pub fn parse_weight_gb(log_text: &str) -> Option<f64> {
    // Per-GPU weights SGLang loaded ("Load weight end ... mem usage=48.87 GB"): target plus drafter, summed per TP rank.
    let mut per_rank: std::collections::BTreeMap<String, f64> = std::collections::BTreeMap::new();
    for line in log_text.lines().filter(|l| l.contains("Load weight end")) {
        let Some(gb) = line.split("mem usage=").nth(1).and_then(|r| r.split_whitespace().next()?.parse::<f64>().ok()) else {
            continue;
        };
        let rank = line.split_whitespace().find(|w| w.starts_with("TP") && w.len() > 2).unwrap_or("").to_string();
        per_rank.entry(rank).and_modify(|v| *v += gb).or_insert(gb);
    }
    per_rank.into_values().reduce(f64::max)
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Learned {
    pub static_gb: f64,
    pub overhead_gb: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum Kind {
    Static,
    Overhead,
    Context,
}

pub const DEFAULT_CONTEXT_GB: f64 = 1.4;
const ANY_MODEL: &str = "*";

fn key(gpu_name: &str, kind: Kind) -> String {
    match kind {
        Kind::Static => format!("{gpu_name} static"),
        Kind::Overhead => gpu_name.to_string(),
        Kind::Context => format!("{gpu_name} context"),
    }
}

pub const PACK_RUNTIME: &str = "goethe-engine";
pub const MAX_LEARNED_OVERHEAD_GB: f64 = 8.0;

fn runtime_key(gpu_name: &str, kind: Kind, runtime: &str) -> String {
    // A figure learned under one serving runtime: the kernel pack, llama.cpp and SGLang hold different memory for the same model.
    format!("{} @{runtime}", key(gpu_name, kind))
}

pub fn model_runtime(home: &Path, model_id: &str) -> String {
    // The runtime a model's card serves with (a card without one is SGLang's).
    let card = std::fs::read_to_string(crate::paths::model_dir(home, model_id).join(crate::install::CARD_FILE)).ok().and_then(|t| t.parse::<toml::Table>().ok()).unwrap_or_default();
    card_runtime(&toml::Value::Table(card))
}

pub fn card_runtime(card: &toml::Value) -> String {
    // Spelling on the card (ternary ≠ launch id sglang; learned rows key on the spelling).
    card.get("runtime").and_then(|v| v.as_str()).unwrap_or("sglang").to_string()
}

pub fn parse_pre_load_gb(log_text: &str) -> Option<f64> {
    // Free memory SGLang sees once its CUDA context exists ("Load weight begin. avail mem=74.86 GB").
    log_text
        .lines()
        .filter(|l| l.contains("Load weight begin"))
        .filter_map(|l| l.split("avail mem=").nth(1))
        .filter_map(|rest| rest.split_whitespace().next()?.parse::<f64>().ok())
        .next_back()
}

pub fn read_context(con: &Connection, gpu_name: &str, tp: u64) -> f64 {
    // What an engine's CUDA context takes on this GPU type and tp before its pools are sized; 1.4 GB until learned.
    let learned = read(con, ANY_MODEL, &key(gpu_name, Kind::Context), tp).unwrap_or(0.0);
    if learned > 0.0 { learned } else { DEFAULT_CONTEXT_GB }
}

pub fn store_context(con: &Connection, gpu_name: &str, tp: u64, gb: f64) -> rusqlite::Result<()> {
    con.execute(
        "INSERT INTO engine_overhead (model_id, gpu_name, tp, extra_gb) VALUES (?1,?2,?3,?4) ON CONFLICT(model_id, gpu_name, tp) DO UPDATE SET extra_gb = MAX(extra_gb, excluded.extra_gb)",
        rusqlite::params![ANY_MODEL, key(gpu_name, Kind::Context), tp as i64, gb],
    )?;
    Ok(())
}

fn read(con: &Connection, model_id: &str, gpu_name: &str, tp: u64) -> Option<f64> {
    con.query_row(
        "SELECT extra_gb FROM engine_overhead WHERE model_id = ?1 AND gpu_name = ?2 AND tp = ?3",
        rusqlite::params![model_id, gpu_name, tp as i64],
        |r| r.get(0),
    )
    .ok()
}

fn read_for(con: &Connection, model_id: &str, gpu_name: &str, tp: u64, kind: Kind, runtime: &str) -> f64 {
    // The row learned under this runtime; rows from before runtimes were recorded count for the plain runtimes only, never for the kernel pack.
    let own = read(con, model_id, &runtime_key(gpu_name, kind, runtime), tp);
    let old = (runtime != PACK_RUNTIME).then(|| read(con, model_id, &key(gpu_name, kind), tp)).flatten();
    own.or(old).unwrap_or(0.0)
}

pub fn read_learned(con: &Connection, model_id: &str, gpu_name: &str, tp: u64, runtime: &str) -> Learned {
    Learned {
        static_gb: read_for(con, model_id, gpu_name, tp, Kind::Static, runtime),
        overhead_gb: read_for(con, model_id, gpu_name, tp, Kind::Overhead, runtime).min(MAX_LEARNED_OVERHEAD_GB),
    }
}

pub fn store_learned(
    con: &Connection,
    model_id: &str,
    gpu_name: &str,
    tp: u64,
    kind: Kind,
    extra_gb: f64,
    runtime: &str,
) -> rusqlite::Result<()> {
    // Static extras keep the larger figure; the overhead is the latest measurement, so a lighter load lowers it. Rows are per runtime.
    let sql = match kind {
        Kind::Overhead => "INSERT INTO engine_overhead (model_id, gpu_name, tp, extra_gb) VALUES (?1,?2,?3,?4) ON CONFLICT(model_id, gpu_name, tp) DO UPDATE SET extra_gb = excluded.extra_gb",
        _ => "INSERT INTO engine_overhead (model_id, gpu_name, tp, extra_gb) VALUES (?1,?2,?3,?4) ON CONFLICT(model_id, gpu_name, tp) DO UPDATE SET extra_gb = MAX(extra_gb, excluded.extra_gb)",
    };
    con.execute(sql, rusqlite::params![model_id, runtime_key(gpu_name, kind, runtime), tp as i64, extra_gb.max(0.0)])?;
    Ok(())
}

pub fn read_log_since(path: &Path, offset: u64) -> String {
    // Log text written after offset (the current spawn), capped to its last 256 KiB.
    let Ok(data) = std::fs::read(path) else {
        return String::new();
    };
    let start = usize::try_from(offset).unwrap_or(usize::MAX).min(data.len());
    let start = start.max(data.len().saturating_sub(256 * 1024));
    String::from_utf8_lossy(&data[start..]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rest_memory_shortfall() {
        let log = "Not enough GPU memory for hybrid ... total_rest_memory=-3.72 GB";
        let gb = parse_shortfall_gb(log).unwrap();
        assert!((gb - 4.72).abs() < 0.01);
    }

    #[test]
    fn log_since_ignores_earlier_spawns() {
        let td = tempfile::tempdir().unwrap();
        let log = td.path().join("e.log");
        std::fs::write(&log, "total_rest_memory=-3.72 GB\n").unwrap();
        let offset = std::fs::metadata(&log).unwrap().len();
        let mut text = std::fs::read_to_string(&log).unwrap();
        text.push_str("Traceback: unrelated failure\n");
        std::fs::write(&log, text).unwrap();
        assert_eq!(parse_shortfall_gb(&read_log_since(&log, offset)), None);
        assert!(parse_shortfall_gb(&read_log_since(&log, 0)).is_some());
    }

    #[test]
    fn static_and_overhead_are_learned_separately_and_only_grow() {
        let td = tempfile::tempdir().unwrap();
        let con = crate::db::open(td.path()).unwrap();
        store_learned(&con, "m", "GPU", 1, Kind::Static, 4.7, "sglang").unwrap();
        store_learned(&con, "m", "GPU", 1, Kind::Static, 3.0, "sglang").unwrap();
        store_learned(&con, "m", "GPU", 1, Kind::Overhead, 2.0, "sglang").unwrap();
        assert_eq!(read_learned(&con, "m", "GPU", 1, "sglang"), Learned { static_gb: 4.7, overhead_gb: 2.0 }, "a static extra only grows");
        store_learned(&con, "m", "GPU", 1, Kind::Overhead, 1.0, "sglang").unwrap();
        assert_eq!(read_learned(&con, "m", "GPU", 1, "sglang").overhead_gb, 1.0, "a lighter load lowers the learned overhead");
        assert_eq!(read_learned(&con, "m", "GPU", 2, "sglang"), Learned::default());
    }

    #[test]
    fn hybrid_state_cache_too_small_triggers_a_retry() {
        let log = "RuntimeError: Hybrid (mamba/linear-attention) state cache is too small to serve any requests. \
                   max_mamba_cache_size=3, mamba_ratio=4, resulting max_num_reqs=0. Try: (1) reduce";
        assert_eq!(parse_shortfall_gb(log), Some(4.0));
    }

    #[test]
    fn the_engine_context_is_learned_per_gpu_type() {
        assert_eq!(parse_pre_load_gb("[t] Load weight begin. avail mem=74.86 GB\n"), Some(74.86));
        let dir = tempfile::tempdir().unwrap();
        let con = crate::db::open(dir.path()).unwrap();
        assert_eq!(read_context(&con, "RTX", 1), DEFAULT_CONTEXT_GB);
        store_context(&con, "RTX", 1, 1.36).unwrap();
        assert_eq!(read_context(&con, "RTX", 1), 1.36);
        assert_eq!(read_context(&con, "RTX", 2), DEFAULT_CONTEXT_GB);
    }

    #[test]
    fn a_torch_oom_names_what_was_missing() {
        let line = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 12.00 GiB. GPU 0 has a total capacity of 178.34 GiB of which 1.15 GiB is free.";
        assert!((parse_shortfall_gb(line).unwrap() - (12.0 - 1.15 + 1.0)).abs() < 1e-9);
        assert_eq!(parse_shortfall_gb("CUDA out of memory"), Some(4.0));
    }

    #[test]
    fn parses_the_loaded_weight_size() {
        let log = "[t] Load weight begin. avail mem=74.86 GB\n[t] Load weight end. elapsed=2.94 s, type=Gemma4, avail mem=25.99 GB, mem usage=48.87 GB.\n";
        assert_eq!(parse_weight_gb(log), Some(48.87));
        assert_eq!(parse_weight_gb("[t] Memory pool end. avail mem=10.21 GB"), None);
        let spec = "[t] Load weight end. type=Qwen4ExpForConditionalGeneration, mem usage=81.35 GB.\n\
                    [t] Load weight end. type=Qwen4ExpForCausalLMMTP, mem usage=0.49 GB.\n";
        assert!((parse_weight_gb(spec).unwrap() - 81.84).abs() < 1e-9, "target plus drafter");
        let tp = "[t TP0] Load weight end. mem usage=40.10 GB.\n[t TP1] Load weight end. mem usage=40.20 GB.\n\
                  [t TP0] Load weight end. mem usage=0.30 GB.\n[t TP1] Load weight end. mem usage=0.30 GB.\n";
        assert!((parse_weight_gb(tp).unwrap() - 40.50).abs() < 1e-9, "per GPU, not summed across ranks");
    }

    #[test]
    fn parses_mem_fraction_hint() {
        let log = "Raise --mem-fraction-static above 0.82 to fit weights";
        assert!(parse_shortfall_gb(log).unwrap() > 0.0);
    }

    #[test]
    fn ternary_card_spelling_stays_ternary_for_learned_rows() {
        let tern: toml::Value = toml::from_str("runtime = \"ternary\"").unwrap();
        let plain: toml::Value = toml::from_str("runtime = \"sglang\"").unwrap();
        assert_eq!(card_runtime(&tern), "ternary");
        assert_eq!(card_runtime(&plain), "sglang");
        assert_eq!(crate::engines::of_card(&tern).unwrap().id, "sglang");
        let td = tempfile::tempdir().unwrap();
        let con = crate::db::open(td.path()).unwrap();
        store_learned(&con, "m", "GPU", 1, Kind::Overhead, 2.5, "ternary").unwrap();
        assert_eq!(read_learned(&con, "m", "GPU", 1, "ternary").overhead_gb, 2.5);
        assert_eq!(read_learned(&con, "m", "GPU", 1, "sglang").overhead_gb, 0.0, "sglang rows stay separate");
    }

    #[test]
    fn a_figure_learned_under_a_plain_runtime_never_blocks_the_kernel_pack() {
        let dir = tempfile::tempdir().unwrap();
        let con = crate::db::open(dir.path()).unwrap();
        let (m, gpu) = ("mradermacher/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF", "NVIDIA GeForce RTX 5090");
        store_learned(&con, m, gpu, 1, Kind::Overhead, 4.791, "llama.cpp").unwrap();
        assert_eq!(read_learned(&con, m, gpu, 1, "llama.cpp").overhead_gb, 4.791);
        assert_eq!(read_learned(&con, m, gpu, 1, PACK_RUNTIME), Learned::default(), "the pack starts from the base overhead");
        assert_eq!(read_learned(&con, m, gpu, 1, "sglang").overhead_gb, 0.0, "another plain runtime has its own row");
        store_learned(&con, m, gpu, 1, Kind::Overhead, 0.5, PACK_RUNTIME).unwrap();
        assert_eq!(read_learned(&con, m, gpu, 1, PACK_RUNTIME).overhead_gb, 0.5);
        assert_eq!(read_learned(&con, m, gpu, 1, "llama.cpp").overhead_gb, 4.791, "and the plain runtime keeps its own");
    }

    #[test]
    fn rows_from_before_runtimes_were_recorded_apply_to_plain_runtimes_only() {
        let dir = tempfile::tempdir().unwrap();
        let con = crate::db::open(dir.path()).unwrap();
        con.execute("INSERT INTO engine_overhead (model_id, gpu_name, tp, extra_gb) VALUES ('m','GPU',1,4.791), ('m','GPU static',1,2.0)", []).unwrap();
        assert_eq!(read_learned(&con, "m", "GPU", 1, "llama.cpp"), Learned { static_gb: 2.0, overhead_gb: 4.791 });
        assert_eq!(read_learned(&con, "m", "GPU", 1, "sglang").overhead_gb, 4.791);
        assert_eq!(read_learned(&con, "m", "GPU", 1, PACK_RUNTIME), Learned::default(), "the old row says nothing about the pack");
        store_learned(&con, "m", "GPU", 1, Kind::Overhead, 1.0, "llama.cpp").unwrap();
        assert_eq!(read_learned(&con, "m", "GPU", 1, "llama.cpp").overhead_gb, 1.0, "a new measurement replaces the old row's figure for that runtime");
    }

    #[test]
    fn the_transcriber_pack_fits_a_32_gb_card_despite_a_row_learned_on_llama_cpp() {
        let dir = tempfile::tempdir().unwrap();
        let con = crate::db::open(dir.path()).unwrap();
        let (m, gpu) = ("mradermacher/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF", "NVIDIA GeForce RTX 5090");
        store_learned(&con, m, gpu, 1, Kind::Overhead, 4.791, "llama.cpp").unwrap();
        let base = crate::calc::Need { weights: 25.1, overhead: 1.8, ..Default::default() };
        let budget = |runtime: &str| crate::chooser::with_learned(base, read_learned(&con, m, gpu, 1, runtime));
        assert_eq!(crate::calc::shortfall(&budget(PACK_RUNTIME), 0, 31.3), None, "the pack: 26.9 GB on a card with 31.3 free");
        assert!(crate::calc::shortfall(&budget("llama.cpp"), 0, 31.3).is_some(), "the plain runtime keeps what it learned");
    }

    #[test]
    fn a_learned_overhead_is_capped() {
        let dir = tempfile::tempdir().unwrap();
        let con = crate::db::open(dir.path()).unwrap();
        store_learned(&con, "m", "GPU", 1, Kind::Overhead, 40.0, "sglang").unwrap();
        assert_eq!(read_learned(&con, "m", "GPU", 1, "sglang").overhead_gb, MAX_LEARNED_OVERHEAD_GB);
    }
}
