// Learned VRAM per model, GPU and tp: static extra from startup shortfalls, overhead measured at RUNNING.

use rusqlite::Connection;
use std::path::Path;

pub fn parse_shortfall_gb(log_text: &str) -> Option<f64> {
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
        if line.contains("CUDA out of memory") {
            return Some(4.0);
        }
    }
    None
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
}

fn key(gpu_name: &str, kind: Kind) -> String {
    match kind {
        Kind::Static => format!("{gpu_name} static"),
        Kind::Overhead => gpu_name.to_string(),
    }
}

fn read(con: &Connection, model_id: &str, gpu_name: &str, tp: u64) -> f64 {
    con.query_row(
        "SELECT extra_gb FROM engine_overhead WHERE model_id = ?1 AND gpu_name = ?2 AND tp = ?3",
        rusqlite::params![model_id, gpu_name, tp as i64],
        |r| r.get(0),
    )
    .unwrap_or(0.0)
}

pub fn read_learned(con: &Connection, model_id: &str, gpu_name: &str, tp: u64) -> Learned {
    Learned {
        static_gb: read(con, model_id, &key(gpu_name, Kind::Static), tp),
        overhead_gb: read(con, model_id, &key(gpu_name, Kind::Overhead), tp),
    }
}

pub fn store_learned(
    con: &Connection,
    model_id: &str,
    gpu_name: &str,
    tp: u64,
    kind: Kind,
    extra_gb: f64,
) -> rusqlite::Result<()> {
    // Keep the larger of the stored and the new figure; static rows carry a " static" suffix on the GPU name.
    con.execute(
        "INSERT INTO engine_overhead (model_id, gpu_name, tp, extra_gb)
         VALUES (?1,?2,?3,?4)
         ON CONFLICT(model_id, gpu_name, tp) DO UPDATE SET extra_gb = MAX(extra_gb, excluded.extra_gb)",
        rusqlite::params![model_id, key(gpu_name, kind), tp as i64, extra_gb],
    )?;
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
        store_learned(&con, "m", "GPU", 1, Kind::Static, 4.7).unwrap();
        store_learned(&con, "m", "GPU", 1, Kind::Overhead, 2.0).unwrap();
        store_learned(&con, "m", "GPU", 1, Kind::Overhead, 1.0).unwrap();
        assert_eq!(read_learned(&con, "m", "GPU", 1), Learned { static_gb: 4.7, overhead_gb: 2.0 });
        assert_eq!(read_learned(&con, "m", "GPU", 2), Learned::default());
    }

    #[test]
    fn parses_mem_fraction_hint() {
        let log = "Raise --mem-fraction-static above 0.82 to fit weights";
        assert!(parse_shortfall_gb(log).unwrap() > 0.0);
    }
}
