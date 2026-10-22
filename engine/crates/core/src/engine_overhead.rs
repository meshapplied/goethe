// Learned extra VRAM from engine startup log shortfalls.

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

pub fn read_learned_extra_gb(
    con: &Connection,
    model_id: &str,
    gpu_name: &str,
    tp: u64,
) -> f64 {
    con.query_row(
        "SELECT extra_gb FROM engine_overhead WHERE model_id = ?1 AND gpu_name = ?2 AND tp = ?3",
        rusqlite::params![model_id, gpu_name, tp as i64],
        |r| r.get(0),
    )
    .unwrap_or(0.0)
}

pub fn store_learned_extra_gb(
    con: &Connection,
    model_id: &str,
    gpu_name: &str,
    tp: u64,
    extra_gb: f64,
) -> rusqlite::Result<()> {
    con.execute(
        "INSERT INTO engine_overhead (model_id, gpu_name, tp, extra_gb)
         VALUES (?1,?2,?3,?4)
         ON CONFLICT(model_id, gpu_name, tp) DO UPDATE SET extra_gb = MAX(extra_gb, excluded.extra_gb)",
        rusqlite::params![model_id, gpu_name, tp as i64, extra_gb],
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
    fn parses_mem_fraction_hint() {
        let log = "Raise --mem-fraction-static above 0.82 to fit weights";
        assert!(parse_shortfall_gb(log).unwrap() > 0.0);
    }
}
