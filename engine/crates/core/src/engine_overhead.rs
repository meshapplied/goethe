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

pub fn read_log_tail(path: &Path, max_bytes: usize) -> String {
    let Ok(data) = std::fs::read(path) else {
        return String::new();
    };
    if data.len() <= max_bytes {
        return String::from_utf8_lossy(&data).into_owned();
    }
    String::from_utf8_lossy(&data[data.len() - max_bytes..]).into_owned()
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
    fn parses_mem_fraction_hint() {
        let log = "Raise --mem-fraction-static above 0.82 to fit weights";
        assert!(parse_shortfall_gb(log).unwrap() > 0.0);
    }
}
