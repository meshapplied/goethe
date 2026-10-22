// Per-instance slot keys (model + sorted GPUs + ordinal).

use rusqlite::Connection;

pub fn gpu_csv_sorted(gpus: &[u32]) -> String {
    let mut g = gpus.to_vec();
    g.sort_unstable();
    g.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

pub fn slot_key(model_id: &str, gpus: &[u32], ordinal: u64) -> String {
    format!("{}|{}|{}", model_id, gpu_csv_sorted(gpus), ordinal)
}

pub fn parse_slot_key(key: &str) -> Option<(String, String, u64)> {
    let (prefix, ord_s) = key.rsplit_once('|')?;
    let (model, gpus) = prefix.rsplit_once('|')?;
    let ord = ord_s.parse().ok()?;
    Some((model.to_string(), gpus.to_string(), ord))
}

pub fn next_ordinal(con: &Connection, model_id: &str, gpu_csv: &str) -> rusqlite::Result<u64> {
    let prefix = format!("{model_id}|{gpu_csv}|");
    let mut max_ord: i64 = -1;
    for table in ["loaded_instances", "desired_loads"] {
        let sql = format!("SELECT slot_key FROM {table} WHERE slot_key LIKE ?1");
        let mut stmt = con.prepare(&sql)?;
        let rows = stmt.query_map([format!("{prefix}%")], |r| r.get::<_, String>(0))?;
        for row in rows.flatten() {
            if let Some((_, _, o)) = parse_slot_key(&row) {
                max_ord = max_ord.max(o as i64);
            }
        }
    }
    Ok((max_ord + 1) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_key_sorts_gpus() {
        assert_eq!(slot_key("m", &[3, 2], 0), "m|2,3|0");
        assert_eq!(slot_key("m", &[2, 3], 1), "m|2,3|1");
    }

    #[test]
    fn roundtrip_parse() {
        let k = slot_key("org/x", &[1, 0], 2);
        let (m, g, o) = parse_slot_key(&k).unwrap();
        assert_eq!(m, "org/x");
        assert_eq!(g, "0,1");
        assert_eq!(o, 2);
    }
}
