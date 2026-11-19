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

pub fn next_ordinal(con: &Connection, model_id: &str, gpus: &[u32]) -> rusqlite::Result<u64> {
    // Lowest ordinal above every live instance and desired slot of this model on the same sorted GPU set.
    let prefix = format!("{model_id}|{}|", gpu_csv_sorted(gpus));
    let mut max_ord: i64 = -1;
    for sql in [
        "SELECT slot_key FROM loaded_instances
         WHERE status NOT IN ('FAILED', 'CANCELLED') AND substr(slot_key, 1, length(?1)) = ?1",
        "SELECT slot_key FROM desired_loads WHERE substr(slot_key, 1, length(?1)) = ?1",
    ] {
        let mut stmt = con.prepare(sql)?;
        let rows = stmt.query_map([&prefix], |r| r.get::<_, String>(0))?;
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
    fn next_ordinal_sorts_gpus_and_skips_tombstones() {
        let con = Connection::open_in_memory().unwrap();
        con.execute_batch(
            "CREATE TABLE loaded_instances (slot_key TEXT, status TEXT);
             CREATE TABLE desired_loads (slot_key TEXT);
             INSERT INTO loaded_instances VALUES ('m|2,3|0', 'RUNNING'), ('m|2,3|4', 'FAILED'), ('m_x|2,3|9', 'RUNNING');
             INSERT INTO desired_loads VALUES ('m|2,3|1');",
        )
        .unwrap();
        assert_eq!(next_ordinal(&con, "m", &[3, 2]).unwrap(), 2);
        assert_eq!(next_ordinal(&con, "m", &[3]).unwrap(), 0);
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
