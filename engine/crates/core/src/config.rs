// In-place goethe_card.toml and installed_models default updates.

use crate::db;
use crate::install::{kv_element_bytes, write_card_table, CARD_FILE};
use crate::paths::{goethe_home, validate_model_id};
use crate::sglang_infer as sgl;
use rusqlite::OptionalExtension;
use std::path::{Path, PathBuf};

pub struct ConfigOpts {
    pub model_id: String,
    pub ctx: Option<u64>,
    pub max_conc: Option<u64>,
    pub tp: Option<u64>,
    pub gpus: Option<Vec<u32>>,
    pub sglang_args: Vec<String>,
    pub clear_sglang_args: bool,
}

pub fn validate_limits(home: &Path, model_id: &str, ctx: Option<u64>, max_conc: Option<u64>) -> Result<(), String> {
    // --ctx must lie between 1 and the installed model's max_ctx, and --max-conc must be at least 1.
    if ctx == Some(0) {
        return Err("--ctx must be at least 1".into());
    }
    if max_conc == Some(0) {
        return Err("--max-conc must be at least 1".into());
    }
    let Some(ctx) = ctx.filter(|_| home.join("state.db").is_file()) else {
        return Ok(());
    };
    let con = db::open(home).map_err(|e| format!("state.db: {e}"))?;
    let max: Option<i64> = con
        .query_row("SELECT max_ctx FROM installed_models WHERE model_id = ?1", [model_id], |r| r.get(0))
        .optional()
        .map_err(|e| format!("state.db: {e}"))?
        .flatten();
    match max.filter(|m| *m > 0) {
        Some(m) if ctx > m as u64 => Err(format!("--ctx {ctx} is above the maximum context of {model_id} ({m} tokens)")),
        _ => Ok(()),
    }
}

pub fn run(opts: ConfigOpts) -> Result<(), String> {
    // Update card fields atomically, then mirror the load defaults into installed_models.
    validate_model_id(&opts.model_id)?;
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| format!("state.db: {e}"))?;
    let row: Option<(String, Option<i64>)> = con
        .query_row(
            "SELECT local_path, kv_elems_per_token FROM installed_models WHERE model_id = ?1",
            [&opts.model_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
        .map_err(|e| format!("state.db: {e}"))?;
    let dir = match &row {
        Some((rel, _)) => home.join(rel),
        None => find_model_dir(&home, &opts.model_id)?,
    };
    let text = std::fs::read_to_string(dir.join(CARD_FILE)).map_err(|e| format!("{}: {e}", dir.join(CARD_FILE).display()))?;
    let mut table: toml::Table = toml::from_str(&text).map_err(|e| format!("{CARD_FILE}: {e}"))?;
    let int = |v: u64| toml::Value::Integer(v as i64);
    if let Some(ctx) = opts.ctx {
        table.insert("context_length".into(), int(ctx));
    }
    if let Some(mc) = opts.max_conc {
        table.insert("max_running_requests".into(), int(mc));
    }
    if let Some(tp) = opts.tp {
        table.insert("tp_size".into(), int(tp));
    }
    if let Some(gpus) = opts.gpus {
        table.insert("preferred_gpus".into(), toml::Value::Array(gpus.into_iter().map(|g| int(g as u64)).collect()));
    }
    let (new_args, kv_dtype) = sgl::take_flag(&opts.sglang_args, "--kv-cache-dtype")?;
    if let Some(msg) = sgl::reserved_flag_error(&new_args) {
        return Err(msg);
    }
    if opts.clear_sglang_args || !new_args.is_empty() {
        let cur: Vec<String> = if opts.clear_sglang_args {
            Vec::new()
        } else {
            table
                .get("extra_sglang_args")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default()
        };
        let merged = sgl::merge_extra_args(&cur, &new_args)?;
        table.insert(
            "extra_sglang_args".into(),
            toml::Value::Array(merged.into_iter().map(toml::Value::String).collect()),
        );
    }
    if let Some(kv) = &kv_dtype {
        table.insert("kv_cache_dtype".into(), toml::Value::String(kv.clone()));
    }
    write_card_table(&dir, &table)?;
    let get = |k: &str, d: i64| table.get(k).and_then(|v| v.as_integer()).unwrap_or(d);
    let (ctx, mc, tp) = (get("context_length", 4096), get("max_running_requests", 8), get("tp_size", 0));
    if row.is_some() {
        con.execute(
            "UPDATE installed_models SET default_ctx = ?1, default_max_conc = ?2, default_tp = ?3 WHERE model_id = ?4",
            rusqlite::params![ctx, mc, tp, opts.model_id],
        )
        .map_err(|e| format!("state.db: {e}"))?;
    }
    if let (Some(kv), Some((_, Some(elems)))) = (&kv_dtype, &row) {
        let cfg = crate::model_facts::read_config(&dir)?;
        let bytes = *elems as u64 * kv_element_bytes(kv, &cfg);
        con.execute(
            "UPDATE installed_models SET kv_bytes_per_token = ?1 WHERE model_id = ?2",
            rusqlite::params![bytes as i64, opts.model_id],
        )
        .map_err(|e| format!("state.db: {e}"))?;
    }
    Ok(())
}

fn find_model_dir(home: &Path, model_id: &str) -> Result<PathBuf, String> {
    // Cards placed under models/ without a state.db row are found by their model_id.
    let models = home.join("models");
    for entry in std::fs::read_dir(&models).map_err(|_| format!("model {model_id} is not installed"))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        let Ok(text) = std::fs::read_to_string(path.join(CARD_FILE)) else {
            continue;
        };
        let id = toml::from_str::<toml::Table>(&text)
            .ok()
            .and_then(|t| t.get("model_id").and_then(|v| v.as_str()).map(String::from));
        if id.as_deref() == Some(model_id) {
            return Ok(path);
        }
    }
    Err(format!("model {model_id} is not installed"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_follow_the_installed_max_ctx() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        assert!(validate_limits(home, "org/m", Some(0), None).unwrap_err().contains("--ctx must be at least 1"));
        assert!(validate_limits(home, "org/m", None, Some(0)).unwrap_err().contains("--max-conc must be at least 1"));
        assert!(validate_limits(home, "org/m", Some(999_999_999), Some(1)).is_ok(), "no state.db yet");
        let con = db::open(home).unwrap();
        con.execute(
            "INSERT INTO installed_models (model_id, local_path, architecture, static_vram_gb, kv_bytes_per_token,
             default_ctx, default_max_conc, default_tp, max_ctx) VALUES ('org/m', 'models/org__m', 'dense', 1.0, 1000,
             1024, 4, 1, 1024)",
            [],
        )
        .unwrap();
        assert!(validate_limits(home, "org/m", Some(1024), Some(4)).is_ok());
        let err = validate_limits(home, "org/m", Some(999_999_999), None).unwrap_err();
        assert!(err.contains("above the maximum context of org/m (1024 tokens)"), "{err}");
        assert!(validate_limits(home, "org/other", Some(1 << 30), None).is_ok());
    }
}
