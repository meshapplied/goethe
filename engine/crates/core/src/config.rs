// In-place goethe_card.toml and installed_models default updates.

use crate::db;
use crate::paths::goethe_home;
use std::path::PathBuf;

pub struct ConfigOpts {
    pub model_id: String,
    pub ctx: Option<u64>,
    pub max_conc: Option<u64>,
    pub tp: Option<u64>,
    pub gpus: Option<Vec<u32>>,
    pub sglang_args: Vec<String>,
    pub clear_sglang_args: bool,
}

pub fn run(opts: ConfigOpts) -> Result<(), String> {
    let home = goethe_home();
    let dir = find_model_dir(&home, &opts.model_id)?;
    let card_path = dir.join("goethe_card.toml");
    let text = std::fs::read_to_string(&card_path).map_err(|e| e.to_string())?;
    let mut root: toml::Value = toml::from_str(&text).map_err(|e| e.to_string())?;
    let table = root
        .as_table_mut()
        .ok_or_else(|| "goethe_card.toml root must be a table".to_string())?;
    if let Some(ctx) = opts.ctx {
        table.insert(
            "context_length".into(),
            toml::Value::Integer(ctx as i64),
        );
    }
    if let Some(mc) = opts.max_conc {
        table.insert(
            "max_running_requests".into(),
            toml::Value::Integer(mc as i64),
        );
    }
    if let Some(tp) = opts.tp {
        table.insert("tp_size".into(), toml::Value::Integer(tp as i64));
    }
    if let Some(gpus) = opts.gpus {
        let arr: Vec<toml::Value> = gpus
            .into_iter()
            .map(|g| toml::Value::Integer(g as i64))
            .collect();
        table.insert("preferred_gpus".into(), toml::Value::Array(arr));
    }
    if opts.clear_sglang_args {
        table.insert("extra_sglang_args".into(), toml::Value::Array(vec![]));
    } else if !opts.sglang_args.is_empty() {
        let mut cur: Vec<String> = table
            .get("extra_sglang_args")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        for a in &opts.sglang_args {
            if !cur.contains(a) {
                cur.push(a.clone());
            }
        }
        table.insert(
            "extra_sglang_args".into(),
            toml::Value::Array(cur.into_iter().map(toml::Value::String).collect()),
        );
    }
    let tp = table
        .get("tp_size")
        .and_then(|v| v.as_integer())
        .unwrap_or(1);
    let ctx = table
        .get("context_length")
        .and_then(|v| v.as_integer())
        .unwrap_or(4096);
    let mc = table
        .get("max_running_requests")
        .and_then(|v| v.as_integer())
        .unwrap_or(4);
    let out = toml::to_string_pretty(&root).map_err(|e| e.to_string())?;
    std::fs::write(&card_path, out).map_err(|e| e.to_string())?;
    if let Ok(con) = db::open(&home) {
        let _ = con.execute(
            "UPDATE installed_models SET default_ctx = ?1, default_max_conc = ?2, default_tp = ?3
             WHERE model_id = ?4",
            rusqlite::params![ctx, mc, tp, opts.model_id],
        );
    }
    Ok(())
}

fn find_model_dir(home: &std::path::Path, model_id: &str) -> Result<PathBuf, String> {
    let models = home.join("models");
    if !models.is_dir() {
        return Err(format!("model {model_id} not installed"));
    }
    for entry in std::fs::read_dir(&models).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let card_path = path.join("goethe_card.toml");
        if !card_path.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&card_path).map_err(|e| e.to_string())?;
        if card_model_id(&text).as_deref() == Some(model_id) {
            return Ok(path);
        }
    }
    Err(format!("model {model_id} not found under models/"))
}

fn card_model_id(text: &str) -> Option<String> {
    let root: toml::Value = toml::from_str(text).ok()?;
    root.get("model_id")
        .and_then(|v| v.as_str())
        .map(String::from)
}
