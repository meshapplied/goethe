// goethe models list and remove, plus per-model host RAM needs.

use crate::chooser::ChooserInput;
use crate::db;
use crate::host;
use crate::install;
use crate::paths::{contained_in_models, goethe_home, shell_quote, validate_model_id};
use crate::recommend::{registry_input, View};
use crate::registry::{fetch, registry_url, Registry};
use crate::rpc;
use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize)]
struct ModelRow {
    id: String,
    status: String,
    gb: f64,
    fit: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    custom: Option<bool>,
    #[serde(skip)]
    verdict: Verdict,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Verdict {
    Unknown,
    Fits { per_gpu: f64, total: f64 },
    TooLarge,
    Unsupported,
}

impl Verdict {
    fn label(&self) -> String {
        match self {
            Verdict::Unknown => "unknown (run goethe hw profile)".to_string(),
            Verdict::Fits { per_gpu, .. } => format!("fits: {per_gpu:.1} GB per GPU"),
            Verdict::TooLarge => "does not fit: not enough VRAM".to_string(),
            Verdict::Unsupported => "does not fit: unsupported architecture".to_string(),
        }
    }
}

struct Installed {
    id: String,
    input: ChooserInput,
    default_tp: i64,
}

const LIVE: &str = "status NOT IN ('FAILED', 'CANCELLED')";

pub fn run_list(json: bool, installed_only: bool, available_only: bool) -> Result<(), String> {
    // Installed rows use the stored budget inputs; registry rows use the registry estimate.
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let loaded = loaded_instances(&con);
    let installed = installed_rows(&home, &con)?;
    let reg = fetch(&registry_url()).ok();
    let view = crate::placement::load_hw(&home).map(|hw| View::capacity(&home, hw));
    let mut rows = Vec::new();
    if !available_only {
        for m in &installed {
            let pinned = (m.default_tp > 0).then_some(m.default_tp as usize);
            let verdict = verdict(view.as_ref(), &m.id, &m.input, pinned);
            let (status, fit) = match loaded.get(&m.id) {
                Some(live) => loaded_state(live),
                None => ("installed".to_string(), verdict.label()),
            };
            rows.push(ModelRow {
                id: m.id.clone(),
                status,
                gb: m.input.static_vram_gb,
                fit,
                custom: is_custom_model(&m.id, reg.as_ref()).then_some(true),
                verdict,
            });
        }
    }
    if let (false, Some(r)) = (installed_only, &reg) {
        for m in r.models.iter().filter(|m| m.manual_download_reason().is_none()) {
            if installed.iter().any(|i| i.id == m.id) {
                continue;
            }
            let pinned = m.default_tp.filter(|t| *t > 0).map(|t| t as usize);
            let verdict = verdict(view.as_ref(), &m.id, &registry_input(m), pinned);
            rows.push(ModelRow {
                id: m.id.clone(),
                status: "available".to_string(),
                gb: m.download_gb.unwrap_or(m.static_vram_gb),
                fit: verdict.label(),
                custom: None,
                verdict,
            });
        }
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?);
    } else {
        print!("{}", list_text(&rows, view.as_ref().map(View::free_text)));
    }
    Ok(())
}

pub fn resolve_installed_id(home: &Path, id: &str) -> Result<String, String> {
    // An installed model id for what the user typed: exact, any case, or its models/ folder name; else a suggestion.
    let con = db::open(home).map_err(|e| e.to_string())?;
    let mut stmt = con.prepare("SELECT model_id, local_path FROM installed_models").map_err(|e| e.to_string())?;
    let rows: Vec<(String, String)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .collect();
    match_installed(id, &rows).map_err(|near| not_installed_message(id, near.as_deref()))
}

fn match_installed(id: &str, rows: &[(String, String)]) -> Result<String, Option<String>> {
    // Exact id, then case-insensitive id, then the folder name (org__name); Err carries the nearest id, if close.
    let folder = |path: &str| Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let lower = id.to_lowercase();
    let found = rows.iter().find(|(m, _)| m == id).or_else(|| rows.iter().find(|(m, _)| m.to_lowercase() == lower)).or_else(|| {
        rows.iter().find(|(m, p)| folder(p).to_lowercase() == lower || m.to_lowercase() == lower.replacen("__", "/", 1))
    });
    if let Some((m, _)) = found {
        return Ok(m.clone());
    }
    let near = rows
        .iter()
        .map(|(m, _)| (edit_distance(&lower, &m.to_lowercase()), m))
        .filter(|(d, m)| *d <= (m.len() / 4).max(2))
        .min_by_key(|(d, _)| *d)
        .map(|(_, m)| m.clone());
    Err(near)
}

fn not_installed_message(id: &str, near: Option<&str>) -> String {
    match near {
        Some(m) => format!("model {id} is not installed; did you mean {m}? (goethe models lists installed models)"),
        None => format!("model {id} is not installed (goethe models lists installed models; goethe install --model {id} adds it)"),
    }
}

fn edit_distance(a: &str, b: &str) -> usize {
    // Levenshtein distance over characters.
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            cur.push((prev[j] + usize::from(ca != *cb)).min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

fn fit_colored(fit: &str) -> String {
    // The verdict word of a fit label in color: fits green, does not fit red; the details stay plain.
    use crate::term::{paint, Color};
    for (word, color) in [("does not fit", Color::Red), ("fits", Color::Green)] {
        if let Some(rest) = fit.strip_prefix(word) {
            return format!("{}{rest}", paint(word, color));
        }
    }
    fit.to_string()
}

fn verdict(view: Option<&View>, id: &str, input: &ChooserInput, pinned: Option<usize>) -> Verdict {
    // Fits (with its VRAM per GPU and in total), or whether the architecture or the free VRAM stops it.
    let Some(view) = view else {
        return Verdict::Unknown;
    };
    if !view.hw.gpus.iter().any(|g| crate::gpu_arch::model_runs_on(&input.supported_sm, &g.compute_capability)) {
        return Verdict::Unsupported;
    }
    match view.fit(id, input, pinned) {
        Ok(f) => Verdict::Fits { per_gpu: f.need.total(), total: f.need.total() * f.tp as f64 },
        Err(_) => Verdict::TooLarge,
    }
}

fn list_text(rows: &[ModelRow], free_gb: Option<String>) -> String {
    // Installed models, then available ones grouped by the VRAM they take, then those that do not fit and why.
    use crate::term::{paint, Color, Table};
    let mut out = String::new();
    if let Some(free) = free_gb {
        out.push_str(&format!("{free}\n\n"));
    }
    let row = |r: &ModelRow| vec![r.id.clone(), crate::term::status(&r.status), format!("{:.1} GB", r.gb), fit_colored(&r.fit)];
    let header = ["MODEL", "STATUS", "DISK", "FIT ON THIS MACHINE"];
    let (installed, available): (Vec<&ModelRow>, Vec<&ModelRow>) = rows.iter().partition(|r| r.status != "available");
    if !installed.is_empty() {
        let mut table = Table::new(&header);
        installed.iter().for_each(|r| table.row(row(r)));
        out.push_str(&format!("{}\n{}\n", paint("Installed", Color::Bold), table.render()));
    }
    let fits: Vec<(f64, Vec<String>)> = available
        .iter()
        .filter_map(|r| match r.verdict {
            Verdict::Fits { total, .. } => Some((total, row(r))),
            _ => None,
        })
        .collect();
    out.push_str(&crate::recommend::banded(&header, fits));
    for (title, kind) in [("Too large for the allowed GPUs", Verdict::TooLarge), ("Unsupported architecture", Verdict::Unsupported), ("Unknown", Verdict::Unknown)] {
        let mut table = Table::new(&header);
        let mut group: Vec<&&ModelRow> = available.iter().filter(|r| r.verdict == kind).collect();
        group.sort_by(|a, b| a.gb.total_cmp(&b.gb));
        group.iter().for_each(|r| table.row(row(r)));
        if !table.is_empty() {
            out.push_str(&format!("{}\n{}\n", paint(title, Color::Bold), table.render()));
        }
    }
    out.trim_end().to_string() + "\n"
}

fn installed_rows(home: &Path, con: &Connection) -> Result<Vec<Installed>, String> {
    // The columns and card fields goethed budgets a load with: KV bytes for the card dtype, Marlin extra, ctx caps.
    let mut stmt = con
        .prepare(
            "SELECT model_id, static_vram_gb, kv_bytes_per_token, default_ctx, default_max_conc, default_tp,
                    max_ctx, num_attention_heads, num_kv_heads, attention, kv_elems_per_token, local_path
             FROM installed_models ORDER BY model_id",
        )
        .map_err(|e| e.to_string())?;
    let rows = stmt
        .query_map([], |r| {
            let kv_bytes: i64 = r.get(2)?;
            let default_ctx: i64 = r.get(3)?;
            let max_ctx = r.get::<_, Option<i64>>(6)?.unwrap_or(default_ctx).max(1) as u64;
            let card = card_value(&home.join(r.get::<_, String>(11)?));
            let elems = r.get::<_, Option<i64>>(10)?.unwrap_or(kv_bytes).max(0) as u64;
            Ok(Installed {
                id: r.get(0)?,
                default_tp: r.get(5)?,
                input: ChooserInput {
                    static_vram_gb: r.get(1)?,
                    kv_bytes_per_token: card_kv_bytes(&card, elems, kv_bytes.max(0) as u64).max(1),
                    num_attention_heads: r.get::<_, Option<i64>>(7)?.unwrap_or(1) as u64,
                    num_kv_heads: r.get::<_, Option<i64>>(8)?.unwrap_or(1) as u64,
                    attention: r.get::<_, Option<String>>(9)?.unwrap_or_else(|| "gqa".to_string()),
                    max_ctx,
                    ctx: (default_ctx.max(1) as u64).min(max_ctx),
                    max_conc: r.get::<_, i64>(4)?.max(1) as u64,
                    marlin_extra_gb: install::card_marlin_extra_gb(&card),
                    supported_sm: install::card_supported_sm(&card),
                },
            })
        })
        .map_err(|e| e.to_string())?;
    rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
}

pub fn fit_input(home: &Path, model_id: &str) -> Result<(ChooserInput, Option<usize>), String> {
    // Budget inputs of one installed model (with its card tp), as goethe models computes them.
    let con = db::open(home).map_err(|e| e.to_string())?;
    let m = installed_rows(home, &con)?
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("{model_id} is not installed (goethe models --installed lists installed models)"))?;
    Ok((m.input, (m.default_tp > 0).then_some(m.default_tp as usize)))
}

fn card_value(dir: &Path) -> toml::Value {
    fs::read_to_string(dir.join(install::CARD_FILE))
        .ok()
        .and_then(|t| toml::from_str(&t).ok())
        .unwrap_or_else(|| toml::Value::Table(Default::default()))
}

fn card_kv_bytes(card: &toml::Value, elems: u64, installed_bytes: u64) -> u64 {
    // KV bytes per token for the card's kv_cache_dtype, as goethed computes them; auto keeps the installed figure.
    match card.get("kv_cache_dtype").and_then(|v| v.as_str()).map(str::to_ascii_lowercase) {
        Some(d) if d.contains("fp8") => elems,
        Some(d) if d.contains("fp4") => elems.div_ceil(2),
        Some(d) if d.contains("16") => elems * 2,
        _ => installed_bytes,
    }
}

fn loaded_instances(con: &Connection) -> HashMap<String, Vec<(String, String)>> {
    // Live instances per model: (status, GPU list), oldest first.
    let mut out: HashMap<String, Vec<(String, String)>> = HashMap::new();
    let sql = "SELECT model_id, status, gpu_indices FROM loaded_instances
               WHERE status IN ('STARTING', 'RUNNING', 'RESTARTING', 'DRAINING') ORDER BY rowid";
    let Ok(mut stmt) = con.prepare(sql) else {
        return out;
    };
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?)));
    for (model, status, gpus) in rows.into_iter().flatten().flatten() {
        out.entry(model).or_default().push((status, gpus));
    }
    out
}

fn loaded_state(live: &[(String, String)]) -> (String, String) {
    // A loaded model shows its instances instead of whether one more copy would fit.
    let status = if live.iter().any(|(s, _)| s == "RUNNING") { "loaded" } else { live[0].0.as_str() };
    let each: Vec<String> = live.iter().map(|(s, g)| format!("GPU {g} {s}")).collect();
    let n = live.len();
    (
        status.to_ascii_lowercase(),
        format!("{n} instance{}: {} (goethe load --replica checks room for another)", if n == 1 { "" } else { "s" }, each.join(", ")),
    )
}

fn is_custom_model(id: &str, reg: Option<&Registry>) -> bool {
    id.starts_with("local/") || reg.is_none_or(|r| !r.models.iter().any(|m| m.id == id))
}

struct RemovalPlan {
    primary: PathBuf,
    delete: Vec<PathBuf>,
    keep: Vec<(PathBuf, String)>,
    source: Option<String>,
}

pub fn run_remove(model_id: &str, yes: bool, force: bool) -> Result<(), String> {
    // Unload when needed, drop the model's rows in one transaction, then delete only what goethe owns.
    validate_model_id(model_id)?;
    let home = goethe_home();
    let mut con = db::open(&home).map_err(|e| e.to_string())?;
    let row: Option<(String, i64)> = con
        .query_row(
            "SELECT local_path, managed FROM installed_models WHERE model_id = ?1",
            [model_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .optional()
        .map_err(|e| format!("state.db: {e}"))?;
    let Some((local_path, managed)) = row else {
        return Err(format!("{model_id} is not installed (goethe models --installed lists installed models)"));
    };
    let plan = removal_plan(&home, &con, model_id, &local_path, managed != 0)?;
    let instances: i64 = con
        .query_row(
            &format!("SELECT COUNT(*) FROM loaded_instances WHERE model_id = ?1 AND {LIVE}"),
            [model_id],
            |r| r.get(0),
        )
        .map_err(|e| format!("state.db: {e}"))?;
    let mut steps = Vec::new();
    if instances > 0 {
        let how = if force { "immediately, without draining" } else { "after draining in-flight requests" };
        steps.push(format!("unload {instances} running instance(s) {how}"));
    }
    for p in &plan.delete {
        if plan.source.is_some() && p == &plan.primary {
            steps.push(format!("remove goethe's links and card in {}", p.display()));
        } else {
            steps.push(format!("delete {} ({})", p.display(), usage_text(dir_usage_gb(p))));
        }
    }
    for (p, why) in &plan.keep {
        steps.push(format!("keep {} ({why})", p.display()));
    }
    if let Some(src) = &plan.source {
        steps.push(format!("unregister the model; your files in {src} stay on disk"));
    }
    host::confirm(&format!("goethe remove --model {model_id} will:\n  {}\nProceed?", steps.join("\n  ")), yes)?;
    if instances > 0 {
        unload(model_id, force)?;
    }
    let tx = con.transaction().map_err(|e| format!("state.db: {e}"))?;
    for sql in [
        "DELETE FROM installed_models WHERE model_id = ?1",
        "DELETE FROM desired_loads WHERE model_id = ?1",
        "DELETE FROM model_aliases WHERE model_id = ?1",
    ] {
        tx.execute(sql, [model_id]).map_err(|e| format!("state.db: {e}"))?;
    }
    tx.commit().map_err(|e| format!("state.db: {e}"))?;
    for p in &plan.delete {
        remove_owned(&home, p)?;
        println!("deleted {}", p.display());
    }
    match &plan.source {
        Some(src) => println!(
            "removed {model_id} from goethe; its files were not deleted. To delete them run:\n  rm -rf {}",
            shell_quote(src)
        ),
        None => println!("removed {model_id}"),
    }
    Ok(())
}

fn removal_plan(home: &Path, con: &Connection, model_id: &str, local_path: &str, managed: bool) -> Result<RemovalPlan, String> {
    // Directories this model owns, minus any another installed model still uses.
    let primary = contained_in_models(home, &home.join(local_path))?;
    let mut candidates = vec![primary.clone()];
    candidates.extend(ternary_dirs_of(home, &primary));
    let mut users: Vec<(String, Vec<PathBuf>)> = Vec::new();
    let mut stmt = con
        .prepare("SELECT model_id, local_path FROM installed_models WHERE model_id != ?1")
        .map_err(|e| e.to_string())?;
    let others = stmt
        .query_map([model_id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))
        .map_err(|e| e.to_string())?;
    for (id, rel) in others.flatten() {
        if let Ok(dir) = contained_in_models(home, &home.join(rel)) {
            let mut owned = vec![dir.clone()];
            owned.extend(ternary_dirs_of(home, &dir));
            users.push((id, owned));
        }
    }
    let mut plan = RemovalPlan {
        primary: primary.clone(),
        delete: Vec::new(),
        keep: Vec::new(),
        source: None,
    };
    for c in candidates {
        if !c.exists() && fs::symlink_metadata(&c).is_err() {
            continue;
        }
        if let Some((id, _)) = users.iter().find(|(_, owned)| owned.contains(&c)) {
            plan.keep.push((c, format!("still used by {id}")));
            continue;
        }
        if c == primary && !managed {
            plan.source = Some(source_of(&primary).unwrap_or_else(|| primary.display().to_string()));
            if !is_link_or_farm(&primary) {
                plan.keep.push((c, "it holds files, not links; delete it yourself".to_string()));
                continue;
            }
        }
        plan.delete.push(c);
    }
    Ok(plan)
}

pub fn ternary_dirs_of(home: &Path, dir: &Path) -> Vec<PathBuf> {
    let Ok(card) = install::read_card(dir) else {
        return Vec::new();
    };
    match (card.runtime.as_deref(), card.pack_id, card.base_model_id) {
        (Some("ternary"), Some(pack), Some(base)) => {
            let d = install::ternary_dirs(home, &pack, &base);
            [d.pack, d.base]
                .into_iter()
                .filter_map(|p| contained_in_models(home, &p).ok())
                .collect()
        }
        _ => Vec::new(),
    }
}

fn source_of(dir: &Path) -> Option<String> {
    // Where an in-place install's files live: the card's source_dir, a legacy link, or a farm link.
    if let Ok(card) = install::read_card(dir) {
        if let Some(s) = card.source_dir {
            return Some(s);
        }
    }
    if let Ok(t) = fs::read_link(dir) {
        return Some(t.display().to_string());
    }
    let t = fs::read_link(dir.join("config.json")).ok()?;
    t.parent().map(|p| p.display().to_string())
}

fn is_link_or_farm(dir: &Path) -> bool {
    fs::symlink_metadata(dir).map(|m| m.file_type().is_symlink()).unwrap_or(false) || install::is_farm(dir)
}

fn remove_owned(home: &Path, p: &Path) -> Result<(), String> {
    // Delete a path inside models/; a symlink is removed itself, never followed.
    let p = contained_in_models(home, p)?;
    let meta = fs::symlink_metadata(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    let r = if meta.is_dir() { fs::remove_dir_all(&p) } else { fs::remove_file(&p) };
    r.map_err(|e| format!("delete {}: {e}", p.display()))
}

fn dir_usage_gb(p: &Path) -> (f64, f64) {
    // GB of files under p (each inode once, links not followed) and the GB deleting p frees.
    fn walk(p: &Path, inodes: &mut HashMap<(u64, u64), (u64, u64, u64)>) {
        let Ok(meta) = fs::symlink_metadata(p) else {
            return;
        };
        if meta.is_dir() {
            for e in fs::read_dir(p).into_iter().flatten().flatten() {
                walk(&e.path(), inodes);
            }
        } else if meta.is_file() {
            inodes.entry((meta.dev(), meta.ino())).or_insert((meta.len(), meta.nlink(), 0)).2 += 1;
        }
    }
    let mut inodes = HashMap::new();
    walk(p, &mut inodes);
    let total: u64 = inodes.values().map(|v| v.0).sum();
    let freed: u64 = inodes.values().filter(|(_, links, seen)| seen >= links).map(|v| v.0).sum();
    (total as f64 / 1e9, freed as f64 / 1e9)
}

fn usage_text((total, freed): (f64, f64)) -> String {
    // Hard links to files kept elsewhere free nothing, so say what deleting actually frees.
    if freed + 0.05 < total {
        format!("{total:.1} GB, frees {freed:.1} GB: the rest are hard links to files kept elsewhere")
    } else {
        format!("{total:.1} GB")
    }
}

fn unload(model_id: &str, force: bool) -> Result<(), String> {
    let reply = rpc::call("unload", json!({"model_id": model_id, "all": false, "force": force}))
        .map_err(|e| format!("cannot unload {model_id} before removing it: {}", crate::paths::daemon_err(e)))?;
    if reply.code != 0 {
        let msg = if reply.stderr.trim().is_empty() { reply.stdout } else { reply.stderr };
        return Err(format!("unload of {model_id} failed, nothing was removed: {}", msg.trim()));
    }
    Ok(())
}

pub fn host_ram_need_gb(home: &Path, model_id: &str) -> Option<f64> {
    // Host RAM held by offloaded weights, for cards that offload; None for cards that do not.
    let con = db::open(home).ok()?;
    let rel: String = con
        .query_row("SELECT local_path FROM installed_models WHERE model_id = ?1", [model_id], |r| r.get(0))
        .ok()?;
    let card = install::read_card(&home.join(rel)).ok()?;
    let flags = card
        .extra_sglang_args
        .iter()
        .any(|a| a == "--ple-offload-embedding" || a.starts_with("--cpu-offload-gb"));
    match card.host_offload_gb {
        Some(gb) => Some(gb),
        None => flags.then_some(0.0),
    }
}

pub fn host_ram_warning(home: &Path, model_id: &str) -> Option<String> {
    // Warn when MemAvailable is below what an offloading card needs plus headroom.
    let need = host_ram_need_gb(home, model_id)?;
    let avail = host::mem_available_gb()?;
    let required = if need > 0.0 { need + 4.0 } else { 16.0 };
    (avail < required).then(|| {
        format!(
            "warning: {model_id} offloads weights to host RAM ({need:.1} GB pinned); only {avail:.1} GB is available, the load may fail or swap"
        )
    })
}

pub fn installed_ids(home: &Path) -> Vec<String> {
    let Ok(con) = db::open(home) else {
        return Vec::new();
    };
    let Ok(mut stmt) = con.prepare("SELECT model_id FROM installed_models") else {
        return Vec::new();
    };
    stmt.query_map([], |r| r.get::<_, String>(0))
        .map(|rows| rows.flatten().collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_ids_resolve_by_case_and_folder_or_suggest_the_nearest() {
        let rows = vec![("Qwen/Qwen3-0.6B".to_string(), "models/Qwen__Qwen3-0.6B".to_string())];
        assert_eq!(match_installed("Qwen/Qwen3-0.6B", &rows), Ok("Qwen/Qwen3-0.6B".to_string()));
        assert_eq!(match_installed("qwen/qwen3-0.6b", &rows), Ok("Qwen/Qwen3-0.6B".to_string()));
        assert_eq!(match_installed("Qwen__Qwen3-0.6B", &rows), Ok("Qwen/Qwen3-0.6B".to_string()));
        assert_eq!(match_installed("Qwen/Qwen3-0.68", &rows), Err(Some("Qwen/Qwen3-0.6B".to_string())));
        assert_eq!(match_installed("meta-llama/Llama-3.1-8B", &rows), Err(None));
        assert_eq!(edit_distance("kitten", "sitting"), 3);
    }

    fn home_with_db() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        crate::paths::ensure_layout(dir.path()).unwrap();
        dir
    }

    fn insert(con: &Connection, id: &str, rel: &str, managed: i64) {
        con.execute(
            "INSERT INTO installed_models (model_id, local_path, architecture, static_vram_gb, kv_bytes_per_token,
             default_ctx, default_max_conc, default_tp, managed) VALUES (?1, ?2, 'dense', 1.0, 1, 1, 1, 0, ?3)",
            rusqlite::params![id, rel, managed],
        )
        .unwrap();
    }

    #[test]
    fn loaded_models_show_their_instances() {
        let one = vec![("RUNNING".to_string(), "2".to_string())];
        let (status, fit) = loaded_state(&one);
        assert_eq!(status, "loaded");
        assert!(fit.starts_with("1 instance: GPU 2 RUNNING"), "{fit}");
        let two = vec![("STARTING".to_string(), "3".to_string()), ("RESTARTING".to_string(), "2,3".to_string())];
        let (status, fit) = loaded_state(&two);
        assert_eq!(status, "starting");
        assert!(fit.starts_with("2 instances: GPU 3 STARTING, GPU 2,3 RESTARTING"), "{fit}");
        let dir = tempfile::tempdir().unwrap();
        let con = db::open(dir.path()).unwrap();
        con.execute(
            "INSERT INTO loaded_instances (instance_id, model_id, pid, pgid, internal_port, gpu_indices,
             allocated_vram_gb_per_gpu, mem_fraction_static, ctx_len, max_conc, status)
             VALUES ('i1', 'org/m', 10, 10, 30001, '2', 5.0, 0.1, 4096, 4, 'RUNNING'),
                    ('i2', 'org/m', 0, 0, 30002, '3', 5.0, 0.1, 4096, 4, 'FAILED')",
            [],
        )
        .unwrap();
        let live = loaded_instances(&con);
        assert_eq!(live.get("org/m").map(Vec::len), Some(1), "tombstones are not loaded instances");
    }

    #[test]
    fn hard_links_free_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let (tree, other) = (dir.path().join("tree"), dir.path().join("other"));
        fs::create_dir_all(&tree).unwrap();
        fs::create_dir_all(&other).unwrap();
        let sparse = |p: PathBuf, len: u64| fs::File::create(p).unwrap().set_len(len).unwrap();
        sparse(tree.join("own.bin"), 300_000_000);
        sparse(other.join("w.bin"), 200_000_000);
        fs::hard_link(other.join("w.bin"), tree.join("w.bin")).unwrap();
        sparse(tree.join("pair.bin"), 100_000_000);
        fs::hard_link(tree.join("pair.bin"), tree.join("pair2.bin")).unwrap();
        let (total, freed) = dir_usage_gb(&tree);
        assert!((total - 0.6).abs() < 1e-9 && (freed - 0.4).abs() < 1e-9, "{total} {freed}");
        assert_eq!(usage_text((0.6, 0.4)), "0.6 GB, frees 0.4 GB: the rest are hard links to files kept elsewhere");
        assert_eq!(usage_text((0.6, 0.6)), "0.6 GB");
    }

    #[test]
    fn plan_deletes_managed_and_keeps_in_place_sources() {
        let home = home_with_db();
        let h = home.path();
        let con = db::open(h).unwrap();
        fs::create_dir_all(h.join("models/org__a")).unwrap();
        fs::write(h.join("models/org__a/w.safetensors"), b"x").unwrap();
        insert(&con, "org/a", "models/org__a", 1);
        let plan = removal_plan(h, &con, "org/a", "models/org__a", true).unwrap();
        assert_eq!(plan.delete.len(), 1);
        assert!(plan.source.is_none());
        let src = tempfile::tempdir().unwrap();
        fs::write(src.path().join("config.json"), "{}").unwrap();
        install::build_farm(src.path(), &h.join("models/local__b")).unwrap();
        insert(&con, "local/b", "models/local__b", 0);
        let plan = removal_plan(h, &con, "local/b", "models/local__b", false).unwrap();
        assert_eq!(plan.delete.len(), 1);
        assert_eq!(plan.source.as_deref(), Some(src.path().to_str().unwrap()));
        remove_owned(h, &plan.delete[0]).unwrap();
        assert!(src.path().join("config.json").is_file());
        assert!(removal_plan(h, &con, "x", "models/..", true).is_err());
        assert!(removal_plan(h, &con, "x", "models", true).is_err());
    }

    #[test]
    fn shared_ternary_base_is_kept() {
        let home = home_with_db();
        let h = home.path();
        let con = db::open(h).unwrap();
        let dirs = install::ternary_dirs(h, "mesh/pack", "base/model");
        fs::create_dir_all(&dirs.pack).unwrap();
        fs::create_dir_all(&dirs.base).unwrap();
        fs::create_dir_all(h.join("models/mesh__pack")).unwrap();
        fs::write(
            h.join("models/mesh__pack").join(install::CARD_FILE),
            "model_id = \"mesh/pack\"\ncontext_length = 1\nmax_running_requests = 1\ntp_size = 0\nep_size = 1\nkv_cache_dtype = \"auto\"\npreferred_gpus = []\nextra_sglang_args = []\nruntime = \"ternary\"\npack_id = \"mesh/pack\"\nbase_model_id = \"base/model\"\n",
        )
        .unwrap();
        insert(&con, "mesh/pack", "models/mesh__pack", 1);
        insert(&con, "base/model", "models/base__model", 1);
        let plan = removal_plan(h, &con, "mesh/pack", "models/mesh__pack", true).unwrap();
        assert_eq!(plan.delete.len(), 2, "{:?}", plan.delete);
        assert_eq!(plan.keep.len(), 1);
        assert_eq!(plan.keep[0].1, "still used by base/model");
    }
}
