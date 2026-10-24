// goethe scan: register model directories the user downloaded and moved into GOETHE_HOME/models.

use crate::db;
use crate::install;
use crate::model_facts::{self, GIB};
use crate::paths::{ensure_layout, goethe_home, validate_model_id};
use crate::registry::{fetch, find_model, registry_url};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

pub struct ScanOpts {
    pub yes: bool,
    pub json: bool,
    pub dir: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ScanRow {
    pub dir: String,
    pub model_id: Option<String>,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SglangInfo {
    pub key: String,
    pub archs: BTreeSet<String>,
    pub quants: BTreeSet<String>,
}

pub struct Gpu {
    pub cc: f64,
    pub total_gb: f64,
    pub free_gb: f64,
}

pub struct Machine {
    pub sglang: Option<SglangInfo>,
    pub gpus: Vec<Gpu>,
}

#[derive(Debug)]
pub enum Verdict {
    Supported { arch: String, kind: String, size_gb: f64, notes: Vec<String> },
    Unsupported(String),
}

#[derive(Debug)]
pub enum State {
    Registered(String),
    Unsupported(String),
    Supported { id: String, arch: String, kind: String, size_gb: f64, notes: Vec<String> },
}

pub struct Finding {
    pub path: PathBuf,
    pub name: String,
    pub state: State,
}

pub fn run(opts: ScanOpts) -> Result<(), String> {
    // Survey models/, ask per supported model (or take --yes), register accepted ones in place.
    let home = goethe_home();
    ensure_layout(&home).map_err(|e| e.to_string())?;
    let dirs = candidate_dirs(&home, opts.dir.as_deref())?;
    let con = db::open(&home).map_err(|e| format!("state.db: {e}"))?;
    let machine = machine(&home, &con);
    let findings = survey(&home, &con, dirs, &machine);
    let models = home.join("models");
    let say = |line: &str| {
        if opts.json {
            eprintln!("{line}");
        } else {
            println!("{line}");
        }
    };
    if findings.iter().all(|f| matches!(f.state, State::Registered(_))) {
        say(&format!(
            "No new model directories in {}. Move a downloaded model folder (config.json plus .safetensors) there, for example {}/<org>__<name>, then run goethe scan.",
            models.display(),
            models.display()
        ));
    }
    let interactive = !opts.yes && std::io::stdin().is_terminal() && std::io::stderr().is_terminal();
    let registry = fetch(&registry_url()).ok();
    let mut rows = Vec::new();
    let (mut pending, mut failed) = (0usize, false);
    for f in findings {
        let mut row = ScanRow {
            dir: f.path.display().to_string(),
            model_id: None,
            status: String::new(),
            reason: None,
            notes: Vec::new(),
        };
        match f.state {
            State::Registered(id) => {
                row.model_id = Some(id);
                row.status = "registered".to_string();
            }
            State::Unsupported(why) => {
                say(&format!("Not supported: {}: {why}", f.name));
                row.status = "unsupported".to_string();
                row.reason = Some(why);
            }
            State::Supported { id, arch, kind, size_gb, notes } => {
                let found = format!("Model found: {} ({arch}, {size_gb:.1} GB, {kind})", f.name);
                row.model_id = Some(id.clone());
                row.notes = notes;
                let add = if opts.yes {
                    say(&found);
                    true
                } else if interactive {
                    for n in &row.notes {
                        eprintln!("  note: {n}");
                    }
                    ask(&format!("{found}. Add to goethe? [y/N] "))
                } else {
                    say(&found);
                    pending += 1;
                    row.status = "declined".to_string();
                    row.reason = Some("confirmation required: re-run with --yes".to_string());
                    false
                };
                if opts.yes || !interactive {
                    for n in &row.notes {
                        say(&format!("  note: {n}"));
                    }
                }
                if add {
                    let entry = registry.as_ref().and_then(|r| find_model(r, &id).cloned());
                    match add_model(&home, &id, &f.path, entry, opts.json) {
                        Ok(()) => row.status = "added".to_string(),
                        Err(e) => {
                            say(&format!("could not add {}: {e}", f.name));
                            failed = true;
                            row.status = "unsupported".to_string();
                            row.reason = Some(e);
                        }
                    }
                } else if row.status.is_empty() {
                    row.status = "declined".to_string();
                }
            }
        }
        rows.push(row);
    }
    if opts.json {
        println!("{}", serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?);
    }
    if pending > 0 {
        eprintln!("re-run with --yes to add {pending} model(s), or run goethe scan in a terminal to choose");
        std::process::exit(3);
    }
    if failed {
        return Err("some models could not be added".to_string());
    }
    Ok(())
}

fn ask(question: &str) -> bool {
    eprint!("{question}");
    std::io::stderr().flush().ok();
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer).is_ok() && matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

fn add_model(home: &Path, id: &str, dir: &Path, entry: Option<crate::registry::RegistryModel>, json: bool) -> Result<(), String> {
    // Register in place with the same card, facts and argument inference as goethe install.
    let planned = install::plan_in_place(home, id, dir, entry)?;
    let mut out: Box<dyn Write> = if json { Box::new(std::io::stderr()) } else { Box::new(std::io::stdout()) };
    install::commit(home, &planned, "added", &mut out)?;
    writeln!(out, "next: goethe load --model {id}").map_err(|e| e.to_string())
}

pub struct Found {
    pub id: String,
    pub path: PathBuf,
    pub in_models: bool,
    pub problem: Option<String>,
}

pub fn found_on_this_machine(home: &Path) -> Vec<Found> {
    // Model folders Goethe could register: unregistered ones in GOETHE_HOME/models, then the Hugging Face download cache.
    let Ok(con) = crate::db::open(home) else {
        return Vec::new();
    };
    let m = machine(home, &con);
    let installed: BTreeSet<String> = registered_dirs(home, &con).into_values().collect();
    let mut out: Vec<Found> = survey(home, &con, candidate_dirs(home, None).unwrap_or_default(), &m)
        .into_iter()
        .filter_map(|f| match f.state {
            State::Registered(_) => None,
            State::Unsupported(why) => Some(Found { id: f.name, path: f.path, in_models: true, problem: Some(why) }),
            State::Supported { id, .. } => Some(Found { id, path: f.path, in_models: true, problem: None }),
        })
        .collect();
    for (id, path) in hf_cache_models() {
        if installed.contains(&id) || out.iter().any(|f| f.id == id) {
            continue;
        }
        let problem = match classify(&path, &m) {
            Verdict::Supported { .. } => None,
            Verdict::Unsupported(why) => Some(why),
        };
        out.push(Found { id, path, in_models: false, problem });
    }
    out
}

fn hf_cache_models() -> Vec<(String, PathBuf)> {
    // org/name and the snapshot folder of every model in the Hugging Face cache (refs/main, else the newest snapshot).
    let hub = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/huggingface/hub")));
    let Some(entries) = hub.and_then(|h| fs::read_dir(h).ok()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(id) = name.strip_prefix("models--").map(|r| r.replacen("--", "/", 1)) else {
            continue;
        };
        let snaps = e.path().join("snapshots");
        let main = fs::read_to_string(e.path().join("refs/main")).ok().map(|r| snaps.join(r.trim()));
        let newest = || {
            fs::read_dir(&snaps).ok()?.flatten().max_by_key(|s| s.metadata().and_then(|m| m.modified()).ok()).map(|s| s.path())
        };
        if let Some(snap) = main.filter(|p| p.is_dir()).or_else(newest).filter(|p| p.join("config.json").is_file()) {
            out.push((id, snap));
        }
    }
    out.sort();
    out
}

pub fn register_found(home: &Path, f: &Found) -> Result<(), String> {
    // A folder in GOETHE_HOME/models is registered like goethe scan does; one elsewhere is linked in like goethe install does.
    if f.in_models {
        return add_model(home, &f.id, &f.path, None, false);
    }
    install::run(install::InstallOpts { model_id: f.path.display().to_string(), name: Some(f.id.clone()), ..Default::default() })
}

fn candidate_dirs(home: &Path, dir: Option<&str>) -> Result<Vec<PathBuf>, String> {
    // Immediate subdirectories of models/, or the one directory given (which must be inside models/).
    let models = home.join("models");
    let Some(d) = dir else {
        let mut out: Vec<PathBuf> = fs::read_dir(&models)
            .map_err(|e| format!("{}: {e}", models.display()))?
            .flatten()
            .map(|e| e.path())
            .collect();
        out.sort();
        return Ok(out);
    };
    let canonical = fs::canonicalize(d).map_err(|e| format!("{d}: {e}"))?;
    let root = fs::canonicalize(&models).map_err(|e| e.to_string())?;
    match canonical.strip_prefix(&root) {
        Ok(rel) if !rel.as_os_str().is_empty() => Ok(vec![models.join(rel)]),
        _ => {
            eprintln!(
                "error: {} is not inside {}: move it there and run goethe scan, or register it where it is with goethe install --model {}",
                canonical.display(),
                models.display(),
                canonical.display()
            );
            std::process::exit(2);
        }
    }
}

fn machine(home: &Path, con: &Connection) -> Machine {
    // GPUs that placement may use (denied ones excluded) and the py-env's SGLang capabilities.
    let denied = crate::gpu_policy::denied_set(con).unwrap_or_default();
    let gpus = crate::hw::profile(home)
        .map(|s| {
            s.gpus
                .into_iter()
                .filter(|g| !denied.contains(&g.index))
                .map(|g| {
                    let mut free_gb = g.vram_free_mb as f64 / 1024.0;
                    if g.unified_memory {
                        free_gb = (free_gb - crate::hw::UNIFIED_OS_RESERVE_GB as f64).max(0.0);
                    }
                    Gpu {
                        cc: g.compute_capability.parse().unwrap_or(0.0),
                        total_gb: g.vram_total_mb as f64 / 1024.0,
                        free_gb,
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    Machine {
        sglang: sglang_info(home),
        gpus,
    }
}

pub fn survey(home: &Path, con: &Connection, dirs: Vec<PathBuf>, m: &Machine) -> Vec<Finding> {
    // Skip hidden and goethe-internal dirs, report registered ones, classify the rest.
    let owners = registered_dirs(home, con);
    let mut out = Vec::new();
    for path in dirs {
        let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let meta = fs::symlink_metadata(&path);
        if name.starts_with('.') || !meta.as_ref().is_ok_and(|m| m.is_dir() || m.file_type().is_symlink()) {
            continue;
        }
        let canonical = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
        if let Some(id) = owners.get(&canonical) {
            out.push(Finding { path, name, state: State::Registered(id.clone()) });
            continue;
        }
        if meta.is_ok_and(|m| m.file_type().is_symlink()) {
            let why = format!("a symlink; register the folder it points to with goethe install --model {}", canonical.display());
            out.push(Finding { path, name, state: State::Unsupported(why) });
            continue;
        }
        if install::is_farm(&path) {
            continue;
        }
        let state = match classify(&path, m) {
            Verdict::Unsupported(why) => State::Unsupported(why),
            Verdict::Supported { arch, kind, size_gb, notes } => match model_id_for(&name, con, &path) {
                Ok(id) => State::Supported { id, arch, kind, size_gb, notes },
                Err(why) => State::Unsupported(why),
            },
        };
        out.push(Finding { path, name, state });
    }
    out
}

fn registered_dirs(home: &Path, con: &Connection) -> BTreeMap<PathBuf, String> {
    // Canonical dirs of installed models plus the ternary pack and base dirs they own.
    let mut out = BTreeMap::new();
    let Ok(mut stmt) = con.prepare("SELECT model_id, local_path FROM installed_models") else {
        return out;
    };
    let rows: Vec<(String, String)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .map(|it| it.flatten().collect())
        .unwrap_or_default();
    for (id, rel) in rows {
        let dir = home.join(&rel);
        for owned in std::iter::once(dir.clone()).chain(crate::models::ternary_dirs_of(home, &dir)) {
            if let Ok(c) = fs::canonicalize(&owned) {
                out.insert(c, id.clone());
            }
        }
    }
    out
}

fn model_id_for(name: &str, con: &Connection, path: &Path) -> Result<String, String> {
    // org__name becomes org/name, anything else local/<dirname>; the id must be free or already this dir's.
    let id = match name.split_once("__") {
        Some((org, rest)) => format!("{org}/{rest}"),
        None => format!("local/{name}"),
    };
    validate_model_id(&id).map_err(|e| format!("{e}; rename the directory to <org>__<name>"))?;
    let taken: Option<String> = con
        .query_row("SELECT local_path FROM installed_models WHERE model_id = ?1", [&id], |r| r.get(0))
        .ok();
    match taken {
        Some(rel) if !path.ends_with(&rel) => Err(format!("model id {id} is already used by {rel}; rename the directory")),
        _ => Ok(id),
    }
}

pub fn classify(dir: &Path, m: &Machine) -> Verdict {
    // Decide whether goethe can serve the model in dir, with a concrete reason when it cannot.
    if let Some(why) = incomplete_download(dir) {
        return Verdict::Unsupported(why);
    }
    let cfg: Value = match fs::read_to_string(dir.join("config.json")) {
        Err(_) => return Verdict::Unsupported("no config.json".to_string()),
        Ok(t) => match serde_json::from_str(&t) {
            Ok(v) => v,
            Err(e) => return Verdict::Unsupported(format!("config.json is not valid JSON: {e}")),
        },
    };
    let archs = architectures(&cfg);
    let Some(arch) = archs.first().cloned() else {
        return Verdict::Unsupported("config.json has no architectures".to_string());
    };
    if let Err(why) = check_weights(dir) {
        return Verdict::Unsupported(why);
    }
    let mut notes = Vec::new();
    match &m.sglang {
        Some(info) if !archs.iter().any(|a| info.archs.contains(a)) => {
            return Verdict::Unsupported(format!("architecture {arch} is not served by the installed SGLang"));
        }
        Some(_) => {}
        None => notes.push("SGLang not found in the py-env; architecture not verified".to_string()),
    }
    let method = quant_method(&cfg, dir);
    if let (Some(q), Some(info)) = (&method, &m.sglang) {
        let known = info.quants.contains(q) || (q == "gptq" && info.quants.contains("gptq_marlin"));
        if !known {
            return Verdict::Unsupported(format!("quantization {q} is not supported by the installed SGLang"));
        }
    }
    if let Err(why) = gpu_fitness(quant_format(&cfg, dir), &m.gpus, &mut notes) {
        return Verdict::Unsupported(why);
    }
    let facts = match model_facts::facts_from_install_dir(dir, None) {
        Ok(f) => f,
        Err(e) => return Verdict::Unsupported(format!("cannot read the model shape: {e}")),
    };
    let kv_dtype = install::checkpoint_kv_dtype(&cfg, dir).unwrap_or_else(|| "auto".to_string());
    let est = install::vram_estimate(dir, &cfg, &facts, &[], &kv_dtype);
    let size_gb = est.weights_gb + est.state_gb;
    let need = size_gb + est.kv_bytes_per_token as f64 * 4096.0 / GIB;
    if m.gpus.is_empty() {
        notes.push("no GPU information; size not checked".to_string());
    } else {
        let total: f64 = m.gpus.iter().map(|g| g.total_gb).sum();
        let free: f64 = m.gpus.iter().map(|g| g.free_gb).sum();
        if need > total {
            return Verdict::Unsupported(format!("needs {need:.1} GB; this machine has {total:.1} GB of GPU memory"));
        }
        if need > free {
            notes.push(format!("not enough free VRAM right now ({free:.1} GB free)"));
        }
    }
    let kind = method.unwrap_or_else(|| dtype_label(&cfg));
    Verdict::Supported { arch, kind, size_gb, notes }
}

fn incomplete_download(dir: &Path) -> Option<String> {
    // Folder names or leftover files that mark a download still running or interrupted.
    let marks = |n: &str| [".incomplete", ".partial", ".part", ".aria2", ".tmp"].iter().any(|s| n.ends_with(s));
    let name = dir.file_name()?.to_string_lossy().to_string();
    if marks(&name) {
        return Some("unfinished download (folder name); finish it, then run goethe scan".to_string());
    }
    let mut stack = vec![dir.to_path_buf(), dir.join(".cache/huggingface/download")];
    while let Some(d) = stack.pop() {
        for e in fs::read_dir(&d).into_iter().flatten().flatten() {
            let n = e.file_name().to_string_lossy().to_string();
            if marks(&n) {
                return Some(format!("unfinished download ({n}); finish it, then run goethe scan"));
            }
            if d != dir && e.file_type().is_ok_and(|t| t.is_dir()) {
                stack.push(e.path());
            }
        }
    }
    None
}

fn architectures(cfg: &Value) -> Vec<String> {
    // Root architectures first, then those of a nested language-model config.
    let mut out = Vec::new();
    for v in std::iter::once(cfg).chain(["text_config", "llm_config", "language_config", "thinker_config"].iter().filter_map(|k| cfg.get(k))) {
        for a in v.get("architectures").and_then(|a| a.as_array()).into_iter().flatten() {
            if let Some(s) = a.as_str() {
                out.push(s.to_string());
            }
        }
    }
    out
}

fn check_weights(dir: &Path) -> Result<(), String> {
    // safetensors only: an index whose shards are all present, or exactly one .safetensors file.
    if dir.join("model.safetensors.index.json").is_file() {
        let index: Value = fs::read_to_string(dir.join("model.safetensors.index.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .ok_or("model.safetensors.index.json is not valid JSON")?;
        let shards: BTreeSet<&str> = index
            .get("weight_map")
            .and_then(|w| w.as_object())
            .ok_or("model.safetensors.index.json has no weight_map")?
            .values()
            .filter_map(|v| v.as_str())
            .collect();
        let missing: Vec<&&str> = shards.iter().filter(|s| !dir.join(s).is_file()).collect();
        if let Some(first) = missing.first() {
            return Err(format!("missing shard {first} ({} of {} present)", shards.len() - missing.len(), shards.len()));
        }
        return Ok(());
    }
    let names: Vec<String> = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .flatten()
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    match names.iter().filter(|n| n.ends_with(".safetensors")).count() {
        1 => Ok(()),
        0 if names.iter().any(|n| n.ends_with(".bin") || n.ends_with(".pth")) => {
            Err("only safetensors checkpoints are supported".to_string())
        }
        0 => Err("no .safetensors weights".to_string()),
        _ => Err("several .safetensors files but no model.safetensors.index.json".to_string()),
    }
}

fn quant_config(cfg: &Value) -> Option<&Value> {
    cfg.get("quantization_config")
        .or_else(|| model_facts::resolve_language_config(cfg).ok().and_then(|lm| lm.get("quantization_config")))
}

fn modelopt_name(algo: Option<&str>) -> String {
    match algo.map(str::to_ascii_uppercase).as_deref() {
        Some("FP8") => "modelopt_fp8".to_string(),
        Some("MIXED_PRECISION") => "modelopt_mixed".to_string(),
        Some(a) if a.contains("FP4") => "modelopt_fp4".to_string(),
        _ => "modelopt".to_string(),
    }
}

fn quant_method(cfg: &Value, dir: &Path) -> Option<String> {
    // The SGLang quantization method name the checkpoint selects.
    if let Some(qc) = quant_config(cfg) {
        let method = qc.get("quant_method").and_then(|v| v.as_str()).unwrap_or("unknown").to_ascii_lowercase();
        return Some(if method == "modelopt" {
            modelopt_name(qc.get("quant_algo").and_then(|v| v.as_str()))
        } else {
            method
        });
    }
    if cfg.get("compression_config").is_some() {
        return Some("compressed-tensors".to_string());
    }
    let hq: Value = serde_json::from_str(&fs::read_to_string(dir.join("hf_quant_config.json")).ok()?).ok()?;
    Some(modelopt_name(hq.pointer("/quantization/quant_algo").and_then(|v| v.as_str())))
}

fn quant_format(cfg: &Value, dir: &Path) -> &'static str {
    let text = format!(
        "{}{}",
        quant_config(cfg).map(|q| q.to_string()).unwrap_or_default(),
        fs::read_to_string(dir.join("hf_quant_config.json")).unwrap_or_default()
    )
    .to_ascii_lowercase();
    if text.contains("nvfp4") {
        "nvfp4"
    } else if text.contains("fp8") || text.contains("float-quantized") {
        "fp8"
    } else {
        ""
    }
}

fn gpu_fitness(format: &str, gpus: &[Gpu], notes: &mut Vec<String>) -> Result<(), String> {
    // Refuse only when no usable GPU can run the format; otherwise warn.
    if gpus.is_empty() || format.is_empty() {
        return Ok(());
    }
    let ccs: Vec<f64> = gpus.iter().map(|g| g.cc).collect();
    let list = ccs.iter().map(|c| format!("{c:.1}")).collect::<Vec<_>>().join(", ");
    match format {
        "nvfp4" if ccs.iter().any(|c| *c >= 10.0 || (*c - 8.6).abs() < 0.05) => Ok(()),
        "nvfp4" if ccs.iter().all(|c| *c < 8.0) => Err(format!(
            "NVFP4 needs compute capability 10.0 or newer (or 8.6 with goethe's Marlin patch); GPUs here are {list}"
        )),
        "nvfp4" => {
            notes.push(format!("NVFP4 on compute capability {list} is not verified"));
            Ok(())
        }
        _ => {
            if !ccs.iter().any(|c| *c >= 8.9) {
                notes.push(format!("FP8 runs natively from compute capability 8.9; GPUs here are {list}"));
            }
            Ok(())
        }
    }
}

fn dtype_label(cfg: &Value) -> String {
    let lm = model_facts::resolve_language_config(cfg).unwrap_or(cfg);
    let dtype = lm
        .get("torch_dtype")
        .or_else(|| lm.get("dtype"))
        .or_else(|| cfg.get("torch_dtype"))
        .and_then(|v| v.as_str())
        .unwrap_or("bfloat16");
    match dtype {
        "float16" => "fp16",
        "float32" => "fp32",
        _ => "bf16",
    }
    .to_string()
}

pub fn sglang_info(home: &Path) -> Option<SglangInfo> {
    // EntryClass names and CUDA quantization methods of the py-env's SGLang, read as text, cached per build.
    let lib = home.join("py-env/lib");
    let site = fs::read_dir(&lib)
        .ok()?
        .flatten()
        .map(|e| e.path().join("site-packages"))
        .find(|p| p.join("sglang/srt/models").is_dir())?;
    let dist = fs::read_dir(&site)
        .ok()?
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .find(|n| n.starts_with("sglang-") && n.ends_with(".dist-info"))?;
    let stamp = fs::metadata(site.join(&dist).join("RECORD"))
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let key = format!("{dist}:{stamp}");
    let cache = home.join("run/sglang-introspection.json");
    if let Some(c) = fs::read_to_string(&cache)
        .ok()
        .and_then(|t| serde_json::from_str::<SglangInfo>(&t).ok())
        .filter(|c| c.key == key)
    {
        return Some(c);
    }
    let srt = site.join("sglang/srt");
    let mut archs = BTreeSet::new();
    for e in fs::read_dir(srt.join("models")).ok()?.flatten() {
        if e.path().extension().and_then(|x| x.to_str()) == Some("py") {
            archs.extend(entry_classes(&fs::read_to_string(e.path()).unwrap_or_default()));
        }
    }
    if archs.is_empty() {
        return None;
    }
    let quants = quant_methods(&fs::read_to_string(srt.join("layers/quantization/__init__.py")).unwrap_or_default());
    let info = SglangInfo { key, archs, quants };
    if let Ok(text) = serde_json::to_string(&info) {
        let _ = fs::write(&cache, text);
    }
    Some(info)
}

pub fn entry_classes(source: &str) -> Vec<String> {
    // Class names assigned to a top-level EntryClass, as a name, list or tuple (possibly multi-line).
    let mut out = Vec::new();
    let mut rest = source;
    while let Some(pos) = rest.find("EntryClass") {
        let at_line_start = pos == 0 || rest[..pos].ends_with('\n');
        let after = rest[pos + "EntryClass".len()..].trim_start_matches([' ', '\t']);
        rest = &rest[pos + "EntryClass".len()..];
        let Some(rhs) = after.strip_prefix('=').filter(|r| at_line_start && !r.starts_with('=')) else {
            continue;
        };
        let rhs = rhs.trim_start();
        let expr = match rhs.chars().next() {
            Some(open @ ('[' | '(')) => {
                let close = if open == '[' { ']' } else { ')' };
                &rhs[..rhs.find(close).map(|i| i + 1).unwrap_or(rhs.len())]
            }
            _ => rhs.lines().next().unwrap_or(""),
        };
        out.extend(
            expr.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .filter(|t| t.chars().next().is_some_and(|c| c.is_ascii_uppercase()))
                .map(String::from),
        );
    }
    out
}

pub fn quant_methods(source: &str) -> BTreeSet<String> {
    // Keys of BASE_QUANTIZATION_METHODS plus .update() blocks guarded by an is_cuda() condition.
    let mut out = BTreeSet::new();
    let mut guard = String::new();
    let mut collecting = false;
    for line in source.lines() {
        let top = !line.starts_with([' ', '\t', '#', ')', '}']) && !line.trim().is_empty();
        if top {
            guard = if line.starts_with("if ") { line.to_string() } else { String::new() };
        }
        if line.starts_with("BASE_QUANTIZATION_METHODS") && line.contains('{') {
            collecting = true;
            continue;
        }
        if line.contains("BASE_QUANTIZATION_METHODS.update(") {
            collecting = guard.contains("is_cuda()");
            continue;
        }
        let t = line.trim();
        if collecting && t.starts_with('}') {
            collecting = false;
        } else if collecting {
            if let Some(key) = t.strip_prefix('"').and_then(|r| r.split_once('"')).filter(|(_, r)| r.trim_start().starts_with(':')) {
                out.insert(key.0.to_string());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn machine(total: f64) -> Machine {
        Machine {
            sglang: Some(SglangInfo {
                key: "k".into(),
                archs: ["LlamaForCausalLM".to_string()].into_iter().collect(),
                quants: ["fp8", "compressed-tensors", "modelopt_fp4"].iter().map(|s| s.to_string()).collect(),
            }),
            gpus: vec![Gpu { cc: 12.0, total_gb: total, free_gb: total * 0.9 }],
        }
    }

    fn model(root: &Path, name: &str, arch: &str) -> PathBuf {
        let d = root.join(name);
        fs::create_dir_all(&d).unwrap();
        let cfg = json!({"architectures": [arch], "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "hidden_size": 64, "max_position_embeddings": 4096, "torch_dtype": "bfloat16"});
        fs::write(d.join("config.json"), cfg.to_string()).unwrap();
        d
    }

    #[test]
    fn classification_matrix() {
        let root = tempfile::tempdir().unwrap();
        let r = root.path();
        let ok = model(r, "ok", "LlamaForCausalLM");
        fs::write(ok.join("model.safetensors"), vec![0u8; 4096]).unwrap();
        match classify(&ok, &machine(96.0)) {
            Verdict::Supported { arch, kind, .. } => assert_eq!((arch.as_str(), kind.as_str()), ("LlamaForCausalLM", "bf16")),
            v => panic!("{v:?}"),
        }
        let unknown = model(r, "unknown", "FooForCausalLM");
        fs::write(unknown.join("model.safetensors"), b"x").unwrap();
        let shard = model(r, "shard", "LlamaForCausalLM");
        fs::write(shard.join("model-00001-of-00002.safetensors"), b"x").unwrap();
        let index = json!({"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}});
        fs::write(shard.join("model.safetensors.index.json"), index.to_string()).unwrap();
        let bin = model(r, "bin", "LlamaForCausalLM");
        fs::write(bin.join("pytorch_model.bin"), b"x").unwrap();
        let big = model(r, "big", "LlamaForCausalLM");
        fs::File::create(big.join("model.safetensors")).unwrap().set_len(3 << 30).unwrap();
        let partial = model(r, "partial", "LlamaForCausalLM");
        fs::write(partial.join("model.safetensors"), b"x").unwrap();
        fs::create_dir_all(partial.join(".cache/huggingface/download")).unwrap();
        fs::write(partial.join(".cache/huggingface/download/model.safetensors.abc.incomplete"), b"x").unwrap();
        let reason = |d: &Path| match classify(d, &machine(2.0)) {
            Verdict::Unsupported(why) => why,
            v => panic!("{} should be unsupported: {v:?}", d.display()),
        };
        assert!(reason(&unknown).contains("FooForCausalLM is not served"));
        assert!(reason(&shard).contains("missing shard model-00002-of-00002.safetensors (1 of 2 present)"));
        assert_eq!(reason(&bin), "only safetensors checkpoints are supported");
        assert!(reason(&big).starts_with("needs 3.0 GB; this machine has 2.0 GB"), "{}", reason(&big));
        assert!(reason(&partial).contains("unfinished download"));
    }

    #[test]
    fn survey_reports_registered_and_names_ids() {
        let home = tempfile::tempdir().unwrap();
        let h = home.path();
        crate::paths::ensure_layout(h).unwrap();
        let con = db::open(h).unwrap();
        let models = h.join("models");
        for name in ["org__done", "org__fresh", "plain"] {
            fs::write(model(&models, name, "LlamaForCausalLM").join("model.safetensors"), b"x").unwrap();
        }
        fs::create_dir_all(models.join(".cache")).unwrap();
        con.execute(
            "INSERT INTO installed_models (model_id, local_path, architecture, static_vram_gb, kv_bytes_per_token,
             default_ctx, default_max_conc, default_tp) VALUES ('org/done', 'models/org__done', 'dense', 1.0, 1, 1, 1, 0)",
            [],
        )
        .unwrap();
        let found = survey(h, &con, candidate_dirs(h, None).unwrap(), &machine(96.0));
        let states: Vec<String> = found
            .iter()
            .map(|f| match &f.state {
                State::Registered(id) => format!("{}=registered:{id}", f.name),
                State::Supported { id, .. } => format!("{}=supported:{id}", f.name),
                State::Unsupported(w) => format!("{}=unsupported:{w}", f.name),
            })
            .collect();
        assert_eq!(
            states,
            vec!["org__done=registered:org/done", "org__fresh=supported:org/fresh", "plain=supported:local/plain"]
        );
    }

    #[test]
    fn parses_entry_classes_and_quant_table() {
        let src = "class A:\n    pass\nEntryClass = [\n    LlamaForCausalLM,\n    Phi3ForCausalLM,\n]\n";
        assert_eq!(entry_classes(src), vec!["LlamaForCausalLM", "Phi3ForCausalLM"]);
        assert_eq!(entry_classes("EntryClass = Qwen3ForCausalLM\nx = EntryClass == 1\n"), vec!["Qwen3ForCausalLM"]);
        assert!(entry_classes("__all__ = [\"EntryClass\"]\n").is_empty());
        let table = "BASE_QUANTIZATION_METHODS: Dict[str, Type] = {\n    \"fp8\": Fp8Config,\n    \"compressed-tensors\": C,\n}\n\nif is_cpu() or is_cuda():\n    BASE_QUANTIZATION_METHODS.update(\n        {\n            \"mxfp4\": M,\n        }\n    )\n\nif is_npu():\n    BASE_QUANTIZATION_METHODS.update(\n        {\n            \"gptq\": G,\n        }\n    )\n";
        let q = quant_methods(table);
        assert_eq!(q.into_iter().collect::<Vec<_>>(), vec!["compressed-tensors", "fp8", "mxfp4"]);
    }
}
