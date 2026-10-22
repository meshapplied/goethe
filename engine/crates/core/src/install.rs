// Model install: Hugging Face or local trees, goethe_card.toml, and state.db registration.

use crate::db;
use crate::hf::{self, DownloadOpts};
use crate::model_facts::{self, GIB};
use crate::paths::{contained_in_models, ensure_layout, goethe_home, model_dir, validate_model_id};
use crate::registry::{fetch, find_model, registry_url, RegistryModel};
use crate::safetensors;
use crate::sglang_infer as sgl;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::io::{IsTerminal, Write};
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

pub const CARD_FILE: &str = "goethe_card.toml";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoetheCard {
    pub model_id: String,
    pub context_length: u64,
    pub max_running_requests: u64,
    pub tp_size: u64,
    pub ep_size: u64,
    pub kv_cache_dtype: String,
    pub preferred_gpus: Vec<u32>,
    pub extra_sglang_args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pack_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<toml::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pythonpath: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub marlin_extra_gb: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_offload_gb: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_dir: Option<String>,
}

const CARD_KEYS: &[&str] = &[
    "model_id",
    "context_length",
    "max_running_requests",
    "tp_size",
    "ep_size",
    "kv_cache_dtype",
    "preferred_gpus",
    "extra_sglang_args",
    "runtime",
    "pack_id",
    "base_model_id",
    "env",
    "pythonpath",
    "marlin_extra_gb",
    "host_offload_gb",
    "source_dir",
];

pub struct InstallOpts {
    pub model_id: String,
    pub ctx: Option<u64>,
    pub max_conc: Option<u64>,
    pub tp: Option<u64>,
    pub gpus: Option<Vec<u32>>,
    pub load: bool,
    pub force: bool,
    pub no_verify: bool,
    pub name: Option<String>,
    pub copy_local: bool,
    pub sglang_args: Vec<String>,
    pub no_wait: bool,
    pub trust_remote_code: bool,
    pub yes: bool,
    pub reset: bool,
}

struct Target {
    model_id: String,
    card_dir: PathBuf,
    entry: Option<RegistryModel>,
    managed: bool,
    source_dir: Option<PathBuf>,
    ternary: Option<TernaryDirs>,
}

pub struct TernaryDirs {
    pub pack: PathBuf,
    pub base: PathBuf,
}

pub fn run(opts: InstallOpts) -> Result<(), String> {
    // Place the files, write the card, register the model, then optionally load it.
    let home = goethe_home();
    ensure_layout(&home).map_err(|e| e.to_string())?;
    let target = if Path::new(&opts.model_id).is_dir() {
        install_local(&home, &opts)?
    } else {
        install_hub(&home, &opts)?
    };
    let model_id = target.model_id.clone();
    register(&home, &opts, target)?;
    if opts.load {
        load_installed(&model_id, &opts)?;
    }
    Ok(())
}

fn install_hub(home: &Path, opts: &InstallOpts) -> Result<Target, String> {
    // Download a Hugging Face model, or a ternary pack plus its base, into models/.
    if opts.model_id.starts_with(['.', '/', '~']) {
        return Err(format!("{}: no such directory", opts.model_id));
    }
    validate_model_id(&opts.model_id)?;
    let entry = fetch(&registry_url())
        .ok()
        .and_then(|r| find_model(&r, &opts.model_id).cloned());
    let card_dir = model_dir(home, &opts.model_id);
    let dl = DownloadOpts {
        revision: entry.as_ref().and_then(|e| e.revision.clone()),
        force: opts.force,
        no_verify: opts.no_verify,
    };
    if let Some(e) = entry.as_ref().filter(|e| e.effective_runtime() == "ternary") {
        crate::runtime::require_ternary_runtime(home)?;
        let pack = e.pack_id.as_deref().ok_or("ternary registry entry has no pack_id")?;
        let base = e.base_model_id.as_deref().ok_or("ternary registry entry has no base_model_id")?;
        validate_model_id(pack)?;
        validate_model_id(base)?;
        let dirs = ternary_dirs(home, pack, base);
        hf::ensure_model(home, pack, &dirs.pack, &dl)?;
        let base_dl = DownloadOpts {
            revision: ternary_base_revision(&dirs.pack, base),
            force: opts.force,
            no_verify: opts.no_verify,
        };
        hf::ensure_model(home, base, &dirs.base, &base_dl)?;
        build_farm(&dirs.base, &card_dir)?;
        return Ok(Target {
            model_id: opts.model_id.clone(),
            card_dir,
            entry: entry.clone(),
            managed: true,
            source_dir: None,
            ternary: Some(dirs),
        });
    }
    hf::ensure_model(home, &opts.model_id, &card_dir, &dl)?;
    Ok(Target {
        model_id: opts.model_id.clone(),
        card_dir,
        entry,
        managed: true,
        source_dir: None,
        ternary: None,
    })
}

pub fn ternary_dirs(home: &Path, pack: &str, base: &str) -> TernaryDirs {
    let pack_dir = model_dir(home, &format!("{pack}/pack"));
    let base_dir = if base == pack {
        model_dir(home, &format!("{base}/base"))
    } else {
        model_dir(home, base)
    };
    TernaryDirs {
        pack: pack_dir,
        base: base_dir,
    }
}

fn ternary_base_revision(pack_dir: &Path, base: &str) -> Option<String> {
    // Pin the base checkpoint to the revision the pack was built against.
    let text = fs::read_to_string(pack_dir.join("ternary_config.json")).ok()?;
    let cfg: Value = serde_json::from_str(&text).ok()?;
    if let Some(named) = cfg.get("base_model").and_then(|v| v.as_str()) {
        if named != base {
            eprintln!("warning: pack was built for base {named}, registry names {base}");
        }
    }
    cfg.get("base_revision").and_then(|v| v.as_str()).map(String::from)
}

fn install_local(home: &Path, opts: &InstallOpts) -> Result<Target, String> {
    // Register a local tree in place (symlink farm) or as a copy; the card always lives under models/<id>.
    let src = fs::canonicalize(&opts.model_id).map_err(|e| format!("{}: {e}", opts.model_id))?;
    if !src.join("config.json").is_file() {
        return Err(format!("{} has no config.json", src.display()));
    }
    if model_facts::weight_files(&src)?.is_empty() {
        return Err(format!("{} has no .safetensors weights", src.display()));
    }
    let dir_name = src
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("{} has no usable directory name; pass --name", src.display()))?;
    let model_id = opts.name.clone().unwrap_or_else(|| format!("local/{dir_name}"));
    validate_model_id(&model_id).map_err(|e| format!("{e}; pass --name org/name"))?;
    let card_dir = model_dir(home, &model_id);
    let models = fs::canonicalize(home.join("models")).map_err(|e| e.to_string())?;
    if src.starts_with(&models) {
        return Err(format!("{} is already inside {}", src.display(), models.display()));
    }
    let keep_copy = prepare_local_target(home, &card_dir, opts)?;
    if opts.copy_local {
        if !keep_copy {
            copy_dir_all(&src, &card_dir)?;
        }
    } else {
        build_farm(&src, &card_dir)?;
    }
    let entry = if model_id.starts_with("local/") {
        None
    } else {
        fetch(&registry_url()).ok().and_then(|r| find_model(&r, &model_id).cloned())
    };
    Ok(Target {
        model_id,
        card_dir,
        entry,
        managed: opts.copy_local,
        source_dir: (!opts.copy_local).then_some(src),
        ternary: None,
    })
}

fn prepare_local_target(home: &Path, card_dir: &Path, opts: &InstallOpts) -> Result<bool, String> {
    // Reuse an existing install of the same kind; switching between copy and in-place needs --force.
    let Ok(meta) = fs::symlink_metadata(card_dir) else {
        return Ok(false);
    };
    if meta.file_type().is_symlink() {
        fs::remove_file(card_dir).map_err(|e| e.to_string())?;
        return Ok(false);
    }
    let farm = is_farm(card_dir);
    if farm && !opts.copy_local {
        return Ok(false);
    }
    if !farm && opts.copy_local && !opts.force {
        return Ok(true);
    }
    if !opts.force {
        return Err(format!(
            "{} exists as a {}; re-run with --force to replace it",
            card_dir.display(),
            if farm { "link to a local folder" } else { "managed copy" }
        ));
    }
    clear_dir_except_card(home, card_dir)?;
    Ok(false)
}

pub fn is_farm(dir: &Path) -> bool {
    // True when dir holds only symlinks and goethe's own card files.
    fs::read_dir(dir)
        .map(|rd| {
            rd.flatten().all(|e| {
                e.file_name().to_string_lossy().starts_with(CARD_FILE)
                    || e.file_type().map(|t| t.is_symlink()).unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

fn clear_dir_except_card(home: &Path, dir: &Path) -> Result<(), String> {
    // Empty an install dir inside models/ without touching link targets.
    for e in fs::read_dir(dir).map_err(|e| e.to_string())?.flatten() {
        if e.file_name() == CARD_FILE {
            continue;
        }
        let p = contained_in_models(home, &e.path())?;
        if e.file_type().map_err(|e| e.to_string())?.is_dir() {
            fs::remove_dir_all(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        } else {
            fs::remove_file(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        }
    }
    Ok(())
}

pub fn build_farm(src: &Path, dir: &Path) -> Result<(), String> {
    // Mirror src as symlinks in dir (refreshing old links) so the engine sees one model directory.
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for e in fs::read_dir(dir).map_err(|e| e.to_string())?.flatten() {
        if e.file_type().map(|t| t.is_symlink()).unwrap_or(false) {
            fs::remove_file(e.path()).map_err(|err| err.to_string())?;
        }
    }
    for e in fs::read_dir(src).map_err(|e| format!("{}: {e}", src.display()))?.flatten() {
        let name = e.file_name();
        if name == CARD_FILE || dir.join(&name).exists() {
            continue;
        }
        symlink(src.join(&name), dir.join(&name)).map_err(|err| format!("{}: {err}", dir.join(&name).display()))?;
    }
    Ok(())
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<(), String> {
    // Copy a tree, following links in the source, and skipping any card it carries.
    fs::create_dir_all(dst).map_err(|e| e.to_string())?;
    for ent in fs::read_dir(src).map_err(|e| e.to_string())? {
        let ent = ent.map_err(|e| e.to_string())?;
        if ent.file_name() == CARD_FILE {
            continue;
        }
        let to = dst.join(ent.file_name());
        if ent.path().is_dir() {
            copy_dir_all(&ent.path(), &to)?;
        } else {
            fs::copy(ent.path(), &to).map_err(|e| format!("{}: {e}", ent.path().display()))?;
        }
    }
    Ok(())
}

fn register(home: &Path, opts: &InstallOpts, t: Target) -> Result<(), String> {
    // Build the card (merged with an existing one unless --reset), write it, and upsert the DB row.
    let dir = &t.card_dir;
    let cfg = model_facts::read_config(dir)?;
    let facts = model_facts::facts_from_install_dir(dir, t.entry.as_ref().map(|e| e.architecture.as_str()))?;
    let existing = if opts.reset { None } else { read_card_table(dir) };
    let old = |k: &str| existing.as_ref().and_then(|c| c.get(k));
    let (cli_args, cli_kv) = sgl::take_flag(&opts.sglang_args, "--kv-cache-dtype")?;
    if let Some(msg) = sgl::reserved_flag_error(&cli_args) {
        return Err(msg);
    }
    let mut notes = Vec::new();
    let registry_args = t.entry.as_ref().map(|e| e.extra_sglang_args.clone()).unwrap_or_default();
    if let Some(msg) = sgl::reserved_flag_error(&registry_args) {
        return Err(format!("registry entry for {}: {msg}", t.model_id));
    }
    let (mut base_args, dir_notes) = sgl::resolve_model_dir_args(&registry_args, dir)?;
    notes.extend(dir_notes);
    if opts.trust_remote_code {
        base_args = sgl::merge_extra_args(&base_args, &["--trust-remote-code".to_string()])?;
        notes.push("--trust-remote-code enabled: SGLang will run code shipped with the model".to_string());
    } else if sgl::wants_remote_code(&cfg) && !sgl::has_flag(&base_args, "--trust-remote-code") {
        notes.push("config.json ships custom code (auto_map); re-run with --trust-remote-code if SGLang needs it".to_string());
    }
    let old_args: Vec<String> = old("extra_sglang_args")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let extra = sgl::merge_extra_args(&sgl::merge_extra_args(&base_args, &old_args)?, &cli_args)
        .map_err(|e| format!("{e} (use --reset to rebuild the card)"))?;
    let kv_dtype = cli_kv
        .or_else(|| old("kv_cache_dtype").and_then(|v| v.as_str()).map(String::from))
        .or_else(|| t.entry.as_ref().and_then(|e| e.kv_cache_dtype.clone()))
        .or_else(|| checkpoint_kv_dtype(&cfg, dir))
        .unwrap_or_else(|| "auto".to_string());
    let kv_elem_bytes = kv_element_bytes(&kv_dtype, &cfg);
    let ple_offload = (sgl::has_flag(&extra, "--ple-offload-embedding") || config_ple_offload(&cfg))
        && !sgl::has_flag(&extra, "--no-ple-offload-embedding");
    let ple_gb = if ple_offload {
        safetensors::tensor_bytes_in_dir(dir, &safetensors::is_ple_embedding).unwrap_or_else(|e| {
            eprintln!("warning: offloaded embedding size unknown: {e}");
            0
        }) as f64
            / GIB
    } else {
        0.0
    };
    let cpu_offload_gb = sgl::flag_value(&extra, "--cpu-offload-gb")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let host_offload_gb = ple_gb + cpu_offload_gb;
    let weights_gb = match (&t.ternary, &t.entry) {
        (Some(_), Some(e)) if e.static_vram_gb > 0.0 => e.static_vram_gb,
        _ => (facts.static_vram_gb - host_offload_gb).max(0.0),
    };
    let state_gb = (facts.swa_kv_elems_per_request * kv_elem_bytes + facts.linear_state_bytes_per_request) as f64 * 4.0 / GIB;
    let static_gb = weights_gb + state_gb;
    let kv_bytes = facts.kv_elems_per_token * kv_elem_bytes;
    let marlin_extra_gb = if safetensors::is_nvfp4_compressed_model(&cfg) {
        match safetensors::marlin_extra_bytes_in_dir(dir) {
            Ok(n) => Some(n as f64 / GIB),
            Err(e) => {
                eprintln!("warning: marlin_extra_gb scan skipped: {e}");
                None
            }
        }
    } else {
        None
    };
    let pick = |cli: Option<u64>, key: &str, reg: Option<u64>, default: u64| {
        cli.or_else(|| old(key).and_then(|v| v.as_integer()).map(|i| i as u64))
            .or(reg)
            .unwrap_or(default)
    };
    let entry = t.entry.as_ref();
    let ctx = pick(opts.ctx, "context_length", entry.and_then(|e| e.default_ctx), facts.max_ctx.min(131072));
    let max_conc = pick(opts.max_conc, "max_running_requests", entry.and_then(|e| e.default_max_conc), 8);
    let tp = pick(opts.tp, "tp_size", entry.and_then(|e| e.default_tp), 0);
    let ep = pick(None, "ep_size", None, 1);
    let gpus = opts.gpus.clone().unwrap_or_else(|| {
        old("preferred_gpus")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_integer().map(|i| i as u32)).collect())
            .unwrap_or_default()
    });
    let card = GoetheCard {
        model_id: t.model_id.clone(),
        context_length: ctx,
        max_running_requests: max_conc,
        tp_size: tp,
        ep_size: ep,
        kv_cache_dtype: kv_dtype.clone(),
        preferred_gpus: gpus,
        extra_sglang_args: extra.clone(),
        runtime: entry.and_then(|e| e.runtime.clone()),
        pack_id: entry.and_then(|e| e.pack_id.clone()),
        base_model_id: entry.and_then(|e| e.base_model_id.clone()),
        env: merged_env(old("env"), entry, t.ternary.as_ref(), home),
        pythonpath: entry.and_then(|e| e.pythonpath.clone()),
        marlin_extra_gb,
        host_offload_gb: (host_offload_gb > 0.0).then_some(host_offload_gb),
        source_dir: t.source_dir.as_ref().map(|p| p.display().to_string()),
    };
    write_card_table(dir, &card_table(&card, existing.as_ref())?)?;
    let rel = format!("models/{}", crate::paths::sanitize_model_id(&t.model_id));
    let con = db::open(home).map_err(|e| e.to_string())?;
    con.execute(
        "INSERT OR REPLACE INTO installed_models
        (model_id, local_path, architecture, static_vram_gb, kv_bytes_per_token,
         default_ctx, default_max_conc, default_tp, kv_elems_per_token, max_ctx,
         num_attention_heads, num_kv_heads, attention, managed, installed_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, datetime('now'))",
        rusqlite::params![
            t.model_id,
            rel,
            facts.architecture,
            static_gb,
            kv_bytes as i64,
            ctx as i64,
            max_conc as i64,
            tp as i64,
            facts.kv_elems_per_token as i64,
            facts.max_ctx as i64,
            facts.num_attention_heads as i64,
            facts.num_kv_heads as i64,
            facts.attention,
            t.managed as i64,
        ],
    )
    .map_err(|e| format!("state.db: {e}"))?;
    println!("installed {}", t.model_id);
    println!("  path      {}", dir.display());
    match &t.source_dir {
        Some(src) => println!("  files     kept in {} (goethe remove only unregisters)", src.display()),
        None => println!("  files     managed by goethe (goethe remove deletes them)"),
    }
    let state_note = if state_gb > 0.0 {
        format!(" + {state_gb:.1} GB per-request state for 4 requests")
    } else {
        String::new()
    };
    println!("  vram      {weights_gb:.1} GB weights{state_note}");
    if host_offload_gb > 0.0 {
        println!("  host ram  {host_offload_gb:.1} GB offloaded weights in pinned memory");
    }
    println!(
        "  kv cache  {:.1} KB per token ({kv_dtype}, {} attention)",
        kv_bytes as f64 / 1024.0,
        facts.attention
    );
    if let Some(q) = sgl::quantization_summary(dir, &cfg) {
        println!("  quant     {q}, detected by SGLang");
    }
    let tp_s = if tp == 0 { "auto".to_string() } else { tp.to_string() };
    println!("  defaults  ctx {ctx}, max-conc {max_conc}, tp {tp_s}");
    if !extra.is_empty() {
        println!("  sglang    {}", extra.join(" "));
    }
    for n in notes {
        println!("  note      {n}");
    }
    Ok(())
}

fn merged_env(
    old: Option<&toml::Value>,
    entry: Option<&RegistryModel>,
    ternary: Option<&TernaryDirs>,
    home: &Path,
) -> Option<toml::Value> {
    // User env from the old card, then registry env, then the ternary dirs install actually used.
    let mut table = old.and_then(|v| v.as_table()).cloned().unwrap_or_default();
    if let Some(reg) = entry.and_then(|e| e.env.as_ref()).and_then(|v| v.as_table()) {
        for (k, v) in reg {
            table.insert(k.clone(), v.clone());
        }
    }
    if let Some(d) = ternary {
        for (k, p) in [("TERNARY_PACK_DIR", &d.pack), ("TERNARY_MODEL_DIR", &d.base)] {
            let rel = p.strip_prefix(home).map(|r| format!("$GOETHE_HOME/{}", r.display()));
            table.insert(k.to_string(), toml::Value::String(rel.unwrap_or_else(|_| p.display().to_string())));
        }
    }
    (!table.is_empty()).then_some(toml::Value::Table(table))
}

fn card_table(card: &GoetheCard, existing: Option<&toml::Table>) -> Result<toml::Table, String> {
    // Serialize the card and keep keys the user added by hand.
    let mut table = match toml::Value::try_from(card).map_err(|e| e.to_string())? {
        toml::Value::Table(t) => t,
        _ => return Err("card did not serialize to a table".to_string()),
    };
    if let Some(old) = existing {
        for (k, v) in old {
            if !CARD_KEYS.contains(&k.as_str()) {
                table.insert(k.clone(), v.clone());
            }
        }
    }
    Ok(table)
}

fn read_card_table(dir: &Path) -> Option<toml::Table> {
    let text = fs::read_to_string(dir.join(CARD_FILE)).ok()?;
    toml::from_str(&text).ok()
}

pub fn write_card_table(dir: &Path, table: &toml::Table) -> Result<(), String> {
    // Write goethe_card.toml through a temp file and rename so readers never see a partial card.
    let text = toml::to_string_pretty(table).map_err(|e| e.to_string())?;
    let tmp = dir.join(format!("{CARD_FILE}.tmp"));
    fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    fs::rename(&tmp, dir.join(CARD_FILE)).map_err(|e| format!("{}: {e}", dir.join(CARD_FILE).display()))
}

pub fn read_card(dir: &Path) -> Result<GoetheCard, String> {
    let text = fs::read_to_string(dir.join(CARD_FILE)).map_err(|e| e.to_string())?;
    toml::from_str(&text).map_err(|e| e.to_string())
}

fn config_ple_offload(cfg: &Value) -> bool {
    model_facts::resolve_language_config(cfg)
        .ok()
        .and_then(|lm| lm.get("ple_offload_embedding"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
}

pub fn checkpoint_kv_dtype(cfg: &Value, dir: &Path) -> Option<String> {
    // An FP8 KV cache the checkpoint was calibrated for (compressed-tensors or modelopt).
    let fp8 = |v: &Value| {
        let scheme = v.get("kv_cache_scheme");
        let ct = scheme.is_some_and(|s| {
            s.get("num_bits").and_then(|n| n.as_u64()) == Some(8) && s.get("type").and_then(|t| t.as_str()) == Some("float")
        });
        ct || v.get("kv_cache_quant_algo").and_then(|a| a.as_str()) == Some("FP8")
    };
    if cfg.get("quantization_config").is_some_and(fp8) {
        return Some("fp8_e4m3".to_string());
    }
    let text = fs::read_to_string(dir.join("hf_quant_config.json")).ok()?;
    let q: Value = serde_json::from_str(&text).ok()?;
    q.get("quantization").filter(|v| fp8(v)).map(|_| "fp8_e4m3".to_string())
}

pub fn kv_element_bytes(kv_cache_dtype: &str, cfg: &Value) -> u64 {
    // Bytes per KV element: the card dtype, or the model dtype for auto.
    let k = kv_cache_dtype.to_ascii_lowercase();
    if k.contains("fp8") {
        return 1;
    }
    if k != "auto" {
        return 2;
    }
    let lm = model_facts::resolve_language_config(cfg).unwrap_or(cfg);
    let dtype = lm
        .get("torch_dtype")
        .or_else(|| lm.get("dtype"))
        .or_else(|| cfg.get("torch_dtype"))
        .or_else(|| cfg.get("dtype"))
        .and_then(|v| v.as_str())
        .unwrap_or("bfloat16");
    if dtype.contains("32") {
        4
    } else {
        2
    }
}

fn load_installed(model_id: &str, opts: &InstallOpts) -> Result<(), String> {
    let gpus_csv = opts
        .gpus
        .as_ref()
        .map(|v| v.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(","));
    let params = json!({
        "model_id": model_id,
        "gpus": gpus_csv,
        "ctx": opts.ctx,
        "max_conc": opts.max_conc,
        "tp": opts.tp,
        "force": false,
        "yes": opts.yes,
        "extra_sglang_args": Vec::<String>::new(),
        "async_finish": true,
    });
    let stdout = send_load(params)?;
    if !stdout.is_empty() {
        print!("{stdout}");
    }
    crate::load_wait::wait_after_load_stdout(&stdout, opts.no_wait)
}

pub fn send_load(mut params: Value) -> Result<String, String> {
    // Load RPC; on a foreign-process notice (code 3) ask on a terminal and resend with yes=true, else exit 3.
    params["is_tty"] = json!(false);
    let call = |p: &Value| crate::rpc::call("load", p.clone()).map_err(crate::rpc::map_daemon_err);
    let mut reply = call(&params)?;
    if reply.code == 3 {
        print_reply_text(&reply);
        if !(std::io::stdin().is_terminal() && std::io::stderr().is_terminal()) {
            eprintln!("not loading next to processes goethe did not start: re-run with --yes (no terminal to ask on)");
            std::process::exit(3);
        }
        eprint!("Load alongside these processes? [y/N] ");
        std::io::stderr().flush().ok();
        let mut answer = String::new();
        std::io::stdin().read_line(&mut answer).map_err(|e| e.to_string())?;
        if !matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            return Err("load aborted".to_string());
        }
        params["yes"] = json!(true);
        reply = call(&params)?;
    }
    if reply.code != 0 {
        print_reply_text(&reply);
        std::process::exit(reply.code);
    }
    Ok(reply.stdout)
}

fn print_reply_text(reply: &crate::rpc::RpcReply) {
    // An error reply may carry the notice in message, error, or both; show each once on stderr.
    let (msg, err) = (reply.stdout.trim_end(), reply.stderr.trim_end());
    if !msg.is_empty() {
        eprintln!("{msg}");
    }
    if !err.is_empty() && err != msg {
        eprintln!("{err}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_bytes_follow_card_dtype() {
        let cfg = json!({"text_config": {"dtype": "bfloat16", "num_hidden_layers": 2, "num_attention_heads": 2}});
        assert_eq!(kv_element_bytes("fp8_e4m3", &cfg), 1);
        assert_eq!(kv_element_bytes("auto", &cfg), 2);
        assert_eq!(kv_element_bytes("auto", &json!({"torch_dtype": "float32", "num_hidden_layers": 1})), 4);
    }

    #[test]
    fn checkpoint_fp8_kv_is_detected() {
        let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
        let dir = fixtures.join("qwen38-27b-nvfp4-ninfer");
        let cfg = model_facts::read_config(&dir).unwrap();
        assert_eq!(checkpoint_kv_dtype(&cfg, &dir).as_deref(), Some("fp8_e4m3"));
        let fnx = fixtures.join("qwen38-flash-next-nvfp4-latest");
        assert_eq!(checkpoint_kv_dtype(&model_facts::read_config(&fnx).unwrap(), &fnx), None);
    }

    #[test]
    fn farm_links_every_entry_but_the_card() {
        let src = tempfile::tempdir().unwrap();
        std::fs::write(src.path().join("config.json"), "{}").unwrap();
        std::fs::write(src.path().join(CARD_FILE), "stale").unwrap();
        std::fs::create_dir(src.path().join("spec")).unwrap();
        let home = tempfile::tempdir().unwrap();
        let dir = home.path().join("models/local__m");
        build_farm(src.path(), &dir).unwrap();
        assert!(fs::symlink_metadata(dir.join("config.json")).unwrap().file_type().is_symlink());
        assert!(dir.join("spec").is_dir());
        assert!(!dir.join(CARD_FILE).exists());
        std::fs::write(dir.join(CARD_FILE), "model_id = \"local/m\"\n").unwrap();
        assert!(is_farm(&dir));
        build_farm(src.path(), &dir).unwrap();
        assert!(dir.join(CARD_FILE).is_file());
        assert_eq!(fs::read_to_string(src.path().join(CARD_FILE)).unwrap(), "stale");
    }

    #[test]
    fn card_merge_keeps_unknown_keys() {
        let card = GoetheCard {
            model_id: "m".into(),
            context_length: 1,
            max_running_requests: 1,
            tp_size: 0,
            ep_size: 1,
            kv_cache_dtype: "auto".into(),
            preferred_gpus: vec![],
            extra_sglang_args: vec![],
            runtime: None,
            pack_id: None,
            base_model_id: None,
            env: None,
            pythonpath: None,
            marlin_extra_gb: None,
            host_offload_gb: None,
            source_dir: None,
        };
        let old: toml::Table = toml::from_str("note = \"mine\"\nmarlin_extra_gb = 3.0\n").unwrap();
        let t = card_table(&card, Some(&old)).unwrap();
        assert_eq!(t.get("note").and_then(|v| v.as_str()), Some("mine"));
        assert!(t.get("marlin_extra_gb").is_none());
    }
}
