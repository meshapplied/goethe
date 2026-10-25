// Registry fetch, validation, and entry inference.

use crate::hf::{self, fetch_config_json};
use crate::model_facts;
use crate::paths::goethe_home;
use crate::sglang_infer;
use crate::sglang_validate;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Registry {
    pub version: u32,
    pub models: Vec<RegistryModel>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RegistryModel {
    pub id: String,
    #[serde(default = "default_arch")]
    pub architecture: String,
    #[serde(default)]
    pub static_vram_gb: f64,
    #[serde(default)]
    pub kv_bytes_per_token: Option<u64>,
    #[serde(default)]
    pub kv_elems_per_token: Option<u64>,
    #[serde(default)]
    pub min_total_vram_gb: Option<f64>,
    #[serde(default)]
    pub supported_sm: Vec<String>,
    #[serde(default)]
    pub default_ctx: Option<u64>,
    #[serde(default)]
    pub max_ctx: Option<u64>,
    #[serde(default)]
    pub default_max_conc: Option<u64>,
    #[serde(default)]
    pub default_tp: Option<u64>,
    #[serde(default)]
    pub num_attention_heads: Option<u64>,
    #[serde(default)]
    pub num_kv_heads: Option<u64>,
    #[serde(default)]
    pub attention: Option<String>,
    #[serde(default)]
    pub kv_cache_dtype: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<crate::memory_spec::MemorySpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_sglang_args: Vec<String>,
    #[serde(default)]
    pub runtime: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pack_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub env: Option<toml::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pythonpath: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gated: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manual_download: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub download_gb: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub released: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
}

const TERNARY_PROFILES: &[&str] = &[
    "flash_next",
    "flash-next",
    "qwen4_exp",
    "qwen38",
    "qwen",
    "deepseek_v41",
    "deepseek",
    "ds",
    "bonsai",
    "bonsai-27b",
    "ternary-bonsai",
];

fn default_arch() -> String {
    "dense".to_string()
}

impl RegistryModel {
    pub fn effective_runtime(&self) -> &str {
        self.runtime.as_deref().unwrap_or("sglang")
    }

    pub fn effective_kv_elems(&self) -> Option<u64> {
        self.kv_elems_per_token.or(self.kv_bytes_per_token)
    }

    pub fn effective_max_ctx(&self) -> Option<u64> {
        self.max_ctx.or(self.default_ctx)
    }

    pub fn manual_download_reason(&self) -> Option<&str> {
        // Entries goethe cannot download: weights published elsewhere, or gated on Hugging Face.
        if self.manual_download.unwrap_or(false) {
            Some("weights are not on Hugging Face; place them in GOETHE_HOME/models and run goethe scan")
        } else if self.gated.unwrap_or(false) {
            Some("gated on Hugging Face; download it with your own account and run goethe scan")
        } else {
            None
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct RegistryCache {
    etag: Option<String>,
    body: String,
}

pub struct ValidateOpts {
    pub online: bool,
    pub allow_individuals: bool,
    pub check_sglang: bool,
}

pub const DEFAULT_REGISTRY_URL: &str = "https://pkg.meshapplied.com/v1/registry.json";

const BUNDLED_REGISTRY: &str = include_str!("../../../cdn/v1/registry.json");

pub fn registry_url() -> String {
    crate::paths::setting("GOETHE_REGISTRY_URL").unwrap_or_else(|| DEFAULT_REGISTRY_URL.to_string())
}

pub fn bundled() -> Result<Registry, String> {
    parse_registry(BUNDLED_REGISTRY)
}

pub fn fetch(url: &str) -> Result<Registry, String> {
    // file:// or http(s) with a cache; the default URL falls back to the catalog built into this binary.
    if let Some(rest) = url.strip_prefix("file://") {
        let path = Path::new(rest);
        let text = fs_read(path)?;
        return parse_registry(&text);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return fetch_http_cached(url).or_else(|e| if url == DEFAULT_REGISTRY_URL { bundled() } else { Err(e) });
    }
    Err(format!("unsupported registry URL scheme: {url}"))
}

const MISS_TTL_SECS: u64 = 300;

fn recent_miss(miss: &Path, url: &str) -> bool {
    // A failed fetch of this same URL within the last few minutes, so commands skip the slow retry.
    let fresh = std::fs::metadata(miss)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_some_and(|age| age.as_secs() < MISS_TTL_SECS);
    fresh && std::fs::read_to_string(miss).is_ok_and(|u| u.trim() == url)
}

fn fetch_http_cached(url: &str) -> Result<Registry, String> {
    // Fetch with ETag revalidation; an unreachable or unparsable registry is remembered for a few minutes.
    let home = goethe_home();
    let miss = home.join("registry.miss");
    if recent_miss(&miss, url) {
        return Err(format!("registry unreachable at {url} (last try under {} min ago)", MISS_TTL_SECS / 60));
    }
    let result = fetch_http_uncached(&home, url);
    if result.is_err() {
        let _ = std::fs::write(&miss, url);
    } else {
        let _ = std::fs::remove_file(&miss);
    }
    result
}

fn fetch_http_uncached(home: &Path, url: &str) -> Result<Registry, String> {
    // One request: a 200 refreshes the cache, a 304 reuses it, and offline mode falls back to it.
    let cache_path = home.join("registry.cache.json");
    let offline = hf::offline_mode();
    let mut req = crate::net::agent().get(url).timeout(std::time::Duration::from_secs(10));
    if let Ok(text) = std::fs::read_to_string(&cache_path) {
        if let Ok(c) = serde_json::from_str::<RegistryCache>(&text) {
            if let Some(etag) = &c.etag {
                req = req.set("If-None-Match", etag);
            }
        }
    }
    let cached = || -> Result<Registry, String> {
        let text = std::fs::read_to_string(&cache_path).map_err(|e| e.to_string())?;
        let c: RegistryCache = serde_json::from_str(&text).map_err(|e| format!("registry cache: {e}"))?;
        parse_registry(&c.body)
    };
    match req.call() {
        Ok(resp) if resp.status() == 304 => cached(),
        Ok(resp) => {
            let etag = resp.header("etag").map(String::from);
            let body = resp.into_string().map_err(|e| format!("registry body: {e}"))?;
            if let Ok(reg) = parse_registry(&body) {
                let cache = RegistryCache {
                    etag,
                    body: body.clone(),
                };
                let _ = std::fs::write(
                    &cache_path,
                    serde_json::to_string(&cache).unwrap_or_default(),
                );
                return Ok(reg);
            }
            parse_registry(&body)
        }
        Err(e) if offline => {
            if !cache_path.is_file() {
                return Err(format!("registry unreachable ({e}) and no registry.cache.json"));
            }
            eprintln!("warning: registry fetch failed ({e}); using cached registry");
            cached()
        }
        Err(e) => Err(format!("registry unreachable at {url}: {e}")),
    }
}

pub fn find_model<'a>(reg: &'a Registry, model_id: &str) -> Option<&'a RegistryModel> {
    reg.models.iter().find(|m| m.id == model_id)
}

pub fn parse_registry(text: &str) -> Result<Registry, String> {
    serde_json::from_str(text).map_err(|e| format!("registry parse failed: {e}"))
}

fn fs_read(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("registry file read failed: {e}"))
}

pub fn run_validate(source: &str, opts: &ValidateOpts) -> Result<(), String> {
    let text = load_registry_source(source)?;
    warn_unknown_fields(&text);
    let reg = parse_registry(&text)?;
    let mut errors = Vec::new();
    let mut ids = std::collections::HashSet::new();
    for m in &reg.models {
        if !ids.insert(m.id.clone()) {
            errors.push(format!("duplicate id {}", m.id));
        }
        validate_entry(m, &mut errors);
    }
    if opts.online {
        for m in &reg.models {
            online_checks(m, opts, &mut errors);
        }
    }
    if opts.check_sglang {
        for e in sglang_validate::validate_registry_sglang(&reg, opts.online) {
            if e.starts_with("note:") || e.starts_with("warning:") {
                eprintln!("{e}");
            } else {
                errors.push(e);
            }
        }
    }
    for e in &errors {
        eprintln!("error: {e}");
    }
    if !errors.is_empty() {
        std::process::exit(1);
    }
    let n = reg.models.len();
    let extra = [(opts.online, "online"), (opts.check_sglang, "sglang")]
        .iter()
        .filter(|(on, _)| *on)
        .map(|(_, what)| *what)
        .collect::<Vec<_>>();
    let checks = if extra.is_empty() { String::new() } else { format!(" ({} checks passed)", extra.join(" and ")) };
    println!("{source}: {n} entr{} valid{checks}", if n == 1 { "y" } else { "ies" });
    Ok(())
}

fn validate_entry(m: &RegistryModel, errors: &mut Vec<String>) {
    if m.static_vram_gb <= 0.0 {
        errors.push(format!("{}: static_vram_gb must be positive", m.id));
    }
    if m.effective_kv_elems().is_none() {
        errors.push(format!("{}: kv_elems_per_token (or kv_bytes_per_token) required", m.id));
    }
    if m.effective_max_ctx().is_none() {
        errors.push(format!("{}: max_ctx (or default_ctx) required", m.id));
    }
    let legacy = m.kv_bytes_per_token.is_some() && m.default_ctx.is_some();
    if m.num_attention_heads.is_none() {
        if m.effective_runtime() == "ternary" || legacy {
            eprintln!("warning: {}: missing num_attention_heads (legacy entry)", m.id);
        } else {
            errors.push(format!("{}: num_attention_heads required", m.id));
        }
    }
    if m.num_kv_heads.is_none() {
        if m.effective_runtime() == "ternary" || legacy {
            eprintln!("warning: {}: missing num_kv_heads (legacy entry)", m.id);
        } else {
            errors.push(format!("{}: num_kv_heads required", m.id));
        }
    }
    if m.architecture != "dense" && m.architecture != "moe" {
        errors.push(format!("{}: architecture must be dense or moe", m.id));
    }
    let rt = m.effective_runtime();
    if rt != "sglang" && rt != "ternary" {
        errors.push(format!("{}: runtime must be sglang or ternary", m.id));
    }
    if rt == "ternary" {
        validate_ternary(m, errors);
    }
    if let Some(msg) = sglang_infer::reserved_flag_error(&m.extra_sglang_args) {
        errors.push(format!("{}: {msg}", m.id));
    }
    if let Err(e) = sglang_infer::parse_groups(&m.extra_sglang_args).and_then(|g| sglang_infer::check_arity(&g)) {
        errors.push(format!("{}: {e}", m.id));
    }
    if sglang_infer::has_flag(&m.extra_sglang_args, "--kv-cache-dtype") {
        errors.push(format!("{}: set kv_cache_dtype instead of passing --kv-cache-dtype", m.id));
    }
    if m.supported_sm.is_empty() {
        errors.push(format!("{}: supported_sm required", m.id));
    }
    if let Some(mem) = &m.memory {
        if mem.kv_bytes_per_token == 0 {
            errors.push(format!("{}: memory.kv_bytes_per_token must be positive", m.id));
        }
        if (mem.state_bytes_per_slot > 0) != (mem.state_slots_per_request > 0) {
            errors.push(format!("{}: memory state needs both state_bytes_per_slot and state_slots_per_request", m.id));
        }
        if mem.swa_bytes_per_token > 0 && mem.swa_window == 0 {
            errors.push(format!("{}: memory.swa_window required with swa_bytes_per_token", m.id));
        }
    }
    for sm in &m.supported_sm {
        if !sm.contains('.') {
            eprintln!("warning: {}: supported_sm entry {sm} should look like 8.6 or 12.0", m.id);
        }
    }
}

fn validate_ternary(m: &RegistryModel, errors: &mut Vec<String>) {
    // Ternary entries must name both checkpoints, a known profile, and the dirs install creates.
    let (Some(pack), Some(base)) = (m.pack_id.as_deref(), m.base_model_id.as_deref()) else {
        errors.push(format!("{}: ternary requires pack_id and base_model_id", m.id));
        return;
    };
    let dirs = crate::install::ternary_dirs(Path::new("$GOETHE_HOME"), pack, base);
    let env = m.env.as_ref().and_then(|v| v.as_table());
    let get = |k: &str| env.and_then(|t| t.get(k)).and_then(|v| v.as_str()).unwrap_or("");
    for (key, want) in [("TERNARY_PACK_DIR", &dirs.pack), ("TERNARY_MODEL_DIR", &dirs.base)] {
        let want = want.display().to_string();
        if get(key) != want {
            errors.push(format!("{}: env {key} must be {want} (got {:?})", m.id, get(key)));
        }
    }
    if !TERNARY_PROFILES.contains(&get("TERNARY_PROFILE")) {
        errors.push(format!(
            "{}: env TERNARY_PROFILE {:?} is not one of {}",
            m.id,
            get("TERNARY_PROFILE"),
            TERNARY_PROFILES.join(", ")
        ));
    }
}

fn online_checks(m: &RegistryModel, opts: &ValidateOpts, errors: &mut Vec<String>) {
    if hf::offline_mode() {
        return;
    }
    if let Some(owner) = hf::hf_owner_org(&m.id) {
        match hf::is_hf_organization(&owner) {
            Ok(true) => {}
            Ok(false) if opts.allow_individuals => {
                eprintln!("warning: {} owner {owner} is not a Hugging Face organization", m.id);
            }
            Ok(false) => {
                errors.push(format!(
                    "{}: owner {owner} is not an organization (use --allow-individuals to warn only)",
                    m.id
                ));
            }
            Err(e) => errors.push(format!("{}: org check failed: {e}", m.id)),
        }
    }
    let rev = m.revision.as_deref().unwrap_or("main");
    let manual = m.manual_download.unwrap_or(false);
    match hf::preflight_model_opts(&m.id, Some(rev), true) {
        Err(e) if manual => eprintln!("note: {}: manual download, Hugging Face says: {e}", m.id),
        Err(e) => errors.push(format!("{}: {e}", m.id)),
        Ok(p) => {
            if manual {
                eprintln!("note: {}: manual download; its weights are not fetched from Hugging Face", m.id);
            } else if !p.siblings.is_empty() && !hf::has_weight_files(&p.siblings) {
                errors.push(format!(
                    "{}: repository has no model weight files (set \"manual_download\": true if they are published elsewhere)",
                    m.id
                ));
            }
            let entry_gated = m.gated.unwrap_or(false);
            if p.gated && entry_gated {
                eprintln!("note: {}: gated; registry checks use no Hugging Face token, so its files are not checked", m.id);
            } else if p.gated && !entry_gated {
                errors.push(format!(
                    "{}: Hugging Face reports gated but registry entry gated is not true",
                    m.id
                ));
            } else if entry_gated && !p.gated {
                eprintln!(
                    "warning: {}: registry gated=true but Hugging Face API reports not gated",
                    m.id
                );
            }
            if let Some(want) = m.revision.as_deref() {
                if let Some(sha) = p.sha.as_deref() {
                    if want != sha && want != "main" {
                        eprintln!(
                            "warning: {}: revision {want} differs from HF default sha {sha}",
                            m.id
                        );
                    }
                }
            }
        }
    }
}

fn warn_unknown_fields(text: &str) {
    let Ok(raw) = serde_json::from_str::<serde_json::Value>(text) else {
        return;
    };
    let known: &[&str] = &[
        "version", "models", "id", "architecture", "static_vram_gb", "kv_bytes_per_token",
        "kv_elems_per_token", "min_total_vram_gb", "supported_sm", "default_ctx", "max_ctx",
        "default_max_conc", "default_tp", "num_attention_heads", "num_kv_heads", "attention",
        "kv_cache_dtype", "memory", "extra_sglang_args", "runtime", "pack_id", "base_model_id", "env",
        "pythonpath", "revision", "gated", "manual_download", "download_gb", "description", "family",
        "license", "released", "modalities",
    ];
    let Some(models) = raw.get("models").and_then(|v| v.as_array()) else {
        return;
    };
    for m in models {
        let Some(obj) = m.as_object() else {
            continue;
        };
        for key in obj.keys() {
            if !known.contains(&key.as_str()) {
                eprintln!("warning: unknown registry field {key}");
            }
        }
    }
}

fn load_registry_source(source: &str) -> Result<String, String> {
    if source.starts_with("http://") || source.starts_with("https://") {
        let resp = crate::net::agent()
            .get(source)
            .timeout(std::time::Duration::from_secs(10))
            .call()
            .map_err(|e| e.to_string())?;
        return resp.into_string().map_err(|e| e.to_string());
    }
    if source.starts_with("file://") {
        return fs_read(Path::new(source.trim_start_matches("file://")));
    }
    fs_read(Path::new(source))
}

pub struct EntryOpts {
    pub strict: bool,
}

pub fn run_entries(
    repo_ids: &[String],
    runtime: Option<&str>,
    revision: Option<&str>,
    opts: &EntryOpts,
) -> Result<(), String> {
    if repo_ids.is_empty() {
        return Err("no repo ids (pass ids or --ids-file)".to_string());
    }
    let mut models = Vec::new();
    let mut failed = 0usize;
    for repo_id in repo_ids {
        match infer_one_entry(repo_id, runtime, revision) {
            Ok((entry, warnings)) => {
                for w in warnings {
                    eprintln!("warning: {w}");
                }
                models.push(entry);
            }
            Err(e) => {
                failed += 1;
                eprintln!("warning: skipped {repo_id}: {e}");
            }
        }
    }
    if models.is_empty() {
        if opts.strict || failed > 0 {
            std::process::exit(1);
        }
        return Err("no entries produced".to_string());
    }
    if opts.strict && failed > 0 {
        std::process::exit(1);
    }
    let out = Registry {
        version: 1,
        models,
    };
    println!("{}", serde_json::to_string_pretty(&out).map_err(|e| e.to_string())?);
    Ok(())
}

pub struct AddOpts {
    pub repo_ids: Vec<String>,
    pub file: String,
    pub overlay_path: Option<String>,
    pub update: bool,
    pub dry_run: bool,
    pub strict: bool,
}

pub fn run_add(opts: &AddOpts) -> Result<(), String> {
    if opts.repo_ids.is_empty() {
        return Err("no repo ids".to_string());
    }
    let path = Path::new(&opts.file);
    let text = fs_read(path)?;
    let mut reg = parse_registry(&text)?;
    let overlay = load_overlay(opts.overlay_path.as_deref())?;
    let mut added = 0usize;
    let mut updated = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;
    for repo_id in &opts.repo_ids {
        match infer_one_entry(repo_id, None, None) {
            Ok((mut entry, warnings)) => {
                for w in warnings {
                    eprintln!("warning: {w}");
                }
                if let Some(o) = overlay.get(&entry.id) {
                    merge_overlay(&mut entry, o).map_err(|e| format!("{}: {e}", entry.id))?;
                }
                if let Some(idx) = reg.models.iter().position(|m| m.id == entry.id) {
                    if !opts.update {
                        skipped += 1;
                        continue;
                    }
                    merge_measured(&mut reg.models[idx], &entry).map_err(|e| format!("{}: {e}", entry.id))?;
                    updated += 1;
                } else {
                    reg.models.push(entry);
                    added += 1;
                }
            }
            Err(e) => {
                failed += 1;
                eprintln!("warning: failed {repo_id}: {e}");
            }
        }
    }
    let mut errors = Vec::new();
    for m in &reg.models {
        validate_entry(m, &mut errors);
    }
    for e in sglang_validate::validate_registry_sglang(&reg, false) {
        if e.starts_with("note:") || e.starts_with("warning:") {
            eprintln!("{e}");
        } else {
            errors.push(e);
        }
    }
    if !errors.is_empty() {
        for e in &errors {
            eprintln!("error: {e}");
        }
        return Err("registry add refused: validation failed".to_string());
    }
    if opts.strict && failed > 0 {
        std::process::exit(1);
    }
    if !opts.dry_run && (added > 0 || updated > 0) {
        write_registry_atomic(path, &reg)?;
    }
    if opts.dry_run {
        println!("dry-run: no file written");
    }
    println!("summary: added {added} updated {updated} skipped {skipped} failed {failed}");
    if added + updated == 0 && failed > 0 && opts.strict {
        std::process::exit(1);
    }
    Ok(())
}

fn write_registry_atomic(path: &Path, reg: &Registry) -> Result<(), String> {
    let text = serde_json::to_string_pretty(reg).map_err(|e| e.to_string())?;
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

fn load_overlay(path: Option<&str>) -> Result<std::collections::HashMap<String, RegistryModel>, String> {
    let Some(p) = path else {
        return Ok(std::collections::HashMap::new());
    };
    let reg = parse_registry(&fs_read(Path::new(p))?)?;
    Ok(reg.models.into_iter().map(|m| (m.id.clone(), m)).collect())
}

fn merge_overlay(dst: &mut RegistryModel, src: &RegistryModel) -> Result<(), String> {
    if src.description.is_some() {
        dst.description = src.description.clone();
    }
    if src.family.is_some() {
        dst.family = src.family.clone();
    }
    if src.license.is_some() {
        dst.license = src.license.clone();
    }
    if src.released.is_some() {
        dst.released = src.released.clone();
    }
    if src.modalities.is_some() {
        dst.modalities = src.modalities.clone();
    }
    if !src.supported_sm.is_empty() {
        dst.supported_sm = src.supported_sm.clone();
    }
    if src.manual_download.is_some() {
        dst.manual_download = src.manual_download;
    }
    dst.extra_sglang_args = sglang_infer::merge_extra_args(&dst.extra_sglang_args, &src.extra_sglang_args)?;
    Ok(())
}

fn merge_measured(dst: &mut RegistryModel, src: &RegistryModel) -> Result<(), String> {
    dst.static_vram_gb = src.static_vram_gb;
    dst.memory = src.memory.clone();
    dst.kv_elems_per_token = src.kv_elems_per_token;
    dst.max_ctx = src.max_ctx;
    dst.num_attention_heads = src.num_attention_heads;
    dst.num_kv_heads = src.num_kv_heads;
    dst.attention = src.attention.clone();
    dst.architecture = src.architecture.clone();
    dst.revision = src.revision.clone();
    dst.gated = src.gated;
    dst.download_gb = src.download_gb;
    dst.extra_sglang_args = sglang_infer::merge_extra_args(&src.extra_sglang_args, &dst.extra_sglang_args)?;
    Ok(())
}

pub fn infer_one_entry(
    repo_id: &str,
    runtime: Option<&str>,
    revision: Option<&str>,
) -> Result<(RegistryModel, Vec<String>), String> {
    let rev = revision.unwrap_or("main");
    let pre = hf::preflight_model_opts(repo_id, Some(rev), true)?;
    if pre.gated {
        return Err("gated on Hugging Face and registry inference uses no token: write this entry by hand (set \"gated\": true)".to_string());
    }
    let sha = pre
        .sha
        .clone()
        .or_else(|| revision.map(String::from))
        .unwrap_or_else(|| rev.to_string());
    let cfg = fetch_config_json(repo_id, &sha)?;
    let mut warnings = Vec::new();
    if let Some(owner) = hf::hf_owner_org(repo_id) {
        if hf::is_hf_organization(&owner) == Ok(false) {
            warnings.push(format!("owner {owner} is not a Hugging Face organization"));
        }
    }
    let facts = model_facts::facts_from_config(&cfg, 0.0, None, repo_id)?;
    if !hf::has_weight_files(&pre.siblings) {
        return Err("repository has no model weight files".to_string());
    }
    let weight_bytes = hf::weight_bytes_from_siblings(repo_id, &sha, &pre.siblings)?;
    let memory = crate::memory_spec::from_config(&cfg, None)?;
    let download_gb = hf::download_bytes(&pre.siblings) as f64 / 1e9;
    let mut extra = Vec::new();
    if sglang_infer::wants_remote_code(&cfg) {
        extra.push("--trust-remote-code".to_string());
        warnings.push(format!("{repo_id}: config.json ships custom code (auto_map); added --trust-remote-code, review it"));
    }
    let entry = RegistryModel {
        id: repo_id.to_string(),
        architecture: facts.architecture,
        static_vram_gb: weight_bytes as f64 / model_facts::GIB,
        kv_bytes_per_token: None,
        kv_elems_per_token: Some(facts.kv_elems_per_token),
        min_total_vram_gb: None,
        supported_sm: crate::gpu_arch::supported_list(),
        default_ctx: None,
        max_ctx: Some(facts.max_ctx),
        default_max_conc: None,
        default_tp: None,
        num_attention_heads: Some(facts.num_attention_heads),
        num_kv_heads: Some(facts.num_kv_heads),
        attention: Some(facts.attention),
        kv_cache_dtype: None,
        memory: Some(memory),
        extra_sglang_args: extra,
        runtime: Some(runtime.unwrap_or("sglang").to_string()),
        pack_id: None,
        base_model_id: None,
        env: None,
        pythonpath: None,
        revision: Some(sha),
        gated: pre.gated.then_some(true),
        manual_download: None,
        download_gb: Some(download_gb),
        description: None,
        family: None,
        license: None,
        released: None,
        modalities: None,
    };
    Ok((entry, warnings))
}

#[cfg(test)]
mod validate_tests {
    use super::*;

    #[test]
    fn legacy_ternary_entry_validates() {
        let json = r#"{"version":1,"models":[{"id":"goethe/x","runtime":"ternary","architecture":"moe",
            "pack_id":"o/p","base_model_id":"o/b","static_vram_gb":1.0,"kv_bytes_per_token":512,
            "supported_sm":["12.0"],"default_ctx":4096,"default_max_conc":4,"default_tp":2,
            "kv_cache_dtype":"fp8_e4m3","env":{"TERNARY_PACK_DIR":"$GOETHE_HOME/models/o__p__pack",
            "TERNARY_MODEL_DIR":"$GOETHE_HOME/models/o__b","TERNARY_PROFILE":"flash_next"}}]}"#;
        let reg = parse_registry(json).unwrap();
        let mut errors = Vec::new();
        validate_entry(&reg.models[0], &mut errors);
        assert!(errors.is_empty(), "{errors:?}");
        let mut bad = reg.models[0].clone();
        bad.env = Some(toml::Value::Table(toml::from_str("TERNARY_PACK_DIR = \"x\"\nTERNARY_PROFILE = \"moe\"").unwrap()));
        bad.extra_sglang_args = vec!["--tool-call-parser".into()];
        let mut errors = Vec::new();
        validate_entry(&bad, &mut errors);
        assert_eq!(errors.len(), 4, "{errors:?}");
    }

    #[test]
    fn reserved_flags_are_refused() {
        let json = r#"{"version":1,"models":[{"id":"o/m","static_vram_gb":1.0,"kv_elems_per_token":2,"max_ctx":8,
            "num_attention_heads":1,"num_kv_heads":1,"supported_sm":["12.0"],
            "extra_sglang_args":["--max-running-requests","8","--kv-cache-dtype","fp8_e4m3"]}]}"#;
        let reg = parse_registry(json).unwrap();
        let mut errors = Vec::new();
        validate_entry(&reg.models[0], &mut errors);
        assert_eq!(errors.len(), 2, "{errors:?}");
    }

    #[test]
    fn legacy_kv_bytes_maps_to_elems() {
        let m = RegistryModel {
            id: "org/x".into(),
            architecture: "dense".into(),
            static_vram_gb: 1.0,
            kv_bytes_per_token: Some(512),
            kv_elems_per_token: None,
            min_total_vram_gb: None,
            supported_sm: vec!["12.0".into()],
            default_ctx: Some(4096),
            max_ctx: None,
            default_max_conc: None,
            default_tp: None,
            num_attention_heads: Some(8),
            num_kv_heads: Some(2),
            attention: None,
            kv_cache_dtype: None,
            memory: None,
            extra_sglang_args: vec![],
            runtime: Some("sglang".into()),
            pack_id: None,
            base_model_id: None,
            env: None,
            pythonpath: None,
            revision: None,
            gated: None,
            manual_download: None,
            download_gb: None,
            description: None,
            family: None,
            license: None,
            released: None,
            modalities: None,
        };
        assert_eq!(m.effective_kv_elems(), Some(512));
        assert_eq!(m.effective_max_ctx(), Some(4096));
    }
}
