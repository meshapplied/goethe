// Registry fetch, validation, and entry inference.

use crate::hf::{self, fetch_config_json};
use crate::model_facts;
use crate::paths::goethe_home;
use crate::sglang_infer;
use crate::sglang_validate;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Registry {
    pub version: u32,
    pub models: Vec<RegistryModel>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gguf_models: Vec<RegistryModel>,
}

fn gguf_catalog() -> bool {
    // A Mac (or GOETHE_CATALOG=gguf) uses the GGUF models of the registry as its catalog, served by llama.cpp.
    cfg!(all(target_os = "macos", not(test))) || std::env::var("GOETHE_CATALOG").is_ok_and(|v| v == "gguf")
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
    pub pack_only: Option<bool>,
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
            Some("weights are not on Hugging Face; place them in GOETHE_HOME/models and run goethe models scan")
        } else if self.gated.unwrap_or(false) {
            Some("gated on Hugging Face; download it with your own account and run goethe models scan")
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

pub fn registry_url() -> String {
    // GOETHE_REGISTRY_URL when set, else the registry of the package goethe installs from.
    crate::paths::setting("GOETHE_REGISTRY_URL").unwrap_or_else(|| format!("{}/v1/registry.json", crate::update::pkg_base_url()))
}

pub fn fetch(url: &str) -> Result<Registry, String> {
    // file:// or http(s). Parse a body only after the release key verifies it.
    if let Some(rest) = url.strip_prefix("file://") {
        let path = Path::new(rest);
        let text = fs_read(path)?;
        let sig = std::fs::read(format!("{}.sig", path.display())).ok();
        return finish_registry(&text, sig.as_deref());
    }
    if let Some(why) = crate::update::plain_http_refusal(url) {
        return Err(why);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        return fetch_http_cached(url);
    }
    Err(format!("unsupported registry URL scheme: {url}"))
}

fn finish_registry(body: &str, sig: Option<&[u8]>) -> Result<Registry, String> {
    let sig = sig.ok_or("registry signature missing")?;
    verify_registry(body.as_bytes(), sig)?;
    parse_registry(body)
}

pub fn cached() -> Option<Registry> {
    // The last registry fetched; it is cached only after its signature checked, so it is read without the network.
    cached_in(&goethe_home())
}

fn cached_in(home: &Path) -> Option<Registry> {
    let text = std::fs::read_to_string(home.join("registry.cache.json")).ok()?;
    parse_registry(&serde_json::from_str::<RegistryCache>(&text).ok()?.body).ok()
}

fn verify_registry(body: &[u8], sig: &[u8]) -> Result<(), String> {
    // The same release-key check as the package list. A bad signature never becomes engine flags.
    crate::release_key::verify("the registry", body, sig)
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
        return cached_in(&home).ok_or_else(|| format!("the model catalog at {url} is unreachable (last try under {} min ago); connect to the internet once, goethe keeps a copy", MISS_TTL_SECS / 60));
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
    // One request: a 200 refreshes the cache, a 304 reuses it, and a failure falls back to it.
    let cache_path = home.join("registry.cache.json");
    let mut req = crate::net::agent().get(url).timeout(std::time::Duration::from_secs(10));
    if let Ok(text) = std::fs::read_to_string(&cache_path) {
        if let Ok(c) = serde_json::from_str::<RegistryCache>(&text) {
            if let Some(etag) = &c.etag {
                req = req.set("If-None-Match", etag);
            }
        }
    }
    let cached_body = || -> Result<String, String> {
        let text = std::fs::read_to_string(&cache_path).map_err(|e| e.to_string())?;
        let c: RegistryCache = serde_json::from_str(&text).map_err(|e| format!("registry cache: {e}"))?;
        Ok(c.body)
    };
    let sig_bytes = |body_url: &str| -> Option<Vec<u8>> {
        let resp = crate::net::agent()
            .get(&format!("{body_url}.sig"))
            .timeout(std::time::Duration::from_secs(10))
            .call()
            .ok()?;
        if resp.status() != 200 {
            return None;
        }
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut resp.into_reader(), &mut buf).ok()?;
        Some(buf)
    };
    let fresh = || -> Result<Registry, (bool, String)> {
        let checked = |body: &str| finish_registry(body, sig_bytes(url).as_deref()).map_err(|e| (false, e));
        match req.call() {
            Ok(resp) if resp.status() == 304 => checked(&cached_body().map_err(|e| (false, e))?),
            Ok(resp) => {
                let etag = resp.header("etag").map(String::from);
                let body = resp.into_string().map_err(|e| (true, format!("registry body: {e}")))?;
                let reg = checked(&body)?;
                let cache = RegistryCache { etag, body };
                let _ = std::fs::write(&cache_path, serde_json::to_string(&cache).unwrap_or_default());
                Ok(reg)
            }
            Err(e) => Err((true, e.to_string())),
        }
    };
    fresh().or_else(|(unreachable, why)| match cached_in(home) {
        Some(reg) => {
            eprintln!("warning: registry fetch failed ({why}); using the last one fetched");
            Ok(reg)
        }
        None if unreachable => Err(format!("the model catalog at {url} is unreachable ({why}); connect to the internet once, goethe keeps a copy")),
        None => Err(why),
    })
}

pub fn known_ids() -> Vec<String> {
    // The models of the last registry fetched; no network.
    cached().map(|r| r.models.into_iter().filter(|m| !m.manual_download.unwrap_or(false)).map(|m| m.id).collect()).unwrap_or_default()
}

pub fn find_model<'a>(reg: &'a Registry, model_id: &str) -> Option<&'a RegistryModel> {
    reg.models.iter().find(|m| m.id == model_id)
}

fn parse_registry(text: &str) -> Result<Registry, String> {
    let mut reg: Registry = serde_json::from_str(text).map_err(|e| format!("registry parse failed: {e}"))?;
    if gguf_catalog() {
        reg.models = std::mem::take(&mut reg.gguf_models);
    }
    Ok(reg)
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
    for m in &reg.gguf_models {
        if !ids.insert(m.id.clone()) {
            errors.push(format!("duplicate id {}", m.id));
        }
        if m.effective_runtime() != "llama.cpp" {
            errors.push(format!("{}: a GGUF model has runtime llama.cpp", m.id));
        }
        if m.static_vram_gb <= 0.0 {
            errors.push(format!("{}: static_vram_gb (the default quant's size) must be above 0", m.id));
        }
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
        if legacy {
            eprintln!("warning: {}: missing num_attention_heads (legacy entry)", m.id);
        } else {
            errors.push(format!("{}: num_attention_heads required", m.id));
        }
    }
    if m.num_kv_heads.is_none() {
        if legacy {
            eprintln!("warning: {}: missing num_kv_heads (legacy entry)", m.id);
        } else {
            errors.push(format!("{}: num_kv_heads required", m.id));
        }
    }
    if m.architecture != "dense" && m.architecture != "moe" {
        errors.push(format!("{}: architecture must be dense or moe", m.id));
    }
    let rt = m.effective_runtime();
    if rt != "sglang" && rt != "llama.cpp" {
        errors.push(format!("{}: runtime must be sglang or llama.cpp", m.id));
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
        "kv_cache_dtype", "memory", "extra_sglang_args", "runtime", "env",
        "pythonpath", "revision", "gated", "manual_download", "pack_only", "download_gb", "description", "family",
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

pub fn build(src: &Value, infer: &dyn Fn(&str, &str) -> Result<RegistryModel, String>) -> Result<Registry, String> {
    // registry.json from registry.src.json: each entry inferred from Hugging Face at its pinned revision with every field the source states laid over it; a gated or manual_download entry, and every GGUF entry, is its source entry as is. O(entries) infers.
    let list = |k: &str| src[k].as_array().cloned().unwrap_or_default();
    let mut models = Vec::new();
    for s in list("models") {
        let id = s["id"].as_str().ok_or("registry.src.json: an entry has no id")?;
        let full = if s["gated"] == true || s["manual_download"] == true {
            s.clone()
        } else {
            let rev = s["revision"].as_str().filter(|r| r.len() == 40 && r.bytes().all(|b| b.is_ascii_hexdigit()));
            let rev = rev.ok_or_else(|| format!("{id}: revision must be the 40-hex commit the entry was checked at"))?;
            let mut v = serde_json::to_value(infer(id, rev).map_err(|e| format!("{id}: {e}"))?).map_err(|e| e.to_string())?;
            for (k, x) in s.as_object().into_iter().flatten() {
                v[k] = x.clone();
            }
            v
        };
        models.push(serde_json::from_value::<RegistryModel>(full).map_err(|e| format!("{id}: {e}"))?);
    }
    let gguf_models = list("gguf_models").into_iter().map(serde_json::from_value).collect::<Result<Vec<RegistryModel>, _>>().map_err(|e| format!("gguf_models: {e}"))?;
    let reg = Registry { version: 1, models, gguf_models };
    let mut errors = Vec::new();
    let mut seen = std::collections::HashMap::new();
    for m in reg.models.iter().chain(&reg.gguf_models) {
        if seen.insert(m.id.clone(), ()).is_some() {
            errors.push(format!("{}: listed twice", m.id));
        }
    }
    let families: std::collections::BTreeSet<&str> = reg.models.iter().chain(&reg.gguf_models).filter_map(|m| m.family.as_deref()).collect();
    for (a, b) in families.iter().zip(families.iter().skip(1)) {
        if a.eq_ignore_ascii_case(b) {
            errors.push(format!("family spelled two ways: {a}, {b}"));
        }
    }
    for m in &reg.models {
        validate_entry(m, &mut errors);
    }
    errors.extend(sglang_validate::validate_registry_sglang(&reg, false).into_iter().filter(|e| !e.starts_with("note:") && !e.starts_with("warning:")));
    if errors.is_empty() {
        Ok(reg)
    } else {
        Err(format!("registry build refused: {}", errors.join("; ")))
    }
}

pub fn run_build(src: &Path, out: &Path) -> Result<(), String> {
    // Infer every entry of src from Hugging Face and write out, only when the whole registry builds and validates.
    let s: Value = serde_json::from_str(&fs_read(src)?).map_err(|e| format!("{}: {e}", src.display()))?;
    let reg = build(&s, &|id, rev| infer_one_entry(id, None, Some(rev)).map(|(m, warnings)| {
        warnings.iter().for_each(|w| eprintln!("warning: {w}"));
        m
    }))?;
    write_registry_atomic(out, &reg)?;
    println!("{}: {} models, {} GGUF models", out.display(), reg.models.len(), reg.gguf_models.len());
    Ok(())
}

fn write_registry_atomic(path: &Path, reg: &Registry) -> Result<(), String> {
    let text = serde_json::to_string_pretty(reg).map_err(|e| e.to_string())? + "\n";
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

fn infer_one_entry(
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
        max_ctx: facts.max_ctx,
        default_max_conc: None,
        default_tp: None,
        num_attention_heads: Some(facts.num_attention_heads),
        num_kv_heads: Some(facts.num_kv_heads),
        attention: Some(facts.attention),
        kv_cache_dtype: None,
        memory: Some(memory),
        extra_sglang_args: extra,
        runtime: Some(runtime.unwrap_or("sglang").to_string()),
        env: None,
        pythonpath: None,
        revision: Some(sha),
        gated: pre.gated.then_some(true),
        manual_download: None,
        pack_only: None,
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

    const SRC: &str = include_str!("../../../registry.src.json");

    #[test]
    fn build_infers_at_the_pinned_revision_and_the_source_wins() {
        // A valid entry, renamed, stands in for what Hugging Face infers.
        let base: RegistryModel = serde_json::from_value(serde_json::json!({"id": "org/base", "architecture": "dense", "static_vram_gb": 5.6,
            "kv_elems_per_token": 131072, "max_ctx": 131072, "supported_sm": ["12.0"], "num_attention_heads": 32, "num_kv_heads": 8})).unwrap();
        let rev = "a".repeat(40);
        let infer = |id: &str, r: &str| {
            assert_eq!(r, "a".repeat(40));
            if id == "org/gone" {
                return Err("404".to_string());
            }
            Ok(RegistryModel { id: id.into(), description: Some("inferred".into()), family: None, ..base.clone() })
        };
        let one = |e: Value| build(&serde_json::json!({"models": [e]}), &infer).map(|r| r.models[0].clone());
        let m = one(serde_json::json!({"id": "org/m", "revision": rev, "description": "curated"})).unwrap();
        assert_eq!((m.description.as_deref(), m.static_vram_gb), (Some("curated"), base.static_vram_gb));
        let hand = RegistryModel { id: "org/gated".into(), gated: Some(true), revision: None, ..base.clone() };
        assert_eq!(one(serde_json::to_value(&hand).unwrap()).unwrap().id, "org/gated", "a gated entry is kept as written, never inferred");
        assert!(one(serde_json::json!({"id": "org/m", "revision": "main"})).unwrap_err().contains("40-hex"));
        assert!(one(serde_json::json!({"id": "org/gone", "revision": rev})).unwrap_err().contains("org/gone: 404"));
        let two = |a: Value, b: Value| build(&serde_json::json!({"models": [a, b]}), &infer).map(|_| ());
        let e = |id: &str, fam: &str| serde_json::json!({"id": id, "revision": rev, "family": fam});
        assert!(two(e("org/m", "Qwen"), e("org/m", "Qwen")).unwrap_err().contains("org/m: listed twice"));
        assert!(two(e("org/a", "Qwen"), e("org/b", "qwen")).unwrap_err().contains("family spelled two ways"));
        assert!(two(e("org/a", "Qwen"), e("org/b", "Qwen")).is_ok());
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
            env: None,
            pythonpath: None,
            revision: None,
            gated: None,
            manual_download: None,
        pack_only: None,
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

    #[test]
    fn unsigned_and_wrong_key_registries_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let reg = dir.path().join("registry.json");
        std::fs::write(&reg, r#"{"version":1,"models":[]}"#).unwrap();
        let url = format!("file://{}", reg.display());
        assert!(fetch(&url).unwrap_err().contains("signature"), "unsigned file");

        let key = dir.path().join("key.pem");
        let pubk = dir.path().join("pub.pem");
        let key2 = dir.path().join("key2.pem");
        let wrong = dir.path().join("wrong.pem");
        let run = |args: &[std::ffi::OsString]| {
            assert!(std::process::Command::new("openssl").args(args).status().unwrap().success());
        };
        run(&["genpkey".into(), "-algorithm".into(), "RSA".into(), "-pkeyopt".into(), "rsa_keygen_bits:2048".into(), "-out".into(), key.clone().into()]);
        run(&["pkey".into(), "-in".into(), key.clone().into(), "-pubout".into(), "-out".into(), pubk.clone().into()]);
        run(&["genpkey".into(), "-algorithm".into(), "RSA".into(), "-pkeyopt".into(), "rsa_keygen_bits:2048".into(), "-out".into(), key2.clone().into()]);
        run(&["pkey".into(), "-in".into(), key2.into(), "-pubout".into(), "-out".into(), wrong.clone().into()]);
        let sig = std::path::PathBuf::from(format!("{}.sig", reg.display()));
        run(&["dgst".into(), "-sha256".into(), "-sign".into(), key.into(), "-out".into(), sig.clone().into(), reg.into()]);

        let held = crate::release_key::test_key::set(&std::fs::read_to_string(&wrong).unwrap());
        assert!(fetch(&url).unwrap_err().contains("release key"), "wrong key");
        drop(held);
        let _held = crate::release_key::test_key::set(&std::fs::read_to_string(&pubk).unwrap());
        assert!(fetch(&url).is_ok(), "matching key");
    }

    #[test]
    fn a_fresh_registry_that_does_not_verify_falls_back_to_the_last_verified_one() {
        // A body and a signature out of step (a publish half way through a cache): the copy that verified before is used, with a warning.
        let home = tempfile::tempdir().unwrap();
        let srv = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/registry.json", srv.local_addr().unwrap());
        std::thread::spawn(move || {
            for (mut s, body) in srv.incoming().flatten().zip([r#"{"version": 1, "models": []}"#, "not a signature"]) {
                let mut buf = [0u8; 4096];
                let _ = std::io::Read::read(&mut s, &mut buf);
                let _ = std::io::Write::write_all(&mut s, format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
            }
        });
        let _held = crate::release_key::test_key::hold();
        let cached = r#"{"version": 1, "models": [{"id": "org/m", "static_vram_gb": 1.0, "kv_elems_per_token": 2, "max_ctx": 8}]}"#;
        std::fs::write(home.path().join("registry.cache.json"), serde_json::to_string(&RegistryCache { etag: None, body: cached.into() }).unwrap()).unwrap();
        assert_eq!(fetch_http_uncached(home.path(), &url).unwrap().models[0].id, "org/m");
    }

    #[test]
    fn offline_the_last_registry_fetched_is_used() {
        let home = tempfile::tempdir().unwrap();
        assert!(cached_in(home.path()).is_none(), "nothing fetched yet: no catalog, never a built-in one");
        let body = r#"{"version": 1, "models": [{"id": "org/m", "static_vram_gb": 1.0, "kv_elems_per_token": 2, "max_ctx": 8}]}"#;
        let cache = RegistryCache { etag: None, body: body.into() };
        std::fs::write(home.path().join("registry.cache.json"), serde_json::to_string(&cache).unwrap()).unwrap();
        assert_eq!(cached_in(home.path()).unwrap().models[0].id, "org/m");
    }

    #[test]
    fn readme_install_ids_are_catalog_ids() {
        let ids: Vec<String> = serde_json::from_str::<Value>(SRC).unwrap()["models"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap().to_string()).collect();
        let readme = include_str!("../../../../README.md");
        for line in readme.lines() {
            let id = line.split("goethe install ").nth(1).and_then(|r| r.split_whitespace().next()).or_else(|| line.split("model=\"").nth(1).and_then(|r| r.split('"').next()));
            if let Some(id) = id {
                assert!(ids.iter().any(|m| m == id), "{id} is not in registry.src.json");
            }
        }
    }
}
