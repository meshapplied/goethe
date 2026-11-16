// Model install: Hugging Face or local trees, goethe_card.toml, and state.db registration.

use crate::db;
use crate::engines::goethe_engine::{PACK_FILE, PACK_VERIFY};
use crate::hf::{self, DownloadOpts};
use crate::model_facts::{self, GIB};
use crate::paths::{contained_in_models, ensure_layout, goethe_home, model_dir, validate_model_id};
use crate::registry::{fetch, find_model, registry_url, RegistryModel};
use crate::safetensors;
use crate::sglang_infer as sgl;
use rusqlite::OptionalExtension;
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_sm: Option<Vec<String>>,
}

pub fn fallback_allowed(flag: bool) -> bool {
    // A model with a kernel pack for this card runs on its plain runtime only when the user opts in: --fallback, or GOETHE_ALLOW_FALLBACK=1.
    allowed(flag, std::env::var("GOETHE_ALLOW_FALLBACK").ok().as_deref())
}

fn allowed(flag: bool, env: Option<&str>) -> bool {
    flag || env == Some("1")
}

const FALLBACK_HINT: &str = "pass --fallback (or set GOETHE_ALLOW_FALLBACK=1) to serve it with the plain runtime instead";

pub const PACK_SKIPPED: &str = "pack_skipped";

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
    "supported_sm",
];

pub fn card_supported_sm(card: &toml::Value) -> Vec<String> {
    // Compute capabilities the card's catalog entry lists; empty (any supported GPU) for cards without one.
    card.get("supported_sm")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default()
}

pub fn card_marlin_extra_gb(card: &toml::Value) -> f64 {
    // The card's Marlin weight copy, or 0 when GOETHE_MARLIN=0 in goethed's or the card's environment.
    let off = |v: Option<&str>| v.is_some_and(|v| v.trim() == "0");
    if off(std::env::var("GOETHE_MARLIN").ok().as_deref())
        || off(card.get("env").and_then(|e| e.get("GOETHE_MARLIN")).and_then(|v| v.as_str()))
    {
        return 0.0;
    }
    card.get("marlin_extra_gb").and_then(|v| v.as_float()).unwrap_or(0.0)
}

#[derive(Default)]
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
    pub quant: Option<String>,
    pub fallback: bool,
}

struct Target {
    model_id: String,
    card_dir: PathBuf,
    entry: Option<RegistryModel>,
    managed: bool,
    source_dir: Option<PathBuf>,
    ternary: Option<TernaryDirs>,
    pack: Option<PackChoice>,
}

struct PackChoice {
    offer: crate::packs::Offer,
    creds: crate::platform::Credentials,
    plan: Option<crate::packs::Plan>,
}

enum PackDecision {
    None,
    Declined,
    Use(Box<PackChoice>),
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
        match install_hub(&home, &opts) {
            Ok(t) => t,
            Err(e) if gated_but_installed(&e, is_installed(&home, &opts.model_id), opts.force) => {
                println!("{} is already installed; --force downloads it again", opts.model_id);
                if opts.load {
                    load_installed(&opts.model_id, &opts)?;
                }
                return Ok(());
            }
            Err(e) => return Err(e),
        }
    };
    let model_id = target.model_id.clone();
    register(&home, &opts, target)?;
    if opts.load {
        load_installed(&model_id, &opts)?;
    }
    Ok(())
}

fn is_installed(home: &Path, model_id: &str) -> bool {
    // The model has a row in the installed registry.
    db::open(home).ok().is_some_and(|c| c.query_row("SELECT 1 FROM installed_models WHERE model_id = ?1", [model_id], |_| Ok(())).is_ok())
}

fn gated_but_installed(err: &str, installed: bool, force: bool) -> bool {
    // Hugging Face answers a gated or private repo without a token as not found: for a model already installed that is no failure, unless --force asks for the download.
    installed && !force && (err.contains("is gated or private on Hugging Face") || err.contains("not found on Hugging Face, or private"))
}

fn install_hub(home: &Path, opts: &InstallOpts) -> Result<Target, String> {
    // Download a Hugging Face model, or a ternary pack plus its base, into models/.
    if opts.model_id.starts_with(['.', '/', '~']) {
        return Err(format!("{}: no such directory", opts.model_id));
    }
    validate_model_id(&opts.model_id)?;
    println!("looking up {} ...", opts.model_id);
    let entry = fetch(&registry_url())
        .ok()
        .and_then(|r| find_model(&r, &opts.model_id).cloned());
    let card_dir = model_dir(home, &opts.model_id);
    dir_owner_is(&card_dir, &opts.model_id)?;
    if opts.model_id == crate::engines::ninfer::NINFER_ID {
        let file = crate::engines::ninfer::acquire_ninfer(&card_dir)?;
        fs::create_dir_all(&card_dir).map_err(|e| e.to_string())?;
        let dest = card_dir.join(crate::engines::ninfer::NINFER_FILE);
        if fs::canonicalize(&dest).ok().as_deref() != Some(file.as_path()) {
            if dest.symlink_metadata().is_ok() {
                fs::remove_file(&dest).map_err(|e| e.to_string())?;
            }
            std::os::unix::fs::symlink(&file, &dest).map_err(|e| format!("link {}: {e}", dest.display()))?;
        }
        let mut profile = crate::recommend::ninfer_profile();
        if let Some(e) = entry {
            profile.description = e.description;
            profile.license = e.license;
            profile.family = e.family;
        }
        return Ok(Target {
            model_id: opts.model_id.clone(),
            card_dir,
            entry: Some(profile),
            managed: false,
            source_dir: file.parent().map(|p| p.to_path_buf()),
            ternary: None,
            pack: None,
        });
    }
    if entry.as_ref().is_some_and(|e| e.manual_download.unwrap_or(false)) {
        return Err(format!(
            "the weights of {} are not on Hugging Face, so goethe cannot download them. Put the model folder \
             (config.json and safetensors) at {} and run: goethe models scan",
            opts.model_id,
            card_dir.display()
        ));
    }
    if let Some(e) = entry.as_ref() {
        crate::recommend::preview(home, e)?;
    }
    let dl = DownloadOpts {
        revision: entry.as_ref().and_then(|e| e.revision.clone()),
        force: opts.force,
        no_verify: opts.no_verify,
        only: None,
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
            only: None,
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
            pack: None,
        });
    }
    let mut pack_refused = false;
    let declined = match pack_decision(home, opts)? {
        PackDecision::Use(mut choice) => 'packed: {
            // Only the files the pack was built for, at its pinned revision; the engine layer once, on first use.
            crate::pyenv::ensure(home, crate::pyenv::Layer::Engine)?;
            let login = crate::packs::Login(choice.creds.clone());
            let broker = crate::packs::HttpBroker::new(&choice.creds.platform_url, &login);
            let plan = crate::packs::plan(home, &opts.model_id, &choice.offer.arch, opts.quant.as_deref(), &broker, &crate::packs::python_runner(home))?
                .ok_or_else(|| format!("the signed pack index has no pack for {} on {}", opts.model_id, choice.offer.arch))?;
            if let Some(why) = refused_before_download(home, &plan, &crate::packs::python_runner(home)) {
                println!("{}", pack_only_refusal(&opts.model_id, entry.as_ref().is_some_and(|e| e.pack_only == Some(true)), &why, fallback_allowed(opts.fallback))?);
                pack_refused = true;
                break 'packed false;
            }
            let rev = choice.offer.revision.clone();
            let pre = hf::preflight_model(&choice.offer.hf_repo, Some(&rev))?;
            let size = |f: &String| pre.siblings.iter().find(|s| &s.rfilename == f).map(|s| s.size);
            let bytes = |v: &Vec<String>| v.iter().filter_map(size).sum::<u64>();
            let fits = |v: &Vec<String>| hf::disk_preflight(home, bytes(v)).and_then(|_| vram_preflight(home, bytes(v)));
            let present: Vec<&Vec<String>> = plan.variants.iter().filter(|v| v.iter().all(|f| size(f).is_some())).collect();
            let files = match present.iter().find(|v| fits(v).is_ok()).or(present.first()) {
                Some(v) => (*v).clone(),
                None => return Err(format!("{}: the model files its kernel pack was built for are not in the repository", choice.offer.hf_repo)),
            };
            fits(&files)?;
            let draft = crate::packs::drafter(&plan.entry);
            let dgb = draft.as_ref().map_or(0, |d| d.files.iter().map(|f| f.1).sum::<u64>());
            let both = bytes(&files) + dgb;
            let short = match (hf::disk_preflight(home, both), vram_preflight(home, both)) {
                (Err(_), _) => Some("disk"),
                (_, Err(_)) => Some("VRAM"),
                _ => None,
            };
            let spec = match (&draft, short) {
                (None, _) => String::new(),
                (Some(d), None) => format!(", speculative decoding {} (+{:.1} GB)", d.name, dgb as f64 / GIB),
                (Some(d), Some(what)) => format!(", speculative decoding off: {} needs {:.1} GB more {what}", d.name, dgb as f64 / GIB),
            };
            confirm(opts, &format!("{} ({}, {:.1} GB) with optimised kernels for {}{spec}", opts.model_id, choice.offer.format, bytes(&files) as f64 / GIB, choice.offer.arch))?;
            let pack_dl = DownloadOpts { revision: Some(rev), force: opts.force, no_verify: opts.no_verify, only: Some(files.clone()) };
            hf::ensure_model(home, &choice.offer.hf_repo, &card_dir, &pack_dl)?;
            if let Some((repo, revision, only)) = crate::packs::aux(&plan.entry) {
                fetch_aux(home, &repo, &card_dir, DownloadOpts { revision, force: opts.force, no_verify: opts.no_verify, only: Some(only) })?;
            }
            if let Some(d) = draft.filter(|_| short.is_none()) {
                let only = Some(d.files.iter().map(|f| f.0.clone()).collect());
                let dl = DownloadOpts { revision: d.revision, force: opts.force, no_verify: opts.no_verify, only };
                if let Err(e) = hf::ensure_model(home, &d.repo, &card_dir.join("drafter"), &dl) {
                    eprintln!("note: speculative decoding off: {e}");
                }
            }
            gguf_metadata(home, &card_dir, &choice.offer.format, &files)?;
            choice.plan = Some(plan);
            return Ok(Target {
                model_id: opts.model_id.clone(),
                card_dir,
                entry,
                managed: true,
                source_dir: None,
                ternary: None,
                pack: Some(*choice),
            });
        }
        PackDecision::Declined => true,
        PackDecision::None => false,
    };
    let local = crate::hw::profile(home).map(|h| h.gpus.iter().map(|g| g.name.clone()).collect::<Vec<_>>().join(", ")).unwrap_or_default();
    let no_pack_gpu = crate::packs::catalog(home).and_then(|ix| crate::packs::unsupported_here(&ix, &opts.model_id, &crate::packs::local_archs(home), &local));
    let built = crate::packs::catalog(home).map(|ix| crate::packs::built_for(&ix, &opts.model_id)).unwrap_or_default();
    pack_only_gate(&opts.model_id, entry.as_ref().is_some_and(|e| e.pack_only == Some(true)) && opts.quant.is_none(), declined, no_pack_gpu.as_deref(), &built)?;
    let gguf = engine_check(home, &opts.model_id, dl.revision.as_deref(), entry.as_ref(), (declined, no_pack_gpu.as_deref()), opts.quant.as_deref())?;
    if !hf::offline_mode() {
        if let Ok(pre) = hf::preflight_model(&opts.model_id, dl.revision.as_deref()) {
            let bytes = match &gguf {
                Some(files) => pre.siblings.iter().filter(|s| files.contains(&s.rfilename)).map(|s| s.size).sum(),
                None => hf::download_bytes(&pre.siblings),
            };
            let (fmt, engine) = if gguf.is_some() { ("gguf", "llama.cpp") } else { ("safetensors", "SGLang") };
            let why = match (declined, pack_refused) {
                (true, _) => "not logged in",
                (_, true) => "its kernel pack does not run on this GPU",
                _ => "none for this model on this GPU",
            };
            confirm(opts, &format!("{} ({fmt}, {:.1} GB) with {engine} (no kernel pack: {why})", opts.model_id, bytes as f64 / GIB))?;
        }
    }
    hf::ensure_model(home, &opts.model_id, &card_dir, &DownloadOpts { only: gguf.clone(), ..dl })?;
    let pack = readable(structure_offer(home, opts, &card_dir), |c| config_for_pack(home, opts, &card_dir, c, gguf.as_deref().unwrap_or_default()), fallback_allowed(opts.fallback))?;
    Ok(Target {
        model_id: opts.model_id.clone(),
        card_dir,
        entry,
        managed: true,
        source_dir: None,
        ternary: None,
        pack,
    })
}

fn readable<T>(choice: Option<T>, meta: impl FnOnce(&T) -> Result<(), String>, allow: bool) -> Result<Option<T>, String> {
    // A pack offered for GGUF weights is used only when their config and tokenizer are in place; otherwise the install fails with the reason, or with --fallback serves them with the plain runtime and says why.
    let Some(c) = choice else { return Ok(None) };
    match (meta(&c), allow) {
        (Ok(()), _) => Ok(Some(c)),
        (Err(e), true) => {
            eprintln!("note: the optimised kernels were not used, serving with the plain runtime: {e}");
            Ok(None)
        }
        (Err(e), false) => Err(format!("the optimised kernels for these weights cannot be used: {e}; {FALLBACK_HINT}")),
    }
}

fn fetch_aux(home: &Path, repo: &str, dir: &Path, dl: DownloadOpts) -> Result<(), String> {
    // The model's own config and tokenizer files beside GGUF weights.
    hf::ensure_model(home, repo, dir, &dl)?;
    drop_orphan_index(dir);
    Ok(())
}

fn drop_orphan_index(dir: &Path) {
    // The repo's safetensors index comes along with its config files and names shards that are not here: without any .safetensors file it goes, or the weights look incomplete.
    let names: Vec<(PathBuf, String)> = fs::read_dir(dir).into_iter().flatten().flatten().map(|e| (e.path(), e.file_name().to_string_lossy().into_owned())).collect();
    if !names.iter().any(|(_, n)| n.ends_with(".safetensors")) {
        for (p, _) in names.iter().filter(|(_, n)| n.ends_with(".safetensors.index.json")) {
            let _ = fs::remove_file(p);
        }
    }
}

fn config_for_pack(home: &Path, opts: &InstallOpts, dir: &Path, c: &PackChoice, gguf: &[String]) -> Result<(), String> {
    // The pack's signed entry names the model's own config and tokenizer files: fetch those beside the weights; only without them are they made from the GGUF header (which needs a transformers that knows the architecture).
    if let Some((repo, revision, only)) = c.plan.as_ref().and_then(|p| crate::packs::aux(&p.entry)) {
        fetch_aux(home, &repo, dir, DownloadOpts { revision, force: opts.force, no_verify: opts.no_verify, only: Some(only) })?;
    }
    if gguf.is_empty() {
        Ok(())
    } else {
        gguf_metadata(home, dir, "gguf", gguf)
    }
}

fn structure_offer(home: &Path, opts: &InstallOpts, dir: &Path) -> Option<PackChoice> {
    // Weights no pack names (a fine-tune, another repo or a folder of the same architecture) on a logged-in node with the engine installed: the pack whose structure they have, if taken. Any miss or error: None, silently.
    let creds = crate::platform::load_credentials(home).ok().flatten()?;
    let archs = crate::packs::local_archs(home);
    let engine = crate::pyenv::has(home, crate::pyenv::Layer::Engine) || crate::pyenv::dev_python(home);
    crate::packs::catalog(home).filter(|ix| engine && !crate::packs::offers_any(ix, &archs).is_empty())?;
    let login = crate::packs::Login(creds.clone());
    let broker = crate::packs::HttpBroker::new(&creds.platform_url, &login);
    let (_, plan) = crate::packs::by_structure(home, dir, &archs, &broker, &crate::packs::python_runner(home)).ok()??;
    let what = format!("optimised kernels available for these weights (same structure as {})", plan.entry.offer.hf_repo);
    if opts.yes || !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        println!("{what}");
    } else if !crate::term::ask(&format!("{what}; install them? [Y/n]"), "y").is_ok_and(|a| a.to_ascii_lowercase().starts_with('y')) {
        return None;
    }
    Some(PackChoice { offer: plan.entry.offer.clone(), creds, plan: Some(plan) })
}

fn confirm(opts: &InstallOpts, plan: &str) -> Result<(), String> {
    // One line before any download: what, how big, which engine; a terminal confirms it unless --yes.
    if opts.yes || !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        println!("{plan}");
        return Ok(());
    }
    match crate::term::ask(&format!("{plan}: download? [Y/n]"), "y")?.to_ascii_lowercase().starts_with('y') {
        true => Ok(()),
        false => Err("nothing was downloaded".into()),
    }
}

fn vram_preflight(home: &Path, bytes: u64) -> Result<(), String> {
    // Weights larger than all allowed GPUs together never load: refused before they are downloaded.
    let Ok(hw) = crate::hw::profile(home) else { return Ok(()) };
    let denied = db::open(home).ok().and_then(|c| crate::gpu_policy::denied_set(&c).ok()).unwrap_or_default();
    let total = hw.gpus.iter().filter(|g| !denied.contains(&g.index)).map(|g| g.vram_total_mb as f64 / 1024.0).sum::<f64>();
    let gb = bytes as f64 / GIB;
    if total > 0.0 && gb > total {
        return Err(format!("the weights ({gb:.1} GB) are larger than the allowed GPUs together ({total:.1} GB); nothing was downloaded"));
    }
    Ok(())
}

fn stale_skip(card: Option<&toml::Table>, force: bool, allow: bool) -> Option<String> {
    // The pack_skipped note of a card, when an explicit fallback is on and the pack is not being forced.
    card?.get(PACK_SKIPPED).and_then(|v| v.as_str()).filter(|_| allow && !force).map(String::from)
}

fn pack_decision(home: &Path, opts: &InstallOpts) -> Result<PackDecision, String> {
    // A pack in the platform's index for this model and a local GPU: used when the node is logged in; otherwise the user may log in now. Without a login no pack is ever downloaded. A pack skipped earlier by an explicit fallback stays skipped while the fallback is on, unless --force; without the opt-in the pack is tried again and fails loudly.
    if let Some(why) = stale_skip(read_card_table(&model_dir(home, &opts.model_id)).as_ref(), opts.force, fallback_allowed(opts.fallback)) {
        eprintln!("note: skipping the kernel pack: {why} (--force tries it again)");
        return Ok(PackDecision::Declined);
    }
    let Some(index) = crate::packs::catalog(home) else { return Ok(PackDecision::None) };
    let archs = crate::packs::local_archs(home);
    let Some(offer) = crate::packs::offers(&index, &opts.model_id, &archs).into_iter().next() else {
        let theirs = crate::packs::pack_archs(&index, &opts.model_id);
        if let Some(note) = crate::packs::engine_too_old(&index, &opts.model_id, &archs) {
            eprintln!("{note}");
        } else if !theirs.is_empty() && !theirs.iter().any(|a| archs.contains(a)) {
            eprintln!("note: kernel packs for {} exist for {} only, not for this machine's GPUs ({})", opts.model_id, theirs.join(", "), archs.join(", "));
        }
        return Ok(PackDecision::None);
    };
    let linked = || crate::platform::load_credentials(home).ok().flatten();
    if let Some(creds) = linked() {
        return Ok(PackDecision::Use(Box::new(PackChoice { offer, creds, plan: None })));
    }
    if !ask_login(&offer, opts)? {
        return Ok(PackDecision::Declined);
    }
    if let Err(e) = crate::platform_cli::login(None, None) {
        eprintln!("note: {e}; continuing without the optimised kernels");
        return Ok(PackDecision::Declined);
    }
    match linked() {
        Some(creds) => Ok(PackDecision::Use(Box::new(PackChoice { offer, creds, plan: None }))),
        None => Ok(PackDecision::Declined),
    }
}

fn ask_login(offer: &crate::packs::Offer, opts: &InstallOpts) -> Result<bool, String> {
    // Asked only on a terminal; elsewhere one note, and the install continues with SGLang.
    let what = format!("Optimised kernels for {} are available for this GPU ({})", opts.model_id, offer.arch);
    if !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        eprintln!("note: {what} to nodes logged in to the Goethe platform (goethe login); continuing without them");
        return Ok(false);
    }
    let a = crate::term::ask(&format!("{what}.\nLog in to the Goethe platform to install them? Otherwise goethe uses SGLang. [Y/n]"), "y")?;
    Ok(a.to_ascii_lowercase().starts_with('y'))
}

fn engine_check(
    home: &Path,
    model_id: &str,
    revision: Option<&str>,
    entry: Option<&RegistryModel>,
    (declined, no_pack_gpu): (bool, Option<&str>),
    quant: Option<&str>,
) -> Result<Option<Vec<String>>, String> {
    // Before any download: SGLang must serve the model's architecture and quantization (catalog entries are vetted); a GGUF-only repo, or a :quant, gets llama.cpp and the files to fetch. A model with no plain runtime and no pack for this GPU is refused with the GPUs its packs are built for.
    if (entry.is_some() && quant.is_none()) || hf::offline_mode() {
        return Ok(None);
    }
    let rev = revision.unwrap_or("main");
    let pre = hf::preflight_model(model_id, Some(rev))?;
    let hint = match (declined, no_pack_gpu) {
        (true, _) => " Optimised kernels exist for it on this GPU: run goethe login, then goethe install again.".to_string(),
        (_, Some(n)) => format!(" {n}."),
        _ => String::new(),
    };
    let names: Vec<&str> = pre.siblings.iter().map(|s| s.rfilename.as_str()).collect();
    let config = names.contains(&"config.json");
    if quant.is_some() || (!config && names.iter().any(|n| n.ends_with(".gguf"))) {
        let files = crate::engines::llama_cpp::pick_gguf(&names, quant).map_err(|e| format!("{model_id}: {e}"))?;
        let none = if declined { "" } else { ", and no optimised kernel pack exists for it on this machine's GPUs" };
        let only = if config { "" } else { " only" };
        return match crate::engines::llama_cpp::server_ready(home) {
            Ok(_) => Ok(Some(files)),
            Err(why) => Err(format!("{model_id} is not supported: it ships GGUF files{only}, which SGLang does not serve{none}; {why}.{hint}")),
        };
    }
    if !config {
        return match no_pack_gpu {
            Some(n) => Err(format!("{model_id} is not supported: it is not a plain checkpoint (no config.json, no GGUF). {n}.")),
            None => Ok(None),
        };
    }
    let cfg = hf::fetch_config_json(model_id, rev)?;
    crate::scan::sglang_serves(&crate::scan::sglang_support(home), &cfg).map(|()| None).map_err(|why| format!("{model_id} is not supported: {why}.{hint}"))
}

const GGUF_META: &str = r#"
import collections, os, sys
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES as M
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES, tokenizer_class_from_name
d, f, out, cfg, tok = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1", sys.argv[5] == "1"
if cfg:
    # The header's config, with per-layer counts (e.g. KV heads of sliding and global layers) folded to the config's
    # scalar and its num_global_* field, which strict config classes require.
    raw = dict(load_gguf_checkpoint(os.path.join(d, f), return_tensors=False)["config"])
    mt = raw.pop("model_type")
    raw.pop("_model_name_or_path", None)
    for k, v in list(raw.items()):
        if isinstance(v, list) and k.startswith("num_") and v and all(isinstance(x, int) for x in v):
            common = collections.Counter(v).most_common()
            raw[k] = common[0][0]
            if len(common) > 1:
                raw.setdefault(k.replace("num_", "num_global_", 1), common[-1][0])
    c = AutoConfig.for_model(mt, **raw)
    if not getattr(c, "architectures", None) and mt in M:
        c.architectures = [M[mt]]
    c.save_pretrained(out)
    from transformers import GenerationConfig
    GenerationConfig.from_model_config(c).save_pretrained(out)   # the model's stop ids (e.g. end of turn and eos)
if tok:
    try:
        t = AutoTokenizer.from_pretrained(d, gguf_file=f)
    except Exception:
        name = TOKENIZER_MAPPING_NAMES.get(AutoConfig.from_pretrained(out).model_type) if cfg else None
        cls = tokenizer_class_from_name(name if isinstance(name, str) else (name[1] or name[0])) if name else PreTrainedTokenizerFast
        t = cls.from_pretrained(d, gguf_file=f)
        # A generic tokenizer knows nothing of the model's specials: take them from the GGUF header (control tokens,
        # and the bos/eos/unk/pad ids), so chat turns end where the model ends them.
        from gguf import GGUFReader
        h = GGUFReader(os.path.join(d, f)).fields
        val = lambda k, i: bytes(h[k].parts[h[k].data[i]]).decode() if h[k].types[-1].name == "STRING" else int(h[k].parts[h[k].data[i]][0])
        n = len(h["tokenizer.ggml.tokens"].data)
        tokens = [val("tokenizer.ggml.tokens", i) for i in range(n)]
        if "tokenizer.ggml.token_type" in h:
            ctrl = [tokens[i] for i in range(n) if val("tokenizer.ggml.token_type", i) == 3]
            t.add_special_tokens({"additional_special_tokens": ctrl}, replace_extra_special_tokens=False)
        for role in ("bos", "eos", "unk", "pad"):
            k = f"tokenizer.ggml.{role}_token_id"
            if k in h:
                setattr(t, f"{role}_token", tokens[val(k, 0)])
    t.save_pretrained(out)
"#;

fn gguf_metadata(home: &Path, dir: &Path, format: &str, files: &[String]) -> Result<(), String> {
    // A GGUF repo ships no config.json or tokenizer.json: transformers writes both from the GGUF header, offline, once. Written to a temp dir and moved in, so a failed run never leaves a half written config behind.
    if format != "gguf" {
        return Ok(());
    }
    let (cfg, tok) = (!dir.join("config.json").is_file(), !dir.join("tokenizer.json").is_file());
    if !cfg && !tok {
        return Ok(());
    }
    let mut files = files.to_vec();
    files.sort();
    let first = files.first().ok_or("the pack names no weights file")?;
    let out = dir.join(".goethe-meta.tmp");
    let _ = fs::remove_dir_all(&out);
    fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let run = std::process::Command::new(crate::engines::resolve_python(home))
        .arg("-I")
        .arg("-c")
        .arg(GGUF_META)
        .arg(dir)
        .arg(first)
        .arg(&out)
        .arg(if cfg { "1" } else { "0" })
        .arg(if tok { "1" } else { "0" })
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1")
        .output()
        .map_err(|e| format!("reading the GGUF metadata: {e}"))?;
    if !run.status.success() {
        let _ = fs::remove_dir_all(&out);
        let err = String::from_utf8_lossy(&run.stderr);
        let last = err.lines().rev().find(|l| !l.trim().is_empty()).unwrap_or("").trim().to_string();
        return Err(format!("could not read the tokenizer and config from {first}: {last}"));
    }
    for e in fs::read_dir(&out).map_err(|e| e.to_string())?.flatten() {
        let to = dir.join(e.file_name());
        if !to.exists() {
            fs::rename(e.path(), &to).map_err(|err| format!("{}: {err}", to.display()))?;
        }
    }
    fs::remove_dir_all(&out).map_err(|e| e.to_string())
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
    let has_pack = src.join(PACK_FILE).is_file();
    let gguf: Vec<String> = crate::engines::llama_cpp::gguf_paths(&src)
        .iter()
        .filter_map(|p| Some(p.strip_prefix(&src).ok()?.to_string_lossy().into_owned()))
        .filter(|n| crate::engines::llama_cpp::is_model_gguf(n))
        .collect();
    if !has_pack && gguf.is_empty() && !src.join("config.json").is_file() {
        return Err(format!("{} has no config.json", src.display()));
    }
    if !has_pack && gguf.is_empty() && model_facts::weight_files(&src)?.is_empty() {
        return Err(format!("{} has no .safetensors or .gguf weights", src.display()));
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
    dir_owner_is(&card_dir, &model_id)?;
    let keep_copy = prepare_local_target(home, &card_dir, opts)?;
    if opts.copy_local {
        if !keep_copy {
            copy_dir_all(&src, &card_dir)?;
        }
    } else {
        build_farm(&src, &card_dir)?;
    }
    if has_pack {
        let arch = crate::packs::local_archs(home).into_iter().next().ok_or("no GPU of this machine runs kernel packs")?;
        crate::pyenv::ensure(home, crate::pyenv::Layer::Engine)?;
        local_pack(home, &card_dir, &arch, &crate::packs::python_runner(home))?;
    }
    let entry = if model_id.starts_with("local/") {
        None
    } else {
        fetch(&registry_url()).ok().and_then(|r| find_model(&r, &model_id).cloned())
    };
    let pack = readable((!has_pack).then(|| structure_offer(home, opts, &card_dir)).flatten(), |c| config_for_pack(home, opts, &card_dir, c, &gguf), fallback_allowed(opts.fallback))?;
    Ok(Target {
        model_id,
        card_dir,
        entry,
        managed: opts.copy_local,
        pack,
        source_dir: (!opts.copy_local).then_some(src),
        ternary: None,
    })
}

fn local_pack(home: &Path, dir: &Path, arch: &str, run: crate::packs::Runner) -> Result<(), String> {
    // goethe-engine verifies the pack for this arch and names the weights it matches; a GGUF gets its config and tokenizer.
    for f in [PACK_FILE.to_string(), format!("{PACK_FILE}.sig")] {
        let p = dir.join(&f);
        if fs::symlink_metadata(&p).is_ok_and(|m| m.file_type().is_symlink()) {
            let from = fs::canonicalize(&p).map_err(|e| format!("{}: {e}", p.display()))?;
            fs::remove_file(&p).and_then(|_| fs::copy(&from, &p)).map_err(|e| format!("{}: {e}", p.display()))?;
        }
    }
    let s = |p: &Path| p.to_string_lossy().into_owned();
    let out = run(&["pack".into(), "verify".into(), s(&dir.join(PACK_FILE)), "--arch".into(), arch.into(), "--partial".into(), "--model-dir".into(), s(dir)])?.1;
    let v: Value = serde_json::from_str(&out).map_err(|e| format!("pack verify: {e}"))?;
    let weights: Vec<String> = v["weights"].as_array().into_iter().flatten().filter_map(|w| w.as_str().map(String::from)).collect();
    if weights.is_empty() {
        return Err(format!("{}: the pack matches none of the weights there", dir.display()));
    }
    let verified = dir.join(PACK_VERIFY);
    let _ = fs::remove_file(&verified);
    fs::write(&verified, &out).map_err(|e| format!("{}: {e}", verified.display()))?;
    gguf_metadata(home, dir, if weights.iter().all(|w| w.ends_with(".gguf")) { "gguf" } else { "safetensors" }, &weights)
}

fn dir_owner_is(dir: &Path, model_id: &str) -> Result<(), String> {
    // models/<id with / as __> is shared by ids that differ only there (x/y and x__y): the one that installed first keeps it.
    match read_card_table(dir).and_then(|c| c.get("model_id").and_then(|v| v.as_str()).map(String::from)) {
        Some(other) if other != model_id => Err(format!("{} already holds {other}, and {model_id} would share it; pick another name", dir.display())),
        _ => Ok(()),
    }
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
    if !farm && !opts.copy_local && is_goethe_local_install(card_dir) {
        clear_dir_except_card(home, card_dir)?;
        return Ok(false);
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

pub fn is_goethe_local_install(dir: &Path) -> bool {
    // A directory goethe made for a local folder install: its card names the source folder (the links and copied pack files in it are goethe's).
    read_card_table(dir).is_some_and(|c| c.get("source_dir").and_then(|v| v.as_str()).is_some())
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

pub struct VramEstimate {
    pub weights_gb: f64,
    pub state_gb: f64,
    pub host_offload_gb: f64,
    pub kv_bytes_per_token: u64,
}

pub fn vram_estimate(dir: &Path, cfg: &Value, facts: &model_facts::ModelFacts, extra: &[String], kv_dtype: &str) -> VramEstimate {
    // Weights left on the GPU after host offload, per-request state for 4 requests, and KV bytes per token.
    let kv_elem_bytes = kv_element_bytes(kv_dtype, cfg);
    let ple_offload = (sgl::has_flag(extra, "--ple-offload-embedding") || config_ple_offload(cfg))
        && !sgl::has_flag(extra, "--no-ple-offload-embedding");
    let ple_gb = if ple_offload {
        safetensors::tensor_bytes_in_dir(dir, &safetensors::is_ple_embedding).unwrap_or_else(|e| {
            eprintln!("warning: offloaded embedding size unknown: {e}");
            0
        }) as f64
            / GIB
    } else {
        0.0
    };
    let cpu_offload_gb = sgl::flag_value(extra, "--cpu-offload-gb")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.0);
    let host_offload_gb = ple_gb + cpu_offload_gb;
    VramEstimate {
        weights_gb: (facts.static_vram_gb - host_offload_gb).max(0.0),
        state_gb: (facts.swa_kv_elems_per_request * kv_elem_bytes + facts.linear_state_bytes_per_request) as f64 * 4.0 / GIB,
        host_offload_gb,
        kv_bytes_per_token: facts.kv_elems_per_token * kv_elem_bytes,
    }
}

pub struct Planned {
    model_id: String,
    dir: PathBuf,
    rel: String,
    table: toml::Table,
    facts: model_facts::ModelFacts,
    vram: VramEstimate,
    kv_dtype: String,
    ctx: u64,
    max_conc: u64,
    tp: u64,
    extra: Vec<String>,
    notes: Vec<String>,
    managed: bool,
    source_dir: Option<PathBuf>,
    quant: Option<String>,
    engine: Option<String>,
}

impl Planned {
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn static_gb(&self) -> f64 {
        self.vram.weights_gb + self.vram.state_gb
    }
}

fn register(home: &Path, opts: &InstallOpts, mut t: Target) -> Result<(), String> {
    // The card is written last: any failure before it (pack refused, SGLang setup failed) leaves no card behind.
    let mut planned = plan(home, opts, &t)?;
    let refused = match &t.pack {
        Some(choice) => use_pack(home, &mut planned, choice, opts.ctx)?,
        None if planned.dir.join(PACK_VERIFY).is_file() => {
            let v: Value = serde_json::from_slice(&fs::read(planned.dir.join(PACK_VERIFY)).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
            pack_card(home, &mut planned, v["arch"].as_str().unwrap_or_default(), "", opts.ctx)?
        }
        None => None,
    };
    if let Some(why) = refused {
        drop_pack(&planned.dir);
        if !fallback_allowed(opts.fallback) {
            return Err(format!("{why}; {FALLBACK_HINT}"));
        }
        t.pack = None;
        planned = plan(home, opts, &t)?;
        println!("{why}; using {}", crate::engines::of_card(&toml::Value::Table(planned.table.clone()))?.id);
    }
    ensure_runtime(home, crate::engines::of_card(&toml::Value::Table(planned.table.clone()))?)?;
    commit(home, &planned, "installed", &mut std::io::stdout())
}

fn ensure_runtime(home: &Path, engine: &crate::engines::Engine) -> Result<(), String> {
    // The card's engine, set up once on first use: its py-env layer, or the pinned prebuilt llama-server.
    if let Some(layer) = engine.layer {
        crate::pyenv::ensure(home, layer)?;
    }
    if engine.id == crate::engines::llama_cpp::ENGINE.id {
        crate::engines::llama_cpp::ensure_server(home)?;
    }
    Ok(())
}

fn drop_pack(dir: &Path) {
    // The pack files goethe placed in the model dir (copies or links; never the user's own files behind a link).
    for f in [PACK_FILE.to_string(), format!("{PACK_FILE}.sig"), PACK_VERIFY.to_string(), "pack.prefix".to_string()] {
        let _ = fs::remove_file(dir.join(f));
    }
}

fn refused_here(dir: &Path, run: crate::packs::Runner) -> Option<String> {
    // goethe-engine `pack runs` on the installed pack: why this device runs no variant of it, in one line; None when it runs one or the engine cannot say (older engine).
    pack_runs(&dir.join(PACK_FILE), None, run)
}

fn pack_runs(file: &Path, sig: Option<&Path>, run: crate::packs::Runner) -> Option<String> {
    // `pack runs` (exit 3: this device runs no variant) on a pack or on its header and manifest prefix; the SKUs come from the engine's own text when it names none.
    let mut args = vec!["pack".to_string(), "runs".into(), file.to_string_lossy().into_owned()];
    if let Some(s) = sig {
        args.extend(["--sig".into(), s.to_string_lossy().into_owned()]);
    }
    let (code, out) = run(&args).ok()?;
    if code != 3 {
        return None;
    }
    let v: Value = serde_json::from_str(&out).unwrap_or_default();
    if let Value::Array(a) = &v["sku"] {
        let skus = a.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", ");
        return Some(format!("kernel pack for {skus} does not run on {} ({})", v["device"].as_str().unwrap_or("this GPU"), v["why"].as_str().unwrap_or("no variant for it")));
    }
    Some(load_refusal(v["why"].as_str().unwrap_or_default()).unwrap_or_else(|| format!("kernel pack for its GPUs does not run on {}", v["device"].as_str().unwrap_or("this GPU"))))
}

fn refused_before_download(home: &Path, plan: &crate::packs::Plan, run: crate::packs::Runner) -> Option<String> {
    // The pack's header and manifest are already fetched: ask the engine whether this GPU runs any variant before a single weight byte is downloaded.
    let work = home.join("packs");
    let (prefix, sig) = (work.join("check.prefix"), work.join("check.sig"));
    fs::write(&prefix, &plan.prefix).ok()?;
    fs::write(&sig, serde_json::to_vec(&plan.entry.sig).ok()?).ok()?;
    let why = pack_runs(&prefix, Some(&sig), run);
    let _ = fs::remove_file(&prefix);
    let _ = fs::remove_file(&sig);
    why
}

fn pack_only_gate(model_id: &str, pack_only: bool, declined: bool, no_pack_gpu: Option<&str>, built: &str) -> Result<(), String> {
    // A pack-only model has no plain route: a node that is not logged in, or whose GPUs the packs are not built for, is refused before any download.
    if !pack_only {
        return Ok(());
    }
    if declined {
        let cards = if built.is_empty() { String::new() } else { format!(" (packs are built for {built})") };
        return Err(format!("{model_id} runs only on its kernel pack: run goethe login first{cards}; nothing was downloaded"));
    }
    match no_pack_gpu {
        Some(why) => Err(format!("{model_id}: {why}")),
        None => Ok(()),
    }
}

fn pack_only_refusal(model_id: &str, pack_only: bool, why: &str, allow: bool) -> Result<String, String> {
    // A pack that cannot run here before any download: a pack-only model is refused with the cards its pack is built for; any other is refused too unless the user opted into its plain runtime.
    if pack_only {
        return Err(format!("{model_id} runs only with its kernel pack, and {why}; nothing was downloaded"));
    }
    if !allow {
        return Err(format!("{why}; nothing was downloaded; {FALLBACK_HINT}"));
    }
    Ok(format!("{why}; using its plain runtime"))
}

fn pack_failure_line(home: &Path, model_id: &str, log: &str) -> Option<String> {
    // For a model that carries a kernel pack: why the pack did not open, in one line, and how to opt into the plain runtime when it has one.
    let dir = model_dir(home, model_id);
    if !dir.join(PACK_FILE).is_file() {
        return None;
    }
    let why = load_failure(log);
    Some(if has_plain_weights(&dir) { format!("{why}; {FALLBACK_HINT}") } else { why })
}

pub fn unpack_failed(home: &Path, model_id: &str, log: &str) -> Result<bool, String> {
    // A pack that failed to open for any reason: drop it, write the plain card with a pack_skipped note, say why in one line; false when there is no pack or no plain runtime to fall back to.
    let dir = model_dir(home, model_id);
    if !dir.join(PACK_FILE).is_file() || !has_plain_weights(&dir) {
        return Ok(false);
    }
    let why = load_failure(log);
    drop_pack(&dir);
    let source_dir = read_card_table(&dir).and_then(|c| c.get("source_dir").and_then(|v| v.as_str()).map(PathBuf::from));
    let entry = fetch(&registry_url()).ok().and_then(|r| find_model(&r, model_id).cloned());
    let t = Target { model_id: model_id.to_string(), card_dir: dir, entry, managed: source_dir.is_none(), source_dir, ternary: None, pack: None };
    let mut p = plan(home, &InstallOpts { model_id: model_id.to_string(), ..InstallOpts::default() }, &t)?;
    let engine = crate::engines::of_card(&toml::Value::Table(p.table.clone()))?;
    ensure_runtime(home, engine)?;
    p.table.insert(PACK_SKIPPED.into(), toml::Value::String(why.clone()));
    println!("{why}; using {}", engine.id);
    commit(home, &p, "installed", &mut std::io::sink())?;
    Ok(true)
}

fn has_plain_weights(dir: &Path) -> bool {
    // Weights a runtime other than the pack can serve sit in the model dir: a GGUF, or safetensors.
    let card = toml::Value::Table(read_card_table(dir).unwrap_or_default());
    crate::engines::llama_cpp::gguf_in(dir, &card, None).is_ok() || model_facts::weight_files(dir).is_ok_and(|w| !w.is_empty())
}

type Cause = fn(&str) -> bool;

fn load_failure(log: &str) -> String {
    // Why a pack did not open, in one line: the engine's device refusal, else the first log line naming a known cause, else the last log line.
    if let Some(refusal) = load_refusal(log) {
        return refusal;
    }
    let causes: [(Cause, &str); 4] = [
        (|l| l.contains("cannot open shared object file"), "a host library is missing"),
        (|l| l.contains("Illegal instruction") || l.contains("SIGILL"), "it crashed at open (illegal instruction)"),
        (|l| l.contains("Segmentation fault") || l.contains("SIGSEGV"), "it crashed at open"),
        (|l| l.contains("needs ") && l.contains("GB"), "it does not fit this GPU"),
    ];
    let lines: Vec<&str> = log.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let short = |l: &str| l.chars().take(160).collect::<String>();
    for (hit, cause) in causes {
        if let Some(l) = lines.iter().find(|l| hit(l)) {
            return format!("kernel pack failed to open ({cause}: {})", short(l));
        }
    }
    format!("kernel pack failed to open (unknown error: {})", lines.last().map_or("no log".to_string(), |l| short(l)))
}

fn load_refusal(log: &str) -> Option<String> {
    // The engine's "device D runs no variant of this pack: SKUS (...): WHY. Fallback: ..." line as "kernel pack for SKUS does not run on D (WHY)".
    let line = log.lines().find(|l| l.contains("runs no variant of this pack"))?;
    let (device, rest) = line.split_once(" runs no variant of this pack")?;
    let device = device.rsplit_once("device ").map_or(device, |(_, d)| d).trim();
    let rest = rest.trim_start_matches(':').trim();
    let (skus, why) = rest.split_once("): ").map_or((rest.to_string(), "no variant for it"), |(s, w)| (format!("{s})"), w.split(". ").next().unwrap_or(w)));
    Some(format!("kernel pack for {skus} does not run on {device} ({why})"))
}

fn use_pack(home: &Path, p: &mut Planned, c: &PackChoice, cli_ctx: Option<u64>) -> Result<Option<String>, String> {
    // Pull with this node's login (goethe-engine verifies index, identity, pack and arch), then point the card at it.
    let login = crate::packs::Login(c.creds.clone());
    let broker = crate::packs::HttpBroker::new(&c.creds.platform_url, &login);
    let run = crate::packs::python_runner(home);
    let plan = c.plan.as_ref().ok_or("no kernel pack plan")?;
    println!("installing the optimised kernels for {} ...", c.offer.arch);
    let variant = crate::packs::finish(home, &p.dir, &c.offer.arch, plan, &broker, &run)?;
    pack_card(home, p, &c.offer.arch, &variant, cli_ctx)
}

fn pack_card(home: &Path, p: &mut Planned, arch: &str, variant: &str, cli_ctx: Option<u64>) -> Result<Option<String>, String> {
    // The card of a verified pack: goethe-engine, the kernels' ctx cap, the pack arch's GPUs; Some(why) when this device runs no variant of it.
    let run = crate::packs::python_runner(home);
    if let Some(why) = refused_here(&p.dir, &run) {
        return Ok(Some(why));
    }
    let max = crate::packs::max_ctx(&p.dir, variant, &run)?;
    if cli_ctx.is_some_and(|x| x > max) {
        p.notes.push(format!("ctx {} is longer than the kernels support; using {max}", cli_ctx.unwrap_or(max)));
    }
    p.ctx = p.ctx.min(max);
    p.extra.clear();
    p.notes.retain(|n| !n.contains("SGLang") && !n.contains("--"));
    p.quant = None;
    p.kv_dtype = "auto".into();
    let t = &mut p.table;
    t.insert("runtime".into(), toml::Value::String("goethe-engine".into()));
    t.insert("context_length".into(), toml::Value::Integer(p.ctx as i64));
    t.insert("extra_sglang_args".into(), toml::Value::Array(Vec::new()));
    t.insert("kv_cache_dtype".into(), toml::Value::String("auto".into()));
    t.insert("supported_sm".into(), toml::Value::Array(crate::packs::arch_ccs(arch).into_iter().map(toml::Value::String).collect()));
    p.engine = Some(format!("goethe-engine with optimised kernels for {arch} (verified)"));
    Ok(None)
}

pub fn plan_in_place(home: &Path, model_id: &str, dir: &Path, entry: Option<RegistryModel>) -> Result<Planned, String> {
    // A directory already inside models/ (goethe models scan): managed, card written next to the weights.
    let opts = InstallOpts {
        model_id: model_id.to_string(),
        ..InstallOpts::default()
    };
    let t = Target {
        model_id: model_id.to_string(),
        card_dir: dir.to_path_buf(),
        entry,
        managed: true,
        source_dir: None,
        ternary: None,
        pack: None,
    };
    plan(home, &opts, &t)
}

fn plan_ninfer(home: &Path, opts: &InstallOpts, t: &Target) -> Result<Planned, String> {
    // The pinned artifact is the model. The card says runtime ninfer so load does not start SGLang.
    let dir = &t.card_dir;
    let path = crate::engines::ninfer::ninfer_in(dir).ok_or_else(crate::engines::ninfer::ninfer_reject)?;
    if fs::metadata(&path).map_err(|e| e.to_string())?.len() != crate::engines::ninfer::NINFER_BYTES {
        return Err(crate::engines::ninfer::ninfer_reject());
    }
    let entry = t.entry.as_ref();
    let facts = model_facts::ModelFacts {
        architecture: entry.map(|e| e.architecture.clone()).filter(|s| !s.is_empty()).unwrap_or_else(|| "dense".into()),
        static_vram_gb: entry.map(|e| e.static_vram_gb).filter(|v| *v > 0.0).unwrap_or(23.566),
        kv_elems_per_token: entry.and_then(|e| e.effective_kv_elems()).unwrap_or(32_768),
        max_ctx: entry.and_then(|e| e.effective_max_ctx()).unwrap_or(262_144),
        num_attention_heads: entry.and_then(|e| e.num_attention_heads).unwrap_or(24),
        num_kv_heads: entry.and_then(|e| e.num_kv_heads).unwrap_or(4),
        attention: entry.and_then(|e| e.attention.clone()).unwrap_or_else(|| "hybrid".into()),
        swa_kv_elems_per_request: 0,
        linear_state_bytes_per_request: 0,
    };
    plan_engine(home, opts, t, facts, crate::engines::ninfer::ENGINE.id, ("fp8_e4m3", 1), Some("nvfp4".into()))
}

fn plan_gguf(home: &Path, opts: &InstallOpts, t: &Target) -> Result<Planned, String> {
    // llama.cpp serves the GGUF: shape from its header, KV in f16, the file recorded in the card; a pooling model as embeddings, a projector with an audio encoder as speech.
    crate::engines::llama_cpp::server_ready(home)?;
    let dir = &t.card_dir;
    let gguf = crate::engines::llama_cpp::gguf_in(dir, &toml::Value::Table(read_card_table(dir).unwrap_or_default()), opts.quant.as_deref())?;
    let facts = crate::engines::llama_cpp::facts(dir, &gguf)?;
    let mut p = plan_engine(home, opts, t, facts, crate::engines::llama_cpp::ENGINE.id, ("f16", 2), None)?;
    let rel = gguf.strip_prefix(dir).unwrap_or(&gguf).to_string_lossy().into_owned();
    p.table.insert("gguf".into(), toml::Value::String(rel));
    if crate::engines::llama_cpp::is_embedding(&gguf) {
        p.table.insert("task".into(), toml::Value::String("embed".into()));
    }
    if let Some(proj) = crate::engines::llama_cpp::mmproj_in(dir) {
        p.table.insert("mmproj".into(), toml::Value::String(proj.strip_prefix(dir).unwrap_or(&proj).to_string_lossy().into_owned()));
        if crate::engines::llama_cpp::has_audio_encoder(&proj) {
            p.table.insert("task".into(), toml::Value::String("transcribe".into()));
        }
    }
    Ok(p)
}

fn plan_engine(
    home: &Path,
    opts: &InstallOpts,
    t: &Target,
    facts: model_facts::ModelFacts,
    runtime: &str,
    (kv_dtype, kv_elem_bytes): (&str, u64),
    quant: Option<String>,
) -> Result<Planned, String> {
    // The card of an engine that takes no SGLang arguments: the model's facts, its KV dtype, 4 parallel requests by default.
    let dir = &t.card_dir;
    let entry = t.entry.as_ref();
    let ctx = opts.ctx.unwrap_or(facts.max_ctx).min(facts.max_ctx);
    let max_conc = opts.max_conc.unwrap_or(4);
    let card = GoetheCard {
        model_id: t.model_id.clone(),
        context_length: ctx,
        max_running_requests: max_conc,
        tp_size: opts.tp.unwrap_or(1),
        ep_size: 1,
        kv_cache_dtype: kv_dtype.into(),
        preferred_gpus: opts.gpus.clone().unwrap_or_default(),
        extra_sglang_args: Vec::new(),
        runtime: Some(runtime.into()),
        pack_id: None,
        base_model_id: None,
        env: None,
        pythonpath: None,
        marlin_extra_gb: None,
        host_offload_gb: None,
        source_dir: t.source_dir.as_ref().map(|p| p.display().to_string()),
        supported_sm: entry.map(|e| e.supported_sm.clone()).filter(|sm| !sm.is_empty()),
    };
    let rel = dir
        .strip_prefix(home)
        .map(|r| r.display().to_string())
        .map_err(|_| format!("{} is not under {}", dir.display(), home.display()))?;
    Ok(Planned {
        model_id: t.model_id.clone(),
        dir: dir.clone(),
        rel,
        table: card_table(&card, None)?,
        vram: VramEstimate {
            weights_gb: facts.static_vram_gb,
            state_gb: 0.0,
            host_offload_gb: 0.0,
            kv_bytes_per_token: facts.kv_elems_per_token * kv_elem_bytes,
        },
        facts,
        kv_dtype: kv_dtype.into(),
        ctx,
        max_conc,
        tp: card.tp_size,
        extra: Vec::new(),
        notes: Vec::new(),
        managed: t.managed,
        source_dir: t.source_dir.clone(),
        quant,
        engine: None,
    })
}

fn plan(home: &Path, opts: &InstallOpts, t: &Target) -> Result<Planned, String> {
    // Compute the card (merged with an existing one unless --reset), facts and VRAM; write nothing.
    let dir = &t.card_dir;
    if crate::engines::ninfer::ninfer_in(dir).is_some() {
        return plan_ninfer(home, opts, t);
    }
    let pack = dir.join(PACK_FILE).is_file() || t.pack.is_some();
    if !pack && !model_facts::weight_files(dir).is_ok_and(|w| !w.is_empty()) && !crate::engines::llama_cpp::gguf_paths(dir).is_empty() {
        return plan_gguf(home, opts, t);
    }
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
    let mut inferred = Vec::new();
    for (flag, value, why) in sgl::infer_parsers(&cfg, dir) {
        if [&base_args, &old_args, &cli_args].iter().all(|a| !sgl::has_flag(a, &flag)) {
            notes.push(format!("{flag} {value} inferred: {why} (change it with goethe models config --sglang-arg)"));
            inferred.extend([flag, value]);
        }
    }
    let ordered = sgl::merge_extra_args(&sgl::merge_extra_args(&inferred, &base_args)?, &old_args)?;
    let extra = sgl::merge_extra_args(&ordered, &cli_args).map_err(|e| format!("{e} (use --reset to rebuild the card)"))?;
    let kv_dtype = cli_kv
        .or_else(|| old("kv_cache_dtype").and_then(|v| v.as_str()).map(String::from))
        .or_else(|| t.entry.as_ref().and_then(|e| e.kv_cache_dtype.clone()))
        .or_else(|| checkpoint_kv_dtype(&cfg, dir))
        .unwrap_or_else(|| "auto".to_string());
    let mut vram = vram_estimate(dir, &cfg, &facts, &extra, &kv_dtype);
    if let (Some(_), Some(e)) = (&t.ternary, &t.entry) {
        if e.static_vram_gb > 0.0 {
            vram.weights_gb = e.static_vram_gb;
        }
    }
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
    let ctx = pick(opts.ctx, "context_length", None, entry.and_then(|e| e.effective_max_ctx()).unwrap_or(facts.max_ctx).min(facts.max_ctx));
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
        host_offload_gb: (vram.host_offload_gb > 0.0).then_some(vram.host_offload_gb),
        source_dir: t.source_dir.as_ref().map(|p| p.display().to_string()),
        supported_sm: entry.map(|e| e.supported_sm.clone()).filter(|sm| !sm.is_empty()),
    };
    let rel = dir
        .strip_prefix(home)
        .map(|r| r.display().to_string())
        .map_err(|_| format!("{} is not under {}", dir.display(), home.display()))?;
    Ok(Planned {
        model_id: t.model_id.clone(),
        dir: dir.clone(),
        rel,
        table: card_table(&card, existing.as_ref())?,
        quant: sgl::quantization_summary(dir, &cfg),
        facts,
        vram,
        kv_dtype,
        ctx,
        max_conc,
        tp,
        extra,
        notes,
        managed: t.managed,
        source_dir: t.source_dir.clone(),
        engine: None,
    })
}

pub fn commit(home: &Path, p: &Planned, verb: &str, out: &mut dyn Write) -> Result<(), String> {
    // Write the card atomically, upsert installed_models, and report what was registered.
    write_card_table(&p.dir, &p.table)?;
    let con = db::open(home).map_err(|e| e.to_string())?;
    con.execute(
        "INSERT OR REPLACE INTO installed_models
        (model_id, local_path, architecture, static_vram_gb, kv_bytes_per_token,
         default_ctx, default_max_conc, default_tp, kv_elems_per_token, max_ctx,
         num_attention_heads, num_kv_heads, attention, managed, installed_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, datetime('now'))",
        rusqlite::params![
            p.model_id,
            p.rel,
            p.facts.architecture,
            p.static_gb(),
            p.vram.kv_bytes_per_token as i64,
            p.ctx as i64,
            p.max_conc as i64,
            p.tp as i64,
            p.facts.kv_elems_per_token as i64,
            p.facts.max_ctx as i64,
            p.facts.num_attention_heads as i64,
            p.facts.num_kv_heads as i64,
            p.facts.attention,
            p.managed as i64,
        ],
    )
    .map_err(|e| format!("state.db: {e}"))?;
    if let Some(note) = drop_shadowing_alias(&con, &p.model_id)? {
        eprintln!("warning: {note}");
    }
    let mut lines = vec![format!("{verb} {}", p.model_id), format!("  path      {}", p.dir.display())];
    lines.push(match &p.source_dir {
        Some(src) => format!("  files     kept in {} (goethe remove only unregisters)", src.display()),
        None => "  files     managed by goethe (goethe remove deletes them)".to_string(),
    });
    let state_note = if p.vram.state_gb > 0.0 {
        format!(" + {:.1} GB per-request state for 4 requests", p.vram.state_gb)
    } else {
        String::new()
    };
    lines.push(format!("  vram      {:.1} GB weights{state_note}", p.vram.weights_gb));
    if p.vram.host_offload_gb > 0.0 {
        lines.push(format!("  host ram  {:.1} GB offloaded weights in pinned memory", p.vram.host_offload_gb));
    }
    lines.push(format!(
        "  kv cache  {:.1} KB per token ({}, {} attention)",
        p.vram.kv_bytes_per_token as f64 / 1024.0,
        p.kv_dtype,
        p.facts.attention
    ));
    if let Some(e) = &p.engine {
        lines.push(format!("  engine    {e}"));
    }
    if let Some(q) = &p.quant {
        lines.push(format!("  quant     {q}, detected by SGLang"));
    }
    let tp_s = if p.tp == 0 { "auto".to_string() } else { p.tp.to_string() };
    lines.push(format!("  defaults  ctx {}, max-conc {}, tp {tp_s}", p.ctx, p.max_conc));
    if !p.extra.is_empty() {
        lines.push(format!("  sglang    {}", p.extra.join(" ")));
    }
    lines.extend(p.notes.iter().map(|n| format!("  note      {n}")));
    writeln!(out, "{}", lines.join("\n")).map_err(|e| e.to_string())
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
    let _ = fs::remove_file(&tmp);
    let mut f = fs::OpenOptions::new().write(true).create_new(true).open(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    f.write_all(text.as_bytes()).map_err(|e| format!("{}: {e}", tmp.display()))?;
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
        "fallback": opts.fallback,
    });
    run_load(params, opts.no_wait, false)
}

pub fn drop_shadowing_alias(con: &rusqlite::Connection, model_id: &str) -> Result<Option<String>, String> {
    // An alias spelled like a newly installed model id would compete with it for that name: remove it and say so.
    let target: Option<String> = con
        .query_row("SELECT model_id FROM model_aliases WHERE alias = ?1", [model_id], |r| r.get(0))
        .optional()
        .map_err(|e| format!("state.db: {e}"))?;
    let Some(target) = target else {
        return Ok(None);
    };
    con.execute("DELETE FROM model_aliases WHERE alias = ?1", [model_id])
        .map_err(|e| format!("state.db: {e}"))?;
    Ok(Some(format!(
        "removed alias {model_id} (it pointed to {target}): the name now belongs to the installed model {model_id}"
    )))
}

pub fn run_load(params: Value, no_wait: bool, json_out: bool) -> Result<(), String> {
    // Send the load, show the plan, wait for this instance, and end with one RUNNING or STARTING line (or JSON).
    let started = std::time::Instant::now();
    let model_id = params.get("model_id").and_then(Value::as_str).unwrap_or_default().to_string();
    crate::pack_update::note_if_newer(&goethe_home(), &model_id);
    let reply = send_load(params.clone())?;
    let message = reply_message(&reply);
    if is_already_loaded(&reply) {
        let fallback = format!("{model_id} is already loaded; add --replica to start another instance");
        println!("{}", if message.is_empty() { fallback } else { message });
        return Ok(());
    }
    if !message.is_empty() {
        if json_out {
            eprintln!("{message}");
        } else {
            println!("{message}");
        }
    }
    let iid = reply
        .result
        .get("instance_id")
        .and_then(Value::as_str)
        .ok_or_else(|| "load did not return instance_id".to_string())?
        .to_string();
    if !no_wait {
        if let Err(e) = crate::load_wait::wait_for_instance(&iid, crate::load_wait::DEFAULT_LOAD_WAIT) {
            let home = goethe_home();
            let log = fs::read_to_string(home.join("logs").join(format!("{iid}.log"))).unwrap_or_default();
            let failed = e.starts_with("the engine failed to start");
            let allow = fallback_allowed(params.get("fallback").and_then(Value::as_bool) == Some(true));
            if failed && allow && unpack_failed(&home, &model_id, &log)? {
                let _ = crate::rpc::call("unload", json!({"model_id": model_id, "all": false, "force": true}));
                return run_load(params, no_wait, json_out);
            }
            return Err(match failed.then(|| pack_failure_line(&home, &model_id, &log)).flatten() {
                Some(line) => format!("{e}\n{line}"),
                None => e,
            });
        }
    }
    let row = instance_row(&iid);
    if json_out {
        let out = if no_wait { &reply.result } else { row.as_ref().unwrap_or(&reply.result) };
        println!("{out}");
    } else {
        let secs = (!no_wait).then(|| started.elapsed().as_secs_f64());
        println!("{}", load_status_line(&model_id, &iid, row.as_ref(), &reply.result, secs));
    }
    Ok(())
}

fn reply_message(reply: &crate::rpc::RpcReply) -> String {
    // The daemon's human lines: stdout without the trailing result JSON.
    let json = serde_json::to_string(&reply.result).unwrap_or_default();
    let text = if reply.result.is_null() { reply.stdout.as_str() } else { reply.stdout.strip_suffix(&json).unwrap_or(&reply.stdout) };
    text.trim().to_string()
}

pub fn is_already_loaded(reply: &crate::rpc::RpcReply) -> bool {
    // Contract C-A: a plain load of a loaded model answers already_loaded instead of adding a replica.
    let r = &reply.result;
    r.get("already_loaded").and_then(Value::as_bool) == Some(true)
        || r.get("status").and_then(Value::as_str) == Some("already_loaded")
        || r.as_str() == Some("already_loaded")
}

fn instance_row(instance_id: &str) -> Option<Value> {
    let ps = crate::rpc::call("ps", Value::Null).ok()?;
    let rows: Value = serde_json::from_str(ps.stdout.trim()).ok()?;
    rows.as_array()?
        .iter()
        .find(|r| r.get("instance_id").and_then(Value::as_str) == Some(instance_id))
        .cloned()
}

pub fn load_status_line(model_id: &str, iid: &str, row: Option<&Value>, result: &Value, secs: Option<f64>) -> String {
    // RUNNING: <id> instance <iid> on GPU <g> port <p> (<t> s), or STARTING for --no-wait.
    let gpus = row
        .and_then(|r| r.get("gpu_indices"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_u64).map(|g| g.to_string()).collect::<Vec<_>>().join(","))
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "?".into());
    let port = row
        .and_then(|r| r.get("internal_port"))
        .or_else(|| result.get("port"))
        .and_then(Value::as_u64)
        .map_or_else(|| "?".to_string(), |p| p.to_string());
    match secs {
        Some(s) => format!("RUNNING: {model_id} instance {iid} on GPU {gpus} port {port} ({s:.1} s)"),
        None => format!("STARTING: {model_id} instance {iid} on GPU {gpus} port {port} (goethe ps shows when it is RUNNING)"),
    }
}

pub fn send_load(mut params: Value) -> Result<crate::rpc::RpcReply, String> {
    // Load RPC; on a foreign-process notice (code 3) ask on a terminal and resend with yes=true, else exit 3.
    params["is_tty"] = json!(false);
    let call = |p: &Value| crate::rpc::call("load", p.clone()).map_err(crate::paths::daemon_err);
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
    if reply.code != 0 && !is_already_loaded(&reply) {
        print_reply_text(&reply);
        std::process::exit(reply.code);
    }
    Ok(reply)
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

    fn gguf_home(server: bool) -> (tempfile::TempDir, PathBuf) {
        // A GOETHE_HOME with a GGUF-only model dir, and llama-server installed when asked.
        let home = tempfile::tempdir().unwrap();
        let dir = home.path().join("models/org/m-GGUF");
        fs::create_dir_all(&dir).unwrap();
        crate::engines::llama_cpp::write_test_gguf(&dir.join("m-Q4_K_M.gguf"));
        if server {
            let bin = home.path().join("engines/llama.cpp/bin");
            fs::create_dir_all(&bin).unwrap();
            fs::write(bin.join("llama-server"), b"").unwrap();
        }
        (home, dir)
    }

    #[test]
    fn a_pack_this_device_runs_no_variant_of_is_named_or_left_to_load() {
        let dir = tempfile::tempdir().unwrap();
        let no = |_: &[String]| Ok((3, r#"{"runs": false, "sku": ["rtx_pro_6000_maxq"], "device": "RTX 5090", "sms": 170, "why": "188 SMs needed"}"#.to_string()));
        let why = refused_here(dir.path(), &no).unwrap();
        assert_eq!(why, "kernel pack for rtx_pro_6000_maxq does not run on RTX 5090 (188 SMs needed)");
        let yes = |_: &[String]| Ok((0, r#"{"runs": true}"#.to_string()));
        assert_eq!(refused_here(dir.path(), &yes), None);
        let old = |_: &[String]| Err("goethe-engine pack runs: exit 2: invalid choice: 'runs'".to_string());
        assert_eq!(refused_here(dir.path(), &old), None, "an engine without pack runs: as before, load decides");
    }

    fn packed_home() -> (tempfile::TempDir, PathBuf) {
        // A GGUF model dir carrying a kernel pack, with a goethe-engine card.
        let (home, _) = gguf_home(true);
        let dir = model_dir(home.path(), "org/m-GGUF");
        fs::create_dir_all(&dir).unwrap();
        crate::engines::llama_cpp::write_test_gguf(&dir.join("m-Q4_K_M.gguf"));
        fs::write(dir.join("config.json"), "{}").unwrap();
        for f in [PACK_FILE, PACK_VERIFY, "pack.gcpack.sig"] {
            fs::write(dir.join(f), b"p").unwrap();
        }
        write_card_table(&dir, &toml::from_str("model_id = \"org/m-GGUF\"\nruntime = \"goethe-engine\"").unwrap()).unwrap();
        (home, dir)
    }

    #[test]
    fn a_pack_refused_at_load_leaves_a_plain_card_once() {
        let (home, dir) = packed_home();
        let log = "goethe-engine: pack REFUSED: device NVIDIA GeForce RTX 5090 cc 12.0 (170 SMs, smem per block 101376) runs no variant of this pack: rtx_pro_6000_maxq (sm_120a, 188 SMs): the plan's grid needs 188 SMs co-resident, the device has 170. Fallback: serve this model with llama.cpp\n";
        let why = "kernel pack for rtx_pro_6000_maxq (sm_120a, 188 SMs) does not run on NVIDIA GeForce RTX 5090 cc 12.0 (170 SMs, smem per block 101376) (the plan's grid needs 188 SMs co-resident, the device has 170)";
        assert_eq!(load_failure(log), why);
        assert!(unpack_failed(home.path(), "org/m-GGUF", log).unwrap());
        assert!(!dir.join(PACK_FILE).exists() && !dir.join(PACK_VERIFY).exists() && dir.join("m-Q4_K_M.gguf").exists());
        let card = read_card_table(&dir).unwrap();
        assert_eq!(card.get("runtime").and_then(|v| v.as_str()), Some("llama.cpp"));
        assert_eq!(card.get(PACK_SKIPPED).and_then(|v| v.as_str()), Some(why));
        assert!(!unpack_failed(home.path(), "org/m-GGUF", log).unwrap(), "no pack left: no second retry");
    }

    #[test]
    fn a_pack_that_runs_no_variant_here_is_known_before_any_download() {
        let home = tempfile::tempdir().unwrap();
        let reply = r#"{"runs": false, "sku": null, "device": "NVIDIA GeForce RTX 5090", "sms": 170, "why": "device NVIDIA GeForce RTX 5090 cc 12.0 (170 SMs, smem per block 101376) runs no variant of this pack: rtx_pro_6000_maxq (sm_120a, 188 SMs): the plan's grid needs 188 SMs co-resident, the device has 170. Fallback: serve this model with llama.cpp"}"#;
        let no = |args: &[String]| {
            assert!(args.contains(&"--sig".to_string()), "a prefix is checked against its signature");
            Ok((3, reply.to_string()))
        };
        let why = pack_runs(&home.path().join("p.prefix"), Some(&home.path().join("p.sig")), &no).unwrap();
        assert!(why.starts_with("kernel pack for rtx_pro_6000_maxq (sm_120a, 188 SMs) does not run on NVIDIA GeForce RTX 5090"), "{why}");
        let yes = |_: &[String]| Ok((0, r#"{"runs": true}"#.to_string()));
        assert_eq!(pack_runs(&home.path().join("p.prefix"), None, &yes), None);
        let old = |_: &[String]| Err("exit 2: invalid choice: 'runs'".to_string());
        assert_eq!(pack_runs(&home.path().join("p.prefix"), None, &old), None, "an engine without pack runs: load decides");
    }

    #[test]
    fn a_pack_only_model_needs_a_login_and_a_card_its_pack_is_built_for() {
        let built = "sm_120a, Blackwell (RTX 5090, RTX PRO 6000)";
        let out = pack_only_gate("nvidia/Qwen3.8-27B-NVFP4", true, true, None, built).unwrap_err();
        assert!(out.contains("goethe login first") && out.contains("sm_120a") && out.ends_with("nothing was downloaded"), "{out}");
        let card = pack_only_gate("nvidia/Qwen3.8-27B-NVFP4", true, false, Some("It runs only with a kernel pack, built for sm_120a; this machine has sm_89"), built).unwrap_err();
        assert!(card.starts_with("nvidia/Qwen3.8-27B-NVFP4: It runs only with a kernel pack") && card.contains("sm_89"), "{card}");
        assert_eq!(pack_only_gate("nvidia/Qwen3.8-27B-NVFP4", true, false, None, built), Ok(()), "logged in with a pack for the card: on to the download plan");
        for gguf in [(true, None), (false, Some("no variant"))] {
            assert_eq!(pack_only_gate("ggml-org/embeddinggemma-2-GGUF", false, gguf.0, gguf.1, built), Ok(()), "a plain GGUF route keeps its own not-logged-in note");
        }
    }

    #[test]
    fn a_pack_only_model_is_refused_before_download_and_any_other_uses_its_plain_runtime() {
        let why = "kernel pack for rtx_pro_6000_maxq (sm_120a, 188 SMs) does not run on NVIDIA GeForce RTX 5090 (no fit)";
        let err = pack_only_refusal("nvidia/Qwen3.8-27B-NVFP4", true, why, true).unwrap_err();
        assert!(err.contains("runs only with its kernel pack") && err.contains("rtx_pro_6000_maxq (sm_120a") && err.ends_with("nothing was downloaded"), "{err}");
        let held = pack_only_refusal("org/m-GGUF", false, why, false).unwrap_err();
        assert!(held.contains("nothing was downloaded") && held.contains("--fallback") && held.contains("GOETHE_ALLOW_FALLBACK"), "{held}");
        let note = pack_only_refusal("org/m-GGUF", false, why, true).unwrap();
        assert!(note.ends_with("using its plain runtime"), "{note}");
    }

    #[test]
    fn the_catalog_marks_the_pack_only_models() {
        let reg = crate::registry::bundled().unwrap();
        assert!(reg.models.iter().find(|m| m.id == "nvidia/Qwen3.8-27B-NVFP4").is_some_and(|m| m.pack_only == Some(true)));
        assert!(reg.models.iter().filter(|m| m.pack_only == Some(true)).count() >= 1);
    }

    #[test]
    fn a_pack_failure_is_one_line_with_the_opt_in_hint_and_fallback_needs_an_opt_in() {
        let (home, _) = packed_home();
        let line = pack_failure_line(home.path(), "org/m-GGUF", "ImportError: libnccl.so.2: cannot open shared object file").unwrap();
        assert!(line.starts_with("kernel pack failed to open (a host library is missing") && line.ends_with("instead"), "{line}");
        assert!(line.contains("--fallback") && !line.contains('\n'), "{line}");
        assert_eq!(pack_failure_line(home.path(), "org/none", "x"), None, "no pack: nothing to add");
        assert!(allowed(true, None) && allowed(false, Some("1")) && !allowed(false, Some("0")) && !allowed(false, None));
    }

    #[test]
    fn a_pack_that_fails_to_open_for_any_reason_falls_back_with_one_line() {
        let cases = [
            ("ImportError: libnccl.so.2: cannot open shared object file: No such file or directory\n", "kernel pack failed to open (a host library is missing: ImportError: libnccl.so.2: cannot open shared object file: No such file or directory)"),
            ("opening the pack\nIllegal instruction (core dumped)\n", "kernel pack failed to open (it crashed at open (illegal instruction): Illegal instruction (core dumped))"),
            ("goethe-engine: bucket b1 needs 40-51 GB, the device has 32\n", "kernel pack failed to open (it does not fit this GPU: goethe-engine: bucket b1 needs 40-51 GB, the device has 32)"),
            ("starting\nsomething odd happened\n", "kernel pack failed to open (unknown error: something odd happened)"),
            ("", "kernel pack failed to open (unknown error: no log)"),
        ];
        for (log, line) in cases {
            let (home, dir) = packed_home();
            assert_eq!(load_failure(log), line);
            assert!(unpack_failed(home.path(), "org/m-GGUF", log).unwrap(), "{log}");
            assert!(!dir.join(PACK_FILE).exists());
            assert_eq!(read_card_table(&dir).unwrap().get(PACK_SKIPPED).and_then(|v| v.as_str()), Some(line));
        }
    }

    #[test]
    fn a_pack_with_no_plain_weights_is_kept_and_the_failure_reported() {
        let home = tempfile::tempdir().unwrap();
        let dir = model_dir(home.path(), "org/p");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(PACK_FILE), b"p").unwrap();
        assert!(!unpack_failed(home.path(), "org/p", "libnccl.so.2: cannot open shared object file").unwrap());
        assert!(dir.join(PACK_FILE).exists());
    }

    #[test]
    fn a_pack_that_failed_here_is_not_offered_again_unless_forced() {
        let (home, dir) = packed_home();
        assert!(unpack_failed(home.path(), "org/m-GGUF", "Illegal instruction").unwrap());
        let opts = InstallOpts { model_id: "org/m-GGUF".into(), fallback: true, ..Default::default() };
        assert!(matches!(pack_decision(home.path(), &opts).unwrap(), PackDecision::Declined));
        assert!(dir.join(CARD_FILE).is_file());
        let card = read_card_table(&dir);
        assert!(stale_skip(card.as_ref(), false, true).is_some(), "an explicit fallback keeps the pack skipped");
        assert_eq!(stale_skip(card.as_ref(), true, true), None, "--force tries the pack again");
        assert_eq!(stale_skip(card.as_ref(), false, false), None, "without the opt-in a stale note never hides the pack");
    }

    #[test]
    fn a_gguf_with_an_audio_projector_is_a_transcribe_model() {
        let (home, dir) = gguf_home(true);
        let proj = dir.join("m.mmproj-f16.gguf");
        crate::engines::llama_cpp::write_test_gguf_with(&proj, &[("clip.vision.image_size", 448)]);
        let p = plan_in_place(home.path(), "org/m-GGUF", &dir, None).unwrap();
        assert_eq!(p.table.get("mmproj").and_then(|v| v.as_str()), Some("m.mmproj-f16.gguf"));
        assert_eq!(p.table.get("task"), None, "a vision projector alone makes no speech model");
        crate::engines::llama_cpp::write_test_gguf_with(&proj, &[("clip.has_audio_encoder", 1)]);
        let p = plan_in_place(home.path(), "org/m-GGUF", &dir, None).unwrap();
        assert_eq!(p.table.get("task").and_then(|v| v.as_str()), Some("transcribe"));
    }

    #[test]
    fn a_safetensors_index_without_its_shards_is_dropped_but_not_beside_real_ones() {
        let d = tempfile::tempdir().unwrap();
        fs::write(d.path().join("model.safetensors.index.json"), "{}").unwrap();
        fs::write(d.path().join("config.json"), "{}").unwrap();
        drop_orphan_index(d.path());
        assert!(!d.path().join("model.safetensors.index.json").exists() && d.path().join("config.json").exists());
        fs::write(d.path().join("model.safetensors.index.json"), "{}").unwrap();
        fs::write(d.path().join("model-00001-of-00001.safetensors"), "x").unwrap();
        drop_orphan_index(d.path());
        assert!(d.path().join("model.safetensors.index.json").exists(), "real shards: the index stays");
    }

    #[test]
    fn an_offered_pack_is_dropped_only_with_a_reason_and_a_projector_is_not_the_model() {
        assert_eq!(readable(Some(1), |_| Ok(()), false), Ok(Some(1)));
        assert_eq!(readable(Some(1), |_| Err("no config".into()), true), Ok(None), "with --fallback: dropped, with a note on stderr");
        let err = readable(Some(1), |_| Err("no config".into()), false).unwrap_err();
        assert!(err.contains("no config") && err.contains("--fallback"), "{err}");
        assert_eq!(readable::<u8>(None, |_| panic!("no offer: nothing to read"), false), Ok(None));
        let src = tempfile::tempdir().unwrap();
        crate::engines::llama_cpp::write_test_gguf(&src.path().join("OrcaCyber-Q4_K_M.gguf"));
        crate::engines::llama_cpp::write_test_gguf_with(&src.path().join("OrcaCyber-mmproj-F16.gguf"), &[("clip.vision.image_size", 448)]);
        let (home, _) = gguf_home(true);
        let opts = InstallOpts { model_id: src.path().display().to_string(), name: Some("me/orca".into()), ..Default::default() };
        let t = install_local(home.path(), &opts).unwrap();
        assert!(t.pack.is_none(), "no login: no offer");
        let names: Vec<String> = crate::engines::llama_cpp::gguf_paths(src.path()).iter().filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())).filter(|n| crate::engines::llama_cpp::is_model_gguf(n)).collect();
        assert_eq!(names, ["OrcaCyber-Q4_K_M.gguf"], "the metadata is made from the model file, never the projector");
    }

    #[test]
    fn a_gated_answer_for_an_installed_model_is_already_installed() {
        let gated = "org/m is gated or private on Hugging Face, so it needs your Hugging Face token.";
        assert!(gated_but_installed(gated, true, false));
        assert!(gated_but_installed("org/m: not found on Hugging Face, or private (check the id)", true, false));
        assert!(!gated_but_installed(gated, true, true), "--force asks for the download");
        assert!(!gated_but_installed(gated, false, false), "not installed: the gated guidance stands");
        assert!(!gated_but_installed("disk full", true, false));
        let (home, _) = gguf_home(false);
        assert!(!is_installed(home.path(), "org/m-GGUF"));
        let con = db::open(home.path()).unwrap();
        con.execute("INSERT INTO installed_models (model_id, local_path, managed) VALUES ('org/m-GGUF', 'models/org/m-GGUF', 1)", []).ok();
    }

    #[test]
    fn a_card_is_never_written_through_a_link_a_model_folder_ships() {
        let (dir, outside) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        let victim = outside.path().join("keep.txt");
        fs::write(&victim, "mine").unwrap();
        std::os::unix::fs::symlink(&victim, dir.path().join(format!("{CARD_FILE}.tmp"))).unwrap();
        write_card_table(dir.path(), &toml::from_str("model_id = \"a/b\"").unwrap()).unwrap();
        assert_eq!(fs::read_to_string(&victim).unwrap(), "mine");
        assert!(fs::symlink_metadata(dir.path().join(CARD_FILE)).unwrap().is_file());
        assert_eq!(read_card_table(dir.path()).unwrap()["model_id"].as_str(), Some("a/b"));
    }

    #[test]
    fn ids_that_share_a_directory_do_not_take_it_over_from_each_other() {
        let (home, _) = gguf_home(true);
        let (a, b) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        for d in [&a, &b] {
            crate::engines::llama_cpp::write_test_gguf(&d.path().join("m-Q8_0.gguf"));
        }
        let first = install_local(home.path(), &InstallOpts { model_id: a.path().display().to_string(), name: Some("x/y".into()), ..Default::default() }).unwrap();
        fs::write(first.card_dir.join(CARD_FILE), format!("model_id = \"x/y\"\nsource_dir = \"{}\"\n", a.path().display())).unwrap();
        let clash = InstallOpts { model_id: b.path().display().to_string(), name: Some("x__y".into()), ..Default::default() };
        assert!(install_local(home.path(), &clash).err().unwrap().contains("would share it"));
        assert!(fs::read_link(first.card_dir.join("m-Q8_0.gguf")).unwrap().starts_with(a.path()), "the first install still points at its own folder");
        assert!(install_local(home.path(), &InstallOpts { name: Some("x/y".into()), ..clash }).is_ok(), "the same id reinstalls");
    }

    #[test]
    fn an_install_over_the_leftovers_of_a_removed_local_install_replaces_them() {
        let (home, _) = gguf_home(true);
        let src = tempfile::tempdir().unwrap();
        crate::engines::llama_cpp::write_test_gguf(&src.path().join("m-Q8_0.gguf"));
        let opts = InstallOpts { model_id: src.path().display().to_string(), name: Some("me/m".into()), ..Default::default() };
        let t = install_local(home.path(), &opts).unwrap();
        fs::write(t.card_dir.join("pack.gcpack"), b"a copied pack").unwrap();
        fs::write(t.card_dir.join(CARD_FILE), format!("model_id = \"me/m\"\nsource_dir = \"{}\"\n", src.path().display())).unwrap();
        let again = install_local(home.path(), &opts).unwrap();
        assert!(!again.card_dir.join("pack.gcpack").exists(), "the old copy is replaced without --force");
        assert!(again.card_dir.join("m-Q8_0.gguf").exists() && src.path().join("m-Q8_0.gguf").is_file());
        let own = model_dir(home.path(), "me/own");
        fs::create_dir_all(&own).unwrap();
        fs::write(own.join("w.gguf"), b"mine").unwrap();
        let o = InstallOpts { name: Some("me/own".into()), ..opts };
        assert!(install_local(home.path(), &o).err().unwrap().contains("--force"), "a folder of the user's own files is never replaced silently");
    }

    #[test]
    fn a_local_gguf_folder_installs_for_llama_cpp_with_or_without_a_config() {
        let (home, dir) = gguf_home(true);
        fs::write(dir.join("config.json"), "{}").unwrap();
        let p = plan_in_place(home.path(), "org/m-GGUF", &dir, None).unwrap();
        assert_eq!(p.table.get("runtime").and_then(|v| v.as_str()), Some("llama.cpp"), "a config.json beside the GGUF changes nothing");
        let src = tempfile::tempdir().unwrap();
        crate::engines::llama_cpp::write_test_gguf(&src.path().join("m-Q8_0.gguf"));
        let opts = InstallOpts { model_id: src.path().display().to_string(), name: Some("me/m".into()), ..Default::default() };
        let t = install_local(home.path(), &opts).unwrap();
        assert!(t.card_dir.join("m-Q8_0.gguf").exists() && t.pack.is_none());
        let empty = tempfile::tempdir().unwrap();
        let none = InstallOpts { model_id: empty.path().display().to_string(), name: Some("me/e".into()), ..Default::default() };
        assert!(install_local(home.path(), &none).err().unwrap().contains("has no config.json"));
    }

    #[test]
    fn a_gguf_only_dir_becomes_a_llama_cpp_card_with_ninfer_row_units() {
        let (home, dir) = gguf_home(true);
        let p = plan_in_place(home.path(), "org/m-GGUF", &dir, None).unwrap();
        assert_eq!(p.table.get("runtime").and_then(|v| v.as_str()), Some("llama.cpp"));
        assert_eq!(p.table.get("gguf").and_then(|v| v.as_str()), Some("m-Q4_K_M.gguf"));
        commit(home.path(), &p, "installed", &mut Vec::new()).unwrap();
        let nin = home.path().join("models/n");
        fs::create_dir_all(&nin).unwrap();
        fs::File::create(nin.join(crate::engines::ninfer::NINFER_FILE)).unwrap().set_len(crate::engines::ninfer::NINFER_BYTES).unwrap();
        commit(home.path(), &plan_in_place(home.path(), "org/n", &nin, None).unwrap(), "installed", &mut Vec::new()).unwrap();
        let con = db::open(home.path()).unwrap();
        let row = |id: &str| {
            con.query_row(
                "SELECT kv_bytes_per_token, kv_elems_per_token, max_ctx, num_kv_heads, static_vram_gb > 0, default_max_conc FROM installed_models WHERE model_id = ?1",
                [id],
                |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?, r.get::<_, i64>(3)?, r.get::<_, bool>(4)?, r.get::<_, i64>(5)?)),
            )
            .unwrap()
        };
        assert_eq!(row("org/m-GGUF"), (768 * 2, 768, 4096, 4, true, 4), "f16 KV: two bytes per element");
        let (bytes, elems, ..) = row("org/n");
        assert_eq!(bytes, elems, "fp8 KV: one byte per element");
    }

    #[test]
    fn a_pack_dir_or_a_missing_llama_server_never_makes_a_llama_cpp_card() {
        let (home, dir) = gguf_home(true);
        fs::write(dir.join(PACK_FILE), b"p").unwrap();
        let err = plan_in_place(home.path(), "org/m-GGUF", &dir, None).map(|p| p.table).unwrap_err();
        assert!(err.contains("config.json"), "{err}");
        if std::env::var_os("LLAMA_SERVER").is_none() {
            let (home, dir) = gguf_home(false);
            let planned = plan_in_place(home.path(), "org/m-GGUF", &dir, None).map(|p| p.table);
            match crate::engines::llama_cpp::server_ready(home.path()) {
                Err(_) => assert!(planned.unwrap_err().contains("llama-server not found")),
                Ok(()) => assert!(planned.is_ok(), "a pinned prebuilt server is fetched at install, never at plan"),
            }
        }
    }

    #[test]
    fn a_local_pack_records_the_weights_it_matched_and_never_writes_into_the_users_folder() {
        let (d, src) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        for f in [PACK_VERIFY, PACK_FILE] {
            fs::write(src.path().join(f), b"theirs").unwrap();
            symlink(src.path().join(f), d.path().join(f)).unwrap();
        }
        let ok = |_: &[String]| Ok((0, r#"{"ok": true, "arch": "sm_120a", "weights": ["m.safetensors"]}"#.to_string()));
        local_pack(d.path(), d.path(), "sm_120a", &ok).unwrap();
        assert!(fs::read_to_string(d.path().join(PACK_VERIFY)).unwrap().contains("m.safetensors"));
        assert_eq!(fs::read(src.path().join(PACK_VERIFY)).unwrap(), b"theirs", "a linked file of the source is never written");
        assert!(fs::symlink_metadata(d.path().join(PACK_FILE)).unwrap().is_file(), "the pack is a regular copy");
        let refused = |_: &[String]| Err("goethe-engine pack verify: exit 1: pack: REFUSED: --arch differs".to_string());
        assert!(local_pack(d.path(), d.path(), "sm_90a", &refused).unwrap_err().contains("REFUSED"));
        let none = |_: &[String]| Ok((0, r#"{"ok": true}"#.to_string()));
        assert!(local_pack(d.path(), d.path(), "sm_120a", &none).unwrap_err().contains("matches none"));
    }

    #[test]
    fn installing_a_model_drops_a_same_named_alias() {
        let dir = tempfile::tempdir().unwrap();
        let con = db::open(dir.path()).unwrap();
        con.execute("INSERT INTO model_aliases (alias, model_id) VALUES ('org/new', 'org/old'), ('fast', 'org/old')", [])
            .unwrap();
        let note = drop_shadowing_alias(&con, "org/new").unwrap().unwrap();
        assert!(note.contains("removed alias org/new (it pointed to org/old)"), "{note}");
        assert_eq!(drop_shadowing_alias(&con, "org/new").unwrap(), None);
        let left: i64 = con.query_row("SELECT COUNT(*) FROM model_aliases", [], |r| r.get(0)).unwrap();
        assert_eq!(left, 1, "other aliases stay");
    }

    #[test]
    fn load_lines_and_already_loaded() {
        let row = json!({"instance_id": "i1", "gpu_indices": [2, 3], "internal_port": 30005, "status": "RUNNING"});
        let result = json!({"instance_id": "i1", "port": 30005, "status": "STARTING"});
        assert_eq!(
            load_status_line("org/m", "i1", Some(&row), &result, Some(23.04)),
            "RUNNING: org/m instance i1 on GPU 2,3 port 30005 (23.0 s)"
        );
        let starting = load_status_line("org/m", "i1", None, &result, None);
        assert!(starting.starts_with("STARTING: org/m instance i1 on GPU ? port 30005"), "{starting}");
        let reply = |result: Value, stdout: &str| crate::rpc::RpcReply { code: 0, stdout: stdout.into(), stderr: String::new(), result };
        assert!(is_already_loaded(&reply(json!({"already_loaded": true}), "")));
        assert!(is_already_loaded(&reply(json!({"status": "already_loaded"}), "")));
        assert!(!is_already_loaded(&reply(result.clone(), "")));
        let both = reply(result.clone(), &format!("auto: tp=1 on GPU 2\n{}", serde_json::to_string(&result).unwrap()));
        assert_eq!(reply_message(&both), "auto: tp=1 on GPU 2");
        assert_eq!(reply_message(&reply(Value::Null, "org/m is already loaded\n")), "org/m is already loaded");
    }

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
            supported_sm: None,
        };
        let old: toml::Table = toml::from_str("note = \"mine\"\nmarlin_extra_gb = 3.0\n").unwrap();
        let t = card_table(&card, Some(&old)).unwrap();
        assert_eq!(t.get("note").and_then(|v| v.as_str()), Some("mine"));
        assert!(t.get("marlin_extra_gb").is_none());
    }
}
