// Pack offer and login decision before download.

use super::{fallback_allowed, model_dir, read_card_table, InstallOpts, PackChoice, PackDecision, PACK_SKIPPED};
use crate::hf;
use std::io::IsTerminal;
use std::path::Path;

pub(super) type Looked = Result<Option<(String, crate::packs::Plan)>, String>;

pub(super) struct OfferDeps<'a> {

    pub(super) creds: Result<Option<crate::platform::Credentials>, String>,
    pub(super) archs: Vec<String>,
    pub(super) index: &'a dyn Fn() -> Option<Vec<u8>>,
    pub(super) ensure_engine: &'a dyn Fn() -> Result<(), String>,
    pub(super) by_structure: &'a dyn Fn(&crate::platform::Credentials) -> Looked,
    pub(super) may_match: &'a dyn Fn(&[u8]) -> bool,
}

pub(super) fn structure_offer(home: &Path, opts: &InstallOpts, dir: &Path) -> Result<Option<PackChoice>, String> {
    // Weights no pack names (a fine-tune, another repo or a folder of the same architecture) on a logged-in node: the pack whose structure they have, if taken. See offer_for.
    let archs = crate::packs::local_archs(home);
    let run = crate::packs::python_runner(home);
    let deps = OfferDeps {
        creds: crate::platform::load_credentials(home),
        index: &|| crate::packs::catalog(home),
        ensure_engine: &|| crate::pyenv::ensure(home, crate::pyenv::Layer::Engine),
        by_structure: &|creds| {
            let token = || Ok(creds.token.expose().to_string());
            let broker = crate::packs::HttpBroker::new(&creds.platform_url, &token);
            crate::packs::by_structure(home, dir, &archs, &broker, &run)
        },
        may_match: &|index| {
            let gguf_arch = crate::engines::llama_cpp::gguf_paths(dir).into_iter().find(|p| p.file_name().is_some_and(|n| crate::engines::llama_cpp::is_model_gguf(&n.to_string_lossy()))).and_then(|p| crate::engines::llama_cpp::architecture(&p));
            let config_archs = if gguf_arch.is_some() { Vec::new() } else { crate::model_facts::read_config(dir).map(|c| crate::scan::architectures(&c)).unwrap_or_default() };
            let pack_archs = |repo: &str, rev: &str| hf::fetch_config_json(repo, rev).map(|c| crate::scan::architectures(&c)).unwrap_or_default();
            crate::packs::may_match(index, &archs, gguf_arch.as_deref(), &config_archs, &pack_archs)
        },
        archs: archs.clone(),
    };
    offer_for(opts, deps)
}

pub(super) fn offer_for(opts: &InstallOpts, deps: OfferDeps) -> Result<Option<PackChoice>, String> {
    // Logged out, or no pack for this machine's GPUs at all: the plain runtime, as before. Logged in with packs for these GPUs: the engine layer is installed first when missing (the pack lookup runs in it), and a lookup that fails stops with its reason and the --fallback hint (or, with --fallback, says so and goes on); weights that match no pack structure say so in one line.
    let creds = match deps.creds {
        Ok(Some(c)) => c,
        Ok(None) => return Ok(None),
        Err(e) => {
            eprintln!("note: the console login cannot be read ({e}), so these weights are not checked for optimised kernels");
            return Ok(None);
        }
    };
    let miss = |why: String| -> Result<Option<PackChoice>, String> {
        eprintln!("note: {why}; using the plain runtime");
        Ok(None)
    };
    let Some(index) = (deps.index)() else {
        eprintln!("note: the kernel pack index is not reachable, so these weights are not checked for optimised kernels");
        return Ok(None);
    };
    if crate::packs::offers_any(&index, &deps.archs).is_empty() {
        return Ok(None);
    }
    if !(deps.may_match)(&index) {
        println!("no optimised kernel pack has the structure of these weights; using the plain runtime");
        return Ok(None);
    }
    if let Err(e) = (deps.ensure_engine)() {
        return miss(format!("optimised kernels exist for this GPU but goethe-engine could not be installed to look for them: {e}"));
    }
    let plan = match (deps.by_structure)(&creds) {
        Ok(Some((_, plan))) => plan,
        Ok(None) => {
            println!("no optimised kernel pack has the structure of these weights; using the plain runtime");
            return Ok(None);
        }
        Err(e) => return miss(format!("looking for optimised kernels for these weights failed: {e}")),
    };
    let what = format!("optimised kernels available for these weights (same structure as {})", plan.entry.offer.hf_repo);
    if opts.yes || !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        println!("{what}");
    } else if !crate::term::ask(&format!("{what}; install them? [Y/n]"), "y").is_ok_and(|a| a.to_ascii_lowercase().starts_with('y')) {
        return Ok(None);
    }
    Ok(Some(PackChoice { offer: plan.entry.offer.clone(), creds, plan: Some(plan) }))
}


pub(super) fn stale_skip(card: Option<&toml::Table>, force: bool, allow: bool) -> Option<String> {
    // The pack_skipped note of a card, when an explicit fallback is on and the pack is not being forced.
    card?.get(PACK_SKIPPED).and_then(|v| v.as_str()).filter(|_| allow && !force).map(String::from)
}

pub(super) fn pack_decision(home: &Path, opts: &InstallOpts) -> Result<PackDecision, String> {
    // A pack in the platform's index for this model and a local GPU: used when the node is logged in; otherwise the user may log in now. Without a login no pack is ever downloaded. A pack skipped earlier by an explicit fallback stays skipped while the fallback is on, unless --force; without the opt-in the pack is tried again and fails loudly.
    if let Some(why) = stale_skip(read_card_table(&model_dir(home, &opts.model_id)).as_ref(), opts.force, fallback_allowed(opts.fallback)) {
        eprintln!("note: skipping the kernel pack: {why} (--force tries it again)");
        return Ok(PackDecision::Declined);
    }
    let Some(index) = crate::packs::catalog(home) else { return Ok(PackDecision::None) };
    let archs = crate::packs::local_archs(home);
    let engine = crate::packs::pack_engine_version(home);
    let Some(offer) = crate::packs::offers_for(&index, &opts.model_id, &archs, &engine).into_iter().next() else {
        let theirs = crate::packs::pack_archs(&index, &opts.model_id);
        if let Some(note) = crate::packs::engine_too_old_at(&index, &opts.model_id, &archs, &engine) {
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

pub(super) fn ask_login(offer: &crate::packs::Offer, opts: &InstallOpts) -> Result<bool, String> {
    // Asked only on a terminal; elsewhere one note, and the install continues with SGLang.
    let what = format!("Optimised kernels for {} are available for this GPU ({})", opts.model_id, offer.arch);
    if !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        eprintln!("note: {what} to nodes logged in to the Goethe console (goethe login); continuing without them");
        return Ok(false);
    }
    let a = crate::term::ask(&format!("{what}.\nLog in to the Goethe console to install them? Otherwise goethe uses SGLang. [Y/n]"), "y")?;
    Ok(a.to_ascii_lowercase().starts_with('y'))
}
