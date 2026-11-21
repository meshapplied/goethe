// Remove a goethe install under GOETHE_HOME only, after proving the directory is one.

use crate::engines::goethe_engine::PACK_FILE;
use crate::engines::ninfer::NINFER_FILE;
use crate::install::CARD_FILE;
use crate::models::UninstallModels;
use crate::paths::{goethe_home, HOME_MARKER};
use crate::shutdown;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

pub struct UninstallOpts {
    pub keep_models: bool,
    pub yes: bool,
}

struct Inventory {
    engines: Vec<String>,
    packs: Vec<String>,
    outside: Vec<PathBuf>,
    outside_bytes: u64,
}

pub fn run(opts: UninstallOpts) -> Result<(), String> {
    // Validate the home, list what goes, confirm, stop goethed (or abort), drop autostart, delete the models, then the rest.
    let home = goethe_home();
    validate_home(&home)?;
    let models = if opts.keep_models {
        None
    } else {
        Some(crate::models::uninstall_plan(&home).unwrap_or(UninstallModels {
            delete: Vec::new(),
            keep: Vec::new(),
            sources: Vec::new(),
            ids: Vec::new(),
            gb: 0.0,
        }))
    };
    let cache = crate::engines::ninfer::ninfer_cache_file().ok();
    let inv = inventory(&home, models.as_ref(), cache.as_deref());
    let mut plan = vec!["stop goethed and remove its autostart entry".to_string()];
    plan.extend(model_lines(&home, models.as_ref(), &inv));
    if inv.engines.is_empty() {
        plan.push("no inference engines are installed".to_string());
    } else {
        plan.push(format!("delete the inference engines: {}", inv.engines.join("; ")));
    }
    if !inv.packs.is_empty() {
        plan.push(format!("delete the kernel packs: {}", inv.packs.join("; ")));
    }
    let traces = home.join("traces");
    let keep_traces = traces.join("manifest.json").is_file();
    if keep_traces {
        plan.push(format!("keep {} (training traces; delete the folder yourself if you do not want them)", traces.display()));
    }
    let linked = crate::platform_cli::is_linked(&home);
    if linked {
        plan.push("log this node out of the console and delete its credentials".to_string());
    }
    plan.push(format!("delete everything else in {}", home.display()));
    plan.push("remove installer symlinks that point into GOETHE_HOME".to_string());
    let default_home = std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".goethe")).is_some_and(|d| d == home);
    if default_home {
        plan.push("remove Tab completion (goethe completion remove)".to_string());
    }
    if opts.yes {
        println!("goethe uninstall will:\n  {}", plan.join("\n  "));
    }
    crate::host::confirm(&format!("goethe uninstall will:\n  {}\nProceed?", plan.join("\n  ")), opts.yes)?;
    shutdown::run_down().map_err(|e| format!("could not stop goethed ({e}); stop it, then run goethe uninstall again"))?;
    crate::service::run_disable()?;
    if linked {
        crate::platform_cli::uninstall_logout(&home);
    }
    remove_installer_symlinks(&home);
    if default_home {
        crate::completion::remove_all();
    }
    let mut left = Vec::new();
    let kept: Vec<PathBuf> = match &models {
        Some(m) => {
            for p in &m.delete {
                if fs::symlink_metadata(p).is_ok() {
                    left.extend(crate::models::delete_owned(&home, p).err());
                }
            }
            for (p, _) in &m.keep {
                left.extend(strip_goethe_files(p));
            }
            m.keep.iter().map(|(p, _)| p.clone()).collect()
        }
        None => Vec::new(),
    };
    left.extend(remove_outside(&inv.outside));
    left.extend(remove_home_contents(&home, models.is_none(), keep_traces, &kept));
    if !left.is_empty() {
        return Err(format!(
            "goethe uninstall could not remove everything; these are left, delete them by hand:\n  {}",
            left.join("\n  ")
        ));
    }
    println!("uninstalled goethe from {}", home.display());
    if keep_traces {
        println!("kept {} (training traces)", traces.display());
    }
    for p in &kept {
        println!("kept {} (your files)", p.display());
    }
    Ok(())
}

fn model_lines(home: &Path, models: Option<&UninstallModels>, inv: &Inventory) -> Vec<String> {
    // The plan's model lines: what goes with its total size, and the folders of your own files that stay.
    let Some(m) = models else {
        return vec![format!("keep {} (--keep-models)", home.join("models").display())];
    };
    let mut out = Vec::new();
    let gb = m.gb + inv.outside_bytes as f64 / 1e9;
    if m.ids.is_empty() {
        out.push("no models are installed".to_string());
    } else {
        out.push(format!("delete {} installed model(s), {gb:.1} GB in total: {}", m.ids.len(), m.ids.join(", ")));
    }
    out.extend(inv.outside.iter().map(|p| format!("delete {} (downloaded by goethe install)", p.display())));
    out.extend(m.sources.iter().map(|s| format!("keep your files in {s} (registered in place; only goethe's links go)")));
    out.extend(m.keep.iter().map(|(p, why)| format!("keep {} ({why}; only goethe's card in it goes)", p.display())));
    if let Ok(t) = fs::read_link(home.join("models")) {
        out.push(format!("remove the models link; its folder {} stays", t.display()));
    }
    out
}

fn inventory(home: &Path, models: Option<&UninstallModels>, ninfer_cache: Option<&Path>) -> Inventory {
    // The engines, kernel packs and goethe downloads outside the home that uninstall deletes, with their sizes.
    let mut inv = Inventory { engines: Vec::new(), packs: Vec::new(), outside: Vec::new(), outside_bytes: 0 };
    let env = home.join("py-env");
    if is_real_dir(&env) {
        let mut layers = vec!["Python environment".to_string()];
        if let Some(site) = crate::pyenv::site_packages(home) {
            layers.extend(crate::pyenv::dist_version(&site, "goethe_engine").map(|v| format!("goethe-engine {v}")));
            layers.extend(crate::pyenv::dist_version(&site, "sglang").map(|v| format!("SGLang {v}")));
        }
        inv.engines.push(format!("{} (py-env, {})", layers.join(", "), size(&env)));
    }
    let mut engines: Vec<_> = fs::read_dir(home.join("engines")).into_iter().flatten().flatten().collect();
    engines.sort_by_key(|e| e.file_name());
    for e in engines {
        let name = e.file_name().to_string_lossy().into_owned();
        let label = if name == "llama.cpp" { name } else { format!("engines/{name}") };
        inv.engines.push(format!("{label} ({})", size(&e.path())));
    }
    let ternary = crate::runtime::ternary_kernels_dir(home);
    if is_real_dir(&ternary) {
        inv.engines.push(format!("ternary kernels ({})", size(&ternary)));
    }
    for dir in models.map(|m| m.delete.as_slice()).unwrap_or_default() {
        let pack = dir.join(PACK_FILE);
        if fs::symlink_metadata(&pack).is_ok_and(|m| m.is_file()) {
            let name = dir.file_name().unwrap_or_default().to_string_lossy().replace("__", "/");
            inv.packs.push(format!("{name} ({})", size(&pack)));
        }
        if let Some(c) = ninfer_cache.filter(|c| links_to(&dir.join(NINFER_FILE), c)) {
            if !inv.outside.iter().any(|p| p == c) {
                inv.outside_bytes += tree_bytes(c);
                inv.outside.push(c.to_path_buf());
                inv.outside.extend(c.parent().and_then(Path::parent).map(|r| r.join("patchkit")).filter(|p| is_real_dir(p)));
            }
        }
    }
    if is_real_dir(&home.join("packs")) {
        inv.packs.push(format!("pack index and catalog cache (packs, {})", size(&home.join("packs"))));
    }
    inv
}

fn links_to(link: &Path, file: &Path) -> bool {
    // True when link is a symlink resolving to file and file is a regular file, not a link.
    fs::symlink_metadata(link).is_ok_and(|m| m.file_type().is_symlink())
        && fs::symlink_metadata(file).is_ok_and(|m| m.is_file())
        && fs::canonicalize(link).ok().is_some_and(|l| fs::canonicalize(file).ok() == Some(l))
}

fn is_real_dir(p: &Path) -> bool {
    fs::symlink_metadata(p).is_ok_and(|m| m.is_dir())
}

fn tree_bytes(p: &Path) -> u64 {
    // Bytes of the files under p; symlinks are not followed.
    let Ok(meta) = fs::symlink_metadata(p) else {
        return 0;
    };
    if !meta.is_dir() {
        return if meta.is_file() { meta.len() } else { 0 };
    }
    fs::read_dir(p).into_iter().flatten().flatten().map(|e| tree_bytes(&e.path())).sum()
}

fn size(p: &Path) -> String {
    crate::progress::size(tree_bytes(p))
}

fn strip_goethe_files(dir: &Path) -> Vec<String> {
    // In a kept folder of your own files, delete only goethe's cards and .goethe_ markers; links and files of yours stay.
    let mut left = Vec::new();
    for e in fs::read_dir(dir).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let regular = e.file_type().is_ok_and(|t| t.is_file());
        if regular && (name.starts_with(CARD_FILE) || name.starts_with(".goethe_")) {
            left.extend(remove_path(&e.path()).err());
        }
    }
    left
}

fn remove_outside(paths: &[PathBuf]) -> Vec<String> {
    // Delete goethe's downloads outside the home (each a regular file or a real folder), then the cache folders they leave empty.
    let mut left = Vec::new();
    for p in paths {
        if fs::symlink_metadata(p).is_ok_and(|m| m.is_file() || m.is_dir()) {
            left.extend(remove_path(p).err());
        }
    }
    for file in paths.iter().filter(|p| p.file_name() == Some(NINFER_FILE.as_ref())) {
        let models = file.parent();
        let root = models.and_then(Path::parent);
        let goethe = root.and_then(Path::parent).filter(|g| g.file_name() == Some("goethe".as_ref()));
        for d in [models, root, goethe].into_iter().flatten() {
            if fs::remove_dir(d).is_err() {
                break;
            }
        }
    }
    left
}

fn remove_path(p: &Path) -> Result<(), String> {
    // Delete a file, a symlink (never its target) or a folder tree; one already gone is fine.
    let r = match fs::symlink_metadata(p) {
        Ok(m) if m.is_dir() => fs::remove_dir_all(p),
        Ok(_) => fs::remove_file(p),
        Err(e) => Err(e),
    };
    match r {
        Err(e) if e.kind() != ErrorKind::NotFound => Err(format!("{}: {e}", p.display())),
        _ => Ok(()),
    }
}

fn validate_home(home: &Path) -> Result<(), String> {
    // Require the goethe marker and a real bin/goethe; never /, $HOME, or a symlinked binary dir.
    let canonical = fs::canonicalize(home).map_err(|_| format!("{} is not a directory", home.display()))?;
    if canonical == Path::new("/") {
        return Err("refusing to uninstall GOETHE_HOME=/".to_string());
    }
    if let Some(user_home) = std::env::var_os("HOME").and_then(|h| fs::canonicalize(h).ok()) {
        if canonical == user_home {
            return Err("refusing to uninstall GOETHE_HOME=$HOME".to_string());
        }
    }
    let regular = |p: PathBuf| fs::symlink_metadata(p).map(|m| m.is_file()).unwrap_or(false);
    if !regular(home.join(HOME_MARKER)) || !regular(home.join("bin/goethe")) {
        return Err(format!(
            "{} does not look like a goethe install (needs {HOME_MARKER} and a regular bin/goethe); nothing was removed",
            home.display()
        ));
    }
    Ok(())
}

fn remove_installer_symlinks(home: &Path) {
    let mut dirs = vec![PathBuf::from("/usr/local/bin"), PathBuf::from("/opt/homebrew/bin")];
    if let Some(h) = std::env::var_os("HOME") {
        dirs.extend([PathBuf::from(&h).join(".local/bin"), PathBuf::from(&h).join("bin")]);
    }
    for path in dirs.iter().flat_map(|d| [d.join("goethe"), d.join("goethed")]) {
        if fs::read_link(&path).is_ok_and(|t| t.starts_with(home)) {
            let _ = fs::remove_file(&path);
        }
    }
}

fn remove_home_contents(home: &Path, keep_models: bool, keep_traces: bool, kept: &[PathBuf]) -> Vec<String> {
    // Everything in the home but a trace corpus; models/ stays with --keep-models, and keeps the folders of your own files otherwise. Returns what is left.
    let entries = match fs::read_dir(home) {
        Ok(rd) => rd,
        Err(e) => return vec![format!("{}: {e}", home.display())],
    };
    let mut left = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if keep_traces && entry.file_name() == "traces" {
            continue;
        }
        if entry.file_name() == "models" {
            if keep_models {
                continue;
            }
            if is_real_dir(&path) && !kept.is_empty() {
                left.extend(remove_except(&path, kept));
                continue;
            }
        }
        left.extend(remove_path(&path).err());
    }
    left
}

fn remove_except(dir: &Path, kept: &[PathBuf]) -> Vec<String> {
    // Delete the entries of dir except kept paths (symlinks are removed, never followed).
    let kept: Vec<PathBuf> = kept.iter().filter_map(|k| fs::canonicalize(k.parent()?).ok().map(|p| p.join(k.file_name().unwrap_or_default()))).collect();
    let (base, entries) = match (fs::canonicalize(dir), fs::read_dir(dir)) {
        (Ok(b), Ok(rd)) => (b, rd),
        (Err(e), _) | (_, Err(e)) => return vec![format!("{}: {e}", dir.display())],
    };
    entries
        .flatten()
        .filter(|e| !kept.contains(&base.join(e.file_name())))
        .filter_map(|e| remove_path(&e.path()).err())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models_of(delete: Vec<PathBuf>) -> UninstallModels {
        UninstallModels { delete, keep: Vec::new(), sources: Vec::new(), ids: vec!["org/a".into()], gb: 1.0 }
    }

    #[test]
    fn home_needs_marker_and_regular_binary() {
        let dir = tempfile::tempdir().unwrap();
        let h = dir.path().join("home");
        fs::create_dir_all(h.join("bin")).unwrap();
        assert!(validate_home(&h).is_err());
        fs::write(h.join(HOME_MARKER), "").unwrap();
        let elsewhere = dir.path().join("real-goethe");
        fs::write(&elsewhere, "bin").unwrap();
        std::os::unix::fs::symlink(&elsewhere, h.join("bin/goethe")).unwrap();
        assert!(validate_home(&h).unwrap_err().contains("regular bin/goethe"));
        fs::remove_file(h.join("bin/goethe")).unwrap();
        fs::write(h.join("bin/goethe"), "bin").unwrap();
        assert!(validate_home(&h).is_ok());
        assert!(validate_home(Path::new("/")).is_err());
    }

    #[test]
    fn models_go_except_folders_of_your_own_files() {
        let dir = tempfile::tempdir().unwrap();
        let h = dir.path().join("home");
        let models = h.join("models");
        for d in ["org__a", "org__mine", ".partial"] {
            fs::create_dir_all(models.join(d)).unwrap();
            fs::write(models.join(d).join("w.safetensors"), "x").unwrap();
        }
        let elsewhere = dir.path().join("user-folder");
        fs::create_dir_all(&elsewhere).unwrap();
        fs::write(elsewhere.join("config.json"), "{}").unwrap();
        std::os::unix::fs::symlink(&elsewhere, models.join("org__linked")).unwrap();
        fs::create_dir_all(h.join("logs")).unwrap();
        assert!(remove_home_contents(&h, false, false, &[models.join("org__mine")]).is_empty());
        assert!(models.join("org__mine/w.safetensors").exists(), "your own files stay");
        assert!(!models.join("org__a").exists() && !models.join(".partial").exists() && !h.join("logs").exists());
        assert!(!models.join("org__linked").exists() && elsewhere.join("config.json").exists(), "links go, their targets stay");
        fs::create_dir_all(h.join("logs")).unwrap();
        assert!(remove_home_contents(&h, true, false, &[]).is_empty());
        assert!(models.join("org__mine").exists() && !h.join("logs").exists(), "--keep-models keeps models/");
        fs::create_dir_all(h.join("traces")).unwrap();
        fs::write(h.join("traces/manifest.json"), "{}").unwrap();
        assert!(remove_home_contents(&h, false, true, &[]).is_empty());
        assert!(!models.exists() && h.join("traces/manifest.json").exists(), "a trace corpus stays");
        assert!(remove_home_contents(&h, false, false, &[]).is_empty());
        assert!(!h.join("traces").exists());
    }

    #[test]
    fn symlinked_models_folder_is_unlinked_not_followed() {
        let dir = tempfile::tempdir().unwrap();
        let h = dir.path().join("home");
        let big = dir.path().join("big-disk/models");
        fs::create_dir_all(big.join("org__other")).unwrap();
        fs::create_dir_all(&h).unwrap();
        std::os::unix::fs::symlink(&big, h.join("models")).unwrap();
        let m = models_of(Vec::new());
        let inv = inventory(&h, Some(&m), None);
        assert!(model_lines(&h, Some(&m), &inv).iter().any(|l| l.contains("remove the models link")));
        assert!(remove_home_contents(&h, false, false, &[]).is_empty());
        assert!(fs::symlink_metadata(h.join("models")).is_err() && big.join("org__other").is_dir());
    }

    #[test]
    fn engines_and_packs_are_listed_and_removed() {
        let dir = tempfile::tempdir().unwrap();
        let h = dir.path().join("home");
        let site = h.join("py-env/lib/python3.11/site-packages");
        for d in ["goethe_engine-1.2.3.dist-info", "sglang-0.5.1.dist-info", "huggingface_hub-1.0.dist-info"] {
            fs::create_dir_all(site.join(d)).unwrap();
        }
        fs::create_dir_all(h.join("py-env/bin")).unwrap();
        fs::write(h.join("py-env/bin/python"), "").unwrap();
        fs::create_dir_all(h.join("engines/llama.cpp")).unwrap();
        fs::write(h.join("engines/llama.cpp/llama-server"), "x").unwrap();
        fs::create_dir_all(crate::runtime::ternary_kernels_dir(&h)).unwrap();
        fs::create_dir_all(h.join("packs/catalog")).unwrap();
        let a = h.join("models/org__a");
        fs::create_dir_all(&a).unwrap();
        fs::write(a.join(PACK_FILE), "pack").unwrap();
        let m = models_of(vec![a.clone()]);
        let inv = inventory(&h, Some(&m), None);
        assert!(inv.engines[0].starts_with("Python environment, goethe-engine 1.2.3, SGLang 0.5.1 (py-env,"), "{:?}", inv.engines);
        assert!(inv.engines.iter().any(|e| e.starts_with("llama.cpp (")) && inv.engines.iter().any(|e| e.starts_with("ternary kernels")));
        assert!(inv.packs[0].starts_with("org/a (") && inv.packs[1].starts_with("pack index and catalog cache"), "{:?}", inv.packs);
        assert_eq!(inventory(&h, None, None).packs.len(), 1, "--keep-models keeps the packs inside the models");
        assert!(remove_home_contents(&h, false, false, &[]).is_empty());
        assert_eq!(fs::read_dir(&h).unwrap().count(), 0, "engines, packs and models all go");
        let empty = inventory(&h, None, None);
        assert!(empty.engines.is_empty() && empty.packs.is_empty(), "a missing folder is fine");
    }

    #[test]
    fn ninfer_download_outside_the_home_goes_only_when_goethe_linked_it() {
        let dir = tempfile::tempdir().unwrap();
        let h = dir.path().join("home");
        let root = dir.path().join("cache/goethe/nvfp4");
        let cache = root.join("models").join(NINFER_FILE);
        fs::create_dir_all(cache.parent().unwrap()).unwrap();
        fs::write(&cache, "weights").unwrap();
        fs::create_dir_all(root.join("patchkit/patches")).unwrap();
        fs::create_dir_all(root.join("ninfer/build")).unwrap();
        let card = h.join("models/meshapplied__ninfer");
        fs::create_dir_all(&card).unwrap();
        let unlinked = inventory(&h, Some(&models_of(vec![card.clone()])), Some(&cache));
        assert!(unlinked.outside.is_empty(), "a cache goethe did not link stays");
        std::os::unix::fs::symlink(&cache, card.join(NINFER_FILE)).unwrap();
        let m = models_of(vec![card.clone()]);
        let inv = inventory(&h, Some(&m), Some(&cache));
        assert_eq!(inv.outside, vec![cache.clone(), root.join("patchkit")]);
        assert_eq!(inv.outside_bytes, 7);
        assert!(model_lines(&h, Some(&m), &inv)[0].contains("1 installed model(s), 1.0 GB in total: org/a"));
        assert!(remove_outside(&inv.outside).is_empty());
        assert!(!cache.exists() && !root.join("models").exists() && !root.join("patchkit").exists());
        assert!(root.join("ninfer/build").is_dir(), "a build of your own next to it stays");
    }

    #[test]
    fn kept_folder_loses_only_goethe_cards() {
        let dir = tempfile::tempdir().unwrap();
        let own = dir.path().join("models/org__own");
        fs::create_dir_all(&own).unwrap();
        for f in [CARD_FILE, ".goethe_hf_revision", "w.safetensors", "config.json"] {
            fs::write(own.join(f), "x").unwrap();
        }
        std::os::unix::fs::symlink(dir.path(), own.join("mine-link")).unwrap();
        assert!(strip_goethe_files(&own).is_empty());
        let mut names: Vec<String> = fs::read_dir(&own).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        assert_eq!(names, ["config.json", "mine-link", "w.safetensors"]);
    }

    #[test]
    fn partial_failure_reports_what_is_left_and_removes_the_rest() {
        use std::os::unix::fs::PermissionsExt;
        if unsafe { libc::geteuid() } == 0 {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let h = dir.path().join("home");
        let stuck = h.join("py-env/lib");
        fs::create_dir_all(&stuck).unwrap();
        fs::write(stuck.join("f"), "x").unwrap();
        fs::create_dir_all(h.join("logs")).unwrap();
        fs::set_permissions(&stuck, fs::Permissions::from_mode(0o555)).unwrap();
        let left = remove_home_contents(&h, false, false, &[]);
        fs::set_permissions(&stuck, fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(left.len(), 1, "{left:?}");
        assert!(left[0].contains("py-env") && !h.join("logs").exists());
        assert!(remove_path(&h.join("gone")).is_ok(), "a missing path is fine");
    }
}
