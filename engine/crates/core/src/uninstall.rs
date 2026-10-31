// Remove a goethe install under GOETHE_HOME only, after proving the directory is one.

use crate::paths::{goethe_home, HOME_MARKER};
use crate::shutdown;
use std::fs;
use std::path::{Path, PathBuf};

pub struct UninstallOpts {
    pub keep_models: bool,
    pub yes: bool,
}

pub fn run(opts: UninstallOpts) -> Result<(), String> {
    // Validate the home, confirm, stop goethed (or abort), drop autostart, delete the installed models, then the rest.
    let home = goethe_home();
    validate_home(&home)?;
    let models = if opts.keep_models {
        None
    } else {
        Some(crate::models::uninstall_plan(&home).unwrap_or(crate::models::UninstallModels {
            delete: Vec::new(),
            keep: Vec::new(),
            sources: Vec::new(),
            ids: Vec::new(),
            gb: 0.0,
        }))
    };
    let mut plan = vec!["stop goethed and remove its autostart entry".to_string()];
    match &models {
        None => plan.push(format!("keep {} (--keep-models)", home.join("models").display())),
        Some(m) => {
            if !m.ids.is_empty() {
                plan.push(format!("delete {} installed model(s), {:.1} GB: {}", m.ids.len(), m.gb, m.ids.join(", ")));
            }
            plan.extend(m.sources.iter().map(|s| format!("keep your files in {s} (registered in place; only goethe's links go)")));
            plan.extend(m.keep.iter().map(|(p, why)| format!("keep {} ({why})", p.display())));
        }
    }
    let traces = home.join("traces");
    let keep_traces = traces.join("manifest.json").is_file();
    if keep_traces {
        plan.push(format!("keep {} (training traces; delete the folder yourself if you do not want them)", traces.display()));
    }
    plan.push(format!("delete everything else in {}", home.display()));
    plan.push("remove installer symlinks that point into GOETHE_HOME".to_string());
    if opts.yes {
        println!("goethe uninstall will:\n  {}", plan.join("\n  "));
    }
    crate::host::confirm(&format!("goethe uninstall will:\n  {}\nProceed?", plan.join("\n  ")), opts.yes)?;
    shutdown::run_down().map_err(|e| format!("could not stop goethed ({e}); stop it, then run goethe uninstall again"))?;
    crate::service::run_disable()?;
    remove_installer_symlinks(&home);
    let kept: Vec<PathBuf> = match &models {
        Some(m) => {
            for p in &m.delete {
                crate::models::delete_owned(&home, p)?;
            }
            m.keep.iter().map(|(p, _)| p.clone()).collect()
        }
        None => Vec::new(),
    };
    remove_home_contents(&home, models.is_none(), keep_traces, &kept)?;
    println!("uninstalled goethe from {}", home.display());
    if keep_traces {
        println!("kept {} (training traces)", traces.display());
    }
    for p in &kept {
        println!("kept {} (your files)", p.display());
    }
    Ok(())
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
    let mut links = vec![PathBuf::from("/usr/local/bin/goethe")];
    if let Some(h) = std::env::var_os("HOME") {
        links.push(PathBuf::from(h).join(".local/bin/goethe"));
    }
    for path in links {
        if fs::read_link(&path).is_ok_and(|t| t.starts_with(home)) {
            let _ = fs::remove_file(&path);
        }
    }
}

fn remove_home_contents(home: &Path, keep_models: bool, keep_traces: bool, kept: &[PathBuf]) -> Result<(), String> {
    // Everything in the home but a trace corpus; models/ stays with --keep-models, and keeps the folders of your own files otherwise.
    for entry in fs::read_dir(home).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let is_dir = entry.file_type().map_err(|e| e.to_string())?.is_dir();
        if keep_traces && entry.file_name() == "traces" {
            continue;
        }
        if entry.file_name() == "models" {
            if keep_models {
                continue;
            }
            if is_dir && !kept.is_empty() {
                remove_except(&path, kept)?;
                continue;
            }
        }
        let r = if is_dir { fs::remove_dir_all(&path) } else { fs::remove_file(&path) };
        r.map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

fn remove_except(dir: &Path, kept: &[PathBuf]) -> Result<(), String> {
    // Delete the entries of dir except kept paths (symlinks are removed, never followed).
    let kept: Vec<PathBuf> = kept.iter().filter_map(|k| fs::canonicalize(k.parent()?).ok().map(|p| p.join(k.file_name().unwrap_or_default()))).collect();
    let base = fs::canonicalize(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        if kept.contains(&base.join(entry.file_name())) {
            continue;
        }
        let path = entry.path();
        let r = if entry.file_type().map_err(|e| e.to_string())?.is_dir() { fs::remove_dir_all(&path) } else { fs::remove_file(&path) };
        r.map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        remove_home_contents(&h, false, false, &[models.join("org__mine")]).unwrap();
        assert!(models.join("org__mine/w.safetensors").exists(), "your own files stay");
        assert!(!models.join("org__a").exists() && !models.join(".partial").exists() && !h.join("logs").exists());
        assert!(!models.join("org__linked").exists() && elsewhere.join("config.json").exists(), "links go, their targets stay");
        fs::create_dir_all(h.join("logs")).unwrap();
        remove_home_contents(&h, true, false, &[]).unwrap();
        assert!(models.join("org__mine").exists() && !h.join("logs").exists(), "--keep-models keeps models/");
        fs::create_dir_all(h.join("traces")).unwrap();
        fs::write(h.join("traces/manifest.json"), "{}").unwrap();
        remove_home_contents(&h, false, true, &[]).unwrap();
        assert!(!models.exists() && h.join("traces/manifest.json").exists(), "a trace corpus stays");
        remove_home_contents(&h, false, false, &[]).unwrap();
        assert!(!h.join("traces").exists());
    }
}
