// Remove a goethe install under GOETHE_HOME only, after proving the directory is one.

use crate::paths::{goethe_home, HOME_MARKER};
use crate::shutdown;
use std::fs;
use std::path::{Path, PathBuf};

pub struct UninstallOpts {
    pub purge_models: bool,
    pub yes: bool,
}

pub fn run(opts: UninstallOpts) -> Result<(), String> {
    // Validate the home, confirm, stop goethed (or abort), drop autostart, then delete.
    let home = goethe_home();
    validate_home(&home)?;
    let mut plan = vec!["stop goethed and remove its autostart entry".to_string(), format!("delete {}", home.display())];
    if !opts.purge_models {
        plan.push(format!("keep {}", home.join("models").display()));
    }
    plan.push("remove installer symlinks that point into GOETHE_HOME".to_string());
    crate::host::confirm(&format!("goethe uninstall will:\n  {}\nProceed?", plan.join("\n  ")), opts.yes)?;
    shutdown::run_down().map_err(|e| format!("could not stop goethed ({e}); stop it, then run goethe uninstall again"))?;
    crate::service::run_disable()?;
    remove_installer_symlinks(&home);
    remove_home_contents(&home, opts.purge_models)?;
    println!("uninstalled goethe from {}", home.display());
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

fn remove_home_contents(home: &Path, purge_models: bool) -> Result<(), String> {
    for entry in fs::read_dir(home).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        if entry.file_name() == "models" && !purge_models {
            continue;
        }
        let path = entry.path();
        let r = if entry.file_type().map_err(|e| e.to_string())?.is_dir() {
            fs::remove_dir_all(&path)
        } else {
            fs::remove_file(&path)
        };
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
}
