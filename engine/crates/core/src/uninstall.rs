// Remove goethe install under GOETHE_HOME only.

use crate::paths::goethe_home;
use crate::shutdown;
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct UninstallOpts {
    pub purge_models: bool,
    pub yes: bool,
}

pub fn run(opts: UninstallOpts) -> Result<(), String> {
    let home = goethe_home();
    validate_home(&home, opts.purge_models)?;
    if !opts.yes {
        if !std::io::stdin().is_terminal() {
            return Err("uninstall requires --yes without a tty".to_string());
        }
        print_plan(&home, opts.purge_models);
        print!("Proceed? [y/N] ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        let mut buf = String::new();
        std::io::stdin().read_line(&mut buf).map_err(|e| e.to_string())?;
        if !buf.trim().eq_ignore_ascii_case("y") {
            return Err("uninstall aborted".to_string());
        }
    } else {
        print_plan(&home, opts.purge_models);
    }
    let _ = shutdown::run_stop();
    remove_installer_symlink(&home);
    remove_systemd_unit_if_ours();
    remove_home_contents(&home, opts.purge_models)?;
    println!("uninstalled goethe from {}", home.display());
    Ok(())
}

fn validate_home(home: &Path, purge_models: bool) -> Result<(), String> {
    let canonical = fs::canonicalize(home).map_err(|_| "GOETHE_HOME is not a directory".to_string())?;
    if canonical == Path::new("/") {
        return Err("refusing to uninstall GOETHE_HOME=/".to_string());
    }
    if let Ok(user_home) = std::env::var("HOME") {
        if let Ok(user_canonical) = fs::canonicalize(&user_home) {
            if canonical == user_canonical {
                return Err("refusing to uninstall GOETHE_HOME=$HOME".to_string());
            }
        }
    }
    let looks_like_goethe = home.join("state.db").is_file() || home.join("bin/goethe").is_file();
    if !looks_like_goethe {
        if only_models_dir(home) && purge_models {
            return Ok(());
        }
        return Err("GOETHE_HOME does not look like a goethe install (missing state.db)".to_string());
    }
    Ok(())
}

fn only_models_dir(home: &Path) -> bool {
    let Ok(entries) = fs::read_dir(home) else {
        return false;
    };
    for ent in entries.flatten() {
        let name = ent.file_name().to_string_lossy().to_string();
        if name == "models" {
            continue;
        }
        return false;
    }
    home.join("models").is_dir()
}

fn print_plan(home: &Path, purge: bool) {
    println!("Will stop goethed and remove:");
    println!("  {}", home.display());
    if !purge {
        println!("  (keeping {}/models/)", home.display());
    }
    println!("  installer symlinks pointing into GOETHE_HOME");
}

fn remove_installer_symlink(home: &Path) {
    for link in [
        "/usr/local/bin/goethe",
        &format!(
            "{}/.local/bin/goethe",
            std::env::var("HOME").unwrap_or_default()
        ),
    ] {
        let path = PathBuf::from(link);
        if path.is_symlink() {
            if let Ok(target) = fs::read_link(&path) {
                if target.starts_with(home) {
                    let _ = fs::remove_file(&path);
                }
            }
        }
    }
}

fn remove_systemd_unit_if_ours() {
    if unsafe { libc::geteuid() } != 0 {
        return;
    }
    let _ = Command::new("systemctl")
        .args(["disable", "--now", "goethed.service"])
        .status();
    let unit = Path::new("/etc/systemd/system/goethed.service");
    if unit.is_file() {
        if let Ok(text) = fs::read_to_string(unit) {
            if text.contains("GOETHE_HOME") || text.contains("/.goethe") {
                let _ = fs::remove_file(unit);
            }
        }
    }
    let _ = Command::new("systemctl").args(["daemon-reload"]).status();
}

fn remove_home_contents(home: &Path, purge_models: bool) -> Result<(), String> {
    for entry in fs::read_dir(home).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name == "models" && !purge_models {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            fs::remove_dir_all(&path).map_err(|e| e.to_string())?;
        } else {
            fs::remove_file(&path).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}
