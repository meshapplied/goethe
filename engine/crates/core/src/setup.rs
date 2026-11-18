// goethe setup: everything an install does after the bootstrap verified this binary, the same for a fresh install, a reinstall and an upgrade.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const DEFAULT_PKG_URL: &str = "https://pkg.goethe.sh";

pub fn env_file_text(pkg: &str, registry: Option<&str>, token: Option<&str>, keep_file_url: bool) -> String {
    // GOETHE_HOME/goethe.env holds only what differs from the defaults; a file:// package is a one-off unless asked to keep it.
    let pkg = pkg.trim_end_matches('/');
    let mut out = String::new();
    let local = pkg.starts_with("file://");
    if pkg != DEFAULT_PKG_URL && !pkg.is_empty() && (!local || keep_file_url) {
        out.push_str(&format!("GOETHE_PKG_URL={pkg}\n"));
    }
    if let Some(r) = registry.map(str::trim).filter(|r| !r.is_empty() && *r != format!("{pkg}/v1/registry.json")) {
        out.push_str(&format!("GOETHE_REGISTRY_URL={r}\n"));
    }
    if let Some(t) = token.map(str::trim).filter(|t| !t.is_empty()) {
        out.push_str(&format!("GOETHE_PKG_TOKEN={t}\n"));
    }
    out
}

pub fn write_env_file(home: &Path, text: &str) -> Result<(), String> {
    // Mode 0600 because a package token may be in it; no file at all when everything is the default.
    let path = home.join("goethe.env");
    if text.is_empty() {
        return match fs::remove_file(&path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(format!("{}: {e}", path.display())),
            _ => Ok(()),
        };
    }
    crate::platform::write_private(&path, text.as_bytes())
}

#[derive(Debug, PartialEq, Eq)]
pub struct LinkPlan {
    pub dir: PathBuf,
    pub on_path: bool,
}

pub fn link_dir(path_env: &str, user_home: &Path, writable: &dyn Fn(&Path) -> bool) -> LinkPlan {
    // The first folder on PATH that is a usual place for commands and that this user can write; else ~/.local/bin, which may need adding to PATH.
    let local = user_home.join(".local/bin");
    let usual = [local.clone(), user_home.join("bin"), PathBuf::from("/usr/local/bin"), PathBuf::from("/opt/homebrew/bin")];
    for dir in path_env.split(':').filter(|d| !d.is_empty()).map(PathBuf::from) {
        if usual.contains(&dir) && writable(&dir) {
            return LinkPlan { dir, on_path: true };
        }
    }
    LinkPlan { dir: local, on_path: false }
}

pub fn dir_writable(dir: &Path) -> bool {
    let c = std::ffi::CString::new(dir.as_os_str().as_encoded_bytes()).unwrap_or_default();
    dir.is_dir() && unsafe { libc::access(c.as_ptr(), libc::W_OK) } == 0
}

pub fn link_commands(home: &Path, dir: &Path) -> Result<(), String> {
    // goethe and goethed both point into GOETHE_HOME/bin, so a copied binary can never shadow the one the service runs.
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for name in ["goethe", "goethed"] {
        let link = dir.join(name);
        let target = home.join("bin").join(name);
        match fs::symlink_metadata(&link) {
            Ok(m) if m.file_type().is_symlink() => fs::remove_file(&link).map_err(|e| format!("{}: {e}", link.display()))?,
            Ok(_) => return Err(format!("{} is a file, not goethe's link; move it away so the goethe in {} is the one you run", link.display(), home.display())),
            Err(_) => {}
        }
        std::os::unix::fs::symlink(&target, &link).map_err(|e| format!("{}: {e}", link.display()))?;
    }
    Ok(())
}

pub const PATH_MARK: &str = "# added by goethe setup";

pub fn rc_file(user_home: &Path, shell: &str, macos: bool) -> PathBuf {
    // The file a new terminal of this shell reads: zsh ~/.zprofile on a Mac and ~/.zshrc elsewhere, bash ~/.bash_profile on a Mac and ~/.bashrc elsewhere.
    let name = Path::new(shell).file_name().and_then(|n| n.to_str()).unwrap_or("");
    let file = match (name, macos) {
        ("zsh", true) => ".zprofile",
        ("zsh", false) => ".zshrc",
        ("bash", true) => ".bash_profile",
        ("bash", false) => ".bashrc",
        _ => ".profile",
    };
    user_home.join(file)
}

pub fn path_line(dir: &Path, user_home: &Path) -> String {
    let shown = dir.strip_prefix(user_home).map_or_else(|_| dir.display().to_string(), |rel| format!("$HOME/{}", rel.display()));
    format!("export PATH=\"{shown}:$PATH\" {PATH_MARK}")
}

pub fn add_to_rc(rc: &Path, line: &str) -> Result<bool, String> {
    // Append the PATH line once; false when a goethe line is already there.
    let text = fs::read_to_string(rc).unwrap_or_default();
    if text.contains(PATH_MARK) {
        return Ok(false);
    }
    let mut f = fs::OpenOptions::new().create(true).append(true).open(rc).map_err(|e| format!("{}: {e}", rc.display()))?;
    let sep = if text.is_empty() || text.ends_with('\n') { "" } else { "\n" };
    writeln!(f, "{sep}{line}").map_err(|e| format!("{}: {e}", rc.display()))?;
    Ok(true)
}

pub fn ask(question: &str, setting: &str) -> bool {
    // Yes or no for a change outside GOETHE_HOME: the setting (1 or 0) decides, else the terminal, else no.
    match std::env::var(setting).as_deref() {
        Ok("1" | "yes") => return true,
        Ok("0" | "no") => return false,
        _ => {}
    }
    let Ok(mut tty) = fs::OpenOptions::new().read(true).write(true).open("/dev/tty") else {
        return false;
    };
    let _ = write!(tty, "{question} [Y/n] ");
    let _ = tty.flush();
    let mut answer = String::new();
    let _ = std::io::BufRead::read_line(&mut std::io::BufReader::new(&tty), &mut answer);
    let a = answer.trim();
    a.is_empty() || a.starts_with(['y', 'Y'])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_env_file_keeps_only_what_differs_from_the_defaults() {
        assert_eq!(env_file_text("https://pkg.goethe.sh/", None, None, false), "");
        assert_eq!(env_file_text("https://mirror.example/goethe", None, None, false), "GOETHE_PKG_URL=https://mirror.example/goethe\n");
        assert_eq!(env_file_text("file:///tmp/gcdn", None, None, false), "", "a local folder is a one-off");
        assert_eq!(env_file_text("file:///srv/cdn", None, None, true), "GOETHE_PKG_URL=file:///srv/cdn\n");
        assert_eq!(env_file_text("https://m.example", Some("https://m.example/v1/registry.json"), None, false), "GOETHE_PKG_URL=https://m.example\n", "the registry follows the package");
        assert_eq!(
            env_file_text(DEFAULT_PKG_URL, Some("file:///r.json"), Some("tok"), false),
            "GOETHE_REGISTRY_URL=file:///r.json\nGOETHE_PKG_TOKEN=tok\n"
        );
    }

    #[test]
    fn the_link_goes_where_path_already_looks() {
        let home = Path::new("/home/u");
        let all = |_: &Path| true;
        let none = |_: &Path| false;
        let only_local = |d: &Path| d.starts_with("/home/u");
        assert_eq!(link_dir("/usr/bin:/usr/local/bin:/home/u/.local/bin", home, &all), LinkPlan { dir: "/usr/local/bin".into(), on_path: true });
        assert_eq!(link_dir("/usr/local/bin:/home/u/.local/bin", home, &only_local), LinkPlan { dir: "/home/u/.local/bin".into(), on_path: true });
        assert_eq!(link_dir("/usr/bin:/bin", home, &all), LinkPlan { dir: "/home/u/.local/bin".into(), on_path: false }, "a Mac's PATH has no ~/.local/bin");
        assert_eq!(link_dir("/opt/homebrew/bin:/usr/bin", home, &all).dir, PathBuf::from("/opt/homebrew/bin"));
        assert!(!link_dir("/usr/local/bin", home, &none).on_path);
    }

    #[test]
    fn links_point_into_the_home_and_never_replace_a_real_file() {
        let t = tempfile::tempdir().unwrap();
        let (home, dir) = (t.path().join("h"), t.path().join("bin"));
        fs::create_dir_all(home.join("bin")).unwrap();
        link_commands(&home, &dir).unwrap();
        link_commands(&home, &dir).unwrap();
        assert_eq!(fs::read_link(dir.join("goethed")).unwrap(), home.join("bin/goethed"));
        fs::remove_file(dir.join("goethe")).unwrap();
        fs::write(dir.join("goethe"), "copy").unwrap();
        assert!(link_commands(&home, &dir).unwrap_err().contains("is a file"));
    }

    #[test]
    fn the_path_line_goes_into_the_shells_startup_file_once() {
        let home = Path::new("/home/u");
        assert_eq!(rc_file(home, "/bin/zsh", true), home.join(".zprofile"));
        assert_eq!(rc_file(home, "/usr/bin/bash", false), home.join(".bashrc"));
        assert_eq!(rc_file(home, "/usr/bin/fish", false), home.join(".profile"));
        assert_eq!(path_line(&home.join(".local/bin"), home), format!("export PATH=\"$HOME/.local/bin:$PATH\" {PATH_MARK}"));
        let t = tempfile::tempdir().unwrap();
        let rc = t.path().join(".zprofile");
        fs::write(&rc, "alias x=y").unwrap();
        assert!(add_to_rc(&rc, &path_line(Path::new("/x"), home)).unwrap());
        assert!(!add_to_rc(&rc, &path_line(Path::new("/x"), home)).unwrap());
        assert_eq!(fs::read_to_string(&rc).unwrap(), format!("alias x=y\nexport PATH=\"/x:$PATH\" {PATH_MARK}\n"));
    }
}
