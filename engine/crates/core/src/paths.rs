// GOETHE_HOME resolution, model id validation, and paths under GOETHE_HOME.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub const HOME_MARKER: &str = ".goethe-home";
const ENV_FILE_KEYS: [&str; 3] = ["GOETHE_PKG_URL", "GOETHE_REGISTRY_URL", "GOETHE_PKG_TOKEN"];

pub fn goethe_home() -> PathBuf {
    // GOETHE_HOME, else the home this binary is installed in, else the default probe (once per process).
    if let Ok(v) = std::env::var("GOETHE_HOME") {
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    static DEFAULT_HOME: OnceLock<PathBuf> = OnceLock::new();
    DEFAULT_HOME
        .get_or_init(|| {
            if let Some(home) = std::env::current_exe().ok().and_then(|exe| home_of_binary(&exe)) {
                return home;
            }
            let user_home = std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/tmp"));
            default_home(Path::new("/"), &user_home)
        })
        .clone()
}

pub fn home_of_binary(exe: &Path) -> Option<PathBuf> {
    // <home>/bin/goethe or <home>/bin/goethed (after resolving links) inside a marked goethe home.
    let exe = std::fs::canonicalize(exe).ok()?;
    let name = exe.file_name()?.to_str()?;
    let bin = exe.parent()?;
    let home = bin.parent()?;
    (matches!(name, "goethe" | "goethed") && bin.file_name()? == "bin" && home.join(HOME_MARKER).is_file())
        .then(|| home.to_path_buf())
}

pub fn setting(key: &str) -> Option<String> {
    // Process env first, then GOETHE_HOME/goethe.env for the installer's package and registry keys.
    let from_env = std::env::var(key).ok().filter(|v| !v.is_empty());
    let file = || std::fs::read_to_string(goethe_home().join("goethe.env")).unwrap_or_default();
    from_env.or_else(|| ENV_FILE_KEYS.contains(&key).then(|| parse_env_file(&file()).remove(key)).flatten())
}

pub fn parse_env_file(text: &str) -> BTreeMap<String, String> {
    // KEY=VALUE lines (optionally quoted) for the allowed keys only; comments and other keys are ignored.
    text.lines()
        .filter_map(|l| l.trim().split_once('='))
        .map(|(k, v)| (k.trim().trim_start_matches("export ").to_string(), v.trim().trim_matches(['"', '\'']).to_string()))
        .filter(|(k, v)| ENV_FILE_KEYS.contains(&k.as_str()) && !v.is_empty())
        .collect()
}

pub fn daemon_err(e: String) -> String {
    // Name the home and socket that were tried when goethed is not reachable.
    let e = crate::rpc::map_daemon_err(e);
    if e.contains("goethed is not running") {
        format!("{e} (looked in GOETHE_HOME {} at socket {})", goethe_home().display(), goethe_sock().display())
    } else {
        e
    }
}

pub fn default_home(root: &Path, user_home: &Path) -> PathBuf {
    // Existing /workspace/.goethe, writable /workspace, writable /var/lib/goethe, then ~/.goethe.
    let workspace = root.join("workspace");
    let existing = workspace.join(".goethe");
    if existing.is_dir() {
        return existing;
    }
    if workspace.is_dir() && is_writable(&workspace) {
        return existing;
    }
    let varlib = root.join("var/lib/goethe");
    if varlib.is_dir() && is_writable(&varlib) {
        return varlib;
    }
    user_home.join(".goethe")
}

pub fn is_writable(path: &Path) -> bool {
    let Ok(c) = CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    unsafe { libc::access(c.as_ptr(), libc::W_OK) == 0 }
}

pub fn goethe_sock() -> PathBuf {
    if let Ok(v) = std::env::var("GOETHE_SOCK") {
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    let euid = unsafe { libc::geteuid() };
    if euid == 0 {
        PathBuf::from("/var/run/goethe.sock")
    } else {
        goethe_home().join("goethe.sock")
    }
}

pub fn ensure_layout(home: &Path) -> std::io::Result<()> {
    // Create the standard subdirectories and the marker that identifies a goethe home.
    for sub in ["bin", "models", "logs", "nginx", "run"] {
        std::fs::create_dir_all(home.join(sub))?;
    }
    let marker = home.join(HOME_MARKER);
    if !marker.exists() {
        std::fs::write(&marker, "goethe home\n")?;
    }
    Ok(())
}

pub fn validate_model_id(model_id: &str) -> Result<(), String> {
    // Accept org/name, local/name, or a bare name; reject anything that could leave models/.
    if model_id.is_empty() {
        return Err("model id must not be empty".to_string());
    }
    if model_id.len() > 200 {
        return Err("model id is longer than 200 characters".to_string());
    }
    if model_id.starts_with('/') {
        return Err(format!("{model_id:?} is a path, not a model id"));
    }
    let parts: Vec<&str> = model_id.split('/').collect();
    if parts.len() > 2 {
        return Err(format!("model id {model_id:?} must look like org/name"));
    }
    for part in parts {
        if part.is_empty() || part == "." || part == ".." {
            return Err(format!("model id {model_id:?} has an empty, \".\" or \"..\" component"));
        }
        if !part
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        {
            return Err(format!(
                "model id {model_id:?} may only contain letters, digits, '.', '_', '-' and one '/'"
            ));
        }
    }
    Ok(())
}

pub fn sanitize_model_id(model_id: &str) -> String {
    model_id.replace('/', "__")
}

pub fn model_dir(home: &Path, model_id: &str) -> PathBuf {
    home.join("models").join(sanitize_model_id(model_id))
}

pub fn contained_in_models(home: &Path, target: &Path) -> Result<PathBuf, String> {
    // Resolve target without following its last component and require it strictly inside models/.
    let models = std::fs::canonicalize(home.join("models"))
        .map_err(|e| format!("{}: {e}", home.join("models").display()))?;
    let name = match target.file_name() {
        Some(n) if n != "." && n != ".." => n.to_owned(),
        _ => return Err(format!("refusing to delete {}", target.display())),
    };
    let parent = target
        .parent()
        .ok_or_else(|| format!("refusing to delete {}", target.display()))?;
    let parent = std::fs::canonicalize(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    let resolved = parent.join(name);
    if resolved == models || !resolved.starts_with(&models) {
        return Err(format!(
            "refusing to delete {}: it is not inside {}",
            resolved.display(),
            models.display()
        ));
    }
    Ok(resolved)
}

pub fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id_validation() {
        for ok in ["org/name", "local/my-model", "k", "Qwen/Qwen3.8-27B", "a_b/c.d"] {
            assert!(validate_model_id(ok).is_ok(), "{ok}");
        }
        for bad in [
            "", ".", "..", "/abs", "org/..", "../x", "a//b", "org/", "/", "a/b/c", "a\0b", "org/na me",
            "local/..",
        ] {
            assert!(validate_model_id(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn remove_target_must_stay_inside_models() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        std::fs::create_dir_all(home.join("models/org__m")).unwrap();
        let ok = contained_in_models(home, &home.join("models/org__m")).unwrap();
        assert!(ok.ends_with("models/org__m"));
        assert!(contained_in_models(home, &home.join("models")).is_err());
        assert!(contained_in_models(home, &home.join("models/..")).is_err());
        assert!(contained_in_models(home, &home.join("models/.")).is_err());
        assert!(contained_in_models(home, home).is_err());
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), home.join("models/link")).unwrap();
        let link = contained_in_models(home, &home.join("models/link")).unwrap();
        assert!(link.ends_with("models/link"));
        std::fs::create_dir_all(outside.path().join("victim")).unwrap();
        assert!(contained_in_models(home, &home.join("models/link/victim")).is_err());
    }

    #[test]
    fn default_home_prefers_existing_workspace_home() {
        let root = tempfile::tempdir().unwrap();
        let user = root.path().join("user");
        assert_eq!(default_home(root.path(), &user), user.join(".goethe"));
        std::fs::create_dir_all(root.path().join("var/lib/goethe")).unwrap();
        assert_eq!(default_home(root.path(), &user), root.path().join("var/lib/goethe"));
        std::fs::create_dir_all(root.path().join("workspace")).unwrap();
        assert_eq!(default_home(root.path(), &user), root.path().join("workspace/.goethe"));
    }

    #[test]
    fn home_follows_the_installed_binary() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("custom-home");
        std::fs::create_dir_all(home.join("bin")).unwrap();
        std::fs::write(home.join("bin/goethe"), "bin").unwrap();
        assert_eq!(home_of_binary(&home.join("bin/goethe")), None);
        std::fs::write(home.join(HOME_MARKER), "").unwrap();
        let canonical = std::fs::canonicalize(&home).unwrap();
        assert_eq!(home_of_binary(&home.join("bin/goethe")), Some(canonical.clone()));
        let link = dir.path().join("local-bin-goethe");
        std::os::unix::fs::symlink(home.join("bin/goethe"), &link).unwrap();
        assert_eq!(home_of_binary(&link), Some(canonical));
        std::fs::write(home.join("bin/other"), "bin").unwrap();
        assert_eq!(home_of_binary(&home.join("bin/other")), None);
    }

    #[test]
    fn env_file_keeps_only_installer_keys() {
        let text = "# installer\nGOETHE_PKG_URL=http://10.0.0.5:8000\nexport GOETHE_REGISTRY_URL=\"file:///r.json\"\nPATH=/evil\nGOETHE_PKG_TOKEN=\n";
        let m = parse_env_file(text);
        assert_eq!(m.get("GOETHE_PKG_URL").map(String::as_str), Some("http://10.0.0.5:8000"));
        assert_eq!(m.get("GOETHE_REGISTRY_URL").map(String::as_str), Some("file:///r.json"));
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn shell_quote_escapes_single_quotes() {
        assert_eq!(shell_quote("/a b/it's"), r"'/a b/it'\''s'");
    }
}
