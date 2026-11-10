// Tab completion setup: the per-shell files, and the marked block in a shell's rc file when the shell has no completion folder of its own.

use std::path::{Path, PathBuf};

const BEGIN: &str = "# goethe completion (goethe completion remove takes this out)";
const END: &str = "# end goethe completion";
const BASH_COMPLETION: &[&str] = &["/usr/share/bash-completion/bash_completion", "/etc/bash_completion"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shell {
    Bash,
    Zsh,
    Fish,
}

impl Shell {
    pub fn parse(name: &str) -> Option<Shell> {
        match Path::new(name.trim()).file_name()?.to_str()? {
            "bash" => Some(Shell::Bash),
            "zsh" => Some(Shell::Zsh),
            "fish" => Some(Shell::Fish),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Shell::Bash => "bash",
            Shell::Zsh => "zsh",
            Shell::Fish => "fish",
        }
    }
}

pub const SHELLS: [Shell; 3] = [Shell::Bash, Shell::Zsh, Shell::Fish];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dirs {
    pub home: PathBuf,
    pub data: PathBuf,
    pub config: PathBuf,
    pub zdot: PathBuf,
    pub bash_completion: bool,
}

impl Dirs {
    pub fn from_env() -> Result<Dirs, String> {
        // XDG folders and ZDOTDIR as the shells use them, falling back to their defaults under HOME.
        let home = std::env::var_os("HOME").map(PathBuf::from).ok_or("HOME is not set")?;
        let xdg = |var: &str, default: &str| std::env::var_os(var).map(PathBuf::from).filter(|p| p.is_absolute()).unwrap_or_else(|| home.join(default));
        Ok(Dirs {
            data: xdg("XDG_DATA_HOME", ".local/share"),
            config: xdg("XDG_CONFIG_HOME", ".config"),
            zdot: std::env::var_os("ZDOTDIR").map(PathBuf::from).unwrap_or_else(|| home.clone()),
            bash_completion: BASH_COMPLETION.iter().any(|p| Path::new(p).is_file()),
            home,
        })
    }

    pub fn script(&self, shell: Shell) -> PathBuf {
        // Where the shell's script goes: its own completion folder for bash and fish, Goethe's data folder for zsh.
        match shell {
            Shell::Bash => self.data.join("bash-completion/completions/goethe"),
            Shell::Zsh => self.data.join("goethe/completion.zsh"),
            Shell::Fish => self.config.join("fish/completions/goethe.fish"),
        }
    }

    pub fn rc(&self, shell: Shell) -> Option<PathBuf> {
        // The rc file that must source the script, when the shell does not load it by itself.
        match shell {
            Shell::Zsh => Some(self.zdot.join(".zshrc")),
            Shell::Bash if !self.bash_completion => Some(self.home.join(".bashrc")),
            _ => None,
        }
    }
}

pub fn rc_block(shell: Shell, script: &Path) -> String {
    let path = script.display().to_string().replace('"', "\\\"");
    let start = match shell {
        Shell::Zsh => "(( $+functions[compdef] )) || { autoload -Uz compinit && compinit; }\n",
        _ => "",
    };
    format!("{BEGIN}\n{start}[ -f \"{path}\" ] && source \"{path}\"\n{END}\n")
}

pub fn without_block(text: &str) -> String {
    // The rc text with Goethe's block taken out, and nothing else touched.
    let mut out = String::new();
    let mut inside = false;
    for line in text.split_inclusive('\n') {
        let bare = line.trim_end_matches('\n');
        if bare == BEGIN {
            inside = true;
        } else if inside && bare == END {
            inside = false;
        } else if !inside {
            out.push_str(line);
        }
    }
    out
}

pub fn with_block(text: &str, block: &str) -> String {
    // The rc text with Goethe's block at the end, replacing an earlier one.
    let mut out = without_block(text);
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    out.push_str(block);
    out
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(path, text).map_err(|e| format!("{}: {e}", path.display()))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Installed {
    Loaded(PathBuf),
    Sourced(PathBuf, PathBuf),
    NeedsRc(PathBuf),
}

pub fn install(dirs: &Dirs, shell: Shell, script: &str, edit_rc: bool) -> Result<Installed, String> {
    // Write the script, then source it from the rc file when the shell needs that (only with edit_rc).
    let path = dirs.script(shell);
    let rc = dirs.rc(shell);
    if let (Some(rc), false) = (&rc, edit_rc) {
        return Ok(Installed::NeedsRc(rc.clone()));
    }
    write(&path, script)?;
    match rc {
        None => Ok(Installed::Loaded(path)),
        Some(rc) => {
            let text = std::fs::read_to_string(&rc).unwrap_or_default();
            write(&rc, &with_block(&text, &rc_block(shell, &path)))?;
            Ok(Installed::Sourced(path, rc))
        }
    }
}

pub fn installed(dirs: &Dirs, shell: Shell) -> bool {
    dirs.script(shell).is_file()
}

pub fn remove(dirs: &Dirs, shell: Shell) -> Result<Vec<PathBuf>, String> {
    // Delete the script and take the block out of the rc files that might hold it; returns what changed.
    let mut changed = Vec::new();
    let path = dirs.script(shell);
    if path.is_file() {
        std::fs::remove_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        changed.push(path);
    }
    let rcs = match shell {
        Shell::Zsh => vec![dirs.zdot.join(".zshrc")],
        Shell::Bash => vec![dirs.home.join(".bashrc")],
        Shell::Fish => Vec::new(),
    };
    for rc in rcs {
        let Ok(text) = std::fs::read_to_string(&rc) else { continue };
        let kept = without_block(&text);
        if kept != text {
            write(&rc, &kept)?;
            changed.push(rc);
        }
    }
    Ok(changed)
}

pub fn remove_all() -> Vec<PathBuf> {
    // goethe uninstall: every shell's setup, quietly.
    let Ok(dirs) = Dirs::from_env() else { return Vec::new() };
    SHELLS.iter().flat_map(|s| remove(&dirs, *s).unwrap_or_default()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dirs(root: &Path, bash_completion: bool) -> Dirs {
        Dirs {
            home: root.to_path_buf(),
            data: root.join(".local/share"),
            config: root.join(".config"),
            zdot: root.to_path_buf(),
            bash_completion,
        }
    }

    #[test]
    fn shells_are_named_by_name_or_path() {
        assert_eq!(Shell::parse("/usr/bin/zsh"), Some(Shell::Zsh));
        assert_eq!(Shell::parse("bash"), Some(Shell::Bash));
        assert_eq!(Shell::parse("fish"), Some(Shell::Fish));
        assert_eq!(Shell::parse("/bin/tcsh"), None);
    }

    #[test]
    fn the_block_is_added_once_replaced_and_removed_leaving_the_rest() {
        let script = Path::new("/h/.local/share/goethe/completion.zsh");
        let rc = "export A=1\nalias ll='ls -l'";
        let once = with_block(rc, &rc_block(Shell::Zsh, script));
        assert!(once.starts_with("export A=1\nalias ll='ls -l'\n# goethe completion"), "{once}");
        assert!(once.contains("compinit") && once.contains("source \"/h/.local/share/goethe/completion.zsh\""));
        assert_eq!(with_block(&once, &rc_block(Shell::Zsh, script)), once, "installing again changes nothing");
        assert_eq!(without_block(&once), "export A=1\nalias ll='ls -l'\n");
        assert_eq!(without_block("no block here\n"), "no block here\n");
        assert!(!rc_block(Shell::Bash, Path::new("/x")).contains("compinit"));
    }

    #[test]
    fn fish_and_bash_completion_get_a_file_zsh_also_an_rc_block() {
        let tmp = tempfile::tempdir().unwrap();
        let d = dirs(tmp.path(), true);
        assert_eq!(install(&d, Shell::Fish, "fish script", false).unwrap(), Installed::Loaded(d.script(Shell::Fish)));
        assert_eq!(std::fs::read_to_string(tmp.path().join(".config/fish/completions/goethe.fish")).unwrap(), "fish script");
        assert_eq!(install(&d, Shell::Bash, "bash script", false).unwrap(), Installed::Loaded(d.script(Shell::Bash)));
        assert_eq!(install(&d, Shell::Zsh, "zsh script", false).unwrap(), Installed::NeedsRc(tmp.path().join(".zshrc")), "--auto never edits an rc file");
        assert!(!installed(&d, Shell::Zsh));
        std::fs::write(tmp.path().join(".zshrc"), "export A=1\n").unwrap();
        assert_eq!(install(&d, Shell::Zsh, "zsh script", true).unwrap(), Installed::Sourced(d.script(Shell::Zsh), tmp.path().join(".zshrc")));
        assert!(std::fs::read_to_string(tmp.path().join(".zshrc")).unwrap().contains("# goethe completion"));
        let bare = dirs(tmp.path(), false);
        assert!(matches!(install(&bare, Shell::Bash, "bash script", true).unwrap(), Installed::Sourced(_, _)), "bash without bash-completion sources it from .bashrc");
        let removed = remove(&d, Shell::Zsh).unwrap();
        assert_eq!(removed.len(), 2);
        assert_eq!(std::fs::read_to_string(tmp.path().join(".zshrc")).unwrap(), "export A=1\n");
        assert_eq!(remove(&bare, Shell::Bash).unwrap().len(), 2);
        assert!(remove(&d, Shell::Bash).unwrap().is_empty(), "removing twice is a no-op");
    }
}
