// The container image's start: the home on the /workspace volume, the console link and API key from the environment, goethed in the foreground, then GOETHE_SETUP.

use crate::ask::parse::{find_shell_metachar, split_words};
use clap::Parser;
use goethe_core::paths::goethe_home;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const IMAGE_FILES: [(&str, &str); 5] = [
    ("bin/goethe", "bin/goethe"),
    ("bin/goethed", "bin/goethed"),
    ("bin/uv", "bin/uv"),
    ("bin/nginx", "nginx-root/usr/sbin/nginx"),
    ("ask/ask-model-v1.bin", "ask/ask-model-v1.bin"),
];
const SETUP_DONE: &str = "container-setup.done";
const SECRET_ENV: [&str; 3] = ["GOETHE_API_KEY", "GOETHE_AUTHKEY", "GOETHE_HF_TOKEN"];
const GOETHED_READY: Duration = Duration::from_secs(60);

pub fn run(check: bool) -> i32 {
    // goethe container: the image's main process under tini, which passes the container's stop signal to goethed; --check is the image's health check.
    if check {
        return i32::from(!goethe_core::service::goethed_running());
    }
    let Some(version) = goethe_core::paths::image() else {
        eprintln!("error: goethe container is the start command of the Goethe container image; on a machine, use goethe up");
        return 2;
    };
    let home = goethe_home();
    println!("goethe {version}, home {}", home.display());
    if let Err(e) = prepare(&home) {
        eprintln!("error: {e}");
        return 1;
    }
    if let Err(e) = api_key(&home) {
        eprintln!("error: GOETHE_API_KEY: {e}");
        return 2;
    }
    link(&home);
    let setup = std::env::var("GOETHE_SETUP").unwrap_or_default();
    let lines = setup_lines(&setup);
    if let Err(e) = &lines {
        eprintln!("error: GOETHE_SETUP is not run: {e}");
    }
    let mut goethed = match without_secrets(&mut Command::new(home.join("bin/goethed"))).stdin(Stdio::null()).spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: cannot start goethed: {e}");
            return 1;
        }
    };
    let since = Instant::now();
    while !goethe_core::service::goethed_running() && since.elapsed() < GOETHED_READY {
        if let Ok(Some(st)) = goethed.try_wait() {
            eprintln!("error: goethed stopped while starting ({st})");
            return st.code().unwrap_or(1);
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    match lines {
        Ok(lines) if !lines.is_empty() => run_setup(&home, &setup, &lines),
        _ => {}
    }
    match goethed.wait() {
        Ok(st) => st.code().unwrap_or(0),
        Err(e) => {
            eprintln!("error: goethed: {e}");
            1
        }
    }
}

fn prepare(home: &Path) -> Result<(), String> {
    // The home's layout, the image's programs linked into it (a newer image takes over on restart), a node id that outlives the container, the TLS certificate and the hardware profile.
    goethe_core::paths::ensure_layout(home).map_err(|e| format!("{}: {e}", home.display()))?;
    std::fs::set_permissions(home, std::os::unix::fs::PermissionsExt::from_mode(0o700)).map_err(|e| format!("{}: {e}", home.display()))?;
    let image = std::env::current_exe().map_err(|e| e.to_string())?.parent().and_then(Path::parent).map(Path::to_path_buf).ok_or("cannot locate the image's programs")?;
    for (in_image, in_home) in IMAGE_FILES {
        let (from, to) = (image.join(in_image), home.join(in_home));
        if !from.is_file() {
            return Err(format!("the image has no {}", from.display()));
        }
        if let Some(dir) = to.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let _ = std::fs::remove_file(&to);
        std::os::unix::fs::symlink(&from, &to).map_err(|e| format!("{}: {e}", to.display()))?;
    }
    goethe_core::client::link::ensure_home_machine_id(home)?;
    goethe_core::nginx::ensure_certs(home)?;
    match goethe_core::hw::profile(home).and_then(|s| goethe_core::hw::write_state(home, &s).map(|_| s)) {
        Ok(s) if !s.gpus.is_empty() => println!("GPUs: {}", s.gpus.iter().map(|g| g.name.as_str()).collect::<Vec<_>>().join(", ")),
        _ => println!("no usable GPU; run the container with --gpus all (goethe doctor says more)"),
    }
    Ok(())
}

fn api_key(home: &Path) -> Result<(), String> {
    // GOETHE_API_KEY registers the user's own secret under GOETHE_API_KEY_NAME (default "default"); only its hash is kept and it is never printed.
    let Some(secret) = std::env::var("GOETHE_API_KEY").ok().filter(|s| !s.is_empty()) else {
        return Ok(());
    };
    let name = std::env::var("GOETHE_API_KEY_NAME").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "default".to_string());
    let changed = goethe_core::security::import_key(home, &name, &secret)?;
    println!("API key {name}: {}", if changed { "set from GOETHE_API_KEY" } else { "unchanged" });
    Ok(())
}

fn without_secrets(cmd: &mut Command) -> &mut Command {
    // The container's secrets were used at start; goethed, its engines and the GOETHE_SETUP lines never inherit them.
    for k in SECRET_ENV {
        cmd.env_remove(k);
    }
    cmd
}

fn link(home: &Path) {
    // GOETHE_AUTHKEY links this node to the Goethe console once; a failure is said and the node serves anyway.
    let Some(key) = std::env::var("GOETHE_AUTHKEY").ok().filter(|s| !s.is_empty()) else {
        return;
    };
    if goethe_core::client::login::is_linked(home) {
        return;
    }
    if let Err(e) = goethe_core::client::login::login(None, Some(&key)) {
        eprintln!("error: console link with GOETHE_AUTHKEY failed: {e}");
    }
}

pub fn setup_lines(text: &str) -> Result<Vec<Vec<String>>, String> {
    // One goethe command per line ("goethe" in front is optional, # starts a comment), each checked by clap before any runs; no shell is involved.
    let mut out = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let at = || format!("line {} ({line})", n + 1);
        if let Some(meta) = find_shell_metachar(line) {
            return Err(format!("{}: these are goethe commands, not a shell: '{meta}'", at()));
        }
        let mut words = split_words(line);
        if words.first().is_some_and(|w| w == "goethe") {
            words.remove(0);
        }
        if words.first().is_some_and(|w| w == "container") {
            return Err(format!("{}: goethe container is the image's own start", at()));
        }
        crate::Cli::try_parse_from(std::iter::once("goethe".to_string()).chain(words.iter().cloned()))
            .map_err(|e| format!("{}: {}", at(), first_paragraph(&e.to_string())))?;
        out.push(words);
    }
    Ok(out)
}

fn first_paragraph(clap_error: &str) -> String {
    // clap's message up to its usage block, on one line ("the following required arguments were not provided: <KEY_NAME>").
    clap_error.lines().take_while(|l| !l.trim().is_empty()).map(str::trim).collect::<Vec<_>>().join(" ").trim_start_matches("error: ").to_string()
}

fn run_setup(home: &Path, text: &str, lines: &[Vec<String>]) {
    // Each line in order, once per version of GOETHE_SETUP; the first failure stops the rest and the next start tries again. GOETHE_HF_TOKEN reaches only the lines that ask for it, on stdin.
    let done = home.join(SETUP_DONE);
    let hash = goethe_core::security::token_hash(text);
    if std::fs::read_to_string(&done).is_ok_and(|h| h.trim() == hash) {
        println!("GOETHE_SETUP: unchanged since it last ran");
        return;
    }
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("error: GOETHE_SETUP: {e}");
            return;
        }
    };
    let hf_token = std::env::var("GOETHE_HF_TOKEN").ok().filter(|t| !t.is_empty());
    for words in lines {
        println!("GOETHE_SETUP: goethe {}", words.join(" "));
        let token = hf_token.as_deref().filter(|_| words.iter().any(|w| w == "--hf-token"));
        let mut cmd = Command::new(&exe);
        without_secrets(&mut cmd).args(words).stdin(if token.is_some() { Stdio::piped() } else { Stdio::null() });
        let ok = cmd.spawn().and_then(|mut child| {
            if let (Some(t), Some(mut stdin)) = (token, child.stdin.take()) {
                writeln!(stdin, "{t}")?;
            }
            child.wait()
        });
        let failed = match ok {
            Ok(st) if st.success() => continue,
            Ok(st) => st.to_string(),
            Err(e) => e.to_string(),
        };
        eprintln!("error: GOETHE_SETUP stopped at goethe {} ({failed}); the next start tries again", words.join(" "));
        return;
    }
    if let Err(e) = std::fs::write(&done, format!("{hash}\n")) {
        eprintln!("error: {}: {e}", done.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_container_secrets_are_not_passed_on() {
        let mut cmd = Command::new("true");
        without_secrets(&mut cmd);
        let removed: Vec<_> = cmd.get_envs().filter(|(_, v)| v.is_none()).map(|(k, _)| k.to_string_lossy().into_owned()).collect();
        assert_eq!(removed, SECRET_ENV);
    }

    #[test]
    fn setup_lines_are_goethe_commands_checked_before_any_runs() {
        // Comments, blank lines and a leading "goethe" are fine; one bad line refuses the whole text, and so does anything shell-like.
        let ok = setup_lines("# serve a model\ngoethe install org/m --load\n\napi expose public --allow 203.0.113.7/32 --yes\n").unwrap();
        assert_eq!(ok, [vec!["install", "org/m", "--load"], vec!["api", "expose", "public", "--allow", "203.0.113.7/32", "--yes"]]);
        assert_eq!(setup_lines("models alias set fast 'org/m'").unwrap(), [vec!["models", "alias", "set", "fast", "org/m"]]);
        assert!(setup_lines("").unwrap().is_empty());
        let bad = setup_lines("install org/m\napi keys create\n").unwrap_err();
        assert!(bad.starts_with("line 2 (api keys create)") && bad.contains("KEY_NAME"), "{bad}");
        assert!(setup_lines("install org/m; rm -rf /").unwrap_err().contains("not a shell"));
        assert!(setup_lines("install $(id)").is_err());
        assert!(setup_lines("container").unwrap_err().contains("own start"));
        assert!(setup_lines("frobnicate").is_err());
    }
}
