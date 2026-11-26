// goethe setup: everything an install does after the bootstrap verified this binary, the same for a fresh install, a reinstall and an upgrade.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const DEFAULT_PKG_URL: &str = "https://pkg.goethe.sh";

pub fn migrate_retired_pkg_env(home: &Path) -> Result<(), String> {
    let text = fs::read_to_string(home.join("goethe.env")).unwrap_or_default();
    if !text.contains("pkg.meshapplied.com") {
        return Ok(());
    }
    let m = crate::paths::parse_env_file(&text);
    write_env_file(
        home,
        &env_file_text(
            m.get("GOETHE_PKG_URL").map(|s| s.as_str()).unwrap_or(DEFAULT_PKG_URL),
            m.get("GOETHE_REGISTRY_URL").map(|s| s.as_str()),
            m.get("GOETHE_PKG_TOKEN").map(|s| s.as_str()),
            true,
        ),
    )
}

fn env_file_text(pkg: &str, registry: Option<&str>, token: Option<&str>, keep_file_url: bool) -> String {
    // GOETHE_HOME/goethe.env holds only what differs from the defaults; a file:// package is a one-off unless asked to keep it.
    let pkg = pkg.trim_end_matches('/');
    let mut out = String::new();
    let local = pkg.starts_with("file://");
    if pkg != DEFAULT_PKG_URL && !pkg.is_empty() && !pkg.contains("pkg.meshapplied.com") && (!local || keep_file_url) {
        out.push_str(&format!("GOETHE_PKG_URL={pkg}\n"));
    }
    if let Some(r) = registry.map(str::trim).filter(|r| !r.is_empty() && !r.contains("pkg.meshapplied.com") && *r != format!("{pkg}/v1/registry.json")) {
        out.push_str(&format!("GOETHE_REGISTRY_URL={r}\n"));
    }
    if let Some(t) = token.map(str::trim).filter(|t| !t.is_empty()) {
        out.push_str(&format!("GOETHE_PKG_TOKEN={t}\n"));
    }
    out
}

fn write_env_file(home: &Path, text: &str) -> Result<(), String> {
    // Mode 0600 because a package token may be in it; no file at all when everything is the default.
    let path = home.join("goethe.env");
    if text.is_empty() {
        return match fs::remove_file(&path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(format!("{}: {e}", path.display())),
            _ => Ok(()),
        };
    }
    crate::client::link::write_private(&path, text.as_bytes())
}

#[derive(Debug, PartialEq, Eq)]
struct LinkPlan {
    pub dir: PathBuf,
    pub on_path: bool,
}

fn link_dir(path_env: &str, user_home: &Path, root: bool, writable: &dyn Fn(&Path) -> bool) -> LinkPlan {
    // Root: /usr/local/bin, on every login PATH even when sudo's secure_path hides it. Others: the first usual folder on PATH they can write, else ~/.local/bin.
    let system = PathBuf::from("/usr/local/bin");
    if root && writable(&system) {
        return LinkPlan { dir: system, on_path: true };
    }
    let local = user_home.join(".local/bin");
    let usual = [local.clone(), user_home.join("bin"), PathBuf::from("/usr/local/bin"), PathBuf::from("/opt/homebrew/bin")];
    for dir in path_env.split(':').filter(|d| !d.is_empty()).map(PathBuf::from) {
        if usual.contains(&dir) && writable(&dir) {
            return LinkPlan { dir, on_path: true };
        }
    }
    LinkPlan { dir: local, on_path: false }
}

fn dir_writable(dir: &Path) -> bool {
    let c = std::ffi::CString::new(dir.as_os_str().as_encoded_bytes()).unwrap_or_default();
    dir.is_dir() && unsafe { libc::access(c.as_ptr(), libc::W_OK) } == 0
}

fn link_commands(home: &Path, dir: &Path) -> Result<(), String> {
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

const PATH_MARK: &str = "# added by goethe setup";

fn rc_file(user_home: &Path, shell: &str, macos: bool) -> PathBuf {
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

fn path_line(dir: &Path, user_home: &Path) -> String {
    let shown = dir.strip_prefix(user_home).map_or_else(|_| dir.display().to_string(), |rel| format!("$HOME/{}", rel.display()));
    format!("export PATH=\"{shown}:$PATH\" {PATH_MARK}")
}

fn add_to_rc(rc: &Path, line: &str) -> Result<bool, String> {
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

pub struct Opts {
    pub pkg: String,
    pub from: Option<String>,
    pub keep_file_url: bool,
    pub platform_url: Option<String>,
    pub no_link: bool,
}

fn say(text: &str) {
    println!("  {} {text}", crate::term::paint("\u{25cf}", crate::term::Color::Green));
}

fn heading(n: usize, of: usize, text: &str) {
    println!("\n{} {}", crate::term::paint(&format!("[{n}/{of}]"), crate::term::Color::Blue), crate::term::paint(text, crate::term::Color::Bold));
}

fn stage_bin(from: &str, sums: &Path, rel: &str, tmp: &Path, dest: &Path, own: Option<&Path>) -> Result<Option<crate::update::Staged>, String> {
    // A binary to replace, or None when the installed one is already the signed build; this goethe is copied instead of downloaded when it is that build.
    let want = crate::update::lookup_sum(sums, rel)?;
    if crate::update::sha256_file(dest).ok().as_deref() == Some(want.as_str()) {
        return Ok(None);
    }
    if let Some(own) = own.filter(|p| crate::update::sha256_file(p).ok().as_deref() == Some(want.as_str())) {
        let part = tmp.join(rel.replace('/', "__"));
        fs::copy(own, &part).map_err(|e| format!("{}: {e}", own.display()))?;
        return Ok(Some(crate::update::Staged { rel: rel.to_string(), part, dest: dest.to_path_buf() }));
    }
    crate::update::stage(from, sums, rel, tmp, dest).map(Some)
}

fn stop_old(home: &Path) -> Result<String, String> {
    // The installed CLI stops its own goethed, so an old daemon gets the command it understands: keep engines when it can, else stop them.
    let old = home.join("bin/goethe");
    let run = |args: &[&str]| std::process::Command::new(&old).args(args).status().map(|s| s.code().unwrap_or(1)).map_err(|e| format!("{}: {e}", old.display()));
    match run(&["down", "--keep-engines"])? {
        0 => Ok("engines kept serving; the new goethed adopts them".to_string()),
        2 => match run(&["stop", "--keep-engines"])? {
            0 => Ok("engines kept serving; the new goethed adopts them".to_string()),
            2 if run(&["stop"])? == 0 => Ok("engines were stopped (this goethed cannot hand them over); the new goethed restores them".to_string()),
            code => Err(format!("could not stop goethed for the upgrade (exit {code}); nothing was replaced")),
        },
        code => Err(format!("could not stop goethed for the upgrade (exit {code}); nothing was replaced")),
    }
}

fn link_step(home: &Path, macos: bool) {
    // Link goethe and goethed onto PATH; when the folder is not on PATH, offer the line for the shell's startup file.
    let user_home = PathBuf::from(std::env::var_os("HOME").unwrap_or_default());
    let plan = link_dir(&std::env::var("PATH").unwrap_or_default(), &user_home, unsafe { libc::geteuid() } == 0, &dir_writable);
    if let Err(e) = link_commands(home, &plan.dir) {
        eprintln!("note: {e}");
        return;
    }
    say(&format!("goethe and goethed linked in {}", plan.dir.display()));
    if plan.on_path {
        return;
    }
    let rc = rc_file(&user_home, &std::env::var("SHELL").unwrap_or_default(), macos);
    let line = path_line(&plan.dir, &user_home);
    if ask(&format!("{} is not on PATH. Add it in {}?", plan.dir.display(), rc.display()), "GOETHE_MODIFY_PATH") {
        match add_to_rc(&rc, &line) {
            Ok(_) => say(&format!("PATH set in {}; open a new terminal, or run: {}", rc.display(), line.split(" #").next().unwrap_or(&line))),
            Err(e) => eprintln!("note: {e}"),
        }
    } else {
        println!("  To run goethe by name, add this line to {}: {}", rc.display(), line.split(" #").next().unwrap_or(&line));
    }
}

fn heartbeat_check(home: &Path) {
    // A linked node with goethed up: wait up to 20 s for the platform to accept a heartbeat, and say plainly when it refuses one.
    if !crate::client::login::is_linked(home) || !crate::service::goethed_running() {
        return;
    }
    let path = crate::client::link::platform_dir(home).join("platform_state.json");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while std::time::Instant::now() < deadline {
        let state: crate::client::link::PlatformState = fs::read_to_string(&path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
        if state.last_ok_at.is_some() && state.last_error.is_none() {
            say("the console accepted this node's heartbeat");
            return;
        }
        if let (Some(e), true) = (&state.last_error, state.failures > 0) {
            eprintln!("note: the console refused this node's heartbeat: {e}; goethe status shows when it recovers");
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
    }
}

fn login_question(link: &crate::client::login::Link) -> Option<&'static str> {
    // Asked when the node is not linked, or its login was revoked or expired (uninstall logs out, so a reinstall lands here); a working link is left alone.
    use crate::client::login::Link;
    match link {
        Link::Linked { revoked: false, .. } | Link::Unreadable(_) => None,
        Link::Linked { revoked: true, .. } => Some("\nThis node's console login was revoked or has expired. Log in again now?"),
        Link::NotLinked => Some("\nLog in to the Goethe console now? Logged-in nodes get optimised kernels for supported models."),
    }
}

fn login_step(home: &Path, platform_url: Option<&str>) {
    // An auth key from the installer links without a browser; otherwise a terminal is offered the device login.
    let goethe = home.join("bin/goethe");
    let mut args = vec!["login".to_string()];
    if let Some(u) = platform_url {
        args.extend(["--platform-url".to_string(), u.to_string()]);
    }
    if let Some(key) = std::env::var("GOETHE_AUTHKEY").ok().filter(|k| !k.is_empty()) {
        args.extend(["--authkey".to_string(), "-".to_string()]);
        let child = std::process::Command::new(&goethe).args(&args).env_remove("GOETHE_AUTHKEY").stdin(std::process::Stdio::piped()).spawn();
        let ok = child.and_then(|mut c| {
            if let Some(mut stdin) = c.stdin.take() {
                writeln!(stdin, "{key}")?;
            }
            c.wait()
        });
        if !ok.is_ok_and(|s| s.success()) {
            eprintln!("WARNING: goethe is installed, but linking to the console failed; retry with: goethe login --authkey <key>");
        }
        return;
    }
    let Some(question) = login_question(&crate::client::login::link_state(home, false)) else {
        return;
    };
    let later = "run goethe login any time to link this node to the Goethe console (logged-in nodes get optimised kernels for supported models)";
    if !ask(question, "GOETHE_LOGIN") {
        println!("Not logged in: {later}");
        return;
    }
    let tty = fs::File::open("/dev/tty").map(std::process::Stdio::from).unwrap_or_else(|_| std::process::Stdio::null());
    if !std::process::Command::new(&goethe).args(&args).stdin(tty).status().is_ok_and(|s| s.success()) {
        eprintln!("Login did not finish; goethe is installed: {later}");
    }
}

pub fn run(o: Opts) -> Result<(), String> {
    // Verify the signed package, place goethe, goethed and nginx (handing a running goethed's engines over), then settings, links, certificate, fixes and login.
    let home = crate::paths::goethe_home();
    crate::paths::ensure_layout(&home).map_err(|e| format!("{}: {e}", home.display()))?;
    fs::set_permissions(&home, std::os::unix::fs::PermissionsExt::from_mode(0o700)).map_err(|e| format!("{}: {e}", home.display()))?;
    let _ = migrate_retired_pkg_env(&home);
    let macos = cfg!(target_os = "macos");
    let from = o.from.clone().unwrap_or_else(|| o.pkg.clone()).trim_end_matches('/').to_string();
    let tmp = crate::update::staging_dir(&home, ".setup-staging")?;
    let result = run_in(&home, &o, &from, &tmp, macos);
    let _ = fs::remove_dir_all(&tmp);
    result
}

fn run_in(home: &Path, o: &Opts, from: &str, tmp: &Path, macos: bool) -> Result<(), String> {
    let total = 3;
    heading(1, total, "Installing goethe, goethed and the gateway");
    if let Some(why) = crate::update::plain_http_refusal(&o.pkg) {
        return Err(why);
    }
    let sums = crate::update::fetch_signed_sums(from, tmp)?;
    let version = fs::read_to_string(crate::update::fetch_verified(from, &sums, "VERSION", tmp)?).map_err(|e| e.to_string())?.trim().to_string();
    let plat = crate::update::platform_dir(&crate::update::normalize_arch());
    let own = std::env::current_exe().ok();
    let mut staged = Vec::new();
    for bin in ["goethe", "goethed"] {
        let rel = format!("bin/{plat}/{bin}");
        staged.extend(stage_bin(from, &sums, &rel, tmp, &home.join("bin").join(bin), own.as_deref().filter(|_| bin == "goethe"))?);
    }
    let nginx_rel = format!("bin/{plat}/nginx");
    let nginx = home.join("nginx-root/usr/sbin/nginx");
    if crate::update::lookup_sum(&sums, &nginx_rel).is_ok() {
        staged.extend(stage_bin(from, &sums, &nginx_rel, tmp, &nginx, None)?);
    } else if !crate::nginx::nginx_bin(home).is_file() {
        return Err(format!("this package has no nginx for {plat} and none is installed; install nginx (or set NGINX_BIN) and run the installer again"));
    }
    let ask = crate::update::ASK_MODEL_REL;
    if crate::update::lookup_sum(&sums, ask).is_ok() {
        staged.extend(stage_bin(from, &sums, ask, tmp, &home.join(ask), None)?);
    } else {
        println!("This package has no Ask model; Ask matches words only until goethe update finds one.");
    }
    let upgrading = home.join("bin/goethe").is_file() && staged.iter().any(|s| !s.rel.ends_with("/nginx") && s.rel != ask);
    let running = upgrading && crate::service::goethed_running();
    let engines = if running { crate::service::live_engine_pids() } else { Vec::new() };
    let note = if running { Some(stop_old(home)?) } else { None };
    let old_version = upgrading.then(|| fs::read_to_string(home.join("VERSION")).unwrap_or_default().trim().to_string()).filter(|v| !v.is_empty());
    crate::update::swap_all(&staged).map_err(|e| if running { format!("{e}; goethed was stopped for the upgrade: start it again with goethe up") } else { e })?;
    crate::service::label_programs(home);
    fs::copy(tmp.join("VERSION"), home.join("VERSION")).map_err(|e| e.to_string())?;
    if running {
        crate::service::refresh_unit(home);
        let kept = engines.iter().filter(|p| crate::service::pid_alive(**p)).count();
        crate::update::start_upgraded(home, engines.len(), kept)?;
    }
    if let Some(n) = note {
        say(&n);
    }
    for f in ["LICENSE", "NOTICE"] {
        if let Ok(p) = crate::update::fetch_verified(from, &sums, f, tmp) {
            let _ = fs::copy(p, home.join(f));
        }
    }
    say(&match (&old_version, staged.is_empty()) {
        (_, true) => format!("goethe {version} is already installed; nothing to replace"),
        (Some(v), false) if v != &version => format!("goethe {v} -> {version}"),
        _ => format!("goethe {version}"),
    });
    say(&format!("nginx: {}", crate::nginx::nginx_bin(home).display()));
    crate::nginx::ensure_certs(home)?;
    say(&format!("TLS certificate: {}", home.join("nginx/certs/server.crt").display()));
    let env = env_file_text(&o.pkg, std::env::var("GOETHE_REGISTRY_URL").ok().as_deref(), std::env::var("GOETHE_PKG_TOKEN").ok().as_deref(), o.keep_file_url);
    write_env_file(home, &env)?;
    if o.pkg.starts_with("file://") && !o.keep_file_url {
        say("installed from a local folder; later updates and downloads come from pkg.goethe.sh");
    }

    heading(2, total, if macos { "Checking this Mac" } else { "Checking this machine" });
    match crate::hw::profile(home).and_then(|s| crate::hw::write_state(home, &s).map(|_| s)) {
        Ok(s) if !s.gpus.is_empty() => say(&s.gpus.iter().map(|g| g.name.clone()).collect::<Vec<_>>().join(", ")),
        Ok(_) | Err(_) => println!("  no usable GPU yet; goethe doctor says why"),
    }
    if !crate::fix::pending(home).is_empty() {
        if let Err(e) = crate::fix::run(false) {
            eprintln!("{e}");
        }
    } else if !macos {
        say("nothing to fix");
    }

    heading(3, total, "Finishing");
    if !o.no_link && std::env::var_os("GOETHE_NO_LINK").is_none() {
        link_step(home, macos);
    }
    if std::env::var("GOETHE_NO_COMPLETION").as_deref() != Ok("1") {
        let _ = std::process::Command::new(home.join("bin/goethe")).args(["completion", "install"]).stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null()).status();
    }
    login_step(home, o.platform_url.as_deref());
    heartbeat_check(home);
    println!("\nGoethe install complete: GOETHE_HOME={}", home.display());
    println!("\nGet started:\n\n\t\u{2192} goethe\n");
    Ok(())
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
        assert_eq!(link_dir("/usr/bin:/usr/local/bin:/home/u/.local/bin", home, false, &all), LinkPlan { dir: "/usr/local/bin".into(), on_path: true });
        assert_eq!(link_dir("/usr/local/bin:/home/u/.local/bin", home, false, &only_local), LinkPlan { dir: "/home/u/.local/bin".into(), on_path: true });
        assert_eq!(link_dir("/usr/bin:/bin", home, false, &all), LinkPlan { dir: "/home/u/.local/bin".into(), on_path: false }, "a Mac's PATH has no ~/.local/bin");
        assert_eq!(link_dir("/opt/homebrew/bin:/usr/bin", home, false, &all).dir, PathBuf::from("/opt/homebrew/bin"));
        assert!(!link_dir("/usr/local/bin", home, false, &none).on_path);
        assert_eq!(link_dir("/sbin:/bin:/usr/sbin:/usr/bin", Path::new("/root"), true, &all), LinkPlan { dir: "/usr/local/bin".into(), on_path: true }, "sudo's secure_path on RHEL");
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
    fn login_is_offered_unless_the_link_works() {
        use crate::client::login::Link;
        let linked = |revoked| Link::Linked { workspace: "w".into(), revoked, heartbeat: String::new(), ok: true };
        assert!(login_question(&Link::NotLinked).unwrap().contains("Log in to the Goethe console now?"));
        assert!(login_question(&linked(true)).unwrap().contains("revoked or has expired"), "a token uninstall revoked");
        assert_eq!(login_question(&linked(false)), None);
    }

    #[test]
    fn retired_pkg_host_is_dropped_from_goethe_env() {
        let t = tempfile::tempdir().unwrap();
        let home = t.path();
        fs::create_dir_all(home).unwrap();
        #[cfg(unix)]
        fs::set_permissions(home, std::os::unix::fs::PermissionsExt::from_mode(0o700)).unwrap();
        crate::client::link::write_private(&home.join("goethe.env"), b"GOETHE_PKG_URL=https://pkg.meshapplied.com\nGOETHE_PKG_TOKEN=tok\n").unwrap();
        migrate_retired_pkg_env(home).unwrap();
        let text = fs::read_to_string(home.join("goethe.env")).unwrap_or_default();
        assert!(!text.contains("pkg.meshapplied.com"), "{text}");
        assert!(text.contains("GOETHE_PKG_TOKEN=tok"), "{text}");
        migrate_retired_pkg_env(home).unwrap();
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
