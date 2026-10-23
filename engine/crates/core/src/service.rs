// Boot-time goethed enablement: systemd (system or user), crontab, or nohup in containers.

use crate::paths::{goethe_home, goethe_sock, shell_quote};
use std::fs;
use std::io::Write;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const CRON_MARKER: &str = "# goethe-goethed-reboot";

pub fn run_enable() -> Result<(), String> {
    // Prefer systemd when it is PID 1; containers without it get cron and a nohup start.
    let home = goethe_home();
    let goethed = home.join("bin/goethed");
    if !goethed.is_file() {
        return Err(format!("goethed not found at {}", goethed.display()));
    }
    let env = crate::host::probe_environment(crate::nginx::public_port_u16(&home));
    if env.systemd_pid1 {
        if try_system_unit(&home)? {
            println!("enabled goethed boot via systemd (system unit)");
            return Ok(());
        }
        if try_user_unit(&home)? {
            println!("enabled goethed boot via systemd (--user unit)");
            return Ok(());
        }
    }
    if env.container.is_some() && !env.systemd_pid1 {
        let cron = install_crontab_reboot(&home).is_ok();
        if !goethed_running() {
            spawn_goethed(&home)?;
            println!("started goethed with nohup (no init system in this container)");
        }
        if cron {
            println!("added a crontab @reboot entry (it only runs if cron runs in the container)");
        }
        println!(
            "containers restart without init: add {} & to the container start command so goethed comes back",
            shell_quote(&goethed.display().to_string())
        );
        return Ok(());
    }
    install_crontab_reboot(&home)?;
    println!("enabled goethed boot via crontab @reboot");
    Ok(())
}

pub fn run_disable() -> Result<(), String> {
    // Remove only the autostart entries that start this GOETHE_HOME; other homes keep theirs.
    let home = goethe_home();
    let mut actions = 0usize;
    if remove_system_unit(&home) {
        println!("disabled goethed systemd (system unit)");
        actions += 1;
    }
    if remove_user_unit(&home) {
        println!("disabled goethed systemd (--user unit)");
        actions += 1;
    }
    if remove_crontab_reboot(&home)? {
        println!("removed goethed crontab @reboot entry");
        actions += 1;
    }
    if actions == 0 {
        println!("goethed boot was not enabled");
    }
    Ok(())
}

pub fn run_status() -> Result<(), String> {
    let home = goethe_home();
    let env = crate::host::probe_environment(crate::nginx::public_port_u16(&home));
    println!("boot method: {}", active_method(&home));
    println!("goethed: {}", if goethed_running() { "running" } else { "stopped" });
    println!(
        "init: {}{}",
        if env.systemd_pid1 { "systemd" } else { "no systemd" },
        env.container.map(|c| format!(", container {c}")).unwrap_or_default()
    );
    Ok(())
}

fn active_method(home: &Path) -> String {
    // This home's boot method, naming a unit that belongs to another home.
    let mut other = String::new();
    for (path, label) in [(system_unit_path(), "systemd (system)"), (user_unit_path(), "systemd (--user)")] {
        match unit_owner(&path, home) {
            UnitOwner::Ours => return label.into(),
            UnitOwner::Other(h) => other = format!(" ({label} unit starts {})", describe_home(h.as_deref())),
            UnitOwner::Absent => {}
        }
    }
    let method = if crontab_entry_for(&read_crontab(), home) { "crontab @reboot" } else { "none" };
    format!("{method}{other}")
}

pub fn goethed_running() -> bool {
    std::os::unix::net::UnixStream::connect(goethe_sock()).is_ok()
}

pub fn live_engine_pids() -> Vec<i64> {
    // Leader PIDs of RUNNING engines, from goethed's ps.
    let Ok(reply) = crate::rpc::call("ps", serde_json::Value::Null) else {
        return Vec::new();
    };
    let rows: serde_json::Value = serde_json::from_str(reply.stdout.trim()).unwrap_or_default();
    rows.as_array()
        .map(|a| {
            a.iter()
                .filter(|r| r.get("status").and_then(|s| s.as_str()) == Some("RUNNING"))
                .filter_map(|r| r.get("pid").and_then(|p| p.as_i64()))
                .filter(|p| *p > 0)
                .collect()
        })
        .unwrap_or_default()
}

pub fn pid_alive(pid: i64) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

pub fn stop_keeping_engines() -> Result<(usize, usize), String> {
    // shutdown::stop_daemon(true) (contract C-D); returns the RUNNING engines before and how many are still alive.
    let before = if goethed_running() { live_engine_pids() } else { Vec::new() };
    crate::shutdown::stop_daemon(true)?;
    Ok((before.len(), before.iter().filter(|p| pid_alive(**p)).count()))
}

pub fn kept_text(before: usize, kept: usize) -> String {
    if kept < before {
        format!("{kept} of {before} engine(s) still run: the stopped goethed tore the rest down; the next goethed restores them")
    } else {
        format!("{kept} engine(s) kept serving")
    }
}

pub fn run_up() -> Result<(), String> {
    // Start this home's goethed, through its systemd unit when that unit starts this home, and wait until it answers.
    let home = goethe_home();
    if goethed_running() {
        println!("goethed is already running (GOETHE_HOME {})", home.display());
        return Ok(());
    }
    match systemctl_restart_args() {
        Some(args) => {
            let args: Vec<String> = args.into_iter().map(|a| if a == "restart" { "start".to_string() } else { a }).collect();
            let ok = Command::new("systemctl").args(&args).status().is_ok_and(|s| s.success());
            if !ok {
                return Err(format!("systemctl {} failed (see journalctl for goethed.service)", args.join(" ")));
            }
        }
        None => spawn_goethed(&home)?,
    }
    wait_goethed(std::time::Duration::from_secs(60))?;
    println!("goethed is up (GOETHE_HOME {}); goethe ps shows loaded models, goethe get api the endpoint", home.display());
    Ok(())
}

pub fn wait_goethed(timeout: std::time::Duration) -> Result<(), String> {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if goethed_running() {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    Err(format!("goethed did not come up within {} s (see goethe logs --daemon)", timeout.as_secs()))
}

pub fn refresh_unit(home: &Path) {
    // Rewrite this home's unit when it predates the current template (KillMode=process lets engines outlive goethed).
    let units = [(system_unit_path(), "multi-user.target", false), (user_unit_path(), "default.target", true)];
    for (path, wanted_by, user) in units {
        let current = fs::read_to_string(&path).ok();
        if !matches!(unit_owner(&path, home), UnitOwner::Ours) || current.as_deref() == Some(unit_text(home, wanted_by).as_str()) {
            continue;
        }
        if (user || unsafe { libc::geteuid() } == 0) && write_unit(&path, home, wanted_by).is_ok() {
            systemctl_quiet(if user { &["--user", "daemon-reload"][..] } else { &["daemon-reload"][..] });
        }
    }
}

pub fn spawn_goethed(home: &Path) -> Result<(), String> {
    spawn(home, true)
}

pub fn respawn_goethed(home: &Path) -> Result<(), String> {
    // After a keep-engines stop: the new goethed takes the persisted ports, where the adopted gateway still listens.
    spawn(home, false)
}

fn spawn(home: &Path, port_env: bool) -> Result<(), String> {
    // Start this home's goethed (else the one beside this CLI) detached, logging to GOETHE_HOME/logs/goethed.log.
    let beside_cli = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("goethed")));
    let goethed = std::iter::once(home.join("bin/goethed"))
        .chain(beside_cli)
        .find(|p| p.is_file())
        .unwrap_or_else(|| home.join("bin/goethed"));
    fs::create_dir_all(home.join("logs")).map_err(|e| e.to_string())?;
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(home.join("logs/goethed.log"))
        .map_err(|e| e.to_string())?;
    let mut cmd = Command::new(&goethed);
    if !port_env {
        cmd.env_remove("GOETHE_MUX_PORT").env_remove("GOETHE_PUBLIC_PORT");
    }
    cmd.env("GOETHE_HOME", home)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone().map_err(|e| e.to_string())?))
        .stderr(Stdio::from(log));
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    cmd.spawn().map_err(|e| format!("start {}: {e}", goethed.display()))?;
    Ok(())
}

pub fn systemctl_restart_args() -> Option<Vec<String>> {
    // Restart through systemd only when the real unit starts this GOETHE_HOME.
    let home = goethe_home();
    let ours = |path: PathBuf| matches!(unit_owner(&path, &home), UnitOwner::Ours);
    if unsafe { libc::geteuid() } == 0 && system_unit_dir() == default_system_unit_dir() && ours(system_unit_path()) {
        return Some(vec!["restart".into(), "goethed.service".into()]);
    }
    if user_unit_dir() == default_user_unit_dir() && ours(user_unit_path()) {
        return Some(vec!["--user".into(), "restart".into(), "goethed.service".into()]);
    }
    None
}

fn default_system_unit_dir() -> PathBuf {
    PathBuf::from("/etc/systemd/system")
}

fn default_user_unit_dir() -> PathBuf {
    std::env::var("HOME")
        .map(|h| PathBuf::from(h).join(".config/systemd/user"))
        .unwrap_or_else(|_| PathBuf::from("/nonexistent"))
}

fn system_unit_dir() -> PathBuf {
    std::env::var("GOETHE_SYSTEMD_UNIT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_system_unit_dir())
}

fn user_unit_dir() -> PathBuf {
    std::env::var("GOETHE_SYSTEMD_USER_UNIT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| default_user_unit_dir())
}

fn system_unit_path() -> PathBuf {
    system_unit_dir().join("goethed.service")
}

fn user_unit_path() -> PathBuf {
    user_unit_dir().join("goethed.service")
}

fn try_system_unit(home: &Path) -> Result<bool, String> {
    if unsafe { libc::geteuid() } != 0 {
        return Ok(false);
    }
    refuse_other_home(&system_unit_path(), home)?;
    write_unit(&system_unit_path(), home, "multi-user.target")?;
    systemctl_quiet(&["daemon-reload"]);
    systemctl_quiet(&enable_args(&[]));
    Ok(true)
}

fn try_user_unit(home: &Path) -> Result<bool, String> {
    // A user unit needs a working user systemd manager.
    let ok = Command::new("systemctl")
        .args(["--user", "status"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return Ok(false);
    }
    refuse_other_home(&user_unit_path(), home)?;
    fs::create_dir_all(user_unit_dir()).map_err(|e| e.to_string())?;
    write_unit(&user_unit_path(), home, "default.target")?;
    systemctl_quiet(&["--user", "daemon-reload"]);
    systemctl_quiet(&enable_args(&["--user"]));
    eprintln!("hint: run `loginctl enable-linger $USER` so goethed starts at boot without a login session");
    Ok(true)
}

enum UnitOwner {
    Absent,
    Ours,
    Other(Option<PathBuf>),
}

fn unit_owner(path: &Path, home: &Path) -> UnitOwner {
    // Whether a goethed unit exists and whether it starts this GOETHE_HOME.
    if !path.is_file() {
        return UnitOwner::Absent;
    }
    match fs::read_to_string(path).ok().and_then(|t| unit_text_home(&t)) {
        Some(h) if same_home(&h, home) => UnitOwner::Ours,
        other => UnitOwner::Other(other),
    }
}

pub fn unit_text_home(text: &str) -> Option<PathBuf> {
    // The GOETHE_HOME a unit pins in its Environment= lines.
    text.lines()
        .filter_map(|l| l.trim().strip_prefix("Environment="))
        .flat_map(env_assignments)
        .find_map(|a| a.strip_prefix("GOETHE_HOME=").map(PathBuf::from))
}

fn env_assignments(value: &str) -> Vec<String> {
    // Split a systemd Environment= value into assignments, honoring double quotes and backslash escapes.
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut chars = value.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => cur.extend(chars.next()),
            '"' => quoted = !quoted,
            c if c.is_whitespace() && !quoted => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

fn same_home(a: &Path, b: &Path) -> bool {
    a == b || matches!((fs::canonicalize(a), fs::canonicalize(b)), (Ok(x), Ok(y)) if x == y)
}

fn describe_home(home: Option<&Path>) -> String {
    home.map(|h| format!("GOETHE_HOME {}", h.display())).unwrap_or_else(|| "an unnamed GOETHE_HOME".into())
}

fn refuse_other_home(path: &Path, home: &Path) -> Result<(), String> {
    // One goethed unit per user: refuse to take over the unit of another home.
    let UnitOwner::Other(other) = unit_owner(path, home) else {
        return Ok(());
    };
    let how = other
        .as_deref()
        .map(|h| format!("GOETHE_HOME={} goethe service disable", shell_quote(&h.display().to_string())))
        .unwrap_or_else(|| format!("goethe service disable from that home, or remove {}", path.display()));
    Err(format!(
        "{} already starts goethed for {}; this home ({}) cannot get a unit too. Disable the other one first ({how})",
        path.display(),
        describe_home(other.as_deref()),
        home.display()
    ))
}

fn enable_args<'a>(prefix: &[&'a str]) -> Vec<&'a str> {
    let mut args = prefix.to_vec();
    args.push("enable");
    if !goethed_running() {
        args.push("--now");
    }
    args.push("goethed.service");
    args
}

fn systemctl_quiet(args: &[&str]) {
    let _ = Command::new("systemctl")
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

fn systemd_quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

pub fn unit_text(home: &Path, wanted_by: &str) -> String {
    // Unit with quoted paths and the installer's goethe.env; goethed stops its own engines, so KillMode=process.
    let h = home.display().to_string();
    format!(
        "[Unit]\nDescription=Goethe control plane daemon\nAfter=network-online.target\nWants=network-online.target\n\n[Service]\nType=simple\nEnvironment={}\nEnvironmentFile=-{h}/goethe.env\nExecStart={}\nKillMode=process\nRestart=on-failure\nRestartSec=3\n\n[Install]\nWantedBy={wanted_by}\n",
        systemd_quote(&format!("GOETHE_HOME={h}")),
        systemd_quote(&format!("{h}/bin/goethed")),
    )
}

fn write_unit(path: &Path, home: &Path, wanted_by: &str) -> Result<(), String> {
    fs::write(path, unit_text(home, wanted_by)).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn cron_line(home: &Path) -> String {
    let h = home.display().to_string();
    let line = format!(
        "@reboot GOETHE_HOME={} {} >>{} 2>&1",
        shell_quote(&h),
        shell_quote(&format!("{h}/bin/goethed")),
        shell_quote(&format!("{h}/logs/goethed.log"))
    );
    line.replace('%', "\\%")
}

fn install_crontab_reboot(home: &Path) -> Result<(), String> {
    let mut existing = read_crontab();
    if crontab_entry_for(&existing, home) {
        return Ok(());
    }
    if !existing.is_empty() && !existing.ends_with('\n') {
        existing.push('\n');
    }
    existing.push_str(&format!("{CRON_MARKER}\n{}\n", cron_line(home)));
    write_crontab(&existing)
}

fn read_crontab() -> String {
    // The user's crontab via `crontab -l`, or GOETHE_CRONTAB_FILE when set (tests).
    if let Ok(path) = std::env::var("GOETHE_CRONTAB_FILE") {
        return fs::read_to_string(path).unwrap_or_default();
    }
    Command::new("crontab")
        .arg("-l")
        .stderr(Stdio::null())
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

fn write_crontab(text: &str) -> Result<(), String> {
    // Replace the user's crontab through `crontab -`, or write GOETHE_CRONTAB_FILE when set.
    if let Ok(path) = std::env::var("GOETHE_CRONTAB_FILE") {
        return fs::write(path, text).map_err(|e| e.to_string());
    }
    let mut child = Command::new("crontab")
        .arg("-")
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| format!("no boot mechanism available (no systemd, crontab not usable: {e})"))?;
    child
        .stdin
        .take()
        .ok_or_else(|| "crontab stdin unavailable".to_string())?
        .write_all(text.as_bytes())
        .map_err(|e| e.to_string())?;
    if child.wait().map_err(|e| e.to_string())?.success() {
        Ok(())
    } else {
        Err("`crontab -` failed".to_string())
    }
}

fn crontab_entry_for(text: &str, home: &Path) -> bool {
    let lines: Vec<&str> = text.lines().collect();
    lines
        .windows(2)
        .any(|w| w[0].trim() == CRON_MARKER && cron_line_is_for(w[1], home))
}

fn cron_line_is_for(line: &str, home: &Path) -> bool {
    // The entry's GOETHE_HOME= token is exactly this home, quoted as written now or bare as older builds wrote it.
    let h = home.display().to_string();
    let quoted = format!("GOETHE_HOME={} ", shell_quote(&h).replace('%', "\\%"));
    line.contains(&quoted) || line.contains(&format!("GOETHE_HOME={h} "))
}

pub fn strip_cron_entry(text: &str, home: &Path) -> Option<String> {
    // Drop our marker and the entry after it, only for this GOETHE_HOME.
    let lines: Vec<&str> = text.lines().collect();
    let mut kept = Vec::new();
    let mut i = 0;
    let mut removed = false;
    while i < lines.len() {
        if lines[i].trim() == CRON_MARKER && lines.get(i + 1).is_some_and(|n| cron_line_is_for(n, home)) {
            i += 2;
            removed = true;
            continue;
        }
        kept.push(lines[i]);
        i += 1;
    }
    if !removed {
        return None;
    }
    let mut body = kept.join("\n");
    if !body.is_empty() {
        body.push('\n');
    }
    Some(body)
}

fn remove_crontab_reboot(home: &Path) -> Result<bool, String> {
    match strip_cron_entry(&read_crontab(), home) {
        Some(body) => write_crontab(&body).map(|_| true),
        None => Ok(false),
    }
}

fn remove_system_unit(home: &Path) -> bool {
    let p = system_unit_path();
    if unsafe { libc::geteuid() } != 0 || !unit_is_ours(&p, home) {
        return false;
    }
    systemctl_quiet(&["disable", "--now", "goethed.service"]);
    let _ = fs::remove_file(&p);
    systemctl_quiet(&["daemon-reload"]);
    true
}

fn remove_user_unit(home: &Path) -> bool {
    let p = user_unit_path();
    if !unit_is_ours(&p, home) {
        return false;
    }
    systemctl_quiet(&["--user", "disable", "--now", "goethed.service"]);
    let _ = fs::remove_file(&p);
    systemctl_quiet(&["--user", "daemon-reload"]);
    true
}

fn unit_is_ours(path: &Path, home: &Path) -> bool {
    // True for this home's unit; another home's unit is left alone with a note.
    match unit_owner(path, home) {
        UnitOwner::Ours => true,
        UnitOwner::Absent => false,
        UnitOwner::Other(h) => {
            eprintln!("left {} alone: it starts goethed for {}", path.display(), describe_home(h.as_deref()));
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_and_cron_quote_paths() {
        let home = Path::new("/srv/my home/.goethe");
        let unit = unit_text(home, "default.target");
        assert!(unit.contains("Environment=\"GOETHE_HOME=/srv/my home/.goethe\""));
        assert!(unit.contains("ExecStart=\"/srv/my home/.goethe/bin/goethed\""));
        assert!(unit.contains("EnvironmentFile=-/srv/my home/.goethe/goethe.env"));
        assert!(unit.contains("\nKillMode=process\n"), "engines must survive a goethed restart under systemd");
        let cron = cron_line(Path::new("/srv/50%/.goethe"));
        assert_eq!(
            cron,
            "@reboot GOETHE_HOME='/srv/50\\%/.goethe' '/srv/50\\%/.goethe/bin/goethed' >>'/srv/50\\%/.goethe/logs/goethed.log' 2>&1"
        );
    }

    #[test]
    fn cron_removal_is_home_specific() {
        let a = Path::new("/a/.goethe");
        let b = Path::new("/b/.goethe");
        let text = format!("0 1 * * * backup\n{CRON_MARKER}\n{}\n{CRON_MARKER}\n{}\n", cron_line(a), cron_line(b));
        let out = strip_cron_entry(&text, a).unwrap();
        assert!(!out.contains("/a/.goethe"));
        assert!(out.contains("/b/.goethe"));
        assert!(out.starts_with("0 1 * * * backup\n"));
        assert!(crontab_entry_for(&out, b));
        assert!(strip_cron_entry(&out, a).is_none());
        let prefix = Path::new("/a/.goeth");
        assert!(strip_cron_entry(&text, prefix).is_none());
        let legacy = format!(
            "{CRON_MARKER}\n@reboot GOETHE_HOME=/a/.goethe /a/.goethe/bin/goethed >>/a/.goethe/logs/goethed.log 2>&1\n"
        );
        assert_eq!(strip_cron_entry(&legacy, a).as_deref(), Some(""));
        assert!(strip_cron_entry(&legacy, Path::new("/a/.goethe2")).is_none());
    }

    #[test]
    fn unit_ownership_follows_its_goethe_home() {
        let mine = Path::new("/srv/my home/.goethe");
        assert_eq!(unit_text_home(&unit_text(mine, "default.target")).as_deref(), Some(mine));
        assert_eq!(
            unit_text_home("[Service]\nEnvironment=FOO=1 GOETHE_HOME=/b/.goethe\n").as_deref(),
            Some(Path::new("/b/.goethe"))
        );
        assert_eq!(unit_text_home("[Service]\nExecStart=/x/bin/goethed\n"), None);
        let dir = std::env::temp_dir().join(format!("goethe-unit-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let unit = dir.join("goethed.service");
        assert!(matches!(unit_owner(&unit, mine), UnitOwner::Absent));
        fs::write(&unit, unit_text(Path::new("/other/.goethe"), "default.target")).unwrap();
        assert!(matches!(unit_owner(&unit, mine), UnitOwner::Other(Some(_))));
        assert!(!unit_is_ours(&unit, mine));
        let err = refuse_other_home(&unit, mine).unwrap_err();
        assert!(err.contains("GOETHE_HOME='/other/.goethe' goethe service disable"), "{err}");
        fs::write(&unit, unit_text(mine, "default.target")).unwrap();
        assert!(matches!(unit_owner(&unit, mine), UnitOwner::Ours));
        assert!(refuse_other_home(&unit, mine).is_ok());
        fs::remove_dir_all(&dir).unwrap();
    }
}
