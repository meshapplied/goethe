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
    let home = goethe_home();
    let mut actions = 0usize;
    if remove_system_unit() {
        println!("disabled goethed systemd (system unit)");
        actions += 1;
    }
    if remove_user_unit() {
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
    if system_unit_path().is_file() {
        "systemd (system)".into()
    } else if user_unit_path().is_file() {
        "systemd (--user)".into()
    } else if crontab_entry_for(&read_crontab(), home) {
        "crontab @reboot".into()
    } else {
        "none".into()
    }
}

pub fn goethed_running() -> bool {
    std::os::unix::net::UnixStream::connect(goethe_sock()).is_ok()
}

pub fn spawn_goethed(home: &Path) -> Result<(), String> {
    // Start goethed detached in its own session, logging to GOETHE_HOME/logs/goethed.log.
    let goethed = home.join("bin/goethed");
    fs::create_dir_all(home.join("logs")).map_err(|e| e.to_string())?;
    let log = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(home.join("logs/goethed.log"))
        .map_err(|e| e.to_string())?;
    let mut cmd = Command::new(&goethed);
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
    if unsafe { libc::geteuid() } == 0 && system_unit_path().is_file() && system_unit_dir() == default_system_unit_dir() {
        return Some(vec!["restart".into(), "goethed.service".into()]);
    }
    if user_unit_path().is_file() && user_unit_dir() == default_user_unit_dir() {
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
    write_unit(&system_unit_path(), home, "multi-user.target")?;
    if system_unit_dir() == default_system_unit_dir() {
        systemctl_quiet(&["daemon-reload"]);
        systemctl_quiet(&enable_args(&[]));
    }
    Ok(true)
}

fn try_user_unit(home: &Path) -> Result<bool, String> {
    // A user unit needs a working user systemd manager; systemctl only sees units in its own dir.
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
    fs::create_dir_all(user_unit_dir()).map_err(|e| e.to_string())?;
    write_unit(&user_unit_path(), home, "default.target")?;
    if user_unit_dir() == default_user_unit_dir() {
        systemctl_quiet(&["--user", "daemon-reload"]);
        systemctl_quiet(&enable_args(&["--user"]));
        eprintln!("hint: run `loginctl enable-linger $USER` so goethed starts at boot without a login session");
    }
    Ok(true)
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
    // Unit with quoted paths and an optional GOETHE_HOME/goethed.env for overrides such as ports.
    let h = home.display().to_string();
    format!(
        "[Unit]\nDescription=Goethe control plane daemon\nAfter=network-online.target\nWants=network-online.target\n\n[Service]\nType=simple\nEnvironment={}\nEnvironmentFile=-{h}/goethed.env\nExecStart={}\nRestart=on-failure\nRestartSec=3\n\n[Install]\nWantedBy={wanted_by}\n",
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
    let h = home.display().to_string();
    let lines: Vec<&str> = text.lines().collect();
    lines
        .windows(2)
        .any(|w| w[0].trim() == CRON_MARKER && w[1].contains(&h))
}

pub fn strip_cron_entry(text: &str, home: &Path) -> Option<String> {
    // Drop our marker and the entry after it, only for this GOETHE_HOME.
    let h = home.display().to_string();
    let lines: Vec<&str> = text.lines().collect();
    let mut kept = Vec::new();
    let mut i = 0;
    let mut removed = false;
    while i < lines.len() {
        if lines[i].trim() == CRON_MARKER && lines.get(i + 1).is_some_and(|n| n.contains(&h)) {
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

fn remove_system_unit() -> bool {
    let p = system_unit_path();
    if unsafe { libc::geteuid() } != 0 || !p.is_file() {
        return false;
    }
    if system_unit_dir() == default_system_unit_dir() {
        systemctl_quiet(&["disable", "--now", "goethed.service"]);
    }
    let _ = fs::remove_file(&p);
    if system_unit_dir() == default_system_unit_dir() {
        systemctl_quiet(&["daemon-reload"]);
    }
    true
}

fn remove_user_unit() -> bool {
    let p = user_unit_path();
    if !p.is_file() {
        return false;
    }
    if user_unit_dir() == default_user_unit_dir() {
        systemctl_quiet(&["--user", "disable", "--now", "goethed.service"]);
    }
    let _ = fs::remove_file(&p);
    if user_unit_dir() == default_user_unit_dir() {
        systemctl_quiet(&["--user", "daemon-reload"]);
    }
    true
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
        assert!(unit.contains("EnvironmentFile=-/srv/my home/.goethe/goethed.env"));
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
    }
}
