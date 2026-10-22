// Boot-time goethed enablement (systemd, user systemd, or crontab).

use crate::paths::goethe_home;
use std::fs;
use std::path::{Path, PathBuf};
use std::io::Write;
use std::process::{Command, Stdio};

const CRON_MARKER: &str = "# goethe-goethed-reboot";

pub fn run_enable() -> Result<(), String> {
    let home = goethe_home();
    let goethed = home.join("bin/goethed");
    if !goethed.is_file() {
        return Err(format!("goethed not found at {}", goethed.display()));
    }
    if try_system_unit(&home)? {
        println!("enabled goethed boot via systemd (system unit)");
        return Ok(());
    }
    if try_user_unit(&home)? {
        println!("enabled goethed boot via systemd (--user unit)");
        return Ok(());
    }
    install_crontab_reboot(&home)?;
    println!("enabled goethed boot via crontab @reboot");
    Ok(())
}

pub fn run_disable() -> Result<(), String> {
    let mut actions = 0usize;
    if remove_system_unit() {
        println!("disabled goethed systemd (system unit)");
        actions += 1;
    }
    if remove_user_unit() {
        println!("disabled goethed systemd (--user unit)");
        actions += 1;
    }
    if remove_crontab_reboot() {
        println!("removed goethed crontab @reboot entry");
        actions += 1;
    }
    if actions == 0 {
        println!("goethed boot was not enabled");
    }
    Ok(())
}

pub fn run_status() -> Result<(), String> {
    let method = active_method();
    let running = goethed_running();
    println!("boot method: {method}");
    println!(
        "goethed: {}",
        if running { "running" } else { "stopped" }
    );
    Ok(())
}

fn active_method() -> String {
    if system_unit_path().is_file() {
        "systemd (system)".into()
    } else if user_unit_path().is_file() {
        "systemd (--user)".into()
    } else if crontab_has_marker() {
        "crontab @reboot".into()
    } else {
        "none".into()
    }
}

fn goethed_running() -> bool {
    let home = goethe_home();
    let Ok(out) = Command::new("pgrep").args(["-f"]).arg(home.join("bin/goethed")).output() else {
        return false;
    };
    out.status.success() && !out.stdout.is_empty()
}

fn systemd_unit_dir() -> PathBuf {
    std::env::var("GOETHE_SYSTEMD_UNIT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/etc/systemd/system"))
}

fn user_unit_dir() -> PathBuf {
    std::env::var("GOETHE_SYSTEMD_USER_UNIT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| PathBuf::from(h).join(".config/systemd/user"))
                .unwrap_or_else(|_| PathBuf::from("/tmp"))
        })
}

fn system_unit_path() -> PathBuf {
    systemd_unit_dir().join("goethed.service")
}

fn user_unit_path() -> PathBuf {
    user_unit_dir().join("goethed.service")
}

fn try_system_unit(home: &Path) -> Result<bool, String> {
    if unsafe { libc::geteuid() } != 0 {
        return Ok(false);
    }
    let pid1 = Command::new("ps").args(["-p", "1", "-o", "comm="]).output();
    let is_systemd = pid1
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .is_some_and(|s| s.trim() == "systemd");
    if !is_systemd {
        return Ok(false);
    }
    write_unit(system_unit_path(), home, "multi-user.target")?;
    systemctl_quiet(&["daemon-reload"]);
    systemctl_quiet(&["enable", "--now", "goethed.service"]);
    Ok(true)
}

fn try_user_unit(home: &Path) -> Result<bool, String> {
    let out = Command::new("systemctl").args(["--user", "status"]).output();
    if out.is_err() || !out.as_ref().unwrap().status.success() {
        return Ok(false);
    }
    fs::create_dir_all(user_unit_dir()).map_err(|e| e.to_string())?;
    write_unit(user_unit_path(), home, "default.target")?;
    systemctl_quiet(&["--user", "daemon-reload"]);
    systemctl_quiet(&["--user", "enable", "--now", "goethed.service"]);
    eprintln!("hint: run `loginctl enable-linger $USER` so goethed starts at boot without a login session");
    Ok(true)
}

fn systemctl_quiet(args: &[&str]) {
    let _ = Command::new("systemctl")
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

fn write_unit(path: PathBuf, home: &Path, wanted_by: &str) -> Result<(), String> {
    let goethed = home.join("bin/goethed");
    let text = format!(
        "[Unit]\nDescription=Goethe control plane daemon\nAfter=network.target\n\n[Service]\nType=simple\nEnvironment=GOETHE_HOME={}\nExecStart={}\nRestart=on-failure\nRestartSec=3\n\n[Install]\nWantedBy={wanted_by}\n",
        home.display(),
        goethed.display()
    );
    fs::write(&path, text).map_err(|e| e.to_string())
}

fn install_crontab_reboot(home: &Path) -> Result<(), String> {
    let goethed = home.join("bin/goethed");
    let line = format!(
        "{CRON_MARKER}\n@reboot GOETHE_HOME={} {} >>{}/logs/goethed.log 2>&1",
        home.display(),
        goethed.display(),
        home.display()
    );
    let mut existing = read_crontab();
    if !existing.contains(CRON_MARKER) {
        if !existing.is_empty() && !existing.ends_with('\n') {
            existing.push('\n');
        }
        existing.push_str(&line);
        existing.push('\n');
        write_crontab(&existing)?;
    }
    Ok(())
}

fn read_crontab() -> String {
    // The user's real crontab via `crontab -l`, or GOETHE_CRONTAB_FILE when set (tests).
    if let Ok(path) = std::env::var("GOETHE_CRONTAB_FILE") {
        return fs::read_to_string(path).unwrap_or_default();
    }
    Command::new("crontab")
        .arg("-l")
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
        .map_err(|e| format!("no boot mechanism available (systemd unit failed, crontab not usable: {e})"))?;
    child
        .stdin
        .take()
        .ok_or_else(|| "crontab stdin unavailable".to_string())?
        .write_all(text.as_bytes())
        .map_err(|e| e.to_string())?;
    let status = child.wait().map_err(|e| e.to_string())?;
    if status.success() {
        Ok(())
    } else {
        Err("`crontab -` failed".to_string())
    }
}

fn crontab_has_marker() -> bool {
    read_crontab().contains(CRON_MARKER)
}

fn remove_crontab_reboot() -> bool {
    // Drop only our marker line and the single entry line that follows it.
    let text = read_crontab();
    if !text.contains(CRON_MARKER) {
        return false;
    }
    let mut kept: Vec<&str> = Vec::new();
    let mut skip_next = false;
    for line in text.lines() {
        if skip_next {
            skip_next = false;
            continue;
        }
        if line.trim() == CRON_MARKER {
            skip_next = true;
            continue;
        }
        kept.push(line);
    }
    let mut body = kept.join("\n");
    if !body.is_empty() {
        body.push('\n');
    }
    let _ = write_crontab(&body);
    true
}

fn remove_system_unit() -> bool {
    if unsafe { libc::geteuid() } != 0 {
        return false;
    }
    let p = system_unit_path();
    let had = p.is_file();
    systemctl_quiet(&["disable", "--now", "goethed.service"]);
    if had {
        let _ = fs::remove_file(p);
    }
    systemctl_quiet(&["daemon-reload"]);
    had
}

fn remove_user_unit() -> bool {
    let p = user_unit_path();
    let had = p.is_file();
    systemctl_quiet(&["--user", "disable", "--now", "goethed.service"]);
    if had {
        let _ = fs::remove_file(p);
    }
    had
}
