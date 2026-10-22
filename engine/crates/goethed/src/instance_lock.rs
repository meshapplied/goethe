// Exclusive goethed instance lock and live socket check.

use goethe_core::paths::{goethe_home, goethe_sock};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};

pub struct InstanceLock {
    _file: File,
}

impl InstanceLock {
    pub fn acquire() -> Result<Self, String> {
        let home = goethe_home();
        let path = home.join("goethed.lock");
        let sock = goethe_sock();
        if sock.exists() {
            if let Some(pid) = read_lock_pid(&path) {
                if pid_alive(pid) {
                    return Err(format!("goethed already running (pid {pid})"));
                }
            }
            reclaim_stale(&home, &path);
        } else if let Some(pid) = read_lock_pid(&path) {
            if pid_alive(pid) {
                return Err(format!("goethed already running (pid {pid})"));
            }
            reclaim_stale(&home, &path);
        }
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&path)
            .map_err(|e| e.to_string())?;
        let fd = file.as_raw_fd();
        let rc = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            if let Some(pid) = read_lock_pid(&path) {
                if pid_alive(pid) {
                    return Err(format!("goethed already running (pid {pid})"));
                }
                reclaim_stale(&home, &path);
                let mut file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .mode(0o600)
                    .open(&path)
                    .map_err(|e| e.to_string())?;
                let fd = file.as_raw_fd();
                if unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) } != 0 {
                    return Err("goethed already running".to_string());
                }
                write_pid(&mut file)?;
                return Ok(Self { _file: file });
            }
            return Err("goethed already running".to_string());
        }
        write_pid(&mut file)?;
        Ok(Self { _file: file })
    }
}

fn write_pid(file: &mut File) -> Result<(), String> {
    let pid = std::process::id();
    file.set_len(0).map_err(|e| e.to_string())?;
    file.write_all(format!("{pid}\n").as_bytes())
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn read_lock_pid(path: &PathBuf) -> Option<u32> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.lines().next()?.parse().ok())
}

fn pid_alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

fn reclaim_stale(home: &Path, lock: &PathBuf) {
    let sock = goethe_sock();
    if sock.exists() {
        if let Some(pid) = read_lock_pid(lock) {
            if !pid_alive(pid) {
                let _ = std::fs::remove_file(&sock);
            }
        } else {
            let _ = std::fs::remove_file(&sock);
        }
    }
    if lock.is_file() {
        if let Some(pid) = read_lock_pid(lock) {
            if !pid_alive(pid) {
                let _ = std::fs::remove_file(lock);
            }
        }
    }
    let _ = home;
}
