// Exclusive goethed instance lock; holding it proves any previous daemon is gone and its socket stale.

use goethe_core::paths::{goethe_home, goethe_sock};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::{FileTypeExt, OpenOptionsExt};
use std::os::unix::net::UnixStream;
use std::path::Path;

pub struct InstanceLock {
    _file: File,
}

impl InstanceLock {
    pub fn acquire() -> Result<Self, String> {
        // flock first (no truncate before it); only the holder may clear a stale socket and write its pid.
        let home = goethe_home();
        std::fs::create_dir_all(&home).map_err(|e| e.to_string())?;
        let path = home.join("goethed.lock");
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(0o600)
            .open(&path)
            .map_err(|e| format!("open {}: {e}", path.display()))?;
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::EWOULDBLOCK) {
                return Err(format!("lock {}: {err}", path.display()));
            }
            return Err(match read_lock_pid(&path) {
                Some(pid) => format!("goethed already running (pid {pid})"),
                None => "goethed already running".to_string(),
            });
        }
        clear_stale_socket(&goethe_sock())?;
        file.set_len(0).map_err(|e| e.to_string())?;
        file.write_all(format!("{}\n", std::process::id()).as_bytes())
            .map_err(|e| e.to_string())?;
        Ok(Self { _file: file })
    }
}

fn read_lock_pid(path: &Path) -> Option<u32> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.lines().next()?.trim().parse().ok())
}

fn clear_stale_socket(sock: &Path) -> Result<(), String> {
    // Remove a leftover socket; refuse if something still answers on it (another home sharing GOETHE_SOCK).
    let Ok(meta) = std::fs::symlink_metadata(sock) else {
        return Ok(());
    };
    if !meta.file_type().is_socket() {
        return Err(format!("{} exists and is not a socket", sock.display()));
    }
    if UnixStream::connect(sock).is_ok() {
        return Err(format!("goethed already running ({} answers)", sock.display()));
    }
    std::fs::remove_file(sock).map_err(|e| format!("remove stale {}: {e}", sock.display()))
}
