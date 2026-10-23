// goethed shutdown coordination and goethe stop, including stops that leave engines running for an upgrade.

use crate::paths::{goethe_home, goethe_sock};
use crate::rpc;
use serde_json::json;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);
static KEEP_ENGINES: AtomicBool = AtomicBool::new(false);

pub fn request_shutdown() {
    SHUTDOWN_REQUESTED.store(true, Ordering::SeqCst);
}

pub fn request_shutdown_keeping_engines() {
    KEEP_ENGINES.store(true, Ordering::SeqCst);
    SHUTDOWN_REQUESTED.store(true, Ordering::SeqCst);
}

pub fn keep_engines_requested() -> bool {
    KEEP_ENGINES.load(Ordering::SeqCst)
}

pub fn shutdown_requested() -> bool {
    SHUTDOWN_REQUESTED.load(Ordering::SeqCst)
}

pub fn clear_shutdown_flag() {
    SHUTDOWN_REQUESTED.store(false, Ordering::SeqCst);
    KEEP_ENGINES.store(false, Ordering::SeqCst);
}

pub fn run_stop() -> Result<(), String> {
    stop_daemon(false)
}

pub fn stop_daemon(keep_engines: bool) -> Result<(), String> {
    // Ask goethed to exit and wait until it is gone; keep_engines leaves engines for the next goethed to adopt.
    let sock = goethe_sock();
    let home = goethe_home();
    if !sock.exists() {
        if !keep_engines && crate::nginx::stop_left_gateway(&home) {
            println!("goethed not running; stopped the gateway a previous goethed left running");
        } else {
            println!("goethed not running");
        }
        return Ok(());
    }
    match rpc::call("shutdown", json!({ "keep_engines": keep_engines })) {
        Err(e) if e.contains("goethed is not running") => {
            return Err(rpc::daemon_unavailable_message().to_string());
        }
        Err(e) => return Err(rpc::map_daemon_err(e)),
        Ok(reply) if reply.code != 0 => {
            return Err(if reply.stderr.is_empty() {
                reply.stdout
            } else {
                reply.stderr
            });
        }
        Ok(_) => {}
    }
    let polls = if keep_engines { 450 } else { 150 };
    for _ in 0..polls {
        if !sock.exists() && daemon_released(&home) {
            let kept = if keep_engines { "; engines left running for the next goethed to adopt" } else { "" };
            println!("goethed stopped{kept}");
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err(format!("goethed did not stop within {}s", polls / 10))
}

fn daemon_released(home: &Path) -> bool {
    // The socket goes before goethed exits; a free instance lock means a new goethed can take its lock and ports.
    match std::fs::File::open(home.join("goethed.lock")) {
        Ok(f) => unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_SH | libc::LOCK_NB) == 0 },
        Err(_) => true,
    }
}

pub fn shutdown_daemon(home: &Path, sock: &Path, keep_engines: bool) {
    // A plain stop tears engines down and keeps desired slots; keep_engines touches no engine, row or run entry.
    if keep_engines {
        crate::supervisor::append_goethed_log(
            home,
            "goethed stopping with keep_engines: engines and the gateway left for adoption\n",
        );
    } else {
        let _ = crate::supervisor::graceful_shutdown(home);
        crate::nginx::stop(home);
    }
    let _ = std::fs::remove_file(sock);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proc_ident;

    #[test]
    fn daemon_counts_as_released_only_when_nobody_holds_its_lock() {
        let td = tempfile::tempdir().unwrap();
        assert!(daemon_released(td.path()), "no lock file: no daemon");
        let held = std::fs::File::create(td.path().join("goethed.lock")).unwrap();
        assert_eq!(unsafe { libc::flock(held.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) }, 0);
        assert!(!daemon_released(td.path()), "a live daemon holds the lock");
        drop(held);
        assert!(daemon_released(td.path()));
    }

    #[test]
    fn keep_engines_leaves_engine_row_and_run_entry_alone() {
        let td = tempfile::tempdir().unwrap();
        crate::paths::ensure_layout(td.path()).unwrap();
        let con = crate::db::open(td.path()).unwrap();
        let mut child = std::process::Command::new("sleep").arg("30").spawn().unwrap();
        let id = proc_ident::capture("inst-keep", child.id() as i32).unwrap();
        con.execute(
            "INSERT INTO loaded_instances (instance_id, model_id, pid, pgid, internal_port, gpu_indices,
             allocated_vram_gb_per_gpu, mem_fraction_static, ctx_len, max_conc, status, boot_id, start_ticks, engine_key)
             VALUES ('inst-keep', 'm', ?1, ?1, 30001, '2', 1.0, 0.1, 4096, 4, 'RUNNING', ?2, ?3, 'k')",
            rusqlite::params![id.pid, id.boot_id, id.start_ticks as i64],
        )
        .unwrap();
        crate::run_registry::register(td.path(), &id).unwrap();
        let sock = td.path().join("goethe.sock");
        std::fs::write(&sock, b"").unwrap();
        shutdown_daemon(td.path(), &sock, true);
        assert!(child.try_wait().unwrap().is_none(), "the engine keeps running");
        let status: String = con
            .query_row("SELECT status FROM loaded_instances WHERE instance_id = 'inst-keep'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(status, "RUNNING");
        assert_eq!(crate::run_registry::list_entries(td.path()), vec![id]);
        assert!(!sock.exists());
        child.kill().unwrap();
        child.wait().unwrap();
    }
}
