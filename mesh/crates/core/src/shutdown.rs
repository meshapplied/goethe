// meshd shutdown coordination and mesh stop CLI.

use crate::paths::mesh_sock;
use crate::rpc;
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

pub fn request_shutdown() {
    SHUTDOWN_REQUESTED.store(true, Ordering::SeqCst);
}

pub fn shutdown_requested() -> bool {
    SHUTDOWN_REQUESTED.load(Ordering::SeqCst)
}

pub fn clear_shutdown_flag() {
    SHUTDOWN_REQUESTED.store(false, Ordering::SeqCst);
}

pub fn run_stop() -> Result<(), String> {
    let sock = mesh_sock();
    if !sock.exists() {
        println!("meshd not running");
        return Ok(());
    }
    match rpc::call("shutdown", json!(null)) {
        Err(e) if e.contains("meshd is not running") => {
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
    for _ in 0..150 {
        if !sock.exists() {
            println!("meshd stopped");
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Err("meshd did not stop within 15s".to_string())
}

pub fn shutdown_daemon(home: &std::path::Path, sock: &std::path::Path) {
    let _ = crate::supervisor::graceful_shutdown(home);
    crate::nginx::stop(home);
    let _ = std::fs::remove_file(sock);
}
