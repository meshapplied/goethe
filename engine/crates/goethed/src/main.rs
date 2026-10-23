// goethed JSON-lines RPC server over GOETHE_SOCK.

mod instance_lock;
mod mux;

use goethe_core::nginx;
use goethe_core::paths::{ensure_layout, goethe_home, goethe_sock};
use goethe_core::security;
use goethe_core::ports;
use goethe_core::shutdown;
use goethe_core::supervisor::{self, LoadParams};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::time::Duration;
use tokio::net::UnixListener as TokioUnixListener;
use tokio::signal::unix::{signal, SignalKind};

#[derive(Debug, Deserialize)]
struct Request {
    method: String,
    params: Value,
}

fn write_resp(stream: &mut UnixStream, ok: bool, code: i32, message: &str, error: &str, result: Value) {
    let resp = json!({
        "ok": ok,
        "code": code,
        "message": message,
        "error": error,
        "result": result,
    });
    let line = serde_json::to_string(&resp).unwrap_or_else(|_| "{}".to_string());
    let _ = stream.write_all(line.as_bytes());
    let _ = stream.write_all(b"\n");
}

fn dispatch(req: Request) -> (bool, i32, String, String, Value) {
    match req.method.as_str() {
        "load" => {
            let params: LoadParams = match serde_json::from_value(req.params) {
                Ok(p) => p,
                Err(e) => return (false, 1, String::new(), e.to_string(), Value::Null),
            };
            match supervisor::handle_load(params) {
                Ok((result, notice)) => (true, 0, notice, String::new(), result),
                Err((3, msg, result)) => (false, 3, msg, String::new(), result),
                Err((code, msg, result)) => (false, code, String::new(), msg, result),
            }
        }
        "unload" => {
            let all = req.params.get("all").and_then(|v| v.as_bool()).unwrap_or(false);
            let model_id = req
                .params
                .get("model_id")
                .and_then(|v| v.as_str())
                .map(String::from);
            let force = req
                .params
                .get("force")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            match supervisor::handle_unload(model_id, all, force) {
                Ok(v) => (true, 0, String::new(), String::new(), v),
                Err((code, msg)) => (false, code, String::new(), msg, Value::Null),
            }
        }
        "ps" => match supervisor::handle_ps() {
            Ok(v) => (true, 0, String::new(), String::new(), v),
            Err((code, msg)) => (false, code, String::new(), msg, Value::Null),
        },
        "shutdown" => {
            shutdown::request_shutdown();
            (true, 0, String::new(), String::new(), Value::Null)
        }
        other => (
            false,
            2,
            String::new(),
            format!("unknown method {other}"),
            Value::Null,
        ),
    }
}

fn handle_client(stream: UnixStream) {
    let mut stream = stream;
    let reader = BufReader::new(stream.try_clone().expect("dup goethe.sock stream"));
    for line in reader.lines() {
        let Ok(line) = line else {
            break;
        };
        if line.trim().is_empty() {
            continue;
        }
        let req: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                write_resp(&mut stream, false, 1, "", &e.to_string(), Value::Null);
                continue;
            }
        };
        let (ok, code, message, error, result) = dispatch(req);
        write_resp(&mut stream, ok, code, &message, &error, result);
    }
}

fn bind_socket(sock: &std::path::Path) -> TokioUnixListener {
    // Bind goethe.sock (owner only); the instance lock already cleared any stale file.
    use std::os::unix::fs::PermissionsExt;
    if let Some(parent) = sock.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let std_listener = UnixListener::bind(sock).unwrap_or_else(|e| {
        eprintln!("bind {}: {e}", sock.display());
        std::process::exit(1);
    });
    let _ = std::fs::set_permissions(sock, std::fs::Permissions::from_mode(0o600));
    std_listener
        .set_nonblocking(true)
        .expect("nonblocking goethe.sock");
    TokioUnixListener::from_std(std_listener).expect("tokio unix listener")
}

#[tokio::main]
async fn main() {
    shutdown::clear_shutdown_flag();
    let _lock = instance_lock::InstanceLock::acquire().unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let home = goethe_home();
    let _ = ensure_layout(&home);
    ports::ensure_service_ports(&home).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let _ = supervisor::reconcile_on_start();
    security::sync_api_keys_conf(&home);
    let sock = goethe_sock();
    let listener = bind_socket(&sock);
    let mut mux = mux::spawn_mux().unwrap_or_else(|e| {
        eprintln!("{e}");
        let _ = std::fs::remove_file(&sock);
        std::process::exit(1);
    });
    let _ = nginx::ensure_running();
    if let Some(line) = nginx::format_gateway_line(&home) {
        eprintln!("{line}");
    }
    eprintln!("goethed listening on {}", sock.display());
    goethe_core::autostart::spawn_restore_background();
    let _ = goethe_core::log_rotation::run_rotation_pass(&home);
    let home_rot = home.clone();
    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    let mut shutdown_tick = tokio::time::interval(Duration::from_millis(100));
    shutdown_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut watchdog_tick = tokio::time::interval(Duration::from_secs(5));
    watchdog_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut log_rotate_tick = tokio::time::interval(Duration::from_secs(3600));
    log_rotate_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut exit_code = 0;
    loop {
        tokio::select! {
            _ = log_rotate_tick.tick() => {
                let h = home_rot.clone();
                tokio::task::spawn_blocking(move || {
                    let _ = goethe_core::log_rotation::run_rotation_pass(&h);
                });
            }
            _ = watchdog_tick.tick() => {
                let h = home_rot.clone();
                tokio::task::spawn_blocking(move || {
                    supervisor::watchdog_tick();
                    goethe_core::proc_ident::reap_deferred();
                    if let Err(e) = goethe_core::nginx::ensure_gateway_running(&h) {
                        eprintln!("gateway check: {e}");
                    }
                });
            }
            res = listener.accept() => {
                match res {
                    Ok((stream, _)) => {
                        let stream = stream.into_std().expect("into_std");
                        tokio::task::spawn_blocking(move || {
                            handle_client(stream);
                        });
                    }
                    Err(e) => eprintln!("accept error: {e}"),
                }
            }
            _ = &mut mux => {
                eprintln!("mux stopped; shutting goethed down");
                exit_code = 1;
                break;
            }
            _ = sigterm.recv() => break,
            _ = sigint.recv() => break,
            _ = shutdown_tick.tick() => {
                if shutdown::shutdown_requested() {
                    break;
                }
            }
        }
    }
    shutdown::request_shutdown();
    shutdown::shutdown_daemon(&home, &sock);
    std::process::exit(exit_code);
}
