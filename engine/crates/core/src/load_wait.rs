// Wait for a load RPC instance to reach RUNNING (shared by goethe load and install --load).

use crate::rpc;
use serde_json::json;
use std::io::{self, Write};
use std::time::{Duration, Instant};

pub const DEFAULT_LOAD_WAIT: Duration = Duration::from_secs(600);

pub fn wait_after_load_stdout(stdout: &str, no_wait: bool) -> Result<(), String> {
    if no_wait {
        return Ok(());
    }
    let instance_id = rpc::parse_load_instance_id(stdout)
        .ok_or_else(|| "load did not return instance_id".to_string())?;
    wait_for_instance(&instance_id, DEFAULT_LOAD_WAIT)
}

pub fn wait_for_instance(instance_id: &str, timeout: Duration) -> Result<(), String> {
    // Poll ps until the instance is RUNNING, printing its notices; a final status or a vanished row ends the wait.
    let deadline = Instant::now() + timeout;
    let mut printed_notices = 0usize;
    while Instant::now() < deadline {
        let ps = rpc::call("ps", json!(null)).map_err(rpc::map_daemon_err)?;
        if ps.code == 0 {
            if let Ok(rows) = serde_json::from_str::<serde_json::Value>(&ps.stdout) {
                if let Some(arr) = rows.as_array() {
                    if !arr.iter().any(|r| r.get("instance_id").and_then(|v| v.as_str()) == Some(instance_id)) {
                        return Err(format!("instance {instance_id} no longer exists (unloaded or stopped)"));
                    }
                    for row in arr {
                        if row.get("instance_id").and_then(|v| v.as_str()) != Some(instance_id) {
                            continue;
                        }
                        if let Some(notices) = row.get("notices").and_then(|v| v.as_array()) {
                            if notices.len() > printed_notices {
                                for n in notices.iter().skip(printed_notices) {
                                    if let Some(s) = n.as_str() {
                                        let _ = writeln!(io::stderr(), "{s}");
                                        io::stderr().flush().ok();
                                    }
                                }
                                printed_notices = notices.len();
                            }
                        }
                        let st = row.get("status").and_then(|v| v.as_str()).unwrap_or("");
                        if st == "RUNNING" {
                            return Ok(());
                        }
                        if st == "CANCELLED" {
                            return Err("load cancelled".to_string());
                        }
                        if st == "FAILED" {
                            return Err(failure_report(&crate::paths::goethe_home(), instance_id));
                        }
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err("load timed out waiting for RUNNING".to_string())
}

const ERROR_MARKERS: &[&str] = &[
    "Error", "error:", "Exception", "Traceback", "out of memory", "CUDA", "Killed", "Assertion", "not supported", "No such file",
];

fn tail_lines(path: &std::path::Path, max_bytes: u64) -> Vec<String> {
    // The last max_bytes of a file as lines; empty when it cannot be read.
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let _ = f.seek(SeekFrom::Start(len.saturating_sub(max_bytes)));
    let mut buf = Vec::new();
    let _ = f.read_to_end(&mut buf);
    String::from_utf8_lossy(&buf).lines().map(str::to_string).collect()
}

pub fn failure_report(home: &std::path::Path, instance_id: &str) -> String {
    // Why a start failed: goethed's lines about this instance, then the error lines (or the tail) of the engine log.
    let mut out = vec![format!("the engine failed to start (instance {instance_id})")];
    let daemon: Vec<String> = tail_lines(&home.join("logs/goethed.log"), 256 * 1024)
        .into_iter()
        .filter(|l| l.contains(instance_id))
        .collect();
    for l in daemon.iter().rev().take(4).rev() {
        out.push(format!("  goethed: {}", l.trim()));
    }
    let engine = tail_lines(&home.join("logs").join(format!("{instance_id}.log")), 512 * 1024);
    let errors: Vec<&String> = engine.iter().filter(|l| ERROR_MARKERS.iter().any(|m| l.contains(m))).collect();
    let picked: Vec<&String> = if errors.is_empty() { engine.iter().rev().take(15).rev().collect() } else { errors.into_iter().rev().take(12).rev().collect() };
    if !picked.is_empty() {
        out.push("  engine log:".to_string());
        out.extend(picked.into_iter().map(|l| format!("    {}", l.trim_end())));
    }
    out.push(format!("full engine log: goethe logs {instance_id} -n 200"));
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failure_report_shows_daemon_reason_and_engine_errors() {
        let td = tempfile::tempdir().unwrap();
        let logs = td.path().join("logs");
        std::fs::create_dir_all(&logs).unwrap();
        std::fs::write(logs.join("goethed.log"), "other line\nwatchdog instance=inst-x reason=process exited\n").unwrap();
        std::fs::write(logs.join("inst-x.log"), "loading\nRuntimeError: CUDA error: no kernel image is available\ndone\n").unwrap();
        let r = failure_report(td.path(), "inst-x");
        assert!(r.contains("goethed: watchdog instance=inst-x reason=process exited"), "{r}");
        assert!(r.contains("RuntimeError: CUDA error: no kernel image"), "{r}");
        assert!(!r.contains("other line") && !r.contains("    loading"), "{r}");
        assert!(r.ends_with("goethe logs inst-x -n 200"));
        let none = failure_report(td.path(), "inst-missing");
        assert!(none.starts_with("the engine failed to start (instance inst-missing)"));
    }
}
