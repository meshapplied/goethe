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
    let deadline = Instant::now() + timeout;
    let mut printed_notices = 0usize;
    while Instant::now() < deadline {
        let ps = rpc::call("ps", json!(null)).map_err(rpc::map_daemon_err)?;
        if ps.code == 0 {
            if let Ok(rows) = serde_json::from_str::<serde_json::Value>(&ps.stdout) {
                if let Some(arr) = rows.as_array() {
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
                            return Err("engine health check failed".to_string());
                        }
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err("load timed out waiting for RUNNING".to_string())
}
