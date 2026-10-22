// JSON-lines RPC client for meshd over MESH_SOCK.

use crate::paths::mesh_sock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;

#[derive(Debug, Serialize)]
struct Request {
    method: String,
    params: Value,
}

#[derive(Debug, Deserialize)]
struct Response {
    ok: bool,
    #[serde(default)]
    result: Value,
    #[serde(default)]
    error: String,
    #[serde(default)]
    code: i32,
    #[serde(default)]
    message: String,
}

pub struct RpcReply {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

pub fn daemon_unavailable_message() -> &'static str {
    "meshd is not running: start it with `mesh service enable` or run meshd"
}

pub fn map_daemon_err(e: String) -> String {
    if e.contains("meshd is not running") {
        daemon_unavailable_message().to_string()
    } else {
        e
    }
}

pub fn parse_load_instance_id(stdout: &str) -> Option<String> {
    let trimmed = stdout.trim();
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
        return v
            .get("instance_id")
            .and_then(|x| x.as_str())
            .map(String::from);
    }
    if let Some(i) = trimmed.rfind('{') {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&trimmed[i..]) {
            return v
                .get("instance_id")
                .and_then(|x| x.as_str())
                .map(String::from);
        }
    }
    None
}

pub fn call(method: &str, params: Value) -> Result<RpcReply, String> {
    call_inner(method, params, 0)
}

fn call_inner(method: &str, params: Value, attempt: u32) -> Result<RpcReply, String> {
    use std::thread;
    use std::time::Duration;
    let sock_path = mesh_sock();
    let mut stream = None;
    for i in 0..40 {
        match UnixStream::connect(&sock_path) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(_e) => {
                if i == 39 {
                    return Err(daemon_unavailable_message().to_string());
                }
                thread::sleep(Duration::from_millis(50));
            }
        }
    }
    let mut stream = stream.expect("connected");
    stream
        .set_read_timeout(Some(Duration::from_secs(600)))
        .ok();
    stream
        .set_write_timeout(Some(Duration::from_secs(30)))
        .ok();
    let req = Request {
        method: method.to_string(),
        params: params.clone(),
    };
    let line = serde_json::to_string(&req).map_err(|e| e.to_string())?;
    if stream.write_all(line.as_bytes()).is_err() || stream.write_all(b"\n").is_err() {
        if attempt < 3 {
            return call_inner(method, params, attempt + 1);
        }
        return Err("meshd connection lost".to_string());
    }
    let mut reader = BufReader::new(stream);
    let mut resp_line = String::new();
    if reader.read_line(&mut resp_line).is_err() {
        if attempt < 3 {
            return call_inner(method, params, attempt + 1);
        }
        return Err("meshd connection lost".to_string());
    }
    let resp: Response = serde_json::from_str(resp_line.trim()).map_err(|e| e.to_string())?;
    if resp.ok {
        let mut stdout = String::new();
        if !resp.message.is_empty() {
            stdout.push_str(resp.message.trim());
            stdout.push('\n');
        }
        if resp.result != Value::Null {
            stdout.push_str(&serde_json::to_string(&resp.result).unwrap_or_default());
        }
        Ok(RpcReply {
            code: 0,
            stdout,
            stderr: String::new(),
        })
    } else {
        Ok(RpcReply {
            code: if resp.code != 0 { resp.code } else { 1 },
            stdout: resp.message.clone(),
            stderr: resp.error,
        })
    }
}
