// JSON-lines RPC client for goethed over GOETHE_SOCK.

use crate::paths::goethe_sock;
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
    pub result: Value,
}

pub fn daemon_unavailable_message() -> &'static str {
    "goethed is not running: start it with `goethe up` (or `goethe service enable` to start it at boot)"
}

pub fn map_daemon_err(e: String) -> String {
    if e.contains("goethed is not running") {
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
    // One request per connection; load and unload are never re-sent after an I/O error or timeout.
    let resend = !matches!(method, "load" | "unload");
    let mut attempt = 0;
    loop {
        match exchange(method, &params) {
            Ok(reply) => return Ok(reply),
            Err((true, _)) if resend && attempt < 3 => attempt += 1,
            Err((true, e)) if !resend => {
                return Err(format!(
                    "goethed connection lost during {method} ({e}); it may still complete, check `goethe ps` before retrying"
                ))
            }
            Err((_, e)) => return Err(e),
        }
    }
}

fn exchange(method: &str, params: &Value) -> Result<RpcReply, (bool, String)> {
    // Connect (retrying briefly while goethed starts), send one line, read one; true flags I/O errors after connect.
    use std::thread;
    use std::time::Duration;
    let sock_path = goethe_sock();
    let mut stream = None;
    for _ in 0..40 {
        match UnixStream::connect(&sock_path) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(_) => thread::sleep(Duration::from_millis(50)),
        }
    }
    let Some(mut stream) = stream else {
        return Err((false, daemon_unavailable_message().to_string()));
    };
    stream.set_read_timeout(Some(Duration::from_secs(600))).ok();
    stream.set_write_timeout(Some(Duration::from_secs(30))).ok();
    let req = Request {
        method: method.to_string(),
        params: params.clone(),
    };
    let mut line = serde_json::to_string(&req).map_err(|e| (false, e.to_string()))?;
    line.push('\n');
    stream
        .write_all(line.as_bytes())
        .map_err(|e| (true, e.to_string()))?;
    let mut reader = BufReader::new(stream);
    let mut resp_line = String::new();
    match reader.read_line(&mut resp_line) {
        Ok(0) => return Err((true, "goethed closed the connection".to_string())),
        Ok(_) => {}
        Err(e) => return Err((true, e.to_string())),
    }
    let resp: Response = serde_json::from_str(resp_line.trim()).map_err(|e| (false, e.to_string()))?;
    if resp.ok {
        let mut stdout = String::new();
        if !resp.message.is_empty() {
            stdout.push_str(resp.message.trim());
            stdout.push('\n');
        }
        if resp.result != Value::Null {
            stdout.push_str(&serde_json::to_string(&resp.result).unwrap_or_default());
        }
        return Ok(RpcReply {
            code: 0,
            stdout,
            stderr: String::new(),
            result: resp.result,
        });
    }
    Ok(RpcReply {
        code: if resp.code != 0 { resp.code } else { 1 },
        stdout: resp.message,
        stderr: resp.error,
        result: resp.result,
    })
}
