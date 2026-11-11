// Platform device API client: the five documented calls, strict response parsing and errors without secrets.

use crate::platform::{clean_text, valid_id, valid_token, Credentials, Secret, Workspace};
use serde::Deserialize;
use serde_json::{json, Value};
use std::io::Read;
use std::time::Duration;

pub const MAX_RESPONSE_BYTES: u64 = 64 * 1024;
pub const MAX_SNAPSHOT_BYTES: usize = 256 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ApiError {
    Network(String),
    Unauthorized,
    RateLimited(Duration),
    Status(u16, String),
    Malformed(String),
}

impl ApiError {
    pub fn retryable(&self) -> bool {
        // Blips worth trying again: the network, rate limits and server errors.
        match self {
            ApiError::Network(_) | ApiError::RateLimited(_) => true,
            ApiError::Status(code, _) => *code == 429 || *code >= 500,
            _ => false,
        }
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApiError::Network(e) => write!(f, "cannot reach the platform: {e}"),
            ApiError::Unauthorized => f.write_str("the platform no longer accepts this node's token (removed or revoked)"),
            ApiError::RateLimited(d) => write!(f, "the platform asked to wait {} s (rate limit)", d.as_secs()),
            ApiError::Status(code, err) if err.is_empty() => write!(f, "the platform answered HTTP {code}"),
            ApiError::Status(code, err) => write!(f, "the platform answered HTTP {code} ({err})"),
            ApiError::Malformed(e) => write!(f, "the platform sent an unexpected answer: {e}"),
        }
    }
}

pub struct Client {
    base: String,
    agent: ureq::Agent,
}

impl Client {
    pub fn new(base: &str, timeout: Duration) -> Result<Client, String> {
        // TLS verification stays on (ureq's default); no redirects, so a token never follows one elsewhere.
        let base = crate::platform::check_url(base)?;
        let agent = crate::net::agent_builder()
            .redirects(0)
            .timeout_read(timeout)
            .timeout_write(timeout)
            .timeout(timeout)
            .user_agent(&format!("goethe/{}", env!("CARGO_PKG_VERSION")))
            .build();
        Ok(Client { base, agent })
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    fn post(&self, path: &str, body: &[u8], bearer: Option<&Secret>) -> Result<Value, ApiError> {
        // One JSON POST; the reply is read up to the size cap and must be a JSON object.
        let mut req = self.agent.post(&format!("{}{path}", self.base)).set("Content-Type", "application/json").set("Accept", "application/json");
        if let Some(t) = bearer {
            req = req.set("Authorization", &format!("Bearer {}", t.expose()));
        }
        let resp = match req.send_bytes(body) {
            Ok(r) => r,
            Err(ureq::Error::Status(401, _)) => return Err(ApiError::Unauthorized),
            Err(ureq::Error::Status(429, r)) => return Err(ApiError::RateLimited(retry_after(r.header("Retry-After")))),
            Err(ureq::Error::Status(code, r)) => return Err(ApiError::Status(code, error_code(&read_capped(r).unwrap_or_default()))),
            Err(ureq::Error::Transport(t)) => return Err(ApiError::Network(transport_text(&t))),
        };
        let text = read_capped(resp)?;
        match serde_json::from_str::<Value>(&text) {
            Ok(v) if v.is_object() => Ok(v),
            _ => Err(ApiError::Malformed(format!("{path} did not return a JSON object"))),
        }
    }

    pub fn auth_start(&self, hostname: &str, fingerprint: &str) -> Result<Started, ApiError> {
        let body = json!({"action": "start", "hostname": hostname, "version": env!("CARGO_PKG_VERSION"), "fingerprint": fingerprint});
        parse_started(&self.post("/api/device/auth", body.to_string().as_bytes(), None)?, self.base.starts_with("http://"))
    }

    pub fn auth_poll(&self, device_code: &Secret) -> Result<Poll, ApiError> {
        let body = json!({"action": "poll", "device_code": device_code.expose()});
        parse_poll(&self.post("/api/device/auth", body.to_string().as_bytes(), None)?)
    }

    pub fn enroll(&self, authkey: &Secret, hostname: &str, fingerprint: &str) -> Result<Approved, ApiError> {
        let body = json!({"authkey": authkey.expose(), "hostname": hostname, "version": env!("CARGO_PKG_VERSION"), "fingerprint": fingerprint});
        parse_approved(&self.post("/api/device/enroll", body.to_string().as_bytes(), None)?)
    }

    pub fn heartbeat(&self, token: &Secret, snapshot: &[u8]) -> Result<HeartbeatReply, ApiError> {
        if snapshot.len() > MAX_SNAPSHOT_BYTES {
            return Err(ApiError::Malformed(format!("the snapshot is {} bytes, over the {MAX_SNAPSHOT_BYTES} byte limit", snapshot.len())));
        }
        parse_heartbeat(&self.post("/api/device/heartbeat", snapshot, Some(token))?)
    }

    pub fn logout(&self, token: &Secret) -> Result<(), ApiError> {
        // A token the platform no longer knows is already logged out.
        match self.post("/api/device/logout", b"{}", Some(token)) {
            Ok(v) if v.get("ok") == Some(&Value::Bool(true)) => Ok(()),
            Ok(_) => Err(ApiError::Malformed("logout did not return ok".into())),
            Err(ApiError::Unauthorized) => Ok(()),
            Err(e) => Err(e),
        }
    }
}

pub fn retry_after(header: Option<&str>) -> Duration {
    // Retry-After in seconds, 2 s when missing or not a number, never more than 60 s.
    let secs = header.and_then(|h| h.trim().parse::<u64>().ok()).unwrap_or(2);
    Duration::from_secs(secs.clamp(1, 60))
}

fn read_capped(resp: ureq::Response) -> Result<String, ApiError> {
    let mut buf = Vec::new();
    resp.into_reader()
        .take(MAX_RESPONSE_BYTES + 1)
        .read_to_end(&mut buf)
        .map_err(|e| ApiError::Network(e.kind().to_string()))?;
    if buf.len() as u64 > MAX_RESPONSE_BYTES {
        return Err(ApiError::Malformed(format!("the answer is larger than {MAX_RESPONSE_BYTES} bytes")));
    }
    String::from_utf8(buf).map_err(|_| ApiError::Malformed("the answer is not UTF-8".into()))
}

fn transport_text(t: &ureq::Transport) -> String {
    // The kind and the cause without the request URL or headers.
    let kind = t.kind().to_string();
    match t.message() {
        Some(m) => clean_text(&format!("{kind}: {m}"), 200),
        None => kind,
    }
}

pub fn error_code(body: &str) -> String {
    // The documented {error:"<code>"} reduced to a short code; anything else says nothing.
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("error").and_then(Value::as_str).map(str::to_string))
        .filter(|c| !c.is_empty() && c.len() <= 64 && c.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.')))
        .unwrap_or_default()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Started {
    pub device_code: Secret,
    pub user_code: String,
    pub verification_uri_complete: String,
    pub expires_in: u64,
    pub interval: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Approved {
    pub node_id: String,
    pub node_token: Secret,
    pub workspace: Workspace,
}

impl Approved {
    pub fn into_credentials(self, platform_url: &str) -> Credentials {
        Credentials {
            platform_url: platform_url.to_string(),
            node_id: self.node_id,
            workspace: self.workspace,
            token: self.node_token,
            linked_at: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Poll {
    Pending,
    SlowDown,
    Expired,
    Denied,
    Approved(Approved),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeartbeatReply {
    pub interval: Option<u64>,
    pub public_ip: Option<Option<String>>,
}

fn malformed(what: &str) -> ApiError {
    ApiError::Malformed(what.to_string())
}

pub fn parse_started(v: &Value, dev: bool) -> Result<Started, ApiError> {
    #[derive(Deserialize)]
    struct Raw {
        device_code: String,
        user_code: String,
        verification_uri_complete: String,
        expires_in: u64,
        interval: u64,
    }
    let r: Raw = serde_json::from_value(v.clone()).map_err(|_| malformed("start did not return device_code, user_code, verification_uri_complete, expires_in and interval"))?;
    if r.device_code.is_empty() || r.device_code.len() > 512 || !r.device_code.chars().all(|c| c.is_ascii_graphic()) {
        return Err(malformed("the device code is not valid"));
    }
    if r.user_code.is_empty() || r.user_code.len() > 32 || !r.user_code.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        return Err(malformed("the user code is not valid"));
    }
    let check = if dev { crate::platform::check_dev_link } else { crate::platform::check_link };
    let uri = check(&r.verification_uri_complete).map_err(|_| malformed("the verification link is not a valid https URL"))?;
    Ok(Started {
        device_code: Secret::new(r.device_code),
        user_code: r.user_code,
        verification_uri_complete: uri,
        expires_in: r.expires_in.clamp(30, 900),
        interval: r.interval.clamp(1, 60),
    })
}

pub fn parse_approved(v: &Value) -> Result<Approved, ApiError> {
    #[derive(Deserialize)]
    struct RawWorkspace {
        id: String,
        name: String,
    }
    #[derive(Deserialize)]
    struct Raw {
        node_id: String,
        node_token: String,
        workspace: RawWorkspace,
    }
    let r: Raw = serde_json::from_value(v.clone()).map_err(|_| malformed("the approval did not carry node_id, node_token and workspace {id, name}"))?;
    if !valid_id(&r.node_id) {
        return Err(malformed("the node id is not valid"));
    }
    if !valid_token(&r.node_token) {
        return Err(malformed("the node token is not valid"));
    }
    if !valid_id(&r.workspace.id) {
        return Err(malformed("the workspace id is not valid"));
    }
    Ok(Approved {
        node_id: r.node_id,
        node_token: Secret::new(r.node_token),
        workspace: Workspace { id: r.workspace.id, name: clean_text(&r.workspace.name, 128) },
    })
}

pub fn parse_poll(v: &Value) -> Result<Poll, ApiError> {
    match v.get("status").and_then(Value::as_str) {
        Some("pending") => Ok(Poll::Pending),
        Some("slow_down") => Ok(Poll::SlowDown),
        Some("expired") => Ok(Poll::Expired),
        Some("denied") => Ok(Poll::Denied),
        Some("approved") => parse_approved(v).map(Poll::Approved),
        _ => Err(malformed("poll returned no known status")),
    }
}

pub fn parse_heartbeat(v: &Value) -> Result<HeartbeatReply, ApiError> {
    if v.get("ok") != Some(&Value::Bool(true)) {
        return Err(malformed("heartbeat did not return ok"));
    }
    let interval = match v.get("interval") {
        None | Some(Value::Null) => None,
        Some(i) => Some(i.as_u64().ok_or_else(|| malformed("heartbeat interval is not a number"))?),
    };
    let public_ip = match v.get("public_ip") {
        None => None,
        Some(Value::Null) => Some(None),
        Some(Value::String(s)) => {
            let ip: std::net::IpAddr = s.trim().parse().map_err(|_| malformed("heartbeat public_ip is not an IP address"))?;
            Some(Some(ip.to_string()))
        }
        Some(_) => return Err(malformed("heartbeat public_ip is not a string")),
    };
    Ok(HeartbeatReply { interval, public_ip })
}

#[cfg(test)]
pub mod mock {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Debug)]
    pub struct Seen {
        pub path: String,
        pub authorization: Option<String>,
        pub body: String,
    }

    pub struct Mock {
        pub url: String,
        pub seen: Arc<Mutex<Vec<Seen>>>,
        pub connections: Arc<Mutex<usize>>,
    }

    pub fn serve(replies: Vec<(u16, String)>) -> Mock {
        // A loopback HTTP server on a free port answering each request with the next scripted reply (the last one repeats).
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let connections = Arc::new(Mutex::new(0usize));
        let (s, c) = (seen.clone(), connections.clone());
        std::thread::spawn(move || {
            let mut n = 0usize;
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                c.lock().map(|mut n| *n += 1).unwrap();
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() {
                    continue;
                }
                let path = line.split_whitespace().nth(1).unwrap_or("").to_string();
                let (mut len, mut auth) = (0usize, None);
                loop {
                    let mut h = String::new();
                    if reader.read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" {
                        break;
                    }
                    let lower = h.to_ascii_lowercase();
                    if let Some(v) = lower.strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap_or(0);
                    }
                    if lower.starts_with("authorization:") {
                        auth = Some(h[14..].trim().to_string());
                    }
                }
                let mut body = vec![0u8; len];
                let _ = reader.read_exact(&mut body);
                s.lock().unwrap().push(Seen { path, authorization: auth, body: String::from_utf8_lossy(&body).into_owned() });
                let (code, text) = replies.get(n).or(replies.last()).cloned().unwrap_or((500, String::new()));
                n += 1;
                let _ = write!(stream, "HTTP/1.1 {code} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}", text.len());
            }
        });
        Mock { url, seen, connections }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOKEN: &str = "mesh_node_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    fn approved_json() -> String {
        format!(r#"{{"status":"approved","node_id":"n1","node_token":"{TOKEN}","workspace":{{"id":"w1","name":"Lab"}}}}"#)
    }

    #[test]
    fn start_and_poll_follow_the_contract() {
        let m = mock::serve(vec![
            (200, r#"{"device_code":"dc1","user_code":"ABCD-1234","verification_uri_complete":"http://127.0.0.1:9/device?code=ABCD-1234","expires_in":900,"interval":3,"extra":1}"#.into()),
            (200, r#"{"status":"pending"}"#.into()),
            (200, r#"{"status":"slow_down"}"#.into()),
            (200, approved_json()),
        ]);
        let c = Client::new(&m.url, Duration::from_secs(5)).unwrap();
        let s = c.auth_start("host", "fp").unwrap();
        assert_eq!((s.user_code.as_str(), s.interval, s.expires_in), ("ABCD-1234", 3, 900));
        assert_eq!(s.verification_uri_complete, "http://127.0.0.1:9/device?code=ABCD-1234");
        assert_eq!(c.auth_poll(&s.device_code).unwrap(), Poll::Pending);
        assert_eq!(c.auth_poll(&s.device_code).unwrap(), Poll::SlowDown);
        let Poll::Approved(a) = c.auth_poll(&s.device_code).unwrap() else { panic!("approved expected") };
        assert_eq!((a.node_id.as_str(), a.workspace.name.as_str(), a.node_token.expose()), ("n1", "Lab", TOKEN));
        let seen = m.seen.lock().unwrap();
        let start: Value = serde_json::from_str(&seen[0].body).unwrap();
        assert_eq!(start["action"], "start");
        assert_eq!(start["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(seen[1].path, "/api/device/auth");
        assert!(seen.iter().all(|r| r.authorization.is_none()));
    }

    #[test]
    fn malformed_answers_are_errors() {
        for body in [
            r#"{"status":"wat"}"#,
            "[]",
            "not json",
            r#"{"status":"approved","node_id":"n1","node_token":"short","workspace":{"id":"w","name":"x"}}"#,
            r#"{"status":"approved","node_id":"../x","node_token":"mesh_node_aaaaaaaaaaaaaaaa","workspace":{"id":"w","name":"x"}}"#,
            r#"{"status":"approved","node_id":"n1","node_token":"mesh_node_aaaaaaaaaaaaaaaa"}"#,
        ] {
            let m = mock::serve(vec![(200, body.into())]);
            let c = Client::new(&m.url, Duration::from_secs(5)).unwrap();
            assert!(matches!(c.auth_poll(&Secret::new("d")), Err(ApiError::Malformed(_))), "{body}");
        }
        let m = mock::serve(vec![(200, format!("{{\"pad\":\"{}\"}}", "x".repeat(70_000)))]);
        let c = Client::new(&m.url, Duration::from_secs(5)).unwrap();
        assert!(matches!(c.auth_poll(&Secret::new("d")), Err(ApiError::Malformed(e)) if e.contains("larger")));
        let m = mock::serve(vec![(200, r#"{"device_code":"d","user_code":"AB\u001b[2J","verification_uri_complete":"http://evil.example.com/x","expires_in":900,"interval":3}"#.into())]);
        let c = Client::new(&m.url, Duration::from_secs(5)).unwrap();
        assert!(c.auth_start("h", "f").is_err());
    }

    #[test]
    fn statuses_map_to_errors_without_detail() {
        let m = mock::serve(vec![
            (401, r#"{"error":"unauthorized"}"#.into()),
            (429, r#"{"error":"rate_limited"}"#.into()),
            (500, r#"{"error":"internal","detail":"SELECT * FROM nodes"}"#.into()),
            (400, r#"{"error":"bad input with spaces"}"#.into()),
            (401, "{}".into()),
        ]);
        let c = Client::new(&m.url, Duration::from_secs(5)).unwrap();
        let t = Secret::new(TOKEN);
        assert_eq!(c.heartbeat(&t, b"{}").unwrap_err(), ApiError::Unauthorized);
        let e = c.heartbeat(&t, b"{}").unwrap_err();
        assert!(e.retryable() && e == ApiError::RateLimited(Duration::from_secs(2)), "{e:?}");
        assert_eq!(retry_after(Some("7")), Duration::from_secs(7));
        assert_eq!(retry_after(Some("600")), Duration::from_secs(60));
        assert_eq!(retry_after(Some("Wed, 21 Oct 2015")), Duration::from_secs(2));
        let e = c.heartbeat(&t, b"{}").unwrap_err();
        assert!(e.retryable() && !e.to_string().contains("SELECT"));
        let e = c.heartbeat(&t, b"{}").unwrap_err();
        assert_eq!(e, ApiError::Status(400, String::new()));
        assert!(!e.retryable());
        assert_eq!(c.logout(&t), Ok(()), "an unknown token is already logged out");
        let seen = m.seen.lock().unwrap();
        assert_eq!(seen[0].authorization.as_deref(), Some(format!("Bearer {TOKEN}").as_str()));
        assert_eq!(seen[4].path, "/api/device/logout");
        for e in [ApiError::Unauthorized, ApiError::Network("x".into()), ApiError::Status(500, "e".into())] {
            assert!(!format!("{e} {e:?}").contains(TOKEN));
        }
    }

    #[test]
    fn heartbeat_reply_and_size_limit() {
        let m = mock::serve(vec![
            (200, r#"{"ok":true,"node_id":"n1","at":"t","interval":10,"public_ip":"203.0.113.5"}"#.into()),
            (200, r#"{"ok":true,"public_ip":null}"#.into()),
            (200, r#"{"ok":true,"public_ip":"nope"}"#.into()),
            (200, r#"{"ok":false}"#.into()),
        ]);
        let c = Client::new(&m.url, Duration::from_secs(5)).unwrap();
        let t = Secret::new(TOKEN);
        let r = c.heartbeat(&t, b"{}").unwrap();
        assert_eq!((r.interval, r.public_ip), (Some(10), Some(Some("203.0.113.5".to_string()))));
        assert_eq!(c.heartbeat(&t, b"{}").unwrap().public_ip, Some(None));
        assert!(c.heartbeat(&t, b"{}").is_err());
        assert!(c.heartbeat(&t, b"{}").is_err());
        let big = vec![b' '; MAX_SNAPSHOT_BYTES + 1];
        assert!(matches!(c.heartbeat(&t, &big), Err(ApiError::Malformed(_))));
        assert_eq!(m.seen.lock().unwrap().len(), 4, "an oversize snapshot is never sent");
    }

    #[test]
    fn unreachable_and_redirects_fail_cleanly() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        drop(listener);
        let c = Client::new(&url, Duration::from_secs(2)).unwrap();
        let e = c.heartbeat(&Secret::new(TOKEN), b"{}").unwrap_err();
        assert!(e.retryable() && !e.to_string().contains(TOKEN), "{e}");
        let m = mock::serve(vec![(302, String::new())]);
        let c = Client::new(&m.url, Duration::from_secs(2)).unwrap();
        assert!(c.heartbeat(&Secret::new(TOKEN), b"{}").is_err());
        assert!(Client::new("http://example.com", Duration::from_secs(1)).is_err());
    }
}
