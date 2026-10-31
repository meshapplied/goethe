// Trace recorder: generation turns saved per session as a corpus for later labelling and training (format in docs/traces.md).

use crate::db;
use crate::paths::sanitize_model_id;
use crate::trace_assemble::{self, Api};
use chrono::{DateTime, SecondsFormat, Utc};
use flate2::write::GzEncoder;
use flate2::Compression;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

pub const SCHEMA: u32 = 1;
pub const DEFAULT_MAX_GB: f64 = 20.0;
pub const MIN_FREE_GB: f64 = 10.0;
pub const RESPONSE_CAP: usize = 32 << 20;
const SETTING: &str = "trace";
const LOCK: &str = ".lock";
const QUEUE: usize = 1024;
const LINK_TTL: Duration = Duration::from_secs(24 * 3600);
const LINK_CAP: usize = 200_000;
const MANIFEST_EVERY: Duration = Duration::from_secs(2);
const SPACE_EVERY: Duration = Duration::from_secs(10);
const MODEL_FILES: &[&str] = &[
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "chat_template.json",
    "model.safetensors.index.json",
    "hf_quant_config.json",
];

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TraceConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub all_keys: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keys: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub except: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dir: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_gb: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stopped: Option<String>,
}

impl TraceConfig {
    pub fn records(&self, key_id: &str) -> bool {
        // While enabled: all keys minus the exceptions, or only the listed keys.
        if !self.enabled {
            false
        } else if self.all_keys {
            !self.except.iter().any(|k| k == key_id)
        } else {
            self.keys.iter().any(|k| k == key_id)
        }
    }

    pub fn on(&self) -> bool {
        self.enabled && self.scoped()
    }

    pub fn scoped(&self) -> bool {
        self.all_keys || !self.keys.is_empty()
    }

    pub fn dir(&self, home: &Path) -> PathBuf {
        self.dir.as_ref().map_or_else(|| home.join("traces"), PathBuf::from)
    }

    pub fn max_gb(&self) -> f64 {
        self.max_gb.unwrap_or(DEFAULT_MAX_GB)
    }

    pub fn turn_on(&mut self, keys: &[String], all_keys: bool) {
        // No keys: resume the saved scope (every key when none is saved); with keys: add them (or drop them from the exceptions); all_keys: every key.
        self.enabled = true;
        if all_keys || (keys.is_empty() && !self.scoped()) {
            self.all_keys = true;
            self.keys.clear();
            self.except.clear();
        } else if keys.is_empty() {
        } else if self.all_keys {
            self.except.retain(|k| !keys.contains(k));
        } else {
            for k in keys {
                if !self.keys.contains(k) {
                    self.keys.push(k.clone());
                }
            }
        }
    }

    pub fn turn_off(&mut self, keys: &[String]) {
        // No keys: stop recording and keep the scope for the next on; with keys: take them out of the scope.
        if keys.is_empty() {
            self.enabled = false;
        } else if self.all_keys {
            for k in keys {
                if !self.except.contains(k) {
                    self.except.push(k.clone());
                }
            }
        } else {
            self.keys.retain(|k| !keys.contains(k));
        }
    }
}

pub fn load_config(con: &Connection) -> TraceConfig {
    con.query_row("SELECT value FROM goethe_settings WHERE key = ?1", [SETTING], |r| r.get::<_, String>(0))
        .ok()
        .and_then(|v| serde_json::from_str(&v).ok())
        .unwrap_or_default()
}

fn save_config(con: &Connection, cfg: &TraceConfig) -> Result<(), String> {
    let text = serde_json::to_string(cfg).map_err(|e| e.to_string())?;
    con.execute(
        "INSERT OR REPLACE INTO goethe_settings (key, value) VALUES (?1, ?2)",
        rusqlite::params![SETTING, text],
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}

fn resolve_keys(con: &Connection, wanted: &[String]) -> Result<Vec<String>, String> {
    // Each argument is a key id or a key name; a name shared by several keys must be given by id.
    let mut stmt = con.prepare("SELECT key_id, name FROM api_keys").map_err(|e| e.to_string())?;
    let rows: Vec<(String, String)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .collect();
    wanted
        .iter()
        .map(|w| {
            if rows.iter().any(|(id, _)| id == w) {
                return Ok(w.clone());
            }
            let named: Vec<&String> = rows.iter().filter(|(_, n)| n == w).map(|(id, _)| id).collect();
            match named.as_slice() {
                [id] => Ok((*id).clone()),
                [] => Err(format!("no API key with id or name {w} (goethe key lists them)")),
                ids => Err(format!(
                    "several keys are named {w}; pass one id: {}",
                    ids.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                )),
            }
        })
        .collect()
}

fn key_names(con: &Connection) -> HashMap<String, String> {
    let Ok(mut stmt) = con.prepare("SELECT key_id, name FROM api_keys") else {
        return HashMap::new();
    };
    stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .map(|rows| rows.filter_map(Result::ok).collect())
        .unwrap_or_default()
}

fn make_private_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::DirBuilder::new().recursive(true).mode(0o700).create(dir)
}

pub fn set(home: &Path, on: Option<bool>, keys: &[String], all_keys: bool, dir: Option<&str>, max_gb: Option<f64>) -> Result<(), String> {
    // Change what is recorded, the folder or the size limit; the daemon picks it up on its next request, no reload needed.
    let con = db::open(home).map_err(|e| e.to_string())?;
    let ids = resolve_keys(&con, keys)?;
    let mut cfg = load_config(&con);
    cfg.stopped = None;
    if let Some(d) = dir {
        let abs = std::path::absolute(d).map_err(|e| format!("trace folder {d}: {e}"))?;
        cfg.dir = Some(abs.to_string_lossy().into_owned());
    }
    if let Some(gb) = max_gb {
        if gb.is_nan() || gb <= 0.0 {
            return Err("--max-gb must be more than 0".into());
        }
        cfg.max_gb = Some(gb);
    }
    match on {
        Some(true) => cfg.turn_on(&ids, all_keys),
        Some(false) => cfg.turn_off(&ids),
        None => {}
    }
    let mut folder = cfg.dir(home);
    if cfg.on() {
        let (bytes, _) = folder_totals(&folder);
        if bytes as f64 >= cfg.max_gb() * 1e9 {
            return Err(format!(
                "the trace folder {} holds {}, over its {} GB limit: raise it with --max-gb or delete traces",
                folder.display(),
                human_bytes(bytes),
                cfg.max_gb()
            ));
        }
        make_private_dir(&folder).map_err(|e| format!("trace folder {}: {e}", folder.display()))?;
        if !crate::paths::is_writable(&folder) {
            return Err(format!("trace folder {} is not writable", folder.display()));
        }
        if let (Some(_), Ok(real)) = (&cfg.dir, std::fs::canonicalize(&folder)) {
            cfg.dir = Some(real.to_string_lossy().into_owned());
            folder = real;
        }
    }
    save_config(&con, &cfg)?;
    if folder.is_dir() {
        let action = match on {
            Some(true) => "on",
            Some(false) => "off",
            None => "settings",
        };
        let line = json!({"at": now_rfc3339(), "action": action, "keys": ids, "config": cfg});
        append_private(&folder.join("history.jsonl"), format!("{line}\n").as_bytes())
            .map_err(|e| format!("trace history: {e}"))?;
    }
    Ok(())
}

pub fn status(home: &Path) -> Result<Value, String> {
    // Config, folder totals from the files on disk, and the writer's counters from the manifest.
    let con = db::open(home).map_err(|e| e.to_string())?;
    let cfg = load_config(&con);
    let names = key_names(&con);
    let named = |ids: &[String]| -> Vec<Value> {
        ids.iter().map(|id| json!({"id": id, "name": names.get(id).cloned().unwrap_or_default()})).collect()
    };
    let dir = cfg.dir(home);
    let (bytes, sessions) = folder_totals(&dir);
    let manifest: Value = std::fs::read(dir.join("manifest.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or(Value::Null);
    Ok(json!({
        "recording": cfg.on(),
        "all_keys": cfg.all_keys,
        "keys": named(&cfg.keys),
        "except": named(&cfg.except),
        "dir": dir.to_string_lossy(),
        "max_gb": cfg.max_gb(),
        "stopped": cfg.stopped,
        "bytes": bytes,
        "sessions": sessions,
        "turns": manifest.get("turns").cloned().unwrap_or(json!(0)),
        "models": manifest.get("models").cloned().unwrap_or(json!(0)),
        "first_turn_at": manifest.get("first_turn_at").cloned().unwrap_or(Value::Null),
        "last_turn_at": manifest.get("last_turn_at").cloned().unwrap_or(Value::Null),
        "dropped": manifest.get("dropped").cloned().unwrap_or(json!(0)),
        "paused": manifest.get("paused").cloned().unwrap_or(Value::Null),
        "free_bytes": free_bytes(&dir),
    }))
}

fn folder_totals(dir: &Path) -> (u64, u64) {
    // Bytes of every file under the folder, and the number of session files.
    let mut bytes = 0;
    let mut sessions = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let Ok(meta) = e.metadata() else {
                continue;
            };
            if meta.is_dir() {
                stack.push(e.path());
            } else {
                bytes += meta.len();
                if e.file_name().to_string_lossy().ends_with(".jsonl.gz") {
                    sessions += 1;
                }
            }
        }
    }
    (bytes, sessions)
}

fn free_bytes(dir: &Path) -> Option<u64> {
    // Free space for an unprivileged writer on the folder's filesystem (or its nearest existing parent).
    let mut p = dir;
    while !p.exists() {
        p = p.parent()?;
    }
    let c = std::ffi::CString::new(p.as_os_str().as_encoded_bytes()).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    (unsafe { libc::statvfs(c.as_ptr(), &mut st) } == 0).then(|| st.f_bavail as u64 * st.f_frsize as u64)
}

pub fn human_bytes(n: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1000.0 && i + 1 < units.len() {
        v /= 1000.0;
        i += 1;
    }
    if i == 0 {
        format!("{n} B")
    } else {
        format!("{v:.1} {}", units[i])
    }
}

pub fn now_rfc3339() -> String {
    Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true)
}

fn append_private(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let mut f = std::fs::OpenOptions::new().create(true).append(true).mode(0o600).open(path)?;
    f.write_all(data)
}

fn write_private_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::OpenOptions::new().create(true).write(true).truncate(true).mode(0o600).open(&tmp)?;
        f.write_all(data)?;
    }
    std::fs::rename(tmp, path)
}

fn random_hex(n: usize) -> String {
    let mut b = vec![0u8; n];
    unsafe {
        libc::getrandom(b.as_mut_ptr().cast(), b.len(), 0);
    }
    b.iter().map(|x| format!("{x:02x}")).collect()
}

type Hash = [u8; 32];

fn hash_step(prev: &Hash, part: &str) -> Hash {
    let mut h = Sha256::new();
    h.update(prev);
    h.update(part.as_bytes());
    h.finalize().into()
}

fn hex(h: &Hash) -> String {
    h.iter().map(|x| format!("{x:02x}")).collect()
}

pub struct Capture {
    pub dir: PathBuf,
    pub max_gb: f64,
    pub path: String,
    pub request_id: String,
    pub key_id: String,
    pub instance_id: String,
    pub model: String,
    pub session_hint: Option<String>,
    pub request: Vec<u8>,
    pub started_at: SystemTime,
    pub duration_s: f64,
    pub ttft_s: Option<f64>,
    pub status: u16,
    pub sse: bool,
    pub complete: bool,
    pub response: Vec<u8>,
    pub truncated: bool,
}

pub struct Recorder {
    tx: SyncSender<Capture>,
    dropped: Arc<AtomicU64>,
}

impl Recorder {
    pub fn start(home: &Path) -> Recorder {
        // One writer thread behind a bounded queue: a slow disk drops traces (counted), never slows a reply.
        let (tx, rx) = mpsc::sync_channel(QUEUE);
        let dropped = Arc::new(AtomicU64::new(0));
        let mut writer = Writer::new(home, Arc::clone(&dropped));
        let _ = std::thread::Builder::new().name("goethe-trace".into()).spawn(move || loop {
            match rx.recv_timeout(MANIFEST_EVERY) {
                Ok(c) => writer.handle(c),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            writer.tick();
        });
        Recorder { tx, dropped }
    }

    pub fn submit(&self, c: Capture) {
        if let Err(TrySendError::Full(_)) = self.tx.try_send(c) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

struct Link {
    at: Instant,
    turn_id: String,
    session_id: String,
    with_reply: bool,
    params: Hash,
}

struct Session {
    file: PathBuf,
    turns: u64,
    last: Instant,
}

#[derive(Default)]
struct Totals {
    turns: u64,
    by_model: BTreeMap<String, u64>,
    sessions: u64,
    bytes: u64,
    paused_dropped: u64,
    errors: u64,
    last_error: Option<String>,
    first_turn_at: Option<String>,
    last_turn_at: Option<String>,
    created: Option<String>,
}

struct Writer {
    home: PathBuf,
    dir: Option<PathBuf>,
    dropped: Arc<AtomicU64>,
    links: HashMap<Hash, Link>,
    order: VecDeque<(Hash, Instant)>,
    sessions: HashMap<String, Session>,
    hinted: HashMap<String, String>,
    fingerprints: HashMap<String, String>,
    models: Map<String, Value>,
    totals: Totals,
    paused: Option<String>,
    dirty: bool,
    last_manifest: Instant,
    last_space: Option<Instant>,
    con: Option<Connection>,
}

impl Writer {
    fn new(home: &Path, dropped: Arc<AtomicU64>) -> Writer {
        Writer {
            home: home.to_path_buf(),
            dir: None,
            dropped,
            links: HashMap::new(),
            order: VecDeque::new(),
            sessions: HashMap::new(),
            hinted: HashMap::new(),
            fingerprints: HashMap::new(),
            models: Map::new(),
            totals: Totals::default(),
            paused: None,
            dirty: false,
            last_manifest: Instant::now(),
            last_space: None,
            con: None,
        }
    }

    fn open_dir(&mut self, dir: &Path) {
        // A new folder starts fresh: totals from its files and manifest, links and sessions forgotten.
        self.dir = Some(dir.to_path_buf());
        self.links.clear();
        self.order.clear();
        self.sessions.clear();
        self.hinted.clear();
        self.fingerprints.clear();
        self.paused = None;
        self.last_space = None;
        let _ = make_private_dir(dir);
        if let Err(e) = prune(dir) {
            eprintln!("goethe trace: {e}");
        }
        let manifest: Value = std::fs::read(dir.join("manifest.json"))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or(Value::Null);
        let text = |k: &str| manifest.get(k).and_then(Value::as_str).map(str::to_string);
        let (bytes, sessions) = folder_totals(dir);
        let by_model = turns_by_model(&manifest);
        self.totals = Totals {
            turns: by_model.values().sum(),
            by_model,
            sessions,
            bytes,
            created: text("created").or_else(|| Some(now_rfc3339())),
            first_turn_at: text("first_turn_at"),
            last_turn_at: text("last_turn_at"),
            ..Totals::default()
        };
        self.models = std::fs::read(dir.join("models.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default();
        self.dirty = true;
        self.write_manifest();
    }

    fn handle(&mut self, c: Capture) {
        if self.dir.as_deref() != Some(c.dir.as_path()) {
            self.open_dir(&c.dir);
        } else if self.totals.by_model.contains_key(&c.model) && !c.dir.join("v1").join(sanitize_model_id(&c.model)).is_dir() {
            self.forget_vanished(&c.dir);
        }
        self.check_space(c.max_gb);
        if self.paused.is_some() {
            self.totals.paused_dropped += 1;
            self.dirty = true;
            return;
        }
        if let Err(e) = self.record(c) {
            eprintln!("goethe trace: {e}");
            self.totals.errors += 1;
            self.totals.last_error = Some(e);
            self.dirty = true;
        }
    }

    fn check_space(&mut self, max_gb: f64) {
        // The size cap is checked on every turn, free disk space every SPACE_EVERY.
        let over_cap = self.totals.bytes as f64 >= max_gb * 1e9;
        let capped = self.paused.as_deref().is_some_and(|p| p.starts_with("size limit"));
        if !over_cap && !capped && self.last_space.is_some_and(|t| t.elapsed() < SPACE_EVERY) {
            return;
        }
        self.last_space = Some(Instant::now());
        let low_disk = self
            .dir
            .as_deref()
            .and_then(free_bytes)
            .is_some_and(|b| (b as f64) < MIN_FREE_GB * 1e9);
        if over_cap && !capped {
            self.turn_off_at_limit(max_gb);
        }
        let paused = if over_cap {
            Some(format!("size limit ({max_gb} GB) reached"))
        } else if low_disk {
            Some(format!("less than {MIN_FREE_GB} GB free on the disk"))
        } else {
            None
        };
        if paused != self.paused {
            if let Some(p) = &paused {
                eprintln!("goethe trace: recording paused: {p}");
            }
            self.paused = paused;
            self.dirty = true;
        }
    }

    fn turn_off_at_limit(&mut self, max_gb: f64) {
        // The size limit turns recording off like goethe trace off, with the reason kept for goethe trace and in history.jsonl.
        if self.con.is_none() {
            self.con = db::open(&self.home).ok();
        }
        let (Some(con), Some(dir)) = (self.con.as_ref(), self.dir.clone()) else {
            return;
        };
        let mut cfg = load_config(con);
        if !cfg.on() || cfg.dir(&self.home) != dir {
            return;
        }
        let before = cfg.clone();
        let at = now_rfc3339();
        cfg.enabled = false;
        cfg.stopped = Some(format!("the {max_gb} GB limit was reached at {at}"));
        if let Err(e) = save_config(con, &cfg) {
            eprintln!("goethe trace: turning off at the size limit: {e}");
            return;
        }
        eprintln!("goethe trace: recording turned off: the {max_gb} GB limit was reached");
        let line = json!({"at": at, "action": "off", "reason": "size limit", "max_gb": max_gb, "was": before, "config": cfg});
        if let Err(e) = append_private(&dir.join("history.jsonl"), format!("{line}\n").as_bytes()) {
            eprintln!("goethe trace: trace history: {e}");
        }
    }

    fn record(&mut self, c: Capture) -> Result<(), String> {
        let api = Api::from_path(&c.path).ok_or_else(|| format!("untraced route {}", c.path))?;
        let mut params = match serde_json::from_slice::<Value>(&c.request) {
            Ok(Value::Object(m)) => m,
            _ => return Err("request body is not a JSON object".into()),
        };
        let messages: Vec<Value> = if api.has_messages() {
            match params.remove("messages") {
                Some(Value::Array(a)) => a,
                _ => Vec::new(),
            }
        } else {
            Vec::new()
        };
        let body = trace_assemble::assemble(api, c.sse, &c.response);
        let ok = c.status == 200 && c.complete && !c.truncated && body.get("error").is_none();
        let params_hash = hash_step(&[0; 32], &trace_assemble::sorted(&Value::Object(params.clone())).to_string());

        let mut parts: Vec<String> = Vec::new();
        let system_offset = match params.get("system") {
            Some(s) if api == Api::Messages && !s.is_null() => {
                parts.push(trace_assemble::canonical(&json!({"role": "system", "content": s})));
                1
            }
            _ => 0,
        };
        parts.extend(messages.iter().map(trace_assemble::canonical));
        let seed = hash_step(&[0; 32], &format!("goethe-trace-v1\0{}\0{}\0{}", c.key_id, c.model, c.session_hint.as_deref().unwrap_or("")));
        let mut chain = vec![seed];
        for p in &parts {
            let next = hash_step(chain.last().unwrap_or(&seed), p);
            chain.push(next);
        }
        let request_hash = *chain.last().unwrap_or(&seed);

        let parent = if api.has_messages() {
            (1..chain.len()).rev().find_map(|k| self.links.get(&chain[k]).map(|l| (k, l))).filter(|(_, l)| self.sessions.contains_key(&l.session_id))
        } else {
            None
        };
        let (messages_from, parent_json, params_same, session_id) = match parent {
            Some((k, l)) => (
                k.saturating_sub(system_offset),
                json!({"turn_id": l.turn_id, "with_reply": l.with_reply}),
                l.params == params_hash,
                l.session_id.clone(),
            ),
            None => {
                let hinted = c
                    .session_hint
                    .as_ref()
                    .map(|h| hint_key(&c, h))
                    .and_then(|h| self.hinted.get(&h).cloned())
                    .filter(|s| self.sessions.contains_key(s));
                let sid = match hinted {
                    Some(s) => s,
                    None => self.new_session(&c)?,
                };
                (0, Value::Null, false, sid)
            }
        };

        let fingerprint = self.fingerprint(&c.instance_id);
        let session = self.sessions.get_mut(&session_id).ok_or("session vanished")?;
        let turn_id = format!("t_{}", random_hex(8));
        let started: DateTime<Utc> = c.started_at.into();
        let reply = if ok { trace_assemble::reply_message(api, &body) } else { None };
        let reply_hash = reply.as_ref().map(|r| hash_step(&request_hash, &trace_assemble::canonical(r)));

        let mut rec = Map::new();
        rec.insert("schema".into(), json!(SCHEMA));
        rec.insert("turn_id".into(), json!(turn_id));
        rec.insert("session_id".into(), json!(session_id));
        rec.insert("seq".into(), json!(session.turns));
        rec.insert("started_at".into(), json!(started.to_rfc3339_opts(SecondsFormat::Millis, true)));
        rec.insert("duration_s".into(), json!((c.duration_s * 1000.0).round() / 1000.0));
        rec.insert("ttft_s".into(), json!(c.ttft_s.map(|t| (t * 1000.0).round() / 1000.0)));
        rec.insert("api".into(), json!(api.name()));
        rec.insert("model".into(), json!(c.model));
        rec.insert("model_fingerprint".into(), json!(fingerprint));
        rec.insert("instance_id".into(), json!(c.instance_id));
        rec.insert("key_id".into(), json!(c.key_id));
        rec.insert("request_id".into(), json!(c.request_id));
        if let Some(h) = &c.session_hint {
            rec.insert("session_hint".into(), json!(h));
        }
        if api.has_messages() {
            rec.insert("parent".into(), parent_json);
            rec.insert("messages_from".into(), json!(messages_from));
            rec.insert("request_hash".into(), json!(hex(&request_hash)));
            rec.insert("reply_hash".into(), json!(reply_hash.as_ref().map(hex)));
        }
        if params_same {
            rec.insert("params_from_parent".into(), json!(true));
        } else {
            rec.insert("params".into(), Value::Object(params));
        }
        if api.has_messages() {
            rec.insert("messages".into(), Value::Array(messages.into_iter().skip(messages_from).collect()));
        }
        rec.insert(
            "response".into(),
            json!({"status": c.status, "stream": c.sse, "complete": c.complete, "truncated": c.truncated, "body": body}),
        );

        let mut line = serde_json::to_vec(&Value::Object(rec)).map_err(|e| e.to_string())?;
        line.push(b'\n');
        let mut gz = GzEncoder::new(Vec::with_capacity(line.len() / 3), Compression::default());
        gz.write_all(&line).map_err(|e| e.to_string())?;
        let packed = gz.finish().map_err(|e| e.to_string())?;
        append_private(&session.file, &packed).map_err(|e| format!("{}: {e}", session.file.display()))?;
        session.turns += 1;
        session.last = Instant::now();

        let now = Instant::now();
        let link = |with_reply| Link { at: now, turn_id: turn_id.clone(), session_id: session_id.clone(), with_reply, params: params_hash };
        if api.has_messages() {
            self.add_link(request_hash, link(false));
            if let Some(h) = reply_hash {
                self.add_link(h, link(true));
            }
        }
        let at = now_rfc3339();
        self.totals.turns += 1;
        self.totals.by_model.entry(c.model).and_modify(|n| *n += 1).or_insert(1);
        self.totals.bytes += packed.len() as u64;
        self.totals.first_turn_at.get_or_insert_with(|| at.clone());
        self.totals.last_turn_at = Some(at);
        self.dirty = true;
        Ok(())
    }

    fn new_session(&mut self, c: &Capture) -> Result<String, String> {
        // File per session under v1/<model>/<day>/, plus one line in sessions.jsonl.
        let dir = self.dir.clone().ok_or("no trace folder")?;
        let started: DateTime<Utc> = c.started_at.into();
        let sid = format!("s_{}", random_hex(8));
        let rel = PathBuf::from(format!("v1/{}/{}", sanitize_model_id(&c.model), started.format("%Y-%m-%d")))
            .join(format!("{}_{sid}.jsonl.gz", started.format("%Y%m%dT%H%M%SZ")));
        let file = dir.join(&rel);
        if let Some(parent) = file.parent() {
            make_private_dir(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        let line = json!({
            "session_id": sid,
            "started_at": started.to_rfc3339_opts(SecondsFormat::Millis, true),
            "model": c.model,
            "key_id": c.key_id,
            "session_hint": c.session_hint,
            "file": rel.to_string_lossy(),
        });
        let line = format!("{line}\n");
        locked(&dir, || append_private(&dir.join("sessions.jsonl"), line.as_bytes())).map_err(|e| format!("sessions.jsonl: {e}"))?;
        self.totals.bytes += line.len() as u64;
        self.sessions.insert(sid.clone(), Session { file, turns: 0, last: Instant::now() });
        if let Some(h) = &c.session_hint {
            self.hinted.insert(hint_key(c, h), sid.clone());
        }
        self.totals.sessions += 1;
        Ok(sid)
    }

    fn add_link(&mut self, h: Hash, link: Link) {
        self.order.push_back((h, link.at));
        self.links.insert(h, link);
    }

    fn tick(&mut self) {
        // Forget links and sessions idle past LINK_TTL (or beyond LINK_CAP links), then refresh the manifest.
        while let Some((h, at)) = self.order.front().copied() {
            if at.elapsed() < LINK_TTL && self.order.len() <= LINK_CAP {
                break;
            }
            self.order.pop_front();
            if self.links.get(&h).is_some_and(|l| l.at == at) {
                self.links.remove(&h);
            }
        }
        self.sessions.retain(|_, s| s.last.elapsed() < LINK_TTL);
        let live = &self.sessions;
        self.hinted.retain(|_, sid| live.contains_key(sid));
        if self.dirty && self.last_manifest.elapsed() >= MANIFEST_EVERY {
            self.write_manifest();
        }
    }

    fn forget_vanished(&mut self, dir: &Path) {
        // A model folder deleted under us (goethe remove, or by hand): prune the index, drop its counts and sessions, keep the rest.
        if let Err(e) = prune(dir) {
            eprintln!("goethe trace: {e}");
        }
        self.totals.by_model.retain(|m, _| dir.join("v1").join(sanitize_model_id(m)).is_dir());
        self.totals.turns = self.totals.by_model.values().sum();
        self.sessions.retain(|_, s| s.file.exists());
        let live = &self.sessions;
        self.hinted.retain(|_, sid| live.contains_key(sid));
        self.links.retain(|_, l| live.contains_key(&l.session_id));
        self.fingerprints.clear();
        self.models = std::fs::read(dir.join("models.json"))
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default();
        (self.totals.bytes, self.totals.sessions) = folder_totals(dir);
        self.dirty = true;
    }

    fn write_manifest(&mut self) {
        let Some(dir) = self.dir.clone() else {
            return;
        };
        if self.totals.by_model.keys().any(|m| !dir.join("v1").join(sanitize_model_id(m)).is_dir()) {
            self.forget_vanished(&dir);
        }
        let t = &self.totals;
        let manifest = json!({
            "schema": SCHEMA,
            "format": "https://github.com/meshapplied/goethe/blob/main/docs/traces.md",
            "layout": "v1/<model>/<YYYY-MM-DD>/<start>_<session_id>.jsonl.gz: one gzip member per turn, one JSON object per line",
            "goethe_version": env!("CARGO_PKG_VERSION"),
            "created": t.created,
            "updated": now_rfc3339(),
            "turns": t.turns,
            "turns_by_model": t.by_model,
            "sessions": t.sessions,
            "models": self.models.len(),
            "bytes": t.bytes,
            "first_turn_at": t.first_turn_at,
            "last_turn_at": t.last_turn_at,
            "dropped": {"queue_full": self.dropped.load(Ordering::Relaxed), "paused": t.paused_dropped, "errors": t.errors},
            "last_error": t.last_error,
            "paused": self.paused,
        });
        let text = serde_json::to_vec_pretty(&manifest).unwrap_or_default();
        if let Err(e) = write_private_atomic(&dir.join("manifest.json"), &text) {
            eprintln!("goethe trace: manifest.json: {e}");
        }
        self.dirty = false;
        self.last_manifest = Instant::now();
    }

    fn fingerprint(&mut self, instance_id: &str) -> String {
        // What produced a turn: weights, tokenizer, chat template and engine flags, hashed; details go to models.json.
        if let Some(fp) = self.fingerprints.get(instance_id) {
            return fp.clone();
        }
        let Some((entry, path)) = self.describe_instance(instance_id) else {
            return "unknown".into();
        };
        let fp = format!("m_{}", &hex(&hash_step(&[0; 32], &entry.to_string()))[..16]);
        self.fingerprints.insert(instance_id.to_string(), fp.clone());
        if !self.models.contains_key(&fp) {
            let mut entry = entry;
            if let Some(m) = entry.as_object_mut() {
                m.insert("path".into(), json!(path));
                m.insert("first_seen".into(), json!(now_rfc3339()));
                m.insert("goethe_version".into(), json!(env!("CARGO_PKG_VERSION")));
            }
            self.models.insert(fp.clone(), entry);
            if let Some(dir) = &self.dir {
                let text = serde_json::to_vec_pretty(&self.models).unwrap_or_default();
                if let Err(e) = locked(dir, || write_private_atomic(&dir.join("models.json"), &text)) {
                    eprintln!("goethe trace: models.json: {e}");
                }
            }
        }
        fp
    }

    fn describe_instance(&mut self, instance_id: &str) -> Option<(Value, Option<String>)> {
        if self.con.is_none() {
            self.con = db::open(&self.home).ok();
        }
        let con = self.con.as_ref()?;
        let (model_id, path, extra): (String, Option<String>, Option<String>) = con
            .query_row(
                "SELECT li.model_id, im.local_path, li.extra_sglang_args FROM loaded_instances li
                 LEFT JOIN installed_models im ON im.model_id = li.model_id WHERE li.instance_id = ?1",
                [instance_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .ok()?;
        let extra: Value = extra.and_then(|e| serde_json::from_str(&e).ok()).unwrap_or(json!([]));
        let dir = path.as_deref().map(|p| self.home.join(p));
        let entry = json!({
            "model_id": model_id,
            "files": dir.as_deref().map(model_files).unwrap_or_default(),
            "weights": dir.as_deref().map(weight_files).unwrap_or(Value::Null),
            "engine_args": extra,
        });
        Some((entry, path))
    }
}

fn hint_key(c: &Capture, hint: &str) -> String {
    format!("{}\0{}\0{hint}", c.key_id, c.model)
}

fn turns_by_model(manifest: &Value) -> BTreeMap<String, u64> {
    manifest
        .get("turns_by_model")
        .and_then(Value::as_object)
        .map(|m| m.iter().filter_map(|(k, v)| Some((k.clone(), v.as_u64()?))).collect())
        .unwrap_or_default()
}

fn locked<T>(dir: &Path, f: impl FnOnce() -> std::io::Result<T>) -> std::io::Result<T> {
    // Exclusive flock on the folder's lock file around index writes shared by goethed and the CLI.
    let lock = std::fs::OpenOptions::new().create(true).append(true).mode(0o600).open(dir.join(LOCK))?;
    if unsafe { libc::flock(std::os::fd::AsRawFd::as_raw_fd(&lock), libc::LOCK_EX) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    f()
}

fn prune(dir: &Path) -> Result<(), String> {
    // Drop sessions.jsonl lines, models.json entries and manifest counts of models whose v1/<model> folder is gone.
    let present = |model: &str| dir.join("v1").join(sanitize_model_id(model)).is_dir();
    locked(dir, || {
        let sessions = dir.join("sessions.jsonl");
        if let Ok(text) = std::fs::read_to_string(&sessions) {
            let kept: String = text
                .lines()
                .filter(|l| serde_json::from_str::<Value>(l).ok().and_then(|v| v["model"].as_str().map(present)).unwrap_or(true))
                .map(|l| format!("{l}\n"))
                .collect();
            if kept.len() != text.len() {
                write_private_atomic(&sessions, kept.as_bytes())?;
            }
        }
        let models_path = dir.join("models.json");
        let mut models = std::fs::read(&models_path)
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default();
        let before = models.len();
        models.retain(|_, m| m["model_id"].as_str().is_none_or(present));
        if models.len() != before {
            write_private_atomic(&models_path, &serde_json::to_vec_pretty(&models).unwrap_or_default())?;
        }
        let path = dir.join("manifest.json");
        let Some(mut manifest) = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) else {
            return Ok(());
        };
        let mut by_model = turns_by_model(&manifest);
        let n = by_model.len();
        by_model.retain(|m, _| present(m));
        if by_model.len() == n && models.len() == before {
            return Ok(());
        }
        let (bytes, sessions) = folder_totals(dir);
        if let Some(m) = manifest.as_object_mut() {
            m.insert("turns".into(), json!(by_model.values().sum::<u64>()));
            m.insert("turns_by_model".into(), json!(by_model));
            m.insert("sessions".into(), json!(sessions));
            m.insert("bytes".into(), json!(bytes));
            m.insert("models".into(), json!(models.len()));
            m.insert("updated".into(), json!(now_rfc3339()));
        }
        write_private_atomic(&path, &serde_json::to_vec_pretty(&manifest).unwrap_or_default())
    })
    .map_err(|e| format!("pruning {}: {e}", dir.display()))
}

pub struct ModelTraces {
    pub folder: PathBuf,
    pub bytes: u64,
    pub sessions: u64,
}

pub fn model_traces(home: &Path, model_id: &str) -> Result<Option<ModelTraces>, String> {
    // The model's session files in the configured trace folder, or None when it has none.
    let con = db::open(home).map_err(|e| e.to_string())?;
    Ok(model_traces_in(&load_config(&con).dir(home), model_id))
}

fn model_traces_in(dir: &Path, model_id: &str) -> Option<ModelTraces> {
    let folder = dir.join("v1").join(sanitize_model_id(model_id));
    let (bytes, sessions) = folder_totals(&folder);
    (folder.is_dir() && sessions > 0).then_some(ModelTraces { folder, bytes, sessions })
}

pub fn delete_model_traces(home: &Path, model_id: &str) -> Result<Option<ModelTraces>, String> {
    // Delete v1/<model> and its index entries; a running goethed notices the missing folder and reloads.
    let con = db::open(home).map_err(|e| e.to_string())?;
    delete_model_traces_in(&load_config(&con).dir(home), model_id)
}

fn delete_model_traces_in(dir: &Path, model_id: &str) -> Result<Option<ModelTraces>, String> {
    let Some(t) = model_traces_in(dir, model_id) else {
        return Ok(None);
    };
    std::fs::remove_dir_all(&t.folder).map_err(|e| format!("{}: {e}", t.folder.display()))?;
    prune(dir)?;
    Ok(Some(t))
}

fn model_files(dir: &Path) -> Map<String, Value> {
    // sha256 of the small files that decide tokenization, the chat template and the architecture.
    MODEL_FILES
        .iter()
        .filter_map(|name| {
            let bytes = std::fs::read(dir.join(name)).ok()?;
            Some((name.to_string(), json!(hex(&Sha256::digest(&bytes).into()))))
        })
        .collect()
}

fn weight_files(dir: &Path) -> Value {
    // Weight file names and sizes: hashing hundreds of GB is too slow, and these change with any re-quantization.
    let mut files: Vec<(String, u64)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let size = std::fs::metadata(e.path()).ok()?.len();
            (name.ends_with(".safetensors") || name.ends_with(".bin") || name.ends_with(".gguf")).then_some((name, size))
        })
        .collect();
    files.sort();
    json!({
        "count": files.len(),
        "bytes": files.iter().map(|(_, s)| s).sum::<u64>(),
        "files": files.into_iter().map(|(n, s)| json!([n, s])).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::MultiGzDecoder;
    use std::io::Read;

    #[test]
    fn scope_follows_on_and_off_per_key() {
        let mut cfg = TraceConfig::default();
        assert!(!cfg.on() && !cfg.records("k1"));
        cfg.turn_on(&["k1".into()], false);
        assert!(cfg.records("k1") && !cfg.records("k2"));
        cfg.turn_off(&[]);
        assert!(!cfg.on() && !cfg.records("k1") && cfg.keys == ["k1"]);
        cfg.turn_on(&[], false);
        assert!(cfg.records("k1") && !cfg.records("k2"));
        cfg.turn_on(&[], true);
        assert!(cfg.records("k1") && cfg.records("k2"));
        cfg.turn_off(&["k2".into()]);
        assert!(cfg.records("k1") && !cfg.records("k2"));
        cfg.turn_off(&[]);
        cfg.turn_on(&[], false);
        assert!(cfg.records("k1") && !cfg.records("k2"));
        cfg.turn_on(&["k2".into()], false);
        assert!(cfg.records("k2") && cfg.except.is_empty());
        let mut fresh = TraceConfig::default();
        fresh.turn_on(&[], false);
        assert!(fresh.records("anything"));
    }

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn capture(dir: &Path, path: &str, request: Value, response: Value) -> Capture {
        Capture {
            dir: dir.to_path_buf(),
            max_gb: 1.0,
            path: path.into(),
            request_id: format!("r{:04}", NEXT.fetch_add(1, Ordering::Relaxed)),
            key_id: "k1".into(),
            instance_id: "i1".into(),
            model: "org/model".into(),
            session_hint: None,
            request: serde_json::to_vec(&request).unwrap(),
            started_at: SystemTime::now(),
            duration_s: 0.5,
            ttft_s: Some(0.1),
            status: 200,
            sse: false,
            complete: true,
            response: serde_json::to_vec(&response).unwrap(),
            truncated: false,
        }
    }

    fn turns(dir: &Path) -> Vec<Value> {
        let mut out = Vec::new();
        let mut files: Vec<PathBuf> = Vec::new();
        let mut stack = vec![dir.join("v1")];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                if e.path().is_dir() {
                    stack.push(e.path());
                } else {
                    files.push(e.path());
                }
            }
        }
        for f in files {
            let mut text = String::new();
            MultiGzDecoder::new(std::fs::File::open(f).unwrap()).read_to_string(&mut text).unwrap();
            out.extend(text.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()));
        }
        out.sort_by_key(|t| t["request_id"].as_str().unwrap_or("").to_string());
        out
    }

    fn reply(text: &str) -> Value {
        json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": text, "reasoning_content": "r"}, "finish_reason": "stop"}]})
    }

    #[test]
    fn follow_up_turns_store_only_new_messages_and_regenerations_share_a_request() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("traces");
        let mut w = Writer::new(tmp.path(), Arc::new(AtomicU64::new(0)));
        let sys = json!({"role": "system", "content": "be brief"});
        let u1 = json!({"role": "user", "content": "hi"});
        let a1 = json!({"role": "assistant", "content": "hello"});
        let u2 = json!({"role": "user", "content": "more"});
        w.handle(capture(&dir, "/chat/completions", json!({"model": "m", "messages": [sys, u1]}), reply("hello")));
        w.handle(capture(&dir, "/chat/completions", json!({"model": "m", "messages": [sys, u1, a1, u2]}), reply("sure")));
        w.handle(capture(&dir, "/chat/completions", json!({"model": "m", "messages": [sys, u1, a1, u2]}), reply("again")));
        w.handle(capture(&dir, "/chat/completions", json!({"model": "m", "max_tokens": 5, "messages": [u2]}), reply("other")));
        let t = turns(&dir);
        assert_eq!(t.len(), 4);
        let (first, follow, regen, other) = (&t[0], &t[1], &t[2], &t[3]);
        assert_eq!(first["parent"], Value::Null);
        assert_eq!(first["messages"].as_array().unwrap().len(), 2);
        assert_eq!(follow["session_id"], first["session_id"]);
        assert_eq!(follow["parent"], json!({"turn_id": first["turn_id"], "with_reply": true}));
        assert_eq!(follow["messages_from"], 3);
        assert_eq!(follow["messages"], json!([u2]));
        assert_eq!(follow["params_from_parent"], true);
        assert_eq!(regen["request_hash"], follow["request_hash"]);
        assert_eq!(regen["parent"]["turn_id"], follow["turn_id"]);
        assert_eq!(regen["parent"]["with_reply"], false);
        assert_eq!(regen["messages"], json!([]));
        assert_ne!(other["session_id"], first["session_id"]);
        assert_eq!(other["params"]["max_tokens"], 5);
        let sessions = std::fs::read_to_string(dir.join("sessions.jsonl")).unwrap();
        assert_eq!(sessions.lines().count(), 2);
        w.write_manifest();
        let manifest: Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!((manifest["turns"].as_u64(), manifest["sessions"].as_u64()), (Some(4), Some(2)));
    }

    #[test]
    fn other_keys_and_failed_replies_do_not_link() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("traces");
        let mut w = Writer::new(tmp.path(), Arc::new(AtomicU64::new(0)));
        let u1 = json!({"role": "user", "content": "hi"});
        let a1 = json!({"role": "assistant", "content": "hello"});
        let u2 = json!({"role": "user", "content": "more"});
        let mut cut = capture(&dir, "/chat/completions", json!({"model": "m", "messages": [u1]}), reply("hello"));
        cut.complete = false;
        w.handle(cut);
        w.handle(capture(&dir, "/chat/completions", json!({"model": "m", "messages": [u1, a1, u2]}), reply("x")));
        let mut other_key = capture(&dir, "/chat/completions", json!({"model": "m", "messages": [u1]}), reply("y"));
        other_key.key_id = "k2".into();
        w.handle(other_key);
        let t = turns(&dir);
        assert_eq!(t[0]["reply_hash"], Value::Null);
        assert_eq!(t[0]["response"]["complete"], false);
        assert_eq!(t[1]["session_id"], t[0]["session_id"]);
        let follow = &t[1];
        assert_eq!(follow["messages"], json!([a1, u2]));
        assert_eq!(follow["parent"]["with_reply"], false);
        assert_eq!(follow["messages_from"], 1);
        let k2 = t.iter().find(|x| x["key_id"] == "k2").unwrap();
        assert_eq!(k2["parent"], Value::Null);
    }

    #[test]
    fn deleting_a_models_traces_prunes_the_index_and_the_writer_reloads() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("traces");
        let mut w = Writer::new(tmp.path(), Arc::new(AtomicU64::new(0)));
        for (model, text) in [("org/model", "a"), ("org/model", "b"), ("other", "c")] {
            let mut c = capture(&dir, "/chat/completions", json!({"model": model, "messages": [{"role": "user", "content": text}]}), reply("x"));
            c.model = model.into();
            w.handle(c);
        }
        w.write_manifest();
        let gone = delete_model_traces_in(&dir, "org/model").unwrap().unwrap();
        assert_eq!(gone.sessions, 2);
        assert!(!dir.join("v1/org__model").exists() && dir.join("v1/other").is_dir());
        assert!(delete_model_traces_in(&dir, "org/model").unwrap().is_none());
        let sessions = std::fs::read_to_string(dir.join("sessions.jsonl")).unwrap();
        assert_eq!(sessions.lines().count(), 1);
        assert!(sessions.contains("\"other\""));
        let read = || serde_json::from_slice::<Value>(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
        assert_eq!((read()["turns"].as_u64(), read()["sessions"].as_u64()), (Some(1), Some(1)));
        let mut c = capture(&dir, "/chat/completions", json!({"model": "other", "messages": [{"role": "user", "content": "c"}]}), reply("y"));
        c.model = "other".into();
        w.handle(c);
        w.write_manifest();
        assert_eq!(read()["turns_by_model"], json!({"other": 2}));
        assert_eq!((w.totals.turns, w.totals.sessions), (2, 1));
    }

    #[test]
    fn same_messages_to_another_model_start_their_own_session() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("traces");
        let mut w = Writer::new(tmp.path(), Arc::new(AtomicU64::new(0)));
        for model in ["org/model", "other"] {
            let mut c = capture(&dir, "/chat/completions", json!({"model": model, "messages": [{"role": "user", "content": "hi"}]}), reply("x"));
            c.model = model.into();
            w.handle(c);
        }
        let t = turns(&dir);
        assert_eq!(t.len(), 2);
        assert_eq!((t[1]["parent"].clone(), t[1]["model"].clone()), (Value::Null, json!("other")));
        assert_ne!(t[0]["session_id"], t[1]["session_id"]);
        assert_eq!(w.totals.by_model.len(), 2);
    }

    #[test]
    fn size_limit_turns_recording_off_until_raised() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("traces");
        let home = tmp.path();
        db::open(home)
            .unwrap()
            .execute("INSERT INTO api_keys (key_id, name, token_hash, raw_prefix) VALUES ('k1', 'app', 'h', 'p')", [])
            .unwrap();
        set(home, Some(true), &["app".into()], false, Some(dir.to_str().unwrap()), Some(1.5e-6)).unwrap();
        let dir = std::fs::canonicalize(&dir).unwrap();
        let mut w = Writer::new(home, Arc::new(AtomicU64::new(0)));
        for _ in 0..20 {
            let mut c = capture(&dir, "/completions", json!({"model": "m", "prompt": "x"}), json!({"choices": []}));
            c.max_gb = 1.5e-6;
            w.handle(c);
        }
        assert!(w.paused.as_deref().is_some_and(|p| p.starts_with("size limit")));
        assert!(w.totals.turns >= 1 && w.totals.paused_dropped >= 1);
        assert_eq!(w.totals.turns + w.totals.paused_dropped, 20);
        let cfg = load_config(&db::open(home).unwrap());
        assert!(!cfg.on() && cfg.keys == ["k1"] && cfg.stopped.as_deref().is_some_and(|s| s.contains("limit was reached")));
        let history = std::fs::read_to_string(dir.join("history.jsonl")).unwrap();
        assert!(history.lines().last().unwrap().contains("\"reason\":\"size limit\""));
        assert!(set(home, Some(true), &[], false, None, None).unwrap_err().contains("over its"));
        set(home, None, &[], false, None, Some(1.0)).unwrap();
        assert!(!load_config(&db::open(home).unwrap()).on());
        set(home, Some(true), &[], false, None, None).unwrap();
        let cfg = load_config(&db::open(home).unwrap());
        assert!(cfg.on() && cfg.stopped.is_none() && cfg.records("k1") && !cfg.records("k2"));
    }
}
