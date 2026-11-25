// Usage log, linked or not: request counts per model, per API key and per dimension tuple, and 5 s hardware samples, as clock-minute rows (48 h) rolled into hourly rows (90 days); counts and timings only, never text; a linked node sends new minutes with its heartbeat.

use rusqlite::types::Value;
use rusqlite::{params_from_iter, Connection};
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const SAMPLE: Duration = Duration::from_secs(if cfg!(target_os = "macos") { 10 } else { 5 });
pub const BOUNDS_MS: [u32; 11] = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000, 120000];
const MINUTE_KEEP_S: i64 = 48 * 3600;
const HOUR_KEEP_S: i64 = 90 * 86400;
const SENT_KEY: &str = "usage_sent_t";
const REPORT_ROWS: usize = 400;
const MAX_PENDING: usize = 2880;
const MAX_MODEL_ID: usize = 200;
const MAX_DIM: usize = 64;
const MAX_TUPLES: usize = 512;
const OTHER: &str = "_other";
const KEY_SEP: char = '\u{1f}';
pub const RATE_LIMITED: &str = "_rate_limited";

static WRITTEN_TO: AtomicI64 = AtomicI64::new(0);

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Requests {
    pub requests: i64,
    pub ok: i64,
    pub client_errors: i64,
    pub server_errors: i64,
    pub rate_limited: i64,
    pub cancelled: i64,
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub decode_tps_sum: f64,
    pub decode_n: i64,
    pub ttft_ms_sum: i64,
    pub ttft_n: i64,
    pub latency_hist: Vec<i64>,
    pub ttft_hist: Vec<i64>,
    pub reasoning_tokens: i64,
    pub cached_tokens: i64,
    pub generation_ms_sum: i64,
    pub generation_hist: Vec<i64>,
    pub router_ms_sum: i64,
    pub router_n: i64,
    pub router_hist: Vec<i64>,
    pub total_ttft_hist: Vec<i64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sample {
    pub code: u16,
    pub cancelled: bool,
    pub duration_s: f64,
    pub ttft_s: Option<f64>,
    pub prompt: i64,
    pub completion: i64,
    pub reasoning: i64,
    pub cached: i64,
    pub router_ms: Option<f64>,
}

impl Sample {
    pub fn new(code: u16, cancelled: bool, duration_s: f64, ttft_s: Option<f64>, prompt: i64, completion: i64) -> Sample {
        Sample { code, cancelled, duration_s, ttft_s, prompt, completion, ..Sample::default() }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Dims {
    pub model: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub engine: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub variant: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub api_key_id: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub user_id: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub modality: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub network_path: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub finish_reason: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub context_length_bucket: String,
}

impl Dims {
    fn parts(&self) -> [&str; 9] {
        [
            &self.model, &self.engine, &self.variant, &self.api_key_id, &self.user_id, &self.modality, &self.network_path,
            &self.finish_reason, &self.context_length_bucket,
        ]
    }

    fn key(&self) -> String {
        // The row key: every dimension, each cut to MAX_DIM characters (the model to MAX_MODEL_ID) and stripped of the separator.
        let cut = |i: usize, v: &str| v.chars().filter(|&c| c != KEY_SEP).take(if i == 0 { MAX_MODEL_ID } else { MAX_DIM }).collect::<String>();
        self.parts().iter().enumerate().map(|(i, v)| cut(i, v)).collect::<Vec<_>>().join(&KEY_SEP.to_string())
    }

    fn from_key(k: &str) -> Dims {
        let mut p = k.split(KEY_SEP).map(str::to_string);
        let mut next = || p.next().unwrap_or_default();
        Dims {
            model: next(),
            engine: next(),
            variant: next(),
            api_key_id: next(),
            user_id: next(),
            modality: next(),
            network_path: next(),
            finish_reason: next(),
            context_length_bucket: next(),
        }
    }

    fn other(&self) -> Dims {
        // The tuple a minute's overflow counts into: the model kept, every other dimension "_other".
        let o = || OTHER.to_string();
        Dims {
            model: self.model.clone(),
            engine: o(),
            variant: o(),
            api_key_id: o(),
            user_id: o(),
            modality: o(),
            network_path: o(),
            finish_reason: o(),
            context_length_bucket: o(),
        }
    }
}

pub fn context_bucket(prompt: i64) -> &'static str {
    // Prompt size tier by token count: up to 1K, 10K, 32K, 128K, then above.
    match prompt {
        i64::MIN..=1024 => "1k",
        1025..=10240 => "10k",
        10241..=32768 => "32k",
        32769..=131072 => "128k",
        _ => "128k+",
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Gpu {
    pub samples: i64,
    pub util_pct: Option<f64>,
    pub util_max_pct: Option<f64>,
    pub membw_pct: Option<f64>,
    pub membw_max_pct: Option<f64>,
    pub vram_used_mb: Option<f64>,
    pub vram_used_max_mb: Option<f64>,
    pub vram_goethe_mb: Option<f64>,
    pub vram_foreign_mb: Option<f64>,
    pub power_w: Option<f64>,
    pub power_max_w: Option<f64>,
    pub power_limit_w: Option<f64>,
    pub temp_c: Option<f64>,
    pub temp_max_c: Option<f64>,
    pub throttle: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Host {
    pub samples: i64,
    pub cpu_pct: Option<f64>,
    pub cpu_max_pct: Option<f64>,
    pub ram_used_mb: Option<f64>,
    pub ram_used_max_mb: Option<f64>,
    pub ram_total_mb: Option<f64>,
    pub disk_free_mb: Option<f64>,
    pub mem_pressure: Option<f64>,
}

fn wmean(a: Option<f64>, an: i64, b: Option<f64>, bn: i64) -> Option<f64> {
    // The mean of two means weighted by their sample counts; a side without a value does not count.
    match (a, b) {
        (Some(x), Some(y)) if an + bn > 0 => Some((x * an as f64 + y * bn as f64) / (an + bn) as f64),
        (Some(x), _) => Some(x),
        (_, y) => y,
    }
}

fn pick(a: Option<f64>, b: Option<f64>, f: fn(f64, f64) -> f64) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(f(x, y)),
        (x, y) => x.or(y),
    }
}

fn add_hist(a: &mut Vec<i64>, b: &[i64]) {
    if a.len() < b.len() {
        a.resize(b.len(), 0);
    }
    a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
}

fn bucket(ms: f64) -> usize {
    BOUNDS_MS.iter().position(|&b| ms <= b as f64).unwrap_or(BOUNDS_MS.len())
}

fn hit(h: &mut Vec<i64>, ms: f64) {
    let i = bucket(ms);
    if h.len() <= i {
        h.resize(i + 1, 0);
    }
    h[i] += 1;
}

impl Requests {
    pub fn add(&mut self, s: &Sample) {
        // One finished request; decode speed and generation time are the tokens and time after the first token (the gateway streams every generation request from the engine); the router hop counts only for requests goethe-server routed.
        let Sample { code, cancelled, duration_s, ttft_s, prompt, completion, reasoning, cached, router_ms } = *s;
        self.requests += 1;
        match code {
            _ if cancelled => self.cancelled += 1,
            200..=399 => self.ok += 1,
            429 => self.rate_limited += 1,
            400..=499 => self.client_errors += 1,
            _ => self.server_errors += 1,
        }
        self.prompt_tokens += prompt.max(0);
        self.completion_tokens += completion.max(0);
        self.reasoning_tokens += reasoning.max(0);
        self.cached_tokens += cached.max(0);
        hit(&mut self.latency_hist, duration_s * 1000.0);
        let router = router_ms.filter(|r| r.is_finite() && *r >= 0.0);
        if let Some(r) = router {
            self.router_n += 1;
            self.router_ms_sum += r.round() as i64;
            hit(&mut self.router_hist, r);
        }
        if let Some(t) = ttft_s.filter(|t| t.is_finite() && *t >= 0.0) {
            self.ttft_n += 1;
            self.ttft_ms_sum += (t * 1000.0).round() as i64;
            hit(&mut self.ttft_hist, t * 1000.0);
            hit(&mut self.total_ttft_hist, t * 1000.0 + router.unwrap_or(0.0));
            let decode = duration_s - t;
            if decode >= 0.0 {
                self.generation_ms_sum += (decode * 1000.0).round() as i64;
                hit(&mut self.generation_hist, decode * 1000.0);
            }
            if completion >= 2 && decode > 0.001 {
                self.decode_tps_sum += (completion - 1) as f64 / decode;
                self.decode_n += 1;
            }
        }
    }
}

pub(crate) trait Row: Default + Clone {
    const KIND: &'static str;
    const COLS: &'static [&'static str];
    fn values(&self) -> Vec<Value>;
    fn from_values(v: &[Value]) -> Self;
    fn merge(&mut self, o: &Self);
}

fn int(v: i64) -> Value {
    Value::Integer(v)
}

fn real(v: Option<f64>) -> Value {
    // Stored to one decimal, which is all a chart needs.
    v.filter(|x| x.is_finite()).map_or(Value::Null, |x| Value::Real((x * 10.0).round() / 10.0))
}

fn hist(h: &[i64]) -> Value {
    let end = h.iter().rposition(|&x| x != 0).map_or(0, |i| i + 1);
    Value::Text(h[..end].iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
}

fn get_i(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        Value::Real(f) => *f as i64,
        _ => 0,
    }
}

fn get_f(v: &Value) -> Option<f64> {
    match v {
        Value::Integer(i) => Some(*i as f64),
        Value::Real(f) => Some(*f),
        _ => None,
    }
}

fn get_hist(v: &Value) -> Vec<i64> {
    match v {
        Value::Text(t) => t.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        _ => Vec::new(),
    }
}

impl Row for Requests {
    const KIND: &'static str = "usage";
    const COLS: &'static [&'static str] = &[
        "requests", "ok", "client_errors", "server_errors", "rate_limited", "cancelled", "prompt_tokens", "completion_tokens",
        "decode_tps_sum", "decode_n", "ttft_ms_sum", "ttft_n", "latency_hist", "ttft_hist", "reasoning_tokens", "cached_tokens",
        "generation_ms_sum", "generation_hist", "router_ms_sum", "router_n", "router_hist", "total_ttft_hist",
    ];
    fn values(&self) -> Vec<Value> {
        vec![
            int(self.requests),
            int(self.ok),
            int(self.client_errors),
            int(self.server_errors),
            int(self.rate_limited),
            int(self.cancelled),
            int(self.prompt_tokens),
            int(self.completion_tokens),
            real(Some(self.decode_tps_sum)),
            int(self.decode_n),
            int(self.ttft_ms_sum),
            int(self.ttft_n),
            hist(&self.latency_hist),
            hist(&self.ttft_hist),
            int(self.reasoning_tokens),
            int(self.cached_tokens),
            int(self.generation_ms_sum),
            hist(&self.generation_hist),
            int(self.router_ms_sum),
            int(self.router_n),
            hist(&self.router_hist),
            hist(&self.total_ttft_hist),
        ]
    }
    fn from_values(v: &[Value]) -> Self {
        Requests {
            requests: get_i(&v[0]),
            ok: get_i(&v[1]),
            client_errors: get_i(&v[2]),
            server_errors: get_i(&v[3]),
            rate_limited: get_i(&v[4]),
            cancelled: get_i(&v[5]),
            prompt_tokens: get_i(&v[6]),
            completion_tokens: get_i(&v[7]),
            decode_tps_sum: get_f(&v[8]).unwrap_or(0.0),
            decode_n: get_i(&v[9]),
            ttft_ms_sum: get_i(&v[10]),
            ttft_n: get_i(&v[11]),
            latency_hist: get_hist(&v[12]),
            ttft_hist: get_hist(&v[13]),
            reasoning_tokens: get_i(&v[14]),
            cached_tokens: get_i(&v[15]),
            generation_ms_sum: get_i(&v[16]),
            generation_hist: get_hist(&v[17]),
            router_ms_sum: get_i(&v[18]),
            router_n: get_i(&v[19]),
            router_hist: get_hist(&v[20]),
            total_ttft_hist: get_hist(&v[21]),
        }
    }
    fn merge(&mut self, o: &Self) {
        self.requests += o.requests;
        self.ok += o.ok;
        self.client_errors += o.client_errors;
        self.server_errors += o.server_errors;
        self.rate_limited += o.rate_limited;
        self.cancelled += o.cancelled;
        self.prompt_tokens += o.prompt_tokens;
        self.completion_tokens += o.completion_tokens;
        self.decode_tps_sum += o.decode_tps_sum;
        self.decode_n += o.decode_n;
        self.ttft_ms_sum += o.ttft_ms_sum;
        self.ttft_n += o.ttft_n;
        add_hist(&mut self.latency_hist, &o.latency_hist);
        add_hist(&mut self.ttft_hist, &o.ttft_hist);
        self.reasoning_tokens += o.reasoning_tokens;
        self.cached_tokens += o.cached_tokens;
        self.generation_ms_sum += o.generation_ms_sum;
        add_hist(&mut self.generation_hist, &o.generation_hist);
        self.router_ms_sum += o.router_ms_sum;
        self.router_n += o.router_n;
        add_hist(&mut self.router_hist, &o.router_hist);
        add_hist(&mut self.total_ttft_hist, &o.total_ttft_hist);
    }
}

impl Row for Gpu {
    const KIND: &'static str = "gpu";
    const COLS: &'static [&'static str] = &[
        "samples", "util_pct", "util_max_pct", "membw_pct", "membw_max_pct", "vram_used_mb", "vram_used_max_mb", "vram_goethe_mb",
        "vram_foreign_mb", "power_w", "power_max_w", "power_limit_w", "temp_c", "temp_max_c", "throttle",
    ];
    fn values(&self) -> Vec<Value> {
        vec![
            int(self.samples),
            real(self.util_pct),
            real(self.util_max_pct),
            real(self.membw_pct),
            real(self.membw_max_pct),
            real(self.vram_used_mb),
            real(self.vram_used_max_mb),
            real(self.vram_goethe_mb),
            real(self.vram_foreign_mb),
            real(self.power_w),
            real(self.power_max_w),
            real(self.power_limit_w),
            real(self.temp_c),
            real(self.temp_max_c),
            int(self.throttle),
        ]
    }
    fn from_values(v: &[Value]) -> Self {
        Gpu {
            samples: get_i(&v[0]),
            util_pct: get_f(&v[1]),
            util_max_pct: get_f(&v[2]),
            membw_pct: get_f(&v[3]),
            membw_max_pct: get_f(&v[4]),
            vram_used_mb: get_f(&v[5]),
            vram_used_max_mb: get_f(&v[6]),
            vram_goethe_mb: get_f(&v[7]),
            vram_foreign_mb: get_f(&v[8]),
            power_w: get_f(&v[9]),
            power_max_w: get_f(&v[10]),
            power_limit_w: get_f(&v[11]),
            temp_c: get_f(&v[12]),
            temp_max_c: get_f(&v[13]),
            throttle: get_i(&v[14]),
        }
    }
    fn merge(&mut self, o: &Self) {
        let (a, b) = (self.samples, o.samples);
        self.util_pct = wmean(self.util_pct, a, o.util_pct, b);
        self.util_max_pct = pick(self.util_max_pct, o.util_max_pct, f64::max);
        self.membw_pct = wmean(self.membw_pct, a, o.membw_pct, b);
        self.membw_max_pct = pick(self.membw_max_pct, o.membw_max_pct, f64::max);
        self.vram_used_mb = wmean(self.vram_used_mb, a, o.vram_used_mb, b);
        self.vram_used_max_mb = pick(self.vram_used_max_mb, o.vram_used_max_mb, f64::max);
        self.vram_goethe_mb = wmean(self.vram_goethe_mb, a, o.vram_goethe_mb, b);
        self.vram_foreign_mb = wmean(self.vram_foreign_mb, a, o.vram_foreign_mb, b);
        self.power_w = wmean(self.power_w, a, o.power_w, b);
        self.power_max_w = pick(self.power_max_w, o.power_max_w, f64::max);
        self.power_limit_w = o.power_limit_w.or(self.power_limit_w);
        self.temp_c = wmean(self.temp_c, a, o.temp_c, b);
        self.temp_max_c = pick(self.temp_max_c, o.temp_max_c, f64::max);
        self.throttle |= o.throttle;
        self.samples = a + b;
    }
}

impl Row for Host {
    const KIND: &'static str = "host";
    const COLS: &'static [&'static str] =
        &["samples", "cpu_pct", "cpu_max_pct", "ram_used_mb", "ram_used_max_mb", "ram_total_mb", "disk_free_mb", "mem_pressure"];
    fn values(&self) -> Vec<Value> {
        vec![
            int(self.samples),
            real(self.cpu_pct),
            real(self.cpu_max_pct),
            real(self.ram_used_mb),
            real(self.ram_used_max_mb),
            real(self.ram_total_mb),
            real(self.disk_free_mb),
            real(self.mem_pressure),
        ]
    }
    fn from_values(v: &[Value]) -> Self {
        Host {
            samples: get_i(&v[0]),
            cpu_pct: get_f(&v[1]),
            cpu_max_pct: get_f(&v[2]),
            ram_used_mb: get_f(&v[3]),
            ram_used_max_mb: get_f(&v[4]),
            ram_total_mb: get_f(&v[5]),
            disk_free_mb: get_f(&v[6]),
            mem_pressure: get_f(&v[7]),
        }
    }
    fn merge(&mut self, o: &Self) {
        let (a, b) = (self.samples, o.samples);
        self.cpu_pct = wmean(self.cpu_pct, a, o.cpu_pct, b);
        self.cpu_max_pct = pick(self.cpu_max_pct, o.cpu_max_pct, f64::max);
        self.ram_used_mb = wmean(self.ram_used_mb, a, o.ram_used_mb, b);
        self.ram_used_max_mb = pick(self.ram_used_max_mb, o.ram_used_max_mb, f64::max);
        self.ram_total_mb = pick(self.ram_total_mb, o.ram_total_mb, f64::max);
        self.disk_free_mb = pick(self.disk_free_mb, o.disk_free_mb, f64::min);
        self.mem_pressure = pick(self.mem_pressure, o.mem_pressure, f64::max);
        self.samples = a + b;
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeyRequests(pub Requests);

impl Row for KeyRequests {
    const KIND: &'static str = "keyuse";
    const COLS: &'static [&'static str] = Requests::COLS;
    fn values(&self) -> Vec<Value> {
        self.0.values()
    }
    fn from_values(v: &[Value]) -> Self {
        KeyRequests(Requests::from_values(v))
    }
    fn merge(&mut self, o: &Self) {
        self.0.merge(&o.0);
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Traffic(pub Requests);

impl Row for Traffic {
    const KIND: &'static str = "req";
    const COLS: &'static [&'static str] = Requests::COLS;
    fn values(&self) -> Vec<Value> {
        self.0.values()
    }
    fn from_values(v: &[Value]) -> Self {
        Traffic(Requests::from_values(v))
    }
    fn merge(&mut self, o: &Self) {
        self.0.merge(&o.0);
    }
}

fn ddl<R: Row>() -> String {
    // One minute and one hour table per kind, keyed by the bucket start (unix seconds) and the model id, GPU index or "" for the host.
    ["minute", "hour"]
        .iter()
        .map(|g| format!("CREATE TABLE IF NOT EXISTS {}_{g} (t INTEGER NOT NULL, k TEXT NOT NULL, {}, PRIMARY KEY (t, k)) WITHOUT ROWID;\n", R::KIND, R::COLS.join(", ")))
        .collect()
}

fn add_columns<R: Row>(con: &Connection) -> rusqlite::Result<()> {
    // Columns a newer goethe added to an existing table; old rows read them as 0 or empty, and an older goethe never selects them.
    for g in ["minute", "hour"] {
        let table = format!("{}_{g}", R::KIND);
        let have: HashSet<String> =
            con.prepare(&format!("PRAGMA table_info({table})"))?.query_map([], |r| r.get::<_, String>(1))?.collect::<rusqlite::Result<_>>()?;
        for c in R::COLS.iter().filter(|c| !have.contains(**c)) {
            add_column(con, &table, c)?;
        }
    }
    Ok(())
}

fn add_column(con: &Connection, table: &str, column: &str) -> rusqlite::Result<()> {
    // Another process (goethed and the CLI after an upgrade) may add the same column first; that is not an error.
    con.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column};")).or_else(|e| {
        let added = con.prepare(&format!("SELECT {column} FROM {table} LIMIT 0")).is_ok();
        if added { Ok(()) } else { Err(e) }
    })
}

pub fn ensure_tables(con: &Connection) -> rusqlite::Result<()> {
    // New tables and columns need no schema version bump: an older goethe opens the database and never looks at them.
    con.execute_batch(&format!("{}{}{}{}{}", ddl::<Requests>(), ddl::<KeyRequests>(), ddl::<Traffic>(), ddl::<Gpu>(), ddl::<Host>()))?;
    add_columns::<Requests>(con)?;
    add_columns::<KeyRequests>(con)?;
    add_columns::<Traffic>(con)
}

fn select<R: Row>(con: &Connection, table: &str, filter: &str, args: &[Value]) -> rusqlite::Result<Vec<(i64, String, R)>> {
    let sql = format!("SELECT t, k, {} FROM {}_{table} {filter} ORDER BY t, k", R::COLS.join(", "), R::KIND);
    let mut st = con.prepare_cached(&sql)?;
    let n = R::COLS.len();
    let rows = st.query_map(params_from_iter(args.iter()), |r| {
        let vals: Vec<Value> = (0..n).map(|i| r.get::<_, Value>(i + 2)).collect::<rusqlite::Result<_>>()?;
        Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, R::from_values(&vals)))
    })?;
    rows.collect()
}

fn put<R: Row>(con: &Connection, table: &str, t: i64, k: &str, row: &R, merge: bool) -> rusqlite::Result<()> {
    // Write one row; merging adds to a row already there (a restart inside the same minute writes it twice).
    let mut row = row.clone();
    if merge {
        if let Some((_, _, old)) = select::<R>(con, table, "WHERE t = ?1 AND k = ?2", &[int(t), Value::Text(k.to_string())])?.into_iter().next() {
            let mut old = old;
            old.merge(&row);
            row = old;
        }
    }
    let marks = (0..R::COLS.len() + 2).map(|i| format!("?{}", i + 1)).collect::<Vec<_>>().join(", ");
    let sql = format!("INSERT OR REPLACE INTO {}_{table} (t, k, {}) VALUES ({marks})", R::KIND, R::COLS.join(", "));
    let mut vals = vec![int(t), Value::Text(k.to_string())];
    vals.extend(row.values());
    con.prepare_cached(&sql)?.execute(params_from_iter(vals.iter()))?;
    Ok(())
}

#[derive(Clone, Debug, Default)]
struct Minute {
    t: i64,
    models: BTreeMap<String, Requests>,
    keys: BTreeMap<String, KeyRequests>,
    tuples: BTreeMap<String, Traffic>,
    gpus: BTreeMap<u32, Gpu>,
    host: Host,
}

impl Minute {
    fn count(&mut self, d: &Dims, s: &Sample) {
        let model: String = d.model.chars().take(MAX_MODEL_ID).collect();
        self.models.entry(model.clone()).or_default().add(s);
        if !d.api_key_id.is_empty() {
            self.keys.entry(format!("{}{KEY_SEP}{model}", d.api_key_id)).or_default().0.add(s);
        }
        let mut k = d.key();
        if !self.tuples.contains_key(&k) && self.tuples.len() >= MAX_TUPLES {
            k = d.other().key();
        }
        self.tuples.entry(k).or_default().0.add(s);
    }

    fn is_empty(&self) -> bool {
        self.models.is_empty() && self.keys.is_empty() && self.tuples.is_empty() && self.gpus.is_empty() && self.host.samples == 0
    }
}

#[derive(Default)]
struct Live {
    cur: Minute,
    done: Vec<Minute>,
}

fn live() -> MutexGuard<'static, Live> {
    static LIVE: OnceLock<Mutex<Live>> = OnceLock::new();
    LIVE.get_or_init(|| Mutex::new(Live::default())).lock().unwrap_or_else(|e| e.into_inner())
}

fn now_s() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

fn minute_of(t: i64) -> i64 {
    t - t.rem_euclid(60)
}

fn hour_of(t: i64) -> i64 {
    t - t.rem_euclid(3600)
}

fn rotate(l: &mut Live, minute: i64) {
    // A later clock minute: the finished one waits for the next write, an empty one is dropped. An earlier one (a request that read the clock just before the sampler rotated, or the clock stepped back) counts into the current minute, so a minute already written is never reopened.
    if minute <= l.cur.t {
        return;
    }
    let old = std::mem::replace(&mut l.cur, Minute { t: minute, ..Minute::default() });
    if !old.is_empty() {
        l.done.push(old);
        if l.done.len() > MAX_PENDING {
            l.done.remove(0);
        }
    }
}

pub fn record(d: &Dims, s: &Sample) {
    // Counted into the minute the request finished in, per model, per API key and model, and per dimension tuple (past MAX_TUPLES tuples in a minute, into the model's "_other" tuple); requests without a model (the model list) are not usage.
    if d.model.is_empty() {
        return;
    }
    let mut l = live();
    rotate(&mut l, minute_of(now_s()));
    l.cur.count(d, s);
}

fn take_finished(now: i64, all: bool) -> (Vec<Minute>, i64) {
    // The minutes ready to write and the start of the first minute that is not; `all` also takes the current one (shutdown).
    let mut l = live();
    rotate(&mut l, minute_of(now));
    let mut out = std::mem::take(&mut l.done);
    let mut upto = l.cur.t;
    if all {
        let t = l.cur.t;
        let cur = std::mem::replace(&mut l.cur, Minute { t, ..Minute::default() });
        if !cur.is_empty() {
            out.push(cur);
        }
        upto = t + 60;
    }
    (out, upto)
}

fn requeue(minutes: Vec<Minute>) {
    // A failed write keeps the minutes for the next try.
    let mut l = live();
    let mut rest = std::mem::take(&mut l.done);
    l.done = minutes;
    l.done.append(&mut rest);
    let over = l.done.len().saturating_sub(MAX_PENDING);
    l.done.drain(..over);
}

fn immediate(con: &Connection, f: impl FnOnce() -> rusqlite::Result<()>) -> rusqlite::Result<()> {
    // One write transaction: all of it lands or none (a crash mid-rollup must not leave an hour with only some of its rows).
    con.execute_batch("BEGIN IMMEDIATE;")?;
    match f() {
        Ok(()) => con.execute_batch("COMMIT;"),
        Err(e) => {
            let _ = con.execute_batch("ROLLBACK;");
            Err(e)
        }
    }
}

fn write(con: &Connection, minutes: &[Minute]) -> rusqlite::Result<()> {
    immediate(con, || {
        for m in minutes {
            for (model, r) in &m.models {
                put(con, "minute", m.t, model, r, true)?;
            }
            for (k, r) in &m.keys {
                put(con, "minute", m.t, k, r, true)?;
            }
            for (k, r) in &m.tuples {
                put(con, "minute", m.t, k, r, true)?;
            }
            for (i, g) in &m.gpus {
                put(con, "minute", m.t, &i.to_string(), g, true)?;
            }
            if m.host.samples > 0 {
                put(con, "minute", m.t, "", &m.host, true)?;
            }
        }
        Ok(())
    })
}

fn roll_kind<R: Row>(con: &Connection, before: i64, now: i64) -> rusqlite::Result<()> {
    // Hours before `before` that have minute rows but no hour row yet, and always the hour just finished; never a partly expired hour.
    let mut hours: Vec<i64> = con
        .prepare_cached(&format!(
            "SELECT DISTINCT t - t % 3600 AS h FROM {k}_minute WHERE t < ?1 AND t - t % 3600 >= ?2 AND h NOT IN (SELECT t FROM {k}_hour)",
            k = R::KIND
        ))?
        .query_map([before, hour_of(now - MINUTE_KEEP_S) + 3600], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    if !hours.contains(&(before - 3600)) {
        hours.push(before - 3600);
    }
    for h in hours {
        let mut sum: BTreeMap<String, R> = BTreeMap::new();
        for (_, k, r) in select::<R>(con, "minute", "WHERE t >= ?1 AND t < ?2", &[int(h), int(h + 3600)])? {
            sum.entry(k).or_default().merge(&r);
        }
        for (k, r) in sum {
            put(con, "hour", h, &k, &r, false)?;
        }
    }
    con.execute(&format!("DELETE FROM {}_minute WHERE t < ?1", R::KIND), [now - MINUTE_KEEP_S])?;
    con.execute(&format!("DELETE FROM {}_hour WHERE t < ?1", R::KIND), [now - HOUR_KEEP_S])?;
    Ok(())
}

fn roll(con: &Connection, now: i64) -> rusqlite::Result<()> {
    let before = hour_of(now);
    immediate(con, || {
        roll_kind::<Requests>(con, before, now)?;
        roll_kind::<KeyRequests>(con, before, now)?;
        roll_kind::<Traffic>(con, before, now)?;
        roll_kind::<Gpu>(con, before, now)?;
        roll_kind::<Host>(con, before, now)
    })
}

struct Sampler {
    home: PathBuf,
    con: Option<Connection>,
    cpu: Option<Vec<u64>>,
    rolled: i64,
}

impl Sampler {
    fn con(&mut self) -> Option<&Connection> {
        if self.con.is_none() {
            self.con = crate::db::open(&self.home).ok();
        }
        self.con.as_ref()
    }

    fn sample(&mut self) -> (Vec<(u32, Gpu)>, Host) {
        let managed: HashSet<i32> = self.con().map(crate::placement::managed_pids).unwrap_or_default();
        let gpus = crate::hw::gpu_usage_shared(&managed).unwrap_or_default().iter().map(|u| (u.index, gpu_sample(u))).collect();
        let cpu = crate::sys::cpu_times().map(|(total, _)| total);
        let cpu_pct = match (&self.cpu, &cpu) {
            (Some(a), Some(b)) => busy_pct(a, b),
            _ => None,
        };
        self.cpu = cpu;
        let mem = crate::sys::memory();
        let ram_total = (mem.total_kb > 0).then(|| mem.total_kb as f64 / 1024.0);
        let ram_used = ram_total.map(|t| t - mem.available_kb as f64 / 1024.0);
        let models = self.home.join("models");
        let disk = crate::hf::available_bytes(if models.exists() { &models } else { &self.home }).ok().map(|b| b as f64 / (1024.0 * 1024.0));
        let host = Host {
            samples: 1,
            cpu_pct,
            cpu_max_pct: cpu_pct,
            ram_used_mb: ram_used,
            ram_used_max_mb: ram_used,
            ram_total_mb: ram_total,
            disk_free_mb: disk,
            mem_pressure: mem_pressure(),
        };
        (gpus, host)
    }

    fn tick(&mut self, all: bool) {
        let now = now_s();
        if !all {
            let (gpus, host) = self.sample();
            let mut l = live();
            rotate(&mut l, minute_of(now));
            for (i, g) in gpus {
                l.cur.gpus.entry(i).or_default().merge(&g);
            }
            l.cur.host.merge(&host);
        }
        let (minutes, upto) = take_finished(now, all);
        let rolled = self.rolled;
        let Some(con) = self.con() else {
            requeue(minutes);
            return;
        };
        if let Err(e) = write(con, &minutes) {
            requeue(minutes);
            self.con = None;
            eprintln!("usage log: {e}");
            return;
        }
        WRITTEN_TO.store(upto, Ordering::Relaxed);
        if hour_of(now) > rolled {
            match roll(con, now) {
                Ok(()) => self.rolled = hour_of(now),
                Err(e) => eprintln!("usage log rollup: {e}"),
            }
        }
    }
}

fn gpu_sample(u: &crate::hw::GpuUsage) -> Gpu {
    // One sample; on a Mac there is no memory-bandwidth figure (memory_util there is the GPU's busy %), power or temperature.
    let f = |v: Option<u32>| v.map(f64::from);
    let watts = |mw: Option<u32>| mw.map(|m| m as f64 / 1000.0);
    let used = Some(u.used_mb as f64);
    Gpu {
        samples: 1,
        util_pct: f(u.gpu_util),
        util_max_pct: f(u.gpu_util),
        membw_pct: if cfg!(target_os = "macos") { None } else { f(u.memory_util) },
        membw_max_pct: if cfg!(target_os = "macos") { None } else { f(u.memory_util) },
        vram_used_mb: used,
        vram_used_max_mb: used,
        vram_goethe_mb: Some(u.managed_mb as f64),
        vram_foreign_mb: Some(u.used_mb.saturating_sub(u.managed_mb) as f64),
        power_w: watts(u.power_mw),
        power_max_w: watts(u.power_mw),
        power_limit_w: watts(u.power_limit_mw),
        temp_c: f(u.temp_c),
        temp_max_c: f(u.temp_c),
        throttle: u.throttle as i64,
    }
}

fn busy_pct(before: &[u64], after: &[u64]) -> Option<f64> {
    // CPU busy % between two /proc/stat style samples: everything but idle and iowait.
    let total: u64 = after.iter().zip(before).map(|(a, b)| a.saturating_sub(*b)).sum();
    let idle: u64 = [3, 4].iter().filter_map(|&i| Some(after.get(i)?.saturating_sub(*before.get(i)?))).sum();
    (total > 0).then(|| (total - idle.min(total)) as f64 * 100.0 / total as f64)
}

fn mem_pressure() -> Option<f64> {
    // macOS memory pressure level: 1 normal, 2 warning, 4 critical. Linux has no single equivalent and leaves it empty.
    #[cfg(target_os = "macos")]
    return crate::sys::sysctl_u64("kern.memorystatus_vm_pressure_level").map(|v| v as f64);
    #[cfg(not(target_os = "macos"))]
    None
}

pub fn spawn(home: PathBuf) {
    // A plain thread, off the async runtime: sample on the SAMPLE grid (5 s, 10 s on a Mac where it runs ioreg), write finished minutes, roll up and expire once an hour; a panic is logged once and sampling goes on.
    let _ = std::thread::Builder::new().name("usage-log".into()).spawn(move || {
        let mut s = Sampler { home, con: None, cpu: None, rolled: 0 };
        let mut noted = false;
        loop {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
            let step = SAMPLE.as_millis() as u64;
            std::thread::sleep(Duration::from_millis(step - (now.as_millis() as u64 % step)).max(Duration::from_millis(50)));
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.tick(false))).is_err() {
                s.con = None;
                if !noted {
                    crate::supervisor::append_goethed_log(&s.home, "usage log: a sample failed; sampling goes on\n");
                    noted = true;
                }
            }
        }
    });
}

pub fn flush(home: &Path) {
    // At shutdown: write everything counted so far, including the minute in progress.
    let mut s = Sampler { home: home.to_path_buf(), con: None, cpu: None, rolled: i64::MAX };
    s.tick(true);
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelRow {
    pub t: i64,
    pub model: String,
    #[serde(flatten)]
    pub stats: Requests,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TupleRow {
    pub t: i64,
    #[serde(flatten)]
    pub dims: Dims,
    #[serde(flatten)]
    pub stats: Requests,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GpuRow {
    pub t: i64,
    pub gpu: u32,
    #[serde(flatten)]
    pub stats: Gpu,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HostRow {
    pub t: i64,
    #[serde(flatten)]
    pub stats: Host,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Report {
    pub from: i64,
    pub to: i64,
    pub bounds_ms: Vec<u32>,
    pub models: Vec<ModelRow>,
    pub requests: Vec<TupleRow>,
    pub gpus: Vec<GpuRow>,
    pub host: Vec<HostRow>,
}

impl Report {
    pub fn halve(&mut self) -> bool {
        // Keep the older half of the minutes, so a report too big for one heartbeat still moves the send cursor; false when one minute is all it holds.
        let mut ts: Vec<i64> = self
            .models
            .iter()
            .map(|r| r.t)
            .chain(self.requests.iter().map(|r| r.t))
            .chain(self.gpus.iter().map(|r| r.t))
            .chain(self.host.iter().map(|r| r.t))
            .collect();
        ts.sort_unstable();
        ts.dedup();
        if ts.len() < 2 {
            return false;
        }
        let mid = ts[ts.len() / 2];
        self.models.retain(|r| r.t < mid);
        self.requests.retain(|r| r.t < mid);
        self.gpus.retain(|r| r.t < mid);
        self.host.retain(|r| r.t < mid);
        self.to = mid;
        true
    }
}

fn sent(con: &Connection) -> Option<i64> {
    crate::db::setting(con, SENT_KEY).and_then(|v| v.parse().ok())
}

pub fn report(con: &Connection) -> Option<Report> {
    // In goethed, up to what the sampler has written; elsewhere (goethe doctor --snapshot) up to the minute before the last.
    let now = now_s();
    let written = WRITTEN_TO.load(Ordering::Relaxed);
    report_at(con, now, if written > 0 { written } else { minute_of(now) - 60 })
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Totals {
    pub models: Vec<(String, Requests)>,
    pub keys: Vec<(String, String, Requests)>,
    pub gpus: Vec<(u32, Gpu)>,
    pub host: Host,
}

fn sum_into<R: Row>(con: &Connection, since: i64, out: &mut BTreeMap<String, R>) -> rusqlite::Result<()> {
    // Hourly rows for the part older than the minute rows, minute rows for the rest; an hour only counts once.
    let mut minutes_from = since;
    if since < now_s() - MINUTE_KEEP_S + 3600 {
        let hours = select::<R>(con, "hour", "WHERE t >= ?1", &[int(hour_of(since))])?;
        minutes_from = hours.iter().map(|(t, _, _)| t + 3600).max().unwrap_or(since).max(since);
        for (_, k, r) in hours {
            out.entry(k).or_default().merge(&r);
        }
    }
    for (_, k, r) in select::<R>(con, "minute", "WHERE t >= ?1", &[int(minutes_from)])? {
        out.entry(k).or_default().merge(&r);
    }
    Ok(())
}

pub fn totals(con: &Connection, since: i64) -> rusqlite::Result<Totals> {
    // Everything since a time, per model, per GPU and for the host, from the written rows (up to a minute behind).
    let (mut m, mut g, mut h) = (BTreeMap::<String, Requests>::new(), BTreeMap::<String, Gpu>::new(), BTreeMap::<String, Host>::new());
    let mut k = BTreeMap::<String, KeyRequests>::new();
    sum_into(con, since, &mut m)?;
    sum_into(con, since, &mut k)?;
    sum_into(con, since, &mut g)?;
    sum_into(con, since, &mut h)?;
    let mut gpus: Vec<(u32, Gpu)> = g.into_iter().filter_map(|(k, v)| Some((k.parse().ok()?, v))).collect();
    gpus.sort_by_key(|(i, _)| *i);
    let keys = k.into_iter().filter_map(|(k, r)| k.split_once(KEY_SEP).map(|(id, model)| (id.to_string(), model.to_string(), r.0))).collect();
    Ok(Totals { models: m.into_iter().collect(), keys, gpus, host: h.remove("").unwrap_or_default() })
}

pub fn p95_ms(hist: &[i64]) -> Option<u32> {
    // The upper bound of the bucket that holds the 95th percentile; None past the last bound or without data.
    let n: i64 = hist.iter().sum();
    if n == 0 {
        return None;
    }
    let mut seen = 0;
    for (i, c) in hist.iter().enumerate() {
        seen += c;
        if seen * 100 >= n * 95 {
            return BOUNDS_MS.get(i).copied();
        }
    }
    None
}

pub fn throttle_words(bits: i64) -> Vec<&'static str> {
    // NVML clock-throttle bits as words: power (software cap, power brake), thermal (software or hardware), slowdown, sync boost.
    let mut out = Vec::new();
    if bits & (0x04 | 0x80) != 0 {
        out.push("power");
    }
    if bits & (0x20 | 0x40) != 0 {
        out.push("thermal");
    }
    if bits & 0x08 != 0 {
        out.push("slowdown");
    }
    if bits & 0x10 != 0 {
        out.push("sync boost");
    }
    out
}

#[cfg(test)]
pub(crate) fn put_minute_for_test(con: &Connection, t: i64, model: &str, completion: i64) {
    let mut r = Requests::default();
    r.add(&Sample::new(200, false, 1.0, Some(0.1), 1, completion));
    put(con, "minute", t, model, &r, true).unwrap();
    put(con, "minute", t, &Dims { model: model.into(), modality: "chat".into(), ..Dims::default() }.key(), &Traffic(r), true).unwrap();
}

fn report_at(con: &Connection, now: i64, written_to: i64) -> Option<Report> {
    // Written minutes after the last confirmed send (at most 48 h back), oldest first, about REPORT_ROWS rows; a covered minute without a model row had no requests.
    let from = sent(con).unwrap_or(0).max(minute_of(now - MINUTE_KEEP_S));
    let end = written_to.min(minute_of(now));
    if from >= end {
        return None;
    }
    let mut per_t: BTreeMap<i64, usize> = BTreeMap::new();
    for kind in [Requests::KIND, Traffic::KIND, Gpu::KIND, Host::KIND] {
        let mut st = con.prepare_cached(&format!("SELECT t, COUNT(*) FROM {kind}_minute WHERE t >= ?1 AND t < ?2 GROUP BY t")).ok()?;
        let rows = st.query_map([from, end], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)? as usize))).ok()?;
        for (t, n) in rows.flatten() {
            per_t.entry(t).and_modify(|c| *c += n).or_insert(n);
        }
    }
    let mut to = end;
    let mut rows = 0;
    for (&t, &n) in &per_t {
        if rows > 0 && rows + n > REPORT_ROWS {
            to = t;
            break;
        }
        rows += n;
    }
    let range = [int(from), int(to)];
    let filter = "WHERE t >= ?1 AND t < ?2";
    Some(Report {
        from,
        to,
        bounds_ms: BOUNDS_MS.to_vec(),
        models: select::<Requests>(con, "minute", filter, &range).ok()?.into_iter().map(|(t, model, stats)| ModelRow { t, model, stats }).collect(),
        requests: select::<Traffic>(con, "minute", filter, &range)
            .ok()?
            .into_iter()
            .map(|(t, k, stats)| TupleRow { t, dims: Dims::from_key(&k), stats: stats.0 })
            .collect(),
        gpus: select::<Gpu>(con, "minute", filter, &range)
            .ok()?
            .into_iter()
            .filter_map(|(t, k, stats)| Some(GpuRow { t, gpu: k.parse().ok()?, stats }))
            .collect(),
        host: select::<Host>(con, "minute", filter, &range).ok()?.into_iter().map(|(t, _, stats)| HostRow { t, stats }).collect(),
    })
}

pub fn mark_sent(home: &Path, to: i64) {
    // The platform has the minutes before `to`; the next heartbeat starts there.
    if let Ok(con) = crate::db::open(home) {
        let _ = crate::db::set_setting(&con, SENT_KEY, &to.to_string());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem() -> Connection {
        let con = Connection::open_in_memory().unwrap();
        con.execute_batch("CREATE TABLE goethe_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);").unwrap();
        ensure_tables(&con).unwrap();
        con
    }

    #[test]
    fn requests_in_one_minute_add_up_exactly() {
        let mut r = Requests::default();
        r.add(&Sample::new(200, false, 2.0, Some(0.5), 40, 132));
        r.add(&Sample::new(200, false, 60.0, Some(1.0), 900, 14712));
        r.add(&Sample::new(429, false, 0.01, None, 0, 0));
        r.add(&Sample::new(500, false, 0.2, None, 0, 0));
        r.add(&Sample::new(200, true, 3.0, Some(0.2), 10, 7));
        assert_eq!((r.requests, r.ok, r.rate_limited, r.server_errors, r.cancelled), (5, 2, 1, 1, 1));
        assert_eq!(r.completion_tokens, 132 + 14712 + 7);
        assert_eq!(r.prompt_tokens, 950);
        assert_eq!(r.decode_n, 3, "only streamed requests with two or more tokens");
        assert!((r.decode_tps_sum - (131.0 / 1.5 + 14711.0 / 59.0 + 6.0 / 2.8)).abs() < 1e-9);
        assert_eq!(r.latency_hist.iter().sum::<i64>(), 5);
        assert_eq!(r.ttft_hist.iter().sum::<i64>(), 3);
    }

    #[test]
    fn speed_needs_a_first_token_and_two_tokens() {
        let mut r = Requests::default();
        r.add(&Sample::new(200, false, 3.0, Some(1.0), 10, 101));
        assert_eq!((r.decode_n, r.decode_tps_sum, r.ttft_n), (1, 50.0, 1), "first to last token only");
        r.add(&Sample::new(200, false, 2.0, None, 10, 100));
        r.add(&Sample::new(200, false, 2.0, Some(0.5), 10, 1));
        assert_eq!((r.decode_n, r.ttft_n), (1, 2), "no first token or a single token gives no speed");
    }

    #[test]
    fn gauges_merge_by_sample_count_and_keep_the_peak() {
        let s = |v: f64| Gpu { samples: 1, util_pct: Some(v), util_max_pct: Some(v), throttle: if v > 90.0 { 0x40 } else { 0 }, ..Gpu::default() };
        let mut g = Gpu::default();
        for v in [10.0, 20.0, 95.0] {
            g.merge(&s(v));
        }
        assert_eq!(g.samples, 3);
        assert!((g.util_pct.unwrap() - 125.0 / 3.0).abs() < 1e-9);
        assert_eq!((g.util_max_pct, g.throttle), (Some(95.0), 0x40));
        let mut h = Gpu { samples: 12, util_pct: Some(50.0), ..Gpu::default() };
        h.merge(&Gpu { samples: 4, util_pct: Some(10.0), ..Gpu::default() });
        assert_eq!(h.util_pct, Some(40.0));
    }

    #[test]
    fn rows_round_trip_and_a_second_write_of_a_minute_adds_to_it() {
        let con = mem();
        let mut r = Requests::default();
        r.add(&Sample::new(200, false, 1.0, Some(0.1), 5, 50));
        put(&con, "minute", 600, "m", &r, true).unwrap();
        put(&con, "minute", 600, "m", &r, true).unwrap();
        let rows = select::<Requests>(&con, "minute", "", &[]).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].2.requests, rows[0].2.completion_tokens, rows[0].2.latency_hist.iter().sum::<i64>()), (2, 100, 2));
        let g = Gpu { samples: 2, util_pct: Some(33.33), power_w: None, throttle: 4, ..Gpu::default() };
        put(&con, "minute", 600, "1", &g, false).unwrap();
        let back = &select::<Gpu>(&con, "minute", "", &[]).unwrap()[0].2;
        assert_eq!((back.util_pct, back.power_w, back.throttle), (Some(33.3), None, 4));
    }

    #[test]
    fn hours_are_rolled_from_minutes_and_old_rows_expire() {
        let con = mem();
        let now = 100 * 86400 + 3 * 3600 + 120;
        let h = hour_of(now) - 3600;
        let mut r = Requests::default();
        r.add(&Sample::new(200, false, 1.0, None, 1, 10));
        for m in 0..3 {
            put(&con, "minute", h + m * 60, "m", &r, true).unwrap();
            put(&con, "minute", h + m * 60, "", &Host { samples: 12, cpu_pct: Some(10.0 * (m + 1) as f64), ..Host::default() }, true).unwrap();
        }
        put(&con, "minute", now - MINUTE_KEEP_S - 600, "m", &r, true).unwrap();
        put(&con, "hour", now - HOUR_KEEP_S - 3600, "m", &r, false).unwrap();
        roll(&con, now).unwrap();
        let hours = select::<Requests>(&con, "hour", "", &[]).unwrap();
        assert_eq!(hours.len(), 1, "the expired hour row is gone, the partly expired hour is not rolled");
        assert_eq!((hours[0].0, hours[0].2.requests, hours[0].2.completion_tokens), (h, 3, 30));
        assert_eq!(select::<Host>(&con, "hour", "", &[]).unwrap()[0].2.cpu_pct, Some(20.0));
        assert_eq!(select::<Requests>(&con, "minute", "", &[]).unwrap().len(), 3);
    }

    #[test]
    fn a_minute_never_reopens_once_a_later_one_started() {
        let mut l = Live::default();
        rotate(&mut l, 600);
        l.cur.host.samples = 1;
        rotate(&mut l, 660);
        l.cur.host.samples = 1;
        rotate(&mut l, 600);
        assert_eq!((l.cur.t, l.done.len()), (660, 1), "a late clock reading stays in the current minute");
        rotate(&mut l, 720);
        assert_eq!(l.done.iter().map(|m| m.t).collect::<Vec<_>>(), vec![600, 660]);
    }

    #[test]
    fn a_report_covers_written_minutes_after_the_last_send() {
        let con = mem();
        let now = 1_000_000 * 60 + 30;
        let base = minute_of(now) - 10 * 60;
        let mut r = Requests::default();
        r.add(&Sample::new(200, false, 1.0, None, 1, 3));
        for m in 0..10 {
            put(&con, "minute", base + m * 60, "0", &Gpu { samples: 12, ..Gpu::default() }, false).unwrap();
        }
        put(&con, "minute", base + 120, "m", &r, false).unwrap();
        assert!(report_at(&con, now, 0).is_none(), "nothing written yet");
        let rep = report_at(&con, now, minute_of(now)).unwrap();
        assert_eq!((rep.to, rep.gpus.len(), rep.models.len()), (minute_of(now), 10, 1));
        assert!(rep.from <= base);
        crate::db::set_setting(&con, SENT_KEY, &(base + 300).to_string()).unwrap();
        let rep = report_at(&con, now, minute_of(now)).unwrap();
        assert_eq!((rep.from, rep.gpus.len(), rep.models.len()), (base + 300, 5, 0));
        let json = serde_json::to_value(&rep).unwrap();
        assert!(json["gpus"][0].get("samples").is_some() && json["gpus"][0].get("stats").is_none(), "rows are flat");
    }

    #[test]
    fn new_counters_split_generation_and_the_router_hop() {
        let mut r = Requests::default();
        r.add(&Sample { reasoning: 30, cached: 900, router_ms: Some(40.0), ..Sample::new(200, false, 3.0, Some(0.5), 1000, 120) });
        r.add(&Sample::new(200, false, 1.0, Some(0.25), 10, 20));
        assert_eq!((r.reasoning_tokens, r.cached_tokens), (30, 900));
        assert_eq!((r.generation_ms_sum, r.router_ms_sum, r.router_n), (3250, 40, 1));
        assert_eq!(r.total_ttft_hist.iter().sum::<i64>(), 2);
        assert_eq!(r.total_ttft_hist[bucket(540.0)] + r.total_ttft_hist[bucket(250.0)], 2, "a routed request waits engine time plus the hop");
        let back = Requests::from_values(&r.values());
        assert_eq!(back, Requests { decode_tps_sum: back.decode_tps_sum, ..r }, "every column round-trips (speed is kept to one decimal)");
    }

    #[test]
    fn tuples_keep_every_dimension_and_overflow_into_other() {
        let mut m = Minute::default();
        let d = |i: usize| Dims { model: "m".into(), api_key_id: format!("k{i}"), network_path: "lan".into(), finish_reason: "stop".into(), ..Dims::default() };
        for i in 0..MAX_TUPLES + 3 {
            m.count(&d(i), &Sample::new(200, false, 1.0, None, 5, 5));
        }
        assert_eq!(m.tuples.len(), MAX_TUPLES + 1);
        assert_eq!(m.models["m"].requests, MAX_TUPLES as i64 + 3);
        let other = m.tuples.get(&d(0).other().key()).expect("overflow tuple");
        assert_eq!((other.0.requests, Dims::from_key(&d(0).other().key()).api_key_id.as_str()), (3, OTHER));
        let first = Dims::from_key(&d(0).key());
        assert_eq!(first, d(0));
        let weird = Dims { model: format!("a{KEY_SEP}b"), user_id: "u".repeat(200), ..Dims::default() };
        let back = Dims::from_key(&weird.key());
        assert_eq!((back.model.as_str(), back.user_id.len()), ("ab", MAX_DIM));
    }

    #[test]
    fn an_older_usage_table_gains_the_new_columns_and_keeps_its_rows() {
        let con = Connection::open_in_memory().unwrap();
        for g in ["minute", "hour"] {
            con.execute_batch(&format!(
                "CREATE TABLE usage_{g} (t INTEGER NOT NULL, k TEXT NOT NULL, requests, ok, client_errors, server_errors, rate_limited, cancelled, prompt_tokens, completion_tokens, decode_tps_sum, decode_n, ttft_ms_sum, ttft_n, latency_hist, ttft_hist, PRIMARY KEY (t, k)) WITHOUT ROWID;"
            ))
            .unwrap();
        }
        con.execute_batch("INSERT INTO usage_minute (t, k, requests, prompt_tokens, latency_hist) VALUES (60, 'm', 2, 7, '0,2');").unwrap();
        ensure_tables(&con).unwrap();
        ensure_tables(&con).unwrap();
        add_column(&con, "usage_minute", "cached_tokens").expect("a column another process added first is not an error");
        assert!(add_column(&con, "no_such_table", "x").is_err());
        let rows = select::<Requests>(&con, "minute", "", &[]).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].2.requests, rows[0].2.prompt_tokens, rows[0].2.cached_tokens, rows[0].2.latency_hist.clone()), (2, 7, 0, vec![0, 2]));
        let mut r = Requests::default();
        r.add(&Sample { cached: 3, ..Sample::new(200, false, 1.0, Some(0.1), 4, 4) });
        put(&con, "minute", 60, "m", &r, true).unwrap();
        assert_eq!(select::<Requests>(&con, "minute", "", &[]).unwrap()[0].2.cached_tokens, 3);
    }

    #[test]
    fn a_report_carries_the_tuple_rows_flat() {
        let con = mem();
        let now = 1_000_000 * 60 + 30;
        let t = minute_of(now) - 120;
        let d = Dims { model: "m".into(), engine: "sglang".into(), modality: "chat".into(), ..Dims::default() };
        let mut r = Traffic::default();
        r.0.add(&Sample::new(200, false, 1.0, Some(0.1), 3, 4));
        put(&con, "minute", t, &d.key(), &r, true).unwrap();
        let rep = report_at(&con, now, minute_of(now)).unwrap();
        assert_eq!(rep.requests.len(), 1);
        let json = serde_json::to_value(&rep.requests[0]).unwrap();
        assert_eq!((json["model"].as_str(), json["engine"].as_str(), json["completion_tokens"].as_i64()), (Some("m"), Some("sglang"), Some(4)));
        assert!(json.get("api_key_id").is_none() && json.get("dims").is_none(), "unset dimensions are left out and rows are flat");
    }

    #[test]
    fn context_buckets_follow_the_prompt_size() {
        let got: Vec<&str> = [0, 1024, 1025, 10240, 30000, 100_000, 200_000].iter().map(|&p| context_bucket(p)).collect();
        assert_eq!(got, ["1k", "1k", "10k", "10k", "32k", "128k", "128k+"]);
    }

    #[test]
    fn a_large_backlog_is_sent_in_parts() {
        let con = mem();
        let now = 2_000_000 * 60;
        for m in 0..200 {
            for g in 0..8 {
                put(&con, "minute", now - (200 - m) * 60, &g.to_string(), &Gpu { samples: 1, ..Gpu::default() }, false).unwrap();
            }
        }
        let rep = report_at(&con, now, now).unwrap();
        assert!(rep.gpus.len() <= REPORT_ROWS && rep.gpus.len() >= REPORT_ROWS - 8);
        assert_eq!(rep.to, rep.gpus.last().unwrap().t + 60);
    }

    #[test]
    fn totals_add_minutes_and_hours_without_counting_an_hour_twice() {
        let con = mem();
        let now = now_s();
        let m = minute_of(now) - 120;
        let mut r = Requests::default();
        r.add(&Sample::new(200, false, 1.0, Some(0.2), 5, 100));
        put(&con, "minute", m, "a", &r, true).unwrap();
        put(&con, "minute", m, "0", &Gpu { samples: 12, util_pct: Some(50.0), throttle: 4, ..Gpu::default() }, false).unwrap();
        put(&con, "hour", hour_of(now) - 5 * 86400, "a", &r, false).unwrap();
        put(&con, "minute", hour_of(now) - 5 * 86400 + 60, "a", &r, true).unwrap();
        let day = totals(&con, now - 86400).unwrap();
        assert_eq!((day.models[0].1.requests, day.gpus[0].0, day.gpus[0].1.util_pct), (1, 0, Some(50.0)));
        let week = totals(&con, now - 7 * 86400).unwrap();
        assert_eq!(week.models[0].1.completion_tokens, 200, "the minute inside an hour row is not added again");
        assert_eq!(p95_ms(&[0, 0, 19, 1]), Some(250));
        assert_eq!(p95_ms(&[0; 3]), None);
        assert_eq!(throttle_words(0x04 | 0x40), vec!["power", "thermal"]);
    }

    #[test]
    fn cpu_busy_leaves_out_idle_and_iowait() {
        assert_eq!(busy_pct(&[0, 0, 0, 0, 0], &[30, 0, 10, 50, 10]), Some(40.0));
        assert_eq!(busy_pct(&[1, 1, 1, 1, 1], &[1, 1, 1, 1, 1]), None);
    }
}
