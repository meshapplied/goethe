// Prometheus text exposition for the loopback mux /metrics handler.

use crate::db;
use crate::paths::goethe_home;
use crate::supervisor;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Mutex, OnceLock};

struct Inner {
    requests: HashMap<(String, String), u64>,
    tokens: HashMap<(String, String), u64>,
    rate_limited: HashMap<String, u64>,
    duration_sum: HashMap<String, f64>,
    duration_count: HashMap<String, u64>,
    completion_window: HashMap<String, Vec<(f64, u64)>>,
}

const COMPLETION_WINDOW_SECS: f64 = 60.0;
const COMPLETION_WINDOW_MAX: usize = 4096;

impl Inner {
    fn new() -> Self {
        Self {
            requests: HashMap::new(),
            tokens: HashMap::new(),
            rate_limited: HashMap::new(),
            duration_sum: HashMap::new(),
            duration_count: HashMap::new(),
            completion_window: HashMap::new(),
        }
    }
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

fn prune_completion_window(samples: &mut Vec<(f64, u64)>, now: f64) {
    let cutoff = now - COMPLETION_WINDOW_SECS;
    while samples.first().is_some_and(|(t, _)| *t < cutoff) {
        samples.remove(0);
    }
    if samples.len() > COMPLETION_WINDOW_MAX {
        let drop = samples.len() - COMPLETION_WINDOW_MAX;
        samples.drain(0..drop);
    }
}

static METRICS: OnceLock<Mutex<Inner>> = OnceLock::new();

fn inner() -> &'static Mutex<Inner> {
    METRICS.get_or_init(|| Mutex::new(Inner::new()))
}

pub fn record_request(model: &str, code: u16, duration_secs: f64, prompt_tokens: i64, completion_tokens: i64) {
    if let Ok(mut g) = inner().lock() {
        let key = (model.to_string(), code.to_string());
        let next = g.requests.get(&key).copied().unwrap_or(0) + 1;
        g.requests.insert(key, next);
        if duration_secs > 0.0 {
            let sum = g.duration_sum.get(model).copied().unwrap_or(0.0) + duration_secs;
            g.duration_sum.insert(model.to_string(), sum);
            let cnt = g.duration_count.get(model).copied().unwrap_or(0) + 1;
            g.duration_count.insert(model.to_string(), cnt);
        }
        if prompt_tokens > 0 {
            let tk = (model.to_string(), "prompt".into());
            let v = g.tokens.get(&tk).copied().unwrap_or(0) + prompt_tokens as u64;
            g.tokens.insert(tk, v);
        }
        if completion_tokens > 0 {
            let tk = (model.to_string(), "completion".into());
            let v = g.tokens.get(&tk).copied().unwrap_or(0) + completion_tokens as u64;
            g.tokens.insert(tk, v);
            let now = now_secs();
            let samples = g
                .completion_window
                .entry(model.to_string())
                .or_default();
            samples.push((now, completion_tokens as u64));
            prune_completion_window(samples, now);
        }
    }
}

pub fn model_tokens_per_s_1m(model_id: &str) -> f64 {
    let now = now_secs();
    let cutoff = now - COMPLETION_WINDOW_SECS;
    let Ok(g) = inner().lock() else {
        return 0.0;
    };
    let Some(samples) = g.completion_window.get(model_id) else {
        return 0.0;
    };
    let sum: u64 = samples
        .iter()
        .filter(|(t, _)| *t >= cutoff)
        .map(|(_, n)| n)
        .sum();
    sum as f64 / COMPLETION_WINDOW_SECS
}

pub fn inc_rate_limited(key_id: &str) {
    if let Ok(mut g) = inner().lock() {
        let v = g.rate_limited.get(key_id).copied().unwrap_or(0) + 1;
        g.rate_limited.insert(key_id.to_string(), v);
    }
}

pub fn render_prometheus() -> String {
    let mut out = String::new();
    write_metric_header(&mut out, "goethe_up", "gauge", "1 if goethe mux is serving metrics");
    let _ = writeln!(out, "goethe_up 1");
    if let Ok(g) = inner().lock() {
        write_metric_header(
            &mut out,
            "goethe_requests_total",
            "counter",
            "HTTP requests through mux by model and status code",
        );
        for ((model, code), n) in &g.requests {
            let _ = writeln!(out, "goethe_requests_total{{model=\"{model}\",code=\"{code}\"}} {n}");
        }
        write_metric_header(
            &mut out,
            "goethe_tokens_total",
            "counter",
            "Token usage through mux",
        );
        for ((model, kind), n) in &g.tokens {
            let _ = writeln!(out, "goethe_tokens_total{{model=\"{model}\",type=\"{kind}\"}} {n}");
        }
        write_metric_header(
            &mut out,
            "goethe_rate_limited_total",
            "counter",
            "Rate-limited requests per API key",
        );
        for (key_id, n) in &g.rate_limited {
            let _ = writeln!(out, "goethe_rate_limited_total{{key_id=\"{key_id}\"}} {n}");
        }
        write_metric_header(
            &mut out,
            "goethe_request_duration_seconds",
            "summary",
            "Request duration seconds",
        );
        for (model, sum) in &g.duration_sum {
            let count = g.duration_count.get(model).copied().unwrap_or(0);
            let _ = writeln!(
                out,
                "goethe_request_duration_seconds{{model=\"{model}\",quantile=\"0.5\"}} {}",
                if count > 0 { sum / count as f64 } else { 0.0 }
            );
            let _ = writeln!(
                out,
                "goethe_request_duration_seconds_sum{{model=\"{model}\"}} {sum}"
            );
            let _ = writeln!(
                out,
                "goethe_request_duration_seconds_count{{model=\"{model}\"}} {count}"
            );
        }
    }
    let home = goethe_home();
    if let Ok(con) = db::open(&home) {
        write_instances(&mut out, &con);
    }
    write_gpu_memory(&mut out);
    out
}

fn write_metric_header(out: &mut String, name: &str, kind: &str, help: &str) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {kind}");
}

fn write_instances(out: &mut String, _con: &rusqlite::Connection) {
    write_metric_header(
        out,
        "goethe_engine_status",
        "gauge",
        "1 when engine instance is RUNNING",
    );
    write_metric_header(
        out,
        "goethe_engine_restarts_total",
        "counter",
        "Watchdog restarts per model",
    );
    let Ok(ps) = supervisor::handle_ps() else {
        return;
    };
    let Value::Array(rows) = ps else { return };
    for row in rows {
        let model = row.get("model_id").and_then(|v| v.as_str()).unwrap_or("");
        let inst = row.get("instance_id").and_then(|v| v.as_str()).unwrap_or("");
        let status = row.get("status").and_then(|v| v.as_str()).unwrap_or("");
        let gpus = row
            .get("gpu_indices")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default();
        let up = if status == "RUNNING" { 1 } else { 0 };
        let _ = writeln!(
            out,
            "goethe_engine_status{{model=\"{model}\",instance=\"{inst}\",gpus=\"{gpus}\"}} {up}"
        );
        let restarts = row.get("restarts").and_then(|v| v.as_i64()).unwrap_or(0);
        let _ = writeln!(out, "goethe_engine_restarts_total{{model=\"{model}\"}} {restarts}");
    }
}

fn write_gpu_memory(out: &mut String) {
    write_metric_header(
        out,
        "goethe_gpu_memory_used_bytes",
        "gauge",
        "GPU memory used (NVML or unified host memory)",
    );
    write_metric_header(
        out,
        "goethe_gpu_memory_total_bytes",
        "gauge",
        "GPU memory total",
    );
    let home = goethe_home();
    let path = home.join("hardware_state.json");
    let Ok(data) = std::fs::read_to_string(&path) else {
        return;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) else {
        return;
    };
    let Some(gpus) = v.get("gpus").and_then(|g| g.as_array()) else {
        return;
    };
    for g in gpus {
        let idx = g.get("index").and_then(|x| x.as_u64()).unwrap_or(0);
        let used_mb = g.get("vram_used_mb").and_then(|x| x.as_u64()).unwrap_or(0);
        let total_mb = g.get("vram_total_mb").and_then(|x| x.as_u64()).unwrap_or(0);
        let used = used_mb * 1024 * 1024;
        let total = total_mb * 1024 * 1024;
        let _ = writeln!(out, "goethe_gpu_memory_used_bytes{{gpu=\"{idx}\"}} {used}");
        let _ = writeln!(out, "goethe_gpu_memory_total_bytes{{gpu=\"{idx}\"}} {total}");
    }
}
