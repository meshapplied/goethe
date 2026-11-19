// Prometheus text exposition for the loopback mux /metrics handler.

use crate::hw;
use crate::supervisor;
use nvml_wrapper::Nvml;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;
use std::sync::{Mutex, MutexGuard, OnceLock};

pub const UNKNOWN_MODEL: &str = "_unknown";
const WINDOW_SECS: u64 = 60;

#[derive(Default)]
struct Summary {
    sum: f64,
    count: u64,
}

impl Summary {
    fn observe(&mut self, v: f64) {
        self.sum += v;
        self.count += 1;
    }
}

struct TokenWindow {
    stamps: [u64; WINDOW_SECS as usize],
    tokens: [u64; WINDOW_SECS as usize],
}

impl TokenWindow {
    fn new() -> Self {
        Self {
            stamps: [0; WINDOW_SECS as usize],
            tokens: [0; WINDOW_SECS as usize],
        }
    }

    fn add(&mut self, now: u64, n: u64) {
        let i = (now % WINDOW_SECS) as usize;
        if self.stamps[i] != now {
            self.stamps[i] = now;
            self.tokens[i] = 0;
        }
        self.tokens[i] += n;
    }

    fn sum(&self, now: u64) -> u64 {
        self.stamps
            .iter()
            .zip(self.tokens.iter())
            .filter(|(t, _)| now.saturating_sub(**t) < WINDOW_SECS)
            .map(|(_, n)| n)
            .sum()
    }
}

#[derive(Default)]
struct Inner {
    requests: BTreeMap<(String, u16), u64>,
    tokens: BTreeMap<(String, &'static str), u64>,
    rate_limited: BTreeMap<String, u64>,
    duration: BTreeMap<String, Summary>,
    ttft: BTreeMap<String, Summary>,
    window: HashMap<String, TokenWindow>,
}

fn inner() -> MutexGuard<'static, Inner> {
    static METRICS: OnceLock<Mutex<Inner>> = OnceLock::new();
    METRICS
        .get_or_init(|| Mutex::new(Inner::default()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub fn record_request(model: &str, code: u16, duration_secs: f64, prompt_tokens: i64, completion_tokens: i64) {
    // Count one finished request; model must already be a known id or UNKNOWN_MODEL.
    let mut g = inner();
    g.requests.entry((model.to_string(), code)).and_modify(|n| *n += 1).or_insert(1);
    if duration_secs > 0.0 {
        g.duration.entry(model.to_string()).or_default().observe(duration_secs);
    }
    if prompt_tokens > 0 {
        g.tokens.entry((model.to_string(), "prompt")).and_modify(|n| *n += prompt_tokens as u64).or_insert(prompt_tokens as u64);
    }
    if completion_tokens > 0 {
        g.tokens
            .entry((model.to_string(), "completion"))
            .and_modify(|n| *n += completion_tokens as u64)
            .or_insert(completion_tokens as u64);
        g.window
            .entry(model.to_string())
            .or_insert_with(TokenWindow::new)
            .add(now_secs(), completion_tokens as u64);
    }
}

pub fn record_ttft(model: &str, secs: f64) {
    inner().ttft.entry(model.to_string()).or_default().observe(secs);
}

pub fn model_tokens_per_s_1m(model_id: &str) -> f64 {
    let g = inner();
    let sum = g.window.get(model_id).map(|w| w.sum(now_secs())).unwrap_or(0);
    sum as f64 / WINDOW_SECS as f64
}

pub fn inc_rate_limited(key_id: &str) {
    inner().rate_limited.entry(key_id.to_string()).and_modify(|n| *n += 1).or_insert(1);
}

fn esc(v: &str) -> String {
    // Escape a Prometheus label value (backslash, quote, newline).
    let mut out = String::with_capacity(v.len());
    for c in v.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out
}

pub fn render_prometheus() -> String {
    // Render counters, engine state and live GPU memory; blocking (DB and NVML), call off the async runtime.
    let mut out = String::new();
    write_metric_header(&mut out, "goethe_up", "gauge", "1 if goethe mux is serving metrics");
    let _ = writeln!(out, "goethe_up 1");
    write_counters(&mut out);
    write_instances(&mut out);
    write_gpu_memory(&mut out);
    out
}

fn write_counters(out: &mut String) {
    // Request, token, rate-limit and latency series from the in-process counters.
    let g = inner();
    write_metric_header(out, "goethe_requests_total", "counter", "HTTP requests through mux by model and status code");
    for ((model, code), n) in &g.requests {
        let _ = writeln!(out, "goethe_requests_total{{model=\"{}\",code=\"{code}\"}} {n}", esc(model));
    }
    write_metric_header(out, "goethe_tokens_total", "counter", "Token usage through mux");
    for ((model, kind), n) in &g.tokens {
        let _ = writeln!(out, "goethe_tokens_total{{model=\"{}\",type=\"{kind}\"}} {n}", esc(model));
    }
    write_metric_header(out, "goethe_rate_limited_total", "counter", "Rate-limited requests per API key");
    for (key_id, n) in &g.rate_limited {
        let _ = writeln!(out, "goethe_rate_limited_total{{key_id=\"{}\"}} {n}", esc(key_id));
    }
    for (name, help, series) in [
        ("goethe_request_duration_seconds", "Request duration seconds, to the last body byte", &g.duration),
        ("goethe_time_to_first_token_seconds", "Streaming time to first token seconds", &g.ttft),
    ] {
        write_metric_header(out, name, "summary", help);
        for (model, s) in series {
            let m = esc(model);
            let _ = writeln!(out, "{name}_sum{{model=\"{m}\"}} {}", s.sum);
            let _ = writeln!(out, "{name}_count{{model=\"{m}\"}} {}", s.count);
        }
    }
}

fn write_metric_header(out: &mut String, name: &str, kind: &str, help: &str) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {kind}");
}

fn write_instances(out: &mut String) {
    // One status sample per instance and one restart total per model.
    write_metric_header(out, "goethe_engine_status", "gauge", "1 when engine instance is RUNNING");
    write_metric_header(out, "goethe_engine_restarts_total", "counter", "Watchdog restarts per model");
    let Ok(Value::Array(rows)) = supervisor::handle_ps() else {
        return;
    };
    let mut restarts: BTreeMap<String, i64> = BTreeMap::new();
    for row in rows {
        let model = row.get("model_id").and_then(|v| v.as_str()).unwrap_or("");
        let inst = row.get("instance_id").and_then(|v| v.as_str()).unwrap_or("");
        let status = row.get("status").and_then(|v| v.as_str()).unwrap_or("");
        let gpus = row
            .get("gpu_indices")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|n| n.to_string()).collect::<Vec<_>>().join(","))
            .unwrap_or_default();
        let up = u8::from(status == "RUNNING");
        let _ = writeln!(
            out,
            "goethe_engine_status{{model=\"{}\",instance=\"{}\",gpus=\"{gpus}\"}} {up}",
            esc(model),
            esc(inst)
        );
        let n = row.get("restarts").and_then(|v| v.as_i64()).unwrap_or(0);
        restarts.entry(model.to_string()).and_modify(|t| *t += n).or_insert(n);
    }
    for (model, n) in restarts {
        let _ = writeln!(out, "goethe_engine_restarts_total{{model=\"{}\"}} {n}", esc(&model));
    }
}

fn nvml() -> Option<&'static Nvml> {
    static NVML: OnceLock<Option<Nvml>> = OnceLock::new();
    NVML.get_or_init(|| crate::nvidia_driver::init_nvml().ok()).as_ref()
}

fn write_gpu_memory(out: &mut String) {
    // Live per-GPU memory from NVML (unified memory hosts report host RAM).
    write_metric_header(out, "goethe_gpu_memory_used_bytes", "gauge", "GPU memory used (NVML or unified host memory)");
    write_metric_header(out, "goethe_gpu_memory_total_bytes", "gauge", "GPU memory total");
    let Some(nvml) = nvml() else {
        return;
    };
    let Ok(count) = nvml.device_count() else {
        return;
    };
    for idx in 0..count {
        let Ok(dev) = nvml.device_by_index(idx) else {
            continue;
        };
        let Ok((total_mb, used_mb, _, _)) = hw::memory_from_nvml_or_unified(dev.memory_info(), false) else {
            continue;
        };
        let _ = writeln!(out, "goethe_gpu_memory_used_bytes{{gpu=\"{idx}\"}} {}", used_mb * 1024 * 1024);
        let _ = writeln!(out, "goethe_gpu_memory_total_bytes{{gpu=\"{idx}\"}} {}", total_mb * 1024 * 1024);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_values_are_escaped() {
        assert_eq!(esc("plain/model-1"), "plain/model-1");
        assert_eq!(esc("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
        record_request("lbl\"} evil 1\n#", 404, 0.01, 0, 0);
        let mut out = String::new();
        write_counters(&mut out);
        let line = out.lines().find(|l| l.contains("evil")).expect("series present");
        assert!(line.starts_with("goethe_requests_total{model=\"lbl\\\"} evil 1\\n#\",code=\"404\"}"), "{line}");
        assert!(!out.lines().any(|l| l.starts_with('#') && l.contains("evil")));
    }

    #[test]
    fn token_window_keeps_every_sample_for_a_minute() {
        let mut w = TokenWindow::new();
        for i in 0..10_000u64 {
            w.add(1_000, 1 + i % 2);
        }
        assert_eq!(w.sum(1_000), 15_000);
        w.add(1_030, 5);
        assert_eq!(w.sum(1_059), 15_005);
        assert_eq!(w.sum(1_060), 5);
        assert_eq!(w.sum(1_090), 0);
    }

    #[test]
    fn duration_summary_has_no_fake_quantile() {
        record_request("sum-model", 200, 0.5, 3, 4);
        record_request("sum-model", 200, 1.5, 3, 4);
        let mut out = String::new();
        write_counters(&mut out);
        assert!(out.contains("goethe_request_duration_seconds_sum{model=\"sum-model\"} 2"));
        assert!(out.contains("goethe_request_duration_seconds_count{model=\"sum-model\"} 2"));
        assert!(!out.contains("quantile"));
        assert!(out.contains("goethe_tokens_total{model=\"sum-model\",type=\"completion\"} 8"));
    }
}
