// The Metrics page of the goethe menu: model usage, API key usage per model, and GPU and host use over a chosen range, from the local usage log, in the Status page's style.

use crate::status_view::{blue, clip, pad, rpad, section, width_for, GAP};
use crate::term::{paint, visible_width, Color};
use crate::usage_log::{self, Gpu, Host, Requests, Row, Totals};
use std::collections::HashMap;
use std::time::Duration;

const RANGES: [(&str, i64); 6] = [("1 h", 3600), ("24 h", 86400), ("7 d", 7 * 86400), ("this month", 0), ("30 d", 30 * 86400), ("90 d", 90 * 86400)];
pub const REFRESH: Duration = Duration::from_secs(5);
const UTIL_SEGMENTS: usize = 10;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetricsView {
    pub range: usize,
    pub totals: Totals,
    pub key_names: HashMap<String, String>,
    pub since: i64,
}

impl MetricsView {
    pub fn new() -> MetricsView {
        let mut v = MetricsView { range: 1, ..MetricsView::default() };
        v.refresh();
        v
    }

    pub fn refresh(&mut self) {
        // Read the range again from state.db; a database that cannot be read shows empty tables.
        let since = since(RANGES[self.range].1, chrono::Local::now());
        self.since = since;
        let Ok(con) = crate::db::open(&crate::paths::goethe_home()) else {
            return;
        };
        self.totals = usage_log::totals(&con, since).unwrap_or_default();
        self.key_names = con
            .prepare("SELECT key_id, name FROM api_keys")
            .and_then(|mut s| s.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect())
            .unwrap_or_default();
    }

    pub fn step(&mut self, down: bool) {
        let n = RANGES.len();
        self.range = if down { (self.range + 1) % n } else { (self.range + n - 1) % n };
        self.refresh();
    }
}

fn since(len: i64, now: chrono::DateTime<chrono::Local>) -> i64 {
    // The start of a range: now minus its length, or for length 0 (this month) midnight on the first of the month, local time.
    use chrono::{Datelike, TimeZone};
    if len > 0 {
        return now.timestamp() - len;
    }
    chrono::Local.with_ymd_and_hms(now.year(), now.month(), 1, 0, 0, 0).earliest().map_or(now.timestamp() - 30 * 86400, |d| d.timestamp())
}

fn num(n: i64) -> String {
    // Counts up to 9999 as they are, then 14.8k, 1.2M, 3.4B; a figure that rounds to 1000.0 moves up a unit.
    if n.unsigned_abs() <= 9_999 {
        return n.to_string();
    }
    let f = n as f64;
    for (div, unit) in [(1e3, "k"), (1e6, "M")] {
        if (f / div * 10.0).round().abs() < 10_000.0 {
            return format!("{:.1}{unit}", f / div);
        }
    }
    format!("{:.1}B", f / 1e9)
}

fn ms(v: Option<f64>) -> String {
    match v {
        Some(x) if x >= 1000.0 => format!("{:.1} s", x / 1000.0),
        Some(x) => format!("{x:.0} ms"),
        None => "-".to_string(),
    }
}

fn p95(r: &Requests) -> String {
    // The histogram only knows the bucket, so the 95th percentile shows as "under" its upper bound.
    match usage_log::p95_ms(&r.latency_hist) {
        Some(b) => format!("< {}", ms(Some(b as f64))),
        None if r.requests > 0 => format!("> {}", ms(usage_log::BOUNDS_MS.last().map(|&b| b as f64))),
        None => "-".to_string(),
    }
}

fn model_cells(r: &Requests) -> Vec<String> {
    // A key that made no request in the range shows zeros all the way.
    if r.requests == 0 {
        return vec!["0".into(), "0".into(), "0".into(), "0".into(), "0 ms".into(), "0.0".into(), "0 ms".into()];
    }
    let avg = |sum: f64, n: i64| (n > 0).then(|| sum / n as f64);
    vec![
        num(r.requests),
        num(r.requests - r.ok),
        num(r.prompt_tokens),
        num(r.completion_tokens),
        ms(avg(r.ttft_ms_sum as f64, r.ttft_n)),
        avg(r.decode_tps_sum, r.decode_n).map_or("-".to_string(), |t| format!("{t:.1}")),
        p95(r),
    ]
}

fn table(head: &[&str], left: usize, rows: &[Vec<String>], total: Option<&[String]>, width: usize) -> Vec<String> {
    // The first `left` columns left-aligned, the last of them as wide as the rest allows; the others right-aligned to their widest cell; a total row in bold.
    let flex = left.max(1) - 1;
    let mut cols: Vec<usize> = head.iter().map(|h| h.chars().count()).collect();
    for r in rows.iter().map(|r| r.as_slice()).chain(total) {
        for (i, c) in r.iter().enumerate() {
            cols[i] = cols[i].max(visible_width(c));
        }
    }
    let rest: usize = cols.iter().enumerate().filter(|(i, _)| *i != flex).map(|(_, w)| w + GAP.len()).sum();
    cols[flex] = cols[flex].min(width.saturating_sub(rest)).max(8);
    let line = |cells: &[String]| -> String {
        let mut out = String::new();
        for (i, c) in cells.iter().enumerate() {
            if i > 0 {
                out.push_str(GAP);
            }
            out.push_str(&if i < left { pad(&clip(c, cols[i]), cols[i]) } else { rpad(c, cols[i]) });
        }
        out.trim_end().to_string()
    };
    let mut out = vec![paint(&line(&head.iter().map(|h| h.to_string()).collect::<Vec<_>>()), Color::Gray)];
    out.extend(rows.iter().map(|r| line(r)));
    if let Some(t) = total {
        out.push(paint(&line(t), Color::Bold));
    }
    out
}

const MODEL_HEAD: [&str; 7] = ["REQUESTS", "ERRORS", "INPUT TOKENS", "OUTPUT TOKENS", "AVG TTFT", "AVG TOK/S", "P95 LATENCY"];

fn model_name(id: &str) -> String {
    // Requests that never reached a model: an id goethe does not serve, or a key over its rate limit (checked before the body is read).
    match id {
        crate::metrics::UNKNOWN_MODEL => paint("(unknown model)", Color::Gray),
        usage_log::RATE_LIMITED => paint("(rate limited)", Color::Gray),
        _ => id.to_string(),
    }
}

fn model_lines(t: &Totals, range: &str, width: usize) -> Vec<String> {
    let mut out = vec![section("\u{25a3}", "Models", width)];
    if t.models.is_empty() {
        out.push(paint(&format!("no requests in {range}"), Color::Gray));
        return out;
    }
    let mut models: Vec<&(String, Requests)> = t.models.iter().collect();
    models.sort_by_key(|(m, _)| m.starts_with('_'));
    let rows: Vec<Vec<String>> = models.iter().map(|(m, r)| [vec![model_name(m)], model_cells(r)].concat()).collect();
    let mut sum = Requests::default();
    for (_, r) in &t.models {
        sum.merge(r);
    }
    let total = [vec!["Total".to_string()], model_cells(&sum)].concat();
    out.extend(table(&[&["MODEL"][..], &MODEL_HEAD].concat(), 1, &rows, Some(&total), width));
    out
}

fn pct(v: Option<f64>) -> String {
    v.map_or("-".to_string(), |x| format!("{x:.0}%"))
}

fn gpu_cells(index: u32, g: &Gpu) -> Vec<String> {
    let util = match g.util_pct {
        Some(u) => format!("{} {}", crate::status_view::colored_bar(u / 100.0, UTIL_SEGMENTS, &crate::theme::secondary()), rpad(&pct(Some(u)), 4)),
        None => format!("{} {}", paint(&"\u{2591}".repeat(UTIL_SEGMENTS), Color::Gray), rpad("-", 4)),
    };
    let power = match (g.power_w, g.power_limit_w) {
        (Some(p), Some(l)) => format!("{p:.0} / {l:.0} W"),
        (Some(p), None) => format!("{p:.0} W"),
        _ => "-".to_string(),
    };
    let words = usage_log::throttle_words(g.throttle);
    vec![
        format!("gpu{index}"),
        util,
        pct(g.util_max_pct),
        pct(g.membw_pct),
        power,
        g.temp_max_c.map_or("-".to_string(), |t| format!("{t:.0} \u{b0}C")),
        if words.is_empty() { "no".to_string() } else { paint(&words.join(", "), Color::Red) },
    ]
}

fn host_line(h: &Host) -> String {
    // CPU and RAM averages with their peaks, and the least free disk seen.
    let label = |t: &str| paint(t, Color::Gray);
    let gb = |mb: f64| format!("{:.1} GB", mb / 1024.0);
    let mut parts = vec![format!("{} {} {}", label("CPU"), pct(h.cpu_pct), label(&format!("(peak {})", pct(h.cpu_max_pct))))];
    if let (Some(u), Some(t)) = (h.ram_used_mb, h.ram_total_mb) {
        parts.push(format!("{} {} / {} {}", label("RAM"), gb(u), gb(t), label(&format!("(peak {})", gb(h.ram_used_max_mb.unwrap_or(u))))));
    }
    if let Some(d) = h.disk_free_mb {
        parts.push(format!("{} {}", label("Disk free"), if d >= 1024.0 * 1000.0 { format!("{:.1} TB", d / 1024.0 / 1024.0) } else { gb(d) }));
    }
    if let Some(p) = h.mem_pressure.filter(|&p| p > 1.0) {
        parts.push(format!("{} {}", label("Memory pressure"), paint(if p >= 4.0 { "critical" } else { "warning" }, Color::Red)));
    }
    parts.join(GAP)
}

fn infra_lines(v: &MetricsView, width: usize) -> Vec<String> {
    let mut out = vec![section("\u{25a3}", "GPUs and host", width)];
    if v.totals.gpus.is_empty() {
        out.push(paint("no GPU samples yet", Color::Gray));
    } else {
        let rows: Vec<Vec<String>> = v.totals.gpus.iter().map(|(i, g)| gpu_cells(*i, g)).collect();
        let band = if cfg!(target_os = "macos") { "-" } else { "MEM BW" };
        out.extend(table(&["GPU", "AVG UTIL", "PEAK", band, "AVG POWER", "PEAK TEMP", "THROTTLED"], 1, &rows, None, width));
    }
    if v.totals.host.samples > 0 {
        out.push(String::new());
        out.push(clip(&host_line(&v.totals.host), width));
    }
    out
}

fn key_lines(v: &MetricsView, width: usize) -> Vec<String> {
    // One block per API key that still exists, busiest first: its name on the first line, then one line per model it used with that model's figures.
    let mut out = vec![section("\u{25a3}", "API keys", width)];
    if v.key_names.is_empty() {
        out.push(paint("no API keys", Color::Gray));
        return out;
    }
    let mut by_key: Vec<(&str, Vec<(&str, &Requests)>)> = Vec::new();
    for (id, model, r) in v.totals.keys.iter().filter(|(id, _, _)| v.key_names.contains_key(id)) {
        match by_key.iter_mut().find(|(k, _)| k == id) {
            Some((_, models)) => models.push((model, r)),
            None => by_key.push((id, vec![(model, r)])),
        }
    }
    let busy = |models: &[(&str, &Requests)]| models.iter().map(|(_, r)| r.requests).sum::<i64>();
    by_key.sort_by_key(|(_, models)| std::cmp::Reverse(busy(models)));
    let idle = Requests::default();
    let mut quiet: Vec<(&String, &String)> = v.key_names.iter().filter(|(id, _)| !by_key.iter().any(|(k, _)| k == id)).collect();
    quiet.sort_by(|a, b| a.1.cmp(b.1));
    by_key.extend(quiet.into_iter().map(|(id, _)| (id.as_str(), vec![("-", &idle)])));
    let mut rows = Vec::new();
    for (id, mut models) in by_key {
        models.sort_by_key(|(m, r)| (m.starts_with('_'), std::cmp::Reverse(r.requests)));
        let name = v.key_names.get(id).cloned().unwrap_or_default();
        for (i, (model, r)) in models.iter().enumerate() {
            let n = if i == 0 { name.clone() } else { String::new() };
            rows.push([vec![n, if *model == "-" { paint("-", Color::Gray) } else { model_name(model) }], model_cells(r)].concat());
        }
    }
    out.extend(table(&[&["KEY", "MODEL"][..], &MODEL_HEAD].concat(), 2, &rows, None, width));
    out
}

fn range_line(range: usize) -> String {
    // The ranges to pick from, the current one bold in the theme's color.
    let items: Vec<String> = RANGES.iter().enumerate().map(|(i, (l, _))| if i == range { paint(&blue(l), Color::Bold) } else { paint(l, Color::Gray) }).collect();
    format!("{}{GAP}{}", blue("Metrics"), items.join("   "))
}

fn since_text(v: &MetricsView) -> String {
    // Where the range starts, local time: with the time of day for the hour and day ranges, the date alone for the longer ones.
    let short = RANGES[v.range].1 > 0 && RANGES[v.range].1 <= 86400;
    match chrono::DateTime::from_timestamp(v.since, 0).filter(|_| v.since > 0) {
        Some(d) => format!("since {}", d.with_timezone(&chrono::Local).format(if short { "%Y-%m-%d %H:%M" } else { "%Y-%m-%d" })),
        None => String::new(),
    }
}

pub fn render(v: &MetricsView, width: usize) -> Vec<String> {
    // The range line, four empty lines like the Status page, then the model, API key, and GPU and host tables.
    let w = width_for(width);
    let range = if RANGES[v.range].1 == 0 { RANGES[v.range].0.to_string() } else { format!("last {}", RANGES[v.range].0) };
    let mut out = vec![String::new(), format!("  {}", range_line(v.range)), format!("  {}", paint(&since_text(v), Color::Gray)), String::new(), String::new(), String::new()];
    let blocks = [model_lines(&v.totals, &range, w), key_lines(v, w), infra_lines(v, w)];
    for (i, b) in blocks.iter().enumerate() {
        if i > 0 {
            out.push(String::new());
        }
        out.extend(b.iter().map(|l| format!("  {}", clip(l, w))));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn plain(lines: &[String]) -> Vec<String> {
        lines.iter().map(|l| crate::status_view::strip(l)).collect()
    }

    fn view() -> MetricsView {
        let mut a = Requests::default();
        a.add(&crate::usage_log::Sample::new(200, false, 2.0, Some(0.5), 40, 132));
        a.add(&crate::usage_log::Sample::new(200, false, 60.0, Some(1.0), 900, 14712));
        let mut b = Requests::default();
        b.add(&crate::usage_log::Sample::new(500, false, 0.2, None, 0, 0));
        let gpu = Gpu { samples: 12, util_pct: Some(73.0), util_max_pct: Some(100.0), membw_pct: Some(40.0), vram_used_mb: Some(80_000.0), vram_goethe_mb: Some(20_000.0), power_w: Some(250.0), power_limit_w: Some(300.0), temp_max_c: Some(71.0), throttle: 4, ..Gpu::default() };
        MetricsView {
            range: 1,
            totals: Totals {
                models: vec![("org/a".into(), a.clone()), ("org/b".into(), b.clone())],
                keys: vec![("k1".into(), "org/a".into(), a), ("k1".into(), "org/b".into(), b.clone()), ("k2".into(), "org/b".into(), b)],
                gpus: vec![(3, gpu)],
                host: Host { samples: 12, cpu_pct: Some(5.0), cpu_max_pct: Some(9.0), ram_used_mb: Some(40_000.0), ram_total_mb: Some(250_000.0), ..Host::default() },
            },
            key_names: HashMap::from([("k1".to_string(), "app".to_string())]),
            since: chrono::Local.with_ymd_and_hms(2026, 11, 18, 21, 30, 0).unwrap().timestamp(),
        }
    }

    #[test]
    fn the_start_of_the_range_is_right_under_the_ranges() {
        let t = plain(&render(&view(), 120));
        assert!(t[1].contains("Metrics") && t[2].trim() == "since 2026-11-18 21:30" && t[3].is_empty(), "{t:?}");
    }

    #[test]
    fn the_model_table_ends_with_the_sums() {
        let t = plain(&render(&view(), 120));
        let head = t.iter().position(|l| l.contains("MODEL") && l.contains("AVG TOK/S")).unwrap();
        assert!(t[head + 1].contains("org/a") && t[head + 1].contains("14.8k"), "{t:?}");
        let total = &t[head + 3];
        assert!(total.trim_start().starts_with("Total") && total.contains(" 3 ") && total.contains(" 1 "), "3 requests, 1 error: {total}");
        assert!(total.contains("750 ms"), "the mean over every request with a first token: {total}");
    }

    #[test]
    fn the_gpu_table_names_throttling_and_the_host_line_follows() {
        let t = plain(&render(&view(), 120));
        let g = t.iter().find(|l| l.trim_start().starts_with("gpu3")).unwrap();
        assert!(g.contains("73%") && g.contains("100%") && g.contains("250 / 300 W") && g.contains("71 \u{b0}C") && g.contains("power"), "{g}");
        assert!(t.iter().any(|l| l.contains("CPU 5%") && l.contains("RAM")), "{t:?}");
        assert!(!t.iter().any(|l| l.contains("updated every minute") || l.contains("AVG VRAM")), "{t:?}");
    }

    #[test]
    fn each_key_lists_its_models_below_each_other_before_the_gpus() {
        let t = plain(&render(&view(), 140));
        let head = t.iter().position(|l| l.contains("KEY") && l.contains("INPUT TOKENS")).unwrap();
        assert!(!t[head].contains(" ID "), "no id column: {}", t[head]);
        assert!(t[head + 1].trim_start().starts_with("app") && !t[head + 1].contains("k1") && t[head + 1].contains("org/a") && t[head + 1].contains("14.8k"), "{}", t[head + 1]);
        let model_at = t[head + 1].find("org/a").unwrap();
        assert_eq!(t[head + 2].find("org/b"), Some(model_at), "the next model of the same key sits under the first: {t:?}");
        assert!(t[head + 2][..model_at].trim().is_empty(), "no name again on its second line");
        assert!(!t.iter().any(|l| l.contains("revoked")) && t[head + 3].trim().is_empty(), "a revoked key is not listed: {t:?}");
        let gpus = t.iter().position(|l| l.contains("GPUs and host")).unwrap();
        assert!(gpus > head, "the API key table comes before the GPU table");
        assert!(t.iter().any(|l| l.contains("OUTPUT TOKENS")) && !t.iter().any(|l| l.contains("TOKENS OUT")));
        assert!(!t.iter().any(|l| l.contains("by whole days")));
        let mut quiet = view();
        quiet.key_names.insert("k3".into(), "idle-app".into());
        let t = plain(&render(&quiet, 140));
        let idle = t.iter().find(|l| l.contains("idle-app")).unwrap();
        assert!(idle.contains(" - ") && idle.contains("0 ms") && idle.contains("0.0"), "a key without requests shows zeros: {idle}");
    }

    #[test]
    fn nothing_overflows_and_empty_ranges_say_so() {
        for w in [60, 80, 100, 140] {
            for l in render(&view(), w) {
                assert!(visible_width(&l) <= width_for(w) + 2, "{w}: {l}");
            }
        }
        let t = plain(&render(&MetricsView::default(), 100));
        assert!(t.iter().any(|l| l.contains("no requests in last 1 h")), "{t:?}");
        let now = chrono::Local.with_ymd_and_hms(2026, 11, 19, 21, 30, 0).unwrap();
        assert_eq!(since(0, now), chrono::Local.with_ymd_and_hms(2026, 11, 1, 0, 0, 0).unwrap().timestamp(), "this month starts on the first");
        assert_eq!(since(3600, now), now.timestamp() - 3600);
        assert_eq!(num(14_844), "14.8k");
        assert_eq!(num(9_999), "9999");
        assert_eq!(num(999_949), "999.9k");
        assert_eq!(num(999_950), "1.0M");
        assert_eq!(num(999_960_000), "1.0B");
        assert_eq!(num(-20_000), "-20.0k");
        assert_eq!(num(i64::MIN), "-9223372036.9B");
    }
}
