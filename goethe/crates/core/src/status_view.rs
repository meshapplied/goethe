// The Status page of the goethe menu: one summary line, then live GPU, CPU and memory tables in the style of mole's status view.

use crate::paths::goethe_home;
use crate::term::{paint, visible_width, Color};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{Duration, Instant};

const VRAM_SEGMENTS: usize = 10;
const BAND_SEGMENTS: usize = 10;
const HALF_SEGMENTS: usize = 16;
pub(crate) const GAP: &str = "   ";
pub(crate) const MAX_WIDTH: usize = 120;
const SLOW: Duration = Duration::from_secs(1);
pub const FAST: Duration = Duration::from_millis(100);

fn sgr(code: &str, text: &str) -> String {
    if text.is_empty() || !crate::term::color_enabled() {
        return text.to_string();
    }
    format!("\x1b[{code}m{text}\x1b[0m")
}

pub(crate) fn blue(text: &str) -> String {
    // Titles in the theme's main color.
    sgr(&crate::theme::main(), text)
}

pub(crate) fn pad(text: &str, width: usize) -> String {
    // Pad a colored string to a visible width.
    format!("{text}{}", " ".repeat(width.saturating_sub(visible_width(text))))
}

pub(crate) fn rpad(text: &str, width: usize) -> String {
    format!("{}{text}", " ".repeat(width.saturating_sub(visible_width(text))))
}

fn blocks(fraction: f64, segments: usize) -> usize {
    ((fraction.clamp(0.0, 1.0) * segments as f64).round() as usize).min(segments)
}

fn bar(fraction: f64, segments: usize) -> String {
    // Filled blocks for the used part, dim dotted blocks for the rest.
    let full = blocks(fraction, segments);
    format!("{}{}", "\u{2588}".repeat(full), paint(&"\u{2591}".repeat(segments - full), Color::Gray))
}

pub(crate) fn colored_bar(fraction: f64, segments: usize, color: &str) -> String {
    let full = blocks(fraction, segments);
    format!("{}{}", sgr(color, &"\u{2588}".repeat(full)), paint(&"\u{2591}".repeat(segments - full), Color::Gray))
}

fn vram_bar(goethe_mb: u64, used_mb: u64, total_mb: u64) -> String {
    // Goethe's models in the theme's main color, other processes in white, free memory dim and dotted.
    let total = total_mb.max(1) as f64;
    let all = blocks(used_mb as f64 / total, VRAM_SEGMENTS);
    let ours = blocks(goethe_mb as f64 / total, VRAM_SEGMENTS).min(all);
    let ours = if goethe_mb > 0 && ours == 0 && all > 0 { 1 } else { ours };
    format!(
        "{}{}{}",
        sgr(&crate::theme::main(), &"\u{2588}".repeat(ours)),
        "\u{2588}".repeat(all - ours),
        paint(&"\u{2591}".repeat(VRAM_SEGMENTS - all), Color::Gray)
    )
}

pub fn legend() -> String {
    format!(
        "{} goethe models   \u{2588} other processes   {} free   {} {}",
        sgr(&crate::theme::main(), "\u{2588}"),
        paint("\u{2591}", Color::Gray),
        sgr(&crate::theme::secondary(), "\u{2588}"),
        if cfg!(target_os = "macos") { "GPU busy" } else { "memory bandwidth" }
    )
}

fn gb(mb: u64) -> String {
    format!("{:.1} GB", mb as f64 / 1024.0)
}

fn size(gb: f64) -> String {
    if gb >= 1000.0 {
        format!("{:.1} TB", gb / 1000.0)
    } else {
        format!("{gb:.1} GB")
    }
}

pub(crate) fn section(icon: &str, title: &str, width: usize) -> String {
    // "◉ CPU  ╌╌╌╌", the title in the brand blue and a dashed rule to the width.
    let head = format!("{icon} {title}  ");
    format!("{}{}", blue(&head), paint(&"\u{254c}".repeat(width.saturating_sub(head.chars().count())), Color::Gray))
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuLine {
    pub index: u32,
    pub used_mb: u64,
    pub goethe_mb: u64,
    pub total_mb: u64,
    pub processes: Vec<Proc>,
    pub denied: bool,
    pub bandwidth: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Cpu {
    pub total: f64,
    pub top: Vec<(usize, f64)>,
    pub load: (f64, f64, f64),
    pub cores: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Memory {
    pub total_kb: u64,
    pub available_kb: u64,
    pub cached_kb: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Summary {
    pub running: bool,
    pub environment: String,
    pub gpus: usize,
    pub vram_mb: u64,
    pub memory_word: &'static str,
    pub disk_gb: Option<f64>,
    pub os: String,
}

pub struct StatusView {
    pub summary: Summary,
    pub gpus: Vec<GpuLine>,
    pub cpu: Cpu,
    pub memory: Memory,
    stat: Option<(Vec<u64>, Vec<Vec<u64>>)>,
    managed: HashSet<i32>,
    models: HashMap<i32, String>,
    denied: HashSet<u32>,
    totals: HashMap<u32, u64>,
    slow_at: Option<Instant>,
}

fn environment(home: &Path) -> String {
    // Bare metal, a cloud VM (by provider) or a container, from the hardware profile; a Mac says laptop or desktop.
    if cfg!(target_os = "macos") {
        return mac_environment();
    }
    let env = crate::hw::load_hardware_state(home).ok().and_then(|h| h.environment);
    match env {
        Some(e) if e.container.is_some() => format!("container ({})", e.container.unwrap_or_default()),
        Some(e) if e.cloud != "none" && !e.cloud.is_empty() => format!("cloud VM ({})", e.cloud),
        _ => "bare metal".to_string(),
    }
}

fn mac_environment() -> String {
    #[cfg(target_os = "macos")]
    return crate::apple::environment_label(crate::apple::form_factor());
    #[cfg(not(target_os = "macos"))]
    String::new()
}

fn disk_gb(path: &Path) -> Option<f64> {
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    (unsafe { libc::statvfs(c.as_ptr(), &mut st) } == 0).then(|| st.f_blocks as f64 * st.f_frsize as f64 / 1e9)
}

fn proc_stat() -> Option<(Vec<u64>, Vec<Vec<u64>>)> {
    crate::sys::cpu_times()
}

fn busy(before: &[u64], after: &[u64]) -> f64 {
    // Share of time not idle or waiting between two samples.
    let sum = |v: &[u64]| v.iter().sum::<u64>();
    let idle = |v: &[u64]| v.get(3).copied().unwrap_or(0) + v.get(4).copied().unwrap_or(0);
    let total = sum(after).saturating_sub(sum(before));
    if total == 0 {
        return 0.0;
    }
    1.0 - idle(after).saturating_sub(idle(before)) as f64 / total as f64
}

pub fn meminfo(text: &str) -> Memory {
    let field = |k: &str| text.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse().ok()).unwrap_or(0);
    Memory { total_kb: field("MemTotal:"), available_kb: field("MemAvailable:"), cached_kb: field("Cached:") + field("Buffers:") }
}

fn load() -> (f64, f64, f64) {
    crate::sys::loadavg()
}

fn instance_of(pid: i32, models: &HashMap<i32, String>) -> Option<(i32, String)> {
    // The instance a GPU process belongs to: its leader's pid and model, found through its parents (scheduler and tensor-parallel ranks).
    let mut at = pid;
    for _ in 0..12 {
        if let Some(m) = models.get(&at) {
            return Some((at, m.clone()));
        }
        at = crate::proc_ident::ppid(at).filter(|p| *p > 1)?;
    }
    None
}

fn role(pid: i32) -> String {
    // What a child of an engine does, from its process title: sglang::scheduler_TP1 is "scheduler TP1".
    let title = crate::sys::argv(pid).and_then(|a| a.into_iter().next()).unwrap_or_default();
    let name = title.rsplit("::").next().unwrap_or("").trim();
    if name.is_empty() || name.contains('/') { "worker".to_string() } else { name.replace('_', " ") }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Proc {
    Foreign(u32),
    Model(u32, String),
    Part(u32, String),
}

pub fn group_processes(procs: &[(u32, bool)], owner: impl Fn(u32) -> Option<(i32, String)>, role: impl Fn(u32) -> String) -> Vec<Proc> {
    // A goethe instance's leader with its model, then each of its other processes on this GPU marked by role (the scheduler); other processes one each.
    let mut ours: Vec<(u32, u32, String)> = Vec::new();
    let mut rest = Vec::new();
    for &(pid, mine) in procs {
        match (mine, owner(pid)) {
            (true, Some((leader, model))) => ours.push((leader as u32, pid, model)),
            (true, None) => rest.push(Proc::Model(pid, String::new())),
            (false, _) => rest.push(Proc::Foreign(pid)),
        }
    }
    let mut leaders: Vec<(u32, String)> = Vec::new();
    for (leader, _, model) in &ours {
        if !leaders.iter().any(|l| l.0 == *leader) {
            leaders.push((*leader, model.clone()));
        }
    }
    let mut out = Vec::new();
    for (leader, model) in leaders {
        out.push(Proc::Model(leader, model));
        out.extend(ours.iter().filter(|(l, pid, _)| *l == leader && *pid != leader).map(|(_, pid, _)| Proc::Part(*pid, role(*pid))));
    }
    out.extend(rest);
    out
}

impl StatusView {
    pub fn new() -> StatusView {
        let mut v = StatusView {
            summary: Summary::default(),
            gpus: Vec::new(),
            cpu: Cpu::default(),
            memory: Memory::default(),
            stat: None,
            managed: HashSet::new(),
            models: HashMap::new(),
            denied: HashSet::new(),
            totals: HashMap::new(),
            slow_at: None,
        };
        v.refresh();
        v
    }

    pub fn refresh(&mut self) {
        // VRAM and bandwidth on every call (every 100 ms while the page is open); processes, CPU, memory and the summary once a second.
        if self.slow_at.is_none_or(|t| t.elapsed() >= SLOW) {
            self.slow_at = Some(Instant::now());
            self.slow();
        }
        let usage = crate::hw::gpu_usage(&self.managed).unwrap_or_default();
        self.gpus = usage
            .iter()
            .map(|u| GpuLine {
                index: u.index,
                used_mb: u.used_mb,
                goethe_mb: u.managed_mb,
                total_mb: self.totals.get(&u.index).copied().unwrap_or(u.used_mb),
                processes: group_processes(&u.processes, |pid| instance_of(pid as i32, &self.models), |pid| role(pid as i32)),
                denied: self.denied.contains(&u.index),
                bandwidth: u.memory_util,
            })
            .collect();
    }

    fn slow(&mut self) {
        let home = goethe_home();
        let con = crate::db::open(&home).ok();
        self.managed = con.as_ref().map(crate::placement::managed_pids).unwrap_or_default();
        self.denied = con.as_ref().and_then(|c| crate::gpu_policy::denied_set(c).ok()).unwrap_or_default();
        self.models = con
            .as_ref()
            .and_then(|c| {
                let mut st = c.prepare("SELECT pid, model_id FROM loaded_instances WHERE status IN ('STARTING','RUNNING','RESTARTING','DRAINING')").ok()?;
                let rows = st.query_map([], |r| Ok((r.get::<_, i32>(0)?, r.get::<_, String>(1)?))).ok()?;
                Some(rows.flatten().collect())
            })
            .unwrap_or_default();
        let hw = crate::hw::load_hardware_state(&home).ok();
        if let Some(h) = &hw {
            self.totals = h.gpus.iter().map(|g| (g.index, g.vram_total_mb)).collect();
        }
        self.summary = Summary {
            running: crate::service::goethed_running(),
            environment: environment(&home),
            gpus: self.totals.len(),
            vram_mb: self.totals.values().sum(),
            memory_word: hw.as_ref().map_or("VRAM", |h| crate::gpu_kind::host_memory_word(&h.gpus)),
            disk_gb: disk_gb(&home),
            os: hw.and_then(|h| h.host).map(|h| h.os.distro).unwrap_or_default(),
        };
        if let Some(now) = proc_stat() {
            if let Some((t0, c0)) = &self.stat {
                let mut cores: Vec<(usize, f64)> = now.1.iter().zip(c0).enumerate().map(|(i, (a, b))| (i, busy(b, a))).collect();
                cores.sort_by(|a, b| b.1.total_cmp(&a.1));
                cores.truncate(2);
                self.cpu = Cpu { total: busy(t0, &now.0), top: cores, load: load(), cores: now.1.len() };
            } else {
                self.cpu = Cpu { load: load(), cores: now.1.len(), ..Cpu::default() };
            }
            self.stat = Some(now);
        }
        let m = crate::sys::memory();
        self.memory = Memory { total_kb: m.total_kb, available_kb: m.available_kb, cached_kb: m.cached_kb };
    }
}

impl Default for StatusView {
    fn default() -> Self {
        StatusView::new()
    }
}

pub fn width_for(terminal: usize) -> usize {
    terminal.saturating_sub(4).min(MAX_WIDTH)
}

pub fn summary_line(s: &Summary, width: usize) -> String {
    // Status, environment, GPUs, VRAM (GPU memory when unified), disk and OS on one line; the last items give way when the terminal is narrow.
    let state = if s.running { paint("running", Color::Green) } else { paint("stopped", Color::Red) };
    let label = |t: &str| paint(t, Color::Gray);
    let mut parts = vec![
        format!("{} {state}", blue("Status")),
        format!("{} {}", label("Environment"), s.environment),
        format!("{} {}", label("GPUs"), s.gpus),
        format!("{} {}", label(s.memory_word), gb(s.vram_mb)),
    ];
    if let Some(d) = s.disk_gb {
        parts.push(format!("{} {}", label("Disk"), size(d)));
    }
    if !s.os.is_empty() {
        parts.push(format!("{} {}", label("OS"), s.os));
    }
    while parts.len() > 1 && visible_width(&parts.join(GAP)) > width {
        parts.pop();
    }
    parts.join(GAP)
}

pub fn gpu_lines(gpus: &[GpuLine], width: usize) -> Vec<String> {
    // One block per GPU: VRAM used, a 10-block bar and the total, then each process on its own line, the status, and the bandwidth bar with its percent.
    let used_w = gpus.iter().map(|g| gb(g.used_mb).len()).max().unwrap_or(7);
    let total_w = gpus.iter().map(|g| gb(g.total_mb).len()).max().unwrap_or(7);
    let name_w = gpus.iter().map(|g| format!("gpu{}", g.index).len()).max().unwrap_or(4);
    let tail = GAP.len() + 7 + GAP.len() + BAND_SEGMENTS + 1 + 4;
    let fixed = name_w + GAP.len() + used_w + 1 + VRAM_SEGMENTS + 1 + total_w + GAP.len();
    let proc_w = width.saturating_sub(fixed + tail).max(8);
    let mut out = vec![section("\u{25a3}", "GPU", width)];
    for g in gpus {
        let vram = format!("{}{GAP}{} {} {}", pad(&format!("gpu{}", g.index), name_w), rpad(&gb(g.used_mb), used_w), vram_bar(g.goethe_mb, g.used_mb, g.total_mb), rpad(&gb(g.total_mb), total_w));
        let procs: Vec<String> = if g.processes.is_empty() {
            vec![paint("no processes", Color::Gray)]
        } else {
            g.processes
                .iter()
                .map(|p| match p {
                    Proc::Foreign(pid) => paint(&pid.to_string(), Color::Red),
                    Proc::Model(pid, m) if m.is_empty() => format!("{pid} goethe"),
                    Proc::Model(pid, m) => format!("{pid} {m}"),
                    Proc::Part(pid, r) => format!("{pid} {}", paint(&format!("\u{2514} {r}"), Color::Gray)),
                })
                .collect()
        };
        let status = if g.denied { paint("DENIED ", Color::Red) } else { "ALLOWED".to_string() };
        let band = match g.bandwidth {
            Some(p) => format!("{} {}", colored_bar(p as f64 / 100.0, BAND_SEGMENTS, &crate::theme::secondary()), rpad(&format!("{p}%"), 4)),
            None => format!("{} {}", paint(&"\u{2591}".repeat(BAND_SEGMENTS), Color::Gray), rpad("-", 4)),
        };
        for (i, p) in procs.iter().enumerate() {
            let p = clip(p, proc_w);
            let line = if i == 0 {
                format!("{vram}{GAP}{}{GAP}{status}{GAP}{band}", pad(&p, proc_w))
            } else {
                format!("{}{GAP}{p}", " ".repeat(fixed - GAP.len()))
            };
            out.push(line.trim_end().to_string());
        }
    }
    if gpus.is_empty() {
        out.push(paint("no GPU found (goethe doctor says why)", Color::Gray));
    }
    out
}

pub(crate) fn clip(text: &str, width: usize) -> String {
    // Cut a colored cell to its width with an ellipsis, dropping its colors only when it is too long.
    if visible_width(text) <= width {
        return text.to_string();
    }
    let plain: String = strip(text);
    plain.chars().take(width.saturating_sub(1)).chain(['\u{2026}']).collect()
}

pub(crate) fn strip(s: &str) -> String {
    let mut out = String::new();
    let mut esc = false;
    for c in s.chars() {
        match (esc, c) {
            (false, '\x1b') => esc = true,
            (true, 'm') => esc = false,
            (true, _) => {}
            (false, c) => out.push(c),
        }
    }
    out
}

pub fn cpu_lines(c: &Cpu, width: usize) -> Vec<String> {
    let row = |label: &str, f: f64| format!("{:<7}{} {:>6}", label, bar(f, HALF_SEGMENTS), format!("{:.1}%", f * 100.0));
    let mut out = vec![section("\u{25a3}", "CPU", width), row("Total", c.total)];
    for (i, f) in &c.top {
        out.push(row(&format!("Core{}", i + 1), *f));
    }
    out.push(format!("{:<7}{:.2} / {:.2} / {:.2}, {} cores", "Load", c.load.0, c.load.1, c.load.2, c.cores));
    out
}

pub fn memory_lines(m: &Memory, width: usize) -> Vec<String> {
    let total = m.total_kb.max(1) as f64;
    let used = m.total_kb.saturating_sub(m.available_kb) as f64;
    let kgb = |kb: f64| format!("{:.1} GB", kb / 1024.0 / 1024.0);
    let row = |label: &str, f: f64| format!("{:<7}{} {:>6}", label, bar(f, HALF_SEGMENTS), format!("{:.1}%", f * 100.0));
    vec![
        section("\u{25a3}", "Memory", width),
        row("Used", used / total),
        row("Free", 1.0 - used / total),
        format!("{:<7}{} / {}", "Total", kgb(used), kgb(total)),
        format!("{:<7}{} \u{b7} Avail {}", "Cache", kgb(m.cached_kb as f64), kgb(m.available_kb as f64)),
    ]
}

pub fn render(v: &StatusView, width: usize) -> Vec<String> {
    // The summary, four empty lines, the GPU table, then CPU and memory side by side in the same total width.
    let w = width_for(width);
    let mut out = vec![String::new(), format!("  {}", summary_line(&v.summary, w)), String::new(), String::new(), String::new(), String::new()];
    out.extend(gpu_lines(&v.gpus, w).into_iter().map(|l| format!("  {l}")));
    out.push(String::new());
    let half = (w - GAP.len()) / 2;
    let (left, right) = (cpu_lines(&v.cpu, half), memory_lines(&v.memory, half));
    for i in 0..left.len().max(right.len()) {
        let l = left.get(i).map(|s| clip(s, half)).unwrap_or_default();
        let r = right.get(i).map(|s| clip(s, half)).unwrap_or_default();
        out.push(format!("  {}{GAP}{r}", pad(&l, half)).trim_end().to_string());
    }
    out.push(String::new());
    out.push(format!("  {}", clip(&legend(), w)));
    out
}

pub fn run_gpus() -> Result<(), String> {
    // goethe gpu on a terminal: the GPU table, live every 100 ms until q or the left arrow; once when the output is not a terminal.
    let mut view = StatusView::new();
    crate::term::live(FAST, || {
        view.refresh();
        let (w, _) = crate::keys_menu::screen_size();
        Ok(gpu_render(&view, w).join("\n"))
    })
}

pub fn gpu_render(v: &StatusView, width: usize) -> Vec<String> {
    // goethe gpu: the GPU table of the Status page and its legend.
    let w = width_for(width);
    let mut out = vec![String::new()];
    out.extend(gpu_lines(&v.gpus, w).into_iter().map(|l| format!("  {l}")));
    out.push(String::new());
    out.push(format!("  {}", clip(&legend(), w)));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view() -> StatusView {
        StatusView {
            summary: Summary { running: true, environment: "bare metal".into(), gpus: 2, vram_mb: 2 * 97_887, memory_word: "VRAM", disk_gb: Some(6000.0), os: "Ubuntu 24.04.4 LTS".into() },
            gpus: vec![
                GpuLine { index: 0, used_mb: 93_696, goethe_mb: 0, total_mb: 97_887, processes: vec![Proc::Foreign(3909)], denied: true, bandwidth: Some(7) },
                GpuLine { index: 3, used_mb: 40_000, goethe_mb: 38_000, total_mb: 97_887, processes: vec![Proc::Model(3164098, "nvidia/Qwen3.8-27B-NVFP4".into()), Proc::Model(3164111, "nvidia/Qwen3.8-27B-NVFP4".into())], denied: false, bandwidth: Some(63) },
            ],
            cpu: Cpu { total: 0.116, top: vec![(0, 0.362), (1, 0.349)], load: (1.93, 1.86, 1.61), cores: 64 },
            memory: Memory { total_kb: 16 * 1024 * 1024, available_kb: 5_368_709, cached_kb: 2_621_440 },
            stat: None,
            managed: HashSet::new(),
            models: HashMap::new(),
            denied: HashSet::new(),
            totals: HashMap::new(),
            slow_at: None,
        }
    }

    fn plain(lines: &[String]) -> Vec<String> {
        lines.iter().map(|l| strip(l)).collect()
    }

    #[test]
    fn the_summary_is_one_line_with_status_environment_gpus_vram_disk_and_os() {
        let t = strip(&summary_line(&view().summary, 200));
        assert_eq!(t, "Status running   Environment bare metal   GPUs 2   VRAM 191.2 GB   Disk 6.0 TB   OS Ubuntu 24.04.4 LTS");
        let stopped = Summary { running: false, ..view().summary };
        assert!(strip(&summary_line(&stopped, 200)).starts_with("Status stopped"));
        assert_eq!(strip(&summary_line(&view().summary, 80)), "Status running   Environment bare metal   GPUs 2   VRAM 191.2 GB   Disk 6.0 TB", "the OS gives way first");
    }

    #[test]
    fn four_empty_lines_then_the_gpu_table_with_one_line_per_process() {
        let t = plain(&render(&view(), 124));
        assert!(t[1].contains("Status running") && t[2..6].iter().all(String::is_empty), "{t:#?}");
        assert!(t[6].trim_start().starts_with("\u{25a3} GPU  \u{254c}"), "{t:#?}");
        let g0 = t.iter().find(|l| l.contains("gpu0")).unwrap();
        assert!(g0.contains("91.5 GB \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588} 95.6 GB"), "{g0}");
        assert!(g0.contains("3909   ") && g0.contains("DENIED") && g0.ends_with("7%"), "{g0}");
        let g3 = t.iter().position(|l| l.contains("gpu3")).unwrap();
        assert!(t[g3].contains("3164098 nvidia/Qwen3.8-27B-NVFP4") && t[g3].contains("ALLOWED") && t[g3].ends_with("63%"), "{}", t[g3]);
        assert!(t[g3 + 1].trim_start().starts_with("3164111 nvidia/Qwen3.8-27B-NVFP4"), "the second process on its own line: {}", t[g3 + 1]);
        let col = |l: &str, s: &str| l.find(s).map(|b| l[..b].chars().count()).unwrap();
        assert_eq!(col(&t[g3 + 1], "3164111"), col(&t[g3], "3164098"), "processes line up");
        assert_eq!(col(g0, "DENIED"), col(&t[g3], "ALLOWED"), "status lines up");
        assert_eq!(col(g0, "GB \u{2588}"), col(&t[g3], "GB \u{2588}"), "bars line up");
    }

    #[test]
    fn cpu_and_memory_sit_side_by_side_like_mole() {
        let t = plain(&render(&view(), 124));
        let head = t.iter().position(|l| l.contains("\u{25a3} CPU")).unwrap();
        assert!(t[head].contains("\u{25a3} Memory"), "{}", t[head]);
        assert!(t[head + 1].contains("Total  \u{2588}\u{2588}") && t[head + 1].contains("11.6%") && t[head + 1].contains("Used   ") && t[head + 1].contains("68.0%"), "{}", t[head + 1]);
        assert!(t[head + 2].contains("Core1") && t[head + 2].contains("36.2%") && t[head + 2].contains("Free") && t[head + 2].contains("32.0%"), "{}", t[head + 2]);
        assert!(t[head + 3].contains("Core2") && t[head + 3].contains("Total  10.9 GB / 16.0 GB"), "{}", t[head + 3]);
        assert!(t[head + 4].contains("Load   1.93 / 1.86 / 1.61, 64 cores") && t[head + 4].contains("Cache  2.5 GB \u{b7} Avail 5.1 GB"), "{}", t[head + 4]);
        let col = |l: &str, s: &str| l.find(s).map(|b| l[..b].chars().count()).unwrap();
        assert_eq!(col(&t[head], "\u{25a3} Memory"), col(&t[head + 1], "Used"), "memory starts at one column");
    }

    #[test]
    fn nothing_overflows_at_any_width() {
        let mut v = view();
        v.gpus[1].processes.push(Proc::Model(1, "some-organisation/an-extremely-long-model-name-that-goes-on-and-on-NVFP4".into()));
        for w in [80, 100, 124, 160, 240] {
            let lines = render(&v, w);
            for l in &lines {
                assert!(visible_width(l) <= w, "{w}: {} > {w}: {l}", visible_width(l));
            }
            let t = plain(&lines);
            let rule = t.iter().find(|l| l.contains("\u{25a3} GPU")).unwrap();
            assert_eq!(rule.chars().count(), 2 + width_for(w), "the GPU rule spans the table: {rule}");
            let cpu = t.iter().find(|l| l.contains("\u{25a3} CPU")).unwrap();
            assert_eq!(cpu.chars().count(), 2 + width_for(w) - (width_for(w) - GAP.len()) % 2, "CPU and memory together span the same width");
        }
    }

    #[test]
    fn a_unified_host_says_gpu_memory() {
        let mac = Summary { memory_word: "GPU memory", environment: "macOS laptop".into(), gpus: 1, ..view().summary };
        assert!(strip(&summary_line(&mac, 200)).contains("Environment macOS laptop   GPUs 1   GPU memory 191.2 GB"));
    }

    #[test]
    fn goethe_vram_is_its_own_part_of_the_bar_and_a_legend_ends_the_page() {
        assert_eq!(strip(&vram_bar(38_000, 40_000, 97_887)), "\u{2588}\u{2588}\u{2588}\u{2588}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}");
        assert_eq!(strip(&vram_bar(500, 50_000, 100_000)), "\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}", "a small share still gets a block");
        let t = plain(&render(&view(), 124));
        let last = t.iter().rposition(|l| !l.is_empty()).unwrap();
        assert!(t[last].contains("goethe models") && t[last].contains("other processes") && t[last].contains("free") && t[last].contains(if cfg!(target_os = "macos") { "GPU busy" } else { "memory bandwidth" }), "{}", t[last]);
        let g = plain(&gpu_render(&view(), 124));
        assert!(g.iter().any(|l| l.contains("gpu3")) && g.iter().any(|l| l.contains("other processes")) && !g.iter().any(|l| l.contains("CPU")));
    }

    #[test]
    fn an_engines_scheduler_is_marked_under_its_model() {
        let owner = |pid: u32| match pid {
            100 | 101 => Some((100, "org/m".to_string())),
            200 => Some((200, "org/n".to_string())),
            _ => None,
        };
        let rows = group_processes(&[(101, true), (100, true), (300, false), (200, true)], owner, |_| "scheduler".to_string());
        assert_eq!(rows, vec![Proc::Model(100, "org/m".into()), Proc::Part(101, "scheduler".into()), Proc::Model(200, "org/n".into()), Proc::Foreign(300)]);
        let only_child = group_processes(&[(101, true)], owner, |_| "scheduler".to_string());
        assert_eq!(only_child, vec![Proc::Model(100, "org/m".into()), Proc::Part(101, "scheduler".into())], "the model is named even when only its scheduler holds this GPU");
        let mut v = view();
        v.gpus[1].processes = rows;
        let t = plain(&render(&v, 124));
        let at = t.iter().position(|l| l.contains("100 org/m")).unwrap();
        assert!(t[at + 1].trim_start() == "101 \u{2514} scheduler", "{}", t[at + 1]);
    }

    #[test]
    fn memory_is_read_from_proc_meminfo() {
        let m = meminfo("MemTotal:       16384000 kB\nMemFree:  100 kB\nMemAvailable:    5000000 kB\nBuffers:    1000 kB\nCached:     2000 kB\n");
        assert_eq!((m.total_kb, m.available_kb, m.cached_kb), (16_384_000, 5_000_000, 3000));
        assert!((busy(&[10, 0, 10, 80, 0], &[20, 0, 20, 160, 0]) - 0.2).abs() < 1e-9);
    }
}
