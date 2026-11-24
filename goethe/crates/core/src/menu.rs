// goethe with no arguments on a terminal: the main menu (Ask, Quick setup, Install models, API keys and gateway, Status, Metrics) and its pages.

use crate::keys_menu::{fit, frame_update, parse_input, screen_size, window, Input, LineEdit, Screen};
use crate::models::{PickData, PickRow};
use crate::paths::goethe_home;
use crate::term::{paint, Color, RawMode};
use std::io::{IsTerminal, Write};

#[path = "menu_draw.rs"]
mod menu_draw;
use menu_draw::{brand, columns, drop_mark, example_below, fine_bar, gb, gray, hl, mark, name_style, screen, sgr, themes_line, GAP};
#[cfg(test)]
use menu_draw::{plain, LOGO, SITE};

enum Flow {
    Stay,
    Leave,
    Quit,
    Pick(usize),
}

trait Page {
    fn render(&self, width: usize, height: usize) -> Vec<String>;
    fn key(&mut self, input: Input) -> Flow;
    fn keys(&mut self, inputs: &[Input]) -> Flow {
        // One read can hold a paste. Default: handle each key; stop when the page leaves.
        let mut flow = Flow::Stay;
        for input in inputs {
            flow = self.key(*input);
            if !matches!(flow, Flow::Stay) {
                break;
            }
        }
        flow
    }
    fn every(&self) -> Option<std::time::Duration> {
        // How often the page refreshes itself while it is open; None for pages that only change on a key.
        None
    }
    fn tick(&mut self) {}
    fn finished(&self) -> bool {
        // A page that closes itself (a load that reached RUNNING) once this turns true.
        false
    }
}

struct Tty {
    screen: Option<Screen>,
    raw: Option<RawMode>,
    last: Vec<String>,
    quit: bool,
}

impl Tty {
    fn open() -> Result<Tty, String> {
        let mut t = Tty { screen: None, raw: None, last: Vec::new(), quit: false };
        t.resume()?;
        Ok(t)
    }

    fn resume(&mut self) -> Result<(), String> {
        // Raw keys on the alternate screen; one session stays open from page to page, so moving between them never shows the shell.
        if self.raw.is_none() {
            self.raw = Some(RawMode::enter().ok_or("could not switch the terminal to raw mode")?);
        }
        if self.screen.is_none() {
            self.screen = Some(Screen::open());
        }
        self.last.clear();
        Ok(())
    }

    fn inside(&mut self, f: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
        self.inside_with(f, true)
    }

    fn inside_with(&mut self, f: impl FnOnce() -> Result<(), String>, wait: bool) -> Result<(), String> {
        // Something that prints and asks as a normal command does (setup, status, an install), on the cleared menu screen; then the left arrow goes back, at once when it was left with the left arrow.
        self.raw = None;
        print!("\x1b[H\x1b[2J\x1b[?25h");
        let _ = std::io::stdout().flush();
        let out = f();
        let cancelled = out.as_ref().is_err_and(|e| e.starts_with("cancelled"));
        report(out.map_err(|e| if cancelled { String::new() } else { e }));
        print!("\x1b[?25l");
        if wait && !cancelled {
            print!("\n\n  {}", gray("\u{2190} back"));
        }
        let _ = std::io::stdout().flush();
        self.resume()?;
        if wait && !cancelled {
            self.quit |= wait_back();
        }
        Ok(())
    }

    fn wait_back(&mut self) -> Result<(), String> {
        // The back hint under what was printed, in raw mode (so each line starts at the margin), then a key to go back.
        print!("\r\n\r\n  {}", gray("\u{2190} back"));
        let _ = std::io::stdout().flush();
        self.quit |= wait_back();
        Ok(())
    }

    fn draw(&mut self, page: &dyn Page) {
        let (w, h) = screen_size();
        let next = page.render(w, h);
        let update = frame_update(&self.last, &next);
        if !update.is_empty() {
            print!("{update}");
            let _ = std::io::stdout().flush();
        }
        self.last.clear();
        self.last.extend(next);
    }

    fn run(&mut self, page: &mut impl Page) -> Result<Option<usize>, String> {
        // Keys go to the page until it leaves, picks something or q quits goethe; a page that refreshes itself is ticked while it is open.
        let mut size = screen_size();
        self.last.clear();
        self.draw(page);
        let mut buf = [0u8; 4096];
        let mut next = std::time::Instant::now();
        loop {
            let every = page.every();
            let left = next.saturating_duration_since(std::time::Instant::now()).as_millis() as i32;
            let mut fds = libc::pollfd { fd: 0, events: libc::POLLIN, revents: 0 };
            if unsafe { libc::poll(&mut fds, 1, if every.is_some() { left.max(1) } else { 250 }) } <= 0 {
                if screen_size() != size {
                    size = screen_size();
                    self.last.clear();
                }
                if let Some(e) = every.filter(|_| std::time::Instant::now() >= next) {
                    page.tick();
                    next = std::time::Instant::now() + e;
                }
                self.draw(page);
                if page.finished() {
                    return Ok(None);
                }
                continue;
            }
            let n = match crate::term::read_keys(&mut buf) {
                Ok(0) => return Ok(None),
                Ok(n) => n,
                Err(e) => return Err(e.to_string()),
            };
            match page.keys(&parse_input(&buf[..n])) {
                Flow::Stay => {}
                Flow::Leave => return Ok(None),
                Flow::Quit => {
                    self.quit = true;
                    return Ok(None);
                }
                Flow::Pick(i) => return Ok(Some(i)),
            }
            self.draw(page);
        }
    }
}

struct StatusPage(crate::status_view::StatusView);

impl Page for StatusPage {
    fn render(&self, width: usize, height: usize) -> Vec<String> {
        screen(crate::status_view::render(&self.0, width), width, height, gray("\u{2190} back"))
    }

    fn key(&mut self, input: Input) -> Flow {
        match input {
            Input::Left | Input::Esc => Flow::Leave,
            Input::Interrupt | Input::Char('q') => Flow::Quit,
            _ => Flow::Stay,
        }
    }

    fn every(&self) -> Option<std::time::Duration> {
        Some(crate::status_view::FAST)
    }

    fn tick(&mut self) {
        self.0.refresh();
    }
}

struct MetricsPage(crate::metrics_view::MetricsView);

impl Page for MetricsPage {
    fn render(&self, width: usize, height: usize) -> Vec<String> {
        screen(crate::metrics_view::render(&self.0, width), width, height, gray("\u{2191}\u{2193} range   \u{2190} back   q quit"))
    }

    fn key(&mut self, input: Input) -> Flow {
        match input {
            Input::Left | Input::Esc => Flow::Leave,
            Input::Interrupt | Input::Char('q') => Flow::Quit,
            Input::Up | Input::Char('k') => {
                self.0.step(true);
                Flow::Stay
            }
            Input::Down | Input::Char('j') => {
                self.0.step(false);
                Flow::Stay
            }
            _ => Flow::Stay,
        }
    }

    fn every(&self) -> Option<std::time::Duration> {
        Some(crate::metrics_view::REFRESH)
    }

    fn tick(&mut self) {
        self.0.refresh();
    }
}

struct Notice(String);

impl Page for Notice {
    fn render(&self, width: usize, height: usize) -> Vec<String> {
        let mut out = brand();
        out.push(format!("  {}", gray(&self.0)));
        screen(out, width, height, String::new())
    }

    fn key(&mut self, _: Input) -> Flow {
        Flow::Stay
    }
}

struct Choice {
    title: &'static str,
    detail: &'static str,
    side: Option<Side>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    Schemas,
    Models,
}

struct SideState {
    kind: Side,
    at: usize,
    models: Vec<(String, bool)>,
    marked: Vec<String>,
    base: String,
    focused: bool,
}

impl SideState {
    fn open(kind: Side) -> SideState {
        let models = if kind == Side::Models { installed_and_loaded(&goethe_home()) } else { Vec::new() };
        SideState { kind, at: 0, models, marked: Vec::new(), base: String::new(), focused: false }
    }

    fn len(&self) -> usize {
        match self.kind {
            Side::Schemas => crate::api_schema::NAMES.len(),
            Side::Models => self.models.len(),
        }
    }
}

pub(crate) fn installed_and_loaded(home: &std::path::Path) -> Vec<(String, bool)> {
    // Installed models by id, each with whether an instance of it is loaded (not failed or cancelled), loaded ones first.
    let Ok(con) = crate::db::open(home) else { return Vec::new() };
    let loaded: Vec<String> = con
        .prepare("SELECT DISTINCT model_id FROM loaded_instances WHERE status NOT IN ('FAILED', 'CANCELLED')")
        .and_then(|mut st| st.query_map([], |r| r.get::<_, String>(0))?.collect())
        .unwrap_or_default();
    let mut rows: Vec<(String, bool)> = crate::models::installed_ids(home)
        .into_iter()
        .map(|id| {
            let on = loaded.contains(&id);
            (id, on)
        })
        .collect();
    rows.sort_by_key(|(_, on)| !*on);
    rows
}

fn side_rows(side: &SideState, job: Option<&LoadJob>) -> Vec<String> {
    // The submenu column: the schema names, or the installed models with "loaded", or the bar of the one loading, beside them.
    let focused = side.focused;
    let cursor = |i: usize| if focused && i == side.at { sgr(&hl(), "\u{27a4}") } else { " ".to_string() };
    let shade = |i: usize, text: &str| match (focused, i == side.at) {
        (true, true) => sgr(&hl(), text),
        (true, false) => text.to_string(),
        (false, _) => gray(text),
    };
    match side.kind {
        Side::Schemas => crate::api_schema::NAMES.iter().enumerate().map(|(i, n)| format!("{} {}", cursor(i), shade(i, n))).collect(),
        Side::Models => side
            .models
            .iter()
            .enumerate()
            .map(|(i, (id, loaded))| {
                let mark = if side.marked.contains(id) { sgr(&mark(), "\u{2713}") } else { " ".to_string() };
                let state = match job.filter(|j| j.model == *id) {
                    Some(j) => format!("  {}  {:>3}%", fine_bar(j.shown), j.shown),
                    None if *loaded => format!("  {}", sgr(&crate::theme::secondary(), "loaded")),
                    None => String::new(),
                };
                format!("{} {mark} {}{state}", cursor(i), shade(i, id))
            })
            .chain([String::new(), gray(GPU_HINT)])
            .collect(),
    }
}

const GPU_HINT: &str = "  Run: goethe gpu allow/deny gpuN to free up or deny GPUs";

enum Phase {
    Up(Option<std::process::Child>),
    Start,
    Loading { instance: String, gpus: Vec<u32>, need_mb: u64, base_mb: u64 },
    Done,
    Failed(String),
}

pub(crate) struct LoadJob {
    pub(crate) model: String,
    ctx: Option<u64>,
    gpus: Vec<u32>,
    phase: Phase,
    frac: f64,
    pub(crate) shown: u32,
    ticks: u32,
}

fn gpu_flag(gpus: &[u32]) -> Option<String> {
    // goethe load's --gpu for the GPUs chosen in the menu, none for goethe's own pick.
    (!gpus.is_empty()).then(|| format!("--gpu={}", gpus.iter().map(|g| format!("gpu{g}")).collect::<Vec<_>>().join(",")))
}

struct GpuRow {
    index: u32,
    name: String,
    free_gb: f64,
    allowed: bool,
    alone: Result<(), String>,
}

struct GpuChoice {
    model: String,
    rows: Vec<GpuRow>,
    at: usize,
    marked: Vec<u32>,
    plan: Result<String, String>,
    view: Option<crate::recommend::View>,
}

impl GpuChoice {
    fn open(model: String) -> GpuChoice {
        // Every GPU with its free memory and whether the model fits on it alone, as goethe load --dry-run would place it now.
        let home = goethe_home();
        let mut c = GpuChoice { model, rows: Vec::new(), at: 0, marked: Vec::new(), plan: Err(String::new()), view: None };
        let Some(hw) = crate::placement::load_hw(&home) else {
            c.plan = Err("no hardware profile yet: goethe doctor --hardware".to_string());
            return c;
        };
        let mut view = crate::recommend::View::new(&home, hw);
        let allowed = view.gpus.clone();
        for g in &view.hw.gpus {
            let live = allowed.iter().find(|a| a.index == g.index);
            view.gpus = live.into_iter().cloned().collect();
            let alone = match live {
                Some(_) => c.fit_on(&view, Some(1)).map(|_| ()),
                None => Err("not allowed".to_string()),
            };
            let free_mb = live.map_or(g.vram_free_mb, |l| l.free_vram_mb);
            c.rows.push(GpuRow { index: g.index, name: g.name.clone(), free_gb: free_mb as f64 / 1024.0, allowed: live.is_some(), alone });
        }
        view.gpus = allowed;
        c.view = Some(view);
        c.replan();
        c
    }

    fn fit_on(&self, view: &crate::recommend::View, tp: Option<usize>) -> Result<String, String> {
        match crate::hw::place(&goethe_home(), view, &self.model, tp, None, None) {
            Ok(Ok((f, _))) => Ok(format!("tp={}  ctx={}  {:.1} GB/GPU", f.tp, f.ctx, f.need.total())),
            Ok(Err(why)) | Err(why) => Err(why),
        }
    }

    fn replan(&mut self) {
        // The plan goethe load would use for Auto or for the marked GPUs: tp, context and memory per GPU.
        let Some(mut view) = self.view.take() else { return };
        let all = view.gpus.clone();
        if !self.marked.is_empty() {
            view.gpus.retain(|g| self.marked.contains(&g.index));
        }
        self.plan = self.fit_on(&view, (!self.marked.is_empty()).then_some(self.marked.len()));
        view.gpus = all;
        self.view = Some(view);
    }

    fn key(&mut self, input: Input) -> Option<bool> {
        // Some(true) loads, Some(false) goes back to the list, None stays.
        match input {
            Input::Up | Input::Char('k') => self.step(false),
            Input::Down | Input::Char('j') => self.step(true),
            Input::Space if self.at == 0 => {
                self.marked.clear();
                self.replan();
            }
            Input::Space if self.rows[self.at - 1].allowed => {
                let g = self.rows[self.at - 1].index;
                match self.marked.iter().position(|m| *m == g) {
                    Some(i) => drop(self.marked.remove(i)),
                    None => self.marked.push(g),
                }
                self.marked.sort_unstable();
                self.replan();
            }
            Input::Enter => return Some(true),
            Input::Left | Input::Esc => return Some(false),
            _ => {}
        }
        None
    }

    fn step(&mut self, down: bool) {
        // Move to the next entry that can be chosen: Auto or an allowed GPU; denied GPUs are skipped.
        let n = self.rows.len() + 1;
        for _ in 0..n {
            self.at = if down { (self.at + 1) % n } else { (self.at + n - 1) % n };
            if self.at == 0 || self.rows[self.at - 1].allowed {
                return;
            }
        }
    }

    fn lines(&self) -> Vec<String> {
        // Auto, then each GPU (gray when not allowed), then the plan the choice gives.
        let cursor = |i: usize| if i == self.at { sgr(&hl(), "\u{27a4}") } else { " ".to_string() };
        let dot = |on: bool| if on { sgr(&mark(), "\u{25cf}") } else { "\u{25cb}".to_string() };
        let mut out = vec![format!("{} {} Auto  {}", cursor(0), dot(self.marked.is_empty()), gray("goethe picks the GPUs"))];
        for (i, r) in self.rows.iter().enumerate() {
            let text = format!("GPU {}  {}  {:.1} GB free", r.index, r.name, r.free_gb);
            out.push(match (&r.alone, r.allowed) {
                (_, false) => format!("{}   {}", cursor(i + 1), gray(&format!("{text}  not allowed"))),
                (Ok(()), true) => format!("{} {} {text}  {}", cursor(i + 1), dot(self.marked.contains(&r.index)), sgr(&crate::theme::secondary(), "fits")),
                (Err(_), true) => format!("{} {} {text}  {}", cursor(i + 1), dot(self.marked.contains(&r.index)), gray("too small alone")),
            });
        }
        out.push(String::new());
        out.push(match &self.plan {
            Ok(desc) => format!("    {desc}"),
            Err(why) if why.is_empty() => String::new(),
            Err(why) => format!("    {}", paint(&format!("does not fit: {why}"), Color::Red)),
        });
        out
    }
}

fn goethe_cmd() -> std::process::Command {
    let mut c = std::process::Command::new(std::env::current_exe().unwrap_or_else(|_| "goethe".into()));
    c.stdin(std::process::Stdio::null()).stderr(std::process::Stdio::piped());
    c
}

impl LoadJob {
    pub(crate) fn new(model: String, ctx: Option<u64>) -> LoadJob {
        LoadJob { model, ctx, gpus: Vec::new(), phase: Phase::Up(None), frac: 0.0, shown: 0, ticks: 0 }
    }


    fn advance(&mut self) {
        // goethed up (started quietly when down), the load sent, then the instance's VRAM against its budget until it runs.
        self.phase = match std::mem::replace(&mut self.phase, Phase::Start) {
            Phase::Up(child) if crate::service::goethed_running() => {
                drop(child);
                Phase::Start
            }
            Phase::Up(None) => match goethe_cmd().args(["up", "--no-boot"]).stdout(std::process::Stdio::null()).spawn() {
                Ok(c) => Phase::Up(Some(c)),
                Err(e) => Phase::Failed(format!("could not start goethed: {e}")),
            },
            Phase::Up(Some(mut c)) => match c.try_wait() {
                Ok(Some(st)) if !st.success() => Phase::Failed("goethed did not start; goethe up shows why".to_string()),
                _ => Phase::Up(Some(c)),
            },
            Phase::Start => match goethe_cmd().args(["load", &self.model, "--no-wait", "--json", "--yes"]).args(self.ctx.map(|c| format!("--ctx={c}"))).args(gpu_flag(&self.gpus)).output() {
                Ok(o) if o.status.success() => match crate::rpc::parse_load_instance_id(&String::from_utf8_lossy(&o.stdout)) {
                    Some(instance) => Phase::Loading { instance, gpus: Vec::new(), need_mb: 0, base_mb: 0 },
                    None if String::from_utf8_lossy(&o.stdout).contains("already loaded") => {
                        self.frac = 1.0;
                        Phase::Done
                    }
                    None => Phase::Failed(String::from_utf8_lossy(&o.stdout).trim().to_string()),
                },
                Ok(o) => Phase::Failed(String::from_utf8_lossy(&o.stderr).trim().lines().last().unwrap_or("the load did not start").to_string()),
                Err(e) => Phase::Failed(e.to_string()),
            },
            Phase::Loading { instance, mut gpus, mut need_mb, mut base_mb } => {
                let ps = crate::rpc::call("ps", serde_json::json!(null)).ok().and_then(|r| serde_json::from_str::<serde_json::Value>(&r.stdout).ok());
                let row = ps.as_ref().and_then(|v| v.as_array()).and_then(|a| a.iter().find(|r| r["instance_id"] == instance.as_str()).cloned());
                match row.as_ref().and_then(|r| r["status"].as_str()) {
                    None => Phase::Failed(format!("{} stopped before it was ready", self.model)),
                    Some("RUNNING") => {
                        self.frac = 1.0;
                        Phase::Done
                    }
                    Some("FAILED") | Some("CANCELLED") => Phase::Failed(crate::load_wait::failure_report(&goethe_home(), &instance)),
                    Some(_) => {
                        if gpus.is_empty() {
                            let r = row.as_ref().unwrap_or(&serde_json::Value::Null);
                            gpus = r["gpu_indices"].as_array().into_iter().flatten().filter_map(|g| g.as_u64()).map(|g| g as u32).collect();
                            need_mb = (r["allocated_vram_gb_per_gpu"].as_f64().unwrap_or(0.0) * 1024.0) as u64 * gpus.len() as u64;
                            base_mb = crate::load_wait::managed_mb(&gpus);
                        }
                        if need_mb > 0 {
                            let grown = crate::load_wait::managed_mb(&gpus).saturating_sub(base_mb) as f64 / need_mb as f64;
                            self.frac = self.frac.max(grown.min(0.95));
                        }
                        Phase::Loading { instance, gpus, need_mb, base_mb }
                    }
                }
            }
            other => other,
        };
    }

    pub(crate) fn tick(&mut self) {
        // The percent climbs one at a time toward what the load measured; goethed is asked every half second.
        if self.ticks.is_multiple_of(8) {
            self.advance();
        }
        self.ticks += 1;
        if self.shown < (self.frac * 100.0).round() as u32 {
            self.shown += 1;
        }
    }

    pub(crate) fn finished(&self) -> bool {
        matches!(self.phase, Phase::Done) && self.shown >= 100
    }

    pub(crate) fn failure(&self) -> Option<String> {
        // The reason a load failed, in two lines at most: the first, and the one that names the engine log.
        let Phase::Failed(why) = &self.phase else { return None };
        let lines: Vec<&str> = why.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        Some(match (lines.first(), lines.iter().rev().find(|l| l.contains("goethe logs"))) {
            (Some(a), Some(b)) if a != b => format!("{a}\n{b}"),
            (Some(a), _) => a.to_string(),
            _ => format!("{} did not load", self.model),
        })
    }
}

fn unload_now(ids: &[String]) -> Result<(), String> {
    // goethe unload for the chosen models, quietly; its last line when it fails.
    let o = goethe_cmd().arg("unload").args(ids).output().map_err(|e| e.to_string())?;
    o.status.success().then_some(()).ok_or_else(|| String::from_utf8_lossy(&o.stderr).trim().lines().last().unwrap_or("unload failed").to_string())
}

struct Menu {
    heading: Vec<&'static str>,
    items: Vec<Choice>,
    at: usize,
    hint: &'static str,
    top: bool,
    gap_before: Option<usize>,
    side: Option<SideState>,
    off: Vec<bool>,
    job: Option<LoadJob>,
    gpus: Option<GpuChoice>,
    note: Option<String>,
    ask_run: Option<Box<dyn Fn() -> (String, bool)>>,
    ask_line: Option<String>,
}

const INSTALL_FIRST: &str = "(install a model first)";

impl Menu {
    fn refresh_off(&mut self) {
        // Load/Unload stays gray and out of reach until a model is installed.
        let none = self.items.iter().any(|c| c.side == Some(Side::Models)) && crate::models::installed_ids(&goethe_home()).is_empty();
        self.off = self.items.iter().map(|c| none && c.side == Some(Side::Models)).collect();
        if self.off.get(self.at).copied().unwrap_or(false) {
            self.step(true);
        }
        self.sync_side();
    }

    fn is_off(&self, i: usize) -> bool {
        self.off.get(i).copied().unwrap_or(false)
    }

    fn step(&mut self, down: bool) {
        // Up or down to the next choice that is not grayed out.
        let n = self.items.len();
        for _ in 0..n {
            self.at = if down { (self.at + 1) % n } else { (self.at + n - 1) % n };
            if !self.is_off(self.at) {
                break;
            }
        }
        self.sync_side();
    }

    fn sync_side(&mut self) {
        // The submenu of the choice under the cursor, shown beside it (read once, not on every frame).
        if self.side.as_ref().is_some_and(|s| s.focused) {
            return;
        }
        self.side = (!self.is_off(self.at)).then(|| self.items[self.at].side.map(SideState::open)).flatten();
    }

    fn refresh_side(&mut self) {
        // After a load or unload: the models again, marks cleared, the cursor and focus kept.
        if let Some(s) = self.side.as_mut() {
            let (at, kind, focused, base) = (s.at, s.kind, s.focused, std::mem::take(&mut s.base));
            let _ = std::mem::replace(s, SideState::open(kind));
            s.at = at.min(s.len().saturating_sub(1));
            s.focused = focused;
            s.base = base;
        }
    }
}

impl Page for Menu {
    fn render(&self, width: usize, height: usize) -> Vec<String> {
        let tw = self.items.iter().map(|c| c.title.chars().count()).max().unwrap_or(0);
        let mut out = brand();
        for h in &self.heading {
            out.push(format!("  {}", paint(h, Color::Bold)));
            out.push(String::new());
        }
        let side = self.side.as_ref();
        let focused = side.is_some_and(|s| s.focused);
        let start = out.len();
        for (i, c) in self.items.iter().enumerate() {
            if self.gap_before == Some(i) {
                out.push(String::new());
            }
            let title = format!("{:<tw$}", c.title);
            if self.is_off(i) {
                out.push(format!("  {}", gray(&format!("{title}{GAP}{INSTALL_FIRST}"))));
                continue;
            }
            let detail = match (&self.ask_line, i) {
                _ if side.is_some() => "",
                (Some(line), ASK_ROW) if self.top => line.as_str(),
                _ => c.detail,
            };
            out.push(match (i == self.at, focused) {
                (true, false) => format!("{} {}{GAP}{}", sgr(&hl(), "\u{27a4}"), sgr(&hl(), &title), detail),
                (true, true) => format!("  {}", paint(&title, Color::Bold)),
                _ => format!("  {title}{GAP}{}", gray(detail)),
            });
        }
        if let Some(s) = side {
            let example = if focused && s.kind == Side::Schemas { crate::api_schema::example(crate::api_schema::NAMES[s.at], &s.base, crate::api_schema::MODEL, crate::api_schema::KEY) } else { Vec::new() };
            let rows = match &self.gpus {
                Some(g) => g.lines(),
                None => side_rows(s, self.job.as_ref()),
            };
            columns(&mut out, start, 2 + tw + GAP.len(), &rows, &[]);
            example_below(&mut out, 2 + tw + GAP.len(), &example);
        }
        if let Some(n) = &self.note {
            out.push(String::new());
            out.extend(n.lines().map(|l| format!("  {}", paint(l, Color::Red))));
        }
        let hint = match side.filter(|s| s.focused).map(|s| s.kind) {
            Some(Side::Models) if self.gpus.is_some() => gray("\u{2191}\u{2193} move   Space choose GPUs (several split the model)   Enter load   \u{2190} back   q quit"),
            Some(Side::Schemas) => gray("\u{2191}\u{2193} move   \u{2190} back   q quit"),
            Some(Side::Models) => gray("\u{2191}\u{2193} move   Enter load or unload   Space mark loaded models to unload together   \u{2190} back   q quit"),
            None if self.top => format!("{}{GAP}{}", gray(self.hint), themes_line()),
            None => gray(self.hint),
        };
        screen(out, width, height, hint)
    }

    fn key(&mut self, input: Input) -> Flow {
        if matches!(input, Input::Char('q') | Input::Interrupt) {
            return Flow::Quit;
        }
        if let Some(mut g) = self.gpus.take() {
            self.note = None;
            match g.key(input) {
                Some(true) if !g.marked.is_empty() && g.plan.is_err() => {
                    self.note = Some("those GPUs cannot hold it; choose others or Auto".to_string());
                    self.gpus = Some(g);
                }
                Some(true) => {
                    let mut job = LoadJob::new(g.model, None);
                    job.gpus = g.marked;
                    self.job = Some(job);
                }
                Some(false) => {}
                None => self.gpus = Some(g),
            }
            return Flow::Stay;
        }
        if let Some(s) = self.side.as_mut().filter(|s| s.focused) {
            let n = s.len().max(1);
            match input {
                Input::Up | Input::Char('k') => s.at = (s.at + n - 1) % n,
                Input::Down | Input::Char('j') => s.at = (s.at + 1) % n,
                Input::Left | Input::Esc => s.focused = false,
                Input::Space if s.kind == Side::Models => {
                    if let Some((id, true)) = s.models.get(s.at).cloned() {
                        match s.marked.iter().position(|m| *m == id) {
                            Some(i) => drop(s.marked.remove(i)),
                            None => s.marked.push(id),
                        }
                    }
                }
                Input::Enter if s.kind == Side::Models => {
                    self.note = None;
                    let unload: Vec<String> = match s.models.get(s.at) {
                        _ if !s.marked.is_empty() => s.marked.clone(),
                        Some((id, true)) => vec![id.clone()],
                        _ => Vec::new(),
                    };
                    if !unload.is_empty() {
                        if let Err(e) = unload_now(&unload) {
                            self.note = Some(e);
                        }
                        self.refresh_side();
                    } else if let Some((id, false)) = s.models.get(s.at).cloned() {
                        match &self.job {
                            Some(j) => self.note = Some(format!("{} is loading; one model loads at a time", j.model)),
                            None => self.gpus = Some(GpuChoice::open(id)),
                        }
                    }
                }
                _ => {}
            }
            return Flow::Stay;
        }
        match input {
            Input::Up | Input::Char('k') => self.step(false),
            Input::Down | Input::Char('j') => self.step(true),
            Input::Char(c @ '1'..='3') if self.top => crate::theme::set(crate::theme::ALL[c as usize - '1' as usize]),
            _ if self.is_off(self.at) => {}
            Input::Enter | Input::Right if self.side.is_some() => {
                if let Some(s) = self.side.as_mut() {
                    s.focused = true;
                    if s.kind == Side::Schemas {
                        s.base = crate::api_schema::base_url(&goethe_home());
                    }
                }
            }
            Input::Enter => return Flow::Pick(self.at),
            Input::Esc | Input::Left if !self.top => return Flow::Leave,
            _ => {}
        }
        Flow::Stay
    }

    fn every(&self) -> Option<std::time::Duration> {
        let ask = self.ask_run.as_ref().map(|_| std::time::Duration::from_millis(250));
        self.job.as_ref().map(|_| std::time::Duration::from_millis(60)).or(ask)
    }

    fn tick(&mut self) {
        // An Ask run in the background shows on the Ask row. The load beside its model: when it runs, the bar gives way to "loaded"; when it fails, the reason shows under the list.
        if let Some(progress) = &self.ask_run {
            let (line, running) = progress();
            self.ask_line = Some(line);
            if !running {
                self.ask_run = None;
            }
        }
        let Some(j) = self.job.as_mut() else { return };
        j.tick();
        if j.finished() {
            self.job = None;
            self.refresh_side();
        } else if let Some(why) = j.failure() {
            self.note = Some(why);
            self.job = None;
            self.refresh_side();
        }
    }
}

struct SchemaPage {
    model: String,
    key: String,
    base: String,
    at: usize,
}

impl Page for SchemaPage {
    fn render(&self, width: usize, height: usize) -> Vec<String> {
        // Quick setup's end: the model and key, then the three schemas with the chosen one's request and SDK call beside them.
        let mut out = brand();
        out.push(format!("  {}", paint("Your API is ready", Color::Bold)));
        out.push(String::new());
        out.push(format!("  Model      {}", self.model));
        out.push(format!("  API key    {}", self.key));
        out.push(String::new());
        let start = out.len();
        let names: Vec<String> = crate::api_schema::NAMES
            .iter()
            .enumerate()
            .map(|(i, n)| if i == self.at { format!("{} {}", sgr(&hl(), "\u{27a4}"), sgr(&hl(), n)) } else { format!("  {n}") })
            .collect();
        columns(&mut out, start, 0, &names, &[]);
        example_below(&mut out, 4, &crate::api_schema::example(crate::api_schema::NAMES[self.at], &self.base, crate::api_schema::MODEL, crate::api_schema::KEY));
        screen(out, width, height, gray("\u{2191}\u{2193} move   \u{2190} back"))
    }

    fn key(&mut self, input: Input) -> Flow {
        let n = crate::api_schema::NAMES.len();
        match input {
            Input::Up | Input::Char('k') => self.at = (self.at + n - 1) % n,
            Input::Down | Input::Char('j') => self.at = (self.at + 1) % n,
            Input::Left | Input::Esc => return Flow::Leave,
            Input::Interrupt | Input::Char('q') => return Flow::Quit,
            _ => {}
        }
        Flow::Stay
    }
}

const ASK_ROW: usize = 0;

fn main_menu() -> Menu {
    Menu {
        heading: Vec::new(),
        items: vec![
            Choice { title: "Ask", detail: "describe what you want in plain English", side: None },
            Choice { title: "Quick setup", detail: "install a model, serve it and open the API", side: None },
            Choice { title: "Load/Unload", detail: "load an installed model, unload loaded ones", side: Some(Side::Models) },
            Choice { title: "Install models", detail: "pick models that fit your GPUs", side: None },
            Choice { title: "API keys and gateway", detail: "keys, who can reach the API, a public address", side: None },
            Choice { title: "Status", detail: "Goethe, the API and the models it serves", side: None },
            Choice { title: "Metrics", detail: "requests, tokens and GPU use over time", side: None },
        ],
        at: 0,
        hint: "\u{2191}\u{2193} move   Enter open   q quit",
        top: true,
        gap_before: Some(5),
        side: None,
        off: Vec::new(),
        job: None,
        gpus: None,
        note: None,
        ask_run: None,
        ask_line: None,
    }
}

fn access_menu() -> Menu {
    Menu {
        heading: vec!["API keys and gateway"],
        items: vec![
            Choice { title: "API keys", detail: "create, scope, show or delete keys", side: None },
            Choice { title: "Network exposure", detail: "local, vpn/private, public", side: None },
            Choice { title: "Public address", detail: "temporarily share models through a hardened gateway", side: None },
            Choice { title: "API schema", detail: "how clients call the API: openai, anthropic, litellm", side: Some(Side::Schemas) },
        ],
        at: 0,
        hint: crate::term::MENU_KEYS,
        top: false,
        gap_before: Some(3),
        side: None,
        off: Vec::new(),
        job: None,
        gpus: None,
        note: None,
        ask_run: None,
        ask_line: None,
    }
}

fn wait_back() -> bool {
    // After a page that printed: the left arrow or Esc goes back to the menu; q (or Ctrl-C) quits goethe, which the return value says.
    let mut buf = [0u8; 16];
    loop {
        let Ok(n) = crate::term::read_keys(&mut buf) else { return false };
        let keys = parse_input(&buf[..n]);
        if n == 0 || keys.iter().any(|i| matches!(i, Input::Char('q') | Input::Interrupt)) {
            return n != 0;
        }
        if keys.iter().any(|i| matches!(i, Input::Left | Input::Esc)) {
            return false;
        }
    }
}

fn report(result: Result<(), String>) {
    if let Err(e) = result.as_ref().map_err(|e| e.trim()).map_err(str::to_string) {
        if !e.is_empty() {
            eprintln!("{}", paint(&e, Color::Red));
        }
    }
}

struct ModelPick {
    models: Vec<(String, bool)>,
    at: usize,
    job: Option<LoadJob>,
    ready: Option<usize>,
    note: Option<String>,
}

impl Page for ModelPick {
    fn render(&self, width: usize, height: usize) -> Vec<String> {
        // goethe run without a model: the installed models at the top, loaded ones first and marked; one that is not loaded loads here, its bar beside it.
        let mut out = vec![String::new(), format!("  {}", paint("Choose a model to run", Color::Bold)), String::new()];
        for (i, (id, loaded)) in self.models.iter().enumerate() {
            let state = match self.job.as_ref().filter(|j| j.model == *id) {
                Some(j) => format!("  {}  {:>3}%", fine_bar(j.shown), j.shown),
                None if *loaded => format!("  {}", sgr(&crate::theme::secondary(), "loaded")),
                None => String::new(),
            };
            out.push(match i == self.at {
                true => format!("{} {}{state}", sgr(&hl(), "\u{27a4}"), sgr(&hl(), id)),
                false => format!("  {id}{state}"),
            });
        }
        if let Some(n) = &self.note {
            out.push(String::new());
            out.extend(n.lines().map(|l| format!("  {}", paint(l, Color::Red))));
        }
        let keys = if self.job.is_some() { "loading...   q quit" } else { "\u{2191}\u{2193} move   Enter run   q quit" };
        screen(out, width, height, gray(keys))
    }

    fn key(&mut self, input: Input) -> Flow {
        let n = self.models.len();
        if self.job.is_some() {
            return match input {
                Input::Interrupt | Input::Char('q') | Input::Esc => Flow::Quit,
                _ => Flow::Stay,
            };
        }
        self.note = None;
        match input {
            Input::Up | Input::Char('k') => self.at = (self.at + n - 1) % n,
            Input::Down | Input::Char('j') => self.at = (self.at + 1) % n,
            Input::Enter if self.models[self.at].1 => return Flow::Pick(self.at),
            Input::Enter => self.job = Some(LoadJob::new(self.models[self.at].0.clone(), crate::run::run_ctx(&goethe_home(), &self.models[self.at].0))),
            Input::Left | Input::Esc | Input::Interrupt | Input::Char('q') => return Flow::Quit,
            _ => {}
        }
        Flow::Stay
    }

    fn every(&self) -> Option<std::time::Duration> {
        self.job.as_ref().map(|_| std::time::Duration::from_millis(60))
    }

    fn tick(&mut self) {
        // The load in place: when it runs the model is chosen, when it fails the reason shows under the list.
        let Some(j) = self.job.as_mut() else { return };
        j.tick();
        if j.finished() {
            self.ready = Some(self.at);
        } else if let Some(why) = j.failure() {
            self.note = Some(why);
            self.job = None;
        }
    }

    fn finished(&self) -> bool {
        self.ready.is_some()
    }
}

pub fn metrics() -> Result<(), String> {
    // goethe metrics: the menu's Metrics page on a terminal (\u{2191}\u{2193} range, \u{2190} or q leaves); printed once when the output is not a terminal.
    let view = crate::metrics_view::MetricsView::new();
    if !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        println!("{}", crate::metrics_view::render(&view, crate::status_view::MAX_WIDTH + 4).join("\n"));
        return Ok(());
    }
    Tty::open()?.run(&mut MetricsPage(view)).map(|_| ())
}

pub fn pick_model() -> Result<Option<(String, bool)>, String> {
    // The model goethe run should use, chosen from the installed ones and loaded on the list when it was not, with whether the list loaded it; None when the list is left.
    let models = installed_and_loaded(&goethe_home());
    if models.is_empty() {
        return Err("no model is installed: goethe install MODEL".to_string());
    }
    let mut page = ModelPick { models, at: 0, job: None, ready: None, note: None };
    let picked = Tty::open()?.run(&mut page)?;
    let started_here = page.ready.is_some();
    if let (Some(j), false) = (&page.job, started_here) {
        crate::run::unload_quietly(&j.model);
    }
    Ok(picked.or(page.ready).and_then(|i| page.models.get(i).map(|(id, _)| (id.clone(), started_here))))
}

pub fn start() -> Result<(), String> {
    // goethe start: Quick setup on its own (install a model, serve it, open the API), then back to the shell.
    if !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        return Err("goethe start needs a terminal; in scripts use goethe up, goethe install MODEL --load and goethe api keys create NAME".to_string());
    }
    let mut tty = Tty::open()?;
    quick_setup(&mut tty)
}

fn quick_setup(tty: &mut Tty) -> Result<(), String> {
    tty.inside_with(crate::onboard::run, false)?;
    match crate::onboard::take_ready() {
        Some((_, model, key)) => drop(tty.run(&mut SchemaPage { model, key, base: crate::api_schema::base_url(&goethe_home()), at: 0 })?),
        None => tty.wait_back()?,
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StepState {
    #[default]
    Planned,
    Running,
    Done,
    Failed,
    Stopped,
}

#[derive(Clone, Debug, Default)]
pub struct AskStep {
    pub cmd: String,
    pub why: String,
    pub state: StepState,
    pub part: usize,
    pub secs: Option<u64>,
    pub latest: Option<String>,
    pub output: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct AskPart {
    pub text: String,
    pub label: String,
    pub options: Vec<(String, String)>,
    pub chosen: Option<usize>,
    pub unsure: bool,
    pub question: Option<String>,
    pub choices: Vec<(String, String)>,
    pub left_out: bool,
    pub unused: Vec<String>,
}

impl AskPart {
    fn open(&self) -> bool {
        !self.left_out && (self.unsure || self.question.is_some() || !self.unused.is_empty())
    }
}

#[derive(Clone, Debug, Default)]
pub struct AskFrame {
    pub said: String,
    pub parts: Vec<AskPart>,
    pub steps: Vec<AskStep>,
    pub note: Option<String>,
    pub confirm: Option<String>,
    pub attached: bool,
    pub running: bool,
    pub ran: bool,
    pub done: bool,
    pub can_run: bool,
}

pub trait AskDriver {
    fn frame(&mut self) -> AskFrame;
    fn submit(&mut self, request: &str) -> AskFrame;
    fn choose(&mut self, part: usize, option: usize) -> AskFrame;
    fn write(&mut self, part: usize, text: &str) -> AskFrame;
    fn answer(&mut self, part: usize, text: &str) -> AskFrame;
    fn pick_value(&mut self, part: usize, index: usize) -> AskFrame;
    fn drop_value(&mut self, part: usize, index: usize) -> AskFrame;
    fn leave_out(&mut self, part: usize) -> AskFrame;
    fn start(&mut self) -> AskFrame;
    fn run_attached(&mut self);
    fn stop(&mut self);
    fn progress(&self) -> Option<Box<dyn Fn() -> (String, bool)>>;
}

pub struct MenuHooks {
    pub ask: Box<dyn AskDriver>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Field {
    None,
    Request,
    Write,
    Answer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Row {
    Option(usize),
    LeaveOut,
    Write,
    Value(usize),
    TypeValue,
    Unused(usize),
}

struct AskPage<'a> {
    driver: &'a mut dyn AskDriver,
    frame: AskFrame,
    request: LineEdit,
    field: LineEdit,
    edit: Field,
    page: usize,
    row: usize,
    confirming: bool,
    attach: bool,
    host: AskHost,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AskHost {
    Menu,
    Alone,
}

const RESULT_LINES: usize = 12;

impl<'a> AskPage<'a> {
    fn new(driver: &'a mut dyn AskDriver, host: AskHost) -> AskPage<'a> {
        let frame = driver.frame();
        let request = LineEdit::of(&frame.said);
        let edit = if frame.said.is_empty() { Field::Request } else { Field::None };
        let mut page = AskPage { driver, frame: AskFrame::default(), request, field: LineEdit::default(), edit, page: 0, row: 0, confirming: false, attach: false, host };
        page.reset(frame);
        page
    }

    fn confirm_page(&self) -> usize {
        self.frame.parts.len()
    }

    fn reset(&mut self, frame: AskFrame) {
        // A new plan: start on the first part while any part still needs an answer, else the confirm page.
        self.frame = frame;
        self.page = if self.frame.parts.iter().any(AskPart::open) { 0 } else { self.confirm_page() };
        self.place();
    }

    fn update(&mut self, frame: AskFrame) {
        // After an answer: stay while this part still needs one, else move on to the next open part (or confirm).
        self.frame = frame;
        if self.frame.parts.get(self.page).is_none_or(|p| !p.open()) {
            let next = self.frame.parts.iter().enumerate().skip(self.page + 1).find(|(_, p)| p.open()).map(|(i, _)| i);
            self.page = next.unwrap_or(self.confirm_page());
        }
        self.place();
    }

    fn place(&mut self) {
        // The cursor on what the page needs: a value question first, else the chosen command.
        self.field = LineEdit::default();
        self.confirming = false;
        let rows = self.rows();
        let value = rows.iter().position(|r| matches!(r, Row::Value(_) | Row::TypeValue | Row::Unused(_)));
        let chosen = self.part().and_then(|p| p.chosen).and_then(|c| rows.iter().position(|r| *r == Row::Option(c)));
        self.row = value.or(chosen).unwrap_or(0);
    }

    fn part(&self) -> Option<&AskPart> {
        self.frame.parts.get(self.page)
    }

    fn rows(&self) -> Vec<Row> {
        let Some(p) = self.part() else {
            return Vec::new();
        };
        let mut rows: Vec<Row> = (0..p.options.len()).map(Row::Option).collect();
        rows.extend([Row::LeaveOut, Row::Write]);
        if p.question.is_some() && !p.left_out {
            rows.extend((0..p.choices.len()).map(Row::Value));
            rows.push(Row::TypeValue);
        }
        if !p.left_out {
            rows.extend((0..p.unused.len()).map(Row::Unused));
        }
        rows
    }

    fn enter(&mut self) -> Flow {
        if self.page == self.confirm_page() {
            return self.enter_confirm();
        }
        let part = self.page;
        let frame = match self.rows().get(self.row).copied() {
            Some(Row::Option(i)) => self.driver.choose(part, i),
            Some(Row::LeaveOut) => self.driver.leave_out(part),
            Some(Row::Value(i)) => self.driver.pick_value(part, i),
            Some(Row::Unused(i)) => self.driver.drop_value(part, i),
            Some(Row::Write) => {
                self.edit = Field::Write;
                return Flow::Stay;
            }
            Some(Row::TypeValue) => {
                self.edit = Field::Answer;
                return Flow::Stay;
            }
            None => return Flow::Stay,
        };
        self.update(frame);
        Flow::Stay
    }

    fn enter_confirm(&mut self) -> Flow {
        // Run the list (a second Enter when it deletes or opens access); a finished run plans again from what is true now.
        if self.frame.ran {
            let said = self.frame.said.clone();
            let frame = self.driver.submit(&said);
            self.reset(frame);
        } else if !self.frame.can_run {
            if let Some(open) = self.frame.parts.iter().position(AskPart::open) {
                self.page = open;
                self.place();
            }
        } else if self.frame.confirm.is_some() && !self.confirming {
            self.confirming = true;
        } else if self.frame.attached {
            self.attach = true;
            return Flow::Leave;
        } else {
            self.frame = self.driver.start();
        }
        Flow::Stay
    }

    fn submit_field(&mut self) {
        let text = match self.edit {
            Field::Request => self.request.get(),
            _ => self.field.get(),
        };
        let text = text.trim().to_string();
        let edit = std::mem::replace(&mut self.edit, Field::None);
        if text.is_empty() {
            return;
        }
        match edit {
            Field::Request => {
                let frame = self.driver.submit(&text);
                self.reset(frame);
            }
            Field::Write => {
                let frame = self.driver.write(self.page, &text);
                self.update(frame);
            }
            Field::Answer => {
                let frame = self.driver.answer(self.page, &text);
                self.update(frame);
            }
            Field::None => {}
        }
    }

    fn tabs(&self) -> String {
        // One entry per part and the confirm page: the current one in the theme color, open ones marked, left out ones gray.
        let mut items: Vec<String> = self
            .frame
            .parts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let name = format!("{} {}{}", i + 1, p.label, if p.open() { " ?" } else { "" });
                if i == self.page {
                    sgr(&hl(), &name)
                } else if p.left_out {
                    gray(&format!("{name} (left out)"))
                } else {
                    name
                }
            })
            .collect();
        let confirm = if self.frame.ran { "Result" } else { "Confirm" };
        items.push(if self.page == self.confirm_page() { sgr(&hl(), confirm) } else { gray(confirm) });
        format!("  {}", items.join(GAP))
    }

    fn part_lines(&self, p: &AskPart) -> Vec<String> {
        let arrow = |on: bool| if on { sgr(&hl(), "\u{27a4}") } else { " ".to_string() };
        let rows = self.rows();
        let here = |r: Row| self.edit == Field::None && rows.get(self.row) == Some(&r);
        let mut out = vec![format!("  {}", gray(&p.text)), String::new()];
        let cw = p.options.iter().map(|(c, _)| crate::term::visible_width(c)).max().unwrap_or(0);
        for (i, (cmd, about)) in p.options.iter().enumerate() {
            let on = here(Row::Option(i));
            let cmd = crate::status_view::pad(cmd, cw);
            let cmd = if on { sgr(&hl(), &cmd) } else if p.chosen == Some(i) { sgr(&format!("1;{}", crate::theme::third()), &cmd) } else { cmd };
            out.push(format!("{} {cmd}{GAP}{}", arrow(on), gray(about)));
        }
        out.push(format!("{} {}", arrow(here(Row::LeaveOut)), gray("leave this part out")));
        if self.edit == Field::Write {
            out.push(format!("{} {}", arrow(true), self.field.draw(true)));
        } else {
            out.push(format!("{} {}", arrow(here(Row::Write)), gray("type something else")));
        }
        if let (Some(q), false) = (&p.question, p.left_out) {
            out.push(String::new());
            out.push(format!("  {}", paint(q, Color::Bold)));
            let vw = p.choices.iter().map(|(v, _)| crate::term::visible_width(v)).max().unwrap_or(0);
            for (i, (v, help)) in p.choices.iter().enumerate() {
                let on = here(Row::Value(i));
                let v = crate::status_view::pad(v, vw);
                out.push(format!("{} {}{GAP}{}", arrow(on), if on { sgr(&hl(), &v) } else { v }, gray(help)));
            }
            if self.edit == Field::Answer {
                out.push(format!("{} {}", arrow(true), self.field.draw(true)));
            } else {
                out.push(format!("{} {}", arrow(here(Row::TypeValue)), gray("type a value")));
            }
        }
        if !p.left_out && !p.unused.is_empty() {
            out.push(String::new());
            out.push(format!("  {}", paint(&format!("Not used: {}", p.unused.join(", ")), Color::Bold)));
            for (i, v) in p.unused.iter().enumerate() {
                out.push(format!("{} {}", arrow(here(Row::Unused(i))), gray(&format!("leave {v} out"))));
            }
        }
        out
    }

    fn confirm_lines(&self) -> Vec<String> {
        // The commands in run order under the part they serve; once running, each one's state, time and newest line; once finished, each part's result.
        let mut out = Vec::new();
        let cw = self.frame.steps.iter().map(|s| crate::term::visible_width(&s.cmd)).max().unwrap_or(0);
        for (i, p) in self.frame.parts.iter().enumerate() {
            let label = format!("{} {}", i + 1, p.label);
            if p.left_out {
                out.push(format!("  {}", gray(&format!("{label}   left out"))));
                continue;
            }
            out.push(format!("  {}", paint(&label, Color::Bold)));
            let steps: Vec<&AskStep> = self.frame.steps.iter().filter(|s| s.part == i).collect();
            for (k, s) in steps.iter().enumerate() {
                let secs = s.secs.map(|t| format!(" {t}s")).unwrap_or_default();
                let state = match s.state {
                    StepState::Planned if self.frame.running => gray("waiting"),
                    StepState::Planned => gray(&s.why),
                    StepState::Running => format!("{}  {}", sgr(&hl(), &format!("running{secs}")), gray(s.latest.as_deref().unwrap_or(""))),
                    StepState::Done => paint(&format!("done{secs}"), Color::Green),
                    StepState::Failed => paint(&format!("failed{secs}"), Color::Red),
                    StepState::Stopped => paint(&format!("stopped{secs}"), Color::Yellow),
                };
                out.push(format!("    {}{GAP}{state}", crate::status_view::pad(&s.cmd, cw)));
                let last = k + 1 == steps.len();
                if (s.state == StepState::Done && last) || matches!(s.state, StepState::Failed | StepState::Stopped) {
                    out.extend(s.output.iter().rev().take(RESULT_LINES).rev().map(|l| format!("      {}", gray(l))));
                }
            }
        }
        out
    }
}

impl Page for AskPage<'_> {
    fn render(&self, width: usize, height: usize) -> Vec<String> {
        let mut out = brand();
        let on_request = self.edit == Field::Request;
        out.push(format!("{} {}", if on_request { sgr(&hl(), "\u{27a4}") } else { " ".to_string() }, self.request.draw(on_request)));
        if !self.frame.parts.is_empty() {
            out.push(String::new());
            out.push(self.tabs());
            out.push(String::new());
            match self.part() {
                Some(p) => out.extend(self.part_lines(p)),
                None => out.extend(self.confirm_lines()),
            }
        }
        if let Some(note) = &self.frame.note {
            out.push(String::new());
            out.push(format!("  {}", paint(note, Color::Red)));
        }
        if let (true, Some(c)) = (self.confirming, &self.frame.confirm) {
            out.push(String::new());
            out.push(format!("  {}", paint(c, Color::Red)));
        }
        if self.frame.done {
            out.push(String::new());
            out.push(format!("  {}", paint("Done.", Color::Green)));
        }
        let hint = match self.edit {
            Field::Request => "Enter plan   Esc cancel",
            Field::Write => "Enter plan this part   Esc cancel",
            Field::Answer => "Enter answer   Esc cancel",
            Field::None if self.frame.running && self.host == AskHost::Alone => "Running   Ctrl-C stop",
            Field::None if self.frame.running => "Running   Esc back to the menu (it keeps running)   Ctrl-C stop",
            Field::None if self.frame.parts.is_empty() => "Type what you want, then Enter   Esc back",
            Field::None if self.confirming => "Enter run   Esc cancel",
            Field::None if self.page == self.confirm_page() && self.frame.ran => "Enter plan again   \u{2190} parts   Esc back",
            Field::None if self.page == self.confirm_page() && self.frame.can_run => "Enter run   \u{2190} parts   \u{2191} edit request   Esc back",
            Field::None if self.page == self.confirm_page() => "Enter answer what is open   \u{2190} parts   Esc back",
            Field::None if matches!(self.rows().get(self.row), Some(Row::Value(_) | Row::TypeValue)) => "\u{2190}\u{2192} part   \u{2191}\u{2193} move   Enter choose   type the value   Esc back",
            Field::None => "\u{2190}\u{2192} part   \u{2191}\u{2193} move   Enter choose   type to reword   Esc back",
        };
        screen(out, width, height, gray(hint))
    }

    fn every(&self) -> Option<std::time::Duration> {
        self.frame.running.then_some(std::time::Duration::from_millis(150))
    }

    fn tick(&mut self) {
        self.frame = self.driver.frame();
    }

    fn keys(&mut self, inputs: &[Input]) -> Flow {
        // A paste arrives as one read: its line breaks become spaces and it is never sent on its own.
        let pasted = inputs.iter().filter(|i| matches!(i, Input::Char(_) | Input::Space)).count() > 1;
        let mut flow = Flow::Stay;
        for (i, input) in inputs.iter().enumerate() {
            flow = match input {
                Input::Enter if pasted && i + 1 == inputs.len() => Flow::Stay,
                Input::Enter if pasted => self.key(Input::Space),
                other => self.key(*other),
            };
            if !matches!(flow, Flow::Stay) {
                break;
            }
        }
        flow
    }

    fn key(&mut self, input: Input) -> Flow {
        // Text fields take every key but Enter and Esc; elsewhere left/right switch pages, up/down move, Enter chooses.
        if self.frame.running {
            return match input {
                Input::Interrupt => {
                    self.driver.stop();
                    Flow::Stay
                }
                Input::Esc if self.host == AskHost::Menu => Flow::Leave,
                _ => Flow::Stay,
            };
        }
        if self.edit != Field::None {
            match input {
                Input::Esc => self.edit = Field::None,
                Input::Interrupt => return Flow::Quit,
                Input::Enter => self.submit_field(),
                Input::Down if self.edit == Field::Request && !self.frame.parts.is_empty() => self.edit = Field::None,
                Input::Left if self.edit == Field::Request && self.request.is_empty() && self.frame.parts.is_empty() => return Flow::Leave,
                other if self.edit == Field::Request => self.request.key(other),
                other => self.field.key(other),
            }
            return Flow::Stay;
        }
        let last = self.rows().len().saturating_sub(1);
        match input {
            Input::Esc if self.confirming => self.confirming = false,
            Input::Esc => return Flow::Leave,
            Input::Interrupt => return Flow::Quit,
            Input::Left if self.page > 0 => {
                self.page -= 1;
                self.place();
            }
            Input::Right if self.page < self.confirm_page() => {
                self.page += 1;
                self.place();
            }
            Input::Up if self.row > 0 => self.row -= 1,
            Input::Up => self.edit = Field::Request,
            Input::Down => self.row = (self.row + 1).min(last),
            Input::Enter => return self.enter(),
            Input::Char(_) | Input::Space if self.page < self.confirm_page() => {
                let on_value = matches!(self.rows().get(self.row), Some(Row::Value(_) | Row::TypeValue));
                self.edit = if on_value { Field::Answer } else { Field::Write };
                self.field.key(input);
            }
            Input::Char(_) | Input::Space => {
                self.edit = Field::Request;
                self.request.key(input);
            }
            _ => {}
        }
        Flow::Stay
    }
}

fn ask_on(tty: &mut Tty, driver: &mut dyn AskDriver, host: AskHost) -> Result<bool, String> {
    // Ask on the open menu screen; its commands run in the background, and a command that is its own screen (a chat) gets the terminal off the menu's alternate screen, then Ask comes back. True when Ctrl-C quits Goethe.
    loop {
        let mut page = AskPage::new(driver, host);
        tty.run(&mut page)?;
        if tty.quit || !page.attach {
            return Ok(tty.quit);
        }
        tty.screen = None;
        tty.inside_with(
            || {
                page.driver.run_attached();
                Ok(())
            },
            false,
        )?;
    }
}

pub fn ask_session(driver: &mut dyn AskDriver) -> Result<bool, String> {
    // `goethe ask` on a terminal: the same page as Ask on the menu (there is no menu to go back to while it runs).
    let mut tty = Tty::open()?;
    ask_on(&mut tty, driver, AskHost::Alone)
}

pub fn run(mut hooks: MenuHooks) -> Result<(), String> {
    // goethe: the main menu until q; each choice runs, then the menu comes back.
    if !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        return Err("goethe alone needs a terminal; goethe help lists the commands".to_string());
    }
    let mut tty = Tty::open()?;
    let mut menu = main_menu();
    loop {
        menu.refresh_off();
        match tty.run(&mut menu)? {
            None if menu.ask_run.is_some() => menu.note = Some("Ask is still running commands; open Ask to follow them, or press Ctrl-C there to stop them.".into()),
            None => return Ok(()),
            Some(ASK_ROW) => {
                menu.ask_run = None;
                menu.ask_line = None;
                if ask_on(&mut tty, hooks.ask.as_mut(), AskHost::Menu)? {
                    tty.quit = true;
                }
                menu.ask_run = hooks.ask.progress();
                menu.ask_line = menu.ask_run.as_ref().map(|p| p().0);
            }
            Some(1) => quick_setup(&mut tty)?,
            Some(2) => {}
            Some(3) => install_page(&mut tty)?,
            Some(4) => access_page(&mut tty)?,
            Some(5) => {
                tty.draw(&Notice("Reading the GPUs...".to_string()));
                let mut page = StatusPage(crate::status_view::StatusView::new());
                tty.run(&mut page)?;
            }
            Some(_) => drop(tty.run(&mut MetricsPage(crate::metrics_view::MetricsView::new()))?),
        }
        if tty.quit {
            return Ok(());
        }
    }
}

fn access_page(tty: &mut Tty) -> Result<(), String> {
    let mut menu = access_menu();
    loop {
        match tty.run(&mut menu)? {
            None => return Ok(()),
            Some(0) => {
                match crate::keys_menu::session() {
                    Ok(session) => match session.summary() {
                        Some(text) => tty.inside(|| {
                            println!("{text}");
                            Ok(())
                        })?,
                        None => tty.quit = session.exit,
                    },
                    Err(e) => report(Err(e)),
                }
                tty.resume()?;
            }
            Some(1) => tty.inside(|| {
                crate::configure::run_api(crate::configure::ApiOpts {
                    mode: None,
                    vpn_ip: None,
                    interface: None,
                    allow: Vec::new(),
                    yes: false,
                    dangerously_allow_all_inbound: false,
                })
            })?,
            Some(_) => {
                let mut page = SharePage::new();
                tty.run(&mut page)?;
            }
        }
        if tty.quit {
            return Ok(());
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Slot {
    Shared(usize),
    Key,
    For,
    Allow,
    Share,
    StopAll,
}

struct SharePage {
    shared: Vec<(String, crate::public_share::Grant)>,
    url: Option<String>,
    names: Vec<String>,
    key: String,
    duration: String,
    allow: String,
    at: usize,
    armed: Option<Slot>,
    status: String,
    error: bool,
}

impl SharePage {
    fn new() -> SharePage {
        let mut p = SharePage {
            shared: Vec::new(),
            url: None,
            names: Vec::new(),
            key: String::new(),
            duration: "5h".to_string(),
            allow: String::new(),
            at: 0,
            armed: None,
            status: String::new(),
            error: false,
        };
        p.reload();
        p.at = p.slots().iter().position(|s| *s == Slot::Key).unwrap_or(0);
        p
    }

    fn reload(&mut self) {
        // What the gateway serves now, its address, and the keys that may be shared (never admin keys).
        let home = goethe_home();
        self.shared = crate::public_share::shared(&home);
        self.url = crate::public_share::active(&home).map(|s| {
            let ip = crate::client::link::lookup_public_ip(&home).map(|i| i.to_string()).unwrap_or_else(|_| "<public IP>".into());
            let host = if ip.contains(':') { format!("[{ip}]") } else { ip };
            format!("https://{host}:{}/v1", s.port)
        });
        self.names = crate::security::key_list_detailed()
            .unwrap_or_default()
            .iter()
            .filter(|k| !k.get("admin").and_then(|v| v.as_bool()).unwrap_or(false))
            .filter_map(|k| k.get("name").and_then(|v| v.as_str()).map(str::to_string))
            .collect();
    }

    fn slots(&self) -> Vec<Slot> {
        let mut out: Vec<Slot> = (0..self.shared.len()).map(Slot::Shared).collect();
        out.extend([Slot::Key, Slot::For, Slot::Allow, Slot::Share]);
        if !self.shared.is_empty() {
            out.push(Slot::StopAll);
        }
        out
    }

    fn slot(&self) -> Slot {
        let slots = self.slots();
        slots[self.at.min(slots.len() - 1)]
    }

    fn text(&mut self) -> Option<&mut String> {
        match self.slot() {
            Slot::Key => Some(&mut self.key),
            Slot::For => Some(&mut self.duration),
            Slot::Allow => Some(&mut self.allow),
            _ => None,
        }
    }

    fn say(&mut self, text: String, error: bool) {
        self.status = text;
        self.error = error;
    }

    fn share(&mut self) {
        // Share the typed key for the typed time from the typed addresses, on the one gateway every shared key uses.
        use crate::public_share as share;
        let name = self.key.trim().to_string();
        if name.is_empty() {
            self.at = self.slots().iter().position(|s| *s == Slot::Key).unwrap_or(0);
            return self.say("Type the name of the API key to share".to_string(), true);
        }
        let secs = match share::parse_duration(&self.duration) {
            Ok(s) => s,
            Err(e) => return self.say(e, true),
        };
        let allow: Vec<String> = self.allow.split([',', ' ']).map(str::trim).filter(|c| !c.is_empty()).map(str::to_string).collect();
        let everyone = allow.len() == 1 && allow[0].eq_ignore_ascii_case("everyone");
        if allow.is_empty() {
            self.at = self.slots().iter().position(|s| *s == Slot::Allow).unwrap_or(0);
            return self.say("Type who may connect: their public IPs or ranges, or everyone to open it to every address".to_string(), true);
        }
        if !everyone && allow.iter().any(|c| matches!(c.as_str(), "0.0.0.0/0" | "::/0" | "0.0.0.0" | "*")) {
            self.at = self.slots().iter().position(|s| *s == Slot::Allow).unwrap_or(0);
            return self.say("To open it to every address, type everyone instead".to_string(), true);
        }
        let allow = if everyone { Vec::new() } else { allow };
        let home = goethe_home();
        match share::open(
            &home,
            share::OpenArgs {
                secs,
                keys: std::slice::from_ref(&name),
                allow: &allow,
                port: None,
                dangerously_allow_all_inbound: everyone,
            },
        ) {
            Ok(o) => {
                self.reload();
                self.key.clear();
                self.allow.clear();
                let warn = o.warnings.first().map(|w| format!("  Note: {w}")).unwrap_or_default();
                let clients = if o.share.allow.is_empty() {
                    String::new()
                } else {
                    format!(" Only: {}.", o.share.allow.join(", "))
                };
                self.say(
                    format!(
                        "Shared {name} until {}; send {} with the key.{clients}{warn}",
                        share::local_time(share::now() + secs),
                        o.cert.display()
                    ),
                    false,
                );
            }
            Err(e) => self.say(e, true),
        }
    }

    fn stop(&mut self, which: Slot) {
        use crate::public_share as share;
        let home = goethe_home();
        let done = match which {
            Slot::Shared(i) => {
                let name = self.shared[i].0.clone();
                share::stop_key(&home, &self.shared[i].1.key).map(|_| format!("{name} is no longer shared"))
            }
            _ => share::shutdown(&home).map(|_| "The public gateway is closed".to_string()),
        };
        self.reload();
        self.at = self.at.min(self.slots().len() - 1);
        match done {
            Ok(t) => self.say(t, false),
            Err(e) => self.say(e, true),
        }
    }
}

impl Page for SharePage {
    fn render(&self, width: usize, height: usize) -> Vec<String> {
        use crate::public_share as share;
        let mut out = brand();
        out.push(format!("  {}", paint("Public address", Color::Bold)));
        out.push(format!("  {}", gray("Temporarily share models through a hardened gateway: TLS, limits per address, only the keys listed here.")));
        out.push(String::new());
        let here = self.slot();
        let cursor = |s: Slot| if here == s { sgr(&hl(), "\u{27a4}") } else { " ".to_string() };
        match &self.url {
            Some(url) => out.push(format!("  {}{url}", paint("Gateway  ", Color::Bold))),
            None => out.push(format!("  {}{}", paint("Gateway  ", Color::Bold), gray("closed"))),
        }
        let nw = self.shared.iter().map(|(n, _)| n.chars().count()).max().unwrap_or(0);
        for (i, (name, g)) in self.shared.iter().enumerate() {
            let from = if g.allow.is_empty() { "any address".to_string() } else { g.allow.join(", ") };
            let left = g.until - share::now();
            let left = if left >= 86_400 { format!("{}d {}h", left / 86_400, left % 86_400 / 3600) } else { format!("{}h {}m", left / 3600, left % 3600 / 60) };
            let row = format!("{name:<nw$}{GAP}until {} ({left} left){GAP}from {from}", share::local_time(g.until));
            let row = if here == Slot::Shared(i) { sgr(&hl(), &row) } else { row };
            out.push(format!("{} {}{row}", cursor(Slot::Shared(i)), " ".repeat(9)));
        }
        out.push(String::new());
        out.push(format!("  {}", paint("Share a key", Color::Bold)));
        let field = |s: Slot, label: &str, value: &str, example: &str| {
            let caret = if here == s { sgr(&mark(), "|") } else { String::new() };
            let shown = if value.is_empty() && here != s { gray(example) } else { format!("{value}{caret}") };
            let tail = if value.is_empty() && here == s { format!("   {}", gray(example)) } else { String::new() };
            format!("{} {}{shown}{tail}", cursor(s), paint(&format!("  {label:<13}"), if here == s { Color::Bold } else { Color::White }))
        };
        let names = if self.names.is_empty() { "no keys yet: create one in API keys".to_string() } else { format!("keys: {}", self.names.join(", ")) };
        out.push(field(Slot::Key, "API key:", &self.key, &names));
        out.push(field(Slot::For, "For:", &self.duration, "e.g. 30m, 5h, 1d; at most 7d"));
        out.push(field(Slot::Allow, "Allowed IPs:", &self.allow, "required: who may connect, e.g. 203.0.113.7, 198.51.100.0/24; everyone opens it to every address"));
        out.push(String::new());
        let action = |s: Slot, text: &str, color: Option<&str>| {
            let t = match (here == s, color) {
                (true, _) => sgr(&hl(), text),
                (false, Some(c)) => sgr(c, text),
                (false, None) => text.to_string(),
            };
            format!("{} {t}", cursor(s))
        };
        out.push(action(Slot::Share, "Share", None));
        if !self.shared.is_empty() {
            out.push(action(Slot::StopAll, "Stop sharing every key", Some("31")));
        }
        let footer = if !self.status.is_empty() {
            paint(&self.status, if self.error { Color::Red } else { Color::Yellow })
        } else {
            gray(match here {
                Slot::Shared(_) => "\u{2191}\u{2193} move   Enter stop sharing this key   \u{2190} back",
                Slot::Key | Slot::For | Slot::Allow => "type   \u{2191}\u{2193} or Enter next   \u{2190} back",
                _ => "\u{2191}\u{2193} move   Enter do it   \u{2190} back",
            })
        };
        screen(out, width, height, footer)
    }

    fn key(&mut self, input: Input) -> Flow {
        let n = self.slots().len();
        let slot = self.slot();
        let armed = self.armed.take();
        self.status.clear();
        match input {
            Input::Up => self.at = (self.at + n - 1) % n,
            Input::Down => self.at = (self.at + 1) % n,
            Input::Left | Input::Esc => return Flow::Leave,
            Input::Interrupt => return Flow::Quit,
            Input::Enter => match slot {
                Slot::Key | Slot::For | Slot::Allow => self.at = (self.at + 1) % n,
                Slot::Share => self.share(),
                s if armed == Some(s) => self.stop(s),
                s => {
                    self.armed = Some(s);
                    let what = match s {
                        Slot::Shared(i) => format!("stop sharing {}", self.shared[i].0),
                        _ => "close the public gateway for every key".to_string(),
                    };
                    self.say(format!("Press Enter again to {what}"), false);
                }
            },
            Input::Backspace => drop(self.text().map(String::pop)),
            Input::Space if slot == Slot::Allow => self.allow.push(' '),
            Input::Char(c) => match self.text() {
                Some(t) if t.chars().count() < 200 => t.push(c),
                Some(_) => {}
                None if c == 'q' => return Flow::Quit,
                None => {}
            },
            _ => {}
        }
        Flow::Stay
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Spot {
    Model(usize),
    Custom,
    Token,
}

struct Picker {
    data: PickData,
    at: usize,
    chosen: Vec<bool>,
    custom: String,
    token: String,
    account: Option<String>,
    status: String,
}

impl Picker {
    fn new(data: PickData) -> Picker {
        let chosen = vec![false; data.rows.len()];
        let account = crate::hf::token_set().then(|| "this session".to_string());
        Picker { data, at: 0, chosen, custom: String::new(), token: String::new(), account, status: String::new() }
    }

    fn spot(&self) -> Spot {
        match self.at.checked_sub(self.data.rows.len()) {
            None => Spot::Model(self.at),
            Some(0) => Spot::Custom,
            Some(_) => Spot::Token,
        }
    }

    fn summary(&self) -> Vec<String> {
        // Two lines over the table: VRAM of the whole machine and of the allowed GPUs, then the storage and what the models take.
        let d = &self.data;
        let label = |s: &str| paint(&format!("{s:<11}"), Color::Bold);
        let gpus = d.allowed_gpus.iter().map(|g| format!("gpu{g}")).collect::<Vec<_>>().join(" ");
        let word = crate::gpu_kind::memory_word(d.unified);
        let mut out = vec![if d.gpu_count <= 1 && !gpus.is_empty() {
            format!("  {}{} total", label(word), gb(d.total_vram_gb))
        } else {
            format!(
                "  {}{} total{GAP}{} allowed ({})",
                label(word),
                gb(d.total_vram_gb),
                gb(d.allowed_vram_gb),
                if gpus.is_empty() { "no GPU allowed".to_string() } else { gpus }
            )
        }];
        out.push(match (d.disk_total_gb, d.disk_free_gb) {
            (Some(t), Some(f)) => format!("  {}{} total{GAP}{} free{GAP}{} used by your models", label("Storage"), gb(t), gb(f), gb(d.models_gb)),
            _ => format!("  {}{} used by your models", label("Storage"), gb(d.models_gb)),
        });
        out
    }

    fn table(&self, width: usize, rows: usize) -> Vec<String> {
        // Header and one line per model, scrolled so the cursor stays in view; a chosen model is indented, installed ones are bold.
        const NUM: usize = 9;
        let note = |r: &PickRow| {
            let parts = [(r.installed, "installed"), (r.gated && !r.installed, "gated"), (r.optimised, "optimised kernels")];
            parts.iter().filter(|(on, _)| *on).map(|(_, t)| *t).collect::<Vec<_>>().join(" | ")
        };
        let longest = self.data.rows.iter().map(|r| r.id.chars().count()).max().unwrap_or(5).max(5);
        let multi = self.data.gpu_count > 1;
        let gpus_w = if multi { 4 + GAP.len() } else { 0 };
        let fixed = 4 + 2 + (NUM + GAP.len()) * 2 + gpus_w;
        let mw = longest.min(width.saturating_sub(fixed + GAP.len())).max(16);
        let nw = width.saturating_sub(fixed + mw + GAP.len());
        let mem = if self.data.unified { "MEMORY" } else { "VRAM" };
        let gpus_head = if multi { format!("{:>4}{GAP}", "GPUS") } else { String::new() };
        let head = format!("    {:<mw$}  {GAP}{:>NUM$}{GAP}{:>NUM$}{GAP}{gpus_head}{}", "MODEL", "SIZE", mem, if nw >= 8 { "NOTE" } else { "" });
        let mut out = vec![paint(head.trim_end(), Color::Bold)];
        let at = self.at.min(self.data.rows.len().saturating_sub(1));
        let start = window(self.data.rows.len(), at, rows);
        for (i, r) in self.data.rows.iter().enumerate().skip(start).take(rows) {
            let (here, picked) = (i == self.at, self.chosen[i]);
            let cursor = if here { sgr(&hl(), "\u{27a4}") } else { " ".to_string() };
            let mark = match (picked, r.installed) {
                (true, false) => sgr(&mark(), "\u{25cf}"),
                (true, true) => sgr(&drop_mark(), "\u{25cf}"),
                (false, true) => sgr(&crate::theme::secondary(), "\u{25cf}"),
                (false, false) => "\u{25cb}".to_string(),
            };
            let name = fit(&format!("{}{}", if picked { "  " } else { "" }, r.id), mw + 2);
            let name = match name_style(r, here) {
                "" => name,
                code => sgr(code, &name),
            };
            let gpus = if multi { format!("{:>4}{GAP}", r.gpus) } else { String::new() };
            let rest = format!("{GAP}{:>NUM$}{GAP}{:>NUM$}{GAP}{gpus}", gb(r.size_gb), gb(r.vram_gb));
            let note = if nw >= 8 { fit(&note(r), nw).trim_end().to_string() } else { String::new() };
            out.push(format!("{cursor} {mark} {name}{rest}{}", gray(&note)).trim_end().to_string());
        }
        if self.data.rows.is_empty() {
            out.push("    No catalog model fits the GPUs of this machine.".to_string());
        }
        out
    }

    fn fields(&self) -> Vec<String> {
        // The two lines under the table: a Hugging Face id to install, and the token for gated or private models.
        let caret = |on: bool| if on { sgr(&mark(), "|") } else { String::new() };
        let cursor = |s: Spot| if self.spot() == s { sgr(&hl(), "\u{27a4}") } else { " ".to_string() };
        let custom = if self.custom.is_empty() && self.spot() != Spot::Custom {
            gray("(org/model)")
        } else {
            format!("{}{}", self.custom, caret(self.spot() == Spot::Custom))
        };
        let token = match (&self.account, self.token.is_empty(), self.spot() == Spot::Token) {
            (Some(a), true, false) => format!("HF token:  {}", gray(&format!("set for {a}"))),
            (_, true, false) => gray("HF token not set:  (optional)"),
            (_, _, _) => format!("HF token:  {}{}", "\u{2022}".repeat(self.token.chars().count()), caret(self.spot() == Spot::Token)),
        };
        vec![format!("{}   Custom model:  {custom}", cursor(Spot::Custom)), format!("{}   {token}", cursor(Spot::Token))]
    }

    fn chosen_ids(&self, installed: bool) -> Vec<String> {
        // The chosen models that are (or are not) installed: what Enter uninstalls (or installs).
        self.data.rows.iter().zip(&self.chosen).filter(|(r, c)| **c && r.installed == installed).map(|(r, _)| r.id.clone()).collect()
    }

    fn installs(&self) -> Vec<String> {
        let mut ids = self.chosen_ids(false);
        let custom = self.custom.trim();
        if !custom.is_empty() && !ids.iter().any(|i| i == custom) && !self.data.rows.iter().any(|r| r.installed && r.id == custom) {
            ids.push(custom.to_string());
        }
        ids
    }

    fn removals(&self) -> Vec<String> {
        self.chosen_ids(true)
    }

    fn plan(&self, width: usize) -> Vec<String> {
        // What Enter would do, one line each: install in the secondary color, uninstall in the third.
        let line = |word: &str, ids: Vec<String>, code: String| {
            (!ids.is_empty()).then(|| sgr(&code, fit(&format!("{word}: {}", ids.join(" ")), width.saturating_sub(2)).trim_end()))
        };
        [line("install", self.installs(), crate::theme::secondary()), line("uninstall", self.removals(), drop_mark())].into_iter().flatten().collect()
    }

    fn hint(&self) -> String {
        let keys = match self.spot() {
            Spot::Model(_) => "\u{2191}\u{2193} move   Space select   Enter apply   \u{2190} back",
            Spot::Custom => "type a Hugging Face id such as org/model   Enter apply   \u{2191}\u{2193} move   \u{2190} back",
            Spot::Token => "paste a Hugging Face token   Enter use it for these installs   \u{2191}\u{2193} move   \u{2190} back",
        };
        keys.to_string()
    }

    fn enter(&mut self) -> Flow {
        if self.spot() == Spot::Token {
            if self.token.trim().is_empty() {
                self.status = "Paste a token first".to_string();
                return Flow::Stay;
            }
            match crate::hf::whoami(&self.token) {
                Ok(account) => {
                    crate::hf::set_token(Some(self.token.clone()));
                    self.status = format!("Token accepted for {account}");
                    self.account = Some(account);
                    self.token.clear();
                }
                Err(e) => self.status = e,
            }
            return Flow::Stay;
        }
        let custom = self.custom.trim();
        if !custom.is_empty() {
            if let Err(e) = crate::paths::validate_model_id(custom) {
                self.status = format!("Custom model: {e}");
                return Flow::Stay;
            }
        }
        if self.installs().is_empty() && self.removals().is_empty() {
            self.status = "Select models with Space, or type a custom model".to_string();
            return Flow::Stay;
        }
        Flow::Pick(0)
    }
}

impl Page for Picker {
    fn render(&self, width: usize, height: usize) -> Vec<String> {
        let mut out = vec![String::new(), format!("  {}", paint("Install models", Color::Bold)), String::new()];
        out.extend(self.summary());
        out.push(String::new());
        let plan = self.plan(width);
        let rows = height.saturating_sub(out.len() + 1 + 3 + 3 + plan.len() + usize::from(!plan.is_empty())).max(1);
        out.extend(self.table(width, rows));
        out.push(String::new());
        out.extend(self.fields());
        if !plan.is_empty() {
            out.push(String::new());
            out.extend(plan.into_iter().map(|l| format!("  {l}")));
        }
        let footer = if self.status.is_empty() { gray(&self.hint()) } else { paint(&self.status, Color::Yellow) };
        screen(out, width, height, footer)
    }

    fn key(&mut self, input: Input) -> Flow {
        let n = self.data.rows.len() + 2;
        let typing = self.spot() != Spot::Model(self.at);
        self.status.clear();
        match input {
            Input::Up => self.at = (self.at + n - 1) % n,
            Input::Down => self.at = (self.at + 1) % n,
            Input::Char('k') if !typing => self.at = (self.at + n - 1) % n,
            Input::Char('j') if !typing => self.at = (self.at + 1) % n,
            Input::PageUp => self.at = self.at.saturating_sub(10),
            Input::PageDown => self.at = (self.at + 10).min(n - 1),
            Input::Enter => return self.enter(),
            Input::Esc | Input::Left => return Flow::Leave,
            Input::Interrupt => return Flow::Quit,
            Input::Char('q') if !typing => return Flow::Quit,
            Input::Space => {
                if let Spot::Model(i) = self.spot() {
                    self.chosen[i] = !self.chosen[i];
                }
            }
            Input::Backspace => match self.spot() {
                Spot::Custom => drop(self.custom.pop()),
                Spot::Token => drop(self.token.pop()),
                Spot::Model(_) => {}
            },
            Input::Char(c) => match self.spot() {
                Spot::Custom if self.custom.chars().count() < 200 => self.custom.push(c),
                Spot::Token if self.token.chars().count() < 200 => self.token.push(c),
                _ => {}
            },
            _ => {}
        }
        Flow::Stay
    }
}

fn install_page(tty: &mut Tty) -> Result<(), String> {
    // The table of models that fit; Enter uninstalls the chosen installed ones, then installs the others, on the normal screen, and the table comes back.
    let home = goethe_home();
    loop {
        tty.draw(&Notice("Reading the catalog and the GPUs...".to_string()));
        let data = crate::placement::ensure_hw_profile(&home).and_then(|_| crate::models::picker_data(&home));
        let mut picker = match data {
            Ok(d) => Picker::new(d),
            Err(e) => return tty.inside(|| Err(e)),
        };
        if tty.run(&mut picker)?.is_none() || tty.quit {
            return Ok(());
        }
        let (removals, installs) = (picker.removals(), picker.installs());
        tty.inside(|| apply_picks(&removals, &installs))?;
        if tty.quit {
            return Ok(());
        }
    }
}

fn apply_picks(removals: &[String], installs: &[String]) -> Result<(), String> {
    // Every chosen change in turn, uninstalls first so their disk space is free for the installs; one failure never stops the others.
    crate::step::set_compact(true);
    let mut failed = Vec::new();
    for id in removals {
        if let Err(e) = crate::models::run_remove(id, true, false, Some(false)) {
            report(Err(e));
            failed.push(id.clone());
        }
    }
    for id in installs {
        let opts = crate::install::InstallOpts {
            model_id: id.clone(),
            ctx: None,
            max_conc: None,
            tp: None,
            gpus: None,
            load: false,
            force: false,
            no_verify: false,
            name: None,
            copy_local: false,
            sglang_args: Vec::new(),
            no_wait: false,
            trust_remote_code: false,
            yes: true,
            reset: false,
            quant: None,
            fallback: false,
        };
        if let Err(e) = crate::install::run(opts) {
            report(Err(e));
            failed.push(id.clone());
        }
    }
    crate::step::set_compact(false);
    match failed.len() {
        0 => Ok(()),
        _ => Err(format!("not done: {}", failed.join(", "))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: &str, installed: bool) -> PickRow {
        PickRow { id: id.into(), size_gb: 16.4, vram_gb: 21.0, gpus: 1, installed, gated: false, optimised: false }
    }

    fn picker() -> Picker {
        let mut p = Picker::new(PickData {
            total_vram_gb: 382.4,
            allowed_vram_gb: 95.6,
            allowed_gpus: vec![3],
            disk_total_gb: Some(3500.0),
            disk_free_gb: Some(1200.0),
            models_gb: 120.4,
            rows: vec![row("org/small", false), row("org/mine", true), row("org/large", false)],
            gpu_count: 4,
            unified: false,
        });
        p.account = None;
        p
    }

    fn text(lines: Vec<String>) -> Vec<String> {
        lines.iter().map(|l| plain(l)).collect()
    }

    #[derive(Default)]
    struct Said {
        frame: AskFrame,
        started: usize,
        stopped: bool,
        written: Vec<(usize, String)>,
        answered: Vec<(usize, String)>,
    }

    fn part(text: &str, chosen: Option<usize>) -> AskPart {
        let options: Vec<(String, String)> = vec![("goethe install".into(), "Download a model".into()), ("goethe api".into(), "The API address".into())];
        AskPart { text: text.into(), label: chosen.map_or(text.into(), |c| options[c].0.trim_start_matches("goethe ").to_string()), options, chosen, unsure: chosen.is_none(), ..AskPart::default() }
    }

    impl Said {
        fn settle(&mut self) -> AskFrame {
            self.frame.can_run = self.frame.parts.iter().all(|p| !p.open());
            self.frame.steps = self.frame.parts.iter().enumerate().filter(|(_, p)| !p.left_out).map(|(i, p)| AskStep { cmd: format!("goethe {}", p.label), why: "w".into(), part: i, ..AskStep::default() }).collect();
            self.frame.clone()
        }
    }

    impl AskDriver for Said {
        fn frame(&mut self) -> AskFrame {
            self.frame.clone()
        }
        fn submit(&mut self, line: &str) -> AskFrame {
            let parts = line
                .split(" and ")
                .enumerate()
                .map(|(i, t)| AskPart { unused: t.split_whitespace().filter(|w| w.starts_with("wg")).map(str::to_string).collect(), ..part(t, (i == 0).then_some(0)) })
                .collect();
            self.frame = AskFrame { said: line.to_string(), parts, ..AskFrame::default() };
            self.settle()
        }
        fn choose(&mut self, at: usize, option: usize) -> AskFrame {
            self.frame.parts[at] = part(&self.frame.parts[at].text.clone(), Some(option));
            self.settle()
        }
        fn write(&mut self, at: usize, text: &str) -> AskFrame {
            self.written.push((at, text.into()));
            self.frame.parts[at] = part(text, Some(0));
            self.settle()
        }
        fn answer(&mut self, at: usize, text: &str) -> AskFrame {
            self.answered.push((at, text.into()));
            self.frame.parts[at].question = None;
            self.settle()
        }
        fn pick_value(&mut self, at: usize, _: usize) -> AskFrame {
            self.frame.parts[at].question = None;
            self.settle()
        }
        fn drop_value(&mut self, at: usize, index: usize) -> AskFrame {
            self.frame.parts[at].unused.remove(index);
            self.settle()
        }
        fn leave_out(&mut self, at: usize) -> AskFrame {
            self.frame.parts[at].left_out = true;
            self.settle()
        }
        fn start(&mut self) -> AskFrame {
            self.started += 1;
            self.frame.running = true;
            self.frame.steps[0].state = StepState::Running;
            self.frame.clone()
        }
        fn run_attached(&mut self) {}
        fn stop(&mut self) {
            self.stopped = true;
        }
        fn progress(&self) -> Option<Box<dyn Fn() -> (String, bool)>> {
            None
        }
    }

    fn typed(page: &mut AskPage<'_>, s: &str) {
        for c in s.chars() {
            page.key(if c == ' ' { Input::Space } else { Input::Char(c) });
        }
    }

    fn asked<'a>(driver: &'a mut Said, request: &str) -> AskPage<'a> {
        let mut page = AskPage::new(driver, AskHost::Menu);
        typed(&mut page, request);
        page.key(Input::Enter);
        page
    }

    #[test]
    fn a_value_no_command_uses_is_shown_until_it_is_left_out() {
        // "Not used: wg0" sits under the part with a row to leave it out; until then the part stays open and nothing can run.
        let mut driver = Said::default();
        let mut page = asked(&mut driver, "install x on wg0");
        assert!(!page.frame.can_run);
        let screen = text(page.render(110, 30));
        assert!(screen.iter().any(|l| l.trim() == "Not used: wg0"), "{screen:?}");
        assert!(screen.iter().any(|l| l.contains("leave wg0 out")), "{screen:?}");
        assert!(page.rows().get(page.row) == Some(&Row::Unused(0)), "the cursor waits on it");
        page.key(Input::Enter);
        assert!(page.frame.parts[0].unused.is_empty() && page.frame.can_run);
    }

    #[test]
    fn ask_edits_the_request_like_a_text_field() {
        // Left moves the cursor, every letter is text, and Esc is the way back.
        let mut driver = Said::default();
        let mut page = AskPage::new(&mut driver, AskHost::Menu);
        typed(&mut page, "load quick jobs");
        for _ in 0.."jobs".len() {
            assert!(matches!(page.key(Input::Left), Flow::Stay), "Left inside text stays on the page");
        }
        typed(&mut page, "q ");
        assert_eq!(page.request.get(), "load quick q jobs");
        page.key(Input::Right);
        page.key(Input::Backspace);
        assert_eq!(page.request.get(), "load quick q obs");
        assert!(matches!(page.key(Input::Enter), Flow::Stay));
        assert_eq!(page.driver.frame().said, "load quick q obs");
        assert!(matches!(page.key(Input::Esc), Flow::Leave));
    }

    #[test]
    fn a_pasted_request_waits_for_enter() {
        // A paste ends in a line break; it must stay editable, not be sent with it.
        let mut driver = Said::default();
        let mut page = AskPage::new(&mut driver, AskHost::Menu);
        let mut paste: Vec<Input> = "run orca now".chars().map(|c| if c == ' ' { Input::Space } else { Input::Char(c) }).collect();
        paste.push(Input::Enter);
        assert!(matches!(page.keys(&paste), Flow::Stay));
        assert_eq!(page.request.get(), "run orca now");
        assert!(page.driver.frame().said.is_empty(), "nothing was sent");
        page.key(Input::Enter);
        assert_eq!(page.driver.frame().said, "run orca now");
    }

    #[test]
    fn each_part_is_a_page_switched_with_left_and_right() {
        // The page opens on the first part while any part needs an answer; left/right move between parts and confirm; Enter on a command settles the part and moves on.
        let mut driver = Said::default();
        let mut page = asked(&mut driver, "install x and give me an api");
        assert_eq!(page.page, 0, "the first part, even though only the second one asks");
        page.key(Input::Left);
        assert_eq!(page.page, 0, "no page before the first");
        page.key(Input::Right);
        assert_eq!(page.page, 1);
        let screen = text(page.render(110, 30));
        assert!(screen.iter().any(|l| l.contains("1 install") && l.contains("2 give me an api ?") && l.contains("Confirm")), "{screen:?}");
        assert!(screen.iter().any(|l| l.contains("goethe api") && l.contains("The API address")));
        page.key(Input::Down);
        page.key(Input::Enter);
        assert_eq!(page.frame.parts[1].chosen, Some(1));
        assert_eq!(page.page, 2, "settled, so on to confirm");
        let confirm = text(page.render(110, 30));
        assert!(confirm.iter().any(|l| l.contains("1 install")) && confirm.iter().any(|l| l.contains("2 api")), "{confirm:?}");
        page.key(Input::Right);
        assert_eq!(page.page, 2, "no page after confirm");
        page.key(Input::Enter);
        assert!(page.frame.running);
    }

    #[test]
    fn typing_rewords_a_part_and_a_part_can_be_left_out() {
        // On a part page every letter starts "type something else" (left/right move its cursor, not the page); leave out takes the part off the plan.
        let mut driver = Said::default();
        let mut page = asked(&mut driver, "install x and give me an api");
        page.key(Input::Right);
        typed(&mut page, "show api");
        assert_eq!(page.edit, Field::Write);
        page.key(Input::Left);
        assert_eq!(page.page, 1, "left moves the cursor inside the text");
        page.key(Input::Enter);
        assert_eq!(page.driver.frame().parts[1].text, "show api");
        assert_eq!(page.page, 2, "the reworded part is settled, so confirm comes next");
        page.key(Input::Left);
        page.key(Input::Left);
        let rows = page.rows();
        page.row = rows.iter().position(|r| *r == Row::LeaveOut).unwrap();
        page.key(Input::Enter);
        assert!(page.frame.parts[0].left_out);
        assert!(text(page.render(110, 30)).iter().any(|l| l.contains("1 install") && l.contains("left out")));
        drop(page);
        assert_eq!(driver.written, [(1, "show api".to_string())]);
    }

    #[test]
    fn a_value_question_sits_under_the_part_and_takes_a_typed_answer() {
        let mut driver = Said::default();
        let mut page = asked(&mut driver, "make a key");
        page.frame.parts[0].question = Some("Key name:".into());
        page.page = 0;
        page.place();
        assert_eq!(page.rows()[page.row], Row::TypeValue, "the cursor goes to what is missing");
        page.key(Input::Enter);
        typed(&mut page, "my-app");
        page.key(Input::Enter);
        page.frame.parts[0].question = Some("Key name:".into());
        page.page = 0;
        page.place();
        typed(&mut page, "ci-bot");
        assert_eq!(page.edit, Field::Answer, "typing on the question answers it, it does not reword the part");
        page.key(Input::Enter);
        drop(page);
        assert_eq!(driver.answered, [(0, "my-app".to_string()), (0, "ci-bot".to_string())]);
        assert!(driver.written.is_empty());
    }

    #[test]
    fn a_run_goes_on_in_the_background_and_esc_returns_to_the_menu() {
        // While it runs, keys do not edit; Esc goes back to the menu (standalone goethe ask stays); Ctrl-C stops it.
        let mut driver = Said::default();
        let mut page = asked(&mut driver, "show status");
        page.key(Input::Enter);
        assert!(page.frame.running && page.every().is_some(), "the page follows the run");
        assert!(text(page.render(110, 30)).iter().any(|l| l.contains("goethe install") && l.contains("running")));
        typed(&mut page, "x");
        assert_eq!(page.request.get(), "show status", "no typing while it runs");
        assert!(matches!(page.key(Input::Interrupt), Flow::Stay));
        assert!(matches!(page.key(Input::Esc), Flow::Leave), "back to the menu, the run goes on");
        page.host = AskHost::Alone;
        assert!(matches!(page.key(Input::Esc), Flow::Stay), "no menu to go back to");
        drop(page);
        assert!(driver.stopped && driver.started == 1);
    }

    #[test]
    fn a_chat_leaves_the_page_for_the_terminal() {
        // A plan marked attached (goethe run's chat screen) is handed the terminal instead of running in the background.
        let mut driver = Said::default();
        let mut page = asked(&mut driver, "chat with it");
        page.frame.attached = true;
        assert!(matches!(page.key(Input::Enter), Flow::Leave));
        assert!(page.attach && !page.frame.running);
    }

    #[test]
    fn a_plan_that_deletes_needs_a_second_enter() {
        let mut driver = Said::default();
        let mut page = asked(&mut driver, "remove it");
        page.frame.confirm = Some("goethe remove org/m cannot be undone. Press Enter again to run it.".into());
        page.key(Input::Enter);
        assert!(!page.frame.running && page.confirming, "first Enter only asks");
        assert!(text(page.render(110, 30)).iter().any(|l| l.contains("cannot be undone")));
        page.key(Input::Esc);
        assert!(!page.confirming, "Esc cancels the question, not Ask");
        page.key(Input::Enter);
        page.key(Input::Enter);
        assert!(page.frame.running);
    }

    #[test]
    fn the_logo_lines_are_one_width() {
        assert!(LOGO.iter().all(|l| l.chars().count() == LOGO[0].chars().count()));
    }

    #[test]
    fn the_menu_lists_the_choices_together_with_the_keys_right_under_them() {
        let t = text(main_menu().render(100, 40));
        assert_eq!(t[1].trim_end(), format!("  {}", LOGO[0]).trim_end());
        assert_eq!(t[5].trim(), format!("Website: {SITE}"));
        let first = t.iter().position(|l| l.contains("Ask") && l.contains("plain English")).unwrap();
        for (i, title) in ["Ask", "Quick setup", "Load/Unload", "Install models", "API keys and gateway"].iter().enumerate() {
            assert!(t[first + i].contains(title), "no gap between choices: {t:?}");
        }
        assert!(t[first + 5].is_empty() && t[first + 6].contains("Status") && t[first + 7].contains("Metrics"), "an empty line before Status: {t:?}");
        assert!(t[first].starts_with('\u{27a4}'));
        assert!(t[first + 8].is_empty() && t[first + 9].contains("Enter open"), "the keys line follows the menu: {t:?}");
        assert!(t[first + 9].contains("theme  ") && t[first + 9].contains("2 sea") && t[first + 9].contains("3 earth"), "{}", t[first + 9]);
        let active = crate::theme::current().name();
        assert!(t[first + 9].contains(&format!("{active}|")), "a bar marks the active theme: {}", t[first + 9]);
        assert!(!text(access_menu().render(100, 40)).iter().any(|l| l.contains("theme  ")), "themes only on the main menu");
        assert_eq!(t.len(), 40);
    }

    #[test]
    fn api_schema_sits_under_a_gap_and_opens_its_schemas_to_the_right() {
        let mut m = access_menu();
        let t = text(m.render(160, 40));
        let public = t.iter().position(|l| l.contains("Public address")).unwrap();
        assert!(t[public + 1].is_empty() && t[public + 2].contains("API schema"), "an empty line before API schema: {t:?}");
        m.at = 3;
        m.sync_side();
        let preview = text(m.render(160, 40));
        let row = preview.iter().position(|l| l.contains("API keys") && !l.contains("gateway")).unwrap();
        assert!(preview[row].contains("openai") && preview[row + 1].contains("anthropic") && preview[row + 2].contains("litellm"), "the submenu starts level with API keys: {preview:?}");
        assert!(!preview.iter().any(|l| l.contains("curl")), "the schema shows once inside");
        if let Some(s) = m.side.as_mut() {
            s.focused = true;
            s.base = "https://127.0.0.1:8443/v1".into();
        }
        let inside = text(m.render(200, 40));
        assert!(inside.iter().any(|l| l.contains("Base URL")), "{inside:?}");
        let curl = inside.iter().position(|l| l.contains("curl -k")).unwrap();
        for l in &inside[curl..curl + 4] {
            assert!(!["API keys", "Network exposure", "Public address", "API schema", "openai", "anthropic", "litellm"].iter().any(|w| l.contains(w)), "a copied curl line carries no menu text: {l}");
        }
        assert!(inside.iter().any(|l| l.contains("Authorization: Bearer API_KEY")));
        m.key(Input::Down);
        let anthropic = text(m.render(200, 40));
        assert!(anthropic.iter().any(|l| l.contains("x-api-key: API_KEY")) && anthropic.iter().any(|l| l.contains("\"model\": \"MODEL_ID\"")));
        assert!(matches!(m.key(Input::Left), Flow::Stay) && !m.side.as_ref().unwrap().focused, "the left arrow goes back to the menu");
        assert!(matches!(m.key(Input::Char('q')), Flow::Quit), "q quits goethe from anywhere");
    }

    fn models_menu() -> Menu {
        let mut m = main_menu();
        m.at = 2;
        m.side = Some(SideState {
            kind: Side::Models,
            at: 0,
            models: vec![("org/a".into(), true), ("org/c".into(), true), ("org/b".into(), false)],
            marked: Vec::new(),
            base: String::new(),
            focused: true,
        });
        m
    }

    #[test]
    fn a_model_loads_in_place_with_its_bar_beside_it_and_then_says_loaded() {
        let mut m = models_menu();
        let t = text(m.render(160, 40));
        let row = |t: &[String], id: &str| t.iter().find(|l| l.contains(id)).cloned().unwrap();
        assert!(row(&t, "org/a").ends_with("loaded") && !row(&t, "org/b").contains("loaded"), "{t:?}");
        assert!(!t.iter().any(|l| l.contains("Not loaded")), "no headings");
        assert_eq!(t.iter().position(|l| l.contains("org/a")), t.iter().position(|l| l.split_whitespace().any(|w| w == "Ask")), "the list starts level with the first choice");
        m.key(Input::Down);
        m.key(Input::Down);
        assert!(matches!(m.key(Input::Enter), Flow::Stay), "loading keeps the menu on screen");
        assert_eq!(m.gpus.as_ref().map(|g| g.model.as_str()), Some("org/b"), "Enter first asks where");
        assert!(matches!(m.key(Input::Enter), Flow::Stay) && m.gpus.is_none(), "Enter on Auto loads");
        assert_eq!(m.job.as_ref().map(|j| j.model.as_str()), Some("org/b"));
        assert!(m.every().is_some(), "the menu refreshes while a model loads");
        m.job.as_mut().unwrap().shown = 30;
        let t = text(m.render(160, 40));
        let b = row(&t, "org/b");
        assert!(b.matches('\u{2588}').count() == 3 && b.contains("30%"), "{b}");
        m.key(Input::Up);
        m.key(Input::Up);
        m.key(Input::Down);
        m.key(Input::Down);
        m.key(Input::Enter);
        assert!(m.note.as_deref().is_some_and(|n| n.contains("one model loads at a time")) || m.job.as_ref().is_some_and(|j| j.model == "org/b"));
        assert!(matches!(m.key(Input::Char('q')), Flow::Quit), "q quits even while a model loads");
    }

    #[test]
    fn the_gpu_choice_marks_allowed_gpus_grays_the_others_and_loads_on_the_marked_ones() {
        let row = |index: u32, allowed: bool, alone: Result<(), String>| GpuRow { index, name: "H100".into(), free_gb: 70.0, allowed, alone };
        let mut g = GpuChoice {
            model: "org/b".into(),
            rows: vec![row(0, false, Err("not allowed".into())), row(1, true, Ok(())), row(2, true, Err("short".into()))],
            at: 0,
            marked: Vec::new(),
            plan: Ok("tp=1  ctx=4096  20.0 GB/GPU".into()),
            view: None,
        };
        let t: Vec<String> = g.lines().iter().map(|l| plain(l)).collect();
        assert!(t[0].contains("\u{25cf} Auto") && t[1].contains("GPU 0") && t[1].contains("not allowed") && !t[1].contains('\u{25cb}'), "{t:?}");
        assert!(t[2].contains("\u{25cb} GPU 1") && t[2].ends_with("fits") && t[3].ends_with("too small alone") && t[5].starts_with("    tp=1"), "{t:?}");
        let col = |l: &str, w: &str| l[..l.find(w).unwrap()].chars().count();
        assert!(col(&t[0], "Auto") == col(&t[1], "GPU") && col(&t[2], "GPU") == col(&t[5], "tp="), "the plan starts under Auto and the GPUs: {t:?}");
        assert_eq!(g.key(Input::Down), None);
        assert_eq!(g.at, 2, "the denied GPU 0 is skipped");
        g.key(Input::Up);
        assert_eq!(g.at, 0, "and skipped on the way back up");
        g.key(Input::Up);
        assert_eq!(g.at, 3, "wrapping from Auto lands on the last allowed GPU");
        g.key(Input::Down);
        g.key(Input::Down);
        g.key(Input::Space);
        g.key(Input::Down);
        g.key(Input::Space);
        assert_eq!(g.marked, vec![1, 2], "several GPUs split one copy");
        assert!(plain(&g.lines()[0]).contains("\u{25cb} Auto"));
        g.at = 0;
        g.key(Input::Space);
        assert!(g.marked.is_empty(), "Auto clears the choice");
        assert_eq!((g.key(Input::Enter), g.key(Input::Left)), (Some(true), Some(false)));
        assert_eq!(gpu_flag(&[1, 3]).as_deref(), Some("--gpu=gpu1,gpu3"));
        assert_eq!(gpu_flag(&[]), None);
    }

    #[test]
    fn only_loaded_models_are_marked_to_unload_together() {
        let mut m = models_menu();
        m.key(Input::Space);
        m.key(Input::Down);
        m.key(Input::Space);
        m.key(Input::Down);
        m.key(Input::Space);
        assert_eq!(m.side.as_ref().unwrap().marked, vec!["org/a".to_string(), "org/c".to_string()], "org/b is not loaded");
        assert!(text(m.render(160, 40)).iter().filter(|l| l.contains('\u{2713}')).count() == 2);
    }

    #[test]
    fn load_unload_is_gray_and_skipped_until_a_model_is_installed() {
        let mut m = main_menu();
        m.at = 1;
        m.off = vec![false, false, true, false, false, false, false];
        let t = text(m.render(160, 40));
        assert!(t.iter().any(|l| l.contains("Load/Unload") && l.contains("(install a model first)")), "{t:?}");
        m.key(Input::Down);
        assert_eq!(m.at, 3, "the arrow keys skip it");
        m.key(Input::Up);
        assert_eq!(m.at, 1);
    }

    #[test]
    fn a_load_climbs_one_percent_at_a_time_on_a_bar_no_wider() {
        let mut j = LoadJob::new("org/m".into(), None);
        j.phase = Phase::Loading { instance: "i".into(), gpus: vec![3], need_mb: 1000, base_mb: 0 };
        j.frac = 0.3;
        j.ticks = 1;
        j.tick();
        assert_eq!(j.shown, 1, "the percent climbs one at a time");
        assert_eq!(plain(&fine_bar(31)).chars().count(), 10, "a finer bar, no wider");
        assert_ne!(plain(&fine_bar(31)), plain(&fine_bar(30)), "one percent shows");
        j.phase = Phase::Done;
        assert!(!j.finished(), "not before the bar is full");
        j.shown = 100;
        assert!(j.finished());
        j.phase = Phase::Failed("the engine failed to start (instance i)\n  noise\n  full engine log: goethe logs i -n 200".into());
        assert_eq!(j.failure().as_deref(), Some("the engine failed to start (instance i)\nfull engine log: goethe logs i -n 200"));
    }

    #[test]
    fn quick_setup_ends_with_the_three_schemas_and_the_example_below_them() {
        let mut p = SchemaPage { model: "org/m".into(), key: "goethe_sk_x".into(), base: "https://100.64.1.2:8443/v1".into(), at: 0 };
        let t = text(p.render(200, 40));
        assert!(t.iter().any(|l| l.contains("Your API is ready")) && t.iter().any(|l| l.contains("goethe_sk_x")));
        assert!(t.iter().any(|l| l.contains("Base URL") && l.contains("https://100.64.1.2:8443/v1")), "{t:?}");
        let curl = t.iter().position(|l| l.contains("curl -k")).unwrap();
        assert!(t[curl..curl + 4].iter().all(|l| !l.contains("openai") && !l.contains("anthropic") && !l.contains("litellm")), "{t:?}");
        assert!(t.iter().any(|l| l.contains("Bearer API_KEY")), "the schema keeps its placeholders");
        p.key(Input::Down);
        p.key(Input::Down);
        assert!(text(p.render(200, 40)).iter().any(|l| l.contains("x-litellm-api-key: API_KEY")));
    }

    #[test]
    fn up_moves_metrics_to_the_next_range_and_down_back() {
        let mut p = MetricsPage(crate::metrics_view::MetricsView { range: 1, ..Default::default() });
        p.key(Input::Up);
        assert_eq!(p.0.range, 2, "up: 24 h to 7 d");
        p.key(Input::Down);
        p.key(Input::Down);
        assert_eq!(p.0.range, 0, "down: back to 1 h");
    }

    #[test]
    fn run_without_a_model_lists_the_installed_ones_at_the_top() {
        let mut p = ModelPick { models: vec![("org/a".into(), true), ("org/b".into(), false)], at: 0, job: None, ready: None, note: None };
        let t = text(p.render(100, 20));
        assert!(t[1].contains("Choose a model to run") && t[3].starts_with('\u{27a4}') && t[3].contains("org/a") && t[3].contains("loaded") && t[4].contains("org/b"), "{t:?}");
        assert!(matches!(p.key(Input::Enter), Flow::Pick(0)), "a loaded model runs at once");
        assert!(matches!(p.key(Input::Char('q')), Flow::Quit) && matches!(p.key(Input::Left), Flow::Quit), "q and the left arrow leave without running");
        assert!(matches!(p.key(Input::Up), Flow::Stay) && p.at == 1, "up from the first wraps to the last");
        assert!(matches!(p.key(Input::Enter), Flow::Stay) && p.job.as_ref().is_some_and(|j| j.model == "org/b"), "one that is not loaded loads on the list");
        p.job.as_mut().unwrap().shown = 40;
        let t = text(p.render(100, 20));
        assert!(t[4].contains("org/b") && t[4].contains("40%") && t.iter().any(|l| l.contains("loading...")), "{t:?}");
        assert!(matches!(p.key(Input::Down), Flow::Stay) && p.at == 1, "the list holds still while it loads");
    }

    #[test]
    fn digits_and_enter_pick_and_q_leaves() {
        let mut m = main_menu();
        m.key(Input::Up);
        assert!(matches!(m.key(Input::Enter), Flow::Pick(6)), "up from the first wraps to the last");
        assert!(matches!(m.key(Input::Right), Flow::Stay), "only Enter opens a choice without a submenu");
        assert!(matches!(m.key(Input::Left), Flow::Stay), "the left arrow on the top menu does not quit goethe");
        assert!(matches!(m.key(Input::Char('q')), Flow::Quit), "q quits goethe");
        assert!(matches!(access_menu().key(Input::Left), Flow::Leave), "a submenu goes back with the left arrow");
        assert!(matches!(access_menu().key(Input::Char('q')), Flow::Quit), "q quits from a submenu too");
        assert!(matches!(access_menu().key(Input::Right), Flow::Stay), "the right arrow opens only what is already shown on the right");
    }

    #[test]
    fn the_picker_shows_vram_and_storage_then_aligned_columns() {
        let t = text(picker().render(120, 30));
        assert!(t.iter().any(|l| l.contains("VRAM") && l.contains("382.4 GB total") && l.contains("95.6 GB allowed (gpu3)")), "{t:?}");
        assert!(t.iter().any(|l| l.contains("Storage") && l.contains("3.5 TB total") && l.contains("1.2 TB free") && l.contains("120.4 GB used by your models")));
        let head = t.iter().position(|l| l.contains("MODEL")).unwrap();
        let chars = |l: &str| l.chars().collect::<Vec<char>>();
        let h = chars(&t[head]);
        let end_of = |w: &str| (0..h.len()).find(|&i| h[i..].starts_with(&w.chars().collect::<Vec<_>>())).unwrap() + w.len();
        for l in &t[head + 1..head + 4] {
            let c = chars(l);
            assert_eq!(c[end_of("SIZE") - 2..end_of("SIZE")].iter().collect::<String>(), "GB", "{l}");
            assert_eq!(c[end_of("VRAM") - 2..end_of("VRAM")].iter().collect::<String>(), "GB", "{l}");
            assert_eq!(c[end_of("GPUS") - 1], '1', "{l}");
        }
    }

    #[test]
    fn a_mac_picker_says_gpu_memory_and_drops_the_gpus_column() {
        let mut p = picker();
        (p.data.gpu_count, p.data.unified, p.data.allowed_gpus) = (1, true, vec![0]);
        let t = text(p.render(120, 30));
        assert!(t.iter().any(|l| l.contains("GPU memory") && l.contains("382.4 GB total") && !l.contains("allowed")), "{t:?}");
        let head = t.iter().find(|l| l.contains("MODEL")).unwrap();
        assert!(head.contains("MEMORY") && !head.contains("VRAM") && !head.contains("GPUS"), "{head}");
    }

    #[test]
    fn custom_model_and_token_lines_sit_under_the_table() {
        let t = text(picker().render(120, 30));
        let custom = t.iter().position(|l| l.contains("Custom model:")).unwrap();
        assert!(t[custom].contains("(org/model)"), "{t:?}");
        assert!(t[custom + 1].contains("HF token not set:  (optional)"), "{t:?}");
        assert!(t[custom - 2].contains("org/large"), "after the last model and a gap: {t:?}");
        assert!(t[custom + 3].contains("Space select"), "then the keys line: {t:?}");
    }

    #[test]
    fn space_selects_models_to_install_and_installed_ones_to_uninstall() {
        let mut p = picker();
        let line = |p: &Picker, id: &str| text(p.render(120, 30)).into_iter().find(|l| l.contains(id)).unwrap();
        assert!(line(&p, "org/mine").contains("\u{25cf} org/mine"), "installed models have a filled dot, not indented");
        assert!(!text(p.render(120, 30)).iter().any(|l| l.contains("install:")), "nothing chosen, no plan");
        p.key(Input::Space);
        assert!(line(&p, "org/small").contains("\u{25cf}   org/small"), "{}", line(&p, "org/small"));
        assert!(line(&p, "org/large").contains("\u{25cb} org/large"));
        p.key(Input::Down);
        p.key(Input::Space);
        assert!(p.chosen[1] && p.status.is_empty());
        assert!(line(&p, "org/mine").contains("\u{25cf}   org/mine"), "chosen for uninstall: indented");
        assert_eq!(name_style(&p.data.rows[1], false), "1", "installed names are bold");
        assert_eq!(name_style(&p.data.rows[0], false), "");
        let t = text(p.render(120, 30));
        let at = t.iter().position(|l| l.trim() == "install: org/small").unwrap_or_else(|| panic!("{t:?}"));
        assert_eq!(t[at + 1].trim(), "uninstall: org/mine");
        assert!(t[at + 3].contains("Enter apply"), "the keys stay under the plan: {t:?}");
        assert!(matches!(p.key(Input::Enter), Flow::Pick(0)));
        assert_eq!((p.installs(), p.removals()), (vec!["org/small".to_string()], vec!["org/mine".to_string()]));
        p.key(Input::Space);
        p.key(Input::Up);
        p.key(Input::Space);
        assert!(p.installs().is_empty() && p.removals().is_empty(), "Space again drops the choice");
        assert!(matches!(p.key(Input::Enter), Flow::Stay) && p.status.starts_with("Select models"));
    }

    #[test]
    fn a_custom_model_that_is_already_chosen_or_installed_is_not_installed_twice() {
        let mut p = picker();
        p.key(Input::Space);
        p.custom = "org/small".into();
        assert_eq!(p.installs(), ["org/small"]);
        p.custom = "org/mine".into();
        assert_eq!(p.installs(), ["org/small"]);
    }

    #[test]
    fn a_custom_model_is_typed_and_installed_with_the_chosen_ones() {
        let mut p = picker();
        p.key(Input::Space);
        for _ in 0..3 {
            p.key(Input::Down);
        }
        assert_eq!(p.spot(), Spot::Custom);
        for c in "qwen/q-1x".chars() {
            p.key(Input::Char(c));
        }
        p.key(Input::Backspace);
        assert_eq!(p.custom, "qwen/q-1", "q and j type while on a field");
        assert!(text(p.render(120, 30)).iter().any(|l| l.contains("Custom model:  qwen/q-1|")));
        assert!(matches!(p.key(Input::Enter), Flow::Pick(0)));
        assert_eq!(p.installs(), ["org/small", "qwen/q-1"]);
        p.custom = "not an id".into();
        assert!(matches!(p.key(Input::Enter), Flow::Stay) && p.status.starts_with("Custom model:"));
    }

    #[test]
    fn the_token_is_masked_while_typed() {
        let mut p = picker();
        p.at = p.data.rows.len() + 1;
        for c in "hf_abc".chars() {
            p.key(Input::Char(c));
        }
        let t = text(p.render(120, 30));
        assert!(t.iter().any(|l| l.contains("HF token:  \u{2022}\u{2022}\u{2022}\u{2022}\u{2022}\u{2022}|")), "{t:?}");
        assert!(!t.iter().any(|l| l.contains("hf_abc")));
    }

    fn share_page(shared: &[(&str, i64, &[&str])]) -> SharePage {
        SharePage {
            shared: shared
                .iter()
                .map(|(n, u, a)| (n.to_string(), crate::public_share::Grant { key: format!("id-{n}"), until: *u, allow: a.iter().map(|c| c.to_string()).collect() }))
                .collect(),
            url: (!shared.is_empty()).then(|| "https://203.0.113.1:8444/v1".to_string()),
            names: vec!["app".into(), "ci".into()],
            key: String::new(),
            duration: "5h".into(),
            allow: String::new(),
            at: shared.len(),
            armed: None,
            status: String::new(),
            error: false,
        }
    }

    #[test]
    fn a_share_needs_who_may_connect_and_everyone_is_said_on_purpose() {
        let mut p = share_page(&[]);
        p.key = "app".into();
        p.share();
        assert!(p.error && p.status.contains("Type who may connect"), "an empty list never opens it: {}", p.status);
        assert_eq!(p.slot(), Slot::Allow, "the cursor goes to the addresses");
        p.allow = "0.0.0.0/0".into();
        p.share();
        assert!(p.error && p.status.contains("type everyone instead"), "{}", p.status);
        assert!(text(p.render(160, 40)).iter().any(|l| l.contains("Allowed IPs:")));
        let mut q = share_page(&[]);
        q.at = q.slots().iter().position(|s| *s == Slot::Allow).unwrap();
        assert!(text(q.render(200, 40)).iter().any(|l| l.contains("required: who may connect") && l.contains("everyone")));
    }

    #[test]
    fn the_public_page_lists_shared_keys_and_takes_a_key_a_time_and_addresses() {
        let soon = crate::public_share::now() + 3 * 3600;
        let mut p = share_page(&[("app", soon, &["203.0.113.7/32"])]);
        let t = text(p.render(140, 40));
        assert!(t.iter().any(|l| l.contains("Gateway") && l.contains("https://203.0.113.1:8444/v1")), "{t:?}");
        assert!(t.iter().any(|l| l.contains("app") && l.contains("left)") && l.contains("from 203.0.113.7/32")), "{t:?}");
        assert!(t.iter().any(|l| l.contains("API key:") && l.contains('|') && l.contains("keys: app, ci")), "typing starts on the key: {t:?}");
        for c in "ci".chars() {
            p.key(Input::Char(c));
        }
        p.key(Input::Enter);
        assert_eq!(p.slot(), Slot::For, "Enter moves to the time");
        assert_eq!(p.duration, "5h");
        p.key(Input::Down);
        for c in "10.0.0.0/8".chars() {
            p.key(Input::Char(c));
        }
        assert_eq!((p.key.as_str(), p.allow.as_str()), ("ci", "10.0.0.0/8"));
        assert!(matches!(p.key(Input::Char('q')), Flow::Stay), "q is a letter while typing");
        assert!(text(p.render(140, 40)).iter().any(|l| l.contains("Stop sharing every key")));
        assert!(matches!(p.key(Input::Left), Flow::Leave), "the left arrow goes back");
    }

    #[test]
    fn stopping_a_key_or_the_gateway_asks_twice() {
        let soon = crate::public_share::now() + 3600;
        let mut p = share_page(&[("app", soon, &[])]);
        p.at = 0;
        assert_eq!(p.slot(), Slot::Shared(0));
        p.key(Input::Enter);
        assert!(p.status.contains("Press Enter again to stop sharing app") && p.armed == Some(Slot::Shared(0)));
        p.key(Input::Down);
        assert!(p.armed.is_none(), "a move disarms it");
        let closed = share_page(&[]);
        assert!(!closed.slots().contains(&Slot::StopAll), "nothing to stop while closed");
        assert!(text(closed.render(140, 40)).iter().any(|l| l.contains("Gateway") && l.contains("closed")));
    }

    #[test]
    fn the_api_menu_names_network_exposure_and_the_public_address() {
        let t = text(access_menu().render(140, 40));
        assert!(t.iter().any(|l| l.contains("Network exposure") && l.contains("local, vpn/private, public")), "{t:?}");
        assert!(t.iter().any(|l| l.contains("Public address") && l.contains("temporarily share models through a hardened gateway")), "{t:?}");
    }

    #[test]
    fn no_line_is_wider_than_the_terminal() {
        let mut p = picker();
        p.data.rows.push(row("some-organisation/a-very-long-model-name-that-goes-on-and-on-NVFP4", false));
        p.chosen.push(false);
        for w in [60, 80, 100, 160] {
            for l in p.render(w, 24).iter().chain(main_menu().render(w, 30).iter()) {
                assert!(crate::term::visible_width(l) <= w, "{w}: {l}");
            }
        }
    }

    #[test]
    fn enter_with_nothing_chosen_explains() {
        let mut p = picker();
        assert!(matches!(p.key(Input::Enter), Flow::Stay));
        assert!(p.status.contains("Space"));
    }
}
