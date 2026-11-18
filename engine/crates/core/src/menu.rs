// goethe with no arguments on a terminal: the main menu (Quick setup, Install models, API keys and gateway, Status) and its pages.

use crate::keys_menu::{fit, frame_update, parse_input, screen_size, window, Input, Screen};
use crate::models::{PickData, PickRow};
use crate::paths::goethe_home;
use crate::term::{paint, Color, RawMode};
use std::io::{IsTerminal, Write};

const LOGO: [&str; 4] = [
    "  ___   ___   ___  _____  _  _   ___ ",
    " / __| / _ \\ | __||_   _|| || | | __|",
    "| (_ || (_) || _|   | |  | __ | | _| ",
    " \\___| \\___/ |___|  |_|  |_||_| |___|",
];
const SITE: &str = "https://goethe.sh/";
const GAP: &str = "   ";
fn hl() -> String {
    // The theme's main color, bold: the highlighted entry, the cursor and the logo.
    format!("1;{}", crate::theme::main())
}

fn mark() -> String {
    // The theme's secondary color: selection dots and text carets.
    format!("1;{}", crate::theme::secondary())
}

fn sgr(code: &str, text: &str) -> String {
    if text.is_empty() || !crate::term::color_enabled() {
        return text.to_string();
    }
    format!("\x1b[{code}m{text}\x1b[0m")
}

fn gray(text: &str) -> String {
    paint(text, Color::Gray)
}

fn plain(s: &str) -> String {
    // The text without its color codes.
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

fn clip(line: String, width: usize) -> String {
    // A line wider than the terminal loses its colors and is cut with an ellipsis, so nothing wraps.
    if crate::term::visible_width(&line) <= width {
        line
    } else {
        fit(&plain(&line), width).trim_end().to_string()
    }
}

fn screen(mut lines: Vec<String>, width: usize, height: usize, footer: String) -> Vec<String> {
    // The content, a gap and the keys line right under it, padded to exactly height lines.
    lines.truncate(height.saturating_sub(2));
    lines.push(String::new());
    lines.push(format!("  {footer}"));
    lines.resize(height, String::new());
    lines.into_iter().map(|l| clip(l, width)).collect()
}

enum Flow {
    Stay,
    Leave,
    Quit,
    Pick(usize),
}

trait Page {
    fn render(&self, width: usize, height: usize) -> Vec<String>;
    fn key(&mut self, input: Input) -> Flow;
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
        let mut buf = [0u8; 16];
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
            for input in parse_input(&buf[..n]) {
                match page.key(input) {
                    Flow::Stay => {}
                    Flow::Leave => return Ok(None),
                    Flow::Quit => {
                        self.quit = true;
                        return Ok(None);
                    }
                    Flow::Pick(i) => return Ok(Some(i)),
                }
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

fn installed_and_loaded(home: &std::path::Path) -> Vec<(String, bool)> {
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
            .collect(),
    }
}

fn pad(line: &str, width: usize) -> String {
    format!("{line}{}", " ".repeat(width.saturating_sub(crate::term::visible_width(line))))
}

fn columns(out: &mut Vec<String>, start: usize, left_width: usize, middle: &[String], right: &[String]) {
    // Lay a submenu column, and the content beside it, to the right of the lines from `start` on.
    let mw = middle.iter().map(|l| crate::term::visible_width(l)).max().unwrap_or(0);
    for k in 0..middle.len().max(right.len()) {
        if out.len() <= start + k {
            out.push(String::new());
        }
        let left = pad(&out[start + k], left_width);
        let mid = middle.get(k).map_or_else(|| " ".repeat(mw), |m| pad(m, mw));
        let rest = right.get(k).map_or(String::new(), |r| format!("{GAP}{r}"));
        out[start + k] = format!("{left}{mid}{rest}").trim_end().to_string();
    }
}

enum Phase {
    Up(Option<std::process::Child>),
    Start,
    Loading { instance: String, gpus: Vec<u32>, need_mb: u64, base_mb: u64 },
    Done,
    Failed(String),
}

struct LoadJob {
    model: String,
    phase: Phase,
    frac: f64,
    shown: u32,
    ticks: u32,
}

fn fine_bar(percent: u32) -> String {
    // Ten cells, each filled in eighths, so the bar moves in steps of about one percent without getting wider.
    let eighths = (percent.min(100) * 80 + 50) / 100;
    let (full, part) = ((eighths / 8) as usize, (eighths % 8) as usize);
    let partial = ["", "\u{258f}", "\u{258e}", "\u{258d}", "\u{258c}", "\u{258b}", "\u{258a}", "\u{2589}"][part];
    let used = full + usize::from(!partial.is_empty());
    format!("{}{}", sgr(&crate::theme::main(), &format!("{}{partial}", "\u{2588}".repeat(full))), gray(&"\u{2591}".repeat(10 - used)))
}

fn goethe_cmd() -> std::process::Command {
    let mut c = std::process::Command::new(std::env::current_exe().unwrap_or_else(|_| "goethe".into()));
    c.stdin(std::process::Stdio::null()).stderr(std::process::Stdio::piped());
    c
}

impl LoadJob {
    fn new(model: String) -> LoadJob {
        LoadJob { model, phase: Phase::Up(None), frac: 0.0, shown: 0, ticks: 0 }
    }

    fn managed_mb(gpus: &[u32]) -> u64 {
        // VRAM goethe's engines hold on these GPUs now.
        let home = goethe_home();
        let managed = crate::db::open(&home).map(|c| crate::placement::managed_pids(&c)).unwrap_or_default();
        crate::hw::gpu_usage(&managed).unwrap_or_default().iter().filter(|u| gpus.contains(&u.index)).map(|u| u.managed_mb).sum()
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
            Phase::Start => match goethe_cmd().args(["load", &self.model, "--no-wait", "--json", "--yes"]).output() {
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
                            base_mb = Self::managed_mb(&gpus);
                        }
                        if need_mb > 0 {
                            let grown = Self::managed_mb(&gpus).saturating_sub(base_mb) as f64 / need_mb as f64;
                            self.frac = self.frac.max(grown.min(0.95));
                        }
                        Phase::Loading { instance, gpus, need_mb, base_mb }
                    }
                }
            }
            other => other,
        };
    }

    fn tick(&mut self) {
        // The percent climbs one at a time toward what the load measured; goethed is asked every half second.
        if self.ticks.is_multiple_of(8) {
            self.advance();
        }
        self.ticks += 1;
        if self.shown < (self.frac * 100.0).round() as u32 {
            self.shown += 1;
        }
    }

    fn finished(&self) -> bool {
        matches!(self.phase, Phase::Done) && self.shown >= 100
    }

    fn failure(&self) -> Option<String> {
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
    note: Option<String>,
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
            let detail = if side.is_some() { "" } else { c.detail };
            out.push(match (i == self.at, focused) {
                (true, false) => format!("{} {}{GAP}{}", sgr(&hl(), "\u{27a4}"), sgr(&hl(), &title), detail),
                (true, true) => format!("  {}", paint(&title, Color::Bold)),
                _ => format!("  {title}{GAP}{}", gray(detail)),
            });
        }
        if let Some(s) = side {
            let right = if focused && s.kind == Side::Schemas { crate::api_schema::example(crate::api_schema::NAMES[s.at], &s.base, crate::api_schema::MODEL, crate::api_schema::KEY) } else { Vec::new() };
            columns(&mut out, start, 2 + tw + GAP.len(), &side_rows(s, self.job.as_ref()), &right);
        }
        if let Some(n) = &self.note {
            out.push(String::new());
            out.extend(n.lines().map(|l| format!("  {}", paint(l, Color::Red))));
        }
        let hint = match side.filter(|s| s.focused).map(|s| s.kind) {
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
                            None => self.job = Some(LoadJob::new(id)),
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
        self.job.as_ref().map(|_| std::time::Duration::from_millis(60))
    }

    fn tick(&mut self) {
        // The load beside its model: when it runs, the bar gives way to "loaded"; when it fails, the reason shows under the list.
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
        columns(&mut out, start, 0, &names, &crate::api_schema::example(crate::api_schema::NAMES[self.at], &self.base, crate::api_schema::MODEL, crate::api_schema::KEY));
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

fn brand() -> Vec<String> {
    // The small GOETHE inscription in the theme's main color, the site below it in the third.
    let mut out = vec![String::new()];
    out.extend(LOGO.iter().map(|l| format!("  {}", sgr(&crate::theme::main(), l))));
    out.push(format!("  {}", sgr(&crate::theme::third(), SITE)));
    out.push(String::new());
    out
}

fn themes_line() -> String {
    // The themes on the main menu's keys line: number and name, the active one in its main color with a bar after it.
    let now = crate::theme::current();
    let items: Vec<String> = crate::theme::ALL
        .iter()
        .enumerate()
        .map(|(i, t)| if *t == now { format!("{}{}", sgr(&hl(), &format!("{} {}", i + 1, t.name())), sgr(&hl(), "|")) } else { gray(&format!("{} {} ", i + 1, t.name())) })
        .collect();
    format!("{}  {}", gray("theme"), items.join("  "))
}

fn main_menu() -> Menu {
    Menu {
        heading: Vec::new(),
        items: vec![
            Choice { title: "Quick setup", detail: "install a model, serve it and open the API", side: None },
            Choice { title: "Load/Unload", detail: "load an installed model, unload loaded ones", side: Some(Side::Models) },
            Choice { title: "Install models", detail: "pick models that fit your GPUs", side: None },
            Choice { title: "API keys and gateway", detail: "keys, who can reach the API, a public address", side: None },
            Choice { title: "Status", detail: "Goethe, the API and the models it serves", side: None },
        ],
        at: 0,
        hint: "\u{2191}\u{2193} move   Enter open   q quit",
        top: true,
        gap_before: None,
        side: None,
        off: Vec::new(),
        job: None,
        note: None,
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
        note: None,
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

pub fn run() -> Result<(), String> {
    // goethe: the main menu until q; each choice runs, then the menu comes back.
    if !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        return Err("goethe alone needs a terminal; goethe help lists the commands".to_string());
    }
    let mut tty = Tty::open()?;
    let mut menu = main_menu();
    loop {
        menu.refresh_off();
        match tty.run(&mut menu)? {
            None => return Ok(()),
            Some(0) => {
                tty.inside_with(crate::onboard::run, false)?;
                match crate::onboard::take_ready() {
                    Some((_, model, key)) => drop(tty.run(&mut SchemaPage { model, key, base: crate::api_schema::base_url(&goethe_home()), at: 0 })?),
                    None => tty.wait_back()?,
                }
            }
            Some(1) => {}
            Some(2) => install_page(&mut tty)?,
            Some(3) => access_page(&mut tty)?,
            Some(_) => {
                tty.draw(&Notice("Reading the GPUs...".to_string()));
                let mut page = StatusPage(crate::status_view::StatusView::new());
                tty.run(&mut page)?;
            }
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
                    Ok(session) => tty.quit = session.exit,
                    Err(e) => report(Err(e)),
                }
                tty.resume()?;
            }
            Some(1) => tty.inside(|| {
                crate::configure::run_api(crate::configure::ApiOpts { mode: None, vpn_ip: None, interface: None, allow: Vec::new(), yes: false })
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
            let ip = crate::platform::lookup_public_ip(&home).map(|i| i.to_string()).unwrap_or_else(|_| "<public IP>".into());
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
        let home = goethe_home();
        match share::open(&home, share::OpenArgs { secs, keys: std::slice::from_ref(&name), allow: &allow, port: None }) {
            Ok(o) => {
                self.reload();
                self.key.clear();
                self.allow.clear();
                let warn = o.warnings.first().map(|w| format!("  Note: {w}")).unwrap_or_default();
                self.say(format!("Shared {name} until {}; send {} with the key.{warn}", share::local_time(share::now() + secs), o.cert.display()), false);
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
        out.push(field(Slot::Allow, "Allowed IPs:", &self.allow, "optional: any address, or e.g. 203.0.113.7, 198.51.100.0/24"));
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

fn gb(v: f64) -> String {
    let v = v.max(0.0);
    if v >= 1000.0 {
        format!("{:.1} TB", v / 1000.0)
    } else {
        format!("{v:.1} GB")
    }
}

fn name_style(r: &PickRow, here: bool) -> &'static str {
    // Installed models and the one under the cursor have bold names.
    if r.installed || here { "1" } else { "" }
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
                (true, _) => sgr(&mark(), "\u{25cf}"),
                (false, true) => sgr(&crate::theme::secondary(), "\u{25cf}"),
                _ => "\u{25cb}".to_string(),
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

    fn installs(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.data.rows.iter().zip(&self.chosen).filter(|(_, c)| **c).map(|(r, _)| r.id.clone()).collect();
        if !self.custom.trim().is_empty() {
            ids.push(self.custom.trim().to_string());
        }
        ids
    }

    fn hint(&self) -> String {
        let n = self.installs().len();
        let lead = if n > 0 { format!("{n} selected   ") } else { String::new() };
        let keys = match self.spot() {
            Spot::Model(_) => "\u{2191}\u{2193} move   Space select   Enter install   \u{2190} back",
            Spot::Custom => "type a Hugging Face id such as org/model   Enter install   \u{2191}\u{2193} move   \u{2190} back",
            Spot::Token => "paste a Hugging Face token   Enter use it for these installs   \u{2191}\u{2193} move   \u{2190} back",
        };
        format!("{lead}{keys}")
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
        if self.installs().is_empty() {
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
        let rows = height.saturating_sub(out.len() + 1 + 3 + 3).max(1);
        out.extend(self.table(width, rows));
        out.push(String::new());
        out.extend(self.fields());
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
            Input::Space => match self.spot() {
                Spot::Model(i) if self.data.rows[i].installed => self.status = format!("{} is already installed", self.data.rows[i].id),
                Spot::Model(i) => self.chosen[i] = !self.chosen[i],
                _ => {}
            },
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
    // The table of models that fit; the chosen ones are installed one after the other on the normal screen.
    let home = goethe_home();
    tty.draw(&Notice("Reading the catalog and the GPUs...".to_string()));
    let data = crate::placement::ensure_hw_profile(&home).and_then(|_| crate::models::picker_data(&home));
    let mut picker = match data {
        Ok(d) => Picker::new(d),
        Err(e) => return tty.inside(|| Err(e)),
    };
    if tty.run(&mut picker)?.is_none() {
        return Ok(());
    }
    let ids = picker.installs();
    tty.inside(|| {
        for id in ids {
            println!("{}", paint(&format!("Installing {id}"), Color::Bold));
            report(crate::install::run(crate::install::InstallOpts {
                model_id: id,
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
            }));
        }
        Ok(())
    })
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

    #[test]
    fn the_logo_lines_are_one_width() {
        assert!(LOGO.iter().all(|l| l.chars().count() == LOGO[0].chars().count()));
    }

    #[test]
    fn the_menu_lists_the_choices_together_with_the_keys_right_under_them() {
        let t = text(main_menu().render(100, 40));
        assert_eq!(t[1].trim_end(), format!("  {}", LOGO[0]).trim_end());
        assert_eq!(t[5].trim(), SITE);
        let first = t.iter().position(|l| l.contains("Quick setup")).unwrap();
        for (i, title) in ["Quick setup", "Load/Unload", "Install models", "API keys and gateway", "Status"].iter().enumerate() {
            assert!(t[first + i].contains(title), "no gap between choices: {t:?}");
        }
        assert!(t[first].starts_with('\u{27a4}'));
        assert!(t[first + 5].is_empty() && t[first + 6].contains("Enter open"), "the keys line follows the menu: {t:?}");
        assert!(t[first + 6].contains("theme  ") && t[first + 6].contains("2 sea") && t[first + 6].contains("3 earth"), "{}", t[first + 6]);
        let active = crate::theme::current().name();
        assert!(t[first + 6].contains(&format!("{active}|")), "a bar marks the active theme: {}", t[first + 6]);
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
        assert!(inside.iter().any(|l| l.contains("openai") && l.contains("Base URL")), "{inside:?}");
        assert!(inside.iter().any(|l| l.contains("Authorization: Bearer API_KEY")));
        m.key(Input::Down);
        let anthropic = text(m.render(200, 40));
        assert!(anthropic.iter().any(|l| l.contains("x-api-key: API_KEY")) && anthropic.iter().any(|l| l.contains("\"model\": \"MODEL_ID\"")));
        assert!(matches!(m.key(Input::Left), Flow::Stay) && !m.side.as_ref().unwrap().focused, "the left arrow goes back to the menu");
        assert!(matches!(m.key(Input::Char('q')), Flow::Quit), "q quits goethe from anywhere");
    }

    fn models_menu() -> Menu {
        let mut m = main_menu();
        m.at = 1;
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
        assert!(t.iter().position(|l| l.contains("org/a")) == t.iter().position(|l| l.contains("Quick setup")), "the list starts level with the first choice");
        m.key(Input::Down);
        m.key(Input::Down);
        assert!(matches!(m.key(Input::Enter), Flow::Stay), "loading keeps the menu on screen");
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
        m.off = vec![false, true, false, false, false];
        let t = text(m.render(160, 40));
        assert!(t.iter().any(|l| l.contains("Load/Unload") && l.contains("(install a model first)")), "{t:?}");
        m.key(Input::Down);
        assert_eq!(m.at, 2, "the arrow keys skip it");
        m.key(Input::Up);
        assert_eq!(m.at, 0);
    }

    #[test]
    fn a_load_climbs_one_percent_at_a_time_on_a_bar_no_wider() {
        let mut j = LoadJob::new("org/m".into());
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
    fn quick_setup_ends_with_the_three_schemas_beside_each_other() {
        let mut p = SchemaPage { model: "org/m".into(), key: "goethe_sk_x".into(), base: "https://100.64.1.2:8443/v1".into(), at: 0 };
        let t = text(p.render(200, 40));
        assert!(t.iter().any(|l| l.contains("Your API is ready")) && t.iter().any(|l| l.contains("goethe_sk_x")));
        assert!(t.iter().any(|l| l.contains("openai") && l.contains("https://100.64.1.2:8443/v1")), "{t:?}");
        assert!(t.iter().any(|l| l.contains("Bearer API_KEY")), "the schema keeps its placeholders");
        p.key(Input::Down);
        p.key(Input::Down);
        assert!(text(p.render(200, 40)).iter().any(|l| l.contains("x-litellm-api-key: API_KEY")));
    }

    #[test]
    fn digits_and_enter_pick_and_q_leaves() {
        let mut m = main_menu();
        m.key(Input::Up);
        assert!(matches!(m.key(Input::Enter), Flow::Pick(4)), "up from the first wraps to the last");
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
    fn space_selects_and_indents_installed_models_cannot_be_picked() {
        let mut p = picker();
        p.key(Input::Space);
        let line = |p: &Picker, id: &str| text(p.render(120, 30)).into_iter().find(|l| l.contains(id)).unwrap();
        assert!(line(&p, "org/small").contains("\u{25cf}   org/small"), "{}", line(&p, "org/small"));
        assert!(line(&p, "org/large").contains("\u{25cb} org/large"));
        p.key(Input::Down);
        p.key(Input::Space);
        assert!(p.status.contains("already installed") && !p.chosen[1]);
        assert_eq!(name_style(&p.data.rows[1], false), "1", "installed names are bold");
        assert!(line(&p, "org/mine").contains("\u{25cf} org/mine"), "installed models have a filled dot, not indented");
        assert_eq!(name_style(&p.data.rows[0], false), "");
        assert!(matches!(p.key(Input::Enter), Flow::Pick(0)));
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
