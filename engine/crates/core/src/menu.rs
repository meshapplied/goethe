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
}

struct Tty {
    screen: Option<Screen>,
    raw: Option<RawMode>,
    last: Vec<String>,
}

impl Tty {
    fn open() -> Result<Tty, String> {
        let mut t = Tty { screen: None, raw: None, last: Vec::new() };
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
        // Something that prints and asks as a normal command does (setup, status, an install), on the cleared menu screen; then the left arrow goes back, at once when it was left with the left arrow.
        self.raw = None;
        print!("\x1b[H\x1b[2J\x1b[?25h");
        let _ = std::io::stdout().flush();
        let out = f();
        let cancelled = out.as_ref().is_err_and(|e| e.starts_with("cancelled"));
        report(out.map_err(|e| if cancelled { String::new() } else { e }));
        print!("\x1b[?25l");
        if !cancelled {
            print!("\n\n  {}", gray("\u{2190} back"));
        }
        let _ = std::io::stdout().flush();
        self.resume()?;
        if !cancelled {
            wait_back();
        }
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
        // Keys go to the page until it leaves or picks something; the screen stays open either way.
        let mut size = screen_size();
        self.last.clear();
        self.draw(page);
        let mut buf = [0u8; 16];
        let every = page.every();
        let wait = every.map_or(250, |d| d.as_millis() as i32);
        let mut next = std::time::Instant::now() + every.unwrap_or_default();
        loop {
            let mut fds = libc::pollfd { fd: 0, events: libc::POLLIN, revents: 0 };
            let left = next.saturating_duration_since(std::time::Instant::now()).as_millis() as i32;
            if unsafe { libc::poll(&mut fds, 1, if every.is_some() { left.max(1) } else { wait }) } <= 0 {
                if screen_size() != size {
                    size = screen_size();
                    self.last.clear();
                }
                if every.is_some() && std::time::Instant::now() >= next {
                    page.tick();
                    next = std::time::Instant::now() + every.unwrap_or_default();
                }
                self.draw(page);
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
            Input::Left | Input::Esc | Input::Interrupt | Input::Char('q') => Flow::Leave,
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
}

struct Menu {
    heading: Vec<&'static str>,
    items: Vec<Choice>,
    at: usize,
    hint: &'static str,
    top: bool,
}

impl Page for Menu {
    fn render(&self, width: usize, height: usize) -> Vec<String> {
        let tw = self.items.iter().map(|c| c.title.chars().count()).max().unwrap_or(0);
        let mut out = brand();
        for h in &self.heading {
            out.push(format!("  {}", paint(h, Color::Bold)));
            out.push(String::new());
        }
        for (i, c) in self.items.iter().enumerate() {
            let title = format!("{:<tw$}", c.title);
            out.push(if i == self.at {
                format!("{} {}{GAP}{}", sgr(&hl(), "\u{27a4}"), sgr(&hl(), &title), c.detail)
            } else {
                format!("  {title}{GAP}{}", gray(c.detail))
            });
        }
        let hint = if self.top { format!("{}{GAP}{}", gray(self.hint), themes_line()) } else { gray(self.hint) };
        screen(out, width, height, hint)
    }

    fn key(&mut self, input: Input) -> Flow {
        let n = self.items.len();
        match input {
            Input::Up | Input::Char('k') => self.at = (self.at + n - 1) % n,
            Input::Down | Input::Char('j') => self.at = (self.at + 1) % n,
            Input::Char(c @ '1'..='3') if self.top => crate::theme::set(crate::theme::ALL[c as usize - '1' as usize]),
            Input::Enter => return Flow::Pick(self.at),
            Input::Char('q') | Input::Interrupt => return Flow::Leave,
            Input::Esc | Input::Left if !self.top => return Flow::Leave,
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
            Choice { title: "Quick setup", detail: "install a model, serve it and open the API" },
            Choice { title: "Install models", detail: "pick models that fit your GPUs" },
            Choice { title: "API keys and gateway", detail: "keys, who can reach the API, a public address" },
            Choice { title: "Status", detail: "Goethe, the API and the models it serves" },
        ],
        at: 0,
        hint: "\u{2191}\u{2193} move   Enter open   q quit",
        top: true,
    }
}

fn access_menu() -> Menu {
    Menu {
        heading: vec!["API keys and gateway"],
        items: vec![
            Choice { title: "API keys", detail: "create, scope, show or delete keys" },
            Choice { title: "Network exposure", detail: "local, vpn/private, public" },
            Choice { title: "Public address", detail: "temporarily share models through a hardened gateway" },
        ],
        at: 0,
        hint: crate::term::MENU_KEYS,
        top: false,
    }
}

fn wait_back() {
    // After a page that printed: the left arrow, Esc, Enter, Backspace or q goes back to the menu.
    let mut buf = [0u8; 16];
    loop {
        let Ok(n) = crate::term::read_keys(&mut buf) else { return };
        let back = |i: &Input| matches!(i, Input::Left | Input::Esc | Input::Enter | Input::Backspace | Input::Interrupt | Input::Char('q'));
        if n == 0 || parse_input(&buf[..n]).iter().any(back) {
            return;
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
        match tty.run(&mut menu)? {
            None => return Ok(()),
            Some(0) => tty.inside(crate::onboard::run)?,
            Some(1) => install_page(&mut tty)?,
            Some(2) => access_page(&mut tty)?,
            Some(_) => {
                tty.draw(&Notice("Reading the GPUs...".to_string()));
                let mut page = StatusPage(crate::status_view::StatusView::new());
                tty.run(&mut page)?;
            }
        }
    }
}

fn access_page(tty: &mut Tty) -> Result<(), String> {
    let mut menu = access_menu();
    loop {
        match tty.run(&mut menu)? {
            None => return Ok(()),
            Some(0) => {
                report(crate::keys_menu::session().map(drop));
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
            Input::Left | Input::Esc | Input::Interrupt => return Flow::Leave,
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
                None if c == 'q' => return Flow::Leave,
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
        let label = |s: &str| paint(&format!("{s:<9}"), Color::Bold);
        let gpus = d.allowed_gpus.iter().map(|g| format!("gpu{g}")).collect::<Vec<_>>().join(" ");
        let mut out = vec![format!(
            "  {}{} total{GAP}{} allowed ({})",
            label("VRAM"),
            gb(d.total_vram_gb),
            gb(d.allowed_vram_gb),
            if gpus.is_empty() { "no GPU allowed".to_string() } else { gpus }
        )];
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
        let fixed = 4 + 2 + (NUM + GAP.len()) * 2 + 4 + GAP.len();
        let mw = longest.min(width.saturating_sub(fixed + GAP.len())).max(16);
        let nw = width.saturating_sub(fixed + mw + GAP.len());
        let head = format!("    {:<mw$}  {GAP}{:>NUM$}{GAP}{:>NUM$}{GAP}{:>4}{GAP}{}", "MODEL", "SIZE", "VRAM", "GPUS", if nw >= 8 { "NOTE" } else { "" });
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
            let rest = format!("{GAP}{:>NUM$}{GAP}{:>NUM$}{GAP}{:>4}{GAP}", gb(r.size_gb), gb(r.vram_gb), r.gpus);
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
            Input::Esc | Input::Interrupt | Input::Left => return Flow::Leave,
            Input::Char('q') if !typing => return Flow::Leave,
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
        for (i, title) in ["Quick setup", "Install models", "API keys and gateway", "Status"].iter().enumerate() {
            assert!(t[first + i].contains(title), "no gap between choices: {t:?}");
        }
        assert!(t[first].starts_with('\u{27a4}'));
        assert!(t[first + 4].is_empty() && t[first + 5].contains("Enter open"), "the keys line follows the menu: {t:?}");
        assert!(t[first + 5].contains("theme  ") && t[first + 5].contains("2 sea") && t[first + 5].contains("3 earth"), "{}", t[first + 5]);
        let active = crate::theme::current().name();
        assert!(t[first + 5].contains(&format!("{active}|")), "a bar marks the active theme: {}", t[first + 5]);
        assert!(!text(access_menu().render(100, 40)).iter().any(|l| l.contains("theme  ")), "themes only on the main menu");
        assert_eq!(t.len(), 40);
    }

    #[test]
    fn digits_and_enter_pick_and_q_leaves() {
        let mut m = main_menu();
        m.key(Input::Up);
        assert!(matches!(m.key(Input::Enter), Flow::Pick(3)), "up from the first wraps to the last");
        assert!(matches!(m.key(Input::Right), Flow::Stay), "only Enter opens");
        assert!(matches!(m.key(Input::Left), Flow::Stay), "the left arrow on the top menu does not quit goethe");
        assert!(matches!(m.key(Input::Char('q')), Flow::Leave));
        assert!(matches!(access_menu().key(Input::Left), Flow::Leave), "a submenu goes back with the left arrow");
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
