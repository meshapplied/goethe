// goethe with no arguments on a terminal: a home screen in two columns; sections on the left, their state and actions on the right.

use crate::keys_menu::{fit, frame_update, line, paint, parse_input, screen_size, window, Input, Line, Screen, Style};
use crate::paths::goethe_home;
use serde_json::Value;
use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

const GAP: usize = 4;
const LEFT_WIDTH: usize = 20;
const MIN_WIDTH: usize = 64;
const MIN_HEIGHT: usize = 12;
const REFRESH: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Section {
    Models,
    Access,
    Keys,
    Gpus,
    Features,
    Platform,
    Done,
}

const SECTIONS: [Section; 7] = [Section::Models, Section::Access, Section::Keys, Section::Gpus, Section::Features, Section::Platform, Section::Done];

fn title(s: Section) -> &'static str {
    match s {
        Section::Models => "Models",
        Section::Access => "API access",
        Section::Keys => "API keys",
        Section::Gpus => "GPUs",
        Section::Features => "Learning & traces",
        Section::Platform => "Platform",
        Section::Done => "Done",
    }
}

#[derive(Clone, Debug, Default)]
struct ModelRow {
    id: String,
    state: String,
    loaded: bool,
}

#[derive(Clone, Debug, Default)]
struct GpuRow {
    index: u32,
    name: String,
    used_gb: f64,
    total_gb: f64,
    denied: bool,
    status: String,
}

#[derive(Clone, Debug, Default)]
struct Snapshot {
    running: bool,
    api: String,
    listens: String,
    allowlist: Vec<String>,
    keys: Vec<(String, String)>,
    models: Vec<ModelRow>,
    gpus: Vec<GpuRow>,
    learn_on: bool,
    learn: String,
    trace_on: bool,
    trace: String,
    platform: Vec<String>,
    linked: bool,
}

fn platform_info(link: crate::platform_cli::Link) -> (Vec<String>, bool) {
    // Plain lines for the Platform section and whether a link exists to log out of.
    use crate::platform_cli::Link;
    match link {
        Link::NotLinked => (vec!["Not linked. Linking is optional: it shows this node on the Goethe platform.".into()], false),
        Link::Unreadable(e) => (vec![format!("Credentials not usable: {e}")], true),
        Link::Linked { revoked: true, workspace, .. } => (vec![format!("Revoked (was workspace {workspace}); log in again to relink.")], true),
        Link::Linked { workspace, heartbeat, .. } => (vec![format!("Linked    workspace {workspace}"), format!("Heartbeat {heartbeat}")], true),
    }
}

fn snapshot() -> Snapshot {
    // Everything the screen shows, read once; a part that cannot be read shows as empty.
    let home = goethe_home();
    let running = crate::service::goethed_running();
    let rows: Vec<Value> = if running {
        crate::rpc::call("ps", serde_json::json!(null))
            .ok()
            .filter(|r| r.code == 0)
            .and_then(|r| serde_json::from_str(r.stdout.trim()).ok())
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    let mut ids = crate::models::installed_ids(&home);
    ids.sort();
    let models = ids
        .into_iter()
        .map(|id| {
            let live: Vec<&Value> = rows.iter().filter(|r| r.get("model_id").and_then(Value::as_str) == Some(id.as_str())).collect();
            let state = match live.first() {
                Some(r) => {
                    let gpus: Vec<String> = r
                        .get("gpu_indices")
                        .and_then(Value::as_array)
                        .map(|a| a.iter().filter_map(Value::as_u64).map(|g| format!("gpu{g}")).collect())
                        .unwrap_or_default();
                    let status = r.get("status").and_then(Value::as_str).unwrap_or("").to_lowercase();
                    let port = r.get("port").and_then(Value::as_u64).map(|p| format!(", port {p}")).unwrap_or_default();
                    format!("{status} on {}{port}", gpus.join("+"))
                }
                None => "not loaded".to_string(),
            };
            ModelRow { id, loaded: !live.is_empty(), state }
        })
        .collect();
    let gpus = crate::db::open(&home)
        .ok()
        .and_then(|con| crate::gpu_policy::list_rows(&con, &home).ok())
        .unwrap_or_default()
        .into_iter()
        .map(|r| GpuRow {
            index: r.index,
            name: r.name.replace("NVIDIA ", ""),
            used_gb: (r.managed_mb + r.foreign_mb) as f64 / 1024.0,
            total_gb: r.vram_total_mb as f64 / 1024.0,
            denied: r.status == "DENIED",
            status: r.status,
        })
        .collect();
    let follows = crate::nginx::bind_interface(&home);
    let listens = match (&follows, crate::nginx::bind_waiting(&home)) {
        (Some(n), Some(_)) => format!("127.0.0.1 only, waiting for {n}"),
        (Some(n), None) => format!("{} (follows {n})", crate::nginx::public_bind(&home)),
        _ => crate::nginx::public_bind(&home),
    };
    let keys = crate::security::key_list_detailed()
        .unwrap_or_default()
        .iter()
        .map(|k| {
            let field = |f: &str| k.get(f).and_then(Value::as_str).unwrap_or("").to_string();
            let models = field("models");
            (field("name"), if models.is_empty() { "every model".to_string() } else { models.replace(',', ", ") })
        })
        .collect();
    let learn = crate::learn::status(&home).unwrap_or_default();
    let learn_on = learn["on"].as_bool().unwrap_or(false);
    let trace = crate::trace::status(&home).unwrap_or_default();
    let trace_on = trace["recording"].as_bool().unwrap_or(false);
    let (platform, linked) = platform_info(crate::platform_cli::link_state(&home, running));
    Snapshot {
        platform,
        linked,
        running,
        api: format!("https://{}/v1", crate::nginx::gateway_connect_addr(&home)),
        listens,
        allowlist: crate::security::allowlist_list().unwrap_or_default(),
        keys,
        models,
        gpus,
        learn_on,
        learn: crate::learn::status_rows(&learn)[0].1.clone(),
        trace_on,
        trace: if trace_on { format!("recording into {}", trace["dir"].as_str().unwrap_or("")) } else { "off".to_string() },
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Action {
    Nothing,
    Quit,
    Run(Vec<String>),
    ToggleGpu(u32, bool),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Row {
    text: String,
    action: Action,
}

struct App {
    snap: Snapshot,
    left: usize,
    right: usize,
    focus_right: bool,
}

fn args(a: &[&str]) -> Vec<String> {
    a.iter().map(|s| s.to_string()).collect()
}

impl App {
    fn section(&self) -> Section {
        SECTIONS[self.left]
    }

    fn info(&self) -> Vec<Line> {
        // The fixed lines above a section's choices.
        let s = &self.snap;
        let mut out = Vec::new();
        match self.section() {
            Section::Models if !s.running => out.push(line("Goethe is not running, so nothing is served.", Style::Yellow)),
            Section::Models => out.push(line(format!("{} installed, {} serving", s.models.len(), s.models.iter().filter(|m| m.loaded).count()), Style::Dim)),
            Section::Access => {
                out.push(line(format!("API       {}", s.api), Style::Plain));
                out.push(line(format!("Listens   {}", s.listens), Style::Plain));
                out.push(line(format!("Allowed   {}", if s.allowlist.is_empty() { "nobody yet".to_string() } else { s.allowlist.join(", ") }), Style::Plain));
            }
            Section::Keys => {
                if s.keys.is_empty() {
                    out.push(line("No keys yet: clients need one.", Style::Yellow));
                }
                let width = s.keys.iter().map(|(n, _)| n.chars().count()).max().unwrap_or(0);
                out.extend(s.keys.iter().map(|(n, m)| line(format!("{n:<width$}  {m}"), Style::Plain)));
            }
            Section::Gpus => out.push(line("Enter allows or denies the hovered GPU for new models.", Style::Dim)),
            Section::Features => {
                out.push(line(format!("Learning  {}", s.learn), Style::Plain));
                out.push(line(format!("Traces    {}", s.trace), Style::Plain));
            }
            Section::Platform => out.extend(s.platform.iter().map(|l| line(l.clone(), Style::Plain))),
            Section::Done => out.push(line("Leave the home screen.", Style::Dim)),
        }
        out
    }

    fn rows(&self) -> Vec<Row> {
        // The choices in the right column for the section on the left.
        let s = &self.snap;
        let row = |text: String, action: Action| Row { text, action };
        match self.section() {
            Section::Models => {
                let mut out = Vec::new();
                if !s.running {
                    out.push(row("Start Goethe".into(), Action::Run(args(&["up"]))));
                }
                let width = s.models.iter().map(|m| m.id.chars().count()).max().unwrap_or(0);
                out.extend(s.models.iter().map(|m| {
                    let verb = if m.loaded { "unload" } else { "load" };
                    row(format!("{:<width$}  {}", m.id, m.state), Action::Run(args(&[verb, &m.id])))
                }));
                out.push(row("Install a model...".into(), Action::Run(args(&["start"]))));
                out
            }
            Section::Access => vec![row("Change who can reach the API".into(), Action::Run(args(&["api", "configure"])))],
            Section::Keys => vec![row("Open the key manager".into(), Action::Run(args(&["api", "manage"])))],
            Section::Gpus => s
                .gpus
                .iter()
                .map(|g| row(format!("gpu{}  {}  {:.1} of {:.1} GB  {}", g.index, g.name, g.used_gb, g.total_gb, g.status), Action::ToggleGpu(g.index, !g.denied)))
                .collect(),
            Section::Features => vec![
                row(format!("Learning: turn {}", if s.learn_on { "off" } else { "on" }), Action::Run(args(&["api", "learn", if s.learn_on { "off" } else { "on" }]))),
                row(format!("Traces: turn {}", if s.trace_on { "off" } else { "on" }), Action::Run(args(&["api", "trace", if s.trace_on { "off" } else { "on" }]))),
            ],
            Section::Platform if s.linked => vec![row("Log out (unlink this node)".into(), Action::Run(args(&["logout"]))), row("Log in again (replace the link)".into(), Action::Run(args(&["login"])))],
            Section::Platform => vec![row("Log in (link this node)".into(), Action::Run(args(&["login"])))],
            Section::Done => Vec::new(),
        }
    }

    fn handle(&mut self, input: Input) -> Action {
        // One key press; anything that changes Goethe comes back as an Action.
        let rows = self.rows();
        match (self.focus_right, input) {
            (_, Input::Interrupt | Input::Char('q')) => Action::Quit,
            (false, Input::Esc) => Action::Quit,
            (false, Input::Up | Input::Char('k')) => {
                self.left = self.left.saturating_sub(1);
                Action::Nothing
            }
            (false, Input::Down | Input::Char('j')) => {
                self.left = (self.left + 1).min(SECTIONS.len() - 1);
                Action::Nothing
            }
            (false, Input::Right | Input::Enter | Input::Char('l')) if self.section() == Section::Done => Action::Quit,
            (false, Input::Right | Input::Enter | Input::Char('l')) => {
                if !rows.is_empty() {
                    self.focus_right = true;
                    self.right = 0;
                }
                Action::Nothing
            }
            (true, Input::Left | Input::Esc | Input::Char('h')) => {
                self.focus_right = false;
                Action::Nothing
            }
            (true, Input::Up | Input::Char('k')) => {
                self.right = self.right.saturating_sub(1);
                Action::Nothing
            }
            (true, Input::Down | Input::Char('j')) => {
                self.right = (self.right + 1).min(rows.len().saturating_sub(1));
                Action::Nothing
            }
            (true, Input::Enter | Input::Space) => rows.get(self.right).map(|r| r.action.clone()).unwrap_or(Action::Nothing),
            _ => Action::Nothing,
        }
    }

    fn help(&self) -> &'static str {
        match (self.focus_right, self.section()) {
            (false, _) => "\u{2191}\u{2193} move   \u{2192} open   q quit",
            (true, Section::Models) => "\u{2191}\u{2193} move   Enter load or unload   \u{2190} back   q quit",
            (true, Section::Gpus) => "\u{2191}\u{2193} move   Enter allow or deny   \u{2190} back   q quit",
            (true, _) => "\u{2191}\u{2193} move   Enter choose   \u{2190} back   q quit",
        }
    }

    fn render(&self, width: usize, height: usize) -> Vec<String> {
        // Title with the state, a gap, the two columns, a gap, the keys to press.
        if width < MIN_WIDTH || height < MIN_HEIGHT {
            let mut out = vec![format!("Make the terminal at least {MIN_WIDTH}x{MIN_HEIGHT} (q quits).")];
            out.resize(height.max(1), String::new());
            return out;
        }
        let state = if self.snap.running { format!("running \u{00B7} {}", self.snap.api) } else { "not running".to_string() };
        let mut out = vec![format!("{}   {}", paint("Goethe", Style::Bold), paint(&state, if self.snap.running { Style::Dim } else { Style::Yellow })), String::new()];
        let left: Vec<Line> = SECTIONS
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let gap = if *s == Section::Done { "\n" } else { "" };
                match (i == self.left, self.focus_right) {
                    (true, false) => line(format!("{gap}\u{25CF} {}", title(*s)), Style::Selected),
                    (true, true) => line(format!("{gap}\u{203A} {}", title(*s)), Style::Bold),
                    _ => line(format!("{gap}  {}", title(*s)), Style::Plain),
                }
            })
            .flat_map(|l| {
                let mut parts: Vec<Line> = l.text.split('\n').map(|t| line(t, l.style)).collect();
                if parts.len() > 1 {
                    parts[0].style = Style::Plain;
                }
                parts
            })
            .collect();
        let mut right = self.info();
        let rows = self.rows();
        if !rows.is_empty() {
            right.push(line("", Style::Plain));
        }
        let first = right.len();
        for (i, r) in rows.iter().enumerate() {
            if self.focus_right && i == self.right {
                right.push(line(format!("\u{25CF} {}", r.text), Style::Selected));
            } else {
                right.push(line(format!("  {}", r.text), Style::Plain));
            }
        }
        let lw = LEFT_WIDTH;
        let rw = width - lw - GAP;
        let rows_n = height - 4;
        let r0 = window(right.len(), first + self.right, rows_n);
        for r in 0..rows_n {
            let l = left.get(r).map(|l| paint(&fit(&l.text, lw), l.style)).unwrap_or_else(|| " ".repeat(lw));
            let rt = right.get(r0 + r).map(|l| paint(&fit(&l.text, rw), l.style)).unwrap_or_default();
            out.push(format!("{l}{}{rt}", " ".repeat(GAP)).trim_end().to_string());
        }
        out.push(String::new());
        out.push(paint(&fit(self.help(), width), Style::Dim));
        out
    }
}

fn run_child(command: &[String]) {
    // A goethe command in the normal terminal, as if typed, then back on Enter.
    let exe = std::env::current_exe().unwrap_or_else(|_| "goethe".into());
    println!("\n$ goethe {}\n", command.join(" "));
    let status = std::process::Command::new(exe).args(command).status();
    if let Err(e) = status {
        println!("could not run it: {e}");
    }
    print!("\nPress Enter to go back to the home screen.");
    let _ = std::io::stdout().flush();
    let mut buf = String::new();
    let _ = std::io::stdin().read_line(&mut buf);
}

fn draw(app: &App, last: &mut Vec<String>) {
    let (w, h) = screen_size();
    let next = app.render(w, h);
    let update = frame_update(last, &next);
    if !update.is_empty() {
        print!("{update}");
        let _ = std::io::stdout().flush();
    }
    last.clear();
    last.extend(next);
}

pub fn run() -> Result<(), String> {
    // The home screen until q or Done; commands chosen here run in the normal terminal and come back.
    if !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        return Err("the home screen needs a terminal; goethe --help lists the commands".to_string());
    }
    let mut app = App { snap: snapshot(), left: 0, right: 0, focus_right: false };
    let mut raw = crate::term::RawMode::enter().ok_or("could not switch the terminal to raw mode")?;
    let mut screen = Screen::open();
    let mut last: Vec<String> = Vec::new();
    let mut size = screen_size();
    let mut loaded = Instant::now();
    draw(&app, &mut last);
    let mut buf = [0u8; 16];
    loop {
        let mut fds = libc::pollfd { fd: 0, events: libc::POLLIN, revents: 0 };
        if unsafe { libc::poll(&mut fds, 1, 250) } <= 0 {
            if screen_size() != size {
                size = screen_size();
                last.clear();
            }
            if loaded.elapsed() >= REFRESH {
                app.snap = snapshot();
                loaded = Instant::now();
            }
            draw(&app, &mut last);
            continue;
        }
        let n = match crate::term::read_keys(&mut buf) {
            Ok(0) | Err(_) => return Ok(()),
            Ok(n) => n,
        };
        for input in parse_input(&buf[..n]) {
            match app.handle(input) {
                Action::Quit => return Ok(()),
                Action::Nothing => {}
                Action::ToggleGpu(index, deny) => {
                    let home = goethe_home();
                    if let Ok(con) = crate::db::open(&home) {
                        let _ = crate::gpu_policy::set_denied(&con, index, deny);
                    }
                    app.snap = snapshot();
                }
                Action::Run(command) => {
                    drop(screen);
                    drop(raw);
                    run_child(&command);
                    raw = crate::term::RawMode::enter().ok_or("could not switch the terminal to raw mode")?;
                    screen = Screen::open();
                    last.clear();
                    app.snap = snapshot();
                    loaded = Instant::now();
                }
            }
        }
        draw(&app, &mut last);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(running: bool) -> Snapshot {
        Snapshot {
            running,
            api: "https://10.0.0.2:8080/v1".into(),
            listens: "10.0.0.2 (follows wg0)".into(),
            allowlist: vec!["10.0.0.0/24".into(), "127.0.0.1/32".into()],
            keys: vec![("dev".into(), "every model".into())],
            models: vec![
                ModelRow { id: "org/a".into(), state: "running on gpu3, port 30001".into(), loaded: true },
                ModelRow { id: "org/b".into(), state: "not loaded".into(), loaded: false },
            ],
            gpus: vec![GpuRow { index: 3, name: "RTX".into(), used_gb: 9.0, total_gb: 95.6, denied: false, status: "ALLOWED".into() }],
            learn_on: false,
            learn: "off".into(),
            trace_on: true,
            trace: "recording into /t".into(),
            platform: vec!["Not linked.".into()],
            linked: false,
        }
    }

    fn app(running: bool) -> App {
        App { snap: snap(running), left: 0, right: 0, focus_right: false }
    }

    fn plain(s: &str) -> String {
        let mut out = String::new();
        let mut chars = s.chars();
        while let Some(c) = chars.next() {
            if c == '\x1b' {
                chars.by_ref().find(|e| e.is_ascii_alphabetic());
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn models_load_or_unload_by_their_state() {
        let mut a = app(true);
        assert_eq!(a.handle(Input::Right), Action::Nothing);
        assert_eq!(a.handle(Input::Enter), Action::Run(args(&["unload", "org/a"])), "a serving model is unloaded");
        a.handle(Input::Down);
        assert_eq!(a.handle(Input::Enter), Action::Run(args(&["load", "org/b"])));
        a.handle(Input::Down);
        assert_eq!(a.handle(Input::Enter), Action::Run(args(&["start"])), "the last row installs");
        a.handle(Input::Down);
        assert_eq!(a.right, 2, "the cursor stops at the last row");
    }

    #[test]
    fn a_stopped_goethe_offers_to_start_first() {
        let mut a = app(false);
        a.handle(Input::Right);
        assert_eq!(a.handle(Input::Enter), Action::Run(args(&["up"])));
        assert!(a.info()[0].text.contains("not running"));
    }

    #[test]
    fn each_section_runs_the_command_for_it() {
        let mut a = app(true);
        let mut pick = |left: usize, down: usize| {
            a.focus_right = false;
            a.left = left;
            a.handle(Input::Right);
            for _ in 0..down {
                a.handle(Input::Down);
            }
            a.handle(Input::Enter)
        };
        assert_eq!(pick(1, 0), Action::Run(args(&["api", "configure"])));
        assert_eq!(pick(2, 0), Action::Run(args(&["api", "manage"])));
        assert_eq!(pick(3, 0), Action::ToggleGpu(3, true), "an allowed GPU is denied");
        assert_eq!(pick(4, 0), Action::Run(args(&["api", "learn", "on"])));
        assert_eq!(pick(4, 1), Action::Run(args(&["api", "trace", "off"])), "recording traces are turned off");
        assert_eq!(pick(5, 0), Action::Run(args(&["login"])));
        let mut b = app(true);
        b.snap.linked = true;
        b.left = 5;
        b.handle(Input::Right);
        assert_eq!(b.handle(Input::Enter), Action::Run(args(&["logout"])));
    }

    #[test]
    fn platform_lines_follow_the_link() {
        use crate::platform_cli::Link;
        assert!(!platform_info(Link::NotLinked).1);
        let (lines, linked) = platform_info(Link::Linked { workspace: "Lab".into(), revoked: false, heartbeat: "OK (2 s ago)".into(), ok: true });
        assert!(linked && lines[0].contains("Lab") && lines[1].contains("OK (2 s ago)"));
        assert!(platform_info(Link::Linked { workspace: "Lab".into(), revoked: true, heartbeat: String::new(), ok: false }).0[0].starts_with("Revoked"));
    }

    #[test]
    fn done_and_q_leave_and_the_left_column_stops_at_its_ends() {
        let mut a = app(true);
        a.handle(Input::Up);
        assert_eq!(a.left, 0);
        for _ in 0..9 {
            a.handle(Input::Down);
        }
        assert_eq!(a.section(), Section::Done);
        assert_eq!(a.handle(Input::Enter), Action::Quit);
        assert_eq!(app(true).handle(Input::Char('q')), Action::Quit);
        let mut b = app(true);
        b.handle(Input::Right);
        assert_eq!(b.handle(Input::Left), Action::Nothing);
        assert!(!b.focus_right, "left goes back to the sections");
    }

    #[test]
    fn the_screen_fits_and_shows_the_state() {
        let mut a = app(true);
        a.left = 1;
        let screen: Vec<String> = a.render(90, 20).iter().map(|l| plain(l)).collect();
        assert_eq!(screen.len(), 20);
        assert!(screen.iter().all(|l| l.chars().count() <= 90));
        assert!(screen[0].contains("running \u{00B7} https://10.0.0.2:8080/v1"), "{}", screen[0]);
        assert!(screen[3].starts_with("\u{25CF} API access") && screen[3].contains("Listens   10.0.0.2 (follows wg0)"), "{}", screen[3]);
        assert!(screen.iter().any(|l| l.trim() == "Done"), "Done sits apart from the sections");
        assert!(screen.iter().all(|l| !l.contains('\u{2502}') && !l.contains('\u{2500}')));
        assert!(app(false).render(90, 20)[0].contains("not running"));
        assert!(app(true).render(40, 10)[0].contains("at least 64x12"));
    }
}
