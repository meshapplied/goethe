// goethe run with no input on a terminal: an ephemeral chat screen. The conversation lives in memory; only e (export) writes files, in the trace format.

use crate::keys_menu::{fit, frame_update, paint, parse_input, screen_size, Input, Screen, Style};
use crate::term::visible_width;
use crate::run::{embed_line, Loaded, Target};
use crate::trace::Capture;
use serde_json::{json, Value};
use std::time::{Duration, Instant, SystemTime};

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";
const TYPING: &str = "[Esc] commands  [Enter] send";
const COMMANDS: &str = "[m] model  [e] export  [r] record  [q] quit  [i] type";
const SURE: &str = "exit? Enter = exit, q = back";
const PICKING: &str = "[Up] [Down] move  [Enter] choose  [Esc] back";

pub fn split_think(raw: &str, last: bool) -> (String, String) {
    // The model's reasoning and its answer: complete think blocks, an unclosed one, and everything before a closing tag that has no opening one are reasoning; while streaming, a tag cut in half stays out of the answer.
    let mut thinking: Vec<String> = Vec::new();
    let mut answer = raw.to_string();
    while let Some(i) = answer.find(OPEN) {
        let rest = &answer[i + OPEN.len()..];
        let (inside, after) = match rest.find(CLOSE) {
            Some(j) => (&rest[..j], &rest[j + CLOSE.len()..]),
            None => (rest, ""),
        };
        thinking.push(inside.trim().to_string());
        answer = format!("{}{after}", &answer[..i]);
    }
    if let Some(i) = answer.find(CLOSE) {
        thinking.insert(0, answer[..i].trim().to_string());
        answer = answer[i + CLOSE.len()..].to_string();
    }
    if !last {
        let held = (1..CLOSE.len()).rev().find(|n| answer.ends_with(&CLOSE[..*n]) || (*n < OPEN.len() && answer.ends_with(&OPEN[..*n])));
        if let Some(n) = held {
            answer.truncate(answer.len() - n);
        }
    }
    thinking.retain(|t| !t.is_empty());
    (thinking.join("\n"), answer.trim_start().to_string())
}

pub fn plain_text(raw: &str, thinking: bool, color: bool) -> String {
    // A finished reply for a pipe or a plain terminal: the answer, with the reasoning dimmed under a thinking marker when asked for; on a terminal without the model's control characters.
    let raw = if color { clean(raw) } else { raw.to_string() };
    let (think, answer) = split_think(&raw, true);
    match (thinking && !think.is_empty(), color) {
        (false, _) => answer,
        (true, true) => format!("\x1b[2mthinking\n{think}\x1b[0m\n\n{answer}"),
        (true, false) => format!("thinking\n{think}\n\n{answer}"),
    }
}

pub(crate) fn wrap(text: &str, width: usize) -> Vec<String> {
    // The lines of text cut to the display width, a line break kept; a character wider than the width gets its own line.
    let mut out = Vec::new();
    for l in text.split('\n') {
        let mut line = String::new();
        for c in l.chars() {
            let w = visible_width(&c.to_string());
            if !line.is_empty() && visible_width(&line) + w > width.max(1) {
                out.push(std::mem::take(&mut line));
            }
            line.push(c);
        }
        out.push(line);
    }
    out
}

pub(crate) fn clean(text: &str) -> String {
    // What a model sent, safe to draw: control characters (escape sequences included) are dropped, a tab is a space.
    text.chars().filter(|c| !c.is_control() || matches!(c, '\n' | '\t')).map(|c| if c == '\t' { ' ' } else { c }).collect()
}

pub(crate) fn top(total: usize, rows: usize, scroll: usize, follow: bool) -> usize {
    // The first line shown: text starts at the top and, once longer than the window, the window follows the end unless it was scrolled up.
    let last = total.saturating_sub(rows);
    if follow { last } else { scroll.min(last) }
}

pub(crate) fn moved(from: usize, total: usize, rows: usize, delta: isize) -> usize {
    // The first line after scrolling by delta lines, kept inside the text.
    from.saturating_add_signed(delta).min(total.saturating_sub(rows))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Typing,
    Command,
    Confirm,
    Pick,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Act {
    None,
    Send(String),
    Audio(String),
    Picker,
    Chose(usize),
    Export,
    Record,
    Quit,
}

#[derive(Clone, Debug, PartialEq)]
enum Entry {
    User(String),
    Reply(String),
    Vector(String),
    Note(String),
    Divider(String),
}

pub(crate) struct Ui {
    mode: Mode,
    back: Mode,
    input: Vec<char>,
    at: usize,
    entries: Vec<Entry>,
    header: String,
    picks: Vec<(String, bool)>,
    pick: usize,
    streaming: bool,
    thinking: bool,
    waiting: Option<Instant>,
    scroll: usize,
    follow: bool,
    size: (usize, usize),
    session: (u64, f64),
    last_reply: Option<(u64, f64)>,
}

impl Ui {
    pub(crate) fn new(header: String, thinking: bool) -> Ui {
        Ui { mode: Mode::Typing, back: Mode::Typing, input: Vec::new(), at: 0, entries: Vec::new(), header, picks: Vec::new(), pick: 0, streaming: false, thinking, waiting: None, scroll: 0, follow: true, size: (80, 24), session: (0, 0.0), last_reply: None }
    }

    pub(crate) fn key(&mut self, key: Input) -> Act {
        // One key: typing edits the line, Esc leaves typing for the command keys (a bare q never quits while typing), Ctrl-C asks before quitting.
        match (self.mode, key) {
            (Mode::Confirm, Input::Enter) => Act::Quit,
            (Mode::Confirm, _) => {
                self.mode = self.back;
                Act::None
            }
            (_, Input::Interrupt) => {
                self.back = self.mode;
                self.mode = Mode::Confirm;
                Act::None
            }
            (Mode::Typing | Mode::Command, Input::PageUp) => self.scroll_by(-self.page()),
            (Mode::Typing | Mode::Command, Input::PageDown) => self.scroll_by(self.page()),
            (Mode::Command, Input::Up | Input::Char('k')) => self.scroll_by(-1),
            (Mode::Command, Input::Down | Input::Char('j')) => self.scroll_by(1),
            (Mode::Command, Input::Char('g')) => self.scroll_by(isize::MIN / 2),
            (Mode::Command, Input::Char('G')) => self.scroll_by(isize::MAX / 2),
            (Mode::Typing, Input::Esc) => {
                self.mode = Mode::Command;
                Act::None
            }
            (Mode::Typing, Input::Enter) => self.line(),
            (Mode::Typing, Input::Backspace) => {
                if self.at > 0 {
                    self.at -= 1;
                    self.input.remove(self.at);
                }
                Act::None
            }
            (Mode::Typing, Input::Left) => {
                self.at = self.at.saturating_sub(1);
                Act::None
            }
            (Mode::Typing, Input::Right) => {
                self.at = (self.at + 1).min(self.input.len());
                Act::None
            }
            (Mode::Typing, Input::Char(c)) => self.insert(c),
            (Mode::Typing, Input::Space) => self.insert(' '),
            (Mode::Command, Input::Char('q')) => {
                self.back = Mode::Command;
                self.mode = Mode::Confirm;
                Act::None
            }
            (Mode::Command, Input::Char('i') | Input::Enter | Input::Esc) => {
                self.mode = Mode::Typing;
                Act::None
            }
            (Mode::Command, Input::Char('m')) => Act::Picker,
            (Mode::Command, Input::Char('e')) => Act::Export,
            (Mode::Command, Input::Char('r')) => Act::Record,
            (Mode::Pick, Input::Up) => {
                self.pick = self.pick.saturating_sub(1);
                Act::None
            }
            (Mode::Pick, Input::Down) => {
                self.pick = (self.pick + 1).min(self.picks.len().saturating_sub(1));
                Act::None
            }
            (Mode::Pick, Input::Enter) => {
                self.mode = Mode::Typing;
                Act::Chose(self.pick)
            }
            (Mode::Pick, Input::Esc | Input::Char('q')) => {
                self.mode = Mode::Command;
                Act::None
            }
            _ => Act::None,
        }
    }

    fn rows(&self) -> usize {
        self.size.1.saturating_sub(3)
    }

    fn page(&self) -> isize {
        self.rows().saturating_sub(1).max(1) as isize
    }

    fn scroll_by(&mut self, delta: isize) -> Act {
        // Scrolling up leaves the stream; reaching the end follows it again.
        let total = self.lines(self.size.0).len();
        let at = top(total, self.rows(), self.scroll, self.follow);
        self.scroll = moved(at, total, self.rows(), delta);
        self.follow = self.scroll >= total.saturating_sub(self.rows());
        Act::None
    }

    pub(crate) fn resize(&mut self, width: usize, height: usize) {
        self.size = (width, height);
    }

    fn insert(&mut self, c: char) -> Act {
        self.input.insert(self.at, c);
        self.at += 1;
        Act::None
    }

    fn line(&mut self) -> Act {
        // The typed line as an action: /audio PATH or text; an empty line does nothing.
        let text: String = self.input.drain(..).collect();
        self.at = 0;
        let text = text.trim().to_string();
        if !text.is_empty() {
            self.follow = true;
        }
        match text.strip_prefix("/audio ") {
            _ if text.is_empty() => Act::None,
            Some(path) => Act::Audio(path.trim().to_string()),
            None => Act::Send(text),
        }
    }

    pub(crate) fn open_picker(&mut self, picks: Vec<(String, bool)>) {
        self.pick = 0;
        self.mode = if picks.is_empty() { Mode::Command } else { Mode::Pick };
        if picks.is_empty() {
            self.note("no installed models");
        }
        self.picks = picks;
    }

    fn note(&mut self, text: &str) {
        self.entries.push(Entry::Note(text.lines().next().unwrap_or_default().to_string()));
    }

    fn lines(&self, width: usize) -> Vec<(String, Style)> {
        // The scrollback wrapped to the width.
        let mut out = Vec::new();
        let wrap = |text: &str, style: Style, out: &mut Vec<(String, Style)>| out.extend(wrap(&clean(text), width).into_iter().map(|l| (l, style)));
        for (i, e) in self.entries.iter().enumerate() {
            match e {
                Entry::User(_) => out.push(("\u{27a4} User".to_string(), Style::Secondary)),
                Entry::Reply(_) | Entry::Vector(_) => out.push(("\u{27a4} Model".to_string(), Style::Third)),
                _ => {}
            }
            match e {
                Entry::User(t) => wrap(t, Style::Bold, &mut out),
                Entry::Vector(t) => wrap(t, Style::Plain, &mut out),
                Entry::Note(t) => wrap(t, Style::Dim, &mut out),
                Entry::Divider(t) => wrap(&format!("-- {t} --"), Style::Dim, &mut out),
                Entry::Reply(raw) => {
                    let (think, answer) = split_think(raw, !self.streaming);
                    match (think.is_empty(), self.thinking) {
                        (true, _) => {}
                        (false, true) => {
                            wrap("thinking", Style::Dim, &mut out);
                            wrap(&think, Style::Dim, &mut out);
                        }
                        (false, false) => wrap(&format!("thinking ({} words)", think.split_whitespace().count()), Style::Dim, &mut out),
                    }
                    let waits = self.waiting.filter(|_| self.streaming && answer.is_empty() && think.is_empty() && i + 1 == self.entries.len());
                    if let Some(since) = waits {
                        wrap(&format!("thinking... {} s", since.elapsed().as_secs()), Style::Dim, &mut out);
                    }
                    wrap(&answer, Style::Plain, &mut out);
                }
            }
        }
        out
    }

    pub(crate) fn count(&mut self, tokens: u64, seconds: f64) {
        // One finished reply (its completion tokens, first to last token in seconds) into the session totals.
        self.session = (self.session.0 + tokens, self.session.1 + seconds);
        self.last_reply = Some((tokens, seconds));
    }

    fn stats(&self, room: usize) -> String {
        // Tokens generated this session and the decode speed, with the last reply's when there is room.
        if self.session.0 == 0 {
            return String::new();
        }
        let speed = |(n, s): (u64, f64)| if s > 0.0 { format!(" \u{b7} {:.1} tok/s", n as f64 / s) } else { String::new() };
        let total = format!("{} tokens{}", self.session.0, speed(self.session));
        let last = self.last_reply.map(|l| format!("{total}  (last {}{})", l.0, speed(l))).unwrap_or_default();
        [last, total].into_iter().find(|t| t.chars().count() <= room).unwrap_or_default()
    }

    pub(crate) fn render(&self, width: usize, height: usize) -> Vec<String> {
        // Header, scrollback (its tail), the input line and the keys of the mode.
        if width < 24 || height < 6 {
            let mut out = vec!["Make the terminal at least 24x6.".to_string()];
            out.resize(height.max(1), String::new());
            return out;
        }
        let body_h = height - 3;
        let mut body: Vec<String> = if self.mode == Mode::Pick {
            self.picks
                .iter()
                .enumerate()
                .map(|(i, (id, loaded))| {
                    let mark = if i == self.pick { "\u{25CF} " } else { "  " };
                    let style = match (i == self.pick, loaded) {
                        (true, _) => Style::Selected,
                        (false, true) => Style::Plain,
                        (false, false) => Style::Dim,
                    };
                    paint(&fit(&format!("{mark}{id}{}", if *loaded { "" } else { "  (not loaded)" }), width), style)
                })
                .collect()
        } else {
            let lines = self.lines(width);
            let first = top(lines.len(), body_h, self.scroll, self.follow);
            lines.iter().skip(first).take(body_h).map(|(t, s)| paint(&fit(t, width), *s)).collect()
        };
        body.truncate(body_h);
        let scrolled = if self.follow { "" } else { "  (scrolled: G follows)" };
        let mut out = vec![paint(&fit(&format!("{}{scrolled}", self.header), width), Style::Bold)];
        let blank = body_h - body.len();
        out.extend(body);
        out.extend(std::iter::repeat_n(String::new(), blank));
        let typed: String = self.input.iter().collect();
        let line = match self.mode {
            Mode::Typing => {
                let (a, b): (String, String) = (self.input[..self.at].iter().collect(), self.input[self.at..].iter().skip(1).collect());
                let cur = self.input.get(self.at).copied().unwrap_or(' ');
                format!("> {a}\x1b[7m{cur}\x1b[0m{b}")
            }
            _ => paint(&format!("> {typed}"), Style::Dim),
        };
        out.push(line);
        let keys = match self.mode {
            Mode::Typing => TYPING,
            Mode::Command => COMMANDS,
            Mode::Confirm => SURE,
            Mode::Pick => PICKING,
        };
        let stats = self.stats(width.saturating_sub(keys.chars().count() + 4));
        let gap = width.saturating_sub(keys.chars().count() + stats.chars().count());
        out.push(paint(&fit(&format!("{keys}{}{stats}", " ".repeat(gap)), width), Style::Dim));
        out
    }
}

struct Turn {
    started: SystemTime,
    seconds: f64,
    ttft: Option<f64>,
    messages: Vec<Value>,
    reply: String,
}

fn captures(model: &str, instance: &str, dir: &std::path::Path, turns: &[Turn]) -> Vec<Capture> {
    // Each turn as the request goethe would have seen and its reply, for the trace writer.
    turns
        .iter()
        .enumerate()
        .map(|(i, t)| Capture {
            dir: dir.to_path_buf(),
            max_gb: 1e9,
            path: "/chat/completions".into(),
            request_id: format!("run-{i}"),
            key_id: "goethe-run".into(),
            instance_id: instance.to_string(),
            model: model.to_string(),
            session_hint: None,
            learn_note: None,
            request: json!({"model": model, "messages": t.messages}).to_string().into_bytes(),
            started_at: t.started,
            duration_s: t.seconds,
            ttft_s: t.ttft,
            status: 200,
            sse: false,
            complete: true,
            response: json!({"object": "chat.completion", "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": t.reply}, "finish_reason": "stop"}]}).to_string().into_bytes(),
            truncated: false,
        })
        .collect()
}

fn export_dir() -> Result<std::path::PathBuf, String> {
    Ok(std::env::current_dir().map_err(|e| e.to_string())?.join(format!("goethe-chat-{}", chrono::Utc::now().format("%Y%m%dT%H%M%SZ"))))
}

fn export(model: &str, instance: &str, turns: &[Turn], vectors: &[(String, Vec<f64>)]) -> Result<std::path::PathBuf, String> {
    // The chat in the trace layout (docs/traces.md) in a new folder here; embeddings as JSON beside it. Nothing is overwritten.
    use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
    if turns.is_empty() && vectors.is_empty() {
        return Err("nothing to export yet".into());
    }
    let dir = export_dir()?;
    if turns.is_empty() {
        std::fs::DirBuilder::new().mode(0o700).create(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    } else {
        crate::trace::export(&crate::paths::goethe_home(), &dir, captures(model, instance, &dir, turns))?;
    }
    if !vectors.is_empty() {
        let rows: Vec<Value> = vectors.iter().map(|(input, e)| json!({"input": input, "embedding": e})).collect();
        let path = dir.join("embeddings.json");
        let mut f = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        std::io::Write::write_all(&mut f, serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?.as_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(dir)
}

fn draw(ui: &mut Ui, last: &mut Vec<String>) {
    let (w, h) = screen_size();
    ui.resize(w, h);
    let next = ui.render(w, h);
    let update = frame_update(last, &next);
    if !update.is_empty() {
        print!("{update}");
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    last.clear();
    last.extend(next);
}

fn header(t: &Target) -> String {
    format!("{}  ({}, {})", t.model, t.runtime, t.kind)
}

fn read_bytes(buf: &mut [u8]) -> usize {
    // What is waiting on the terminal, read from the descriptor itself: std's buffered stdin would hold typed-ahead keys where poll cannot see them.
    usize::try_from(unsafe { libc::read(0, buf.as_mut_ptr().cast(), buf.len()) }).unwrap_or(0)
}

fn stop_asked() -> bool {
    // Ctrl-C or Esc pressed while a reply streams: stop it, keep what arrived.
    let mut fds = libc::pollfd { fd: 0, events: libc::POLLIN, revents: 0 };
    if unsafe { libc::poll(&mut fds, 1, 0) } <= 0 {
        return false;
    }
    let mut buf = [0u8; 256];
    let n = read_bytes(&mut buf);
    parse_input(&buf[..n]).iter().any(|k| matches!(k, Input::Interrupt | Input::Esc))
}

struct State {
    t: Target,
    ui: Ui,
    msgs: Vec<Value>,
    turns: Vec<Turn>,
    vectors: Vec<(String, Vec<f64>)>,
    last: Vec<String>,
}

impl State {
    fn send(&mut self, text: String) {
        match self.t.kind.as_str() {
            "embedding" => {
                self.ui.entries.push(Entry::User(text.clone()));
                match self.t.api.embed(&self.t.model, &text) {
                    Ok(e) => {
                        self.ui.entries.push(Entry::Vector(embed_line(&e)));
                        self.vectors.push((text, e));
                    }
                    Err(e) => self.ui.note(&e),
                }
            }
            "speech" => self.ui.note("this is a speech model: /audio PATH sends a file, r records"),
            _ => self.chat(text),
        }
    }

    fn chat(&mut self, text: String) {
        self.ui.entries.push(Entry::User(text.clone()));
        self.msgs.push(json!({"role": "user", "content": text}));
        self.ui.entries.push(Entry::Reply(String::new()));
        self.ui.streaming = true;
        let (started, clock) = (SystemTime::now(), Instant::now());
        self.ui.waiting = Some(clock);
        draw(&mut self.ui, &mut self.last);
        let (mut ttft, mut first, mut end) = (None, None, clock);
        let State { t, ui, msgs, last, .. } = self;
        let result = t.api.chat(&t.model, msgs, &mut |piece| {
            ttft.get_or_insert_with(|| clock.elapsed().as_secs_f64());
            if !piece.is_empty() {
                end = Instant::now();
                first.get_or_insert(end);
            }
            if let Some(Entry::Reply(raw)) = ui.entries.last_mut() {
                raw.push_str(piece);
            }
            draw(ui, last);
            !stop_asked()
        });
        self.ui.streaming = false;
        self.ui.waiting = None;
        match result {
            Ok((reply, tokens)) => {
                self.ui.count(tokens, first.map_or(0.0, |f| (end - f).as_secs_f64()));
                self.turns.push(Turn { started, seconds: clock.elapsed().as_secs_f64(), ttft, messages: self.msgs.clone(), reply: reply.clone() });
                self.msgs.push(json!({"role": "assistant", "content": reply}));
            }
            Err(e) => {
                self.ui.entries.pop();
                self.msgs.pop();
                self.ui.note(&e);
            }
        }
    }

    fn audio(&mut self, path: &str) {
        if self.t.kind != "speech" {
            return self.ui.note("/audio needs a speech model");
        }
        self.ui.entries.push(Entry::User(format!("[audio] {path}")));
        match self.t.api.transcribe(&self.t.model, std::path::Path::new(path), self.t.llama) {
            Ok(text) => self.ui.entries.push(Entry::Reply(text)),
            Err(e) => self.ui.note(&e),
        }
    }

    fn record(&mut self) {
        if self.t.kind != "speech" {
            return self.ui.note("r records for a speech model; this one takes text");
        }
        match crate::run::record() {
            Ok(wav) => {
                self.audio(&wav.to_string_lossy());
                let _ = std::fs::remove_file(&wav);
                let _ = wav.parent().map(std::fs::remove_dir);
            }
            Err(e) => self.ui.note(&e),
        }
        self.last.clear();
    }

    fn export(&mut self) {
        match export(&self.t.model, &self.t.instance, &self.turns, &self.vectors) {
            Ok(dir) => self.ui.note(&format!("exported to {}", dir.display())),
            Err(e) => self.ui.note(&e),
        }
    }

    fn switch(&mut self, to: Target) {
        self.ui.entries.push(Entry::Divider(format!("new conversation with {}", to.model)));
        self.ui.header = header(&to);
        self.msgs.clear();
        self.t = to;
    }
}

fn run_child(command: &[String]) {
    // A goethe command in the normal terminal, as if typed, then back on Enter.
    let exe = std::env::current_exe().unwrap_or_else(|_| "goethe".into());
    println!("\n$ goethe {}\n", command.join(" "));
    if let Err(e) = std::process::Command::new(exe).args(command).status() {
        println!("could not run it: {e}");
    }
    print!("\nPress Enter to go back.");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let _ = std::io::stdin().read_line(&mut String::new());
}

pub(crate) fn run(t: Target, loaded: &mut Loaded, thinking: bool) -> Result<(), String> {
    // The screen until quit; the terminal is restored on every way out (guards drop, and a panic leaves the alternate screen first).
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        eprint!("\x1b[?25h\x1b[?1049l");
        prev(info);
    }));
    let mut _raw = Some(crate::term::RawMode::enter().ok_or("could not switch the terminal to raw mode")?);
    let mut _screen = Some(Screen::open());
    let mut st = State { ui: Ui::new(header(&t), thinking), t, msgs: Vec::new(), turns: Vec::new(), vectors: Vec::new(), last: Vec::new() };
    let mut size = screen_size();
    let mut buf = [0u8; 256];
    loop {
        draw(&mut st.ui, &mut st.last);
        let mut fds = libc::pollfd { fd: 0, events: libc::POLLIN, revents: 0 };
        if unsafe { libc::poll(&mut fds, 1, Duration::from_millis(250).as_millis() as i32) } <= 0 {
            if screen_size() != size {
                size = screen_size();
                st.last.clear();
            }
            continue;
        }
        let n = read_bytes(&mut buf);
        if n == 0 {
            return Ok(());
        }
        for input in parse_input(&buf[..n]) {
            match st.ui.key(input) {
                Act::None => {}
                Act::Quit => return Ok(()),
                Act::Send(text) => st.send(text),
                Act::Audio(path) => st.audio(&path),
                Act::Export => st.export(),
                Act::Record => st.record(),
                Act::Picker => st.ui.open_picker(crate::menu::installed_and_loaded(&crate::paths::goethe_home())),
                Act::Chose(i) => {
                    let (id, up) = st.ui.picks[i].clone();
                    if id == st.t.model {
                        continue;
                    }
                    if !up {
                        _screen = None;
                        _raw = None;
                        run_child(&["load".to_string(), id.clone()]);
                        _raw = crate::term::RawMode::enter();
                        _screen = Some(Screen::open());
                        st.last.clear();
                    }
                    match crate::run::target(&id) {
                        Ok(next) => {
                            if !up {
                                loaded.add(&id);
                            }
                            st.switch(next);
                        }
                        Err(e) => st.ui.note(&e),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn keys(ui: &mut Ui, ks: &[Input]) -> Vec<Act> {
        ks.iter().map(|k| ui.key(*k)).collect()
    }

    fn typed(ui: &mut Ui, s: &str) {
        for c in s.chars() {
            ui.key(if c == ' ' { Input::Space } else { Input::Char(c) });
        }
    }

    #[test]
    fn a_reply_that_has_shown_nothing_yet_says_the_model_is_thinking_and_for_how_long() {
        let mut ui = Ui::new("m".into(), false);
        ui.entries.push(Entry::User("hi".into()));
        ui.entries.push(Entry::Reply(String::new()));
        ui.streaming = true;
        ui.waiting = Some(Instant::now() - Duration::from_secs(25));
        let shown: Vec<String> = ui.lines(40).into_iter().map(|l| l.0).collect();
        assert!(shown.contains(&"thinking... 25 s".to_string()), "{shown:?}");
        ui.entries[1] = Entry::Reply("hello".into());
        assert!(!ui.lines(40).iter().any(|l| l.0.starts_with("thinking...")), "an answer replaces the indicator");
        ui.entries[1] = Entry::Reply(String::new());
        ui.streaming = false;
        assert!(!ui.lines(40).iter().any(|l| l.0.starts_with("thinking...")), "none once the reply is over");
    }

    #[test]
    fn the_stripper_drops_reasoning_in_every_shape() {
        let both = |raw: &str| split_think(raw, true);
        assert_eq!(both("The user wants a word.\n</think>\npong"), ("The user wants a word.".into(), "pong".into()), "a closing tag alone");
        assert_eq!(both("<think>hmm</think>\nanswer"), ("hmm".into(), "answer".into()), "a full block");
        assert_eq!(both("plain answer"), (String::new(), "plain answer".into()), "no tags");
        assert_eq!(both("thinking...</think>"), ("thinking...".into(), String::new()), "an empty answer");
        assert_eq!(both("a <think>x</think>b<think>y</think> c"), ("x\ny".into(), "a b c".into()), "blocks inside an answer");
        assert_eq!(both("<think>never closed"), ("never closed".into(), String::new()), "still reasoning");
        assert_eq!(both("a < b"), (String::new(), "a < b".into()), "a lone angle bracket at the end of a finished reply stays");
    }

    #[test]
    fn a_tag_split_across_chunks_never_leaks_into_the_answer() {
        let chunks = ["The user", " asks.</th", "ink>\n", "po", "ng"];
        let mut raw = String::new();
        let mut seen = Vec::new();
        for c in chunks {
            raw.push_str(c);
            seen.push(split_think(&raw, false).1);
        }
        assert_eq!(seen[1], "The user asks.", "before the closing tag arrives the reasoning looks like an answer, minus the half tag");
        assert_eq!(seen[2], "", "once </think> is whole the reasoning leaves the answer");
        assert_eq!(seen[4], "pong");
        assert_eq!(split_think("answer <", false).1, "answer ", "a possible tag start is held back while streaming");
    }

    #[test]
    fn a_bare_q_while_typing_is_text_and_the_quit_prompt_needs_enter() {
        let mut ui = Ui::new("m".into(), false);
        typed(&mut ui, "quick");
        assert_eq!(ui.input.iter().collect::<String>(), "quick");
        assert_eq!(ui.key(Input::Esc), Act::None);
        assert_eq!(ui.mode, Mode::Command);
        assert_eq!(ui.key(Input::Char('q')), Act::None);
        assert_eq!(ui.mode, Mode::Confirm);
        assert_eq!(ui.key(Input::Char('q')), Act::None, "q goes back");
        assert_eq!(ui.mode, Mode::Command);
        assert_eq!(keys(&mut ui, &[Input::Char('q'), Input::Enter]), vec![Act::None, Act::Quit], "Enter exits");
        let mut ui = Ui::new("m".into(), false);
        assert_eq!(ui.key(Input::Interrupt), Act::None);
        assert_eq!(ui.key(Input::Char('x')), Act::None, "any other key returns");
        assert_eq!(ui.mode, Mode::Typing);
        assert_eq!(keys(&mut ui, &[Input::Interrupt, Input::Enter]), vec![Act::None, Act::Quit], "Ctrl-C is the quit prompt");
    }

    #[test]
    fn command_keys_and_line_editing() {
        let mut ui = Ui::new("m".into(), false);
        typed(&mut ui, "h\u{e9}llo");
        keys(&mut ui, &[Input::Left, Input::Left, Input::Left, Input::Backspace]);
        assert_eq!(ui.input.iter().collect::<String>(), "hllo", "a two-byte character goes as one");
        ui.key(Input::Char('e'));
        assert_eq!(ui.input.iter().collect::<String>(), "hello");
        assert_eq!(ui.key(Input::Enter), Act::Send("hello".into()));
        assert_eq!(ui.key(Input::Enter), Act::None, "an empty line sends nothing");
        typed(&mut ui, "/audio /tmp/a.wav ");
        assert_eq!(ui.key(Input::Enter), Act::Audio("/tmp/a.wav".into()));
        ui.key(Input::Esc);
        assert_eq!([Input::Char('m'), Input::Char('e'), Input::Char('r')].map(|k| ui.key(k)), [Act::Picker, Act::Export, Act::Record]);
        assert_eq!(ui.key(Input::Char('i')), Act::None);
        assert_eq!(ui.mode, Mode::Typing);
    }

    #[test]
    fn the_picker_lists_models_and_hands_back_the_choice() {
        let mut ui = Ui::new("m".into(), false);
        ui.key(Input::Esc);
        ui.open_picker(vec![("org/a".into(), true), ("org/b".into(), false)]);
        assert_eq!(ui.mode, Mode::Pick);
        let shown = ui.render(60, 10).join("\n");
        assert!(shown.contains("org/a") && shown.contains("org/b  (not loaded)") && shown.contains("[Enter] choose"), "{shown}");
        assert_eq!(keys(&mut ui, &[Input::Down, Input::Enter]), vec![Act::None, Act::Chose(1)]);
    }

    #[test]
    fn the_screen_shows_the_model_the_scrollback_the_input_and_the_keys_of_the_mode() {
        let mut ui = Ui::new("org/m  (llama.cpp, text)".into(), false);
        ui.entries = vec![Entry::User("hi".into()), Entry::Reply("thought it over.</think>\nHello!".into())];
        typed(&mut ui, "ab");
        let typing = ui.render(40, 8);
        assert_eq!(typing.len(), 8);
        assert!(typing[0].contains("org/m  (llama.cpp, text)") && typing[7].contains("[Esc] commands  [Enter] send"));
        let all = typing.join("\n");
        assert!(all.contains("\u{27a4} User ") && all.contains("hi") && all.contains("\u{27a4} Model") && all.contains("Hello!") && all.contains("thinking (3 words)") && !all.contains("thought it over"), "{all}");
        assert!(typing[6].starts_with("> ab"), "{:?}", typing[6]);
        ui.key(Input::Esc);
        assert!(ui.render(70, 8)[7].contains("[m] model  [e] export  [r] record  [q] quit  [i] type"));
        ui.key(Input::Char('q'));
        assert!(ui.render(40, 8)[7].contains("exit? Enter = exit, q = back"));
        let mut shown = Ui::new("m".into(), true);
        shown.entries = vec![Entry::Reply("a long thought</think>done".into())];
        let text = shown.render(40, 8).join("\n");
        assert!(text.contains("thinking") && text.contains("a long thought") && text.contains("done"), "{text}");
        assert_eq!(ui.render(10, 3).len(), 3, "a tiny terminal gets a one-line message");
    }

    #[test]
    fn each_turn_is_labelled_with_who_said_it_in_the_theme_colors() {
        let mut ui = Ui::new("m".into(), false);
        ui.entries = vec![Entry::User("hi".into()), Entry::Reply("Hello!".into()), Entry::Note("a note".into())];
        let lines = ui.lines(40);
        let user = ("\u{27a4} User".to_string(), Style::Secondary);
        let model = ("\u{27a4} Model".to_string(), Style::Third);
        assert_eq!(lines, [user, ("hi".into(), Style::Bold), model, ("Hello!".into(), Style::Plain), ("a note".into(), Style::Dim)]);
        ui.entries = vec![Entry::User("x".repeat(50))];
        assert_eq!(ui.lines(40).iter().map(|l| visible_width(&l.0)).collect::<Vec<_>>(), [6, 40, 10], "text wraps by itself, the label apart");
    }

    #[test]
    fn the_keys_line_shows_the_session_tokens_and_decode_speed_once_a_reply_is_in() {
        let mut ui = Ui::new("m".into(), false);
        assert!(!ui.render(80, 8)[7].contains("tokens"), "nothing before the first reply");
        ui.count(100, 4.0);
        ui.count(300, 6.0);
        let keys = ui.render(80, 8)[7].clone();
        assert!(keys.contains("[Esc] commands  [Enter] send") && keys.contains("400 tokens \u{b7} 40.0 tok/s  (last 300 \u{b7} 50.0 tok/s)"), "{keys}");
        assert_eq!(visible_width(&keys), 80);
        let narrow = ui.render(60, 8)[7].clone();
        assert!(narrow.contains("400 tokens \u{b7} 40.0 tok/s") && !narrow.contains("last"), "{narrow}");
        assert!(!ui.render(40, 8)[7].contains("tokens"), "no room, no stats");
        ui.count(1, 0.0);
        assert!(ui.render(80, 8)[7].contains("401 tokens \u{b7} 40.1 tok/s  (last 1)"));
    }

    #[test]
    fn a_reply_printed_on_a_terminal_cannot_send_escape_sequences() {
        assert_eq!(plain_text("hi\x1b]52;c;ZXZpbA==\x07 there", false, true), "hi]52;c;ZXZpbA== there");
        assert_eq!(plain_text("raw\x1b[1m", false, false), "raw\x1b[1m", "a pipe gets the reply byte for byte");
    }

    #[test]
    fn wrap_cuts_by_display_width_and_keeps_line_breaks_and_utf8() {
        assert_eq!(wrap("abcdefgh", 3), ["abc", "def", "gh"]);
        assert_eq!(wrap("ab\n\ncd", 5), ["ab", "", "cd"], "blank lines stay");
        assert_eq!(wrap("h\u{e9}llo w\u{f6}rld", 6), ["h\u{e9}llo ", "w\u{f6}rld"], "a two-byte character is one column");
        assert_eq!(wrap("x", 0), ["x"], "a zero width still makes progress");
        assert_eq!(clean("a\x1b[31mred\x07\tz\nb"), "a[31mred z\nb", "a model cannot send the terminal an escape sequence");
    }

    #[test]
    fn the_window_starts_at_the_top_and_follows_the_end_unless_scrolled() {
        assert_eq!(top(5, 10, 0, true), 0, "short text sits at the top");
        assert_eq!(top(30, 10, 0, true), 20, "long text follows the stream");
        assert_eq!(top(30, 10, 4, false), 4, "scrolled up it stays");
        assert_eq!(top(30, 10, 99, false), 20, "never past the end");
        assert_eq!(moved(4, 30, 10, -9), 0);
        assert_eq!(moved(18, 30, 10, 9), 20);
        assert_eq!(moved(0, 5, 10, 3), 0, "nothing to scroll");
    }

    fn long_chat(ui: &mut Ui) {
        ui.resize(40, 13);
        ui.entries = vec![Entry::User("first".into()), Entry::Reply((1..=40).map(|n| format!("line {n}")).collect::<Vec<_>>().join("\n"))];
    }

    #[test]
    fn text_flows_from_the_top_and_scrolling_in_command_mode_leaves_and_rejoins_the_stream() {
        let mut ui = Ui::new("m".into(), false);
        ui.resize(40, 13);
        ui.entries = vec![Entry::User("first".into()), Entry::Reply("short".into())];
        let screen = ui.render(40, 13);
        assert!(screen[1].contains("\u{27a4} User") && screen[2].contains("first") && screen[4].contains("short"), "the first line is on the first body row: {screen:?}");
        long_chat(&mut ui);
        let body = |ui: &Ui| ui.render(40, 13)[1..=10].join("|");
        assert!(body(&ui).contains("line 40") && !body(&ui).contains("first"), "following: the end is shown, the start scrolled off");
        ui.key(Input::Esc);
        assert_eq!(ui.key(Input::Char('g')), Act::None);
        assert!(body(&ui).contains("first") && ui.render(40, 13)[0].contains("scrolled: G follows"), "g goes to the top and leaves the stream");
        ui.key(Input::Down);
        ui.key(Input::Char('j'));
        assert!(!body(&ui).contains("first") && body(&ui).contains("line 1"), "j and Down scroll one line");
        ui.key(Input::PageDown);
        assert!(body(&ui).starts_with("line 9 ") && !body(&ui).contains("line 8"), "a page is the window minus one line: {}", body(&ui));
        ui.key(Input::Char('k'));
        ui.key(Input::PageUp);
        assert!(body(&ui).contains("first") && !body(&ui).contains("User"), "{}", body(&ui));
        ui.key(Input::PageUp);
        assert!(body(&ui).contains("\u{27a4} User"), "{}", body(&ui));
        ui.key(Input::Char('G'));
        assert!(body(&ui).contains("line 40") && ui.follow && !ui.render(40, 13)[0].contains("scrolled"), "G jumps to the end and follows again");
        ui.key(Input::Char('g'));
        ui.key(Input::Char('i'));
        typed(&mut ui, "more");
        assert_eq!(ui.key(Input::Enter), Act::Send("more".into()));
        assert!(ui.follow, "a new send re-enables follow");
    }

    #[test]
    fn page_keys_scroll_while_typing_too_and_resize_rewraps() {
        let mut ui = Ui::new("m".into(), false);
        long_chat(&mut ui);
        ui.key(Input::PageUp);
        assert!(!ui.follow);
        let wide = ui.lines(40).len();
        assert!(ui.lines(10).len() >= wide, "a narrower terminal wraps into more lines");
        assert_eq!(wrap("a long enough line to wrap", 10).len(), 3);
        assert_eq!(parse_input(b"\x1b[5~\x1b[6~\x15\x04"), [Input::PageUp, Input::PageDown, Input::PageUp, Input::PageDown]);
    }

    #[test]
    fn an_export_is_in_the_trace_layout_with_private_modes_and_never_overwrites() {
        let home = tempfile::tempdir().unwrap();
        let out = tempfile::tempdir().unwrap();
        let dir = out.path().join("goethe-chat-x");
        let user = |t: &str| json!({"role": "user", "content": t});
        let turns = vec![
            Turn { started: SystemTime::now(), seconds: 1.5, ttft: Some(0.2), messages: vec![user("hi")], reply: "<think>hmm</think>Hello!".into() },
            Turn {
                started: SystemTime::now(),
                seconds: 0.5,
                ttft: None,
                messages: vec![user("hi"), json!({"role": "assistant", "content": "<think>hmm</think>Hello!"}), user("and?")],
                reply: "Fine.".into(),
            },
        ];
        crate::trace::export(home.path(), &dir, captures("org/m", "inst-1", &dir, &turns)).unwrap();
        let mut files = Vec::new();
        let mut stack = vec![dir.clone()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).unwrap().flatten() {
                match e.path().is_dir() {
                    true => stack.push(e.path()),
                    false => files.push(e.path()),
                }
            }
        }
        let session = files.iter().find(|f| f.to_string_lossy().ends_with(".jsonl.gz")).expect("a session file");
        assert_eq!(std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
        assert!(files.iter().all(|f| std::fs::metadata(f).unwrap().permissions().mode() & 0o077 == 0), "files are private");
        let mut text = String::new();
        std::io::Read::read_to_string(&mut flate2::read::MultiGzDecoder::new(std::fs::File::open(session).unwrap()), &mut text).unwrap();
        let rows: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0]["response"]["body"]["choices"][0]["message"]["content"], "<think>hmm</think>Hello!", "the raw reply, reasoning included");
        assert_eq!(rows[1]["parent"]["turn_id"], rows[0]["turn_id"], "turns of one chat are one session");
        assert_eq!((rows[1]["messages_from"].as_u64(), rows[1]["session_id"].clone()), (Some(2), rows[0]["session_id"].clone()));
        assert_eq!((rows[0]["api"].as_str(), rows[0]["model"].as_str()), (Some("chat.completions"), Some("org/m")));
        let again = crate::trace::export(home.path(), &dir, captures("org/m", "inst-1", &dir, &turns)).unwrap_err();
        assert!(again.contains("exists"), "{again}");
    }
}
