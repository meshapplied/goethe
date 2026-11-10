// goethe api manage: a full-screen, two-column key manager in the style of ranger; the left picks what to act on, the right shows and changes it.

use crate::paths::goethe_home;
use crate::security;
use serde_json::Value;
use std::io::{IsTerminal, Write};

const SCRIPT_HINT: &[&str] = &[
    "goethe key --create my-app",
    "goethe key --create my-app --models <model-id>",
    "goethe key --revoke <key-id>",
];
const DEFAULT_RPM: i64 = 120;
const MIN_WIDTH: usize = 64;
const KEY_WIDTH: usize = 44;
const GAP: usize = 4;
const MIN_HEIGHT: usize = 12;
const LEFT_MAX: usize = 36;
const TEXT_MAX: usize = 64;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyRow {
    pub id: String,
    pub name: String,
    pub models: Option<String>,
    pub admin: bool,
    pub token: Option<String>,
    pub prefix: String,
    pub rpm: i64,
    pub expires: Option<String>,
}

fn keys() -> Result<Vec<KeyRow>, String> {
    let field = |r: &Value, k: &str| r.get(k).and_then(Value::as_str).map(str::to_string).filter(|v| !v.is_empty());
    Ok(security::key_list_detailed()?
        .iter()
        .map(|r| KeyRow {
            id: field(r, "id").unwrap_or_default(),
            name: field(r, "name").unwrap_or_default(),
            models: field(r, "models"),
            admin: r.get("admin").and_then(Value::as_bool).unwrap_or(false),
            token: field(r, "key"),
            prefix: field(r, "prefix").unwrap_or_default(),
            rpm: r.get("rpm").and_then(Value::as_i64).unwrap_or(DEFAULT_RPM),
            expires: field(r, "expires"),
        })
        .collect())
}

fn installed_models() -> Vec<String> {
    let Ok(con) = crate::db::open(&goethe_home()) else { return Vec::new() };
    let Ok(mut stmt) = con.prepare("SELECT model_id FROM installed_models ORDER BY model_id") else { return Vec::new() };
    stmt.query_map([], |r| r.get::<_, String>(0)).map(|rows| rows.flatten().collect()).unwrap_or_default()
}

fn rpm_text(rpm: i64) -> String {
    if rpm == 0 {
        "unlimited".to_string()
    } else {
        format!("{rpm} per minute")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Input {
    Up,
    Down,
    Left,
    Right,
    Enter,
    Space,
    Esc,
    Backspace,
    Interrupt,
    Char(char),
}

fn parse_input(bytes: &[u8]) -> Vec<Input> {
    // Arrows (CSI or SS3), Enter, Space, Backspace, Ctrl-C, a lone Esc, and typed characters; other escape sequences are skipped.
    let chars: Vec<char> = String::from_utf8_lossy(bytes).chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        i += 1;
        match c {
            '\x1b' if i < chars.len() && matches!(chars[i], '[' | 'O') => {
                let mut j = i + 1;
                while j < chars.len() && !(chars[j].is_ascii_alphabetic() || chars[j] == '~') {
                    j += 1;
                }
                if let Some(end) = chars.get(j) {
                    out.extend(match end {
                        'A' => Some(Input::Up),
                        'B' => Some(Input::Down),
                        'C' => Some(Input::Right),
                        'D' => Some(Input::Left),
                        _ => None,
                    });
                }
                i = j + 1;
            }
            '\x1b' => out.push(Input::Esc),
            '\r' | '\n' => out.push(Input::Enter),
            ' ' => out.push(Input::Space),
            '\x7f' | '\x08' => out.push(Input::Backspace),
            '\x03' => out.push(Input::Interrupt),
            c if c.is_control() => {}
            c => out.push(Input::Char(c)),
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LeftItem {
    Create,
    Done,
    Key(usize),
}

fn left_items(keys: usize) -> Vec<LeftItem> {
    [LeftItem::Create, LeftItem::Done].into_iter().chain((0..keys).map(LeftItem::Key)).collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Form {
    name: String,
    every: bool,
    models: Vec<String>,
    ticks: Vec<bool>,
    rpm: i64,
    expires: Option<String>,
}

impl Form {
    fn blank(installed: &[String]) -> Form {
        Form {
            name: String::new(),
            every: true,
            models: installed.to_vec(),
            ticks: vec![false; installed.len()],
            rpm: DEFAULT_RPM,
            expires: None,
        }
    }

    fn of(key: &KeyRow, installed: &[String]) -> Form {
        // Installed models plus any the key already names, so a scope never loses a model that is not installed now.
        let named: Vec<String> = key.models.iter().flat_map(|m| m.split(',').map(|s| s.trim().to_string())).collect();
        let mut models = installed.to_vec();
        models.extend(named.iter().filter(|m| !installed.contains(m)).cloned());
        let ticks = models.iter().map(|m| named.contains(m)).collect();
        Form { name: key.name.clone(), every: key.models.is_none(), models, ticks, rpm: key.rpm, expires: key.expires.clone() }
    }

    fn scope(&self) -> Result<Option<String>, String> {
        if self.every {
            return Ok(None);
        }
        let chosen: Vec<&str> = self.models.iter().zip(&self.ticks).filter(|(_, t)| **t).map(|(m, _)| m.as_str()).collect();
        if chosen.is_empty() {
            return Err("Tick at least one model, or Every model.".to_string());
        }
        Ok(Some(chosen.join(",")))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Field {
    Name,
    Every,
    Model(usize),
    Rpm,
    Expires,
    Apply,
    Delete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Armed {
    Delete,
    Discard,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Effect {
    Nothing,
    Quit,
    Create { name: String, scope: Option<String>, rpm: i64, expires: Option<String> },
    Save { id: String, scope: Option<String>, rpm: i64, expires: Option<String> },
    Delete { id: String, name: String },
}

struct App {
    keys: Vec<KeyRow>,
    installed: Vec<String>,
    left: usize,
    right: usize,
    focus_right: bool,
    form: Form,
    base: Form,
    editing: Option<(Field, String)>,
    armed: Option<Armed>,
    status: String,
}

impl App {
    fn new(keys: Vec<KeyRow>, installed: Vec<String>) -> App {
        let form = Form::blank(&installed);
        App {
            keys,
            installed,
            left: 0,
            right: 0,
            focus_right: false,
            base: form.clone(),
            form,
            editing: None,
            armed: None,
            status: String::new(),
        }
    }

    fn item(&self) -> LeftItem {
        left_items(self.keys.len())[self.left]
    }

    fn hover(&mut self, left: usize) {
        // Point the left column at an entry and load its form on the right.
        self.left = left.min(left_items(self.keys.len()).len() - 1);
        self.form = match self.item() {
            LeftItem::Key(i) => Form::of(&self.keys[i], &self.installed),
            _ => Form::blank(&self.installed),
        };
        self.base = self.form.clone();
        self.right = 0;
        self.editing = None;
        self.armed = None;
    }

    fn reload(&mut self, keys: Vec<KeyRow>, hover_name: Option<&str>) {
        // New key list after a change; hover a named key, else stay near the old position.
        self.keys = keys;
        let at = hover_name.and_then(|n| self.keys.iter().position(|k| k.name == n)).map(|i| i + 2).unwrap_or(self.left);
        self.hover(at);
    }

    fn fields(&self) -> Vec<Field> {
        let models = (0..self.form.models.len()).map(Field::Model);
        match self.item() {
            LeftItem::Create => [Field::Name, Field::Every].into_iter().chain(models).chain([Field::Rpm, Field::Expires, Field::Apply]).collect(),
            LeftItem::Key(_) => [Field::Every].into_iter().chain(models).chain([Field::Rpm, Field::Expires, Field::Apply, Field::Delete]).collect(),
            LeftItem::Done => Vec::new(),
        }
    }

    fn field(&self) -> Option<Field> {
        self.fields().get(self.right).copied()
    }

    fn dirty(&self) -> bool {
        self.form != self.base
    }

    fn unsaved_note(&self) -> Option<String> {
        // What leaving now drops, said after the screen closes; nothing unsaved is ever written.
        if !self.dirty() {
            return None;
        }
        Some(match self.item() {
            LeftItem::Key(i) => format!("Unsaved changes to {} were not saved.", self.keys[i].name),
            _ => "The new key was not created.".to_string(),
        })
    }

    fn handle(&mut self, input: Input) -> Effect {
        // One key press; anything that touches stored keys comes back as an Effect for the caller to carry out.
        if input == Input::Interrupt {
            return Effect::Quit;
        }
        if self.editing.is_some() {
            self.edit(input);
            return Effect::Nothing;
        }
        let armed = self.armed.take();
        if input == Input::Char('q') || (!self.focus_right && input == Input::Esc) {
            return Effect::Quit;
        }
        self.status.clear();
        if self.focus_right {
            self.handle_right(input, armed)
        } else {
            self.handle_left(input)
        }
    }

    fn handle_left(&mut self, input: Input) -> Effect {
        let n = left_items(self.keys.len()).len();
        match input {
            Input::Up | Input::Char('k') => self.hover(self.left.saturating_sub(1)),
            Input::Down | Input::Char('j') => self.hover((self.left + 1).min(n - 1)),
            Input::Right | Input::Enter | Input::Char('l') => match self.item() {
                LeftItem::Done => return Effect::Quit,
                _ => {
                    self.focus_right = true;
                    self.right = 0;
                }
            },
            _ => {}
        }
        Effect::Nothing
    }

    fn handle_right(&mut self, input: Input, armed: Option<Armed>) -> Effect {
        let last = self.fields().len().saturating_sub(1);
        match input {
            Input::Up | Input::Char('k') => self.right = self.right.saturating_sub(1),
            Input::Down | Input::Char('j') => self.right = (self.right + 1).min(last),
            Input::Left | Input::Esc | Input::Char('h') => {
                if self.dirty() && armed != Some(Armed::Discard) {
                    self.armed = Some(Armed::Discard);
                    self.status = "Unsaved changes. Save them, or press \u{2190} again to discard them.".to_string();
                } else {
                    self.form = self.base.clone();
                    self.focus_right = false;
                }
            }
            Input::Space | Input::Enter => return self.activate(input, armed),
            _ => {}
        }
        Effect::Nothing
    }

    fn activate(&mut self, input: Input, armed: Option<Armed>) -> Effect {
        // Space or Enter on the hovered field: tick, start an edit, create, save or delete.
        let Some(field) = self.field() else { return Effect::Nothing };
        match field {
            Field::Every => {
                self.form.every = !self.form.every;
                if !self.form.every && !self.form.ticks.contains(&true) {
                    self.status = "Now tick the models this key may use.".to_string();
                }
            }
            Field::Model(i) => {
                if self.form.every {
                    self.form.every = false;
                    self.form.ticks = vec![false; self.form.models.len()];
                }
                self.form.ticks[i] = !self.form.ticks[i];
            }
            Field::Name => self.editing = Some((Field::Name, self.form.name.clone())),
            Field::Rpm => self.editing = Some((Field::Rpm, self.form.rpm.to_string())),
            Field::Expires => {
                self.editing = Some((Field::Expires, self.form.expires.clone().unwrap_or_else(|| "never".to_string())))
            }
            Field::Apply => return self.apply(),
            Field::Delete => {
                let LeftItem::Key(i) = self.item() else { return Effect::Nothing };
                if input == Input::Enter && armed == Some(Armed::Delete) {
                    return Effect::Delete { id: self.keys[i].id.clone(), name: self.keys[i].name.clone() };
                }
                self.armed = Some(Armed::Delete);
                self.status = format!("Press Enter again to delete {}. Clients using it will be refused.", self.keys[i].name);
            }
        }
        Effect::Nothing
    }

    fn apply(&mut self) -> Effect {
        let scope = match self.form.scope() {
            Ok(s) => s,
            Err(e) => {
                self.status = e;
                return Effect::Nothing;
            }
        };
        let (rpm, expires) = (self.form.rpm, self.form.expires.clone());
        match self.item() {
            LeftItem::Create => {
                let name = self.form.name.trim().to_string();
                if name.is_empty() {
                    self.status = "Give the key a name first.".to_string();
                    self.right = 0;
                    return Effect::Nothing;
                }
                if self.keys.iter().any(|k| k.name == name) {
                    self.status = format!("A key named {name} already exists; pick another name.");
                    self.right = 0;
                    return Effect::Nothing;
                }
                Effect::Create { name, scope, rpm, expires }
            }
            LeftItem::Key(i) if self.dirty() => Effect::Save { id: self.keys[i].id.clone(), scope, rpm, expires },
            LeftItem::Key(_) => {
                self.status = "Nothing to save.".to_string();
                Effect::Nothing
            }
            LeftItem::Done => Effect::Nothing,
        }
    }

    fn edit(&mut self, input: Input) {
        // Typing into Name, Rate limit or Expires; Enter keeps a valid value, Esc drops the edit.
        let Some((field, mut text)) = self.editing.take() else { return };
        match input {
            Input::Char(c) if text.chars().count() < TEXT_MAX => text.push(c),
            Input::Space if field == Field::Name && text.chars().count() < TEXT_MAX => text.push(' '),
            Input::Backspace => {
                text.pop();
            }
            Input::Esc => {
                self.status.clear();
                return;
            }
            Input::Enter => match self.commit(field, &text) {
                Ok(()) => {
                    self.status.clear();
                    return;
                }
                Err(e) => self.status = e,
            },
            _ => {}
        }
        self.editing = Some((field, text));
    }

    fn commit(&mut self, field: Field, text: &str) -> Result<(), String> {
        let text = text.trim();
        match field {
            Field::Name => {
                security::key_name_check(text).map_err(|_| "Names are 1 to 64 letters, digits, spaces or punctuation.".to_string())?;
                self.form.name = text.to_string();
            }
            Field::Rpm => {
                self.form.rpm = text
                    .parse::<i64>()
                    .ok()
                    .filter(|n| *n >= 0)
                    .ok_or("The rate limit is a whole number of requests per minute, 0 for unlimited.")?;
            }
            Field::Expires => {
                self.form.expires = match text {
                    "" | "never" => None,
                    t => Some(
                        security::parse_expires(t).map_err(|_| "Expires takes never, days like 30d, or a date like 2027-01-31.".to_string())?,
                    ),
                };
            }
            _ => {}
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Style {
    Plain,
    Bold,
    Dim,
    Red,
    Yellow,
    Selected,
    Hovered,
}

struct Line {
    text: String,
    style: Style,
}

fn line(text: impl Into<String>, style: Style) -> Line {
    Line { text: text.into(), style }
}

fn paint(text: &str, style: Style) -> String {
    let code = match style {
        Style::Plain => return text.to_string(),
        Style::Bold | Style::Hovered => "1",
        Style::Dim => "2",
        Style::Red => "31",
        Style::Yellow => "33",
        Style::Selected => "1;38;5;117",
    };
    format!("\x1b[{code}m{text}\x1b[0m")
}

fn fit(text: &str, width: usize) -> String {
    // Cut to the width with an ellipsis, then pad with spaces.
    let n = text.chars().count();
    let cut: String = if n > width { text.chars().take(width.saturating_sub(1)).chain(['\u{2026}']).collect() } else { text.to_string() };
    let pad = width.saturating_sub(cut.chars().count());
    format!("{cut}{}", " ".repeat(pad))
}

fn left_lines(app: &App) -> (Vec<Line>, usize) {
    // Create and Done, a gap, then the keys; returns the lines and the hovered line.
    let width = app.keys.iter().map(|k| k.name.chars().count()).max().unwrap_or(0);
    let mut lines = vec![line("+ Create new key", Style::Plain), line("Done", Style::Plain), line("", Style::Plain)];
    lines.extend(app.keys.iter().map(|k| line(format!("{:<width$}  {}", k.name, k.id), Style::Plain)));
    if app.keys.is_empty() {
        lines.push(line("No keys yet.", Style::Dim));
    }
    let hovered = if app.left < 2 { app.left } else { app.left + 1 };
    let style = if app.focus_right { Style::Hovered } else { Style::Selected };
    let marker = if app.focus_right { "\u{203A} " } else { "\u{25CF} " };
    for (i, l) in lines.iter_mut().enumerate() {
        if i == hovered {
            l.text = format!("{marker}{}", l.text);
            l.style = style;
        } else if !l.text.is_empty() {
            l.text = format!("  {}", l.text);
        }
    }
    (lines, hovered)
}

fn field_text(app: &App, field: Field) -> (String, Style) {
    let f = &app.form;
    let editing = app.editing.as_ref().filter(|(e, _)| *e == field).map(|(_, t)| format!("{t}\u{2588}"));
    let value = |shown: String| editing.clone().unwrap_or(shown);
    match field {
        Field::Name => {
            let shown = if f.name.is_empty() { "(Enter to type a name)".to_string() } else { f.name.clone() };
            (format!("Name:        {}", value(shown)), Style::Plain)
        }
        Field::Every => (format!("[{}] Every model", if f.every { "x" } else { " " }), Style::Plain),
        Field::Model(i) => {
            let tick = if f.ticks[i] && !f.every { "x" } else { " " };
            (format!("[{tick}] {}", f.models[i]), if f.every { Style::Dim } else { Style::Plain })
        }
        Field::Rpm => (format!("Rate limit:  {}", value(rpm_text(f.rpm))), Style::Plain),
        Field::Expires => (format!("Expires:     {}", value(f.expires.clone().unwrap_or_else(|| "never".to_string()))), Style::Plain),
        Field::Apply if app.item() == LeftItem::Create => ("Create key".to_string(), Style::Bold),
        Field::Apply if app.dirty() => ("Save changes (unsaved)".to_string(), Style::Yellow),
        Field::Apply => ("Save changes".to_string(), Style::Dim),
        Field::Delete => ("Delete key".to_string(), Style::Red),
    }
}

fn right_lines(app: &App) -> (Vec<Line>, usize) {
    // The hovered entry: a key's value and fixed facts on top, then what can change; returns the lines and the hovered field's line.
    let mut lines = Vec::new();
    match app.item() {
        LeftItem::Done => {
            lines.push(line("Leave goethe api manage.", Style::Bold));
            lines.push(line("Saved changes already apply; goethe api lists the keys.", Style::Dim));
            return (lines, 0);
        }
        LeftItem::Create => lines.push(line("New key", Style::Bold)),
        LeftItem::Key(i) => {
            let k = &app.keys[i];
            lines.push(line("API key:", Style::Bold));
            lines.push(match &k.token {
                Some(t) => line(format!("  {t}"), Style::Plain),
                None => line(format!("  {}... (made before keys were stored)", k.prefix), Style::Dim),
            });
            lines.push(line(format!("  Name:        {}", k.name), Style::Dim));
            lines.push(line(format!("  ID:          {}{}", k.id, if k.admin { "  (admin)" } else { "" }), Style::Dim));
        }
    }
    let mut at = 0;
    for (n, field) in app.fields().iter().enumerate() {
        match field {
            Field::Every => {
                lines.push(line("", Style::Plain));
                lines.push(line("Allowed models", Style::Bold));
            }
            Field::Rpm | Field::Apply => lines.push(line("", Style::Plain)),
            _ => {}
        }
        let (text, style) = field_text(app, *field);
        let indent = if matches!(field, Field::Every | Field::Model(_)) { "  " } else { "" };
        if app.focus_right && n == app.right {
            at = lines.len();
            lines.push(line(format!("\u{25CF} {indent}{text}"), Style::Selected));
        } else {
            lines.push(line(format!("  {indent}{text}"), style));
        }
        if *field == Field::Every && app.form.models.is_empty() {
            lines.push(line("    (no models installed yet)", Style::Dim));
        }
    }
    (lines, at)
}

fn window(len: usize, hovered: usize, rows: usize) -> usize {
    // First line to show so the hovered one stays visible with a line of context.
    if len <= rows || hovered + 2 <= rows {
        return 0;
    }
    (hovered + 2 - rows).min(len - rows)
}

const RPM_FORMAT: &str = "requests per minute, a whole number such as 60; 0 means unlimited";
const EXPIRES_FORMAT: &str = "never, a number of days such as 30d, or a date such as 2027-01-31";
const NAME_FORMAT: &str = "1 to 64 letters, digits, spaces or punctuation";

fn help(app: &App) -> String {
    // The keys to press, or for a value the formats it takes.
    if let Some((field, _)) = &app.editing {
        let format = match field {
            Field::Rpm => RPM_FORMAT,
            Field::Expires => EXPIRES_FORMAT,
            _ => NAME_FORMAT,
        };
        return format!("{format}.  Enter keeps, Esc cancels");
    }
    if !app.focus_right {
        return "\u{2191}\u{2193} move   \u{2192} open   q quit".to_string();
    }
    match app.field() {
        Some(Field::Rpm) => format!("Rate limit: {RPM_FORMAT}.  Enter to change"),
        Some(Field::Expires) => format!("Expires: {EXPIRES_FORMAT}.  Enter to change"),
        Some(Field::Name) => format!("Name: {NAME_FORMAT}.  Enter to type"),
        _ => "\u{2191}\u{2193} move   Space tick   Enter change   \u{2190} back   q quit".to_string(),
    }
}

fn render(app: &App, width: usize, height: usize) -> Vec<String> {
    // Exactly height lines: title, a gap, the two columns, a gap, then the status or what the keys do.
    if width < MIN_WIDTH || height < MIN_HEIGHT {
        let mut out = vec![format!("Make the terminal at least {MIN_WIDTH}x{MIN_HEIGHT} (q quits).")];
        out.resize(height.max(1), String::new());
        return out;
    }
    let (left, left_at) = left_lines(app);
    let (right, right_at) = right_lines(app);
    let lw = left.iter().map(|l| l.text.chars().count()).max().unwrap_or(0).clamp(16, LEFT_MAX).min(width - GAP - KEY_WIDTH);
    let rw = width - lw - GAP;
    let rows = height - 4;
    let (l0, r0) = (window(left.len(), left_at, rows), window(right.len(), right_at, rows));
    let mut out = vec![paint(&fit("API keys", width), Style::Bold), String::new()];
    for r in 0..rows {
        let cell = |lines: &[Line], start: usize, w: usize| match lines.get(start + r) {
            Some(l) => paint(&fit(&l.text, w), l.style),
            None => " ".repeat(w),
        };
        out.push(format!("{}{}{}", cell(&left, l0, lw), " ".repeat(GAP), cell(&right, r0, rw)).trim_end().to_string());
    }
    out.push(String::new());
    out.push(if app.status.is_empty() { paint(&fit(&help(app), width), Style::Dim) } else { paint(&fit(&app.status, width), Style::Yellow) });
    out
}

fn screen_size() -> (usize, usize) {
    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    if unsafe { libc::ioctl(1, libc::TIOCGWINSZ, &mut ws) } == 0 && ws.ws_col > 0 && ws.ws_row > 0 {
        (usize::from(ws.ws_col), usize::from(ws.ws_row))
    } else {
        (80, 24)
    }
}

struct Screen;

impl Screen {
    fn open() -> Screen {
        // The alternate screen, cleared, cursor hidden; the user's scrollback comes back on close.
        print!("\x1b[?1049h\x1b[H\x1b[2J\x1b[?25l");
        let _ = std::io::stdout().flush();
        Screen
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        print!("\x1b[?25h\x1b[?1049l");
        let _ = std::io::stdout().flush();
    }
}

fn frame_update(last: &[String], next: &[String]) -> String {
    // Only the lines that changed, each rewritten in place; everything when the size changed.
    if last.len() != next.len() {
        return format!("\x1b[H\x1b[2J{}", next.join("\r\n"));
    }
    let mut out = String::new();
    for (i, line) in next.iter().enumerate() {
        if last[i] != *line {
            out.push_str(&format!("\x1b[{};1H{line}\x1b[K", i + 1));
        }
    }
    out
}

fn draw(app: &App, last: &mut Vec<String>) {
    let (w, h) = screen_size();
    let next = render(app, w, h);
    let update = frame_update(last, &next);
    if !update.is_empty() {
        print!("{update}");
        let _ = std::io::stdout().flush();
    }
    last.clear();
    last.extend(next);
}

fn carry_out(app: &mut App, effect: Effect, made: &mut Vec<(String, String)>) -> Result<(), String> {
    // Write one change, reload the list and say what happened on the status line.
    let result = match &effect {
        Effect::Create { name, scope, rpm, expires } => {
            security::key_create(name, true, Some(*rpm), expires.as_deref(), scope.as_deref(), false).map(|token| {
                made.push((name.clone(), token));
                format!("Created {name}. Its key is at the top right; goethe api shows it again.")
            })
        }
        Effect::Save { id, scope, rpm, expires } => security::key_set_models(id, scope.as_deref())
            .and_then(|()| security::key_set_limits(id, *rpm, expires.as_deref()))
            .map(|()| "Saved. The gateway uses it from the next request.".to_string()),
        Effect::Delete { id, name } => security::key_revoke(id).map(|()| format!("Deleted {name}.")),
        Effect::Nothing | Effect::Quit => return Ok(()),
    };
    let hover = match &effect {
        Effect::Create { name, .. } => Some(name.clone()),
        Effect::Save { .. } => app.left.checked_sub(2).and_then(|i| app.keys.get(i)).map(|k| k.name.clone()),
        _ => None,
    };
    let focus_right = matches!(effect, Effect::Save { .. });
    let right = app.right;
    match result {
        Ok(msg) => {
            app.reload(keys()?, hover.as_deref());
            app.focus_right = focus_right;
            if focus_right {
                app.right = right.min(app.fields().len().saturating_sub(1));
            }
            app.status = msg;
        }
        Err(e) => app.status = format!("Not done: {e}"),
    }
    Ok(())
}

pub fn run() -> Result<(), String> {
    // Full screen until q or Done; keys made here are printed afterwards, since the screen they were shown on goes away.
    if !(std::io::stdin().is_terminal() && std::io::stdout().is_terminal()) {
        return Err(format!("goethe api manage needs a terminal. {}", crate::term::commands("In scripts:", SCRIPT_HINT).trim_end()));
    }
    let mut app = App::new(keys()?, installed_models());
    let mut made: Vec<(String, String)> = Vec::new();
    let raw = crate::term::RawMode::enter().ok_or("could not switch the terminal to raw mode")?;
    let screen = Screen::open();
    let mut size = screen_size();
    let mut last: Vec<String> = Vec::new();
    draw(&app, &mut last);
    let mut buf = [0u8; 16];
    let outcome = 'outer: loop {
        let mut fds = libc::pollfd { fd: 0, events: libc::POLLIN, revents: 0 };
        if unsafe { libc::poll(&mut fds, 1, 250) } <= 0 {
            if screen_size() != size {
                size = screen_size();
                last.clear();
                draw(&app, &mut last);
            }
            continue;
        }
        let n = match crate::term::read_keys(&mut buf) {
            Ok(0) => break Ok(()),
            Ok(n) => n,
            Err(e) => break Err(e.to_string()),
        };
        for input in parse_input(&buf[..n]) {
            match app.handle(input) {
                Effect::Quit => break 'outer Ok(()),
                effect => {
                    if let Err(e) = carry_out(&mut app, effect, &mut made) {
                        break 'outer Err(e);
                    }
                }
            }
        }
        draw(&app, &mut last);
    };
    let unsaved = app.unsaved_note();
    drop(screen);
    drop(raw);
    if let Some(note) = unsaved {
        println!("{note}");
    }
    for (name, token) in &made {
        println!("Created {name}: {token}");
    }
    if !made.is_empty() {
        println!("goethe api shows every key again.");
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(id: &str, name: &str, models: Option<&str>) -> KeyRow {
        KeyRow {
            id: id.into(),
            name: name.into(),
            models: models.map(String::from),
            admin: false,
            token: Some(format!("goethe_sk_{id}")),
            prefix: String::new(),
            rpm: 120,
            expires: None,
        }
    }

    fn app() -> App {
        App::new(vec![key("id-a", "dev", None), key("id-b", "chat", Some("m/beta"))], vec!["m/alpha".into(), "m/beta".into()])
    }

    fn press(app: &mut App, keys: &[Input]) -> Effect {
        keys.iter().fold(Effect::Nothing, |_, k| app.handle(*k))
    }

    fn typed(s: &str) -> Vec<Input> {
        s.chars().map(|c| if c == ' ' { Input::Space } else { Input::Char(c) }).collect()
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
    fn input_parses_arrows_keys_and_text() {
        use Input::*;
        assert_eq!(parse_input(b"\x1b[A\x1b[B\x1bOC\x1b[D"), vec![Up, Down, Right, Left]);
        assert_eq!(parse_input(b"\r \x7f\x03\x1b"), vec![Enter, Space, Backspace, Interrupt, Esc]);
        assert_eq!(parse_input(b"q\x1b[3~j"), vec![Char('q'), Char('j')], "Delete's sequence is skipped");
        assert_eq!(parse_input("\u{e9}".as_bytes()), vec![Char('\u{e9}')]);
    }

    #[test]
    fn the_left_column_is_create_done_then_keys() {
        let mut a = app();
        assert_eq!(left_items(2), vec![LeftItem::Create, LeftItem::Done, LeftItem::Key(0), LeftItem::Key(1)]);
        press(&mut a, &[Input::Down, Input::Down]);
        assert_eq!(a.item(), LeftItem::Key(0));
        press(&mut a, &[Input::Down, Input::Down, Input::Down]);
        assert_eq!(a.item(), LeftItem::Key(1), "the cursor stops at the last key");
        press(&mut a, &[Input::Up, Input::Up, Input::Up, Input::Up]);
        assert_eq!(a.item(), LeftItem::Create);
        assert_eq!(press(&mut a, &[Input::Down, Input::Right]), Effect::Quit, "Done leaves");
    }

    #[test]
    fn a_new_key_is_named_scoped_and_created_on_the_right() {
        let mut a = app();
        press(&mut a, &[Input::Right, Input::Enter]);
        assert!(a.editing.is_some(), "Enter on Name starts typing");
        press(&mut a, &typed("my laptop"));
        press(&mut a, &[Input::Enter]);
        assert_eq!(a.form.name, "my laptop");
        press(&mut a, &[Input::Down, Input::Down, Input::Space]);
        assert_eq!((a.form.every, a.form.ticks.clone()), (false, vec![true, false]), "ticking a model ends Every model");
        press(&mut a, &[Input::Down, Input::Down, Input::Enter]);
        press(&mut a, &typed("0"));
        press(&mut a, &[Input::Backspace, Input::Char('6'), Input::Char('0'), Input::Enter]);
        assert_eq!(a.form.rpm, 12060, "an edit starts from the current value");
        press(&mut a, &[Input::Enter]);
        press(&mut a, &[Input::Backspace; 5]);
        press(&mut a, &[Input::Char('6'), Input::Char('0'), Input::Enter]);
        assert_eq!(a.form.rpm, 60);
        press(&mut a, &[Input::Down, Input::Enter]);
        press(&mut a, &[Input::Backspace; 5]);
        press(&mut a, &typed("2099-01-31"));
        press(&mut a, &[Input::Enter]);
        assert_eq!(a.form.expires.as_deref(), Some("2099-01-31"));
        let made = press(&mut a, &[Input::Down, Input::Enter]);
        assert_eq!(made, Effect::Create { name: "my laptop".into(), scope: Some("m/alpha".into()), rpm: 60, expires: Some("2099-01-31".into()) });
    }

    #[test]
    fn bad_values_are_refused_where_they_are_typed() {
        let mut a = app();
        press(&mut a, &[Input::Right, Input::Down, Input::Down, Input::Down, Input::Down, Input::Enter]);
        assert_eq!(a.field(), Some(Field::Rpm));
        press(&mut a, &[Input::Char('x'), Input::Enter]);
        assert!(a.editing.is_some() && a.status.contains("whole number"), "{}", a.status);
        press(&mut a, &[Input::Esc]);
        assert_eq!((a.editing.is_none(), a.form.rpm), (true, 120), "Esc drops the edit");
        press(&mut a, &[Input::Down, Input::Down, Input::Enter]);
        assert!(a.status.contains("name first"), "{}", a.status);
        assert_eq!(a.field(), Some(Field::Name), "the cursor goes to the missing name");
        press(&mut a, &[Input::Enter]);
        press(&mut a, &typed("dev"));
        press(&mut a, &[Input::Enter]);
        press(&mut a, &[Input::Down, Input::Space, Input::Down, Input::Down, Input::Down, Input::Down, Input::Down, Input::Enter]);
        assert_eq!(a.field(), Some(Field::Apply));
        assert!(a.status.contains("at least one model"), "{}", a.status);
        press(&mut a, &[Input::Up, Input::Up, Input::Up, Input::Space, Input::Down, Input::Down, Input::Down, Input::Enter]);
        assert!(a.status.contains("already exists"), "{}", a.status);
    }

    #[test]
    fn an_existing_key_is_changed_saved_and_deleted_from_the_right() {
        let mut a = app();
        press(&mut a, &[Input::Down, Input::Down, Input::Down, Input::Right]);
        assert_eq!((a.item(), a.field()), (LeftItem::Key(1), Some(Field::Every)));
        assert_eq!((a.form.every, a.form.ticks.clone()), (false, vec![false, true]), "the key's scope is ticked");
        press(&mut a, &[Input::Down, Input::Space]);
        assert!(a.dirty());
        let left = press(&mut a, &[Input::Left]);
        assert_eq!((left, a.focus_right), (Effect::Nothing, true), "unsaved changes hold the first back");
        assert!(a.status.contains("again to discard"));
        let saved = press(&mut a, &[Input::Down, Input::Down, Input::Down, Input::Down, Input::Enter]);
        assert_eq!(saved, Effect::Save { id: "id-b".into(), scope: Some("m/alpha,m/beta".into()), rpm: 120, expires: None });
        press(&mut a, &[Input::Down, Input::Enter]);
        assert!(a.status.contains("Press Enter again"));
        assert_eq!(press(&mut a, &[Input::Enter]), Effect::Delete { id: "id-b".into(), name: "chat".into() });
        press(&mut a, &[Input::Enter]);
        press(&mut a, &[Input::Up]);
        assert_eq!(press(&mut a, &[Input::Down, Input::Enter]), Effect::Nothing, "a move disarms the delete");
    }

    #[test]
    fn q_leaves_at_once_and_saves_nothing() {
        let mut a = app();
        press(&mut a, &[Input::Down, Input::Down, Input::Right, Input::Space]);
        assert!(a.dirty());
        assert_eq!(press(&mut a, &[Input::Char('q')]), Effect::Quit, "no Save effect, so nothing is written");
        assert_eq!(a.unsaved_note().as_deref(), Some("Unsaved changes to dev were not saved."));
        let mut c = app();
        press(&mut c, &[Input::Right, Input::Enter, Input::Char('x'), Input::Enter]);
        assert_eq!(press(&mut c, &[Input::Char('q')]), Effect::Quit);
        assert_eq!(c.unsaved_note().as_deref(), Some("The new key was not created."));
        let mut e = app();
        press(&mut e, &[Input::Right, Input::Enter]);
        assert_eq!(press(&mut e, &[Input::Char('q')]), Effect::Nothing, "while typing, q is a letter");
        assert_eq!(e.editing.as_ref().map(|(_, t)| t.as_str()), Some("q"));
        let mut b = app();
        assert_eq!(press(&mut b, &[Input::Char('q')]), Effect::Quit, "nothing changed: q leaves at once");
        assert_eq!(press(&mut app(), &[Input::Interrupt]), Effect::Quit);
    }

    #[test]
    fn the_screen_has_two_columns_with_the_key_on_top() {
        let mut a = app();
        press(&mut a, &[Input::Down, Input::Down]);
        let screen: Vec<String> = render(&a, 100, 24).iter().map(|l| plain(l)).collect();
        assert_eq!(screen.len(), 24);
        assert!(screen.iter().all(|l| l.chars().count() <= 100));
        assert!(screen[2].starts_with("  + Create new key") && screen[2].contains("    API key:"), "{}", screen[2]);
        assert!(screen.iter().all(|l| !l.contains('\u{2502}') && !l.contains('\u{2500}')), "no table lines");
        assert!(screen[3].starts_with("  Done") && screen[3].contains("goethe_sk_id-a"), "{}", screen[3]);
        assert!(screen[5].starts_with("\u{25CF} dev   id-a"), "{}", screen[5]);
        assert!(screen.iter().any(|l| l.contains("    Name:        dev")) && screen.iter().any(|l| l.contains("    ID:          id-a")));
        assert!(screen.iter().any(|l| l.contains("[x] Every model")) && screen.iter().any(|l| l.contains("Delete key")));
        assert!(screen[23].contains("\u{2192} open"));
        let tiny: Vec<String> = render(&a, 40, 10);
        assert_eq!(tiny.len(), 10);
        assert!(tiny[0].contains("at least 64x12"));
        let narrow: Vec<String> = render(&a, 64, 16).iter().map(|l| plain(l)).collect();
        assert!(narrow[3].ends_with("goethe_sk_id-a"), "the key is never cut: {}", narrow[3]);
        let long = App::new(vec![KeyRow { token: Some(format!("goethe_sk_{}", "f".repeat(32))), ..key("id-a", "dev", None) }], Vec::new());
        let mut long = long;
        press(&mut long, &[Input::Down, Input::Down]);
        let narrow: Vec<String> = render(&long, 64, 16).iter().map(|l| plain(l)).collect();
        assert!(narrow[3].contains(&"f".repeat(32)), "a full 42-character key fits at the minimum width: {}", narrow[3]);
    }

    #[test]
    fn the_info_bar_names_the_formats_for_limits() {
        let mut a = app();
        press(&mut a, &[Input::Down, Input::Down, Input::Right, Input::Down, Input::Down, Input::Down]);
        assert_eq!(a.field(), Some(Field::Rpm));
        assert!(help(&a).contains("0 means unlimited"), "{}", help(&a));
        press(&mut a, &[Input::Down]);
        assert!(help(&a).contains("30d") && help(&a).contains("2027-01-31") && help(&a).contains("never"), "{}", help(&a));
        press(&mut a, &[Input::Enter]);
        assert!(help(&a).contains("30d") && help(&a).contains("Esc cancels"), "editing keeps the formats in view: {}", help(&a));
        let shown = plain(render(&a, 100, 24).last().unwrap());
        assert!(shown.contains("2027-01-31") && shown.contains("Esc cancels"), "the whole hint fits at 100 columns: {shown}");
        press(&mut a, &[Input::Esc, Input::Up]);
        let shown = plain(render(&a, 80, 24).last().unwrap());
        assert!(shown.contains("0 means unlimited"), "the format fits at 80 columns: {shown}");
        assert!(shown.starts_with("Rate limit: requests per minute"), "{shown}");
    }

    #[test]
    fn only_changed_lines_are_redrawn() {
        let a: Vec<String> = ["one", "two", "three"].iter().map(|s| s.to_string()).collect();
        let mut b = a.clone();
        assert_eq!(frame_update(&a, &b), "", "nothing changed: nothing is written");
        b[1] = "TWO".into();
        assert_eq!(frame_update(&a, &b), "\x1b[2;1HTWO\x1b[K");
        assert!(frame_update(&a, &b[..2]).starts_with("\x1b[H\x1b[2J"), "a new size repaints");
    }

    #[test]
    fn long_lists_scroll_with_the_cursor() {
        let models: Vec<String> = (0..40).map(|i| format!("org/model-{i:02}")).collect();
        let mut a = App::new(vec![key("id-a", "dev", None)], models);
        press(&mut a, &[Input::Down, Input::Down, Input::Right]);
        for _ in 0..39 {
            press(&mut a, &[Input::Down]);
        }
        let screen: Vec<String> = render(&a, 90, 20).iter().map(|l| plain(l)).collect();
        assert!(screen.iter().any(|l| l.contains("\u{25CF}   [ ] org/model-38")), "the hovered model is on screen: {screen:#?}");
    }
}
