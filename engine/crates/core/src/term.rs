// Human CLI output: aligned tables and status colors, with color only on an interactive stdout.

use std::io::{IsTerminal, Write};
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Color {
    Red,
    Green,
    Yellow,
    Bold,
}

pub fn color_enabled() -> bool {
    // Color on a terminal stdout only; NO_COLOR (any non-empty value) or TERM=dumb turn it off.
    static ON: OnceLock<bool> = OnceLock::new();
    let on = ON.get_or_init(|| {
        let no_color = std::env::var_os("NO_COLOR").is_some_and(|v| !v.is_empty());
        let dumb = std::env::var("TERM").is_ok_and(|t| t == "dumb");
        !no_color && !dumb && std::io::stdout().is_terminal()
    });
    on.to_owned()
}

pub fn paint(text: &str, color: Color) -> String {
    paint_if(color_enabled(), text, color)
}

fn paint_if(on: bool, text: &str, color: Color) -> String {
    // Wrap text in one SGR color and a reset; plain text when color is off or the text is empty.
    if !on || text.is_empty() {
        return text.to_string();
    }
    let code = match color {
        Color::Red => "31",
        Color::Green => "32",
        Color::Yellow => "33",
        Color::Bold => "1",
    };
    format!("\x1b[{code}m{text}\x1b[0m")
}

pub fn status_color(status: &str) -> Option<Color> {
    // The color of a status word, judged by its first word so "RESTARTING (attempt 2)" still matches.
    let word = status.split_whitespace().next().unwrap_or("").trim_end_matches(':').to_ascii_uppercase();
    match word.as_str() {
        "OK" | "PASS" | "RUNNING" | "ALLOWED" | "LOADED" | "READY" | "FITS" | "ENABLED" => Some(Color::Green),
        "WARN" | "WARNING" | "SKIP" | "STARTING" | "RESTARTING" | "STOPPING" | "DRAINING" | "PENDING" => {
            Some(Color::Yellow)
        }
        "FAIL" | "FAILED" | "DENIED" | "ERROR" | "CANCELLED" | "UNSUPPORTED" => Some(Color::Red),
        _ => None,
    }
}

pub fn status(text: &str) -> String {
    // A status word in its color, or unchanged when it has none.
    match status_color(text) {
        Some(c) => paint(text, c),
        None => text.to_string(),
    }
}

pub fn visible_width(s: &str) -> usize {
    // Characters a terminal shows: SGR escape sequences take no columns.
    let mut width = 0;
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for e in chars.by_ref() {
                if e.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            width += 1;
        }
    }
    width
}

pub fn commands(intro: &str, lines: &[&str]) -> String {
    // A lead-in, a blank line, then each command on its own indented line behind an arrow that marks it as a command.
    let mut out = format!("{intro}\n\n");
    for line in lines {
        out.push_str(&format!("{}\n", command(line)));
    }
    out
}

pub fn command(line: &str) -> String {
    format!("\t\u{2192} {line}")
}

pub struct Table {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Table {
    pub fn new(header: &[&str]) -> Table {
        Table { header: header.iter().map(|h| h.to_string()).collect(), rows: Vec::new() }
    }

    pub fn row(&mut self, cells: Vec<String>) {
        self.rows.push(cells);
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn render(&self) -> String {
        render_table(&self.header, &self.rows, color_enabled())
    }

    pub fn print(&self) {
        print!("{}", self.render());
    }
}

fn render_table(header: &[String], rows: &[Vec<String>], color: bool) -> String {
    // Columns padded to their widest visible line; a cell with newlines spans several lines of its row.
    let cols = header.len().max(rows.iter().map(Vec::len).max().unwrap_or(0));
    let mut widths = vec![0usize; cols];
    for cells in std::iter::once(header).chain(rows.iter().map(Vec::as_slice)) {
        for (i, cell) in cells.iter().enumerate() {
            for line in cell.lines() {
                widths[i] = widths[i].max(visible_width(line));
            }
        }
    }
    let bold: Vec<String> = header.iter().map(|h| paint_if(color, h, Color::Bold)).collect();
    let mut out = String::new();
    for cells in std::iter::once(bold.as_slice()).chain(rows.iter().map(Vec::as_slice)) {
        let split: Vec<Vec<&str>> = (0..cols).map(|i| cells.get(i).map_or(vec![], |c| c.lines().collect())).collect();
        let height = split.iter().map(Vec::len).max().unwrap_or(0).max(1);
        for n in 0..height {
            let mut line = String::new();
            for (i, parts) in split.iter().enumerate() {
                let text = parts.get(n).copied().unwrap_or("");
                line.push_str(text);
                if i + 1 < cols {
                    line.push_str(&" ".repeat(widths[i] - visible_width(text) + 2));
                }
            }
            out.push_str(line.trim_end());
            out.push('\n');
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Key {
    Up,
    Down,
    Enter,
    Cancel,
    Digit(usize),
    Other,
}

fn parse_key(bytes: &[u8]) -> Key {
    // One read from a raw terminal: arrow escape sequences, vi keys, digits, Enter, Ctrl-C, q or a lone Esc.
    match bytes {
        [0x1b, b'[' | b'O', b'A', ..] | [b'k', ..] => Key::Up,
        [0x1b, b'[' | b'O', b'B', ..] | [b'j', ..] => Key::Down,
        [b'\r' | b'\n', ..] => Key::Enter,
        [0x03, ..] | [0x1b] | [b'q', ..] => Key::Cancel,
        [d @ b'1'..=b'9', ..] => Key::Digit(usize::from(d - b'1')),
        _ => Key::Other,
    }
}

fn parse_keys(bytes: &[u8]) -> Vec<Key> {
    // Every key in one read, which can hold several when keys come fast or are pasted.
    let mut keys = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let len = if bytes[i] == 0x1b && i + 2 < bytes.len() && matches!(bytes[i + 1], b'[' | b'O') { 3 } else { 1 };
        keys.push(parse_key(&bytes[i..i + len]));
        i += len;
    }
    keys
}

fn step(options: &[String], selected: usize, key: Key) -> usize {
    // The highlighted row after one key: arrows wrap around and skip blank spacer rows, digit n jumps to the nth choice.
    let choices: Vec<usize> = (0..options.len()).filter(|&i| !options[i].is_empty()).collect();
    let at = choices.iter().position(|&i| i == selected).unwrap_or(0);
    let n = choices.len().max(1);
    match key {
        Key::Up => choices[(at + n - 1) % n],
        Key::Down => choices[(at + 1) % n],
        Key::Digit(d) if d < choices.len() => choices[d],
        _ => selected,
    }
}

pub fn ask(question: &str, default: &str) -> Result<String, String> {
    // One line typed on the terminal; an empty answer takes the default.
    eprint!("{question} ");
    std::io::stderr().flush().ok();
    let mut buf = String::new();
    std::io::stdin().read_line(&mut buf).map_err(|e| e.to_string())?;
    let answer = buf.trim();
    Ok(if answer.is_empty() { default.to_string() } else { answer.to_string() })
}

pub fn ask_secret(question: &str) -> Result<String, String> {
    // Like ask, without echoing what is typed (tokens and passwords).
    let mut saved: libc::termios = unsafe { std::mem::zeroed() };
    let quiet = unsafe { libc::tcgetattr(0, &mut saved) } == 0 && {
        let mut t = saved;
        t.c_lflag &= !libc::ECHO;
        unsafe { libc::tcsetattr(0, libc::TCSANOW, &t) == 0 }
    };
    let answer = ask(question, "");
    if quiet {
        unsafe { libc::tcsetattr(0, libc::TCSANOW, &saved) };
        eprintln!();
    }
    answer
}

struct RawMode(libc::termios);

impl RawMode {
    fn enter() -> Option<RawMode> {
        // Turn off line buffering, echo and signal keys on stdin; the saved settings come back on drop.
        let mut saved: libc::termios = unsafe { std::mem::zeroed() };
        if unsafe { libc::tcgetattr(0, &mut saved) } != 0 {
            return None;
        }
        let mut raw = saved;
        raw.c_lflag &= !(libc::ICANON | libc::ECHO | libc::ISIG);
        raw.c_cc[libc::VMIN] = 1;
        raw.c_cc[libc::VTIME] = 0;
        (unsafe { libc::tcsetattr(0, libc::TCSANOW, &raw) } == 0).then_some(RawMode(saved))
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        unsafe { libc::tcsetattr(0, libc::TCSANOW, &self.0) };
        eprint!("\x1b[?25h");
    }
}

fn terminal_width() -> usize {
    // Columns of the terminal on stderr, 80 when it cannot be read.
    let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
    if unsafe { libc::ioctl(2, libc::TIOCGWINSZ, &mut ws) } == 0 && ws.ws_col > 0 {
        usize::from(ws.ws_col)
    } else {
        80
    }
}

fn draw_menu(options: &[String], selected: usize, redraw: bool) {
    // Draw the options on stderr, the selected one in light blue behind a light blue dot; a redraw first moves back up over them.
    let mut out = String::new();
    if redraw {
        out.push_str(&format!("\x1b[{}A", options.len()));
    }
    let width = terminal_width().saturating_sub(3).max(10);
    for (i, o) in options.iter().enumerate() {
        let o: String = if o.chars().count() > width { o.chars().take(width - 1).chain(['.']).collect() } else { o.clone() };
        let line = if i == selected { format!("\x1b[1;38;5;117m\u{25CF} {o}\x1b[0m") } else { format!("  {o}") };
        out.push_str(&format!("\r\x1b[2K{line}\n"));
    }
    eprint!("{out}");
    std::io::stderr().flush().ok();
}

static HINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn select(options: &[String], default: usize) -> Result<Option<usize>, String> {
    // Arrow-key menu on a terminal: Ok(Some(i)) on Enter, Ok(None) when cancelled, Err when no terminal or raw mode.
    use std::io::Read;
    if options.is_empty() || !(std::io::stdin().is_terminal() && std::io::stderr().is_terminal()) {
        return Err("no terminal for a menu".to_string());
    }
    let raw = RawMode::enter().ok_or("could not switch the terminal to raw mode")?;
    if !HINTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        eprintln!("(up/down to move, Enter to select, q to go back)");
    }
    eprint!("\x1b[?25l");
    let mut selected = default.min(options.len() - 1);
    if options[selected].is_empty() {
        selected = step(options, selected, Key::Down);
    }
    draw_menu(options, selected, false);
    let mut buf = [0u8; 8];
    loop {
        let n = std::io::stdin().read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            drop(raw);
            return Ok(None);
        }
        for key in parse_keys(&buf[..n]) {
            match key {
                Key::Enter => {
                    drop(raw);
                    return Ok(Some(selected));
                }
                Key::Cancel => {
                    drop(raw);
                    return Ok(None);
                }
                _ => {
                    let next = step(options, selected, key);
                    if next != selected {
                        selected = next;
                        draw_menu(options, selected, true);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commands_sit_under_their_lead_in_with_a_tab() {
        assert_eq!(commands("Start it with:", &["goethe up"]), "Start it with:\n\n\t\u{2192} goethe up\n");
    }

    fn strip(s: &str) -> String {
        // Text without SGR sequences.
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
    fn columns_line_up_whatever_the_name_length_or_color() {
        let header = vec!["GPU".to_string(), "NAME".to_string(), "STATUS".to_string()];
        let rows = vec![
            vec!["gpu0".into(), "NVIDIA RTX PRO 6000 Blackwell Max-Q".into(), paint_if(true, "DENIED", Color::Red)],
            vec!["gpu10".into(), "RTX 3090".into(), paint_if(true, "ALLOWED", Color::Green)],
        ];
        let out = render_table(&header, &rows, true);
        let starts: Vec<usize> = out
            .lines()
            .map(|l| {
                let plain = strip(l);
                plain.len() - plain.split_whitespace().last().unwrap_or("").len()
            })
            .collect();
        assert!(starts.iter().all(|s| *s == starts[0]), "status column starts at the same column: {out}");
    }

    #[test]
    fn multiline_cells_stack_inside_their_column() {
        let header = vec!["GPU".to_string(), "PROCESSES".to_string(), "X".to_string()];
        let rows = vec![vec!["gpu2".into(), "PID 1  1.0 GB\nPID 22  2.0 GB".into(), "y".into()]];
        let out = render_table(&header, &rows, false);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[2].find("PID 22"), lines[1].find("PID 1"));
        assert!(lines[2].starts_with("    "), "continuation lines leave earlier columns blank");
    }

    #[test]
    fn menu_keys_move_wrap_jump_and_finish() {
        assert_eq!(parse_key(b"\x1b[A"), Key::Up);
        assert_eq!(parse_key(b"\x1bOB"), Key::Down);
        assert_eq!(parse_key(b"j"), Key::Down);
        assert_eq!(parse_key(b"\r"), Key::Enter);
        assert_eq!(parse_key(b"\x03"), Key::Cancel);
        assert_eq!(parse_key(b"\x1b"), Key::Cancel);
        assert_eq!(parse_key(b"3"), Key::Digit(2));
        assert_eq!(parse_keys(b"\x1b[B\x1b[B\r"), vec![Key::Down, Key::Down, Key::Enter]);
        let three: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        assert_eq!(step(&three, 0, Key::Up), 2);
        assert_eq!(step(&three, 2, Key::Down), 0);
        assert_eq!(step(&three, 1, Key::Digit(0)), 0);
        assert_eq!(step(&three, 1, Key::Digit(7)), 1);
        assert_eq!(step(&three, 1, Key::Other), 1);
        let spaced: Vec<String> = ["a", "", "b"].iter().map(|s| s.to_string()).collect();
        assert_eq!(step(&spaced, 0, Key::Down), 2, "the blank spacer row is skipped");
        assert_eq!(step(&spaced, 2, Key::Up), 0);
        assert_eq!(step(&spaced, 0, Key::Digit(1)), 2, "digits count choices, not rows");
    }

    #[test]
    fn widths_ignore_escape_sequences_and_status_words_map_to_colors() {
        assert_eq!(visible_width(&paint_if(true, "OK", Color::Green)), 2);
        assert_eq!(paint_if(false, "OK", Color::Green), "OK");
        assert_eq!(status_color("OK"), Some(Color::Green));
        assert_eq!(status_color("RESTARTING (attempt 2, next in 4s)"), Some(Color::Yellow));
        assert_eq!(status_color("DENIED"), Some(Color::Red));
        assert_eq!(status_color("installed"), None);
    }
}
