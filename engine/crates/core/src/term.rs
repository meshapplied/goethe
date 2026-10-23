// Human CLI output: aligned tables and status colors, with color only on an interactive stdout.

use std::io::IsTerminal;
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn widths_ignore_escape_sequences_and_status_words_map_to_colors() {
        assert_eq!(visible_width(&paint_if(true, "OK", Color::Green)), 2);
        assert_eq!(paint_if(false, "OK", Color::Green), "OK");
        assert_eq!(status_color("OK"), Some(Color::Green));
        assert_eq!(status_color("RESTARTING (attempt 2, next in 4s)"), Some(Color::Yellow));
        assert_eq!(status_color("DENIED"), Some(Color::Red));
        assert_eq!(status_color("installed"), None);
    }
}
