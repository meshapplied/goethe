// Terminal draw helpers for the interactive menu.

use crate::keys_menu::fit;
use crate::models::PickRow;
use crate::term::{paint, Color};
pub(super) use crate::status_view::{pad, strip as plain};

pub(super) const LOGO: [&str; 4] = [
    "  ___   ___   ___  _____  _  _   ___ ",
    " / __| / _ \\ | __||_   _|| || | | __|",
    "| (_ || (_) || _|   | |  | __ | | _| ",
    " \\___| \\___/ |___|  |_|  |_||_| |___|",
];
pub(super) const SITE: &str = "https://goethe.sh/";
pub(super) const GAP: &str = "   ";
pub(super) fn hl() -> String {
    // The theme's main color, bold: the highlighted entry, the cursor and the logo.
    format!("1;{}", crate::theme::main())
}

pub(super) fn mark() -> String {
    // The theme's secondary color: selection dots and text carets.
    format!("1;{}", crate::theme::secondary())
}

pub(super) fn drop_mark() -> String {
    // The theme's third color: an installed model chosen to be uninstalled.
    format!("1;{}", crate::theme::third())
}

pub(super) fn sgr(code: &str, text: &str) -> String {
    if text.is_empty() || !crate::term::color_enabled() {
        return text.to_string();
    }
    format!("\x1b[{code}m{text}\x1b[0m")
}

pub(super) fn gray(text: &str) -> String {
    paint(text, Color::Gray)
}

pub(super) fn clip(line: String, width: usize) -> String {
    // A line wider than the terminal loses its colors and is cut with an ellipsis, so nothing wraps.
    if crate::term::visible_width(&line) <= width {
        line
    } else {
        fit(&plain(&line), width).trim_end().to_string()
    }
}

pub(super) fn screen(mut lines: Vec<String>, width: usize, height: usize, footer: String) -> Vec<String> {
    // The content, a gap and the keys line right under it, padded to exactly height lines.
    lines.truncate(height.saturating_sub(2));
    lines.push(String::new());
    lines.push(format!("  {footer}"));
    lines.resize(height, String::new());
    lines.into_iter().map(|l| clip(l, width)).collect()
}
pub(super) fn example_below(out: &mut Vec<String>, indent: usize, example: &[String]) {
    // A schema example under the menu, one empty line apart, at the submenu's indent: its rows hold nothing else, so a copied curl command carries no menu text into the shell.
    if example.is_empty() {
        return;
    }
    out.push(String::new());
    out.extend(example.iter().map(|l| if l.is_empty() { String::new() } else { format!("{}{l}", " ".repeat(indent)) }));
}

pub(super) fn columns(out: &mut Vec<String>, start: usize, left_width: usize, middle: &[String], right: &[String]) {
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
pub(super) fn fine_bar(percent: u32) -> String {
    // Ten cells, each filled in eighths, so the bar moves in steps of about one percent without getting wider.
    let eighths = (percent.min(100) * 80 + 50) / 100;
    let (full, part) = ((eighths / 8) as usize, (eighths % 8) as usize);
    let partial = ["", "\u{258f}", "\u{258e}", "\u{258d}", "\u{258c}", "\u{258b}", "\u{258a}", "\u{2589}"][part];
    let used = full + usize::from(!partial.is_empty());
    format!("{}{}", sgr(&crate::theme::main(), &format!("{}{partial}", "\u{2588}".repeat(full))), gray(&"\u{2591}".repeat(10 - used)))
}
pub(super) fn brand() -> Vec<String> {
    // The small GOETHE inscription in the theme's main color, the site below it in the third.
    let mut out = vec![String::new()];
    out.extend(LOGO.iter().map(|l| format!("  {}", sgr(&crate::theme::main(), l))));
    out.push(format!("  {}", sgr(&crate::theme::third(), SITE)));
    out.push(String::new());
    out
}

pub(super) fn themes_line() -> String {
    // The themes on the main menu's keys line: number and name, the active one in its main color with a bar after it.
    let now = crate::theme::current();
    let items: Vec<String> = crate::theme::ALL
        .iter()
        .enumerate()
        .map(|(i, t)| if *t == now { format!("{}{}", sgr(&hl(), &format!("{} {}", i + 1, t.name())), sgr(&hl(), "|")) } else { gray(&format!("{} {} ", i + 1, t.name())) })
        .collect();
    format!("{}  {}", gray("theme"), items.join("  "))
}
pub(super) fn gb(v: f64) -> String {
    let v = v.max(0.0);
    if v >= 1000.0 {
        format!("{:.1} TB", v / 1000.0)
    } else {
        format!("{v:.1} GB")
    }
}

pub(super) fn name_style(r: &PickRow, here: bool) -> &'static str {
    // Installed models and the one under the cursor have bold names.
    if r.installed || here { "1" } else { "" }
}
