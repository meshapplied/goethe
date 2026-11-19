// Color themes for the goethe menu and live views: default (yellow, light purple), sea (turquoise, pink) and earth (green, brown).

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Theme {
    Default,
    Sea,
    Earth,
}

pub const ALL: [Theme; 3] = [Theme::Default, Theme::Sea, Theme::Earth];

struct Rgb(u8, u8, u8, u8);

impl Theme {
    pub fn name(self) -> &'static str {
        match self {
            Theme::Default => "default",
            Theme::Sea => "sea",
            Theme::Earth => "earth",
        }
    }

    fn colors(self) -> [Rgb; 3] {
        // Main, secondary and third: 24-bit, then the nearest of the 256 colors; all light enough for a dark background.
        match self {
            Theme::Default => [Rgb(245, 208, 110, 221), Rgb(201, 160, 255, 183), Rgb(168, 180, 255, 147)],
            Theme::Sea => [Rgb(135, 215, 255, 117), Rgb(255, 135, 200, 212), Rgb(95, 215, 192, 79)],
            Theme::Earth => [Rgb(90, 214, 90, 77), Rgb(160, 108, 60, 94), Rgb(230, 200, 138, 186)],
        }
    }

    fn parse(name: &str) -> Option<Theme> {
        ALL.into_iter().find(|t| t.name().eq_ignore_ascii_case(name.trim()))
    }
}

static CURRENT: AtomicU8 = AtomicU8::new(u8::MAX);

fn file() -> std::path::PathBuf {
    crate::paths::goethe_home().join("theme")
}

pub fn current() -> Theme {
    // GOETHE_THEME, else the theme chosen in the menu, else default.
    let at = CURRENT.load(Ordering::Relaxed);
    if let Some(t) = ALL.get(at as usize) {
        return *t;
    }
    let chosen = std::env::var("GOETHE_THEME")
        .ok()
        .and_then(|v| Theme::parse(&v))
        .or_else(|| std::fs::read_to_string(file()).ok().and_then(|v| Theme::parse(&v)))
        .unwrap_or(Theme::Default);
    CURRENT.store(ALL.iter().position(|t| *t == chosen).unwrap_or(0) as u8, Ordering::Relaxed);
    chosen
}

pub fn set(theme: Theme) {
    // Use the theme now and remember it for the next goethe.
    CURRENT.store(ALL.iter().position(|t| *t == theme).unwrap_or(0) as u8, Ordering::Relaxed);
    let _ = std::fs::write(file(), format!("{}\n", theme.name()));
}

fn truecolor() -> bool {
    std::env::var("COLORTERM").is_ok_and(|v| v.contains("truecolor") || v.contains("24bit"))
}

fn code(i: usize) -> String {
    let c = &current().colors()[i];
    if truecolor() {
        format!("38;2;{};{};{}", c.0, c.1, c.2)
    } else {
        format!("38;5;{}", c.3)
    }
}

pub fn main() -> String {
    code(0)
}

pub fn secondary() -> String {
    code(1)
}

pub fn third() -> String {
    code(2)
}

pub fn paint(sgr: &str, text: &str) -> String {
    if text.is_empty() || !crate::term::color_enabled() {
        return text.to_string();
    }
    format!("\x1b[{sgr}m{text}\x1b[0m")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_themes_with_three_distinct_light_colors_each() {
        for t in ALL {
            let c = t.colors();
            assert!(c[0].3 != c[1].3 && c[1].3 != c[2].3 && c[0].3 != c[2].3, "{}", t.name());
            for rgb in &c {
                let luma = 0.2126 * rgb.0 as f64 + 0.7152 * rgb.1 as f64 + 0.0722 * rgb.2 as f64;
                assert!(luma > 110.0, "{} is too dark for a dark terminal: {luma}", t.name());
            }
            assert_eq!(Theme::parse(&t.name().to_uppercase()), Some(t));
        }
        assert_eq!(Theme::Sea.colors()[0].3, 117, "sea keeps the turquoise used before");
        let [green, brown, _] = Theme::Earth.colors();
        assert!(green.1 as i32 - green.0.max(green.2) as i32 > 100, "earth's green is clearly green");
        assert!(brown.0 > brown.1 && brown.1 > brown.2 && brown.0 < 190, "earth's brown is red over green over blue, and not light");
    }
}
