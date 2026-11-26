// Byte-based progress lines for long downloads, and the learned warm-up time.

use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub const INTERVAL: Duration = Duration::from_secs(5);
pub static NET_BYTES: AtomicU64 = AtomicU64::new(0);

pub struct Counted<R>(pub R);

impl<R: Read> Read for Counted<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.0.read(buf)?;
        NET_BYTES.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }
}

pub fn dir_bytes(dir: &Path) -> u64 {
    // Every byte under a directory, partial downloads included.
    std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| match e.metadata() {
        Ok(m) if m.is_dir() => dir_bytes(&e.path()),
        Ok(m) => m.len(),
        Err(_) => 0,
    }).sum()
}

pub fn size(bytes: u64) -> String {
    match bytes {
        b if b >= 1 << 30 => format!("{:.1} GB", b as f64 / (1u64 << 30) as f64),
        b => format!("{:.0} MB", b as f64 / (1u64 << 20) as f64),
    }
}

fn span(seconds: u64) -> String {
    match seconds {
        s if s >= 5400 => format!("{} h", s.div_ceil(3600)),
        s if s >= 90 => format!("{} min", s.div_ceil(60)),
        s => format!("{s} s"),
    }
}

pub fn bar(percent: u32) -> String {
    // Twenty cells filled in eighths in the theme's main color, then the percent: one line that moves about every half percent.
    let eighths = (percent.min(100) * 160 + 50) / 100;
    let (full, part) = ((eighths / 8) as usize, (eighths % 8) as usize);
    let partial = ["", "\u{258f}", "\u{258e}", "\u{258d}", "\u{258c}", "\u{258b}", "\u{258a}", "\u{2589}"][part];
    let rest = "\u{2591}".repeat(20 - full - usize::from(!partial.is_empty()));
    let done = crate::theme::paint(&crate::theme::main(), &format!("{}{partial}", "\u{2588}".repeat(full)));
    format!("{done}{rest} {:>3}%", percent.min(100))
}

pub fn line(label: &str, done: u64, total: Option<u64>, bytes_per_s: f64) -> String {
    // "model: 1.2 GB of 4.7 GB, 85 MB/s, about 41 s left"; compact, a bar with the sizes.
    if crate::step::compact() {
        return match total {
            Some(t) if t > 0 => format!("{}  {} of {}", bar((done.min(t) * 100 / t) as u32), size(done.min(t)), size(t)),
            _ => size(done),
        };
    }
    let rate = format!("{:.0} MB/s", bytes_per_s / (1u64 << 20) as f64);
    match total {
        Some(t) if done >= t => format!("{label}: {} complete, finishing up", size(t)),
        Some(t) if bytes_per_s >= 1.0 && t > done => {
            format!("{label}: {} of {}, {rate}, about {} left", size(done), size(t), span(((t - done) as f64 / bytes_per_s).ceil() as u64))
        }
        Some(t) => format!("{label}: {} of {}, {rate}", size(done.min(t)), size(t)),
        None => format!("{label}: {}, {rate}", size(done)),
    }
}

pub fn spaced(every: Duration, running: Duration) -> Duration {
    // A plain-line cadence (5 s) widens to 8 s after the first minute, so a long download stays near one line per 8 s; a terminal's 1 s line does not change.
    if every >= INTERVAL && running > Duration::from_secs(60) {
        Duration::from_secs(8)
    } else {
        every
    }
}

pub struct Ticker {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Ticker {
    pub fn start(label: &str, total: Option<u64>, every: Duration, done: impl Fn() -> u64 + Send + 'static, say: impl Fn(String) + Send + 'static) -> Ticker {
        // One line per interval until dropped, with the rate over the last interval.
        Ticker::fetching(label, "", total, every, done, say)
    }

    pub fn fetching(label: &str, source: &str, total: Option<u64>, every: Duration, done: impl Fn() -> u64 + Send + 'static, say: impl Fn(String) + Send + 'static) -> Ticker {
        // The same, and while compact and not even 1% has arrived "Fetching from <source>" stands where the bar will be.
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        ACTIVE.fetch_add(1, Ordering::Relaxed);
        let label = label.to_string();
        let waiting = (!source.is_empty()).then(|| format!("Fetching from {source}"));
        let handle = thread::spawn(move || {
            let (mut last, mut at, began) = (done(), Instant::now(), Instant::now());
            let compact = crate::step::compact();
            let started = move |now: u64| now > total.map_or(0, |t| t / 100);
            let shown = |now: u64, rate: f64| match &waiting {
                Some(w) if compact && !started(now) => w.clone(),
                _ => line(&label, now, total, rate),
            };
            if compact {
                say(shown(last, 0.0));
            }
            while !flag.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(20).min(every));
                if at.elapsed() < spaced(every, began.elapsed()) {
                    continue;
                }
                let now = done();
                let rate = now.saturating_sub(last) as f64 / at.elapsed().as_secs_f64();
                say(shown(now, rate));
                (last, at) = (now, Instant::now());
            }
            if compact {
                say(line(&label, done(), total, 0.0));
            }
        });
        Ticker { stop, handle: Some(handle) }
    }
}

impl Ticker {
    pub fn start_with(every: Duration, value: impl Fn() -> u64 + Send + 'static, say: impl Fn(u64) + Send + 'static) -> Ticker {
        // Calls `say` with `value()` every interval until dropped.
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        ACTIVE.fetch_add(1, Ordering::Relaxed);
        let handle = thread::spawn(move || {
            let mut at = Instant::now();
            while !flag.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(20).min(every));
                if at.elapsed() >= every {
                    say(value());
                    at = Instant::now();
                }
            }
        });
        Ticker { stop, handle: Some(handle) }
    }
}

impl Drop for Ticker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        ACTIVE.fetch_sub(1, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        end_line();
    }
}

pub fn every() -> Duration {
    cadence(std::io::IsTerminal::is_terminal(&std::io::stderr()))
}

fn cadence(tty: bool) -> Duration {
    // One updating line a second on a terminal, one plain line every 5 s otherwise.
    if tty {
        Duration::from_secs(1)
    } else {
        INTERVAL
    }
}

pub struct Beat(#[allow(dead_code)] Ticker);

impl Beat {
    pub fn start(label: &str, say: impl Fn(String) + Send + 'static) -> Beat {
        // Silent while a byte counter is already talking (the counters and this beat are all counted in ACTIVE).
        Beat::every(label, every(), move |l| {
            if ACTIVE.load(Ordering::Relaxed) <= 1 && !crate::step::compact() {
                say(l);
            }
        })
    }

    pub fn every(label: &str, interval: Duration, say: impl Fn(String) + Send + 'static) -> Beat {
        // "starting the engine, 12 s" every interval until dropped, for a step with no byte count; silent while a byte counter is already talking.
        let began = Instant::now();
        let label = label.to_string();
        let value = move || began.elapsed().as_secs();
        Beat(Ticker::start_with(interval, value, move |s| say(format!("{label}, {s} s"))))
    }
}

pub fn eprint_line(text: String) {
    // Rewrites the line in place on a terminal; the Ticker ends it with a newline.
    if std::io::IsTerminal::is_terminal(&std::io::stderr()) {
        eprint!("\r\x1b[2K{text}");
        UPDATING.store(true, Ordering::Relaxed);
    } else {
        eprintln!("{text}");
    }
}

static UPDATING: AtomicBool = AtomicBool::new(false);
static ACTIVE: AtomicUsize = AtomicUsize::new(0);

pub fn end_line() {
    // Ends an updating line left on a terminal, so the next line starts clean.
    if UPDATING.swap(false, Ordering::Relaxed) {
        eprintln!();
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn the_line_has_sizes_rate_and_eta() {
        let gb = 1u64 << 30;
        let mb = 1u64 << 20;
        assert_eq!(line("model", gb, Some(3 * gb), 100.0 * mb as f64), "model: 1.0 GB of 3.0 GB, 100 MB/s, about 21 s left");
        assert_eq!(line("pack", 10 * mb, Some(12 * gb), 1.0 * mb as f64), "pack: 10 MB of 12.0 GB, 1 MB/s, about 4 h left");
        assert_eq!(line("pack", 5 * mb, None, 0.0), "pack: 5 MB, 0 MB/s");
        assert_eq!(line("pack", 9 * gb, Some(3 * gb), 0.0), "pack: 3.0 GB complete, finishing up");
        assert_eq!(line("pack", 3 * gb, Some(3 * gb), 5.0 * mb as f64), "pack: 3.0 GB complete, finishing up");
    }

    #[test]
    fn off_a_terminal_a_line_comes_every_5_seconds() {
        assert_eq!(cadence(false), Duration::from_secs(5));
        assert_eq!(cadence(true), Duration::from_secs(1), "a terminal's single line updates each second");
    }

    #[test]
    fn a_beat_says_how_long_a_step_has_run() {
        let lines = Arc::new(Mutex::new(Vec::new()));
        let sink = lines.clone();
        let beat = Beat::every("starting the engine", Duration::from_millis(60), move |l| sink.lock().unwrap().push(l));
        thread::sleep(Duration::from_millis(300));
        drop(beat);
        let seen = lines.lock().unwrap().clone();
        assert!(seen.len() >= 3 && seen[0] == "starting the engine, 0 s", "{seen:?}");
    }

    #[test]
    fn plain_lines_slow_down_after_the_first_minute_and_a_terminal_line_does_not() {
        let (s, m) = (Duration::from_secs, Duration::from_secs(61));
        assert_eq!(spaced(s(5), s(30)), s(5));
        assert_eq!(spaced(s(5), m), s(8));
        assert_eq!(spaced(s(1), m), s(1));
    }

    #[test]
    fn the_ticker_says_a_line_every_interval_and_stops_on_drop() {
        let n = Arc::new(AtomicU64::new(0));
        let lines = Arc::new(Mutex::new(Vec::new()));
        let (counter, sink) = (n.clone(), lines.clone());
        let t = Ticker::start("x", Some(10 << 20), Duration::from_millis(60), move || counter.fetch_add(1 << 20, Ordering::Relaxed), move |l| sink.lock().unwrap().push(l));
        thread::sleep(Duration::from_millis(400));
        drop(t);
        let seen = lines.lock().unwrap().len();
        assert!((3..=8).contains(&seen), "{seen}");
        thread::sleep(Duration::from_millis(150));
        assert_eq!(lines.lock().unwrap().len(), seen, "no line after the drop");
    }

    #[test]
    fn a_directory_counts_partial_files() {
        let d = std::env::temp_dir().join(format!("goethe-progress-{}", std::process::id()));
        std::fs::create_dir_all(d.join(".cache/dl")).unwrap();
        std::fs::write(d.join("a"), [0u8; 10]).unwrap();
        std::fs::write(d.join(".cache/dl/b.incomplete"), [0u8; 5]).unwrap();
        assert_eq!(dir_bytes(&d), 15);
        let _ = std::fs::remove_dir_all(&d);
    }
}
