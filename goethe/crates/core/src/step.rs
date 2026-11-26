// What goethe prints and what it keeps: key facts on screen, every underlying process in a log file.

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

const TAIL_LINES: usize = 15;
pub const SLOW: Duration = Duration::from_secs(3);

static VERBOSE: AtomicBool = AtomicBool::new(false);
static LOG: Mutex<Option<Log>> = Mutex::new(None);

struct Log {
    path: PathBuf,
    file: File,
}

fn log() -> MutexGuard<'static, Option<Log>> {
    LOG.lock().unwrap_or_else(|e| e.into_inner())
}

static COMPACT: AtomicBool = AtomicBool::new(false);

pub fn set_compact(on: bool) {
    // On a terminal, install, remove and load show only the model, progress bars and their result; the rest goes to the log.
    COMPACT.store(on && !verbose(), Ordering::Relaxed);
}

pub fn compact() -> bool {
    COMPACT.load(Ordering::Relaxed)
}

pub fn say(text: &str) {
    // A detail line on stdout, kept in the log; only the log when compact.
    log_line(text);
    if !compact() {
        println!("{text}");
    }
}

pub fn note(text: &str) {
    // A note or warning on stderr, kept in the log; only the log when compact.
    log_line(text);
    if !compact() {
        eprintln!("{text}");
    }
}

pub fn set_verbose(on: bool) {
    VERBOSE.store(on, Ordering::Relaxed);
}

pub fn verbose() -> bool {
    VERBOSE.load(Ordering::Relaxed) || std::env::var("GOETHE_VERBOSE").is_ok_and(|v| v == "1")
}

pub fn begin(home: &Path, command: &str) {
    // The log of this command: <home>/logs/<command>-<UTC time>.log, owner-only, created on the first line written.
    let stamp = chrono::Utc::now().format("%Y%m%dT%H%M%SZ");
    let name: String = command.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    log().replace(Log { path: home.join("logs").join(format!("{name}-{stamp}.log")), file: placeholder() });
}

fn placeholder() -> File {
    File::open("/dev/null").unwrap_or_else(|_| unreachable!("/dev/null opens"))
}

pub fn log_path() -> Option<PathBuf> {
    log().as_ref().filter(|l| l.path.exists()).map(|l| l.path.clone())
}

pub fn log_line(line: &str) {
    let mut guard = log();
    let Some(l) = guard.as_mut() else { return };
    if !l.path.exists() {
        if let Some(dir) = l.path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match std::fs::OpenOptions::new().create(true).append(true).mode(0o600).open(&l.path) {
            Ok(f) => l.file = f,
            Err(_) => return,
        }
    }
    let _ = writeln!(l.file, "{line}");
}

pub struct Captured {
    pub status: ExitStatus,
    pub tail: Vec<String>,
    pub error: Option<String>,
}

fn names_an_error(line: &str) -> bool {
    let low = line.to_lowercase();
    ["error", "failed", "cannot", "could not", "timed out", "refused", "no such", "not found"].iter().any(|w| low.contains(w))
}

fn pump(src: impl Read, tail: &Mutex<VecDeque<String>>, error: &Mutex<Option<String>>, echo: Option<&(dyn Fn(&str) + Sync)>) {
    // Lines (a carriage return ends one too, as progress bars use it) go to the log, to the screen with -v, and into the tail.
    let mut reader = BufReader::new(src);
    let mut buf = Vec::new();
    while reader.read_until(b'\n', &mut buf).unwrap_or(0) > 0 {
        let text = String::from_utf8_lossy(&buf).to_string();
        for part in text.split(['\n', '\r']).map(str::trim_end).filter(|p| !p.is_empty()) {
            log_line(part);
            if let Some(show) = echo {
                show(part);
            }
            if names_an_error(part) {
                error.lock().unwrap_or_else(|e| e.into_inner()).replace(part.to_string());
            }
            let mut t = tail.lock().unwrap_or_else(|e| e.into_inner());
            if t.len() == TAIL_LINES {
                t.pop_front();
            }
            t.push_back(part.to_string());
        }
        buf.clear();
    }
}

pub fn run(cmd: &mut Command) -> std::io::Result<Captured> {
    // The command with stdout and stderr captured; with -v its lines also go to stderr as they come.
    run_with(cmd, verbose().then_some(&|l: &str| eprintln!("{l}")))
}

fn run_with(cmd: &mut Command, echo: Option<&(dyn Fn(&str) + Sync)>) -> std::io::Result<Captured> {
    // Into the log, the last lines kept for a failure, each line also handed to `echo` when there is one.
    let mut child = cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    let (tail, error) = (Mutex::new(VecDeque::new()), Mutex::new(None));
    let (out, err) = (child.stdout.take(), child.stderr.take());
    let status = std::thread::scope(|s| {
        if let Some(o) = out {
            s.spawn(|| pump(o, &tail, &error, echo));
        }
        if let Some(e) = err {
            s.spawn(|| pump(e, &tail, &error, echo));
        }
        child.wait()
    })?;
    let tail = tail.into_inner().unwrap_or_else(|e| e.into_inner()).into_iter().collect();
    let error = error.into_inner().unwrap_or_else(|e| e.into_inner());
    Ok(Captured { status, tail, error })
}

pub fn failure(what: &str, c: &Captured, next: &str) -> String {
    // The step and its exit code, the last line of the output that names an error, where the whole log is, and what to do; with -v the output was already on screen.
    let code = c.status.code().map_or("killed by a signal".to_string(), |n| format!("exit {n}"));
    let mut out = format!("{what} failed ({code})");
    if !verbose() {
        if let Some(line) = headline(c) {
            out.push_str(&format!(": {line}"));
        }
    }
    if let Some(p) = log_path() {
        out.push_str(&format!("\nfull log: {}", p.display()));
    }
    out.push_str(&format!("\n{next}"));
    out
}

fn headline(c: &Captured) -> Option<String> {
    // The last line that says something went wrong, else the last line there is.
    c.error.as_ref().or(c.tail.last()).map(|l| l.trim().chars().take(200).collect())
}

pub fn took(spent: Duration) -> String {
    match spent {
        d if d >= SLOW => format!(" ({} s)", d.as_secs()),
        _ => String::new(),
    }
}

pub fn row(name: &str, rest: &str) -> String {
    // A step's result line: a name column, then the fact.
    format!("{name:<8} {rest}")
}

pub fn step<T>(name: &str, work: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    // One line with the step's result, and its time when it took over 3 s; an error is the caller's to print.
    let started = Instant::now();
    let value = work()?;
    println!("{name}{}", took(started.elapsed()));
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    static SERIAL: Mutex<()> = Mutex::new(());

    fn serial() -> MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn fake() -> Command {
        let mut c = Command::new("bash");
        c.arg("-c").arg("for i in $(seq 300); do echo \"noise $i\"; done; echo \"error: broke\" >&2; exit ${FAKE_EXIT:-0}");
        c
    }

    fn fake_lines(p: &Path) -> usize {
        // Lines of the fake process only: other tests running in parallel write their own lines to the same process-wide log.
        std::fs::read_to_string(p).unwrap().lines().filter(|l| l.contains("noise ") || l.contains("error: broke")).count()
    }

    #[test]
    fn a_noisy_process_is_captured_whole_in_a_private_log_and_the_screen_gets_a_line() {
        let _one = serial();
        let home = tempfile::tempdir().unwrap();
        begin(home.path(), "install");
        set_verbose(false);
        let shown = step("fake step", || {
            let c = run(&mut fake()).map_err(|e| e.to_string())?;
            assert!(c.status.success() && c.tail.len() == TAIL_LINES);
            Ok(c.tail.len())
        });
        assert_eq!(shown, Ok(TAIL_LINES));
        let path = log_path().expect("the log exists once something was written");
        assert_eq!(fake_lines(&path), 301, "all 300 lines and the stderr line");
        assert!(path.starts_with(home.path().join("logs")) && path.file_name().unwrap().to_string_lossy().starts_with("install-"));
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        assert_eq!(took(Duration::from_secs(2)), "");
        assert_eq!(took(Duration::from_secs(12)), " (12 s)");
    }

    #[test]
    fn a_failure_shows_the_tail_the_log_path_and_the_next_step_and_nothing_else() {
        let _one = serial();
        let home = tempfile::tempdir().unwrap();
        begin(home.path(), "install");
        let mut c = fake();
        c.env("FAKE_EXIT", "3");
        let got = run(&mut c).unwrap();
        let text = failure("installing the Python environment", &got, "run the command again");
        assert_eq!(text.lines().count(), 3, "{text}");
        assert!(text.starts_with("installing the Python environment failed (exit 3): error: broke"), "{text}");
        assert!(!text.contains("noise"), "the noise stays in the log");
        assert!(text.contains("full log: ") && text.contains(&home.path().display().to_string()) && text.ends_with("run the command again"));
    }

    #[test]
    fn the_screen_gets_the_lines_only_when_asked() {
        let _one = serial();
        let home = tempfile::tempdir().unwrap();
        begin(home.path(), "load");
        let shown = std::sync::atomic::AtomicUsize::new(0);
        let count = |_: &str| {
            shown.fetch_add(1, Ordering::Relaxed);
        };
        run_with(&mut fake(), None).unwrap();
        assert_eq!(shown.load(Ordering::Relaxed), 0, "the default shows none of the process's lines");
        run_with(&mut fake(), Some(&count)).unwrap();
        assert_eq!(shown.load(Ordering::Relaxed), 301, "-v shows every line");
        assert_eq!(fake_lines(&log_path().unwrap()), 602, "both runs are in the log");
    }

    #[test]
    fn verbose_is_a_flag_or_the_environment() {
        let _one = serial();
        set_verbose(true);
        assert!(verbose());
        set_verbose(false);
    }
}
