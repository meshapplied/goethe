// Error codes and ask's outcomes: stable labels (cdn/install.sh writes the INSTALL_ ones itself; a test keeps both equal), the log at <home>/errors.log, and the rows a linked node sends.

use serde::Serialize;
use std::fs;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

const FILE: &str = "errors.log";
const MAX_BYTES: usize = 64 * 1024;
const MAX_DETAIL: usize = 120;
const REPEAT_S: i64 = 86_400;
const SEND_MAX: usize = 20;
const SENT_KEY: &str = "errors_sent_t";

static RECORDED: AtomicBool = AtomicBool::new(false);

macro_rules! codes {
    ($($v:ident = $l:literal),* $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Code { $($v),* }
        impl Code {
            pub const ALL: &'static [Code] = &[$(Code::$v),*];
            pub fn label(self) -> &'static str { match self { $(Code::$v => $l),* } }
        }
    };
}

codes! {
    InstallArgs = "INSTALL_ARGS",
    InstallOs = "INSTALL_OS",
    InstallArch = "INSTALL_ARCH",
    InstallGlibc = "INSTALL_GLIBC",
    InstallTools = "INSTALL_TOOLS",
    InstallUrl = "INSTALL_URL",
    InstallNoKey = "INSTALL_NO_KEY",
    InstallDownload = "INSTALL_DOWNLOAD",
    InstallSignature = "INSTALL_SIGNATURE",
    InstallNoBuild = "INSTALL_NO_BUILD",
    InstallBinary = "INSTALL_BINARY",
    ModelDownload = "MODEL_DOWNLOAD",
    ModelFiles = "MODEL_FILES",
    ModelFit = "MODEL_FIT",
    ModelPack = "MODEL_PACK",
    EnvDeps = "ENV_DEPS",
    EnvFix = "ENV_FIX",
    CmdFailed = "CMD_FAILED",
    DoctorHardware = "DOCTOR_HARDWARE",
    DoctorDriver = "DOCTOR_DRIVER",
    DoctorConsole = "DOCTOR_CONSOLE",
    DoctorGlibc = "DOCTOR_GLIBC",
    DoctorNginx = "DOCTOR_NGINX",
    DoctorGoetheHome = "DOCTOR_GOETHE_HOME",
    DoctorDisk = "DOCTOR_DISK",
    DoctorRam = "DOCTOR_RAM",
    DoctorStateDb = "DOCTOR_STATE_DB",
    DoctorEnvironment = "DOCTOR_ENVIRONMENT",
    DoctorCloud = "DOCTOR_CLOUD",
    DoctorPyEnv = "DOCTOR_PY_ENV",
    DoctorPorts = "DOCTOR_PORTS",
    DoctorGoethed = "DOCTOR_GOETHED",
    DoctorGateway = "DOCTOR_GATEWAY",
    DoctorNetwork = "DOCTOR_NETWORK",
    DoctorTls = "DOCTOR_TLS",
    DoctorHuggingface = "DOCTOR_HUGGINGFACE",
    DoctorRegistry = "DOCTOR_REGISTRY",
    DoctorNvidiaDriver = "DOCTOR_NVIDIA_DRIVER",
    DoctorCompiler = "DOCTOR_COMPILER",
    DoctorFfmpeg = "DOCTOR_FFMPEG",
    DoctorAskModel = "DOCTOR_ASK_MODEL",
    DoctorLlamaCpp = "DOCTOR_LLAMA_CPP",
    RemoteActionUnknown = "REMOTE_ACTION_UNKNOWN",
    RemoteArgs = "REMOTE_ARGS",
    RemoteWrongNode = "REMOTE_WRONG_NODE",
    RemoteSig = "REMOTE_SIG",
    RemoteKeyUnknown = "REMOTE_KEY_UNKNOWN",
    RemoteKeyCap = "REMOTE_KEY_CAP",
    RemotePolicy = "REMOTE_POLICY",
    RemoteExpired = "REMOTE_EXPIRED",
    RemoteReplay = "REMOTE_REPLAY",
    RemoteBusy = "REMOTE_BUSY",
    AskRan = "ASK_RAN",
    AskChanged = "ASK_CHANGED",
    AskDropped = "ASK_DROPPED",
}

impl Code {
    pub fn for_check(name: &str) -> Option<Code> {
        // A doctor row's code is its fixed name in capitals.
        let want: String = name.chars().map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' }).collect();
        Code::ALL.iter().copied().find(|c| c.label().strip_prefix("DOCTOR_") == Some(want.as_str()))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Event {
    pub t: i64,
    pub code: String,
    pub v: String,
    pub detail: String,
}

pub fn record(code: Code, detail: &str) {
    // Best effort and never fatal: a failure that cannot be logged is still reported to the user as before.
    record_at(&crate::paths::goethe_home(), code, detail);
}

pub fn record_at(home: &Path, code: Code, detail: &str) {
    // The same row in a named home, for goethed acting on a command.
    RECORDED.store(true, Ordering::Relaxed);
    let t = chrono::Utc::now().timestamp();
    let _ = append(home, t, code.label(), env!("CARGO_PKG_VERSION"), &scrub(detail));
}

pub fn coded(code: Code, msg: String) -> String {
    // Logs the failure under its code and hands the message back, so a site reads `Err(coded(Code::X, format!(..)))`.
    record(code, &msg);
    msg
}

pub fn recorded() -> bool {
    RECORDED.load(Ordering::Relaxed)
}

fn append(home: &Path, t: i64, code: &str, v: &str, detail: &str) -> std::io::Result<()> {
    // The same failure (code and detail) once a day, and one row per code and second, the key the console and the send cursor use; over the size cap the newest half stays.
    let path = home.join(FILE);
    let old = fs::read_to_string(&path).unwrap_or_default();
    if rows(&old).any(|e| e.code == code && (e.t == t || (e.detail == detail && t - e.t < REPEAT_S))) {
        return Ok(());
    }
    fs::create_dir_all(home)?;
    let mut line = format!("{t}\t{code}\t{v}\t{detail}\n");
    if old.len() + line.len() > MAX_BYTES {
        let keep = old.len() / 2;
        let from = old.as_bytes()[keep..].iter().position(|&b| b == b'\n').map_or(old.len(), |i| keep + i + 1);
        line = format!("{}{line}", &old[from..]);
        crate::paths::write_private(&path, line.as_bytes())?;
        return Ok(());
    }
    fs::OpenOptions::new().create(true).append(true).mode(0o600).open(path)?.write_all(line.as_bytes())
}

fn rows(text: &str) -> impl Iterator<Item = Event> + '_ {
    // Lines that do not parse (a half-written one) are skipped.
    text.lines().filter_map(|l| {
        let mut f = l.splitn(4, '\t');
        Some(Event { t: f.next()?.parse().ok()?, code: f.next()?.to_string(), v: f.next()?.to_string(), detail: f.next()?.to_string() })
    })
}

fn scrub(detail: &str) -> String {
    // First line only; no token-looking word, address, URL, email or path (a model id such as org/name stays); at most MAX_DETAIL characters.
    let words: Vec<&str> = detail.lines().next().unwrap_or("").split_whitespace().map(redact).collect();
    words.join(" ").chars().filter(|c| !c.is_control()).take(MAX_DETAIL).collect()
}

fn redact(word: &str) -> &str {
    let bare = word.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != ':' && c != '.');
    let tail = word.rsplit('=').next().unwrap_or(word).trim_start_matches(['(', '[', '"', '\'']);
    if ["goethe_sk_", "mesh_", "msh_", "hf_", "sk-", "ghp_"].iter().any(|p| bare.starts_with(p)) {
        "<secret>"
    } else if bare.parse::<std::net::IpAddr>().is_ok() || bare.trim_end_matches(':').parse::<std::net::IpAddr>().is_ok() {
        "<ip>"
    } else if word.contains("://") || word.contains('@') {
        "<redacted>"
    } else if tail.starts_with(['/', '~', '.']) || tail.matches('/').count() > 1 {
        "<path>"
    } else {
        word
    }
}

pub fn pending(home: &Path, after: &(i64, String)) -> Vec<Event> {
    // The oldest unsent rows, at most SEND_MAX per heartbeat, ordered by (time, code): a row's key, since a code is logged at most once a second, so a second with many rows (one doctor run) is never cut.
    let text = fs::read_to_string(home.join(FILE)).unwrap_or_default();
    let mut rows: Vec<Event> = rows(&text).filter(|e| (e.t, &e.code) > (after.0, &after.1)).collect();
    rows.sort_by(|a, b| (a.t, &a.code).cmp(&(b.t, &b.code)));
    rows.truncate(SEND_MAX);
    rows
}

pub fn sent_to(con: &rusqlite::Connection) -> (i64, String) {
    crate::db::setting(con, SENT_KEY).and_then(|v| v.split_once(' ').and_then(|(t, code)| Some((t.parse().ok()?, code.to_string())))).unwrap_or((0, String::new()))
}

pub fn mark_sent(home: &Path, last: &Event) {
    // The console has every row up to and including `last`.
    if let Ok(con) = crate::db::open(home) {
        let _ = crate::db::set_setting(&con, SENT_KEY, &format!("{} {}", last.t, last.code));
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn labels_are_unique_and_shout() {
        let mut seen = std::collections::HashSet::new();
        for c in Code::ALL {
            assert!(seen.insert(c.label()), "{} twice", c.label());
            assert!(c.label().chars().all(|ch| ch.is_ascii_uppercase() || ch == '_'), "{}", c.label());
        }
    }

    #[test]
    fn install_sh_uses_exactly_the_install_labels() {
        let sh = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../cdn/install.sh")).unwrap();
        for c in Code::ALL.iter().filter(|c| c.label().starts_with("INSTALL_")) {
            assert!(sh.contains(&format!("die {} ", c.label())), "{} unused in install.sh", c.label());
        }
        for w in sh.split_whitespace().filter(|w| w.starts_with("INSTALL_") && w.chars().all(|c| c.is_ascii_uppercase() || c == '_')) {
            assert!(Code::ALL.iter().any(|c| c.label() == w), "{w} is not a code");
        }
    }

    #[test]
    fn every_doctor_row_has_a_code() {
        for name in crate::doctor::check_names() {
            assert!(Code::for_check(name).is_some(), "doctor row {name} has no code");
        }
    }

    #[test]
    fn the_same_failure_is_logged_once_a_day_and_different_ones_each() {
        let h = tempfile::tempdir().unwrap();
        append(h.path(), 1000, "CMD_FAILED", "1.0", "login").unwrap();
        append(h.path(), 1500, "CMD_FAILED", "1.0", "install").unwrap();
        append(h.path(), 1500, "CMD_FAILED", "1.0", "load").unwrap();
        append(h.path(), 1600, "CMD_FAILED", "1.0", "login").unwrap();
        append(h.path(), 1600, "ENV_FIX", "1.0", "x").unwrap();
        append(h.path(), 1000 + REPEAT_S, "CMD_FAILED", "1.0", "login").unwrap();
        let text = fs::read_to_string(h.path().join(FILE)).unwrap();
        let got: Vec<(i64, String)> = rows(&text).map(|e| (e.t, e.detail)).collect();
        assert_eq!(got, [(1000, "login".into()), (1500, "install".into()), (1600, "x".into()), (1000 + REPEAT_S, "login".into())]);
    }

    #[test]
    fn the_log_stays_under_its_cap_and_keeps_the_newest() {
        let h = tempfile::tempdir().unwrap();
        for i in 0..2000 {
            append(h.path(), i * REPEAT_S, "ENV_DEPS", "1.0", &"x".repeat(100)).unwrap();
        }
        let text = fs::read_to_string(h.path().join(FILE)).unwrap();
        assert!(text.len() <= MAX_BYTES);
        assert_eq!(rows(&text).last().unwrap().t, 1999 * REPEAT_S);
    }

    #[test]
    fn the_cap_cuts_on_a_line_even_inside_non_ascii_text() {
        let h = tempfile::tempdir().unwrap();
        for i in 0..2000 {
            append(h.path(), i * REPEAT_S, "ENV_DEPS", "1.0", &"\u{e9}".repeat(50)).unwrap();
        }
        let text = fs::read_to_string(h.path().join(FILE)).unwrap();
        assert!(text.len() <= MAX_BYTES && rows(&text).all(|e| e.detail.chars().all(|c| c == '\u{e9}')));
    }

    #[test]
    fn pending_sends_oldest_first_after_the_cursor() {
        let h = tempfile::tempdir().unwrap();
        for i in 1..=25 {
            append(h.path(), i, &format!("C{i}"), "1.0", "d").unwrap();
        }
        let got = pending(h.path(), &(3, "C3".into()));
        assert_eq!((got.len(), got[0].t, got[SEND_MAX - 1].t), (SEND_MAX, 4, 23));
    }

    #[test]
    fn a_second_with_more_rows_than_one_heartbeat_takes_is_sent_whole() {
        let h = tempfile::tempdir().unwrap();
        for i in 0..SEND_MAX + 5 {
            append(h.path(), 7, &format!("DOCTOR_{i:02}"), "1.0", "d").unwrap();
        }
        let first = pending(h.path(), &(0, String::new()));
        let last = first.last().unwrap();
        let rest = pending(h.path(), &(last.t, last.code.clone()));
        assert_eq!((first.len(), rest.len()), (SEND_MAX, 5));
    }

    #[test]
    fn detail_loses_paths_tokens_addresses_urls_emails_and_newlines() {
        let out = scrub("failed in /srv/g/models and (~/x) dir=/mnt/a/b with mesh_node_abc on 10.1.2.3: https://h.example/p?q=1 a@b.co org/name ok\nsecond line");
        assert_eq!(out, "failed in <path> and <path> <path> with <secret> on <ip> <redacted> <redacted> org/name ok");
        assert_eq!(scrub("key goethe_sk_0123abcd and msh_ak_xyz"), "key <secret> and <secret>");
    }
}
