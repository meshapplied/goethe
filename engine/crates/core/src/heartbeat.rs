// One heartbeat step for goethed: check the link, build and send a snapshot, record the outcome, choose the next delay.

use crate::platform::{self, PlatformState};
use crate::platform_api::{ApiError, Client};
use crate::snapshot::{self, Collector};
use std::path::PathBuf;
use std::time::Duration;

pub const INTERVAL: Duration = Duration::from_secs(10);
pub const MAX_BACKOFF: Duration = Duration::from_secs(60);
pub const SEND_TIMEOUT: Duration = Duration::from_secs(5);
pub const DEBOUNCE: Duration = Duration::from_secs(1);
pub const MIN_GAP: Duration = Duration::from_secs(2);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    Start,
    Interval,
    Change,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Unlinked,
    Unreadable(String),
    Revoked,
    Sent,
    RateLimited(Duration),
    Failed(String),
}

pub struct Worker {
    home: PathBuf,
    collector: Collector,
    client: Option<Client>,
    failures: u32,
    last_loaded: Option<usize>,
    server_interval: Option<u64>,
    last_note: Option<String>,
    seed: u64,
}

impl Worker {
    pub fn new(home: PathBuf, collector: Collector) -> Worker {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1)
            ^ ((std::process::id() as u64) << 32);
        Worker { home, collector, client: None, failures: 0, last_loaded: None, server_interval: None, last_note: None, seed: seed | 1 }
    }

    pub fn step(&mut self, reason: Reason) -> (Outcome, Duration) {
        // Without credentials nothing is built or sent; a revoked link waits for a new login.
        let creds = match platform::load_credentials(&self.home) {
            Ok(Some(c)) => c,
            Ok(None) => {
                self.forget();
                return (Outcome::Unlinked, INTERVAL);
            }
            Err(e) => {
                self.forget();
                self.note(&format!("platform: credentials not used: {e}"));
                return (Outcome::Unreadable(e), MAX_BACKOFF);
            }
        };
        let mut state = platform::load_state(&self.home, &platform::link_key(&creds));
        if state.revoked_at.is_some() {
            self.client = None;
            return (Outcome::Revoked, INTERVAL);
        }
        if self.client.as_ref().map(|c| c.base()) != Some(creds.platform_url.as_str()) {
            match Client::new(&creds.platform_url, SEND_TIMEOUT) {
                Ok(c) => self.client = Some(c),
                Err(e) => return (self.failed(&mut state, e), self.backoff()),
            }
        }
        let snap = self.collector.build(&self.home, &self.reason_word(reason));
        let loaded = snap.models.loaded.len();
        let data = match snapshot::encode(&snap) {
            Ok(d) => d,
            Err(e) => return (self.failed(&mut state, e), self.backoff()),
        };
        let Some(client) = self.client.as_ref() else {
            return (Outcome::Failed("no client".into()), self.backoff());
        };
        match client.heartbeat(&creds.token, &data) {
            Ok(reply) => {
                self.last_loaded = Some(loaded);
                self.server_interval = reply.interval;
                if self.failures > 0 || state.last_ok_at.is_none() {
                    self.note(&format!("platform: heartbeat OK, linked to workspace {}", platform::clean_text(&creds.workspace.name, 64)));
                }
                self.failures = 0;
                state.last_ok_at = Some(snapshot::now_rfc3339());
                state.last_error = None;
                state.failures = 0;
                let _ = platform::save_state(&self.home, &state);
                (Outcome::Sent, self.next_interval())
            }
            Err(ApiError::Unauthorized) => {
                let now = snapshot::now_rfc3339();
                state.revoked_at = Some(now.clone());
                state.last_error = Some(ApiError::Unauthorized.to_string());
                state.last_error_at = Some(now);
                let _ = platform::save_state(&self.home, &state);
                self.note("platform: the token was revoked; heartbeats stop until goethe login");
                self.client = None;
                (Outcome::Revoked, INTERVAL)
            }
            Err(ApiError::RateLimited(wait)) => (Outcome::RateLimited(wait), wait),
            Err(e) => (self.failed(&mut state, e.to_string()), self.backoff()),
        }
    }

    fn reason_word(&self, reason: Reason) -> String {
        // load or unload by whether fewer instances are live than at the last send.
        match reason {
            Reason::Start => "start",
            Reason::Interval => "interval",
            Reason::Change => {
                let now = crate::db::open(&self.home)
                    .ok()
                    .and_then(|c| c.query_row("SELECT COUNT(*) FROM loaded_instances WHERE status IN ('STARTING', 'RUNNING', 'RESTARTING', 'DRAINING', 'STOPPING')", [], |r| r.get::<_, i64>(0)).ok());
                match (self.last_loaded, now) {
                    (Some(before), Some(n)) if (n as usize) < before => "unload",
                    _ => "load",
                }
            }
        }
        .to_string()
    }

    fn failed(&mut self, state: &mut PlatformState, e: String) -> Outcome {
        self.failures = self.failures.saturating_add(1);
        state.failures = self.failures;
        state.last_error = Some(platform::clean_text(&e, 300));
        state.last_error_at = Some(snapshot::now_rfc3339());
        let _ = platform::save_state(&self.home, state);
        if self.failures == 1 {
            self.note(&format!("platform: heartbeat failing: {e}"));
        }
        Outcome::Failed(e)
    }

    fn forget(&mut self) {
        self.client = None;
        self.failures = 0;
        self.last_loaded = None;
        self.server_interval = None;
    }

    fn note(&mut self, line: &str) {
        // One goethed.log line per change of situation, never repeated every tick.
        if self.last_note.as_deref() != Some(line) {
            crate::supervisor::append_goethed_log(&self.home, &format!("{line}\n"));
            self.last_note = Some(line.to_string());
        }
    }

    pub fn backoff(&self) -> Duration {
        backoff_for(self.failures)
    }

    fn next_interval(&mut self) -> Duration {
        // The server's interval within 5 s to 60 s, else 10 s, with up to 1 s of jitter either way.
        let base = self.server_interval.map(|s| Duration::from_secs(s.clamp(5, 60))).unwrap_or(INTERVAL);
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 7;
        self.seed ^= self.seed << 17;
        let jitter = (self.seed % 2001) as i64 - 1000;
        Duration::from_millis((base.as_millis() as i64 + jitter).max(1000) as u64)
    }
}

pub fn backoff_for(failures: u32) -> Duration {
    // 10 s, 20 s, 40 s, then 60 s for every later failure in a row.
    if failures == 0 {
        return INTERVAL;
    }
    let secs = INTERVAL.as_secs().saturating_mul(1u64 << (failures - 1).min(6));
    Duration::from_secs(secs).min(MAX_BACKOFF)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::{Credentials, Secret, Workspace};
    use crate::platform_api::mock;
    use std::os::unix::fs::PermissionsExt;

    const TOKEN: &str = "mesh_node_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    fn home() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        dir
    }

    fn creds(url: &str) -> Credentials {
        Credentials {
            platform_url: url.into(),
            node_id: "n1".into(),
            workspace: Workspace { id: "w1".into(), name: "Lab".into() },
            token: Secret::new(TOKEN),
            linked_at: "t".into(),
        }
    }

    fn key() -> String {
        platform::link_key(&creds("x"))
    }

    fn link(home: &std::path::Path, url: &str) {
        platform::save_credentials(home, &creds(url)).unwrap();
    }

    fn worker(home: &std::path::Path) -> Worker {
        Worker::new(home.to_path_buf(), Collector::new(false, "2026-11-11T10:00:00Z".into()))
    }

    #[test]
    fn without_credentials_nothing_touches_the_network() {
        let m = mock::serve(vec![(200, r#"{"ok":true}"#.into())]);
        let dir = home();
        let mut w = worker(dir.path());
        for reason in [Reason::Start, Reason::Interval, Reason::Change] {
            assert_eq!(w.step(reason), (Outcome::Unlinked, INTERVAL));
        }
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(*m.connections.lock().unwrap(), 0, "no connection without a link");
        assert!(!platform::state_path(dir.path()).exists(), "no state file without a link");
        assert!(!dir.path().join("logs").exists());
    }

    #[test]
    fn a_link_sends_the_snapshot_with_the_token() {
        let m = mock::serve(vec![(200, r#"{"ok":true,"node_id":"n1","at":"t","interval":30}"#.into()), (200, r#"{"ok":true,"interval":600}"#.into())]);
        let dir = home();
        link(dir.path(), &m.url);
        let mut w = worker(dir.path());
        let (out, next) = w.step(Reason::Start);
        assert_eq!(out, Outcome::Sent);
        assert!(next >= Duration::from_secs(29) && next <= Duration::from_secs(31), "{next:?}");
        let (_, next) = w.step(Reason::Interval);
        assert!(next <= Duration::from_secs(61), "the server interval is capped at 60 s: {next:?}");
        let seen = m.seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[0].path, "/api/device/heartbeat");
        assert_eq!(seen[0].authorization.as_deref(), Some(format!("Bearer {TOKEN}").as_str()));
        assert!(!seen[0].body.contains(TOKEN));
        let v: serde_json::Value = serde_json::from_str(&seen[0].body).unwrap();
        assert_eq!((v["schema"].as_u64(), v["reason"].as_str()), (Some(1), Some("start")));
        let st = platform::load_state(dir.path(), &key());
        assert!(st.last_ok_at.is_some() && st.last_error.is_none() && st.revoked_at.is_none());
    }

    #[test]
    fn failures_back_off_and_recover() {
        let m = mock::serve(vec![(503, "{}".into()), (500, "{}".into()), (200, r#"{"ok":true}"#.into())]);
        let dir = home();
        link(dir.path(), &m.url);
        let mut w = worker(dir.path());
        let (out, next) = w.step(Reason::Interval);
        assert!(matches!(out, Outcome::Failed(_)) && next == Duration::from_secs(10));
        let (_, next) = w.step(Reason::Interval);
        assert_eq!(next, Duration::from_secs(20));
        let st = platform::load_state(dir.path(), &key());
        assert_eq!(st.failures, 2);
        assert!(st.last_error.unwrap().contains("500"));
        assert_eq!(w.step(Reason::Interval).0, Outcome::Sent);
        assert_eq!(platform::load_state(dir.path(), &key()).failures, 0);
        assert_eq!(backoff_for(3), Duration::from_secs(40));
        assert_eq!(backoff_for(4), MAX_BACKOFF);
        assert_eq!(backoff_for(40), MAX_BACKOFF);
    }

    #[test]
    fn a_429_waits_without_counting_as_a_failure() {
        let m = mock::serve(vec![(200, r#"{"ok":true}"#.into()), (429, r#"{"error":"rate_limited"}"#.into()), (200, r#"{"ok":true}"#.into())]);
        let dir = home();
        link(dir.path(), &m.url);
        let mut w = worker(dir.path());
        assert_eq!(w.step(Reason::Start).0, Outcome::Sent);
        assert_eq!(w.step(Reason::Change), (Outcome::RateLimited(Duration::from_secs(2)), Duration::from_secs(2)));
        let st = platform::load_state(dir.path(), &key());
        assert!(st.failures == 0 && st.last_error.is_none() && st.revoked_at.is_none());
        assert_eq!(w.backoff(), INTERVAL, "no backoff growth");
        assert_eq!(w.step(Reason::Change).0, Outcome::Sent);
    }

    #[test]
    fn a_401_marks_the_link_revoked_and_stops_sending() {
        let m = mock::serve(vec![(401, r#"{"error":"unauthorized"}"#.into())]);
        let dir = home();
        link(dir.path(), &m.url);
        let mut w = worker(dir.path());
        assert_eq!(w.step(Reason::Interval).0, Outcome::Revoked);
        assert_eq!(w.step(Reason::Change).0, Outcome::Revoked);
        assert_eq!(w.step(Reason::Interval).0, Outcome::Revoked);
        assert_eq!(m.seen.lock().unwrap().len(), 1, "nothing is sent after a 401");
        assert!(platform::load_state(dir.path(), &key()).revoked_at.is_some());
        platform::remove_credentials(dir.path()).unwrap();
        assert_eq!(w.step(Reason::Interval).0, Outcome::Unlinked);
    }

    #[test]
    fn unreadable_credentials_are_not_used() {
        let m = mock::serve(vec![(200, r#"{"ok":true}"#.into())]);
        let dir = home();
        link(dir.path(), &m.url);
        std::fs::set_permissions(platform::credentials_path(dir.path()), std::fs::Permissions::from_mode(0o640)).unwrap();
        let mut w = worker(dir.path());
        assert!(matches!(w.step(Reason::Start).0, Outcome::Unreadable(e) if e.contains("chmod 600")));
        assert_eq!(*m.connections.lock().unwrap(), 0);
    }
}
