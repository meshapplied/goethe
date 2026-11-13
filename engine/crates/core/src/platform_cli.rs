// goethe login, goethe logout and the platform lines of goethe status and the home screen.

use crate::paths::goethe_home;
use crate::platform::{self, Credentials, Secret};
use crate::platform_api::{Approved, ApiError, Client, Poll, Started};
use crate::term::{command, paint, Color};
use std::io::BufRead;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const LOGIN_TIMEOUT: Duration = Duration::from_secs(15 * 60);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
static CANCELLED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, PartialEq, Eq)]
pub enum WaitError {
    Cancelled,
    Failed(String),
}

extern "C" fn on_sigint(_: libc::c_int) {
    // First Ctrl-C asks the login to stop; the default action comes back so a second one ends goethe at once.
    CANCELLED.store(true, Ordering::SeqCst);
    unsafe {
        libc::signal(libc::SIGINT, libc::SIG_DFL);
    }
}

struct CtrlC {
    previous: libc::sighandler_t,
}

impl CtrlC {
    fn catch() -> CtrlC {
        CANCELLED.store(false, Ordering::SeqCst);
        let handler = on_sigint as extern "C" fn(libc::c_int) as libc::sighandler_t;
        CtrlC { previous: unsafe { libc::signal(libc::SIGINT, handler) } }
    }
}

impl Drop for CtrlC {
    fn drop(&mut self) {
        unsafe {
            libc::signal(libc::SIGINT, self.previous);
        }
    }
}

fn sleep_unless_cancelled(d: Duration) -> bool {
    // Sleep in short slices; false as soon as Ctrl-C was pressed.
    let end = Instant::now() + d;
    while Instant::now() < end {
        if CANCELLED.load(Ordering::SeqCst) {
            return false;
        }
        std::thread::sleep((end - Instant::now()).min(Duration::from_millis(100)));
    }
    !CANCELLED.load(Ordering::SeqCst)
}

pub fn wait_for_approval(
    started: &Started,
    poll: &mut dyn FnMut() -> Result<Poll, ApiError>,
    sleep: &mut dyn FnMut(Duration) -> bool,
    on_blip: &mut dyn FnMut(&ApiError),
) -> Result<Approved, WaitError> {
    // RFC 8628 polling: wait interval between polls, 5 s more on slow_down, retry network blips, stop at expiry or 15 min.
    let deadline = Duration::from_secs(started.expires_in).min(LOGIN_TIMEOUT);
    let mut interval = Duration::from_secs(started.interval);
    let mut waited = Duration::ZERO;
    loop {
        if !sleep(interval) {
            return Err(WaitError::Cancelled);
        }
        waited += interval;
        if waited >= deadline {
            return Err(WaitError::Failed("the code expired before it was approved; run goethe login again".into()));
        }
        let t0 = Instant::now();
        let reply = poll();
        waited += t0.elapsed();
        match reply {
            Ok(Poll::Pending) => {}
            Ok(Poll::SlowDown) => interval = (interval + Duration::from_secs(5)).min(Duration::from_secs(60)),
            Ok(Poll::Expired) => return Err(WaitError::Failed("the code expired before it was approved; run goethe login again".into())),
            Ok(Poll::Denied) => return Err(WaitError::Failed("the login was denied on the platform; nothing was linked".into())),
            Ok(Poll::Approved(a)) => return Ok(a),
            Err(ApiError::Status(404, code)) if code == "invalid_device_code" => {
                return Err(WaitError::Failed("the platform no longer knows this code (already used or unknown); run goethe login again".into()))
            }
            Err(e @ ApiError::Status(404, _)) => on_blip(&e),
            Err(e) if e.retryable() => on_blip(&e),
            Err(e) => return Err(WaitError::Failed(e.to_string())),
        }
    }
}

fn with_retries<T>(mut f: impl FnMut() -> Result<T, ApiError>) -> Result<T, ApiError> {
    // Three tries 2 s and 4 s apart for network blips and server errors.
    let mut delay = Duration::from_secs(2);
    for _ in 0..2 {
        match f() {
            Err(e) if e.retryable() => {
                if !sleep_unless_cancelled(delay) {
                    return Err(e);
                }
                delay *= 2;
            }
            other => return other,
        }
    }
    f()
}

fn read_authkey(raw: &str) -> Result<Secret, String> {
    // The auth key as given, or from the first line of stdin with --authkey -, so it need not sit in the process list.
    let key = if raw == "-" {
        let mut line = String::new();
        std::io::stdin().lock().read_line(&mut line).map_err(|e| format!("cannot read the auth key from stdin: {e}"))?;
        line.trim().to_string()
    } else {
        raw.trim().to_string()
    };
    if !key.starts_with("msh_ak_") || key.len() < 12 || key.len() > 512 || !key.chars().all(|c| c.is_ascii_graphic()) {
        return Err("that is not an auth key: they start with msh_ak_ (create one on the platform under Connect a node)".into());
    }
    Ok(Secret::new(key))
}

fn check_home_writable(home: &Path) -> Result<(), String> {
    // Fail before contacting the platform when the credentials could not be stored safely.
    platform::check_home(home)?;
    if home.exists() && !crate::paths::is_writable(home) {
        return Err(format!("{} is not writable", home.display()));
    }
    Ok(())
}

fn existing_link(home: &Path) -> Option<Credentials> {
    match platform::load_credentials(home) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("note: the saved link is not used ({e}); logging in replaces it");
            None
        }
    }
}

fn logout_remote(c: &Credentials, timeout: Duration) -> Result<(), String> {
    Client::new(&c.platform_url, timeout).and_then(|client| client.logout(&c.token).map_err(|e| e.to_string()))
}

fn is_revoked(home: &Path, c: &Credentials) -> bool {
    platform::load_state(home, &platform::link_key(c)).revoked_at.is_some()
}

fn relink_intro(workspace: &str, revoked: bool) -> String {
    if revoked {
        format!("The previous link (workspace {workspace}) was revoked or expired on the platform; linking again.")
    } else {
        format!("This node is linked to workspace {workspace}; logging in again replaces that link.")
    }
}

fn replace_old_link(home: &Path, old: Option<Credentials>) -> Result<(), String> {
    // A new login replaces the old link: log a live old node out (best effort; a revoked one is already gone), then drop its credentials.
    let Some(old) = old.filter(|o| !is_revoked(home, o)) else {
        return platform::remove_credentials(home).map(|_| ());
    };
    let ws = platform::clean_text(&old.workspace.name, 64);
    match logout_remote(&old, REQUEST_TIMEOUT) {
        Ok(()) => println!("Logged out the previous link (workspace {ws})."),
        Err(e) => println!("Could not log out the previous link (workspace {ws}): {e}. It will show offline on the platform."),
    }
    platform::remove_credentials(home).map(|_| ())
}

fn finish(home: &Path, approved: Approved, url: &str) -> Result<(), String> {
    // Store the credentials, tell goethed, and report the first heartbeat when goethed is up.
    let creds = approved.into_credentials(url);
    platform::save_credentials(home, &creds)?;
    let ws = platform::clean_text(&creds.workspace.name, 64);
    println!("{} this node to workspace {ws} on {url}.", paint("Linked", Color::Green));
    if !platform::poke_daemon() {
        if crate::service::goethed_running() {
            println!("goethed is running an older version without platform support; it starts sending heartbeats after goethe update.");
        } else {
            println!("goethed is not running; it sends heartbeats once it starts:\n\n{}", command("goethe up"));
        }
        return Ok(());
    }
    let end = Instant::now() + Duration::from_secs(6);
    while Instant::now() < end {
        let st = platform::load_state(home, &platform::link_key(&creds));
        if st.revoked_at.is_some() {
            return Err("the platform refused the new token right away; run goethe login again".into());
        }
        if st.last_ok_at.is_some() {
            println!("First heartbeat: {}", paint("OK", Color::Green));
            return Ok(());
        }
        if let Some(e) = st.last_error {
            println!("First heartbeat failed ({e}); goethed keeps trying. goethe status shows how it goes.");
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    println!("goethed sends the first heartbeat shortly; goethe status shows it.");
    Ok(())
}

pub fn login(platform_url: Option<&str>, authkey: Option<&str>) -> Result<(), String> {
    // goethe login: the device flow in a browser, or an auth key for machines without one.
    let home = goethe_home();
    let url = match platform_url {
        Some(u) => platform::check_url(u)?,
        None => platform::platform_url()?,
    };
    check_home_writable(&home)?;
    let key = authkey.map(read_authkey).transpose()?;
    let old = existing_link(&home);
    if let Some(o) = &old {
        println!("{}", relink_intro(&platform::clean_text(&o.workspace.name, 64), is_revoked(&home, o)));
    }
    let client = Client::new(&url, REQUEST_TIMEOUT)?;
    let (hostname, fingerprint) = (platform::hostname(), platform::fingerprint());
    let _ctrl_c = CtrlC::catch();
    if let Some(key) = key {
        replace_old_link(&home, old)?;
        let approved = with_retries(|| client.enroll(&key, &hostname, &fingerprint)).map_err(|e| match e {
            ApiError::Unauthorized => "the platform did not accept the auth key (wrong, used or expired)".to_string(),
            e => format!("{e}; nothing was linked"),
        })?;
        return finish(&home, approved, &url);
    }
    let started = with_retries(|| client.auth_start(&hostname, &fingerprint)).map_err(|e| format!("{e}; nothing was changed"))?;
    replace_old_link(&home, old)?;
    println!("\nTo link this node ({hostname}), open:\n\n{}\n\nand check that the page shows the code {}.\n", command(&started.verification_uri_complete), paint(&started.user_code, Color::Bold));
    println!("Waiting for approval (Ctrl-C cancels)...");
    let mut blips = 0u32;
    let result = wait_for_approval(
        &started,
        &mut || client.auth_poll(&started.device_code),
        &mut sleep_unless_cancelled,
        &mut |e| {
            blips += 1;
            if blips == 1 {
                eprintln!("note: {e}; still trying");
            }
        },
    );
    match result {
        Ok(approved) => finish(&home, approved, &url),
        Err(WaitError::Cancelled) => Err("login cancelled; this node is not linked".into()),
        Err(WaitError::Failed(e)) => Err(e),
    }
}

pub fn logout() -> Result<(), String> {
    // goethe logout: tell the platform when it answers, and always remove the local link.
    let home = goethe_home();
    let creds = match platform::load_credentials(&home) {
        Ok(Some(c)) => Some(c),
        Ok(None) => {
            if platform::remove_credentials(&home)? {
                println!("Removed the local link.");
            } else {
                println!("This node is not linked to the platform.");
            }
            return Ok(());
        }
        Err(e) => {
            eprintln!("note: {e}");
            None
        }
    };
    let remote = match &creds {
        Some(c) => logout_remote(c, REQUEST_TIMEOUT),
        None => Err("the credentials could not be read".into()),
    };
    platform::remove_credentials(&home)?;
    platform::poke_daemon();
    let ws = creds.as_ref().map(|c| platform::clean_text(&c.workspace.name, 64)).unwrap_or_default();
    match remote {
        Ok(()) => println!("Logged out: this node was removed from workspace {ws}."),
        Err(e) => println!(
            "Removed the local link, but could not tell the platform ({e}).\nThe node will show offline there until you remove it on the platform."
        ),
    }
    Ok(())
}

pub fn parse_lifetime(raw: &str) -> Result<Option<u64>, String> {
    // "never", or a number with h, d or w (hours, days, weeks); at least one hour.
    let s = raw.trim().to_ascii_lowercase();
    if s == "never" {
        return Ok(None);
    }
    let (num, unit) = s.split_at(s.find(|c: char| !c.is_ascii_digit()).unwrap_or(s.len()));
    let n: u64 = num.parse().map_err(|_| format!("{raw:?}: use never, or a number with h, d or w (for example 30d)"))?;
    let secs = match unit {
        "h" => n * 3600,
        "d" => n * 86_400,
        "w" => n * 7 * 86_400,
        _ => return Err(format!("{raw:?}: use never, or a number with h, d or w (for example 30d)")),
    };
    if secs < 3600 {
        return Err("the shortest lifetime is 1h".into());
    }
    Ok(Some(secs))
}

pub fn token_lifetime(raw: &str) -> Result<(), String> {
    // Hidden goethe token-lifetime: limit how long this node's login lasts (the workspace's setting still applies).
    let ttl = parse_lifetime(raw)?;
    let home = goethe_home();
    let c = platform::load_credentials(&home)?.ok_or("this node is not logged in (goethe login)")?;
    let client = Client::new(&c.platform_url, REQUEST_TIMEOUT)?;
    let expires = client.token_lifetime(&c.token, ttl).map_err(|e| match e {
        ApiError::Unauthorized => "the platform did not accept this node's login (revoked or expired); run goethe login".to_string(),
        e => e.to_string(),
    })?;
    match expires {
        Some(at) => println!("This node's login expires at {at}."),
        None => println!("This node's login does not expire."),
    }
    Ok(())
}

pub fn uninstall_logout(home: &Path) {
    // Best effort for goethe uninstall: log the node out quickly; the credentials go with the home.
    if let Ok(Some(c)) = platform::load_credentials(home) {
        match logout_remote(&c, Duration::from_secs(5)) {
            Ok(()) => println!("logged this node out of the platform"),
            Err(e) => println!("could not log this node out of the platform ({e}); it will show offline there"),
        }
    }
    let _ = platform::remove_credentials(home);
}

pub fn is_linked(home: &Path) -> bool {
    platform::credentials_path(home).exists() || home.join("credentials").exists()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Link {
    NotLinked,
    Unreadable(String),
    Linked { workspace: String, revoked: bool, heartbeat: String, ok: bool },
}

pub fn link_state(home: &Path, daemon_running: bool) -> Link {
    // The saved link and the last heartbeat goethed recorded for it.
    let creds = match platform::load_credentials(home) {
        Ok(Some(c)) => c,
        Ok(None) => return Link::NotLinked,
        Err(e) => return Link::Unreadable(e),
    };
    let st = platform::load_state(home, &platform::link_key(&creds));
    let workspace = platform::clean_text(&creds.workspace.name, 64);
    let (heartbeat, ok) = heartbeat_text(&st, daemon_running, chrono::Utc::now());
    Link::Linked { workspace, revoked: st.revoked_at.is_some(), heartbeat, ok }
}

fn age(ts: &str, now: chrono::DateTime<chrono::Utc>) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(ts).ok().map(|t| (now - t.with_timezone(&chrono::Utc)).num_seconds().max(0))
}

fn ago(secs: i64) -> String {
    match secs {
        s if s < 120 => format!("{s} s ago"),
        s if s < 7200 => format!("{} min ago", s / 60),
        s => format!("{} h ago", s / 3600),
    }
}

pub fn heartbeat_text(st: &platform::PlatformState, daemon_running: bool, now: chrono::DateTime<chrono::Utc>) -> (String, bool) {
    // OK with its age, failing with the reason, or why none is sent.
    if st.revoked_at.is_some() {
        return ("stopped: the platform revoked this node".into(), false);
    }
    let ok_age = st.last_ok_at.as_deref().and_then(|t| age(t, now));
    let err_age = st.last_error_at.as_deref().and_then(|t| age(t, now));
    let failing = st.last_error.is_some() && err_age.is_some_and(|e| ok_age.is_none_or(|o| e <= o));
    if !daemon_running {
        let last = ok_age.map(|a| format!(", last OK {}", ago(a))).unwrap_or_default();
        return (format!("not sent, goethed is not running (goethe up){last}"), false);
    }
    match (failing, ok_age) {
        (true, _) => (format!("failing: {}", st.last_error.clone().unwrap_or_default()), false),
        (false, Some(a)) => (format!("OK ({})", ago(a)), true),
        _ => ("waiting for the first one".into(), false),
    }
}

pub fn status_lines(home: &Path, daemon_running: bool) -> Vec<String> {
    // The lines goethe status prints under the API.
    match link_state(home, daemon_running) {
        Link::NotLinked => vec![format!("Platform: Not linked ({})", "goethe login")],
        Link::Unreadable(e) => vec![format!("Platform: {} ({e})", paint("credentials not usable", Color::Red))],
        Link::Linked { revoked: true, .. } => vec![format!("Platform: {} (goethe login)", paint("Revoked or expired", Color::Red))],
        Link::Linked { workspace, heartbeat, ok, .. } => vec![
            format!("Platform: {} (workspace {workspace})", paint("Linked", Color::Green)),
            format!("Heartbeat: {}", if ok { paint(&heartbeat, Color::Green) } else { paint(&heartbeat, Color::Yellow) }),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_lifetimes_parse_never_hours_days_weeks() {
        assert_eq!(parse_lifetime("never").unwrap(), None);
        assert_eq!(parse_lifetime("2h").unwrap(), Some(7200));
        assert_eq!(parse_lifetime("30d").unwrap(), Some(30 * 86_400));
        assert_eq!(parse_lifetime("1W").unwrap(), Some(7 * 86_400));
        for bad in ["0h", "30", "30m", "d", "", "-1d"] {
            assert!(parse_lifetime(bad).is_err(), "{bad}");
        }
    }
    use crate::platform::{PlatformState, Workspace};
    use crate::platform_api::mock;

    fn started(interval: u64, expires: u64) -> Started {
        Started { device_code: Secret::new("dc"), user_code: "ABCD-1234".into(), verification_uri_complete: "https://p/device".into(), expires_in: expires, interval }
    }

    fn approved() -> Approved {
        Approved { node_id: "n1".into(), node_token: Secret::new("mesh_node_cccccccccccccccc"), workspace: Workspace { id: "w".into(), name: "Lab".into() } }
    }

    fn run(replies: Vec<Result<Poll, ApiError>>, s: &Started, cancel_after: Option<usize>) -> (Result<Approved, WaitError>, Vec<Duration>, usize) {
        let mut replies = replies.into_iter();
        let mut slept = Vec::new();
        let mut blips = 0;
        let r = wait_for_approval(
            s,
            &mut || replies.next().unwrap_or(Ok(Poll::Pending)),
            &mut |d| {
                slept.push(d);
                cancel_after.is_none_or(|n| slept.len() <= n)
            },
            &mut |_| blips += 1,
        );
        (r, slept, blips)
    }

    #[test]
    fn polls_at_the_interval_and_slows_down_when_told() {
        let (r, slept, _) = run(vec![Ok(Poll::Pending), Ok(Poll::SlowDown), Ok(Poll::Pending), Ok(Poll::Approved(approved()))], &started(3, 900), None);
        assert_eq!(r.unwrap().node_id, "n1");
        let secs: Vec<u64> = slept.iter().map(|d| d.as_secs()).collect();
        assert_eq!(secs, [3, 3, 8, 8]);
    }

    #[test]
    fn network_blips_are_retried_and_hard_errors_stop() {
        let (r, _, blips) = run(vec![Err(ApiError::Network("reset".into())), Err(ApiError::Status(503, String::new())), Ok(Poll::Approved(approved()))], &started(1, 900), None);
        assert!(r.is_ok());
        assert_eq!(blips, 2);
        let (r, _, _) = run(vec![Err(ApiError::Status(400, "bad_code".into()))], &started(1, 900), None);
        assert!(matches!(r, Err(WaitError::Failed(e)) if e.contains("bad_code")));
        let (r, slept, _) = run(vec![Ok(Poll::Pending), Err(ApiError::Status(404, "invalid_device_code".into()))], &started(3, 900), None);
        assert!(matches!(r, Err(WaitError::Failed(e)) if e.contains("goethe login again")));
        assert_eq!(slept.len(), 2, "a used or unknown code stops the polling");
        let (r, _, blips) = run(vec![Err(ApiError::RateLimited(Duration::from_secs(2))), Ok(Poll::Approved(approved()))], &started(3, 900), None);
        assert!(r.is_ok() && blips == 1);
        let (r, _, blips) = run(vec![Err(ApiError::Status(404, String::new())), Ok(Poll::Pending), Ok(Poll::Approved(approved()))], &started(1, 900), None);
        assert!(r.is_ok() && blips == 1, "a 404 page without the platform's own code (a deploy or a proxy) is only a blip");
    }

    #[test]
    fn expiry_denial_timeout_and_ctrl_c_end_the_wait() {
        let (r, _, _) = run(vec![Ok(Poll::Expired)], &started(3, 900), None);
        assert!(matches!(r, Err(WaitError::Failed(e)) if e.contains("expired")));
        let (r, _, _) = run(vec![Ok(Poll::Denied)], &started(3, 900), None);
        assert!(matches!(r, Err(WaitError::Failed(e)) if e.contains("denied")));
        let (r, slept, _) = run(vec![], &started(60, 900), None);
        assert!(matches!(r, Err(WaitError::Failed(e)) if e.contains("expired")));
        assert_eq!(slept.len(), 15, "15 min at 60 s");
        let (r, slept, _) = run(vec![], &started(5, 30), None);
        assert!(r.is_err() && slept.len() == 6, "the server's expiry counts too");
        let (r, slept, _) = run(vec![], &started(3, 900), Some(2));
        assert_eq!(r.unwrap_err(), WaitError::Cancelled);
        assert_eq!(slept.len(), 3);
    }

    #[test]
    fn device_flow_against_a_mock_platform() {
        let m = mock::serve(vec![
            (200, r#"{"device_code":"dc9","user_code":"WXYZ-0001","verification_uri_complete":"http://127.0.0.1:1/device?code=WXYZ-0001","expires_in":900,"interval":1}"#.into()),
            (502, "".into()),
            (200, r#"{"status":"pending"}"#.into()),
            (200, r#"{"status":"approved","node_id":"n7","node_token":"mesh_node_dddddddddddddddd","workspace":{"id":"w7","name":"Prod"}}"#.into()),
        ]);
        let c = Client::new(&m.url, Duration::from_secs(5)).unwrap();
        let s = c.auth_start("h", "f").unwrap();
        let mut blips = 0;
        let a = wait_for_approval(&s, &mut || c.auth_poll(&s.device_code), &mut |_| true, &mut |_| blips += 1).unwrap();
        assert_eq!((a.node_id.as_str(), a.workspace.name.as_str(), blips), ("n7", "Prod", 1));
        let polls: Vec<serde_json::Value> = m.seen.lock().unwrap()[1..].iter().map(|r| serde_json::from_str(&r.body).unwrap()).collect();
        assert!(polls.iter().all(|p| p["action"] == "poll" && p["device_code"] == "dc9"));
    }

    #[test]
    fn heartbeat_text_reads_the_state() {
        let now = chrono::DateTime::parse_from_rfc3339("2026-11-11T10:00:10Z").unwrap().with_timezone(&chrono::Utc);
        let ok = PlatformState { link: "n".into(), last_ok_at: Some("2026-11-11T10:00:08Z".into()), ..Default::default() };
        assert_eq!(heartbeat_text(&ok, true, now), ("OK (2 s ago)".into(), true));
        let mut failing = ok.clone();
        failing.last_error = Some("cannot reach the platform: timeout".into());
        failing.last_error_at = Some("2026-11-11T10:00:09Z".into());
        assert_eq!(heartbeat_text(&failing, true, now).0, "failing: cannot reach the platform: timeout");
        failing.last_ok_at = Some("2026-11-11T10:00:10Z".into());
        assert!(heartbeat_text(&failing, true, now).1, "a later success wins");
        assert!(heartbeat_text(&ok, false, now).0.starts_with("not sent, goethed is not running"));
        let fresh = PlatformState::default();
        assert_eq!(heartbeat_text(&fresh, true, now).0, "waiting for the first one");
    }

    #[test]
    fn a_revoked_link_is_replaced_without_a_logout_call() {
        use std::os::unix::fs::PermissionsExt;
        let m = mock::serve(vec![(200, r#"{"ok":true}"#.into())]);
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        std::fs::set_permissions(home, std::fs::Permissions::from_mode(0o700)).unwrap();
        let c = Credentials {
            platform_url: m.url.clone(),
            node_id: "n1".into(),
            workspace: Workspace { id: "w".into(), name: "Personal".into() },
            token: Secret::new("mesh_node_gggggggggggggggggggg"),
            linked_at: "t".into(),
        };
        platform::save_credentials(home, &c).unwrap();
        assert!(!is_revoked(home, &c));
        assert!(relink_intro("Personal", false).contains("logging in again replaces that link"));
        let st = PlatformState { link: platform::link_key(&c), revoked_at: Some("t".into()), ..Default::default() };
        platform::save_state(home, &st).unwrap();
        assert!(is_revoked(home, &c));
        assert_eq!(relink_intro("Personal", true), "The previous link (workspace Personal) was revoked or expired on the platform; linking again.");
        replace_old_link(home, Some(c.clone())).unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(*m.connections.lock().unwrap(), 0, "no logout call for a revoked link");
        assert_eq!(platform::load_credentials(home).unwrap(), None);
        platform::save_credentials(home, &c).unwrap();
        replace_old_link(home, Some(c)).unwrap();
        assert_eq!(m.seen.lock().unwrap()[0].path, "/api/device/logout", "a live link is logged out");
    }

    #[test]
    fn auth_keys_are_checked_before_use() {
        assert!(read_authkey("msh_ak_0123456789").is_ok());
        for bad in ["", "mesh_node_x", "msh_ak_", "msh_ak_ with space", "msh_ak_\u{7}abcdefgh"] {
            assert!(read_authkey(bad).is_err(), "{bad:?}");
        }
        assert!(!format!("{:?}", read_authkey("msh_ak_0123456789").unwrap()).contains("msh_ak_"));
    }
}
