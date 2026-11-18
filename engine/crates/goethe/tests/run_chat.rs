// goethe run with no input on a terminal: the chat screen against a fake engine on a pty; one line sent, the reply streams, Esc q Enter leaves, no file is written.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn fake_engine() -> u16 {
    // /v1/models lists the model; /v1/chat/completions streams "hi " then "there".
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let mut stream = stream;
            let mut got = Vec::new();
            let mut buf = [0u8; 4096];
            let head = loop {
                let n = stream.read(&mut buf).unwrap_or(0);
                got.extend_from_slice(&buf[..n]);
                let text = String::from_utf8_lossy(&got).to_string();
                let Some((head, body)) = text.split_once("\r\n\r\n") else {
                    if n == 0 {
                        break text;
                    }
                    continue;
                };
                let len = head.to_ascii_lowercase().split("content-length:").nth(1).and_then(|v| v.split_whitespace().next()?.parse::<usize>().ok()).unwrap_or(0);
                if body.len() >= len || n == 0 {
                    break head.to_string();
                }
            };
            let (kind, body) = if head.starts_with("GET /v1/models") {
                ("application/json", r#"{"data": [{"id": "org/m"}]}"#.to_string())
            } else {
                let piece = |s: &str| format!("data: {{\"choices\": [{{\"delta\": {{\"content\": \"{s}\"}}}}]}}\n\n");
                let long: String = (1..=70).map(|n| piece(&format!("row {n}\\n"))).collect();
                ("text/event-stream", format!("{}{}{long}data: [DONE]\n\n", piece("hi "), piece("there")))
            };
            let _ = write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
        }
    });
    port
}

fn pty() -> (OwnedFd, OwnedFd) {
    let (mut master, mut slave) = (0, 0);
    let mut size = libc::winsize { ws_row: 24, ws_col: 80, ws_xpixel: 0, ws_ypixel: 0 };
    let ok = unsafe { libc::openpty(&mut master, &mut slave, std::ptr::null_mut(), std::ptr::null_mut(), &mut size as *mut libc::winsize as _) };
    assert_eq!(ok, 0, "openpty");
    unsafe { (OwnedFd::from_raw_fd(master), OwnedFd::from_raw_fd(slave)) }
}

fn pump(master: &OwnedFd, seen: &mut String, ms: i32) {
    // Whatever the terminal wrote, waiting up to ms for it.
    let mut fds = libc::pollfd { fd: master.as_raw_fd(), events: libc::POLLIN, revents: 0 };
    if unsafe { libc::poll(&mut fds, 1, ms) } > 0 {
        let mut buf = [0u8; 4096];
        let n = unsafe { libc::read(master.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len()) };
        if n > 0 {
            seen.push_str(&String::from_utf8_lossy(&buf[..n as usize]));
        }
    }
}

fn read_until(master: &OwnedFd, seen: &mut String, want: &str) {
    let end = Instant::now() + Duration::from_secs(15);
    while !seen.contains(want) {
        assert!(Instant::now() < end, "waited for {want:?}, screen so far: {seen:?}");
        pump(master, seen, 200);
    }
}

fn settle(master: &OwnedFd, seen: &mut String) {
    // Whatever else the screen draws in the next 300 ms.
    let end = Instant::now() + Duration::from_millis(300);
    while Instant::now() < end {
        pump(master, seen, 50);
    }
}

fn screen_of(raw: &str) -> Vec<String> {
    // The 24 rows a terminal shows after these bytes: full redraws and per-row rewrites, colors dropped.
    let (mut rows, mut row, mut cur) = (vec![String::new(); 24], 0usize, String::new());
    let chars: Vec<char> = raw.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '\x1b' if chars.get(i + 1) == Some(&'[') => {
                let mut j = i + 2;
                while j < chars.len() && !chars[j].is_ascii_alphabetic() {
                    j += 1;
                }
                let params: String = chars[i + 2..j].iter().collect();
                match (chars[j], params.as_str()) {
                    ('H', p) => {
                        rows[row] = std::mem::take(&mut cur);
                        row = p.split(';').next().and_then(|r| r.parse::<usize>().ok()).map_or(0, |r| r - 1);
                    }
                    ('J', "2") => {
                        rows.iter_mut().for_each(String::clear);
                        cur.clear();
                    }
                    _ => {}
                }
                i = j;
            }
            '\r' => {}
            '\n' => {
                rows[row] = std::mem::take(&mut cur);
                row = (row + 1).min(23);
            }
            c => cur.push(c),
        }
        i += 1;
    }
    rows[row] = cur;
    rows.iter().map(|r| r.trim_end().to_string()).collect()
}

fn type_keys(master: &OwnedFd, keys: &str) {
    assert_eq!(unsafe { libc::write(master.as_raw_fd(), keys.as_ptr().cast(), keys.len()) }, keys.len() as isize);
}

#[test]
fn the_chat_screen_streams_a_reply_and_leaves_cleanly_without_writing_a_file() {
    let (home, cwd) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let port = fake_engine();
    let con = goethe_core::db::open(home.path()).unwrap();
    con.execute(
        "INSERT INTO installed_models (model_id, local_path, architecture, static_vram_gb, kv_bytes_per_token, default_ctx, default_max_conc, default_tp) VALUES ('org/m', 'models/org__m', 'dense', 1.0, 1, 4096, 1, 1)",
        [],
    )
    .unwrap();
    con.execute(
        &format!("INSERT INTO loaded_instances (instance_id, model_id, pid, pgid, internal_port, gpu_indices, allocated_vram_gb_per_gpu, mem_fraction_static, ctx_len, max_conc, status, engine_key) VALUES ('inst-1', 'org/m', 1, 1, {port}, '0', 1.0, 0.5, 4096, 1, 'RUNNING', 'k')"),
        [],
    )
    .unwrap();
    drop(con);
    let (master, slave) = pty();
    let dup = |fd: &OwnedFd| Stdio::from(fd.try_clone().unwrap());
    let mut child = Command::new(env!("CARGO_BIN_EXE_goethe"))
        .args(["run", "org/m"])
        .env("GOETHE_HOME", home.path())
        .env("TERM", "xterm")
        .current_dir(cwd.path())
        .stdin(dup(&slave))
        .stdout(dup(&slave))
        .stderr(dup(&slave))
        .spawn()
        .unwrap();
    drop(slave);
    let mut seen = String::new();
    read_until(&master, &mut seen, "[Esc] commands  [Enter] send");
    assert!(seen.contains("org/m"), "the model is on the top line: {seen:?}");
    type_keys(&master, "hello, this line is longer than sixteen bytes\r");
    read_until(&master, &mut seen, "row 70");
    settle(&master, &mut seen);
    let following = screen_of(&seen);
    assert!(following[1..=20].iter().any(|r| r.contains("row 70")), "following the stream shows its end: {following:?}");
    assert!(!following.iter().any(|r| r.contains("> hello")), "the first lines scrolled off while following: {following:?}");
    type_keys(&master, "\x1b");
    read_until(&master, &mut seen, "[q] quit");
    type_keys(&master, "g");
    read_until(&master, &mut seen, "scrolled: G follows");
    settle(&master, &mut seen);
    let top = screen_of(&seen);
    assert!(top[1].contains("> hello") && top[2].contains("hi there"), "text flows from the top row: {top:?}");
    assert!(!top.iter().any(|r| r.contains("row 70")), "scrolled up, the end is below the window: {top:?}");
    type_keys(&master, "G");
    settle(&master, &mut seen);
    let end_again = screen_of(&seen);
    assert!(end_again[1..=20].iter().any(|r| r.contains("row 70")) && !end_again[0].contains("scrolled"), "G follows again: {end_again:?}");
    type_keys(&master, "q");
    read_until(&master, &mut seen, "exit? Enter = exit, q = back");
    type_keys(&master, "\r");
    let end = Instant::now() + Duration::from_secs(10);
    let status = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break s;
        }
        assert!(Instant::now() < end, "the chat did not exit");
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(status.success(), "{status:?}");
    read_until(&master, &mut seen, "\x1b[?1049l");
    assert_eq!(std::fs::read_dir(cwd.path()).unwrap().count(), 0, "closing writes nothing");
}
