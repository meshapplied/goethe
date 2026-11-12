// Gated kernel-pack pull. goethe does network and auth; goethe-engine decides trust (signature, rollback, identity, arch).

use crate::engine_cmd::{resolve_python, PACK_FILE, PACK_VERIFY};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The login token. The login flow fills this in; until it does, GOETHE_PACKS_TOKEN stands in for it.
pub trait TokenSource {
    fn token(&self) -> Result<String, String>;
}

pub struct EnvToken;

impl TokenSource for EnvToken {
    fn token(&self) -> Result<String, String> {
        std::env::var("GOETHE_PACKS_TOKEN").map_err(|_| "not logged in to the pack store".to_string())
    }
}

/// The three broker routes. Every error is final: the caller never falls back to unverified bytes.
pub trait Broker {
    fn index(&self) -> Result<Vec<u8>, String>;
    fn index_sig(&self) -> Result<Vec<u8>, String>;
    fn pack(&self, file_sha256: &str) -> Result<Box<dyn Read>, String>;
}

pub struct HttpBroker<'a> {
    pub store: String,
    pub tokens: &'a dyn TokenSource,
}

impl HttpBroker<'_> {
    fn get(&self, path: &str) -> Result<Box<dyn Read>, String> {
        // Redirects are followed by hand: the bearer token goes to the store, never to the redirect target.
        let agent = ureq::AgentBuilder::new().redirects(0).build();
        let token = self.tokens.token()?;
        let mut req = agent.get(&format!("{}{path}", self.store.trim_end_matches('/'))).set("Authorization", &format!("Bearer {token}"));
        for _ in 0..2 {
            match req.call() {
                Ok(r) if (300..400).contains(&r.status()) => {
                    let to = r.header("Location").ok_or(format!("{path}: redirect without Location"))?;
                    if !to.starts_with("https://") && !to.starts_with("http://") {
                        return Err(format!("{path}: redirect to a relative URL"));
                    }
                    req = agent.get(to);
                }
                Ok(r) => return Ok(Box::new(r.into_reader())),
                Err(ureq::Error::Status(c, _)) => return Err(format!("{path}: HTTP {c}")),
                Err(e) => return Err(format!("{path}: {e}")),
            }
        }
        Err(format!("{path}: too many redirects"))
    }
}

impl Broker for HttpBroker<'_> {
    fn index(&self) -> Result<Vec<u8>, String> {
        read_all(self.get("/packs/v2/index")?)
    }
    fn index_sig(&self) -> Result<Vec<u8>, String> {
        read_all(self.get("/packs/v2/index.sig")?)
    }
    fn pack(&self, file_sha256: &str) -> Result<Box<dyn Read>, String> {
        self.get(&format!("/packs/v2/pack/{file_sha256}"))
    }
}

fn read_all(mut r: impl Read) -> Result<Vec<u8>, String> {
    let mut v = Vec::new();
    r.read_to_end(&mut v).map_err(|e| e.to_string())?;
    Ok(v)
}

/// Runs `goethe-engine ARGS`: (exit code, stdout).
pub type Runner<'a> = &'a dyn Fn(&[String]) -> Result<(i32, String), String>;

pub fn python_runner(home: &Path) -> impl Fn(&[String]) -> Result<(i32, String), String> {
    let py = resolve_python(home);
    move |args| {
        let out = Command::new(&py).args(["-m", "goethe_engine"]).args(args).output().map_err(|e| format!("{}: {e}", py.display()))?;
        let err = String::from_utf8_lossy(&out.stderr);
        let code = out.status.code().unwrap_or(-1);
        if code != 0 && code != 3 {
            return Err(format!("goethe-engine {}: exit {code}: {}", args.iter().take(2).cloned().collect::<Vec<_>>().join(" "), err.trim()));
        }
        Ok((code, String::from_utf8_lossy(&out.stdout).into_owned()))
    }
}

/// The store URL, from packs.toml in the goethe home (`url = "https://..."`). Absent: no store, nothing changes.
pub fn store_url(home: &Path) -> Result<Option<String>, String> {
    let Ok(text) = std::fs::read_to_string(home.join("packs.toml")) else { return Ok(None) };
    let t: toml::Table = toml::from_str(&text).map_err(|e| format!("packs.toml: {e}"))?;
    Ok(t.get("url").and_then(|v| v.as_str()).map(String::from))
}

fn serial_path(home: &Path) -> PathBuf {
    home.join("packs").join("serial")
}

fn last_serial(home: &Path) -> u64 {
    std::fs::read_to_string(serial_path(home)).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0)
}

fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

fn download(mut r: Box<dyn Read>, want: &str, dest: &Path) -> Result<(), String> {
    let mut f = std::fs::File::create(dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    let (mut h, mut buf) = (Sha256::new(), vec![0u8; 1 << 20]);
    loop {
        let n = r.read(&mut buf).map_err(|e| format!("pack download: {e}"))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        f.write_all(&buf[..n]).map_err(|e| e.to_string())?;
    }
    let got = format!("{:x}", h.finalize());
    if got != want {
        return Err(format!("pack download: sha256 {got} differs from the index entry {want}"));
    }
    Ok(())
}

/// Index, match on the files in `dir`, download, verify. Ok(true): `dir` holds the verified pack, its signature and the
/// verification output, and the index serial is stored. Ok(false): no pack for these files (nothing written but the index).
/// Err: nothing usable is left in `dir`.
pub fn pull(home: &Path, dir: &Path, arch: &str, broker: &dyn Broker, run: Runner) -> Result<bool, String> {
    let work = home.join("packs");
    std::fs::create_dir_all(&work).map_err(|e| e.to_string())?;
    let (ix, sg) = (work.join("index.json"), work.join("index.sig"));
    std::fs::write(&ix, broker.index()?).map_err(|e| e.to_string())?;
    std::fs::write(&sg, broker.index_sig()?).map_err(|e| e.to_string())?;
    let last = last_serial(home).to_string();
    let v = |x: &str| x.to_string();
    let base = [v("--index"), s(&ix), v("--sig"), s(&sg), v("--last-serial"), last.clone()];
    let verified = run(&[&[v("index"), v("verify")][..], &base].concat())?.1;
    let serial = serde_json::from_str::<serde_json::Value>(&verified).map_err(|e| format!("index verify: {e}"))?["serial"].as_u64();
    let serial = serial.ok_or("index verify: no serial")?;
    let (code, out) = run(&[&[v("pack"), v("match")][..], &base, &[v("--model-dir"), s(dir), v("--arch"), v(arch)]].concat())?;
    if code == 3 {
        return Ok(false);
    }
    let entry: serde_json::Value = serde_json::from_str(&out).map_err(|e| format!("pack match: {e}"))?;
    let sha = entry["file_sha256"].as_str().ok_or("entry: no file_sha256")?.to_string();
    if sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()) {
        return Err("entry: file_sha256 is not lowercase sha256 hex".into());
    }
    let sig = serde_json::to_vec(&entry["sig"]).map_err(|e| e.to_string())?;
    let (pack, psig, pv) = (dir.join(PACK_FILE), dir.join(format!("{PACK_FILE}.sig")), dir.join(PACK_VERIFY));
    let undo = |e: String| {
        for p in [&pack, &psig, &pv] {
            let _ = std::fs::remove_file(p);
        }
        e
    };
    download(broker.pack(&sha)?, &sha, &pack).map_err(undo)?;
    std::fs::write(&psig, sig).map_err(|e| undo(e.to_string()))?;
    let ok = run(&[v("pack"), v("verify"), s(&pack), v("--arch"), v(arch)]).map_err(undo)?.1;
    std::fs::write(&pv, ok).map_err(|e| undo(e.to_string()))?;
    std::fs::write(serial_path(home), format!("{serial}\n")).map_err(|e| undo(e.to_string()))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    const BYTES: &[u8] = b"pack bytes";

    fn sha(b: &[u8]) -> String {
        format!("{:x}", Sha256::digest(b))
    }

    struct Fake {
        index: Result<Vec<u8>, String>,
        pack: Vec<u8>,
        fetched: RefCell<u32>,
    }

    impl Fake {
        fn ok() -> Self {
            Fake { index: Ok(b"{}".to_vec()), pack: BYTES.to_vec(), fetched: RefCell::new(0) }
        }
    }

    impl Broker for Fake {
        fn index(&self) -> Result<Vec<u8>, String> {
            self.index.clone()
        }
        fn index_sig(&self) -> Result<Vec<u8>, String> {
            Ok(b"sig".to_vec())
        }
        fn pack(&self, _: &str) -> Result<Box<dyn Read>, String> {
            *self.fetched.borrow_mut() += 1;
            Ok(Box::new(std::io::Cursor::new(self.pack.clone())))
        }
    }

    // A fake goethe-engine: exit code and stdout per subcommand ("index verify", "pack match", "pack verify").
    fn engine(codes: &[(&str, i32)]) -> impl Fn(&[String]) -> Result<(i32, String), String> {
        let codes: HashMap<String, i32> = codes.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        move |a| {
            let key = format!("{} {}", a[0], a[1]);
            let code = codes.get(&key).copied().unwrap_or(0);
            if code == 1 {
                return Err(format!("goethe-engine {key}: exit 1"));
            }
            let out = match key.as_str() {
                "index verify" => r#"{"serial": 7}"#.to_string(),
                "pack match" => serde_json::json!({ "file_sha256": sha(BYTES), "sig": { "keyid": "k" } }).to_string(),
                _ => r#"{"ok": true, "weights": ["t.gguf"]}"#.to_string(),
            };
            Ok((code, out))
        }
    }

    fn dirs() -> (tempfile::TempDir, tempfile::TempDir) {
        (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap())
    }

    fn nothing_left(dir: &Path) {
        for n in [PACK_FILE.to_string(), format!("{PACK_FILE}.sig"), PACK_VERIFY.to_string()] {
            assert!(!dir.join(n).exists());
        }
    }

    #[test]
    fn a_match_writes_the_pack_its_signature_and_the_verification_and_stores_the_serial() {
        let (home, dir) = dirs();
        assert!(pull(home.path(), dir.path(), "sm_120a", &Fake::ok(), &engine(&[])).unwrap());
        assert_eq!(std::fs::read(dir.path().join(PACK_FILE)).unwrap(), BYTES);
        assert_eq!(std::fs::read_to_string(dir.path().join(format!("{PACK_FILE}.sig"))).unwrap(), r#"{"keyid":"k"}"#);
        assert!(std::fs::read_to_string(dir.path().join(PACK_VERIFY)).unwrap().contains("t.gguf"));
        assert_eq!(last_serial(home.path()), 7);
    }

    #[test]
    fn no_matching_pack_changes_nothing() {
        let (home, dir) = dirs();
        let b = Fake::ok();
        assert!(!pull(home.path(), dir.path(), "sm_120a", &b, &engine(&[("pack match", 3)])).unwrap());
        assert_eq!(*b.fetched.borrow(), 0);
        nothing_left(dir.path());
        assert_eq!(last_serial(home.path()), 0);
    }

    #[test]
    fn pack_bytes_that_differ_from_the_entry_hash_fail_and_leave_nothing() {
        let (home, dir) = dirs();
        let b = Fake { pack: b"tampered".to_vec(), ..Fake::ok() };
        assert!(pull(home.path(), dir.path(), "sm_120a", &b, &engine(&[])).unwrap_err().contains("differs"));
        nothing_left(dir.path());
        assert_eq!(last_serial(home.path()), 0);
    }

    #[test]
    fn a_broker_refusal_such_as_401_fails() {
        let (home, dir) = dirs();
        let b = Fake { index: Err("/packs/v2/index: HTTP 401".into()), ..Fake::ok() };
        assert!(pull(home.path(), dir.path(), "sm_120a", &b, &engine(&[])).unwrap_err().contains("401"));
        nothing_left(dir.path());
    }

    #[test]
    fn an_index_the_engine_refuses_fails_before_any_download() {
        // foreign key, expired index, serial below the stored one: all are exit 1 of index verify
        let (home, dir) = dirs();
        let b = Fake::ok();
        assert!(pull(home.path(), dir.path(), "sm_120a", &b, &engine(&[("index verify", 1)])).is_err());
        assert_eq!(*b.fetched.borrow(), 0);
        nothing_left(dir.path());
    }

    #[test]
    fn a_pack_the_engine_refuses_is_removed_and_the_serial_is_not_stored() {
        // modified after signing, wrong arch: pack verify exit 1
        let (home, dir) = dirs();
        assert!(pull(home.path(), dir.path(), "sm_120a", &Fake::ok(), &engine(&[("pack verify", 1)])).is_err());
        nothing_left(dir.path());
        assert_eq!(last_serial(home.path()), 0);
    }

    #[test]
    fn a_malformed_entry_hash_is_refused() {
        let (home, dir) = dirs();
        let run = |a: &[String]| {
            if a[1] == "match" {
                return Ok((0, r#"{"file_sha256": "../x", "sig": {}}"#.to_string()));
            }
            engine(&[])(a)
        };
        assert!(pull(home.path(), dir.path(), "sm_120a", &Fake::ok(), &run).unwrap_err().contains("not lowercase"));
    }

    #[test]
    fn no_store_url_means_no_store() {
        let home = tempfile::tempdir().unwrap();
        assert_eq!(store_url(home.path()).unwrap(), None);
        std::fs::write(home.path().join("packs.toml"), "url = \"https://s.example\"\n").unwrap();
        assert_eq!(store_url(home.path()).unwrap().as_deref(), Some("https://s.example"));
    }

    struct Tok(Result<String, String>);
    impl TokenSource for Tok {
        fn token(&self) -> Result<String, String> {
            self.0.clone()
        }
    }

    // One-shot HTTP server: replies to one request, returns the request head it saw.
    fn serve(reply: String) -> (String, std::thread::JoinHandle<String>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", l.local_addr().unwrap());
        let h = std::thread::spawn(move || {
            let (mut c, _) = l.accept().unwrap();
            let mut head = String::new();
            let mut r = BufReader::new(c.try_clone().unwrap());
            loop {
                let mut line = String::new();
                if r.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                    break;
                }
                head += &line;
            }
            c.write_all(reply.as_bytes()).unwrap();
            head
        });
        (url, h)
    }

    #[test]
    fn the_bearer_token_goes_to_the_store_and_never_to_the_redirect_target() {
        let (target, target_seen) = serve("HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\npack bytes".into());
        let (store, store_seen) = serve(format!("HTTP/1.1 302 Found\r\nLocation: {target}/blob\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"));
        let tok = Tok(Ok("SECRET".into()));
        let b = HttpBroker { store, tokens: &tok };
        assert_eq!(read_all(b.pack(&sha(BYTES)).unwrap()).unwrap(), BYTES);
        assert!(store_seen.join().unwrap().contains("Bearer SECRET"));
        assert!(!target_seen.join().unwrap().to_lowercase().contains("authorization"));
    }

    #[test]
    fn http_401_and_a_missing_login_are_errors() {
        let (store, _h) = serve("HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into());
        let b = HttpBroker { store, tokens: &Tok(Ok("bad".into())) };
        assert!(b.index().unwrap_err().contains("HTTP 401"));
        let b = HttpBroker { store: "http://127.0.0.1:1".into(), tokens: &Tok(Err("not logged in".into())) };
        assert!(b.index().unwrap_err().contains("not logged in"));
    }
}
