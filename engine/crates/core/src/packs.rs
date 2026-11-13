// Gated kernel-pack pull. goethe does network and auth; goethe-engine decides trust (signature, rollback, identity, arch).

use crate::engine_cmd::{resolve_python, PACK_FILE, PACK_VERIFY};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// The bearer token for pack downloads. Only the platform login supplies one: packs never install without it.
pub trait TokenSource {
    fn token(&self) -> Result<String, String>;
}

/// The node token of this machine's platform link.
pub struct Login(pub crate::platform::Credentials);

impl TokenSource for Login {
    fn token(&self) -> Result<String, String> {
        Ok(self.0.token.expose().to_string())
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
        let agent = crate::net::agent_builder().redirects(0).timeout_read(std::time::Duration::from_secs(60)).build();
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
                Err(ureq::Error::Status(401, _)) => {
                    return Err("the platform did not accept this node's login (logged out or removed); run goethe login".into())
                }
                Err(ureq::Error::Status(403, _)) => return Err("the platform refused the kernel pack for this workspace".into()),
                Err(ureq::Error::Status(429, _)) => return Err("the platform is rate limiting pack downloads; try again in a minute".into()),
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

/// Pack index ABI and runtimes the pinned goethe-engine runs (goethe_engine.pack ABI and RUNTIMES); a catalog entry
/// outside them can never match, so it is never offered.
const ENGINE_ABI_MAJOR: u64 = 2;
const ENGINE_RUNTIMES: &[(&str, u64)] = &[("gc-c", 1)];
const CATALOG_MISS_SECS: u64 = 300;

/// A pack the platform's public index lists for a model and one of this machine's GPU architectures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Offer {
    pub arch: String,
    pub hf_repo: String,
    pub revision: String,
    pub format: String,
    pub files: Vec<String>,
}

/// The platform this machine talks to for packs: the one it is linked to, else the configured default.
pub fn platform_base(home: &Path) -> Result<String, String> {
    match crate::platform::load_credentials(home) {
        Ok(Some(c)) => Ok(c.platform_url),
        _ => crate::platform::platform_url(),
    }
}

/// The platform's public pack index (unverified bytes), revalidated with its ETag and cached under packs/catalog.
/// Only decides what to offer: everything installed is verified by goethe-engine against the signed index again.
/// None when the platform is unreachable and nothing is cached; a failure is remembered for a few minutes.
pub fn catalog(home: &Path) -> Option<Vec<u8>> {
    let dir = home.join("packs").join("catalog");
    let (body, etag, miss) = (dir.join("index.json"), dir.join("etag"), dir.join("miss"));
    let cached = || std::fs::read(&body).ok();
    let recent_miss = std::fs::metadata(&miss)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.elapsed().ok())
        .is_some_and(|age| age.as_secs() < CATALOG_MISS_SECS);
    if recent_miss || crate::hf::offline_mode() {
        return cached();
    }
    let base = platform_base(home).ok()?;
    let mut req = crate::net::agent().get(&format!("{}/packs/v2/index", base.trim_end_matches('/'))).timeout(std::time::Duration::from_secs(10));
    if let (Some(tag), true) = (std::fs::read_to_string(&etag).ok(), body.is_file()) {
        req = req.set("If-None-Match", tag.trim());
    }
    let _ = std::fs::create_dir_all(&dir);
    match req.call() {
        Ok(r) if r.status() == 304 => {
            let _ = std::fs::remove_file(&miss);
            cached()
        }
        Ok(r) => {
            let tag = r.header("ETag").map(String::from);
            let bytes = read_all(r.into_reader()).ok()?;
            let _ = std::fs::write(&body, &bytes);
            match tag {
                Some(t) => std::fs::write(&etag, t).ok(),
                None => std::fs::remove_file(&etag).ok(),
            };
            let _ = std::fs::remove_file(&miss);
            Some(bytes)
        }
        Err(_) => {
            let _ = std::fs::write(&miss, b"");
            cached()
        }
    }
}

/// Entries of `index` for `model_id` (case-insensitive) on one of `archs`, in `archs` order, that the pinned engine can run.
pub fn offers(index: &[u8], model_id: &str, archs: &[String]) -> Vec<Offer> {
    offers_where(index, archs, |repo| repo.eq_ignore_ascii_case(model_id))
}

/// Every entry of `index` on one of `archs` that the pinned engine can run, whatever the model.
pub fn offers_any(index: &[u8], archs: &[String]) -> Vec<Offer> {
    offers_where(index, archs, |_| true)
}

fn offers_where(index: &[u8], archs: &[String], model: impl Fn(&str) -> bool) -> Vec<Offer> {
    let Ok(ix) = serde_json::from_slice::<serde_json::Value>(index) else { return Vec::new() };
    if ix["schema"] != "gc-pack-index" || ix["version"] != 2 {
        return Vec::new();
    }
    let engine = version(crate::pyenv::ENGINE_VERSION);
    let mut out = Vec::new();
    for arch in archs {
        for e in ix["entries"].as_array().into_iter().flatten() {
            let m = &e["model"];
            let runs = ENGINE_RUNTIMES.iter().any(|(k, v)| e["runtime"]["kind"] == *k && e["runtime"]["version"] == *v);
            let fits = e["arch"].as_str() == Some(arch.as_str())
                && e["abi_major"].as_u64() == Some(ENGINE_ABI_MAJOR)
                && runs
                && e["min_engine"].as_str().and_then(version).is_some_and(|v| Some(v) <= engine)
                && m["hf_repo"].as_str().is_some_and(&model);
            let files: Vec<String> = m["files"].as_object().map(|f| f.keys().cloned().collect()).unwrap_or_default();
            if !fits || files.is_empty() || files.iter().any(|f| !safe_name(f)) {
                continue;
            }
            let offer = Offer {
                arch: arch.clone(),
                hf_repo: m["hf_repo"].as_str().unwrap_or_default().to_string(),
                revision: m["revision"].as_str().unwrap_or("main").to_string(),
                format: m["format"].as_str().unwrap_or_default().to_string(),
                files,
            };
            if !out.contains(&offer) {
                out.push(offer);
            }
        }
    }
    out
}

fn version(s: &str) -> Option<(u64, u64, u64)> {
    let mut it = s.split('.').map(|x| x.parse::<u64>().ok());
    let v = (it.next()??, it.next()??, it.next()??);
    it.next().is_none().then_some(v)
}

fn safe_name(f: &str) -> bool {
    // A plain file name in the repo root: the download and the model dir never see a path from the index.
    !f.is_empty() && !f.contains(['/', '\\']) && f != "." && f != ".." && !f.starts_with('.')
}

/// The distinct pack architectures (sm_XX tags) of this machine's usable GPUs, most GPUs first.
pub fn local_archs(home: &Path) -> Vec<String> {
    let Ok(hw) = crate::hw::profile(home) else { return Vec::new() };
    let denied = crate::db::open(home).ok().and_then(|c| crate::gpu_policy::denied_set(&c).ok()).unwrap_or_default();
    let mut count: Vec<(String, usize)> = Vec::new();
    for g in hw.gpus.into_iter().filter(|g| !denied.contains(&g.index)) {
        if let Some(a) = crate::gpu_arch::lookup(&g.compute_capability) {
            match count.iter_mut().find(|(t, _)| *t == a.tag()) {
                Some(c) => c.1 += 1,
                None => count.push((a.tag(), 1)),
            }
        }
    }
    count.sort_by_key(|c| std::cmp::Reverse(c.1));
    count.into_iter().map(|(t, _)| t).collect()
}

/// The compute capabilities whose pack architecture is `arch`: the card's supported_sm, so placement keeps the model
/// on GPUs the pack was built for.
pub fn arch_ccs(arch: &str) -> Vec<String> {
    crate::gpu_arch::supported_list().into_iter().filter(|cc| crate::gpu_arch::lookup(cc).is_some_and(|a| a.tag() == arch)).collect()
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

/// The longest context any bucket of the verified pack in `dir` serves (`goethe-engine pack show`): the card's ctx cap.
pub fn max_ctx(dir: &Path, run: Runner) -> Result<u64, String> {
    let (_, out) = run(&["pack".into(), "show".into(), s(&dir.join(PACK_FILE))])?;
    out.lines()
        .filter_map(|l| l.split_whitespace().skip_while(|w| *w != "maxctx").nth(1)?.parse::<u64>().ok())
        .max()
        .ok_or_else(|| "pack show: no bucket lists a maxctx".to_string())
}

fn already(pack: &Path, want: &str) -> bool {
    // The pack file in the model dir is the one the entry names (a reinstall downloads nothing again).
    let Ok(mut f) = std::fs::File::open(pack) else { return false };
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h).is_ok() && format!("{:x}", h.finalize()) == want
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
    if !already(&pack, &sha) {
        download(broker.pack(&sha).map_err(undo)?, &sha, &pack).map_err(undo)?;
    }
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

    struct Refuses;
    impl Broker for Refuses {
        fn index(&self) -> Result<Vec<u8>, String> {
            Ok(b"{}".to_vec())
        }
        fn index_sig(&self) -> Result<Vec<u8>, String> {
            Ok(b"sig".to_vec())
        }
        fn pack(&self, _: &str) -> Result<Box<dyn Read>, String> {
            Err("the platform did not accept this node's login".into())
        }
    }

    #[test]
    fn a_refused_pack_download_leaves_no_pack_files_from_an_earlier_install() {
        let (home, dir) = dirs();
        for f in [format!("{PACK_FILE}.sig"), PACK_VERIFY.to_string()] {
            std::fs::write(dir.path().join(f), b"old").unwrap();
        }
        assert!(pull(home.path(), dir.path(), "sm_120a", &Refuses, &engine(&[])).unwrap_err().contains("login"));
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

    fn index(entries: serde_json::Value) -> Vec<u8> {
        serde_json::json!({"schema": "gc-pack-index", "version": 2, "serial": 3, "entries": entries}).to_string().into_bytes()
    }

    fn entry(repo: &str, arch: &str) -> serde_json::Value {
        serde_json::json!({"model": {"hf_repo": repo, "revision": "abc", "format": "gguf", "files": {"m-Q8_0.gguf": "00"}},
            "arch": arch, "abi_major": 2, "runtime": {"kind": "gc-c", "version": 1}, "min_engine": "0.1.0", "file_sha256": "11"})
    }

    #[test]
    fn offers_follow_the_model_the_local_archs_and_what_the_engine_runs() {
        let archs = vec!["sm_120a".to_string(), "sm_90a".to_string()];
        let mut abi1 = entry("org/m-GGUF", "sm_120a");
        abi1["abi_major"] = 1.into();
        let mut newer = entry("org/m-GGUF", "sm_120a");
        newer["min_engine"] = "9.0.0".into();
        let mut path = entry("org/m-GGUF", "sm_120a");
        path["model"]["files"] = serde_json::json!({"../x.gguf": "00"});
        let ix = index(serde_json::json!([abi1, newer, path, entry("org/other", "sm_120a"), entry("org/m-GGUF", "sm_100a"), entry("Org/M-gguf", "sm_90a")]));
        let o = offers(&ix, "org/m-gguf", &archs);
        assert_eq!(o.len(), 1);
        assert_eq!((o[0].arch.as_str(), o[0].files.clone(), o[0].revision.as_str()), ("sm_90a", vec!["m-Q8_0.gguf".to_string()], "abc"));
        assert_eq!(offers_any(&ix, &archs).len(), 2);
        assert!(offers(b"not json", "org/m-gguf", &archs).is_empty());
        let mut v1: serde_json::Value = serde_json::from_slice(&ix).unwrap();
        v1["version"] = 1.into();
        assert!(offers(v1.to_string().as_bytes(), "org/m-gguf", &archs).is_empty());
    }

    #[test]
    fn the_card_names_the_compute_capabilities_of_the_pack_arch() {
        let ccs = arch_ccs("sm_120a");
        assert!(ccs.contains(&"12.0".to_string()) && !ccs.contains(&"9.0".to_string()));
    }

    #[test]
    fn max_ctx_is_the_longest_bucket() {
        let d = tempfile::tempdir().unwrap();
        let run = |_: &[String]| Ok((0, "a  b4  ncol   1 maxctx   4096 3 files\nb  b4  ncol   8 maxctx  32768 3 files\n".to_string()));
        assert_eq!(max_ctx(d.path(), &run).unwrap(), 32768);
        let none = |_: &[String]| Ok((0, String::new()));
        assert!(max_ctx(d.path(), &none).is_err());
    }

    #[test]
    fn a_pack_already_in_place_is_not_downloaded_again() {
        let (home, dir) = dirs();
        std::fs::write(dir.path().join(PACK_FILE), BYTES).unwrap();
        let b = Fake { pack: b"other".to_vec(), ..Fake::ok() };
        assert!(pull(home.path(), dir.path(), "sm_120a", &b, &engine(&[])).unwrap());
        assert_eq!(*b.fetched.borrow(), 0);
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
        assert!(b.index().unwrap_err().contains("goethe login"));
        let b = HttpBroker { store: "http://127.0.0.1:1".into(), tokens: &Tok(Err("not logged in".into())) };
        assert!(b.index().unwrap_err().contains("not logged in"));
    }
}
