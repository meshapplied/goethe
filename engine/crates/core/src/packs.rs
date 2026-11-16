// Gated kernel-pack pull. goethe does network and auth; goethe-engine decides trust (signature, rollback, identity, arch).

use crate::engines::{goethe_engine::{PACK_FILE, PACK_VERIFY}, resolve_python};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

pub trait TokenSource {
    fn token(&self) -> Result<String, String>;
}

pub struct Login(pub crate::platform::Credentials);

impl TokenSource for Login {
    fn token(&self) -> Result<String, String> {
        Ok(self.0.token.expose().to_string())
    }
}

pub trait Broker {
    fn index(&self) -> Result<Vec<u8>, String>;
    fn index_sig(&self) -> Result<Vec<u8>, String>;
    fn range(&self, file: &str, off: u64, len: u64) -> Result<Vec<u8>, String>;
}

pub struct HttpBroker<'a> {
    store: String,
    tokens: &'a dyn TokenSource,
    link: RefCell<Option<(String, String, Instant)>>,
}

const LINK_REUSE: Duration = Duration::from_secs(240);
const MAX_HEADER: u64 = 64 << 20;
const MAX_MANIFEST: u64 = 16 << 20;
const MAX_INDEX: u64 = 64 << 20;
const MERGE_GAP: u64 = 1 << 20;

impl<'a> HttpBroker<'a> {
    pub fn new(store: &str, tokens: &'a dyn TokenSource) -> HttpBroker<'a> {
        HttpBroker { store: store.trim_end_matches('/').to_string(), tokens, link: RefCell::new(None) }
    }

    fn agent() -> ureq::Agent {
        crate::net::agent_builder().redirects(0).timeout_read(Duration::from_secs(120)).build()
    }

    fn platform(&self, path: &str) -> Result<ureq::Response, String> {
        // One request to the platform with the node's token; a redirect comes back to the caller, which follows it without the token.
        let token = self.tokens.token()?;
        match Self::agent().get(&format!("{}{path}", self.store)).set("Authorization", &format!("Bearer {token}")).call() {
            Ok(r) => Ok(r),
            Err(ureq::Error::Status(401, _)) => {
                Err("the platform did not accept this node's login (logged out, removed, or its login expired); run goethe login".into())
            }
            Err(ureq::Error::Status(403, _)) => Err("the platform refused the kernel pack for this workspace".into()),
            Err(ureq::Error::Status(404, _)) => Err(format!("{path}: not in the pack store")),
            Err(ureq::Error::Status(429, _)) => Err("the platform is rate limiting pack downloads; try again in a minute".into()),
            Err(ureq::Error::Status(c, _)) => Err(format!("{path}: HTTP {c}")),
            Err(e) => Err(format!("{path}: {e}")),
        }
    }

    fn location(r: &ureq::Response, path: &str) -> Result<String, String> {
        let to = r.header("Location").ok_or(format!("{path}: redirect without Location"))?;
        if !to.starts_with("https://") && !to.starts_with("http://") {
            return Err(format!("{path}: redirect to a relative URL"));
        }
        Ok(to.to_string())
    }

    fn bytes(&self, path: &str) -> Result<Vec<u8>, String> {
        let r = self.platform(path)?;
        if (300..400).contains(&r.status()) {
            let to = Self::location(&r, path)?;
            return Self::agent().get(&to).call().map_err(|e| format!("{path}: {e}")).and_then(|r| read_all(r.into_reader()));
        }
        read_all(r.into_reader())
    }

    fn link(&self, file: &str, fresh: bool) -> Result<String, String> {
        // The short-lived download link of a pack, reused for a few minutes so a partial download asks the platform once.
        if let Some((f, url, at)) = self.link.borrow().as_ref() {
            if !fresh && f == file && at.elapsed() < LINK_REUSE {
                return Ok(url.clone());
            }
        }
        let path = format!("/packs/v2/pack/{file}");
        let r = self.platform(&path)?;
        if !(300..400).contains(&r.status()) {
            return Err(format!("{path}: expected a download link, got HTTP {}", r.status()));
        }
        let url = Self::location(&r, &path)?;
        self.link.replace(Some((file.to_string(), url.clone(), Instant::now())));
        Ok(url)
    }
}

impl Broker for HttpBroker<'_> {
    fn index(&self) -> Result<Vec<u8>, String> {
        self.bytes("/packs/v2/index.json")
    }
    fn index_sig(&self) -> Result<Vec<u8>, String> {
        self.bytes("/packs/v2/index.sig")
    }
    fn range(&self, file: &str, off: u64, len: u64) -> Result<Vec<u8>, String> {
        if len == 0 {
            return Ok(Vec::new());
        }
        let mut fresh = false;
        loop {
            let url = self.link(file, fresh)?;
            let req = Self::agent().get(&url).set("Range", &format!("bytes={off}-{}", off + len - 1));
            match req.call() {
                Ok(r) if r.status() == 206 => return read_exact(r.into_reader(), len),
                Ok(r) => {
                    let mut rd = r.into_reader();
                    std::io::copy(&mut (&mut rd).take(off), &mut std::io::sink()).map_err(|e| format!("pack download: {e}"))?;
                    return read_exact(rd, len);
                }
                Err(ureq::Error::Status(403, _)) if !fresh => fresh = true,
                Err(e) => return Err(format!("pack download: {e}")),
            }
        }
    }
}

fn read_all(r: impl Read) -> Result<Vec<u8>, String> {
    read_capped(r, MAX_INDEX)
}

fn read_capped(r: impl Read, max: u64) -> Result<Vec<u8>, String> {
    // A platform response (index, signature) read whole, refused past `max` bytes.
    let mut v = Vec::new();
    r.take(max + 1).read_to_end(&mut v).map_err(|e| e.to_string())?;
    if v.len() as u64 > max {
        return Err(format!("the platform's response is longer than {max} bytes; refused"));
    }
    Ok(v)
}

fn read_exact(r: impl Read, len: u64) -> Result<Vec<u8>, String> {
    let mut v = Vec::with_capacity(len.min(1 << 26) as usize);
    r.take(len).read_to_end(&mut v).map_err(|e| format!("pack download: {e}"))?;
    if v.len() as u64 != len {
        return Err(format!("pack download: {} of {len} bytes", v.len()));
    }
    Ok(v)
}

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

const INDEX_VERSION: u64 = 3;
const ENGINE_ABI_MAJOR: u64 = 3;
const ENGINE_RUNTIMES: &[(&str, u64)] = &[("gc-c", 1)];
const CATALOG_MISS_SECS: u64 = 300;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Offer {
    pub arch: String,
    pub hf_repo: String,
    pub revision: String,
    pub format: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Overlay {
    pub file: String,
    pub base: String,
    pub size: u64,
    pub sha256: String,
    pub sig: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Entry {
    pub offer: Offer,
    pub file: String,
    pub manifest_sha256: String,
    pub size: u64,
    pub sig: Value,
    pub aux: Value,
    pub drafter: Value,
    pub min_engine: (u64, u64, u64),
    pub overlays: Vec<Overlay>,
}

pub fn platform_base(home: &Path) -> Result<String, String> {
    // The platform this machine talks to for packs: the one it is linked to, else the configured default.
    match crate::platform::load_credentials(home) {
        Ok(Some(c)) => Ok(c.platform_url),
        _ => crate::platform::platform_url(),
    }
}

pub fn catalog(home: &Path) -> Option<Vec<u8>> {
    // The platform's public pack index (unverified bytes), revalidated with its ETag and cached under packs/catalog. Only decides what to offer: everything installed is verified by goethe-engine against the signed index again. None when the platform is unreachable and nothing is cached; a failure is remembered for a few minutes.
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
    let mut req = crate::net::agent().get(&format!("{}/packs/v2/index.json", base.trim_end_matches('/'))).timeout(std::time::Duration::from_secs(10));
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

pub fn offers(index: &[u8], model_id: &str, archs: &[String]) -> Vec<Offer> {
    // Packs of `index` for `model_id` (case-insensitive) on one of `archs`, in `archs` order, that the pinned engine can run.
    dedup(entries(index, archs, |repo| repo.eq_ignore_ascii_case(model_id)).into_iter().filter(runnable).map(|e| e.offer).collect())
}

pub fn offers_any(index: &[u8], archs: &[String]) -> Vec<Offer> {
    // Every pack of `index` on one of `archs` that the pinned engine can run, whatever the model.
    dedup(entries(index, archs, |_| true).into_iter().filter(runnable).map(|e| e.offer).collect())
}

fn dedup(v: Vec<Offer>) -> Vec<Offer> {
    let mut out: Vec<Offer> = Vec::new();
    for o in v {
        if !out.contains(&o) {
            out.push(o);
        }
    }
    out
}

pub fn runnable(e: &Entry) -> bool {
    // The pinned goethe-engine (the one goethe installs) is at least the pack's min_engine.
    Some(e.min_engine) <= version(crate::pyenv::ENGINE_VERSION)
}

pub fn engine_too_old(index: &[u8], model_id: &str, archs: &[String]) -> Option<String> {
    // A note when the index has packs for `model_id` on `archs` but each needs a newer goethe-engine than the pinned one.
    let es = entries(index, archs, |r| r.eq_ignore_ascii_case(model_id));
    if es.iter().any(runnable) {
        return None;
    }
    let (a, b, c) = es.iter().map(|e| e.min_engine).min()?;
    Some(format!("engine_too_old: {model_id} needs goethe-engine >= {a}.{b}.{c}, installed {}: run goethe update", crate::pyenv::ENGINE_VERSION))
}

pub fn pack_archs(index: &[u8], model_id: &str) -> Vec<String> {
    // The archs the index has a pack of `model_id` for, whatever GPUs this machine has.
    let ix: Value = serde_json::from_slice(index).unwrap_or_default();
    let all: Vec<String> = ix["entries"].as_array().into_iter().flatten().filter_map(|e| e["arch"].as_str().map(String::from)).collect();
    let mut out: Vec<String> = entries(index, &all, |r| r.eq_ignore_ascii_case(model_id)).into_iter().map(|e| e.offer.arch).collect();
    out.sort();
    out.dedup();
    out
}

pub fn built_for(index: &[u8], model_id: &str) -> String {
    // The GPU architectures (with their families) the index has a pack of `model_id` for.
    pack_archs(index, model_id).iter().map(|a| crate::gpu_arch::SUPPORTED.iter().find(|x| &x.tag() == a).map_or(a.clone(), |x| format!("{a}, {}", x.family))).collect::<Vec<_>>().join("; ")
}

pub fn unsupported_here(index: &[u8], model_id: &str, archs: &[String], local: &str) -> Option<String> {
    // One line when the index has packs of `model_id` but none for this machine's GPUs: which GPUs the packs are built for, and that this one is not among them.
    let theirs = pack_archs(index, model_id);
    if theirs.is_empty() || theirs.iter().any(|a| archs.contains(a)) {
        return None;
    }
    let built = built_for(index, model_id);
    let here = if local.is_empty() { archs.join(", ") } else { format!("{local} ({})", archs.join(", ")) };
    Some(format!("It runs only with a kernel pack, built for {built}; this machine has {here}, so it is not supported here and nothing was downloaded"))
}

pub fn entries(index: &[u8], archs: &[String], model: impl Fn(&str) -> bool) -> Vec<Entry> {
    // Index entries on one of `archs` (in that order) for a model `model` accepts, in a format, ABI and runtime this goethe runs, whatever min_engine they need (`runnable` filters).
    let Ok(ix) = serde_json::from_slice::<Value>(index) else { return Vec::new() };
    if ix["schema"] != "gc-pack-index" || ix["version"] != INDEX_VERSION {
        return Vec::new();
    }
    let mut out = Vec::new();
    for arch in archs {
        for e in ix["entries"].as_array().into_iter().flatten() {
            let m = &e["model"];
            let runs = ENGINE_RUNTIMES.iter().any(|(k, v)| e["runtime"]["kind"] == *k && e["runtime"]["version"] == *v);
            let file = e["file"].as_str().unwrap_or_default();
            let manifest = e["manifest_sha256"].as_str().unwrap_or_default();
            let Some(min_engine) = e["min_engine"].as_str().and_then(version) else { continue };
            let fits = e["arch"].as_str() == Some(arch.as_str())
                && e["abi_major"].as_u64() == Some(ENGINE_ABI_MAJOR)
                && runs
                && m["hf_repo"].as_str().is_some_and(&model)
                && pack_file_name(file)
                && is_sha256(manifest)
                && e["size"].as_u64().is_some();
            if !fits {
                continue;
            }
            out.push(Entry {
                offer: Offer {
                    arch: arch.clone(),
                    hf_repo: m["hf_repo"].as_str().unwrap_or_default().to_string(),
                    revision: m["revision"].as_str().filter(|r| !r.is_empty()).unwrap_or("main").to_string(),
                    format: m["format"].as_str().unwrap_or_default().to_string(),
                },
                file: file.to_string(),
                manifest_sha256: manifest.to_string(),
                size: e["size"].as_u64().unwrap_or(0),
                sig: e["sig"].clone(),
                aux: e["aux"].clone(),
                drafter: e["drafter"].clone(),
                min_engine,
                overlays: overlays_of(e),
            });
        }
    }
    out
}

fn overlays_of(e: &Value) -> Vec<Overlay> {
    // The overlays an index entry lists: a small signed file per base manifest, with its size and sha256; a malformed one is left out.
    e["overlays"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|o| {
            let (file, base, sha256) = (o["file"].as_str()?, o["base"].as_str()?, o["sha256"].as_str()?);
            (safe_name(file) && is_sha256(base) && is_sha256(sha256) && o["size"].as_u64().is_some_and(|n| n > 0 && n < (64 << 20))).then_some(())?;
            Some(Overlay { file: file.to_string(), base: base.to_string(), size: o["size"].as_u64()?, sha256: sha256.to_string(), sig: o["sig"].clone() })
        })
        .collect()
}

fn version(s: &str) -> Option<(u64, u64, u64)> {
    let mut it = s.split('.').map(|x| x.parse::<u64>().ok());
    let v = (it.next()??, it.next()??, it.next()??);
    it.next().is_none().then_some(v)
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn pack_file_name(f: &str) -> bool {
    // The pack's asset name (goethe-engine's slug): it becomes a URL path segment, never a local path.
    f.len() <= 134
        && f.ends_with(".gcpack")
        && f.len() > ".gcpack".len()
        && f.as_bytes()[0].is_ascii_alphanumeric()
        && f.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

pub fn aux(e: &Entry) -> Option<(String, Option<String>, Vec<String>)> {
    // The model's own config and tokenizer files the signed entry names (repo, revision, files): a GGUF repo ships none.
    let repo = e.aux["repo"].as_str().filter(|r| crate::paths::validate_model_id(r).is_ok())?;
    let files: Vec<String> = e.aux["files"].as_array()?.iter().filter_map(|f| f.as_str()).filter(|f| safe_name(f)).map(String::from).collect();
    (!files.is_empty()).then(|| (repo.to_string(), e.aux["revision"].as_str().map(String::from), files))
}

pub struct Drafter {
    pub name: String,
    pub repo: String,
    pub revision: Option<String>,
    pub files: Vec<(String, u64, String)>,
}

pub fn drafter(e: &Entry) -> Option<Drafter> {
    // The speculative-decoding drafter the signed entry names: aux's rules, plus each file's bytes and sha256; any bad field drops it whole.
    let d = &e.drafter;
    let repo = d["repo"].as_str().filter(|r| crate::paths::validate_model_id(r).is_ok())?.to_string();
    let files = d["files"].as_array()?.iter().map(|f| Some((f["name"].as_str().filter(|n| safe_name(n))?.to_string(), f["bytes"].as_u64()?, f["sha256"].as_str().filter(|h| is_sha256(h))?.to_string())));
    let files: Vec<(String, u64, String)> = files.collect::<Option<_>>().filter(|f: &Vec<_>| !f.is_empty())?;
    Some(Drafter { name: d["name"].as_str().unwrap_or("drafter").to_string(), repo, revision: d["revision"].as_str().map(String::from), files })
}

fn sha256_file(p: &Path) -> Option<String> {
    let mut h = Sha256::new();
    std::io::copy(&mut std::fs::File::open(p).ok()?, &mut h).ok()?;
    Some(format!("{:x}", h.finalize()))
}

pub fn safe_name(f: &str) -> bool {
    // A plain file name in the repo root: the download and the model dir never see a path from a pack.
    !f.is_empty() && !f.contains(['/', '\\']) && f != "." && f != ".." && !f.starts_with('.')
}

pub fn local_archs(home: &Path) -> Vec<String> {
    // The distinct pack architectures (sm_XX tags) of this machine's usable GPUs, most GPUs first.
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

pub fn arch_ccs(arch: &str) -> Vec<String> {
    // The compute capabilities whose pack architecture is `arch`: the card's supported_sm, so placement keeps the model on GPUs the pack was built for.
    crate::gpu_arch::supported_list().into_iter().filter(|cc| crate::gpu_arch::lookup(cc).is_some_and(|a| a.tag() == arch)).collect()
}

pub fn model_gb(home: &Path, o: &Offer) -> Option<f64> {
    // Download size of a pack's model when its repo has one model file (projector files aside), cached per repo revision.
    let path = home.join("packs").join("catalog").join("sizes.json");
    let key = format!("{}@{}", o.hf_repo, o.revision);
    let mut cache: serde_json::Map<String, Value> = std::fs::read(&path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    if let Some(v) = cache.get(&key) {
        return v.as_f64();
    }
    if crate::hf::offline_mode() {
        return None;
    }
    let pre = crate::hf::preflight_model(&o.hf_repo, Some(&o.revision)).ok()?;
    let ext = if o.format == "gguf" { ".gguf" } else { ".safetensors" };
    let models: Vec<u64> = pre.siblings.iter().filter(|s| s.rfilename.ends_with(ext) && !s.rfilename.contains("mmproj")).map(|s| s.size).collect();
    let gb = match (o.format.as_str(), models.as_slice()) {
        ("gguf", [one]) => Some(*one as f64 / crate::model_facts::GIB),
        ("safetensors", all) if !all.is_empty() => Some(all.iter().sum::<u64>() as f64 / crate::model_facts::GIB),
        _ => None,
    };
    cache.insert(key, gb.map_or(Value::Null, Value::from));
    let _ = std::fs::write(&path, Value::Object(cache).to_string());
    gb
}

pub fn store_serial(home: &Path, serial: u64) -> Result<(), String> {
    // Remember the newest index serial installed, so an older index is refused.
    std::fs::write(serial_path(home), format!("{serial}\n")).map_err(|e| e.to_string())
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

#[derive(Debug, Clone)]
pub struct Plan {
    pub entry: Entry,
    pub prefix: Vec<u8>,
    pub variants: Vec<Vec<String>>,
    pub serial: u64,
}

pub fn plan(home: &Path, model_id: &str, arch: &str, quant: Option<&str>, broker: &dyn Broker, run: Runner) -> Result<Option<Plan>, String> {
    // The signed index's pack for this model and arch, and its manifest's weight files per variant (what to download), only the variant a quant names (id, desc.quant or desc.format, any case). None: no pack.
    let work = home.join("packs");
    std::fs::create_dir_all(&work).map_err(|e| e.to_string())?;
    let (ix, sg) = (work.join("index.json"), work.join("index.sig"));
    let raw = broker.index()?;
    std::fs::write(&ix, &raw).map_err(|e| e.to_string())?;
    std::fs::write(&sg, broker.index_sig()?).map_err(|e| e.to_string())?;
    let v = |x: &str| x.to_string();
    let verified = run(&[v("index"), v("verify"), v("--index"), s(&ix), v("--sig"), s(&sg), v("--last-serial"), last_serial(home).to_string()])?.1;
    let serial = serde_json::from_str::<Value>(&verified).map_err(|e| format!("index verify: {e}"))?["serial"].as_u64().ok_or("index verify: no serial")?;
    let Some(entry) = entries(&raw, &[arch.to_string()], |r| r.eq_ignore_ascii_case(model_id)).into_iter().find(runnable) else { return Ok(None) };
    let head = broker.range(&entry.file, 0, 16)?;
    if &head[..8] != b"GCPACK3\n" {
        return Err(format!("{}: not a pack of ABI 3", entry.file));
    }
    let n = u64::from_le_bytes(head[8..16].try_into().map_err(|_| "pack header")?);
    if n == 0 || n > MAX_HEADER || 16 + n > entry.size {
        return Err(format!("{}: bad container header length", entry.file));
    }
    let header = broker.range(&entry.file, 16, n)?;
    let h: Value = serde_json::from_slice(&header).map_err(|e| format!("{}: container header: {e}", entry.file))?;
    let (off, len) = (h["manifest"][0].as_u64().ok_or("container header: manifest")?, h["manifest"][1].as_u64().ok_or("container header: manifest")?);
    let end = off.checked_add(len).and_then(|x| x.checked_add(16 + n));
    if len == 0 || len > MAX_MANIFEST || !end.is_some_and(|e| e <= entry.size) {
        return Err(format!("{}: bad manifest range", entry.file));
    }
    let body = broker.range(&entry.file, 16 + n, off + len)?;
    let manifest = &body[off as usize..];
    if format!("{:x}", Sha256::digest(manifest)) != entry.manifest_sha256 {
        return Err(format!("{}: manifest differs from the signed index", entry.file));
    }
    let m: Value = serde_json::from_slice(manifest).map_err(|e| format!("{}: manifest: {e}", entry.file))?;
    let names: Vec<String> = m["model"]["weights"].as_array().into_iter().flatten().map(|w| w["file"].as_str().unwrap_or_default().to_string()).collect();
    let (mut variants, mut all): (Vec<Vec<String>>, Vec<&str>) = (Vec::new(), Vec::new());
    for (id, var) in m["variants"].as_object().into_iter().flatten() {
        let d = &var["desc"];
        all.push(d["quant"].as_str().unwrap_or(id));
        if quant.is_some_and(|q| ![Some(id.as_str()), d["quant"].as_str(), d["format"].as_str()].into_iter().flatten().any(|t| t.eq_ignore_ascii_case(q))) {
            continue;
        }
        let files: Vec<String> = var["weights"].as_array().into_iter().flatten().filter_map(|i| names.get(i.as_u64()? as usize).cloned()).collect();
        if !files.is_empty() && files.iter().all(|f| safe_name(f)) && !variants.contains(&files) {
            variants.push(files);
        }
    }
    if variants.is_empty() {
        return Err(match quant {
            Some(q) => format!("{model_id}: its kernel pack has no variant {q} (available: {})", all.join(", ")),
            None => format!("{}: the manifest names no model files", entry.file),
        });
    }
    let prefix = [head, header, body].concat();
    Ok(Some(Plan { entry, prefix, variants, serial }))
}

pub fn by_structure(home: &Path, dir: &Path, archs: &[String], broker: &dyn Broker, run: Runner) -> Result<Option<(String, Plan)>, String> {
    // The signed index's pack, on the first of `archs` with one, whose structure (tensor table and metadata keys, never weight values) the weights in `dir` have, whatever repo they came from.
    let work = home.join("packs");
    std::fs::create_dir_all(&work).map_err(|e| e.to_string())?;
    let (ix, sg) = (work.join("index.json"), work.join("index.sig"));
    std::fs::write(&ix, broker.index()?).map_err(|e| e.to_string())?;
    std::fs::write(&sg, broker.index_sig()?).map_err(|e| e.to_string())?;
    let v = |x: &str| x.to_string();
    for arch in archs {
        let (code, out) = run(&[v("pack"), v("match"), v("--index"), s(&ix), v("--sig"), s(&sg), v("--model-dir"), s(dir), v("--arch"), v(arch), v("--last-serial"), last_serial(home).to_string()])?;
        if code == 3 {
            continue;
        }
        let hit: Value = serde_json::from_str(&out).map_err(|e| format!("pack match: {e}"))?;
        let repo = hit["model"]["hf_repo"].as_str().ok_or("pack match: no model.hf_repo")?;
        if let Some(p) = plan(home, repo, arch, None, broker, run)?.filter(|p| hit["file"] == p.entry.file.as_str()) {
            return Ok(Some((arch.clone(), p)));
        }
    }
    Ok(None)
}

fn merged(mut ranges: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    // Byte ranges sorted, with neighbours closer than MERGE_GAP joined, so a partial download takes few requests.
    ranges.sort();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (o, l) in ranges {
        match out.last_mut() {
            Some((po, pl)) if o <= *po + *pl + MERGE_GAP => *pl = (*pl).max(o + l - *po),
            _ => out.push((o, l)),
        }
    }
    out
}

pub fn finish(home: &Path, dir: &Path, arch: &str, plan: &Plan, broker: &dyn Broker, run: Runner) -> Result<String, String> {
    // Match the downloaded model files, fetch only the byte ranges of the pack they need, verify, store the serial; the variant id.
    let (variant, _) = finish_as(home, dir, arch, plan, broker, run, Next { suffix: "", donor: None })?;
    store_serial(home, plan.serial)?;
    Ok(variant)
}

pub struct Donor {
    pub pack: PathBuf,
    pub ranges: Vec<(u64, u64, String)>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Delta {
    pub fetched: u64,
    pub reused: u64,
}

#[derive(Clone, Copy)]
pub struct Next<'a> {
    pub suffix: &'a str,
    pub donor: Option<&'a Donor>,
}

pub fn finish_as(
    home: &Path,
    dir: &Path,
    arch: &str,
    plan: &Plan,
    broker: &dyn Broker,
    run: Runner,
    next: Next,
) -> Result<(String, Delta), String> {
    // finish into files named with `suffix` (an update builds the next pack beside the installed one); blocks the donor pack already holds by hash are copied, only the rest is fetched.
    let Next { suffix, donor } = next;
    let e = &plan.entry;
    let (pack, psig, pv, pre) = (
        dir.join(format!("{PACK_FILE}{suffix}")),
        dir.join(format!("{PACK_FILE}.sig{suffix}")),
        dir.join(format!("{PACK_VERIFY}{suffix}")),
        dir.join(format!("pack.prefix{suffix}")),
    );
    let mut delta = Delta::default();
    let undo = |err: String| {
        for p in [&pack, &psig, &pv, &pre] {
            let _ = std::fs::remove_file(p);
        }
        err
    };
    let work = home.join("packs");
    let v = |x: &str| x.to_string();
    let (code, out) = run(&[
        v("pack"), v("match"), v("--index"), s(&work.join("index.json")), v("--sig"), s(&work.join("index.sig")),
        v("--model-dir"), s(dir), v("--arch"), v(arch), v("--last-serial"), last_serial(home).to_string(),
    ]).map_err(undo)?;
    if code == 3 {
        return Err(undo(format!("the downloaded files of {} do not match its kernel pack", e.offer.hf_repo)));
    }
    let hit: Value = serde_json::from_str(&out).map_err(|err| undo(format!("pack match: {err}")))?;
    if hit["file"].as_str() != Some(e.file.as_str()) {
        return Err(undo(format!("pack match chose {} instead of {}", hit["file"], e.file)));
    }
    let variant = hit["variant"].as_str().unwrap_or_default().to_string();
    std::fs::write(&pre, &plan.prefix).map_err(|err| undo(err.to_string()))?;
    let need: Value = serde_json::from_str(&run(&[v("pack"), v("need"), s(&pre), v("--model-dir"), s(dir), v("--arch"), v(arch)]).map_err(undo)?.1)
        .map_err(|err| undo(format!("pack need: {err}")))?;
    if need["manifest_sha256"].as_str() != Some(e.manifest_sha256.as_str()) {
        return Err(undo("pack need: the manifest differs from the signed index".into()));
    }
    let ranges: Vec<(u64, u64, String, bool)> = need["ranges"].as_array().into_iter().flatten()
        .filter_map(|r| Some((r["off"].as_u64()?, r["len"].as_u64()?, r["sha256"].as_str()?.to_string(), matches!(r["codec"].as_str(), None | Some("none") | Some("raw")))))
        .collect();
    if ranges.iter().any(|(o, l, h, _)| !o.checked_add(*l).is_some_and(|end| end <= e.size) || !is_sha256(h)) {
        return Err(undo("pack need: a range lies outside the pack".into()));
    }
    let sig = serde_json::to_vec(&e.sig).map_err(|err| undo(err.to_string()))?;
    std::fs::write(&psig, sig).map_err(|err| undo(err.to_string()))?;
    let verify = || run(&[v("pack"), v("verify"), s(&pack), v("--sig"), s(&psig), v("--arch"), v(arch), v("--partial"), v("--model-dir"), s(dir)]);
    let reuse = std::fs::metadata(&pack).is_ok_and(|m| m.len() == e.size) && verify().is_ok_and(|(c, _)| c == 0);
    if !reuse {
        let mut f = std::fs::File::create(&pack).map_err(|err| undo(format!("{}: {err}", pack.display())))?;
        f.set_len(e.size).map_err(|err| undo(err.to_string()))?;
        f.write_all(&plan.prefix).map_err(|err| undo(err.to_string()))?;
        let mut missing = Vec::new();
        for r in &ranges {
            let held = donor.and_then(|d| d.ranges.iter().find(|(_, l, h)| *l == r.1 && *h == r.2).map(|(from, ..)| (d, *from)));
            let copied = match held {
                Some((d, from)) if r.3 => {
                    let mut bytes = vec![0u8; r.1 as usize];
                    let read = std::fs::File::open(&d.pack).and_then(|mut old| old.seek(SeekFrom::Start(from)).and_then(|_| std::io::Read::read_exact(&mut old, &mut bytes)));
                    let good = read.is_ok() && format!("{:x}", Sha256::digest(&bytes)) == r.2;
                    if good {
                        f.seek(SeekFrom::Start(r.0)).and_then(|_| f.write_all(&bytes)).map_err(|err| undo(err.to_string()))?;
                        delta.reused += r.1;
                    }
                    good
                }
                _ => false,
            };
            if !copied {
                missing.push(r.clone());
            }
        }
        for (o, l) in merged(missing.iter().map(|(o, l, ..)| (*o, *l)).collect()) {
            delta.fetched += l;
            let bytes = broker.range(&e.file, o, l).map_err(undo)?;
            for (ro, rl, h, _) in missing.iter().filter(|(ro, rl, _, stored)| *stored && *ro >= o && ro + rl <= o + l) {
                let part = &bytes[(ro - o) as usize..(ro - o + rl) as usize];
                if format!("{:x}", Sha256::digest(part)) != *h {
                    return Err(undo(format!("pack download: a block of {} does not hash to its name", e.file)));
                }
            }
            f.seek(SeekFrom::Start(o)).and_then(|_| f.write_all(&bytes)).map_err(|err| undo(err.to_string()))?;
        }
        f.sync_all().map_err(|err| undo(err.to_string()))?;
    }
    let mut ok = verify().map_err(undo)?.1;
    if let Some(d) = drafter(e).filter(|d| dir.join("drafter").join(&d.files[0].0).is_file()) {
        match d.files.iter().all(|(n, _, h)| sha256_file(&dir.join("drafter").join(n)).as_deref() == Some(h.as_str())) {
            true => {
                let mut v: Value = serde_json::from_str(&ok).map_err(|err| undo(format!("pack verify: {err}")))?;
                v["drafter"] = d.files.iter().map(|(n, ..)| format!("drafter/{n}")).collect();
                ok = v.to_string();
            }
            false => eprintln!("note: the {} drafter files differ from the signed index; speculative decoding off", d.name),
        }
    }
    std::fs::write(&pv, ok).map_err(|err| undo(err.to_string()))?;
    let _ = std::fs::remove_file(&pre);
    Ok((variant, delta))
}

pub fn max_ctx(dir: &Path, variant: &str, run: Runner) -> Result<u64, String> {
    // The longest context a bucket of `variant` in the verified pack serves (`goethe-engine pack show`): the card's ctx cap.
    let (_, out) = run(&["pack".into(), "show".into(), s(&dir.join(PACK_FILE))])?;
    out.lines()
        .filter(|l| variant.is_empty() || l.split_whitespace().nth(1) == Some(variant))
        .filter_map(|l| l.split_whitespace().skip_while(|w| *w != "maxctx").nth(1)?.parse::<u64>().ok())
        .max()
        .ok_or_else(|| "pack show: no bucket lists a maxctx".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;

    const BLOB: &[u8] = b"kernel blob bytes";

    fn sha(b: &[u8]) -> String {
        format!("{:x}", Sha256::digest(b))
    }

    struct Pack {
        bytes: Vec<u8>,
        manifest: Vec<u8>,
        blob_off: u64,
    }

    fn container() -> Pack {
        let manifest = serde_json::json!({"model": {"weights": [{"file": "m-Q8_0.gguf"}, {"file": "m-Q4_0.gguf"}]},
            "variants": {"q8": {"weights": [0]}, "q4": {"weights": [1]}}}).to_string().into_bytes();
        let header = serde_json::json!({"manifest": [0, manifest.len()], "blobs": {sha(BLOB): [manifest.len() + 100, BLOB.len()]}}).to_string().into_bytes();
        let mut bytes = b"GCPACK3\n".to_vec();
        bytes.extend((header.len() as u64).to_le_bytes());
        bytes.extend(&header);
        bytes.extend(&manifest);
        bytes.extend(vec![0u8; 100]);
        let blob_off = bytes.len() as u64;
        bytes.extend(BLOB);
        Pack { bytes, manifest, blob_off }
    }

    fn index_of(p: &Pack, manifest_sha: &str) -> Vec<u8> {
        serde_json::json!({"schema": "gc-pack-index", "version": 3, "serial": 7, "entries": [{
            "model": {"hf_repo": "org/M-GGUF", "revision": "abc", "format": "gguf", "structure": []},
            "arch": "sm_120a", "abi_major": 3, "runtime": {"kind": "gc-c", "version": 1}, "min_engine": "0.2.0", "tier": "base",
            "file": "m.gcpack", "manifest_sha256": manifest_sha, "size": p.bytes.len(), "sig": {"keyid": "k"},
            "aux": {"repo": "org/M", "revision": "def", "files": ["config.json", "tokenizer.json", "../x"]},
            "drafter": {"name": "DFlash2", "repo": "org/D", "revision": "r", "files": [{"name": "d.gguf", "bytes": 3, "sha256": sha(b"abc")}]}}]})
        .to_string()
        .into_bytes()
    }

    struct Fake {
        pack: Pack,
        index: Vec<u8>,
        tamper: bool,
        refuse: bool,
        ranges: RefCell<Vec<(u64, u64)>>,
    }

    fn fake() -> Fake {
        let pack = container();
        let index = index_of(&pack, &sha(&pack.manifest));
        Fake { pack, index, tamper: false, refuse: false, ranges: RefCell::new(Vec::new()) }
    }

    impl Broker for Fake {
        fn index(&self) -> Result<Vec<u8>, String> {
            Ok(self.index.clone())
        }
        fn index_sig(&self) -> Result<Vec<u8>, String> {
            Ok(b"sig".to_vec())
        }
        fn range(&self, _: &str, off: u64, len: u64) -> Result<Vec<u8>, String> {
            if self.refuse {
                return Err("the platform did not accept this node's login".into());
            }
            self.ranges.borrow_mut().push((off, len));
            let mut b = self.pack.bytes[off as usize..(off + len) as usize].to_vec();
            if self.tamper && off + len > self.pack.blob_off {
                let last = b.len() - 1;
                b[last] ^= 1;
            }
            Ok(b)
        }
    }

    fn engine(p: &Pack, codes: &[(&str, i32)]) -> impl Fn(&[String]) -> Result<(i32, String), String> {
        // A fake goethe-engine: exit code and stdout per subcommand ("index verify", "pack match", "pack need", "pack verify").
        let codes: HashMap<String, i32> = codes.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        let (msha, off) = (sha(&p.manifest), p.blob_off);
        move |a| {
            let key = format!("{} {}", a[0], a[1]);
            let code = codes.get(&key).copied().unwrap_or(0);
            if code == 1 {
                return Err(format!("goethe-engine {key}: exit 1"));
            }
            let out = match key.as_str() {
                "index verify" => r#"{"serial": 7}"#.to_string(),
                "pack match" => r#"{"file": "m.gcpack", "variant": "q8", "model": {"hf_repo": "org/M-GGUF"}}"#.to_string(),
                "pack need" => serde_json::json!({"manifest_sha256": msha, "ranges": [{"off": off, "len": BLOB.len(), "sha256": sha(BLOB)}]}).to_string(),
                "pack show" => "a q8 K1 ncol 1 maxctx 4096 3 files\nb q4 K1 ncol 1 maxctx 8192 3 files\n".to_string(),
                _ => r#"{"ok": true, "weights": ["m-Q8_0.gguf"]}"#.to_string(),
            };
            Ok((code, out))
        }
    }

    fn dirs() -> (tempfile::TempDir, tempfile::TempDir) {
        (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap())
    }

    fn nothing_left(dir: &Path) {
        for f in [PACK_FILE.to_string(), format!("{PACK_FILE}.sig"), PACK_VERIFY.to_string(), "pack.prefix".to_string()] {
            assert!(!dir.join(&f).exists(), "{f} left behind");
        }
    }

    #[test]
    fn a_platform_response_past_the_cap_is_refused() {
        assert_eq!(read_capped(std::io::repeat(b'x').take(10), 10).unwrap().len(), 10);
        assert!(read_capped(std::io::repeat(b'x'), 1 << 20).unwrap_err().contains("longer than"));
    }

    #[test]
    fn a_plan_reads_the_manifest_and_names_the_files_of_each_variant() {
        let (home, _) = dirs();
        let b = fake();
        let p = plan(home.path(), "org/m-gguf", "sm_120a", None, &b, &engine(&b.pack, &[])).unwrap().unwrap();
        assert_eq!(p.variants, vec![vec!["m-Q4_0.gguf".to_string()], vec!["m-Q8_0.gguf".to_string()]]);
        assert_eq!(p.prefix, b.pack.bytes[..p.prefix.len()]);
        let files = vec!["config.json".to_string(), "tokenizer.json".to_string()];
        assert_eq!(aux(&p.entry), Some(("org/M".to_string(), Some("def".to_string()), files)), "a path in aux files is dropped");
        let q8 = plan(home.path(), "org/m-gguf", "sm_120a", Some("Q8"), &b, &engine(&b.pack, &[])).unwrap().unwrap();
        assert_eq!(q8.variants, vec![vec!["m-Q8_0.gguf".to_string()]], "a quant names one variant by id");
        let e = plan(home.path(), "org/m-gguf", "sm_120a", Some("q5"), &b, &engine(&b.pack, &[])).unwrap_err();
        assert_eq!(e, "org/m-gguf: its kernel pack has no variant q5 (available: q4, q8)");
        assert!(plan(home.path(), "org/other", "sm_120a", None, &b, &engine(&b.pack, &[])).unwrap().is_none());
        assert!(plan(home.path(), "org/m-gguf", "sm_90a", None, &b, &engine(&b.pack, &[])).unwrap().is_none());
    }

    #[test]
    fn weights_of_another_repo_find_the_pack_whose_structure_they_have() {
        let (home, dir) = dirs();
        let b = fake();
        let archs = vec!["sm_90a".to_string(), "sm_120a".to_string()];
        let (arch, p) = by_structure(home.path(), dir.path(), &archs, &b, &engine(&b.pack, &[])).unwrap().unwrap();
        assert_eq!((arch.as_str(), p.entry.file.as_str(), p.entry.offer.hf_repo.as_str()), ("sm_120a", "m.gcpack", "org/M-GGUF"), "sm_90a has no entry");
        assert!(by_structure(home.path(), dir.path(), &archs, &b, &engine(&b.pack, &[("pack match", 3)])).unwrap().is_none());
        assert!(by_structure(home.path(), dir.path(), &archs, &b, &engine(&b.pack, &[("pack match", 1)])).is_err());
    }

    struct Spy(Fake);

    impl Broker for Spy {
        fn index(&self) -> Result<Vec<u8>, String> {
            self.0.index()
        }
        fn index_sig(&self) -> Result<Vec<u8>, String> {
            self.0.index_sig()
        }
        fn range(&self, file: &str, off: u64, len: u64) -> Result<Vec<u8>, String> {
            if off <= 16 {
                return self.0.range(file, off, len);
            }
            Err(format!("asked for {len} bytes at {off}"))
        }
    }

    #[test]
    fn a_container_header_whose_manifest_range_overflows_is_refused_before_any_request() {
        let (home, _) = dirs();
        let header = serde_json::json!({"manifest": [u64::MAX - 100, 200]}).to_string().into_bytes();
        let mut bytes = b"GCPACK3\n".to_vec();
        bytes.extend((header.len() as u64).to_le_bytes());
        bytes.extend(&header);
        let pack = Pack { manifest: b"{}".to_vec(), blob_off: 0, bytes };
        let index = index_of(&pack, &sha(&pack.manifest));
        let b = Spy(Fake { pack, index, tamper: false, refuse: false, ranges: RefCell::new(Vec::new()) });
        let e = plan(home.path(), "org/m-gguf", "sm_120a", None, &b, &engine(&b.0.pack, &[])).unwrap_err();
        assert!(e.contains("bad manifest range"), "{e}");
    }

    #[test]
    fn a_needed_range_that_overflows_is_refused_not_wrapped() {
        let (home, dir) = dirs();
        let b = fake();
        let inner = engine(&b.pack, &[]);
        let run = move |a: &[String]| {
            let (c, out) = inner(a)?;
            if a[0] == "pack" && a[1] == "need" {
                let mut v: Value = serde_json::from_str(&out).map_err(|e| e.to_string())?;
                v["ranges"] = serde_json::json!([{"off": u64::MAX - 5, "len": 10, "sha256": sha(BLOB)}]);
                return Ok((c, v.to_string()));
            }
            Ok((c, out))
        };
        let p = plan(home.path(), "org/m-gguf", "sm_120a", None, &b, &run).unwrap().unwrap();
        let e = finish(home.path(), dir.path(), "sm_120a", &p, &b, &run).unwrap_err();
        assert!(e.contains("outside the pack"), "{e}");
        nothing_left(dir.path());
    }

    #[test]
    fn a_manifest_that_differs_from_the_signed_index_is_refused() {
        let (home, _) = dirs();
        let mut b = fake();
        b.index = index_of(&b.pack, &"0".repeat(64));
        assert!(plan(home.path(), "org/m-gguf", "sm_120a", None, &b, &engine(&b.pack, &[])).unwrap_err().contains("manifest differs"));
    }

    #[test]
    fn an_index_the_engine_refuses_fails_before_any_download() {
        let (home, _) = dirs();
        let b = fake();
        assert!(plan(home.path(), "org/m-gguf", "sm_120a", None, &b, &engine(&b.pack, &[("index verify", 1)])).is_err());
        assert!(b.ranges.borrow().is_empty());
    }

    #[test]
    fn a_compressed_block_is_named_by_its_raw_bytes_so_only_pack_verify_checks_it() {
        // The real engine names an xz block by the sha256 of what it decompresses to (`pack need` says codec xz); hashing the stored bytes would refuse every real pack.
        let (home, dir) = dirs();
        let b = fake();
        let inner = engine(&b.pack, &[]);
        let run = move |a: &[String]| {
            let (c, out) = inner(a)?;
            if a[0] == "pack" && a[1] == "need" {
                let mut v: Value = serde_json::from_str(&out).map_err(|e| e.to_string())?;
                for r in v["ranges"].as_array_mut().into_iter().flatten() {
                    r["codec"] = "xz".into();
                    r["sha256"] = sha(b"the raw bytes").into();
                }
                return Ok((c, v.to_string()));
            }
            Ok((c, out))
        };
        let p = plan(home.path(), "org/m-gguf", "sm_120a", None, &b, &run).unwrap().unwrap();
        assert_eq!(finish(home.path(), dir.path(), "sm_120a", &p, &b, &run).unwrap(), "q8");
        let got = std::fs::read(dir.path().join(PACK_FILE)).unwrap();
        assert_eq!(&got[b.pack.blob_off as usize..], BLOB, "the stored bytes are written as they are");
    }

    #[test]
    fn finish_fetches_only_the_needed_blob_and_verifies_and_stores_the_serial() {
        let (home, dir) = dirs();
        let b = fake();
        let run = engine(&b.pack, &[]);
        let p = plan(home.path(), "org/m-gguf", "sm_120a", None, &b, &run).unwrap().unwrap();
        let d = drafter(&p.entry).unwrap();
        assert_eq!((d.name.as_str(), d.repo.as_str(), d.files[0].1), ("DFlash2", "org/D", 3));
        std::fs::create_dir(dir.path().join("drafter")).unwrap();
        std::fs::write(dir.path().join("drafter/d.gguf"), b"abc").unwrap();
        b.ranges.borrow_mut().clear();
        assert_eq!(finish(home.path(), dir.path(), "sm_120a", &p, &b, &run).unwrap(), "q8");
        let v: Value = serde_json::from_slice(&std::fs::read(dir.path().join(PACK_VERIFY)).unwrap()).unwrap();
        assert_eq!(v["drafter"], serde_json::json!(["drafter/d.gguf"]), "a drafter with its signed sha256 is listed");
        std::fs::write(dir.path().join("drafter/d.gguf"), b"abd").unwrap();
        finish(home.path(), dir.path(), "sm_120a", &p, &b, &run).unwrap();
        let v: Value = serde_json::from_slice(&std::fs::read(dir.path().join(PACK_VERIFY)).unwrap()).unwrap();
        assert!(v["drafter"].is_null(), "a drafter that differs from the index is not listed");
        assert_eq!(*b.ranges.borrow(), vec![(b.pack.blob_off, BLOB.len() as u64)]);
        let got = std::fs::read(dir.path().join(PACK_FILE)).unwrap();
        assert_eq!(got.len(), b.pack.bytes.len());
        assert_eq!(&got[b.pack.blob_off as usize..], BLOB);
        assert!(dir.path().join(format!("{PACK_FILE}.sig")).is_file() && dir.path().join(PACK_VERIFY).is_file());
        assert!(!dir.path().join("pack.prefix").exists());
        assert_eq!(std::fs::read_to_string(home.path().join("packs/serial")).unwrap().trim(), "7");
        assert_eq!(max_ctx(dir.path(), "q8", &run).unwrap(), 4096);
    }

    #[test]
    fn a_pack_already_in_place_is_not_downloaded_again() {
        let (home, dir) = dirs();
        let b = fake();
        let run = engine(&b.pack, &[]);
        let p = plan(home.path(), "org/m-gguf", "sm_120a", None, &b, &run).unwrap().unwrap();
        std::fs::write(dir.path().join(PACK_FILE), &b.pack.bytes).unwrap();
        b.ranges.borrow_mut().clear();
        finish(home.path(), dir.path(), "sm_120a", &p, &b, &run).unwrap();
        assert!(b.ranges.borrow().is_empty());
    }

    #[test]
    fn a_tampered_block_fails_and_leaves_nothing() {
        let (home, dir) = dirs();
        let mut b = fake();
        let run = engine(&b.pack, &[]);
        let p = plan(home.path(), "org/m-gguf", "sm_120a", None, &b, &run).unwrap().unwrap();
        b.tamper = true;
        assert!(finish(home.path(), dir.path(), "sm_120a", &p, &b, &run).unwrap_err().contains("does not hash"));
        nothing_left(dir.path());
    }

    #[test]
    fn files_that_match_no_pack_and_a_refused_download_fail_and_leave_nothing() {
        let (home, dir) = dirs();
        let mut b = fake();
        let p = plan(home.path(), "org/m-gguf", "sm_120a", None, &b, &engine(&b.pack, &[])).unwrap().unwrap();
        assert!(finish(home.path(), dir.path(), "sm_120a", &p, &b, &engine(&b.pack, &[("pack match", 3)])).unwrap_err().contains("do not match"));
        nothing_left(dir.path());
        for f in [format!("{PACK_FILE}.sig"), PACK_VERIFY.to_string()] {
            std::fs::write(dir.path().join(f), b"old").unwrap();
        }
        b.refuse = true;
        assert!(finish(home.path(), dir.path(), "sm_120a", &p, &b, &engine(&b.pack, &[])).unwrap_err().contains("login"));
        nothing_left(dir.path());
        assert!(!home.path().join("packs/serial").exists());
    }

    #[test]
    fn a_pack_the_engine_refuses_is_removed() {
        let (home, dir) = dirs();
        let b = fake();
        let p = plan(home.path(), "org/m-gguf", "sm_120a", None, &b, &engine(&b.pack, &[])).unwrap().unwrap();
        assert!(finish(home.path(), dir.path(), "sm_120a", &p, &b, &engine(&b.pack, &[("pack verify", 1)])).is_err());
        nothing_left(dir.path());
    }

    #[test]
    fn a_pack_only_model_on_an_unbuilt_gpu_is_named_before_any_download() {
        let p = container();
        let ix = index_of(&p, &sha(&p.manifest));
        let here = |a: &str| vec![a.to_string()];
        let line = unsupported_here(&ix, "org/M-GGUF", &here("sm_89"), "NVIDIA GeForce RTX 4090").unwrap();
        for part in ["sm_120a", "Blackwell", "NVIDIA GeForce RTX 4090 (sm_89)", "not supported here", "nothing was downloaded"] {
            assert!(line.contains(part), "{part} in {line}");
        }
        assert_eq!(unsupported_here(&ix, "org/M-GGUF", &here("sm_120a"), "RTX 5090"), None, "a pack for this GPU");
        assert_eq!(unsupported_here(&ix, "org/other", &here("sm_89"), "RTX 4090"), None, "a model with no pack");
    }

    #[test]
    fn offers_follow_the_model_the_local_archs_and_what_the_engine_runs() {
        let archs = vec!["sm_120a".to_string(), "sm_90a".to_string()];
        let p = container();
        let mut ix: Value = serde_json::from_slice(&index_of(&p, &sha(&p.manifest))).unwrap();
        let base = ix["entries"][0].clone();
        let with = |k: &str, v: Value| {
            let mut e = base.clone();
            e[k] = v;
            e
        };
        ix["entries"] = serde_json::json!([
            with("abi_major", 2.into()), with("min_engine", "9.0.0".into()), with("file", "../x.gcpack".into()),
            with("arch", "sm_100a".into()), with("arch", "sm_90a".into()), base.clone()
        ]);
        let o = offers(ix.to_string().as_bytes(), "ORG/m-gguf", &archs);
        assert_eq!(o.iter().map(|o| o.arch.as_str()).collect::<Vec<_>>(), ["sm_120a", "sm_90a"]);
        assert_eq!(offers_any(ix.to_string().as_bytes(), &archs).len(), 2);
        assert_eq!(pack_archs(ix.to_string().as_bytes(), "org/M-GGUF"), ["sm_100a", "sm_120a", "sm_90a"]);
        ix["version"] = 2.into();
        assert!(offers(ix.to_string().as_bytes(), "org/m-gguf", &archs).is_empty());
        assert!(offers(b"not json", "org/m-gguf", &archs).is_empty());
    }

    #[test]
    fn a_pack_only_a_newer_engine_runs_is_named() {
        let archs = vec!["sm_120a".to_string()];
        let p = container();
        let mut ix: Value = serde_json::from_slice(&index_of(&p, &sha(&p.manifest))).unwrap();
        assert_eq!(engine_too_old(ix.to_string().as_bytes(), "org/m-gguf", &archs), None);
        ix["entries"][0]["min_engine"] = "9.0.0".into();
        let e = engine_too_old(ix.to_string().as_bytes(), "org/m-gguf", &archs).unwrap();
        assert!(e.starts_with("engine_too_old: org/m-gguf needs goethe-engine >= 9.0.0, installed "), "{e}");
    }

    #[test]
    fn neighbouring_ranges_are_fetched_together() {
        assert_eq!(merged(vec![(100, 10), (0, 10), (MERGE_GAP * 5, 1)]), vec![(0, 110), (MERGE_GAP * 5, 1)]);
    }

    #[test]
    fn the_card_names_the_compute_capabilities_of_the_pack_arch() {
        let ccs = arch_ccs("sm_120a");
        assert!(ccs.contains(&"12.0".to_string()) && !ccs.contains(&"9.0".to_string()));
    }

    struct Tok(Result<String, String>);
    impl TokenSource for Tok {
        fn token(&self) -> Result<String, String> {
            self.0.clone()
        }
    }

    fn serve(reply: String) -> (String, std::thread::JoinHandle<String>) {
        // One-shot HTTP server: replies to one request, returns the request head it saw.
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
    fn the_token_goes_to_the_platform_and_the_range_request_to_the_link_without_it() {
        let (target, target_seen) = serve("HTTP/1.1 206 Partial Content\r\nContent-Length: 4\r\nConnection: close\r\n\r\nabcd".into());
        let (store, store_seen) = serve(format!("HTTP/1.1 302 Found\r\nLocation: {target}/blob\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"));
        let tok = Tok(Ok("SECRET".into()));
        let b = HttpBroker::new(&store, &tok);
        assert_eq!(b.range("m.gcpack", 10, 4).unwrap(), b"abcd");
        let (s, t) = (store_seen.join().unwrap(), target_seen.join().unwrap());
        assert!(s.contains("GET /packs/v2/pack/m.gcpack") && s.contains("Bearer SECRET"));
        assert!(t.contains("GET /blob") && t.to_ascii_lowercase().contains("range: bytes=10-13") && !t.contains("SECRET"));
    }

    #[test]
    fn http_401_and_a_missing_login_are_errors() {
        let (store, _h) = serve("HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".into());
        let tok = Tok(Ok("bad".into()));
        assert!(HttpBroker::new(&store, &tok).index().unwrap_err().contains("goethe login"));
        let none = Tok(Err("not logged in".into()));
        assert!(HttpBroker::new("http://127.0.0.1:1", &none).range("m.gcpack", 0, 16).unwrap_err().contains("not logged in"));
    }
}
