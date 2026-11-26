// Kernel pack updates for installed models: only the blocks the installed pack lacks are fetched, the swap is atomic, one rollback is kept.

use crate::engines::goethe_engine::{PACK_FILE, PACK_VERIFY};
use crate::packs::{Broker, Delta, Donor, Entry, Runner};
use serde_json::Value;
use std::path::{Path, PathBuf};

const PREV: &str = ".prev";
const NEW: &str = ".new";
const OVERLAYS: &str = "pack.overlays.json";

#[derive(Debug, PartialEq, Eq)]
pub enum Outcome {
    Unchanged,
    Updated(Delta),
    Overlaid(u64),
    Skipped(String),
}

fn sidecars() -> [String; 3] {
    [PACK_FILE.to_string(), format!("{PACK_FILE}.sig"), PACK_VERIFY.to_string()]
}

fn installed_manifest(dir: &Path) -> Option<String> {
    // The manifest sha256 the installed pack was verified under.
    let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join(PACK_VERIFY)).ok()?).ok()?;
    v["manifest_sha256"].as_str().map(String::from)
}

pub fn pending(index: &[u8], model_id: &str, arch: &str, installed: &str, have_overlays: &[String]) -> Option<Entry> {
    // The index entry for this model and arch when it names another pack than the installed one, or overlays the installed one lacks; no network, the index is already local.
    let e = crate::packs::entries(index, &[arch.to_string()], |r| r.eq_ignore_ascii_case(model_id)).into_iter().next()?;
    let new_overlays = e.overlays.iter().any(|o| o.base == installed && !have_overlays.contains(&o.file));
    (e.manifest_sha256 != installed || new_overlays).then_some(e)
}

pub(crate) fn read_prefix(pack: &Path) -> Result<Vec<u8>, String> {
    // Header and manifest of an installed pack: its first 16 + header + manifest bytes.
    use std::io::Read;
    let mut f = std::fs::File::open(pack).map_err(|e| format!("{}: {e}", pack.display()))?;
    let mut head = [0u8; 16];
    f.read_exact(&mut head).map_err(|e| format!("{}: {e}", pack.display()))?;
    let size = f.metadata().map_err(|e| format!("{}: {e}", pack.display()))?.len();
    let within = |end: Option<u64>| end.filter(|e| *e <= size).ok_or_else(|| format!("{}: the container header points past the end of the file", pack.display()));
    let n = u64::from_le_bytes(head[8..16].try_into().map_err(|_| "pack header")?);
    within(n.checked_add(16))?;
    let mut header = vec![0u8; n as usize];
    f.read_exact(&mut header).map_err(|e| format!("{}: {e}", pack.display()))?;
    let h: Value = serde_json::from_slice(&header).map_err(|e| format!("{}: container header: {e}", pack.display()))?;
    let (off, len) = (h["manifest"][0].as_u64().ok_or("container header: manifest")?, h["manifest"][1].as_u64().ok_or("container header: manifest")?);
    let rest_len = within(off.checked_add(len).and_then(|r| r.checked_add(16 + n)))? - 16 - n;
    let mut rest = vec![0u8; rest_len as usize];
    f.read_exact(&mut rest).map_err(|e| format!("{}: {e}", pack.display()))?;
    let mut out = head.to_vec();
    out.extend(header);
    out.extend(rest);
    Ok(out)
}

fn donor_of(dir: &Path, arch: &str, run: Runner) -> Result<Donor, String> {
    // The blocks the installed pack holds, by hash and place: `pack need` of its own prefix.
    let (pack, pre) = (dir.join(PACK_FILE), dir.join("pack.prefix.old"));
    std::fs::write(&pre, read_prefix(&pack)?).map_err(|e| e.to_string())?;
    let out = run(&["pack".into(), "need".into(), pre.to_string_lossy().into_owned(), "--model-dir".into(), dir.to_string_lossy().into_owned(), "--arch".into(), arch.into()]);
    let _ = std::fs::remove_file(&pre);
    let need: Value = serde_json::from_str(&out?.1).map_err(|e| format!("pack need: {e}"))?;
    let ranges = need["ranges"].as_array().into_iter().flatten().filter_map(|r| Some((r["off"].as_u64()?, r["len"].as_u64()?, r["sha256"].as_str()?.to_string()))).collect();
    Ok(Donor { pack, ranges })
}

fn rename(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::rename(from, to).map_err(|e| format!("{} to {}: {e}", from.display(), to.display()))
}

fn swap_in(dir: &Path) -> Result<(), String> {
    // The new pack files take the installed names; the installed ones become .prev. Any failure puts everything back.
    let names = sidecars();
    let mut moved: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut step = || -> Result<(), String> {
        for n in &names {
            let (cur, prev) = (dir.join(n), dir.join(format!("{n}{PREV}")));
            if cur.exists() {
                rename(&cur, &prev)?;
                moved.push((prev, cur));
            }
        }
        for n in &names {
            let (new, cur) = (dir.join(format!("{n}{NEW}")), dir.join(n));
            rename(&new, &cur)?;
            moved.push((cur, new));
        }
        Ok(())
    };
    let done = step();
    if done.is_err() {
        for (a, b) in moved.iter().rev() {
            let _ = std::fs::rename(a, b);
        }
    }
    done
}

fn drop_new(dir: &Path) {
    for n in sidecars() {
        let _ = std::fs::remove_file(dir.join(format!("{n}{NEW}")));
    }
    let _ = std::fs::remove_file(dir.join(format!("pack.prefix{NEW}")));
}

fn verified_index(home: &Path, broker: &impl Broker, run: Runner) -> Result<(Vec<u8>, u64), String> {
    // The platform's index, checked by the engine against the pinned keys and refused when older than the last one installed; nothing of it is kept unless it passes.
    let dir = crate::paths::private_temp_dir("goethe-idx")?;
    let checked = (|| {
        let (raw, sig) = (broker.index()?, broker.index_sig()?);
        let (ix, sg) = (dir.join("index.json"), dir.join("index.sig"));
        std::fs::write(&ix, &raw).map_err(|e| e.to_string())?;
        std::fs::write(&sg, sig).map_err(|e| e.to_string())?;
        let s = |p: &Path| p.to_string_lossy().into_owned();
        let args = ["index".to_string(), "verify".into(), "--index".into(), s(&ix), "--sig".into(), s(&sg), "--last-serial".into(), crate::packs::last_serial(home).to_string()];
        let (code, out) = run(&args).map_err(|e| format!("the pack index did not verify: {e}"))?;
        let serial = serde_json::from_str::<Value>(&out).ok().and_then(|v| v["serial"].as_u64());
        match (code, serial) {
            (0, Some(serial)) => Ok((raw, serial)),
            _ => Err("the pack index did not verify: the engine gave no serial".to_string()),
        }
    })();
    let _ = std::fs::remove_dir_all(&dir);
    checked
}

pub fn refresh(home: &Path, model_id: &str, dir: &Path, arch: &str, broker: &impl Broker, run: Runner) -> Result<Outcome, String> {
    // Bring an installed model's pack to the index's: nothing when it already is, the changed blocks only when it is not, loud when it cannot.
    let Some(installed) = installed_manifest(dir) else { return Err(format!("{model_id}: no installed kernel pack to update")) };
    let have = installed_overlays(dir);
    let (index, serial) = verified_index(home, broker, run).map_err(|e| format!("{model_id}: {e}; the installed pack is kept"))?;
    let Some(entry) = pending(&index, model_id, arch, &installed, &have) else { return Ok(Outcome::Unchanged) };
    let engine = crate::packs::pack_engine_version(home);
    if !crate::packs::runnable_at(&entry, &engine) {
        let hint = if engine == crate::pyenv::ENGINE_VERSION { "goethe update" } else { "goethe update engine" };
        return Ok(Outcome::Skipped(format!(
            "the new pack of {model_id} needs goethe-engine {}.{}.{}, installed {engine}: run {hint}",
            entry.min_engine.0, entry.min_engine.1, entry.min_engine.2
        )));
    }
    if entry.manifest_sha256 == installed {
        let fetched = overlay_only(dir, &entry, &installed, broker, run)?;
        crate::packs::store_serial(home, serial)?;
        return Ok(Outcome::Overlaid(fetched));
    }
    let plan = crate::packs::plan(home, model_id, arch, None, broker, run)?.ok_or_else(|| format!("the signed index has no pack for {model_id} on {arch}"))?;
    let donor = donor_of(dir, arch, run)?;
    let built = crate::packs::finish_as(home, dir, arch, &plan, broker, run, crate::packs::Next { suffix: NEW, donor: Some(&donor) });
    let (_, delta) = match built {
        Ok(b) => b,
        Err(e) => {
            drop_new(dir);
            return Err(format!("{model_id}: the pack update was refused and the installed pack is kept: {e}"));
        }
    };
    if let Err(e) = swap_in(dir) {
        drop_new(dir);
        return Err(format!("{model_id}: {e}; the installed pack is kept"));
    }
    crate::packs::store_serial(home, plan.serial)?;
    Ok(Outcome::Updated(delta))
}

pub fn installed_overlays(dir: &Path) -> Vec<String> {
    // The overlay files stacked on the installed pack.
    let v: Value = std::fs::read_to_string(dir.join(OVERLAYS)).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    v["files"].as_array().into_iter().flatten().filter_map(|f| f.as_str().map(String::from)).collect()
}

fn overlay_only(dir: &Path, entry: &Entry, installed: &str, broker: &impl Broker, run: Runner) -> Result<u64, String> {
    // The entry lists overlays for the installed base: fetch only those (small), check each against its hash and the engine, then stack them.
    let have = installed_overlays(dir);
    let mut files = have.clone();
    let mut fetched = 0;
    let odir = dir.join("overlays");
    std::fs::create_dir_all(&odir).map_err(|e| e.to_string())?;
    for o in entry.overlays.iter().filter(|o| o.base == installed && !have.contains(&o.file)) {
        let bytes = broker.range(&o.file, 0, o.size)?;
        if crate::sha::digest_hex(&bytes) != o.sha256 {
            return Err(format!("overlay {}: does not hash to the signed index's figure; the installed pack is kept", o.file));
        }
        let (path, sig) = (odir.join(&o.file), odir.join(format!("{}.sig", o.file)));
        std::fs::write(&path, &bytes).map_err(|e| e.to_string())?;
        std::fs::write(&sig, serde_json::to_vec(&o.sig).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        let s = |p: &Path| p.to_string_lossy().into_owned();
        let checked = run(&["pack".into(), "verify".into(), s(&dir.join(PACK_FILE)), "--sig".into(), s(&dir.join(format!("{PACK_FILE}.sig"))), "--overlay".into(), s(&path)]);
        if !matches!(checked, Ok((0, _))) {
            let _ = std::fs::remove_file(&path);
            let _ = std::fs::remove_file(&sig);
            return Err(format!("overlay {}: the engine refused it ({}); the installed pack is kept", o.file, checked.err().unwrap_or_else(|| "exit not 0".into())));
        }
        fetched += o.size;
        files.push(o.file.clone());
    }
    std::fs::write(dir.join(OVERLAYS), serde_json::json!({ "files": files }).to_string()).map_err(|e| e.to_string())?;
    Ok(fetched)
}

pub fn rollback(dir: &Path) -> Result<(), String> {
    // The previous pack becomes the installed one and the installed one the previous (a second rollback goes forward again).
    let names = sidecars();
    if !dir.join(format!("{PACK_FILE}{PREV}")).is_file() {
        return Err("no previous kernel pack to restore".into());
    }
    let tmp = |n: &str| dir.join(format!("{n}.swap"));
    for n in &names {
        let (cur, prev) = (dir.join(n), dir.join(format!("{n}{PREV}")));
        if cur.exists() {
            rename(&cur, &tmp(n))?;
        }
        if prev.exists() {
            rename(&prev, &cur)?;
        }
        if tmp(n).exists() {
            rename(&tmp(n), &prev)?;
        }
    }
    let _ = std::fs::remove_file(dir.join(OVERLAYS));
    Ok(())
}

fn installed_packs(home: &Path) -> Vec<(String, PathBuf, String)> {
    // Installed models with a kernel pack: id, directory and the architecture the pack was verified for.
    let Ok(con) = crate::db::open(home) else { return Vec::new() };
    let Ok(mut stmt) = con.prepare("SELECT model_id, local_path FROM installed_models ORDER BY model_id") else { return Vec::new() };
    let rows: Vec<(String, String)> = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?))).map(|m| m.flatten().collect()).unwrap_or_default();
    rows.into_iter()
        .filter_map(|(id, rel)| {
            let dir = crate::paths::contained_in_models(home, &home.join(rel)).ok()?;
            let v: Value = serde_json::from_str(&std::fs::read_to_string(dir.join(PACK_VERIFY)).ok()?).ok()?;
            Some((id, dir, v["arch"].as_str()?.to_string()))
        })
        .collect()
}

pub fn run_all(check_only: bool) -> Result<(), String> {
    // goethe update for kernel packs: each installed pack against the signed index; a running instance keeps its pack until reloaded.
    let home = crate::paths::goethe_home();
    let packs = installed_packs(&home);
    if packs.is_empty() {
        return Ok(());
    }
    let Some(creds) = crate::client::link::load_credentials(&home).ok().flatten() else {
        println!("kernel packs: not checked, this node is not logged in (goethe login)");
        return Ok(());
    };
    let token = || Ok(creds.token.expose().to_string());
    let broker = crate::packs::HttpBroker::new(&creds.platform_url, &token);
    let run = crate::packs::python_runner(&home);
    let mut failed = Vec::new();
    for (id, dir, arch) in packs {
        if check_only {
            let (index, _) = verified_index(&home, &broker, &run)?;
            let installed = installed_manifest(&dir).unwrap_or_default();
            match pending(&index, &id, &arch, &installed, &installed_overlays(&dir)) {
                Some(e) => println!("{id}: a newer kernel pack is available ({} bytes); goethe update installs it", e.size),
                None => println!("{id}: kernel pack is current"),
            }
            continue;
        }
        match refresh(&home, &id, &dir, &arch, &broker, &run) {
            Ok(Outcome::Unchanged) => println!("{id}: kernel pack is current"),
            Ok(Outcome::Updated(d)) => println!("{id}: kernel pack updated ({} bytes fetched, {} reused from the installed pack); updated: reload to use (goethe update rollback undoes it)", d.fetched, d.reused),
            Ok(Outcome::Overlaid(n)) => println!("{id}: kernel pack patched ({n} bytes fetched); updated: reload to use"),
            Ok(Outcome::Skipped(why)) => println!("{id}: {why}"),
            Err(e) => {
                eprintln!("error: {e}");
                failed.push(id);
            }
        }
    }
    if failed.is_empty() { Ok(()) } else { Err(format!("kernel pack update failed for {}", failed.join(", "))) }
}

pub fn run_rollback() -> Result<(), String> {
    // Every installed model that has a previous pack gets it back.
    let home = crate::paths::goethe_home();
    let mut any = false;
    for (id, dir, _) in installed_packs(&home) {
        if dir.join(format!("{PACK_FILE}{PREV}")).is_file() {
            rollback(&dir)?;
            println!("{id}: the previous kernel pack is back; reload to use it");
            any = true;
        }
    }
    if !any {
        println!("no installed model has a previous kernel pack");
    }
    Ok(())
}

pub fn note_if_newer(home: &Path, model_id: &str) {
    // At goethe load: at most once a day, say when the index has a newer pack for this model; the check reads the cached index and never delays a load otherwise.
    let Some((_, dir, arch)) = installed_packs(home).into_iter().find(|(id, ..)| id == model_id) else { return };
    let marker = home.join("packs").join("update-check");
    let today = crate::sys::unix_secs() as u64 / 86400;
    if std::fs::read_to_string(&marker).ok().and_then(|t| t.trim().parse::<u64>().ok()) == Some(today) {
        return;
    }
    let _ = std::fs::write(&marker, today.to_string());
    let Some(index) = crate::packs::catalog(home) else { return };
    let installed = installed_manifest(&dir).unwrap_or_default();
    if pending(&index, model_id, &arch, &installed, &installed_overlays(&dir)).is_some() {
        eprintln!("note: a newer kernel pack exists for {model_id}: goethe update installs it (a loaded copy keeps running until reloaded)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::collections::HashMap;

    fn sha(b: &[u8]) -> String {
        crate::sha::digest_hex(b)
    }

    struct PackV {
        bytes: Vec<u8>,
        manifest_sha: String,
        ranges: Vec<(u64, u64, String)>,
    }

    fn pack_v(tag: &str, blobs: &[Vec<u8>]) -> PackV {
        // A container: header, manifest, then the blobs one after another; the manifest differs per tag.
        let manifest = serde_json::json!({"tag": tag, "model": {"weights": [{"file": "m-Q8_0.gguf"}]}, "variants": {"q8": {"weights": [0]}}}).to_string().into_bytes();
        let header = serde_json::json!({"manifest": [0, manifest.len()]}).to_string().into_bytes();
        let mut bytes = b"GCPACK3\n".to_vec();
        bytes.extend((header.len() as u64).to_le_bytes());
        bytes.extend(&header);
        bytes.extend(&manifest);
        let mut ranges = Vec::new();
        for b in blobs {
            ranges.push((bytes.len() as u64, b.len() as u64, sha(b)));
            bytes.extend(b);
        }
        PackV { manifest_sha: sha(&manifest), bytes, ranges }
    }

    fn blob(tag: u8) -> Vec<u8> {
        (0..4000u32).map(|i| (i as u8).wrapping_mul(7).wrapping_add(tag)).collect()
    }

    struct World {
        index: Vec<u8>,
        files: HashMap<String, Vec<u8>>,
        needs: HashMap<String, Vec<(u64, u64, String)>>,
    }

    fn world(new: &PackV, min_engine: &str, overlays: Value) -> World {
        let index = serde_json::json!({"schema": "gc-pack-index", "version": 3, "serial": 9, "entries": [{
            "model": {"hf_repo": "org/M-GGUF", "revision": "abc", "format": "gguf", "structure": []},
            "arch": "sm_120a", "abi_major": 3, "runtime": {"kind": "gc-c", "version": 1}, "min_engine": min_engine, "tier": "base",
            "file": "m.gcpack", "manifest_sha256": new.manifest_sha, "size": new.bytes.len(), "sig": {"keyid": "k"}, "overlays": overlays}]})
        .to_string()
        .into_bytes();
        let mut files = HashMap::new();
        files.insert("m.gcpack".to_string(), new.bytes.clone());
        World { index, files, needs: HashMap::new() }
    }

    struct Broker2<'a> {
        w: &'a World,
        tamper: bool,
        fetched: Cell<u64>,
        calls: Cell<u32>,
    }

    impl Broker for Broker2<'_> {
        fn index(&self) -> Result<Vec<u8>, String> {
            self.calls.set(self.calls.get() + 1);
            Ok(self.w.index.clone())
        }
        fn index_sig(&self) -> Result<Vec<u8>, String> {
            Ok(b"sig".to_vec())
        }
        fn range(&self, file: &str, off: u64, len: u64) -> Result<Vec<u8>, String> {
            self.calls.set(self.calls.get() + 1);
            self.fetched.set(self.fetched.get() + len);
            let mut b = self.w.files.get(file).ok_or("no such file")?[off as usize..(off + len) as usize].to_vec();
            if self.tamper && len == 4000 {
                let last = b.len() - 1;
                b[last] ^= 1;
            }
            Ok(b)
        }
    }

    fn broker(w: &World) -> Broker2<'_> {
        Broker2 { w, tamper: false, fetched: Cell::new(0), calls: Cell::new(0) }
    }

    fn engine<'a>(w: &'a World, installed: &'a PackV, bad_sig: bool) -> impl Fn(&[String]) -> Result<(i32, String), String> + 'a {
        // A fake goethe-engine: blocks by manifest hash for `pack need`, a refusal for `pack verify` when asked.
        move |a| {
            let key = format!("{} {}", a[0], a[1]);
            match key.as_str() {
                "index verify" => Ok((0, r#"{"serial": 9}"#.into())),
                "pack match" => Ok((0, r#"{"file": "m.gcpack", "variant": "q8"}"#.into())),
                "pack need" => {
                    let prefix = std::fs::read(&a[2]).unwrap();
                    let n = u64::from_le_bytes(prefix[8..16].try_into().unwrap()) as usize;
                    let msha = sha(&prefix[16 + n..]);
                    let ranges = if msha == installed.manifest_sha { &installed.ranges } else { w.needs.get(&msha).ok_or("unknown manifest")? };
                    let r: Vec<Value> = ranges.iter().map(|(o, l, h)| serde_json::json!({"off": o, "len": l, "sha256": h})).collect();
                    Ok((0, serde_json::json!({"manifest_sha256": msha, "ranges": r}).to_string()))
                }
                "pack verify" if bad_sig => Err("goethe-engine pack verify: exit 1: signature does not verify".into()),
                "pack verify" => {
                    let bytes = std::fs::read(&a[2]).unwrap();
                    let (h, n) = (&bytes[..16], u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize);
                    let _ = h;
                    let header: Value = serde_json::from_slice(&bytes[16..16 + n]).unwrap();
                    let end = 16 + n + header["manifest"][1].as_u64().unwrap() as usize;
                    Ok((0, serde_json::json!({"ok": true, "manifest_sha256": sha(&bytes[16 + n..end]), "weights": ["m-Q8_0.gguf"]}).to_string()))
                }
                _ => Ok((0, r#"{"ok": true}"#.into())),
            }
        }
    }

    fn install(v1: &PackV) -> (tempfile::TempDir, tempfile::TempDir) {
        let (home, dir) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
        std::fs::write(dir.path().join(PACK_FILE), &v1.bytes).unwrap();
        std::fs::write(dir.path().join(format!("{PACK_FILE}.sig")), b"{\"keyid\":\"old\"}").unwrap();
        std::fs::write(dir.path().join(PACK_VERIFY), serde_json::json!({"ok": true, "manifest_sha256": v1.manifest_sha}).to_string()).unwrap();
        (home, dir)
    }

    fn no_leftovers(dir: &Path) {
        for n in sidecars() {
            assert!(!dir.join(format!("{n}{NEW}")).exists(), "{n}{NEW} left behind");
        }
    }

    #[test]
    fn nothing_changed_means_no_download() {
        let v1 = pack_v("1", &[blob(1), blob(2), blob(3)]);
        let w = world(&v1, "0.2.0", Value::Null);
        let (home, dir) = install(&v1);
        let b = broker(&w);
        let out = refresh(home.path(), "org/m-gguf", dir.path(), "sm_120a", &b, &engine(&w, &v1, false)).unwrap();
        assert_eq!(out, Outcome::Unchanged);
        assert_eq!(b.fetched.get(), 0, "no pack byte fetched");
    }

    #[test]
    fn a_changed_pack_fetches_only_the_blocks_the_installed_one_lacks() {
        let v1 = pack_v("1", &[blob(1), blob(2), blob(3)]);
        let v2 = pack_v("2", &[blob(1), blob(2), blob(9)]);
        let mut w = world(&v2, "0.2.0", Value::Null);
        w.needs.insert(v2.manifest_sha.clone(), v2.ranges.clone());
        let (home, dir) = install(&v1);
        let b = broker(&w);
        let out = refresh(home.path(), "org/m-gguf", dir.path(), "sm_120a", &b, &engine(&w, &v1, false)).unwrap();
        let full = v2.bytes.len() as u64;
        let prefix = v2.ranges[0].0;
        assert_eq!(out, Outcome::Updated(Delta { fetched: 4000, reused: 8000 }), "one 4000-byte block fetched, two reused from the installed pack");
        assert_eq!(b.fetched.get(), prefix + 4000, "the prefix ({prefix} bytes) and the one new block, of a {full}-byte pack");
        assert_eq!(std::fs::read(dir.path().join(PACK_FILE)).unwrap(), v2.bytes, "the new pack is whole");
        assert_eq!(std::fs::read(dir.path().join(format!("{PACK_FILE}.prev"))).unwrap(), v1.bytes, "the old one is kept for a rollback");
        assert_eq!(installed_manifest(dir.path()).as_deref(), Some(v2.manifest_sha.as_str()));
        assert_eq!(std::fs::read_to_string(home.path().join("packs/serial")).unwrap().trim(), "9");
        no_leftovers(dir.path());
        let again = refresh(home.path(), "org/m-gguf", dir.path(), "sm_120a", &b, &engine(&w, &v1, false)).unwrap();
        assert_eq!(again, Outcome::Unchanged);
    }

    #[test]
    fn a_bad_signature_or_a_bad_block_refuses_the_update_and_keeps_the_old_pack() {
        let v1 = pack_v("1", &[blob(1), blob(2)]);
        let v2 = pack_v("2", &[blob(1), blob(7)]);
        let mut w = world(&v2, "0.2.0", Value::Null);
        w.needs.insert(v2.manifest_sha.clone(), v2.ranges.clone());
        let (home, dir) = install(&v1);
        let err = refresh(home.path(), "org/m-gguf", dir.path(), "sm_120a", &broker(&w), &engine(&w, &v1, true)).unwrap_err();
        assert!(err.contains("installed pack is kept") && err.contains("signature"), "{err}");
        assert_eq!(std::fs::read(dir.path().join(PACK_FILE)).unwrap(), v1.bytes);
        no_leftovers(dir.path());
        let mut b = broker(&w);
        b.tamper = true;
        let err = refresh(home.path(), "org/m-gguf", dir.path(), "sm_120a", &b, &engine(&w, &v1, false)).unwrap_err();
        assert!(err.contains("does not hash") && err.contains("installed pack is kept"), "{err}");
        assert_eq!(std::fs::read(dir.path().join(PACK_FILE)).unwrap(), v1.bytes);
        assert!(!dir.path().join(format!("{PACK_FILE}.prev")).exists());
        no_leftovers(dir.path());
    }

    #[test]
    fn a_pack_that_needs_a_newer_engine_is_skipped_with_the_reason() {
        let v1 = pack_v("1", &[blob(1)]);
        let v2 = pack_v("2", &[blob(2)]);
        let w = world(&v2, "99.0.0", Value::Null);
        let (home, dir) = install(&v1);
        let b = broker(&w);
        match refresh(home.path(), "org/m-gguf", dir.path(), "sm_120a", &b, &engine(&w, &v1, false)).unwrap() {
            Outcome::Skipped(why) => assert!(why.contains("needs goethe-engine 99.0.0") && why.contains("goethe update"), "{why}"),
            other => panic!("{other:?}"),
        }
        assert_eq!(b.fetched.get(), 0);
        assert_eq!(std::fs::read(dir.path().join(PACK_FILE)).unwrap(), v1.bytes);
    }

    #[test]
    fn an_overlay_for_the_installed_base_is_fetched_alone_and_stacked() {
        let v1 = pack_v("1", &[blob(1), blob(2)]);
        let overlay = b"overlay manifest and blocks".to_vec();
        let ov = serde_json::json!([{"file": "fix1.gcoverlay", "base": v1.manifest_sha, "size": overlay.len(), "sha256": sha(&overlay), "sig": {"keyid": "k"}}]);
        let mut w = world(&v1, "0.2.0", ov);
        w.files.insert("fix1.gcoverlay".into(), overlay.clone());
        let (home, dir) = install(&v1);
        let b = broker(&w);
        let out = refresh(home.path(), "org/m-gguf", dir.path(), "sm_120a", &b, &engine(&w, &v1, false)).unwrap();
        assert_eq!(out, Outcome::Overlaid(overlay.len() as u64));
        assert_eq!(b.fetched.get(), overlay.len() as u64, "only the overlay crossed the network");
        assert_eq!(std::fs::read(dir.path().join(PACK_FILE)).unwrap(), v1.bytes, "the base pack is untouched");
        assert_eq!(std::fs::read(dir.path().join("overlays/fix1.gcoverlay")).unwrap(), overlay);
        assert_eq!(installed_overlays(dir.path()), vec!["fix1.gcoverlay".to_string()]);
        let again = refresh(home.path(), "org/m-gguf", dir.path(), "sm_120a", &b, &engine(&w, &v1, false)).unwrap();
        assert_eq!(again, Outcome::Unchanged, "an overlay already stacked is not fetched again");
        let mut bad = world(&v1, "0.2.0", serde_json::json!([{"file": "fix2.gcoverlay", "base": v1.manifest_sha, "size": 4, "sha256": sha(b"good"), "sig": {}}]));
        bad.files.insert("fix2.gcoverlay".into(), b"evil".to_vec());
        let err = refresh(home.path(), "org/m-gguf", dir.path(), "sm_120a", &broker(&bad), &engine(&bad, &v1, false)).unwrap_err();
        assert!(err.contains("does not hash") && !dir.path().join("overlays/fix2.gcoverlay").exists(), "{err}");
    }

    fn index_refused<'a>(inner: impl Fn(&[String]) -> Result<(i32, String), String> + 'a, why: &'a str) -> impl Fn(&[String]) -> Result<(i32, String), String> + 'a {
        // The engine as it answers an index it cannot verify: an error exit for `index verify`.
        move |a| if a[0] == "index" { Err(format!("goethe-engine index verify: exit 1: {why}")) } else { inner(a) }
    }

    fn untouched(home: &Path, dir: &Path, v1: &PackV, b: &Broker2) {
        assert_eq!(b.fetched.get(), 0, "no pack or overlay byte fetched");
        assert_eq!(std::fs::read(dir.join(PACK_FILE)).unwrap(), v1.bytes);
        assert!(!dir.join("overlays").exists() && installed_overlays(dir).is_empty());
        assert!(!dir.join(format!("{PACK_FILE}.prev")).exists());
        no_leftovers(dir);
        assert!(!home.join("packs/index.json").exists(), "an unverified index is not kept");
    }

    #[test]
    fn an_unsigned_index_that_lists_an_overlay_or_a_new_pack_changes_nothing() {
        let v1 = pack_v("1", &[blob(1), blob(2)]);
        let overlay = b"overlay manifest and blocks".to_vec();
        let ov = serde_json::json!([{"file": "fix1.gcoverlay", "base": v1.manifest_sha, "size": overlay.len(), "sha256": sha(&overlay), "sig": {"keyid": "evil"}}]);
        let mut w = world(&v1, "0.2.0", ov);
        w.files.insert("fix1.gcoverlay".into(), overlay);
        let (home, dir) = install(&v1);
        let b = broker(&w);
        let err = refresh(home.path(), "org/m-gguf", dir.path(), "sm_120a", &b, &index_refused(engine(&w, &v1, false), "no signature")).unwrap_err();
        assert!(err.contains("did not verify") && err.contains("installed pack is kept"), "{err}");
        untouched(home.path(), dir.path(), &v1, &b);
        assert!(!home.path().join("packs/serial").exists());
        let v2 = pack_v("2", &[blob(1), blob(7)]);
        let mut w2 = world(&v2, "0.2.0", Value::Null);
        w2.needs.insert(v2.manifest_sha.clone(), v2.ranges.clone());
        let b2 = broker(&w2);
        assert!(refresh(home.path(), "org/m-gguf", dir.path(), "sm_120a", &b2, &index_refused(engine(&w2, &v1, false), "no signature")).is_err());
        untouched(home.path(), dir.path(), &v1, &b2);
        assert!(run_all_check_refuses(home.path(), &b2, &index_refused(engine(&w2, &v1, false), "no signature")), "update --check does not announce an unverified pack either");
    }

    fn run_all_check_refuses(home: &Path, b: &Broker2, run: Runner) -> bool {
        verified_index(home, b, run).is_err()
    }

    #[test]
    fn an_index_older_than_the_last_installed_one_is_refused() {
        let v1 = pack_v("1", &[blob(1), blob(2)]);
        let overlay = b"overlay manifest and blocks".to_vec();
        let ov = serde_json::json!([{"file": "fix1.gcoverlay", "base": v1.manifest_sha, "size": overlay.len(), "sha256": sha(&overlay), "sig": {"keyid": "k"}}]);
        let mut w = world(&v1, "0.2.0", ov);
        w.files.insert("fix1.gcoverlay".into(), overlay);
        let (home, dir) = install(&v1);
        crate::packs::store_serial(home.path(), 12).unwrap();
        let inner = engine(&w, &v1, false);
        let old_serial = move |a: &[String]| {
            let last: u64 = a.iter().position(|x| x == "--last-serial").and_then(|i| a.get(i + 1)).and_then(|n| n.parse().ok()).unwrap_or(0);
            if a[0] == "index" && last > 9 { Err("goethe-engine index verify: exit 1: serial 9 is older than 12".to_string()) } else { inner(a) }
        };
        let b = broker(&w);
        let err = refresh(home.path(), "org/m-gguf", dir.path(), "sm_120a", &b, &old_serial).unwrap_err();
        assert!(err.contains("did not verify"), "{err}");
        untouched(home.path(), dir.path(), &v1, &b);
        assert_eq!(std::fs::read_to_string(home.path().join("packs/serial")).unwrap().trim(), "12");
    }

    #[test]
    fn a_verified_index_stacks_its_overlay_and_records_the_serial() {
        let v1 = pack_v("1", &[blob(1), blob(2)]);
        let overlay = b"overlay manifest and blocks".to_vec();
        let ov = serde_json::json!([{"file": "fix1.gcoverlay", "base": v1.manifest_sha, "size": overlay.len(), "sha256": sha(&overlay), "sig": {"keyid": "k"}}]);
        let mut w = world(&v1, "0.2.0", ov);
        w.files.insert("fix1.gcoverlay".into(), overlay.clone());
        let (home, dir) = install(&v1);
        let out = refresh(home.path(), "org/m-gguf", dir.path(), "sm_120a", &broker(&w), &engine(&w, &v1, false)).unwrap();
        assert_eq!(out, Outcome::Overlaid(overlay.len() as u64));
        assert_eq!(std::fs::read_to_string(home.path().join("packs/serial")).unwrap().trim(), "9");
        assert!(!home.path().join("packs/index.json").exists());
    }

    #[test]
    fn rollback_restores_the_previous_pack_and_goes_forward_again() {
        let v1 = pack_v("1", &[blob(1), blob(2)]);
        let v2 = pack_v("2", &[blob(1), blob(7)]);
        let mut w = world(&v2, "0.2.0", Value::Null);
        w.needs.insert(v2.manifest_sha.clone(), v2.ranges.clone());
        let (home, dir) = install(&v1);
        assert_eq!(rollback(dir.path()).unwrap_err(), "no previous kernel pack to restore");
        refresh(home.path(), "org/m-gguf", dir.path(), "sm_120a", &broker(&w), &engine(&w, &v1, false)).unwrap();
        rollback(dir.path()).unwrap();
        assert_eq!(std::fs::read(dir.path().join(PACK_FILE)).unwrap(), v1.bytes);
        assert_eq!(installed_manifest(dir.path()).as_deref(), Some(v1.manifest_sha.as_str()));
        rollback(dir.path()).unwrap();
        assert_eq!(std::fs::read(dir.path().join(PACK_FILE)).unwrap(), v2.bytes);
    }

    #[test]
    fn a_damaged_pack_header_is_an_error_not_a_huge_allocation() {
        let v1 = pack_v("1", &[blob(1)]);
        let dir = tempfile::tempdir().unwrap();
        let pack = dir.path().join(PACK_FILE);
        std::fs::write(&pack, &v1.bytes).unwrap();
        assert_eq!(read_prefix(&pack).unwrap(), v1.bytes[..v1.ranges[0].0 as usize]);
        let mut bad = v1.bytes.clone();
        bad[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        std::fs::write(&pack, &bad).unwrap();
        assert!(read_prefix(&pack).unwrap_err().contains("past the end of the file"));
        std::fs::write(&pack, &v1.bytes[..20]).unwrap();
        assert!(read_prefix(&pack).is_err());
    }
}
