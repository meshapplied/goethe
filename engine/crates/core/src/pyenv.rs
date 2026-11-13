// The py-env in layers: base everywhere, goethe-engine or SGLang only once a model needs them, from the signed CDN.

use crate::update::{fetch_signed_sums, fetch_verified, lookup_sum, normalize_arch, pkg_base_url, sha256_file};
use std::fs;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    Base,
    Engine,
    Sglang,
}

const PIN: &str = include_str!("../../../goethe-engine/PIN");

pub const ENGINE_VERSION: &str = pin_value(PIN, "GOETHE_ENGINE_VERSION");
const ENGINE_WHEEL_URL: &str = pin_value(PIN, "GOETHE_ENGINE_WHEEL_URL");
const ENGINE_WHEEL_SHA256: &str = pin_value(PIN, "GOETHE_ENGINE_WHEEL_SHA256");

const fn pin_value(text: &'static str, key: &str) -> &'static str {
    // The value of a KEY=value line, at compile time (a missing key fails the build).
    let (t, k) = (text.as_bytes(), key.as_bytes());
    let mut i = 0;
    while i < t.len() {
        let mut j = 0;
        while j < k.len() && i + j < t.len() && t[i + j] == k[j] {
            j += 1;
        }
        if j == k.len() && i + j < t.len() && t[i + j] == b'=' {
            let start = i + j + 1;
            let mut end = start;
            while end < t.len() && t[end] != b'\n' && t[end] != b'\r' {
                end += 1;
            }
            let (_, rest) = t.split_at(start);
            let (value, _) = rest.split_at(end - start);
            match std::str::from_utf8(value) {
                Ok(v) => return v,
                Err(_) => panic!("goethe-engine/PIN is not UTF-8"),
            }
        }
        while i < t.len() && t[i] != b'\n' {
            i += 1;
        }
        i += 1;
    }
    panic!("goethe-engine/PIN lacks a key")
}

impl Layer {
    pub fn name(self) -> &'static str {
        match self {
            Layer::Base => "base",
            Layer::Engine => "engine",
            Layer::Sglang => "sglang",
        }
    }

    fn lock_stem(self) -> &'static str {
        // The requirements file of the layer: requirements.base, requirements.engine, or the full requirements lock.
        match self {
            Layer::Base => "requirements.base",
            Layer::Engine => "requirements.engine",
            Layer::Sglang => "requirements",
        }
    }

    fn need_gb(self) -> u64 {
        // Free disk the layer needs at GOETHE_HOME while it installs, with room to spare.
        match self {
            Layer::Base => 1,
            Layer::Engine => 7,
            Layer::Sglang => 12,
        }
    }
}

pub fn site_packages(home: &Path) -> Option<PathBuf> {
    // The py-env's site-packages, when its python is on disk.
    let env = home.join("py-env");
    if !env.join("bin/python").exists() {
        return None;
    }
    let mut sites: Vec<PathBuf> = fs::read_dir(env.join("lib"))
        .ok()?
        .flatten()
        .map(|e| e.path().join("site-packages"))
        .filter(|p| p.is_dir())
        .collect();
    sites.sort();
    sites.pop()
}

pub fn dist_version(site: &Path, dist: &str) -> Option<String> {
    // The installed version of a distribution, from its dist-info directory name (dist in its _ normalized form).
    fs::read_dir(site).ok()?.flatten().find_map(|e| {
        let name = e.file_name().to_string_lossy().into_owned();
        let rest = name.strip_suffix(".dist-info")?;
        let (n, v) = rest.split_once('-')?;
        n.eq_ignore_ascii_case(dist).then(|| v.to_string())
    })
}

pub fn has(home: &Path, layer: Layer) -> bool {
    // From dist-info dirs only: base has the hf CLI's package, engine the pinned goethe-engine and its runtime, sglang sglang.
    let Some(site) = site_packages(home) else {
        return false;
    };
    let present = |d: &str| dist_version(&site, d).is_some();
    let base = present("huggingface_hub");
    match layer {
        Layer::Base => base,
        Layer::Engine => {
            base && dist_version(&site, "goethe_engine").as_deref() == Some(ENGINE_VERSION)
                && ["torch", "transformers", "gguf"].iter().all(|d| present(d))
        }
        Layer::Sglang => base && present("sglang") && present("torch"),
    }
}

pub fn engine_installed(home: &Path) -> Option<String> {
    // The goethe-engine version in the py-env, whatever the pin says.
    dist_version(&site_packages(home)?, "goethe_engine")
}

fn dev_python(home: &Path) -> bool {
    // A source or test setup runs engines with GOETHE_PYTHON and has no py-env: goethe installs nothing there.
    std::env::var_os("GOETHE_PYTHON").is_some() && !home.join("py-env/bin/python").exists()
}

pub fn require(home: &Path, layer: Layer, model_id: &str) -> Result<(), String> {
    // Loading never installs: a missing layer is a clear error that names the command that installs it.
    if has(home, layer) || dev_python(home) {
        return Ok(());
    }
    match layer {
        Layer::Base => Err("the Python environment is missing: run the installer again".to_string()),
        Layer::Engine => Err(format!(
            "{model_id} runs on goethe-engine {ENGINE_VERSION}, which is not installed: run goethe install {model_id} to install it"
        )),
        Layer::Sglang => Err(format!(
            "{model_id} runs on SGLang, which is not installed yet: run goethe install {model_id} to install it (about 8 GB, only once)"
        )),
    }
}

pub fn ensure(home: &Path, layer: Layer) -> Result<(), String> {
    // Install the layer (and the ones below it) from the signed CDN unless it is already there; one install at a time.
    if has(home, layer) || dev_python(home) {
        return Ok(());
    }
    let _lock = lock(home)?;
    if has(home, layer) {
        return Ok(());
    }
    check_disk(home, layer)?;
    println!("{}", progress_line(home, layer));
    let tmp = home.join(".pyenv-staging");
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
    let pkg = pkg_base_url();
    let result = fetch_signed_sums(home, &pkg, &tmp)
        .map_err(|e| format!("could not get the signed package list from {pkg} ({e}); nothing was changed, run the command again"))
        .and_then(|sums| install_from(home, &pkg, &sums, &tmp, layer));
    let _ = fs::remove_dir_all(&tmp);
    result?;
    if !has(home, layer) {
        return Err(format!("the {} layer of the Python environment is still incomplete after installing it; run the command again", layer.name()));
    }
    Ok(())
}

fn progress_line(home: &Path, layer: Layer) -> String {
    // One line on what is about to download, smaller when PyTorch is already in the py-env.
    let torch = site_packages(home).is_some_and(|s| dist_version(&s, "torch").is_some());
    match (layer, torch) {
        (Layer::Base, _) => "Setting up the Python environment (about 30 MB, only once)".to_string(),
        (Layer::Engine, false) => format!("Installing goethe-engine {ENGINE_VERSION} with PyTorch and the CUDA libraries (about 5 GB, only once)"),
        (Layer::Engine, true) => format!("Installing goethe-engine {ENGINE_VERSION} (PyTorch is already installed)"),
        (Layer::Sglang, false) => "Installing SGLang and the CUDA libraries (about 8 GB, only once)".to_string(),
        (Layer::Sglang, true) => "Installing SGLang and its CUDA libraries (about 4 GB more, only once; PyTorch is already installed)".to_string(),
    }
}

fn check_disk(home: &Path, layer: Layer) -> Result<(), String> {
    // Stop before downloading when the disk at GOETHE_HOME cannot hold the layer.
    let Some(free) = free_bytes(home) else {
        return Ok(());
    };
    let need = layer.need_gb();
    if free < need * 1024 * 1024 * 1024 {
        return Err(format!(
            "only {} GB free at {}; the {} layer of the Python environment needs about {need} GB: free space and run the command again",
            free / (1024 * 1024 * 1024),
            home.display(),
            layer.name()
        ));
    }
    Ok(())
}

fn free_bytes(path: &Path) -> Option<u64> {
    let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    (unsafe { libc::statvfs(c.as_ptr(), &mut st) } == 0).then(|| st.f_bavail as u64 * st.f_frsize as u64)
}

pub struct Lock(#[allow(dead_code)] fs::File);

pub fn lock(home: &Path) -> Result<Lock, String> {
    // Exclusive flock on GOETHE_HOME/run/pyenv.lock for the whole install; a second goethe waits its turn.
    let dir = home.join("run");
    fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let path = dir.join("pyenv.lock");
    let f = fs::OpenOptions::new().create(true).truncate(false).write(true).open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        println!("Another goethe is installing into the Python environment; waiting for it to finish");
        if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err(format!("{}: {}", path.display(), std::io::Error::last_os_error()));
        }
    }
    Ok(Lock(f))
}

fn lock_rel(layer: Layer, arch: &str) -> String {
    // The CDN path of the layer's lock for this arch, named like bootstrap.sh expects it.
    let suffix = if arch == "aarch64" { ".aarch64" } else { "" };
    format!("pyenv/{}{suffix}.lock", layer.lock_stem())
}

fn place(part: &Path, dest: &Path, mode: u32) -> Result<(), String> {
    // Copy a verified download into place through a .new file and a rename.
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let new = dest.with_extension("new");
    fs::copy(part, &new).map_err(|e| format!("{}: {e}", new.display()))?;
    fs::set_permissions(&new, std::os::unix::fs::PermissionsExt::from_mode(mode)).map_err(|e| e.to_string())?;
    fs::rename(&new, dest).map_err(|e| format!("{}: {e}", dest.display()))
}

pub(crate) fn install_from(home: &Path, pkg: &str, sums: &Path, tmp: &Path, layer: Layer) -> Result<(), String> {
    // Fetch what bootstrap.sh needs for the layer (verified against the signed SHA256SUMS), then run it.
    let arch = normalize_arch();
    let pyenv = home.join("pyenv");
    let fetch = |rel: &str| fetch_verified(pkg, sums, rel, tmp).map_err(|e| format!("downloading the Python environment from {pkg}: {e}"));
    let uv = home.join("bin/uv");
    if !uv.is_file() {
        place(&fetch(&format!("bin/linux-{arch}/uv"))?, &uv, 0o755)?;
    }
    let mut files = vec!["pyenv/bootstrap.sh".to_string(), lock_rel(Layer::Base, &arch)];
    if layer != Layer::Base {
        files.push(lock_rel(layer, &arch));
    }
    for rel in &files {
        place(&fetch(rel)?, &pyenv.join(rel.trim_start_matches("pyenv/")), 0o644)?;
    }
    let mut cmd = Command::new("bash");
    cmd.arg(pyenv.join("bootstrap.sh"))
        .arg(layer.name())
        .env("GOETHE_HOME", home)
        .env("GOETHE_ARCH", &arch)
        .env("UV", &uv);
    let wheel = if layer == Layer::Sglang { Some(sglang_wheel(home, pkg, sums, tmp)?) } else { None };
    if let Some(w) = &wheel {
        cmd.env("GOETHE_SGLANG_WHEEL", w);
    }
    if layer == Layer::Engine {
        cmd.env("GOETHE_ENGINE_WHEEL_URL", ENGINE_WHEEL_URL).env("GOETHE_ENGINE_WHEEL_SHA256", ENGINE_WHEEL_SHA256);
    }
    let status = cmd.status().map_err(|e| format!("could not run bootstrap.sh: {e}"))?;
    if !status.success() {
        return Err(format!(
            "installing the {} layer of the Python environment failed ({}; the output above says why). What was installed before still works; run the command again to retry",
            layer.name(),
            status.code().map_or("killed".to_string(), |c| format!("bootstrap.sh exit {c}"))
        ));
    }
    if let Some(w) = wheel {
        let name = w.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        fs::write(pyenv.join("LATEST"), format!("{name}\n")).map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub(crate) fn sglang_wheel(home: &Path, pkg: &str, sums: &Path, tmp: &Path) -> Result<PathBuf, String> {
    // The SGLang wheel wheels/LATEST names, kept in GOETHE_HOME/pyenv; a copy with the signed sha is not downloaded again.
    let latest = fetch_verified(pkg, sums, "wheels/LATEST", tmp)?;
    let name = fs::read_to_string(&latest).map_err(|e| e.to_string())?.trim().to_string();
    if !name.ends_with(".whl") || name.contains('/') || name.starts_with('.') {
        return Err(format!("wheels/LATEST names an invalid wheel {name:?}"));
    }
    let rel = format!("wheels/{name}");
    let dest = home.join("pyenv").join(&name);
    if sha256_file(&dest).ok() == Some(lookup_sum(sums, &rel)?) {
        return Ok(dest);
    }
    place(&fetch_verified(pkg, sums, &rel, tmp)?, &dest, 0o644)?;
    Ok(dest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn lock_entries(text: &str) -> BTreeMap<String, (String, Vec<String>)> {
        // name -> (requirement line, sorted hashes) of a uv hash-locked requirements file.
        let mut out = BTreeMap::new();
        let mut cur: Option<String> = None;
        for line in text.lines() {
            let t = line.trim();
            if t.starts_with('#') || t.is_empty() {
                continue;
            }
            if let Some(h) = t.strip_prefix("--hash=") {
                let name = cur.clone().expect("hash before any requirement");
                out.entry(name).and_modify(|e: &mut (String, Vec<String>)| e.1.push(h.trim_end_matches(" \\").to_string()));
                continue;
            }
            let req = t.trim_end_matches('\\').trim().to_string();
            let name = req.split([' ', '=', '@']).next().unwrap().to_lowercase().replace(['_', '.'], "-");
            assert!(out.insert(name.clone(), (req, Vec::new())).is_none(), "{name} listed twice");
            cur = Some(name);
        }
        for e in out.values_mut() {
            e.1.sort();
        }
        out
    }

    #[test]
    fn layer_locks_are_subsets_of_the_main_lock() {
        // Every layer lock pins exactly what the main lock pins (version or URL and hashes), so layers never reinstall.
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pyenv");
        let read = |n: &str| fs::read_to_string(dir.join(n)).unwrap_or_else(|e| panic!("{n}: {e}"));
        for suffix in ["", ".aarch64"] {
            let main = lock_entries(&read(&format!("requirements{suffix}.lock")));
            let base = lock_entries(&read(&format!("requirements.base{suffix}.lock")));
            let engine = lock_entries(&read(&format!("requirements.engine{suffix}.lock")));
            for (layer, entries) in [("base", &base), ("engine", &engine)] {
                assert!(!entries.is_empty(), "{layer}{suffix} lock is empty");
                for (name, pin) in entries.iter() {
                    assert!(!pin.1.is_empty(), "{layer}{suffix}: {name} has no hashes");
                    assert_eq!(main.get(name), Some(pin), "{layer}{suffix}: {name} differs from requirements{suffix}.lock; rerun pyenv/layers.sh");
                }
            }
            for name in base.keys() {
                assert!(engine.contains_key(name), "engine{suffix} lock lacks base package {name}");
            }
            for name in ["huggingface-hub", "torch", "transformers", "gguf"] {
                assert!(engine.contains_key(name), "engine{suffix} lock lacks {name}");
            }
        }
    }

    #[test]
    #[ignore = "installs from the CDN in GOETHE_PKG_URL into GOETHE_TEST_PYENV_HOME"]
    fn ensure_from_cdn() {
        // Manual check: GOETHE_TEST_PYENV_LAYERS=engine,sglang installs those layers in that order, then again as no-ops.
        let home = PathBuf::from(std::env::var("GOETHE_TEST_PYENV_HOME").expect("set GOETHE_TEST_PYENV_HOME"));
        let layers = std::env::var("GOETHE_TEST_PYENV_LAYERS").unwrap_or_else(|_| "base".to_string());
        for name in layers.split(',') {
            let layer = [Layer::Base, Layer::Engine, Layer::Sglang].into_iter().find(|l| l.name() == name).expect("base, engine or sglang");
            ensure(&home, layer).unwrap();
            assert!(has(&home, layer));
            ensure(&home, layer).unwrap();
        }
    }

    #[test]
    fn pin_is_compiled_in() {
        assert_eq!(pin_value("A=1\nGOETHE_X=2\nX=3\n", "X"), "3");
        assert_eq!(pin_value("A=1\r\nB=x y\r\n", "B"), "x y");
        assert!(!ENGINE_VERSION.is_empty() && ENGINE_WHEEL_URL.contains(ENGINE_VERSION));
        assert_eq!(ENGINE_WHEEL_SHA256.len(), 64);
    }

    fn fake_env(home: &Path, dists: &[&str]) {
        let site = home.join("py-env/lib/python3.11/site-packages");
        fs::create_dir_all(&site).unwrap();
        fs::create_dir_all(home.join("py-env/bin")).unwrap();
        fs::write(home.join("py-env/bin/python"), b"").unwrap();
        for d in dists {
            fs::create_dir_all(site.join(format!("{d}.dist-info"))).unwrap();
        }
    }

    #[test]
    fn layers_are_read_from_dist_info() {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path();
        assert!(!has(home, Layer::Base));
        fake_env(home, &["huggingface_hub-1.32.0"]);
        assert!(has(home, Layer::Base) && !has(home, Layer::Engine) && !has(home, Layer::Sglang));
        let pinned = format!("goethe_engine-{ENGINE_VERSION}");
        fake_env(home, &["torch-2.13.0+cu130", "transformers-5.12.1", "goethe_engine-0.0.1"]);
        assert!(!has(home, Layer::Engine), "gguf missing and an old goethe-engine");
        fake_env(home, &["gguf-0.19.0"]);
        assert!(!has(home, Layer::Engine), "old goethe-engine");
        assert_eq!(engine_installed(home).as_deref(), Some("0.0.1"));
        fs::remove_dir(home.join("py-env/lib/python3.11/site-packages/goethe_engine-0.0.1.dist-info")).unwrap();
        fake_env(home, &[&pinned]);
        assert!(has(home, Layer::Engine) && !has(home, Layer::Sglang));
        fake_env(home, &["sglang-0.0.0.dev1+g2d2576775", "sglang_kernel-0.4.7"]);
        assert!(has(home, Layer::Sglang));
        assert_eq!(dist_version(&site_packages(home).unwrap(), "sglang").as_deref(), Some("0.0.0.dev1+g2d2576775"));
    }

    #[test]
    fn layer_files_follow_bootstrap_names() {
        assert_eq!(lock_rel(Layer::Base, "x86_64"), "pyenv/requirements.base.lock");
        assert_eq!(lock_rel(Layer::Engine, "aarch64"), "pyenv/requirements.engine.aarch64.lock");
        assert_eq!(lock_rel(Layer::Sglang, "aarch64"), "pyenv/requirements.aarch64.lock");
        assert_eq!(lock_rel(Layer::Sglang, "x86_64"), "pyenv/requirements.lock");
    }

    #[test]
    fn missing_layers_name_the_install_command() {
        let dir = tempfile::tempdir().unwrap();
        fake_env(dir.path(), &["huggingface_hub-1.32.0"]);
        let e = require(dir.path(), Layer::Sglang, "org/m").unwrap_err();
        assert!(e.contains("goethe install org/m") && e.contains("SGLang"), "{e}");
        assert!(require(dir.path(), Layer::Engine, "org/m").unwrap_err().contains(ENGINE_VERSION));
        assert!(require(dir.path(), Layer::Base, "org/m").is_ok());
        let _l = lock(dir.path()).unwrap();
        assert!(dir.path().join("run/pyenv.lock").is_file());
    }
}
