// The one trust anchor for every release-key signature goethe checks: the key compiled into this binary, never one from the home or the environment.

use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::process::{Command, Stdio};

const COMPILED: &str = include_str!("../../../packaging/release_pub.pem");

pub const MISMATCH: &str = "the signature does not match this goethe's release key; update goethe with the installer (curl -fsSL https://goethe.sh/install.sh | bash)";

pub fn verify(what: &str, body: &[u8], sig: &[u8]) -> Result<(), String> {
    // Ok when `sig` is the release key's signature of `body` (package list, registry); goethe-engine pins and checks the pack keys itself, install.sh embeds this same key and still writes it to the home for old tools that nothing here reads.
    let key = key_pem()?;
    verify_with_key(&key, what, body, sig)
}

fn key_pem() -> Result<String, String> {
    // The compiled key; a rollover is a new binary carrying the new key whose package list the old key signs; only a test-release-key build reads GOETHE_RELEASE_PUB, a release build refuses to run while it is set.
    #[cfg(test)]
    if let Some(pem) = test_key::get() {
        return Ok(pem);
    }
    if std::env::var_os("GOETHE_RELEASE_PUB").is_some() {
        #[cfg(feature = "test-release-key")]
        {
            let path = std::env::var_os("GOETHE_RELEASE_PUB").unwrap_or_default();
            return std::fs::read_to_string(&path).map_err(|e| format!("GOETHE_RELEASE_PUB {}: {e}", Path::new(&path).display()));
        }
        #[cfg(not(feature = "test-release-key"))]
        return Err("GOETHE_RELEASE_PUB is set, and a release goethe never takes its release key from the environment; unset it".to_string());
    }
    Ok(COMPILED.to_string())
}

fn verify_with_key(pem: &str, what: &str, body: &[u8], sig: &[u8]) -> Result<(), String> {
    let dir = crate::paths::private_temp_dir("goethe-sig")?;
    let checked = check(&dir, pem, body, sig);
    let _ = std::fs::remove_dir_all(&dir);
    match checked? {
        true => Ok(()),
        false => Err(format!("{what}: {MISMATCH}")),
    }
}

fn check(dir: &Path, pem: &str, body: &[u8], sig: &[u8]) -> Result<bool, String> {
    let write = |name: &str, bytes: &[u8]| -> Result<std::path::PathBuf, String> {
        let path = dir.join(name);
        let mut f = std::fs::OpenOptions::new().write(true).create_new(true).mode(0o600).open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        f.write_all(bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(path)
    };
    let (key, body, sig) = (write("key.pem", pem.as_bytes())?, write("body", body)?, write("body.sig", sig)?);
    let status = Command::new(openssl()?)
        .args(["dgst", "-sha256", "-verify"])
        .arg(&key)
        .arg("-signature")
        .arg(&sig)
        .arg(&body)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("cannot check a signature: openssl did not run ({e}); install openssl"))?;
    Ok(status.success())
}

const OPENSSL: [&str; 3] = ["/usr/bin/openssl", "/bin/openssl", "/usr/local/bin/openssl"];

fn openssl() -> Result<&'static str, String> {
    // The system openssl by absolute path, never the first one on PATH, which a planted binary that always answers yes could be.
    OPENSSL
        .into_iter()
        .find(|p| Path::new(p).is_file())
        .ok_or_else(|| format!("cannot check a signature: no openssl in {}; install openssl", OPENSSL.join(", ")))
}

#[cfg(test)]
pub(crate) mod test_key {
    use std::sync::{Mutex, MutexGuard};

    static KEY: Mutex<Option<String>> = Mutex::new(None);
    static SERIAL: Mutex<()> = Mutex::new(());

    pub struct Held(#[allow(dead_code)] MutexGuard<'static, ()>);

    impl Drop for Held {
        fn drop(&mut self) {
            KEY.lock().unwrap_or_else(|e| e.into_inner()).take();
        }
    }

    pub fn set(pem: &str) -> Held {
        let held = hold();
        KEY.lock().unwrap_or_else(|e| e.into_inner()).replace(pem.to_string());
        held
    }

    pub fn hold() -> Held {
        let held = Held(SERIAL.lock().unwrap_or_else(|e| e.into_inner()));
        KEY.lock().unwrap_or_else(|e| e.into_inner()).take();
        held
    }

    pub fn get() -> Option<String> {
        KEY.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keypair(dir: &Path, name: &str) -> (std::path::PathBuf, String) {
        let key = dir.join(format!("{name}.key"));
        let run = |args: &[&str]| assert!(Command::new("openssl").args(args).stderr(Stdio::null()).status().unwrap().success());
        run(&["ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", key.to_str().unwrap()]);
        let out = Command::new("openssl").args(["ec", "-in", key.to_str().unwrap(), "-pubout"]).stderr(Stdio::null()).output().unwrap();
        (key, String::from_utf8(out.stdout).unwrap())
    }

    fn sign(dir: &Path, key: &Path, body: &[u8]) -> Vec<u8> {
        let (data, sig) = (dir.join("signed.body"), dir.join("signed.sig"));
        std::fs::write(&data, body).unwrap();
        assert!(Command::new("openssl").args(["dgst", "-sha256", "-sign", key.to_str().unwrap(), "-out", sig.to_str().unwrap(), data.to_str().unwrap()]).status().unwrap().success());
        std::fs::read(sig).unwrap()
    }

    #[test]
    fn a_list_signed_by_the_key_verifies_a_tampered_one_and_another_key_do_not() {
        let dir = tempfile::tempdir().unwrap();
        let ((key, pem), (_, other)) = (keypair(dir.path(), "a"), keypair(dir.path(), "b"));
        let sig = sign(dir.path(), &key, b"SHA256SUMS body");
        assert_eq!(verify_with_key(&pem, "the package list", b"SHA256SUMS body", &sig), Ok(()));
        let tampered = verify_with_key(&pem, "the package list", b"SHA256SUMS body!", &sig).unwrap_err();
        assert_eq!(tampered, format!("the package list: {MISMATCH}"));
        assert!(verify_with_key(&other, "the package list", b"SHA256SUMS body", &sig).is_err());
        assert!(!MISMATCH.contains("release_pub") && MISMATCH.contains("update goethe with the installer"));
    }

    #[test]
    fn the_openssl_that_checks_comes_from_a_system_folder_not_path() {
        let found = openssl().unwrap();
        assert!(OPENSSL.contains(&found) && found.starts_with('/'));
        let src = include_str!("release_key.rs");
        let code = src.split("\n#[cfg(test)]\npub(crate) mod test_key").next().unwrap();
        assert_eq!(code.matches("Command::new(").count(), 1, "one openssl call");
        assert!(code.contains("Command::new(openssl()?)"));
    }

    #[test]
    fn a_fresh_home_without_any_key_file_checks_against_the_compiled_key() {
        let dir = tempfile::tempdir().unwrap();
        let (key, pem) = keypair(dir.path(), "a");
        let sig = sign(dir.path(), &key, b"list");
        let _held = test_key::set(&pem);
        assert_eq!(verify("the package list", b"list", &sig), Ok(()), "the key in force verifies, no file in any home is read");
        assert!(COMPILED.contains("BEGIN PUBLIC KEY"));
    }

    #[test]
    fn an_attacker_key_is_not_the_compiled_key_and_the_environment_cannot_name_one() {
        let dir = tempfile::tempdir().unwrap();
        let (attacker_key, attacker_pub) = keypair(dir.path(), "attacker");
        let sig = sign(dir.path(), &attacker_key, b"evil list");
        let _held = test_key::hold();
        let err = verify("the package list", b"evil list", &sig).unwrap_err();
        assert!(err.ends_with(MISMATCH), "a list signed by another key is refused: {err}");
        assert_eq!(key_pem().unwrap(), COMPILED, "the key in force is the compiled one");
        std::fs::write(dir.path().join("env.pem"), &attacker_pub).unwrap();
        std::env::set_var("GOETHE_RELEASE_PUB", dir.path().join("env.pem"));
        let named = key_pem();
        std::env::remove_var("GOETHE_RELEASE_PUB");
        let refused = "GOETHE_RELEASE_PUB is set, and a release goethe never takes its release key from the environment; unset it";
        assert_eq!(named, if cfg!(feature = "test-release-key") { Ok(attacker_pub) } else { Err(refused.to_string()) });
    }

    #[test]
    fn every_signature_check_goes_through_this_module() {
        fn rust_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
            for e in std::fs::read_dir(dir).unwrap().flatten() {
                let p = e.path();
                if p.is_dir() {
                    rust_files(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }
        let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut files = Vec::new();
        rust_files(&crates, &mut files);
        assert!(files.len() > 20);
        for f in files {
            if f.ends_with("core/src/release_key.rs") {
                continue;
            }
            let text = std::fs::read_to_string(&f).unwrap();
            let text = text.split("#[cfg(test)]").next().unwrap_or_default();
            for word in ["release_pub", "RELEASE_PUB", "dgst", "\"-verify\""] {
                assert!(!text.contains(word), "{} names {word}: signature checks with the release key belong to release_key.rs", f.display());
            }
        }
    }
}
