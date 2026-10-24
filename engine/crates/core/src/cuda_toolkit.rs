// The CUDA toolkit bundled in the py-env, laid out so run-time kernel builds link without a system CUDA install.

use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

const DRIVER_LIB_DIRS: &[&str] = &[
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/aarch64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
    "/usr/lib/wsl/lib",
    "/usr/local/nvidia/lib64",
    "/usr/local/nvidia/lib",
    "/run/nvidia/driver/usr/lib/x86_64-linux-gnu",
];

pub fn bundled_home(home: &Path) -> Option<PathBuf> {
    // The toolkit pip-installed in the py-env (nvidia/cu13, else cu12) when it carries nvcc.
    let lib = std::fs::read_dir(home.join("py-env/lib")).ok()?;
    let mut sites: Vec<PathBuf> = lib.flatten().map(|e| e.path().join("site-packages/nvidia")).collect();
    sites.sort();
    ["cu13", "cu12"]
        .iter()
        .flat_map(|v| sites.iter().map(move |s| s.join(v)))
        .find(|d| d.join("bin/nvcc").is_file())
}

pub fn driver_library(name: &str) -> Option<PathBuf> {
    // A driver library such as libcuda.so.1 from the loader cache, else the usual host, WSL and container paths.
    let cache = std::process::Command::new("ldconfig").arg("-p").output().ok();
    let listed = cache.and_then(|o| {
        String::from_utf8_lossy(&o.stdout)
            .lines()
            .filter(|l| l.trim_start().starts_with(&format!("{name} ")))
            .filter_map(|l| l.rsplit("=> ").next().map(|p| PathBuf::from(p.trim())))
            .find(|p| p.exists())
    });
    listed.or_else(|| DRIVER_LIB_DIRS.iter().map(|d| Path::new(d).join(name)).find(|p| p.exists()))
}

pub fn prepare(home: &Path) -> Result<Option<PathBuf>, String> {
    // Add what pip wheels leave out and linkers expect: lib64, unversioned libraries and a libcuda.so stub link.
    let Some(root) = bundled_home(home) else {
        return Ok(None);
    };
    let lib = root.join("lib");
    link_if_missing(Path::new("lib"), &root.join("lib64"))?;
    for entry in std::fs::read_dir(&lib).map_err(|e| format!("{}: {e}", lib.display()))?.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some((stem, version)) = name.split_once(".so.") {
            if !version.contains('.') && !stem.contains(".alt") {
                link_if_missing(Path::new(&name), &lib.join(format!("{stem}.so")))?;
            }
        }
    }
    if let Some(libcuda) = driver_library("libcuda.so.1") {
        let stubs = lib.join("stubs");
        std::fs::create_dir_all(&stubs).map_err(|e| format!("{}: {e}", stubs.display()))?;
        link_if_missing(&libcuda, &stubs.join("libcuda.so"))?;
    }
    Ok(Some(root))
}

fn link_if_missing(target: &Path, link: &Path) -> Result<(), String> {
    if link.symlink_metadata().is_ok() {
        return Ok(());
    }
    symlink(target, link).map_err(|e| format!("link {}: {e}", link.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_adds_lib64_and_unversioned_libraries_once() {
        let td = tempfile::tempdir().unwrap();
        let cu = td.path().join("py-env/lib/python3.11/site-packages/nvidia/cu13");
        std::fs::create_dir_all(cu.join("bin")).unwrap();
        std::fs::create_dir_all(cu.join("lib")).unwrap();
        assert_eq!(prepare(td.path()).unwrap(), None, "no nvcc, no toolkit");
        std::fs::write(cu.join("bin/nvcc"), b"").unwrap();
        for f in ["libcudart.so.13", "libnvrtc.alt.so.13", "libnvrtc-builtins.so.13.0", "libcublas.so"] {
            std::fs::write(cu.join("lib").join(f), b"").unwrap();
        }
        assert_eq!(prepare(td.path()).unwrap(), Some(cu.clone()));
        assert!(prepare(td.path()).is_ok(), "idempotent");
        assert_eq!(std::fs::read_link(cu.join("lib64")).unwrap(), PathBuf::from("lib"));
        assert_eq!(std::fs::read_link(cu.join("lib/libcudart.so")).unwrap(), PathBuf::from("libcudart.so.13"));
        assert!(cu.join("lib/libcublas.so").symlink_metadata().unwrap().is_file(), "an existing file is kept");
        assert!(!cu.join("lib/libnvrtc.alt.so").exists() && !cu.join("lib/libnvrtc-builtins.so").exists());
    }
}
