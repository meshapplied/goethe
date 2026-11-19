// CUDA 13 on drivers 535 to 579: NVIDIA's forward-compatibility libraries in GOETHE_HOME/cuda-compat, used only below 580.

use std::ffi::OsString;
use std::path::{Path, PathBuf};

pub const DIR: &str = "cuda-compat";
pub const DRIVER_FLOOR: u32 = 535;

pub fn usable_with(major: u32) -> bool {
    // NVIDIA supports forward compatibility from these branches; on 580+ the libraries are refused (error 803).
    (DRIVER_FLOOR..crate::nvidia_driver::DRIVER_MIN).contains(&major)
}

pub fn installed(home: &Path) -> bool {
    home.join(DIR).join("libcuda.so.1").exists()
}

pub fn driver_major() -> Option<u32> {
    let text = std::fs::read_to_string("/proc/driver/nvidia/version").ok()?;
    crate::nvidia_driver::module_version(&text)?.split('.').next()?.parse().ok()
}

pub fn lib_dir(home: &Path) -> Option<PathBuf> {
    // The libraries to load first, when they are installed and the loaded driver needs them.
    (installed(home) && driver_major().is_some_and(usable_with)).then(|| home.join(DIR))
}

pub fn env(home: &Path) -> Vec<(String, OsString)> {
    // LD_LIBRARY_PATH with the compat libraries first, and Triton's libcuda directory, when they apply.
    let Some(dir) = lib_dir(home) else { return Vec::new() };
    vec![
        ("LD_LIBRARY_PATH".into(), library_path(&dir, std::env::var_os("LD_LIBRARY_PATH"))),
        ("TRITON_LIBCUDA_PATH".into(), dir.into_os_string()),
    ]
}

fn library_path(dir: &Path, inherited: Option<OsString>) -> OsString {
    let mut dirs = vec![dir.to_path_buf()];
    dirs.extend(inherited.iter().flat_map(std::env::split_paths).filter(|d| d != dir && !d.as_os_str().is_empty()));
    std::env::join_paths(dirs).unwrap_or_default()
}

pub fn check(home: &Path) -> Result<(), String> {
    // cuInit through the compat libraries in a child process; NVIDIA refuses some GPUs and drivers only at this point.
    let dir = home.join(DIR);
    let out = std::process::Command::new(home.join("py-env/bin/python"))
        .args(["-I", "-c", "import ctypes; print(ctypes.CDLL('libcuda.so.1').cuInit(0))"])
        .env("LD_LIBRARY_PATH", library_path(&dir, std::env::var_os("LD_LIBRARY_PATH")))
        .output()
        .map_err(|e| format!("could not run the py-env python: {e}"))?;
    let code = String::from_utf8_lossy(&out.stdout).trim().parse::<i64>().unwrap_or(-1);
    match code {
        0 => Ok(()),
        803 => Err("the driver is too new or too old for NVIDIA's CUDA 13 forward compatibility (error 803)".into()),
        804 => Err("NVIDIA does not offer CUDA 13 forward compatibility for this GPU, only for data-center GPUs (error 804)".into()),
        -1 => Err(format!("the forward-compatibility check failed: {}", String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or("").trim())),
        c => Err(format!("cuInit through the forward-compatibility libraries returned CUDA error {c}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_drivers_535_to_579_use_it() {
        assert!(usable_with(535) && usable_with(570) && usable_with(575));
        assert!(!usable_with(530) && !usable_with(580) && !usable_with(595));
    }

    #[test]
    fn compat_libraries_come_first_once() {
        let dir = Path::new("/h/cuda-compat");
        assert_eq!(library_path(dir, None), OsString::from("/h/cuda-compat"));
        assert_eq!(library_path(dir, Some("/opt/lib::/h/cuda-compat".into())), OsString::from("/h/cuda-compat:/opt/lib"));
    }
}
