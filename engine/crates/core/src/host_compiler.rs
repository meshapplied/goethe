// The host C++ compiler nvcc and torch need to build run-time kernels: find it, check it, and set it for engines.

use std::path::{Path, PathBuf};
use std::process::Command;

pub const GCC_MIN: u32 = 9;
pub const GCC_MAX: u32 = 15;
pub const CLANG_MAX: u32 = 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Compiler {
    pub cxx: PathBuf,
    pub cc: Option<PathBuf>,
    pub clang: bool,
    pub major: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Ok(String),
    TooNew(String),
    Unusable(String),
}

pub fn find() -> Option<Compiler> {
    // CXX when set, else c++ or g++ on PATH; its C sibling (CC, cc or gcc) goes with it.
    let pick = |env: &str, names: &[&str]| {
        std::env::var_os(env)
            .map(PathBuf::from)
            .filter(|p| which(p).is_some())
            .and_then(|p| which(&p))
            .or_else(|| names.iter().find_map(|n| which(Path::new(n))))
    };
    let cxx = pick("CXX", &["c++", "g++", "clang++"])?;
    let text = output(&cxx, &["--version"]);
    let clang = text.contains("clang");
    let major = output(&cxx, &["-dumpversion"]).trim().split('.').next().and_then(|m| m.parse().ok()).unwrap_or(0);
    let cc = pick("CC", if clang { &["clang", "cc"] } else { &["cc", "gcc"] });
    Some(Compiler { cxx, cc, clang, major })
}

fn which(p: &Path) -> Option<PathBuf> {
    if p.components().count() > 1 {
        return p.is_file().then(|| p.to_path_buf());
    }
    std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join(p)).find(|c| c.is_file())
}

fn output(bin: &Path, args: &[&str]) -> String {
    Command::new(bin).args(args).output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default()
}

pub fn verdict(c: Option<&Compiler>, distro: &str) -> Verdict {
    // Whether nvcc 13.0 can use this compiler: missing or too old cannot build kernels, newer than tested is allowed.
    let Some(c) = c else {
        return Verdict::Unusable(format!(
            "no C++ compiler: models whose kernels are built on first load (NVFP4 and other quantized models) cannot start; {}",
            install_hint(distro)
        ));
    };
    let (name, max) = if c.clang { ("clang", CLANG_MAX) } else { ("gcc", GCC_MAX) };
    let label = format!("{name} {} ({})", c.major, c.cxx.display());
    if !c.clang && c.major < GCC_MIN {
        return Verdict::Unusable(format!("{label} is older than {GCC_MIN}, which kernel builds need; {}", install_hint(distro)));
    }
    if c.major > max {
        return Verdict::TooNew(format!(
            "{label} is newer than the bundled nvcc 13.0 supports ({name} {max}); Goethe passes -allow-unsupported-compiler, and if a kernel build still fails install {name}-{max} and set CXX"
        ));
    }
    Verdict::Ok(label)
}

pub fn install_hint(distro: &str) -> String {
    let cmd = if distro.contains("ubuntu") || distro.contains("debian") {
        "sudo apt install build-essential"
    } else if distro.contains("rhel") || distro.contains("fedora") || distro.contains("centos") {
        "sudo dnf install gcc-c++"
    } else if distro.contains("suse") {
        "sudo zypper install gcc-c++"
    } else {
        "install gcc and g++ from your distribution"
    };
    format!("install one with {cmd} (or rerun the Goethe installer, which offers it)")
}

pub fn engine_env(c: Option<&Compiler>, inherited_nvcc_flags: Option<&str>) -> Vec<(String, String)> {
    // CXX/CC for torch and flashinfer builds, CUDAHOSTCXX for nvcc, and the override nvcc needs for a newer compiler.
    let Some(c) = c else { return Vec::new() };
    let mut env = vec![("CXX".to_string(), c.cxx.display().to_string()), ("CUDAHOSTCXX".to_string(), c.cxx.display().to_string())];
    if let Some(cc) = &c.cc {
        env.push(("CC".to_string(), cc.display().to_string()));
    }
    if matches!(verdict(Some(c), ""), Verdict::TooNew(_)) {
        let flags = inherited_nvcc_flags.unwrap_or("");
        if !flags.contains("allow-unsupported-compiler") {
            env.push(("NVCC_APPEND_FLAGS".to_string(), format!("{flags} -allow-unsupported-compiler").trim().to_string()));
        }
    }
    env
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gcc(major: u32) -> Compiler {
        Compiler { cxx: "/usr/bin/g++".into(), cc: Some("/usr/bin/gcc".into()), clang: false, major }
    }

    #[test]
    fn verdicts_follow_the_nvcc_limits() {
        assert!(matches!(verdict(None, "ubuntu debian"), Verdict::Unusable(w) if w.contains("apt install build-essential")));
        assert!(matches!(verdict(None, "rocky rhel"), Verdict::Unusable(w) if w.contains("dnf install gcc-c++")));
        assert_eq!(verdict(Some(&gcc(13)), ""), Verdict::Ok("gcc 13 (/usr/bin/g++)".into()));
        assert!(matches!(verdict(Some(&gcc(8)), ""), Verdict::Unusable(w) if w.contains("older than 9")));
        assert!(matches!(verdict(Some(&gcc(16)), ""), Verdict::TooNew(w) if w.contains("gcc 15")));
        let clang = Compiler { clang: true, major: 21, ..gcc(0) };
        assert!(matches!(verdict(Some(&clang), ""), Verdict::TooNew(w) if w.contains("clang 20")));
    }

    #[test]
    fn engines_get_the_compiler_and_the_override_only_when_needed() {
        assert!(engine_env(None, None).is_empty());
        let ok = engine_env(Some(&gcc(13)), None);
        assert!(ok.contains(&("CXX".into(), "/usr/bin/g++".into())) && ok.contains(&("CC".into(), "/usr/bin/gcc".into())));
        assert!(!ok.iter().any(|(k, _)| k == "NVCC_APPEND_FLAGS"));
        let new = engine_env(Some(&gcc(16)), Some("-O2"));
        assert!(new.contains(&("NVCC_APPEND_FLAGS".into(), "-O2 -allow-unsupported-compiler".into())));
        assert!(!engine_env(Some(&gcc(16)), Some("-allow-unsupported-compiler")).iter().any(|(k, _)| k == "NVCC_APPEND_FLAGS"));
    }
}
