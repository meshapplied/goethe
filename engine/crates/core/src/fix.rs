// goethe doctor --fix: the system changes the installer used to make from curl | bash (NVIDIA driver, CUDA forward compatibility, a C++ compiler), each one asked for.

use std::path::Path;
use std::process::Command;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fix {
    Load,
    Open,
    Install,
    Compat,
    Compiler(Vec<String>),
}

#[derive(Clone, Debug, Default)]
pub struct Facts {
    pub pci_gpus: usize,
    pub container: bool,
    pub wsl: bool,
    pub driver: Option<String>,
    pub module_installed: bool,
    pub started_gpus: usize,
    pub open_module: bool,
    pub blackwell: bool,
    pub geforce_only: bool,
    pub compat_installed: bool,
    pub compiler_packages: Option<Vec<String>>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    Fine(String),
    Elsewhere(String),
    Fixes(Vec<(Fix, String)>),
}

pub fn decide(f: &Facts) -> Verdict {
    // What is wrong and what fixes it, in the order the installer used to check; a container or WSL is fixed on its host.
    let mut fixes = Vec::new();
    let major = f.driver.as_deref().and_then(|v| v.split('.').next()?.parse::<u32>().ok());
    let min = crate::nvidia_driver::DRIVER_MIN;
    if f.pci_gpus > 0 || f.driver.is_some() {
        if f.container || f.wsl {
            let host = if f.wsl { "Windows (the NVIDIA driver for WSL, 580 or newer)" } else { "the host, and start the container with --gpus all" };
            if f.driver.is_none() || major.is_some_and(|m| m < min && !f.compat_installed) {
                return Verdict::Elsewhere(format!("the NVIDIA driver is fixed outside this {}: on {host}", if f.wsl { "WSL" } else { "container" }));
            }
        } else {
            match (major, &f.driver) {
                (_, None) if f.module_installed => fixes.push((Fix::Load, "the NVIDIA driver is installed but its kernel module is not loaded".to_string())),
                (_, None) => fixes.push((Fix::Install, "an NVIDIA GPU is here but no NVIDIA driver is installed".to_string())),
                (Some(m), Some(_)) if m < min && f.compat_installed => {}
                (Some(m), Some(v)) if (535..min).contains(&m) && !f.geforce_only => {
                    fixes.push((Fix::Compat, format!("NVIDIA driver {v} is older than {min}; NVIDIA's forward-compatibility libraries run CUDA 13 on it without a driver change")))
                }
                (Some(m), Some(v)) if m < min => fixes.push((Fix::Install, format!("NVIDIA driver {v} is older than {min} (CUDA 13)"))),
                _ if f.started_gpus == 0 && !f.open_module && f.blackwell => {
                    fixes.push((Fix::Open, "the proprietary NVIDIA kernel module cannot start Blackwell GPUs; they need the open module".to_string()))
                }
                _ => {}
            }
        }
    }
    if let Some(pkgs) = &f.compiler_packages {
        fixes.push((Fix::Compiler(pkgs.clone()), "SGLang builds GPU kernels on a model's first load and needs a C++ compiler (gcc 9 to 15)".to_string()));
    }
    if fixes.is_empty() {
        let state = match (&f.driver, f.pci_gpus) {
            (Some(v), _) => format!("NVIDIA driver {v}"),
            (None, 0) => "no NVIDIA GPU on this machine".to_string(),
            (None, _) => "no driver".to_string(),
        };
        return Verdict::Fine(format!("{state}: nothing to fix"));
    }
    Verdict::Fixes(fixes)
}

pub fn question(fix: &Fix, distro: &str, sudo: bool) -> String {
    let via = if sudo { " (uses sudo)" } else { "" };
    let distro = distro.split_whitespace().next().unwrap_or("the distribution's");
    match fix {
        Fix::Load => format!("Load it now with modprobe nvidia{via}?"),
        Fix::Open => format!("Switch to NVIDIA's open kernel module {} with {distro} packages{via}? It needs a reboot", crate::nvidia_driver::DRIVER_MIN),
        Fix::Install => format!("Install the NVIDIA driver {}+ with {distro} packages{via}? It may need a reboot", crate::nvidia_driver::DRIVER_MIN),
        Fix::Compat => "Download NVIDIA's CUDA 13 forward-compatibility libraries (about 60 MB) into goethe's folder?".to_string(),
        Fix::Compiler(pkgs) => format!("Install {}{via}?", pkgs.join(" ")),
    }
}

pub fn compiler_packages(distro: &str) -> Vec<String> {
    let words: Vec<&str> = distro.split_whitespace().collect();
    let has = |d: &str| words.contains(&d);
    if has("ubuntu") || has("debian") {
        vec!["build-essential".to_string()]
    } else if has("suse") || has("opensuse") || has("sles") {
        vec!["gcc-c++".to_string(), "make".to_string()]
    } else {
        vec!["gcc".to_string(), "gcc-c++".to_string(), "make".to_string()]
    }
}

pub fn package_command(distro: &str, pkgs: &[String]) -> Vec<String> {
    let words: Vec<&str> = distro.split_whitespace().collect();
    let mut cmd: Vec<String> = if words.contains(&"ubuntu") || words.contains(&"debian") {
        vec!["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y", "-qq"]
    } else if words.iter().any(|w| w.contains("suse") || *w == "sles") {
        vec!["zypper", "-n", "install"]
    } else {
        vec!["dnf", "-y", "install"]
    }
    .into_iter()
    .map(String::from)
    .collect();
    cmd.extend(pkgs.iter().cloned());
    cmd
}

fn facts(home: &Path) -> Facts {
    let p = crate::nvidia_driver::probe();
    let names: Vec<String> = crate::nvidia_driver::nvml()
        .ok()
        .map(|n| (0..n.device_count().unwrap_or(0)).filter_map(|i| n.device_by_index(i).ok()?.name().ok()).collect())
        .unwrap_or_default();
    let compiler = matches!(crate::host_compiler::verdict(crate::host_compiler::find().as_ref(), &p.distro), crate::host_compiler::Verdict::Unusable(_));
    Facts {
        pci_gpus: p.pci_gpus.len(),
        container: p.container,
        wsl: p.wsl,
        driver: p.module_version.clone(),
        module_installed: Command::new("modinfo").arg("nvidia").output().is_ok_and(|o| o.status.success()),
        started_gpus: crate::nvidia_driver::started_bus_ids().len(),
        open_module: p.open_module,
        blackwell: p.blackwell_or_newer,
        geforce_only: !names.is_empty() && names.iter().all(|n| { let l = n.to_lowercase(); l.contains("geforce") || l.contains("titan") }),
        compat_installed: crate::cuda_compat::installed(home),
        compiler_packages: (compiler && crate::pyenv::has(home, crate::pyenv::Layer::Sglang)).then(|| compiler_packages(&p.distro)),
    }
}

pub fn pending(home: &Path) -> Vec<(Fix, String)> {
    // The fixes doctor mentions under its table; none on a Mac.
    if cfg!(target_os = "macos") {
        return Vec::new();
    }
    match decide(&facts(home)) {
        Verdict::Fixes(f) => f,
        _ => Vec::new(),
    }
}

fn root_prefix() -> Result<Vec<String>, String> {
    if unsafe { libc::geteuid() } == 0 {
        return Ok(Vec::new());
    }
    if which("sudo") {
        return Ok(vec!["sudo".to_string()]);
    }
    Err("this needs root: run goethe doctor --fix as root, or install sudo".to_string())
}

fn which(cmd: &str) -> bool {
    std::env::var("PATH").unwrap_or_default().split(':').any(|d| Path::new(d).join(cmd).is_file())
}

fn run_script(home: &Path, action: &str, as_root: bool) -> Result<(), String> {
    const SCRIPT: &str = include_str!("fix_driver.sh");
    let dir = home.join("run");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join("fix_driver.sh");
    std::fs::write(&path, SCRIPT).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut argv = if as_root { root_prefix()? } else { Vec::new() };
    argv.extend(["bash".to_string(), path.display().to_string(), action.to_string()]);
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..]).env("GOETHE_HOME", home);
    if action == "compat" {
        crate::pyenv::ensure(home, crate::pyenv::Layer::Base)?;
        cmd.env("PYTHON", home.join("py-env/bin/python"));
    }
    let status = cmd.status().map_err(|e| format!("{}: {e}", argv[0]))?;
    let _ = std::fs::remove_file(&path);
    status.success().then_some(()).ok_or_else(|| format!("the {action} step did not finish; goethe doctor shows what is still wrong"))
}

pub fn apply(home: &Path, fix: &Fix) -> Result<(), String> {
    match fix {
        Fix::Load => run_script(home, "load", true),
        Fix::Open => run_script(home, "open", true),
        Fix::Install => run_script(home, "install", true),
        Fix::Compat => run_script(home, "compat", false),
        Fix::Compiler(pkgs) => {
            let mut argv = root_prefix()?;
            argv.extend(package_command(&crate::nvidia_driver::probe().distro, pkgs));
            let ok = Command::new(&argv[0]).args(&argv[1..]).status().map_err(|e| e.to_string())?.success();
            ok.then_some(()).ok_or_else(|| format!("installing {} failed", pkgs.join(" ")))
        }
    }
}

pub fn run(yes: bool) -> Result<(), String> {
    // Each fix is asked for (GOETHE_INSTALL_DRIVER or GOETHE_INSTALL_DEPS answer for scripts, --yes for all), then run; doctor checks again after.
    let home = crate::paths::goethe_home();
    if cfg!(target_os = "macos") {
        println!("Nothing to fix: a Mac needs no GPU driver or compiler for goethe.");
        return Ok(());
    }
    let f = facts(&home);
    let distro = crate::nvidia_driver::probe().distro;
    match decide(&f) {
        Verdict::Fine(text) => {
            println!("{text}.");
            Ok(())
        }
        Verdict::Elsewhere(text) => Err(text),
        Verdict::Fixes(fixes) => {
            let sudo = unsafe { libc::geteuid() } != 0;
            let mut failed = None;
            for (fix, why) in fixes {
                println!("{why}.");
                let setting = if matches!(fix, Fix::Compiler(_)) { "GOETHE_INSTALL_DEPS" } else { "GOETHE_INSTALL_DRIVER" };
                let needs_root = !matches!(fix, Fix::Compat);
                if !yes && !crate::setup::ask(&question(&fix, &distro, sudo && needs_root), setting) {
                    println!("skipped; run goethe doctor --fix again any time");
                    continue;
                }
                if let Err(e) = apply(&home, &fix) {
                    eprintln!("{e}");
                    failed = Some(e);
                }
            }
            failed.map_or(Ok(()), Err)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card() -> Facts {
        Facts { pci_gpus: 1, started_gpus: 1, driver: Some("580.82.07".into()), ..Facts::default() }
    }

    #[test]
    fn a_working_driver_or_no_gpu_needs_nothing() {
        assert!(matches!(decide(&card()), Verdict::Fine(t) if t.contains("580.82.07")));
        assert!(matches!(decide(&Facts::default()), Verdict::Fine(t) if t.contains("no NVIDIA GPU")));
    }

    #[test]
    fn each_driver_problem_gets_its_fix() {
        let no_driver = Facts { driver: None, started_gpus: 0, ..card() };
        assert_eq!(decide(&no_driver), Verdict::Fixes(vec![(Fix::Install, "an NVIDIA GPU is here but no NVIDIA driver is installed".into())]));
        assert!(matches!(decide(&Facts { module_installed: true, ..no_driver.clone() }), Verdict::Fixes(f) if f[0].0 == Fix::Load));
        let old = Facts { driver: Some("550.54.15".into()), ..card() };
        assert!(matches!(decide(&old), Verdict::Fixes(f) if f[0].0 == Fix::Compat));
        assert!(matches!(decide(&Facts { geforce_only: true, ..old.clone() }), Verdict::Fixes(f) if f[0].0 == Fix::Install), "GeForce gets no forward compatibility");
        assert!(matches!(decide(&Facts { compat_installed: true, ..old }), Verdict::Fine(_)), "compat already serves it");
        assert!(matches!(decide(&Facts { driver: Some("525.1".into()), ..card() }), Verdict::Fixes(f) if f[0].0 == Fix::Install));
        let blackwell = Facts { blackwell: true, started_gpus: 0, ..card() };
        assert!(matches!(decide(&blackwell), Verdict::Fixes(f) if f[0].0 == Fix::Open));
        assert!(matches!(decide(&Facts { open_module: true, ..blackwell }), Verdict::Fine(_)), "the open module that fails to start a card is not a driver choice");
    }

    #[test]
    fn a_container_or_wsl_is_fixed_on_its_host() {
        assert!(matches!(decide(&Facts { container: true, driver: None, ..card() }), Verdict::Elsewhere(t) if t.contains("--gpus all")));
        assert!(matches!(decide(&Facts { wsl: true, driver: Some("550.1".into()), ..card() }), Verdict::Elsewhere(t) if t.contains("Windows")));
        assert!(matches!(decide(&Facts { container: true, ..card() }), Verdict::Fine(_)));
    }

    #[test]
    fn a_compiler_is_offered_only_when_asked_for_and_by_distribution() {
        let f = Facts { compiler_packages: Some(compiler_packages("ubuntu debian")), ..card() };
        assert_eq!(decide(&f), Verdict::Fixes(vec![(Fix::Compiler(vec!["build-essential".into()]), "SGLang builds GPU kernels on a model's first load and needs a C++ compiler (gcc 9 to 15)".into())]));
        assert_eq!(compiler_packages("rocky rhel centos fedora"), vec!["gcc", "gcc-c++", "make"]);
        assert_eq!(package_command("debian ", &["build-essential".into()])[2..], ["apt-get", "install", "-y", "-qq", "build-essential"]);
        assert_eq!(package_command("rhel", &["gcc".into()]), ["dnf", "-y", "install", "gcc"]);
    }

    #[test]
    fn the_fix_script_parses_and_knows_its_actions() {
        let out = Command::new("bash").args(["-n", concat!(env!("CARGO_MANIFEST_DIR"), "/src/fix_driver.sh")]).output().unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let usage = Command::new("bash").args([concat!(env!("CARGO_MANIFEST_DIR"), "/src/fix_driver.sh"), "nonsense"]).output().unwrap();
        assert_eq!(usage.status.code(), Some(2));
    }
}
