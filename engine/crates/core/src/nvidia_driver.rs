// Why the NVIDIA driver is not usable here (no GPU, no kernel module, no device nodes, missing libraries) and the fix.

use nvml_wrapper::Nvml;
use std::path::Path;

pub const DRIVER_MIN: u32 = 580;
const INSTALLER: &str = "rerun the Goethe installer, which offers to install it,";

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Probe {
    pub pci_gpus: Vec<String>,
    pub module_version: Option<String>,
    pub device_nodes: bool,
    pub libcuda: bool,
    pub libnvml: bool,
    pub container: bool,
    pub distro: String,
}

pub fn nvml() -> Result<Nvml, String> {
    // NVML, or an error that says why the driver is unusable and how to fix it.
    Nvml::init().map_err(|e| match diagnose(&probe()) {
        Some(why) => format!("NVML init failed: {why}"),
        None => format!("NVML init failed: {e}"),
    })
}

pub fn probe() -> Probe {
    // What the machine shows: NVIDIA PCI functions, the loaded kernel module, /dev nodes and driver libraries.
    let pci_gpus = std::fs::read_dir("/sys/bus/pci/devices")
        .map(|dir| {
            let mut out: Vec<String> = dir
                .flatten()
                .filter(|e| {
                    let read = |f: &str| std::fs::read_to_string(e.path().join(f)).unwrap_or_default();
                    read("vendor").trim() == "0x10de" && read("class").trim().starts_with("0x03")
                })
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect();
            out.sort();
            out
        })
        .unwrap_or_default();
    let module_version = std::fs::read_to_string("/proc/driver/nvidia/version").ok().and_then(|t| module_version(&t));
    let os = std::fs::read_to_string("/etc/os-release").unwrap_or_default();
    Probe {
        pci_gpus,
        module_version,
        device_nodes: Path::new("/dev/nvidiactl").exists(),
        libcuda: crate::cuda_toolkit::driver_library("libcuda.so.1").is_some(),
        libnvml: crate::cuda_toolkit::driver_library("libnvidia-ml.so.1").is_some(),
        container: Path::new("/.dockerenv").exists() || Path::new("/run/.containerenv").exists(),
        distro: os_id(&os),
    }
}

fn module_version(text: &str) -> Option<String> {
    // "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  580.82.07  Release Build ..." into "580.82.07".
    let line = text.lines().find(|l| l.starts_with("NVRM version"))?;
    line.split_whitespace().find(|w| w.split('.').count() >= 2 && w.split('.').all(|p| p.parse::<u32>().is_ok())).map(String::from)
}

fn os_id(os_release: &str) -> String {
    let field = |k: &str| os_release.lines().find_map(|l| l.strip_prefix(k)).map(|v| v.trim_matches('"').to_string());
    format!("{} {}", field("ID=").unwrap_or_default(), field("ID_LIKE=").unwrap_or_default())
}

fn install_command(distro: &str) -> &'static str {
    if distro.contains("ubuntu") {
        "sudo ubuntu-drivers install --gpgpu nvidia:580-server-open (or sudo apt install nvidia-driver-580-server-open), then reboot"
    } else if distro.contains("debian") {
        "install nvidia-open 580 or newer from NVIDIA's CUDA repository for Debian (sudo apt install nvidia-open), then reboot"
    } else if distro.contains("rhel") || distro.contains("fedora") || distro.contains("centos") {
        "add NVIDIA's CUDA repository and run sudo dnf install nvidia-open (580 or newer), then reboot"
    } else {
        "install the NVIDIA driver 580 or newer from your distribution or https://www.nvidia.com/drivers, then reboot"
    }
}

pub fn diagnose(p: &Probe) -> Option<String> {
    // None when the driver looks usable; otherwise the first blocker with what to do about it.
    if p.pci_gpus.is_empty() && p.module_version.is_none() && !p.device_nodes {
        return Some(if p.container {
            "no NVIDIA GPU in this container: start it with the GPU attached (docker run --gpus all, which needs the NVIDIA Container Toolkit on the host)".to_string()
        } else {
            "no NVIDIA GPU visible to this machine; in a virtual machine, attach the GPU with PCI passthrough or vGPU first".to_string()
        });
    }
    let Some(version) = &p.module_version else {
        return Some(format!(
            "NVIDIA GPU at {} but no NVIDIA driver is loaded; {INSTALLER} or {}",
            p.pci_gpus.join(", "),
            install_command(&p.distro)
        ));
    };
    let major = version.split('.').next().and_then(|m| m.parse::<u32>().ok()).unwrap_or(0);
    if major < DRIVER_MIN {
        return Some(format!(
            "NVIDIA driver {version} is older than {DRIVER_MIN} (CUDA 13); {INSTALLER} or {}",
            install_command(&p.distro)
        ));
    }
    if !p.device_nodes {
        return Some(format!(
            "driver {version} is loaded but /dev/nvidia* is missing here; in a container pass the GPU through (docker run --gpus all, or the /dev/nvidia* devices), on a host run sudo nvidia-modprobe -u -c=0"
        ));
    }
    if !p.libcuda || !p.libnvml {
        return Some(format!(
            "driver {version} is loaded but its libraries (libcuda.so.1, libnvidia-ml.so.1) are not installed here; in a container use the NVIDIA Container Toolkit, otherwise install the user-space driver of the same version (Ubuntu: sudo apt install libnvidia-compute-{major})"
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready() -> Probe {
        Probe {
            pci_gpus: vec!["0000:01:00.0".into()],
            module_version: Some("580.82.07".into()),
            device_nodes: true,
            libcuda: true,
            libnvml: true,
            container: false,
            distro: "ubuntu debian".into(),
        }
    }

    #[test]
    fn reads_the_module_version() {
        let t = "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  595.84  Release Build  (dvs-builder@U22)  Fri\nGCC version: gcc 13";
        assert_eq!(module_version(t).as_deref(), Some("595.84"));
        assert_eq!(os_id("ID=rocky\nID_LIKE=\"rhel centos fedora\"\n"), "rocky rhel centos fedora");
    }

    #[test]
    fn each_blocker_gets_its_own_fix() {
        assert_eq!(diagnose(&ready()), None);
        let vm = Probe { module_version: None, device_nodes: false, libcuda: false, libnvml: false, ..ready() };
        assert!(diagnose(&vm).unwrap().starts_with("NVIDIA GPU at 0000:01:00.0 but no NVIDIA driver is loaded; rerun the Goethe installer"));
        assert!(diagnose(&vm).unwrap().contains("or sudo ubuntu-drivers install --gpgpu nvidia:580-server-open"));
        let old = Probe { module_version: Some("550.120".into()), ..ready() };
        assert!(diagnose(&old).unwrap().contains("older than 580"));
        let no_nodes = Probe { device_nodes: false, ..ready() };
        assert!(diagnose(&no_nodes).unwrap().contains("/dev/nvidia* is missing"));
        let no_libs = Probe { libcuda: false, ..ready() };
        assert!(diagnose(&no_libs).unwrap().contains("libnvidia-compute-580"));
        let bare = Probe { pci_gpus: vec![], module_version: None, device_nodes: false, ..ready() };
        assert!(diagnose(&bare).unwrap().contains("PCI passthrough"));
        assert!(diagnose(&Probe { container: true, ..bare }).unwrap().contains("--gpus all"));
        let rocky = Probe { module_version: None, distro: "rocky rhel centos fedora".into(), ..ready() };
        assert!(diagnose(&rocky).unwrap().contains("dnf install nvidia-open"));
    }
}
