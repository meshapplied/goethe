// Why the NVIDIA driver is not usable here (no GPU, no kernel module, no device nodes, missing libraries) and the fix.

use nvml_wrapper::Nvml;
use std::path::Path;

pub const DRIVER_MIN: u32 = 580;
const BLACKWELL_FIRST_DEVICE_ID: u16 = 0x2900;
const WSL_FIX: &str = "WSL gets the GPU from the Windows NVIDIA driver: install or update it on Windows (580 or newer), then run wsl --shutdown and reopen WSL; never install a Linux NVIDIA driver inside WSL";
const INSTALLER: &str = "rerun the Goethe installer, which offers to install it,";

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Probe {
    pub pci_gpus: Vec<String>,
    pub assigned_gpus: Vec<String>,
    pub wsl: bool,
    pub blackwell_or_newer: bool,
    pub module_version: Option<String>,
    pub open_module: bool,
    pub device_nodes: bool,
    pub libcuda: bool,
    pub libnvml: bool,
    pub container: bool,
    pub cuda_compat: bool,
    pub distro: String,
}

pub fn init_nvml() -> Result<Nvml, nvml_wrapper::error::NvmlError> {
    // libnvidia-ml.so.1 first: GPU containers mount only the versioned library, not the unversioned dev link.
    Nvml::builder().lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1")).init().or_else(|_| Nvml::init())
}

pub fn nvml() -> Result<Nvml, String> {
    // NVML, or an error that says why the driver is unusable and how to fix it.
    init_nvml().map_err(|e| match diagnose(&probe()) {
        Some(why) => format!("NVML init failed: {why}"),
        None => format!("NVML init failed: {e}"),
    })
}

pub fn probe() -> Probe {
    // What the machine shows: NVIDIA PCI functions, the loaded kernel module, /dev nodes and driver libraries.
    let mut devices: Vec<(String, u16)> = std::fs::read_dir("/sys/bus/pci/devices")
        .map(|dir| {
            dir.flatten()
                .filter_map(|e| {
                    let read = |f: &str| std::fs::read_to_string(e.path().join(f)).unwrap_or_default();
                    let id = u16::from_str_radix(read("device").trim().trim_start_matches("0x"), 16).unwrap_or(0);
                    (read("vendor").trim() == "0x10de" && read("class").trim().starts_with("0x03"))
                        .then(|| (e.file_name().to_string_lossy().into_owned(), id))
                })
                .collect()
        })
        .unwrap_or_default();
    devices.sort();
    let version_text = std::fs::read_to_string("/proc/driver/nvidia/version").unwrap_or_default();
    let assigned_gpus = devices
        .iter()
        .map(|(addr, _)| addr.clone())
        .filter(|addr| {
            let info = std::fs::read_to_string(format!("/proc/driver/nvidia/gpus/{addr}/information")).unwrap_or_default();
            device_minor(&info).is_none_or(|m| Path::new(&format!("/dev/nvidia{m}")).exists())
        })
        .collect();
    let os = std::fs::read_to_string("/etc/os-release").unwrap_or_default();
    Probe {
        blackwell_or_newer: devices.iter().any(|(_, id)| *id >= BLACKWELL_FIRST_DEVICE_ID),
        pci_gpus: devices.into_iter().map(|(addr, _)| addr).collect(),
        assigned_gpus,
        wsl: Path::new("/dev/dxg").exists() || Path::new("/usr/lib/wsl/lib").is_dir(),
        module_version: module_version(&version_text),
        open_module: version_text.contains("Open Kernel Module"),
        device_nodes: Path::new("/dev/nvidiactl").exists(),
        libcuda: crate::cuda_toolkit::driver_library("libcuda.so.1").is_some(),
        libnvml: crate::cuda_toolkit::driver_library("libnvidia-ml.so.1").is_some(),
        container: crate::host::container_kind().is_some(),
        cuda_compat: crate::cuda_compat::installed(&crate::paths::goethe_home()),
        distro: os_id(&os),
    }
}

fn device_minor(information: &str) -> Option<u32> {
    // "Device Minor: 2" from /proc/driver/nvidia/gpus/<bus>/information.
    information.lines().find_map(|l| l.strip_prefix("Device Minor:")).and_then(|v| v.trim().parse().ok())
}

pub(crate) fn module_version(text: &str) -> Option<String> {
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
    if p.wsl {
        return (!p.libcuda || !p.libnvml).then(|| WSL_FIX.to_string());
    }
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
    let compat = crate::cuda_compat::usable_with(major);
    if major < DRIVER_MIN && !(compat && p.cuda_compat) {
        let alternative = if compat {
            "; data-center GPUs can keep this driver and use CUDA 13 forward compatibility, which the installer sets up"
        } else {
            ""
        };
        return Some(format!(
            "NVIDIA driver {version} is older than {DRIVER_MIN} (CUDA 13); {INSTALLER} or {}{alternative}",
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

pub fn unstarted(p: &Probe, started: &[String]) -> Option<(bool, String)> {
    // GPUs on the PCI bus that NVML cannot open; (true, why) when none started, (false, note) when only some did.
    if p.container || p.wsl {
        return None;
    }
    let started: Vec<&str> = started.iter().map(|b| bus_tail(b)).collect();
    let missing: Vec<&str> = p.assigned_gpus.iter().map(|b| b.as_str()).filter(|b| !started.contains(&bus_tail(b))).collect();
    if missing.is_empty() {
        return None;
    }
    let version = p.module_version.as_deref().unwrap_or("?");
    let at = missing.join(", ");
    if !started.is_empty() {
        return Some((false, format!("the GPU at {at} did not start and is skipped; {}", failed_init_fix())));
    }
    if !p.open_module && p.blackwell_or_newer {
        return Some((
            true,
            format!(
                "driver {version} is NVIDIA's proprietary kernel module, which cannot start Blackwell GPUs (at {at}); {INSTALLER} or switch to the open module: {}",
                install_command(&p.distro)
            ),
        ));
    }
    Some((true, format!("driver {version} is loaded but the GPU at {at} did not start; {}", failed_init_fix())))
}

fn failed_init_fix() -> &'static str {
    "the card failed to initialise (sudo dmesg | grep -i nvrm shows why): power the machine fully off and on (on a VM stop and start the instance; the host may need a power cycle), check the GPU power cables and that Above 4G decoding is on in the BIOS"
}

fn bus_tail(bus_id: &str) -> &str {
    // "00000000:02:00.0" (NVML) and "0000:02:00.0" (sysfs) both end in "02:00.0".
    bus_id.get(bus_id.len().saturating_sub(7)..).unwrap_or(bus_id)
}

pub fn started_bus_ids() -> Vec<String> {
    // PCI bus ids of the GPUs NVML can open; empty when NVML itself does not start.
    let Ok(nvml) = init_nvml() else { return Vec::new() };
    (0..nvml.device_count().unwrap_or(0))
        .filter_map(|i| nvml.device_by_index(i).ok()?.pci_info().ok().map(|p| p.bus_id.to_lowercase()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready() -> Probe {
        Probe {
            pci_gpus: vec!["0000:01:00.0".into()],
            assigned_gpus: vec!["0000:01:00.0".into()],
            wsl: false,
            blackwell_or_newer: false,
            module_version: Some("580.82.07".into()),
            open_module: true,
            device_nodes: true,
            libcuda: true,
            libnvml: true,
            container: false,
            cuda_compat: false,
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
        assert!(diagnose(&old).unwrap().contains("forward compatibility"));
        assert_eq!(diagnose(&Probe { cuda_compat: true, ..old.clone() }), None);
        let ancient = Probe { module_version: Some("525.147".into()), cuda_compat: true, ..ready() };
        assert!(!diagnose(&ancient).unwrap().contains("forward compatibility"));
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

    #[test]
    fn gpus_the_driver_could_not_start_are_named() {
        let two = vec!["0000:02:00.0".to_string(), "0000:03:00.0".to_string()];
        let p = Probe { pci_gpus: two.clone(), assigned_gpus: two, ..ready() };
        assert_eq!(unstarted(&p, &["00000000:02:00.0".into(), "00000000:03:00.0".into()]), None);
        let (blocking, note) = unstarted(&p, &["00000000:02:00.0".into()]).unwrap();
        assert!(!blocking && note.starts_with("the GPU at 0000:03:00.0 did not start"), "{note}");
        let (blocking, why) = unstarted(&p, &[]).unwrap();
        assert!(blocking && why.contains("did not start") && why.contains("power the machine fully off"), "{why}");
        let closed = Probe { open_module: false, blackwell_or_newer: true, ..p.clone() };
        assert!(unstarted(&closed, &[]).unwrap().1.contains("proprietary kernel module, which cannot start Blackwell"));
        let ampere = Probe { open_module: false, ..p.clone() };
        assert!(!unstarted(&ampere, &[]).unwrap().1.contains("proprietary"));
        assert_eq!(unstarted(&Probe { container: true, ..p.clone() }, &[]), None, "a container sees the host's PCI bus");
        let given_one = Probe { assigned_gpus: vec!["0000:02:00.0".into()], ..p };
        assert_eq!(unstarted(&given_one, &["00000000:02:00.0".into()]), None, "GPUs without a /dev node belong to someone else");
    }

    #[test]
    fn wsl_takes_the_gpu_from_windows() {
        let wsl = Probe { wsl: true, pci_gpus: vec![], assigned_gpus: vec![], module_version: None, device_nodes: false, ..ready() };
        assert_eq!(diagnose(&wsl), None);
        assert!(diagnose(&Probe { libcuda: false, ..wsl.clone() }).unwrap().contains("Windows NVIDIA driver"));
        assert_eq!(unstarted(&wsl, &[]), None);
        assert_eq!(device_minor("Model: \t NVIDIA\nDevice Minor: \t 2\nBus Type: PCIe"), Some(2));
    }
}
