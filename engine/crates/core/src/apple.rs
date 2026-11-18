// Apple Silicon: the one GPU of a Mac and the share of unified memory macOS lets it use, read with built-in tools only.

pub const GPU_UUID: &str = "apple-gpu";
pub const COMPUTE: &str = "metal";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Accel {
    pub model: String,
    pub cores: u32,
    pub busy_pct: Option<u32>,
    pub in_use_mb: u64,
}

pub fn parse_ioreg(text: &str) -> Accel {
    // `ioreg -r -d 1 -w 0 -c IOAccelerator`: the GPU's model, core count and PerformanceStatistics (busy % and memory in use).
    let num = |key: &str| -> Option<u64> {
        let at = text.find(&format!("\"{key}\""))? + key.len() + 2;
        let rest = text[at..].trim_start().trim_start_matches('=').trim_start();
        rest.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()
    };
    let model = text
        .find("\"model\"")
        .and_then(|at| text[at..].split('"').nth(3))
        .map(str::to_string)
        .unwrap_or_default();
    Accel {
        model,
        cores: num("gpu-core-count").unwrap_or(0) as u32,
        busy_pct: num("Device Utilization %").map(|v| v.min(100) as u32),
        in_use_mb: num("In use system memory").unwrap_or(0) / (1024 * 1024),
    }
}

pub fn gpu_limit_mb(total_mb: u64, wired_limit_mb: u64) -> u64 {
    // The unified memory the GPU may use: iogpu.wired_limit_mb when set, else macOS's default (2/3 up to 36 GB, 3/4 above).
    if wired_limit_mb > 0 {
        return wired_limit_mb.min(total_mb);
    }
    if total_mb > 36 * 1024 { total_mb * 3 / 4 } else { total_mb * 2 / 3 }
}

pub fn free_mb(limit_mb: u64, in_use_mb: u64, available_ram_mb: u64) -> u64 {
    // What a new model can still take: under the GPU limit, and never more than the memory the system has free.
    limit_mb.saturating_sub(in_use_mb).min(available_ram_mb)
}

pub fn form_factor_of(battery_ioreg: &str) -> &'static str {
    // `ioreg -r -c AppleSmartBattery` lists a battery on a MacBook and nothing on a Mac mini, iMac, Studio or Pro.
    if battery_ioreg.contains("AppleSmartBattery") { "laptop" } else { "desktop" }
}

pub fn environment_label(form_factor: Option<&str>) -> String {
    match form_factor {
        Some(f @ ("laptop" | "desktop")) => format!("macOS {f}"),
        _ => "macOS".to_string(),
    }
}

pub fn parse_version(text: &str) -> Option<(u32, u32)> {
    let mut it = text.trim().split('.');
    Some((it.next()?.parse().ok()?, it.next().and_then(|m| m.parse().ok()).unwrap_or(0)))
}

#[cfg(target_os = "macos")]
mod imp {
    use super::*;
    use crate::host::{CpuInfo, DiskInfo, EnvironmentInfo, HostInfo, MemoryInfo, OsInfo};
    use crate::hw::{GpuState, HardwareState};
    use crate::sys;
    use std::path::Path;

    pub fn accel() -> Accel {
        std::process::Command::new("/usr/sbin/ioreg")
            .args(["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
            .output()
            .map(|o| parse_ioreg(&String::from_utf8_lossy(&o.stdout)))
            .unwrap_or_default()
    }

    fn total_mb() -> u64 {
        sys::sysctl_u64("hw.memsize").unwrap_or(0) / (1024 * 1024)
    }

    pub fn form_factor() -> Option<&'static str> {
        let out = std::process::Command::new("/usr/sbin/ioreg").args(["-r", "-d", "1", "-c", "AppleSmartBattery"]).output().ok()?;
        out.status.success().then(|| form_factor_of(&String::from_utf8_lossy(&out.stdout)))
    }

    pub fn limit_mb() -> u64 {
        gpu_limit_mb(total_mb(), sys::sysctl_u64("iogpu.wired_limit_mb").unwrap_or(0))
    }

    pub fn chip() -> String {
        sys::sysctl_text("machdep.cpu.brand_string").unwrap_or_else(|| "Apple Silicon".to_string())
    }

    pub fn macos_version() -> String {
        sys::sysctl_text("kern.osproductversion").unwrap_or_default()
    }

    pub fn apple_silicon() -> bool {
        sys::sysctl_u64("hw.optional.arm64").unwrap_or(0) == 1
    }

    pub fn gpu(managed: &[(i32, u64)]) -> GpuState {
        // One GPU: its limit as the total, what it uses now, and what a new model could still take.
        let a = accel();
        let limit = limit_mb();
        let avail = sys::memory().available_kb / 1024;
        let name = if a.cores > 0 { format!("{} ({}-core GPU)", chip(), a.cores) } else { chip() };
        GpuState {
            index: 0,
            uuid: GPU_UUID.to_string(),
            name,
            compute_capability: COMPUTE.to_string(),
            vram_total_mb: limit,
            vram_used_mb: a.in_use_mb.min(limit),
            vram_free_mb: free_mb(limit, a.in_use_mb, avail),
            pcie_gen: 0,
            pcie_width: 0,
            pcie_gen_max: 0,
            pcie_width_max: 0,
            running_processes: managed.iter().map(|&(pid, mb)| crate::hw::RunningProcess { pid: pid as u32, used_vram_mb: mb }).collect(),
            unified_memory: false,
            numa_node: None,
        }
    }

    fn disk(role: &str, path: &Path) -> Option<DiskInfo> {
        let c = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
        let mut st: libc::statfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statfs(c.as_ptr(), &mut st) } != 0 {
            return None;
        }
        let cstr = |b: &[libc::c_char]| unsafe { std::ffi::CStr::from_ptr(b.as_ptr()) }.to_string_lossy().into_owned();
        let gb = |blocks: u64| blocks as f64 * st.f_bsize as f64 / 1e9;
        let fs_type = cstr(&st.f_fstypename);
        Some(DiskInfo {
            role: role.to_string(),
            path: path.display().to_string(),
            mount_point: cstr(&st.f_mntonname),
            network_fs: matches!(fs_type.as_str(), "smbfs" | "nfs" | "afpfs" | "webdav"),
            fs_type,
            total_gb: gb(st.f_blocks),
            free_gb: gb(st.f_bavail),
        })
    }

    pub fn host_info(home: &Path) -> HostInfo {
        let m = sys::memory();
        let cores = |k: &str| sys::sysctl_u64(k).unwrap_or(0) as u32;
        HostInfo {
            cpu: CpuInfo {
                vendor: "Apple".to_string(),
                model: chip(),
                sockets: 1,
                physical_cores: cores("hw.physicalcpu"),
                logical_threads: cores("hw.logicalcpu"),
                numa_nodes: Vec::new(),
                flags: Vec::new(),
            },
            memory: MemoryInfo { total_mb: m.total_kb / 1024, available_mb: m.available_kb / 1024, ..MemoryInfo::default() },
            disks: [("goethe_home", home.to_path_buf()), ("models", home.join("models"))].iter().filter_map(|(r, p)| disk(r, p)).collect(),
            os: OsInfo {
                kernel: sys::sysctl_text("kern.osrelease").unwrap_or_default(),
                distro: format!("macOS {}", macos_version()).trim().to_string(),
                glibc: String::new(),
            },
            nvidia_driver: None,
            cuda_driver: None,
        }
    }

    pub fn environment() -> EnvironmentInfo {
        EnvironmentInfo { container: None, systemd_pid1: false, cloud: "none".to_string(), ..EnvironmentInfo::default() }
    }

    pub fn profile(home: &Path, managed: &[(i32, u64)]) -> HardwareState {
        HardwareState {
            profiled_at: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
            driver: format!("macOS {}", macos_version()),
            gpus: vec![gpu(managed)],
            links: Vec::new(),
            unavailable: Vec::new(),
            host: Some(host_info(home)),
            environment: Some(environment()),
        }
    }
}

#[cfg(target_os = "macos")]
pub use imp::*;

#[cfg(test)]
mod tests {
    use super::*;

    const IOREG: &str = r#"+-o AGXAcceleratorG15X  <class AGXAcceleratorG15X, id 0x1000009c4, registered, matched, active, busy 0 (0 ms), retain 63>
    {
      "model" = "Apple M3 Max"
      "gpu-core-count" = 40
      "PerformanceStatistics" = {"In use system memory"=4294967296,"Device Utilization %"=37,"Renderer Utilization %"=12,"Alloc system memory"=9663676416}
    }"#;

    #[test]
    fn the_gpu_entry_gives_model_cores_busy_and_memory() {
        assert_eq!(parse_ioreg(IOREG), Accel { model: "Apple M3 Max".into(), cores: 40, busy_pct: Some(37), in_use_mb: 4096 });
        assert_eq!(parse_ioreg(""), Accel::default(), "no GPU entry, no numbers");
    }

    #[test]
    fn the_gpu_may_use_two_thirds_or_three_quarters_unless_set() {
        assert_eq!(gpu_limit_mb(16 * 1024, 0), 16 * 1024 * 2 / 3);
        assert_eq!(gpu_limit_mb(64 * 1024, 0), 48 * 1024);
        assert_eq!(gpu_limit_mb(64 * 1024, 56 * 1024), 56 * 1024, "the wired limit the user set wins");
        assert_eq!(gpu_limit_mb(16 * 1024, 99 * 1024), 16 * 1024, "never above the memory there is");
    }

    #[test]
    fn free_memory_is_bounded_by_the_limit_and_by_free_ram() {
        assert_eq!(free_mb(48_000, 8_000, 60_000), 40_000);
        assert_eq!(free_mb(48_000, 8_000, 12_000), 12_000, "other apps hold the rest of memory");
        assert_eq!(free_mb(10_000, 12_000, 30_000), 0);
    }

    #[test]
    fn a_battery_makes_a_laptop() {
        assert_eq!(form_factor_of("+-o AppleSmartBattery  <class AppleSmartBattery, id 0x100000585>"), "laptop");
        assert_eq!(form_factor_of(""), "desktop");
        assert_eq!(environment_label(Some("laptop")), "macOS laptop");
        assert_eq!(environment_label(None), "macOS");
    }

    #[test]
    fn macos_versions_parse() {
        assert_eq!(parse_version("15.1"), Some((15, 1)));
        assert_eq!(parse_version("26"), Some((26, 0)));
        assert_eq!(parse_version(""), None);
    }
}
