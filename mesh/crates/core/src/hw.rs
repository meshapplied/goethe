// NVML hardware probe and hardware_state.json writer.

use crate::paths::{ensure_layout, mesh_home};
use nvml_wrapper::enum_wrappers::device::TopologyLevel;
use nvml_wrapper::enums::device::UsedGpuMemory;
use nvml_wrapper::Nvml;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareState {
    pub profiled_at: String,
    pub driver: String,
    pub gpus: Vec<GpuState>,
    pub links: Vec<LinkState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuState {
    pub index: u32,
    pub uuid: String,
    pub name: String,
    pub compute_capability: String,
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    pub vram_free_mb: u64,
    pub pcie_gen: u32,
    pub pcie_width: u32,
    pub running_processes: Vec<RunningProcess>,
    #[serde(default)]
    pub unified_memory: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numa_node: Option<u32>,
}

pub const UNIFIED_OS_RESERVE_GB: u64 = 8;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningProcess {
    pub pid: u32,
    pub used_vram_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkState {
    pub a: u32,
    pub b: u32,
    #[serde(rename = "type")]
    pub link_type: String,
}

pub fn profile(home: &Path) -> Result<HardwareState, String> {
    ensure_layout(home).map_err(|e| format!("layout: {e}"))?;
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let driver = nvml
        .sys_driver_version()
        .map_err(|e| format!("driver version: {e}"))?;
    let count = nvml
        .device_count()
        .map_err(|e| format!("device count: {e}"))?;
    let mut gpus = Vec::new();
    let mut devices = Vec::new();
    for idx in 0..count {
        let dev = nvml
            .device_by_index(idx)
            .map_err(|e| format!("device {idx}: {e}"))?;
        devices.push(dev);
    }
    for (idx, dev) in devices.iter().enumerate() {
        gpus.push(probe_gpu(idx as u32, dev)?);
    }
    let mut links = Vec::new();
    for i in 0..count {
        for j in (i + 1)..count {
            links.push(classify_link(&nvml, i, j)?);
        }
    }
    let profiled_at = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    Ok(HardwareState {
        profiled_at,
        driver,
        gpus,
        links,
    })
}

pub fn write_state(home: &Path, state: &HardwareState) -> Result<(), String> {
    let path = home.join("hardware_state.json");
    let data = serde_json::to_vec_pretty(state).map_err(|e| format!("json encode: {e}"))?;
    write_atomic(&path, &data).map_err(|e| format!("write hardware_state.json: {e}"))
}

pub fn run_profile(json_out: bool) -> Result<(), String> {
    let home = mesh_home();
    let state = profile(&home)?;
    write_state(&home, &state)?;
    if json_out {
        println!("{}", serde_json::to_string_pretty(&state).map_err(|e| e.to_string())?);
    } else {
        print_summary(&state);
    }
    Ok(())
}

fn fake_unified_enabled() -> bool {
    std::env::var("MESH_TEST_FAKE_UNIFIED")
        .ok()
        .is_some_and(|v| v == "1")
}

fn host_unified_mem_mb() -> Result<(u64, u64, u64), String> {
    let text = std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
    let mut total_kb = 0u64;
    let mut avail_kb = 0u64;
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("MemTotal:") {
            total_kb = v.split_whitespace().next().unwrap_or("0").parse().unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("MemAvailable:") {
            avail_kb = v.split_whitespace().next().unwrap_or("0").parse().unwrap_or(0);
        }
    }
    let total_mb = total_kb / 1024;
    let free_mb = avail_kb / 1024;
    let used_mb = total_mb.saturating_sub(free_mb);
    Ok((total_mb, used_mb, free_mb))
}

pub fn memory_from_nvml_or_unified(
    mem_result: Result<nvml_wrapper::struct_wrappers::device::MemoryInfo, nvml_wrapper::error::NvmlError>,
    force_unified: bool,
) -> Result<(u64, u64, u64, bool), String> {
    if force_unified {
        let (t, u, f) = host_unified_mem_mb()?;
        return Ok((t, u, f, true));
    }
    match mem_result {
        Ok(mem) => Ok((
            mem.total / (1024 * 1024),
            mem.used / (1024 * 1024),
            mem.free / (1024 * 1024),
            false,
        )),
        Err(nvml_wrapper::error::NvmlError::NotSupported) => {
            let (t, u, f) = host_unified_mem_mb()?;
            Ok((t, u, f, true))
        }
        Err(e) => Err(format!("memory: {e}")),
    }
}

fn probe_gpu(index: u32, dev: &nvml_wrapper::Device<'_>) -> Result<GpuState, String> {
    let uuid = dev.uuid().map_err(|e| format!("uuid: {e}"))?;
    let name = dev.name().map_err(|e| format!("name: {e}"))?;
    let cc = dev
        .cuda_compute_capability()
        .map_err(|e| format!("compute cap: {e}"))?;
    let compute_capability = format!("{}.{}", cc.major, cc.minor);
    let (vram_total_mb, vram_used_mb, vram_free_mb, unified_memory) =
        memory_from_nvml_or_unified(dev.memory_info(), fake_unified_enabled())?;
    let running_processes = if unified_memory {
        list_compute_pids_no_vram(dev)?
    } else {
        gpu_running_processes(dev, vram_used_mb)?
    };
    let pcie_gen = dev
        .current_pcie_link_gen()
        .map_err(|e| format!("pcie gen: {e}"))?;
    let pcie_width = dev
        .current_pcie_link_width()
        .map_err(|e| format!("pcie width: {e}"))?;
    let numa_node = read_numa_node(dev);
    Ok(GpuState {
        index,
        uuid,
        name,
        compute_capability,
        vram_total_mb,
        vram_used_mb,
        vram_free_mb,
        pcie_gen,
        pcie_width,
        running_processes,
        unified_memory,
        numa_node,
    })
}

fn read_numa_node(dev: &nvml_wrapper::Device<'_>) -> Option<u32> {
    let pci = dev.pci_info().ok()?;
    let path = format!(
        "/sys/bus/pci/devices/{}/numa_node",
        pci.bus_id.to_lowercase()
    );
    let text = std::fs::read_to_string(&path).ok()?;
    let node: i32 = text.trim().parse().ok()?;
    if node < 0 {
        None
    } else {
        Some(node as u32)
    }
}

fn list_compute_pids_no_vram(dev: &nvml_wrapper::Device<'_>) -> Result<Vec<RunningProcess>, String> {
    let procs = dev
        .running_compute_processes()
        .map_err(|e| format!("processes: {e}"))?;
    Ok(procs
        .into_iter()
        .map(|p| RunningProcess {
            pid: p.pid,
            used_vram_mb: 0,
        })
        .collect())
}

fn gpu_running_processes(
    dev: &nvml_wrapper::Device<'_>,
    vram_used_mb: u64,
) -> Result<Vec<RunningProcess>, String> {
    use std::collections::HashMap;
    use std::thread;
    use std::time::Duration;
    let mut merged: HashMap<u32, u64> = HashMap::new();
    for attempt in 0..60 {
        read_compute_procs(dev, &mut merged)?;
        let sum: u64 = merged.values().sum();
        if !needs_proc_reread(vram_used_mb, sum) {
            break;
        }
        if attempt >= 59 {
            break;
        }
        thread::sleep(Duration::from_millis(100));
    }
    Ok(merged
        .into_iter()
        .map(|(pid, used_vram_mb)| RunningProcess { pid, used_vram_mb })
        .collect())
}

pub fn needs_proc_reread(used_mb: u64, sum_mb: u64) -> bool {
    if used_mb <= sum_mb.saturating_add(256) {
        return false;
    }
    if sum_mb == 0 && used_mb < 4096 {
        return false;
    }
    true
}

fn read_compute_procs(
    dev: &nvml_wrapper::Device<'_>,
    merged: &mut std::collections::HashMap<u32, u64>,
) -> Result<(), String> {
    let procs = dev
        .running_compute_processes()
        .map_err(|e| format!("processes: {e}"))?;
    for p in procs {
        let mb = match p.used_gpu_memory {
            UsedGpuMemory::Used(bytes) => bytes / (1024 * 1024),
            UsedGpuMemory::Unavailable => 0,
        };
        merged
            .entry(p.pid)
            .and_modify(|v| *v = (*v).max(mb))
            .or_insert(mb);
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct GpuPlacementSnapshot {
    pub index: u32,
    pub foreign_compute: bool,
    pub free_vram_mb: u64,
}

pub fn list_gpus_for_placement(managed: &HashSet<i32>) -> Result<Vec<GpuPlacementSnapshot>, String> {
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let count = nvml
        .device_count()
        .map_err(|e| format!("device count: {e}"))?;
    let mut out = Vec::new();
    for idx in 0..count {
        let dev = nvml
            .device_by_index(idx)
            .map_err(|e| format!("device {idx}: {e}"))?;
        let mem = dev.memory_info().map_err(|e| format!("memory: {e}"))?;
        let free_vram_mb = mem.free / (1024 * 1024);
        let mut foreign_compute = false;
        if let Ok(procs) = dev.running_compute_processes() {
            for p in procs {
                if !managed.contains(&(p.pid as i32)) {
                    foreign_compute = true;
                    break;
                }
            }
        }
        out.push(GpuPlacementSnapshot {
            index: idx,
            foreign_compute,
            free_vram_mb,
        });
    }
    Ok(out)
}

pub fn gpu_used_mb_live(gpu_index: u32) -> Result<u64, String> {
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let dev = nvml
        .device_by_index(gpu_index)
        .map_err(|e| format!("device {gpu_index}: {e}"))?;
    let mem = dev.memory_info().map_err(|e| format!("memory: {e}"))?;
    Ok(mem.used / (1024 * 1024))
}

pub fn managed_vram_listed_mb(gpu_index: u32, managed: &HashSet<i32>) -> Result<u64, String> {
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let dev = nvml
        .device_by_index(gpu_index)
        .map_err(|e| format!("device {gpu_index}: {e}"))?;
    let mut sum = 0u64;
    for procs in [
        dev.running_compute_processes(),
        dev.running_graphics_processes(),
    ] {
        let Ok(procs) = procs else {
            continue;
        };
        for p in procs {
            if !managed.contains(&(p.pid as i32)) {
                continue;
            }
            let mb = match p.used_gpu_memory {
                UsedGpuMemory::Used(bytes) => bytes / (1024 * 1024),
                UsedGpuMemory::Unavailable => 0,
            };
            sum = sum.saturating_add(mb);
        }
    }
    Ok(sum)
}

pub fn gpu_free_mb_live(gpu_index: u32) -> Result<u64, String> {
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let dev = nvml
        .device_by_index(gpu_index)
        .map_err(|e| format!("device {gpu_index}: {e}"))?;
    let (_, _, free, unified) = memory_from_nvml_or_unified(dev.memory_info(), fake_unified_enabled())?;
    let free = if unified {
        free.saturating_sub(UNIFIED_OS_RESERVE_GB * 1024)
    } else {
        free
    };
    Ok(free)
}

pub fn foreign_compute_on_gpu(
    gpu_index: u32,
    managed: &HashSet<i32>,
) -> Result<Option<(i32, f64)>, String> {
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let dev = nvml
        .device_by_index(gpu_index)
        .map_err(|e| format!("device {gpu_index}: {e}"))?;
    for procs in [
        dev.running_compute_processes(),
        dev.running_graphics_processes(),
    ] {
        let Ok(procs) = procs else {
            continue;
        };
        for p in procs {
            let pid = p.pid as i32;
            if managed.contains(&pid) {
                continue;
            }
            let gb = match p.used_gpu_memory {
                UsedGpuMemory::Used(bytes) => bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                UsedGpuMemory::Unavailable => 0.0,
            };
            return Ok(Some((pid, gb)));
        }
    }
    Ok(None)
}

pub fn refresh_gpu_vram(gpu_index: u32) -> Result<(u64, u64, u64, Vec<RunningProcess>), String> {
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let dev = nvml
        .device_by_index(gpu_index)
        .map_err(|e| format!("device {gpu_index}: {e}"))?;
    let mem = dev.memory_info().map_err(|e| format!("memory: {e}"))?;
    let total = mem.total / (1024 * 1024);
    let used = mem.used / (1024 * 1024);
    let free = mem.free / (1024 * 1024);
    let procs = gpu_running_processes(&dev, used)?;
    Ok((total, used, free, procs))
}

fn classify_link(nvml: &Nvml, a: u32, b: u32) -> Result<LinkState, String> {
    let dev_a = nvml
        .device_by_index(a)
        .map_err(|e| format!("device {a}: {e}"))?;
    let dev_b = nvml
        .device_by_index(b)
        .map_err(|e| format!("device {b}: {e}"))?;
    if nvlink_active(&dev_a, &dev_b) {
        return Ok(LinkState {
            a,
            b,
            link_type: "NVLINK".to_string(),
        });
    }
    let level = dev_a
        .topology_common_ancestor(dev_b)
        .map_err(|e| format!("topology {a}-{b}: {e}"))?;
    let link_type = topology_level_label(level).to_string();
    Ok(LinkState { a, b, link_type })
}

fn topology_level_label(level: TopologyLevel) -> &'static str {
    match level {
        TopologyLevel::Internal | TopologyLevel::Single => "PIX",
        TopologyLevel::Multiple => "PXB",
        TopologyLevel::HostBridge => "PHB",
        TopologyLevel::Node => "NODE",
        TopologyLevel::System => "SYS",
    }
}

pub fn load_hardware_state(home: &Path) -> Result<HardwareState, String> {
    let path = home.join("hardware_state.json");
    if !path.is_file() {
        return profile(home);
    }
    let data = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
    serde_json::from_str(&data).map_err(|e| format!("hardware_state.json: {e}"))
}

pub fn profile_gpu_indices(home: &Path) -> Result<Vec<u32>, String> {
    let state = load_hardware_state(home)?;
    Ok(state.gpus.iter().map(|g| g.index).collect())
}

pub fn link_rank(link_type: &str) -> u32 {
    match link_type {
        "NVLINK" | "PIX" => 0,
        "PXB" => 1,
        "PHB" => 2,
        "NODE" => 3,
        "SYS" => 4,
        "PCIE_SWITCH" => 1,
        "SAME_NUMA" => 3,
        "CROSS_SOCKET" => 4,
        _ => 5,
    }
}

pub fn worst_link_rank(hw: &HardwareState, indices: &[u32]) -> u32 {
    let mut worst = 0u32;
    for i in 0..indices.len() {
        for j in (i + 1)..indices.len() {
            let a = indices[i];
            let b = indices[j];
            let ty = hw
                .links
                .iter()
                .find(|l| (l.a == a && l.b == b) || (l.a == b && l.b == a))
                .map(|l| l.link_type.as_str())
                .unwrap_or("SYS");
            worst = worst.max(link_rank(ty));
        }
    }
    worst
}

fn nvlink_active(dev_a: &nvml_wrapper::Device<'_>, dev_b: &nvml_wrapper::Device<'_>) -> bool {
    let Ok(peer) = dev_b.pci_info() else {
        return false;
    };
    for link in 0..18u32 {
        let wrapper = dev_a.link_wrapper_for(link);
        if wrapper.is_active().ok() != Some(true) {
            continue;
        }
        let Ok(remote) = wrapper.remote_pci_info() else {
            continue;
        };
        if remote.bus == peer.bus && remote.device == peer.device && remote.domain == peer.domain {
            return true;
        }
    }
    false
}

fn write_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(tmp, path)
}

fn print_summary(state: &HardwareState) {
    println!("Mesh hardware profile at {}", state.profiled_at);
    println!("Driver: {}", state.driver);
    for gpu in &state.gpus {
        let um = if gpu.unified_memory { " unified memory" } else { "" };
        let numa = gpu
            .numa_node
            .map(|n| format!(" NUMA {n}"))
            .unwrap_or_default();
        println!(
            "GPU {}: {} ({}){}{} {} GB total, {} GB free",
            gpu.index,
            gpu.name,
            gpu.compute_capability,
            um,
            numa,
            gpu.vram_total_mb / 1024,
            gpu.vram_free_mb / 1024
        );
        for proc in &gpu.running_processes {
            println!("  PID {} using {} MB", proc.pid, proc.used_vram_mb);
        }
    }
    for link in &state.links {
        println!("Link GPU {} - GPU {}: {}", link.a, link.b, link.link_type);
    }
}

#[cfg(test)]
mod unified_tests {
    use super::*;
    use nvml_wrapper::error::NvmlError;

    #[test]
    fn not_supported_uses_host_mem() {
        let (t, u, f, uni) =
            memory_from_nvml_or_unified(Err(NvmlError::NotSupported), false).expect("host mem");
        assert!(uni);
        assert!(t > 0);
        assert!(f <= t);
        assert!(u <= t);
    }
}
