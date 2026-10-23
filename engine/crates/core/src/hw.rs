// NVML hardware probe, host profile, and hardware_state.json writer.

use crate::host::{self, EnvironmentInfo, HostInfo};
use crate::paths::{ensure_layout, goethe_home};
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host: Option<HostInfo>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub environment: Option<EnvironmentInfo>,
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
    #[serde(default)]
    pub pcie_gen_max: u32,
    #[serde(default)]
    pub pcie_width_max: u32,
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
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub topo: String,
}

pub fn profile(home: &Path) -> Result<HardwareState, String> {
    // Probe every GPU and pair over NVML, then add the host and runtime environment.
    ensure_layout(home).map_err(|e| format!("layout: {e}"))?;
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let driver = nvml.sys_driver_version().map_err(|e| format!("driver version: {e}"))?;
    let count = nvml.device_count().map_err(|e| format!("device count: {e}"))?;
    let mut gpus = Vec::new();
    for idx in 0..count {
        let dev = nvml.device_by_index(idx).map_err(|e| format!("device {idx}: {e}"))?;
        gpus.push(probe_gpu(idx, &dev)?);
    }
    let mut links = Vec::new();
    for i in 0..count {
        for j in (i + 1)..count {
            links.push(classify_link(&nvml, i, j)?);
        }
    }
    let environment = host::probe_environment(crate::nginx::public_port_u16(home));
    let mut host_info = host::probe_host(home, &environment.cloud);
    host_info.nvidia_driver = Some(driver.clone());
    host_info.cuda_driver = nvml.sys_cuda_driver_version().ok().map(|v| {
        format!(
            "{}.{}",
            nvml_wrapper::cuda_driver_version_major(v),
            nvml_wrapper::cuda_driver_version_minor(v)
        )
    });
    Ok(HardwareState {
        profiled_at: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
        driver,
        gpus,
        links,
        host: Some(host_info),
        environment: Some(environment),
    })
}

pub fn write_state(home: &Path, state: &HardwareState) -> Result<(), String> {
    let path = home.join("hardware_state.json");
    let data = serde_json::to_vec_pretty(state).map_err(|e| format!("json encode: {e}"))?;
    write_atomic(&path, &data).map_err(|e| format!("write hardware_state.json: {e}"))
}

pub fn run_pick_gpus(tp: Option<u64>, model: Option<&str>) -> Result<(), String> {
    // Where goethe load would place a model now (its full budget), or an empty engine of a given tp.
    let home = goethe_home();
    let view = crate::recommend::View::new(&home, profile(&home)?);
    let pinned = tp.map(|t| t.max(1) as usize);
    let (what, fit) = match model {
        Some(id) => {
            let (input, card_tp, marlin) = crate::models::fit_input(&home, id)?;
            let fit = view.fit(id, &input, pinned.or(card_tp), marlin);
            (format!("{id} "), fit.map(|f| (f.describe(&input), f.gpus)))
        }
        None => {
            let t = pinned.unwrap_or(1);
            let empty = crate::chooser::ChooserInput {
                static_vram_gb: 0.0,
                kv_bytes_per_token: 1,
                num_attention_heads: t as u64,
                num_kv_heads: t as u64,
                attention: "gqa".to_string(),
                max_ctx: 1,
                ctx: 1,
                max_conc: 1,
            };
            let fit = view.fit("", &empty, Some(t), None);
            (String::new(), fit.map(|f| (format!("tp={} on GPU {}", f.tp, crate::placement::format_gpu_list(&f.gpus)), f.gpus)))
        }
    };
    let (desc, gpus) = fit.map_err(|why| format!("would not place {what}now: {why}"))?;
    let empty = if model.is_none() { " (an empty engine; --model <id> places a real budget)" } else { "" };
    println!("would place {what}{desc}{empty}");
    for g in view.gpus.iter().filter(|g| gpus.contains(&g.index)) {
        let others = if g.foreign_process { "other software is running there" } else { "no other processes" };
        println!("  GPU {}: {:.1} GB free, {others}", g.index, g.free_vram_mb as f64 / 1024.0);
    }
    Ok(())
}

pub fn run_profile(json_out: bool) -> Result<(), String> {
    let home = goethe_home();
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
    std::env::var("GOETHE_TEST_FAKE_UNIFIED").ok().is_some_and(|v| v == "1")
}

fn host_unified_mem_mb() -> Result<(u64, u64, u64), String> {
    let text = std::fs::read_to_string("/proc/meminfo").map_err(|e| e.to_string())?;
    let m = host::parse_meminfo(&text);
    Ok((m.total_mb, m.total_mb.saturating_sub(m.available_mb), m.available_mb))
}

pub fn memory_from_nvml_or_unified(
    mem_result: Result<nvml_wrapper::struct_wrappers::device::MemoryInfo, nvml_wrapper::error::NvmlError>,
    force_unified: bool,
) -> Result<(u64, u64, u64, bool), String> {
    // NVML memory in MB, or host memory when the GPU shares it (GB10 reports NotSupported).
    if force_unified {
        let (t, u, f) = host_unified_mem_mb()?;
        return Ok((t, u, f, true));
    }
    match mem_result {
        Ok(mem) => Ok((mem.total / (1024 * 1024), mem.used / (1024 * 1024), mem.free / (1024 * 1024), false)),
        Err(nvml_wrapper::error::NvmlError::NotSupported) => {
            let (t, u, f) = host_unified_mem_mb()?;
            Ok((t, u, f, true))
        }
        Err(e) => Err(format!("memory: {e}")),
    }
}

fn device_memory(dev: &nvml_wrapper::Device<'_>) -> Result<(u64, u64, u64, bool), String> {
    memory_from_nvml_or_unified(dev.memory_info(), fake_unified_enabled())
}

fn probe_gpu(index: u32, dev: &nvml_wrapper::Device<'_>) -> Result<GpuState, String> {
    let uuid = dev.uuid().map_err(|e| format!("uuid: {e}"))?;
    let name = dev.name().map_err(|e| format!("name: {e}"))?;
    let cc = dev.cuda_compute_capability().map_err(|e| format!("compute cap: {e}"))?;
    let (vram_total_mb, vram_used_mb, vram_free_mb, unified_memory) = device_memory(dev)?;
    Ok(GpuState {
        index,
        uuid,
        name,
        compute_capability: format!("{}.{}", cc.major, cc.minor),
        vram_total_mb,
        vram_used_mb,
        vram_free_mb,
        pcie_gen: dev.current_pcie_link_gen().unwrap_or(0),
        pcie_width: dev.current_pcie_link_width().unwrap_or(0),
        pcie_gen_max: dev.max_pcie_link_gen().unwrap_or(0),
        pcie_width_max: dev.max_pcie_link_width().unwrap_or(0),
        running_processes: compute_processes(dev, unified_memory)?,
        unified_memory,
        numa_node: read_numa_node(dev),
    })
}

fn read_numa_node(dev: &nvml_wrapper::Device<'_>) -> Option<u32> {
    let pci = dev.pci_info().ok()?;
    let path = format!("/sys/bus/pci/devices/{}/numa_node", pci.bus_id.to_lowercase());
    let node: i32 = std::fs::read_to_string(&path).ok()?.trim().parse().ok()?;
    u32::try_from(node).ok()
}

fn compute_processes(dev: &nvml_wrapper::Device, unified: bool) -> Result<Vec<RunningProcess>, String> {
    // One NVML read; unified-memory GPUs report no per-process VRAM.
    let procs = dev.running_compute_processes().map_err(|e| format!("processes: {e}"))?;
    Ok(procs
        .into_iter()
        .map(|p| RunningProcess {
            pid: p.pid,
            used_vram_mb: match p.used_gpu_memory {
                UsedGpuMemory::Used(bytes) if !unified => bytes / (1024 * 1024),
                _ => 0,
            },
        })
        .collect())
}

#[derive(Debug, Clone)]
pub struct GpuPlacementSnapshot {
    pub index: u32,
    pub foreign_compute: bool,
    pub free_vram_mb: u64,
}

pub fn list_gpus_for_placement(managed: &HashSet<i32>) -> Result<Vec<GpuPlacementSnapshot>, String> {
    // Free memory (after the unified-memory reserve) and foreign-process presence per GPU.
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let count = nvml.device_count().map_err(|e| format!("device count: {e}"))?;
    let mut out = Vec::new();
    for idx in 0..count {
        let dev = nvml.device_by_index(idx).map_err(|e| format!("device {idx}: {e}"))?;
        let (_, _, free, unified) = device_memory(&dev)?;
        let free_vram_mb = if unified { free.saturating_sub(UNIFIED_OS_RESERVE_GB * 1024) } else { free };
        let foreign_compute = dev
            .running_compute_processes()
            .map(|procs| procs.iter().any(|p| !managed.contains(&(p.pid as i32))))
            .unwrap_or(false);
        out.push(GpuPlacementSnapshot {
            index: idx,
            foreign_compute,
            free_vram_mb,
        });
    }
    Ok(out)
}

pub struct LiveGpu {
    pub index: u32,
    pub uuid: String,
    pub free_mb: u64,
    pub pids: Vec<i32>,
}

pub fn live_gpus() -> Result<Vec<LiveGpu>, String> {
    // Live free VRAM (after the unified-memory reserve), uuid and process ids per GPU.
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let count = nvml.device_count().map_err(|e| format!("device count: {e}"))?;
    let mut out = Vec::new();
    for index in 0..count {
        let dev = nvml.device_by_index(index).map_err(|e| format!("device {index}: {e}"))?;
        let (_, _, free, unified) = device_memory(&dev)?;
        let mut pids: Vec<i32> = [dev.running_compute_processes(), dev.running_graphics_processes()]
            .into_iter()
            .flatten()
            .flatten()
            .map(|p| p.pid as i32)
            .collect();
        pids.sort_unstable();
        pids.dedup();
        out.push(LiveGpu {
            index,
            uuid: dev.uuid().unwrap_or_default(),
            free_mb: if unified { free.saturating_sub(UNIFIED_OS_RESERVE_GB * 1024) } else { free },
            pids,
        });
    }
    Ok(out)
}

pub fn gpu_used_mb_live(gpu_index: u32) -> Result<u64, String> {
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let dev = nvml.device_by_index(gpu_index).map_err(|e| format!("device {gpu_index}: {e}"))?;
    Ok(device_memory(&dev)?.1)
}

pub fn managed_vram_listed_mb(gpu_index: u32, managed: &HashSet<i32>) -> Result<u64, String> {
    // VRAM NVML attributes to goethe-managed PIDs on one GPU.
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let dev = nvml.device_by_index(gpu_index).map_err(|e| format!("device {gpu_index}: {e}"))?;
    let mut sum = 0u64;
    for procs in [dev.running_compute_processes(), dev.running_graphics_processes()] {
        let Ok(procs) = procs else {
            continue;
        };
        for p in procs {
            if !managed.contains(&(p.pid as i32)) {
                continue;
            }
            if let UsedGpuMemory::Used(bytes) = p.used_gpu_memory {
                sum = sum.saturating_add(bytes / (1024 * 1024));
            }
        }
    }
    Ok(sum)
}

pub fn gpu_free_mb_live(gpu_index: u32) -> Result<u64, String> {
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let dev = nvml.device_by_index(gpu_index).map_err(|e| format!("device {gpu_index}: {e}"))?;
    let (_, _, free, unified) = device_memory(&dev)?;
    Ok(if unified { free.saturating_sub(UNIFIED_OS_RESERVE_GB * 1024) } else { free })
}

pub fn refresh_gpu_vram(gpu_index: u32) -> Result<(u64, u64, u64, Vec<RunningProcess>), String> {
    let nvml = Nvml::init().map_err(|e| format!("NVML init failed: {e}"))?;
    let dev = nvml.device_by_index(gpu_index).map_err(|e| format!("device {gpu_index}: {e}"))?;
    let (total, used, free, unified) = device_memory(&dev)?;
    Ok((total, used, free, compute_processes(&dev, unified)?))
}

fn classify_link(nvml: &Nvml, a: u32, b: u32) -> Result<LinkState, String> {
    // CONTRACT C class plus the raw nvidia-smi topo code (NV#, PIX, PXB, PHB, NODE, SYS).
    let dev_a = nvml.device_by_index(a).map_err(|e| format!("device {a}: {e}"))?;
    let dev_b = nvml.device_by_index(b).map_err(|e| format!("device {b}: {e}"))?;
    let nvlinks = nvlink_count(&dev_a, &dev_b);
    let topo = if nvlinks > 0 {
        format!("NV{nvlinks}")
    } else {
        match dev_a.topology_common_ancestor(dev_b) {
            Ok(level) => topology_level_label(level).to_string(),
            Err(e) => return Err(format!("topology {a}-{b}: {e}")),
        }
    };
    Ok(LinkState {
        a,
        b,
        link_type: link_class(&topo).to_string(),
        topo,
    })
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

pub fn link_class(topo: &str) -> &'static str {
    match topo {
        t if t.starts_with("NV") => "NVLINK",
        "PIX" | "PXB" => "PCIE_SWITCH",
        "PHB" | "NODE" => "SAME_NUMA",
        _ => "CROSS_SOCKET",
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
        t if t.starts_with("NV") => 0,
        "PIX" => 0,
        "PXB" | "PCIE_SWITCH" => 1,
        "PHB" => 2,
        "NODE" | "SAME_NUMA" => 3,
        "SYS" | "CROSS_SOCKET" => 4,
        _ => 5,
    }
}

pub fn worst_link_rank(hw: &HardwareState, indices: &[u32]) -> u32 {
    // Rank by the raw topo code when present (finer), else by the link class.
    let mut worst = 0u32;
    for i in 0..indices.len() {
        for j in (i + 1)..indices.len() {
            let (a, b) = (indices[i], indices[j]);
            let ty = hw
                .links
                .iter()
                .find(|l| (l.a == a && l.b == b) || (l.a == b && l.b == a))
                .map(|l| if l.topo.is_empty() { l.link_type.as_str() } else { l.topo.as_str() })
                .unwrap_or("SYS");
            worst = worst.max(link_rank(ty));
        }
    }
    worst
}

fn nvlink_count(dev_a: &nvml_wrapper::Device<'_>, dev_b: &nvml_wrapper::Device<'_>) -> u32 {
    // Active NVLink lanes from a that land on b's PCI address.
    let Ok(peer) = dev_b.pci_info() else {
        return 0;
    };
    (0..18u32)
        .filter(|link| {
            let wrapper = dev_a.link_wrapper_for(*link);
            wrapper.is_active().ok() == Some(true)
                && wrapper
                    .remote_pci_info()
                    .is_ok_and(|r| r.bus == peer.bus && r.device == peer.device && r.domain == peer.domain)
        })
        .count() as u32
}

fn write_atomic(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(tmp, path)
}

fn print_summary(state: &HardwareState) {
    // GPUs, links, then host and environment lines.
    println!("Goethe hardware profile at {}", state.profiled_at);
    println!("Driver: {}", state.driver);
    for gpu in &state.gpus {
        let um = if gpu.unified_memory { " unified memory" } else { "" };
        let numa = gpu.numa_node.map(|n| format!(" NUMA {n}")).unwrap_or_default();
        println!(
            "GPU {}: {} ({}){}{} {} GB total, {} GB free, PCIe gen {}/{} x{}",
            gpu.index,
            gpu.name,
            gpu.compute_capability,
            um,
            numa,
            gpu.vram_total_mb / 1024,
            gpu.vram_free_mb / 1024,
            gpu.pcie_gen,
            gpu.pcie_gen_max,
            gpu.pcie_width
        );
        for proc in &gpu.running_processes {
            println!("  PID {} using {} MB", proc.pid, proc.used_vram_mb);
        }
    }
    for link in &state.links {
        println!("Link GPU {} - GPU {}: {} ({})", link.a, link.b, link.link_type, link.topo);
    }
    if let (Some(h), Some(e)) = (&state.host, &state.environment) {
        let home = crate::paths::goethe_home();
        let bind = if home.join("state.db").is_file() { crate::nginx::public_bind(&home) } else { "127.0.0.1".into() };
        for line in host::summary_lines(h, e, &bind) {
            println!("{line}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nvml_wrapper::error::NvmlError;

    #[test]
    fn not_supported_uses_host_mem() {
        let (t, u, f, uni) = memory_from_nvml_or_unified(Err(NvmlError::NotSupported), false).expect("host mem");
        assert!(uni);
        assert!(t > 0);
        assert!(f <= t);
        assert!(u <= t);
    }

    #[test]
    fn link_classes_and_ranks() {
        assert_eq!(link_class("NV4"), "NVLINK");
        assert_eq!(link_class("PXB"), "PCIE_SWITCH");
        assert_eq!(link_class("NODE"), "SAME_NUMA");
        assert_eq!(link_class("SYS"), "CROSS_SOCKET");
        assert!(link_rank("NV12") < link_rank("PCIE_SWITCH"));
        assert!(link_rank("PHB") < link_rank("SAME_NUMA"));
        let state: HardwareState = serde_json::from_str(
            r#"{"profiled_at":"t","driver":"d","gpus":[],"links":[{"a":0,"b":1,"type":"SAME_NUMA","topo":"PHB"},{"a":0,"b":2,"type":"SAME_NUMA"}]}"#,
        )
        .unwrap();
        assert_eq!(worst_link_rank(&state, &[0, 1]), 2);
        assert_eq!(worst_link_rank(&state, &[0, 2]), 3);
        assert!(state.host.is_none());
    }
}
