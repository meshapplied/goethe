// Wait for a load RPC instance to reach RUNNING (shared by goethe load and install --load).

use crate::rpc;
use serde_json::json;
use std::time::{Duration, Instant};

pub const DEFAULT_LOAD_WAIT: Duration = Duration::from_secs(600);

const BANNER: &str = "Loading into VRAM...";

fn heartbeat(secs: u64) -> String {
    // One line that says the load is alive, whatever the engine is doing: "loading into VRAM, 12 s".
    format!("loading into VRAM, {secs} s")
}

pub fn managed_mb(gpus: &[u32]) -> u64 {
    // VRAM goethe's engines hold on these GPUs now.
    let home = crate::paths::goethe_home();
    let managed = crate::db::open(&home).map(|c| crate::placement::managed_pids(&c)).unwrap_or_default();
    crate::hw::gpu_usage(&managed).unwrap_or_default().iter().filter(|u| gpus.contains(&u.index)).map(|u| u.managed_mb).sum()
}

struct Facts {
    ctx: i64,
    max_conc: i64,
    gb_per_gpu: f64,
    gpus: Vec<u32>,
}

fn facts(instance_id: &str) -> Option<Facts> {
    // The context, concurrency, VRAM budget and GPUs goethed chose for an instance.
    let con = crate::db::open(&crate::paths::goethe_home()).ok()?;
    con.query_row(
        "SELECT ctx_len, max_conc, allocated_vram_gb_per_gpu, gpu_indices FROM loaded_instances WHERE instance_id = ?1",
        [instance_id],
        |r| {
            let csv: String = r.get(3)?;
            let gpus = csv.split(',').filter_map(|g| g.trim().parse().ok()).collect();
            Ok(Facts { ctx: r.get(0)?, max_conc: r.get(1)?, gb_per_gpu: r.get(2)?, gpus })
        },
    )
    .ok()
}

fn head_lines(f: &Facts) -> [String; 3] {
    let gpus: Vec<String> = f.gpus.iter().map(u32::to_string).collect();
    [
        format!("Context window: {} tokens", f.ctx),
        format!("Max concurrency: {}", f.max_conc),
        format!("Max VRAM: {:.1} GB (GPU {})", f.gb_per_gpu * f.gpus.len().max(1) as f64, gpus.join(",")),
    ]
}

struct Bar {
    gpus: Vec<u32>,
    need_mb: u64,
    base_mb: u64,
    frac: f64,
}

impl Bar {
    fn show(&mut self) {
        // The share of its budget the instance holds in VRAM so far, at most 95% until it runs.
        if self.need_mb > 0 {
            let grown = managed_mb(&self.gpus).saturating_sub(self.base_mb) as f64 / self.need_mb as f64;
            self.frac = self.frac.max(grown.min(0.95));
        }
        crate::progress::eprint_line(crate::progress::bar((self.frac * 100.0) as u32));
    }
}

pub fn wait_for_instance(instance_id: &str, timeout: Duration) -> Result<(), String> {
    // Poll ps until the instance is RUNNING, printing its notices; a final status or a vanished row ends the wait. Compact: the instance's context, concurrency and VRAM, then a bar.
    let compact = crate::step::compact();
    if !compact {
        eprintln!("{BANNER}");
    }
    let mut bar: Option<Bar> = None;
    let (started, mut last) = (Instant::now(), Instant::now());
    let deadline = started + timeout;
    while Instant::now() < deadline {
        if compact {
            if bar.is_none() {
                if let Some(f) = facts(instance_id) {
                    head_lines(&f).iter().for_each(|l| eprintln!("{l}"));
                    let need_mb = (f.gb_per_gpu * 1024.0) as u64 * f.gpus.len() as u64;
                    bar = Some(Bar { base_mb: managed_mb(&f.gpus), gpus: f.gpus, need_mb, frac: 0.0 });
                }
            }
            if let Some(b) = bar.as_mut() {
                b.show();
            }
        } else if last.elapsed() >= crate::progress::every() {
            crate::progress::eprint_line(heartbeat(started.elapsed().as_secs()));
            last = Instant::now();
        }
        let ps = rpc::call("ps", json!(null)).map_err(rpc::map_daemon_err)?;
        if ps.code == 0 {
            if let Ok(rows) = serde_json::from_str::<serde_json::Value>(&ps.stdout) {
                if let Some(arr) = rows.as_array() {
                    if !arr.iter().any(|r| r.get("instance_id").and_then(|v| v.as_str()) == Some(instance_id)) {
                        return Err(format!("instance {instance_id} no longer exists (unloaded or stopped)"));
                    }
                    for row in arr {
                        if row.get("instance_id").and_then(|v| v.as_str()) != Some(instance_id) {
                            continue;
                        }
                        let st = row.get("status").and_then(|v| v.as_str()).unwrap_or("");
                        if st == "RUNNING" {
                            if compact {
                                crate::progress::eprint_line(crate::progress::bar(100));
                                crate::progress::end_line();
                            }
                            return Ok(());
                        }
                        if st == "CANCELLED" {
                            return Err("load cancelled".to_string());
                        }
                        if st == "FAILED" {
                            return Err(failure_report(&crate::paths::goethe_home(), instance_id));
                        }
                    }
                }
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err("load timed out waiting for RUNNING".to_string())
}

const ERROR_MARKERS: &[&str] = &[
    "Error", "error:", "Exception", "Traceback", "out of memory", "CUDA", "Killed", "Assertion", "not supported", "No such file",
];

fn tail_lines(path: &std::path::Path, max_bytes: u64) -> Vec<String> {
    // The last max_bytes of a file as lines; empty when it cannot be read.
    use std::io::{Read, Seek, SeekFrom};
    let Ok(mut f) = std::fs::File::open(path) else {
        return Vec::new();
    };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let _ = f.seek(SeekFrom::Start(len.saturating_sub(max_bytes)));
    let mut buf = Vec::new();
    let _ = f.read_to_end(&mut buf);
    String::from_utf8_lossy(&buf).lines().map(str::to_string).collect()
}

pub fn failure_report(home: &std::path::Path, instance_id: &str) -> String {
    // Why a start failed: goethed's lines about this instance, then the error lines (or the tail) of the engine log.
    let mut out = vec![format!("the engine failed to start (instance {instance_id})")];
    let daemon: Vec<String> = tail_lines(&home.join("logs/goethed.log"), 256 * 1024)
        .into_iter()
        .filter(|l| l.contains(instance_id))
        .collect();
    for l in daemon.iter().rev().take(4).rev() {
        out.push(format!("  goethed: {}", l.trim()));
    }
    let engine = tail_lines(&home.join("logs").join(format!("{instance_id}.log")), 512 * 1024);
    if let Some(cause) = likely_cause(&engine) {
        out.push(format!("  likely cause: {cause}"));
    }
    let mut in_traceback = false;
    let kept: Vec<&String> = engine
        .iter()
        .filter(|l| {
            let t = l.trim_start();
            in_traceback = t.starts_with("Traceback") || (in_traceback && l.starts_with(char::is_whitespace));
            !in_traceback
        })
        .collect();
    let errors: Vec<&String> = kept.iter().copied().filter(|l| ERROR_MARKERS.iter().any(|m| l.contains(m))).collect();
    let picked: Vec<&String> = if errors.is_empty() { kept[kept.len().saturating_sub(15)..].to_vec() } else { errors.into_iter().rev().take(12).rev().collect() };
    if !picked.is_empty() {
        out.push("  engine log:".to_string());
        out.extend(picked.into_iter().map(|l| format!("    {}", l.trim_end())));
    }
    out.push(format!("full engine log: goethe logs {instance_id} -n 200"));
    out.join("\n")
}

const COMPILER_MARKERS: &[&str] = &[
    "unsupported GNU version",
    "unsupported clang version",
    "Failed to find C compiler",
    "No such file or directory: 'c++'",
    "No such file or directory: 'g++'",
    "c++: not found",
    "g++: not found",
    "gcc: not found",
    "cc1plus",
    "command 'c++' failed",
    "failed to preprocess host compiler properties",
    "No such file or directory: 'ninja'",
];

fn likely_cause(engine: &[String]) -> Option<String> {
    // A plain reason for the engine failures Goethe can name: a pack that does not run on this GPU, no compiler, no kernels for this GPU, an old driver.
    if let Some(refusal) = crate::install::load_refusal(&engine.join("\n")) {
        return Some(refusal);
    }
    let has = |markers: &[&str]| engine.iter().any(|l| markers.iter().any(|m| l.contains(m)));
    if has(COMPILER_MARKERS) {
        let distro = crate::nvidia_driver::probe().distro;
        return Some(match crate::host_compiler::verdict(crate::host_compiler::find().as_ref(), &distro) {
            crate::host_compiler::Verdict::Ok(label) => format!("a run-time kernel build failed with {label}; see the engine log lines below"),
            crate::host_compiler::Verdict::TooNew(why) | crate::host_compiler::Verdict::Unusable(why) => why,
        });
    }
    if has(&["OutOfMemoryError", "CUDA out of memory", "alloc chain KV: out of memory"]) {
        let loading = has(&["process_weights_after_loading", "load_weights", "Load weight begin"]) && !has(&["Load weight end"]);
        return Some(if loading {
            "the GPU ran out of memory while loading the weights; the model needs more memory than these GPUs have free during loading: split it across more GPUs with --gpu".into()
        } else {
            "the GPU ran out of memory; free memory on it, lower --ctx or --max-conc, or split the model across more GPUs with --gpu".into()
        });
    }
    if has(&["kIOGPUCommandBufferCallbackErrorOutOfMemory", "ggml_metal_buffer_init: error", "failed to allocate Metal buffer"]) || (has(&["ggml_metal"]) && has(&["failed to allocate"])) {
        return Some("the Mac ran out of GPU memory; quit some apps, or lower --ctx or --max-conc".into());
    }
    if has(&["no kernel image is available"]) {
        return Some("the engine has no kernels for this GPU's architecture; goethe doctor lists the supported ones".into());
    }
    if has(&["CUDA driver version is insufficient", "cudaErrorInsufficientDriver"]) {
        let why = crate::nvidia_driver::diagnose(&crate::nvidia_driver::probe());
        return Some(why.unwrap_or_else(|| format!("the NVIDIA driver is older than {} (CUDA 13)", crate::nvidia_driver::DRIVER_MIN)));
    }
    if has(&["ninja: build stopped", "Error building extension"]) {
        return Some("a run-time kernel build failed; goethe doctor checks the compiler and CUDA toolkit".into());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_waiting_load_only_says_it_is_loading() {
        assert_eq!(BANNER, "Loading into VRAM...");
        assert_eq!(heartbeat(12), "loading into VRAM, 12 s", "the same line whether weights load, kernels build or warm up");
    }

    #[test]
    fn known_engine_failures_get_a_plain_cause() {
        let lines = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(likely_cause(&lines(&["RuntimeError: CUDA error: no kernel image is available for execution"])).unwrap().contains("architecture"));
        assert!(likely_cause(&lines(&["Error building extension 'x'", "ninja: build stopped"])).unwrap().contains("kernel build failed"));
        assert!(likely_cause(&lines(&["error -- unsupported GNU version! gcc versions later than 15"])).is_some());
        let metal = likely_cause(&lines(&["ggml_metal_init: allocating", "llama_kv_cache: failed to allocate buffer for kv cache", "ggml_metal_buffer_init: error: failed to allocate buffer"])).unwrap();
        assert!(metal.contains("Mac ran out of GPU memory") && !metal.contains("NVIDIA") && !metal.contains("CUDA"), "{metal}");
        assert_eq!(likely_cause(&lines(&["Traceback", "ValueError: bad config"])), None);
    }

    #[test]
    fn a_failure_report_never_shows_a_python_traceback_and_names_a_pack_that_does_not_run() {
        let td = tempfile::tempdir().unwrap();
        let logs = td.path().join("logs");
        std::fs::create_dir_all(&logs).unwrap();
        let log = "starting\nTraceback (most recent call last):\n  File \"/py-env/lib/x.py\", line 9, in open\n    raise RuntimeError(why)\nRuntimeError: device NVIDIA GeForce RTX 5090 cc 12.0 (170 SMs) runs no variant of this pack: rtx_pro_6000_maxq (sm_120a, 188 SMs): the plan\'s grid needs 188 SMs co-resident, the device has 170. Fallback: serve this model with llama.cpp\n";
        std::fs::write(logs.join("inst-y.log"), log).unwrap();
        let r = failure_report(td.path(), "inst-y");
        assert!(!r.contains("Traceback") && !r.contains("File \"") && !r.contains("raise RuntimeError"), "{r}");
        assert!(r.contains("likely cause: kernel pack for rtx_pro_6000_maxq (sm_120a, 188 SMs) does not run on NVIDIA GeForce RTX 5090 cc 12.0 (170 SMs)"), "{r}");
        assert!(r.contains("RuntimeError: device NVIDIA GeForce RTX 5090") && r.ends_with("goethe logs inst-y -n 200"), "{r}");
    }

    #[test]
    fn failure_report_shows_daemon_reason_and_engine_errors() {
        let td = tempfile::tempdir().unwrap();
        let logs = td.path().join("logs");
        std::fs::create_dir_all(&logs).unwrap();
        std::fs::write(logs.join("goethed.log"), "other line\nwatchdog instance=inst-x reason=process exited\n").unwrap();
        std::fs::write(logs.join("inst-x.log"), "loading\nRuntimeError: CUDA error: no kernel image is available\ndone\n").unwrap();
        let r = failure_report(td.path(), "inst-x");
        assert!(r.contains("goethed: watchdog instance=inst-x reason=process exited"), "{r}");
        assert!(r.contains("RuntimeError: CUDA error: no kernel image"), "{r}");
        assert!(!r.contains("other line") && !r.contains("    loading"), "{r}");
        assert!(r.ends_with("goethe logs inst-x -n 200"));
        let none = failure_report(td.path(), "inst-missing");
        assert!(none.starts_with("the engine failed to start (instance inst-missing)"));
    }
}
