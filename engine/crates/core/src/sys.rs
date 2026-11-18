// Process and host facts the rest of goethe needs, one API for Linux (/proc) and macOS (libproc, sysctl, mach).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProcStat {
    pub state: char,
    pub ppid: i32,
    pub start_ticks: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Memory {
    pub total_kb: u64,
    pub available_kb: u64,
    pub cached_kb: u64,
}

pub fn environ_var(pid: i32, key: &str) -> Option<String> {
    // One variable from a process's environment, when this user may read it.
    let prefix = format!("{key}=");
    environ(pid)?.into_iter().find_map(|kv| kv.strip_prefix(&prefix).map(str::to_string))
}

pub fn children(pid: i32) -> Vec<i32> {
    pids().into_iter().filter(|p| stat(*p).is_some_and(|s| s.ppid == pid)).collect()
}

pub fn random_bytes(buf: &mut [u8]) {
    // Kernel randomness: getrandom on Linux, getentropy (256 bytes per call) on macOS.
    #[cfg(target_os = "linux")]
    unsafe {
        libc::getrandom(buf.as_mut_ptr().cast(), buf.len(), 0);
    }
    #[cfg(target_os = "macos")]
    for chunk in buf.chunks_mut(256) {
        unsafe {
            libc::getentropy(chunk.as_mut_ptr().cast(), chunk.len());
        }
    }
}

#[cfg(target_os = "linux")]
mod imp {
    use super::{Memory, ProcStat};
    use std::path::PathBuf;

    pub fn pids() -> Vec<i32> {
        std::fs::read_dir("/proc")
            .map(|rd| rd.flatten().filter_map(|e| e.file_name().to_str().and_then(|s| s.parse().ok())).collect())
            .unwrap_or_default()
    }

    pub fn stat(pid: i32) -> Option<ProcStat> {
        // State, ppid and start time (fields 3, 4 and 22) of /proc/<pid>/stat.
        if pid <= 0 {
            return None;
        }
        let text = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        parse_stat(&text)
    }

    pub fn parse_stat(text: &str) -> Option<ProcStat> {
        let rest = text.get(text.rfind(')')? + 2..)?;
        let fields: Vec<&str> = rest.split_whitespace().collect();
        Some(ProcStat { state: fields.first()?.chars().next()?, ppid: fields.get(1)?.parse().ok()?, start_ticks: fields.get(19)?.parse().ok()? })
    }

    fn nul_list(raw: Vec<u8>) -> Vec<String> {
        raw.split(|b| *b == 0).filter(|s| !s.is_empty()).map(|s| String::from_utf8_lossy(s).into_owned()).collect()
    }

    pub fn argv(pid: i32) -> Option<Vec<String>> {
        std::fs::read(format!("/proc/{pid}/cmdline")).ok().map(nul_list)
    }

    pub fn environ(pid: i32) -> Option<Vec<String>> {
        std::fs::read(format!("/proc/{pid}/environ")).ok().map(nul_list)
    }

    pub fn comm(pid: i32) -> Option<String> {
        std::fs::read_to_string(format!("/proc/{pid}/comm")).ok().map(|s| s.trim().to_string())
    }

    pub fn exe(pid: i32) -> Option<PathBuf> {
        std::fs::read_link(format!("/proc/{pid}/exe")).ok()
    }

    pub fn exists(pid: i32) -> bool {
        pid > 0 && std::path::Path::new(&format!("/proc/{pid}")).exists()
    }

    pub fn uid(pid: i32) -> Option<u32> {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata(format!("/proc/{pid}")).ok().map(|m| m.uid())
    }

    pub fn rss_bytes(pid: i32) -> Option<u64> {
        let text = std::fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
        let kb: u64 = text.lines().find(|l| l.starts_with("VmRSS:"))?.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }

    pub fn footprint_bytes(pid: i32) -> Option<u64> {
        rss_bytes(pid)
    }

    pub fn cpu_ticks(pid: i32) -> Option<u64> {
        // User and system time of the process and its reaped children, in clock ticks (fields 14 to 17 of stat).
        let text = std::fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
        let fields: Vec<&str> = text.rsplit_once(')').map_or("", |(_, rest)| rest).split_whitespace().collect();
        Some((11..=14).filter_map(|i| fields.get(i)?.parse::<u64>().ok()).sum())
    }

    pub fn boot_id() -> String {
        std::fs::read_to_string("/proc/sys/kernel/random/boot_id").map(|s| s.trim().to_string()).unwrap_or_default()
    }

    pub fn boot_epoch() -> Option<i64> {
        let text = std::fs::read_to_string("/proc/stat").ok()?;
        text.lines().find_map(|l| l.strip_prefix("btime ")).and_then(|v| v.trim().parse().ok())
    }

    pub fn ticks_per_second() -> u64 {
        let t = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
        if t > 0 { t as u64 } else { 100 }
    }

    pub fn memory() -> Memory {
        let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let field = |k: &str| text.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse().ok()).unwrap_or(0);
        Memory { total_kb: field("MemTotal:"), available_kb: field("MemAvailable:"), cached_kb: field("Cached:") + field("Buffers:") }
    }

    pub fn cpu_times() -> Option<(Vec<u64>, Vec<Vec<u64>>)> {
        // Total and per-core jiffies in /proc/stat order (user, nice, system, idle, iowait, ...).
        let text = std::fs::read_to_string("/proc/stat").ok()?;
        let nums = |l: &str| l.split_whitespace().skip(1).filter_map(|v| v.parse().ok()).collect::<Vec<u64>>();
        let total = nums(text.lines().next()?);
        let cores = text.lines().skip(1).take_while(|l| l.starts_with("cpu")).map(nums).collect();
        Some((total, cores))
    }

    pub fn loadavg() -> (f64, f64, f64) {
        let text = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
        let v: Vec<f64> = text.split_whitespace().take(3).filter_map(|x| x.parse().ok()).collect();
        (v.first().copied().unwrap_or(0.0), v.get(1).copied().unwrap_or(0.0), v.get(2).copied().unwrap_or(0.0))
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use super::{Memory, ProcStat};
    use std::ffi::CStr;
    use std::path::PathBuf;

    pub fn pids() -> Vec<i32> {
        // proc_listallpids fills as many pids as fit; ask with room to spare.
        let n = unsafe { libc::proc_listallpids(std::ptr::null_mut(), 0) };
        if n <= 0 {
            return Vec::new();
        }
        let mut buf = vec![0i32; n as usize + 64];
        let bytes = (buf.len() * std::mem::size_of::<i32>()) as libc::c_int;
        let got = unsafe { libc::proc_listallpids(buf.as_mut_ptr().cast(), bytes) };
        buf.truncate(got.max(0) as usize);
        buf.retain(|p| *p > 0);
        buf
    }

    fn bsdinfo(pid: i32) -> Option<libc::proc_bsdinfo> {
        if pid <= 0 {
            return None;
        }
        let mut info: libc::proc_bsdinfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::proc_bsdinfo>() as libc::c_int;
        let got = unsafe { libc::proc_pidinfo(pid, libc::PROC_PIDTBSDINFO, 0, (&mut info as *mut libc::proc_bsdinfo).cast(), size) };
        (got == size).then_some(info)
    }

    pub fn stat(pid: i32) -> Option<ProcStat> {
        // Start time in microseconds stands in for Linux start ticks: it only has to stay the same for the life of a process.
        let i = bsdinfo(pid)?;
        let state = if i.pbi_status == libc::SZOMB { 'Z' } else { 'S' };
        Some(ProcStat { state, ppid: i.pbi_ppid as i32, start_ticks: i.pbi_start_tvsec * 1_000_000 + i.pbi_start_tvusec })
    }

    fn procargs(pid: i32) -> Option<(Vec<String>, Vec<String>)> {
        // KERN_PROCARGS2: argc, the executable path, padding, argv, then the environment.
        let mut argmax: libc::c_int = 0;
        let mut size = std::mem::size_of::<libc::c_int>();
        let mut mib = [libc::CTL_KERN, libc::KERN_ARGMAX];
        if unsafe { libc::sysctl(mib.as_mut_ptr(), 2, (&mut argmax as *mut libc::c_int).cast(), &mut size, std::ptr::null_mut(), 0) } != 0 {
            return None;
        }
        let mut buf = vec![0u8; argmax.max(4096) as usize];
        let mut len = buf.len();
        let mut mib = [libc::CTL_KERN, libc::KERN_PROCARGS2, pid];
        if unsafe { libc::sysctl(mib.as_mut_ptr(), 3, buf.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0) } != 0 {
            return None;
        }
        buf.truncate(len);
        parse_procargs(&buf)
    }

    pub fn parse_procargs(buf: &[u8]) -> Option<(Vec<String>, Vec<String>)> {
        let argc = i32::from_ne_bytes(buf.get(..4)?.try_into().ok()?).max(0) as usize;
        let mut at = 4;
        while at < buf.len() && buf[at] != 0 {
            at += 1;
        }
        while at < buf.len() && buf[at] == 0 {
            at += 1;
        }
        let mut words = buf[at..].split(|b| *b == 0).map(|s| String::from_utf8_lossy(s).into_owned());
        let argv: Vec<String> = words.by_ref().take(argc).collect();
        let env: Vec<String> = words.take_while(|s| !s.is_empty()).collect();
        Some((argv, env))
    }

    pub fn argv(pid: i32) -> Option<Vec<String>> {
        procargs(pid).map(|a| a.0)
    }

    pub fn environ(pid: i32) -> Option<Vec<String>> {
        procargs(pid).map(|a| a.1)
    }

    pub fn comm(pid: i32) -> Option<String> {
        let i = bsdinfo(pid)?;
        let name = unsafe { CStr::from_ptr(i.pbi_name.as_ptr()) }.to_string_lossy().into_owned();
        if name.is_empty() { Some(unsafe { CStr::from_ptr(i.pbi_comm.as_ptr()) }.to_string_lossy().into_owned()) } else { Some(name) }
    }

    pub fn exe(pid: i32) -> Option<PathBuf> {
        let mut buf = vec![0u8; 4096];
        let n = unsafe { libc::proc_pidpath(pid, buf.as_mut_ptr().cast(), buf.len() as u32) };
        (n > 0).then(|| PathBuf::from(String::from_utf8_lossy(&buf[..n as usize]).into_owned()))
    }

    pub fn exists(pid: i32) -> bool {
        bsdinfo(pid).is_some()
    }

    pub fn uid(pid: i32) -> Option<u32> {
        bsdinfo(pid).map(|i| i.pbi_uid)
    }

    pub fn footprint_bytes(pid: i32) -> Option<u64> {
        // Physical footprint, which on Apple Silicon includes the process's Metal (GPU) memory.
        let mut info: libc::rusage_info_v4 = unsafe { std::mem::zeroed() };
        let rc = unsafe { libc::proc_pid_rusage(pid, libc::RUSAGE_INFO_V4, (&mut info as *mut libc::rusage_info_v4).cast()) };
        (rc == 0).then_some(info.ri_phys_footprint)
    }

    pub fn rss_bytes(pid: i32) -> Option<u64> {
        let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
        let got = unsafe { libc::proc_pidinfo(pid, libc::PROC_PIDTASKINFO, 0, (&mut info as *mut libc::proc_taskinfo).cast(), size) };
        (got == size).then_some(info.pti_resident_size)
    }

    pub fn cpu_ticks(pid: i32) -> Option<u64> {
        // User and system time in hundredths of a second, the unit Linux clock ticks usually have.
        let mut info: libc::proc_taskinfo = unsafe { std::mem::zeroed() };
        let size = std::mem::size_of::<libc::proc_taskinfo>() as libc::c_int;
        let got = unsafe { libc::proc_pidinfo(pid, libc::PROC_PIDTASKINFO, 0, (&mut info as *mut libc::proc_taskinfo).cast(), size) };
        (got == size).then(|| (info.pti_total_user + info.pti_total_system) / 10_000_000)
    }

    fn sysctl_string(name: &str) -> Option<String> {
        let c = std::ffi::CString::new(name).ok()?;
        let mut len = 0usize;
        if unsafe { libc::sysctlbyname(c.as_ptr(), std::ptr::null_mut(), &mut len, std::ptr::null_mut(), 0) } != 0 || len == 0 {
            return None;
        }
        let mut buf = vec![0u8; len];
        if unsafe { libc::sysctlbyname(c.as_ptr(), buf.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0) } != 0 {
            return None;
        }
        Some(String::from_utf8_lossy(&buf[..len]).trim_end_matches('\0').trim().to_string())
    }

    pub fn sysctl_u64(name: &str) -> Option<u64> {
        let c = std::ffi::CString::new(name).ok()?;
        let mut v: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let rc = unsafe { libc::sysctlbyname(c.as_ptr(), (&mut v as *mut u64).cast(), &mut len, std::ptr::null_mut(), 0) };
        (rc == 0).then_some(if len == 4 { v & 0xffff_ffff } else { v })
    }

    pub fn sysctl_text(name: &str) -> Option<String> {
        sysctl_string(name)
    }

    pub fn boot_id() -> String {
        sysctl_string("kern.bootsessionuuid").unwrap_or_default()
    }

    pub fn boot_epoch() -> Option<i64> {
        let mut tv: libc::timeval = unsafe { std::mem::zeroed() };
        let mut len = std::mem::size_of::<libc::timeval>();
        let mut mib = [libc::CTL_KERN, libc::KERN_BOOTTIME];
        let rc = unsafe { libc::sysctl(mib.as_mut_ptr(), 2, (&mut tv as *mut libc::timeval).cast(), &mut len, std::ptr::null_mut(), 0) };
        (rc == 0).then_some(tv.tv_sec)
    }

    pub fn ticks_per_second() -> u64 {
        1_000_000
    }

    pub fn memory() -> Memory {
        // Total from hw.memsize; available = free, inactive and speculative pages; cached = file-backed pages.
        let total = sysctl_u64("hw.memsize").unwrap_or(0);
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(4096) as u64;
        let mut vm: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
        let mut count = (std::mem::size_of::<libc::vm_statistics64>() / std::mem::size_of::<libc::integer_t>()) as libc::mach_msg_type_number_t;
        #[allow(deprecated)]
        let rc = unsafe { libc::host_statistics64(libc::mach_host_self(), libc::HOST_VM_INFO64, (&mut vm as *mut libc::vm_statistics64).cast(), &mut count) };
        if rc != 0 {
            return Memory { total_kb: total / 1024, ..Memory::default() };
        }
        let pages = |n: u64| n * page / 1024;
        Memory {
            total_kb: total / 1024,
            available_kb: pages(vm.free_count as u64 + vm.inactive_count as u64 + vm.speculative_count as u64),
            cached_kb: pages(vm.external_page_count as u64),
        }
    }

    pub fn cpu_times() -> Option<(Vec<u64>, Vec<Vec<u64>>)> {
        // Per-core ticks from host_processor_info, in the /proc/stat order (user, nice, system, idle, iowait).
        let mut n: libc::natural_t = 0;
        let mut info: libc::processor_info_array_t = std::ptr::null_mut();
        let mut count: libc::mach_msg_type_number_t = 0;
        #[allow(deprecated)]
        let rc = unsafe { libc::host_processor_info(libc::mach_host_self(), libc::PROCESSOR_CPU_LOAD_INFO, &mut n, &mut info, &mut count) };
        if rc != 0 || info.is_null() {
            return None;
        }
        let raw = unsafe { std::slice::from_raw_parts(info, count as usize) };
        let cores: Vec<Vec<u64>> = raw
            .chunks(libc::CPU_STATE_MAX as usize)
            .take(n as usize)
            .map(|t| {
                let at = |i: libc::c_int| t.get(i as usize).copied().unwrap_or(0) as u32 as u64;
                vec![at(libc::CPU_STATE_USER), at(libc::CPU_STATE_NICE), at(libc::CPU_STATE_SYSTEM), at(libc::CPU_STATE_IDLE), 0]
            })
            .collect();
        #[allow(deprecated)]
        unsafe {
            libc::vm_deallocate(libc::mach_task_self(), info as libc::vm_address_t, (count as usize * std::mem::size_of::<libc::integer_t>()) as libc::vm_size_t);
        }
        let total = (0..5).map(|i| cores.iter().map(|c| c[i]).sum()).collect();
        Some((total, cores))
    }

    pub fn loadavg() -> (f64, f64, f64) {
        let mut v = [0f64; 3];
        let n = unsafe { libc::getloadavg(v.as_mut_ptr(), 3) };
        if n < 3 { (0.0, 0.0, 0.0) } else { (v[0], v[1], v[2]) }
    }
}

pub use imp::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn this_process_is_visible_with_its_parent_and_start_time() {
        let me = std::process::id() as i32;
        let s = stat(me).expect("own stat");
        assert!(s.ppid > 0 && s.start_ticks > 0 && s.state != 'Z');
        assert_eq!(stat(me).unwrap().start_ticks, s.start_ticks, "stable while the process lives");
        assert!(pids().contains(&me) && exists(me));
        assert!(argv(me).is_some_and(|a| !a.is_empty()));
        assert!(rss_bytes(me).is_some_and(|r| r > 0));
        assert!(!boot_id().is_empty() && boot_epoch().is_some_and(|b| b > 0));
        let m = memory();
        assert!(m.total_kb > 0 && m.available_kb > 0 && m.available_kb <= m.total_kb);
        let (total, cores) = cpu_times().expect("cpu times");
        assert!(!cores.is_empty() && total.len() >= 4);
    }

    #[test]
    fn our_own_environment_can_be_read_back() {
        let me = std::process::id() as i32;
        assert!(environ(me).is_some_and(|e| e.iter().any(|kv| kv.starts_with("PATH="))));
    }

    #[test]
    fn random_bytes_fill_the_buffer() {
        let mut a = [0u8; 600];
        random_bytes(&mut a);
        assert!(a.iter().filter(|b| **b != 0).count() > 500);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn procargs_split_argv_from_the_environment() {
        let mut buf = 2i32.to_ne_bytes().to_vec();
        buf.extend(b"/bin/x\0\0\0x\0-v\0PATH=/bin\0HOME=/h\0\0");
        let (argv, env) = imp::parse_procargs(&buf).unwrap();
        assert_eq!((argv, env), (vec!["x".to_string(), "-v".to_string()], vec!["PATH=/bin".to_string(), "HOME=/h".to_string()]));
    }
}

