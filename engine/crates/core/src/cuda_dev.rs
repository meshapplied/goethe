// The SM count and opt-in shared memory of one GPU from the CUDA driver (no context, no new dependency): what a pack variant's grid is checked against.

use std::ffi::{c_char, c_int, c_void, CStr};

type Init = unsafe extern "C" fn(u32) -> c_int;
type Count = unsafe extern "C" fn(*mut c_int) -> c_int;
type Get = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
type Uuid = unsafe extern "C" fn(*mut [u8; 16], c_int) -> c_int;
type Attr = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;

const MULTIPROCESSOR_COUNT: c_int = 16;
const MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;

fn symbol<T: Copy>(lib: *mut c_void, name: &CStr) -> Option<T> {
    // A driver entry point, or None when the driver lacks it.
    let p = unsafe { libc::dlsym(lib, name.as_ptr() as *const c_char) };
    (!p.is_null()).then(|| unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) })
}

pub fn sm_info(uuid: &str) -> Option<(u64, u64)> {
    // (SMs, opt-in shared memory per block in bytes) of the GPU whose NVML uuid is `uuid`; None without a CUDA driver or when that GPU is not visible to CUDA.
    let want: String = uuid.trim_start_matches("GPU-").chars().filter(|c| *c != '-').collect::<String>().to_ascii_lowercase();
    unsafe {
        let lib = libc::dlopen(c"libcuda.so.1".as_ptr(), libc::RTLD_NOW);
        if lib.is_null() {
            return None;
        }
        let init: Init = symbol(lib, c"cuInit")?;
        let count: Count = symbol(lib, c"cuDeviceGetCount")?;
        let get: Get = symbol(lib, c"cuDeviceGet")?;
        let id: Uuid = symbol(lib, c"cuDeviceGetUuid")?;
        let attr: Attr = symbol(lib, c"cuDeviceGetAttribute")?;
        if init(0) != 0 {
            return None;
        }
        let mut n = 0;
        if count(&mut n) != 0 {
            return None;
        }
        for ordinal in 0..n {
            let (mut dev, mut bytes) = (0, [0u8; 16]);
            if get(&mut dev, ordinal) != 0 || id(&mut bytes, dev) != 0 {
                continue;
            }
            let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            if hex != want {
                continue;
            }
            let (mut sms, mut smem) = (0, 0);
            if attr(&mut sms, MULTIPROCESSOR_COUNT, dev) != 0 {
                return None;
            }
            let _ = attr(&mut smem, MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev);
            return Some((sms as u64, smem.max(0) as u64));
        }
    }
    None
}
