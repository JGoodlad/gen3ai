//! The shared-memory LAYOUT of the process front end and a std-only `mmap` (the crate takes no
//! `libc` dependency; std already links libc on Linux, so the two symbols are declared here).
//! `m5_loader.ShmLayout` is the Python twin — keep the two identical.

use std::ffi::c_void;
use std::os::unix::io::AsRawFd;

use pokesim::encoder::OBS_DIM;

use crate::core::ACT;

pub const HEADER: usize = 64; // [magic u64, n u64, k u64, status i64, core_ns u64, …]
pub const ERR_LEN: usize = 4096;
pub const K_MAX: usize = 1024;
pub const MAGIC: u64 = 0x4d35_5052_4f54_4f31; // "M5PROTO1"

fn align(x: usize) -> usize {
    (x + 63) & !63
}

/// Byte offsets: [err, obs, mask, need, reward, done, actions, rows, ok, total].
pub fn layout(n: usize) -> [usize; 10] {
    let sizes = [ERR_LEN, n * 2 * OBS_DIM * 4, n * 2 * ACT, n * 2, n * 4, n, n * 2 * 4, K_MAX * OBS_DIM * 4, K_MAX];
    let mut out = [0usize; 10];
    let mut at = HEADER;
    for (i, s) in sizes.iter().enumerate() {
        out[i] = at;
        at = align(at + s);
    }
    out[9] = at;
    out
}

extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
}
const PROT_RW: i32 = 0x1 | 0x2;
const MAP_SHARED: i32 = 0x01;

/// Map the WHOLE file at `path` read-write, shared. The mapping outlives the `File` (POSIX).
pub fn map(path: &str) -> Result<(*mut u8, usize), String> {
    let f = std::fs::OpenOptions::new().read(true).write(true).open(path).map_err(|e| format!("open {path}: {e}"))?;
    let len = f.metadata().map_err(|e| e.to_string())?.len() as usize;
    let p = unsafe { mmap(std::ptr::null_mut(), len, PROT_RW, MAP_SHARED, f.as_raw_fd(), 0) };
    if p as isize == -1 {
        return Err(format!("mmap {path} failed"));
    }
    Ok((p as *mut u8, len))
}
