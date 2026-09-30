//! FRONT END B's TRANSPORT — the shared-memory mapping the process front end's columns live in
//! (M5 Lane B, `designs/endstate/program_rust_core.md` §2 M5). The child is `src/bin/rust_env_proc.rs`;
//! the Python host is `src/utils/rust_env/proc.py` (`ProcCore`).
//!
//! The host creates an ANONYMOUS `memfd` (it never has a name in `/dev/shm`, so no exit path — not
//! even a SIGKILL of either process — can leak one), sizes it with [`layout`] and passes it to the
//! child by fd. The child maps it ([`Mapping::from_fd`]), writes the HEADER (magic, wire id, n,
//! obs_dim, column count, total, every column's offset) for the host to compare with its own layout,
//! and binds the columns to the core ONCE (`Core::freeze`).
//!
//! Layout: [`HEADER_BYTES`] of header, then every column of `columns::COLUMNS` in table order, each
//! at an [`ALIGN`]-aligned offset. `proc.layout` is the Python twin; the header comparison at startup
//! is what keeps the two honest at run time, `proc_test.py` what keeps the wire table honest at
//! commit time.
//!
//! Std-only (like the port): std already links libc on Linux, so `mmap` / `munmap` are declared here.

use std::ffi::c_void;
use std::fs::File;
use std::os::fd::{FromRawFd, RawFd};

use crate::core::columns::{col_bytes, ColAddrs, N_COLUMNS, OBS_DIM};

// ---- @generated-begin by `python -m utils.rust_env.proc --write` — DO NOT EDIT this region.
// Source of truth: the wire table in `src/utils/rust_env/proc.py`; pinned by `proc_test.py` (routine).

/// `proc.wire_id()` — FNV-1a-64 of the wire's canonical text; compared at the handshake.
pub const WIRE_ID: &str = "89d942296dcf35bb";
pub const WIRE_ID_U64: u64 = 0x89d942296dcf35bb;
/// Bytes before the first column (the header words live at its start).
pub const HEADER_BYTES: usize = 4096;
/// Every column's offset is a multiple of this.
pub const ALIGN: usize = 64;
/// Header word 0.
pub const MAGIC: u64 = 0x31435250564e4552;
/// The handshake line's first field.
pub const HANDSHAKE_TAG: &str = "rust_env_proc";

/// The header's u64 word indices (`OFFSETS` is followed by one word per column).
pub mod hdr {
    pub const MAGIC: usize = 0;
    pub const WIRE_ID: usize = 1;
    pub const N: usize = 2;
    pub const OBS_DIM: usize = 3;
    pub const N_COLUMNS: usize = 4;
    pub const TOTAL: usize = 5;
    pub const OFFSETS: usize = 6;
}

/// The front end's own request bytes (every other byte is a core opcode).
pub mod ctl {
    /// STARTUP, the child's FIRST request (once): `[u32 LE length][spec JSON]`; `Core::new`, map the columns, write the header, `freeze`
    pub const INIT: u8 = b'I';
    /// reply payload: the refusal bank, a JSON array of `Banked::json`
    pub const BANK: u8 = b'B';
    /// reply payload: the episodes that ENDED in the last op, a JSON array of `Finished::json` (M5 Lane H)
    pub const FINISHED: u8 = b'F';
    /// reply OK, drop the core (joins its workers), unmap, exit 0
    pub const QUIT: u8 = b'Q';
    /// TEST HOOK: a panic inside the child's guarded op section — status PANIC, and the child's core is POISONED (every later op LIFECYCLE)
    pub const PANIC_PROBE: u8 = b'P';
    /// TEST HOOK: `std::process::abort()` — what no in-process front end survives
    pub const ABORT_PROBE: u8 = b'A';
}

// ---- @generated-end

/// `(offsets, total bytes)` of the mapping for a pool of `n` envs (`proc.layout`'s twin).
pub fn layout(n: usize) -> ([usize; N_COLUMNS], usize) {
    let sizes = col_bytes(n);
    let mut off = [0usize; N_COLUMNS];
    let mut at = HEADER_BYTES;
    for (i, s) in sizes.iter().enumerate() {
        off[i] = at;
        at = (at + s).div_ceil(ALIGN) * ALIGN;
    }
    (off, at)
}

extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
}
const PROT_READ_WRITE: i32 = 0x1 | 0x2;
const MAP_SHARED: i32 = 0x01;

/// The child's view of the host's mapping. Unmapped on drop (drop the core FIRST: its bound column
/// addresses point in here).
pub struct Mapping {
    base: *mut u8,
    len: usize,
    n: usize,
    off: [usize; N_COLUMNS],
}

impl Mapping {
    /// Map the WHOLE file behind `fd` (inherited from the host) read-write, shared, for a pool of `n`;
    /// refuse one smaller than [`layout`]`(n)`.
    ///
    /// # Safety
    /// `fd` is an open file descriptor this process owns (it is consumed: closed once mapped).
    pub unsafe fn from_fd(fd: RawFd, n: usize) -> Result<Mapping, String> {
        let f = File::from_raw_fd(fd);
        let len = f.metadata().map_err(|e| format!("fd {fd}: {e}"))?.len() as usize;
        let (off, total) = layout(n);
        if len < total {
            return Err(format!("the mapping behind fd {fd} is {len} bytes; a pool of {n} needs {total}"));
        }
        let p = mmap(std::ptr::null_mut(), len, PROT_READ_WRITE, MAP_SHARED, fd, 0);
        if p as isize == -1 {
            return Err(format!("mmap of fd {fd} ({len} bytes) failed: {}", std::io::Error::last_os_error()));
        }
        drop(f); // the mapping outlives the descriptor (POSIX)
        Ok(Mapping { base: p as *mut u8, len, n, off })
    }

    /// Write the header the host compares with its own layout before the first op.
    pub fn write_header(&mut self) {
        let (_, total) = layout(self.n);
        let mut words = vec![0u64; hdr::OFFSETS + N_COLUMNS];
        words[hdr::MAGIC] = MAGIC;
        words[hdr::WIRE_ID] = WIRE_ID_U64;
        words[hdr::N] = self.n as u64;
        words[hdr::OBS_DIM] = OBS_DIM as u64;
        words[hdr::N_COLUMNS] = N_COLUMNS as u64;
        words[hdr::TOTAL] = total as u64;
        for (i, o) in self.off.iter().enumerate() {
            words[hdr::OFFSETS + i] = *o as u64;
        }
        // SAFETY: the header fits in HEADER_BYTES (`proc._check_table`) and the mapping is page-aligned.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), self.base as *mut u64, words.len()) };
    }

    /// Every column's address inside the mapping, in table order (the set the core freezes on).
    pub fn addrs(&self) -> ColAddrs {
        let mut a = [0usize; N_COLUMNS];
        for (i, o) in self.off.iter().enumerate() {
            a[i] = self.base as usize + o;
        }
        ColAddrs(a)
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: base/len are exactly what mmap returned.
        unsafe { munmap(self.base as *mut c_void, self.len) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layout_is_aligned_disjoint_and_leaves_the_header_alone() {
        for n in [1, 2, 7, 48] {
            let (off, total) = layout(n);
            let b = col_bytes(n);
            assert_eq!(off[0], HEADER_BYTES);
            for i in 0..N_COLUMNS {
                assert_eq!(off[i] % ALIGN, 0);
                let end = off[i] + b[i];
                assert!(end <= if i + 1 < N_COLUMNS { off[i + 1] } else { total });
            }
            assert!((hdr::OFFSETS + N_COLUMNS) * 8 <= HEADER_BYTES);
        }
        assert_eq!(WIRE_ID_U64, u64::from_str_radix(WIRE_ID, 16).unwrap());
    }
}
