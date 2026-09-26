//! FRONT END 1 — in-process FFI (EnvPool-style). A C ABI over [`crate::core::Pool`]; Python loads
//! the cdylib through `ctypes.CDLL`, which RELEASES THE GIL for the duration of every foreign call,
//! so the core's worker threads run while other Python threads proceed. No battle logic here:
//! each function turns caller pointers into the core's `Ptrs` and returns a status.
//!
//! A core panic is CAUGHT (per env, in `core::run_block`, and here) and becomes status -2 + an
//! error string; what CANNOT be caught in-process — an abort, a stack overflow, OOM, a fault in
//! unsafe code — takes the Python process with it. That is the crash-isolation difference
//! against the process front end.

use std::ffi::{c_char, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Mutex;

use crate::core::{Pool, Ptrs, Spec, STAMP};

static LAST_ERR: Mutex<Option<CString>> = Mutex::new(None);

fn set_err(e: String) {
    *LAST_ERR.lock().unwrap() = Some(CString::new(e.replace('\0', " ")).unwrap());
}

fn status(r: std::thread::Result<Result<(), String>>) -> i32 {
    match r {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_err(e);
            -1
        }
        Err(_) => {
            set_err("PANIC inside the core".into());
            -2
        }
    }
}

#[no_mangle]
pub extern "C" fn m5_stamp() -> *const c_char {
    STAMP.as_ptr() as *const c_char
}

#[no_mangle]
pub extern "C" fn m5_obs_dim() -> usize {
    pokesim::encoder::OBS_DIM
}

#[no_mangle]
pub extern "C" fn m5_error() -> *const c_char {
    match LAST_ERR.lock().unwrap().as_ref() {
        Some(c) => c.as_ptr(),
        None => b"\0".as_ptr() as *const c_char,
    }
}

#[no_mangle]
pub extern "C" fn m5_new(n: usize, threads: usize, seed: u64, opp_external: i32) -> *mut Pool {
    let r = catch_unwind(|| Pool::new(Spec { n, threads, seed, opp_external: opp_external != 0 }));
    match r {
        Ok(Ok(p)) => Box::into_raw(Box::new(p)),
        Ok(Err(e)) => {
            set_err(e);
            std::ptr::null_mut()
        }
        Err(_) => {
            set_err("PANIC in m5_new".into());
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `pool` from [`m5_new`]; every column sized per `core`'s contract for the pool's n.
#[no_mangle]
pub unsafe extern "C" fn m5_reset(pool: *mut Pool, obs: *mut f32, mask: *mut u8, need: *mut u8, reward: *mut f32, done: *mut u8) -> i32 {
    let p = Ptrs { obs: obs as usize, mask: mask as usize, need: need as usize, reward: reward as usize, done: done as usize, actions: 0 };
    status(catch_unwind(AssertUnwindSafe(|| (*pool).reset(p))))
}

/// # Safety
/// As [`m5_reset`]; `actions` is i32 [n][2].
#[no_mangle]
pub unsafe extern "C" fn m5_step(pool: *mut Pool, actions: *const i32, obs: *mut f32, mask: *mut u8, need: *mut u8, reward: *mut f32, done: *mut u8) -> i32 {
    let p = Ptrs { obs: obs as usize, mask: mask as usize, need: need as usize, reward: reward as usize, done: done as usize, actions: actions as usize };
    status(catch_unwind(AssertUnwindSafe(|| (*pool).step(p))))
}

/// # Safety
/// `rows` is f32 [k][OBS_DIM], `ok` u8 [k]; the pool is n = 1, threads = 1.
#[no_mangle]
pub unsafe extern "C" fn m5_successors(pool: *mut Pool, k: usize, rows: *mut f32, ok: *mut u8) -> i32 {
    let rows = std::slice::from_raw_parts_mut(rows, k * pokesim::encoder::OBS_DIM);
    let ok = std::slice::from_raw_parts_mut(ok, k);
    status(catch_unwind(AssertUnwindSafe(|| (*pool).successors(k, rows, ok))))
}

/// # Safety
/// `pool` from [`m5_new`].
#[no_mangle]
pub unsafe extern "C" fn m5_core_ns(pool: *const Pool) -> u64 {
    (*pool).last_core_ns
}

/// # Safety
/// `pool` from [`m5_new`], not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn m5_free(pool: *mut Pool) {
    if !pool.is_null() {
        drop(Box::from_raw(pool));
    }
}

/// # Safety
/// `pool` from [`m5_new`].
#[no_mangle]
pub unsafe extern "C" fn m5_refusals(pool: *const Pool) -> usize {
    (*pool).refused.len()
}
