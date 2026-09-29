//! FRONT END A — the env core as a C ABI (M5 Lane A, `designs/endstate/program_rust_core.md` §2 M5).
//!
//! Python loads the `cdylib` with `ctypes` (`src/utils/rust_env/ffi.py`), which RELEASES THE GIL for
//! every foreign call. A THIN front end: each function forwards to [`crate::core::Core`]
//! (`Core::new(Spec::from_json)` → `freeze(cols)` → `dispatch(op, cols)`) and returns a
//! `columns::status` code; the typed error is `rust_env_last_error()` (this thread's, as
//! `DispatchError::json`). No battle logic lives here.
//!
//! THE SIGNATURES ARE GENERATED (the region between the markers, from `FUNCTIONS` in `ffi.py`):
//! each wrapper calls `imp::<name>` with the SAME arguments, so an implementation whose types drift
//! from the table does not compile, and the loader sets `ctypes` argtypes from the same rows.
//!
//! PANICS never cross the boundary: every export runs inside [`guard`] (`catch_unwind`) and a panic
//! becomes status `PANIC` — unwinding out of an `extern "C"` function ABORTS the process. A panic
//! inside a core's locked section POISONS that handle (every later call is `LIFECYCLE`), as a
//! batch failure poisons the pool inside the core. What cannot be caught in-process (an abort, a
//! stack overflow, OOM, a fault in unsafe code) still takes the process — the process front end's
//! reason to exist.
//!
//! One handle is SINGLE-CALLER: a second thread calling into a handle already in a call gets
//! `LIFECYCLE`, never a data race (`try_lock`).

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe, UnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock, TryLockError};

use crate::core::columns::{status, ColAddrs, NCOUNTERS, N_COLUMNS, SCHEMA_ID};
use crate::core::{Core, DispatchError, Spec};

/// The opaque handle `rust_env_new` returns.
pub struct Handle {
    core: Mutex<Core>,
    /// Set by a panic inside the locked section: the core's state is no longer trusted.
    poisoned: AtomicBool,
}

thread_local! {
    /// This thread's last failure (`DispatchError::json`), NUL-terminated.
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
    /// This thread's last returned string (`rust_env_bank_json`).
    static OUT: RefCell<CString> = RefCell::new(CString::default());
}

fn cstring(s: String) -> CString {
    CString::new(s.replace('\0', " ")).expect("NULs replaced")
}

fn set_err(e: &DispatchError) {
    LAST_ERROR.with(|c| *c.borrow_mut() = cstring(e.json()));
}

fn err(st: i32, kind: &str, message: impl Into<String>) -> DispatchError {
    DispatchError { status: st, env: None, kind: kind.into(), class: None, message: message.into(), script: None }
}

fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "a panic with a non-string payload".into())
}

/// What an export returns when a panic was caught at the boundary.
trait FfiRet {
    fn on_panic() -> Self;
}
impl FfiRet for i32 {
    fn on_panic() -> i32 {
        status::PANIC
    }
}
impl FfiRet for usize {
    fn on_panic() -> usize {
        usize::MAX
    }
}
impl FfiRet for *const c_char {
    fn on_panic() -> *const c_char {
        std::ptr::null()
    }
}
impl FfiRet for *mut Handle {
    fn on_panic() -> *mut Handle {
        std::ptr::null_mut()
    }
}
impl FfiRet for () {
    fn on_panic() {}
}

/// THE BOUNDARY: no unwind leaves an export. A caught panic is recorded as this thread's error.
fn guard<R: FfiRet, F: FnOnce() -> R + UnwindSafe>(f: F) -> R {
    match catch_unwind(f) {
        Ok(r) => r,
        Err(p) => {
            set_err(&err(status::PANIC, "panic", format!("PANIC across the FFI boundary: {}", panic_msg(p))));
            R::on_panic()
        }
    }
}

fn code(r: Result<(), DispatchError>) -> i32 {
    match r {
        Ok(()) => status::OK,
        Err(e) => {
            set_err(&e);
            e.status
        }
    }
}

/// Lock handle `h` (refusing null, poisoned and concurrent use) and run `f` on its core inside a
/// `catch_unwind` that POISONS the handle on a panic.
unsafe fn in_core<R>(h: *mut Handle, f: impl FnOnce(&mut Core) -> Result<R, DispatchError>) -> Result<R, DispatchError> {
    let hd: &Handle = h.as_ref().ok_or_else(|| err(status::CALLER, "caller", "a null rust env handle"))?;
    if hd.poisoned.load(Ordering::Acquire) {
        return Err(err(
            status::LIFECYCLE,
            "lifecycle",
            "the handle is POISONED by an earlier panic across the FFI boundary; free it and build a new core",
        ));
    }
    let mut g: MutexGuard<Core> = match hd.core.try_lock() {
        Ok(g) => g,
        Err(TryLockError::WouldBlock) => {
            return Err(err(status::LIFECYCLE, "lifecycle", "a concurrent call on one rust env handle (a core is single-caller)"))
        }
        Err(TryLockError::Poisoned(_)) => return Err(err(status::LIFECYCLE, "lifecycle", "the handle's lock is poisoned")),
    };
    match catch_unwind(AssertUnwindSafe(|| f(&mut g))) {
        Ok(r) => r,
        Err(p) => {
            hd.poisoned.store(true, Ordering::Release);
            Err(err(status::PANIC, "panic", format!("PANIC inside the core (the handle is now poisoned): {}", panic_msg(p))))
        }
    }
}

unsafe fn read_addrs(addrs: *const usize) -> Result<ColAddrs, DispatchError> {
    if addrs.is_null() {
        return Err(err(status::CALLER, "caller", "a null column-address array"));
    }
    let mut a = [0usize; N_COLUMNS];
    a.copy_from_slice(std::slice::from_raw_parts(addrs, N_COLUMNS));
    Ok(ColAddrs(a))
}

/// The hand-written implementations the generated wrappers call (the SAME argument lists).
mod imp {
    use super::*;

    const STAMP_C: &str = concat!(env!("POKESIM_ENV_STAMP"), "\0");

    pub unsafe fn stamp() -> *const c_char {
        STAMP_C.as_ptr() as *const c_char
    }

    pub unsafe fn ffi_sig() -> *const c_char {
        FFI_SIG_ID_C.as_ptr() as *const c_char
    }

    pub unsafe fn schema_id() -> *const c_char {
        static S: OnceLock<CString> = OnceLock::new();
        S.get_or_init(|| cstring(SCHEMA_ID.to_string())).as_ptr()
    }

    pub unsafe fn obs_dim() -> usize {
        crate::core::columns::OBS_DIM
    }

    pub unsafe fn n_columns() -> usize {
        N_COLUMNS
    }

    pub unsafe fn last_error() -> *const c_char {
        LAST_ERROR.with(|c| c.borrow().as_ptr())
    }

    pub unsafe fn new(spec_json: *const c_char) -> *mut Handle {
        let r = (|| {
            if spec_json.is_null() {
                return Err(err(status::CALLER, "caller", "a null spec"));
            }
            let text = CStr::from_ptr(spec_json).to_str().map_err(|e| err(status::CALLER, "caller", format!("spec: not UTF-8: {e}")))?;
            let spec = Spec::from_json(text).map_err(|e| err(status::CALLER, "caller", e))?;
            let core = Core::new(spec).map_err(|e| err(status::CALLER, "startup", e))?;
            Ok(Box::into_raw(Box::new(Handle { core: Mutex::new(core), poisoned: AtomicBool::new(false) })))
        })();
        r.unwrap_or_else(|e| {
            set_err(&e);
            std::ptr::null_mut()
        })
    }

    pub unsafe fn n(h: *mut Handle) -> usize {
        in_core(h, |c| Ok(c.spec().n)).unwrap_or_else(|e| {
            set_err(&e);
            usize::MAX
        })
    }

    pub unsafe fn freeze(h: *mut Handle, addrs: *const usize) -> i32 {
        code(in_core(h, |c| c.freeze(read_addrs(addrs)?)))
    }

    pub unsafe fn dispatch(h: *mut Handle, op: u8, addrs: *const usize) -> i32 {
        code(in_core(h, |c| {
            let a = read_addrs(addrs)?;
            match c.dispatch(op, a) {
                status::OK => Ok(()),
                s => Err(c.last_error().cloned().unwrap_or_else(|| err(s, "fault", "a failed dispatch recorded no error"))),
            }
        }))
    }

    pub unsafe fn counters(h: *mut Handle, out: *mut u64, len: usize) -> i32 {
        code(in_core(h, |c| {
            if out.is_null() || len != NCOUNTERS {
                return Err(err(status::CALLER, "caller", format!("counters: need a non-null u64[{NCOUNTERS}], got len {len}")));
            }
            std::slice::from_raw_parts_mut(out, len).copy_from_slice(c.counters());
            Ok(())
        }))
    }

    pub unsafe fn bank_json(h: *mut Handle) -> *const c_char {
        match in_core(h, |c| Ok(format!("[{}]", c.bank().items().iter().map(|b| b.json()).collect::<Vec<_>>().join(",")))) {
            Ok(s) => OUT.with(|o| {
                *o.borrow_mut() = cstring(s);
                o.borrow().as_ptr()
            }),
            Err(e) => {
                set_err(&e);
                std::ptr::null()
            }
        }
    }

    pub unsafe fn free(h: *mut Handle) {
        if !h.is_null() {
            drop(Box::from_raw(h));
        }
    }

    pub unsafe fn panic_probe(h: *mut Handle, kind: i32) -> i32 {
        match kind {
            0 => panic!("rust_env_panic_probe: a deliberate panic (string payload)"),
            1 => std::panic::panic_any(7u32),
            2 => code(in_core(h, |_| -> Result<(), DispatchError> { panic!("rust_env_panic_probe: a deliberate panic inside the core's locked section") })),
            k => code(Err(err(status::CALLER, "caller", format!("panic_probe: unknown kind {k}")))),
        }
    }
}

// ---- @generated-begin by `python -m utils.rust_env.ffi --write` — DO NOT EDIT this region.
// Source of truth: `FUNCTIONS` in `src/utils/rust_env/ffi.py`; pinned by `ffi_test.py` (routine).
// Each wrapper runs `imp::<name>` (hand-written, the SAME arguments) inside `guard`.

/// `ffi.sig_id()` — FNV-1a-64 of the table's canonical text; compared by the loader.
pub const FFI_SIG_ID: &str = "24b1abe303459a42";
/// The same, NUL-terminated, for `rust_env_ffi_sig`.
const FFI_SIG_ID_C: &str = "24b1abe303459a42\0";

/// the build stamp (`stamp.py`'s format); static
///
/// # Safety
/// The caller passes what the table's row says (a handle from `rust_env_new`, a
/// NUL-terminated string, an array of `N_COLUMNS` live, aligned column addresses).
#[no_mangle]
pub unsafe extern "C" fn rust_env_stamp() -> *const c_char {
    guard(AssertUnwindSafe(|| imp::stamp()))
}

/// this table's id (`ffi.sig_id()`), compiled in; static
///
/// # Safety
/// The caller passes what the table's row says (a handle from `rust_env_new`, a
/// NUL-terminated string, an array of `N_COLUMNS` live, aligned column addresses).
#[no_mangle]
pub unsafe extern "C" fn rust_env_ffi_sig() -> *const c_char {
    guard(AssertUnwindSafe(|| imp::ffi_sig()))
}

/// the column schema id (`columns.schema_id()`); static
///
/// # Safety
/// The caller passes what the table's row says (a handle from `rust_env_new`, a
/// NUL-terminated string, an array of `N_COLUMNS` live, aligned column addresses).
#[no_mangle]
pub unsafe extern "C" fn rust_env_schema_id() -> *const c_char {
    guard(AssertUnwindSafe(|| imp::schema_id()))
}

/// the encoder's row length (`pokesim::encoder::OBS_DIM`)
///
/// # Safety
/// The caller passes what the table's row says (a handle from `rust_env_new`, a
/// NUL-terminated string, an array of `N_COLUMNS` live, aligned column addresses).
#[no_mangle]
pub unsafe extern "C" fn rust_env_obs_dim() -> usize {
    guard(AssertUnwindSafe(|| imp::obs_dim()))
}

/// `columns::N_COLUMNS` (the length of every `addrs` array)
///
/// # Safety
/// The caller passes what the table's row says (a handle from `rust_env_new`, a
/// NUL-terminated string, an array of `N_COLUMNS` live, aligned column addresses).
#[no_mangle]
pub unsafe extern "C" fn rust_env_n_columns() -> usize {
    guard(AssertUnwindSafe(|| imp::n_columns()))
}

/// the last failure ON THIS THREAD as `DispatchError::json` (valid until this thread's next call)
///
/// # Safety
/// The caller passes what the table's row says (a handle from `rust_env_new`, a
/// NUL-terminated string, an array of `N_COLUMNS` live, aligned column addresses).
#[no_mangle]
pub unsafe extern "C" fn rust_env_last_error() -> *const c_char {
    guard(AssertUnwindSafe(|| imp::last_error()))
}

/// STARTUP: `Core::new(Spec::from_json)`; null on failure (then `rust_env_last_error`)
///
/// # Safety
/// The caller passes what the table's row says (a handle from `rust_env_new`, a
/// NUL-terminated string, an array of `N_COLUMNS` live, aligned column addresses).
#[no_mangle]
pub unsafe extern "C" fn rust_env_new(spec_json: *const c_char) -> *mut Handle {
    guard(AssertUnwindSafe(|| imp::new(spec_json)))
}

/// the pool's N (usize::MAX on a null handle or a panic)
///
/// # Safety
/// The caller passes what the table's row says (a handle from `rust_env_new`, a
/// NUL-terminated string, an array of `N_COLUMNS` live, aligned column addresses).
#[no_mangle]
pub unsafe extern "C" fn rust_env_n(h: *mut Handle) -> usize {
    guard(AssertUnwindSafe(|| imp::n(h)))
}

/// FREEZE: `Core::freeze` (once)
///
/// # Safety
/// The caller passes what the table's row says (a handle from `rust_env_new`, a
/// NUL-terminated string, an array of `N_COLUMNS` live, aligned column addresses).
#[no_mangle]
pub unsafe extern "C" fn rust_env_freeze(h: *mut Handle, addrs: *const usize) -> i32 {
    guard(AssertUnwindSafe(|| imp::freeze(h, addrs)))
}

/// THE ENTRY: `Core::dispatch(op, cols)`
///
/// # Safety
/// The caller passes what the table's row says (a handle from `rust_env_new`, a
/// NUL-terminated string, an array of `N_COLUMNS` live, aligned column addresses).
#[no_mangle]
pub unsafe extern "C" fn rust_env_dispatch(h: *mut Handle, op: u8, addrs: *const usize) -> i32 {
    guard(AssertUnwindSafe(|| imp::dispatch(h, op, addrs)))
}

/// copy the pool counters (`len` must be NCOUNTERS) — readable before freeze too
///
/// # Safety
/// The caller passes what the table's row says (a handle from `rust_env_new`, a
/// NUL-terminated string, an array of `N_COLUMNS` live, aligned column addresses).
#[no_mangle]
pub unsafe extern "C" fn rust_env_counters(h: *mut Handle, out: *mut u64, len: usize) -> i32 {
    guard(AssertUnwindSafe(|| imp::counters(h, out, len)))
}

/// the refusal bank as a JSON array of `Banked::json` (valid until this thread's next call)
///
/// # Safety
/// The caller passes what the table's row says (a handle from `rust_env_new`, a
/// NUL-terminated string, an array of `N_COLUMNS` live, aligned column addresses).
#[no_mangle]
pub unsafe extern "C" fn rust_env_bank_json(h: *mut Handle) -> *const c_char {
    guard(AssertUnwindSafe(|| imp::bank_json(h)))
}

/// drop the core (joins its workers); null is a no-op
///
/// # Safety
/// The caller passes what the table's row says (a handle from `rust_env_new`, a
/// NUL-terminated string, an array of `N_COLUMNS` live, aligned column addresses).
#[no_mangle]
pub unsafe extern "C" fn rust_env_free(h: *mut Handle) {
    guard(AssertUnwindSafe(|| imp::free(h)))
}

/// TEST HOOK: panic on purpose (0: a string payload; 1: a non-string payload; 2: inside the handle's locked section, which poisons it) — proves a panic becomes status PANIC, never an abort
///
/// # Safety
/// The caller passes what the table's row says (a handle from `rust_env_new`, a
/// NUL-terminated string, an array of `N_COLUMNS` live, aligned column addresses).
#[no_mangle]
pub unsafe extern "C" fn rust_env_panic_probe(h: *mut Handle, kind: i32) -> i32 {
    guard(AssertUnwindSafe(|| imp::panic_probe(h, kind)))
}

// ---- @generated-end

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_static_strings_are_nul_terminated_and_name_this_build() {
        unsafe {
            assert_eq!(CStr::from_ptr(rust_env_ffi_sig()).to_str().unwrap(), FFI_SIG_ID);
            assert_eq!(CStr::from_ptr(rust_env_schema_id()).to_str().unwrap(), SCHEMA_ID);
            assert_eq!(CStr::from_ptr(rust_env_stamp()).to_str().unwrap(), crate::core::STAMP);
            assert_eq!(rust_env_n_columns(), N_COLUMNS);
        }
    }

    #[test]
    fn a_panic_never_crosses_the_boundary() {
        unsafe {
            assert_eq!(rust_env_panic_probe(std::ptr::null_mut(), 0), status::PANIC);
            let e = CStr::from_ptr(rust_env_last_error()).to_str().unwrap().to_string();
            assert!(e.contains("\"status\":3") && e.contains("deliberate panic"), "{e}");
            assert_eq!(rust_env_panic_probe(std::ptr::null_mut(), 1), status::PANIC);
            assert!(CStr::from_ptr(rust_env_last_error()).to_str().unwrap().contains("non-string payload"));
            assert_eq!(rust_env_panic_probe(std::ptr::null_mut(), 2), status::CALLER, "a null handle is a caller error");
        }
    }
}
