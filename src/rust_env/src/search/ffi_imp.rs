//! The FFI implementations of Lane I's rows of `ffi.FUNCTIONS` (`rust_env_search_*`,
//! `rust_env_playout_*`). `crate::ffi`'s GENERATED wrappers call these by name (`imp` re-exports
//! this module), inside its `guard` (a panic never crosses the boundary).
//!
//! THE HANDLE CONTRACT is `crate::ffi::Handle`'s: single caller (a concurrent call is `LIFECYCLE`),
//! a panic inside the locked section POISONS the handle (every later call is `LIFECYCLE`). A search
//! ERROR (an unknown node, a refused request, a battle the port refuses) is status `CALLER`, kind
//! `search`, and does NOT poison: the tree and the playout table are left as they were before the
//! call for a refused request, and a failed arm or branch discards only what it built — a caller
//! may go on (open a new root).
//!
//! Returned strings are valid until THIS THREAD's next call into this module (a thread-local).

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, TryLockError};

use pokesim::encoder::OBS_DIM;
use pokesim::json::Json;

use super::tree::RowSink;
use super::{SearchCore, SearchHandle, SearchSpec};
use crate::core::columns::{status, ACT};
use crate::core::DispatchError;
use crate::ffi::{cstring, err, panic_msg, set_err};

thread_local! {
    static OUT: RefCell<CString> = RefCell::new(CString::default());
}

fn out_str(s: String) -> *const c_char {
    OUT.with(|o| {
        *o.borrow_mut() = cstring(s);
        o.borrow().as_ptr()
    })
}

fn search_err(msg: impl Into<String>) -> DispatchError {
    err(status::CALLER, "search", msg)
}

unsafe fn text<'a>(p: *const c_char, what: &str) -> Result<&'a str, DispatchError> {
    if p.is_null() {
        return Err(search_err(format!("a null {what}")));
    }
    CStr::from_ptr(p).to_str().map_err(|e| search_err(format!("{what}: not UTF-8: {e}")))
}

unsafe fn json(p: *const c_char, what: &str) -> Result<Json, DispatchError> {
    Json::parse(text(p, what)?).map_err(|e| search_err(format!("{what}: not JSON: {e}")))
}

/// Lock `h` (refusing null, poisoned and concurrent use) and run `f` inside a `catch_unwind` that
/// POISONS the handle on a panic.
unsafe fn in_search<R>(h: *mut SearchHandle, f: impl FnOnce(&mut SearchCore) -> Result<R, DispatchError>) -> Result<R, DispatchError> {
    let hd: &SearchHandle = h.as_ref().ok_or_else(|| err(status::CALLER, "caller", "a null search handle"))?;
    if hd.poisoned.load(Ordering::Acquire) {
        return Err(err(status::LIFECYCLE, "lifecycle", "the search handle is POISONED by an earlier panic; free it and build a new one"));
    }
    let mut g = match hd.core.try_lock() {
        Ok(g) => g,
        Err(TryLockError::WouldBlock) => {
            return Err(err(status::LIFECYCLE, "lifecycle", "a concurrent call on one search handle (a handle is single-caller)"))
        }
        Err(TryLockError::Poisoned(_)) => return Err(err(status::LIFECYCLE, "lifecycle", "the search handle's lock is poisoned")),
    };
    match catch_unwind(AssertUnwindSafe(|| f(&mut g))) {
        Ok(r) => r,
        Err(p) => {
            hd.poisoned.store(true, Ordering::Release);
            Err(err(status::PANIC, "panic", format!("PANIC inside the search core (the handle is now poisoned): {}", panic_msg(p))))
        }
    }
}

fn to_cstr(r: Result<String, DispatchError>) -> *const c_char {
    match r {
        Ok(s) => out_str(s),
        Err(e) => {
            set_err(&e);
            std::ptr::null()
        }
    }
}

pub unsafe fn search_new(spec_json: *const c_char) -> *mut SearchHandle {
    let r = (|| {
        let spec = SearchSpec::from_json(text(spec_json, "search spec")?).map_err(|e| err(status::CALLER, "caller", e))?;
        Ok(Box::into_raw(Box::new(SearchHandle { core: Mutex::new(SearchCore::new(spec)), poisoned: AtomicBool::new(false) })))
    })();
    r.unwrap_or_else(|e: DispatchError| {
        set_err(&e);
        std::ptr::null_mut()
    })
}

pub unsafe fn search_free(h: *mut SearchHandle) {
    if !h.is_null() {
        drop(Box::from_raw(h));
    }
}

pub unsafe fn search_open_root(h: *mut SearchHandle, req_json: *const c_char) -> *const c_char {
    to_cstr(in_search(h, |c| {
        let req = json(req_json, "open_root request")?;
        c.tree.open_root(&req).map(|body| format!("{{{body}}}")).map_err(search_err)
    }))
}

pub unsafe fn search_expand(h: *mut SearchHandle, req_json: *const c_char, rows: *mut f32, cap: usize) -> *const c_char {
    to_cstr(in_search(h, |c| {
        let req = json(req_json, "expand_many request")?;
        if rows.is_null() && cap > 0 {
            return Err(search_err("expand: a null row buffer"));
        }
        let buf: &mut [f32] = if cap == 0 { &mut [] } else { std::slice::from_raw_parts_mut(rows, cap * OBS_DIM) };
        let mut sink = RowSink { rows: buf, used: 0 };
        c.tree.expand(&req, &mut sink).map(|body| format!("{{{body},\"rows_used\":{}}}", sink.used)).map_err(search_err)
    }))
}

pub unsafe fn search_stats(h: *mut SearchHandle) -> *const c_char {
    to_cstr(in_search(h, |c| {
        Ok(format!(
            "{{\"tree_nodes\":{},\"nodes_made\":{},\"arms\":{},\"max_nodes\":{},\"playout_branches_live\":{},\"playout_answered\":{},\"playout_finished\":{},\"max_branches\":{}}}",
            c.tree.len(),
            c.tree.nodes_made,
            c.tree.arms,
            c.spec.max_nodes,
            c.playouts.live(),
            c.playouts.answered,
            c.playouts.finished,
            c.spec.max_branches
        ))
    }))
}

pub unsafe fn playout_open(h: *mut SearchHandle, req_json: *const c_char) -> *const c_char {
    to_cstr(in_search(h, |c| {
        let req = json(req_json, "playout request")?;
        c.playouts.open(&req).map_err(search_err)
    }))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn playout_step(
    h: *mut SearchHandle,
    actions: *const i32,
    n: usize,
    rows: *mut f32,
    masks: *mut u8,
    who: *mut u32,
    cap: usize,
) -> usize {
    in_search(h, |c| {
        if (n > 0 && actions.is_null()) || (cap > 0 && (rows.is_null() || masks.is_null() || who.is_null())) {
            return Err(search_err("playout_step: a null buffer"));
        }
        let acts: &[i32] = if n == 0 { &[] } else { std::slice::from_raw_parts(actions, n) };
        let (r, m, w): (&mut [f32], &mut [u8], &mut [u32]) = if cap == 0 {
            (&mut [], &mut [], &mut [])
        } else {
            (
                std::slice::from_raw_parts_mut(rows, cap * OBS_DIM),
                std::slice::from_raw_parts_mut(masks, cap * ACT),
                std::slice::from_raw_parts_mut(who, cap),
            )
        };
        c.playouts.step(acts, r, m, w).map_err(search_err)
    })
    .unwrap_or_else(|e| {
        set_err(&e);
        usize::MAX
    })
}

pub unsafe fn playout_results(h: *mut SearchHandle) -> *const c_char {
    to_cstr(in_search(h, |c| Ok(c.playouts.results())))
}
