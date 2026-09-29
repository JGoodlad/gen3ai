//! THE ONE ENTRY — `Core::dispatch(op, cols)` (M5 Lane 0). Both front ends forward an opcode and
//! a column set here and turn the returned STATUS into a typed Python error (`protocol.py`); neither
//! names an op's semantics.
//!
//! [`Core`] is the pool plus what the contract promises around it: the frozen column binding, the
//! quarantine bank, the pool counters (written into the `counters` column after EVERY dispatch,
//! whatever its status) and the poison flag (after a batch failure every later op is refused).

use std::time::Instant;

use super::columns::{self, counter, op, status, ColAddrs, Dtype, COLUMNS, NCOUNTERS, N_COLUMNS};
use super::pool::{Job, Pool};
use super::refusal::{json_str, Bank, Banked, Class};
use super::spec::Spec;

/// A failed dispatch, typed. `json()` is what a front end hands to `protocol.error_for`.
#[derive(Clone, Debug)]
pub struct DispatchError {
    pub status: i32,
    pub env: Option<usize>,
    pub kind: String,
    pub class: Option<String>,
    pub message: String,
    /// The failing env's input log (a replayable script), when there is one.
    pub script: Option<String>,
}

impl DispatchError {
    fn lifecycle(msg: impl Into<String>) -> DispatchError {
        DispatchError { status: status::LIFECYCLE, env: None, kind: "lifecycle".into(), class: None, message: msg.into(), script: None }
    }

    pub fn json(&self) -> String {
        format!(
            "{{\"status\":{},\"env\":{},\"kind\":{},\"class\":{},\"message\":{},\"script\":{}}}",
            self.status,
            self.env.map_or("null".into(), |e| e.to_string()),
            json_str(&self.kind),
            self.class.as_deref().map_or("null".into(), json_str),
            json_str(&self.message),
            self.script.as_deref().map_or("null".into(), json_str),
        )
    }
}

/// See the module docs.
pub struct Core {
    pool: Pool,
    bank: Bank,
    counters: [u64; NCOUNTERS],
    reset_done: bool,
    poisoned: Option<String>,
    last_error: Option<DispatchError>,
}

impl Core {
    /// STARTUP: build the pool (every resource acquired) and reserve the bank.
    pub fn new(spec: Spec) -> Result<Core, String> {
        let budget = spec.refusal_budget;
        Ok(Core { pool: Pool::new(spec)?, bank: Bank::new(budget), counters: [0; NCOUNTERS], reset_done: false, poisoned: None, last_error: None })
    }

    pub fn spec(&self) -> &Spec {
        &self.pool.ctx.spec
    }

    /// FREEZE: bind the caller's column set (validated: every address non-null and aligned for its
    /// dtype). Once only; every later dispatch must name exactly this set.
    pub fn freeze(&mut self, cols: ColAddrs) -> Result<(), DispatchError> {
        if self.pool.is_frozen() {
            return Err(DispatchError::lifecycle("freeze: the pool is already frozen (a column set is bound once)"));
        }
        for (i, spec) in COLUMNS.iter().enumerate() {
            let a = cols.0[i];
            let align = match spec.dtype {
                Dtype::U8 => 1,
                Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
                Dtype::U64 | Dtype::I64 => 8,
            };
            if a == 0 || a % align != 0 {
                return Err(DispatchError::lifecycle(format!("freeze: column `{}` at {a:#x} is null or not {align}-aligned", spec.name)));
            }
        }
        self.pool.bound = Some(cols);
        Ok(())
    }

    /// Env `i` of an inline pool (harness access; see `Pool::inline_env`).
    pub fn inline_env(&self, i: usize) -> Option<&super::pool::Env> {
        self.pool.inline_env(i)
    }

    pub fn counters(&self) -> &[u64; NCOUNTERS] {
        &self.counters
    }

    pub fn bank(&self) -> &Bank {
        &self.bank
    }

    pub fn last_error(&self) -> Option<&DispatchError> {
        self.last_error.as_ref()
    }

    /// THE ENTRY. Returns a `columns::status` code; on failure [`Core::last_error`] holds the typed
    /// error.
    pub fn dispatch(&mut self, opcode: u8, cols: ColAddrs) -> i32 {
        let t0 = Instant::now();
        self.counters[counter::DISPATCHES] += 1;
        let r = self.dispatch_inner(opcode, cols);
        let ns = t0.elapsed().as_nanos() as u64;
        self.counters[counter::CORE_NS_LAST] = ns;
        self.counters[counter::CORE_NS_TOTAL] += ns;
        self.counters[counter::THREADS_SPAWNED_AFTER_FREEZE] = self.pool.threads_after_freeze;
        self.counters[counter::ENVS_ADDED_AFTER_FREEZE] = self.pool.envs_after_freeze;
        self.counters[counter::BANK_GROWTH_AFTER_FREEZE] = self.bank.growth;
        if let Some(bound) = self.pool.bound {
            // SAFETY: the bound set was validated at freeze and names the caller's live columns.
            unsafe { bound.counters() }.copy_from_slice(&self.counters);
        }
        match r {
            Ok(()) => {
                self.last_error = None;
                status::OK
            }
            Err(e) => {
                let s = e.status;
                if s != status::LIFECYCLE && self.poisoned.is_none() {
                    self.poisoned = Some(format!("status {s}: {}", e.message));
                }
                self.last_error = Some(e);
                s
            }
        }
    }

    fn dispatch_inner(&mut self, opcode: u8, cols: ColAddrs) -> Result<(), DispatchError> {
        let bound = self.pool.bound.ok_or_else(|| DispatchError::lifecycle("dispatch before freeze: bind the columns first"))?;
        if cols != bound {
            self.counters[counter::COLUMN_REBINDS_AFTER_FREEZE] += 1;
            return Err(DispatchError::lifecycle("dispatch named a column set other than the frozen binding (refused)"));
        }
        if let Some(p) = &self.poisoned {
            return Err(DispatchError::lifecycle(format!("the pool is POISONED by an earlier failure ({p}); build a new one")));
        }
        let job = match opcode {
            op::RESET => Job::Reset,
            op::STEP if self.reset_done => Job::Step,
            op::STEP => return Err(DispatchError::lifecycle("STEP before the first RESET")),
            other => return Err(DispatchError::lifecycle(format!("unknown opcode {other} (known: {:?})", op::ALL))),
        };
        let reports = self.pool.run(job, cols).map_err(|m| DispatchError {
            status: status::FAULT,
            env: None,
            kind: "fault".into(),
            class: None,
            message: m,
            script: None,
        })?;
        if job == Job::Reset {
            self.reset_done = true;
        }
        let mut first: Option<DispatchError> = None;
        for rep in reports {
            self.counters[counter::DECISIONS] += rep.tally.decisions;
            self.counters[counter::EPISODES_STARTED] += rep.tally.started;
            self.counters[counter::EPISODES_ENDED] += rep.tally.ended;
            for (e, log, episode) in rep.quarantined {
                self.counters[counter::REFUSALS] += 1;
                let b = Banked { env: rep.env, episode, error: e, log };
                if let Some(d) = &self.spec().bank_dir {
                    let _ = b.write_to(d);
                }
                if let Err(b) = self.bank.push(b) {
                    first.get_or_insert(DispatchError {
                        status: status::BUDGET,
                        env: Some(rep.env),
                        kind: b.error.kind.into(),
                        class: b.error.py_class.map(str::to_string),
                        message: format!(
                            "quarantine {} exceeds the refusal budget {} declared at startup: {}",
                            self.counters[counter::REFUSALS],
                            self.bank.len(),
                            b.error.message
                        ),
                        script: Some(b.log.script("")),
                    });
                }
            }
            if let Some((e, log, _episode)) = rep.failed {
                let st = match e.class {
                    Class::Caller => status::CALLER,
                    Class::Panic => status::PANIC,
                    Class::Fault | Class::Quarantine => status::FAULT,
                };
                first.get_or_insert(DispatchError {
                    status: st,
                    env: Some(rep.env),
                    kind: e.kind.into(),
                    class: e.py_class.map(str::to_string),
                    message: format!("env {}: {}", rep.env, e.message),
                    script: Some(log.script("")),
                });
            }
        }
        match first {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}

/// Owned, aligned buffers for every column — for Rust-side callers (tests, harnesses, Lane J).
/// Each column is its own `u64`-backed allocation, so every dtype is aligned.
pub struct OwnedCols {
    bufs: Vec<Vec<u64>>,
    pub n: usize,
}

impl OwnedCols {
    pub fn new(n: usize) -> OwnedCols {
        let bytes = columns::col_bytes(n);
        let bufs = bytes.iter().map(|b| vec![0u64; b.div_ceil(8)]).collect();
        let mut o = OwnedCols { bufs, n };
        o.slice_mut::<i32>(columns::col::ACTION).fill(-1);
        o
    }

    pub fn addrs(&mut self) -> ColAddrs {
        let mut a = [0usize; N_COLUMNS];
        for (i, b) in self.bufs.iter_mut().enumerate() {
            a[i] = b.as_mut_ptr() as usize;
        }
        ColAddrs(a)
    }

    fn elems<T>(&self, col: usize) -> usize {
        columns::col_bytes(self.n)[col] / std::mem::size_of::<T>()
    }

    /// Column `col` as `T` (the caller names the column's own dtype).
    pub fn slice<T: Copy>(&self, col: usize) -> &[T] {
        assert_eq!(std::mem::size_of::<T>(), COLUMNS[col].dtype.size(), "column {} is not this dtype", COLUMNS[col].name);
        // SAFETY: the buffer is u64-aligned and at least col_bytes long; T is a plain number type.
        unsafe { std::slice::from_raw_parts(self.bufs[col].as_ptr() as *const T, self.elems::<T>(col)) }
    }

    pub fn slice_mut<T: Copy>(&mut self, col: usize) -> &mut [T] {
        assert_eq!(std::mem::size_of::<T>(), COLUMNS[col].dtype.size(), "column {} is not this dtype", COLUMNS[col].name);
        let n = self.elems::<T>(col);
        // SAFETY: as `slice`, exclusively borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.bufs[col].as_mut_ptr() as *mut T, n) }
    }

    /// The raw bytes of column `col` (for hashing / byte comparisons).
    pub fn bytes(&self, col: usize) -> &[u8] {
        let b = columns::col_bytes(self.n)[col];
        // SAFETY: the allocation holds at least `b` bytes.
        unsafe { std::slice::from_raw_parts(self.bufs[col].as_ptr() as *const u8, b) }
    }
}
