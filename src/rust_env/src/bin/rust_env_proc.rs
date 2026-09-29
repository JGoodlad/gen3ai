//! FRONT END B — the env core in its OWN PROCESS (M5 Lane B, `designs/endstate/program_rust_core.md`
//! §2 M5). The Python host (`src/utils/rust_env/proc.py`, `ProcCore`) spawns it with the columns'
//! shared mapping as an inherited fd; a SIGKILL, an abort or an OOM kill here takes only this process —
//! the host sees EOF on the reply pipe, raises a typed error and spawns a fresh core.
//!
//!   rust_env_proc --fd <memfd>
//!
//! 1. HANDSHAKE, before reading a byte: one line on stdout —
//!    `rust_env_proc\twire=<id>\tobs_dim=<d>\tn_columns=<k>\tschema=<id>\tstamp=<stamp>\n`.
//!    The host REFUSES a foreign build here (it closes stdin; this process exits having run nothing).
//! 2. INIT (the first request, once): `[b'I'][u32 LE len][spec JSON]` → `Core::new(Spec::from_json)`,
//!    map the fd ([`shm::Mapping`]), write the header, `Core::freeze` on the mapped columns (the ONE
//!    binding: a rebind cannot be expressed through this front end).
//! 3. THE LOOP: one request byte → one reply frame `[status u8][len u32 LE][payload]`. A control byte
//!    (`shm::ctl`) is the front end's own; EVERY other byte is forwarded to `Core::dispatch` unread,
//!    so the core — not this file — decides what an opcode means. EOF on stdin (the host closed it, or
//!    died) ends the loop: no orphan outlives its host.
//!
//! A THIN front end: no battle logic here. A panic in an op is caught (status `PANIC`) and POISONS
//! this child's core, as a panic inside the FFI's locked section poisons that handle.

use std::io::{Read, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};

use pokesim_env::core::columns::{status, N_COLUMNS, OBS_DIM, SCHEMA_ID};
use pokesim_env::core::{Core, DispatchError, Spec, STAMP};
use pokesim_env::shm::{ctl, Mapping, HANDSHAKE_TAG, WIRE_ID};

fn err(st: i32, kind: &str, message: impl Into<String>) -> DispatchError {
    DispatchError { status: st, env: None, kind: kind.into(), class: None, message: message.into(), script: None }
}

fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    p.downcast_ref::<String>()
        .cloned()
        .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_else(|| "a panic with a non-string payload".into())
}

/// One reply frame, written with ONE `write` and flushed. `false` when the host is gone.
fn reply(out: &mut impl Write, st: i32, payload: &str) -> bool {
    let mut f = Vec::with_capacity(5 + payload.len());
    f.push(st as u8);
    f.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    f.extend_from_slice(payload.as_bytes());
    out.write_all(&f).and_then(|_| out.flush()).is_ok()
}

fn reply_result(out: &mut impl Write, r: Result<String, DispatchError>) -> bool {
    match r {
        Ok(payload) => reply(out, status::OK, &payload),
        Err(e) => reply(out, e.status, &e.json()),
    }
}

/// STARTUP: the spec → the core → the mapping → the header → FREEZE.
fn init(fd: i32, text: &str) -> Result<(Core, Mapping), DispatchError> {
    let spec = Spec::from_json(text).map_err(|e| err(status::CALLER, "caller", e))?;
    let n = spec.n;
    let mut core = Core::new(spec).map_err(|e| err(status::CALLER, "startup", e))?;
    // SAFETY: `fd` is the host's mapping, inherited by this process for exactly this call.
    let mut map = unsafe { Mapping::from_fd(fd, n) }.map_err(|e| err(status::FAULT, "transport", e))?;
    map.write_header();
    core.freeze(map.addrs())?;
    Ok((core, map))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let fd = match args.as_slice() {
        [_, flag, fd] if flag == "--fd" => fd.parse::<i32>().ok(),
        _ => None,
    };
    let Some(fd) = fd else {
        eprintln!("usage: rust_env_proc --fd <memfd>");
        std::process::exit(2);
    };
    let mut out = std::io::stdout().lock();
    let mut inp = std::io::stdin().lock();

    // 1. HANDSHAKE — before anything is read or built.
    let hello = format!(
        "{HANDSHAKE_TAG}\twire={WIRE_ID}\tobs_dim={OBS_DIM}\tn_columns={N_COLUMNS}\tschema={SCHEMA_ID}\tstamp={STAMP}\n"
    );
    if out.write_all(hello.as_bytes()).and_then(|_| out.flush()).is_err() {
        return;
    }

    // 2. INIT.
    let mut op = [0u8; 1];
    if inp.read_exact(&mut op).is_err() {
        return; // the host refused the handshake (or went away): nothing ran
    }
    if op[0] != ctl::INIT {
        reply_result(&mut out, Err(err(status::LIFECYCLE, "lifecycle", format!("the first request must be INIT, got byte {}", op[0]))));
        std::process::exit(1);
    }
    let mut len = [0u8; 4];
    let mut text = Vec::new();
    if inp.read_exact(&mut len).is_err() || inp.by_ref().take(u32::from_le_bytes(len) as u64).read_to_end(&mut text).is_err() {
        return;
    }
    let started = catch_unwind(AssertUnwindSafe(|| {
        let text = String::from_utf8(text).map_err(|e| err(status::CALLER, "caller", format!("spec: not UTF-8: {e}")))?;
        init(fd, &text)
    }));
    let (mut core, map) = match started {
        Ok(Ok(x)) => x,
        Ok(Err(e)) => {
            reply_result(&mut out, Err(e));
            std::process::exit(1);
        }
        Err(p) => {
            reply_result(&mut out, Err(err(status::PANIC, "panic", format!("PANIC at startup: {}", panic_msg(p)))));
            std::process::exit(1);
        }
    };
    let addrs = map.addrs();
    if !reply(&mut out, status::OK, "") {
        return;
    }

    // 3. THE LOOP.
    let mut poisoned: Option<String> = None;
    loop {
        if inp.read_exact(&mut op).is_err() {
            break; // EOF: the host closed stdin or died
        }
        let code = op[0];
        if code == ctl::QUIT {
            reply(&mut out, status::OK, "");
            break;
        }
        if code == ctl::ABORT_PROBE {
            std::process::abort();
        }
        let r = if let Some(p) = &poisoned {
            Err(err(status::LIFECYCLE, "lifecycle", format!("the child's core is POISONED by an earlier panic ({p}); respawn it")))
        } else {
            match catch_unwind(AssertUnwindSafe(|| -> Result<String, DispatchError> {
                match code {
                    ctl::BANK => Ok(format!("[{}]", core.bank().items().iter().map(|b| b.json()).collect::<Vec<_>>().join(","))),
                    ctl::PANIC_PROBE => panic!("rust_env_proc PANIC_PROBE: a deliberate panic inside the child's op section"),
                    op => match core.dispatch(op, addrs) {
                        status::OK => Ok(String::new()),
                        s => Err(core.last_error().cloned().unwrap_or_else(|| err(s, "fault", "a failed dispatch recorded no error"))),
                    },
                }
            })) {
                Ok(r) => r,
                Err(p) => {
                    let m = panic_msg(p);
                    poisoned = Some(m.clone());
                    Err(err(status::PANIC, "panic", format!("PANIC inside the child (its core is now poisoned): {m}")))
                }
            }
        };
        if !reply_result(&mut out, r) {
            break; // the host is gone (EPIPE: Rust ignores SIGPIPE)
        }
    }
    drop(core); // joins the workers, which hold addresses inside the mapping
    drop(map);
}
