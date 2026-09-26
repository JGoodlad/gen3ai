//! FRONT END 2 — a separate Rust env PROCESS over shared memory. Python creates a /dev/shm file
//! laid out by `shm::layout(n)`, writes the actions into it, and sends ONE opcode byte on stdin
//! per batch; this process runs the SAME `core::Pool` on the mapped columns and answers ONE status
//! byte on stdout. No battle logic here — only the opcode loop and the header.
//!
//!   m5_envproc <shm-path> <n> <threads> <seed> <opp_external 0|1>
//!   opcodes: b'R' reset · b'S' step · b'F' successors (k from the header) · b'Q' quit
//!   status:  0 ok · 1 error (the text in the error region, NUL-terminated)
//!   first line of stdout at start-up: the build stamp, then b'\n'.

use std::io::{Read, Write};

use m5proto::core::{Pool, Ptrs, Spec, STAMP};
use m5proto::shm::{layout, map, ERR_LEN, MAGIC};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 6 {
        eprintln!("usage: m5_envproc <shm> <n> <threads> <seed> <opp_external>");
        std::process::exit(2);
    }
    let n: usize = a[2].parse().unwrap();
    let spec = Spec { n, threads: a[3].parse().unwrap(), seed: a[4].parse().unwrap(), opp_external: a[5] == "1" };
    let (base, len) = map(&a[1]).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(3)
    });
    let off = layout(n);
    if len < off[9] {
        eprintln!("shm too small: {len} < {}", off[9]);
        std::process::exit(3);
    }
    let hdr = base as *mut u64;
    let mut out = std::io::stdout().lock();
    let mut pool = match Pool::new(spec) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("pool: {e}");
            std::process::exit(4)
        }
    };
    unsafe {
        *hdr = MAGIC;
        *hdr.add(1) = n as u64;
    }
    out.write_all(STAMP.trim_end_matches('\0').as_bytes()).unwrap();
    out.write_all(b"\n").unwrap();
    out.flush().unwrap();
    let at = |i: usize| unsafe { base.add(off[i]) } as usize;
    let p = Ptrs { obs: at(1), mask: at(2), need: at(3), reward: at(4), done: at(5), actions: at(6) };
    let mut stdin = std::io::stdin().lock();
    let mut op = [0u8; 1];
    loop {
        if stdin.read_exact(&mut op).is_err() {
            break;
        }
        let r = match op[0] {
            b'R' => unsafe { pool.reset(Ptrs { actions: 0, ..p }) },
            b'S' => unsafe { pool.step(p) },
            b'F' => {
                let k = unsafe { *hdr.add(2) } as usize;
                let rows = unsafe { std::slice::from_raw_parts_mut(at(7) as *mut f32, k * pokesim::encoder::OBS_DIM) };
                let ok = unsafe { std::slice::from_raw_parts_mut(at(8) as *mut u8, k) };
                pool.successors(k, rows, ok)
            }
            b'Q' => break,
            x => Err(format!("unknown opcode {x}")),
        };
        unsafe {
            *hdr.add(4) = pool.last_core_ns;
            *hdr.add(5) = pool.refused.len() as u64;
        }
        let st = match r {
            Ok(()) => 0u8,
            Err(e) => {
                let b = e.as_bytes();
                let m = b.len().min(ERR_LEN - 1);
                unsafe {
                    std::ptr::copy_nonoverlapping(b.as_ptr(), at(0) as *mut u8, m);
                    *(at(0) as *mut u8).add(m) = 0;
                }
                1u8
            }
        };
        if out.write_all(&[st]).and_then(|_| out.flush()).is_err() {
            break;
        }
    }
}
