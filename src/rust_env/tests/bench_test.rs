//! The core's per-decision cost through `Core::dispatch` (IGNORED — a measurement, not a gate):
//!
//!     (cd src/rust_env && CARGO_TARGET_DIR=$PWD/target cargo test --release --test bench_test -- --ignored --nocapture)
//!
//! The Phase-A prototype's TRAINING shape (both sides' rows out, seeded random policy, the bridge
//! corpus) at N = 48 and T = 1 / 8, on the RELEASE build (zero-filled rows, no self-check). Reports
//! the median over blocks of µs per written row and the box's contention factor alongside — the rule
//! is WARN, never stretch: a number read on a loaded box is labelled, not corrected.

mod common;

use std::time::Instant;

use pokesim_env::core::columns::col;
use pokesim_env::core::{Core, OwnedCols};

fn load1() -> f64 {
    std::fs::read_to_string("/proc/loadavg").ok().and_then(|s| s.split_whitespace().next()?.parse().ok()).unwrap_or(f64::NAN)
}

fn shape(n: usize, threads: usize, blocks: usize, steps: usize) -> (f64, u64) {
    let teams = common::corpus_teams();
    let nt = teams.len();
    let mut core = Core::new(common::spec(n, threads, teams)).unwrap();
    let mut cols = OwnedCols::new(n);
    let (mut sr, mut ar) = (common::Rng(1), common::Rng(2));
    for i in 0..n {
        common::stage(&mut cols, i, &mut sr, nt);
    }
    let a = cols.addrs();
    core.freeze(a).unwrap();
    assert_eq!(core.dispatch(b'R', a), 0);
    let mut per = Vec::new();
    let mut rows_total = 0u64;
    for b in 0..=blocks {
        let mut rows = 0u64;
        let mut ns = 0u128;
        for _ in 0..steps {
            common::random_actions(&mut cols, &mut ar);
            let t0 = Instant::now();
            assert_eq!(core.dispatch(b'S', a), 0);
            ns += t0.elapsed().as_nanos();
            rows += cols.slice::<u8>(col::NEED).iter().map(|&x| x as u64).sum::<u64>();
            for i in 0..n {
                if cols.slice::<u8>(col::DONE)[i] == 1 {
                    common::stage(&mut cols, i, &mut sr, nt);
                }
            }
        }
        if b > 0 {
            per.push(ns as f64 / 1e3 / rows as f64);
            rows_total += rows;
        }
    }
    per.sort_by(|x, y| x.partial_cmp(y).unwrap());
    (per[per.len() / 2], rows_total)
}

#[test]
#[ignore]
fn bench_core_training_shape() {
    let l0 = load1();
    for (n, t) in [(48, 1), (48, 8), (48, 1), (48, 8)] {
        let (us, rows) = shape(n, t, 5, 200);
        eprintln!("N = {n}, T = {t}: {us:.1} µs per row (median of 5 blocks, {rows} rows), load1 {:.1}", load1());
    }
    eprintln!("load1 before {l0:.1} (16 cores); a loaded box inflates every figure — label, never correct");
}
