//! GATE ② — determinism: seed → bytes, thread-count-invariant (M5 Lane 0).
//!
//! The same staged inputs and the same seeded actions give BYTE-identical output columns (every
//! per-env output + the deterministic counters) at EVERY step, and an identical refusal bank,
//! whatever the worker count (1, 4, 5 = uneven blocks, 12 = one env per worker) and on a rerun.
//! Teeth: a different seed differs. Non-vacuity: episodes end (auto-reset is exercised) and rows
//! are written.

mod common;

use pokesim_env::core::columns::counter;

const N: usize = 12;
const STEPS: usize = 250;

#[test]
fn seed_to_bytes_is_invariant_under_thread_count_and_rerun() {
    let base = common::run(N, 1, 7, STEPS);
    assert!(base.counters[counter::EPISODES_ENDED] >= 3, "too few episodes ended to exercise auto-reset: {:?}", base.counters);
    assert!(base.counters[counter::DECISIONS] > (N * STEPS) as u64, "rows written: {:?}", base.counters);
    eprintln!("threads=1 counters {:?} ({} banked)", base.counters, base.bank.len());
    for t in [4, 5, 12, 1] {
        let got = common::run(N, t, 7, STEPS);
        let first = got.digests.iter().zip(&base.digests).position(|(a, b)| a != b);
        assert_eq!(first, None, "threads = {t} differs from threads = 1 at step {first:?}");
        assert_eq!(got.bank, base.bank, "threads = {t}: the refusal bank differs");
        for k in [counter::DECISIONS, counter::EPISODES_STARTED, counter::EPISODES_ENDED, counter::REFUSALS] {
            assert_eq!(got.counters[k], base.counters[k], "threads = {t}: counter {}", counter::NAMES[k]);
        }
        for &k in &counter::AFTER_FREEZE {
            assert_eq!(got.counters[k], 0, "threads = {t}: {} moved", counter::NAMES[k]);
        }
    }
    // Teeth: the gate can fail.
    let other = common::run(N, 1, 8, 40);
    assert_ne!(other.digests[1..], base.digests[1..41], "a different seed must differ");
}
