//! `gen3_expand_many_timing_v1` — OPT-IN per-phase wall accounting inside the persistent
//! search driver's `expand_many`.
//!
//! # Why this exists in the binary rather than in a profiler
//!
//! `SearchSession.expand_many` is ONE span from Python's side: the request write, whatever the
//! child does, and `json.loads` of the reply are indistinguishable from outside the pipe. A
//! profile that cannot say which of the three it is cannot choose between "make the payload
//! smaller" and "make the sim faster", and those are opposite changes. So the child reports its
//! OWN split, and Python subtracts it from the span it measures; the remainder is transport.
//!
//! # 🚨 IT IS OFF UNLESS `POKESIM_SEARCH_TIMING=1`, AND OFF MEANS BYTE-IDENTICAL
//!
//! The flag is read ONCE per process (`OnceLock`). When it is off, [`ArmClock`] takes no
//! `Instant`, the accumulator stays zero, and [`ArmTimings::render_field`] returns the EMPTY
//! STRING — so not one byte of the response differs from a build without this module. That is
//! what lets the cross-impl parity harness (`harness/search_impl_parity.py`, which sets no env)
//! keep comparing the node driver against this one field-for-field: a timing field that appeared
//! unconditionally would be a new key on every arm, and the node driver has none.
//!
//! The numbers are MICROSECONDS, summed over the arms of one `expand_many` batch, and they are a
//! partition of the handler: `sim + chunks + render + core + core_render == total` up to the clock's own
//! resolution. The record they were taken for is
//! `designs/research_state/measurements/expand_many_2026-09-22/README.md`.

use std::sync::OnceLock;
use std::time::Instant;

static ENABLED: OnceLock<bool> = OnceLock::new();

/// Is per-phase timing on? Read from `POKESIM_SEARCH_TIMING` exactly once per process.
///
/// Once per process rather than per call on purpose: a flag that could change mid-run would make
/// one batch's rows a partition and the next batch's rows a subset, and nothing downstream could
/// tell which it was holding.
pub fn timing_enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("POKESIM_SEARCH_TIMING").as_deref() == Ok("1"))
}

/// Micros spent in each phase of `expand_many`, summed over the batch's arms.
#[derive(Default, Clone, Copy)]
pub struct ArmTimings {
    /// Clone the parent node + resolve the joint turn — the ENGINE, the only irreducible row.
    pub sim_us: u64,
    /// The per-side protocol chunk arrays, including their JSON quoting.
    pub chunks_us: u64,
    /// Everything else the handler renders (outcome, requests, choices_used, the child node).
    pub render_us: u64,
    /// The whole handler, arms included — the denominator the other rows partition.
    pub total_us: u64,
    /// The core road (`gen3_core_search_v1`): folding the arm's lines into its versions (the
    /// end-of-turn child and, on a D10 arm, the leaf), from their text.
    pub core_us: u64,
    /// The core road: rendering the leaf's payload.
    pub core_render_us: u64,
}

impl ArmTimings {
    /// The `,"timing_us":{…}` suffix, or **the empty string when timing is off**.
    ///
    /// The leading comma belongs to the field, so the caller appends unconditionally and an
    /// off build renders the historical body byte-for-byte.
    pub fn render_field(&self) -> String {
        if !timing_enabled() {
            return String::new();
        }
        format!(
            ",\"timing_us\":{{\"sim\":{},\"chunks\":{},\"render\":{},\"total\":{},\
             \"core\":{},\"core_render\":{}}}",
            self.sim_us, self.chunks_us, self.render_us, self.total_us, self.core_us,
            self.core_render_us
        )
    }
}

/// A phase stopwatch that costs NOTHING when timing is off (it holds no `Instant`).
pub struct ArmClock(Option<Instant>);

impl ArmClock {
    /// Start a phase.
    pub fn start() -> ArmClock {
        ArmClock(if timing_enabled() { Some(Instant::now()) } else { None })
    }

    /// End the phase, adding its micros to `sink`.
    pub fn stop(self, sink: &mut u64) {
        if let Some(t0) = self.0 {
            *sink += t0.elapsed().as_micros() as u64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The load-bearing negative: OFF must render NOTHING, not a zeroed object.
    ///
    /// A zeroed object would still be a new key on every arm, which is exactly what the
    /// cross-impl parity harness would then have to forgive.
    #[test]
    fn a_disabled_clock_renders_no_field_at_all() {
        // The process-wide flag is unset under `cargo test` (no `POKESIM_SEARCH_TIMING`), so
        // this asserts the shipped default.
        assert!(!timing_enabled(), "POKESIM_SEARCH_TIMING must not be set in the test env");
        let t = ArmTimings::default();
        assert_eq!(t.render_field(), "");
    }

    /// A disabled clock still ACCEPTS the calls — the handler is not branchy at the call sites.
    #[test]
    fn a_disabled_clock_accumulates_nothing_but_does_not_panic() {
        let mut t = ArmTimings::default();
        let c = ArmClock::start();
        c.stop(&mut t.sim_us);
        assert_eq!(t.sim_us, 0);
    }

    /// The field's SHAPE, asserted without the env var, by rendering it by hand — so a rename of
    /// a key is caught here rather than in a Python `KeyError` three layers away.
    #[test]
    fn the_rendered_field_is_the_documented_shape() {
        let t = ArmTimings { sim_us: 1, chunks_us: 3, render_us: 4, total_us: 10, core_us: 5,
                             core_render_us: 6 };
        let rendered = format!(
            ",\"timing_us\":{{\"sim\":{},\"chunks\":{},\"render\":{},\"total\":{},\
             \"core\":{},\"core_render\":{}}}",
            t.sim_us, t.chunks_us, t.render_us, t.total_us, t.core_us, t.core_render_us
        );
        assert_eq!(
            rendered,
            ",\"timing_us\":{\"sim\":1,\"chunks\":3,\"render\":4,\"total\":10,\"core\":5,\
             \"core_render\":6}"
        );
    }
}
