//! PRNG — bit-for-bit port of `deps/pokemon-showdown/sim/prng.ts`.
//!
//! Showdown records a battle as `(seed + teams + choice log)` and replays it to
//! get the SAME result. That guarantee is the foundation everything else in this
//! crate stands on, and the foundation our existing tooling already assumes
//! (reconstruction, counterfactual re-roll, clone-and-branch search). So this is
//! the first thing ported and the most heavily pinned: `tests/prng_golden.rs`
//! checks every public method against ~2900 vectors captured from the REAL
//! `prng.js` (regenerate via `harness/gen_prng_vectors.js`).
//!
//! Two backends, chosen by seed prefix, exactly like `PRNG.setSeed`:
//!   - `sodium,<hex>`           -> [`sodium::SodiumRng`] (ChaCha20; the DEFAULT)
//!   - `gen5,<hex16>` / `m,n,o,p` (decimal) -> [`gen5::Gen5Rng`] (64-bit LCG)
//!
//! The high-level API (`random*`, `sample`, `shuffle`) reproduces prng.ts's
//! float arithmetic with `f64`, which is exact for the small ranges Showdown
//! actually uses (damage roll of 16, accuracy of 100, team sizes, etc.).

pub mod gen5;
pub mod sodium;

use gen5::Gen5Rng;
use sodium::SodiumRng;

/// A seed / serialized-state string, e.g. `"sodium,deadbeef…"` or `"1,2,3,4"`.
/// This is exactly what `PRNG.getSeed()` returns, so any seed round-trips
/// through [`Prng::new`] — the property the differential harness exploits.
pub type PrngSeed = String;

const TWO32: f64 = 4_294_967_296.0; // 2^32

/// `POKESIM_PRNG_TRACE=1` → [`Prng::next`] logs every draw to stderr. Read ONCE per
/// process (a `OnceLock`), so the per-draw cost when off is a single relaxed load.
static TRACE_ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
/// Monotonic draw counter for the trace (process-global, matching the sim probe's
/// per-battle numbering when one battle is replayed per process — which is how
/// `ab_replay <repro-dir>` runs).
static TRACE_DRAW_N: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(0);

#[inline]
fn trace_enabled() -> bool {
    *TRACE_ON.get_or_init(|| std::env::var("POKESIM_PRNG_TRACE").map(|v| v == "1").unwrap_or(false))
}

/// Shared low-level RNG interface (`interface RNG` in prng.ts).
pub(crate) trait RngCore {
    /// A raw 32-bit random number (`next()`).
    fn next_u32(&mut self) -> u32;
    /// The current seed string (`getSeed()`).
    fn get_seed(&self) -> String;
}

enum Backend {
    Sodium(SodiumRng),
    Gen5(Gen5Rng),
}

/// High-level PRNG. Mirrors `class PRNG`.
pub struct Prng {
    backend: Backend,
}

impl Prng {
    /// Build a PRNG for a seed string, dispatching on prefix like `setSeed`.
    ///
    /// Panics on an unrecognized seed (matching prng.ts, which throws).
    pub fn new(seed: &str) -> Self {
        let backend = if let Some(hex) = seed.strip_prefix("sodium,") {
            Backend::Sodium(SodiumRng::from_hex(hex))
        } else if let Some(rest) = seed.strip_prefix("gen5,") {
            // four big-endian 16-bit words as 4-hex-digit groups
            let word = |a: usize, b: usize| {
                u16::from_str_radix(&rest[a..b], 16).expect("gen5 hex seed word")
            };
            Backend::Gen5(Gen5Rng::new([word(0, 4), word(4, 8), word(8, 12), word(12, 16)]))
        } else if seed.as_bytes().first().map_or(false, |c| c.is_ascii_digit()) {
            let parts: Vec<u16> = seed
                .split(',')
                .map(|p| p.parse().expect("gen5 decimal seed word"))
                .collect();
            assert_eq!(parts.len(), 4, "gen5 seed needs exactly 4 words");
            Backend::Gen5(Gen5Rng::new([parts[0], parts[1], parts[2], parts[3]]))
        } else {
            panic!("Unrecognized RNG seed {seed}");
        };
        Prng { backend }
    }

    /// Raw 32-bit draw (`rng.next()`).
    ///
    /// When `POKESIM_PRNG_TRACE=1` this also prints ONE stderr line per draw:
    /// `[prng] #<n> -> <value> seed_before=<seed>`. That is the port-side half of the
    /// draw-divergence workflow whose sim-side half is
    /// `harness/probe_repro_simtrace.js` (which prints the sim's draws WITH call
    /// sites) — line up the two by draw index to find the exact draw where a
    /// `kind=seed` repro desyncs, instead of guessing at the mechanism.
    ///
    /// Made PERMANENT (`gen3_prng_trace_env_v1`) because three separate rounds have
    /// hand-patched this same hook into `next()` and then removed it. COST WHEN OFF:
    /// one relaxed atomic load of a `OnceLock<bool>` — the env var is read exactly
    /// once per process, never per draw, so the bridge/training hot path is unaffected.
    pub fn next(&mut self) -> u32 {
        if trace_enabled() {
            let before = self.get_seed();
            let v = self.next_raw();
            let n = TRACE_DRAW_N.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
            eprintln!("[prng] #{n} -> {v} seed_before={before}");
            return v;
        }
        self.next_raw()
    }

    #[inline]
    fn next_raw(&mut self) -> u32 {
        match &mut self.backend {
            Backend::Sodium(r) => r.next_u32(),
            Backend::Gen5(r) => r.next_u32(),
        }
    }

    /// Current seed string (`getSeed()`).
    pub fn get_seed(&self) -> PrngSeed {
        match &self.backend {
            Backend::Sodium(r) => r.get_seed(),
            Backend::Gen5(r) => r.get_seed(),
        }
    }

    /// `random()` — a real number in `[0, 1)`, like `Math.random()`.
    pub fn random_float(&mut self) -> f64 {
        self.next() as f64 / TWO32
    }

    /// `random(n)` — an integer in `[0, n)`.
    pub fn random_below(&mut self, n: u32) -> u32 {
        (self.next() as f64 * n as f64 / TWO32).floor() as u32
    }

    /// `random(m, n)` — an integer in `[m, n)`.
    pub fn random_range(&mut self, from: u32, to: u32) -> u32 {
        (self.next() as f64 * (to - from) as f64 / TWO32).floor() as u32 + from
    }

    /// `randomChance(numerator, denominator)` — true with prob `num/den`.
    pub fn random_chance(&mut self, numerator: u32, denominator: u32) -> bool {
        self.random_below(denominator) < numerator
    }

    /// `sample(items)` — a uniformly random element. Panics on an empty slice.
    pub fn sample<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        assert!(!items.is_empty(), "cannot sample an empty slice");
        &items[self.random_below(items.len() as u32) as usize]
    }

    /// `shuffle(items)` — Fisher-Yates over the whole slice. This is how the
    /// game resolves speed ties, so it MUST consume the RNG in the same order
    /// and count as Showdown.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        let end = items.len();
        self.shuffle_range(items, 0, end);
    }

    /// `shuffle(items, start, end)` — Fisher-Yates over `[start, end)`.
    pub fn shuffle_range<T>(&mut self, items: &mut [T], mut start: usize, end: usize) {
        while start + 1 < end {
            let next_index = self.random_range(start as u32, end as u32) as usize;
            if start != next_index {
                items.swap(start, next_index);
            }
            start += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Smoke vectors (independent of the golden corpus) so a broken build fails
    // fast even before the full differential test runs.
    #[test]
    fn sodium_first_draw_matches_showdown() {
        let mut p = Prng::new("sodium,deadbeef");
        assert_eq!(p.next(), 3_406_896_987);
        assert_eq!(
            p.get_seed(),
            "sodium,cc282df82625252cb0e7fdc7463d75852267f0a45737f921948d0798debda58b"
        );
    }

    #[test]
    fn gen5_decimal_roundtrips() {
        let mut p = Prng::new("1,2,3,4");
        let _ = p.next();
        // getSeed stays in the 4-word decimal form
        assert_eq!(p.get_seed().split(',').count(), 4);
    }
}
