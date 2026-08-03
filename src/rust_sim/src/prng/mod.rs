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

/// Normalize a `>start` seed into a [`Prng::new`]-acceptable string. Accepts:
/// - a bracketed JSON array `[1,2,3,4]` (the form `>start {"seed":[..]}` carries)
///   → `1,2,3,4`;
/// - any already-valid seed string (`sodium,…`, `gen5,…`, `1,2,3,4`) → trimmed verbatim.
///
/// Lives here (not in `state`) so the bridge's START parser and `BattleState::start`
/// share ONE definition of the accepted spelling — the whole
/// `gen3_bridge_seedless_fixed_seed_v1` bug class was two parsers disagreeing about
/// which forms exist.
pub fn normalize_seed(s: &str) -> String {
    let t = s.trim();
    if let Some(inner) = t.strip_prefix('[').and_then(|x| x.strip_suffix(']')) {
        // `[1, 2, 3, 4]` → `1,2,3,4`
        inner.split(',').map(str::trim).collect::<Vec<_>>().join(",")
    } else {
        t.to_string()
    }
}

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
    /// Panics on an unrecognized seed (matching prng.ts, which throws). Callers that
    /// hold a seed from OUTSIDE the process (the bridge's `START` JSON) should use
    /// [`Prng::validate_seed`] first and report a clean error instead — a seed that is
    /// present but unusable must never fall back to some other stream.
    pub fn new(seed: &str) -> Self {
        match Self::try_new(seed) {
            Ok(p) => p,
            Err(e) => panic!("{e}"),
        }
    }

    /// The fallible form of [`Prng::new`] — the SINGLE definition of "an acceptable seed".
    /// Mirrors `PRNG.setSeed`'s three accepted forms and its `throw` on anything else.
    pub fn try_new(seed: &str) -> Result<Self, String> {
        let backend = if let Some(hex) = seed.strip_prefix("sodium,") {
            if hex.is_empty() || !hex.bytes().all(|c| c.is_ascii_hexdigit()) {
                return Err(format!("Unrecognized RNG seed {seed}"));
            }
            Backend::Sodium(SodiumRng::from_hex(hex))
        } else if let Some(rest) = seed.strip_prefix("gen5,") {
            // four big-endian 16-bit words as 4-hex-digit groups
            if rest.len() < 16 || !rest.bytes().take(16).all(|c| c.is_ascii_hexdigit()) {
                return Err(format!("Unrecognized RNG seed {seed}"));
            }
            let word = |a: usize, b: usize| u16::from_str_radix(&rest[a..b], 16).unwrap();
            Backend::Gen5(Gen5Rng::new([word(0, 4), word(4, 8), word(8, 12), word(12, 16)]))
        } else if seed.as_bytes().first().is_some_and(|c| c.is_ascii_digit()) {
            let parts: Vec<u16> = seed
                .split(',')
                .map(|p| p.trim().parse::<u16>().map_err(|_| format!("Unrecognized RNG seed {seed}")))
                .collect::<Result<_, _>>()?;
            if parts.len() != 4 {
                return Err(format!("gen5 seed needs exactly 4 words: {seed}"));
            }
            Backend::Gen5(Gen5Rng::new([parts[0], parts[1], parts[2], parts[3]]))
        } else {
            return Err(format!("Unrecognized RNG seed {seed}"));
        };
        Ok(Prng { backend })
    }

    /// `Ok(())` iff `seed` is a form [`Prng::new`] accepts. The consumer-side throwing
    /// guard for any seed that crossed a process boundary.
    pub fn validate_seed(seed: &str) -> Result<(), String> {
        Self::try_new(seed).map(|_| ())
    }

    /// `PRNG.generateSeed()` — a FRESH random seed in the sim's own default form,
    /// `sodium,<32 hex>` (128 bits, right-padded to the 256-bit ChaCha20 key by
    /// [`SodiumRng::from_hex`], exactly like `SodiumRNG.setSeed`'s `padEnd(64,"0")`).
    ///
    /// This is what Showdown does when `>start` carries no `seed` (`PRNG`'s constructor:
    /// `if (!seed) seed = PRNG.generateSeed()`), so a seedless battle is a NEW battle.
    /// The port must mint too: a constant fallback makes every seedless battle replay ONE
    /// dice stream, which is exactly the `gen3_bridge_seedless_fixed_seed_v1` bug.
    ///
    /// Entropy comes from `/dev/urandom`; if that cannot be read (a sandbox with no
    /// `/dev`), it degrades to a time ⊕ pid ⊕ address mix — still per-call distinct,
    /// which is the property that matters here. NOT called anywhere on the deterministic
    /// battle path: only when a caller supplied no seed at all.
    pub fn generate_seed() -> PrngSeed {
        let mut bytes = [0u8; 16];
        let ok = std::fs::File::open("/dev/urandom")
            .and_then(|mut f| {
                use std::io::Read;
                f.read_exact(&mut bytes)
            })
            .is_ok();
        if !ok {
            // Fallback mix: nanosecond clock ⊕ pid ⊕ a stack address (ASLR).
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0);
            let pid = std::process::id() as u64;
            let addr = &bytes as *const _ as u64;
            let mut x = nanos ^ (pid << 32) ^ addr;
            for chunk in bytes.chunks_mut(8) {
                // SplitMix64 — a decent avalanche over a weak seed.
                x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = x;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                chunk.copy_from_slice(&z.to_be_bytes()[..chunk.len()]);
            }
        }
        let mut s = String::with_capacity(7 + 32);
        s.push_str("sodium,");
        for b in bytes {
            s.push_str(&format!("{b:02x}"));
        }
        s
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

    /// `gen3_bridge_seed_forms_v1` — `validate_seed` must accept exactly the set
    /// `PRNG.setSeed` accepts, and reject everything else WITHOUT panicking (the bridge
    /// turns the `Err` into a clean `__ERR__` naming the offending field).
    #[test]
    fn validate_seed_matches_showdown_accept_set() {
        for good in [
            "1,2,3,4",
            "0,0,0,0",
            "65535,65535,65535,65535",
            "gen5,0001000200030004",
            "sodium,deadbeef",
            "sodium,cc282df82625252cb0e7fdc7463d75852267f0a45737f921948d0798debda58b",
        ] {
            assert!(Prng::validate_seed(good).is_ok(), "rejected a valid seed: {good}");
        }
        for bad in [
            "", "not-a-seed", "1,2,3", "1,2,3,4,5", "1,2,x,4", "99999,1,1,1",
            "gen5,zz", "gen5,", "sodium,", "sodium,xyz", "[1,2,3,4]",
        ] {
            assert!(Prng::validate_seed(bad).is_err(), "accepted an invalid seed: {bad:?}");
        }
        // The bracketed array spelling is normalized FIRST, then validated.
        assert!(Prng::validate_seed(&normalize_seed("[1, 2, 3, 4]")).is_ok());
    }

    /// `gen3_bridge_seedless_fixed_seed_v1` — a minted seed must be FRESH per call and
    /// in the sim's own default `sodium,<32 hex>` form, not some constant.
    #[test]
    fn generate_seed_is_fresh_and_well_formed() {
        let a = Prng::generate_seed();
        let b = Prng::generate_seed();
        assert_ne!(a, b, "generate_seed returned the same seed twice");
        for s in [&a, &b] {
            assert!(s.starts_with("sodium,"), "not the sim's default form: {s}");
            assert_eq!(s.len(), "sodium,".len() + 32, "expected 128 bits of hex: {s}");
            assert!(Prng::validate_seed(s).is_ok(), "minted an unusable seed: {s}");
        }
        // ...and it actually seeds a distinct stream.
        assert_ne!(Prng::new(&a).next(), Prng::new(&b).next());
    }

    #[test]
    fn gen5_decimal_roundtrips() {
        let mut p = Prng::new("1,2,3,4");
        let _ = p.next();
        // getSeed stays in the 4-word decimal form
        assert_eq!(p.get_seed().split(',').count(), 4);
    }
}
