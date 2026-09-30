//! CPython's `random.Random` — MT19937 with CPython's seeding and draw primitives, word for word
//! (M5 Lane F).
//!
//! THE DECISION (program §2 M5, Lane F): a bot's randomness is reproduced as THE SAME STREAM, not
//! passed in as explicit draws. A seeded Python bot and the Rust bot are then interchangeable draw
//! for draw — which is what lets a bot battle be the same battle on both paths (Lane H's "same seed
//! set on both paths") — and the gate checks the draw COUNT per decision (in 32-bit words), so a bot
//! that consumes one draw too many or too few fails even when its action happens to agree.
//!
//! Mirrors CPython 3.11 `Modules/_randommodule.c` + `Lib/random.py`: `seed(int)` (`init_by_array`
//! over the 32-bit little-endian words of `abs(n)`), `random()` (`genrand_res53`), `getrandbits(k)`
//! for `k <= 32`, `_randbelow_with_getrandbits(n)` and `choice`.

const N: usize = 624;
const M: usize = 397;

/// See the module docs.
#[derive(Clone)]
pub struct PyRandom {
    mt: [u32; N],
    idx: usize,
    /// 32-bit outputs consumed since seeding — the stream OFFSET the corpus records.
    pub words: u64,
}

impl PyRandom {
    /// `random.Random(seed)` for a non-negative int seed.
    pub fn new(seed: u64) -> PyRandom {
        let mut key: Vec<u32> = Vec::new();
        let mut n = seed;
        while n > 0 {
            key.push(n as u32);
            n >>= 32;
        }
        if key.is_empty() {
            key.push(0);
        }
        let mut r = PyRandom { mt: [0; N], idx: N + 1, words: 0 };
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            self.mt[i] = 1812433253u32.wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        self.idx = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let (mut i, mut j) = (1usize, 0usize);
        let mut k = N.max(key.len());
        while k > 0 {
            self.mt[i] = (self.mt[i] ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1664525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            self.mt[i] = (self.mt[i] ^ (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
    }

    /// `genrand_uint32`.
    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= N {
            for kk in 0..N {
                let y = (self.mt[kk] & 0x8000_0000) | (self.mt[(kk + 1) % N] & 0x7fff_ffff);
                let mag = if y & 1 == 1 { 0x9908_b0df } else { 0 };
                self.mt[kk] = self.mt[(kk + M) % N] ^ (y >> 1) ^ mag;
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        self.words += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `random()` — `genrand_res53`, two words.
    pub fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// `getrandbits(k)` for `1 <= k <= 32` (one word; `k = 0` is 0 and consumes nothing).
    pub fn getrandbits(&mut self, k: u32) -> u32 {
        assert!(k <= 32, "getrandbits({k}): only k <= 32 is ported (no bot draws more)");
        if k == 0 {
            return 0;
        }
        self.next_u32() >> (32 - k)
    }

    /// `_randbelow_with_getrandbits(n)`.
    pub fn randbelow(&mut self, n: usize) -> usize {
        let k = usize::BITS - n.leading_zeros();
        let mut r = self.getrandbits(k) as usize;
        while r >= n {
            r = self.getrandbits(k) as usize;
        }
        r
    }

    /// `choice(seq)` as an index into a sequence of length `len` (`None` = Python's `IndexError`).
    pub fn choice(&mut self, len: usize) -> Option<usize> {
        (len > 0).then(|| self.randbelow(len))
    }

    /// Advance the stream to absolute offset `words` (the corpus's recorded position); refuses to
    /// go backwards.
    pub fn skip_to(&mut self, words: u64) -> Result<(), String> {
        if words < self.words {
            return Err(format!("rng: cannot rewind from word {} to {words}", self.words));
        }
        while self.words < words {
            self.next_u32();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::PyRandom;

    // Golden values from CPython 3.11: `random.Random(s)` — three `random()` bit patterns, then a
    // fresh stream's `choice(range(n))` for n in (1, 2, 3, 5, 6, 7, 11), then a fresh stream's
    // `getrandbits(k)` for k in (1, 3, 32). The last seed is `bot_corpus.stream_seed(70000, "staller", "protect")`.
    const GOLDEN: &[(u64, [u64; 3], [usize; 7], [u32; 3])] = &[
        (0, [0x3feb0580f98a7dbe, 0x3fe84129978f9c1a, 0x3fdaeaa51052e978], [0, 1, 0, 2, 4, 3, 6], [1, 3, 3255389356]),
        (1, [0x3fc132d8f91b7584, 0x3feb1e2d5b3584f8, 0x3fe870d778409f13], [0, 0, 1, 0, 3, 6, 7], [0, 4, 3639700191]),
        (12345, [0x3fdaa9e665dc8a18, 0x3f84d392d1f6f840, 0x3fea68177b361de3], [0, 0, 1, 2, 1, 2, 9], [0, 5, 43676229]),
        (1099511627783, [0x3fe3a3761afd4291, 0x3fea13cb57c0715a, 0x3fee3d88c68edf49], [0, 1, 2, 1, 4, 3, 8], [1, 5, 3500038837]),
        (70006220523, [0x3fe4c9cbb15f8cb4, 0x3fe0be673d6f7e41, 0x3fb1e06f4dc1cb30], [0, 1, 0, 3, 5, 6, 10], [1, 2, 2247309811]),
    ];

    #[test]
    fn the_stream_is_cpythons() {
        for (seed, rand, choices, bits) in GOLDEN {
            let mut r = PyRandom::new(*seed);
            for want in rand {
                assert_eq!(r.random().to_bits(), *want, "seed {seed}");
            }
            assert_eq!(r.words, 6);
            let mut r = PyRandom::new(*seed);
            let got: Vec<usize> = [1, 2, 3, 5, 6, 7, 11].iter().map(|n| r.choice(*n).unwrap()).collect();
            assert_eq!(got, choices.to_vec(), "seed {seed}");
            let mut r = PyRandom::new(*seed);
            let got: Vec<u32> = [1, 3, 32].iter().map(|k| r.getrandbits(*k)).collect();
            assert_eq!(got, bits.to_vec(), "seed {seed}");
        }
    }

    #[test]
    fn skip_to_lands_on_the_same_stream() {
        let mut a = PyRandom::new(99);
        for _ in 0..7 {
            a.random();
        }
        let mut b = PyRandom::new(99);
        b.skip_to(14).unwrap();
        assert_eq!(a.random().to_bits(), b.random().to_bits());
        assert!(b.skip_to(3).is_err());
    }
}
