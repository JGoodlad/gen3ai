//! Gen5RNG — the legacy backend that emulates the on-cartridge Gen 5 PRNG: a
//! 64-bit Linear Congruential Generator `x' = (a*x + c) mod 2^64`, stored as
//! four big-endian 16-bit words. Output is the upper 32 bits of the new state.
//!
//! Our seeds appear in decimal (`"1,2,3,4"`) or `gen5,<16 hex digits>` form; both
//! land here via [`super::Prng::new`]. `getSeed()` always reports the decimal
//! 4-word form, matching prng.ts.

use super::RngCore;

const A: [u16; 4] = [0x5d58, 0x8b65, 0x6c07, 0x8965]; // 0x5D588B656C078965
const C: [u16; 4] = [0, 0, 0x26, 0x9ec3]; // 0x00269EC3

pub struct Gen5Rng {
    seed: [u16; 4],
}

impl Gen5Rng {
    pub fn new(seed: [u16; 4]) -> Self {
        Gen5Rng { seed }
    }

    /// `a * b + c` over four 16-bit words (long multiplication with carry).
    /// prng.ts keeps the running carry in a JS double; we use `u64`, which holds
    /// it exactly and sidesteps the `>>> 16` 32-bit-truncation subtlety.
    fn multiply_add(a: [u16; 4], b: [u16; 4], c: [u16; 4]) -> [u16; 4] {
        let mut out = [0u16; 4];
        let mut carry: u64 = 0;
        for out_index in (0..4).rev() {
            for b_index in out_index..4 {
                let a_index = 3 - (b_index - out_index);
                carry += a[a_index] as u64 * b[b_index] as u64;
            }
            carry += c[out_index] as u64;
            out[out_index] = (carry & 0xffff) as u16;
            carry >>= 16;
        }
        out
    }

    fn next_frame(seed: [u16; 4]) -> [u16; 4] {
        Gen5Rng::multiply_add(seed, A, C)
    }
}

impl RngCore for Gen5Rng {
    fn next_u32(&mut self) -> u32 {
        self.seed = Gen5Rng::next_frame(self.seed); // advance
        ((self.seed[0] as u32) << 16) | (self.seed[1] as u32) // upper 32 bits
    }

    fn get_seed(&self) -> String {
        format!(
            "{},{},{},{}",
            self.seed[0], self.seed[1], self.seed[2], self.seed[3]
        )
    }
}
