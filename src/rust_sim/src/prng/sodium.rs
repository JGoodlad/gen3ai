//! SodiumRNG — the default backend. A drop-in for libsodium's
//! `randombytes_buf_deterministic`, implemented over a hand-rolled ChaCha20 so
//! the whole thing is dependency-free and bit-auditable (Showdown reaches the
//! same bytes via the `ts-chacha20` package).
//!
//! `next()` encrypts a 36-byte zero buffer (== the first 36 ChaCha20 keystream
//! bytes, counter 0) with the 32-byte seed as key and the fixed nonce
//! `"LibsodiumDRG"`. Bytes `[0,32)` become the next seed; bytes `[32,36)` are
//! the big-endian u32 output.

use super::RngCore;

/// Nonce chosen for libsodium compatibility (12 ASCII bytes).
const NONCE: [u8; 12] = *b"LibsodiumDRG";

pub struct SodiumRng {
    seed: [u8; 32],
}

impl SodiumRng {
    /// Build from the hex string after the `sodium,` prefix. Right-padded with
    /// `'0'` to 32 bytes (64 hex chars) exactly like `padEnd(64, '0')`; extra
    /// hex beyond 32 bytes is ignored (matching the 32-byte write buffer).
    pub fn from_hex(hex: &str) -> Self {
        let mut padded = String::with_capacity(64);
        padded.push_str(hex);
        while padded.len() < 64 {
            padded.push('0');
        }
        let bytes = &padded.as_bytes()[..64];
        let mut seed = [0u8; 32];
        for (i, b) in seed.iter_mut().enumerate() {
            let pair = std::str::from_utf8(&bytes[i * 2..i * 2 + 2]).expect("ascii hex");
            *b = u8::from_str_radix(pair, 16).expect("sodium hex seed");
        }
        SodiumRng { seed }
    }
}

impl RngCore for SodiumRng {
    fn next_u32(&mut self) -> u32 {
        let block = chacha20_block(&self.seed, 0, &NONCE);
        self.seed.copy_from_slice(&block[0..32]);
        u32::from_be_bytes([block[32], block[33], block[34], block[35]])
    }

    fn get_seed(&self) -> String {
        let mut s = String::with_capacity(7 + 64);
        s.push_str("sodium,");
        for b in &self.seed {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }
}

// --- ChaCha20 block function (RFC 8439 IETF variant: 96-bit nonce, 32-bit counter) ---

#[inline]
fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] = (s[d] ^ s[a]).rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_left(7);
}

/// 64 keystream bytes for `(key, counter, nonce)`.
fn chacha20_block(key: &[u8; 32], counter: u32, nonce: &[u8; 12]) -> [u8; 64] {
    let mut state = [0u32; 16];
    // "expand 32-byte k"
    state[0] = 0x6170_7865;
    state[1] = 0x3320_646e;
    state[2] = 0x7962_2d32;
    state[3] = 0x6b20_6574;
    for i in 0..8 {
        state[4 + i] =
            u32::from_le_bytes([key[i * 4], key[i * 4 + 1], key[i * 4 + 2], key[i * 4 + 3]]);
    }
    state[12] = counter;
    for i in 0..3 {
        state[13 + i] = u32::from_le_bytes([
            nonce[i * 4],
            nonce[i * 4 + 1],
            nonce[i * 4 + 2],
            nonce[i * 4 + 3],
        ]);
    }

    let mut w = state;
    for _ in 0..10 {
        // 20 rounds = 10 double-rounds
        quarter_round(&mut w, 0, 4, 8, 12);
        quarter_round(&mut w, 1, 5, 9, 13);
        quarter_round(&mut w, 2, 6, 10, 14);
        quarter_round(&mut w, 3, 7, 11, 15);
        quarter_round(&mut w, 0, 5, 10, 15);
        quarter_round(&mut w, 1, 6, 11, 12);
        quarter_round(&mut w, 2, 7, 8, 13);
        quarter_round(&mut w, 3, 4, 9, 14);
    }

    let mut out = [0u8; 64];
    for i in 0..16 {
        let v = w[i].wrapping_add(state[i]);
        out[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    out
}
