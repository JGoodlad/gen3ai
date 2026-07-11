// prng_reference.js — a FROM-SCRATCH, dependency-free reimplementation of
// Pokémon Showdown's PRNG (deps/pokemon-showdown/sim/prng.ts).
//
// WHY THIS FILE EXISTS
// --------------------
// The Rust port in ../src/prng/ must be bit-for-bit identical to Showdown's
// PRNG. Showdown's SodiumRNG drives ChaCha20 through the `ts-chacha20`
// dependency, so we cannot read the algorithm off the TypeScript alone. This
// file re-derives the WHOLE thing (ChaCha20 block function included) with zero
// dependencies. `gen_prng_vectors.js` runs this side-by-side with the REAL
// prng.js and asserts they agree on every value before emitting golden vectors.
//
// So this file is two things at once:
//   1. The executable SPEC the Rust code mirrors line-for-line.
//   2. A self-check: if it ever diverges from the real lib, vector generation
//      fails loudly instead of baking a wrong "golden" answer.
//
// Keep this in lockstep with ../src/prng/. A change here without a change there
// (or vice-versa) is a bug.

'use strict';

// ----------------------------------------------------------------------------
// ChaCha20 block function (RFC 8439 / IETF variant: 32-bit counter, 96-bit nonce)
// ----------------------------------------------------------------------------
// libsodium's randombytes_buf_deterministic (which SodiumRNG emulates) is
// crypto_stream_chacha20_ietf keystream with counter starting at 0. The nonce
// is the 12 ASCII bytes "LibsodiumDRG".

function rotl32(x, n) {
	return ((x << n) | (x >>> (32 - n))) >>> 0;
}

function readLE32(bytes, off) {
	return (bytes[off] | (bytes[off + 1] << 8) | (bytes[off + 2] << 16) | (bytes[off + 3] << 24)) >>> 0;
}

function writeLE32(bytes, off, v) {
	bytes[off] = v & 0xff;
	bytes[off + 1] = (v >>> 8) & 0xff;
	bytes[off + 2] = (v >>> 16) & 0xff;
	bytes[off + 3] = (v >>> 24) & 0xff;
}

function quarterRound(s, a, b, c, d) {
	s[a] = (s[a] + s[b]) >>> 0; s[d] = rotl32(s[d] ^ s[a], 16);
	s[c] = (s[c] + s[d]) >>> 0; s[b] = rotl32(s[b] ^ s[c], 12);
	s[a] = (s[a] + s[b]) >>> 0; s[d] = rotl32(s[d] ^ s[a], 8);
	s[c] = (s[c] + s[d]) >>> 0; s[b] = rotl32(s[b] ^ s[c], 7);
}

// Returns 64 keystream bytes for the given (32-byte key, u32 counter, 12-byte nonce).
function chacha20Block(key, counter, nonce) {
	const state = new Uint32Array(16);
	// "expand 32-byte k"
	state[0] = 0x61707865; state[1] = 0x3320646e; state[2] = 0x79622d32; state[3] = 0x6b206574;
	for (let i = 0; i < 8; i++) state[4 + i] = readLE32(key, i * 4);
	state[12] = counter >>> 0;
	for (let i = 0; i < 3; i++) state[13 + i] = readLE32(nonce, i * 4);

	const w = state.slice();
	for (let i = 0; i < 10; i++) { // 20 rounds = 10 double-rounds
		quarterRound(w, 0, 4, 8, 12);
		quarterRound(w, 1, 5, 9, 13);
		quarterRound(w, 2, 6, 10, 14);
		quarterRound(w, 3, 7, 11, 15);
		quarterRound(w, 0, 5, 10, 15);
		quarterRound(w, 1, 6, 11, 12);
		quarterRound(w, 2, 7, 8, 13);
		quarterRound(w, 3, 4, 9, 14);
	}
	const out = new Uint8Array(64);
	for (let i = 0; i < 16; i++) writeLE32(out, i * 4, (w[i] + state[i]) >>> 0);
	return out;
}

const SODIUM_NONCE = Uint8Array.from([...'LibsodiumDRG'].map(c => c.charCodeAt(0)));

// ----------------------------------------------------------------------------
// SodiumRNG — mirrors sim/prng.ts SodiumRNG
// ----------------------------------------------------------------------------
class SodiumRNG {
	constructor(hex) {
		// seed[1] hex, right-padded with '0' to 64 chars (32 bytes).
		const padded = hex.padEnd(64, '0');
		const seed = new Uint8Array(32);
		for (let i = 0; i < 32; i++) seed[i] = parseInt(padded.slice(i * 2, i * 2 + 2), 16);
		this.seed = seed;
	}
	getSeed() {
		let hex = '';
		for (const b of this.seed) hex += b.toString(16).padStart(2, '0');
		return `sodium,${hex}`;
	}
	next() {
		// Encrypt a 36-byte zero buffer == first 36 keystream bytes (counter 0).
		const buf = chacha20Block(this.seed, 0, SODIUM_NONCE).slice(0, 36);
		this.seed = buf.slice(0, 32);
		// big-endian read of bytes [32,36)
		return ((buf[32] * 256 + buf[33]) * 256 + buf[34]) * 256 + buf[35];
	}
}

// ----------------------------------------------------------------------------
// Gen5RNG — mirrors sim/prng.ts Gen5RNG (64-bit LCG over four 16-bit words)
// ----------------------------------------------------------------------------
class Gen5RNG {
	constructor(seed4) { this.seed = [...seed4]; }
	getSeed() { return this.seed.join(','); }
	multiplyAdd(a, b, c) {
		const out = [0, 0, 0, 0];
		let carry = 0;
		for (let outIndex = 3; outIndex >= 0; outIndex--) {
			for (let bIndex = outIndex; bIndex < 4; bIndex++) {
				const aIndex = 3 - (bIndex - outIndex);
				carry += a[aIndex] * b[bIndex];
			}
			carry += c[outIndex];
			out[outIndex] = carry & 0xffff;
			carry = Math.floor(carry / 65536); // >>>16 but carry can exceed 32 bits
		}
		return out;
	}
	nextFrame(seed) {
		const a = [0x5d58, 0x8b65, 0x6c07, 0x8965];
		const c = [0, 0, 0x26, 0x9ec3];
		return this.multiplyAdd(seed, a, c);
	}
	next() {
		this.seed = this.nextFrame(this.seed);
		return (((this.seed[0] << 16) >>> 0) + this.seed[1]) >>> 0;
	}
}

// NOTE on multiplyAdd carry: prng.ts uses `carry >>>= 16`, which truncates carry
// to 32 bits first. The cross-products a[i]*b[j] sum to well under 2^32 per
// column here (four 16-bit*16-bit terms ~= 2^34 max... ) — Showdown relies on
// JS doubles holding the full carry, and `>>>16` only works because the running
// carry stays < 2^32 at the point of the shift. We use Math.floor(carry/65536)
// which is identical for the in-range values and avoids the 32-bit-truncation
// trap. gen_prng_vectors.js verifies this against the real lib.

// ----------------------------------------------------------------------------
// PRNG — high-level API, mirrors sim/prng.ts PRNG
// ----------------------------------------------------------------------------
class PRNG {
	constructor(seed) {
		this.startingSeed = seed;
		if (seed.startsWith('sodium,')) {
			this.rng = new SodiumRNG(seed.split(',')[1]);
		} else if (seed.startsWith('gen5,')) {
			const s = [seed.slice(5, 9), seed.slice(9, 13), seed.slice(13, 17), seed.slice(17, 21)];
			this.rng = new Gen5RNG(s.map(n => parseInt(n, 16)));
		} else if (/[0-9]/.test(seed.charAt(0))) {
			this.rng = new Gen5RNG(seed.split(',').map(Number));
		} else {
			throw new Error(`Unrecognized RNG seed ${seed}`);
		}
	}
	getSeed() { return this.rng.getSeed(); }

	random(from, to) {
		const result = this.rng.next();
		if (from) from = Math.floor(from);
		if (to) to = Math.floor(to);
		if (from === undefined) {
			return result / 2 ** 32;
		} else if (!to) {
			return Math.floor(result * from / 2 ** 32);
		} else {
			return Math.floor(result * (to - from) / 2 ** 32) + from;
		}
	}
	randomChance(numerator, denominator) {
		return this.random(denominator) < numerator;
	}
	sample(items) {
		if (items.length === 0) throw new RangeError('Cannot sample an empty array');
		return items[this.random(items.length)];
	}
	shuffle(items, start = 0, end = items.length) {
		while (start < end - 1) {
			const nextIndex = this.random(start, end);
			if (start !== nextIndex) {
				[items[start], items[nextIndex]] = [items[nextIndex], items[start]];
			}
			start++;
		}
		return items;
	}
}

module.exports = { PRNG, SodiumRNG, Gen5RNG, chacha20Block, SODIUM_NONCE };
