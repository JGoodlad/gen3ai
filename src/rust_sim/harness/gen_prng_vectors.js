// gen_prng_vectors.js — PRNG differential fuzz harness (level 1: the RNG).
//
// Loads the REAL Pokémon Showdown PRNG (deps/pokemon-showdown/dist/sim/prng.js)
// and emits SELF-CONTAINED golden assertions. Before writing anything, it
// cross-checks our dependency-free reference (prng_reference.js) against the
// real lib value-by-value; on ANY divergence it aborts nonzero so we never bake
// a "golden" answer the reference (hence the Rust port) couldn't reproduce.
//
// Each output line is an INDEPENDENT assertion keyed on a pre-state seed string,
// so the Rust golden test (tests/prng_golden.rs) can reconstruct a PRNG from
// that seed and check one call — no JSON parser, no shared call-order coupling:
//
//   NEXT    <pre_seed> <u32 value> <post_seed>      # rng.next(): value + new getSeed()
//   RANDN   <pre_seed> <n> <int>                    # random(n)
//   RANDMN  <pre_seed> <m> <n> <int>                # random(m, n)
//   CHANCE  <pre_seed> <num> <den> <0|1>            # randomChance(num, den)
//   SHUFFLE <pre_seed> <len> <csv>                  # shuffle([0..len])
//
// Any getSeed() string is a valid constructor arg for the same backend, so every
// line round-trips. Output: ../tests/vectors/prng_golden.txt
//
// Run:  node src/rust_sim/harness/gen_prng_vectors.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { PRNG: RealPRNG } = require(path.join(PS, 'dist/sim/prng.js'));
const { PRNG: RefPRNG } = require('./prng_reference.js');

const OUT = path.resolve(__dirname, '../tests/vectors/prng_golden.txt');

const SEEDS = [
	'sodium,deadbeef',
	'sodium,0123456789abcdef0123456789abcdef',
	'sodium,00000000000000000000000000000000',
	'sodium,ffffffffffffffffffffffffffffffff',
	'sodium,1',
	'gen5,00000000000000000000',
	'gen5,123456789abcdef01234',
	'1,2,3,4',
	'9999,8888,7777,6666',
];

const N_NEXT = 64;          // next() draws (also seeds the pool of pre-states)
const NS = [2, 16, 100, 256, 6, 24];
const MNS = [[0, 1], [1, 5], [2, 14], [0, 2], [3, 10]];
const CHANCES = [[1, 256], [1, 5], [3, 4], [85, 100]];
const LENS = [2, 3, 6, 12];

let mismatches = 0;
function check(label, a, b) {
	if (JSON.stringify(a) !== JSON.stringify(b)) {
		if (++mismatches <= 20) {
			console.error(`MISMATCH [${label}]\n  real= ${JSON.stringify(a)}\n  ref = ${JSON.stringify(b)}`);
		}
	}
}

const lines = [];

for (const seed of SEEDS) {
	// next() chain — also collects the pool of pre-state seeds we reuse below.
	const real = new RealPRNG(seed), ref = new RefPRNG(seed);
	const preStates = [seed];
	for (let i = 0; i < N_NEXT; i++) {
		const pre = real.getSeed();
		check(`${seed} pre[${i}]`, pre, ref.getSeed());
		const rv = real.rng.next(), fv = ref.rng.next();
		check(`${seed} next[${i}]`, rv, fv);
		const post = real.getSeed();
		check(`${seed} post[${i}]`, post, ref.getSeed());
		lines.push(`NEXT\t${pre}\t${rv}\t${post}`);
		preStates.push(post);
	}

	// Independent high-level calls, each from a distinct pre-state seed.
	preStates.forEach((s, idx) => {
		const n = NS[idx % NS.length];
		const rN = new RealPRNG(s).random(n), fN = new RefPRNG(s).random(n);
		check(`${s} random(${n})`, rN, fN);
		lines.push(`RANDN\t${s}\t${n}\t${rN}`);

		const [m, nn] = MNS[idx % MNS.length];
		const rMN = new RealPRNG(s).random(m, nn), fMN = new RefPRNG(s).random(m, nn);
		check(`${s} random(${m},${nn})`, rMN, fMN);
		lines.push(`RANDMN\t${s}\t${m}\t${nn}\t${rMN}`);

		const [num, den] = CHANCES[idx % CHANCES.length];
		const rC = new RealPRNG(s).randomChance(num, den), fC = new RefPRNG(s).randomChance(num, den);
		check(`${s} randomChance(${num},${den})`, rC, fC);
		lines.push(`CHANCE\t${s}\t${num}\t${den}\t${rC ? 1 : 0}`);

		const len = LENS[idx % LENS.length];
		const realArr = Array.from({ length: len }, (_, k) => k);
		const refArr = Array.from({ length: len }, (_, k) => k);
		new RealPRNG(s).shuffle(realArr);
		new RefPRNG(s).shuffle(refArr);
		check(`${s} shuffle(${len})`, realArr, refArr);
		lines.push(`SHUFFLE\t${s}\t${len}\t${realArr.join(',')}`);
	});
}

if (mismatches > 0) {
	console.error(`\n✗ ${mismatches} mismatch(es): prng_reference.js is NOT bit-exact vs prng.js. Not writing golden vectors.`);
	process.exit(1);
}

const header = [
	'# Golden PRNG vectors from the REAL pokemon-showdown prng.js.',
	'# Regenerate: node src/rust_sim/harness/gen_prng_vectors.js',
	'# Each line is an independent assertion keyed on a pre-state seed.',
	'# Format: see harness/gen_prng_vectors.js header.',
	'',
];
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, header.concat(lines).join('\n') + '\n');
console.log(`✓ reference matches real prng.js on all ${SEEDS.length} seeds`);
console.log(`✓ wrote ${lines.length} assertions -> ${path.relative(process.cwd(), OUT)}`);
