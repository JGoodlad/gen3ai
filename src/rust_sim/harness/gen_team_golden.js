// gen_team_golden.js — team pack/unpack differential harness (the bridge gate).
//
// Loads the REAL Pokémon Showdown Teams lib (deps/pokemon-showdown/dist/sim/
// index.js) and emits SELF-CONTAINED golden assertions for the Rust port
// (src/rust_sim/src/team.rs). The packed string is exactly what the bridge
// feeds `>player p1 {team}`, so reproducing Teams.pack/unpack bit-for-bit is
// the requirement.
//
// We construct ~10 representative gen-3 sets covering the edge cases that bite
// (default vs custom EVs/IVs, a non-default Hidden-Power IV spread, a shiny,
// level < 100, a nickname differing from the species, 1-4 moves, item present/
// absent, happiness, an hpType tail). For EACH we emit two inputs — Showdown's
// canonical pack AND a poke-env lowercase-id variant (our real producer) — plus
// a set of hand-crafted RAW_CASES that pin malformed-but-legal field decoding
// (a ']' inside a nickname, a short IV field, a trailing-comma moves field, an
// empty species field). Each input yields a TAB-delimited triple the Rust test
// parses with std only:
//
//   IN      <packed string>            # the input we feed Rust unpack()
//   UNPACK  <field>=<value> ...        # real Teams.unpack(IN)[0] canonical fields
//   PACK    <packed string>            # real Teams.pack(unpack(IN)) — canonical
//
// The Rust test asserts: unpack(IN) == the UNPACK fields, AND pack(unpack(IN))
// == the PACK string. PACK is the CANONICAL re-pack, which may differ from IN
// (e.g. a poke-env lowercase input re-packs to Showdown's case-preserving bytes).
//
// Run:  node src/rust_sim/harness/gen_team_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Teams } = require(path.join(PS, 'dist/sim/index.js'));
const { toID } = require(path.join(PS, 'dist/sim/dex.js'));

const OUT = path.resolve(__dirname, '../tests/vectors/team_golden.txt');

// Default IV/EV objects (Teams.pack reads per-stat).
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };

// ~10 representative gen-3 sets. Each is ONE set (we pack a single-set team so
// every golden line is independent).
const CASES = [
	// 1. Canonical special attacker: custom EVs, a 0-IV (HP-type/min-Atk),
	//    gender, item, ability, 4 moves.
	{
		name: 'Jynx', species: 'Jynx', item: 'Leftovers', ability: 'Oblivious',
		moves: ['Ice Beam', 'Calm Mind', 'Substitute', 'Lovely Kiss'],
		nature: 'Timid',
		evs: { hp: 36, atk: 0, def: 0, spa: 252, spd: 0, spe: 220 },
		ivs: { ...IV31, atk: 0 }, gender: 'F', level: 100,
	},
	// 2. Explicit non-default IV line (Atk 2, SpA 30) — survives verbatim.
	{
		name: 'Suicune', species: 'Suicune', item: 'Leftovers', ability: 'Pressure',
		moves: ['Calm Mind', 'Hydro Pump', 'Ice Beam', 'Hidden Power Grass'],
		nature: 'Timid',
		evs: { hp: 56, atk: 0, def: 0, spa: 220, spd: 0, spe: 232 },
		ivs: { ...IV31, atk: 2, spa: 30 }, level: 100,
	},
	// 3. Hidden-Power IV spread (HP Bug): Def/Spe 30.
	{
		name: 'Dugtrio', species: 'Dugtrio', item: 'Choice Band', ability: 'Arena Trap',
		moves: ['Earthquake', 'Beat Up', 'Hidden Power Bug', 'Aerial Ace'],
		nature: 'Jolly',
		evs: { hp: 40, atk: 144, def: 0, spa: 0, spd: 100, spe: 224 },
		ivs: { ...IV31, spa: 30, spe: 30 }, level: 100,
	},
	// 4. No item, all-default EVs and IVs (both fields collapse to empty).
	{
		name: 'Skarmory', species: 'Skarmory', item: '', ability: 'Keen Eye',
		moves: ['Spikes', 'Whirlwind', 'Toxic', 'Protect'],
		nature: 'Impish', evs: EV0, ivs: IV31, level: 100,
	},
	// 5. Shiny + a real nickname (differs from species) + 3 moves.
	{
		name: 'Sparky', species: 'Zapdos', item: 'Leftovers', ability: 'Pressure',
		moves: ['Thunderbolt', 'Hidden Power Ice', 'Roar'],
		nature: 'Modest',
		evs: { hp: 248, atk: 0, def: 0, spa: 252, spd: 8, spe: 0 },
		ivs: { ...IV31, atk: 30, spe: 30 }, shiny: true, level: 100,
	},
	// 6. Level < 100 (LC-style) + 1 move + genderless.
	{
		name: 'Magnemite', species: 'Magnemite', item: 'Oran Berry', ability: 'Sturdy',
		moves: ['Thunderbolt'], nature: 'Modest',
		evs: { hp: 0, atk: 0, def: 0, spa: 240, spd: 0, spe: 240 },
		ivs: IV31, gender: 'N', level: 5,
	},
	// 7. Custom happiness (Return user).
	{
		name: 'Snorlax', species: 'Snorlax', item: 'Leftovers', ability: 'Immunity',
		moves: ['Return', 'Body Slam', 'Self-Destruct', 'Earthquake'],
		nature: 'Adamant',
		evs: { hp: 188, atk: 168, def: 152, spa: 0, spd: 0, spe: 0 },
		ivs: IV31, happiness: 200, level: 100,
	},
	// 8. hpType tail set explicitly (the misc sub-list fires).
	{
		name: 'Gengar', species: 'Gengar', item: 'Leftovers', ability: 'Levitate',
		moves: ['Explosion', 'Hidden Power', 'Thunderbolt', 'Will-O-Wisp'],
		nature: 'Timid',
		evs: { hp: 168, atk: 0, def: 0, spa: 0, spd: 164, spe: 176 },
		ivs: { ...IV31, hp: 30, spd: 30 }, hpType: 'Grass', level: 100,
	},
	// 9. Adamant physical wall-breaker, mixed EVs, male, no nickname.
	{
		name: 'Claydol', species: 'Claydol', item: 'Leftovers', ability: 'Levitate',
		moves: ['Rapid Spin', 'Earthquake', 'Psychic', 'Explosion'],
		nature: 'Adamant',
		evs: { hp: 244, atk: 204, def: 0, spa: 32, spd: 20, spe: 8 },
		ivs: IV31, gender: 'M', level: 100,
	},
	// 10. A 0-Spe min-speed IV (Trick Room / Beat Up edge) + a name with
	//     punctuation (Mr. Mime) to exercise packName stripping.
	{
		name: 'Mr. Mime', species: 'Mr. Mime', item: 'Salac Berry', ability: 'Soundproof',
		moves: ['Baton Pass', 'Calm Mind', 'Substitute', 'Encore'],
		nature: 'Calm',
		evs: { hp: 252, atk: 0, def: 120, spa: 0, spd: 136, spe: 0 },
		ivs: { ...IV31, spe: 0 }, level: 100,
	},
];

// Hand-crafted (NOT pack-normalized) packed strings that pin the EXACT decode of
// malformed-but-legal fields — the cases pack()-round-tripping can never generate.
// Each is captured straight from real Teams.unpack by the harness below; the PACK
// line shows the CANONICAL re-pack (which may differ from the input).
const RAW_CASES = [
	// 1. ']' inside a nickname — must stay ONE set (sequential parse, not split(']')).
	'Snor]lax|Snorlax|Leftovers|Immunity|BodySlam|Adamant||||||',
	// 2. Short IV field '30,' — slot0=30, slot1 ('')=31, slots2-5 (absent)=0.
	'Regice||Leftovers|ClearBody|IceBeam,ThunderWave,Toxic,Rest|Calm|252,,,,200,||30,|||',
	// 3. Trailing-comma moves 'BodySlam,' — keeps the empty token (byte-exact re-pack).
	'Snorlax||Leftovers|Immunity|BodySlam,|Adamant||||||',
	// 4. Empty species field — species falls back to the nickname.
	'Pikachu|||Static|Thunderbolt|Timid||||||',
];

// Mimic poke-env's TeambuilderPokemon.packed (our REAL producer): lowercase the
// id fields (species/item/ability/moves), leaving name + nature as-is. Our unpack
// is case-insensitive, so this must decode identically and re-pack to Showdown's
// case-preserving canonical bytes. Single-set strings only (no ']').
function lowercaseIds(packed) {
	const f = packed.split('|');
	for (const k of [1, 2, 3, 4]) if (f[k] !== undefined) f[k] = f[k].toLowerCase();
	return f.join('|');
}

// Format one unpacked set as canonical TAB-delimited key=value fields. We
// materialize the same concrete defaults the Rust PokemonSet uses (Teams.unpack
// leaves some keys undefined — level/happiness/evs/ivs — so we resolve them
// here to the documented defaults, matching the Rust decode). Moves keep empty
// tokens (Showdown does too), so a trailing-comma moves field round-trips.
function unpackFields(u) {
	const evs = u.evs || EV0;
	const ivs = u.ivs || IV31;
	const ev = (s) => Number(evs[s] ?? 0);
	const iv = (s) => Number(ivs[s] ?? 31);
	const f = [];
	f.push(`name=${u.name || ''}`);
	f.push(`species=${toID(u.species || '')}`);
	f.push(`item=${toID(u.item || '')}`);
	f.push(`ability=${toID(u.ability || '')}`);
	f.push(`moves=${(u.moves || []).map(toID).join(',')}`);
	f.push(`nature=${u.nature || ''}`);
	f.push(`evs=${ev('hp')},${ev('atk')},${ev('def')},${ev('spa')},${ev('spd')},${ev('spe')}`);
	f.push(`ivs=${iv('hp')},${iv('atk')},${iv('def')},${iv('spa')},${iv('spd')},${iv('spe')}`);
	f.push(`gender=${u.gender || ''}`);
	f.push(`shiny=${u.shiny ? '1' : '0'}`);
	f.push(`level=${u.level === undefined ? 100 : u.level}`);
	f.push(`happiness=${u.happiness === undefined ? 255 : u.happiness}`);
	f.push(`hptype=${u.hpType || ''}`);
	return f.join('\t');
}

// Emit one (IN, UNPACK, PACK) triple for an input packed string, decoded by REAL
// Showdown. Contract the Rust test checks: unpack(IN) == UNPACK fields, and
// pack(unpack(IN)) == PACK (the canonical re-pack).
function emit(lines, input) {
	const u = Teams.unpack(input);
	if (!u || u.length !== 1) {
		throw new Error(`Showdown unpack did not yield 1 set for ${JSON.stringify(input)}: ${JSON.stringify(u)}`);
	}
	lines.push(`IN\t${input}`);
	lines.push(`UNPACK\t${unpackFields(u[0])}`);
	lines.push(`PACK\t${Teams.pack(u)}`);
}

function main() {
	const lines = [];
	lines.push('# team_golden.txt — Showdown Teams.pack/unpack golden (gen-3).');
	lines.push('# Triples: IN <str> / UNPACK <k=v...> / PACK <str>.');
	lines.push('# Rust: unpack(IN) == UNPACK fields; pack(unpack(IN)) == PACK.');

	let n = 0;
	for (const set of CASES) {
		const packed = Teams.pack([set]);
		// Sanity: pack is idempotent on its own output (mirrors the PRNG harness's
		// reference cross-check — abort rather than bake a wrong golden).
		const re = Teams.pack(Teams.unpack(packed));
		if (re !== packed) {
			throw new Error(`pack not idempotent for ${set.species}:\n  ${packed}\n  ${re}`);
		}
		emit(lines, packed); n++;                 // Showdown canonical form
		emit(lines, lowercaseIds(packed)); n++;   // poke-env lowercase-id form
	}
	for (const raw of RAW_CASES) { emit(lines, raw); n++; }

	fs.writeFileSync(OUT, lines.join('\n') + '\n');
	console.error(`team golden: ${n} cases (${n * 3} golden lines) -> ${OUT}`);
}

main();
