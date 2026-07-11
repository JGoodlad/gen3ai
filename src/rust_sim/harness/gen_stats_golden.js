// gen_stats_golden.js — Gen-3 in-battle STAT computation differential harness.
//
// Loads the REAL Pokémon Showdown sim and, for a spread of (species, level,
// nature, EV-spread, IV-spread) inputs covering the floor / nature / level edge
// cases, reads the sim's OWN computed stats — the ground truth — by driving an
// in-process OMNISCIENT BattleStream exactly like damage_probe.js (no live
// server): >start gen3customgame, >player with the set packed via Teams.pack,
// tick until the sim quiesces, then snapshot the live Pokemon object.
//
// CRITICAL (verified spec): `a.storedStats` is {atk,def,spa,spd,spe} and has NO
// hp key (pokemon.js:185). HP lives in `a.maxhp` (== a.baseStoredStats.hp). So we
// assemble [hp,atk,def,spa,spd,spe] = [a.maxhp, storedStats.atk..spe].
//
// Each input emits ONE self-contained TAB-delimited golden line the Rust test
// (tests/stats_test.rs) parses with std only. We emit BOTH the canonical PACKED
// string (which the Rust `team::unpack` reconstructs the exact input from) AND
// the 6 real Showdown stats:
//
//   STAT  <packed>  <hp> <atk> <def> <spa> <spd> <spe>   (all TAB-separated)
//
// The Rust test does: unpack(packed) -> compute_stats(set, dex) == [the 6 stats].
//
// Run:  node src/rust_sim/harness/gen_stats_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/stats_golden.txt');

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const EVMAX = { hp: 252, atk: 252, def: 252, spa: 252, spd: 252, spe: 252 };

// A dummy opponent set so the battle can start (its stats are irrelevant).
const DUMMY = {
	species: 'Snorlax', item: '', ability: 'Immunity',
	moves: ['Body Slam'], evs: EV0, ivs: IV31, nature: 'Hardy', level: 100,
};

// The cases. Each carries a `mv` (a real legal move so the validator is happy)
// — for stat reading the move is immaterial, but a packable set needs >=1 move.
// We deliberately span: all-0 EV/IV; max 252 EV / 31 IV; a nature boosting AND
// one lowering EACH stat (atk/def/spa/spd/spe — HP is never natured); level 100
// AND a low level (5); Shedinja (base HP 1); a 0-Atk min IV; odd base stats for
// floor edges (Blissey 255 HP, Shuckle 230 Def/SpD).
const CASES = [
	// 1. Canonical: Adamant 252HP/252Atk Tyranitar (spec-verified -> [404,403,256,203,236,158]).
	{ species: 'Tyranitar', nature: 'Adamant', level: 100,
	  evs: { hp: 252, atk: 252, def: 0, spa: 0, spd: 0, spe: 0 }, ivs: IV31, mv: 'Crunch' },
	// 2. Bold 252HP/252Def Skarmory (spec-verified -> [334,176,416,116,177,176]; Atk hindered).
	{ species: 'Skarmory', nature: 'Bold', level: 100,
	  evs: { hp: 252, atk: 0, def: 252, spa: 0, spd: 0, spe: 0 }, ivs: IV31, mv: 'Spikes' },
	// 3. Serious (neutral) 0-EV Blissey (spec-verified -> [651,56,56,186,306,146]).
	{ species: 'Blissey', nature: 'Serious', level: 100, evs: EV0, ivs: IV31, mv: 'Soft-Boiled' },
	// 4. All-0 EV / all-0 IV, neutral, level 100 (minimum-stat floor edge).
	{ species: 'Blissey', nature: 'Hardy', level: 100,
	  evs: EV0, ivs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, mv: 'Soft-Boiled' },
	// 5. Max 252 EV across the board / 31 IV, neutral, level 100.
	{ species: 'Snorlax', nature: 'Hardy', level: 100, evs: EVMAX, ivs: IV31, mv: 'Body Slam' },
	// 6. A nature boosting SpA and hindering Spe (Quiet) on a fast special mon.
	{ species: 'Starmie', nature: 'Quiet', level: 100,
	  evs: { hp: 0, atk: 0, def: 0, spa: 252, spd: 0, spe: 252 }, ivs: IV31, mv: 'Hydro Pump' },
	// 7. A nature boosting Def and hindering SpA (Bold) on a wall.
	{ species: 'Suicune', nature: 'Bold', level: 100,
	  evs: { hp: 252, atk: 0, def: 252, spa: 0, spd: 4, spe: 0 }, ivs: IV31, mv: 'Surf' },
	// 8. A nature boosting Spe and hindering Atk (Timid).
	{ species: 'Gengar', nature: 'Timid', level: 100,
	  evs: { hp: 0, atk: 0, def: 0, spa: 252, spd: 4, spe: 252 }, ivs: IV31, mv: 'Thunderbolt' },
	// 9. A nature boosting SpD and hindering Def (Careful).
	{ species: 'Snorlax', nature: 'Careful', level: 100,
	  evs: { hp: 188, atk: 168, def: 0, spa: 0, spd: 152, spe: 0 }, ivs: IV31, mv: 'Body Slam' },
	// 10. A nature boosting Atk and hindering Def (Lonely).
	{ species: 'Heracross', nature: 'Lonely', level: 100, evs: EVMAX, ivs: IV31, mv: 'Megahorn' },
	// 11. Level 5 (LC-style) — small-number floor behaviour.
	{ species: 'Magnemite', nature: 'Modest', level: 5,
	  evs: { hp: 0, atk: 0, def: 0, spa: 240, spd: 0, spe: 240 }, ivs: IV31, mv: 'Thunderbolt' },
	// 12. Level 5, all-0 EV/IV neutral (extreme low-stat floor).
	{ species: 'Gengar', nature: 'Hardy', level: 5,
	  evs: EV0, ivs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 }, mv: 'Thunderbolt' },
	// 13. Shedinja — base HP 1; the fixed-HP species (must yield HP 1).
	{ species: 'Shedinja', nature: 'Adamant', level: 100,
	  evs: { hp: 252, atk: 252, def: 0, spa: 0, spd: 0, spe: 4 }, ivs: IV31, mv: 'Shadow Ball' },
	// 14. 0-Atk min IV (HP-type / confusion-min) on a special attacker.
	{ species: 'Jynx', nature: 'Timid', level: 100,
	  evs: { hp: 36, atk: 0, def: 0, spa: 252, spd: 0, spe: 220 },
	  ivs: { ...IV31, atk: 0 }, mv: 'Ice Beam' },
	// 15. Shuckle — extreme base Def/SpD (230) for floor/round edges.
	{ species: 'Shuckle', nature: 'Bold', level: 100,
	  evs: { hp: 252, atk: 0, def: 252, spa: 0, spd: 4, spe: 0 }, ivs: IV31, mv: 'Toxic' },
	// 16. An EV that is NOT a multiple of 4 (ev/4 floor edge): 251/253 etc.
	{ species: 'Blissey', nature: 'Calm', level: 100,
	  evs: { hp: 251, atk: 0, def: 0, spa: 0, spd: 253, spe: 6 }, ivs: IV31, mv: 'Soft-Boiled' },
	// 17. Level 78 (an odd mid level) for *level/100 floor behaviour.
	{ species: 'Dragonite', nature: 'Adamant', level: 78,
	  evs: { hp: 0, atk: 252, def: 0, spa: 0, spd: 4, spe: 252 }, ivs: IV31, mv: 'Earthquake' },
	// 18. Level 1 (extreme floor).
	{ species: 'Snorlax', nature: 'Hardy', level: 1, evs: EV0, ivs: IV31, mv: 'Body Slam' },
];

function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function realStats(set) {
	const stream = new BattleStream();
	const streams = getPlayerStreams(stream);
	// Drain the omniscient stream so the sim runs; we read the live object after.
	(async () => { for await (const _ of streams.omniscient) { /* discard */ } })();

	streams.omniscient.write('>start {"formatid":"gen3customgame"}');
	streams.omniscient.write('>player p1 ' + JSON.stringify({ name: 'P1', team: Teams.pack([set]) }));
	streams.omniscient.write('>player p2 ' + JSON.stringify({ name: 'P2', team: Teams.pack([DUMMY]) }));

	for (let i = 0; i < 6; i++) await tick(); // quiesce so setSpecies populated stats

	const a = stream.battle.sides[0].active[0];
	if (!a) throw new Error('no active mon for ' + set.species);
	const s = a.storedStats; // {atk,def,spa,spd,spe} — NO hp key
	const hp = a.maxhp;       // == a.baseStoredStats.hp
	if (hp !== a.baseStoredStats.hp) {
		throw new Error(`maxhp != baseStoredStats.hp for ${set.species}: ${hp} vs ${a.baseStoredStats.hp}`);
	}
	const stats = [hp, s.atk, s.def, s.spa, s.spd, s.spe];
	try { streams.omniscient.destroy(); } catch (e) { /* best effort */ }
	return stats;
}

async function main() {
	const lines = [];
	lines.push('# stats_golden.txt — Gen-3 in-battle stat computation golden.');
	lines.push('# Lines: STAT <packed> <hp> <atk> <def> <spa> <spd> <spe>  (TAB-separated).');
	lines.push('# Rust: compute_stats(unpack(<packed>)[0], Dex::for_gen(3)) == [the 6 stats].');

	let n = 0;
	for (const c of CASES) {
		const set = {
			species: c.species, item: '', ability: '', moves: [c.mv],
			evs: c.evs, ivs: c.ivs, nature: c.nature, level: c.level,
		};
		const stats = await realStats(set);
		// Pack the EXACT input the Rust test reconstructs. Teams.pack drops level
		// 100 / default IV/EV — fine: Rust unpack restores the same defaults.
		const packed = Teams.pack([set]);
		lines.push(`STAT\t${packed}\t${stats.join('\t')}`);
		n++;
	}

	fs.writeFileSync(OUT, lines.join('\n') + '\n');
	console.error(`stats golden: ${n} cases -> ${OUT}`);
	process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
