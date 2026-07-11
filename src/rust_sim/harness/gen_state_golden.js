// gen_state_golden.js — Gen-3 CONSTRUCTION-TIME battle-state differential harness.
//
// Starts a REAL gen3ou battle over an in-process OMNISCIENT BattleStream (the
// damage_probe.js / gen_stats_golden.js pattern — NO server), advances to the
// first MOVE request, and dumps each (side, slot) mon's CONSTRUCTION-time state:
// the sim's OWN computed stats (storedStats + maxhp / baseStoredStats.hp),
// species id, level, and the active-lead flag.
//
// WHY this is the right read point + what we DO NOT assert (verified spec):
//   * gen3ou has NO team preview, so after `>start` + two `>player` lines + a few
//     ticks the sim lands directly on requestState==='move', turn 1, with each
//     side's lead in active[0] (=== side.pokemon[0], gen3 singles).
//   * The stats/maxhp/species/level we read are CONSTRUCTION-time + seed-
//     INDEPENDENT — set in the Pokemon ctor -> setSpecies -> spreadModify, BEFORE
//     any switch-in event. The Rust `Battle::start` builds EXACTLY these.
//   * We deliberately include Sand Stream (Tyranitar) and Intimidate (Salamence /
//     Gyarados) LEADS to PROVE the Rust matches the construction-time stats even
//     though those abilities FIRE at switch-in. We DO NOT emit/compare the
//     post-event boosts or field.weather (those need the event engine, the next
//     step). The golden carries stats/maxhp/species/level/active-flag ONLY.
//   * speciesid is read as `p.species.id` (NOT `p.speciesid`, which is undefined).
//   * storedStats has NO hp key; HP = p.maxhp == p.baseStoredStats.hp. We assemble
//     [hp, atk, def, spa, spd, spe] = [maxhp, storedStats.atk..spe].
//
// We ALSO emit the exact packed team strings used, so the Rust test (state_test.rs)
// feeds byte-identical input to `Battle::start`.
//
// Output: tests/vectors/state_golden.txt, TAB-delimited, std-parseable:
//   TEAM  <p1|p2>  <packed team string>
//   SEED  <m,n,o,p>
//   MON   <p1|p2>  <slot>  <speciesid>  <level>  <maxhp>  <hp> <atk> <def> <spa> <spd> <spe>  <active 0|1>
//
// Run:  node src/rust_sim/harness/gen_state_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/state_golden.txt');

// Fixed gen5 seed (matches the verified-spec [1,2,3,4]); seed only affects post-
// switch-in RNG, never the construction-time fields we assert — but we pin it so
// the Rust test feeds the identical >start.
const SEED = [1, 2, 3, 4];

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };

// Two representative gen3ou teams. P1 LEADS with a Sand Stream Tyranitar and also
// carries an Intimidate Salamence (both abilities FIRE at switch-in for the lead,
// but we only assert the construction-time stats). P2 LEADS with an Intimidate
// Gyarados. The benches span common gen3ou mons so the stat corpus is broad.
const P1 = [
	// Lead: Sand Stream (weather fires at switch-in; we don't assert weather).
	{ species: 'Tyranitar', ability: 'Sand Stream', item: 'Leftovers', nature: 'Adamant',
	  evs: { hp: 252, atk: 252, def: 0, spa: 0, spd: 4, spe: 0 }, ivs: IV31, level: 100,
	  moves: ['Rock Slide', 'Earthquake', 'Crunch', 'Dragon Dance'] },
	// Intimidate (the foe's lead would be -1 atk at the read point — we don't read boosts).
	{ species: 'Salamence', ability: 'Intimidate', item: 'Choice Band', nature: 'Adamant',
	  evs: { hp: 0, atk: 252, def: 0, spa: 0, spd: 4, spe: 252 }, ivs: IV31, level: 100,
	  moves: ['Earthquake', 'Rock Slide', 'Hidden Power Flying', 'Brick Break'] },
	{ species: 'Magneton', ability: 'Magnet Pull', item: 'Leftovers', nature: 'Modest',
	  evs: { hp: 4, atk: 0, def: 0, spa: 252, spd: 0, spe: 252 }, ivs: IV31, level: 100,
	  moves: ['Thunderbolt', 'Hidden Power Grass', 'Thunder Wave', 'Toxic'] },
	{ species: 'Skarmory', ability: 'Keen Eye', item: 'Leftovers', nature: 'Impish',
	  evs: { hp: 252, atk: 0, def: 252, spa: 0, spd: 4, spe: 0 }, ivs: IV31, level: 100,
	  moves: ['Spikes', 'Roar', 'Drill Peck', 'Protect'] },
	{ species: 'Blissey', ability: 'Natural Cure', item: 'Leftovers', nature: 'Calm',
	  evs: { hp: 252, atk: 0, def: 0, spa: 0, spd: 252, spe: 4 }, ivs: IV31, level: 100,
	  moves: ['Soft-Boiled', 'Seismic Toss', 'Toxic', 'Aromatherapy'] },
	{ species: 'Claydol', ability: 'Levitate', item: 'Leftovers', nature: 'Relaxed',
	  evs: { hp: 252, atk: 0, def: 128, spa: 0, spd: 128, spe: 0 }, ivs: IV31, level: 100,
	  moves: ['Earthquake', 'Rapid Spin', 'Psychic', 'Explosion'] },
];

const P2 = [
	// Lead: Intimidate Gyarados (boost on the foe fires at switch-in — not asserted).
	{ species: 'Gyarados', ability: 'Intimidate', item: 'Leftovers', nature: 'Adamant',
	  evs: { hp: 156, atk: 252, def: 0, spa: 0, spd: 0, spe: 100 }, ivs: IV31, level: 100,
	  moves: ['Dragon Dance', 'Earthquake', 'Hidden Power Flying', 'Taunt'] },
	{ species: 'Jirachi', ability: 'Serene Grace', item: 'Leftovers', nature: 'Careful',
	  evs: { hp: 252, atk: 0, def: 4, spa: 0, spd: 252, spe: 0 }, ivs: IV31, level: 100,
	  moves: ['Wish', 'Body Slam', 'Toxic', 'Protect'] },
	{ species: 'Suicune', ability: 'Pressure', item: 'Leftovers', nature: 'Bold',
	  evs: { hp: 252, atk: 0, def: 252, spa: 0, spd: 4, spe: 0 }, ivs: IV31, level: 100,
	  moves: ['Surf', 'Calm Mind', 'Rest', 'Sleep Talk'] },
	{ species: 'Metagross', ability: 'Clear Body', item: 'Choice Band', nature: 'Adamant',
	  evs: { hp: 0, atk: 252, def: 0, spa: 0, spd: 4, spe: 252 }, ivs: IV31, level: 100,
	  moves: ['Meteor Mash', 'Earthquake', 'Explosion', 'Rock Slide'] },
	{ species: 'Snorlax', ability: 'Immunity', item: 'Leftovers', nature: 'Careful',
	  evs: { hp: 188, atk: 128, def: 0, spa: 0, spd: 192, spe: 0 }, ivs: IV31, level: 100,
	  moves: ['Body Slam', 'Curse', 'Rest', 'Shadow Ball'] },
	// Gengar — Levitate (ground-immune) special attacker, broad stat spread.
	{ species: 'Gengar', ability: 'Levitate', item: 'Leftovers', nature: 'Timid',
	  evs: { hp: 4, atk: 0, def: 0, spa: 252, spd: 0, spe: 252 }, ivs: IV31, level: 100,
	  moves: ['Thunderbolt', 'Ice Punch', 'Hidden Power Fire', 'Will-O-Wisp'] },
];

function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function startBattle(p1Sets, p2Sets) {
	const stream = new BattleStream();
	const streams = getPlayerStreams(stream);
	// Drain the omniscient stream so the sim runs; we read live objects after.
	(async () => { for await (const _ of streams.omniscient) { /* discard */ } })();

	const p1packed = Teams.pack(p1Sets);
	const p2packed = Teams.pack(p2Sets);

	streams.omniscient.write(`>start {"formatid":"gen3ou","seed":[${SEED.join(',')}]}`);
	streams.omniscient.write('>player p1 ' + JSON.stringify({ name: 'P1', team: p1packed }));
	streams.omniscient.write('>player p2 ' + JSON.stringify({ name: 'P2', team: p2packed }));

	for (let i = 0; i < 8; i++) await tick(); // quiesce to the first move request

	return { stream, streams, p1packed, p2packed };
}

function snapMon(p) {
	const s = p.storedStats; // {atk,def,spa,spd,spe} — NO hp key
	const maxhp = p.maxhp;   // == baseStoredStats.hp
	const liveHp = p.hp;     // the sim's LIVE current HP (read independently)
	if (maxhp !== p.baseStoredStats.hp) {
		throw new Error(`maxhp != baseStoredStats.hp for ${p.species.id}: ${maxhp} vs ${p.baseStoredStats.hp}`);
	}
	// At the construction / first-request boundary, live HP must equal maxhp (no
	// residual has run). Assert it against the real sim rather than emitting maxhp
	// twice — so the <hp> column carries an independent live-HP signal.
	if (liveHp !== maxhp) {
		throw new Error(`live hp != maxhp at construction for ${p.species.id}: ${liveHp} vs ${maxhp}`);
	}
	if ('hp' in s) throw new Error(`storedStats unexpectedly has hp key for ${p.species.id}`);
	if (p.speciesid !== undefined) throw new Error('p.speciesid should be undefined (read p.species.id)');
	return {
		speciesid: p.species.id,
		level: p.level,
		maxhp,
		hp: liveHp,
		stats: [maxhp, s.atk, s.def, s.spa, s.spd, s.spe],
		position: p.position,
	};
}

async function main() {
	const { stream, streams, p1packed, p2packed } = await startBattle(P1, P2);
	const b = stream.battle;

	// Sanity: we are at the construction/first-request boundary we expect.
	if (!b || !b.started) throw new Error('battle did not start');
	if (b.gameType !== 'singles') throw new Error('expected singles gameType, got ' + b.gameType);
	if (b.gen !== 3) throw new Error('expected gen 3, got ' + b.gen);
	if (b.turn !== 1) throw new Error('expected turn 1 at first request, got ' + b.turn);

	const lines = [];
	lines.push('# state_golden.txt — Gen-3 CONSTRUCTION-TIME battle state golden.');
	lines.push('# TEAM <p1|p2> <packed>   /   SEED <m,n,o,p>');
	lines.push('# MON  <side> <slot> <speciesid> <level> <maxhp> <hp> <atk> <def> <spa> <spd> <spe> <active 0|1>');
	lines.push('# Rust: Battle::start(opts{seed, p1=packed, p2=packed}) reproduces stats/maxhp/species/level/lead.');
	lines.push('# NOTE: boosts + field.weather are EVENT-driven (Sand Stream/Intimidate fire here) and NOT asserted.');
	lines.push(`TEAM\tp1\t${p1packed}`);
	lines.push(`TEAM\tp2\t${p2packed}`);
	lines.push(`SEED\t${SEED.join(',')}`);

	let monCount = 0;
	b.sides.forEach((side, si) => {
		const tag = side.id; // 'p1' / 'p2'
		const lead = side.active[0];
		side.pokemon.forEach((p, slot) => {
			const m = snapMon(p);
			if (m.position !== slot) {
				throw new Error(`position mismatch ${tag} slot ${slot}: position=${m.position}`);
			}
			const isActive = lead === p ? 1 : 0;
			lines.push([
				'MON', tag, slot, m.speciesid, m.level, m.maxhp,
				m.hp, m.stats[1], m.stats[2], m.stats[3], m.stats[4], m.stats[5],
				isActive,
			].join('\t'));
			monCount++;
		});
		// The lead must be slot 0 in gen3 singles.
		if (side.active[0] !== side.pokemon[0]) {
			throw new Error(`${tag} lead is not pokemon[0]`);
		}
	});

	try { streams.omniscient.destroy(); } catch (e) { /* best effort */ }

	fs.writeFileSync(OUT, lines.join('\n') + '\n');
	console.error(`state golden: ${monCount} mons (2 sides) -> ${OUT}`);
	process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
