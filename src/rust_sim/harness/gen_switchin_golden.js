// gen_switchin_golden.js — Gen-3 >start SWITCH-IN-EVENT differential harness.
//
// Starts REAL gen3 battles over an in-process OMNISCIENT BattleStream (the
// damage_probe.js / gen_state_golden.js pattern — NO server), advances to the
// first MOVE request, and dumps the POST-switch-in EVENT outputs the construction
// step deferred:
//   * each active lead's Atk boost stage (the Intimidate -1 drop), and
//   * field.weather + weatherState.duration (the Sand Stream / Drizzle / Drought
//     weather, set PERMANENT — duration 0 — for an ability source in gen<=5).
//
// WHY this is the right read point (verified spec + empirical PRNG instrument):
//   * gen3ou has NO team preview, so after `>start` + two `>player` lines the sim
//     lands on requestState==='move', turn 1, with each lead's switch-in events
//     ALREADY fired (runSwitch -> singleEvent('Start', ability)).
//   * The switch-in abilities fire in raw-Speed order (faster lead first). When
//     BOTH leads set weather, the SLOWER lead fires LAST so its weather wins
//     (verified: Kyogre Drizzle vs slower Tyranitar Sand Stream -> sandstorm).
//   * The ONLY PRNG draw inside the SwitchIn dispatch is speedSort's
//     per-tie-group Fisher-Yates shuffle, which fires only on a handler speed
//     TIE. For DISTINCT-speed leads it draws ZERO, so the boosts/weather end
//     state is fully deterministic. (Two other real draws bracket the >start
//     window — the per-mon gender sample in the Pokemon ctor and the gen3 Quick
//     Claw randomChance(1,5) in endTurn — but they belong to the construction /
//     turn-loop phases, do NOT affect these boosts/weather, and are out of scope
//     for this bounded switch-in step.)
//
// Scenarios (all DISTINCT-speed leads => no shuffle draw => deterministic):
//   A  Tyranitar(Sand Stream) vs Gyarados(Intimidate)   — sand + foe atk -1, ASYMMETRIC
//   B  Tyranitar(Sand Stream) vs Salamence(Intimidate)  — faster Intimidate drops the
//                                                          weather-setter; weather still set
//   C  Kyogre(Drizzle) vs Blissey(no weather)           — raindance (single setter)
//   D  Groudon(Drought) vs Blissey(no weather)          — sunnyday (single setter)
//   E  Kyogre(Drizzle, faster) vs Tyranitar(Sand Stream, slower) — SLOWER wins => sandstorm
//                                                          (ORDER-dependent: proves slower-last)
//
// (The speed-TIE shuffle-draw path is exercised in Rust by a focused unit test;
//  these golden scenarios isolate EVENT correctness from the PRNG shuffle.)
//
// Output: tests/vectors/switchin_golden.txt, TAB-delimited, std-parseable:
//   SCEN  <name>
//   TEAM  <scen>  <p1|p2>  <packed team string>
//   SEED  <scen>  <m,n,o,p>
//   WEATHER  <scen>  <none|sandstorm|raindance|sunnyday>  <duration>
//   BOOST    <scen>  <p1|p2>  <atk-stage>
//
// Run:  node src/rust_sim/harness/gen_switchin_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/switchin_golden.txt');

// Fixed gen5 seed (matches the verified-spec [1,2,3,4]). For DISTINCT-speed leads
// the switch-in dispatch draws ZERO, so the asserted boosts/weather are
// seed-independent — but we pin it so the Rust test feeds the identical >start
// and builds the SAME Prng.
const SEED = [1, 2, 3, 4];

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };

// --- Reusable lead sets (distinct speeds across every pairing) ---
const TTAR = { species: 'Tyranitar', ability: 'Sand Stream', item: 'Leftovers', nature: 'Adamant',
  evs: { hp: 252, atk: 252, spd: 4 }, ivs: IV31, level: 100,
  moves: ['Rock Slide', 'Earthquake', 'Crunch', 'Dragon Dance'] };            // spe 158
const SALAMENCE = { species: 'Salamence', ability: 'Intimidate', item: 'Choice Band', nature: 'Adamant',
  evs: { atk: 252, spd: 4, spe: 252 }, ivs: IV31, level: 100,
  moves: ['Earthquake', 'Rock Slide', 'Hidden Power Flying', 'Brick Break'] }; // spe 299
const GYARADOS = { species: 'Gyarados', ability: 'Intimidate', item: 'Leftovers', nature: 'Adamant',
  evs: { hp: 156, atk: 252, spe: 100 }, ivs: IV31, level: 100,
  moves: ['Dragon Dance', 'Earthquake', 'Hidden Power Flying', 'Taunt'] };     // spe 223
const KYOGRE = { species: 'Kyogre', ability: 'Drizzle', item: 'Leftovers', nature: 'Modest',
  evs: { hp: 4, spa: 252, spe: 252 }, ivs: IV31, level: 100,
  moves: ['Surf', 'Ice Beam', 'Thunder', 'Calm Mind'] };                       // spe 279
const GROUDON = { species: 'Groudon', ability: 'Drought', item: 'Leftovers', nature: 'Adamant',
  evs: { hp: 4, atk: 252, spe: 252 }, ivs: IV31, level: 100,
  moves: ['Earthquake', 'Rock Slide', 'Swords Dance', 'Hidden Power Bug'] };   // spe 279
const BLISSEY = { species: 'Blissey', ability: 'Natural Cure', item: 'Leftovers', nature: 'Calm',
  evs: { hp: 252, spd: 252, spe: 4 }, ivs: IV31, level: 100,
  moves: ['Soft-Boiled', 'Seismic Toss', 'Toxic', 'Aromatherapy'] };           // spe 147

// Each scenario is a single-lead pair (one mon per side keeps the read trivial;
// the switch-in dispatch is the same with a full team — only the leads' abilities
// fire at start).
const SCENARIOS = [
  { name: 'sand_vs_intim_gyarados', p1: [TTAR], p2: [GYARADOS] },
  { name: 'sand_vs_intim_salamence', p1: [TTAR], p2: [SALAMENCE] },
  { name: 'drizzle_vs_neutral', p1: [KYOGRE], p2: [BLISSEY] },
  { name: 'drought_vs_neutral', p1: [GROUDON], p2: [BLISSEY] },
  { name: 'drizzle_faster_vs_sand_slower', p1: [KYOGRE], p2: [TTAR] },
];

function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function startBattle(p1Sets, p2Sets) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const _ of streams.omniscient) { /* discard */ } })();

  const p1packed = Teams.pack(p1Sets);
  const p2packed = Teams.pack(p2Sets);

  streams.omniscient.write(`>start {"formatid":"gen3ou","seed":[${SEED.join(',')}]}`);
  streams.omniscient.write('>player p1 ' + JSON.stringify({ name: 'P1', team: p1packed }));
  streams.omniscient.write('>player p2 ' + JSON.stringify({ name: 'P2', team: p2packed }));

  for (let i = 0; i < 12; i++) await tick(); // quiesce to the first move request

  return { stream, streams, p1packed, p2packed };
}

async function main() {
  const lines = [];
  lines.push('# switchin_golden.txt — Gen-3 >start SWITCH-IN-EVENT outputs (boosts + weather).');
  lines.push('# SCEN <name>  /  TEAM <scen> <p1|p2> <packed>  /  SEED <scen> <m,n,o,p>');
  lines.push('# WEATHER <scen> <none|sandstorm|raindance|sunnyday> <duration>');
  lines.push('# BOOST   <scen> <p1|p2> <atk-stage>  (the active lead Atk boost; -1 = Intimidate hit)');
  lines.push('# Rust: Battle::start_with_switchins reproduces these from the SAME teams + seed.');

  let scenCount = 0;
  for (const sc of SCENARIOS) {
    const { stream, streams, p1packed, p2packed } = await startBattle(sc.p1, sc.p2);
    const b = stream.battle;

    if (!b || !b.started) throw new Error(`scenario ${sc.name}: battle did not start`);
    if (b.gen !== 3) throw new Error(`scenario ${sc.name}: expected gen 3, got ${b.gen}`);
    if (b.turn !== 1) throw new Error(`scenario ${sc.name}: expected turn 1, got ${b.turn}`);

    const weather = b.field.weather || 'none';
    const wdur = b.field.weatherState ? (b.field.weatherState.duration || 0) : 0;
    const a0 = b.sides[0].active[0];
    const a1 = b.sides[1].active[0];

    // Sanity: the read is at the construction/first-request boundary (leads slot 0).
    if (a0 !== b.sides[0].pokemon[0] || a1 !== b.sides[1].pokemon[0]) {
      throw new Error(`scenario ${sc.name}: lead is not pokemon[0]`);
    }

    lines.push(`SCEN\t${sc.name}`);
    lines.push(`TEAM\t${sc.name}\tp1\t${p1packed}`);
    lines.push(`TEAM\t${sc.name}\tp2\t${p2packed}`);
    lines.push(`SEED\t${sc.name}\t${SEED.join(',')}`);
    lines.push(`WEATHER\t${sc.name}\t${weather}\t${wdur}`);
    lines.push(`BOOST\t${sc.name}\tp1\t${a0.boosts.atk}`);
    lines.push(`BOOST\t${sc.name}\tp2\t${a1.boosts.atk}`);

    try { streams.omniscient.destroy(); } catch (e) { /* best effort */ }
    scenCount++;
  }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(`switchin golden: ${scenCount} scenarios -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
