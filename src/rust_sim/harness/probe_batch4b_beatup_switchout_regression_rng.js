// probe_batch4b_beatup_switchout_regression_rng.js — GROUND TRUTH for the BEAT UP
// switch-out clearVolatile pin (MC46, `gen3_move_coverage_batch4b_v1`): the `beatup`
// `duration: 1` volatile is dropped by clearVolatile on switch-out (voluntary OR phaze-drag),
// exactly like its sibling focus_punch/pursuit volatiles. The port used to STRAND a stale
// `beat_up = true` on a mon phazed out the same turn it Beat Up'd (Roar priority -6 resolves
// after the move), so when the mon RE-ENTERED the active-only residual gather pushed a spurious
// duration handler → at a residual speed tie an extra `random(0,2)` shuffle vs the sim.
//
// Choreography (all Charizards → equal speed → residual ties):
//   Turn 1: p1 Charizard(slot0) Beat Ups (sets the beatup volatile), p2 Charizard Roars
//           (priority -6, resolves AFTER Beat Up) → drags p1 (n=1 eligible → the slot1
//           Charizard) → the Beat Up user is BENCHED. In the sim clearVolatile drops its
//           beatup volatile on the way out.
//   Turn 2: p1 SWITCHES the Beat Up user (now array slot 2) BACK IN, p2 Charizard Beat Ups
//           (registers ITS OWN beatup residual handler). At the turn-2 residual the returned
//           mon is ACTIVE *before* the next turn-top clear_flinch would fire — so a STALE
//           beat_up (the bug) would register a 2nd NO_ORDER/subOrder-2 handler that TIES p2's,
//           drawing one extra shuffle. The sim (clean volatile) has only p2's handler → 0
//           residual shuffle draws. The post-turn-2 seed below is the CLEAN (== fix) truth.
//
// Run:  node src/rust_sim/harness/probe_batch4b_beatup_switchout_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

function tick() { return new Promise((r) => setTimeout(r, 0)); }

const RAW = [44317, 42357, 9927, 48760];

async function run(label, p1, p2, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  b.prng = new PRNG(RAW.slice());
  const a0 = () => b.sides[0].active[0], a1 = () => b.sides[1].active[0];
  console.log(`\n=== ${label} (raw seed ${RAW.join(',')}) ===`);
  let i = 0, safety = 0;
  while (!b.ended && safety < 60 && i < plan.length) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = log.length;
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const fmt = (m) => (m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.status ? ' ' + m.status : ''} beatup=${!!(m.volatiles && m.volatiles.beatup)}` : '-');
    console.log(`  dec ${i - 1} [${rs}] ${JSON.stringify(entry)}  seedAfter=${b.prng.getSeed()}`);
    console.log(`      p1=${fmt(a0())}  bench0=${fmt(b.sides[0].pokemon[1])}  left=${b.sides[0].pokemonLeft}`);
    console.log(`      p2=${fmt(a1())}  left=${b.sides[1].pokemonLeft}`);
    log.slice(before)
      .filter((l) => /\|move\||-damage|-activate|drag|switch|faint|-crit|-immune/.test(l))
      .forEach((l) => console.log(`      L ${l}`));
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // p1: two identical Beat Up Charizards (n=1 phaze drag → deterministic target). slot0 Beat Ups
  //     then is Roared out; slot1 becomes active; turn 2 slot0 (array pos 2) switches back.
  const p1 =
    'Charizard|||Blaze|beatup,splash|Modest|,,,252,,252|N||||' +
    ']Charizard|||Blaze|beatup,splash|Modest|,,,252,,252|N||||';
  // p2: a Roar+Beat Up Charizard + a bench Magikarp (2 healthy → p2 Beat Up = 2 strikes turn 2).
  const p2 =
    'Charizard|||Blaze|roar,beatup|Modest|,,,252,,252|N||||' +
    ']Magikarp|||keeneye|splash|Serious|,,,,,|N|||5|';
  await run('MC46 Beat Up user phazed out → clearVolatile drops the beatup volatile', p1, p2, [
    { p1: 'move 1', p2: 'move 1' },   // turn 1: p1 Beat Up, p2 Roar (drags p1)
    { p1: 'switch 2', p2: 'move 2' }, // turn 2: p1 switch the BU user back, p2 Beat Up (its own residual handler)
  ]);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
