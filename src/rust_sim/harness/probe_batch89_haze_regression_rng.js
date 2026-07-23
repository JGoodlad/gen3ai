// probe_batch89_haze_regression_rng.js — GROUND TRUTH for the `gen3_haze_v1` HAZE regression pins
// (HZ1 clears both actives' boosts incl. the user's own; HZ2 the draw-free no-op) in
// tests/regression_test.rs.
//
// Drives the OMNISCIENT in-process BattleStream (no server) over a CONSTRUCTED gen3customgame
// board, RESEEDED to a RAW seed right before the first decision (matching the Rust's draw-free
// `start_with_switchins`), and prints each decision's seedAfter + both actives' 7 boost stages.
// Haze is DRAW-FREE, so a Haze turn's seedAfter must equal a Splash control's bit-for-bit.
//
// Run:  node src/rust_sim/harness/probe_batch89_haze_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }
const bs = (a) => a ? [a.boosts.atk||0,a.boosts.def||0,a.boosts.spa||0,a.boosts.spd||0,a.boosts.spe||0,a.boosts.accuracy||0,a.boosts.evasion||0].join(',') : '-';

async function run(label, p1, p2, rawSeed, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  b.prng = new PRNG(rawSeed.slice());
  console.log(`\n=== ${label} (raw seed ${rawSeed.join(',')}) ===`);
  let i = 0;
  for (const entry of plan) {
    const before = log.length;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const cab = log.slice(before).some((l) => l.startsWith('|-clearallboost'));
    const a0 = b.sides[0].active[0], a1 = b.sides[1].active[0];
    console.log(`  dec ${i} ${JSON.stringify(entry)} clearallboost=${cab}`);
    console.log(`    seedAfter=${b.prng.getSeed()}`);
    console.log(`    p1 hp=${a0.hp}/${a0.maxhp} boosts=[${bs(a0)}] | p2 hp=${a1.hp}/${a1.maxhp} boosts=[${bs(a1)}]`);
    i++;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // HZ1: p1 Snorlax SD +4 Atk, p2 Alakazam CM +2 SpA/SpD, then p1 HAZE -> BOTH reset to all-0.
  const snorlax = "Snorlax||stick|Immunity|swordsdance,haze,bodyslam,splash|Careful|252,4,,,252,|N||||".replace('stick','');
  const alakazam = "Alakazam|||Synchronize|calmmind,psychic|Modest|4,,252,,252,|N||||";
  const seed = [7, 19, 23, 31];
  await run('HZ1 haze clears both', snorlax, alakazam, seed, [
    { p1: 'move 1', p2: 'move 1' },   // dec0: SD+2 / CM+1
    { p1: 'move 1', p2: 'move 1' },   // dec1: SD+4 / CM+2
    { p1: 'move 2', p2: 'move 1' },   // dec2: HAZE -> both 0 (Alakazam CMs first at +3, then Haze wipes)
  ]);
  // HZ2: the draw-free no-op — Haze at dec0 (no boosts) vs a Splash control -> IDENTICAL seedAfter.
  await run('HZ2 haze no-boost', snorlax, alakazam, seed, [
    { p1: 'move 2', p2: 'move 2' },   // dec0: Haze (no boosts) / Psychic
  ]);
  await run('HZ2 splash control', snorlax, alakazam, seed, [
    { p1: 'move 4', p2: 'move 2' },   // dec0: Splash / Psychic -> same draws as Haze
  ]);
}
main();
