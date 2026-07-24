// probe_soundproof_regression_rng.js — GROUND TRUTH for the `gen3_ability_batch2_v1` DAMAGING
// Soundproof regression pin (soundproof_blocks_a_damaging_sound_move) in tests/regression_test.rs.
//
// Drives the OMNISCIENT in-process BattleStream (no server), RESEEDED to a RAW seed right before
// the first decision (matching the Rust's draw-free `start_with_switchins`), printing each
// decision's seedAfter + both actives' HP + whether a Soundproof block fired. A DAMAGING sound
// move (Hyper Voice) into a Soundproof holder draws ONLY its accuracy roll then `-immune` (NO
// crit/damage), so the Soundproof board and its no-Soundproof control DIVERGE in seed + HP.
//
// Run:  node src/rust_sim/harness/probe_soundproof_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }

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
    if (b.ended) { console.log(`  (battle ended before dec ${i})`); break; }
    const before = log.length;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const sp = log.slice(before).some((l) => l.includes('|-immune|') && l.includes('[from] ability: Soundproof'));
    const emit = log.slice(before).filter((l) => /Hyper Voice|-immune|-damage/.test(l));
    const a0 = b.sides[0].active[0], a1 = b.sides[1].active[0];
    const f = (a) => a ? `${a.hp}/${a.maxhp}${a.status ? ' ' + a.status : ''}${a.fainted ? ' FNT' : ''}` : 'GONE';
    console.log(`  dec ${i} ${JSON.stringify(entry)} soundproofBlock=${sp}`);
    console.log(`    seedAfter=${b.prng.getSeed()}  p1=${f(a0)} p2=${f(a1)}`);
    console.log(`    emit: ${JSON.stringify(emit)}`);
    i++;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // Snorlax uses Hyper Voice (a DAMAGING flags.sound move); Mr. Mime SOUNDPROOF is immune.
  const snorlax = "Snorlax|||Immunity|hypervoice,bodyslam|Serious|,,,,,|N||||";
  const mimeSP = "Mr. Mime|||Soundproof|calmmind,tackle|Serious|,,,,,|N||||";
  const mimeNO = "Mr. Mime|||Own Tempo|calmmind,tackle|Serious|,,,,,|N||||";
  const seed = [13, 29, 41, 53];

  // Soundproof — Hyper Voice (move 1) is BLOCKED: Mr. Mime takes NO damage, the block draws ONLY
  //   the accuracy roll (`-immune [from] ability: Soundproof`).
  await run('SP damaging Hyper Voice BLOCKED (Soundproof)', snorlax, mimeSP, seed, [{ p1: 'move 1', p2: 'move 1' }]);
  // Control (no Soundproof) — Hyper Voice HITS + damages, a DIFFERENT seed (it drew crit+damage).
  await run('SP control (Own Tempo — Hyper Voice HITS)', snorlax, mimeNO, seed, [{ p1: 'move 1', p2: 'move 1' }]);
}
main();
