// probe_batch89_wonderguard_regression_rng.js — GROUND TRUTH for the `gen3_wonder_guard_v1`
// WONDER GUARD regression pins (WG1 neutral block / WG2 resisted never-miss block / WG3 leech
// residual KO bypass) in tests/regression_test.rs.
//
// Drives the OMNISCIENT in-process BattleStream (no server), RESEEDED to a RAW seed right before
// the first decision (matching the Rust's draw-free `start_with_switchins`), printing each
// decision's seedAfter + both actives' HP/STATUS + whether a Wonder Guard block / a Leech Seed
// residual fired. A BLOCKED damaging move draws ONLY its accuracy roll (a never-miss Magical Leaf
// draws NOTHING), so the WG board and its no-WG control DIVERGE in both seed and HP.
//
// Run:  node src/rust_sim/harness/probe_batch89_wonderguard_regression_rng.js
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
    const wg = log.slice(before).some((l) => l.includes('|-immune|') && l.includes('[from] ability: Wonder Guard'));
    const leech = log.slice(before).some((l) => l.includes('|-damage|') && l.includes('[from] Leech Seed'));
    const a0 = b.sides[0].active[0], a1 = b.sides[1].active[0];
    const f = (a) => a ? `${a.hp}/${a.maxhp}${a.status ? ' ' + a.status : ''}${a.fainted ? ' FNT' : ''}` : 'GONE';
    console.log(`  dec ${i} ${JSON.stringify(entry)} wgBlock=${wg} leechResidual=${leech}`);
    console.log(`    seedAfter=${b.prng.getSeed()}  p1=${f(a0)} p2=${f(a1)}  winner=${b.winner ?? '-'}`);
    i++;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  const chariWG = "Charizard|||Blaze|watergun,magicalleaf,ember,bodyslam|Serious|,,,,,|N||||";
  const shedWG = "Shedinja|||Wonder Guard|splash|Serious|,,,,,|N||||";
  const shedNO = "Shedinja|||Compound Eyes|splash|Serious|,,,,,|N||||";
  const venuWG = "Venusaur|||Overgrow|leechseed,ember|Modest|252,,,252,,|N||||";
  const gengarWG = "Gengar|||Levitate|thunderwave,shadowball|Timid|,,,252,,252|N||||";
  const seed = [13, 29, 41, 53];

  // WG1 — a NEUTRAL move (Water Gun, Water 1× vs Bug/Ghost) is BLOCKED: Shedinja stays 1/1, the
  //   block draws ONLY the accuracy roll. The no-WG control HITS + KOs (Shedinja 0/1).
  await run('WG1 neutral Water Gun BLOCKED (Wonder Guard)', chariWG, shedWG, seed, [{ p1: 'move 1', p2: 'move 1' }]);
  await run('WG1 control (no WG — Water Gun HITS + KOs)', chariWG, shedNO, seed, [{ p1: 'move 1', p2: 'move 1' }]);

  // WG2 — a RESISTED NEVER-MISS move (Magical Leaf, Grass 0.5× vs Bug/Ghost) is BLOCKED drawing
  //   NOTHING (no accuracy roll): Shedinja stays 1/1. The no-WG control HITS + KOs.
  await run('WG2 resisted never-miss Magical Leaf BLOCKED (Wonder Guard)', chariWG, shedWG, seed, [{ p1: 'move 2', p2: 'move 1' }]);
  await run('WG2 control (no WG — Magical Leaf HITS + KOs)', chariWG, shedNO, seed, [{ p1: 'move 2', p2: 'move 1' }]);

  // WG3 — Leech Seed (Status → WG bypass) plants; the end-of-turn residual (clampIntRange(_,1))
  //   drains the 1-HP Shedinja → it FAINTS → P1 wins (a residual bypasses the WG MOVE hook).
  await run('WG3 leech residual KOs the 1-HP Shedinja (bypass)', venuWG, shedWG, seed, [{ p1: 'move 1', p2: 'move 1' }]);

  // WG4 (context) — Thunder Wave (Status → WG bypass) paralyzes Shedinja; then Shadow Ball
  //   (Ghost SE vs Ghost) CONNECTS + KOs (WG lets an SE move through).
  await run('WG4 status bypass (par) then SE Shadow Ball KO', gengarWG, shedWG, seed, [
    { p1: 'move 1', p2: 'move 1' },
    { p1: 'move 2', p2: 'move 1' },
  ]);
}
main();
