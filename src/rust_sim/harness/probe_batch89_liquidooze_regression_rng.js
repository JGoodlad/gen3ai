// probe_batch89_liquidooze_regression_rng.js — GROUND TRUTH for the `gen3_liquid_ooze_v1`
// LIQUID OOZE regression pins (LO1 drain reversal / LO2 leech reversal / LO3 drain-KO) in
// tests/regression_test.rs.
//
// Drives the OMNISCIENT in-process BattleStream (no server), RESEEDED to a RAW seed right before
// the first decision (matching the Rust's draw-free `start_with_switchins`), printing each
// decision's seedAfter + both actives' HP + whether a Liquid-Ooze reversal `-damage` fired. The
// reversal is DRAW-FREE (a drain into a Liquid Ooze mon draws the SAME chain as into a Clear-Body
// control), so LO1's Liquid-Ooze board and its Clear-Body control share the same seedAfter.
//
// Run:  node src/rust_sim/harness/probe_batch89_liquidooze_regression_rng.js
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
    const before = log.length;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const ooze = log.slice(before).some((l) => l.includes('|-damage|') && l.includes('[from] ability: Liquid Ooze'));
    const a0 = b.sides[0].active[0], a1 = b.sides[1].active[0];
    const f = (a) => a ? `${a.hp}/${a.maxhp}${a.fainted?' FNT':''}` : 'GONE';
    console.log(`  dec ${i} ${JSON.stringify(entry)} ooze=${ooze}`);
    console.log(`    seedAfter=${b.prng.getSeed()}  p1=${f(a0)} p2=${f(a1)}`);
    i++;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  const venuGiga = "Venusaur|||Overgrow|gigadrain,sludgebomb|Modest|252,,4,252,,|N||||";
  const venuLeech = "Venusaur|||Overgrow|leechseed,sludgebomb|Modest|252,,4,252,,|N||||";
  const tentaLO = "Tentacruel|||Liquid Ooze|barrier,splash|Bold|252,,,,252,|N||||";
  const tentaCB = "Tentacruel|||Clear Body|barrier,splash|Bold|252,,,,252,|N||||";
  const seed = [11, 23, 37, 41];

  // LO1: Giga Drain into a Liquid Ooze Tentacruel -> Venusaur TAKES damage (reversal).
  await run('LO1 drain reversal (Liquid Ooze)', venuGiga, tentaLO, seed, [{ p1: 'move 1', p2: 'move 2' }]);
  // LO1 control: Giga Drain into a CLEAR BODY Tentacruel -> Venusaur HEALS; SAME seedAfter (draw-free).
  await run('LO1 control (Clear Body — heals)', venuGiga, tentaCB, seed, [{ p1: 'move 1', p2: 'move 2' }]);

  // LO2: Leech Seed the Liquid Ooze Tentacruel -> at the residual the SEEDER (Venusaur) takes damage.
  await run('LO2 leech reversal', venuLeech, tentaLO, seed, [
    { p1: 'move 1', p2: 'move 2' },   // dec0: plant + first residual -> Venusaur takes reversal
    { p1: 'move 2', p2: 'move 2' },   // dec1: another residual
  ]);

  // LO3: a FRAIL Venusaur (0 HP EV) drains a Liquid Ooze Tentacruel that Surfs it low first, so a
  //   later Giga Drain reversal KOs Venusaur. Sweep-ish: run a fixed 4-turn script.
  const venuFrail = "Venusaur|||Overgrow|gigadrain,synthesis|Modest|,,,252,,252|N||||";
  const tentaLOatk = "Tentacruel|||Liquid Ooze|surf,splash|Modest|252,,,252,,|N||||";
  await run('LO3 drain reversal KO', venuFrail, tentaLOatk, seed, [
    { p1: 'move 1', p2: 'move 1' },   // Surf chips Venusaur / Giga Drain reversal chips it more
    { p1: 'move 1', p2: 'move 1' },
    { p1: 'move 1', p2: 'move 1' },
    { p1: 'move 1', p2: 'move 1' },
    { p1: 'move 1', p2: 'move 1' },
  ]);
}
main();
