// probe_batch4b_edge_regression_rng.js — GROUND TRUTH for the two BATCH 4b EDGE-branch pins the
// distinct-speed dedicated golden never realizes (`gen3_move_coverage_batch4b_v1`):
//   MC47  WATER SPOUT at 1 HP → the `max(bp, 1)` min-BP-1 clamp (floor(150·1/maxhp)=0 → bp 1)
//   MC48  BEAT UP with NO eligible party member (a STATUSED active user, all-else fainted/statused)
//         → the basePowerCallback returns null → the move FIZZLES draw-free.
//
// Run:  node src/rust_sim/harness/probe_batch4b_edge_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

function tick() { return new Promise((r) => setTimeout(r, 0)); }
const RAW = [44317, 42357, 9927, 48760];

async function run(label, p1, p2, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  if (inject) for (const inj of inject) {
    const m = b.sides[inj.side].pokemon[inj.slot || 0];
    if (inj.hp !== undefined) m.hp = inj.hp;
    if (inj.status !== undefined) { m.status = inj.status; m.statusState = { id: inj.status }; }
  }
  b.prng = new PRNG(RAW.slice());
  const a0 = () => b.sides[0].active[0], a1 = () => b.sides[1].active[0];
  console.log(`\n=== ${label} (raw seed ${RAW.join(',')}) ===`);
  let i = 0, safety = 0;
  while (!b.ended && safety < 40 && i < plan.length) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = log.length;
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const fmt = (m) => (m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.status ? ' ' + m.status : ''}` : '-');
    console.log(`  dec ${i - 1} [${rs}] ${JSON.stringify(entry)}  seedAfter=${b.prng.getSeed()}`);
    console.log(`      p1=${fmt(a0())}  p2=${fmt(a1())}`);
    log.slice(before)
      .filter((l) => /\|move\||-damage|-activate|-fail|-immune|-hitcount|cant|faint|-crit/.test(l))
      .forEach((l) => console.log(`      L ${l}`));
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // MC47: Kyogre Water Spout at 1 HP → floor(150*1/341)=0 → bp clamped to 1 (a MIN-damage HIT,
  // NOT a fail). Snorlax bulky, splashes (draw-free).
  const kyogre = 'Kyogre|||keeneye|waterspout,seismictoss|Serious|,,252,,,252|||||';
  const snorlax = 'Snorlax|||immunity|splash|Serious|252,,,,,|||||';
  await run('MC47 Water Spout at 1 HP (bp clamped to 1)', kyogre, snorlax,
    [{ p1: 'move 1', p2: 'move 1' }], [{ side: 0, slot: 0, hp: 1 }]);

  // MC48: a BURNED single-mon Beat Up user → the ONLY party member is statused → 0 eligible
  // strikes → the basePowerCallback returns null → the move FIZZLES (no strikes, no damage).
  const beatupSolo = 'Slaking|||keeneye|beatup,seismictoss|Serious|252,252,,,,|||||';
  const gengar = 'Gengar|||levitate|splash|Serious|,,,,,252|||||';
  await run('MC48 Beat Up with no eligible party (burned solo user) fizzles', beatupSolo, gengar,
    [{ p1: 'move 1', p2: 'move 1' }], [{ side: 0, slot: 0, status: 'brn' }]);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
