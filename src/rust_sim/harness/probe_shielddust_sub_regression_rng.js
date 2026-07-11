// probe_shielddust_sub_regression_rng.js — GROUND TRUTH for the `gen3_shielddust_sub_v1`
// regression pin (tests/regression_test.rs::shield_dust_behind_a_substitute_still_draws_the_secondary).
//
// THE BUG (the A/B fuzzer's #1 sub×secondary SEED cluster, auto_0708_0304 — 347/365 ShieldDust-team
// repros flip ok on the fix): the port filtered a Shield Dust DEFENDER's incoming secondaries
// UNCONDITIONALLY (no random(100) draw). But Shield Dust's filter is a TARGET-gathered
// ModifySecondaries handler — when a SUBSTITUTE absorbs the hit the target list is `null`, the
// filter never gathers, and the secondary `random(100)` STILL DRAWS (held AND breaking sub; the
// effect stays sub-suppressed). Same for the Tri Attack gate and the King's Rock appended
// secondary. Settled by harness/probe_sub_break_secondary_rng.js (2a/2b/2c, 3a/3b, 4a/4b).
//
// Each scenario drives the OMNISCIENT BattleStream over a CONSTRUCTED gen3customgame board whose
// EXACT packed teams + raw seed the Rust pin replays (prng reseeded at the first decision).
// Run:  node src/rust_sim/harness/probe_shielddust_sub_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
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
  let i = 0, safety = 0;
  while (!b.ended && safety < 60 && i < plan.length) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const a1 = b.sides[1].active[0];
    const sub = (a1.volatiles && a1.volatiles.substitute) ? `SUB(${a1.volatiles.substitute.hp})` : 'nosub';
    console.log(`  dec ${i - 1} ${JSON.stringify(entry)} seedAfter=${b.prng.getSeed()}`);
    console.log(`    p2=${a1.species.name} ${a1.hp}/${a1.maxhp} st=${a1.status || '-'} ${sub}`);
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

const VENOMOTH_SUB = 'Venomoth|||ShieldDust|substitute,splash|Serious|252,,,,,|N||||';
const SEED = [21, 32, 43, 54];
async function main() {
  // SD-a: the move's OWN secondary (Flamethrower brn10) into the Shield Dust sub — the
  // random(100) must draw (bare control SD-a2 filters it: one fewer draw).
  await run('SD-a Flamethrower brn10 into SUBBED Shield Dust Venomoth',
    'Magcargo|||FlameBody|flamethrower,splash|Serious||N||||', VENOMOTH_SUB, SEED,
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }]);
  await run('SD-a2 BARE control: same board, no sub (filtered — no random(100))',
    'Magcargo|||FlameBody|flamethrower,splash|Serious||N||||', VENOMOTH_SUB, SEED,
    [{ p1: 'move 2', p2: 'move 2' }, { p1: 'move 1', p2: 'move 2' }]);
  // SD-b: Tri Attack's gate random(100) into the Shield Dust sub (sample stays suppressed).
  await run('SD-b Tri Attack into SUBBED Shield Dust Venomoth',
    'Dodrio|||EarlyBird|triattack,splash|Serious||N||||', VENOMOTH_SUB, SEED,
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }]);
  // SD-c: King's Rock appended secondary into the Shield Dust sub (flinch stays suppressed).
  await run('SD-c KR Hidden Power Dark into SUBBED Shield Dust Venomoth',
    'Sceptile||kingsrock|Overgrow|hiddenpowerdark,splash|Serious||N||||', VENOMOTH_SUB, SEED,
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
