// probe_facade_defrost_regression_rng.js — GROUND TRUTH for the `gen3_facade_v1` +
// `gen3_defrost_v1` regression pins (tests/regression_test.rs::
// facade_status_doubles_bp_and_composes / frozen_defrost_move_bypasses_the_cant).
//
// THE BUGS (the A/B fuzzer's post-ShieldDust #1 + #2 clusters, auto_0709_0805 —
// 143/145 facade-team + 8/10 sacredfire/flamewheel-team repros flip ok on the fixes):
//  1. FACADE: the dist onBasePower (`chainModify(2)` when the user has a non-slp major
//     status) was priced FLAT BP 70 by the port. Settled by harness/probe_facade_gen3.js:
//     psn/tox/par ×2; brn ×2 AND the gen3 burn-halve STILL applies; burned GUTS composes
//     (Atk ×1.5 + halve-suppressed + BP ×2); Pink Bow (DIRECT ×1.1 float) + Facade
//     COMPOSE — `70 * 1.1 == 77` exactly in f64, so the runEvent-tail integer-guard
//     PASSES and the accumulated chain (×2) re-applies → BP 154 (the old "Direct
//     discards the chain" port shortcut was wrong).
//  2. DEFROST: a FROZEN user of a `flags.defrost` move (Sacred Fire / Flame Wheel)
//     still DRAWS the 1/5 thaw roll, but on a FAILED roll it PROCEEDS anyway and thaws
//     draw-free via `frz.onModifyMove` (`|-curestatus|<mon>|frz|[from] move: <Move>`
//     BEFORE the `|move|` line). The port's old model cant'd every failed roll — a
//     draw-COUNT desync. Settled by harness/probe_sacredfire_defrost.js.
//
// Each scenario drives the OMNISCIENT BattleStream over a CONSTRUCTED gen3customgame
// board whose EXACT packed teams + raw seed the Rust pin replays (prng reseeded at the
// first decision; the user's status FORCE-SET right after the reseed, draw-free).
// Run:  node src/rust_sim/harness/probe_facade_defrost_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1, p2, rawSeed, plan, pre) {
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
  if (pre) pre(b);
  console.log(`\n=== ${label} (raw seed ${rawSeed.join(',')}) ===`);
  const a0 = b.sides[0].active[0];
  console.log(`  pre: p1=${a0.species.name} st=${a0.status || '-'}`);
  let i = 0, safety = 0;
  const logStart = b.log.length;
  while (!b.ended && safety < 60 && i < plan.length) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const u = b.sides[0].active[0];
    const t = b.sides[1].active[0];
    console.log(`  dec ${i - 1} ${JSON.stringify(entry)} seedAfter=${b.prng.getSeed()}`);
    console.log(`    p1=${u.species.name} ${u.hp}/${u.maxhp} st=${u.status || '-'}  p2=${t.species.name} ${t.hp}/${t.maxhp} st=${t.status || '-'}`);
  }
  const interesting = b.log.slice(logStart).filter(l =>
    l.startsWith('|move|') || l.includes('curestatus') || l.startsWith('|cant|') || l.startsWith('|-damage|'));
  console.log('  lines: ' + JSON.stringify(interesting));
  try { streams.omniscient.destroy(); } catch (e) {}
}

// FACADE boards: Raticate (No Ability unless Guts) facade+splash vs Snorlax splash.
const RATICATE = (item, ability) => `Raticate||${item || ''}|${ability || ''}|facade,splash|Serious||N||||`;
const SNORLAX = 'Snorlax||||splash,splash|Serious||N||||';
// DEFROST boards.
const HOOH = 'Ho-Oh||||sacredfire,flamethrower|Serious||N||||';
const ENTEI = 'Entei||||flamewheel,splash|Serious||N||||';
const BLISSEY = 'Blissey||||splash,splash|Serious||N||||';
const SEED = [21, 32, 43, 54];
const ONE_TURN = [{ p1: 'move 1', p2: 'move 1' }];

async function main() {
  const psn = (b) => b.sides[0].active[0].setStatus('psn');
  const brn = (b) => b.sides[0].active[0].setStatus('brn');
  const frz = (b) => b.sides[0].active[0].setStatus('frz');

  await run('FA-a POISONED Facade (BP 140)', RATICATE('', ''), SNORLAX, SEED, ONE_TURN, psn);
  await run('FA-b BURNED Facade (BP 140 + burn-halve)', RATICATE('', ''), SNORLAX, SEED, ONE_TURN, brn);
  await run('FA-c BURNED GUTS Facade (Atk ×1.5, halve suppressed, BP 140)', RATICATE('', 'Guts'), SNORLAX, SEED, ONE_TURN, brn);
  await run('FA-d PINK BOW + POISONED Facade (integer-guard: BP 154)', RATICATE('pinkbow', ''), SNORLAX, SEED, ONE_TURN, psn);
  await run('FA-e control: UNSTATUSED Facade (BP 70)', RATICATE('', ''), SNORLAX, SEED, ONE_TURN, null);

  // DEFROST: seed chosen so the 1/5 thaw roll FAILS (the defrost path) — verified in
  // the output (the cure line must be `[from] move:`, not `[msg]`).
  await run('DF-a FROZEN Ho-Oh Sacred Fire (defrost: proceeds + thaws)', HOOH, BLISSEY, SEED, ONE_TURN, frz);
  await run('DF-b FROZEN Entei Flame Wheel (defrost: proceeds + thaws)', ENTEI, BLISSEY, SEED, ONE_TURN, frz);
  await run('DF-c control: FROZEN Ho-Oh Flamethrower (non-defrost: cant)', HOOH, BLISSEY, SEED,
    [{ p1: 'move 2', p2: 'move 1' }], frz);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
