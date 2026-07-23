// probe_batch89_trick_regression_rng.js — GROUND TRUTH for the `gen3_trick_v1` TRICK regression pins
// (TR1 two-item swap, TR2 Sticky-Hold `-immune`, TR3 both-itemless fail, TR4 Choice-Band trick-away
// lock RELEASE, TR5 Substitute block) in tests/regression_test.rs.
//
// Drives the OMNISCIENT in-process BattleStream (no server), RESEEDED to a RAW seed right before the
// first decision (matching the Rust's draw-free `start_with_switchins`), and prints each decision's
// seedAfter + both actives' item id + a swap/immune/fail marker. Trick draws ONE accuracy roll then a
// DRAW-FREE swap: a swap, a Sticky-Hold `-immune`, and a both-itemless / substitute FAIL all draw the
// SAME count (accuracy + endTurn Quick Claw), so their post-turn seeds coincide at the same init.
//
// Run:  node src/rust_sim/harness/probe_batch89_trick_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, format, p1, p2, rawSeed, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${format}","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  b.prng = new PRNG(rawSeed.slice());
  console.log(`\n=== ${label} [${format}] (raw seed ${rawSeed.join(',')}) ===`);
  let i = 0;
  for (const entry of plan) {
    const before = log.length;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const chunk = log.slice(before);
    const swap = chunk.some((l) => l.includes('move: Trick'));
    const immune = chunk.some((l) => l.startsWith('|-immune|'));
    const fail = chunk.some((l) => l.startsWith('|-fail|'));
    const a0 = b.sides[0].active[0], a1 = b.sides[1].active[0];
    console.log(`  dec ${i} ${JSON.stringify(entry)} swap=${swap} immune=${immune} fail=${fail}`);
    console.log(`    seedAfter=${b.prng.getSeed()}`);
    console.log(`    p1 item=${a0 ? a0.item || '""' : '-'} | p2 item=${a1 ? a1.item || '""' : '-'}`);
    i++;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  const zam = (moves, item) => `Alakazam||${item}|Synchronize|${moves}|Timid|4,,,252,,252|N||||`;
  const seed = [23, 44, 61, 82];

  // TR1: a full two-item swap (Silk Scarf <-> Leftovers). DRAW-FREE past accuracy.
  await run('TR1 two-item swap', 'gen3customgame',
    zam('trick,splash', 'silkscarf'),
    'Snorlax||leftovers|Immunity|splash|Careful|252,,252,,,|N||||',
    seed, [{ p1: 'move 1', p2: 'move 1' }]);

  // TR2: Sticky Hold -> PLAIN -immune, NO swap. SAME draw count (accuracy + Quick Claw) as TR1.
  await run('TR2 sticky-hold immune', 'gen3customgame',
    zam('trick,splash', 'silkscarf'),
    'Muk||leftovers|StickyHold|splash|Careful|252,,252,,,|N||||',
    seed, [{ p1: 'move 1', p2: 'move 1' }]);

  // TR3: both itemless -> FAIL ([still]+-fail), NO swap. SAME draw count as TR1/TR2.
  await run('TR3 both itemless fail', 'gen3customgame',
    zam('trick,splash', ''),
    'Snorlax|||Immunity|splash|Careful|252,,252,,,|N||||',
    seed, [{ p1: 'move 1', p2: 'move 1' }]);

  // TR4: a CHOICE BAND user Tricks its Band AWAY, then uses a DIFFERENT slot (Psychic) next turn —
  //   the lock is RELEASED (the sim offers all moves). dec0 swap, dec1 Psychic runs.
  await run('TR4 CB trick-away lock release', 'gen3customgame',
    zam('trick,psychic,splash', 'choiceband'),
    'Snorlax||leftovers|Immunity|splash|Careful|252,,252,,,|N||||',
    seed, [
      { p1: 'move 1', p2: 'move 1' },  // dec0: Trick (CB -> Snorlax, Leftovers -> Alakazam)
      { p1: 'move 2', p2: 'move 1' },  // dec1: Psychic — a DIFFERENT slot (the lock was released)
    ]);

  // TR5: the foe SUBSTITUTES then p1 Tricks into the sub -> [still]+-fail, NO swap.
  await run('TR5 substitute block', 'gen3customgame',
    zam('trick,splash', 'silkscarf'),
    'Snorlax||leftovers|Immunity|substitute,splash|Careful|252,,252,,,|N||||',
    seed, [
      { p1: 'move 2', p2: 'move 1' },  // dec0: p1 Splash / p2 Substitute
      { p1: 'move 1', p2: 'move 2' },  // dec1: p1 Trick into the sub (fail) / p2 Splash
    ]);
}
main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
