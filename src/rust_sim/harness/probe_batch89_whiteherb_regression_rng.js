// probe_batch89_whiteherb_regression_rng.js — GROUND TRUTH for the `gen3_white_herb_v1` WHITE HERB
// regression pins (WH1 self-drop restore + single-use / WH2 foe-Charm restore / WH3 net-positive
// no-trigger / WH4 lead-Intimidate construction restore / WH5 mid-battle-Intimidate run_switch
// restore) in tests/regression_test.rs.
//
// Drives the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED gen3customgame boards,
// RESEEDED to a RAW seed right before the first decision (matching the Rust's draw-free
// `start_with_switchins`), and prints each decision's seedAfter + both actives' 7 boost stages +
// held item. White Herb is DRAW-FREE, so a restore turn's seedAfter equals a no-White-Herb control's
// bit-for-bit (the boosts + item are the effect proof).
//
// Run:  node src/rust_sim/harness/probe_batch89_whiteherb_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }
const bs = (a) => a ? [a.boosts.atk||0,a.boosts.def||0,a.boosts.spa||0,a.boosts.spd||0,a.boosts.spe||0,a.boosts.accuracy||0,a.boosts.evasion||0].join(',') : '-';
const it = (a) => a ? JSON.stringify(a.item || '') : '-';

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
  // Report the POST-CONSTRUCTION state (dec -1) too — the lead-Intimidate restore fired here.
  {
    const a0 = b.sides[0].active[0], a1 = b.sides[1].active[0];
    console.log(`  [ctor] p1 boosts=[${bs(a0)}] item=${it(a0)} | p2 boosts=[${bs(a1)}] item=${it(a1)}`);
  }
  let i = 0;
  for (const entry of plan) {
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const a0 = b.sides[0].active[0], a1 = b.sides[1].active[0];
    console.log(`  dec ${i} ${JSON.stringify(entry)}`);
    console.log(`    seedAfter=${b.prng.getSeed()}`);
    console.log(`    p1 hp=${a0?a0.hp:'-'} boosts=[${bs(a0)}] item=${it(a0)} | p2 hp=${a1?a1.hp:'-'} boosts=[${bs(a1)}] item=${it(a1)}`);
    i++;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  const seed = [11, 29, 37, 53];
  // Teams: Species|nick|item|ability|moves|nature|evs(hp,atk,def,spa,spd,spe)|gender|ivs|...
  const whSnorlaxSP = "Snorlax||whiteherb|Own Tempo|superpower,splash|Adamant|252,252,4,,,|N||||";
  const noItemSnorlaxSP = "Snorlax|||Own Tempo|superpower,splash|Adamant|252,252,4,,,|N||||";
  const bulkyFoe = "Snorlax|||Own Tempo|splash|Careful|252,,252,,4,|N||||";
  const whSnorlaxBS = "Snorlax||whiteherb|Own Tempo|bodyslam,splash|Adamant|252,252,4,,,|N||||";
  const whSnorlaxSD = "Snorlax||whiteherb|Own Tempo|swordsdance,bodyslam,splash|Adamant|252,252,4,,,|N||||";
  const charmFoe = "Alakazam|||Own Tempo|charm,splash|Timid|4,,,,,252|N||||";
  const salamence = "Salamence|||Intimidate|earthquake,splash|Adamant|4,252,,,,252|N||||";
  const whSnorlaxSplash = "Snorlax||whiteherb|Own Tempo|splash|Careful|252,,252,,4,|N||||";

  // WH1: self-drop (Superpower) — dec0 restores Atk/Def to 0 + consumes; dec1 drops −1/−1 (single-use).
  //      The control (no White Herb) shares the seedAfter (the selfDrops random(100) draws either way).
  await run('WH1 self-drop restore + single-use', whSnorlaxSP, bulkyFoe, seed, [
    { p1: 'move 1', p2: 'move 1' },   // Superpower -> self -1/-1 -> WH restore
    { p1: 'move 1', p2: 'move 1' },   // Superpower -> self -1/-1 (item gone, no restore)
  ]);
  await run('WH1 control (no White Herb) — same seedAfter', noItemSnorlaxSP, bulkyFoe, seed, [
    { p1: 'move 1', p2: 'move 1' },
    { p1: 'move 1', p2: 'move 1' },
  ]);

  // WH2: foe Charm (−2 Atk) -> restore to 0 + consume (apply_secondary_boost path).
  await run('WH2 foe-Charm restore', whSnorlaxBS, charmFoe, seed, [
    { p1: 'move 2', p2: 'move 1' },   // p1 splash, p2 Charm -> p1 atk -2 -> WH restore
  ]);

  // WH3: net-positive — SD (+2) then Charm (−2) -> net 0, NO negative -> NO trigger, item RETAINED.
  await run('WH3 net-positive no-trigger', whSnorlaxSD, charmFoe, seed, [
    { p1: 'move 1', p2: 'move 2' },   // p1 SD +2, p2 splash
    { p1: 'move 3', p2: 'move 1' },   // p1 splash, p2 Charm -2 -> net 0, item RETAINED
  ]);

  // WH4: LEAD Intimidate -> construction restore (dec -1 shows p2 atk 0, item "").
  await run('WH4 lead-Intimidate construction restore', salamence, whSnorlaxSplash, seed, [
    { p1: 'move 1', p2: 'move 1' },   // p1 EQ, p2 splash (post-construction)
  ]);

  // WH5: MID-BATTLE Intimidate (run_switch) — p1 switches to a Salamence, its Intimidate drops p2's
  //      Atk -> WH restore. p1 = [filler Snorlax, Salamence]; p2 = WH Snorlax.
  const fillerThenMence = "Snorlax|||Own Tempo|splash|Adamant|252,,252,,4,|N||||]Salamence|||Intimidate|earthquake,splash|Adamant|4,252,,,,252|N||||";
  await run('WH5 mid-battle-Intimidate run_switch restore', fillerThenMence, whSnorlaxSplash, seed, [
    { p1: 'switch 2', p2: 'move 1' }, // p1 -> Salamence (Intimidate drops p2 atk -> WH restore), p2 splash
  ]);
}
main();
