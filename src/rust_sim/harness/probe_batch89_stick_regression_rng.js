// probe_batch89_stick_regression_rng.js — GROUND TRUTH for the `gen3_crit_item_v1` STICK
// regression pins (CI1 Stick crits / CI2 the no-item control does not) in
// tests/regression_test.rs.
//
// Drives the OMNISCIENT in-process BattleStream (no server) over a CONSTRUCTED gen3customgame
// board — a Farfetch'd (Keen Eye) using Cut vs a Snorlax (Immunity) — RESEEDED to a RAW seed
// right before the first decision (matching the Rust's draw-free `start_with_switchins`). It
// sweeps seeds for one where the Stick Farfetch'd CRITS (critRatio 1+2=3 → 1/4) but the SAME
// board WITHOUT the item does NOT crit (critRatio 1 → 1/16), proving the +2 species-gated fold
// lands. The crit roll draws ONE `random(denom)` either way (same PRNG consumption), so the
// two boards share the SAME post-turn seedAfter — the draw-free-denominator-shift proof.
//
// Run:  node src/rust_sim/harness/probe_batch89_stick_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(t1, t2, rawSeed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: t1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: t2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  b.prng = new PRNG(rawSeed.slice());
  const before = log.length;
  streams.omniscient.write(`>p1 move 1`);
  streams.omniscient.write(`>p2 move 1`);
  for (let k = 0; k < 18; k++) await tick();
  const crit = log.slice(before).some((l) => l.startsWith('|-crit|p2a'));
  const def = b.sides[1].active[0];
  const out = { crit, p2hp: `${def.hp}/${def.maxhp}`, seedAfter: String(b.prng.getSeed()) };
  try { streams.omniscient.destroy(); } catch (e) {}
  return out;
}

async function main() {
  const stick = "Farfetch'd|||KeenEye|cut,rest|Adamant|,252,,,,252|N||||";
  const noitem = "Farfetch'd|||KeenEye|cut,rest|Adamant|,252,,,,252|N||||"; // item added below
  const snorlax = "Snorlax|||Immunity|bodyslam,rest|Careful|252,,,,252,|N||||";
  // add @ Stick to the stick set
  const stickSet = stick.replace("Farfetch'd||", "Farfetch'd||stick");
  for (let s = 1; s < 400; s++) {
    const seed = [s * 7 + 1, s * 13 + 2, s * 5 + 4, s * 11 + 3];
    const rs = await run(stickSet, snorlax, seed);
    const rc = await run(noitem, snorlax, seed);
    if (rs.crit && !rc.crit && rs.seedAfter === rc.seedAfter) {
      console.log(`FOUND seed=${seed.join(',')}`);
      console.log(`  STICK:   crit=${rs.crit} p2hp=${rs.p2hp} seedAfter=${rs.seedAfter}`);
      console.log(`  CONTROL: crit=${rc.crit} p2hp=${rc.p2hp} seedAfter=${rc.seedAfter}`);
      return;
    }
  }
  console.log('no qualifying seed found in sweep');
}
main();
