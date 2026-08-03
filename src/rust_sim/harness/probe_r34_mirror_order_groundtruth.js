// probe_r34_mirror_order_groundtruth.js — ORACLE ground truth for the ROUND-34 regression
// test (`gen3_turn0_construction_mirror_order_v1`).
//
// For a same-species Intimidate MIRROR at a speed-TIED lead, the turn-0 `insertChoice`
// tie window is broken by ONE `random(firstIndex, lastIndex+1)` draw, so the two leads'
// `runSwitch` actions — and hence the emitted `-ability|…|Intimidate|boost` block — fire
// p1-FIRST or p2-FIRST depending on the seed. The rust regression test needs BOTH cases
// (a test that only ever sees one order would pass on a hard-coded flip), so this script
// scans candidate `>start` seeds against the REAL sim and prints, per seed:
//   * which side's Intimidate is emitted FIRST,
//   * the POST-construction PRNG seed (the extra pin the rust test asserts), and
//   * a DISTINCT-speed control (the fallback ordering the draw-free path also uses).
//
// Run: node src/rust_sim/harness/probe_r34_mirror_order_groundtruth.js

'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream'));

const MIRROR = 'Masquerain|||Intimidate|splash|Serious||M||||';
// A distinct-speed control: Masquerain (spe 156 @0 EVs) vs a FASTER Intimidate mon.
const MASQ = 'Masquerain|||Intimidate|splash|Serious||M||||';
const TAUROS_SLOW = 'Tauros|||Intimidate|splash|Serious|252,,,,,|M||||';
// A MIXED weather-setter tie (both spe 216): the LAST setter to fire wins the field, so the
// drawn order shows up in BOTH the `-weather` line order AND `field.weather` — the
// emission-vs-state consistency the mirror case (symmetric by construction) cannot test.
const KYOGRE = 'Kyogre|||Drizzle|splash|Serious||N||||';
const GROUDON = 'Groudon|||Drought|splash|Serious||N||||';

function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(p1, p2, seed) {
  const stream = new BattleStream();
  const lines = [];
  (async () => { for await (const ch of stream) for (const l of String(ch).split('\n')) if (l) lines.push(l); })();
  stream.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  stream.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  stream.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 20; i++) await tick();
  const b = stream.battle;
  const abil = lines.filter((l) => /^\|-ability\|p\da: .*\|Intimidate\|boost$/.test(l));
  const order = abil.map((l) => l.split('|')[2].slice(0, 3));
  return {
    order,
    weatherLines: lines.filter((l) => l.startsWith('|-weather|')),
    weather: b.field.weather,
    postSeed: String(b.prng.getSeed()),
    spe: [b.sides[0].active[0].storedStats.spe, b.sides[1].active[0].storedStats.spe],
  };
}

(async () => {
  console.log('=== MIRROR (Masquerain/Masquerain, Intimidate, speed TIE) ===');
  for (let i = 1; i <= 8; i++) {
    const seed = [i, 2, 3, 4];
    const r = await run(MIRROR, MIRROR, seed);
    if (r.order.length !== 2) { console.log(`  seed=${JSON.stringify(seed)}  !! ${r.order.length} Intimidate lines`); continue; }
    if (r.spe[0] !== r.spe[1]) { console.log(`  seed=${JSON.stringify(seed)}  !! not speed-tied ${r.spe}`); continue; }
    console.log(`  seed=${JSON.stringify(seed)}  first=${r.order[0]}  order=${r.order.join(',')}  postSeed=${r.postSeed}  spe=${r.spe[0]}`);
  }
  console.log('\n=== DISTINCT SPEED control (Masquerain vs Tauros, both Intimidate) ===');
  const c = await run(MASQ, TAUROS_SLOW, [1, 2, 3, 4]);
  console.log(`  seed=[1,2,3,4]  first=${c.order[0]}  order=${c.order.join(',')}  postSeed=${c.postSeed}  spe=${c.spe.join('/')}`);

  console.log('\n=== MIXED WEATHER-SETTER tie (p1 Kyogre Drizzle vs p2 Groudon Drought) ===');
  for (const seed of [[1, 2, 3, 4], [3, 2, 3, 4]]) {
    const w = await run(KYOGRE, GROUDON, seed);
    if (w.spe[0] !== w.spe[1]) { console.log(`  seed=${JSON.stringify(seed)}  !! not speed-tied ${w.spe}`); continue; }
    if (w.weatherLines.length !== 2) { console.log(`  seed=${JSON.stringify(seed)}  !! ${w.weatherLines.length} weather lines`); continue; }
    console.log(`  seed=${JSON.stringify(seed)}  spe=${w.spe[0]}  field.weather=${w.weather}  postSeed=${w.postSeed}`);
    for (const l of w.weatherLines) console.log(`     ${l}`);
  }
})().catch((e) => { console.error(e); process.exit(1); });
