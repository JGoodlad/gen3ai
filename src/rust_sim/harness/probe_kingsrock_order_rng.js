// probe_kingsrock_order_rng.js — the two ORDER cruxes probe_kingsrock_rng.js left open:
//  O1: a LANDED Muddy Water (own 30% acc-drop secondary) + King's Rock — TWO trailing
//      rolls, own secondary FIRST then the appended KR roll (list order).
//  O2: the TRIPLE order on one hit: own secondary + KR secondary + the DEFENDER's
//      contact proc (Static) — [own sec][KR sec][contact proc].
//      Use a listed CONTACT move with its own secondary: Crunch? (not listed). Use
//      DragonBreath (listed, 30% par, non-contact) for O1b and Slash(contact? no).
//      Contact+listed+own-secondary: 'volttackle' (listed, contact? gen3 no volttackle
//      secondaries?) — use Slash (listed, contact, NO own sec) vs Static for the
//      [KR][proc] pair order, and Muddy Water for [own][KR].
// Run: node src/rust_sim/harness/probe_kingsrock_order_rng.js
'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

async function main() {
  console.log('=== O1: landed Muddy Water + KR: own random(100) then KR random(100)');
  const mw = [
    [mon('Zangoose', ['muddywater'], { ability: 'Immunity', item: "King's Rock", evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Thick Fat' })],
  ];
  for (let s = 1; s <= 14; s++) {
    const seed = [s, s * 3 + 1, s * 5 + 2, s * 7 + 3];
    const r = await run(mw, seed, Array(1).fill(['move 1', 'move 1']));
    const d = r.perDecision[0];
    const hit = d.lines.some((l) => l.includes('|-damage|'));
    if (hit) {
      const cant = d.lines.filter((l) => l.includes('|cant|') || l.includes('unboost'));
      console.log(`  seed=${s} HIT: [${fmtCalls(d.calls)}] ev=${JSON.stringify(cant)}`);
    }
  }
  console.log('=== O2: Slash (listed, contact, no own sec) + KR into STATIC: [KR roll] then [proc roll]');
  const st = [
    [mon('Zangoose', ['slash'], { ability: 'Immunity', item: "King's Rock", evs: { spe: 252 } })],
    [mon('Electrode', ['splash'], { ability: 'Static', evs: { hp: 252 } })],
  ];
  for (let s = 1; s <= 6; s++) {
    const seed = [s, s + 1, s + 2, s + 3];
    const r = await run(st, seed, Array(1).fill(['move 1', 'move 1']));
    console.log(`  seed=${s}: [${fmtCalls(r.perDecision[0].calls)}]`);
  }
  console.log('=== O3: DragonBreath (listed, own 30% par) + KR: [own][KR] both trailing');
  const db = [
    [mon('Salamence', ['dragonbreath'], { ability: 'Intimidate', item: "King's Rock", evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Thick Fat' })],
  ];
  for (let s = 1; s <= 6; s++) {
    const seed = [s * 2, s * 3, s * 5, s * 7];
    const r = await run(db, seed, Array(1).fill(['move 1', 'move 1']));
    const ev = r.perDecision[0].lines.filter((l) => l.includes('-status') || l.includes('|cant|'));
    console.log(`  seed=${s}: [${fmtCalls(r.perDecision[0].calls)}] ev=${JSON.stringify(ev)}`);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
