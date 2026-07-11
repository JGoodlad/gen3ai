// probe_kingsrock_rng.js — settle KING'S ROCK vs the resolved gen3 sim. Hypotheses
// (resolved dist onModifyMove, priority -1): for move.id in a FIXED 100+-move list,
// PUSH {chance:10, volatileStatus:'flinch'} onto move.secondaries — i.e. it becomes an
// ORDINARY trailing secondary: rolled in secondaries() AFTER the move's own secondary,
// BEFORE the foe's contact proc (DamagingHit); Serene Grace doubles it (20); Shield
// Dust (defender) filters it (no draw); behind a Sub it draws but doesn't apply (the
// gen3 quirk); Inner Focus draws + blocks at apply. NON-listed moves gain nothing.
// Run: node src/rust_sim/harness/probe_kingsrock_rng.js

'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

async function main() {
  console.log('=== listed no-secondary move (Slash) + item control: the extra roll + flinch cant');
  const mk = (item, defAb) => [
    [mon('Zangoose', ['slash', 'tackle', 'muddywater'], { ability: 'Immunity', item, evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: defAb || 'Thick Fat' })],
  ];
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]) {
    for (const item of ["King's Rock", '']) {
      const r = await run(mk(item), seed, Array(2).fill(['move 1', 'move 1']));
      r.perDecision.forEach((d, i) => {
        const cant = d.lines.filter((l) => l.includes('|cant|'));
        console.log(`slash item=${item || 'none'} seed=${seed[0]} t${i + 1}: [${fmtCalls(d.calls)}] cant=${JSON.stringify(cant)}`);
      });
    }
  }

  console.log('=== NON-listed move (Tackle): no extra roll');
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2]]) {
    const r = await run(mk("King's Rock"), seed, Array(1).fill(['move 2', 'move 1']));
    console.log(`tackle seed=${seed[0]}: [${fmtCalls(r.perDecision[0].calls)}]`);
  }

  console.log('=== listed move WITH own secondary (Muddy Water, 30% acc-drop): TWO rolls?');
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2]]) {
    const r = await run(mk("King's Rock"), seed, Array(1).fill(['move 3', 'move 1']));
    console.log(`muddywater seed=${seed[0]}: [${fmtCalls(r.perDecision[0].calls)}]`);
  }

  console.log('=== Serene Grace attacker: the KR chance doubled? (roll args/threshold)');
  const sg = (item) => [
    [mon('Blissey', ['slash'], { ability: 'Serene Grace', item, evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Thick Fat' })],
  ];
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2], [5, 5, 5, 5], [7, 7, 7, 7], [11, 11, 11, 11], [13, 13, 13, 13]]) {
    const r = await run(sg("King's Rock"), seed, Array(1).fill(['move 1', 'move 1']));
    const cant = r.perDecision[0].lines.filter((l) => l.includes('|cant|'));
    console.log(`SG seed=${seed[0]}: [${fmtCalls(r.perDecision[0].calls)}] cant=${JSON.stringify(cant)}`);
  }

  console.log('=== Shield Dust defender: KR roll suppressed?');
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2]]) {
    const r = await run(mk("King's Rock", 'Shield Dust'), seed, Array(1).fill(['move 1', 'move 1']));
    console.log(`shielddust seed=${seed[0]}: [${fmtCalls(r.perDecision[0].calls)}]`);
  }

  console.log('=== Inner Focus defender: KR roll drawn, flinch blocked');
  for (const seed of [[3, 3, 3, 3], [4, 4, 4, 4]]) {
    const r = await run(mk("King's Rock", 'Inner Focus'), seed, Array(2).fill(['move 1', 'move 1']));
    r.perDecision.forEach((d, i) => {
      const cant = d.lines.filter((l) => l.includes('|cant|'));
      console.log(`innerfocus seed=${seed[0]} t${i + 1}: [${fmtCalls(d.calls)}] cant=${JSON.stringify(cant)}`);
    });
  }

  console.log('=== behind a Sub: KR roll drawn? flinch applied?');
  const subT = [
    [mon('Zangoose', ['slash'], { ability: 'Immunity', item: "King's Rock", evs: { spe: 252 } })],
    [mon('Snorlax', ['substitute', 'splash'], { ability: 'Thick Fat' })],
  ];
  for (const seed of [[3, 3, 3, 3], [4, 4, 4, 4]]) {
    const r = await run(subT, seed, [['move 1', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 2']]);
    r.perDecision.forEach((d, i) => {
      const cant = d.lines.filter((l) => l.includes('|cant|'));
      console.log(`sub seed=${seed[0]} t${i + 1}: [${fmtCalls(d.calls)}] cant=${JSON.stringify(cant)}`);
    });
  }

  console.log('=== listed FIXED-damage move (Seismic Toss) + KR');
  const st = [
    [mon('Zangoose', ['seismictoss'], { ability: 'Immunity', item: "King's Rock", evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Thick Fat' })],
  ];
  for (const seed of [[1, 1, 1, 1], [6, 6, 6, 6]]) {
    const r = await run(st, seed, Array(1).fill(['move 1', 'move 1']));
    const cant = r.perDecision[0].lines.filter((l) => l.includes('|cant|'));
    console.log(`seismictoss seed=${seed[0]}: [${fmtCalls(r.perDecision[0].calls)}] cant=${JSON.stringify(cant)}`);
  }

  console.log('=== KR flinch when the holder moves SECOND (flinch on an already-moved foe = inert)');
  const slow = [
    [mon('Snorlax', ['slash'], { ability: 'Thick Fat', item: "King's Rock" })],
    [mon('Zangoose', ['splash'], { ability: 'Immunity', evs: { spe: 252 } })],
  ];
  for (const seed of [[1, 1, 1, 1], [3, 3, 3, 3]]) {
    const r = await run(slow, seed, Array(2).fill(['move 1', 'move 1']));
    r.perDecision.forEach((d, i) => {
      const cant = d.lines.filter((l) => l.includes('|cant|'));
      console.log(`slowKR seed=${seed[0]} t${i + 1}: [${fmtCalls(d.calls)}] cant=${JSON.stringify(cant)}`);
    });
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
