// probe_perish_double_faint_order.js — settle the gen3 PERISH-0 emission + faint ORDER on a
// MUTUAL perish-out, against the OMNISCIENT in-process BattleStream. The sim is the ONLY oracle.
//
// WHY: the live external-consistency gate found `soak3/divergences/sbd_msb1zfxs_b134`:
//     node: |-start|p2a: Aipom|perish0  then  |-start|p1a: Misdreavus|perish0
//           |faint|p2a: Aipom           then  |faint|p1a: Misdreavus
//     port: p1a FIRST in BOTH.
// Crucially the perish3 / perish2 / perish1 ticks match EXACTLY in both engines (both p2a-first),
// so ONLY the FINAL, duration-ENDING tick reorders. That points at the duration-end branch rather
// than the residual speed-sort: batch 6 models "the perish `onEnd` faint is ENQUEUED but the
// per-handler `faintMessages` is SKIPPED (the fieldEvent duration-end `continue`)", and the emit
// order then comes from the enqueue order rather than the handler order.
//
// WHAT THIS PINS (per tick, for both sides):
//   * the ORDER of the `|-start|<mon>|perish<N>` lines at every tick 3..0, and
//   * the ORDER of the resulting `|faint|` lines at tick 0,
// across three speed relationships, because the residual handler sort is speed-ordered and the
// question is whether tick 0 still follows it:
//   A. p1 FASTER than p2
//   B. p2 FASTER than p1   (the b134 shape — Aipom 239 spe vs Misdreavus 209)
//   C. a SPEED TIE          (the shuffle case — both orders are legal, so this one is only
//                            reported, never asserted; it is here to show whether tick 0 is
//                            shuffle-driven at all)
//
// Run:  node src/rust_sim/harness/probe_perish_double_faint_order.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, spe) {
  return {
    species, item: '', ability: 'No Ability', moves,
    evs: { ...EV0, spe }, ivs: IV31, nature: 'Serious', level: 100, gender: 'N',
  };
}
const tick = () => new Promise((r) => setTimeout(r, 0));

// Both sides carry Perish Song + Splash. One cast puts the counter on BOTH actives, so a mutual
// perish-out follows 3 turns later with neither side able to escape (single-mon teams).
async function run(label, p1spe, p2spe) {
  const p1 = [mon('Misdreavus', ['perishsong', 'splash'], p1spe)];
  const p2 = [mon('Aipom', ['perishsong', 'splash'], p2spe)];

  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify([7, 11, 13, 17])}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const spe = (s) => battle.sides[s].active[0].getStat('spe');
  const rel = spe(0) > spe(1) ? 'p1 faster' : spe(0) < spe(1) ? 'p2 faster' : 'TIE';

  const plan = [{ p1: 'move perishsong', p2: 'move splash' },
                { p1: 'move splash', p2: 'move splash' },
                { p1: 'move splash', p2: 'move splash' },
                { p1: 'move splash', p2: 'move splash' },
                { p1: 'move splash', p2: 'move splash' }];
  for (const step of plan) {
    if (battle.ended) break;
    if (battle.requestState === 'move') {
      if (step.p1) streams.omniscient.write(`>p1 ${step.p1}`);
      if (step.p2) streams.omniscient.write(`>p2 ${step.p2}`);
    }
    for (let k = 0; k < 20; k++) await tick();
  }

  const perish = log.filter((l) => /\|-start\|p[12]a: \w+\|perish\d/.test(l))
    .map((l) => `${l.match(/p([12])a/)[1]}:${l.match(/perish(\d)/)[1]}`);
  const faints = log.filter((l) => /^\|faint\|/.test(l)).map((l) => l.match(/p([12])a/)[1]);
  // group the perish marks by tick number
  const byTick = {};
  for (const p of perish) { const [s, n] = p.split(':'); (byTick[n] = byTick[n] || []).push(s); }
  console.log(`\n=== ${label} ===  spe p1=${spe(0)} p2=${spe(1)} (${rel})`);
  for (const n of ['3', '2', '1', '0']) {
    if (byTick[n]) console.log(`    perish${n}: emitted for side(s) [${byTick[n].join(' then ')}]`);
  }
  console.log(`    faint order: [${faints.join(' then ')}]   ended=${battle.ended} winner=${battle.winner || '(tie)'}`);
}

(async () => {
  await run('A p1 FASTER', 252, 0);
  await run('B p2 FASTER  <-- the b134 shape', 0, 252);
  await run('C SPEED TIE (reported only — both orders legal)', 0, 0);
})();
