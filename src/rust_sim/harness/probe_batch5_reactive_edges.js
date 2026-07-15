// probe_batch5_reactive_edges.js — settle the two RECORDER edges the main batch-5 reactive
// probe (`probe_batch5_reactive.js`) did not cover, vs the OMNISCIENT gen3 BattleStream:
//
//   1. BEAT UP into a Counter user / a Mirror Coat user — Beat Up's strikes are typeless
//      '???' multihit hits whose `effect.category` is the resolved gen3 move category
//      (Dark → Special?). Which reactive move do they arm, and is the return 2x the LAST
//      strike (the multihit overwrite rule)?
//   2. STRUGGLE into a Counter user — Struggle is typeless '???' but category Physical;
//      the recorder checks `effect.category === 'Physical'` → recorded (2x)?
//
// Run:  node src/rust_sim/harness/probe_batch5_reactive_edges.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  const seed = (inject && inject.seed) || [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  for (const inj of (inject && inject.acts) || []) {
    const m = battle.sides[inj.side].active[0];
    if (inj.hp !== undefined) m.hp = inj.hp;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.zeropp) for (const s of m.moveSlots) s.pp = 0;
  }
  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let draws = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws++; return v; };

  let i = 0, safety = 0;
  while (!battle.ended && safety < 16) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    draws = 0;
    const logLen0 = log.length;
    const before = battle.prng.getSeed();
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp}` : '-';
    console.log(`  [${rs}] ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} draws=${draws} seed ${before}->${after}`);
    console.log(`        p1=${fmt(battle.sides[0].active[0])}  p2=${fmt(battle.sides[1].active[0])}`);
    for (const l of log.slice(logLen0)) {
      if (/\|move\||-damage|-fail|-immune|-miss|-crit|-hitcount|cant|faint/.test(l)) console.log(`        LINE ${l}`);
    }
    if (entry.stop) break;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  const d = Dex.forFormat(FORMAT);
  console.log(`resolved gen3 beatup category=${d.moves.get('beatup').category} struggle category=${d.moves.get('struggle').category}`);

  // Beat Up (5 healthy teammates -> 5+ strikes) into a COUNTER Snorlax and a MIRROR COAT
  // Blissey. Beat Up user slow (counter -5 anyway executes after).
  await run('BU-C: beatup into counter user',
    [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    [mon('Tyranitar', ['beatup', 'splash'], { ability: 'No Ability', evs: { hp: 252 } }),
     mon('Houndoom', ['splash']), mon('Sneasel', ['splash'])],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  await run('BU-M: beatup into mirrorcoat user',
    [mon('Blissey', ['mirrorcoat', 'splash'], { evs: { hp: 252 } })],
    [mon('Tyranitar', ['beatup', 'splash'], { ability: 'No Ability', evs: { hp: 252 } }),
     mon('Houndoom', ['splash']), mon('Sneasel', ['splash'])],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // Struggle into a Counter user (p2 has one 1-pp move, exhaust it first via Trick? —
  // simplest: give p2 ONLY splash with pp exhausted by many turns is slow; instead use a
  // 5-pp move and burn turns). Use Endeavor's trick: p2 mon with only 'tackle' — burn 35+
  // turns is too slow. Instead: inject pp=0 directly.
  await run('ST-C: struggle into counter user',
    [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['tackle'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 1, zeropp: true }] });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
