// probe_intimidate_substitute_rng.js — GROUND TRUTH for the Intimidate-vs-SUBSTITUTE gate
// (`gen3_trapping_v1`'s e2e regen surfaced it: e2e_171 / e2e_204 — a mon that Substituted
// the turn before an Intimidate switch-in kept Atk 0 in the sim while the port dropped -1).
//
// SETTLES (vs the omniscient in-process BattleStream — the sim is the oracle):
//   1. gen-3 Intimidate does NOT drop the Atk of a foe behind a SUBSTITUTE (the gen3 mod's
//      per-foe substitute skip): sub up -> NO |-unboost|, boosts unchanged.
//   2. The block is DRAW-FREE + SEED-NEUTRAL: the sub scenario and the no-sub control draw
//      IDENTICAL counts and land on IDENTICAL seeds every turn (boost() consumes no PRNG
//      either way) -> the fix is STATE-only, no seed suite can shift.
//
// CONFIRMED (run this file):
//   A (sub up):  switch-in turn draws=1, p1 atkBoost=0,  seedAfter=41762,18770,8812,43906
//   B (control): switch-in turn draws=1, p1 atkBoost=-1, seedAfter=41762,18770,8812,43906
//   (identical seeds all three turns; only the boost differs)
//
// Pin: tests/regression_test.rs::intimidate_into_a_substitute_is_a_noop (STATE pin).
//
// Run:  node src/rust_sim/harness/probe_intimidate_substitute_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }
async function run(label, subFirst) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[3,5,7,9]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } })]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([mon('Regice', ['icebeam', 'splash'], { evs: { hp: 252 } }), mon('Salamence', ['dragonclaw', 'splash'], { ability: 'Intimidate' })]) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };
  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  const plan = subFirst
    ? [{ p1: 'move 1', p2: 'move 2' }, { p1: 'move 2', p2: 'switch 2' }, { p1: 'move 2', p2: 'move 2' }]
    : [{ p1: 'move 2', p2: 'move 2' }, { p1: 'move 2', p2: 'switch 2' }, { p1: 'move 2', p2: 'move 2' }];
  for (const e of plan) {
    const dc0 = drawCount;
    const l0 = log.length;
    streams.omniscient.write(`>p1 ${e.p1}`);
    streams.omniscient.write(`>p2 ${e.p2}`);
    for (let k = 0; k < 16; k++) await tick();
    const a0 = battle.sides[0].active[0];
    console.log(`  ${JSON.stringify(e)} draws=${drawCount - dc0} p1 atkBoost=${a0.boosts.atk} sub=${!!a0.volatiles.substitute} seedAfter=${battle.prng.getSeed()}`);
    for (const l of log.slice(l0)) if (/-unboost|-fail|-activate|-ability/.test(l)) console.log(`      ${l}`);
  }
}
async function main() {
  await run('A: Snorlax SUBS first, then the Intimidate Salamence switches in (sub up -> NO drop)', true);
  await run('B: CONTROL no sub — the Intimidate switch-in DOES drop Atk -1', false);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
