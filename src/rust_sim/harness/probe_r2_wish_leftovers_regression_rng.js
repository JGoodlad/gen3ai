// probe_r2_wish_leftovers_regression_rng.js — GROUND-TRUTH for the R2 Leftovers
// residual-tie -heal EMIT ORDER pin (`gen3_leftovers_slotcond_gather_order_v1`).
//
// A Jolteon mirror (both Leftovers, equal speed) under sandstorm, with a PENDING p2 Wish
// (order-7 slot condition). At the residual the pre-sort handler array is
// [Sandstorm(o8), Leftovers_p1(o10), Leftovers_p2(o10), Wish_p2(o7)] — and the sim's
// NON-STABLE selection sort's swaps reverse the tied Leftovers pair, so the two -heal lines
// emit in an order that depends on where Wish sits in the pre-sort array. The port used to
// gather Wish FIRST (a pre-loop) → the OPPOSITE permutation at the SAME shuffle value.
// This finds a seed where the sim heals p1a-first (pre-fix the port healed p2a-first) and
// prints the seedAfter + the -heal order.
//   node src/rust_sim/harness/probe_r2_wish_leftovers_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve('/home/goodlad/dev/gen3ai/deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: 100, gender: 'N',
  };
}
const tick = () => new Promise((r) => setTimeout(r, 0));

async function run(seed, printPacked) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) if (l) log.push(l); })();
  const p1 = [
    mon('Tyranitar', ['splash'], { ability: 'Sand Stream', item: 'Leftovers' }),
    mon('Jolteon', ['splash'], { ability: 'Volt Absorb', item: 'Leftovers', evs: { hp: 4 } }),
  ];
  const p2 = [
    mon('Tyranitar', ['splash'], { ability: 'Sand Stream', item: 'Leftovers' }),
    mon('Jolteon', ['wish', 'splash'], { ability: 'Volt Absorb', item: 'Leftovers', evs: { hp: 4 } }),
  ];
  if (printPacked) { console.log('P1:', Teams.pack(p1)); console.log('P2:', Teams.pack(p2)); }
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const initSeed = battle.prng.getSeed();
  // Turn1: both switch to Jolteon. Turn2: p1 splash, p2 Wish (leaves a pending p2 Wish).
  const plan = [{ p1: 'switch 2', p2: 'switch 2' }, { p1: 'move 1', p2: 'move 1' }];
  let i = 0, safety = 0, heals = null, seedAfter = null;
  while (!battle.ended && safety < 30) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const llen = log.length;
    const e = plan[Math.min(i, plan.length - 1)];
    i++;
    if (e.p1) streams.omniscient.write(`>p1 ${e.p1}`);
    if (e.p2) streams.omniscient.write(`>p2 ${e.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    seedAfter = battle.prng.getSeed();
    if (i === 2) heals = log.slice(llen).filter((l) => /-heal\|.*Leftovers/.test(l)).map((l) => (l.includes('p1a') ? 'p1' : 'p2'));
    if (i >= plan.length) break;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
  return { initSeed, seedAfter, heals };
}

async function main() {
  await run([1, 1, 1, 1], true);
  // Scan seeds; report ones where the turn-2 residual heals p1-first (the R2 pin premise).
  for (const s of [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [7, 7, 7, 7], [9, 9, 9, 9], [1, 2, 3, 4], [5, 9, 1, 3]]) {
    const r = await run(s);
    console.log(`seed=${JSON.stringify(s)} initSeed=${r.initSeed} heals=${JSON.stringify(r.heals)} seedAfter=${r.seedAfter}`);
  }
}
main();
