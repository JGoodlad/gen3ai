// probe_residual_abilities.js — settle the RESIDUAL ability class (Speed Boost / Rain Dish)
// draw model + the exact resolved onResidual order/subOrder, so the port schedules them at
// the right place in the residual ladder (relative to Leftovers sub 4 / leech sub 5 / status
// DoT sub 6) and draws nothing extra.
//
// It dumps, for a Speed-Boost mon and a Rain-Dish mon (in rain), the resolved residual
// handler list (findEventHandlers) with each handler's order/subOrder + whether the residual
// turn draws more than a no-op control (both should be DRAW-FREE — a deterministic +1 spe /
// heal). Speed Boost feeds NEXT turn's cached speed.
//
// Run: node src/rust_sim/harness/probe_residual_abilities.js

'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Dex, Teams } = require(path.join(PS, 'dist/sim'));

const d3 = Dex.mod('gen3');
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: 'Serious', level: 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// 1) Dump the resolved onResidual order/subOrder from the ability data directly.
console.log('=== resolved onResidual order/subOrder (from Dex.mod("gen3")) ===');
for (const id of ['speedboost', 'raindish', 'shedskin', 'leftovers']) {
  const e = id === 'leftovers' ? d3.items.get(id) : d3.abilities.get(id);
  console.log(`  ${id.padEnd(12)} onResidualOrder=${e.onResidualOrder} onResidualSubOrder=${e.onResidualSubOrder}`);
}

async function run(p2ability, weatherSetter, seed, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of String(ch).split('\n')) lines.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  const p1 = weatherSetter || mon('Snorlax', ['recover', 'recover'], { ability: 'Shell Armor' });
  // p2: the probe mon. Damage it first so a heal is observable (Surf from p1 if a setter).
  const p2 = mon(opts.species || 'Ludicolo', ['rest', 'rest'], { ability: p2ability });
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([p1]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([p2]) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  let n = 0; rng.next = (...a) => { n += 1; return realNext(...a); };
  const per = [];
  const speTrace = [];
  const hpTrace = [];
  for (let t = 0; t < 4; t++) {
    const b = n;
    streams.omniscient.write('>p1 move 1');
    streams.omniscient.write('>p2 move 1');
    for (let k = 0; k < 10; k++) await tick();
    per.push(n - b);
    speTrace.push(battle.sides[1].active[0].boosts.spe || 0);
    hpTrace.push(battle.sides[1].active[0].hp);
  }
  return { totalDraws: n, per, speTrace, hpTrace, maxhp: battle.sides[1].active[0].maxhp };
}

(async () => {
  const seeds = [[1, 2, 3, 4], [7, 11, 13, 17], [5, 5, 5, 5]];
  console.log('\n=== Speed Boost residual: +1 spe/turn (draw-free), feeds NEXT turn speed ===');
  for (const seed of seeds) {
    const sb = await run('Speed Boost', null, seed, { species: 'Ninjask' });
    const ctl = await run('Shell Armor', null, seed, { species: 'Ninjask' });
    console.log(`  seed ${JSON.stringify(seed)}: SpeedBoost per=${JSON.stringify(sb.per)} speBoosts=${JSON.stringify(sb.speTrace)} | ctl per=${JSON.stringify(ctl.per)} spe=${JSON.stringify(ctl.speTrace)} | drawMatch=${JSON.stringify(sb.per) === JSON.stringify(ctl.per)}`);
  }
  console.log('\n=== Rain Dish residual: heal maxhp/16 in rain (draw-free) ===');
  const rain = mon('Kyogre', ['surf', 'surf'], { ability: 'Drizzle' });
  for (const seed of seeds) {
    const rd = await run('Rain Dish', rain, seed, { species: 'Ludicolo' });
    const ctl = await run('Shell Armor', rain, seed, { species: 'Ludicolo' });
    console.log(`  seed ${JSON.stringify(seed)}: RainDish per=${JSON.stringify(rd.per)} hp=${JSON.stringify(rd.hpTrace)}/${rd.maxhp} | ctl per=${JSON.stringify(ctl.per)} hp=${JSON.stringify(ctl.hpTrace)} | drawMatch=${JSON.stringify(rd.per) === JSON.stringify(ctl.per)}`);
  }
  console.log('\n(Draw-free if per-turn counts match the control; the STATE — +1 spe / +maxhp/16 — is the effect.)');
})();
