// probe_phaze_actions.js — log the runAction sequence + each draw's action context.
// Wraps battle.runAction (via the queue) + draws, to map WHICH action each shuffle belongs to.
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }
async function run(label, seed, p1team, p2team, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) {} })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 10; i++) await tick();
  const battle = stream.battle;
  let logging = false;
  const realRunAction = battle.runAction.bind(battle);
  battle.runAction = function (action) {
    if (logging) console.log(`    >> runAction ${action.choice}${action.pokemon ? ' ' + action.pokemon : ''}${action.move ? ' ' + action.move.id : ''}`);
    return realRunAction(action);
  };
  for (const fn of ['random', 'randomChance', 'sample']) {
    const real = battle.prng[fn].bind(battle.prng);
    battle.prng[fn] = function (...a) {
      const r = real(...a);
      if (logging && fn !== 'random' || (logging && fn === 'random' && a.length))
        console.log(`        ${fn}(${a.join(',')}) -> ${fn === 'sample' ? (r && r.species ? r.species.name : r) : JSON.stringify(r)}`);
      return r;
    };
  }
  console.log(`\n=== ${label} (seed ${JSON.stringify(seed)}) ===`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 40) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    logging = !!entry.log;
    if (entry.log) console.log(`  --- decision [${rs}] ${JSON.stringify(entry)} ---`);
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    logging = false;
    if (entry.stop) break;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}
async function main() {
  await run('A: Roar (2 eligible, no spikes)', [1, 2, 3, 4],
    [mon('Suicune', ['roar', 'surf'], { ability: 'Pressure', evs: { hp: 252, def: 252 }, nature: 'Relaxed' })],
    [mon('Blissey', ['softboiled', 'icebeam'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Skarmory', ['steelwing'], { ability: 'Keen Eye', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', log: true, stop: true }]);
  await run('D: Roar into 2-layer Spikes (drag Tyranitar)', [1, 2, 3, 4],
    [mon('Skarmory', ['spikes', 'roar', 'steelwing'], { ability: 'Keen Eye', evs: { hp: 252, def: 252 }, nature: 'Relaxed' })],
    [mon('Blissey', ['softboiled'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Tyranitar', ['crunch'], { ability: 'Sand Stream', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1', log: true, stop: true }]);
  await run('E: Roar into Spikes-KO (chain replace)', [1, 2, 3, 4],
    [mon('Skarmory', ['spikes', 'roar', 'drillpeck'], { ability: 'Keen Eye', evs: { hp: 252, atk: 252 } })],
    [mon('Blissey', ['softboiled'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Diglett', ['scratch'], { level: 1, ability: 'Sand Veil' }),
     mon('Sandshrew', ['scratch'], { level: 1, ability: 'Sand Veil' })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' },
     { p1: 'move 2', p2: 'move 1', log: true }, { p2: 'switch 3', log: true, stop: true }]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
