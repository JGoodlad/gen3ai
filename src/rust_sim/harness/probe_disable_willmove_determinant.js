// probe_disable_willmove_determinant.js — determine EXACTLY what drives willMove(target) at
// Disable's onStart: the move order (who acted first). We hook runMove to log the actual move
// order, and addVolatile to log willMove + post-onStart duration, across scenarios where the
// disabler is faster vs slower and the target uses an attack vs a status move. This tells the
// port the CORRECT predicate for the duration offset.
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const { Pokemon } = require(path.join(PS, 'dist/sim/pokemon'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1team, p2team, plan, targetSide) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":[7,11,13,17]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  const lines = [];
  let rolled = null;
  const realRandom = battle.random.bind(battle);
  battle.random = function (from, to) { const r = realRandom(from, to); if (from === 2 && to === 6) rolled = r; return r; };

  const realRunMove = battle.actions.runMove.bind(battle.actions);
  battle.actions.runMove = function (move, pokemon, ...rest) {
    const mv = typeof move === 'string' ? move : (move && move.id);
    lines.push(`    runMove: ${pokemon.name} uses ${mv}`);
    return realRunMove(move, pokemon, ...rest);
  };
  const realAdd = Pokemon.prototype.addVolatile;
  Pokemon.prototype.addVolatile = function (status, source, sourceEffect, linkedStatus) {
    const id = (typeof status === 'string') ? this.battle.dex.conditions.get(status).id : status.id;
    if (id === 'disable') {
      const willMove = !!this.battle.queue.willMove(this);
      rolled = null;
      const ret = realAdd.call(this, status, source, sourceEffect, linkedStatus);
      const post = this.volatiles.disable ? this.volatiles.disable.duration : null;
      lines.push(`    >>> Disable onStart: willMove(target)=${willMove}  rolled=${rolled}  post-onStart=${post} (== rolled${post === rolled ? '' : post === rolled + 1 ? '+1' : '-1'})`);
      return ret;
    }
    return realAdd.call(this, status, source, sourceEffect, linkedStatus);
  };

  let i = 0, safety = 0, done = false;
  while (!battle.ended && safety < 12 && !done) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)];
    lines.push(`  turn ${i + 1}: p1=${entry.p1} p2=${entry.p2}`);
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {}
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {}
    for (let k = 0; k < 18; k++) await tick();
    i++;
    if (i >= 2 && battle.sides[targetSide].active[0].volatiles.disable) done = true;
  }
  Pokemon.prototype.addVolatile = realAdd;
  try { streams.omniscient.destroy(); } catch (e) {}
  console.log(`\n==== ${label} ====`);
  console.log(lines.join('\n'));
}

async function main() {
  // A: disabler SLOWER, target uses a STATUS move (Rest) on the disable turn.
  await run('A: SLOWER disabler, target uses Rest (status) on disable turn',
    [mon('Blissey', ['disable', 'softboiled'], { evs: { hp: 252, def: 252 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })],
    [mon('Snorlax', ['bodyslam', 'rest'], { evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }], 1);

  // B: disabler SLOWER, target uses an ATTACK (Body Slam) on the disable turn.
  await run('B: SLOWER disabler, target uses Body Slam (attack) on disable turn',
    [mon('Blissey', ['disable', 'softboiled'], { evs: { hp: 252, def: 252 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })],
    [mon('Snorlax', ['bodyslam', 'rest'], { evs: { hp: 252, atk: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }], 1);

  // C: disabler FASTER, target uses an ATTACK on the disable turn.
  await run('C: FASTER disabler, target uses Rock Slide (attack) on disable turn',
    [mon('Aerodactyl', ['disable', 'rockslide'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['bodyslam', 'blizzard'], { evs: { hp: 252, atk: 128, spd: 128 }, nature: 'Brave', ivs: { ...IV31, spe: 0 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }], 1);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
