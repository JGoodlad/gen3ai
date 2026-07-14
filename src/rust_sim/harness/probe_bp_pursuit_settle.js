// Settle: does the sim's Pursuit interrupt STRIKE a Baton-Pass passer, or run
// Pursuit normally against the entrant? Exact diagnosis scenario.
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const FORMAT = 'gen3customgame';
const IV31 = { hp:31,atk:31,def:31,spa:31,spd:31,spe:31 };
const EV0 = { hp:0,atk:0,def:0,spa:0,spd:0,spe:0 };
function mon(species, moves, opts={}) {
  return { species, item:opts.item||'', ability:opts.ability||'No Ability', moves,
    evs:{...EV0,...(opts.evs||{})}, ivs:opts.ivs||IV31, nature:opts.nature||'Serious',
    level:opts.level||100, gender:opts.gender||'N' };
}
async function run() {
  const p1 = [ mon('Snorlax', ['pursuit','bodyslam'], {evs:{spa:252}}) ];
  const p2 = [ mon('Jolteon', ['batonpass','thunderbolt'], {evs:{spe:252,spa:252}}),
               mon('Vaporeon', ['surf']) ];
  const stream = new BattleStream();
  const seed = [7,11,13,17];
  const chunks = [];
  (async () => { for await (const c of stream) chunks.push(c); })();
  // hook runMove + faint
  const Battle = require(path.join(PS,'dist/sim/battle')).Battle;
  const BA = require(path.join(PS,'dist/sim/battle-actions')).BattleActions;
  const origRun = BA.prototype.runMove;
  BA.prototype.runMove = function(moveOrId, pokemon, ...rest) {
    const id = (typeof moveOrId === 'string') ? moveOrId : (moveOrId && moveOrId.id);
    console.log(`  runMove ${id} by ${pokemon && pokemon.name} hp=${pokemon && pokemon.hp}`);
    return origRun.call(this, moveOrId, pokemon, ...rest);
  };
  stream.write(`>start ${JSON.stringify({formatid:FORMAT, seed})}`);
  stream.write(`>player p1 ${JSON.stringify({name:'A', team:Teams.pack(p1)})}`);
  stream.write(`>player p2 ${JSON.stringify({name:'B', team:Teams.pack(p2)})}`);
  await new Promise(r=>setTimeout(r,0));
  console.log('--- dec0: p1 pursuit, p2 batonpass ---');
  stream.write('>p1 move 1');
  stream.write('>p2 move 1');
  await new Promise(r=>setTimeout(r,0));
  console.log('--- dec1: p2 switch 2 ---');
  stream.write('>p2 switch 2');
  await new Promise(r=>setTimeout(r,0));
  // report state via the omniscient battle
  const battle = stream.battle;
  for (const side of battle.sides) {
    for (const pk of side.pokemon) {
      console.log(`  ${side.id} ${pk.name} hp=${pk.hp}/${pk.maxhp} fnt=${pk.fainted} active=${pk.isActive} boosts=${JSON.stringify(pk.boosts)}`);
    }
  }
  const s = battle.prng.getSeed ? battle.prng.getSeed() : battle.prng.seed;
  console.log('  seedAfter =', JSON.stringify(s));
}
run();
