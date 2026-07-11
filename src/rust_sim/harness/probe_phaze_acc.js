// probe_phaze_acc.js — pin EXACTLY why a gen3 Roar draws randomChance(100,100).
// Wrap battle.runEvent to log the 'Accuracy'/'ModifyAccuracy' event relayVar for the phaze move.
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
async function main() {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) {} })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(
    [mon('Suicune', ['roar', 'surf'], { ability: 'Pressure', evs: { hp: 252, def: 252 }, nature: 'Relaxed' })]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(
    [mon('Blissey', ['softboiled'], { ability: 'Pressure', evs: { hp: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Pressure', evs: { hp: 252 } })]) })}`);
  for (let i = 0; i < 10; i++) await tick();
  const battle = stream.battle;
  let logging = false;
  const realRun = battle.runEvent.bind(battle);
  battle.runEvent = function (eventid, target, source, effect, relayVar, ...rest) {
    const out = realRun(eventid, target, source, effect, relayVar, ...rest);
    if (logging && (eventid === 'Accuracy' || eventid === 'ModifyAccuracy')) {
      const mv = effect && effect.id ? effect.id : (effect || '');
      console.log(`    runEvent('${eventid}') move=${mv} in=${JSON.stringify(relayVar)} out=${JSON.stringify(out)}`);
    }
    return out;
  };
  logging = true;
  streams.omniscient.write('>p1 move 1');
  streams.omniscient.write('>p2 move 1');
  for (let k = 0; k < 18; k++) await tick();
  console.log('roar move.accuracy in dex =', battle.dex.moves.get('roar').accuracy);
  try { streams.omniscient.destroy(); } catch (e) {}
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
