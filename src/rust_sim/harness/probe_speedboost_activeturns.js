// probe_speedboost_activeturns.js — nail the Speed Boost activeTurns gate timing: on which
// turn does a LEAD Speed Boost mon first boost, and does a mon that SWITCHES IN mid-battle
// skip the boost on its entry turn? Instrument the boost + read activeTurns at the residual.
//
// Run: node src/rust_sim/harness/probe_speedboost_activeturns.js

'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: 'Serious', level: 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

(async () => {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const boostLines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of String(ch).split('\n')) if (l.startsWith('|-boost|') && l.includes('spe')) boostLines.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  // p1: Ninjask(Speed Boost) + a bench Ninjask2 to switch to. p2: a punching bag.
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([mon('Ninjask', ['protect', 'protect']), mon('Yanma', ['protect', 'protect'], { ability: 'Speed Boost' })]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([mon('Snorlax', ['recover', 'recover'])]) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  const at = () => battle.sides[0].active[0].activeTurns;
  const boost = () => battle.sides[0].active[0].boosts.spe || 0;
  const spec = () => battle.sides[0].active[0].species.name;

  console.log('LEAD Ninjask (Speed Boost). Track activeTurns + spe boost across turns:');
  for (let t = 1; t <= 4; t++) {
    console.log(`  before turn ${t}: active=${spec()} activeTurns=${at()} speBoost=${boost()}`);
    streams.omniscient.write('>p1 move 1');
    streams.omniscient.write('>p2 move 1');
    for (let k = 0; k < 10; k++) await tick();
  }
  console.log(`  after turn 4: active=${spec()} activeTurns=${at()} speBoost=${boost()}`);
  console.log('  boost lines:', JSON.stringify(boostLines));

  // Now a mid-battle switch: on turn 5 switch to Yanma (Speed Boost). Does Yanma boost on its
  // ENTRY turn (activeTurns=0 at that residual → NO boost)?
  console.log('\n  --- switch Ninjask -> Yanma on turn 5 (entry turn) ---');
  console.log(`  before switch: active=${spec()} boost=${boost()}`);
  streams.omniscient.write('>p1 switch 2');
  streams.omniscient.write('>p2 move 1');
  for (let k = 0; k < 10; k++) await tick();
  console.log(`  after entry turn: active=${spec()} activeTurns=${at()} speBoost=${boost()} (0 boost = entry-turn skip; activeTurns should be 1 now after endTurn)`);
  streams.omniscient.write('>p1 move 1');
  streams.omniscient.write('>p2 move 1');
  for (let k = 0; k < 10; k++) await tick();
  console.log(`  after next turn: active=${spec()} activeTurns=${at()} speBoost=${boost()} (should be +1 now)`);
})();
