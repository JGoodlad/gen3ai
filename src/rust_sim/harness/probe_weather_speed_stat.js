// probe_weather_speed_stat.js — settle the EXACT weather-speed ×2 fold in getStat('spe'),
// including its composition with paralysis ×0.25 (both are `onModifySpe chainModify`) and
// with a boost, so the port's effective_speed is bit-for-bit.
//
// The port's effective_speed does: boost-table floor → (paralysis ? modify(spe,1,4)). The
// weather ×2 is a `chainModify(2)`. runEvent('ModifySpe') ACCUMULATES all onModifySpe
// chainModify handlers into ONE 4096 modifier applied once (getStat). So a paralyzed Swift
// Swim mon in rain should be `modify(spe, 2*0.25=0.5)` = one chain, NOT sequential rounds.
// We read the sim's getStat('spe') directly for several (base, boost, para, weather) combos.
//
// Run: node src/rust_sim/harness/probe_weather_speed_stat.js

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
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: opts.nature || 'Serious', level: 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function getSpe(ability, weatherSetter, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  // p1: the probe mon (Kingdra/Exeggutor with the weather ability). p2: the weather setter.
  const probe = mon(opts.species || 'Kingdra', ['rest', 'rest'], { ability, evs: opts.evs || {} });
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([probe]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([weatherSetter]) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const p1 = battle.sides[0].active[0];
  // Optionally paralyze / boost the probe mon directly.
  if (opts.para) p1.setStatus('par');
  if (opts.boost) battle.boost({ spe: opts.boost }, p1);
  const rawSpe = p1.storedStats.spe;
  const actionSpe = p1.getActionSpeed();
  const getStatSpe = p1.getStat('spe');
  const weather = battle.field.effectiveWeather();
  return { rawSpe, actionSpe, getStatSpe, weather };
}

(async () => {
  const rain = mon('Kyogre', ['recover', 'recover'], { ability: 'Drizzle' });
  const noWeather = mon('Snorlax', ['recover', 'recover'], { ability: 'Shell Armor' });
  console.log('=== Swift Swim getStat(spe) composition ===');
  const cases = [
    ['no ability, no weather', 'Shell Armor', noWeather, {}],
    ['Swift Swim, no weather', 'Swift Swim', noWeather, {}],
    ['Swift Swim, RAIN', 'Swift Swim', rain, {}],
    ['Swift Swim, RAIN, para', 'Swift Swim', rain, { para: true }],
    ['Swift Swim, RAIN, +1 boost', 'Swift Swim', rain, { boost: 1 }],
    ['Swift Swim, RAIN, +1 boost, para', 'Swift Swim', rain, { boost: 1, para: true }],
    ['no ability, RAIN', 'Shell Armor', rain, {}],
    ['para only, no weather', 'Shell Armor', noWeather, { para: true }],
  ];
  for (const [label, ab, setter, opts] of cases) {
    const r = await getSpe(ab, setter, opts);
    console.log(`  ${label.padEnd(38)} raw=${r.rawSpe} weather=${r.weather} getStat(spe)=${r.getStatSpe} actionSpeed=${r.actionSpe}`);
  }
  console.log('\nInterpretation: compare "Swift Swim RAIN" == 2×raw; "RAIN+para" == floor(2*raw*0.25) as ONE chain (0.5) vs sequential; the port must match getStat(spe).');
})();
