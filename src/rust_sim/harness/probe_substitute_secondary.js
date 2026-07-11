// probe_substitute_secondary.js — settle the EXACT secondary draw model vs a sub in gen3.
//
// Hooks battle.random / battle.randomChance to log EVERY draw's arguments during a
// single Body-Slam-into-sub turn vs a Body-Slam-into-bare turn, so we can see whether
// the per-move secondary `random(100)` is or is not drawn when the target has a sub.
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

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

async function run(label, p1team, p2team, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  const seed = [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  let log = [];
  const realRandom = battle.random.bind(battle);
  battle.random = function (from, to) { const v = realRandom(from, to); log.push(`random(${from},${to})=${v}`); return v; };
  const realRC = battle.randomChance.bind(battle);
  battle.randomChance = function (num, den) { const v = realRC(num, den); log.push(`randomChance(${num},${den})=${v}`); return v; };

  console.log(`\n=== ${label} ===`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 50) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= plan.length) break;
    const entry = plan[i]; i++;
    log = [];
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const subOf = (m) => (m && m.volatiles && m.volatiles['substitute']) ? `SUB(${m.volatiles['substitute'].hp})` : '';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} ${subOf(m)}` : '-';
    console.log(`  ${JSON.stringify(entry)} -> p1=${fmt(a0)} | p2=${fmt(a1)}`);
    console.log(`      draws: ${log.join('  ')}`);
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // Body Slam (par 30 secondary) into a BARE Blissey: expect the secondary random(100).
  await run('Body Slam into BARE Blissey',
    [mon('Snorlax', ['bodyslam'], { evs: { atk: 252 } })],
    [mon('Blissey', ['splash'], { ability: 'Natural Cure', evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // Body Slam into a SUBBED Blissey: is the secondary random(100) drawn?
  await run('Body Slam into SUBBED Blissey',
    [mon('Snorlax', ['bodyslam', 'splash'], { evs: { atk: 252 } })],
    [mon('Blissey', ['substitute', 'softboiled'], { ability: 'Natural Cure', evs: { hp: 252, def: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' }, // Snorlax Splash, Blissey Substitute → sub up
      { p1: 'move 1', p2: 'move 2' }, // Snorlax Body Slam INTO sub ; Blissey Soft-Boiled
    ]);

  // Crunch (-1 SpD secondary) into a SUBBED Gengar: is the secondary random(100) drawn?
  await run('Crunch into SUBBED Gengar',
    [mon('Tyranitar', ['crunch', 'splash'], { evs: { atk: 252 } })],
    [mon('Gengar', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' }, // TTar Splash, Gengar Substitute
      { p1: 'move 1', p2: 'move 2' }, // TTar Crunch INTO sub ; Gengar Splash
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
