// probe_status_fail_accuracy.js — pinpoint WHETHER a standalone status move draws
// its accuracy roll when the target is already-statused (same vs different status),
// by wrapping battle.randomChance + battle.random and logging the caller move.
//
// Run:  node src/rust_sim/harness/probe_status_fail_accuracy.js
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
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: 'Serious', level: 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p2status) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify([9, 8, 7, 6])}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([mon('Tyranitar', ['thunderwave', 'crunch'], { ability: 'Sand Stream' })]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([mon('Blissey', ['softboiled'], {})]) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  console.log(`\n=== ${label} ===`);
  const realRC = battle.randomChance.bind(battle);
  battle.randomChance = function (n, d) { console.log(`    randomChance(${n},${d}) activeMove=${battle.activeMove ? battle.activeMove.name : '-'}`); return realRC(n, d); };
  const realRnd = battle.random.bind(battle);
  battle.random = function (...a) { console.log(`    random(${a.join(',')}) activeMove=${battle.activeMove ? battle.activeMove.name : '-'}`); return realRnd(...a); };
  // pre-status p2
  if (p2status) battle.sides[1].active[0].setStatus(p2status, battle.sides[1].active[0], null, true);
  console.log('  --- submit TWave/Soft-Boiled ---');
  streams.omniscient.write('>p1 move 1');
  streams.omniscient.write('>p2 move 1');
  for (let k = 0; k < 20; k++) await tick();
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  await run('par into par (same status)', 'par');
  await run('par into brn (different status)', 'brn');
  await run('par into unstatused (baseline)', null);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
