// probe_statdrop_substitute.js — the EXACT omniscient bytes a STAT-DROP status move
// (Screech / Growl / Leer / Sand-Attack — non-bypasssub) emits when BLOCKED BY A
// SUBSTITUTE. The port's stat-drop arm currently emits NOTHING on the sub-block
// (turn.rs:7418) — this settles whether the sim emits `[still]`+`-fail`, `-immune`,
// or `-activate Substitute`. THE SIM IS THE ORACLE — capture, don't guess.
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

async function run(label, p1team, p2team, plan, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const protoLog = [];
  (async () => { for await (const ch of streams.omniscient) { protoLog.push(ch); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  console.log(`\n=== ${label} (seed ${seed}) ===`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 50) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= plan.length) break;
    const entry = plan[i]; i++;
    protoLog.length = 0;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    // Dump EVERY omniscient line of this turn's chunk (split, normalize |t:|).
    const lines = protoLog.join('').split('\n').filter((l) => l.length && !l.startsWith('|t:'));
    console.log(`  >> ${JSON.stringify(entry)}`);
    for (const l of lines) console.log(`     ${l}`);
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // SCREECH (Def -2, SOUND) into a subbed foe.
  await run('Screech into SUBBED Snorlax',
    [mon('Gengar', ['screech', 'splash'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' }, // Gengar Splash, Snorlax Substitute (up)
      { p1: 'move 1', p2: 'move 2' }, // Gengar SCREECH into the sub ; Snorlax Splash
    ], [7, 11, 13, 17]);

  // GROWL (Atk -1, NOT sound) into a subbed foe.
  await run('Growl into SUBBED Snorlax',
    [mon('Meowth', ['growl', 'splash'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 2' }, // Growl into the sub
    ], [7, 11, 13, 17]);

  // SAND-ATTACK (accuracy -1) into a subbed foe.
  await run('Sand-Attack into SUBBED Snorlax',
    [mon('Sandslash', ['sandattack', 'splash'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 2' },
    ], [7, 11, 13, 17]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
