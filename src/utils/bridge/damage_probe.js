// damage_probe.js
// Single-turn DAMAGE ORACLE over a local, in-process Pokémon Showdown BattleStream.
// Unlike local_sim_bridge.js (a per-side protocol relay for poke-env), this drives the
// OMNISCIENT (referee) stream so we can read EXACT both-side HP + the sim's OWN computed
// stats — the clean ground truth for validating our differentiable damage operator's gen3
// physics, with ZERO measurement confounds (no percent-rounding, no stale HP, no switches).
//
// Batch request/response over stdio (same pattern as validate_team.js):
//   stdin : ONE json  { "scenarios": [ <scenario>, ... ] }
//     scenario = {
//       id:        any (echoed back),
//       formatid:  e.g. "gen3customgame",
//       seed?:     [n,n,n,n]  (force the PRNG; e.g. force/suppress crits in tests),
//       p1: [<set>, ...],  p2: [<set>, ...]   // FULL sets — we pack them here
//       choices:   [ ["p1","move 1"], ["p2","move 1"], ... ]  // written in order
//     }
//     set = {species,item,ability,moves:[...],evs:{...},ivs:{...},nature,level,gender}
//   stdout: one json line PER scenario, in order:
//     { id, log:[omniscient protocol lines], p1:<snap>, p2:<snap> }  or  { id, error }
//     snap = {species, maxhp, hp, stats:{hp,atk,def,spa,spd,spe}, boosts, status, item,
//             types:[...], sideConditions:[...]}   // END-state (design the measured attack LAST)
//
// One BattleStream per scenario (fresh sim → no cross-scenario state). Scenarios run
// sequentially; each awaits a quiescence tick so the sim fully resolves before we snapshot.
'use strict';

const path = require('path');
const psPath = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(psPath, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(psPath, 'dist/sim'));

function snapSide(side) {
  const a = side && side.active && side.active[0];
  if (!a) return null;
  let types;
  try { types = a.getTypes(); } catch (e) { types = a.types; }
  return {
    species: (a.species && a.species.name) || a.speciesid || String(a.species),
    maxhp: a.maxhp,
    hp: a.hp,
    stats: a.storedStats,                 // the sim's OWN computed stats (truth, no guessing)
    boosts: a.boosts,
    status: a.status || '',
    item: a.item || '',
    ability: a.ability || '',
    types: types,
    sideConditions: Object.keys(side.sideConditions || {}),
  };
}

function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function runScenario(sc) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  // Drain the omniscient stream into `log` in the background; we snapshot the live battle
  // object directly (no reliance on the stream closing).
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();

  const seedClause = sc.seed ? `,"seed":${JSON.stringify(sc.seed)}` : '';
  streams.omniscient.write(`>start {"formatid":"${sc.formatid}"${seedClause}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(sc.p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(sc.p2) })}`);
  for (const [side, choice] of (sc.choices || [])) streams.omniscient.write(`>${side} ${choice}`);

  // Let the sim fully process the queued input (a few macrotask ticks is ample for a
  // handful of turns; the sim resolves synchronously between awaits).
  for (let i = 0; i < 6; i++) await tick();

  const weather = (stream.battle.field && stream.battle.field.weather) || '';
  const out = { id: sc.id, log, weather, p1: snapSide(stream.battle.sides[0]), p2: snapSide(stream.battle.sides[1]) };
  try { streams.omniscient.destroy(); } catch (e) { /* best effort */ }
  return out;
}

let input = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (c) => { input += c; });
process.stdin.on('end', async () => {
  let scenarios;
  try { scenarios = JSON.parse(input).scenarios; }
  catch (e) { process.stdout.write(JSON.stringify({ fatal: String(e) }) + '\n'); process.exit(1); }
  for (const sc of scenarios) {
    try {
      const res = await runScenario(sc);
      process.stdout.write(JSON.stringify(res) + '\n');
    } catch (e) {
      process.stdout.write(JSON.stringify({ id: sc && sc.id, error: e && e.stack ? e.stack : String(e) }) + '\n');
    }
  }
  process.exit(0);
});
process.on('uncaughtException', (e) => {
  process.stdout.write(JSON.stringify({ fatal: e && e.stack ? e.stack : String(e) }) + '\n');
  process.exit(1);
});
