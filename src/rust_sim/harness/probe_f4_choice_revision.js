// probe_f4_choice_revision.js — the F4 oracle: does the sim's `side.choose` CLEAR and
// re-parse on a SECOND pre-commit `>pN` write (last-write-wins), or MERGE/append?
// The real bridge sends ONE choice per request, so this never triggers in production —
// but the port's per-side accumulator must match the sim's semantics (or the gap must be
// documented). We write TWO different `>p1` choices for the SAME move request BEFORE p2
// commits, then commit p2, and observe which p1 choice the sim executed (from the |move|
// line) and the resulting PRNG seed.
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
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: opts.nature || 'Serious', level: 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, writes) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) { if (l && !l.startsWith('|t:|') && !l.startsWith('|split') && l !== '|') lines.push(l); } } })();
  const seed = [11, 22, 33, 44];
  const p1 = [mon('Snorlax', ['bodyslam', 'earthquake', 'rest', 'curse'], { item: 'Leftovers', ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })];
  const p2 = [mon('Skarmory', ['drillpeck', 'spikes', 'roar', 'toxic'], { item: 'Leftovers', ability: 'Keen Eye', nature: 'Impish', evs: { hp: 252, def: 252 } })];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  for (const w of writes) {
    streams.omniscient.write(w);
    for (let i = 0; i < 4; i++) await tick();
  }
  for (let i = 0; i < 16; i++) await tick();
  const seedAfter = stream.battle.prng.getSeed();
  const moves = lines.filter((l) => l.startsWith('|move|'));
  console.log(`--- ${label} ---`);
  console.log('  writes: ' + JSON.stringify(writes));
  console.log('  |move| lines: ' + JSON.stringify(moves));
  console.log('  seedAfter: ' + JSON.stringify(seedAfter));
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // Baseline A: p1 commits move 1 (Body Slam) once, then p2.
  await run('A: single p1 move 1', ['>p1 move 1', '>p2 move 1']);
  // Baseline B: p1 commits move 2 (Earthquake) once, then p2.
  await run('B: single p1 move 2', ['>p1 move 2', '>p2 move 1']);
  // F4 test: p1 writes move 1 THEN move 2 (both before p2 commits), then p2 commits.
  //   last-write-wins → executes move 2 (matches B). first-wins → executes move 1 (matches A).
  await run('C: p1 move 1 THEN move 2, then p2', ['>p1 move 1', '>p1 move 2', '>p2 move 1']);
  // F4 test 2: three revisions p1 move 1 -> 2 -> 3(Rest), then p2.
  await run('D: p1 move 1->2->3, then p2', ['>p1 move 1', '>p1 move 2', '>p1 move 3', '>p2 move 1']);
}
main();
