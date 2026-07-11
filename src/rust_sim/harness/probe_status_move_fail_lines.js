// probe_status_move_fail_lines.js — characterize the EXACT protocol lines a
// STANDALONE major-status move (Thunder Wave / Toxic / Will-O-Wisp) emits when it
// FAILS on an already-statused / type-immune / already-same-status target, so the
// port's `run_status_move` emits byte-exact `|-fail|` / `|-immune|` lines.
//
// The gap the protocol test surfaced: Thunder Wave into an already-PARALYZED Blissey
// emits `|-fail|p2a: Blissey|par` in the golden, which the port did not emit. This
// probe dumps the omniscient stream for each fail flavor so we get the exact tokens.
//
// Run:  node src/rust_sim/harness/probe_status_move_fail_lines.js
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

async function run(label, p1team, p2team, plan, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) lines.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(opts.seed || [1, 2, 3, 4])}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const startLine = lines.length;
  let i = 0, safety = 0;
  while (!battle.ended && safety < 40) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= plan.length) break;
    const entry = plan[i]; i++;
    if (entry.inject) {
      for (const inj of entry.inject) {
        const m = battle.sides[inj.side].active[0];
        if (inj.status) m.setStatus(inj.status, m, null, true);
      }
    }
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    if (entry.stop) break;
  }
  console.log(`\n=== ${label} ===`);
  for (const l of lines.slice(startLine)) {
    if (/\|(move|-status|-fail|-immune|-miss|-activate|cant|-curestatus)\|/.test(l)) console.log('  ' + l);
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // Thunder Wave into an ALREADY-PARALYZED target → |-fail|
  await run('TWave into already-par target',
    [mon('Tyranitar', ['thunderwave', 'crunch'], { ability: 'Sand Stream' })],
    [mon('Blissey', ['seismictoss'], {})],
    [
      { p1: 'move 1', p2: 'move 1' },            // TWave → par Blissey
      { p1: 'move 1', p2: 'move 1' },            // TWave AGAIN into par → -fail
      { stop: true },
    ]);

  // Toxic into an already-poisoned target → |-fail|
  await run('Toxic into already-psn target',
    [mon('Gengar', ['toxic', 'shadowball'], {})],
    [mon('Snorlax', ['bodyslam'], { ability: 'No Ability' })],
    [
      { p1: 'move 1', p2: 'move 1' },            // Toxic → tox Snorlax
      { p1: 'move 1', p2: 'move 1' },            // Toxic AGAIN → -fail
      { stop: true },
    ]);

  // Will-O-Wisp into an already-burned target → |-fail|
  await run('WoW into already-brn target',
    [mon('Gengar', ['willowisp', 'shadowball'], {})],
    [mon('Snorlax', ['bodyslam'], { ability: 'No Ability' })],
    [
      { inject: [{ side: 1, status: 'brn' }] },  // pre-burn Snorlax (via inject, draws may vary)
    ].concat([
      { p1: 'move 1', p2: 'move 1' },            // WoW into burned → -fail
      { stop: true },
    ]));

  // Thunder Wave into a GROUND target → |-immune| (already handled by port)
  await run('TWave into Ground target',
    [mon('Tyranitar', ['thunderwave', 'crunch'], { ability: 'Sand Stream' })],
    [mon('Swampert', ['earthquake'], {})],
    [
      { p1: 'move 1', p2: 'move 1' },            // TWave → -immune (Ground)
      { stop: true },
    ]);

  // Thunder Wave into a DIFFERENTLY-statused target (burned) → |-fail| too?
  await run('TWave into a burned target (different status)',
    [mon('Tyranitar', ['thunderwave', 'crunch'], { ability: 'Sand Stream' })],
    [mon('Snorlax', ['bodyslam'], { ability: 'No Ability' })],
    [
      { inject: [{ side: 1, status: 'brn' }] },
      { p1: 'move 1', p2: 'move 1' },            // TWave into burned → -fail
      { stop: true },
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
