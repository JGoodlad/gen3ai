// probe_status_move_fail_draws.js — count PRNG draws + dump lines for a STANDALONE
// status move into an already-statused target, to confirm the draw model of the
// already-statused fail (same-status vs different-status) is DRAW-FREE past accuracy.
//
// Run:  node src/rust_sim/harness/probe_status_move_fail_draws.js
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
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31, nature: 'Serious', level: 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1team, p2team, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) lines.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify([9, 8, 7, 6])}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  let draws = 0;
  const rng = battle.prng.rng; const realNext = rng.next.bind(rng);
  rng.next = function (...a) { draws++; return realNext(...a); };
  console.log(`\n=== ${label} ===`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 40) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= plan.length) break;
    const entry = plan[i]; i++;
    if (entry.inject) for (const inj of entry.inject) { const m = battle.sides[inj.side].active[0]; if (inj.status) m.setStatus(inj.status, m, null, true); }
    const d0 = draws; const before = lines.length;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    console.log(`  turn draws=${draws - d0}`);
    for (const l of lines.slice(before)) if (/\|(move|-status|-fail|-immune|-miss)\|/.test(l)) console.log('    ' + l);
    if (entry.stop) break;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // Baseline: TWave into unstatused (draws accuracy). Compare to already-statused.
  await run('TWave same-status (par into par)',
    [mon('Tyranitar', ['thunderwave', 'crunch'], { ability: 'Sand Stream' })],
    [mon('Blissey', ['softboiled'], {})],
    [{ inject: [{ side: 1, status: 'par' }] }, { p1: 'move 1', p2: 'move 1' }, { stop: true }]);
  await run('TWave different-status (par into brn)',
    [mon('Tyranitar', ['thunderwave', 'crunch'], { ability: 'Sand Stream' })],
    [mon('Blissey', ['softboiled'], {})],
    [{ inject: [{ side: 1, status: 'brn' }] }, { p1: 'move 1', p2: 'move 1' }, { stop: true }]);
  await run('TWave into unstatused (baseline)',
    [mon('Tyranitar', ['thunderwave', 'crunch'], { ability: 'Sand Stream' })],
    [mon('Blissey', ['softboiled'], {})],
    [{ p1: 'move 1', p2: 'move 1' }, { stop: true }]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
