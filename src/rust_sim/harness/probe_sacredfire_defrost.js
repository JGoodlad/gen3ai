// probe_sacredfire_defrost.js — settle the gen3 FROZEN-user × defrost-move model
// (the ~61-repro "sacredfire-at-the-diverging-decision" A/B tail). The RESOLVED
// `Dex.forFormat('gen3customgame')` sim is the ONLY oracle.
//
// HYPOTHESIS under test: a FROZEN user of a gen3 `flags.defrost` move (Sacred Fire,
// Flame Wheel) BYPASSES the frozen `randomChance(1,5)` thaw roll (moves draw-free
// through the frz onBeforeMove gate) — a draw-COUNT difference vs the port's uniform
// freeze model (which always draws the 1-in-5 for a frozen user).
// Also: does the defrost move thaw the USER on use (and via which handler — draw-free?).
//
// Method: direct in-process Battle (gen3customgame), the user FORCE-FROZEN via
// pokemon.setStatus('frz') before the measured turn (outside the draw count). Per
// scenario we count raw backend PRNG draws for the turn, log whether the user moved,
// and the user's status after. Controls: the same seed with a NON-defrost Fire move
// (Flamethrower) and with a non-Fire move (Strength).
//
// Run:  node src/rust_sim/harness/probe_sacredfire_defrost.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31,
    nature: opts.nature || 'Hardy', level: opts.level || 100, gender: opts.gender || 'N',
  };
}

// One turn: p1 (maybe frozen) uses `moveId`; p2 splashes. Count raw draws; report.
function runTurn({ seed, moveId, frozen, attacker }) {
  const battle = new Battle({ formatid: 'gen3customgame', seed: [0, 0, 0, seed] });
  battle.setPlayer('p1', { name: 'A', team: Teams.pack([attacker]) });
  battle.setPlayer('p2', { name: 'B', team: Teams.pack([mon('Blissey', ['splash'])]) });
  const atk = battle.sides[0].active[0];
  if (frozen) atk.setStatus('frz');
  let draws = 0;
  const backend = battle.prng.rng;
  const orig = backend.next.bind(backend);
  backend.next = (...a) => { draws++; return orig(...a); };
  const logStart = battle.log.length;
  battle.choose('p1', 'move ' + moveId);
  battle.choose('p2', 'move splash');
  const turnLog = battle.log.slice(logStart);
  const moved = turnLog.some(l => l.startsWith('|move|p1a:'));
  const cant = turnLog.some(l => l.startsWith('|cant|p1a:'));
  const cure = turnLog.find(l => l.includes('-curestatus') && l.includes('p1a:')) || '';
  return { draws, moved, cant, cure, statusAfter: atk.status || 'none' };
}

function sweep(label, cfg, n = 25) {
  const rows = [];
  for (let s = 1; s <= n; s++) rows.push(runTurn({ ...cfg, seed: s }));
  const drawCounts = {};
  let moved = 0, cant = 0, thawed = 0;
  for (const r of rows) {
    drawCounts[r.draws] = (drawCounts[r.draws] || 0) + 1;
    if (r.moved) moved++;
    if (r.cant) cant++;
    if (r.statusAfter !== 'frz') thawed++;
  }
  console.log(`  ${label}: draws=${JSON.stringify(drawCounts)} moved=${moved}/${rows.length} cant=${cant} thawed=${thawed}`);
  const sample = rows.find(r => r.cure) || rows[0];
  console.log(`    sample cure line: ${sample.cure || '(none)'} statusAfter=${sample.statusAfter}`);
  return rows;
}

function main() {
  const dex3 = Dex.forFormat('gen3customgame');
  console.log('=== resolved gen3 move flags + frz handlers ===');
  for (const id of ['sacredfire', 'flamewheel', 'flamethrower', 'overheat']) {
    const m = dex3.moves.get(id);
    console.log(`  ${id}: flags=${JSON.stringify(m.flags)} thawsTarget=${m.thawsTarget}`);
  }
  const frz = dex3.conditions.get('frz');
  console.log('  frz.onBeforeMove =', frz.onBeforeMove.toString().replace(/\s+/g, ' '));
  console.log('  frz.onModifyMove =', frz.onModifyMove ? frz.onModifyMove.toString().replace(/\s+/g, ' ') : 'NONE');
  console.log('  frz.onHit        =', frz.onHit ? frz.onHit.toString().replace(/\s+/g, ' ') : 'NONE');

  const hoOh = mon('Ho-Oh', ['sacredfire', 'flamethrower', 'strength']);
  const enteiFW = mon('Entei', ['flamewheel', 'flamethrower', 'strength']);

  console.log('\n=== FROZEN user draw model (25 seeds each) ===');
  console.log('  [control] frozen + Flamethrower (non-defrost): expect 1/5 thaw roll — ~20% move');
  sweep('frozen flamethrower', { moveId: 'flamethrower', frozen: true, attacker: hoOh });
  console.log('  [control] frozen + Strength (non-Fire, non-defrost):');
  sweep('frozen strength   ', { moveId: 'strength', frozen: true, attacker: hoOh });
  console.log('  [hypothesis] frozen + Sacred Fire (flags.defrost):');
  sweep('frozen sacredfire ', { moveId: 'sacredfire', frozen: true, attacker: hoOh });
  console.log('  [hypothesis] frozen + Flame Wheel (flags.defrost):');
  sweep('frozen flamewheel ', { moveId: 'flamewheel', frozen: true, attacker: enteiFW });

  console.log('\n=== NON-frozen user (draw baseline at the same seeds) ===');
  sweep('healthy sacredfire', { moveId: 'sacredfire', frozen: false, attacker: hoOh });
  sweep('healthy flamewheel', { moveId: 'flamewheel', frozen: false, attacker: enteiFW });
  sweep('healthy flamethrower', { moveId: 'flamethrower', frozen: false, attacker: hoOh });

  console.log('\n=== per-seed diff: frozen sacredfire vs healthy sacredfire (first 6 seeds) ===');
  for (let s = 1; s <= 6; s++) {
    const fz = runTurn({ moveId: 'sacredfire', frozen: true, attacker: hoOh, seed: s });
    const ok = runTurn({ moveId: 'sacredfire', frozen: false, attacker: hoOh, seed: s });
    console.log(`  seed=${s}: frozen{draws=${fz.draws},moved=${fz.moved},after=${fz.statusAfter}}  healthy{draws=${ok.draws},moved=${ok.moved}}`);
  }
}

main();
