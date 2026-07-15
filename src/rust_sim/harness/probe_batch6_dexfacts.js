// probe_batch6_dexfacts.js — batch-6 RESOLVED gen3 dex facts (the mod-chain law: read
// the resolved Dex.mod('gen3'), never a single data file).
//   * condition noCopy for the batch-6 volatiles (Baton Pass copyability)
//   * failencore / failmimic flag carriers across the whole gen3 move dex
//   * endure priority / memento accuracy / charge boosts / mimic+painsplit flags
//   * perishsong condition residual order + duration
// Run: node src/rust_sim/harness/probe_batch6_dexfacts.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Dex } = require(path.join(PS, 'dist/sim/dex'));

const g3 = Dex.mod('gen3');

function condFacts(id) {
  const c = g3.conditions.get(id);
  return {
    id,
    exists: c.exists,
    noCopy: c.noCopy === undefined ? '(undef)' : c.noCopy,
    duration: c.duration,
    durationCallback: !!c.durationCallback,
    onResidualOrder: c.onResidualOrder,
    onResidualSubOrder: c.onResidualSubOrder,
    keys: Object.keys(c).filter((k) => k.startsWith('on')),
  };
}

console.log('=== CONDITIONS (resolved gen3) ===');
for (const id of ['trapped', 'trapper', 'partiallytrapped', 'perishsong', 'destinybond',
                  'encore', 'charge', 'endure', 'stall', 'mustrecharge']) {
  console.log(JSON.stringify(condFacts(id)));
}

console.log('\n=== MOVE FACTS (resolved gen3) ===');
for (const id of ['encore', 'destinybond', 'endure', 'perishsong', 'meanlook', 'spiderweb',
                  'block', 'bellydrum', 'charge', 'memento', 'mimic', 'painsplit', 'psychup',
                  'snatch']) {
  const m = g3.moves.get(id);
  const cond = m.condition ? {
    noCopy: m.condition.noCopy === undefined ? '(undef)' : m.condition.noCopy,
    duration: m.condition.duration,
    durationCallback: !!m.condition.durationCallback,
    onResidualOrder: m.condition.onResidualOrder,
    onResidualSubOrder: m.condition.onResidualSubOrder,
    onKeys: Object.keys(m.condition).filter((k) => k.startsWith('on')),
  } : null;
  console.log(JSON.stringify({
    id, accuracy: m.accuracy, priority: m.priority, pp: m.pp, target: m.target,
    type: m.type, category: m.category, flags: m.flags, boosts: m.boosts || null,
    volatileStatus: m.volatileStatus || null, selfdestruct: m.selfdestruct || null,
    condition: cond,
  }));
}

console.log('\n=== failencore carriers (gen3 dex) ===');
const fe = [];
const fm = [];
for (const m of g3.moves.all()) {
  if (m.isNonstandard || m.num > 354 || m.num <= 0) continue; // gen<=3 real moves
  if (m.flags && m.flags.failencore) fe.push(m.id);
  if (m.flags && m.flags.failmimic) fm.push(m.id);
}
console.log('failencore:', JSON.stringify(fe.sort()));
console.log('failmimic:', JSON.stringify(fm.sort()));
