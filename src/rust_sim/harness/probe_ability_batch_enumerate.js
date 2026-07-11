// probe_ability_batch_enumerate.js — STEP 0 of the ability-batch workflow.
//
// Enumerate EVERY gen-3-legal ability (num <= 76) from the RESOLVED Dex.mod('gen3'),
// cross-reference against the e2e MODELED_ABILITIES + NOOP_ABILITIES sets, and for
// each NOT-YET-ADMITTED ability print its resolved handler inventory + the
// dump_gen3_mechanics class, so the batch can classify each by DRAW MODEL:
//   (a) provable no-op in the modeled move/item universe
//   (b) draw-free / structural (gate / stat-damage mod / weather-negate / residual)
//   (c) draw-bearing proc (a NEW random roll — DEFER to batch 2)
//
// The RESOLVED dist is the ONLY oracle (base data/*.ts is a hypothesis — the mod chain
// replaces/deletes handlers). This prints the handler-source so the reviewer can see
// exactly which callbacks exist and whether any is draw-bearing.
//
// Run: node src/rust_sim/harness/probe_ability_batch_enumerate.js

'use strict';

const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Dex } = require(path.join(PS, 'dist/sim/dex'));
const e2e = require('./gen_e2e_fuzz.js');

const d3 = Dex.mod('gen3');

const IGNORED_HANDLERS = new Set(['onDrive', 'onMemory', 'onPlate']);

function handlerKeys(entry) {
  const keys = Object.keys(entry).filter(
    (k) => k.startsWith('on') && entry[k] !== undefined && !IGNORED_HANDLERS.has(k)
  );
  const hasFn = keys.some((k) => typeof entry[k] === 'function');
  const out = [];
  for (const k of keys) {
    if (!hasFn && typeof entry[k] !== 'function') continue;
    out.push(k);
  }
  return out;
}

function fnSrc(entry, keys) {
  return keys
    .map((k) => (typeof entry[k] === 'function' ? String(entry[k]) : `${k}=${entry[k]}`))
    .join(' || ')
    .replace(/\s+/g, ' ');
}

// A conservative DRAW-BEARING heuristic: the handler source calls a PRNG method.
// (This is a HINT, not the oracle — the dedicated per-ability probe settles it.)
const DRAW_RE = /\b(this\.)?(random|randomChance|sample|shuffle|speedSort)\s*\(/;

const MODELED = e2e.MODELED_ABILITIES;
const NOOP = e2e.NOOP_ABILITIES;

// Pull the dump's classification for cross-reference.
const abilities = [];
for (const id of Object.keys(d3.data.Abilities || {})) {
  const ab = d3.abilities.get(id);
  if (!ab || !ab.exists) continue;
  if (typeof ab.num !== 'number' || ab.num < 1 || ab.num > 76) continue; // gen-3-legal
  const keys = handlerKeys(ab);
  const src = fnSrc(ab, keys);
  abilities.push({
    id: ab.id,
    name: ab.name,
    num: ab.num,
    modeled: MODELED.has(ab.id),
    noop: NOOP.has(ab.id),
    keys,
    drawHint: DRAW_RE.test(src),
    src,
  });
}
abilities.sort((a, b) => a.num - b.num);

const admitted = abilities.filter((a) => a.modeled || a.noop);
const notAdmitted = abilities.filter((a) => !a.modeled && !a.noop);

console.log(`=== gen-3-legal abilities (num <= 76): ${abilities.length} total ===`);
console.log(`ADMITTED (MODELED ${abilities.filter((a) => a.modeled).length} + NOOP ${abilities.filter((a) => a.noop).length}): ${admitted.length}`);
console.log(`NOT ADMITTED: ${notAdmitted.length}`);
console.log('');
console.log('=== ADMITTED already ===');
console.log('  MODELED:', abilities.filter((a) => a.modeled).map((a) => a.id).join(', '));
console.log('  NOOP   :', abilities.filter((a) => a.noop).map((a) => a.id).join(', '));
console.log('');
console.log('=== NOT-YET-ADMITTED abilities (the batch targets) ===');
for (const a of notAdmitted) {
  console.log(`\n[${a.num}] ${a.id} (${a.name})${a.drawHint ? '  <<< DRAW-HINT (PRNG call in a handler)' : ''}`);
  console.log(`    handlers: ${a.keys.join(', ') || '(none)'}`);
  console.log(`    src: ${a.src.slice(0, 800)}`);
}
