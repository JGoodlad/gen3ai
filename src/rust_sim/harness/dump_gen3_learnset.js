// dump_gen3_learnset.js — Showdown's OWN gen-3 move legality: the oracle for data/pokemon/gen3_learnset.json.
//
// WHY (gen3_learnset_prevo_chain_v1): the table was built from each species' OWN learnset entry, so a move
// reachable only through a PRE-EVOLUTION (Tyranitar's Dragon Dance: `["9M","8M"]` on its own entry, but Larvitar
// carries `3E` + `3S0`) read as illegal — 1151 (species, move) pairs over 182 species, plus Smeargle's Sketch pool.
// The authority is Showdown's team validator, whose `checkCanLearn` walks the whole pre-evolution chain
// (`dex.species.getFullLearnset`) and treats a Sketch learner's pool as every sketchable move. This harness asks
// that validator directly, for every base species x every gen-3 move, at level 100 (a single move on a mon of
// any legal origin — NOT a whole-set check: move combinations and event-only sets are not in question here).
//
// Outputs:
//   node dump_gen3_learnset.js --json    -> prints {species_id: [move_id, ...]} (sorted) the validator accepts
//   node dump_gen3_learnset.js --check   -> verifies data/pokemon/gen3_learnset.json EXACTLY equals it
//                                           (exit 1 listing every species that drifts)
//
// The typed Hidden Powers (`hiddenpowerfire`, ...) are OUR ids (the type is an IV choice, not a learnset entry), so
// the table carries only the bare `hiddenpower` and the oracle skips the typed ids.
//
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md. A dev/test ORACLE, never a runtime path.)

'use strict';

const path = require('path');
const fs = require('fs');

const REPO = path.resolve(__dirname, '../../..');
const PS = path.join(REPO, 'deps/pokemon-showdown');
const { Dex } = require(path.join(PS, 'dist/sim/dex'));
const { TeamValidator } = require(path.join(PS, 'dist/sim/team-validator'));

const readJson = (rel) => JSON.parse(fs.readFileSync(path.join(REPO, rel), 'utf8'));

function oracle() {
  const species = readJson('data/pokemon/gen3_species.json');
  const moves = readJson('data/pokemon/gen3_moves.json');
  const dex = Dex.forGen(3);
  const validator = TeamValidator.get('gen3ou');
  const baseIds = Object.keys(species).filter((s) => !species[s].baseSpecies).sort();
  const moveIds = Object.keys(moves)
    .filter((m) => !(m.startsWith('hiddenpower') && m !== 'hiddenpower'))
    .sort();
  const out = {};
  for (const sid of baseIds) {
    const sp = dex.species.get(sid);
    if (!sp.exists) throw new Error(`Showdown has no gen-3 species ${sid}`);
    const legal = [];
    for (const mid of moveIds) {
      const mv = dex.moves.get(mid);
      if (!mv.exists) throw new Error(`Showdown has no gen-3 move ${mid}`);
      // null = legal; a string = the reason it is not. FRESH sources per call: checkCanLearn MUTATES them.
      if (validator.checkCanLearn(mv, sp, validator.allSources(sp), { level: 100 }) === null) legal.push(mid);
    }
    if (legal.length) out[sid] = legal;
  }
  return out;
}

function check() {
  const want = oracle();
  const have = readJson('data/pokemon/gen3_learnset.json');
  const drift = [];
  for (const sid of new Set([...Object.keys(want), ...Object.keys(have)])) {
    const w = new Set(want[sid] || []);
    const h = new Set(have[sid] || []);
    const missing = [...w].filter((m) => !h.has(m)).sort();
    const extra = [...h].filter((m) => !w.has(m)).sort();
    if (missing.length || extra.length) drift.push({ sid, missing, extra });
  }
  const pairs = Object.values(want).reduce((n, l) => n + l.length, 0);
  if (drift.length) {
    console.error(`gen3_learnset.json DRIFTS from Showdown's validator in ${drift.length} species:`);
    for (const d of drift.slice(0, 40)) {
      console.error(`  ${d.sid}: missing ${JSON.stringify(d.missing.slice(0, 12))}${d.missing.length > 12 ? '...' : ''}` +
        ` (${d.missing.length}), extra ${JSON.stringify(d.extra.slice(0, 12))} (${d.extra.length})`);
    }
    process.exit(1);
  }
  console.log(`gen3_learnset.json == Showdown validator verdicts: ${Object.keys(want).length} species, ${pairs} pairs`);
}

if (process.argv.includes('--check')) {
  check();
} else if (process.argv.includes('--json')) {
  process.stdout.write(JSON.stringify(oracle()) + '\n');
} else {
  console.error('usage: node dump_gen3_learnset.js --json | --check');
  process.exit(2);
}
