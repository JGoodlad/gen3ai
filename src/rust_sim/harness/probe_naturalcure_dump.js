// probe_naturalcure_dump.js — dump the RESOLVED gen3 NATURAL CURE ability handler
// inventory (the DRAW-MODEL crux: onSwitchOut vs onCheckShow vs both). THE PROBE IS
// THE ONLY ORACLE — reading base data/*.ts is a hypothesis (gen3 `inherit: true`s
// from gen4 which REPLACES/DELETES handlers). This resolves `Dex.mod('gen3')`.
//
// Run:  node src/rust_sim/harness/probe_naturalcure_dump.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Dex } = require(path.join(PS, 'dist/sim'));

function dumpAbility(id) {
  const d = Dex.mod('gen3');
  const ab = d.abilities.get(id);
  console.log(`=== RESOLVED gen3 ability '${id}' (name=${ab.name}, num=${ab.num}) ===`);
  const handlerKeys = Object.keys(ab).filter((k) => k.startsWith('on'));
  console.log('  handler keys:', handlerKeys.join(', ') || '(none)');
  for (const k of Object.keys(ab)) {
    const fn = ab[k];
    if (typeof fn === 'function') {
      console.log(`  ${k}: ${fn.toString().replace(/\s+/g, ' ').slice(0, 600)}`);
    } else if (k !== 'condition') {
      console.log(`  ${k}: ${JSON.stringify(fn)}`);
    }
  }
  const cond = ab.condition;
  if (cond) {
    console.log('  --- ability.condition ---');
    for (const k of Object.keys(cond)) {
      const fn = cond[k];
      if (typeof fn === 'function') {
        console.log(`    ${k}: ${fn.toString().replace(/\s+/g, ' ').slice(0, 600)}`);
      } else {
        console.log(`    ${k}: ${JSON.stringify(fn)}`);
      }
    }
  }
}

dumpAbility('naturalcure');

// Also inspect where onSwitchOut / onCheckShow are dispatched in the sim so we know the
// exact call site + whether a draw is involved. Find switchOut / checkShow references.
console.log('\n=== BattleActions.switchIn / dragIn source (the switch-out call chain) ===');
const { Dex: D2 } = require(path.join(PS, 'dist/sim'));
// The switch-out event is fired by pokemon.ts::switchOut / battle.ts. Dump the relevant fns.
try {
  const battleActionsSrc = require('fs').readFileSync(
    path.join(PS, 'dist/sim/battle-actions.js'), 'utf8');
  // Find runSwitch / switchIn definitions.
  for (const name of ['switchIn', 'dragIn', 'runSwitch']) {
    const re = new RegExp(`\\b${name}\\s*\\(`);
    const idx = battleActionsSrc.search(re);
    if (idx >= 0) {
      console.log(`  --- battle-actions.js ${name} (around char ${idx}) ---`);
      console.log('   ', battleActionsSrc.slice(idx, idx + 500).replace(/\n/g, '\n    '));
    }
  }
} catch (e) { console.log('  (could not read battle-actions.js:', e.message, ')'); }

console.log('\n=== pokemon.js switchOut / checkShow references ===');
try {
  const pokeSrc = require('fs').readFileSync(path.join(PS, 'dist/sim/pokemon.js'), 'utf8');
  for (const needle of ['SwitchOut', 'CheckShow', 'switchFlag', 'clearVolatile']) {
    let from = 0;
    let count = 0;
    while (count < 3) {
      const idx = pokeSrc.indexOf(needle, from);
      if (idx < 0) break;
      const line = pokeSrc.slice(pokeSrc.lastIndexOf('\n', idx) + 1, pokeSrc.indexOf('\n', idx));
      console.log(`  ${needle} @${idx}: ${line.trim().slice(0, 160)}`);
      from = idx + 1;
      count += 1;
    }
  }
} catch (e) { console.log('  (could not read pokemon.js:', e.message, ')'); }

// Where does runEvent('SwitchOut') fire? Search battle.js.
console.log('\n=== battle.js SwitchOut / runSwitch references ===');
try {
  const battleSrc = require('fs').readFileSync(path.join(PS, 'dist/sim/battle.js'), 'utf8');
  for (const needle of ['SwitchOut', 'CheckShow']) {
    let from = 0;
    let count = 0;
    while (count < 4) {
      const idx = battleSrc.indexOf(needle, from);
      if (idx < 0) break;
      const line = battleSrc.slice(battleSrc.lastIndexOf('\n', idx) + 1, battleSrc.indexOf('\n', idx));
      console.log(`  ${needle} @${idx}: ${line.trim().slice(0, 160)}`);
      from = idx + 1;
      count += 1;
    }
  }
} catch (e) { console.log('  (could not read battle.js:', e.message, ')'); }
