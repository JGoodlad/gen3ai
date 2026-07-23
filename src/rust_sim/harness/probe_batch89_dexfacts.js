// probe_batch89_dexfacts.js — dump the RESOLVED gen3 handlers/metadata for every
// Batch-8/9 move/ability/item. This is the SOURCE-READ hypothesis layer only — every
// draw-model claim must still be confirmed by a live BattleStream probe (the mod-chain
// has burned us repeatedly). We read `Dex.mod('gen3')` (== forFormat gen3customgame)
// so we see the gen3-inherited+overridden dist, NEVER the base .ts.
//
// Run: node src/rust_sim/harness/probe_batch89_dexfacts.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Dex } = require(path.join(PS, 'dist/sim'));

const d = Dex.forFormat('gen3customgame');

function handlerKeys(obj) {
  if (!obj) return [];
  const ks = [];
  for (const k of Object.keys(obj)) {
    if (/^on[A-Z]/.test(k) || /Priority$/.test(k) || /Order$/.test(k)) ks.push(k);
  }
  return ks;
}
function fnBody(f) {
  if (typeof f !== 'function') return String(f);
  return f.toString().replace(/\s+/g, ' ').slice(0, 400);
}

function dumpMove(id) {
  const m = d.moves.get(id);
  if (!m.exists) { console.log(`  MOVE ${id}: DOES NOT EXIST`); return; }
  console.log(`\n=== MOVE ${id} (${m.name}) ===`);
  console.log(`  num=${m.num} type=${m.type} cat=${m.category} bp=${m.basePower} acc=${JSON.stringify(m.accuracy)} pp=${m.pp} prio=${m.priority} target=${m.target}`);
  console.log(`  flags=${JSON.stringify(m.flags)}`);
  for (const f of ['volatileStatus', 'sideCondition', 'slotCondition', 'status', 'boosts', 'self', 'secondary', 'secondaries', 'forceSwitch', 'stallingMove', 'ignoreImmunity', 'breaksProtect', 'sleepUsable', 'smartTarget', 'multihit', 'multiaccuracy', 'nonGhostTarget', 'pseudoWeather', 'onDisableMove']) {
    if (m[f] !== undefined) console.log(`  ${f}=${JSON.stringify(m[f])}`);
  }
  for (const k of handlerKeys(m)) console.log(`  ${k}: ${typeof m[k] === 'function' ? fnBody(m[k]) : JSON.stringify(m[k])}`);
  if (m.condition) {
    console.log(`  .condition:`);
    if (m.condition.duration !== undefined) console.log(`    duration=${m.condition.duration}`);
    if (m.condition.durationCallback) console.log(`    durationCallback: ${fnBody(m.condition.durationCallback)}`);
    for (const k of handlerKeys(m.condition)) console.log(`    ${k}: ${typeof m.condition[k] === 'function' ? fnBody(m.condition[k]) : JSON.stringify(m.condition[k])}`);
  }
}

function dumpAbility(id) {
  const a = d.abilities.get(id);
  if (!a.exists) { console.log(`  ABILITY ${id}: DOES NOT EXIST`); return; }
  console.log(`\n=== ABILITY ${id} (${a.name}) ===`);
  console.log(`  num=${a.num} rating=${a.rating}`);
  for (const k of handlerKeys(a)) console.log(`  ${k}: ${typeof a[k] === 'function' ? fnBody(a[k]) : JSON.stringify(a[k])}`);
  if (a.condition) {
    console.log(`  .condition:`);
    for (const k of handlerKeys(a.condition)) console.log(`    ${k}: ${typeof a.condition[k] === 'function' ? fnBody(a.condition[k]) : JSON.stringify(a.condition[k])}`);
  }
}

function dumpItem(id) {
  const it = d.items.get(id);
  if (!it.exists) { console.log(`  ITEM ${id}: DOES NOT EXIST`); return; }
  console.log(`\n=== ITEM ${id} (${it.name}) ===`);
  console.log(`  num=${it.num} fling=${JSON.stringify(it.fling)} isBerry=${!!it.isBerry} ignoreKlutz=${!!it.ignoreKlutz}`);
  for (const f of ['onlyPokemon', 'itemUser', 'boosts', 'critRatio', 'megaStone']) {
    if (it[f] !== undefined) console.log(`  ${f}=${JSON.stringify(it[f])}`);
  }
  for (const k of handlerKeys(it)) console.log(`  ${k}: ${typeof it[k] === 'function' ? fnBody(it[k]) : JSON.stringify(it[k])}`);
  if (it.condition) {
    console.log(`  .condition:`);
    for (const k of handlerKeys(it.condition)) console.log(`    ${k}: ${typeof it.condition[k] === 'function' ? fnBody(it.condition[k]) : JSON.stringify(it.condition[k])}`);
  }
}

function main() {
  console.log('################ BATCH 8 ################');
  console.log('---- haze ----'); dumpMove('haze');
  console.log('\n---- trick / switcheroo ----'); dumpMove('trick'); dumpMove('switcheroo');
  console.log('\n---- yawn ----'); dumpMove('yawn');
  // yawn resolves a sleep — inspect the sleep condition source too
  console.log('\n---- slp status condition (for yawn resolve draw) ----');
  const slp = d.conditions.get('slp') || d.data.Conditions && d.data.Conditions.slp;
  if (slp) { for (const k of handlerKeys(slp)) console.log(`  slp.${k}: ${fnBody(slp[k])}`); if (slp.durationCallback) console.log(`  slp.durationCallback: ${fnBody(slp.durationCallback)}`); }
  console.log('\n---- partial-trap family (wrap/bind/firespin/clamp/sandtomb/whirlpool) ----');
  for (const id of ['wrap', 'bind', 'firespin', 'clamp', 'sandtomb', 'whirlpool']) dumpMove(id);
  console.log('\n---- partiallytrapped condition ----');
  const pt = d.conditions.get('partiallytrapped');
  if (pt && pt.exists !== false) { console.log(`  duration=${pt.duration}`); if (pt.durationCallback) console.log(`  durationCallback: ${fnBody(pt.durationCallback)}`); for (const k of handlerKeys(pt)) console.log(`  ${k}: ${fnBody(pt[k])}`); }

  console.log('\n\n################ BATCH 9 ################');
  console.log('---- transform ----'); dumpMove('transform');
  const tf = d.conditions.get('transform');
  if (tf && tf.exists !== false) { console.log('  transform.condition:'); for (const k of handlerKeys(tf)) console.log(`    ${k}: ${fnBody(tf[k])}`); }
  console.log('\n---- wonderguard ----'); dumpAbility('wonderguard');
  console.log('\n---- forecast ----'); dumpAbility('forecast');
  console.log('\n---- liquidooze ----'); dumpAbility('liquidooze');
  console.log('\n---- whiteherb ----'); dumpItem('whiteherb');
  console.log('\n---- stick / leek ----'); dumpItem('stick'); dumpItem('leek');

  // Castform formes (for forecast forme map)
  console.log('\n---- Castform formes ----');
  for (const sp of ['castform', 'castformsunny', 'castformrainy', 'castformsnowy']) {
    const s = d.species.get(sp);
    console.log(`  ${sp}: exists=${s.exists} name=${s.name} num=${s.num} types=${JSON.stringify(s.types)} baseSpecies=${s.baseSpecies} forme=${s.forme}`);
  }
}
main();
