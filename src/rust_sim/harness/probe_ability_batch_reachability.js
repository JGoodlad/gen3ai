// probe_ability_batch_reachability.js — the e2e PAYOFF analysis for batch-1 abilities.
//
// For the CURRENT teams pool, compute (a) each candidate ability's team-carry count (how
// many validated teams carry it at all), (b) how many teams are blocked ONLY on ability
// gaps that a given candidate SET would clear (item gaps still block), and (c) the
// filter-clean count if a given candidate ability SET is admitted. This ranks which
// abilities actually move the e2e filter-clean pool (the batch's payoff metric) so we wire
// what matters + honestly report what a class admits.
//
// Run: node src/rust_sim/harness/probe_ability_batch_reachability.js

'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Teams } = require(path.join(PS, 'dist/sim'));
const e2e = require('./gen_e2e_fuzz.js');

const { loadTeams, MODELED_ABILITIES, NOOP_ABILITIES, MODELED_ITEMS } = e2e;
function toId(s) { return ('' + (s || '')).toLowerCase().replace(/[^a-z0-9]/g, ''); }

const { teams } = loadTeams();
console.log(`Validated teams: ${teams.length}`);

// Current allow-set (ability).
const curAbil = new Set([...MODELED_ABILITIES, ...NOOP_ABILITIES, '']);
function itemOk(it) { return MODELED_ITEMS.has(toId(it)); }

// Per-team: the set of ability gaps + item gaps under the CURRENT allow-set.
const teamGaps = teams.map((t) => {
  const team = Teams.unpack(t.packed);
  const aGaps = new Set(), iGaps = new Set();
  for (const set of team) {
    const a = toId(set.ability), it = toId(set.item);
    if (!curAbil.has(a)) aGaps.add(a);
    if (!itemOk(it)) iGaps.add(it);
  }
  return { file: t.file, aGaps, iGaps };
});

const curClean = teamGaps.filter((g) => g.aGaps.size === 0 && g.iGaps.size === 0).length;
console.log(`Currently filter-clean (baseline): ${curClean}`);
console.log('');

// The batch-1 candidate abilities (class-a no-op + class-b draw-free/structural), i.e.
// everything NOT yet admitted that we intend to wire or admit in THIS batch.
const CANDIDATES = [
  // class-b draw-free/structural (to WIRE):
  'battlearmor', 'shellarmor', 'chlorophyll', 'swiftswim', 'cloudnine', 'airlock',
  'speedboost', 'raindish', 'suctioncups', 'soundproof', 'damp',
  // structural gates (assess): wonderguard, truant, innerfocus
  'wonderguard', 'truant', 'innerfocus',
  // class-a no-op candidates (to ADMIT as no-ops if proven):
  'lightningrod', 'stickyhold', 'plus', 'minus', 'forecast', 'colorchange',
  // class-c draw-bearing (DEFER — listed for completeness of the pool):
  'static', 'poisonpoint', 'flamebody', 'cutecharm', 'effectspore', 'roughskin',
  'synchronize', 'trace', 'shedskin', 'shadowtag', 'liquidooze',
];

// team-carry count for each candidate.
console.log('=== team-carry count (validated teams carrying this ability at all) ===');
const carry = {};
for (const a of CANDIDATES) carry[a] = 0;
for (const t of teams) {
  const team = Teams.unpack(t.packed);
  const seen = new Set(team.map((s) => toId(s.ability)));
  for (const a of CANDIDATES) if (seen.has(a)) carry[a] += 1;
}
for (const a of CANDIDATES.slice().sort((x, y) => carry[y] - carry[x])) {
  if (carry[a] > 0) console.log(`   ${a.padEnd(14)} ${carry[a]}`);
}
console.log('');

// Incremental filter-clean if we admit a SET of abilities (item gaps still block).
function cleanIfAdmit(abilitySet) {
  const allow = new Set([...curAbil, ...abilitySet]);
  let clean = 0;
  for (const g of teamGaps) {
    // Recompute ability gaps under the expanded allow-set; item gaps unchanged.
    let aOk = true;
    for (const a of g.aGaps) if (!allow.has(a)) { aOk = false; break; }
    if (aOk && g.iGaps.size === 0) clean += 1;
  }
  return clean;
}

// The batch we can wire cleanly (class-a no-ops + class-b structural we prove).
const BATCH_B = ['battlearmor', 'shellarmor', 'chlorophyll', 'swiftswim', 'cloudnine', 'airlock', 'speedboost', 'raindish', 'suctioncups', 'soundproof', 'damp'];
const BATCH_A_NOOP = ['wonderguard', 'truant', 'innerfocus', 'lightningrod', 'stickyhold', 'plus', 'minus', 'forecast', 'colorchange'];

console.log('=== filter-clean if we admit candidate sets (item gaps still block) ===');
console.log(`   baseline                       : ${curClean}`);
console.log(`   + class-b (crit/weather/residual/block: ${BATCH_B.join(',')}): ${cleanIfAdmit(BATCH_B)}`);
console.log(`   + class-a no-ops (${BATCH_A_NOOP.join(',')}): ${cleanIfAdmit(BATCH_A_NOOP)}`);
console.log(`   + BOTH class-a & class-b        : ${cleanIfAdmit([...BATCH_B, ...BATCH_A_NOOP])}`);
console.log(`   + EVERYTHING incl deferred draw-bearing: ${cleanIfAdmit(CANDIDATES)}`);
console.log('');

// Which teams are blocked ONLY on ability gaps that the FULL non-draw-bearing batch clears
// (i.e., would become clean if items also cleared)? And of those, which are ALSO item-clean
// (the real new-clean teams)?
const nonDraw = [...BATCH_B, ...BATCH_A_NOOP];
const allowNon = new Set([...curAbil, ...nonDraw]);
let newlyClean = 0, blockedOnItemsOnly = 0;
for (const g of teamGaps) {
  let aOk = true; for (const a of g.aGaps) if (!allowNon.has(a)) { aOk = false; break; }
  const wasClean = g.aGaps.size === 0 && g.iGaps.size === 0;
  if (aOk && g.iGaps.size === 0 && !wasClean) newlyClean += 1;
  if (aOk && g.iGaps.size > 0) blockedOnItemsOnly += 1;
}
console.log(`Teams that become NEWLY filter-clean under the non-draw-bearing batch: ${newlyClean}`);
console.log(`Teams whose ABILITY gaps clear under the batch but still item-blocked: ${blockedOnItemsOnly}`);

// Report the residual ability-gap ranking (what's LEFT after this batch) — the taxonomy update.
console.log('');
console.log('=== residual ability gaps AFTER the non-draw-bearing batch (still-blocking, ranked) ===');
const resid = {};
for (const g of teamGaps) {
  for (const a of g.aGaps) if (!allowNon.has(a)) resid[a] = (resid[a] || 0) + 1;
}
for (const [a, n] of Object.entries(resid).sort((x, y) => y[1] - x[1]).slice(0, 20)) {
  console.log(`   ${a.padEnd(16)} ${n}`);
}
