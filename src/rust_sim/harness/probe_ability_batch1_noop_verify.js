// probe_ability_batch1_noop_verify.js — PROVE (or refute) that each class-(a) candidate no-op
// ability is a TRUE no-op in the e2e's MODELED move/item universe (damaging-move-only, no-PP-
// pressure, no item-removal, no attract/sleep/OHKO/recoil/drain), INCLUDING under weather.
//
// The rigorous test: run a full battle with the CANDIDATE ability on a mon, then the IDENTICAL
// battle with a NO-OP control ability (Insomnia), same teams/seed/choices — and assert the
// per-decision STATE + the post-turn SEED are BIT-IDENTICAL. If they diverge, the ability is NOT
// a no-op in this universe and must NOT be admitted. Weather (Sand via a Tyranitar foe) is present
// so a weather-reactive ability (Forecast) is STRESSED.
//
// The RESOLVED Dex.mod('gen3') is the oracle. Candidates (from the enumerate probe's handler read):
//   plus / minus  — `onModifySpA` gated on a PARTNER with the paired ability → impossible in
//                   SINGLES (no partner) → no-op.
//   lightningrod  — `onFoeRedirectTarget` (redirect Electric moves) → N/A in singles (one target).
//   stickyhold    — `onTakeItem` (block Thief/Knock Off) → no item-removal move is modeled.
//   forecast      — `onWeatherChange` CHANGES CASTFORM's forme+TYPE in weather → NOT a pure no-op
//                   if a Castform is on the field with weather (its type changes → damage changes).
//                   This probe SHOWS whether it diverges (it should, under weather) → DEFER it.
//
// Run: node src/rust_sim/harness/probe_ability_batch1_noop_verify.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: opts.nature || 'Serious', level: 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Drive a full battle to game-end with a fixed plan; capture per-decision (seedAfter, both HP,
// both status, weather). Returns the trace.
async function run(p1sets, p2sets, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1sets) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2sets) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  const trace = [];
  let safety = 0;
  while (!b.ended && safety < 200) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    // Always: p1 move 1, p2 move 1 (or forced switch 2).
    if (rs === 'switch') {
      for (let i = 0; i < 2; i++) {
        const req = b.sides[i].activeRequest;
        if (req && req.forceSwitch && req.forceSwitch[0]) streams.omniscient.write(`>p${i + 1} switch 2`);
      }
    } else {
      streams.omniscient.write('>p1 move 1');
      streams.omniscient.write('>p2 move 1');
    }
    for (let k = 0; k < 16; k++) await tick();
    const a0 = b.sides[0].active[0], a1 = b.sides[1].active[0];
    trace.push([b.prng.getSeed(), a0.hp, a0.status || '-', a1.hp, a1.status || '-', b.field.weather || '-', a0.species.name, a1.species.name].join('|'));
  }
  try { streams.omniscient.destroy(); } catch (e) {}
  return trace;
}

async function verify(label, candAbility, holderSpecies, holderMoves, opts = {}) {
  // Same board, once with the candidate ability, once with Insomnia (a no-op control). A Tyranitar
  // foe keeps SAND up (stresses a weather-reactive ability like Forecast).
  const seed = [33, 71, 155, 4021];
  const foe = [mon('Tyranitar', ['crunch', 'rest'], { ability: 'Sand Stream', nature: 'Careful', evs: { hp: 252, spd: 252 } })];
  const cand = [mon(holderSpecies, holderMoves, { ability: candAbility, ...opts })];
  const ctrl = [mon(holderSpecies, holderMoves, { ability: 'Insomnia', ...opts })];
  const tc = await run(cand, foe, seed);
  const tk = await run(ctrl, foe, seed);
  const n = Math.min(tc.length, tk.length);
  let firstDiff = -1;
  for (let i = 0; i < n; i++) { if (tc[i] !== tk[i]) { firstDiff = i; break; } }
  const same = firstDiff === -1 && tc.length === tk.length;
  console.log(`${same ? 'NO-OP ✓' : 'DIVERGES ✗'}  ${label} (${candAbility} on ${holderSpecies})  [${tc.length} vs ${tk.length} decisions]`);
  if (!same) {
    console.log(`    first diff at decision ${firstDiff}:`);
    console.log(`      cand: ${tc[firstDiff] || '(ended)'}`);
    console.log(`      ctrl: ${tk[firstDiff] || '(ended)'}`);
  }
  return same;
}

async function main() {
  console.log('=== class-(a) no-op verification (candidate vs Insomnia control, SAND up) ===');
  const results = {};
  // plus/minus: put on a special attacker; no partner in singles → no SpA boost.
  results.plus = await verify('Plus', 'Plus', 'Manectric', ['thunderbolt', 'rest']);
  results.minus = await verify('Minus', 'Minus', 'Manectric', ['thunderbolt', 'rest']);
  // lightningrod: redirect is N/A in singles.
  results.lightningrod = await verify('Lightning Rod', 'Lightning Rod', 'Rhydon', ['earthquake', 'rest'], { evs: { hp: 252, atk: 252 }, nature: 'Adamant' });
  // stickyhold: no item-removal move modeled.
  results.stickyhold = await verify('Sticky Hold', 'Sticky Hold', 'Muk', ['sludgebomb', 'rest'], { evs: { hp: 252, spa: 252 }, nature: 'Modest' });
  // forecast: Castform changes forme+TYPE under weather → EXPECT DIVERGES.
  results.forecast = await verify('Forecast', 'Forecast', 'Castform', ['icebeam', 'rest'], { evs: { spa: 252 }, nature: 'Modest' });

  console.log('\n=== VERDICT ===');
  const safe = Object.entries(results).filter(([, v]) => v).map(([k]) => k);
  const unsafe = Object.entries(results).filter(([, v]) => !v).map(([k]) => k);
  console.log(`  ADMIT as no-op: ${safe.join(', ') || '(none)'}`);
  console.log(`  DEFER (NOT a no-op): ${unsafe.join(', ') || '(none)'}`);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
