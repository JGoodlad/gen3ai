// probe_r35_forecast_ties.js — ROUND 35. The DRAW + ORDER questions that only a SPEED TIE
// can answer (the ROUND 32 lesson: never assume a handler sort can't tie).
//
//   T1  EXPIRING move-weather under CLOUD NINE with TIED actives — `field.clearWeather()`
//       fires `eachEvent('WeatherChange')` UNCONDITIONALLY (field.ts:97) while the port gates
//       the expiry shuffle on `effective`. Does the sim draw where the port would not?
//       (Control: the same board with a non-suppressing ability.)
//   T2  the CLOUD NINE holder FAINTS under standing rain — does the corpse's ability `onEnd`
//       WeatherChange re-forme a Castform, and in what byte position?
//   T3  TWO tied Castforms + weather EXPIRY — the `-formechange` emission order is the
//       shuffle's permutation (not a fixed side walk)?
//   T4  a Castform switching in UNDER a suppressor, then the SUPPRESSOR switching out —
//       exact line order around the `|switch|`.
//   T5  Forecast + SKILL SWAP / a non-Castform holder of `forecast` (the baseSpecies gate).
//   T6  a Castform whose ability is swapped AWAY while formed — does it keep the forme?
//
// Run: node src/rust_sim/harness/probe_r35_forecast_ties.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));
const { mon } = require('./probe_batch4_lib');

function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(teams, seed, choices, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const omni = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of String(ch).split('\n')) omni.push(l); })();
  streams.omniscient.write(`>start {"formatid":"${opts.fmt || 'gen3customgame'}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(teams[0]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(teams[1]) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const battle = stream.battle;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng); let nextCount = 0;
  const shuffles = [];
  rng.next = (...a) => { nextCount += 1; return realNext(...a); };
  const realRandom = battle.random.bind(battle);
  battle.random = (m, n) => { const v = realRandom(m, n); shuffles.push([m === undefined ? null : m, n === undefined ? null : n]); return v; };
  const per = [];
  let lo = omni.length, nLo = 0, sLo = 0;
  for (const [c1, c2] of choices) {
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 16; k++) await tick();
    per.push({
      omni: omni.slice(lo), draws: nextCount - nLo, calls: shuffles.slice(sLo),
      state: opts.onBoundary ? opts.onBoundary(battle) : null,
    });
    lo = omni.length; nLo = nextCount; sLo = shuffles.length;
    if (battle.ended) break;
  }
  return { battle, per, omni };
}

const INTEREST = /formechange|-weather|^\|switch\||^\|drag\||faint|-activate|-endability|-ability|-damage/;
function show(tag, d) {
  console.log(`  ${tag} draws=${d.draws} calls=${JSON.stringify(d.calls)}`);
  for (const l of d.omni.filter((l) => INTEREST.test(l))) console.log(`      ${JSON.stringify(l)}`);
  if (d.state) console.log(`      STATE ${JSON.stringify(d.state)}`);
}
const cfState = (b) => {
  const all = [];
  for (const s of b.sides) for (const p of s.pokemon) {
    if (p.baseSpecies.baseSpecies === 'Castform' || p.species.baseSpecies === 'Castform') {
      all.push({ n: `${p.side.id}:${p.name}`, sp: p.species.id, ty: p.types, ab: p.ability, fnt: p.fainted });
    }
  }
  return {
    cf: all, raw: b.field.weather,
    spe: b.sides.map((s) => s.active[0] && s.active[0].speed),
    eff: b.sides[0].active[0] ? b.sides[0].active[0].effectiveWeather() : null,
  };
};

async function main() {
  // -------------------------------------------------------------------- T1
  // Castform (base spe 70) vs Ledian (base spe 85) -> tune EVs so both land 70-ish? Simpler:
  // use two mons whose *final* speed ties. Castform 70 base, 0 EV/IV31, L100 -> 236.
  // Use a MIRROR: both sides Castform (spe tie by construction), one with Cloud Nine
  // is impossible (Castform's only ability is Forecast) -> instead pair Castform vs a
  // Cloud-Nine mon EV-tuned to the same final speed.
  console.log('############ T1: EXPIRING hail under CLOUD NINE, TIED speeds — does clearWeather draw?');
  {
    const dex = Dex.mod('gen3');
    // Castform: base 70, 31 IV, 0 EV, L100, neutral -> 2*70+31+5 = 176... compute properly.
    // stat = floor((2*base + iv + floor(ev/4)) * level/100) + 5
    const st = (base, ev) => Math.floor(((2 * base + 31 + Math.floor(ev / 4)) * 100) / 100) + 5;
    const cfSpe = st(70, 0);
    // Psyduck base spe 55 -> needs (2*55+31+ev/4)+5 == cfSpe -> ev/4 = cfSpe-5-141
    const need = (cfSpe - 5 - (2 * 55 + 31)) * 4;
    console.log(`      castform spe=${cfSpe} psyduck spe_evs_needed=${need}`);
    for (const [tag, ab] of [['cloudnine', 'Cloud Nine'], ['control', 'Damp']]) {
      const r = await run([
        [mon('Castform', ['hail', 'splash'], { ability: 'Forecast' })],
        [mon('Psyduck', ['splash'], { ability: ab, evs: { spe: need } })],
      ], [3, 5, 7, 9], Array.from({ length: 7 }, (_, i) => [i === 0 ? 'move 1' : 'move 2', 'move 1']), { onBoundary: cfState });
      console.log(`  --- ${tag} perTurnDraws=${JSON.stringify(r.per.map((d) => d.draws))}`);
      r.per.forEach((d, i) => show(`t${i + 1}`, d));
    }
  }

  // -------------------------------------------------------------------- T2
  console.log('\n############ T2: the CLOUD NINE holder FAINTS under standing rain');
  {
    const r = await run([
      [mon('Castform', ['raindance', 'hydropump'], { ability: 'Forecast' })],
      [mon('Psyduck', ['splash'], { ability: 'Cloud Nine', evs: { hp: 0 } }), mon('Blissey', ['splash'])],
    ], [3, 5, 7, 9], [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'switch 2'], ['move 2', 'move 1']], { onBoundary: cfState });
    r.per.forEach((d, i) => show(`t${i + 1}`, d));
  }

  // -------------------------------------------------------------------- T3
  console.log('\n############ T3: TWO Castforms + weather EXPIRY — formechange order vs shuffle');
  for (const seed of [[1, 2, 3, 4], [5, 5, 5, 5], [9, 8, 7, 6], [2, 2, 2, 2]]) {
    const r = await run([
      [mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' })],
      [mon('Castform', ['splash'], { ability: 'Forecast' })],
    ], seed, [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1']], { onBoundary: cfState });
    console.log(`  --- seed=${JSON.stringify(seed)} draws=${JSON.stringify(r.per.map((d) => d.draws))}`);
    r.per.forEach((d, i) => show(`t${i + 1}`, d));
  }

  // -------------------------------------------------------------------- T4
  console.log('\n############ T4: Castform switches IN under a suppressor, suppressor leaves');
  {
    const r = await run([
      [mon('Rattata', ['raindance'], { ability: 'Guts' }), mon('Castform', ['splash'], { ability: 'Forecast' })],
      [mon('Psyduck', ['splash'], { ability: 'Cloud Nine' }), mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ], [3, 5, 7, 9], [['move 1', 'move 1'], ['switch 2', 'move 1'], ['move 1', 'switch 2'], ['move 1', 'switch 2']], { onBoundary: cfState });
    r.per.forEach((d, i) => show(`t${i + 1}`, d));
  }

  // -------------------------------------------------------------------- T5/T6
  console.log('\n############ T5: SKILL SWAP — forecast onto a non-Castform / off a Castform');
  {
    const r = await run([
      [mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' })],
      [mon('Alakazam', ['skillswap', 'splash'], { ability: 'Synchronize' })],
    ], [3, 5, 7, 9], [['move 1', 'move 2'], ['move 2', 'move 1'], ['move 2', 'move 2'], ['move 2', 'move 2'], ['move 2', 'move 2'], ['move 2', 'move 2']], { onBoundary: cfState });
    r.per.forEach((d, i) => show(`t${i + 1}`, d));
  }
  console.log('\n############ T6: ROLE PLAY — an Alakazam copying Forecast, then rain');
  {
    const r = await run([
      [mon('Castform', ['splash'], { ability: 'Forecast' })],
      [mon('Alakazam', ['roleplay', 'raindance'], { ability: 'Synchronize' })],
    ], [3, 5, 7, 9], [['move 1', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 1']], { onBoundary: cfState });
    r.per.forEach((d, i) => show(`t${i + 1}`, d));
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
