// probe_r35_forecast_edges.js — ROUND 35. The EDGES the two committed r35 probes left open:
// the turn-0 CONSTRUCTION window (a Castform LEAD), the faint sites, the Cloud-Nine
// composition's DRAW question on an EXPIRING weather, Transform x Forecast lifecycle, and the
// downstream type consequences (STAB / effectiveness / status immunity).
//
// Questions (each answered by raw bytes + draw counts):
//   E1  a Castform LEAD vs a weather-setting LEAD: where does the `-formechange` land in the
//       battle-start FRAMING (which lead's block), and in what order at a speed tie?
//   E2  a Castform LEAD under NO weather: silent (no line)?
//   E3  the Castform FAINTS while formed — any line? does the corpse still forme?
//   E4  the WEATHER-SETTER faints (its permanent weather persists) — no WeatherChange.
//   E5  an EXPIRING move-weather under Cloud Nine: does `clearWeather`'s eachEvent(
//       'WeatherChange') still DRAW (the port gates the expiry shuffle on `effective`)?
//   E6  Cloud Nine mon FAINTS while Castform is Normal under suppressed rain — does the
//       corpse's ability `onEnd` fire the WeatherChange -> forme?
//   E7  Transform lifecycle: rain ENDS while a Ditto is `castformrainy`; and a CASTFORM that
//       itself transformed then reverts.
//   E8  type consequences: a Rainy Castform's STAB + the effectiveness BOTH ways, and a
//       Snowy Castform's Ice-type status immunity (freeze).
//
// Run: node src/rust_sim/harness/probe_r35_forecast_edges.js
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
  const framing = omni.slice();
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng); let nextCount = 0;
  rng.next = (...a) => { nextCount += 1; return realNext(...a); };
  const per = [];
  let lo = omni.length, nLo = 0;
  for (const [c1, c2] of choices) {
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 16; k++) await tick();
    per.push({ omni: omni.slice(lo), draws: nextCount - nLo, state: opts.onBoundary ? opts.onBoundary(battle) : null });
    lo = omni.length; nLo = nextCount;
    if (battle.ended) break;
  }
  return { battle, per, framing, omni };
}

const INTEREST = /formechange|-weather|^\|switch\||^\|drag\||^\|transform\||-damage|-supereffective|-resisted|-immune|-status|faint|-crit|-ability|-end\|/;
function show(tag, d) {
  console.log(`  ${tag} draws=${d.draws}`);
  for (const l of d.omni.filter((l) => INTEREST.test(l))) console.log(`      ${JSON.stringify(l)}`);
  if (d.state) console.log(`      STATE ${JSON.stringify(d.state)}`);
}
const cfState = (b) => {
  const all = [];
  for (const s of b.sides) for (const p of s.pokemon) {
    if (p.baseSpecies.baseSpecies === 'Castform' || p.species.baseSpecies === 'Castform') {
      all.push({ n: p.name, sp: p.species.id, ty: p.types, tr: !!p.transformed, fnt: p.fainted, act: p.isActive });
    }
  }
  return { cf: all, raw: b.field.weather, eff: b.sides[0].active[0] ? b.sides[0].active[0].effectiveWeather() : null };
};

async function main() {
  // ------------------------------------------------------------------- E1/E2
  console.log('############ E1: Castform LEAD + a weather LEAD — framing placement');
  for (const [tag, foe] of [
    ['drizzle-foe', mon('Politoed', ['splash'], { ability: 'Drizzle' })],
    ['sandstream-foe', mon('Tyranitar', ['splash'], { ability: 'Sand Stream' })],
    ['drought-foe', mon('Ninetales', ['splash'], { ability: 'Drought' })],
    ['plain-foe', mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
  ]) {
    const r = await run([
      [mon('Castform', ['splash'], { ability: 'Forecast' })],
      [foe],
    ], [3, 5, 7, 9], [], {});
    console.log(`  --- ${tag}`);
    for (const l of r.framing.filter((l) => INTEREST.test(l) || l.startsWith('|turn|'))) console.log(`      ${JSON.stringify(l)}`);
    console.log(`      STATE ${JSON.stringify(cfState(r.battle))}`);
  }

  console.log('\n############ E1b: BOTH leads Castform + a rain move (tie order in framing)');
  for (const seed of [[1, 2, 3, 4], [5, 5, 5, 5]]) {
    const r = await run([
      [mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' })],
      [mon('Castform', ['splash'], { ability: 'Forecast' })],
    ], seed, [['move 1', 'move 1']], { onBoundary: cfState });
    console.log(`  --- seed=${JSON.stringify(seed)}`);
    show('t1', r.per[0]);
  }

  // --------------------------------------------------------------------- E3
  console.log('\n############ E3: the formed Castform FAINTS (rain up) — lines + corpse forme');
  {
    const r = await run([
      [mon('Castform', ['raindance', 'splash'], { ability: 'Forecast', evs: { hp: 0 } }), mon('Rattata', ['scratch'])],
      [mon('Machamp', ['crosschop', 'splash'], { ability: 'Guts' })],
    ], [3, 5, 7, 9], [['move 1', 'move 2'], ['move 2', 'move 1'], ['switch 2', ''], ['move 1', 'move 2']], { onBoundary: cfState });
    r.per.forEach((d, i) => show(`t${i + 1}`, d));
  }

  // --------------------------------------------------------------------- E4
  console.log('\n############ E4: the SAND-STREAM setter faints — permanent sand persists');
  {
    const r = await run([
      [mon('Castform', ['hail', 'splash'], { ability: 'Forecast' }), mon('Rattata', ['scratch'])],
      [mon('Tyranitar', ['splash'], { ability: 'Sand Stream', evs: { hp: 0 } }), mon('Blissey', ['splash'])],
    ], [3, 5, 7, 9], [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1']], { onBoundary: cfState });
    r.per.forEach((d, i) => show(`t${i + 1}`, d));
  }

  // --------------------------------------------------------------------- E5
  console.log('\n############ E5: EXPIRING move-weather under CLOUD NINE — does clearWeather draw?');
  for (const [tag, ab] of [['cloudnine', 'Cloud Nine'], ['control-levitate', 'Levitate']]) {
    const r = await run([
      [mon('Castform', ['hail', 'splash'], { ability: 'Forecast' })],
      [mon('Psyduck', ['splash'], { ability: ab })],
    ], [3, 5, 7, 9], [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1']], { onBoundary: cfState });
    console.log(`  --- ${tag} perTurnDraws=${JSON.stringify(r.per.map((d) => d.draws))}`);
    r.per.forEach((d, i) => show(`t${i + 1}`, d));
  }

  // --------------------------------------------------------------------- E6
  console.log('\n############ E6: the CLOUD NINE mon FAINTS under standing rain');
  {
    const r = await run([
      [mon('Castform', ['raindance', 'hydropump'], { ability: 'Forecast' })],
      [mon('Psyduck', ['splash'], { ability: 'Cloud Nine', evs: { hp: 0 } }), mon('Blissey', ['splash'])],
    ], [3, 5, 7, 9], [['move 1', 'move 1'], ['move 2', 'move 1'], ['', 'switch 2']], { onBoundary: cfState });
    r.per.forEach((d, i) => show(`t${i + 1}`, d));
  }

  // --------------------------------------------------------------------- E7
  console.log('\n############ E7: Transform x Forecast lifecycle');
  {
    // Ditto copies a RAINY Castform, then the rain EXPIRES.
    const r = await run([
      [mon('Ditto', ['transform'], { ability: 'Limber' }), mon('Rattata', ['scratch'])],
      [mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' })],
    ], [3, 5, 7, 9], [['move 1', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['switch 2', 'move 2'], ['switch 2', 'move 2']], { onBoundary: cfState });
    r.per.forEach((d, i) => show(`ditto-t${i + 1}`, d));
  }
  {
    // The CASTFORM itself transforms (into Rattata) under rain, then reverts on switch-out.
    const r = await run([
      [mon('Castform', ['raindance', 'transform'], { ability: 'Forecast' }), mon('Rattata', ['scratch'])],
      [mon('Snorlax', ['splash'], { ability: 'Immunity' })],
    ], [3, 5, 7, 9], [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 1', 'move 1'], ['switch 2', 'move 1'], ['switch 2', 'move 1']], { onBoundary: cfState });
    r.per.forEach((d, i) => show(`cf-transform-t${i + 1}`, d));
  }

  // --------------------------------------------------------------------- E8
  console.log('\n############ E8: type consequences (STAB / effectiveness / status immunity)');
  {
    // Rainy Castform (Water): Water Pulse gets STAB; incoming Thunderbolt is SE (2x).
    const r = await run([
      [mon('Castform', ['raindance', 'waterpulse'], { ability: 'Forecast', evs: { hp: 252 } })],
      [mon('Zapdos', ['thunderbolt', 'splash'], { ability: 'Pressure' })],
    ], [3, 5, 7, 9], [['move 1', 'move 2'], ['move 2', 'move 1'], ['move 2', 'move 1']], { onBoundary: cfState });
    r.per.forEach((d, i) => show(`water-t${i + 1}`, d));
  }
  {
    // Snowy Castform (Ice): cannot be FROZEN (Ice-type freeze immunity).
    const r = await run([
      [mon('Castform', ['hail', 'splash'], { ability: 'Forecast' })],
      [mon('Articuno', ['icebeam', 'splash'], { ability: 'Pressure' })],
    ], [3, 5, 7, 9], [['move 1', 'move 2'], ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1']], { onBoundary: cfState });
    r.per.forEach((d, i) => show(`ice-t${i + 1}`, d));
  }
  {
    // Sunny Castform (Fire): burn immunity + hail/sand chip immunity check.
    const r = await run([
      [mon('Castform', ['sunnyday', 'splash'], { ability: 'Forecast' })],
      [mon('Blissey', ['willowisp', 'splash'], { ability: 'Natural Cure' })],
    ], [3, 5, 7, 9], [['move 1', 'move 2'], ['move 2', 'move 1'], ['move 2', 'move 1']], { onBoundary: cfState });
    r.per.forEach((d, i) => show(`fire-t${i + 1}`, d));
  }

  console.log('\n############ E9: dex — every gen3 species whose baseSpecies differs (forme rows)');
  {
    const dex = Dex.mod('gen3');
    const rows = [];
    for (const s of dex.species.all()) {
      if (s.gen > 3 || s.gen === 0) continue;
      if (s.battleOnly || s.forme) rows.push({ id: s.id, name: s.name, base: s.baseSpecies, battleOnly: s.battleOnly, types: s.types, num: s.num });
    }
    console.log(`      ${JSON.stringify(rows)}`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
