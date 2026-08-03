// probe_r35_forecast_order.js — ROUND 35. The DRAW + EMISSION-ORDER half of Forecast.
//
// The port already models the `eachEvent('WeatherChange')` speed-tie SHUFFLE as a bare DRAW
// (`each_event_shuffle`) because Forecast — the ONLY gen3 WeatherChange handler — was
// unmodeled, so nothing consumed the resulting ORDER. Modelling Forecast makes that order
// OBSERVABLE (two Castforms both emit `-formechange`, in eachEvent order). This probe settles:
//
//   O1  which `eachEvent` ids fire, in what order, on: a weather-SET move, a weather EXPIRY,
//       an ability weather set (Drizzle switch-in), and a Cloud-Nine leave.
//   O2  with TWO Forecast Castforms (speed-TIED), does the `-formechange` emission order
//       follow the eachEvent speed-sort/shuffle? (=> the port must CONSUME the shuffle order.)
//   O3  a Castform switching in the SAME turn as a Drizzle mon — the interleaving of
//       `onStart`(-2) vs the Drizzle-driven WeatherChange.
//   O4  does adding the Forecast handler change the DRAW COUNT anywhere? (`runEvent`'s own
//       `speedSort(handlers)` short-circuits at <2 handlers; forecast is the ONLY gen3
//       WeatherChange handler => structurally 1 handler => no new draw. Verified empirically.)
//   O5  Transform x Forecast: the `pokemon.transformed` / `baseSpecies.baseSpecies` gates.
//
// Run: node src/rust_sim/harness/probe_r35_forecast_order.js
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
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(teams[0]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(teams[1]) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const battle = stream.battle;

  const trace = [];
  const realEach = battle.eachEvent.bind(battle);
  battle.eachEvent = (eventid, effect, relayVar) => {
    trace.push({ t: 'eachEvent', eventid });
    return realEach(eventid, effect, relayVar);
  };
  const realShuffle = battle.prng.shuffle.bind(battle.prng);
  battle.prng.shuffle = (...a) => { trace.push({ t: 'shuffle', n: a.length > 1 ? a.slice(1) : [0, a[0].length] }); return realShuffle(...a); };
  const rng = battle.prng.rng; const realNext = rng.next.bind(rng);
  let draws = 0; rng.next = (...a) => { draws += 1; return realNext(...a); };

  const per = [];
  let oLo = omni.length, tLo = 0, dLo = 0;
  for (let i = 0; i < choices.length; i++) {
    const [c1, c2] = choices[i];
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 14; k++) await tick();
    per.push({ omni: omni.slice(oLo), trace: trace.slice(tLo), draws: draws - dLo, state: opts.onBoundary ? opts.onBoundary(battle) : null });
    oLo = omni.length; tLo = trace.length; dLo = draws;
    if (battle.ended) break;
  }
  return { battle, per };
}

const SEED = [3, 1, 4, 1];
const KEEP = /formechange|-weather|^\|switch\||^\|drag\||-ability|-activate|-transform|^\|turn\|/;

function show(tag, d) {
  console.log(`  ${tag} draws=${d.draws} eachEvents=${JSON.stringify(d.trace.filter((x) => x.t === 'eachEvent').map((x) => x.eventid))} shuffles=${JSON.stringify(d.trace.filter((x) => x.t === 'shuffle').map((x) => x.n))}`);
  for (const l of d.omni.filter((l) => KEEP.test(l))) console.log(`      ${JSON.stringify(l)}`);
  if (d.state) console.log(`      STATE ${JSON.stringify(d.state)}`);
}

async function main() {
  console.log('############ O1: eachEvent ids on weather SET (move) / EXPIRY');
  {
    const teams = [
      [mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' })],
      [mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ];
    const r = await run(teams, SEED, Array(6).fill(0).map((_, i) => [i === 0 ? 'move 1' : 'move 2', 'move 1']));
    r.per.forEach((d, i) => show(`t${i + 1}`, d));
  }

  console.log('\n############ O1b: eachEvent ids on an ABILITY weather set (Drizzle switch-in)');
  {
    const teams = [
      [mon('Castform', ['splash'], { ability: 'Forecast' })],
      [mon('Blissey', ['splash'], { ability: 'Natural Cure' }), mon('Politoed', ['splash'], { ability: 'Drizzle' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1'], ['move 1', 'switch 2'], ['move 1', 'move 1']]);
    r.per.forEach((d, i) => show(`t${i + 1}`, d));
  }

  console.log('\n############ O1c: CLOUD NINE leaves -> WeatherChange -> forme');
  {
    const teams = [
      [mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' })],
      [mon('Psyduck', ['splash'], { ability: 'Cloud Nine' }), mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 1'],
      ['move 2', 'switch 2'],
      ['move 2', 'switch 2'],
    ], { onBoundary: (b) => ({ sp: b.p1.active[0].species.id, ty: b.p1.active[0].types, raw: b.field.weather, eff: b.p1.active[0].effectiveWeather() }) });
    r.per.forEach((d, i) => show(`t${i + 1}`, d));
  }

  console.log('\n############ O2: TWO speed-tied Forecast Castforms — emission order vs shuffle');
  for (const seed of [[1, 2, 3, 4], [9, 8, 7, 6], [5, 5, 5, 5], [2, 2, 2, 2]]) {
    const teams = [
      [mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' })],
      [mon('Castform', ['sunnyday', 'splash'], { ability: 'Forecast' })],
    ];
    const r = await run(teams, seed, [['move 1', 'move 2'], ['move 2', 'move 2']]);
    const fc = r.per[0].omni.filter((l) => l.includes('formechange'));
    console.log(`  seed=${JSON.stringify(seed)} draws=${r.per[0].draws} shuffles=${JSON.stringify(r.per[0].trace.filter((x) => x.t === 'shuffle').map((x) => x.n))}`);
    for (const l of r.per[0].omni.filter((l) => /-weather|formechange/.test(l))) console.log(`      ${JSON.stringify(l)}`);
  }

  console.log('\n############ O3: Castform switches IN the same turn a Drizzle mon does');
  {
    const teams = [
      [mon('Blissey', ['splash'], { ability: 'Natural Cure' }), mon('Castform', ['splash'], { ability: 'Forecast' })],
      [mon('Blissey', ['splash'], { ability: 'Natural Cure' }), mon('Politoed', ['splash'], { ability: 'Drizzle' })],
    ];
    const r = await run(teams, SEED, [['switch 2', 'switch 2'], ['move 1', 'move 1']],
      { onBoundary: (b) => ({ sp: b.p1.active[0].species.id, ty: b.p1.active[0].types }) });
    r.per.forEach((d, i) => show(`t${i + 1}`, d));
  }

  console.log('\n############ O4: draw parity Forecast vs control across MANY weather changes');
  {
    for (const ab of ['Forecast', 'Levitate']) {
      const teams = [
        [mon('Castform', ['raindance', 'sunnyday', 'hail', 'sandstorm'], { ability: ab })],
        [mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
      ];
      const r = await run(teams, SEED, [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 3', 'move 1'], ['move 4', 'move 1'], ['move 1', 'move 1'], ['move 2', 'move 1']]);
      console.log(`  ${ab}: draws=${JSON.stringify(r.per.map((d) => d.draws))} eachEvents=${JSON.stringify(r.per.map((d) => d.trace.filter((x) => x.t === 'eachEvent').map((x) => x.eventid)))}`);
    }
  }

  console.log('\n############ O5: Transform x Forecast gates');
  {
    const teams = [
      [mon('Ditto', ['transform'], { ability: 'Limber' })],
      [mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1'], ['move 1', 'move 2']],
      { onBoundary: (b) => ({ dittoSp: b.p1.active[0].species.id, dittoBase: b.p1.active[0].baseSpecies.id, dittoTypes: b.p1.active[0].types, dittoTransformed: b.p1.active[0].transformed, dittoAbility: b.p1.active[0].ability, cfSp: b.p2.active[0].species.id }) });
    r.per.forEach((d, i) => show(`t${i + 1}`, d));
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
