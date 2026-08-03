// probe_r35_forecast_reporting.js — ROUND 35. Settle the FORECAST **REPORTING SURFACE**
// with BYTES, not reasoning. This is the piece the deferral hinged on (BATCH89_RESEARCH.md
// "OWNER DECISION (reporting surface)"): the mechanic/draw-model/Cloud-Nine composition were
// already probed, but NOBODY had read the actual `|switch|` details / `|drag|` details /
// per-side `|request|` JSON for a FORMED Castform.
//
// Questions this settles (each answered by raw bytes):
//   Q1  the exact `-formechange` line bytes + ORDER relative to `-weather`.
//   Q2  does `|switch|` details carry the BASE species or the FORME? (switch-out+in under
//       standing weather; the re-forme happens at onStart AFTER the switch line?)
//   Q3  does the per-side `|request|` JSON `side.pokemon[].details`/`ident` carry base or forme?
//       (this is what poke-env parses — the RL observation surface.)
//   Q4  does `|drag|` (Roar/Whirlwind) details carry base or forme?
//   Q5  REDUNDANT transitions — a weather change that maps to the SAME forme: does it emit a
//       second `-formechange`, or nothing? (rain->rain re-set; sand while already Normal.)
//   Q6  weather EXPIRY revert bytes + ordering vs `|-weather|none`.
//   Q7  a fainted/benched formed Castform: what does the request/`|switch|` report?
//   Q8  the DRAW question re-verified on THIS seed (Forecast vs a control ability).
//
// Run: node src/rust_sim/harness/probe_r35_forecast_reporting.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));
const { mon } = require('./probe_batch4_lib');

function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Run a scripted battle capturing the OMNISCIENT stream AND both PER-SIDE streams
// (the per-side stream is where `|request|` lives — the poke-env surface).
async function runSided(teams, seed, choices, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const omni = [];
  const side = { p1: [], p2: [] };
  (async () => { for await (const ch of streams.omniscient) for (const l of String(ch).split('\n')) omni.push(l); })();
  (async () => { for await (const ch of streams.p1) for (const l of String(ch).split('\n')) side.p1.push(l); })();
  (async () => { for await (const ch of streams.p2) for (const l of String(ch).split('\n')) side.p2.push(l); })();
  streams.omniscient.write(`>start {"formatid":"${opts.fmt || 'gen3customgame'}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(teams[0]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(teams[1]) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const battle = stream.battle;
  // instrument draws
  const prng = battle.prng; const rng = prng.rng;
  const realNext = rng.next.bind(rng); let nextCount = 0;
  rng.next = (...a) => { nextCount += 1; return realNext(...a); };

  const per = [];
  let oLo = omni.length, p1Lo = side.p1.length, p2Lo = side.p2.length, nLo = 0;
  for (let i = 0; i < choices.length; i++) {
    const [c1, c2] = choices[i];
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 14; k++) await tick();
    per.push({
      omni: omni.slice(oLo), p1: side.p1.slice(p1Lo), p2: side.p2.slice(p2Lo),
      draws: nextCount - nLo,
      state: opts.onBoundary ? opts.onBoundary(battle, i) : null,
    });
    oLo = omni.length; p1Lo = side.p1.length; p2Lo = side.p2.length; nLo = nextCount;
    if (battle.ended) break;
  }
  return { battle, per, omni, side };
}

const SEED = [7, 11, 13, 17];
const INTEREST = /formechange|-weather|^\|switch\||^\|drag\||^\|replace\||-start|-end|faint/;

function showTurn(tag, d, opts = {}) {
  const lines = d.omni.filter((l) => INTEREST.test(l));
  console.log(`  ${tag} draws=${d.draws}`);
  for (const l of lines) console.log(`      OMNI ${JSON.stringify(l)}`);
  if (opts.req) {
    for (const l of d.p1) {
      if (!l.startsWith('|request|')) continue;
      const j = JSON.parse(l.slice('|request|'.length));
      const act = (j.active || []).length ? 'active' : (j.forceSwitch ? 'forceSwitch' : (j.wait ? 'wait' : '?'));
      console.log(`      P1REQ[${act}] pokemon=${JSON.stringify((j.side.pokemon || []).map((p) => ({ ident: p.ident, details: p.details, active: p.active })))}`);
    }
  }
  if (d.state) console.log(`      STATE ${JSON.stringify(d.state)}`);
}

const readState = (b) => {
  const a = b.p1.active[0];
  const cf = b.p1.pokemon.find((p) => p.baseSpecies.baseSpecies === 'Castform');
  return {
    activeSpecies: a && a.species.id, activeTypes: a && a.types,
    cfSpecies: cf && cf.species.id, cfDetails: cf && cf.details,
    cfBase: cf && cf.baseSpecies.id, cfName: cf && cf.name,
    rawWeather: b.field.weather, effWeather: a ? a.effectiveWeather() : null,
  };
};

async function main() {
  // ---------------------------------------------------------------- Q1/Q3/Q6
  console.log('############ S1: rain cycle — formechange bytes, request JSON, expiry revert');
  {
    const teams = [
      [mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' }), mon('Rattata', ['scratch'], { ability: 'Guts' })],
      [mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ];
    const r = await runSided(teams, SEED, [
      ['move 1', 'move 1'], // t1 RainDance -> Rainy
      ['move 2', 'move 1'], // t2
      ['move 2', 'move 1'], // t3
      ['move 2', 'move 1'], // t4
      ['move 2', 'move 1'], // t5 rain ends -> revert
      ['move 2', 'move 1'], // t6
    ], { onBoundary: readState });
    r.per.forEach((d, i) => showTurn(`t${i + 1}`, d, { req: i < 2 || i === 4 }));
  }

  // ---------------------------------------------------------------- Q2/Q3/Q7
  console.log('\n############ S2: switch OUT then IN under standing rain — |switch| details + request');
  {
    const teams = [
      [mon('Castform', ['splash'], { ability: 'Forecast' }), mon('Rattata', ['scratch'], { ability: 'Guts' })],
      [mon('Politoed', ['splash'], { ability: 'Drizzle' })],
    ];
    const r = await runSided(teams, SEED, [
      ['move 1', 'move 1'],   // t1 drizzle rain up at construction -> Castform formes
      ['switch 2', 'move 1'], // t2 Castform OUT (benched forme?)
      ['switch 2', 'move 1'], // t3 Castform back IN -> |switch| details?
    ], { onBoundary: readState });
    r.per.forEach((d, i) => showTurn(`t${i + 1}`, d, { req: true }));
  }

  // ---------------------------------------------------------------- Q4
  console.log('\n############ S3: DRAG a formed Castform (Roar) — |drag| details');
  {
    const teams = [
      [mon('Castform', ['splash'], { ability: 'Forecast' }), mon('Rattata', ['scratch'], { ability: 'Guts' })],
      [mon('Politoed', ['roar', 'splash'], { ability: 'Drizzle' })],
    ];
    const r = await runSided(teams, SEED, [
      ['move 1', 'move 2'],   // t1 rain (drizzle) -> Castform Rainy
      ['move 1', 'move 1'],   // t2 Roar drags Castform out, Rattata in
      ['move 1', 'move 1'],   // t3 Roar again -> drags Castform back IN (details?)
    ], { onBoundary: readState });
    r.per.forEach((d, i) => showTurn(`t${i + 1}`, d, { req: true }));
  }

  // ---------------------------------------------------------------- Q5
  console.log('\n############ S4: REDUNDANT transitions — same-forme weather change');
  {
    // sand while Normal (Normal->Normal: emits?); then sun (Normal->Sunny); then
    // sand again (Sunny->Normal); then sand again while sand (fails, no weather change).
    const teams = [
      [mon('Castform', ['sandstorm', 'sunnyday', 'splash'], { ability: 'Forecast' })],
      [mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ];
    const r = await runSided(teams, SEED, [
      ['move 1', 'move 1'], // t1 Sandstorm  (Normal -> Normal: any formechange line?)
      ['move 1', 'move 1'], // t2 Sandstorm again (fails - already sand)
      ['move 2', 'move 1'], // t3 SunnyDay   (Normal -> Sunny)
      ['move 1', 'move 1'], // t4 Sandstorm  (Sunny -> Normal)
      ['move 3', 'move 1'], // t5
    ], { onBoundary: readState });
    r.per.forEach((d, i) => showTurn(`t${i + 1}`, d));
  }

  // ---------------------------------------------------------------- Q5b hail
  console.log('\n############ S5: hail -> Snowy, then rain -> Rainy (forme->forme direct)');
  {
    const teams = [
      [mon('Castform', ['hail', 'raindance', 'splash'], { ability: 'Forecast' })],
      [mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ];
    const r = await runSided(teams, SEED, [
      ['move 1', 'move 1'], // hail -> Snowy
      ['move 2', 'move 1'], // rain -> Rainy (direct forme->forme)
      ['move 3', 'move 1'],
    ], { onBoundary: readState });
    r.per.forEach((d, i) => showTurn(`t${i + 1}`, d));
  }

  // ---------------------------------------------------------------- Q8
  console.log('\n############ S6: DRAW-FREENESS re-verify (Forecast vs control, same seed)');
  {
    for (const ab of ['Forecast', 'Levitate']) {
      const teams = [
        [mon('Castform', ['raindance', 'splash'], { ability: ab })],
        [mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
      ];
      const r = await runSided(teams, SEED, [
        ['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'],
        ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'],
      ], {});
      console.log(`  ${ab}: perTurnDraws=${JSON.stringify(r.per.map((d) => d.draws))}`);
    }
  }

  // ---------------------------------------------------------------- dex facts
  console.log('\n############ S7: dex facts (gen3-resolved)');
  {
    const g3 = Dex.mod('gen3');
    for (const id of ['castform', 'castformsunny', 'castformrainy', 'castformsnowy']) {
      const s = g3.species.get(id);
      console.log(`  ${id}: name=${JSON.stringify(s.name)} types=${JSON.stringify(s.types)} baseSpecies=${JSON.stringify(s.baseSpecies)} battleOnly=${JSON.stringify(s.battleOnly)} num=${s.num} bs=${JSON.stringify(s.baseStats)} gender=${JSON.stringify(s.gender)} weighthg=${s.weighthg} exists=${s.exists} isNonstandard=${JSON.stringify(s.isNonstandard)}`);
    }
    const ab = g3.abilities.get('forecast');
    console.log(`  ability forecast: num=${ab.num} handlers=${JSON.stringify(Object.keys(ab).filter((k) => /^on/.test(k)))} onSwitchInPriority=${ab.onSwitchInPriority} onAnyWeatherChange=${ab.onAnyWeatherChange}`);
    console.log(`  forecast.onWeatherChange source:\n${String(ab.onWeatherChange)}`);
    console.log(`  forecast.onStart source:\n${String(ab.onStart)}`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
