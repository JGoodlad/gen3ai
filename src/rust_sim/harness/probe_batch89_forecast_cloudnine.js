// probe_batch89_forecast_cloudnine.js — the UNPROBED forecast piece: the Cloud Nine /
// Air Lock effective-weather composition, plus the exact forme-change timing/reporting.
// Run: node src/rust_sim/harness/probe_batch89_forecast_cloudnine.js
'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');
const SEED = [5, 4, 3, 2];

function showDec(tag, r) {
  r.perDecision.forEach((d, i) => {
    const ev = d.lines.filter((l) => l.includes('formechange') || l.includes('-weather') || l.includes('|switch|') || l.includes('|drag|'));
    console.log(`  ${tag} t${i + 1}: draws=${d.nexts} forme/weather=${JSON.stringify(ev)} state=${JSON.stringify(r.states[i])}`);
  });
}
const st = (b) => ({ species: b.sides[0].active[0].species.id, types: b.sides[0].active[0].types, effWeather: b.sides[0].active[0].effectiveWeather(), rawWeather: b.field.weather });

async function main() {
  // 1. Castform under rain, then a CLOUD NINE mon comes in on the OTHER side.
  //    Does Castform revert to Normal (effectiveWeather == '') while Cloud Nine is active?
  console.log('############ Forecast under Cloud Nine (opp) ############');
  {
    const teams = [
      [mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' })],
      [mon('Psyduck', ['splash'], { ability: 'Cloud Nine' }), mon('Rattata', ['splash'], { ability: 'Guts' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 2'],  // t1: p1 RainDance -> but Cloud Nine active -> Castform stays Normal?
      ['move 2', 'switch 2'],// t2: opp switches Psyduck(CloudNine) OUT to Rattata -> Castform re-formes to Rainy?
      ['move 2', 'switch 2'],// t3: opp switches Rattata OUT, Psyduck(CloudNine) back IN -> Castform reverts to Normal?
    ], { onBoundary: st });
    showDec('CN-opp', r);
  }

  // 2. Cloud Nine on p1's OWN partner? (singles -> the Castform itself; use Air Lock Rayquaza as opp instead).
  console.log('\n############ Forecast under Air Lock (opp Rayquaza) ############');
  {
    const teams = [
      [mon('Castform', ['sunnyday', 'splash'], { ability: 'Forecast' })],
      [mon('Rayquaza', ['splash'], { ability: 'Air Lock' }), mon('Rattata', ['splash'], { ability: 'Guts' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 1'],  // t1: SunnyDay, Air Lock suppresses -> Castform stays Normal?
      ['move 2', 'switch 2'],// t2: Air Lock leaves -> Castform -> Sunny (Fire)?
    ], { onBoundary: st });
    showDec('AL-opp', r);
  }

  // 3. Forecast mon SWITCHING IN while weather already up + Cloud Nine already up: does onStart forme?
  console.log('\n############ Castform switches in under standing rain w/ Cloud Nine already active ############');
  {
    const teams = [
      [mon('Politoed', ['splash'], { ability: 'Drizzle' }), mon('Castform', ['splash'], { ability: 'Forecast' })],
      [mon('Psyduck', ['splash'], { ability: 'Cloud Nine' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 1'],  // t1: Drizzle rain set but Cloud Nine active -> no forme yet
      ['switch 2', 'move 1'],// t2: Castform switches in under Cloud-Nine-suppressed rain -> stays Normal?
    ], { onBoundary: st });
    showDec('CN-switchin', r);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
