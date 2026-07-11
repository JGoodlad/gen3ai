// probe_forecast_rng.js — settle FORECAST (Castform forme+type follows effective
// weather) vs the resolved gen3 sim. Hypotheses (resolved dist): onWeatherChange —
// sun→Castform-Sunny(Fire), rain→Castform-Rainy(Water), hail→Castform-Snowy(Ice),
// sand/none→Castform(Normal); formeChange draw-free; onStart (switch-in, priority -2)
// re-fires it; weather END reverts; switch-out reverts (forme persistence?).
// Also: what species id does the sim REPORT at a decision boundary (the e2e golden's
// species field) — castform vs castformrainy?
// Run: node src/rust_sim/harness/probe_forecast_rng.js

'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

const SEED = [5, 4, 3, 2];

async function main() {
  console.log('=== rain cycle: RainDance -> forme; weather end -> revert; chart read');
  const teams = [
    [mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' }), mon('Rattata', ['scratch'], { ability: 'Guts' })],
    [mon('Charizard', ['flamethrower', 'splash'], { ability: 'Blaze' })],
  ];
  const r = await run(teams, SEED, [
    ['move 1', 'move 2'],  // t1 RainDance -> Castform-Rainy?
    ['move 2', 'move 1'],  // t2 Flamethrower into Water-Castform: resisted?
    ['move 2', 'move 2'],  // t3
    ['move 2', 'move 2'],  // t4
    ['move 2', 'move 2'],  // t5 rain ends (5 turns) -> revert?
    ['move 2', 'move 1'],  // t6 Flamethrower into Normal-Castform: neutral?
  ], { onBoundary: (b) => ({ species: b.p1.active[0].species.id, types: b.p1.active[0].types, weather: b.field.weather }) });
  r.perDecision.forEach((d, i) => {
    const ev = d.lines.filter((l) => l.includes('formechange') || l.includes('-weather') || l.includes('resisted') || l.includes('supereffective'));
    console.log(`t${i + 1}: [${fmtCalls(d.calls)}] ev=${JSON.stringify(ev)} ${JSON.stringify(r.states[i])}`);
  });

  console.log('=== draw-freeness: Castform vs control under a Drizzle lead (same seeds)');
  const mk = (ab) => [
    [mon('Castform', ['splash'], { ability: ab })],
    [mon('Politoed', ['splash'], { ability: 'Drizzle' })],
  ];
  for (const ab of ['Forecast', 'Levitate']) {
    const r2 = await run(mk(ab), SEED, Array(3).fill(['move 1', 'move 1']), {
      onBoundary: (b) => ({ species: b.p1.active[0].species.id, types: b.p1.active[0].types }),
    });
    console.log(`${ab}: perTurn=${JSON.stringify(r2.perDecision.map((d) => d.nexts))} states=${JSON.stringify(r2.states)}`);
  }

  console.log('=== switch out/in under standing rain: forme at re-entry; forme while benched');
  const teams3 = [
    [mon('Castform', ['splash'], { ability: 'Forecast' }), mon('Rattata', ['scratch'], { ability: 'Guts' })],
    [mon('Politoed', ['splash'], { ability: 'Drizzle' })],
  ];
  const r3 = await run(teams3, SEED, [
    ['move 1', 'move 1'],   // t1: rainy forme (drizzle from lead switch-in)
    ['switch 2', 'move 1'], // t2: Castform out — benched forme?
    ['switch 2', 'move 1'], // t3: back in — re-formes at switch-in?
  ], { onBoundary: (b) => ({ active: b.p1.active[0].species.id, benchCastform: b.p1.pokemon.find((p) => p.baseSpecies.baseSpecies === 'Castform').species.id }) });
  r3.perDecision.forEach((d, i) => {
    const ev = d.lines.filter((l) => l.includes('formechange'));
    console.log(`t${i + 1}: ev=${JSON.stringify(ev)} ${JSON.stringify(r3.states[i])}`);
  });

  console.log('=== sand + hail: sand -> stays Normal; hail -> Snowy');
  const teams4 = [
    [mon('Castform', ['sandstorm', 'hail', 'splash'], { ability: 'Forecast' })],
    [mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
  ];
  const r4 = await run(teams4, SEED, [['move 1', 'move 3'], ['move 2', 'move 1'], ['move 3', 'move 1']], {
    onBoundary: (b) => ({ species: b.p1.active[0].species.id, types: b.p1.active[0].types, weather: b.field.weather, hp: b.p1.active[0].hp }),
  });
  r4.perDecision.forEach((d, i) => {
    const ev = d.lines.filter((l) => l.includes('formechange') || l.includes('-weather'));
    console.log(`t${i + 1}: ev=${JSON.stringify(ev)} ${JSON.stringify(r4.states[i])}`);
  });
}

main().catch((e) => { console.error(e); process.exit(1); });
