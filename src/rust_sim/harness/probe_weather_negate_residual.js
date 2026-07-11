// probe_weather_negate_residual.js — settle the WEATHER_NEGATE (Cloud Nine / Air Lock)
// draw model at the RESIDUAL: when a negater is up under Sand, does the sandstorm chip
// residual handler still get GATHERED (participating in the residual handler-sort
// tie-shuffle) but no-op, or is it not scheduled at all? And does the chip HP apply?
//
// This is the draw-COUNT crux for the port's `run_residuals` weather-chip gate. We compare
// (A) Sand + a negater on the field vs (B) Sand + a no-op, counting raw draws + reading the
// chip HP, over several seeds. If A's draw count == a NO-WEATHER control (not == B), the
// negater removes the weather residual handler entirely (draw-count change); if A == B, the
// handler stays but no-ops.
//
// Run: node src/rust_sim/harness/probe_weather_negate_residual.js

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
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: 'Serious', level: 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// p1 leads a Sand Stream Tyranitar (sets sand); p2 the probe mon (negater vs no-op vs a
// no-weather setter). Both non-Rock/Ground/Steel mons take the chip normally.
async function run(scenario, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  let p1lead, p2lead;
  if (scenario === 'negater') {
    p1lead = mon('Tyranitar', ['recover', 'recover'], { ability: 'Sand Stream' });
    p2lead = mon('Golduck', ['recover', 'recover'], { ability: 'Cloud Nine' });
  } else if (scenario === 'noop') {
    p1lead = mon('Tyranitar', ['recover', 'recover'], { ability: 'Sand Stream' });
    p2lead = mon('Golduck', ['recover', 'recover'], { ability: 'Shell Armor' });
  } else {
    // no weather at all (both no-op abilities).
    p1lead = mon('Snorlax', ['recover', 'recover'], { ability: 'Shell Armor' });
    p2lead = mon('Golduck', ['recover', 'recover'], { ability: 'Shell Armor' });
  }
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([p1lead]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([p2lead]) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  let n = 0; rng.next = (...a) => { n += 1; return realNext(...a); };
  const per = [];
  const p2 = battle.sides[1].active[0];
  const hpBefore = [];
  for (let t = 0; t < 3; t++) {
    const b = n;
    hpBefore.push(p2.hp);
    streams.omniscient.write('>p1 move 1');
    streams.omniscient.write('>p2 move 1');
    for (let k = 0; k < 10; k++) await tick();
    per.push(n - b);
  }
  return { totalDraws: n, per, effWeather: battle.field.effectiveWeather(), rawWeather: battle.field.weather, p2hp: p2.hp, p2maxhp: p2.maxhp };
}

(async () => {
  const seeds = [[1, 2, 3, 4], [7, 11, 13, 17], [5, 5, 5, 5], [42, 42, 42, 42]];
  console.log('=== WEATHER_NEGATE residual draw model (Sand + negater vs no-op vs no-weather) ===');
  for (const seed of seeds) {
    const neg = await run('negater', seed);
    const noop = await run('noop', seed);
    const none = await run('noweather', seed);
    console.log(`  seed ${JSON.stringify(seed)}:`);
    console.log(`    Sand+CloudNine: draws=${neg.totalDraws} per=${JSON.stringify(neg.per)} effW=${neg.effWeather} rawW=${neg.rawWeather} p2hp=${neg.p2hp}/${neg.p2maxhp}`);
    console.log(`    Sand+no-op    : draws=${noop.totalDraws} per=${JSON.stringify(noop.per)} effW=${noop.effWeather} rawW=${noop.rawWeather} p2hp=${noop.p2hp}/${noop.p2maxhp}`);
    console.log(`    no-weather    : draws=${none.totalDraws} per=${JSON.stringify(none.per)} effW=${none.effWeather} p2hp=${none.p2hp}/${none.p2maxhp}`);
  }
  console.log('');
  console.log('Interpretation:');
  console.log('  - p2hp under CloudNine should stay FULL (no chip) — the STATE effect.');
  console.log('  - If CloudNine draws == no-op (both include the sand residual handler), the weather handler is GATHERED but no-ops.');
  console.log('  - If CloudNine draws == no-weather, the negater REMOVES the weather residual handler (draw-count change).');
})();
