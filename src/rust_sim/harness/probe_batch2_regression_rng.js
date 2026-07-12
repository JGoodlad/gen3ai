// probe_batch2_regression_rng.js — GROUND-TRUTH seeds for the MOVE-COVERAGE BATCH 2
// regression pins (MC9…) in tests/regression_test.rs, captured from the OMNISCIENT
// in-process BattleStream (no server). Each scenario is a CONSTRUCTED gen3customgame board
// at a fixed raw seed; we print the per-decision post-turn PRNG seed + the key STATE the
// pin asserts. Copy the printed `seedAfter` verbatim into the pins.
//
//   MC9  — Refresh self-cures paralysis (DRAW-FREE).
//   MC10 — Heal Bell cures the whole team + SKIPS a Soundproof ally (DRAW-FREE).
//   MC11 — Aromatherapy cures the whole team via clearStatus (DRAW-FREE).
//   MC12 — Rain Dance sets a 5-turn Rain (distinct speed → DRAW-FREE) — the SET turn seed.
//   MC13 — Rain Dance into an ALREADY-active Rain FAILS (DRAW-FREE, weather unchanged).
//   MC14 — Screech −2 Def (accuracy roll drawn) vs a plain foe.
//   MC15 — Screech BLOCKED by Clear Body (accuracy still drawn, no drop).
//   MC16 — Light Screen (5-turn side condition, DRAW-FREE).
//   MC17 — the DOUBLE-SCREEN ModifyDamagePhase1 SHUFFLE: a physical hit into a side with
//          BOTH Reflect + Light Screen up draws ONE extra `random(0,2)` (the crux). Compare
//          a ONE-screen control (no shuffle) — the seeds must DIFFER by that one draw.
//
// Run:  node src/rust_sim/harness/probe_batch2_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1team, p2team, plans, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  const seed = opts.seed || [11, 22, 33, 44];
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 10; i++) await tick();
  const b = stream.battle;
  for (const inj of (opts.inject || [])) {
    if (inj.weather) { b.field.setWeather(inj.weather, b.sides[0].active[0]); b.field.weatherState.duration = 0; }
    if (inj.side !== undefined) {
      const s = b.sides[inj.side];
      if (inj.status) s.active[0].setStatus(inj.status, s.active[0], null, true);
      if (inj.benchStatus && s.pokemon[1]) s.pokemon[1].setStatus(inj.benchStatus, s.pokemon[1], null, true);
      for (const sc of (inj.side_conditions || [])) s.addSideCondition(sc, s.active[0]);
    }
  }
  console.log(`\n=== ${label} ===  initSeed=${b.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!b.ended && safety < 30) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const plan = plans[Math.min(i, plans.length - 1)]; i++;
    if (plan.p1) streams.omniscient.write(`>p1 ${plan.p1}`);
    if (plan.p2) streams.omniscient.write(`>p2 ${plan.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const a0 = b.sides[0].active[0], a1 = b.sides[1].active[0];
    const scr = (s) => Object.keys(b.sides[s].sideConditions).map((k) => `${k}:${b.sides[s].sideConditions[k].duration}`).join(',');
    console.log(`  [${rs}] ${JSON.stringify(plan)} seedAfter=${b.prng.getSeed()}`);
    console.log(`     p1=${a0.species.name} ${a0.hp}/${a0.maxhp} ${a0.status || '-'} def${a0.boosts.def} scr[${scr(0)}]  benchStat=${b.sides[0].pokemon[1] ? (b.sides[0].pokemon[1].status || '-') : 'n/a'}`);
    console.log(`     p2=${a1.species.name} ${a1.hp}/${a1.maxhp} ${a1.status || '-'} def${a1.boosts.def} atk${a1.boosts.atk} scr[${scr(1)}]  weather=${b.field.weather || '-'}(${b.field.weatherState ? b.field.weatherState.duration : 0})`);
    if (plan.stop) break;
  }
  console.log(`  ended=${b.ended} winner=${b.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, o = {}) {
  return { species, item: o.item || '', ability: o.ability || 'No Ability', moves, evs: { ...EV0, ...(o.evs || {}) }, ivs: IV31, nature: o.nature || 'Serious', level: o.level || 100, gender: 'N' };
}

async function main() {
  // MC9: Refresh cures self par (draw-free). Injected par on the caster.
  await run('MC9 Refresh cures self paralysis (draw-free)',
    [mon('Vaporeon', ['refresh', 'surf'], { ability: 'Water Absorb', evs: { spa: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { inject: [{ side: 0, status: 'par' }] });

  // MC10: Heal Bell cures active tox + bench par, SKIPS the Soundproof bench ally.
  await run('MC10 Heal Bell cures team + skips Soundproof ally (draw-free)',
    [mon('Miltank', ['healbell', 'bodyslam'], { ability: 'Thick Fat', evs: { hp: 252 } }),
     mon('Electrode', ['thunderbolt'], { ability: 'Soundproof', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { inject: [{ side: 0, status: 'tox', benchStatus: 'par' }] });

  // MC11: Aromatherapy cures the team (clearStatus banner). Active brn + bench slp.
  await run('MC11 Aromatherapy cures team (draw-free)',
    [mon('Vileplume', ['aromatherapy', 'gigadrain'], { ability: 'Chlorophyll', evs: { hp: 252 } }),
     mon('Snorlax', ['bodyslam'], { ability: 'Thick Fat', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { inject: [{ side: 0, status: 'brn', benchStatus: 'slp' }] });

  // MC12: Rain Dance sets 5-turn Rain (distinct speed → draw-free). Electrode fast, Snorlax slow.
  await run('MC12 Rain Dance sets 5-turn Rain (distinct speed, draw-free)',
    [mon('Electrode', ['raindance', 'thunderbolt'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }]);

  // MC13: Rain Dance into ALREADY-active Rain FAILS (draw-free), weather stays permanent.
  await run('MC13 Rain Dance into active Rain FAILS (draw-free)',
    [mon('Electrode', ['raindance', 'thunderbolt'], { evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { inject: [{ weather: 'raindance' }] });

  // MC14: Screech −2 Def, accuracy 85 (drawn). Persian into a plain Snorlax.
  await run('MC14 Screech foe -2 Def (accuracy drawn)',
    [mon('Persian', ['screech', 'slash'], { ability: 'Limber', evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // MC15: Screech BLOCKED by Clear Body (accuracy still drawn, no drop).
  await run('MC15 Screech vs Clear Body (blocked, accuracy drawn)',
    [mon('Persian', ['screech', 'slash'], { ability: 'Limber', evs: { spe: 252 } })],
    [mon('Metagross', ['meteormash'], { ability: 'Clear Body', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // MC16: Light Screen alone (5-turn side condition, draw-free).
  await run('MC16 Light Screen sets a 5-turn side condition (draw-free)',
    [mon('Blissey', ['lightscreen', 'softboiled'], { ability: 'Natural Cure', evs: { hp: 252, def: 252 } })],
    [mon('Snorlax', ['pound'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }]);

  // MC17: the DOUBLE-SCREEN ModifyDamagePhase1 SHUFFLE. A physical Pound into a side with
  //  BOTH Reflect + Light Screen up draws ONE extra random(0,2). CONTROL: one screen only.
  await run('MC17 DOUBLE screen: physical hit draws the ModifyDamagePhase1 shuffle',
    [mon('Blissey', ['softboiled'], { ability: 'Natural Cure', evs: { hp: 252, def: 252 } })],
    [mon('Snorlax', ['pound'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { inject: [{ side: 0, side_conditions: ['reflect', 'lightscreen'] }] });
  await run('MC17-control ONE screen: physical hit does NOT draw the shuffle',
    [mon('Blissey', ['softboiled'], { ability: 'Natural Cure', evs: { hp: 252, def: 252 } })],
    [mon('Snorlax', ['pound'], { ability: 'Immunity', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }],
    { inject: [{ side: 0, side_conditions: ['reflect'] }] });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
