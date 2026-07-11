// probe_flashfire_regression_rng.js — GROUND-TRUTH seeds + HP for the FLASH FIRE regression pins
// (`tests/regression_test.rs`, `gen3_flashfire_boost_v1`). Drives the OMNISCIENT in-process
// BattleStream (no server) over the EXACT constructed `gen3customgame` scenarios the Rust pins
// replay, and dumps the per-decision post-turn SEED + both actives' HP so a pin can assert them.
//
// The pins (each revert-verified against a specific behavior):
//   FF1 — THE ×1.5 BOOST. p1 Ninetales (Flash Fire) vs p2 Charizard (Fire Blast). T1: Ninetales
//     (slower — Modest, no Spe) is hit by Fire Blast → 0 damage + FF ARMS; its own Flamethrower
//     that turn is UNBOOSTED (armed AFTER it moved? no — Ninetales is slower, so Charizard fires
//     first → FF arms → Ninetales' T1 Flamethrower is ALREADY boosted). T2: definitely boosted.
//     We dump Charizard's HP after each turn so the pin asserts the BOOSTED damage (revert the
//     fold → Charizard keeps more HP).
//   FF2 — ACTIVATION GATE (a MISSED Fire move does NOT arm). p1 Ninetales vs p2 a low-accuracy
//     Fire move; a MISS seed leaves FF disarmed. We dump the flashfire volatile after a miss vs
//     a hit so the pin asserts arm-on-hit / no-arm-on-miss.
//   FF3 — SWITCH-CLEAR. p1 [Ninetales, Umbreon]; Ninetales arms, pivots OUT and back → FF cleared.
//     We dump the flashfire volatile after the pivot cycle.
//
// Run:  node src/rust_sim/harness/probe_flashfire_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, o = {}) {
  return {
    species, item: o.item || '', ability: o.ability || 'No Ability', moves,
    evs: { ...EV0, ...(o.evs || {}) }, ivs: o.ivs || IV31,
    nature: o.nature || 'Serious', level: o.level || 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1, p2, plan, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  const A = () => battle.sides[0].active[0];
  const B = () => battle.sides[1].active[0];
  const ff = (m) => !!(m && m.volatiles && m.volatiles.flashfire);
  for (const [i, step] of plan.entries()) {
    if (step.p1) { try { streams.omniscient.write(`>p1 ${step.p1}`); } catch (e) {} }
    if (step.p2) { try { streams.omniscient.write(`>p2 ${step.p2}`); } catch (e) {} }
    for (let k = 0; k < 12; k++) await tick();
    const a = A(), b = B();
    console.log(`  T${i + 1} ${JSON.stringify(step)}  seedAfter=${battle.prng.getSeed()}`);
    console.log(`     p1=${a ? a.species.name : '-'} hp=${a ? a.hp : '-'}/${a ? a.maxhp : '-'} ff=${ff(a)}` +
      `   p2=${b ? b.species.name : '-'} hp=${b ? b.hp : '-'}/${b ? b.maxhp : '-'} ff=${ff(b)}`);
  }
  return battle;
}

async function main() {
  // FF1 — the ×1.5 boost. Ninetales Modest (SLOWER than Charizard Modest? both no Spe EVs, base
  //   Spe Ninetales 100 vs Charizard 100 — a tie. To make Charizard fire FIRST (so FF arms before
  //   Ninetales moves on T1), give Charizard a Spe edge (Timid) so it out-speeds. Then T1 Ninetales
  //   Flamethrower is BOOSTED. We dump Charizard's HP.
  await run('FF1_boost',
    [mon('Ninetales', ['flamethrower', 'rest'], { ability: 'Flash Fire', nature: 'Modest', evs: { spa: 252 } })],
    [mon('Charizard', ['fireblast', 'rest'], { ability: 'Blaze', nature: 'Timid', evs: { hp: 252, spe: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2' }],
    [11, 22, 33, 44]);

  // FF1b — a NON-activated baseline: the SAME Ninetales but p2 never fires a Fire move (Snorlax),
  //   so its Flamethrower is UNBOOSTED. Dump Snorlax HP for the un-boosted reference (the pin can
  //   compare boosted vs unboosted damage if desired; primarily FF1 asserts the boosted number).
  await run('FF1b_unboosted_ref',
    [mon('Ninetales', ['flamethrower', 'rest'], { ability: 'Flash Fire', nature: 'Modest', evs: { spa: 252 } })],
    [mon('Snorlax', ['bodyslam', 'rest'], { ability: 'No Ability', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    [{ p1: 'move 1', p2: 'move 2' }, { p1: 'move 1', p2: 'move 2' }],
    [11, 22, 33, 44]);

  // FF2 — activation gate (miss does NOT arm). Ninetales vs a foe firing Fire Blast (85% acc).
  //   Sweep seeds; report a MISS seed (ff stays false) + a HIT seed (ff true) for the pin.
  {
    console.log('\n=== FF2_activation_gate: find a MISS seed (no arm) + a HIT seed (arm) ===');
    let missSeed = null, hitSeed = null;
    for (let s = 1; s <= 120 && !(missSeed && hitSeed); s++) {
      const seed = [s * 7 + 1, s * 13 + 2, s * 5 + 3, s * 3 + 4];
      const stream = new BattleStream();
      const streams = getPlayerStreams(stream);
      (async () => { for await (const ch of streams.omniscient) { void ch; } })();
      streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
      streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([mon('Ninetales', ['flamethrower', 'rest'], { ability: 'Flash Fire', nature: 'Modest', evs: { spa: 252 } })]) })}`);
      streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([mon('Charizard', ['fireblast'], { ability: 'Blaze', nature: 'Timid', evs: { hp: 252, spe: 252 } })]) })}`);
      for (let i = 0; i < 12; i++) await tick();
      const b = stream.battle;
      // The POST-switch-in seed — the state the port's `start_with_switchins` seeds AT (its
      // switch-in is draw-free, so a pin must use THIS, not the raw >start seed).
      const initSeed = b.prng.getSeed();
      streams.omniscient.write('>p1 move 2'); // Ninetales rest (idle, no attack)
      streams.omniscient.write('>p2 move 1'); // Charizard Fire Blast (may miss)
      for (let k = 0; k < 12; k++) await tick();
      const nt = b.sides[0].active[0];
      const armed = !!(nt && nt.volatiles && nt.volatiles.flashfire);
      const missed = /\|-miss\|/.test(b.log.join('\n'));
      if (missed && !missSeed) { missSeed = seed; console.log(`  MISS raw=${JSON.stringify(seed)} PORT-INIT=${initSeed}: flashfire=${armed} (expect false — miss does NOT arm), seedAfter=${b.prng.getSeed()}`); }
      if (!missed && !hitSeed) { hitSeed = seed; console.log(`  HIT  raw=${JSON.stringify(seed)} PORT-INIT=${initSeed}: flashfire=${armed} (expect true — a landed absorb arms), seedAfter=${b.prng.getSeed()}`); }
    }
    if (!missSeed) console.log('  (no MISS seed found — raise the pool)');
  }

  // FF3 — switch-clear. Ninetales arms (T1 Charizard Fire Blast), pivots to Umbreon (T2) + back
  //   (T3) → FF cleared. Dump the Ninetales flashfire after the cycle.
  await run('FF3_switch_clear',
    [mon('Ninetales', ['flamethrower', 'rest'], { ability: 'Flash Fire', nature: 'Modest', evs: { spa: 252 } }),
     mon('Umbreon', ['rest', 'protect'], { ability: 'Synchronize', nature: 'Calm', evs: { hp: 252, spd: 252 } })],
    [mon('Charizard', ['fireblast', 'rest'], { ability: 'Blaze', nature: 'Timid', evs: { hp: 252, spe: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },   // T1: Fire Blast arms Ninetales
     { p1: 'switch 2', p2: 'move 2' }, // T2: Ninetales OUT → Umbreon (Charizard roosts)
     { p1: 'switch 2', p2: 'move 2' }, // T3: Umbreon OUT → Ninetales back (FF now CLEARED)
     { p1: 'move 1', p2: 'move 2' }],  // T4: Ninetales Flamethrower — UNBOOSTED (re-armed only if hit again)
    [11, 22, 33, 44]);

  console.log('\n=== DONE ===');
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
