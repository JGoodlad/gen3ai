// probe_confusion_screen_shuffle_rng.js — the REAL-Showdown ground truth for the
// CONFUSION self-hit × ModifyDamagePhase1 screen shuffle
// (`gen3_confusion_self_hit_screen_shuffle_v1`, the random-mode byte-fuzz find ab_3_17
// @ master-seed 100125).
//
// gen-4 confusion (gen-3-inherited) runs the FULL `getDamage(self,self,40)` →
// `modifyDamage` → `runEvent('ModifyDamagePhase1')`, which GATHERS the screens'
// `onAnyModifyDamagePhase1` SIDE handlers exactly like a normal hit — once per side across
// BOTH sides. So BOTH sides carrying Reflect → 2 tied handlers → a size-2 Fisher-Yates
// speed-sort shuffle draws ONE `random(0,2)` DURING the self-hit (verified: the draw sits
// AFTER the confusion `randomChance(1,2)`, BEFORE the `random(16)` randomizer). The screen
// handlers' `target !== source` guard makes them a DAMAGE no-op for a self-hit (no
// reduction), but they still gather → the shuffle draws.
//
// Prints the post-turn seed for BOTH boards so the Rust pin asserts (a) both-Reflect draws
// the shuffle (== this seed) and (b) it DIFFERS from a no-screen control (the extra draw).
//
// Run from src/rust_sim:  node harness/probe_confusion_screen_shuffle_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { Teams } = require(path.join(PS, 'dist/sim/teams.js'));

const FORMAT = 'gen3customgame';
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, seed, p1team, p2team, inject, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  inject(battle);
  const before = battle.prng.getSeed();
  if (plan.p1) streams.omniscient.write(`>p1 ${plan.p1}`);
  if (plan.p2) streams.omniscient.write(`>p2 ${plan.p2}`);
  for (let k = 0; k < 18; k++) await tick();
  const after = battle.prng.getSeed();
  const a0 = battle.sides[0].active[0];
  console.log(`\n=== ${label} ===  seed=${seed.join(',')}`);
  console.log(`  seedBefore=${before}`);
  console.log(`  seedAfter =${after}`);
  console.log(`  p1=${a0.species.name} ${a0.hp}/${a0.maxhp} selfHit=${a0.hp < a0.maxhp ? 'Y' : 'n'} ` +
    `p1Reflect=${!!battle.sides[0].sideConditions['reflect']} ` +
    `p2LightScreen=${!!battle.sides[1].sideConditions['lightscreen']}`);
  try { streams.omniscient.destroy(); } catch (e) { void e; }
}

async function main() {
  const p1 = [mon('Snorlax', ['splash'], { ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })];
  const p2 = [mon('Blissey', ['splash'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252, def: 252 } })];

  // Seed [2,2,2,2]: the SUB3 precedent proves this raw seed yields a confusion self-hit
  // (randomChance(1,2) -> false). We inject confusion(time=4) on the Snorlax + Reflect on BOTH
  // sides, then Splash/Splash. The self-hit fires getDamage -> the ModifyDamagePhase1 shuffle
  // (len=2, both Reflects) -> random(16).
  const SEED = [2, 2, 2, 2];
  // Reflect on p1's side + Light Screen on p2's side: BOTH gather (`onAny`) at the self-hit's
  // ModifyDamagePhase1 -> 2 tied handlers -> the shuffle DRAWS. But at the residual their duration
  // handlers sit at DIFFERENT orders (reflect onSideResidualOrder 1, lightscreen 2) -> NO tie ->
  // NO residual shuffle, so the seed isolates ONLY the confusion self-hit shuffle. (The real
  // ab_3_17 board was both-Reflect, which ALSO draws a residual both-screen shuffle -- a
  // correctly-modeled confound we avoid here.)
  await run('C-2SCR: confused Snorlax self-hits with 2 screens (reflect+lightscreen) (shuffle DRAWS)',
    SEED, p1, p2,
    (battle) => {
      const lax = battle.sides[0].active[0];
      lax.addVolatile('confusion');
      lax.volatiles['confusion'].time = 4; // long confusion; counter pinned in the Rust test
      battle.sides[0].addSideCondition('reflect', battle.sides[0].active[0]); // draw-free onStart
      battle.sides[1].addSideCondition('lightscreen', battle.sides[1].active[0]);
    },
    { p1: 'move 1', p2: 'move 1' });

  // CONTROL: identical board but NO screens -> the self-hit gathers 0 screen handlers -> NO
  // shuffle -> ONE fewer draw -> a DIFFERENT post-turn seed (the extra-draw proof / MC17 style).
  await run('C-NONE: confused Snorlax self-hits with NO screens (shuffle SKIPPED)',
    SEED, p1, p2,
    (battle) => {
      const lax = battle.sides[0].active[0];
      lax.addVolatile('confusion');
      lax.volatiles['confusion'].time = 4;
    },
    { p1: 'move 1', p2: 'move 1' });
}
main().catch((e) => { console.error(e); process.exit(1); });
