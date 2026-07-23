// probe_screen_residual_regression_rng.js — GROUND-TRUTH seeds for the BOTH-SIDES-SAME-SCREEN
// residual tie-shuffle pin (`gen3_screen_residual_tie_shuffle_v1`). Drives the REAL Showdown sim
// (gen3customgame) at a FIXED raw seed with genderless mons (draw-free construction, so the port's
// `opts_cg` at the same raw seed aligns), INJECTS Light Screen on one/both sides, runs a
// draw-minimal splash/splash turn, and prints the post-decision PRNG seed.
//
// The both-sides case draws ONE extra `random(0,2)` at the end-of-turn residual (fieldEvent's
// speed-sort ties the two Light Screen `onSideResidual` duration handlers — order 2, speed 0),
// so its seedAfter DIFFERS from the one-side control by that draw. Copy the printed `seedAfter`s
// verbatim into the pin. Run from src/rust_sim:  node harness/probe_screen_residual_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

function tick() { return new Promise((r) => setTimeout(r, 0)); }
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, o = {}) {
  return { species, item: '', ability: o.ability || 'No Ability', moves, evs: { ...EV0 }, ivs: IV31, nature: 'Serious', level: 100, gender: 'N' };
}

async function run(label, injectSides, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const _ of streams.omniscient) { /* drain */ } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([mon('Blissey', ['splash'], { ability: 'Natural Cure' })]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([mon('Snorlax', ['splash'], { ability: 'Immunity' })]) })}`);
  for (let i = 0; i < 10; i++) await tick();
  const b = stream.battle;
  // Inject Light Screen (duration 5, gen3 has no Light Clay) on the requested sides — AFTER
  // switch-in, BEFORE the first decision (draw-free, mirrors the pin's state injection).
  for (const s of injectSides) b.sides[s].addSideCondition('lightscreen', b.sides[s].active[0]);
  console.log(`\n=== ${label} ===  initSeed=${b.prng.getSeed()}  sides=[${injectSides.join(',')}]`);
  // One draw-minimal turn: both Splash (draw-free move phase → the residual + Quick Claw are the
  // only draws; distinct speeds Blissey 55 vs Snorlax 30 → no action-order/eachEvent tie).
  streams.omniscient.write('>p1 move 1');
  streams.omniscient.write('>p2 move 1');
  for (let k = 0; k < 18; k++) await tick();
  const scr = (s) => Object.keys(b.sides[s].sideConditions).map((k) => `${k}:${b.sides[s].sideConditions[k].duration}`).join(',');
  console.log(`  seedAfter=${b.prng.getSeed()}`);
  console.log(`  p1 scr[${scr(0)}]  p2 scr[${scr(1)}]`);
  try { streams.omniscient.destroy(); } catch (e) {}
  return b.prng.getSeed();
}

async function main() {
  const seed = [11, 22, 33, 44];
  console.log('Raw >start seed:', JSON.stringify(seed), '(gen3customgame, genderless → draw-free construction)');
  const both = await run('BOTH sides Light Screen (the residual tie-shuffle DRAWS)', [0, 1], seed);
  const one = await run('ONE side Light Screen (control — NO residual tie, NO shuffle)', [0], seed);
  console.log('\n=== COPY INTO THE PIN ===');
  console.log('  both-sides seedAfter =', both);
  console.log('  one-side  seedAfter =', one);
  console.log('  differ =', both !== one, '(MUST be true — the residual shuffle draw)');
}
main().catch((e) => { console.error(e); process.exit(1); });
