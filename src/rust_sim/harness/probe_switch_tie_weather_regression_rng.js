// probe_switch_tie_weather_regression_rng.js — GROUND TRUTH for the
// `switch_into_a_tie_under_sand_draws_the_weather_change_shuffle_seed` regression pin.
//
// THE BUG (e2e_84 dec4): a MID-TURN switch-in whose entrant TIES the opposing active on
// cached speed, AND sets/changes WEATHER, draws ONE `eachEvent('WeatherChange')` speed-tie
// shuffle (`Field.setWeather` → field.ts:87) that the port MISSED. This probe drives the
// OMNISCIENT in-process BattleStream (no server) over a CONSTRUCTED gen3customgame scenario
// whose EXACT packed teams + seed the Rust regression test replays, and prints the
// post-decision PRNG `seedAfter` for each boundary — copied verbatim into the test as the
// real-Showdown ground truth. The pin FAILS if the WeatherChange shuffle fix is reverted.
//
// The scenario (211/221-class 213-vs-213 mirror, exact speed tie):
//   p1: [Suicune lead (spe 221), Tyranitar (Sand Stream, spe 221)]
//   p2: [Suicune (spe 221)]
//   turn 1 (move): p1 SWITCHES Tyranitar in (slot 2) while p2 Suicune Splashes. The switch
//     (order 103) runs first → Tyranitar sets sandstorm on the runSwitch ability Start →
//     the actives TIE (221 == 221) → `eachEvent('WeatherChange')` draws ONE shuffle (the
//     missing draw), then p2's Splash + the residual sand chip + the nested weather shuffle.
//
// Run:  node src/rust_sim/harness/probe_switch_tie_weather_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

const FORMAT = 'gen3customgame';
const SEED = [52903, 53571, 56373, 31187]; // the e2e_84-class init seed (a tie exerciser)

// The EXACT packed teams the Rust regression test uses (the `||...||` Showdown pack form).
// Suicune 60-spe-EV serious = 221 spe; Tyranitar 252-spe-EV serious = 221 spe → exact TIE.
const P1 =
  'Suicune|||pressure|surf,splash|Serious|252,,,,,60|||||' +
  ']Tyranitar|||sandstream|crunch,rockslide|Serious|252,,,,,252|||||';
const P2 = 'Suicune|||pressure|surf,splash|Serious|252,,,,,60|||||';

// turn 1: p1 SWITCHES Tyranitar in (slot 2); p2 Suicune Splashes (move 2).
const PLAN = [{ p1: 'switch 2', p2: 'move 2' }];

function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function main() {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  // Pack via the importer so the bytes match what team::unpack ingests (we hand-pass the
  // Showdown-canonical packed string directly).
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(SEED)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: P1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: P2 })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;

  // RESEED to the RAW seed right before the decision. The Rust `start_with_switchins` places
  // the leads DRAW-FREE and leaves `prng = new Prng(RAW_SEED)` — so for a bit-for-bit-comparable
  // ground truth we reset the sim's prng to the SAME raw seed here (the `>start` switch-in
  // setup draws the sim makes — the lead Suicune-tie `eachEvent('Update')` shuffles + the turn-1
  // Quick Claw — are NOT modeled by the bounded Rust `start_with_switchins`, exactly as
  // documented; the e2e seeds at the post-`>start` state to absorb them, but a CONSTRUCTED
  // regression pin reseeds to raw so the DECISION draws line up with the Rust's). After this,
  // `seedBefore == RAW_SEED` and `seedAfter` is the value the Rust must reproduce.
  battle.prng = new PRNG(SEED.slice());

  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`=== switch-into-tie-under-sand regression ground truth ===`);
  console.log(`rawSeed (== Rust start_with_switchins prng) = ${battle.prng.getSeed()}`);
  const a0 = () => battle.sides[0].active[0], a1 = () => battle.sides[1].active[0];
  console.log(`speeds: p1=${a0().getStat('spe')} p2=${a1().getStat('spe')}`);

  let i = 0, safety = 0;
  while (!battle.ended && safety < 50) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= PLAN.length) break;
    const dc0 = drawCount;
    const entry = PLAN[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} spe=${m.getStat('spe')}` : '-';
    console.log(`  decision ${i - 1} [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0}`);
    console.log(`    seedAfter = ${after}   weather=${battle.field.weather || 'none'}`);
    console.log(`    p1=${fmt(a0())} | p2=${fmt(a1())}`);
  }
  console.log(`ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
