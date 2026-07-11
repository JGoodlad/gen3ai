// probe_weather_eachevent_tie_regression_rng.js — GROUND TRUTH for the
// `sun_rain_weather_turn_tie_draws_the_eachevent_weather_shuffle_seed` regression pin
// (`gen3_ability_batch1_v1`, the STEP-1 weather-eachEvent fix).
//
// THE BUG: gen3 sun (`sunnyday`) + rain (`raindance`) fire `this.eachEvent('Weather')` at EVERY
// end-of-turn UNCONDITIONALLY (the resolved `onFieldResidual` body is a bare `this.add('-weather',
// …,'[upkeep]'); this.eachEvent('Weather');` — NO `isWeather` guard, unlike sand/hail). That
// `eachEvent('Weather')` speed-sorts the actives → on a speed TIE it draws ONE `random(0,2)`
// Fisher-Yates shuffle. The port used to gate the END-OF-TURN weather tie-shuffle on `Sand | Hail`
// ONLY → a WEATHER-TURN speed TIE under sun/rain MISSED that draw (a 1-draw desync on every later
// turn). The fix schedules the field weather-residual (which fires the shuffle) off the RAW
// `field.weather` for sun/rain (so it fires even under a Cloud Nine / Air Lock negater — verified),
// and off `effectiveWeather()` for sand/hail (a negater suppresses those).
//
// This probe drives the OMNISCIENT in-process BattleStream (no server) over a CONSTRUCTED
// gen3customgame Kyogre-vs-Kyogre MIRROR (both Drizzle → RAIN on switch-in, exact speed TIE), and
// prints the post-decision PRNG `seedAfter` — copied verbatim into the Rust regression test as the
// real-Showdown ground truth. The pin FAILS if the sun/rain eachEvent('Weather') shuffle fix is
// reverted (the port would draw one fewer call on the rain-turn tie → the seed diverges).
//
// The scenario (a 90/90 Kyogre mirror, exact speed tie, both Splash under self-set rain):
//   p1: Kyogre lead (Drizzle, spe 90-base, 0 spe EV) + a Splash bench
//   p2: Kyogre lead (Drizzle, spe 90-base, 0 spe EV)
//   turn 1 (move): BOTH Splash. The actives TIE (spe 90 == 90) so the action-order + per-action
//     eachEvent shuffles draw; then the END-OF-TURN `eachEvent('Weather')` under RAIN draws ONE MORE
//     `random(0,2)` (the FIX — rain has no chip, so this shuffle is the whole field-residual), then
//     the Quick Claw. A model that gates the end-of-turn weather shuffle on Sand|Hail draws one
//     FEWER → the seed diverges here.
//
// Run:  node src/rust_sim/harness/probe_weather_eachevent_tie_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

const FORMAT = 'gen3customgame';
const SEED = [41231, 8877, 60013, 25519]; // a tie exerciser (arbitrary fixed init)

// The EXACT packed teams the Rust regression test uses (the `||...||` Showdown pack form).
// Kyogre 0-spe-EV serious = spe 90*2+31+5 mapped through the formula (both sides identical → TIE).
const P1 =
  'Kyogre|||drizzle|splash,surf|Serious|252,,,,,|||||' +
  ']Snorlax|||immunity|splash,bodyslam|Serious|252,,,,,|||||';
const P2 = 'Kyogre|||drizzle|splash,surf|Serious|252,,,,,|||||';

// turn 1: BOTH Splash (move 1).
const PLAN = [{ p1: 'move 1', p2: 'move 1' }];

function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function main() {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(SEED)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: P1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: P2 })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;

  // RESEED to the RAW seed right before the decision. The Rust `start_with_switchins` places the
  // leads DRAW-FREE (the Drizzle rain-set is a draw-free ability Start) and leaves
  // `prng = new Prng(RAW_SEED)` — so for a bit-for-bit-comparable ground truth we reset the sim's
  // prng to the SAME raw seed here (the `>start` switch-in setup draws the sim makes — the lead
  // Kyogre-tie `eachEvent('Update')` shuffles + the turn-1 Quick Claw + the gender samples — are
  // NOT modeled by the bounded Rust `start_with_switchins`, exactly as documented; the e2e seeds at
  // the post-`>start` state to absorb them, but a CONSTRUCTED regression pin reseeds to raw so the
  // DECISION draws line up). After this, `seedBefore == RAW_SEED` and `seedAfter` is the value the
  // Rust must reproduce.
  battle.prng = new PRNG(SEED.slice());

  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`=== sun/rain weather-turn-tie eachEvent('Weather') regression ground truth ===`);
  console.log(`rawSeed (== Rust start_with_switchins prng) = ${battle.prng.getSeed()}`);
  const a0 = () => battle.sides[0].active[0], a1 = () => battle.sides[1].active[0];
  console.log(`weather=${battle.field.weather || 'none'}  speeds: p1=${a0().getStat('spe')} p2=${a1().getStat('spe')}  tie=${a0().getStat('spe') === a1().getStat('spe')}`);

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
