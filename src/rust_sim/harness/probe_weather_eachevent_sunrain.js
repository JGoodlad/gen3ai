// probe_weather_eachevent_sunrain.js — CONFIRM the STEP-1 bug: gen3 sun/rain fire
// `eachEvent('Weather')` at EVERY end-of-turn UNCONDITIONALLY (not just sand/hail).
//
// THE CLAIM (reviewer, to re-verify here against the RESOLVED Dex.mod('gen3')):
//   gen3 conditions.js does NOT redefine sunnyday/raindance, so the gen4 mod inherits the
//   base `onFieldResidual` body, which UNCONDITIONALLY calls `this.eachEvent('Weather')`.
//   `eachEvent('Weather')` speed-sorts the active mons → on a SPEED TIE it draws ONE
//   `random(0,2)` Fisher-Yates shuffle. So a WEATHER-TURN speed tie under SUN or RAIN
//   draws that end-of-turn shuffle EXACTLY as sand/hail does — even though sun/rain do NO
//   chip damage. The port currently gates the end-of-turn weather tie-shuffle on Sand|Hail
//   only → it MISSES this draw (a 1-draw desync on a sun/rain weather-turn tie).
//
// METHOD (the draw-count differential — the shuffle is the ONLY difference):
//   Build a same-species MIRROR at an EXACT speed tie (so eachEvent's speedSort DRAWS its
//   size-2 shuffle). Both mons Splash (a no-op status move: never-miss, no secondary, no
//   damage → the ONLY draws are the per-action + end-of-turn eachEvent tie-shuffles + the
//   turn-end Quick Claw). We run the SAME mirror:
//     (A) under RAIN  (a Damp Rock / Rain Dance not needed — set field weather directly),
//     (B) under SUN,
//     (C) with NO weather (control).
//   The end-of-turn `eachEvent('Weather')` shuffle fires in (A) and (B) but NOT (C) → the
//   per-turn draw count under rain/sun must be EXACTLY (control + 1). We ALSO dump, for a
//   weather turn, the raw draw trace so the extra draw is unambiguously the WEATHER one
//   (it fires at the residual field-event, AFTER the move-phase shuffles).
//
// We drive the OMNISCIENT in-process BattleStream (no server) and instrument prng.next.
//
// Run: node src/rust_sim/harness/probe_weather_eachevent_sunrain.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Dex, Teams } = require(path.join(PS, 'dist/sim'));

const d3 = Dex.mod('gen3');

// Re-CONFIRM the mod-chain claim by READING the resolved data: gen3's `sunnyday`/`raindance`
// conditions must NOT define their own onFieldResidual (so the gen4-inherited base body,
// which calls eachEvent('Weather'), is what runs). Also confirm sand/hail DO (the control).
function dumpWeatherCondition(id) {
  const c = d3.conditions.get(id);
  const own = Object.prototype.hasOwnProperty;
  const hasOwnFR = c && own.call(c, 'onFieldResidual');
  console.log(`  cond '${id}': exists=${!!c} ownOnFieldResidual=${!!hasOwnFR} ` +
    `onWeather=${c && !!c.onWeather}`);
  return { hasOwnFR: !!hasOwnFR };
}

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Run a 2-turn Splash mirror, optionally forcing `weather` on the field before each turn.
// Returns per-turn draw counts + a trace of the LAST turn.
async function run(weather, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => {
    for await (const ch of streams.omniscient) {
      for (const l of String(ch).split('\n')) lines.push(l);
    }
  })();
  // A same-species Snorlax mirror at IDENTICAL speed (0 EV, Serious) → exact tie.
  const team = Teams.pack([mon('Snorlax', ['splash'])]);
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team })}`);
  for (let i = 0; i < 14; i++) await tick();
  const battle = stream.battle;

  const a0 = () => battle.sides[0].active[0], a1 = () => battle.sides[1].active[0];
  const spe0 = a0().getActionSpeed(), spe1 = a1().getActionSpeed();

  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  let n = 0;
  let trace = null;
  rng.next = function (...a) { n += 1; if (trace) trace.push(n); return realNext(...a); };

  const perTurn = [];
  for (let t = 0; t < 2; t++) {
    // Force the weather onto the field WITHOUT drawing (set the internal field state so we
    // isolate the end-of-turn eachEvent('Weather'), independent of any set-weather draw).
    if (weather) {
      battle.field.weather = weather;
      battle.field.weatherState = { id: weather, duration: 9 };
    } else {
      battle.field.weather = '';
      battle.field.weatherState = {};
    }
    if (t === 1) trace = []; // capture the 2nd turn's draw positions
    const b = n;
    streams.omniscient.write('>p1 move 1');
    streams.omniscient.write('>p2 move 1');
    for (let k = 0; k < 14; k++) await tick();
    perTurn.push(n - b);
    if (battle.ended) break;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
  return { perTurn, spe0, spe1, trace, weatherAfter: battle.field.weather };
}

async function main() {
  console.log('=== gen3 weather-condition onFieldResidual ownership (mod-chain law) ===');
  const sun = dumpWeatherCondition('sunnyday');
  const rain = dumpWeatherCondition('raindance');
  const sand = dumpWeatherCondition('sandstorm');
  const hail = dumpWeatherCondition('hail');
  console.log(`  => sun/rain define own onFieldResidual? ${sun.hasOwnFR || rain.hasOwnFR} ` +
    `(EXPECT false — they inherit the base eachEvent('Weather') body)`);
  console.log('');

  const SEED = [12, 34, 56, 78];
  console.log('=== 2-turn Snorlax-mirror Splash, per-turn draw counts ===');
  const none = await run('', SEED);
  const rainR = await run('raindance', SEED);
  const sunR = await run('sunnyday', SEED);
  const sandR = await run('sandstorm', SEED);

  console.log(`  speeds (must tie): p1=${none.spe0} p2=${none.spe1}  tie=${none.spe0 === none.spe1}`);
  console.log(`  NO weather : perTurn=${JSON.stringify(none.perTurn)}`);
  console.log(`  RAIN       : perTurn=${JSON.stringify(rainR.perTurn)}  (weatherAfter=${rainR.weatherAfter})`);
  console.log(`  SUN        : perTurn=${JSON.stringify(sunR.perTurn)}  (weatherAfter=${sunR.weatherAfter})`);
  console.log(`  SAND       : perTurn=${JSON.stringify(sandR.perTurn)}  (weatherAfter=${sandR.weatherAfter})`);
  console.log('');

  // The claim: rain/sun draw EXACTLY one MORE per turn than no-weather (the end-of-turn
  // eachEvent('Weather') shuffle), and match sand (which also chips but chip is draw-free).
  const d_rain = rainR.perTurn[1] - none.perTurn[1];
  const d_sun = sunR.perTurn[1] - none.perTurn[1];
  const d_sand = sandR.perTurn[1] - none.perTurn[1];
  console.log(`  Δdraws vs no-weather (turn 2):  rain=+${d_rain}  sun=+${d_sun}  sand=+${d_sand}`);
  console.log(`  EXPECT rain==sun==sand==+1 (the end-of-turn eachEvent('Weather') tie shuffle)`);
  const pass = d_rain === 1 && d_sun === 1 && d_sand === 1 && none.spe0 === none.spe1;
  console.log('');
  console.log(pass
    ? 'CONFIRMED: sun/rain fire the end-of-turn eachEvent(Weather) shuffle UNCONDITIONALLY (== sand).'
    : 'NOT CONFIRMED — inspect the per-turn counts above.');
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
