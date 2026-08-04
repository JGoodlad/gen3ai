// probe_r35_forecast_expiry_draw.js — ROUND 35, the EXPIRY-TURN eachEvent question.
//
// `probe_r35_forecast_ties.js` T1 showed, on SPEED-TIED actives with a Cloud Nine holder up,
// an expiring HAIL turn drawing 8 while an ordinary upkeep turn drew 7 — one EXTRA draw that
// the control (Levitate, weather EFFECTIVE) does not show. That is the signature of:
//
//   * upkeep turn  -> residualEvent runs the weather's `onFieldResidual` -> eachEvent('Weather')
//                     ... which the PORT gates OFF for a SUPPRESSED sand/hail (RM3), and
//   * expiry turn  -> residualEvent's `duration-- === 0` branch calls `handler.end()` =
//                     `field.clearWeather()` -> eachEvent('WeatherChange') and `continue`s
//                     (so NO eachEvent('Weather') at all) — and that WeatherChange is
//                     UNCONDITIONAL, suppressed or not.
//
// This probe settles it DIRECTLY by tracing the eachEvent ids + the prng.shuffle calls on the
// expiry turn, for the 2x2 {suppressed, effective} x {hail, rain}. It is the ground truth for
// whether `residuals.rs`'s expiry branch may keep its `Sun|Rain || effective` gate.
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const { mon } = require('./probe_batch4_lib');

function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(teams, seed, choices) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const omni = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of String(ch).split('\n')) omni.push(l); })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(teams[0]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(teams[1]) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const battle = stream.battle;

  const trace = [];
  const realEach = battle.eachEvent.bind(battle);
  battle.eachEvent = (eventid, effect, relayVar) => { trace.push(`each:${eventid}`); return realEach(eventid, effect, relayVar); };
  const realShuffle = battle.prng.shuffle.bind(battle.prng);
  battle.prng.shuffle = (...a) => { trace.push('SHUFFLE'); return realShuffle(...a); };
  const rng = battle.prng.rng; const realNext = rng.next.bind(rng);
  let draws = 0; rng.next = (...a) => { draws += 1; return realNext(...a); };

  const per = [];
  let oLo = omni.length, tLo = 0, dLo = 0;
  for (const [c1, c2] of choices) {
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 14; k++) await tick();
    per.push({ omni: omni.slice(oLo), trace: trace.slice(tLo), draws: draws - dLo });
    oLo = omni.length; tLo = trace.length; dLo = draws;
    if (battle.ended) break;
  }
  return per;
}

// Castform 70/70/70/70/70/70, L100, no EVs -> spe 176. Psyduck base spe 55 -> needs EVs to tie.
// Instead use a SECOND Castform-statline mon so the tie is structural: Psyduck base 55.
// We give the foe explicit EVs to land exactly on 176.
function foe(ability) {
  // base spe 55 -> 2*55+31+X/4+5 = 176 -> 141+X/4 = 176 -> X/4 = 35 -> X = 140
  return mon('Psyduck', ['splash'], { ability, evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 140 } });
}

async function main() {
  for (const weather of ['hail', 'raindance']) {
    for (const [tag, ability] of [['SUPPRESSED(cloudnine)', 'Cloud Nine'], ['EFFECTIVE(levitate)', 'Levitate']]) {
      const teams = [
        [mon('Castform', [weather, 'splash'], { ability: 'Forecast' })],
        [foe(ability)],
      ];
      const per = await run(teams, [7, 7, 7, 7], [
        ['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'],
        ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'],
      ]);
      console.log(`\n### ${weather} ${tag}`);
      per.forEach((d, i) => {
        const ev = d.trace.filter((t) => t.startsWith('each:') || t === 'SHUFFLE');
        const isExpiry = d.omni.some((l) => l === '|-weather|none');
        console.log(`  t${i + 1}${isExpiry ? ' [EXPIRY]' : ''} draws=${d.draws} ${JSON.stringify(ev)}`);
      });
    }
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
