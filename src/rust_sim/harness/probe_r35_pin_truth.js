// probe_r35_pin_truth.js — ROUND 35 regression-pin ground truth (the real sim's seeds).
//
// Prints, for each pin board: the POST-CONSTRUCTION initSeed (the seed the port's
// draw-free `start_with_switchins` must be given — the round-29 methodology) and the
// per-decision seedAfter + key lines. Boards:
//   PT-A  Snorlax(hail)+splash MIRROR (tied 96) — set / upkeep / EXPIRY / post. The expiry
//         turn's seed pins the UNCONDITIONAL clearWeather WeatherChange draw.
//   PT-B  Suicune(hail) vs Cloud Nine Psyduck (tied 206) — the SUPPRESSED board: upkeep
//         turns draw NO Weather event; the EXPIRY turn draws the WeatherChange (T1 8-vs-7).
//   PT-C  Castform(raindance,splash) vs Snorlax(splash) — forme Rainy at the set, revert at
//         expiry (the forme STATE + `-formechange` bytes at untied speeds, draw-light).
//   PT-D  Castform(splash) vs Ninetales-Drought — the START-window forme (framing).
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const { mon } = require('./probe_batch4_lib');

function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(name, teams, seed, choices) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const omni = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of String(ch).split('\n')) omni.push(l); })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(teams[0]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(teams[1]) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const battle = stream.battle;
  console.log(`\n### ${name}`);
  console.log(`  initSeed(post-construction) = ${battle.prng.getSeed()}`);
  console.log(`  quickClawRoll(turn0) = ${battle.quickClawRoll}`);
  let oLo = omni.length;
  let i = 0;
  for (const [c1, c2] of choices) {
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 14; k++) await tick();
    i += 1;
    const lines = omni.slice(oLo).filter((l) => /-weather|formechange|damage.*(Hail|Sandstorm)|still|-fail/.test(l));
    console.log(`  dec${i} seedAfter=${battle.prng.getSeed()} lines=${JSON.stringify(lines)}`);
    oLo = omni.length;
    if (battle.ended) break;
  }
  // The final board's Castform state where relevant.
  for (const side of battle.sides) {
    for (const p of side.pokemon) {
      if (p.baseSpecies.baseSpecies === 'Castform') {
        console.log(`  ${side.id} castform: species=${p.species.id} types=${JSON.stringify(p.types)}`);
      }
    }
  }
}

async function main() {
  const lax = (moves) => mon('Snorlax', moves, { ability: 'Immunity' });
  const M6 = [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'],
              ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1']];
  await run('PT-A snorlax-hail-mirror tied96', [[lax(['hail', 'splash'])], [lax(['splash', 'splash'])]], [11, 22, 33, 44], M6);
  await run('PT-B suicune-hail vs cloudnine-psyduck tied206',
    [[mon('Suicune', ['hail', 'splash'], { ability: 'Pressure' })],
     [mon('Psyduck', ['splash', 'splash'], { ability: 'Cloud Nine', evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 240 } })]],
    [11, 22, 33, 44], M6);
  await run('PT-C castform-raindance vs snorlax',
    [[mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' })], [lax(['splash', 'splash'])]],
    [11, 22, 33, 44], M6);
  await run('PT-D castform vs ninetales-drought (start-window forme)',
    [[mon('Castform', ['splash', 'splash'], { ability: 'Forecast' })],
     [mon('Ninetales', ['splash', 'splash'], { ability: 'Drought' })]],
    [11, 22, 33, 44], [['move 1', 'move 1'], ['move 1', 'move 1']]);
}

// FC5-FC7 boards (appended): run with `node probe_r35_pin_truth.js --more`.
async function more() {
  // FC5: Cloud Nine Psyduck LEAVES while rain is up -> the Castform formes on the
  // negate-onEnd WeatherChange (the ending-negater exclusion), BEFORE the |switch| line.
  await run('PT-E cloudnine leaves under rain (castform formes on the End site)',
    [[mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' })],
     [mon('Psyduck', ['splash', 'splash'], { ability: 'Cloud Nine' }), mon('Blissey', ['splash'], { ability: 'Natural Cure' })]],
    [11, 22, 33, 44],
    [['move 1', 'move 1'], ['move 2', 'switch 2'], ['move 2', 'move 1']]);
  // FC6: the formed Castform pivots OUT (silent revert) and back IN (re-forme onStart).
  await run('PT-F formed castform switches out then back in',
    [[mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' }), mon('Rattata', ['splash'], { ability: 'Guts' })],
     [mon('Snorlax', ['splash', 'splash'], { ability: 'Immunity' })]],
    [11, 22, 33, 44],
    [['move 1', 'move 1'], ['switch 2', 'move 1'], ['switch 2', 'move 1'], ['move 2', 'move 1']]);
  // FC7: the Castform MIRROR at a tie — the -formechange order follows the shuffle (two seeds).
  for (const seed of [[11, 22, 33, 44], [44, 33, 22, 11]]) {
    await run(`PT-G castform mirror tie seed=${JSON.stringify(seed)}`,
      [[mon('Castform', ['raindance', 'splash'], { ability: 'Forecast' })],
       [mon('Castform', ['splash', 'splash'], { ability: 'Forecast' })]],
      seed, [['move 1', 'move 1'], ['move 2', 'move 1']]);
  }
}
(process.argv.includes('--more') ? more() : main())
  .catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
