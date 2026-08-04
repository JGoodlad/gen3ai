// probe_r35_weather_moves.js — ROUND 35: the HAIL + SANDSTORM weather-set MOVES.
//
// Forecast needs timed HAIL (Castform-Snowy), so the two remaining C_WEATHER_SET members
// must be modeled first. The batch-2 machinery (`modeled_weather_set_move`) is generic over
// Weather, but "the sim is the oracle": this probe settles, for each of {hail, sandstorm},
// on a genuinely SPEED-TIED board (the expiry_draw probe's board was NOT tied — its EV math
// dropped the +5):
//   W1 the set-turn byte form (`|-weather|Hail` / `|-weather|Sandstorm`) + the WeatherChange
//      tie draw;
//   W2 the upkeep tick + per-active chip lines (`[from] Hail`, Ice immune; `[from]
//      Sandstorm`, Rock/Ground/Steel immune) + the per-turn draw counts;
//   W3 the fail-into-same form (`[still]` + `|-fail|`) and its draw count;
//   W4 the EXPIRY turn (`|-weather|none` + the UNCONDITIONAL WeatherChange draw);
//   W5 the SUPPRESSED (Cloud Nine) upkeep + expiry draw counts — hail's upkeep Weather event
//      is SKIPPED under a suppressor while the expiry WeatherChange still fires (the T1
//      8-vs-7 lead, re-derived on a VERIFIED-tied board).
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
  // NON-VACUITY: the tie is the whole point — assert it or the probe silently tests nothing.
  const spds = battle.sides.map((s) => s.active[0].getStat('spe'));
  if (spds[0] !== spds[1]) throw new Error(`NOT TIED: spe ${spds[0]} vs ${spds[1]}`);

  const trace = [];
  const realEach = battle.eachEvent.bind(battle);
  battle.eachEvent = (eventid, effect, relayVar) => { trace.push(`each:${eventid}`); return realEach(eventid, effect, relayVar); };
  const rng = battle.prng.rng; const realNext = rng.next.bind(rng);
  let draws = 0; rng.next = (...a) => { draws += 1; return realNext(...a); };

  const per = [];
  let oLo = omni.length, tLo = 0, dLo = 0;
  for (const [c1, c2] of choices) {
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 14; k++) await tick();
    per.push({ omni: omni.slice(oLo).filter((l) => l.startsWith('|')), trace: trace.slice(tLo), draws: draws - dLo });
    oLo = omni.length; tLo = trace.length; dLo = draws;
    if (battle.ended) break;
  }
  return per;
}

async function scenario(name, teams, choices, showLines) {
  const per = await run(teams, [7, 7, 7, 7], choices);
  console.log(`\n### ${name}`);
  per.forEach((d, i) => {
    const ev = d.trace.filter((t) => t.startsWith('each:'));
    console.log(`  t${i + 1} draws=${d.draws} ${JSON.stringify(ev)}`);
    if (showLines) {
      for (const l of d.omni) {
        if (/-weather|damage.*(Hail|Sandstorm)|still|-fail|move\|/.test(l)) console.log(`      ${l}`);
      }
    }
  });
}

async function main() {
  // W1/W2/W4: Snorlax mirror (spe 96 both) — Normal types, both CHIP under hail AND sand.
  const lax = (moves) => mon('Snorlax', moves, { ability: 'Immunity' });
  for (const wmove of ['hail', 'sandstorm']) {
    await scenario(`${wmove} set/upkeep/expiry — Snorlax mirror (tied 96)`,
      [[lax([wmove, 'splash'])], [lax(['splash', 'splash'])]],
      [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'],
       ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1']], true);
    // W3: fail-into-same (cast on t1, re-cast on t2).
    await scenario(`${wmove} re-cast FAILS into same — Snorlax mirror`,
      [[lax([wmove, 'splash'])], [lax(['splash', 'splash'])]],
      [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 2', 'move 1']], true);
  }
  // W2b: hail chip IMMUNITY — Regice (Ice) mirror, no chip lines expected.
  await scenario('hail chip immunity — Regice mirror (Ice, tied 122)',
    [[mon('Regice', ['hail', 'splash'], { ability: 'Clear Body' })],
     [mon('Regice', ['splash', 'splash'], { ability: 'Clear Body' })]],
    [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1']], true);
  // W5: SUPPRESSED hail on a TIED board — Suicune (206) vs Psyduck Cloud Nine EV'd to 206:
  // 2*55+31+240/4+5 = 206. The T1 re-derivation: upkeep turns skip the Weather event
  // (fewer draws than effective), the EXPIRY turn still draws its WeatherChange.
  await scenario('hail SUPPRESSED — Suicune 206 vs CloudNine Psyduck 206 (tied)',
    [[mon('Suicune', ['hail', 'splash'], { ability: 'Pressure' })],
     [mon('Psyduck', ['splash', 'splash'], { ability: 'Cloud Nine', evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 240 } })]],
    [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'],
     ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1']], true);
  // W5b: the EFFECTIVE control at the same speeds (Levitate foe — hail chips it).
  await scenario('hail EFFECTIVE control — Suicune 206 vs Levitate Psyduck 206 (tied)',
    [[mon('Suicune', ['hail', 'splash'], { ability: 'Pressure' })],
     [mon('Psyduck', ['splash', 'splash'], { ability: 'Levitate', evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 240 } })]],
    [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'],
     ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1']], true);
}

main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
