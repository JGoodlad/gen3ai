// probe_batch4_pursuit_bench_regression_rng.js — GROUND TRUTH for the PURSUIT-INTERRUPT
// bench-order fix pins (MC36 + the two batch-4 nit pins, `gen3_move_coverage_batch4_v1`).
//
// MC36  pursuit_does_not_intercept_a_baton_pass_selfswitch — a pursued BATON-PASS passer is
//       NOT struck (the sim SUPPRESSES BeforeSwitchOut for a selfSwitch via
//       `batonpass.self.onHit → skipBeforeSwitchOutEventFlag=true`); the passer survives, passes
//       its boosts, and Pursuit runs NORMALLY against the ENTRANT next decision.
// MC37  pursuit_interrupt_into_entry_hazards — a VOLUNTARY switch into a Pursuit STRIKE, then the
//       replacement enters through the runSwitch EntryHazard (Spikes chip). The strike→swap→
//       runSwitch(spikes) composition.
// MC38  pursuit_speed_tie_interrupt — the pursuer and the SWITCHER tie on speed, so the strike's
//       in-tryMoveHit `eachEvent('Update')` draws ONE tie-shuffle (the post-strike each_event draw).
//
// Reseeds the sim to the RAW seed before the first decision (the port's start_with_switchins is
// draw-free), then prints per-decision seedAfter + STATE. Run:
//   node src/rust_sim/harness/probe_batch4_pursuit_bench_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }
const RAW = [44317, 42357, 9927, 48760];

async function run(label, p1, p2, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  if (inject) inject(b);
  b.prng = new PRNG(RAW.slice());
  const fmt = (m) => (m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.status ? ' ' + m.status : ''} atk${m.boosts.atk}` : '-');
  console.log(`\n=== ${label} (raw ${RAW.join(',')}) ===`);
  let i = 0, safety = 0;
  while (!b.ended && safety < 60 && i < plan.length) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = log.length;
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    console.log(`  dec ${i - 1} [${rs}] ${JSON.stringify(entry)}  seedAfter=${b.prng.getSeed()}`);
    console.log(`      p1 active=${fmt(b.sides[0].active[0])} left=${b.sides[0].pokemonLeft}`);
    console.log(`      p2 active=${fmt(b.sides[1].active[0])} left=${b.sides[1].pokemonLeft}`);
    // bench dump
    b.sides[1].pokemon.forEach((pk, idx) => { if (!pk.isActive) console.log(`      p2 bench[${idx}]=${fmt(pk)}`); });
    log.slice(before).filter((l) => /\|move\||-damage|-activate|cant|switch|faint|-crit|-supereffective|-boost|Spikes/.test(l))
      .forEach((l) => console.log(`      L ${l}`));
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  const ttarPursuit = 'Tyranitar|||pressure|pursuit,crunch|Serious|,,,252,,|||||';
  const jolteonBP = 'Jolteon|||voltabsorb|batonpass,thunderbolt|Serious|,,,,,252|||||]Vaporeon|||waterabsorb|surf|Serious|252,,,,,|||||';

  // MC36 main: Jolteon (+2 Atk injected) Baton Passes; Pursuit does NOT strike it; Vaporeon
  //   enters with +2 Atk and takes a NORMAL Pursuit.
  await run('MC36 BP not intercepted (+2 Atk injected)', ttarPursuit, jolteonBP,
    [{ p1: 'move 1', p2: 'move 1' }, { p2: 'switch 2' }],
    (b) => { b.sides[1].active[0].boosts.atk = 2; });

  // MC36b low-HP passer: Jolteon injected to 10 HP; a pre-fix strike would KO it. The sim leaves
  //   it ALIVE on the bench at 10 HP.
  await run('MC36b BP low-HP passer survives (hp=10 injected)', ttarPursuit, jolteonBP,
    [{ p1: 'move 1', p2: 'move 1' }, { p2: 'switch 2' }],
    (b) => { b.sides[1].active[0].hp = 10; });

  // MC37 hazards: p1 lays Spikes, then Pursuit-strikes a voluntary switcher; the replacement
  //   takes the Spikes chip on entry.
  const ttarSpikes = 'Tyranitar|||pressure|pursuit,spikes|Serious|,,,252,,|||||';
  const jolSnor = 'Jolteon|||voltabsorb|thunderbolt|Serious|,,,,,252|||||]Snorlax|||immunity|bodyslam|Serious|252,,,,,|||||';
  await run('MC37 pursuit interrupt into entry hazards', ttarSpikes, jolSnor,
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'switch 2' }]);

  // MC38 speed-tie: pursuer + switcher are the SAME species/spread → tie → the strike's
  //   in-tryMoveHit eachEvent('Update') draws one tie-shuffle.
  const ttarTie = 'Tyranitar|||pressure|pursuit,crunch|Serious|,,,252,,|||||';
  const ttarTie2 = 'Tyranitar|||pressure|crunch|Serious|,,,252,,|||||]Snorlax|||immunity|bodyslam|Serious|252,,,,,|||||';
  await run('MC38 pursuer/switcher speed-tie interrupt', ttarTie, ttarTie2,
    [{ p1: 'move 1', p2: 'switch 2' }]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
