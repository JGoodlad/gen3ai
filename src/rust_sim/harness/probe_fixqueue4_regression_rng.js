// probe_fixqueue4_regression_rng.js — GROUND TRUTH for the 2026-07-10 A/B
// fix-queue-#4 regression pins (tests/regression_test.rs):
//   FQ1 double_faint_processes_corpses_in_enqueue_order       (`gen3_faint_queue_order_v1`)
//   FS1 fainted_swift_swim_corpse_sorts_at_plain_speed        (`gen3_fainted_no_ability_speed_v1`)
//   TX1 tox_stage_resets_when_the_runswitch_runs              (`gen3_tox_stage_persists_v1`)
//   TX2 tox_stage_persists_when_the_runswitch_is_cancelled    (`gen3_tox_stage_persists_v1`)
//
// THE BUGS (the auto_0709_2205 steady-state 9-repro corpus, all probe-settled):
//  FQ1: `faintMessages` drains `faintQueue` in ENQUEUE order, fully processing each
//       corpse (`fainted=true`, `isActive=false`) before the next corpse's ability-End —
//       so when an Explosion USER (enqueued first by the self-KO `faint(user)`) and a
//       CLOUD NINE target double-faint, the Cloud Nine End's `eachEvent('WeatherChange')`
//       gathers ONLY itself → NO tie-shuffle even on a cached-speed tie (ab_723_13 /
//       ab_464_16: the port walked side order → processed the CN corpse first → a
//       phantom draw).
//  FS1: a FAINTED mon's ability handlers no longer gather — a Swift Swim corpse under
//       rain sorts the replacement instaswitch at its PLAIN getActionSpeed (alive 368 →
//       fainted 184 on the ab_894_12 board), TYING an equal plain corpse → the shuffle
//       draw the port missed.
//  TX1/TX2: the gen3 `tox.onSwitchIn(){stage=0}` reset fires via the gen4-override
//       runSwitch's `runEvent('SwitchIn')` — NOT at the raw switch swap. So the stage
//       RESETS when the entrant's runSwitch RUNS (TX1), but PERSISTS when that queued
//       runSwitch is CANCELLED by the gen3 faint-cancels-all rule (TX2 — the ab_1166_22
//       Mew: its co-replacement died to Spikes, cancelling Mew's runSwitch; Mew's next
//       residual kept ramping from its prior stage and KO'd it).
//
// Run:  node src/rust_sim/harness/probe_fixqueue4_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1, p2, rawSeed, plan, pre) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  b.prng = new PRNG(rawSeed.slice());
  if (pre) pre(b);
  console.log(`\n=== ${label} (raw seed ${rawSeed.join(',')}) ===`);
  let i = 0, safety = 0;
  while (!b.ended && safety < 80 && i < plan.length) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const u = b.sides[0].active[0];
    const t = b.sides[1].active[0];
    console.log(`  dec ${i - 1} ${JSON.stringify(entry)} req=${b.requestState} seedAfter=${b.prng.getSeed()}`);
    console.log(`    p1=${u.species.name} ${u.hp}/${u.maxhp} st=${u.status || '-'}  p2=${t.species.name} ${t.hp}/${t.maxhp} st=${t.status || '-'} p2stage=${t.statusState ? t.statusState.stage : '-'}`);
  }
  return { b, log };
}

const S = (name, item, abil, moves, lvl = 100) =>
  `${name}||${item}|${abil}|${moves}|Hardy|85,85,85,85,85,85|M|||${lvl}|`;
const snorlax = S('Snorlax', 'Leftovers', 'ThickFat', 'splash,bodyslam');

async function main() {
  // FQ1: p1 Golduck-L81 Cloud Nine (184) vs p2 Smeargle-L89 (184) — cached-speed TIE.
  // Golduck at 1 HP; Smeargle Explodes → the USER is enqueued first, so the Cloud Nine
  // corpse's End WeatherChange gathers only itself → NO extra tie draw.
  const golduckCN = S('Golduck', 'Leftovers', 'CloudNine', 'splash,surf', 81);
  const smeargleBoom = S('Smeargle', 'Leftovers', 'OwnTempo', 'explosion,splash', 89);
  await run('FQ1_enqueue_order (Explosion user first; CN corpse End alone)',
    golduckCN + ']' + snorlax, smeargleBoom + ']' + snorlax, [0, 0, 0, 41],
    [{ p1: 'move 1', p2: 'move 1' },     // splash vs Explosion → mutual double faint
     { p1: 'switch 2', p2: 'switch 2' }, // double replacement
     { p1: 'move 1', p2: 'move 1' }],
    (b) => { b.sides[0].active[0].hp = 1; });

  // FS1: p2 leads Kyogre (Drizzle → permanent rain, the ab_894_12 rain source —
  // Rain Dance the MOVE is unmodeled), pivots to Kingdra-L81 Swift Swim (alive-in-
  // rain 368). The mutual Explosion KO → both corpses sort at PLAIN 184 == 184 →
  // the instaswitch tie shuffle + the resumed Quick Claw.
  const kyogre = `Kyogre||Leftovers|Drizzle|splash,surf|Hardy|85,85,85,85,85,85|N|||67|`;
  const kingdraSS = S('Kingdra', 'Leftovers', 'SwiftSwim', 'splash,icebeam', 81);
  await run('FS1_swiftswim_corpse (Drizzle rain up; corpse sorts plain → tie)',
    smeargleBoom + ']' + snorlax,
    kyogre + ']' + kingdraSS + ']' + S('Blissey', 'Leftovers', 'NaturalCure', 'splash,icebeam'),
    [0, 0, 0, 43],
    [{ p1: 'move 2', p2: 'switch 2' },   // Kingdra in under the permanent rain
     { p1: 'move 1', p2: 'move 1' },     // Explosion vs splash → mutual double faint
     { p1: 'switch 2', p2: 'switch 2' }, // double replacement — the tie draw + QC (Kyogre back)
     { p1: 'move 1', p2: 'move 1' }],
    (b) => { b.sides[1].pokemon[1].hp = 1; });

  // TX1: Smeargle toxics Swampert; Swampert ramps stage 1, pivots out, re-enters
  // (voluntary; its runSwitch RUNS) → the stage RESETS: next residuals are 1×, 2×.
  const smeargleTox = S('Smeargle', 'Leftovers', 'OwnTempo', 'toxic,splash', 89);
  const swampert = S('Swampert', 'Leftovers', 'Torrent', 'splash,surf');
  await run('TX1_reset_on_runswitch (voluntary out-and-back resets the stage)',
    smeargleTox, swampert + ']' + snorlax, [0, 0, 0, 47],
    [{ p1: 'move 1', p2: 'move 1' },     // Toxic lands (verify!) → residual stage 1
     { p1: 'move 2', p2: 'switch 2' },   // Swampert out
     { p1: 'move 2', p2: 'switch 2' },   // Swampert back IN (runSwitch runs → reset) → stage 1
     { p1: 'move 2', p2: 'move 1' }]);   // stage 2

  // TX2: p2 Smeargle lays Spikes on p1's side, then the mutual Explosion double-faint.
  // p1 replaces with a 1-HP Electrode (FAST → its runSwitch runs first, dies to
  // Spikes → the gen3 faint-cancel removes p2's pending runSwitch), p2 replaces with a
  // pre-ramped tox Swampert (stage 2). Swampert's runSwitch was CANCELLED → NO reset:
  // the resumed residual ramps 2→3 (3×25 = 75 chip on a 404-maxhp Swampert).
  const smeargleSpk = S('Smeargle', 'Leftovers', 'OwnTempo', 'spikes,splash', 89);
  const electrode = S('Electrode', 'Leftovers', 'Static', 'splash,thunderbolt');
  await run('TX2_persist_on_cancelled_runswitch (co-replacement Spikes-faint cancels)',
    smeargleBoom + ']' + electrode + ']' + snorlax,
    smeargleSpk + ']' + swampert + ']' + S('Blissey', 'Leftovers', 'NaturalCure', 'splash,icebeam'),
    [0, 0, 0, 53],
    [{ p1: 'move 2', p2: 'move 1' },     // splash vs Spikes (p1 side gets a layer)
     { p1: 'move 1', p2: 'move 2' },     // Explosion vs splash → mutual double faint
     { p1: 'switch 2', p2: 'switch 2' }, // Electrode (1 HP) + tox Swampert; Electrode's runSwitch spikes-KOs it → cancels Swampert's
     { p1: 'switch 3' },                 // Snorlax in; resumed residual: Swampert tox ramps 2→3 (NO reset)
     { p1: 'move 1', p2: 'move 1' }],
    (b) => {
      b.sides[0].pokemon[1].hp = 1;                 // Electrode dies to the Spikes chip
      b.sides[1].pokemon[1].setStatus('tox');       // Swampert pre-ramped badly-poisoned
      b.sides[1].pokemon[1].statusState.stage = 2;
    });
}
main();
