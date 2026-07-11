// probe_intimidate_floor.js — GROUND TRUTH for the mid-battle Intimidate atk-stage-FLOOR
// emit (the bridge A/B fuzzer's "|-unboost|…|atk|0 at the −6 Atk floor" find).
//
// The port (turn.rs::emit_ability_start_lines) hardcodes `self.log.boost(&foe_ref, 0, -1)`
// → ALWAYS emits `|-unboost|<foe>|atk|1`. But Showdown emits the CLAMPED-APPLIED delta: a
// foe already at −6 Atk drops by 0 → the sim emits `|-unboost|<foe>|atk|0`; a foe at −5 →
// `atk|1` (lands −6). This probe pins the EXACT emitted line in each case.
//
// SETTLES (vs the omniscient in-process BattleStream — the sim is the oracle):
//   1. Intimidate into a foe at −6 Atk → what exactly is emitted? (`atk|0`? omitted? -fail?)
//   2. Intimidate into a foe at −5 → `atk|1`, lands −6.
//   3. Intimidate into Clear Body / White Smoke / Hyper Cutter → the `-fail … [from] ability:`
//      form regardless of stage (already handled — confirm unaffected).
//   4. Does the −6-floor result differ between a LEAD and a MID-BATTLE switch-in? (both match)
//
// Run:  node src/rust_sim/harness/probe_intimidate_floor.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// p1 = a single sturdy Splash mon (the Intimidate TARGET, whose Atk we drive down).
// p2 = 6 Intimidate Salamences that pivot in one after another, each dropping p1's Atk by 1
//      until it floors at −6, then one MORE switch-in at the floor. We record the atkBoost
//      BEFORE each switch-in + the emitted `|-unboost|`/`-fail` line so we see the exact delta.
async function run(label, p1mon, p2team, extraPlan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[3,5,7,9]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([p1mon]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const battle = stream.battle;
  console.log(`\n=== ${label} ===`);
  // The lead Salamence (p2 slot 1) already Intimidated at the >start switch-in.
  {
    const a0 = battle.sides[0].active[0];
    console.log(`  [LEAD switch-in] p1 atkBoost now = ${a0.boosts.atk}`);
    for (const l of log) if (/-unboost|-fail|-ability.*Intimidate|-immune/.test(l)) console.log(`      ${l}`);
  }
  // Now pivot each subsequent Salamence in; before each, print the CURRENT p1 atk stage.
  const plan = extraPlan;
  for (const e of plan) {
    const a0before = battle.sides[0].active[0].boosts.atk;
    const l0 = log.length;
    streams.omniscient.write(`>p1 ${e.p1}`);
    streams.omniscient.write(`>p2 ${e.p2}`);
    for (let k = 0; k < 16; k++) await tick();
    const a0 = battle.sides[0].active[0];
    console.log(`  ${JSON.stringify(e)}  p1 atkBoost ${a0before} -> ${a0.boosts.atk}  seedAfter=${battle.prng.getSeed()}`);
    for (const l of log.slice(l0)) if (/-unboost|-fail|-ability.*Intimidate|-immune/.test(l)) console.log(`      ${l}`);
  }
}

async function main() {
  // Scenario 1: drive p1 to −6 via 6 Intimidate pivots, then a 7th switch-in AT the floor.
  const salas = [];
  for (let i = 0; i < 7; i++) salas.push(mon('Salamence', ['dragonclaw', 'splash'], { ability: 'Intimidate' }));
  // p2 pivots: after the lead Intimidated (-1), switch in slots 2..7 (each -1) then re-pivot.
  // Team-slot indices shift as mons swap to active; we just keep switching to whatever bench
  // Salamence is available. Use `switch 2` repeatedly (the active-index-2 bench walk).
  const plan1 = [
    { p1: 'move 2', p2: 'switch 2' }, // -2
    { p1: 'move 2', p2: 'switch 2' }, // -3
    { p1: 'move 2', p2: 'switch 2' }, // -4
    { p1: 'move 2', p2: 'switch 2' }, // -5
    { p1: 'move 2', p2: 'switch 2' }, // -6
    { p1: 'move 2', p2: 'switch 2' }, // AT FLOOR: -6 -> -6 (applied delta 0)
  ];
  const p1target = mon('Snorlax', ['bodyslam', 'splash'], { evs: { hp: 252 } });
  await run('MID-BATTLE Intimidate drives p1 Atk to −6 then a floor switch-in', p1target, salas, plan1);

  // Scenario 2: Intimidate into Clear Body (the -fail form, stage-independent).
  const cb = [mon('Metagross', ['meteormash', 'splash'], { ability: 'Clear Body' })];
  const sala2 = [mon('Salamence', ['dragonclaw', 'splash'], { ability: 'Intimidate' }),
                 mon('Salamence', ['dragonclaw', 'splash'], { ability: 'Intimidate' })];
  await run('Intimidate into CLEAR BODY (lead + a mid-battle re-pivot)', cb[0], sala2, [
    { p1: 'move 2', p2: 'switch 2' },
  ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
