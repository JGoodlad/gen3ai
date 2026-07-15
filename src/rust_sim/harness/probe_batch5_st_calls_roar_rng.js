// probe_batch5_st_calls_roar_rng.js — REAL-Showdown ground truth for the
// `gen3_move_coverage_batch5_v1` Sleep-Talk-CALLED-ROAR composition pin
// (`regression_test.rs::sleep_talk_called_roar_drags_the_foe`, MC78).
//
// The review's coverage gap: the batch-5 golden's 23 scenarios call only Rest/attacks
// via Sleep Talk — the called-Roar drag composition (the called move's resolution
// PROPAGATES: a called Roar rides `force_switch_foe` → the runAction-tail `drag_in`
// `sample`) was coded but unpinned. This scenario constructs it deterministically:
//
//   p1 Suicune [Sleep Talk, Roar] (pool = [roar] only — Sleep Talk itself is
//     flags.nosleeptalk → excluded → the n=1 sample) vs
//   p2 Parasect [Spore, Splash] + a bench Snorlax (exactly one eligible drag target).
//
//   dec0: p1 Sleep Talk AWAKE (the silent onTry fail — a bare announce, zero draws)
//         + p2 Spore → p1 asleep (accuracy + the slp random(2,6)).
//   dec1: p1 Sleep Talk ASLEEP → |cant|slp (counter decrement, sleepUsable proceeds)
//         → the n=1 `sample` picks Roar → the CALLED Roar draws its accuracy roll →
//         drag: the n=1 `sample` pulls the lone bench Snorlax in; p2 Splash.
//
// Seed scan: the default seed must leave p1 STILL ASLEEP at dec1 (slp time >= 2).
// Prints per-boundary seedAfter + state — copied verbatim into the pin's constants.
//
// Run:  node src/rust_sim/harness/probe_batch5_st_calls_roar_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, seed, p1team, p2team, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  console.log(`\nTEAM p1 ${Teams.pack(p1team)}`);
  console.log(`TEAM p2 ${Teams.pack(p2team)}`);
  console.log(`=== ${label} ===  seed=${JSON.stringify(seed)} initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  const out = { dragged: false, asleepAtDec1: false };
  while (!battle.ended && safety < 30) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= plan.length) break;
    const entry = plan[i]; i++;
    const before = battle.prng.getSeed();
    const l0 = log.length;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const slp = (m) => (m && m.status === 'slp' && m.statusState) ? ` slp(t=${m.statusState.time})` : '';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${slp(m)}` : '-';
    console.log(`  dec${i - 1} [${rs}] ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} seed ${before} -> ${battle.prng.getSeed()}`);
    console.log(`      p1=${fmt(a0)}  p2=${fmt(a1)}`);
    const key = log.slice(l0).filter((l) => /move\||-damage|-fail|-immune|-miss|cant|faint|drag|switch\||-status/.test(l));
    for (const l of key) console.log(`      LINE ${l}`);
    if (i - 1 === 1) {
      out.asleepAtDec1 = a0 && a0.status === 'slp';
      out.dragged = key.some((l) => l.startsWith('|drag|'));
    }
  }
  try { streams.omniscient.destroy(); } catch (e) {}
  return out;
}

async function main() {
  const p1 = [mon('Suicune', ['sleeptalk', 'roar'], { evs: { hp: 252 } })];
  const p2 = [
    mon('Parasect', ['spore', 'splash'], { evs: { hp: 252 } }),
    mon('Snorlax', ['splash'], { evs: { hp: 252 } }),
  ];
  // Scan raw seeds until dec1 has p1 STILL ASLEEP + the called-Roar |drag| realized.
  for (let s = 7; s < 60; s++) {
    const r = await run(`MC78 seed [${s},${s + 4},${s + 6},${s + 10}]`, [s, s + 4, s + 6, s + 10], p1, p2, [
      { p1: 'move 1', p2: 'move 1' },  // ST awake-fail ; Spore lands
      { p1: 'move 1', p2: 'move 2' },  // ST asleep → n=1 sample → CALLED Roar → drag ; Splash
    ]);
    if (r.asleepAtDec1 && r.dragged) {
      console.log(`\n*** MC78 REALIZED at raw seed [${s},${s + 4},${s + 6},${s + 10}] — copy the boundary seeds above into the pin.`);
      return;
    }
    console.log(`  (seed [${s},…] did not realize: asleep=${r.asleepAtDec1} dragged=${r.dragged} — rescanning)`);
  }
  console.log('NO seed realized the scenario in the scan window — widen the scan.');
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
