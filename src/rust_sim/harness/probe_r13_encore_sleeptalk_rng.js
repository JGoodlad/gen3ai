// probe_r13_encore_sleeptalk_rng.js — REAL-Showdown ground truth for the R13 pin
// `gen3_encore_sleeptalk_trylhit_v1`: Sleep Talk's onTryHit
// (`return !volatiles['choicelock'] && !volatiles['encore']`) FAILS DRAW-FREE for a
// mon carrying the `encore` volatile. A faster foe Encores an asleep RestTalker whose
// lastMove is Sleep Talk (Encore locks Sleep Talk); that mon can then NEVER resolve
// Sleep Talk while encored — it fails at onTryHit (`|move|…Sleep Talk||[still]` +
// `|-fail|`), drawing NOTHING (no sample, no called move). The port used to SAMPLE +
// run the picked move (the R13 pool-fuzz divergence ab_15_15 dec 47, +4 draws).
//
// Run:  node src/rust_sim/harness/probe_r13_encore_sleeptalk_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: 'Serious', level: 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }
async function run(seed) {
  const p1 = [mon('Electrode', ['encore', 'spore', 'splash'], { evs: { spe: 252 } })];
  const p2 = [mon('Snorlax', ['sleeptalk', 'splash'], { evs: { hp: 252 } })];
  const stream = new BattleStream(); const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  console.log(`seed=${JSON.stringify(seed)} initSeed=${battle.prng.getSeed()}`);
  // plan: Spore(sleep Snorlax); Splash while Snorlax SleepTalks (lastMove->sleeptalk);
  //       Encore (locks sleeptalk) while Snorlax SleepTalk -> onTryHit encore FAILS.
  const plan = [
    { p1: 'move 2', p2: 'move 2' }, // Spore ; Splash
    { p1: 'move 3', p2: 'move 1' }, // Splash ; Sleep Talk (asleep)
    { p1: 'move 1', p2: 'move 1' }, // Encore ; Sleep Talk (encored -> fail)
  ];
  let i = 0, safety = 0;
  while (!battle.ended && safety < 20 && i < plan.length) {
    safety++;
    if (battle.requestState !== 'move' && battle.requestState !== 'switch') { await tick(); continue; }
    const e = plan[i]; i++;
    const before = battle.prng.getSeed();
    const l0 = log.length;
    if (e.p1) streams.omniscient.write(`>p1 ${e.p1}`);
    if (e.p2) streams.omniscient.write(`>p2 ${e.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const a1 = battle.sides[1].active[0];
    const slp = (a1 && a1.status === 'slp') ? ` slp(t=${a1.statusState.time})` : '';
    console.log(`dec${i-1} ${JSON.stringify(e)} seed ${before} -> ${battle.prng.getSeed()}  p2=${a1.hp}/${a1.maxhp}${slp} enc=${a1.volatiles['encore']?a1.volatiles['encore'].move:'-'} last=${a1.lastMove?a1.lastMove.id:null}`);
    for (const l of log.slice(l0).filter((x)=>/move\||-fail|-start|cant|-end|Sleep Talk/.test(x))) console.log('   LINE '+l);
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}
(async () => { await run([7, 11, 13, 17]); })();
