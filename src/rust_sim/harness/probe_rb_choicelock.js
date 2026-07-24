// probe_rb_choicelock.js — SIM ORACLE for the CHOICE-LOCK volatile's LAZY removal
// (`gen3_choicelock_lazy_release_v1`, the rmrzcanyf_ab_71_14 `kind=seed` repro).
//
// The resolved gen3 `choicelock` condition:
//     onDisableMove(pokemon) {
//       if (!pokemon.getItem().isChoice || !pokemon.hasMove(this.effectState.move)) {
//         pokemon.removeVolatile('choicelock'); return;
//       } … }
// so the volatile SURVIVES a Knock Off / Trick that removes the Choice item, and is only
// dropped at the NEXT endTurn `runEvent('DisableMove')` — where it is STILL GATHERED (and so
// still counts toward that event's handler-sort TIE-SHUFFLE) before removing itself.
//
// This probe counts the PRNG calls per decision (the sodium seed advance is argument-blind, so
// the DRAW COUNT is the observable) on a board where an ENCORED, Choice-Banded mon has its Band
// knocked off, and prints each active's live volatiles at every boundary.
//
// Run:  node src/rust_sim/harness/probe_rb_choicelock.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }
const vols = (a) => a ? Object.keys(a.volatiles).join('+') || '-' : '-';

async function run(label, p1, p2, rawSeed, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 14; i++) await tick();
  const b = stream.battle;
  b.prng = new PRNG(rawSeed.slice());
  let draws = 0;
  const realRandom = b.prng.random.bind(b.prng);
  b.prng.random = (...a) => { draws++; return realRandom(...a); };
  console.log(`\n================ ${label} ================`);
  let mark = log.length;
  let i = 0;
  for (const entry of plan) {
    draws = 0;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 22; k++) await tick();
    const a0 = b.sides[0].active[0], a1 = b.sides[1].active[0];
    console.log(`--- dec ${i} ${JSON.stringify(entry)}  DRAWS=${draws}`);
    console.log(`      p1 vols=[${vols(a0)}] item=${a0 && a0.item} | p2 vols=[${vols(a1)}] item=${JSON.stringify(a1 && a1.item)}`);
    for (const l of log.slice(mark)) {
      if (l.startsWith('|t:|') || l.startsWith('|debug|')) continue;
      console.log(`      ${l}`);
    }
    mark = log.length;
    i++;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  const seed = [11, 29, 37, 53];
  const sableye = "Sableye||leftovers|Keen Eye|encore,knockoff,splash|Hardy|85,85,85,85,85,85|M||||";
  const aipomCB = "Aipom||choiceband|Run Away|return,shadowball|Hardy|85,85,85,85,85,85|M||||";
  const aipomNo = "Aipom|||Run Away|return,shadowball|Hardy|85,85,85,85,85,85|M||||";
  // dec0 p2 Return (arms choicelock) / p1 Splash          -> endTurn: choicelock only (n=1)
  // dec1 p1 Encore                                        -> endTurn: encore+choicelock (n=2, SHUFFLE)
  // dec2 p1 Knock Off (removes the Band)                  -> endTurn: encore+choicelock STILL gathered
  //                                                          (n=2, SHUFFLE) then choicelock self-removes
  // dec3 p1 Splash                                        -> endTurn: encore only (n=1, NO shuffle)
  await run('CL1 knock the Band off an ENCORED Choice-locked mon', sableye, aipomCB, seed, [
    { p1: 'move 3', p2: 'move 1' },
    { p1: 'move 1', p2: 'move 1' },
    { p1: 'move 2', p2: 'move 1' },
    { p1: 'move 3', p2: 'move 1' },
    { p1: 'move 3', p2: 'move 1' },
  ]);
  // Control: the same script with NO Choice Band — Aipom never gets a choicelock volatile, so
  // every endTurn has at most the encore handler (n<=1) and NEVER draws the shuffle.
  await run('CL2 control — no Choice Band (no choicelock volatile ever)', sableye, aipomNo, seed, [
    { p1: 'move 3', p2: 'move 1' },
    { p1: 'move 1', p2: 'move 1' },
    { p1: 'move 2', p2: 'move 1' },
    { p1: 'move 3', p2: 'move 1' },
    { p1: 'move 3', p2: 'move 1' },
  ]);
}

main().catch((e) => { console.error(e); process.exit(1); });
