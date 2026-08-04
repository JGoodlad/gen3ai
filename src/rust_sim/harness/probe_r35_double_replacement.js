// probe_r35_double_replacement.js — ROUND 35 honest-scope closer: the DOUBLE-replacement
// Castform × Intimidate ordering (`forecast.onSwitchInPriority: -2`).
//
// A mutual Explosion double-KO under standing rain; p1 replaces with a Forecast Castform,
// p2 with an Intimidate Salamence. Questions the round disclosed as unprobed:
//   (1) the LINE ORDER of the two entrants' switch-in effects (|switch| ×2, -ability/-unboost,
//       -formechange) — does the -2 priority reorder anything observable?
//   (2) the DRAW count/order across the boundary (the insertChoice splice + any Start draws).
// Prints per-decision seeds + the boundary's full line block, for the port comparison.
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
  console.log(`  initSeed=${battle.prng.getSeed()} qc=${battle.quickClawRoll}`);
  let oLo = omni.length;
  let i = 0;
  for (const [c1, c2] of choices) {
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 14; k++) await tick();
    i += 1;
    const lines = omni.slice(oLo).filter((l) => l.startsWith('|') && !l.startsWith('|t:|') && !l.startsWith('|debug'));
    console.log(`  dec${i} seedAfter=${battle.prng.getSeed()}`);
    for (const l of lines) console.log(`      ${l}`);
    oLo = omni.length;
    if (battle.ended) break;
  }
}

async function main() {
  // p1: Snorlax (Explosion, Adamant 252 Atk) + Castform(Forecast). p2: Abra (frail — the
  // Explosion DOUBLE-KOs) + Salamence (Intimidate). t1: p2 Abra... has no Rain Dance — the
  // rain comes from p1 Snorlax casting it t1. t2 Snorlax explodes (self-KO + Abra dies) →
  // a TRUE double replacement: Castform + Salamence enter simultaneously under rain.
  const lax = mon('Snorlax', ['explosion', 'raindance'], { ability: 'Immunity', nature: 'Adamant', evs: { hp: 0, atk: 252, def: 0, spa: 0, spd: 0, spe: 0 } });
  const abra = mon('Abra', ['splash', 'splash'], { ability: 'Inner Focus', evs: { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 } });
  await run('DR: castform + intimidate TRUE double-replacement under rain',
    [[lax, mon('Castform', ['icebeam', 'splash'], { ability: 'Forecast' })],
     [abra, mon('Salamence', ['splash', 'splash'], { ability: 'Intimidate' })]],
    [3, 5, 7, 9],
    [['move 2', 'move 1'], ['move 1', 'move 1'], ['switch 2', 'switch 2'], ['move 2', 'move 1'], ['move 2', 'move 1']]);
  // The MIRROR: sides swapped (Intimidate side explodes... keep the exploder+castform on p2).
  await run('DR-mirror: castform on p2, intimidate on p1',
    [[abra, mon('Salamence', ['splash', 'splash'], { ability: 'Intimidate' })],
     [lax, mon('Castform', ['icebeam', 'splash'], { ability: 'Forecast' })]],
    [3, 5, 7, 9],
    [['move 1', 'move 2'], ['move 1', 'move 1'], ['switch 2', 'switch 2'], ['move 2', 'move 1'], ['move 2', 'move 1']]);
}

main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
