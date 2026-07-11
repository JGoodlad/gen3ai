// probe_accuracy_pins.js — ground-truth for the AC regression pins.
// Runs a handful of CONSTRUCTED accuracy scenarios in the OMNISCIENT gen3 BattleStream and
// prints, per decision, the post-decision seed + both actives' HP. The Rust pins assert
// the PORT reproduces these — so a revert of the acc fold flips a roll and diverges.
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
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(name, p1, p2, seed, plan1, plan2, nDec) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([p1]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([p2]) })}`);
  for (let i = 0; i < 10; i++) await tick();
  const b = stream.battle;
  console.log(`\n### ${name}  >start seed=${seed.join(',')}`);
  console.log(`  p1=${Teams.pack([p1])}`);
  console.log(`  p2=${Teams.pack([p2])}`);
  // The prng seed AT decision 0 (post-switch-in) — the seed the Rust pin must START from
  // (start_with_switchins reproduces the switch-in advance from the >start seed, but the pin
  // asserts against this decision-0 seed, matching the golden test's use of rec.initSeed).
  console.log(`  DECISION-0 PRNG SEED (use this in the Rust pin): ${String(b.prng.getSeed())}`);
  for (let dno = 0; dno < nDec && !b.ended; dno++) {
    const lenB = log.length;
    const c1 = plan1[dno % plan1.length];
    const c2 = plan2[dno % plan2.length];
    if (c1) { try { streams.omniscient.write(`>p1 ${c1}`); } catch (e) {} }
    if (c2) { try { streams.omniscient.write(`>p2 ${c2}`); } catch (e) {} }
    for (let i = 0; i < 16; i++) await tick();
    const a1 = b.sides[0].active[0], a2 = b.sides[1].active[0];
    const miss = log.slice(lenB).some((l) => l.split('|')[1] === '-miss');
    console.log(`  dec ${dno}: seedAfter=${String(b.prng.getSeed())} ` +
      `p1(${a1 ? a1.hp + '/' + a1.maxhp : 'fnt'}) p2(${a2 ? a2.hp + '/' + a2.maxhp : 'fnt'}) ` +
      `miss=${miss} p1status=${a1 ? (a1.status || '-') : '-'} p2status=${a2 ? (a2.status || '-') : '-'}`);
  }
  console.log(`  ended=${b.ended} winner=${b.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // AC1 — the acc-STAGE flip (the fuzzer's cluster). Mud-Slap (100% acc-drop) lowers a
  // foe whose acc-80 Cross Chop then misses. Pick a seed + decision where a MISS occurs.
  await run('AC1 mudslap_stage_flip',
    mon('Dugtrio', ['mudslap', 'rest'], { nature: 'Jolly', evs: { hp: 252, spe: 252 } }),
    mon('Snorlax', ['crosschop', 'rest'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    [7, 11, 23, 45], ['move 1'], ['move 1'], 8);

  // AC2 — Compound Eyes lifts Thunder(70→91): a hit that a raw-70 roll would sometimes miss.
  await run('AC2 compoundeyes',
    mon('Butterfree', ['thunder', 'rest'], { ability: 'Compound Eyes', nature: 'Modest', evs: { hp: 252, spa: 252 } }),
    mon('Blissey', ['softboiled', 'seismictoss'], { nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    [3, 5, 7, 9], ['move 1'], ['move 2', 'move 1'], 6);

  // AC3 — Bright Powder drops the attacker's Cross Chop 80→72: an extra miss.
  await run('AC3 brightpowder',
    mon('Tauros', ['crosschop', 'rest'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    mon('Snorlax', ['bodyslam', 'rest'], { item: 'brightpowder', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    [13, 17, 19, 23], ['move 1'], ['move 1'], 6);

  // AC4 — Hustle: the Atk ×1.5 (dmgMod) + acc ×0.8 (accMod). Delibird Cross Chop into Snorlax.
  await run('AC4 hustle',
    mon('Delibird', ['crosschop', 'rest'], { ability: 'Hustle', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    mon('Snorlax', ['bodyslam', 'rest'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    [29, 31, 37, 41], ['move 1'], ['move 1'], 6);

  console.log('\n=== PINS PROBE COMPLETE ===');
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
