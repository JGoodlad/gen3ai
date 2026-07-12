// probe_nickname_perside.js — capture the PER-SIDE p1 stream (getPlayerStreams) +
// the post-`>start` construction seed for the NICKNAMED-Zapdos scenario, so the Rust
// `bridge_test.rs` nickname pin can assert byte-equality of the ident tokens against
// the real sim. Prints the p1 chunks (the owner sees exact HP) filtered to the ident-
// bearing lines, plus the pre-first-decision (post-`>start`) prng seed.
'use strict';

const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
function mon(species, name, ability, moves, gender) {
  return { species, name: name || '', item: '', ability, moves, evs: {}, ivs: IV31,
           nature: 'Serious', level: 100, gender: gender || 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function main() {
  const p1 = [mon('Zapdos', 'Electhor', 'Pressure', ['thunderbolt', 'roar'], 'N')];
  const p2 = [mon('Snorlax', '', 'Immunity', ['bodyslam', 'splash'], 'M'),
              mon('Regice', '', 'Clear Body', ['icebeam', 'splash'], 'N')];
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const p1chunks = [];
  (async () => { for await (const c of streams.p1) p1chunks.push(c); })();
  (async () => { for await (const c of streams.p2) void c; })();
  (async () => { for await (const c of streams.omniscient) void c; })();

  const packed1 = Teams.pack(p1);
  const packed2 = Teams.pack(p2);
  streams.omniscient.write(
    `>start {"formatid":"gen3customgame","seed":[7,11,13,17]}\n` +
    `>player p1 {"name":"P1","team":"${packed1}"}\n` +
    `>player p2 {"name":"P2","team":"${packed2}"}`);
  await tick(); await tick();
  // The post-`>start` seed (what run_full_battle_bridge seeds via advance_seed_for_construction).
  console.log('CONSTRUCTION_SEED=' + JSON.stringify(stream.battle.prng.getSeed()));
  streams.omniscient.write('>p1 move 1\n>p2 move 1');
  await tick(); await tick();

  const lines = p1chunks.join('\n').split('\n');
  console.log('=== p1 per-side ident-bearing lines ===');
  for (const l of lines) {
    if (l.startsWith('|switch|') || l.startsWith('|move|')) console.log(JSON.stringify(l));
  }
}
main().then(() => process.exit(0));
