// probe_accuracy_stage_pin.js — find a DETERMINISTIC acc-stage-flip pin for AC1.
// Pre-sets the attacker's accuracy stage to a strong drop, then searches seeds for a
// decision where the dropped-accuracy move MISSES (Body Slam acc-100 → 100×3/9=33% at
// stage −6). The Rust pin sets the same stage and asserts the same seed/HP; reverting the
// stage fold makes the raw-100 roll HIT → the seed/HP flips.
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

// p1 slow bulky wall (Snorlax) that just Rests; p2 attacker (Snorlax Body Slam) whose
// accuracy we pre-drop to -6. We want a decision where p2's Body Slam MISSES.
async function trySeed(seed, stage) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([
    mon('Miltank', ['recover', 'bodyslam'], { nature: 'Bold', evs: { hp: 252, def: 252 } })]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([
    mon('Snorlax', ['bodyslam', 'rest'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })]) })}`);
  for (let i = 0; i < 10; i++) await tick();
  const b = stream.battle;
  // Pre-set p2's accuracy stage.
  b.sides[1].active[0].boosts.accuracy = stage;
  const initSeed = String(b.prng.getSeed());
  // One decision: p1 Recover (draw-free heal), p2 Body Slam (rolls its dropped accuracy).
  const lenB = log.length;
  streams.omniscient.write('>p1 move 1');
  streams.omniscient.write('>p2 move 1');
  for (let i = 0; i < 16; i++) await tick();
  const miss = log.slice(lenB).some((l) => l.split('|')[1] === '-miss');
  const seedAfter = String(b.prng.getSeed());
  const a1 = b.sides[0].active[0];
  try { streams.omniscient.destroy(); } catch (e) {}
  return { miss, initSeed, seedAfter, p1hp: a1 ? a1.hp : 0, p1max: a1 ? a1.maxhp : 0 };
}

async function main() {
  // Search a handful of seeds at stage -6 (Body Slam 100 → 33%) for a MISS.
  const stage = -6;
  for (let i = 1; i <= 40; i++) {
    const seed = [i * 101 % 65535 || 1, i * 211 % 65535 || 1, i * 307 % 65535 || 1, i * 401 % 65535 || 1];
    const r = await trySeed(seed, stage);
    if (r.miss) {
      console.log(`FOUND miss at seed=${seed.join(',')} stage=${stage}`);
      console.log(`  DECISION-0 PRNG SEED (Rust pin start): ${r.initSeed}`);
      console.log(`  seedAfter=${r.seedAfter} p1(Miltank)=${r.p1hp}/${r.p1max} (Recover full — p2 Body Slam MISSED)`);
      // Also record the NON-dropped (stage 0) result at the SAME seed to prove the flip.
      const r0 = await trySeed(seed, 0);
      console.log(`  CONTROL stage=0: miss=${r0.miss} seedAfter=${r0.seedAfter} p1=${r0.p1hp}/${r0.p1max} (raw-100 → the flip)`);
      return;
    }
  }
  console.log('no miss found in the search window — widen it');
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
