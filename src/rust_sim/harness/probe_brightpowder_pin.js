// probe_brightpowder_pin.js — find a DECISIVE Bright Powder pin for AC3.
// A Cross Chop (acc 80) into a Bright Powder holder rolls at 80×0.9=72. We want a seed where
// the roll is in [72,80): it HITS at 80 (Bright Powder reverted) but MISSES at 72 (the mod).
// The Rust pin asserts the MISS (defender full); a revert of the accMod flips it to a hit.
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

// p1 Tauros Cross Chop into p2 Miltank (Bright Powder). p2 just Recovers (draw-free). We hook
// randomChance(_,100) to read the roll number, so we know if the flip window applies.
async function trySeed(seed, withItem) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([
    mon('Tauros', ['crosschop', 'rest'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([
    mon('Miltank', ['recover', 'bodyslam'], { item: withItem ? 'brightpowder' : '', nature: 'Bold', evs: { hp: 252, def: 252 } })]) })}`);
  for (let i = 0; i < 10; i++) await tick();
  const b = stream.battle;
  const initSeed = String(b.prng.getSeed());
  const rolls = [];
  const realRC = b.prng.randomChance.bind(b.prng);
  b.prng.randomChance = function (n, den) { const r = realRC(n, den); if (den === 100) rolls.push(n); return r; };
  const lenB = log.length;
  streams.omniscient.write('>p1 move 1'); // Cross Chop
  streams.omniscient.write('>p2 move 1'); // Recover
  for (let i = 0; i < 16; i++) await tick();
  const miss = log.slice(lenB).some((l) => l.split('|')[1] === '-miss');
  const a2 = b.sides[1].active[0];
  const seedAfter = String(b.prng.getSeed());
  try { streams.omniscient.destroy(); } catch (e) {}
  return { miss, initSeed, seedAfter, effAcc: rolls[0], p2hp: a2 ? a2.hp : 0, p2max: a2 ? a2.maxhp : 0 };
}

async function main() {
  for (let i = 1; i <= 120; i++) {
    const seed = [i * 137 % 65535 || 1, i * 251 % 65535 || 1, i * 331 % 65535 || 1, i * 449 % 65535 || 1];
    const rItem = await trySeed(seed, true);   // Bright Powder → 72%
    const rNone = await trySeed(seed, false);  // no item → 80%
    // The FLIP: with Bright Powder it MISSES, without it HITS (the roll ∈ [72,80)).
    if (rItem.miss && !rNone.miss) {
      console.log(`FOUND flip at seed=${seed.join(',')}`);
      console.log(`  DECISION-0 PRNG SEED (Rust pin start): ${rItem.initSeed}`);
      console.log(`  WITH Bright Powder (effAcc=${rItem.effAcc}): MISS, seedAfter=${rItem.seedAfter} p2(Miltank)=${rItem.p2hp}/${rItem.p2max}`);
      console.log(`  WITHOUT (revert, effAcc=${rNone.effAcc}): HIT, seedAfter=${rNone.seedAfter} p2=${rNone.p2hp}/${rNone.p2max}`);
      return;
    }
  }
  console.log('no flip found — widen the window');
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
