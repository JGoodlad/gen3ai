// probe_pressure_curse_regression_rng.js — GROUND TRUTH for the gen3_pressure_allyteam_v1
// D1 (byte-fuzz 5_6) regression pin: a NON-GHOST user's Curse is RE-TARGETED to `self`
// at runtime (curse.onModifyMove -> nonGhostTarget), so under a Pressure foe it deducts
// ONE PP, not two (the static dex target is `normal`, but the RUNTIME target is `self`
// -> the foe is NOT in the move's pressureTargets). WRONG (pre-fix pressure_targets_foe
// reading the static `normal`): 2 PP/turn -> Curse drains ~1 cycle early -> Struggle turns
// the sim still Curses -> the deep seed desync. Captures Curse PP + the post-turn seed for
// a constructed gen3customgame board so the Rust pin asserts bit-for-bit.
//
// Usage: node harness/probe_pressure_curse_regression_rng.js   (run from src/rust_sim)
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(PS + '/dist/sim/battle-stream');
const tick = () => new Promise(r => setImmediate(r));

// p1 Swampert (NON-GHOST): Curse=slot0 (16 PP), fillers. p2 Zapdos with Pressure + Agility
// (a self-boost, draw-free, keeps Zapdos faster than the Curse-slowed Swampert -> no ties).
const P1 = 'Swampert||Leftovers|Torrent|Curse,Surf,Earthquake,IcePunch|Relaxed|252,,252,,4,|N||||';
const P2 = 'Zapdos||Leftovers|Pressure|Agility,Thunderbolt,Rest,ThunderWave|Modest|252,,,252,,|N||||';
const SEED = [30982, 33910, 19571, 50263];

(async () => {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const _ of streams.omniscient) {} })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(SEED)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'A', team: P1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'B', team: P2 })}`);
  for (let i = 0; i < 10; i++) await tick();
  const b = stream.battle;
  const swamp = b.sides[0].active[0];
  const cursepp = () => swamp.moveSlots[0].pp;
  console.log('SEED_BEFORE', JSON.stringify(b.prng.getSeed()));
  console.log('Curse PP init', cursepp());

  for (let dec = 0; dec < 9; dec++) {
    streams.omniscient.write('>p1 move 1'); // Curse
    streams.omniscient.write('>p2 move 1'); // Agility (self, draw-free)
    for (let i = 0; i < 14; i++) await tick();
    console.log(`dec${dec}: CursePP ${cursepp()}  atkBoost ${swamp.boosts.atk}  SEED_AFTER ${JSON.stringify(b.prng.getSeed())}`);
  }
  try { streams.omniscient.destroy(); } catch (e) {}
})().catch(e => { console.error(e); process.exit(1); });
