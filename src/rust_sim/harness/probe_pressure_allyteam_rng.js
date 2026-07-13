// probe_pressure_allyteam_rng.js — GROUND TRUTH for the gen3_pressure_allyteam_v1
// regression pin (the e2e_182 root cause): an `allyTeam` move (Aromatherapy / Heal Bell)
// under a Pressure foe deducts ONE PP, NOT two (the Pressure extra fires only when the
// Pressure foe is in the move's `pressureTargets` — a foe-directed target). Captures the
// post-turn seed + the Blissey moveSlot PP for a constructed gen3customgame board so the
// Rust pin asserts bit-for-bit.
//
// Usage: node harness/probe_pressure_allyteam_rng.js   (run from src/rust_sim)
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(PS + '/dist/sim/battle-stream');
const tick = () => new Promise(r => setImmediate(r));

// p1 Blissey (Aromatherapy=allyTeam slot0, ThunderWave=foe slot1); p2 Zapdos with Pressure.
// (Aromatherapy needs a statused ally to "do something", but the PP deduct happens on ANY
//  use — Blissey burns itself is not needed; the deduct is unconditional post-BeforeMove.)
const P1 = 'Blissey||Leftovers|NaturalCure|Aromatherapy,ThunderWave,SeismicToss,SoftBoiled|Bold|252,,252,,4,|F||||';
const P2 = 'Zapdos||Leftovers|Pressure|Thunderbolt,Roost,Rest,ThunderWave|Modest|252,,,252,,|N||||';
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
  const bliss = b.sides[0].active[0];
  const pp = () => bliss.moveSlots.map(s => s.pp).join(',');
  console.log('SEED_BEFORE', b.prng.getSeed());
  console.log('PP init (aroma,twave,stoss,sboiled)', pp());

  // dec0: p1 Aromatherapy (allyTeam) into Pressure Zapdos; p2 Rest (self)
  streams.omniscient.write('>p1 move 1');
  streams.omniscient.write('>p2 move 3');
  for (let i = 0; i < 12; i++) await tick();
  console.log('dec0 (Aromatherapy under Pressure foe): PP', pp(), '  SEED_AFTER', b.prng.getSeed());

  // dec1: p1 ThunderWave (foe move) into Pressure Zapdos; p2 Rest (self)
  streams.omniscient.write('>p1 move 2');
  streams.omniscient.write('>p2 move 3');
  for (let i = 0; i < 12; i++) await tick();
  console.log('dec1 (ThunderWave into Pressure foe): PP', pp(), '  SEED_AFTER', b.prng.getSeed());
  try { streams.omniscient.destroy(); } catch (e) {}
})().catch(e => { console.error(e); process.exit(1); });
