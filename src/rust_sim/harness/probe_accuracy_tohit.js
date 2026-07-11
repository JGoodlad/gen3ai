// probe_accuracy_tohit.js — SETTLE the gen3 to-hit computation bit-for-bit.
//
// Wraps battle.runEvent ('ModifyBoost'/'ModifyAccuracy'/'Accuracy') and
// prng.randomChance to capture, for a given move + acc/eva boost stages + acc
// item/ability in play, the EXACT effective accuracy that reaches the single
// randomChance(effAcc, 100) draw. We instrument, not assume — this is the ONLY
// oracle (the mod-chain law). Confirms:
//   1. the acc/eva STAGE TABLE (gen3 [1, 4/3, 5/3, 2, 7/3, 8/3, 3], the 3/3-base
//      ±6 form) + the exact float math + the runEvent integer-guard.
//   2. the MODIFIER ORDER: inline stages → ModifyAccuracy (item+ability) →
//      Accuracy — and the RESOLVED multiplier of brightpowder/laxincense/
//      compoundeyes/sandveil/hustle.
//   3. that the draw is ONE randomChance(effAcc,100) and stage/item/ability
//      mods are DRAW-FREE.
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
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// ---- run ONE scenario and capture the accuracy pipeline for the p1 move ----
async function scenario(desc, p1mon, p2mon, p1move, p2move, setup) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) {} })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([p1mon]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([p2mon]) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  // Force boosts directly on the mons (setup applies the acc/eva stages).
  if (setup) setup(battle);

  let logging = false;
  const captured = [];
  const realRun = battle.runEvent.bind(battle);
  battle.runEvent = function (eventid, target, source, effect, relayVar, ...rest) {
    const inV = relayVar;
    const out = realRun(eventid, target, source, effect, relayVar, ...rest);
    if (logging && (eventid === 'ModifyAccuracy' || eventid === 'Accuracy' || eventid === 'ModifyBoost')) {
      const mv = effect && effect.id ? effect.id : (effect || '');
      captured.push({ ev: eventid, move: mv, in: inV, out });
    }
    return out;
  };
  // Capture the randomChance draw args (the effAcc that reaches the roll).
  const rc = [];
  const realRC = battle.prng.randomChance.bind(battle.prng);
  battle.prng.randomChance = function (num, den) {
    const res = realRC(num, den);
    if (logging) rc.push({ num, den, res });
    return res;
  };
  logging = true;
  streams.omniscient.write(`>p1 move ${p1move}`);
  streams.omniscient.write(`>p2 move ${p2move}`);
  for (let k = 0; k < 20; k++) await tick();
  logging = false;

  console.log(`\n### ${desc}`);
  for (const c of captured) {
    console.log(`  runEvent('${c.ev}') move=${c.move} in=${JSON.stringify(c.in)} out=${JSON.stringify(c.out)}`);
  }
  for (const r of rc) {
    console.log(`  randomChance(${r.num}, ${r.den}) -> ${r.res}`);
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

function setBoost(battle, side, key, val) {
  // side 0 = p1, 1 = p2
  battle.sides[side].active[0].boosts[key] = val;
}

async function main() {
  console.log('=== gen3 to-hit PROBE: stage table, mod order, draw position ===');
  console.log('boostTable = [1, 4/3, 5/3, 2, 7/3, 8/3, 3] (gen3 tryMoveHit)');
  console.log('acc *= boostTable[+accStage] or /= boostTable[-accStage]');
  console.log('acc /= boostTable[+evaStage] or *= boostTable[-evaStage]');
  console.log('runEvent final modify() applies ONLY if relayVar is a non-neg INTEGER (battle.ts:929)');

  // Attacker: a plain acc-100 physical move (Body Slam has a secondary; use Tackle acc-100).
  // Use Return? bp varies. Use a clean acc-100 physical: "tackle".  Defender inert.
  const atkNoItem = (species, moves, opts) => mon(species, moves, opts);

  // 1) BASELINE no stages, no mods: acc 100 → randomChance(100,100)
  await scenario('baseline acc100 no-stage no-mod (Tackle)',
    atkNoItem('Tauros', ['tackle'], {}),
    atkNoItem('Snorlax', ['tackle'], { evs: { hp: 252 } }),
    1, 1, null);

  // 2) attacker accuracy stage -1: acc 100 /= (4/3) = 75
  await scenario('attacker acc stage -1 (100 / (4/3) = 75)',
    atkNoItem('Tauros', ['tackle'], {}),
    atkNoItem('Snorlax', ['tackle'], { evs: { hp: 252 } }),
    1, 1, (b) => setBoost(b, 0, 'accuracy', -1));

  // 3) attacker accuracy stage -2: acc 100 /= (5/3) = 60
  await scenario('attacker acc stage -2 (100 / (5/3) = 60)',
    atkNoItem('Tauros', ['tackle'], {}),
    atkNoItem('Snorlax', ['tackle'], { evs: { hp: 252 } }),
    1, 1, (b) => setBoost(b, 0, 'accuracy', -2));

  // 4) target evasion stage +1: acc 100 /= (4/3) = 75  (Sand-Attack raises... no, Sand-Attack LOWERS foe acc)
  await scenario('target eva stage +1 (100 / (4/3) = 75)',
    atkNoItem('Tauros', ['tackle'], {}),
    atkNoItem('Snorlax', ['tackle'], { evs: { hp: 252 } }),
    1, 1, (b) => setBoost(b, 1, 'evasion', 1));

  // 5) target evasion stage -1: acc 100 *= (4/3) = 133.33
  await scenario('target eva stage -1 (100 * (4/3) = 133.33)',
    atkNoItem('Tauros', ['tackle'], {}),
    atkNoItem('Snorlax', ['tackle'], { evs: { hp: 252 } }),
    1, 1, (b) => setBoost(b, 1, 'evasion', -1));

  // 6) attacker acc stage +1: acc 100 *= (4/3) = 133.33
  await scenario('attacker acc stage +1 (100 * (4/3) = 133.33)',
    atkNoItem('Tauros', ['tackle'], {}),
    atkNoItem('Snorlax', ['tackle'], { evs: { hp: 252 } }),
    1, 1, (b) => setBoost(b, 0, 'accuracy', 1));

  // 7) a shaky move (Thunder acc 70) + no stages -> baseline for items
  await scenario('Thunder acc70 no-mod',
    atkNoItem('Raikou', ['thunder'], {}),
    atkNoItem('Snorlax', ['tackle'], { evs: { hp: 252 } }),
    1, 1, null);

  // ---- ACC ITEMS / ABILITIES ----
  // 8) Bright Powder on DEFENDER (chainModify [3686,4096]) vs Thunder(70): expect floor
  await scenario('Thunder70 vs Bright Powder defender (chainModify 3686/4096)',
    atkNoItem('Raikou', ['thunder'], {}),
    atkNoItem('Snorlax', ['tackle'], { item: 'brightpowder', evs: { hp: 252 } }),
    1, 1, null);

  // 9) Bright Powder on DEFENDER vs Tackle(100): 100 -> 90
  await scenario('Tackle100 vs Bright Powder defender (expect 90)',
    atkNoItem('Tauros', ['tackle'], {}),
    atkNoItem('Snorlax', ['tackle'], { item: 'brightpowder', evs: { hp: 252 } }),
    1, 1, null);

  // 10) Lax Incense on DEFENDER (gen3: accuracy * 0.95) vs Tackle(100): 100 -> ?
  await scenario('Tackle100 vs Lax Incense defender (gen3 *0.95)',
    atkNoItem('Tauros', ['tackle'], {}),
    atkNoItem('Snorlax', ['tackle'], { item: 'laxincense', evs: { hp: 252 } }),
    1, 1, null);

  // 11) Compound Eyes on ATTACKER (chainModify [5325,4096] ~1.3) vs Thunder(70): 70 -> ?
  await scenario('Thunder70 + Compound Eyes attacker (expect ~91)',
    atkNoItem('Butterfree', ['thunder'], { ability: 'Compound Eyes' }),
    atkNoItem('Snorlax', ['tackle'], { evs: { hp: 252 } }),
    1, 1, null);

  // 12) Compound Eyes on ATTACKER vs Tackle(100): 100 -> 130 (>100 always hits)
  await scenario('Tackle100 + Compound Eyes attacker (expect 130)',
    atkNoItem('Butterfree', ['tackle'], { ability: 'Compound Eyes' }),
    atkNoItem('Snorlax', ['tackle'], { evs: { hp: 252 } }),
    1, 1, null);

  // 13) Sand Veil on DEFENDER in sandstorm (chainModify [3277,4096] ~0.8) vs Tackle(100)
  await scenario('Tackle100 vs Sand Veil defender in sand (expect 80)',
    atkNoItem('Tauros', ['tackle'], {}),
    atkNoItem('Cacturne', ['tackle'], { ability: 'Sand Veil', evs: { hp: 252 } }),
    1, 1, (b) => { b.field.setWeather('sandstorm', b.sides[0].active[0]); });

  // 14) Sand Veil on DEFENDER NOT in sand -> no effect (100)
  await scenario('Tackle100 vs Sand Veil defender NO sand (expect 100)',
    atkNoItem('Tauros', ['tackle'], {}),
    atkNoItem('Cacturne', ['tackle'], { ability: 'Sand Veil', evs: { hp: 252 } }),
    1, 1, null);

  // 15) Hustle acc side: a PHYSICAL-TYPE move (Tackle=Normal in the gen3 list) -> 0.8
  await scenario('Tackle100 (Normal=physical-type) + Hustle attacker (expect 80)',
    atkNoItem('Snorlax', ['tackle'], { ability: 'Hustle' }),
    atkNoItem('Snorlax', ['tackle'], { evs: { hp: 252 } }),
    1, 1, null);

  // 16) Hustle acc side: a NON-physical-TYPE move (Thunder=Electric) -> unaffected
  await scenario('Thunder70 (Electric NOT in physical-type list) + Hustle attacker (expect 70)',
    atkNoItem('Ampharos', ['thunder'], { ability: 'Hustle' }),
    atkNoItem('Snorlax', ['tackle'], { evs: { hp: 252 } }),
    1, 1, null);

  // 17) STACK: target eva +1 (float 133.33) THEN Bright Powder (chainModify) —
  //     does the non-integer skip chainModify? (the mod-chain subtlety)
  await scenario('eva+1 float(133.33) THEN Bright Powder chainModify — INTEGER GUARD test',
    atkNoItem('Tauros', ['tackle'], {}),
    atkNoItem('Snorlax', ['tackle'], { item: 'brightpowder', evs: { hp: 252 } }),
    1, 1, (b) => setBoost(b, 1, 'evasion', -1));

  // 18) STACK: target eva +1 (float 133.33) THEN Lax Incense (direct *0.95) —
  //     direct multiply always applies
  await scenario('eva+1 float(133.33) THEN Lax Incense *0.95 (direct, always applies)',
    atkNoItem('Tauros', ['tackle'], {}),
    atkNoItem('Snorlax', ['tackle'], { item: 'laxincense', evs: { hp: 252 } }),
    1, 1, (b) => setBoost(b, 1, 'evasion', -1));

  // 19) STACK: acc stage that keeps integer (acc -3 -> 100/2=50) THEN Bright Powder (int -> chainModify applies)
  await scenario('acc-3 -> 50 (integer) THEN Bright Powder (int guard PASSES -> 45)',
    atkNoItem('Tauros', ['tackle'], {}),
    atkNoItem('Snorlax', ['tackle'], { item: 'brightpowder', evs: { hp: 252 } }),
    1, 1, (b) => setBoost(b, 0, 'accuracy', -3));

  console.log('\n=== PROBE COMPLETE ===');
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
