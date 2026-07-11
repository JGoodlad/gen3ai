// probe_plus_minus_gen3.js — settle the gen3 PLUS / MINUS model bit-for-bit (the A/B
// fuzzer's thunderbolt-vs-Plusle/Minun STATE cluster). The RESOLVED gen3 sim is the
// ONLY oracle. The old NOOP classification assumed "partner-less in singles → no-op",
// which never tested the OPPOSING active carrying the paired ability.
//
// Questions (each answered by direct measurement):
//  (a) Does MINUS boost the holder's SpA when the FOE active has PLUS (gen3 singles)?
//      And the mirror (Plus holder vs Minus foe)?
//  (b) Same-ability pairing — Plus vs Plus / Minus vs Minus: boost or not?
//  (c) Magnitude + fold point — ×1.5? Special only (a physical move unboosted)?
//  (d) Is the boost DRAW-FREE (post-turn seed identical to a control)?
//  (e) Does it react to the foe LEAVING (boost only while the partner is active)?
//
// Run:  node src/rust_sim/harness/probe_plus_minus_gen3.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31,
    nature: opts.nature || 'Hardy', level: opts.level || 100, gender: opts.gender || 'M',
  };
}

// One turn: p1 attacker uses moveId into p2 defender (splash). Returns delta/crit/seed.
function runOneTurn({ seed, attacker, defender, moveId, benchA, benchB }) {
  const battle = new Battle({ formatid: 'gen3customgame', seed });
  battle.setPlayer('p1', { name: 'A', team: Teams.pack([attacker, ...(benchA || [])]) });
  battle.setPlayer('p2', { name: 'B', team: Teams.pack([defender, ...(benchB || [])]) });
  const def = battle.sides[1].active[0];
  const hpBefore = def.hp;
  const logStart = battle.log.length;
  battle.choose('p1', 'move ' + moveId);
  battle.choose('p2', 'move splash');
  const turnLog = battle.log.slice(logStart).join('\n');
  return {
    delta: hpBefore - def.hp,
    crit: turnLog.includes('|-crit|'),
    seedAfter: String(battle.prng.getSeed()),
  };
}

function maxRoll(cfg, nSeeds = 400) {
  const deltas = new Set();
  for (let s = 1; s <= nSeeds; s++) {
    const r = runOneTurn({ ...cfg, seed: [0, 0, 0, s] });
    if (!r.crit && r.delta > 0) deltas.add(r.delta);
  }
  const sorted = [...deltas].sort((a, b) => b - a);
  return { max: sorted[0], nDistinct: sorted.length };
}

function main() {
  const abil = Dex.mod('gen3').abilities.get('minus');
  console.log('resolved gen3 minus handlers:', Object.keys(abil).filter(k => k.startsWith('on')).join(','));

  const tbolt = 'thunderbolt';
  const cases = [
    ['minus attacker vs PLUS foe', mon('Minun', [tbolt, 'splash'], { ability: 'Minus' }), mon('Plusle', ['splash'], { ability: 'Plus' })],
    ['minus attacker vs NOOP foe', mon('Minun', [tbolt, 'splash'], { ability: 'Minus' }), mon('Plusle', ['splash'], { ability: 'Sturdy' })],
    ['plus attacker vs MINUS foe', mon('Plusle', [tbolt, 'splash'], { ability: 'Plus' }), mon('Minun', ['splash'], { ability: 'Minus' })],
    ['plus attacker vs PLUS foe', mon('Plusle', [tbolt, 'splash'], { ability: 'Plus' }), mon('Minun', ['splash'], { ability: 'Plus' })],
    ['minus attacker vs MINUS foe', mon('Minun', [tbolt, 'splash'], { ability: 'Minus' }), mon('Plusle', ['splash'], { ability: 'Minus' })],
  ];
  for (const [name, a, d] of cases) {
    const r = maxRoll({ attacker: a, defender: d, moveId: 1 });
    console.log(`${name}: maxRoll=${r.max} (${r.nDistinct} rolls)`);
  }

  // (c) physical control: quick attack from Minus vs Plus foe / vs noop foe.
  const p1 = maxRoll({ attacker: mon('Minun', ['quickattack', 'splash'], { ability: 'Minus' }), defender: mon('Plusle', ['splash'], { ability: 'Plus' }), moveId: 1 });
  const p2 = maxRoll({ attacker: mon('Minun', ['quickattack', 'splash'], { ability: 'Minus' }), defender: mon('Plusle', ['splash'], { ability: 'Sturdy' }), moveId: 1 });
  console.log(`physical quickattack: vs Plus foe=${p1.max} vs noop foe=${p2.max} (expect EQUAL — SpA-only)`);

  // (d) draw-freeness: same seed, boost vs control — post-turn seed identical?
  const s1 = runOneTurn({ seed: [0, 0, 0, 7], attacker: mon('Minun', [tbolt, 'splash'], { ability: 'Minus' }), defender: mon('Plusle', ['splash'], { ability: 'Plus' }), moveId: 1 });
  const s2 = runOneTurn({ seed: [0, 0, 0, 7], attacker: mon('Minun', [tbolt, 'splash'], { ability: 'Minus' }), defender: mon('Plusle', ['splash'], { ability: 'Sturdy' }), moveId: 1 });
  console.log(`draw-free check: boosted seedAfter=${s1.seedAfter} control=${s2.seedAfter} identical=${s1.seedAfter === s2.seedAfter}`);

  // (e) foe leaves: Plusle switches out to a non-partner -> Minun's NEXT tbolt unboosted?
  {
    const battle = new Battle({ formatid: 'gen3customgame', seed: [0, 0, 0, 9] });
    battle.setPlayer('p1', { name: 'A', team: Teams.pack([mon('Minun', [tbolt, 'splash'], { ability: 'Minus' })]) });
    battle.setPlayer('p2', {
      name: 'B', team: Teams.pack([
        mon('Plusle', ['splash'], { ability: 'Plus', evs: { hp: 252 } }),
        mon('Snorlax', ['splash'], { ability: 'Thick Fat' }),
      ]),
    });
    const spaBoosted = battle.sides[0].active[0].getStat('spa');
    battle.choose('p1', 'move splash');
    battle.choose('p2', 'switch 2');
    const spaAfter = battle.sides[0].active[0].getStat('spa');
    console.log(`live getStat(spa): partner-active=${spaBoosted} partner-benched=${spaAfter} (expect drop by /1.5)`);
  }
}
main();
