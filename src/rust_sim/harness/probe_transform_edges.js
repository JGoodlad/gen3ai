// probe_transform_edges.js — ROUND 33 (TRANSFORM) live probe vs the omniscient gen3 sim.
//
// Re-verifies the BATCH89_RESEARCH Transform notes and settles the questions that note did
// NOT answer:
//   A  the FULL copy inventory incl. the STALE `pokemon.speed` cache (setSpecies sets speed
//      from spreadModify(TARGET base, OWN set) and transformInto then overwrites storedStats
//      WITHOUT re-setting speed) + `details` (never refreshed) + baseStoredStats.
//   B  the copied slots' pp/maxpp when the TRANSFORMER has fewer than 4 moves (`this.ppUps[i]
//      || 0`).
//   C  the PER-SIDE `|request|` bytes for a transformed mon (roster details/stats/moves +
//      active moves incl. the `target` key Mimic omits).
//   D  revert on switch-out / faint.
//   E-… the interaction edges (Mimic both ways, Substitute, Ditto mirror, Hidden Power,
//      Baton Pass, choice lock, ability End).
//
// Run: node harness/probe_transform_edges.js
'use strict';
const path = require('path');
const { mon, run, fmtCalls } = require('./probe_batch4_lib');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const SEED = [5, 4, 3, 2];
const KEEP = (l) => l && !l.startsWith('|t:|') && l !== '|' && !l.startsWith('|upkeep');
const tick = () => new Promise((r) => setTimeout(r, 0));

function dump(u) {
  return {
    name: u.name, fullname: u.fullname, details: u.details,
    species: u.species.id, transformed: u.transformed, weighthg: u.weighthg,
    types: u.types, addedType: u.addedType,
    hp: `${u.hp}/${u.maxhp}`, baseMaxhp: u.baseMaxhp,
    stored: u.storedStats, baseStored: u.baseStoredStats,
    speedCache: u.speed,
    ability: u.ability, baseAbility: u.baseAbility,
    item: u.item, status: u.status, boosts: u.boosts,
    hpType: u.hpType, hpPower: u.hpPower, baseHpType: u.baseHpType, baseHpPower: u.baseHpPower,
    ppUps: u.ppUps,
    slots: u.moveSlots.map((m) => ({ move: m.move, id: m.id, pp: m.pp, maxpp: m.maxpp, target: m.target, virtual: m.virtual, used: m.used, disabled: m.disabled })),
    baseSlots: u.baseMoveSlots.map((m) => `${m.id}:${m.pp}/${m.maxpp}`),
    vol: Object.keys(u.volatiles),
    movesGetter: u.moves,
  };
}

function showDec(tag, r) {
  r.perDecision.forEach((d, i) => {
    console.log(`  ${tag} d${i}: draws=${d.nexts} calls=[${fmtCalls(d.calls)}]`);
    console.log(`        lines=${JSON.stringify(d.lines.filter(KEEP))}`);
  });
}

// ---------------------------------------------------------------- per-side request runner
async function perSide(teams, seed, choices, fmt = 'gen3customgame') {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const p1 = [], omni = [];
  (async () => { for await (const c of streams.p1) for (const l of String(c).split('\n')) p1.push(l); })();
  (async () => { for await (const c of streams.p2) void c; })();
  (async () => { for await (const c of streams.omniscient) for (const l of String(c).split('\n')) omni.push(l); })();
  streams.omniscient.write(`>start {"formatid":"${fmt}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(teams[0]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(teams[1]) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const reqs = [];
  let lo = 0;
  const grab = () => { for (; lo < p1.length; lo++) if (p1[lo].startsWith('|request|')) reqs.push(p1[lo].slice(9)); };
  grab();
  for (const [c1, c2] of choices) {
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 14; k++) await tick();
    grab();
    if (stream.battle.ended) break;
  }
  return { reqs, battle: stream.battle, omni };
}

async function main() {
  // ================================================================= A. copy inventory
  console.log('############ A. copy inventory (Ditto set != target set, so a stale speed shows) ############');
  {
    const teams = [
      // Ditto: Serious, 0 EVs, 31 IVs.  Target Snorlax: Adamant, 252 spe EVs, 0 spe IV.
      [mon('Ditto', ['transform', 'splash'], { ability: 'Limber', item: 'Metal Powder' }),
       mon('Gengar', ['splash'], { ability: 'Levitate' })],
      [mon('Snorlax', ['swordsdance', 'bodyslam', 'rest', 'splash'],
        { item: 'Leftovers', ability: 'Thick Fat', nature: 'Adamant',
          evs: { hp: 252, atk: 252, spe: 4 }, ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 3 } })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1'], ['move 2', 'move 4']],
      { onBoundary: (b) => ({ p1: dump(b.sides[0].active[0]), p2spe: b.sides[1].active[0].storedStats.spe, p2speed: b.sides[1].active[0].speed }) });
    showDec('A', r);
    r.states.forEach((s, i) => console.log(`  A state${i} = ${JSON.stringify(s, null, 1)}`));
    const sl = Dex.mod('gen3').species.get('Snorlax');
    console.log(`  Snorlax baseStats=${JSON.stringify(sl.baseStats)}`);
  }

  // ================================================================= B. ppUps indexing
  console.log('\n############ B. copied pp/maxpp vs the TRANSFORMER move count (ppUps[i]||0) ############');
  for (const dittoMoves of [['transform'], ['transform', 'splash'], ['transform', 'splash', 'recover', 'rest']]) {
    const teams = [
      [mon('Ditto', dittoMoves, { ability: 'Limber' })],
      [mon('Snorlax', ['bodyslam', 'swordsdance', 'rest', 'splash'], { ability: 'Thick Fat' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 4']], { onBoundary: (b) => dump(b.sides[0].active[0]) });
    const s = r.states[0];
    console.log(`  ditto moves=${JSON.stringify(dittoMoves)} ppUps=${JSON.stringify(s.ppUps)}`);
    console.log(`     slots=${JSON.stringify(s.slots)}`);
  }

  // ================================================================= C. per-side request
  console.log('\n############ C. PER-SIDE |request| bytes for a transformed mon ############');
  {
    const teams = [
      [mon('Ditto', ['transform'], { ability: 'Limber', item: 'Metal Powder' }),
       mon('Gengar', ['splash'], { ability: 'Levitate' })],
      [mon('Snorlax', ['bodyslam', 'curse', 'hiddenpower', 'splash'],
        { ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 },
          ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 30 } })],
    ];
    const r = await perSide(teams, SEED, [['move 1', 'move 4'], ['move 1', 'move 4'], ['switch 2', 'move 4'], ['switch 2', 'move 4']]);
    r.reqs.forEach((q, i) => console.log(`  req${i}: ${q}`));
    console.log('  omni: ' + JSON.stringify(r.omni.filter(KEEP)));
  }

  // ================================================================= D. revert on faint
  console.log('\n############ D. revert on FAINT (self-KO via copied Explosion) ############');
  {
    const teams = [
      [mon('Ditto', ['transform', 'splash'], { ability: 'Limber' }), mon('Gengar', ['splash'], { ability: 'Levitate' })],
      [mon('Snorlax', ['explosion', 'splash'], { ability: 'Thick Fat' }), mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 2'], ['move 1', 'move 2'], ['switch 2', null]],
      { onBoundary: (b) => ({ p1a: b.sides[0].active[0].species.id, dumps: b.sides[0].pokemon.map((p) => `${p.species.id}:${p.transformed}:${p.moves.join('/')}`) }) });
    showDec('D', r);
    r.states.forEach((s, i) => console.log(`  D state${i}=${JSON.stringify(s)}`));
  }

  // ================================================================= E. Mimic both ways
  console.log('\n############ E1. MIMIC after TRANSFORM (source.transformed ⇒ fail) ############');
  {
    // Ditto knows mimic+transform: transform first, then the COPIED slots have no mimic —
    // so instead give the TARGET mimic, and have the transformed Ditto use the copied Mimic.
    const teams = [
      [mon('Ditto', ['transform', 'splash'], { ability: 'Limber' })],
      [mon('Snorlax', ['mimic', 'bodyslam', 'splash', 'rest'], { ability: 'Thick Fat' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 2'], ['move 1', 'move 2']],
      { onBoundary: (b) => dump(b.sides[0].active[0]) });
    showDec('E1', r);
    r.states.forEach((s, i) => console.log(`  E1 state${i} slots=${JSON.stringify(s.slots)} transformed=${s.transformed}`));
  }
  console.log('\n############ E2. TRANSFORM after MIMIC (moveSlots wholesale; revert on switch) ############');
  {
    const teams = [
      [mon('Ditto', ['mimic', 'transform', 'splash'], { ability: 'Limber' }), mon('Gengar', ['splash'], { ability: 'Levitate' })],
      [mon('Snorlax', ['bodyslam', 'splash'], { ability: 'Thick Fat' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1'], ['move 2', 'move 2'], ['switch 2', 'move 2'], ['switch 2', 'move 2']],
      { onBoundary: (b) => ({ act: dump(b.sides[0].active[0]), d: b.sides[0].pokemon.map((p) => `${p.species.id}:${p.moves.join('/')}`) }) });
    showDec('E2', r);
    r.states.forEach((s, i) => console.log(`  E2 state${i} slots=${JSON.stringify(s.act.slots)} base=${JSON.stringify(s.act.baseSlots)} team=${JSON.stringify(s.d)}`));
  }

  // ================================================================= F. Substitute / mirror / fails
  console.log('\n############ F1. TRANSFORM into a SUBBED target (bypasssub ⇒ succeeds) ############');
  {
    const teams = [
      [mon('Ditto', ['transform', 'splash'], { ability: 'Limber' })],
      [mon('Snorlax', ['substitute', 'splash'], { ability: 'Thick Fat' })],
    ];
    const r = await run(teams, SEED, [['move 2', 'move 1'], ['move 1', 'move 2']],
      { onBoundary: (b) => dump(b.sides[0].active[0]) });
    showDec('F1', r);
    console.log(`  F1 species=${r.states[1].species} transformed=${r.states[1].transformed}`);
  }
  console.log('\n############ F2. DITTO MIRROR: 2nd transform FAILS (target.transformed) ############');
  {
    const teams = [
      [mon('Ditto', ['transform', 'splash'], { ability: 'Limber' })],
      [mon('Ditto', ['transform', 'splash'], { ability: 'Limber' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1'], ['move 1', 'move 1']],
      { onBoundary: (b) => [dump(b.sides[0].active[0]), dump(b.sides[1].active[0])] });
    showDec('F2', r);
    r.states.forEach((s, i) => console.log(`  F2 state${i} p1=${s[0].species}/${s[0].transformed}/${JSON.stringify(s[0].slots.map((m) => m.id))} p2=${s[1].species}/${s[1].transformed}/${JSON.stringify(s[1].slots.map((m) => m.id))}`));
  }
  console.log('\n############ F3. TRANSFORM into a FAINTED target? (target faints first) ############');
  {
    const teams = [
      [mon('Ditto', ['transform', 'splash'], { ability: 'Limber' }), mon('Gengar', ['splash'], { ability: 'Levitate' })],
      [mon('Snorlax', ['explosion', 'splash'], { ability: 'Thick Fat' }), mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1']], { onBoundary: (b) => dump(b.sides[0].active[0]) });
    showDec('F3', r);
  }

  // ================================================================= G. Hidden Power
  console.log('\n############ G. copied HIDDEN POWER (hpType/hpPower for gen<5) ############');
  {
    // IVs giving Hidden Power Ice 70 for the target; Ditto has 31s (HP Dark 70).
    const hpIce = { hp: 31, atk: 2, def: 30, spa: 31, spd: 31, spe: 31 };
    const teams = [
      [mon('Ditto', ['transform'], { ability: 'Limber' })],
      [mon('Snorlax', ['hiddenpower', 'splash'], { ability: 'Thick Fat', ivs: hpIce })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 2'], ['move 1', 'move 2']],
      { onBoundary: (b) => dump(b.sides[0].active[0]) });
    showDec('G', r);
    r.states.forEach((s, i) => console.log(`  G state${i} hpType=${s.hpType}/${s.hpPower} base=${s.baseHpType}/${s.baseHpPower} slots=${JSON.stringify(s.slots)} movesGetter=${JSON.stringify(s.movesGetter)}`));
    const q = await perSide(teams, SEED, [['move 1', 'move 2']]);
    q.reqs.forEach((x, i) => console.log(`  G req${i}: ${x}`));
  }

  // ================================================================= H. Baton Pass
  console.log('\n############ H. BATON PASS from a transformed mon (transform is NOT a volatile) ############');
  {
    const teams = [
      [mon('Ditto', ['transform', 'splash'], { ability: 'Limber' }), mon('Gengar', ['splash'], { ability: 'Levitate' })],
      [mon('Snorlax', ['batonpass', 'swordsdance', 'splash'], { ability: 'Thick Fat' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 2'], ['move 1', 'move 3'], ['switch 2', 'move 3']],
      { onBoundary: (b) => ({ act: dump(b.sides[0].active[0]) }) });
    showDec('H', r);
    r.states.forEach((s, i) => console.log(`  H state${i} sp=${s.act.species} tr=${s.act.transformed} boosts=${JSON.stringify(s.act.boosts)} slots=${JSON.stringify(s.act.slots.map((m) => m.id))}`));
  }

  // ================================================================= I. ability End in gen3
  console.log('\n############ I. gen3: NO ability onStart on transform (gen>3 gate); the old ability End DOES fire ############');
  {
    const teams = [
      [mon('Ditto', ['transform', 'splash'], { ability: 'Limber' })],
      [mon('Gyarados', ['splash'], { ability: 'Intimidate' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1'], ['move 2', 'move 1']],
      { onBoundary: (b) => ({ p1ab: b.sides[0].active[0].ability, p2boost: b.sides[1].active[0].boosts, p1boost: b.sides[0].active[0].boosts }) });
    showDec('I', r);
    r.states.forEach((s, i) => console.log(`  I state${i}=${JSON.stringify(s)}`));
  }

  // ================================================================= J. gen3 randbats Ditto
  console.log('\n############ J. gen3 randbats: what set does Ditto actually get? ############');
  {
    const dex = Dex.mod('gen3');
    const sp = dex.species.get('Ditto');
    console.log(`  Ditto: types=${JSON.stringify(sp.types)} baseStats=${JSON.stringify(sp.baseStats)} abilities=${JSON.stringify(sp.abilities)} weighthg=${sp.weighthg}`);
    const mv = dex.moves.get('transform');
    console.log(`  transform move: ${JSON.stringify({ num: mv.num, cat: mv.category, acc: mv.accuracy, pp: mv.pp, target: mv.target, flags: mv.flags, type: mv.type, priority: mv.priority })}`);
  }

  // ================================================================= K. choice lock
  console.log('\n############ K. CHOICE BAND lock across transform ############');
  {
    const teams = [
      [mon('Ditto', ['transform', 'splash'], { ability: 'Limber', item: 'Choice Band' })],
      [mon('Snorlax', ['bodyslam', 'splash', 'rest', 'curse'], { ability: 'Thick Fat' })],
    ];
    const r = await perSide(teams, SEED, [['move 1', 'move 2'], ['move 1', 'move 2']]);
    r.reqs.forEach((q, i) => console.log(`  K req${i}: ${q}`));
  }
}
main().then(() => process.exit(0)).catch((e) => { console.error(e); process.exit(1); });
