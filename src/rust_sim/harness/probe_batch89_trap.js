// probe_batch89_trap.js — LIVE draw-model probe for the Batch-8 PARTIAL-TRAP family
// (wrap/bind/firespin/clamp/sandtomb/whirlpool) vs the OMNISCIENT gen3 BattleStream.
//
// Settles the bit-for-bit crux:
//   1. DURATION DRAW: the `partiallytrapped` durationCallback in gen3 resolves (via gen4
//      override, inherited) to `this.random(3, 7)` — a SINGLE draw at CAST. How many CHIP
//      turns result (duration semantics: 3-6 draws => 2-5 chip turns?).
//   2. CHIP: baseMaxhp/16 each end-of-turn (no Binding Band/Grip Claw in gen3).
//   3. RELEASE: trapper switch-out / faint / activeTurns==0 -> trap ends silently.
//   4. SWITCH-BLOCK: the trapped mon can't voluntarily switch (onTrapPokemon -> tryTrap).
//   5. EMISSION bytes: -activate move: X [of] SRC ; -damage [from] move: X [partiallytrapped] ; -end.
//   6. Interaction with the endTurn draw baseline + residual tie-shuffle order.
//
// Run: node src/rust_sim/harness/probe_batch89_trap.js
'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

function showDec(tag, r) {
  r.perDecision.forEach((d, i) => {
    const lines = d.lines.filter((l) => l && !l.startsWith('|t:|') && !l.startsWith('|upkeep') && l !== '|');
    console.log(`  ${tag} t${i + 1}: draws=${d.nexts} calls=[${fmtCalls(d.calls)}]`);
    console.log(`        lines=${JSON.stringify(lines)}`);
    if (r.states[i]) console.log(`        state=${JSON.stringify(r.states[i])}`);
  });
}

const boundState = (b) => {
  const t = b.sides[1].active[0];
  const v = t && t.volatiles.partiallytrapped;
  return { p2hp: t ? `${t.hp}/${t.maxhp}` : '-', pt: v ? { dur: v.duration, time: v.time } : null, p2trapped: t ? t.trapped : '-', p2species: t ? t.species.id : '-' };
};

async function main() {
  // Find a seed where Wrap LANDS (acc 85), then trace the full lifecycle to release.
  const mkTeams = () => [
    [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus' })],
    [mon('Snorlax', ['splash', 'return'], { ability: 'Own Tempo' }), mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
  ];
  console.log('############ WRAP full lifecycle (search a landing seed) ############');
  const seeds = [[5, 4, 3, 2], [1, 2, 3, 4], [7, 11, 13, 17], [2, 2, 2, 2], [9, 8, 7, 6], [3, 1, 4, 1]];
  for (const seed of seeds) {
    const r = await run(mkTeams(), seed, [
      ['move 1', 'move 1'],  // t1: Wrap (does it land? -activate + duration draw)
      ['move 2', 'move 1'],  // t2: chip
      ['move 2', 'move 1'],  // t3: chip
      ['move 2', 'move 1'],  // t4: chip
      ['move 2', 'move 1'],  // t5: chip / maybe -end
      ['move 2', 'move 1'],  // t6: -end / re-castable
      ['move 2', 'move 1'],  // t7
    ], { onBoundary: boundState });
    const landed = r.perDecision[0].lines.some((l) => l.includes('-activate') && l.includes('Wrap'));
    if (!landed) { console.log(`  seed ${JSON.stringify(seed)}: Wrap MISSED (acc 85), skip`); continue; }
    console.log(`\n=== seed ${JSON.stringify(seed)}: Wrap LANDED — full trace ===`);
    showDec('WRAP', r);
    break;
  }

  // SWITCH-BLOCK: a trapped mon's voluntary switch is rejected.
  console.log('\n############ SWITCH-BLOCK: trapped mon tries to switch ############');
  for (const seed of seeds) {
    const r = await run(mkTeams(), seed, [
      ['move 1', 'move 1'],  // t1: Wrap lands?
      ['move 2', 'switch 2'],// t2: trapped Snorlax tries to switch to Blissey -> rejected?
    ], { onBoundary: boundState });
    const landed = r.perDecision[0].lines.some((l) => l.includes('-activate') && l.includes('Wrap'));
    if (!landed) continue;
    console.log(`\n=== seed ${JSON.stringify(seed)} ===`);
    r.perDecision.forEach((d, i) => {
      const errs = d.lines.filter((l) => l.startsWith('|error|'));
      console.log(`  t${i + 1}: draws=${d.nexts} switchErrs=${JSON.stringify(errs)} state=${JSON.stringify(r.states[i])}`);
    });
    // also check: is the SWITCH rejected (state should show p2 still Snorlax)?
    break;
  }

  // RELEASE on TRAPPER switch-out: p1 (wrapper) switches out -> trap ends.
  console.log('\n############ RELEASE on trapper switch-out ############');
  for (const seed of seeds) {
    const teams = [
      [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus' }), mon('Gengar', ['splash'], { ability: 'Levitate' })],
      [mon('Snorlax', ['splash', 'return'], { ability: 'Own Tempo' })],
    ];
    const r = await run(teams, seed, [
      ['move 1', 'move 1'],  // t1: Wrap lands
      ['switch 2', 'move 1'],// t2: wrapper Dragonite switches out -> trap should END on Snorlax
      ['move 1', 'move 1'],  // t3: confirm Snorlax no longer trapped/chipping
    ], { onBoundary: boundState });
    const landed = r.perDecision[0].lines.some((l) => l.includes('-activate') && l.includes('Wrap'));
    if (!landed) continue;
    console.log(`\n=== seed ${JSON.stringify(seed)} ===`);
    showDec('RELEASE', r);
    break;
  }

  // DURATION distribution: sample the random(3,7) draw across many seeds + count chip turns.
  console.log('\n############ DURATION distribution (count chip turns per landing) ############');
  const durCounts = {};
  const chipCounts = {};
  let landCount = 0;
  for (let s = 0; s < 120; s++) {
    const seed = [s * 7 + 1, s * 3 + 2, s * 5 + 4, s * 11 + 3];
    const r = await run(mkTeams(), seed, [
      ['move 1', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'],
      ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'], ['move 2', 'move 1'],
    ], { onBoundary: boundState });
    const cast = r.perDecision[0];
    if (!cast.lines.some((l) => l.includes('-activate') && l.includes('Wrap'))) continue;
    landCount++;
    const durCall = cast.calls.find((c) => c.kind === 'random' && Array.isArray(c.args) && c.args[0] === 3 && c.args[1] === 7);
    if (durCall) durCounts[durCall.ret] = (durCounts[durCall.ret] || 0) + 1;
    // count chip turns: number of decisions whose lines carry a [partiallytrapped] -damage
    let chips = 0;
    for (const d of r.perDecision) if (d.lines.some((l) => l.includes('[partiallytrapped]') && l.includes('-damage'))) chips++;
    chipCounts[chips] = (chipCounts[chips] || 0) + 1;
  }
  console.log(`  landed=${landCount}/120`);
  console.log(`  random(3,7) return distribution: ${JSON.stringify(durCounts)}`);
  console.log(`  chip-turn-count distribution: ${JSON.stringify(chipCounts)}`);
}
main().catch((e) => { console.error(e); process.exit(1); });
