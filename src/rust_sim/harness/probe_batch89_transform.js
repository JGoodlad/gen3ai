// probe_batch89_transform.js — LIVE probe for TRANSFORM (gen3) vs the omniscient sim.
// Settles: draw model (draw-free copy?), WHAT is copied (species/stats/moves/PP/types/boosts/
// ability), the transform volatile, fail conditions (vs a transformed/subbed target),
// revert on switch-out, emission bytes (-transform), and move PP of the copied moves.
// Run: node src/rust_sim/harness/probe_batch89_transform.js
'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');
const SEED = [5, 4, 3, 2];

function showDec(tag, r) {
  r.perDecision.forEach((d, i) => {
    const lines = d.lines.filter((l) => l && !l.startsWith('|t:|') && !l.startsWith('|upkeep') && l !== '|');
    console.log(`  ${tag} t${i + 1}: draws=${d.nexts} calls=[${fmtCalls(d.calls)}]`);
    console.log(`        lines=${JSON.stringify(lines)}`);
    if (r.states[i]) console.log(`        state=${JSON.stringify(r.states[i], null, 0)}`);
  });
}

const copyState = (b) => {
  const u = b.sides[0].active[0];
  return {
    species: u.species.id,
    types: u.types,
    transformed: u.transformed,
    boosts: u.boosts,
    ability: u.ability,
    stats: u.storedStats,
    moves: u.moveSlots.map((m) => `${m.id}:${m.pp}/${m.maxpp}`),
    vol: Object.keys(u.volatiles),
  };
};

async function main() {
  // 1. Ditto Transforms into a boosted Snorlax: copy species/stats/moves(PP=5)/types/boosts.
  console.log('############ TRANSFORM: Ditto -> boosted Snorlax ############');
  {
    const teams = [
      [mon('Ditto', ['transform'], { ability: 'Limber' }), mon('Gengar', ['splash'], { ability: 'Levitate' })],
      [mon('Snorlax', ['swordsdance', 'bodyslam', 'rest', 'splash'], { item: 'Leftovers', ability: 'Thick Fat' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 1'],  // t1: p2 SD (+2 atk), p1 Ditto Transform -> copies Snorlax incl +2 atk?
    ], { onBoundary: copyState });
    showDec('T', r);
  }

  // 2. Draw-freeness: Transform vs a control (Splash), same seed, count draws.
  console.log('\n############ TRANSFORM draw count vs Splash control ############');
  {
    const mk = (m1) => [
      [mon('Ditto', [m1], { ability: 'Limber' })],
      [mon('Snorlax', ['splash'], { ability: 'Thick Fat' })],
    ];
    for (const m1 of ['transform', 'splash']) {
      const r = await run(mk(m1), SEED, [['move 1', 'move 1']], { onBoundary: copyState });
      console.log(`  ${m1}: draws=${r.perDecision[0].nexts} calls=[${fmtCalls(r.perDecision[0].calls)}]`);
      console.log(`     lines=${JSON.stringify(r.perDecision[0].lines.filter((l) => l && l !== '|' && !l.startsWith('|t:|') && !l.startsWith('|upkeep')))}`);
    }
  }

  // 3. Use a copied move (does the copied PP deduct? does it run correctly?).
  console.log('\n############ Use a copied move after Transform ############');
  {
    const teams = [
      [mon('Ditto', ['transform'], { ability: 'Limber' })],
      [mon('Alakazam', ['psychic', 'calmmind', 'recover', 'splash'], { ability: 'Synchronize' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 4'],  // t1: Transform into Alakazam
      ['move 2', 'move 4'],  // t2: Ditto (now Alakazam) uses copied Calm Mind -> boost + PP deduct
    ], { onBoundary: copyState });
    showDec('USE', r);
  }

  // 4. Revert on switch-out.
  console.log('\n############ Revert on switch-out ############');
  {
    const teams = [
      [mon('Ditto', ['transform'], { ability: 'Limber' }), mon('Gengar', ['splash'], { ability: 'Levitate' })],
      [mon('Snorlax', ['splash'], { ability: 'Thick Fat' })],
    ];
    const r = await run(teams, SEED, [
      ['move 1', 'move 1'],  // t1: Transform into Snorlax
      ['switch 2', 'move 1'],// t2: Ditto switches to Gengar
      ['switch 2', 'move 1'],// t3: Ditto back in -> reverted to Ditto?
    ], { onBoundary: (b) => ({ activeSpecies: b.sides[0].active[0].species.id, benchDitto: (b.sides[0].pokemon.find((p) => p.baseSpecies.id === 'ditto') || {}).species && b.sides[0].pokemon.find((p) => p.baseSpecies.id === 'ditto').species.id }) });
    showDec('REVERT', r);
  }

  // 5. Fail: Transform into an already-transformed target; Transform into a substituted target.
  console.log('\n############ FAIL: vs transformed target / vs substitute ############');
  {
    // vs substitute (transform flags include bypasssub -> does it copy THROUGH the sub?)
    const teams = [
      [mon('Ditto', ['transform', 'splash'], { ability: 'Limber' })],
      [mon('Snorlax', ['substitute', 'splash'], { ability: 'Thick Fat' })],
    ];
    const r = await run(teams, SEED, [
      ['move 2', 'move 1'],  // t1: p2 Substitute, p1 Splash
      ['move 1', 'move 2'],  // t2: p1 Transform into subbed Snorlax (bypasssub in flags)
    ], { onBoundary: copyState });
    showDec('vs-sub', r);
  }
  {
    // Ditto vs Ditto -> mutual transform edge.
    const teams = [
      [mon('Ditto', ['transform'], { ability: 'Limber' })],
      [mon('Ditto', ['transform'], { ability: 'Limber' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1']], { onBoundary: copyState });
    showDec('ditto-mirror', r);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
