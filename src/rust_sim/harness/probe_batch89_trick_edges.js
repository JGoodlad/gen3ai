// probe_batch89_trick_edges.js — Trick edge cases: vs Substitute, vs Mail (untradeable),
// vs a Berry, and confirm switcheroo is NOT gen3-legal (learnset check).
// Run: node src/rust_sim/harness/probe_batch89_trick_edges.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Dex } = require(path.join(PS, 'dist/sim'));
const { mon, run, fmtCalls } = require('./probe_batch4_lib');
const SEED = [5, 4, 3, 2];

function showDec(tag, r) {
  r.perDecision.forEach((d, i) => {
    const lines = d.lines.filter((l) => l && !l.startsWith('|t:|') && !l.startsWith('|upkeep'));
    console.log(`  ${tag} t${i + 1}: draws=${d.nexts} calls=[${fmtCalls(d.calls)}]`);
    console.log(`        lines=${JSON.stringify(lines)}`);
    if (r.states[i]) console.log(`        state=${JSON.stringify(r.states[i])}`);
  });
}

async function main() {
  const d = Dex.forFormat('gen3customgame');
  // --- switcheroo gen3-legality (num 415 => gen4) ---
  const sw = d.moves.get('switcheroo');
  console.log(`switcheroo: num=${sw.num} exists=${sw.exists} (gen3 move nums cap ~354; 415 => gen4-only, NOT gen3-legal)`);
  const tr = d.moves.get('trick');
  console.log(`trick: num=${tr.num} isMax=${!!tr.isMax}`);

  // --- Trick INTO a Substitute (bypasssub NOT set) ---
  console.log('\n## Trick into a Substitute');
  {
    const teams = [
      [mon('Alakazam', ['trick', 'splash'], { item: 'Choice Band', ability: 'Synchronize' })],
      [mon('Snorlax', ['substitute', 'splash'], { item: 'Leftovers', ability: 'Own Tempo' })],
    ];
    const r = await run(teams, SEED, [
      ['move 2', 'move 1'],  // p1 Splash, p2 Substitute
      ['move 1', 'move 2'],  // p1 Trick into subbed Snorlax
    ], { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item, p2sub: b.sides[1].active[0].volatiles.substitute ? b.sides[1].active[0].volatiles.substitute.hp : null }) });
    showDec('sub', r);
  }

  // --- Trick a MAIL holder (untradeable) ---
  console.log('\n## Trick a Mail holder (untradeable -> whole move fails?)');
  {
    const teams = [
      [mon('Alakazam', ['trick'], { item: 'Choice Band', ability: 'Synchronize' })],
      [mon('Snorlax', ['splash'], { item: 'Bead Mail', ability: 'Own Tempo' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1']],
      { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item }) });
    showDec('mail', r);
  }

  // --- p1 holds Mail, p2 holds item ---
  console.log('\n## Trick where the USER holds Mail');
  {
    const teams = [
      [mon('Alakazam', ['trick'], { item: 'Bead Mail', ability: 'Synchronize' })],
      [mon('Snorlax', ['splash'], { item: 'Leftovers', ability: 'Own Tempo' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1']],
      { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item }) });
    showDec('mail-user', r);
  }

  // --- Trick a BERRY holder (berries tradeable in gen3?) ---
  console.log('\n## Trick a Berry holder (Lum Berry)');
  {
    const teams = [
      [mon('Alakazam', ['trick'], { item: 'Choice Band', ability: 'Synchronize' })],
      [mon('Snorlax', ['splash'], { item: 'Lum Berry', ability: 'Own Tempo' })],
    ];
    const r = await run(teams, SEED, [['move 1', 'move 1']],
      { onBoundary: (b) => ({ p1item: b.sides[0].active[0].item, p2item: b.sides[1].active[0].item }) });
    showDec('berry', r);
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
