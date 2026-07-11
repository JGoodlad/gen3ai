// probe_innerfocus_rng.js — settle INNER FOCUS (flinch immunity) vs the resolved gen3 sim.
// Hypothesis (resolved dist): onTryAddVolatile — status.id==='flinch' → null, i.e. the
// block happens at the volatile APPLY: the move's own flinch-secondary random(100) is
// STILL DRAWN (draw-count-neutral), the flinch just never sticks. CONTRAST Shield Dust,
// which FILTERS the secondary list (onModifySecondaries) → the random(100) is NOT drawn.
// Also: King's Rock's ADDED flinch secondary must behave the same way (drawn, blocked).
// Run: node src/rust_sim/harness/probe_innerfocus_rng.js

'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

// Fast Bite user into a slow defender (defender moves second → a landed flinch cants).
// Find seeds where the 30% flinch secondary PASSES, then compare abilities.
async function main() {
  const mk = (defAbility) => [
    [mon('Jolteon', ['bite'], { ability: 'Sturdy', evs: { spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: defAbility })],
  ];
  for (const seed of [[1, 2, 3, 4], [9, 9, 9, 9], [5, 6, 7, 8], [2, 4, 6, 8], [10, 20, 30, 40]]) {
    const out = {};
    for (const ab of ['Thick Fat', 'Inner Focus', 'Shield Dust']) {
      const r = await run(mk(ab), seed, Array(3).fill(['move 1', 'move 1']));
      out[ab] = r.perDecision.map((d) => ({
        draws: fmtCalls(d.calls),
        cant: d.lines.filter((l) => l.includes('|cant|')).length,
      }));
    }
    for (let t = 0; t < 3; t++) {
      console.log(`seed=${JSON.stringify(seed)} turn ${t + 1}:`);
      for (const ab of ['Thick Fat', 'Inner Focus', 'Shield Dust']) {
        const d = out[ab][t];
        if (d) console.log(`  ${ab.padEnd(12)} cant=${d.cant} [${d.draws}]`);
      }
    }
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
