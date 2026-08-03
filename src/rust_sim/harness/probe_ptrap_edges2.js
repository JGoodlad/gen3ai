// probe_ptrap_edges2.js — the second ROUND-32 edge probe for the gen3 PARTIAL-TRAP family.
// Settles the four questions `probe_ptrap_edges.js` left open:
//
//   D. the TRAPPER FAINTS         — release form + which residual; the mid-turn replacement
//   E. the VICTIM faints to the CHIP — the `-damage`/`|faint|` emission + boundary
//   O. duration-0 vs trapper-gone PRECEDENCE — both true on the same residual: which wins?
//   P. the trapped mon SWITCHES via a forced replacement after a faint? (n/a) + BATON PASS
//      chain: does the passed volatile keep releasing off the ORIGINAL trapper?
//
// Run: node harness/probe_ptrap_edges2.js
'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

const KEEP = (l) => l && !l.startsWith('|t:|') && l !== '|' && !l.startsWith('|upkeep');
function show(tag, r) {
  r.perDecision.forEach((d, i) => {
    console.log(`  ${tag} t${i + 1}: draws=${d.nexts} calls=[${fmtCalls(d.calls).slice(0, 300)}]`);
    console.log(`        lines=${JSON.stringify(d.lines.filter(KEEP))}`);
    if (r.states[i] !== undefined) console.log(`        state=${JSON.stringify(r.states[i])}`);
  });
}
const st = (b) => {
  const o = {};
  for (const s of [0, 1]) {
    const m = b.sides[s].active[0];
    if (!m) { o[`p${s + 1}`] = '-'; continue; }
    const v = m.volatiles.partiallytrapped;
    o[`p${s + 1}`] = `${m.species.id} ${m.hp}/${m.maxhp}${v ? ` PT(dur=${v.duration},src=${v.source && v.source.species.id})` : ''}${m.trapped ? ' TRAPPED' : ''}`;
  }
  return o;
};
const SEEDS = [];
for (let s = 0; s < 60; s++) SEEDS.push([s * 7 + 1, s * 3 + 2, s * 5 + 4, s * 11 + 3]);

async function main() {
  // ── D. TRAPPER FAINTS mid-turn (replacement chosen, then the residual) ─────
  console.log('############ D. the TRAPPER faints mid-turn ############');
  for (const seed of SEEDS) {
    const teams = [
      [mon('Diglett', ['wrap', 'splash'], { ability: 'Sand Veil', level: 5 }), mon('Gengar', ['splash'], { ability: 'Levitate' })],
      [mon('Snorlax', ['return', 'splash'], { ability: 'Own Tempo', evs: { atk: 252 } }), mon('Blissey', ['splash'])],
    ];
    const r = await run(teams, seed, [
      ['move 1', 'move 2'],   // t1: Diglett Wraps
      ['move 2', 'move 1'],   // t2: Snorlax Returns -> Diglett faints (mid-turn request)
      ['switch 2', null],     // t2b: p1 replacement Gengar -> then the residual runs
      ['move 1', 'move 2'],   // t3
    ], { onBoundary: st });
    if (!r.perDecision[0].lines.some((l) => l.includes('move: Wrap'))) continue;
    if (!r.perDecision[1].lines.some((l) => l.startsWith('|faint|p1a'))) continue;
    console.log(`=== seed ${JSON.stringify(seed)} ===`); show('TRAPPERFNT', r); break;
  }

  // ── E. the VICTIM faints to the residual CHIP ─────────────────────────────
  console.log('\n############ E. the VICTIM faints to the residual chip ############');
  {
    // p1 re-casts Wrap every turn (bp 15): the move chips, the trap chips, and eventually
    // one of the two lands the KO. Sweep seeds until the KO comes from the RESIDUAL.
    let found = false;
    for (const seed of SEEDS) {
      const teams = [
        [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus' })],
        [mon('Blissey', ['splash'], { ability: 'Natural Cure' }), mon('Snorlax', ['splash'])],
      ];
      const ch = [];
      for (let i = 0; i < 26; i++) ch.push(['move 1', 'move 1']);
      const r = await run(teams, seed, ch, { onBoundary: st });
      const idx = r.perDecision.findIndex((d) => {
        const ls = d.lines.filter(KEEP);
        const k = ls.findIndex((l) => l.includes('[partiallytrapped]') && l.includes('0 fnt'));
        return k >= 0;
      });
      if (idx < 0) continue;
      console.log(`=== seed ${JSON.stringify(seed)} (chip-KO at decision ${idx + 1}) ===`);
      for (let i = Math.max(0, idx - 1); i < Math.min(r.perDecision.length, idx + 3); i++) {
        console.log(`  CHIPKO t${i + 1}: draws=${r.perDecision[i].nexts}`);
        console.log(`        lines=${JSON.stringify(r.perDecision[i].lines.filter(KEEP))}`);
        console.log(`        state=${JSON.stringify(r.states[i])}`);
      }
      found = true; break;
    }
    if (!found) console.log('  (no residual chip-KO found in the sweep)');
  }

  // ── O. duration-0 vs trapper-gone on the SAME residual ────────────────────
  console.log('\n############ O. duration hits 0 on the SAME residual the trapper leaves ############');
  {
    // Find a seed whose random(3,7) == 3 (=> 2 chips: cast turn + turn 2; the `-end` lands on
    // turn 3's residual). Then switch the trapper out ON TURN 3 and see which branch emits.
    let done = false;
    for (const seed of SEEDS) {
      const teams = [
        [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus' }), mon('Gengar', ['splash'], { ability: 'Levitate' })],
        [mon('Snorlax', ['splash'], { ability: 'Own Tempo' })],
      ];
      const r = await run(teams, seed, [
        ['move 1', 'move 1'],   // t1: Wrap
        ['move 2', 'move 1'],   // t2: chip
        ['switch 2', 'move 1'], // t3: trapper leaves AND duration would hit 0
        ['move 1', 'move 1'],
      ], { onBoundary: st });
      const cast = r.perDecision[0];
      const dur = cast.calls.find((c) => c.kind === 'random' && c.args[0] === 3 && c.args[1] === 7);
      if (!dur || dur.ret !== 3) continue;
      console.log(`=== seed ${JSON.stringify(seed)} duration=${dur.ret} ===`); show('PRECEDENCE', r);
      done = true; break;
    }
    if (!done) console.log('  (no duration==3 seed found)');
  }

  // ── P. BATON PASS chain: does the passed volatile still key off the ORIGINAL trapper? ──
  console.log('\n############ P. BATON-PASSED trap + the ORIGINAL trapper switching out ############');
  for (const seed of SEEDS) {
    const teams = [
      [mon('Dragonite', ['wrap', 'splash'], { ability: 'Inner Focus' }), mon('Gengar', ['splash'], { ability: 'Levitate' })],
      [mon('Snorlax', ['batonpass', 'splash'], { ability: 'Own Tempo' }), mon('Blissey', ['splash'], { ability: 'Natural Cure' })],
    ];
    const r = await run(teams, seed, [
      ['move 1', 'move 2'],   // t1: Wrap lands on Snorlax
      ['move 2', 'move 1'],   // t2: Snorlax Baton Passes
      [null, 'switch 2'],     // t2b: Blissey in, inherits the trap
      ['switch 2', 'move 1'], // t3: the ORIGINAL trapper (Dragonite) switches out -> release?
      ['move 1', 'move 1'],
    ], { onBoundary: st });
    if (!r.perDecision[0].lines.some((l) => l.includes('move: Wrap'))) continue;
    console.log(`=== seed ${JSON.stringify(seed)} ===`); show('BPCHAIN', r); break;
  }
}
main().catch((e) => { console.error(e); process.exit(1); });
