// probe_truant_edges_rng.js — the two TRUANT arming edges probe_truant_rng.js left open:
//  E1: a replacement that enters AFTER the residual (a status-DoT KO at end of turn) —
//      onSwitchIn sets truantTurn=true and NO residual toggles it that turn → does it
//      LOAF its first full turn? (Reachable in e2e: DoT KOs happen.)
//  E2: a Roar DRAG-in mid-turn (before the residual) — toggled same turn → moves next?
// Run: node src/rust_sim/harness/probe_truant_edges_rng.js

'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

const SEED = [3, 1, 4, 1];

async function e1() {
  console.log('=== E1 residual-DoT-KO replacement: Slaking enters AFTER the residual');
  // p1: Shedinja (1 HP — the burn DoT KOs it AT the residual), then Slaking. p2: slow WoW user.
  const teams = [
    [mon('Shedinja', ['splash'], { ability: 'Wonder Guard' }), mon('Slaking', ['scratch'], { ability: 'Truant' })],
    [mon('Dusclops', ['willowisp', 'splash'], { ability: 'Pressure' })],
  ];
  const r = await run(teams, SEED, [
    ['move 1', 'move 1'],  // t1: WoW burns Shedinja; the residual burn chip KOs it (AFTER the residual)
    ['switch 2', null],    // forced replacement: Slaking in (post-residual)
    ['move 1', 'move 2'],  // t2: does Slaking LOAF its first turn?
    ['move 1', 'move 2'],  // t3
    ['move 1', 'move 2'],  // t4
  ], { onBoundary: (b) => ({ active: b.p1.active[0].species.id, hp: b.p1.active[0].hp, truantTurn: b.p1.active[0].truantTurn, turn: b.turn }) });
  r.perDecision.forEach((d, i) => {
    const ev = d.lines.filter((l) => l.includes('|cant|') || l.includes('|faint|') || l.includes('-status'));
    console.log(`boundary ${i + 1}: ev=${JSON.stringify(ev)} state=${JSON.stringify(r.states[i])}`);
  });
}

async function e2() {
  console.log('=== E2 Roar drag-in mid-turn: dragged Slaking — moves or loafs next turn?');
  const teams = [
    [mon('Zangoose', ['scratch'], { ability: 'Immunity' }), mon('Slaking', ['scratch'], { ability: 'Truant' })],
    [mon('Skarmory', ['roar', 'splash'], { ability: 'Keen Eye' })],
  ];
  const r = await run(teams, SEED, [
    ['move 1', 'move 1'],  // t1: Roar drags Zangoose out -> Slaking in (mid-turn, before residual)
    ['move 1', 'move 2'],  // t2: does Slaking move?
    ['move 1', 'move 2'],  // t3
  ], { onBoundary: (b) => ({ active: b.p1.active[0].species.id, truantTurn: b.p1.active[0].truantTurn }) });
  r.perDecision.forEach((d, i) => {
    const ev = d.lines.filter((l) => l.includes('|cant|') || l.includes('|drag|'));
    console.log(`turn ${i + 1}: ev=${JSON.stringify(ev)} state=${JSON.stringify(r.states[i])}`);
  });
}

(async () => { await e1(); await e2(); })().catch((e) => { console.error(e); process.exit(1); });
