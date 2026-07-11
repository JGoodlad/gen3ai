// probe_truant_rng.js — settle the TRUANT draw model against the resolved gen3 sim
// (Mandate 1 — the sim is the only oracle; the resolved handlers are the HYPOTHESIS:
// onBeforeMovePriority 9, onBeforeMove cant iff pokemon.truantTurn (draw-free),
// onSwitchIn truantTurn = (turn !== 0), onResidual (order 27) truantTurn = !truantTurn).
//
// Questions:
//  Q1 cadence + draw-freeness: lead Slaking moves turn 1, loafs turn 2, alternates;
//     the loaf turn draws NOTHING for the loafer (no acc/crit/dmg), no PP deducted.
//  Q2 ladder position vs sleep (10) / flinch (8) / para (1): an asleep Slaking on a
//     loaf turn shows WHICH cant? does the sleep counter still decrement? a paralyzed
//     Slaking's loaf turn draws NO para roll (truant 9 > par 1 short-circuits).
//  Q3 arming: a VOLUNTARY mid-battle switch-in (turn != 0) sets truantTurn=true, but
//     that turn's residual toggles it back → moves next turn. A POST-FAINT replacement
//     (after the residual) keeps truantTurn=true → LOAFS its first turn. Turn-0 lead
//     arms false → moves turn 1.
//  Q4 the order-27 residual handler: does a speed-TIED Slaking mirror draw a residual
//     tie-shuffle (2 truant handlers, same order 27) vs a control? and does truant tick
//     (toggle) even on a loaf turn / while asleep?
//
// Run: node src/rust_sim/harness/probe_truant_rng.js

'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

const SEED = [11, 22, 33, 44];

async function q1() {
  console.log('=== Q1 cadence + draws + PP (Slaking Scratch vs Shuckle Splash, customgame)');
  const teams = [
    [mon('Slaking', ['scratch'], { ability: 'Truant' })],
    [mon('Shuckle', ['splash'], { ability: 'Sturdy' })],
  ];
  const r = await run(teams, SEED, Array(6).fill(['move 1', 'move 1']), {
    onBoundary: (b) => ({
      pp: b.p1.active[0].moveSlots[0].pp,
      truantTurn: b.p1.active[0].truantTurn,
      shuckleHp: b.p2.active[0].hp,
    }),
  });
  r.perDecision.forEach((d, i) => {
    const cant = d.lines.filter((l) => l.includes('|cant|'));
    console.log(`turn ${i + 1}: draws=[${fmtCalls(d.calls)}] nexts=${d.nexts} cant=${JSON.stringify(cant)} state=${JSON.stringify(r.states[i])}`);
  });
}

async function q2() {
  console.log('=== Q2 ladder: asleep Slaking (Spore turn 1) — which cant on a loaf turn? counter decrement?');
  // Fast Shuckle? No — Shuckle is slow. Use a fast Spore user: Breloom (customgame).
  const teams = [
    [mon('Slaking', ['scratch'], { ability: 'Truant' })],
    [mon('Breloom', ['spore', 'splash'], { ability: 'Effect Spore', evs: { spe: 252 } })],
  ];
  // t1: Slaking moves (truantTurn false), Breloom Spores it (sleep random(2,6) drawn? gen3 sleep 2-5 starts).
  // t2..: Slaking asleep + loafing — watch cant lines + sleep decrement + truantTurn.
  const r = await run(teams, SEED, [['move 1', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2']], {
    onBoundary: (b) => ({
      status: b.p1.active[0].status,
      sleepTime: b.p1.active[0].statusState && b.p1.active[0].statusState.time,
      truantTurn: b.p1.active[0].truantTurn,
    }),
  });
  r.perDecision.forEach((d, i) => {
    const cant = d.lines.filter((l) => l.includes('|cant|') || l.includes('curestatus'));
    console.log(`turn ${i + 1}: draws=[${fmtCalls(d.calls)}] cant=${JSON.stringify(cant)} state=${JSON.stringify(r.states[i])}`);
  });

  console.log('--- Q2b paralyzed Slaking loaf turn: no para roll?');
  const teams2 = [
    [mon('Slaking', ['scratch'], { ability: 'Truant' })],
    [mon('Jolteon', ['thunderwave', 'splash'], { ability: 'Sturdy' })],
  ];
  const r2 = await run(teams2, SEED, [['move 1', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2'], ['move 1', 'move 2']], {
    onBoundary: (b) => ({ status: b.p1.active[0].status, truantTurn: b.p1.active[0].truantTurn }),
  });
  r2.perDecision.forEach((d, i) => {
    const cant = d.lines.filter((l) => l.includes('|cant|'));
    console.log(`turn ${i + 1}: draws=[${fmtCalls(d.calls)}] cant=${JSON.stringify(cant)} state=${JSON.stringify(r2.states[i])}`);
  });
}

async function q3() {
  console.log('=== Q3 arming: voluntary switch-in vs post-faint replacement');
  // Voluntary: Slaking leads, switches out t2, back in t4; does it move t5?
  const teams = [
    [mon('Slaking', ['scratch'], { ability: 'Truant' }), mon('Zangoose', ['scratch'], { ability: 'Immunity' })],
    [mon('Shuckle', ['splash'], { ability: 'Sturdy' })],
  ];
  const r = await run(teams, SEED, [
    ['move 1', 'move 1'],   // t1 Slaking moves
    ['switch 2', 'move 1'], // t2 out (Zangoose in)
    ['move 1', 'move 1'],   // t3 Zangoose scratch
    ['switch 2', 'move 1'], // t4 Slaking back in (turn != 0)
    ['move 1', 'move 1'],   // t5 does Slaking move or loaf?
    ['move 1', 'move 1'],   // t6
  ], { onBoundary: (b) => ({ active: b.p1.active[0].species.id, truantTurn: b.p1.active[0].truantTurn }) });
  r.perDecision.forEach((d, i) => {
    const cant = d.lines.filter((l) => l.includes('|cant|'));
    console.log(`turn ${i + 1}: cant=${JSON.stringify(cant)} state=${JSON.stringify(r.states[i])} draws=[${fmtCalls(d.calls)}]`);
  });

  console.log('--- Q3b post-faint replacement: does the replacement Slaking loaf its first turn?');
  const teams2 = [
    [mon('Caterpie', ['splash'], { ability: 'Sturdy', level: 1 }), mon('Slaking', ['scratch'], { ability: 'Truant' })],
    [mon('Machamp', ['crosschop', 'splash'], { ability: 'Guts', evs: { atk: 252 } })],
  ];
  // t1: Machamp KOs Caterpie -> forced replacement (Slaking in AFTER the residual).
  const r2 = await run(teams2, SEED, [
    ['move 1', 'move 1'],  // t1 caterpie splash, machamp KO
    ['switch 2', null],    // forced replacement: Slaking in
    ['move 1', 'move 2'],  // t2: does Slaking loaf?
    ['move 1', 'move 2'],  // t3
    ['move 1', 'move 2'],  // t4
  ], { onBoundary: (b) => ({ active: b.p1.active[0].species.id, truantTurn: b.p1.active[0].truantTurn }) });
  r2.perDecision.forEach((d, i) => {
    const cant = d.lines.filter((l) => l.includes('|cant|') || l.includes('faint'));
    console.log(`boundary ${i + 1}: cant/faint=${JSON.stringify(cant)} state=${JSON.stringify(r2.states[i])}`);
  });
}

async function q4() {
  console.log('=== Q4 order-27 residual: speed-tied Slaking mirror draws vs a Vigoroth(=control) — extra tie-shuffle?');
  // Both p1+p2 Slaking (same species → speed tie). Control: replace ONE side's ability via
  // a different truant-less mon of identical speed? Simplest: compare the mirror draw count
  // to hypothesis; and an unequal-speed pair (EVs) for the no-tie case.
  const mk = (spe) => [
    [mon('Slaking', ['scratch'], { ability: 'Truant', evs: { spe } })],
    [mon('Slaking', ['scratch'], { ability: 'Truant', evs: { spe: 0 } })],
  ];
  for (const [label, spe] of [['tied', 0], ['untied', 252]]) {
    const r = await run(mk(spe), SEED, Array(4).fill(['move 1', 'move 1']));
    r.perDecision.forEach((d, i) => {
      console.log(`${label} turn ${i + 1}: nexts=${d.nexts} draws=[${fmtCalls(d.calls)}]`);
    });
  }
  // Control pair with NO truant at the same tie (Zangoose mirror): baseline residual draws.
  const ctl = [
    [mon('Zangoose', ['scratch'], { ability: 'Immunity' })],
    [mon('Zangoose', ['scratch'], { ability: 'Immunity' })],
  ];
  const rc = await run(ctl, SEED, Array(3).fill(['move 1', 'move 1']));
  rc.perDecision.forEach((d, i) => console.log(`control(tied, no truant) turn ${i + 1}: nexts=${d.nexts} draws=[${fmtCalls(d.calls)}]`));
}

(async () => { await q1(); await q2(); await q3(); await q4(); })().catch((e) => { console.error(e); process.exit(1); });
