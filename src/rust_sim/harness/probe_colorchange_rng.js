// probe_colorchange_rng.js — settle COLOR CHANGE (on-hit type override) vs the resolved
// gen3 sim. Hypotheses (resolved dist onDamagingHit): damage>0 && target.hp>0 &&
// category!=='Status' && type!=='???' && !hasType(type) → setType([move.type]),
// '-start ... typechange'. DRAW-FREE. Struggle (onModifyMove type='???') never changes.
// Probes: the change's effect on LATER type reads (chart eff, STAB, status-type-immunity,
// sand-chip immunity), behind a SUB, on the KO hit, switch-out revert, repeat-type no-op.
// Run: node src/rust_sim/harness/probe_colorchange_rng.js

'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

const SEED = [8, 8, 8, 8];

async function main() {
  console.log('=== chart + revert: Kecleon hit by Thunderbolt, then Fire; switch out/in');
  const teams = [
    [mon('Kecleon', ['splash'], { ability: 'Color Change' }), mon('Rattata', ['scratch'], { ability: 'Guts' })],
    [mon('Jolteon', ['thunderbolt', 'flamethrower', 'earthquake', 'struggle'], { ability: 'Sturdy', evs: { spe: 252 } })],
  ];
  const r = await run(teams, SEED, [
    ['move 1', 'move 1'],   // TBolt -> typechange Electric?
    ['move 1', 'move 3'],   // EQ into Electric-Kecleon: super effective? then typechange Ground
    ['switch 2', 'move 1'], // Kecleon out
    ['switch 2', 'move 1'], // Kecleon back in — types reverted?
    ['move 1', 'move 2'],   // Flamethrower -> typechange Fire?
  ], { onBoundary: (b) => ({ p1active: b.p1.active[0].species.id, types: b.p1.active[0].types, hp: b.p1.active[0].hp }) });
  r.perDecision.forEach((d, i) => {
    const ev = d.lines.filter((l) => l.includes('typechange') || l.includes('supereffective') || l.includes('-resisted') || l.includes('-immune'));
    console.log(`t${i + 1}: [${fmtCalls(d.calls)}] ev=${JSON.stringify(ev)} ${JSON.stringify(r.states[i])}`);
  });

  console.log('=== draw-freeness: Kecleon vs a Sturdy control (same seed/choices) — identical draws?');
  const mk = (ab) => [
    [mon('Kecleon', ['splash'], { ability: ab })],
    [mon('Jolteon', ['thunderbolt'], { ability: 'Sturdy', evs: { spe: 252 } })],
  ];
  for (const ab of ['Color Change', 'Sturdy']) {
    const r2 = await run(mk(ab), SEED, Array(3).fill(['move 1', 'move 1']));
    console.log(`${ab}: perTurn=${JSON.stringify(r2.perDecision.map((d) => d.nexts))}`);
  }

  console.log('=== Struggle + repeat-type + status move');
  const teams3 = [
    [mon('Kecleon', ['splash'], { ability: 'Color Change' })],
    [mon('Jolteon', ['struggle', 'thunderwave', 'thunderbolt'], { ability: 'Sturdy', evs: { spe: 252 } })],
  ];
  const r3 = await run(teams3, SEED, [['move 1', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 3'], ['move 1', 'move 3']], {
    onBoundary: (b) => ({ types: b.p1.active[0].types, status: b.p1.active[0].status }),
  });
  r3.perDecision.forEach((d, i) => {
    const ev = d.lines.filter((l) => l.includes('typechange'));
    console.log(`t${i + 1}: ev=${JSON.stringify(ev)} ${JSON.stringify(r3.states[i])}`);
  });

  console.log('=== status-type-immunity via the override: Kecleon -> Poison (Sludge Bomb) then Toxic');
  const teams4 = [
    [mon('Kecleon', ['splash'], { ability: 'Color Change' })],
    [mon('Gengar', ['sludgebomb', 'toxic'], { ability: 'Levitate', evs: { spe: 252 } })],
  ];
  const r4 = await run(teams4, SEED, [['move 1', 'move 1'], ['move 1', 'move 2'], ['move 1', 'move 2']], {
    onBoundary: (b) => ({ types: b.p1.active[0].types, status: b.p1.active[0].status }),
  });
  r4.perDecision.forEach((d, i) => {
    const ev = d.lines.filter((l) => l.includes('typechange') || l.includes('-status') || l.includes('-immune'));
    console.log(`t${i + 1}: [${fmtCalls(d.calls)}] ev=${JSON.stringify(ev)} ${JSON.stringify(r4.states[i])}`);
  });

  console.log('=== behind a SUB: does the sub-absorbed hit change the type?');
  const teams5 = [
    [mon('Kecleon', ['substitute', 'splash'], { ability: 'Color Change' })],
    [mon('Jolteon', ['thunderbolt'], { ability: 'Sturdy', evs: { spe: 252 } })],
  ];
  const r5 = await run(teams5, SEED, [['move 1', 'move 1'], ['move 2', 'move 1']], {
    onBoundary: (b) => ({ types: b.p1.active[0].types, sub: !!b.p1.active[0].volatiles['substitute'] }),
  });
  r5.perDecision.forEach((d, i) => {
    const ev = d.lines.filter((l) => l.includes('typechange'));
    console.log(`t${i + 1}: ev=${JSON.stringify(ev)} ${JSON.stringify(r5.states[i])}`);
  });

  console.log('=== KO hit: no typechange on the killing blow (target.hp 0)');
  const teams6 = [
    [mon('Kecleon', ['splash'], { ability: 'Color Change', level: 5 })],
    [mon('Jolteon', ['thunderbolt'], { ability: 'Sturdy', evs: { spe: 252 } })],
  ];
  const r6 = await run(teams6, SEED, [['move 1', 'move 1']]);
  console.log(`ev=${JSON.stringify(r6.perDecision[0].lines.filter((l) => l.includes('typechange') || l.includes('faint')))}`);

  console.log('=== sand-chip immunity via the override: Kecleon -> Rock (Rock Slide) under sand');
  const teams7 = [
    [mon('Kecleon', ['splash'], { ability: 'Color Change' })],
    [mon('Tyranitar', ['rockslide', 'splash'], { ability: 'Sand Stream', evs: { spe: 252 } })],
  ];
  const r7 = await run(teams7, SEED, [['move 1', 'move 1'], ['move 1', 'move 2']], {
    onBoundary: (b) => ({ types: b.p1.active[0].types, hp: b.p1.active[0].hp }),
  });
  r7.perDecision.forEach((d, i) => {
    const ev = d.lines.filter((l) => l.includes('typechange') || l.includes('-damage') && l.includes('Sandstorm'));
    console.log(`t${i + 1}: ev=${JSON.stringify(ev)} ${JSON.stringify(r7.states[i])}`);
  });
}

main().catch((e) => { console.error(e); process.exit(1); });
