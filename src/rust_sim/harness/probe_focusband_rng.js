// probe_focusband_rng.js — settle FOCUS BAND vs the resolved gen3 sim. Hypotheses
// (resolved dist onDamage, priority -40): `if (randomChance(1,10) && damage >= hp &&
// effect.effectType === 'Move') return hp - 1` — the && ORDER means the roll draws
// FIRST, i.e. on EVERY Damage event into the holder (lethal or not, move or chip?),
// and the survive fires only when the (already-drawn) roll passed AND the hit was a
// lethal MOVE. Probes: non-lethal hit (+1 draw?), lethal hit (survive at 1), residual
// burn/sand chip (draw? no survive?), confusion self-hit, recoil to an FB ATTACKER,
// behind a Sub (no draw?), Spikes switch-in, Struggle recoil, whether the deferred-
// faint protocol is bypassed (no faint on survive).
// Run: node src/rust_sim/harness/probe_focusband_rng.js

'use strict';
const { mon, run, fmtCalls } = require('./probe_batch4_lib');

async function main() {
  console.log('=== non-lethal hit: FB vs no-item — one extra draw per hit?');
  const mk = (item) => [
    [mon('Snorlax', ['splash'], { ability: 'Thick Fat', item })],
    [mon('Zangoose', ['slash'], { ability: 'Immunity', evs: { spe: 252 } })],
  ];
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2]]) {
    for (const item of ['Focus Band', '']) {
      const r = await run(mk(item), seed, Array(2).fill(['move 1', 'move 1']));
      r.perDecision.forEach((d, i) => console.log(`nonlethal item=${item || 'none'} seed=${seed[0]} t${i + 1}: [${fmtCalls(d.calls)}]`));
    }
  }

  console.log('=== lethal hit: the survive at 1 HP (lv5 holder, strong attacker)');
  const lethal = [
    [mon('Rattata', ['splash'], { ability: 'Guts', item: 'Focus Band', level: 5 })],
    [mon('Machamp', ['crosschop'], { ability: 'Guts', evs: { atk: 252, spe: 252 } })],
  ];
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9], [10, 10, 10, 10], [11, 11, 11, 11], [12, 12, 12, 12]]) {
    const r = await run(lethal, seed, Array(1).fill(['move 1', 'move 1']), {
      onBoundary: (b) => ({ hp: b.p1.pokemon[0].hp, fainted: b.p1.pokemon[0].fainted }),
    });
    const act = r.perDecision[0].lines.filter((l) => l.includes('Focus Band') || l.includes('faint'));
    console.log(`lethal seed=${seed[0]}: [${fmtCalls(r.perDecision[0].calls)}] ev=${JSON.stringify(act)} ${JSON.stringify(r.states[0])}`);
  }

  console.log('=== residual burn chip on the FB holder: draw? survive?');
  const burn = [
    [mon('Shedinja', ['splash'], { ability: 'Wonder Guard', item: 'Focus Band' })],
    [mon('Dusclops', ['willowisp', 'splash'], { ability: 'Pressure' })],
  ];
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]) {
    const r = await run(burn, seed, Array(1).fill(['move 1', 'move 1']));
    const ev = r.perDecision[0].lines.filter((l) => l.includes('Focus Band') || l.includes('faint'));
    console.log(`burnchip seed=${seed[0]}: [${fmtCalls(r.perDecision[0].calls)}] ev=${JSON.stringify(ev)}`);
  }

  console.log('=== sand chip: FB Shedinja under Sand Stream (chip draw? survive?)');
  const sand = [
    [mon('Shedinja', ['splash'], { ability: 'Wonder Guard', item: 'Focus Band' })],
    [mon('Tyranitar', ['splash'], { ability: 'Sand Stream' })],
  ];
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2]]) {
    const r = await run(sand, seed, Array(1).fill(['move 1', 'move 1']));
    const ev = r.perDecision[0].lines.filter((l) => l.includes('Focus Band') || l.includes('faint'));
    console.log(`sandchip seed=${seed[0]}: [${fmtCalls(r.perDecision[0].calls)}] ev=${JSON.stringify(ev)}`);
  }

  console.log('=== confusion self-hit into an FB holder: draw?');
  const conf = [
    [mon('Snorlax', ['splash'], { ability: 'Thick Fat', item: 'Focus Band' })],
    [mon('Jolteon', ['confuseray', 'splash'], { ability: 'Sturdy', evs: { spe: 252 } })],
  ];
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [5, 5, 5, 5]]) {
    const r = await run(conf, seed, Array(3).fill(['move 1', 'move 2']));
    r.perDecision.forEach((d, i) => console.log(`confusion seed=${seed[0]} t${i + 1}: [${fmtCalls(d.calls)}]`));
  }

  console.log('=== recoil: FB DOUBLE-EDGE attacker (self-recoil draw? survive?)');
  const recoil = [
    [mon('Rattata', ['doubleedge'], { ability: 'Guts', item: 'Focus Band', level: 10 })],
    [mon('Snorlax', ['splash'], { ability: 'Thick Fat' })],
  ];
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2]]) {
    const r = await run(recoil, seed, Array(1).fill(['move 1', 'move 1']));
    const ev = r.perDecision[0].lines.filter((l) => l.includes('Focus Band') || l.includes('faint'));
    console.log(`recoil seed=${seed[0]}: [${fmtCalls(r.perDecision[0].calls)}] ev=${JSON.stringify(ev)}`);
  }

  console.log('=== behind a Sub: FB holder subbed, foe hits the sub — draw?');
  const sub = [
    [mon('Snorlax', ['substitute', 'splash'], { ability: 'Thick Fat', item: 'Focus Band' })],
    [mon('Zangoose', ['slash'], { ability: 'Immunity', evs: { spe: 252 } })],
  ];
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2]]) {
    const r = await run(sub, seed, [['move 1', 'move 1'], ['move 2', 'move 1']]);
    r.perDecision.forEach((d, i) => console.log(`sub seed=${seed[0]} t${i + 1}: [${fmtCalls(d.calls)}]`));
  }

  console.log('=== Spikes switch-in damage onto an FB holder: draw?');
  const spikes = [
    [mon('Zangoose', ['scratch'], { ability: 'Immunity' }), mon('Snorlax', ['splash'], { ability: 'Thick Fat', item: 'Focus Band' })],
    [mon('Skarmory', ['spikes', 'splash'], { ability: 'Keen Eye', evs: { spe: 252 } })],
  ];
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2]]) {
    const r = await run(spikes, seed, [['move 1', 'move 1'], ['switch 2', 'move 2']]);
    r.perDecision.forEach((d, i) => console.log(`spikes seed=${seed[0]} t${i + 1}: [${fmtCalls(d.calls)}]`));
  }

  console.log('=== lethal EXPLOSION into an FB holder + the exploder self-KO');
  const expl = [
    [mon('Rattata', ['splash'], { ability: 'Guts', item: 'Focus Band', level: 5 })],
    [mon('Snorlax', ['explosion'], { ability: 'Thick Fat', evs: { atk: 252 } })],
  ];
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9], [10, 10, 10, 10]]) {
    const r = await run(expl, seed, Array(1).fill(['move 1', 'move 1']), {
      onBoundary: (b) => ({ p1hp: b.p1.pokemon[0].hp, p2fainted: b.p2.pokemon[0].fainted }),
    });
    const ev = r.perDecision[0].lines.filter((l) => l.includes('Focus Band') || l.includes('faint'));
    console.log(`explosion seed=${seed[0]}: [${fmtCalls(r.perDecision[0].calls)}] ev=${JSON.stringify(ev)} ${JSON.stringify(r.states[0])}`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
