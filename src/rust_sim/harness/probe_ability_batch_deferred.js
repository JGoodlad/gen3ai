// probe_ability_batch_deferred.js — confirm the DRAW MODEL of the abilities the batch will
// DEFER (class-c draw-bearing) or must classify carefully (roughskin/colorchange), so the
// deferral is PROVEN, not assumed.
//
// For each, run a probed-vs-control battle (probed ability vs a no-op) and report whether
// the draw count DIFFERS (a new roll / a state-driven cascade) and whether the handler is
// REACHABLE by a modeled contact/status move. A draw-bearing proc DEFERS to batch 2.
//
// Run: node src/rust_sim/harness/probe_ability_batch_deferred.js

'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: opts.nature || 'Serious', level: 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(teamsFn, ability, seed, choices) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of String(ch).split('\n')) lines.push(l); } })();
  const [p1, p2] = teamsFn(ability);
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  let n = 0; rng.next = (...a) => { n += 1; return realNext(...a); };
  const per = [];
  for (const [c1, c2] of choices) {
    const b = n;
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 10; k++) await tick();
    per.push(n - b);
    if (battle.ended) break;
  }
  return { totalDraws: n, per, lines };
}

async function compare(label, teamsFn, probed, control, choices, note) {
  const seeds = [[1, 2, 3, 4], [7, 11, 13, 17], [100, 200, 300, 400], [5, 5, 5, 5], [42, 42, 42, 42], [8, 8, 8, 8]];
  let anyDiff = false;
  const rows = [];
  for (const seed of seeds) {
    const a = await run(teamsFn, probed, seed, choices);
    const b = await run(teamsFn, control, seed, choices);
    const match = a.totalDraws === b.totalDraws && JSON.stringify(a.per) === JSON.stringify(b.per);
    if (!match) anyDiff = true;
    rows.push({ seed, a: a.per, at: a.totalDraws, b: b.per, bt: b.totalDraws, match });
  }
  console.log(`\n### ${label}  [probed=${probed} vs control=${control}]  — ${note}`);
  for (const r of rows) console.log(`   seed ${JSON.stringify(r.seed)}: probed=${JSON.stringify(r.a)}(${r.at}) control=${JSON.stringify(r.b)}(${r.bt}) match=${r.match}`);
  console.log(`   => draw count DIFFERS on some seed (draw-bearing / cascade): ${anyDiff}`);
  return anyDiff;
}

(async () => {
  // CONTACT_PROC: Static (para 1/3 on contact) — Body Slam (contact) into a Static mon.
  await compare('CONTACT_PROC Static', (ab) => [
    [mon('Snorlax', ['bodyslam', 'bodyslam'])],
    [mon('Jolteon', ['recover', 'recover'], { ability: ab })],
  ], 'Static', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], 'contact para proc → new randomChance(1,3)');

  // CONTACT_PROC: Effect Spore (slp/par/psn 1/10 + sample on contact).
  await compare('CONTACT_PROC Effect Spore', (ab) => [
    [mon('Snorlax', ['bodyslam', 'bodyslam'])],
    [mon('Breloom', ['recover', 'recover'], { ability: ab })],
  ], 'Effect Spore', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], 'contact status proc → randomChance(1,10) + sample');

  // Rough Skin (contact recoil baseMaxhp/16 — draw-free? state cascade?).
  await compare('CONTACT Rough Skin', (ab) => [
    [mon('Snorlax', ['bodyslam', 'bodyslam'])],
    [mon('Rhydon', ['rest', 'rest'], { ability: ab })], // Rough Skin isn't on Rhydon; but override works in customgame
  ], 'Rough Skin', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], 'contact recoil damage (no direct PRNG) — cascade check');

  // Color Change (type change on hit — draw-free, but changes types → downstream draws).
  await compare('ON_HIT Color Change', (ab) => [
    [mon('Snorlax', ['bodyslam', 'bodyslam'])], // Normal move → Color Change makes target Normal
    [mon('Kecleon', ['recover', 'recover'], { ability: ab })],
  ], 'Color Change', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], 'type change on hit — downstream draw cascade?');

  // Synchronize (reflect status). A Thunder Wave into a Synchronize mon reflects para back.
  await compare('REFLECT Synchronize', (ab) => [
    [mon('Jolteon', ['thunderwave', 'recover'])],
    [mon('Alakazam', ['recover', 'recover'], { ability: ab })],
  ], 'Synchronize', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], 'reflect status back to source (trySetStatus)');

  // Trace (copy random foe ability at switch-in). Needs a switch to a Trace mon.
  await compare('SWITCH_IN Trace', (ab) => [
    [mon('Snorlax', ['recover', 'recover']), mon('Gengar', ['recover', 'recover'])],
    // p2 leads Porygon2(Trace) — traces p1's Snorlax ability at battle start.
    [mon('Porygon2', ['recover', 'recover'], { ability: ab })],
  ], 'Trace', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1']], 'copy random foe ability (randomFoe — potential draw)');

  // Shed Skin (residual randomChance(33,100) status cure).
  await compare('RESIDUAL Shed Skin', (ab) => [
    [mon('Jolteon', ['thunderwave', 'recover'])], // para the shed skin mon
    [mon('Dragonair', ['recover', 'recover'], { ability: ab })],
  ], 'Shed Skin', 'Shell Armor', [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], 'residual randomChance(33,100) cure');

  console.log('\n\n=> Abilities whose draw count DIFFERS are DRAW-BEARING → DEFER to batch 2. A "false" here means draw-neutral in this scenario (re-check reachability before admitting).');
})();
