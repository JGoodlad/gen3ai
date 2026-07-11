// probe_synchronize_rng.js — settle Synchronize's draw model precisely (a STRETCH for
// batch 2). onAfterSetStatus: when the holder is inflicted a major status by a FOE, reflect
// it back to the source (slp/frz EXEMPT, tox→psn). The crux: does the reflected trySetStatus
// re-enter runEvent('SetStatus') → another 2-clause shuffle (gen3ou)? What's its POSITION
// (right after the holder's setStatus)? Does it draw a sleep random(2,6)? (No — slp exempt.)
// customgame + gen3ou. Run: node .../probe_synchronize_rng.js

'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves, evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: opts.nature || 'Serious', level: 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }
function siteLabel(stack) {
  for (const l of String(stack).split('\n').slice(2)) {
    const m = l.match(/at (?:Battle|BattleActions|Pokemon|Side|Field)\.?(\w+)/);
    if (m && !['random', 'randomChance', 'sample', 'shuffle'].includes(m[1])) return m[1];
  }
  return '?';
}
async function run(p1, p2, seed, choices, fmt) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of String(ch).split('\n')) lines.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${fmt || 'gen3customgame'}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 14; i++) await tick();
  const battle = stream.battle;
  const prng = battle.prng;
  const calls = [];
  const wrap = (name) => { const orig = prng[name].bind(prng); prng[name] = (...a) => { const r = orig(...a); calls.push({ kind: name, args: a, site: siteLabel(new Error().stack), ret: r }); return r; }; };
  wrap('random'); wrap('randomChance'); wrap('sample');
  const per = [];
  for (const [c1, c2] of choices) {
    const before = calls.length;
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 12; k++) await tick();
    per.push(calls.slice(before));
    if (battle.ended) break;
  }
  return { per, lines };
}
function fmtCalls(cs) { return cs.map((c) => `${c.kind}(${JSON.stringify(c.args)})@${c.site}=>${JSON.stringify(c.ret)}`).join('  '); }

(async () => {
  // p1 Thunder Wave's the Synchronize holder (p2). Para reflects back to p1.
  console.log('=== Synchronize: Thunder Wave (par) into holder → reflect par to source (customgame) ===');
  for (const seed of [[1, 2, 3, 4], [7, 11, 13, 17]]) {
    const a = await run(
      [mon('Jolteon', ['thunderwave', 'thunderwave'])],
      [mon('Alakazam', ['recover', 'recover'], { ability: 'Synchronize' })],
      seed, [['move 1', 'move 1']], 'gen3customgame');
    const b = await run(
      [mon('Jolteon', ['thunderwave', 'thunderwave'])],
      [mon('Alakazam', ['recover', 'recover'], { ability: 'Shell Armor' })],
      seed, [['move 1', 'move 1']], 'gen3customgame');
    console.log(`  seed ${JSON.stringify(seed)}`);
    console.log(`    SYNC   : ${fmtCalls(a.per[0])}`);
    console.log(`    control: ${fmtCalls(b.per[0])}`);
    console.log(`    sync status lines: ${JSON.stringify(a.lines.filter((l) => /-status/.test(l)))}`);
  }

  console.log('\n=== Synchronize gen3ou (does the reflected setStatus draw its OWN clause shuffle?) ===');
  for (const seed of [[1, 2, 3, 4], [7, 11, 13, 17]]) {
    const a = await run(
      [mon('Jolteon', ['thunderwave', 'thunderwave'])],
      [mon('Alakazam', ['recover', 'recover'], { ability: 'Synchronize' })],
      seed, [['move 1', 'move 1']], 'gen3ou');
    console.log(`  seed ${JSON.stringify(seed)} SYNC: ${fmtCalls(a.per[0])}`);
    console.log(`    status lines: ${JSON.stringify(a.lines.filter((l) => /-status/.test(l)))}`);
  }

  console.log('\n=== Synchronize: Toxic (tox) → reflect as psn (tox→psn); Will-O-Wisp (brn) → brn ===');
  for (const [move, lbl] of [['toxic', 'Toxic→psn'], ['willowisp', 'WoW→brn']]) {
    const a = await run(
      [mon('Gengar', [move, move])],
      [mon('Alakazam', ['recover', 'recover'], { ability: 'Synchronize' })],
      [1, 2, 3, 4], [['move 1', 'move 1']], 'gen3customgame');
    console.log(`  ${lbl}: ${fmtCalls(a.per[0])}`);
    console.log(`    status lines: ${JSON.stringify(a.lines.filter((l) => /-status/.test(l)))}`);
  }

  console.log('\n=== Synchronize: Spore (slp) → NO reflect (slp exempt) ===');
  {
    const a = await run(
      [mon('Breloom', ['spore', 'spore'])],
      [mon('Alakazam', ['recover', 'recover'], { ability: 'Synchronize' })],
      [1, 2, 3, 4], [['move 1', 'move 1']], 'gen3customgame');
    console.log(`  Spore: ${fmtCalls(a.per[0])}`);
    console.log(`    status lines: ${JSON.stringify(a.lines.filter((l) => /-status/.test(l)))}`);
  }

  console.log('\n=== Synchronize: reflected status into an IMMUNE source (para into a Ground? no; a burn into a Fire source) ===');
  // Flame Body... no. Will-O-Wisp'd holder reflects brn to a FIRE-type source → immune.
  {
    const a = await run(
      [mon('Charizard', ['willowisp', 'willowisp'], { nature: 'Modest' })],
      [mon('Alakazam', ['recover', 'recover'], { ability: 'Synchronize' })],
      [1, 2, 3, 4], [['move 1', 'move 1']], 'gen3customgame');
    console.log(`  WoW from Fire source (reflect brn→Fire immune): ${fmtCalls(a.per[0])}`);
    console.log(`    status lines: ${JSON.stringify(a.lines.filter((l) => /-status/.test(l)))}`);
  }
})();
