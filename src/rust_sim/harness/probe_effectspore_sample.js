// probe_effectspore_sample.js — find seeds where Effect Spore's randomChance(1,10) PASSES,
// to nail the NESTED draw: random(10) gate -> sample(3) [slp/par/psn] -> trySetStatus. And
// confirm: a sampled SLEEP draws the slp.onStart random(2,6); a sampled status in gen3ou
// draws the SetStatus 2-clause shuffle. Run: node .../probe_effectspore_sample.js

'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves, evs: { ...EV0 }, ivs: IV31, nature: 'Serious', level: 100, gender: 'N' };
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
  streams.omniscient.write(`>start {"formatid":"${fmt}","seed":${JSON.stringify(seed)}}`);
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
  console.log('=== Effect Spore PASS hunt (customgame) — 40 seeds, print any with a sample(3) ===');
  let found = 0;
  for (let s = 0; s < 400 && found < 8; s++) {
    const seed = [s * 7 + 1, s * 13 + 2, s * 5 + 3, s * 3 + 4];
    const r = await run(
      [mon('Snorlax', ['bodyslam', 'bodyslam'])],
      [mon('Suicune', ['recover', 'recover'], { ability: 'Effect Spore' })],
      seed, [['move 1', 'move 1']], 'gen3customgame');
    const hasSample = r.per[0].some((c) => c.kind === 'sample');
    if (hasSample) {
      found++;
      const st = r.lines.filter((l) => /-status\|p1a: Snorlax/.test(l));
      console.log(`  seed ${JSON.stringify(seed)} status=${JSON.stringify(st)}:\n    ${fmtCalls(r.per[0])}`);
    }
  }
  if (!found) console.log('  (no pass found in 400 seeds — rare 1/10)');

  console.log('\n=== Effect Spore PASS hunt (gen3ou) — confirm sample -> trySetStatus -> SetStatus shuffle ===');
  found = 0;
  for (let s = 0; s < 400 && found < 6; s++) {
    const seed = [s * 7 + 1, s * 13 + 2, s * 5 + 3, s * 3 + 4];
    const r = await run(
      [mon('Snorlax', ['bodyslam', 'bodyslam'])],
      [mon('Suicune', ['recover', 'recover'], { ability: 'Effect Spore' })],
      seed, [['move 1', 'move 1']], 'gen3ou');
    const hasSample = r.per[0].some((c) => c.kind === 'sample');
    if (hasSample) {
      found++;
      const st = r.lines.filter((l) => /-status\|p1a: Snorlax/.test(l));
      console.log(`  seed ${JSON.stringify(seed)} status=${JSON.stringify(st)}:\n    ${fmtCalls(r.per[0])}`);
    }
  }
})();
