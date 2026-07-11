// probe_contact_proc_lands.js — settle the LANDED-proc draw model + the gen3ou SetStatus
// shuffle interaction for the CONTACT_PROC class. The count probe showed one randomChance
// per contact hit; this nails: (1) when the proc's randomChance PASSES, does trySetStatus
// draw the gen3ou SetStatus 2-clause handler-sort shuffle? (2) Effect Spore: on a PASS, the
// sample(3) fires, THEN (if the sampled status lands) does trySetStatus draw the shuffle?
// (3) does a proc into an already-statused / type-immune / ability-immune ATTACKER still
// draw the randomChance (yes) but not the shuffle?  gen3ou (sleep+freeze clause) format.
//
// Run: node src/rust_sim/harness/probe_contact_proc_lands.js

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
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: opts.nature || 'Serious',
    level: 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }
function siteLabel(stack) {
  const lines = String(stack).split('\n').slice(2);
  for (const l of lines) {
    const m = l.match(/at (?:Battle|BattleActions|Pokemon|Side|Field)\.?(\w+)/);
    if (m && !['random', 'randomChance', 'sample', 'shuffle', 'speedSort'].includes(m[1])) return m[1];
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
  const wrap = (name) => {
    const orig = prng[name].bind(prng);
    prng[name] = (...a) => { const r = orig(...a); calls.push({ kind: name, args: a, site: siteLabel(new Error().stack), ret: r }); return r; };
  };
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
  // Find a seed where Static's randomChance(1,3) PASSES on the FIRST Body Slam, in gen3ou,
  // so we can see if trySetStatus(par) then draws the SetStatus 2-clause shuffle.
  console.log('=== Static PASS in gen3ou: does trySetStatus draw the SetStatus shuffle? ===');
  for (const seed of [[1, 2, 3, 4], [7, 11, 13, 17], [2, 2, 2, 2], [9, 9, 9, 9], [3, 14, 15, 92]]) {
    const r = await run(
      [mon('Snorlax', ['bodyslam', 'bodyslam'])],
      [mon('Suicune', ['recover', 'recover'], { ability: 'Static' })],
      seed, [['move 1', 'move 1']], 'gen3ou');
    // Did par land on the attacker (Snorlax)?
    const parLanded = r.lines.some((l) => /-status\|p1a: Snorlax\|par\|\[from\] ability: Static/.test(l));
    console.log(`  seed ${JSON.stringify(seed)} parLanded=${parLanded}: ${fmtCalls(r.per[0])}`);
  }

  console.log('\n=== Static PASS in gen3customgame (no clauses): trySetStatus draws NO shuffle ===');
  for (const seed of [[1, 2, 3, 4], [7, 11, 13, 17], [2, 2, 2, 2]]) {
    const r = await run(
      [mon('Snorlax', ['bodyslam', 'bodyslam'])],
      [mon('Suicune', ['recover', 'recover'], { ability: 'Static' })],
      seed, [['move 1', 'move 1']], 'gen3customgame');
    const parLanded = r.lines.some((l) => /-status\|p1a: Snorlax\|par\|\[from\] ability: Static/.test(l));
    console.log(`  seed ${JSON.stringify(seed)} parLanded=${parLanded}: ${fmtCalls(r.per[0])}`);
  }

  console.log('\n=== Effect Spore: on a PASS, sample(3) fires, then trySetStatus (customgame) ===');
  for (const seed of [[1, 2, 3, 4], [7, 11, 13, 17], [2, 2, 2, 2], [9, 9, 9, 9], [3, 14, 15, 92], [40, 41, 42, 43], [5, 6, 7, 8]]) {
    const r = await run(
      [mon('Snorlax', ['bodyslam', 'bodyslam'])],
      [mon('Suicune', ['recover', 'recover'], { ability: 'Effect Spore' })],
      seed, [['move 1', 'move 1']], 'gen3customgame');
    const st = r.lines.filter((l) => /-status\|p1a: Snorlax/.test(l));
    console.log(`  seed ${JSON.stringify(seed)} attackerStatus=${JSON.stringify(st)}: ${fmtCalls(r.per[0])}`);
  }

  console.log('\n=== Static into a FIRE-type attacker (brn-immune analog): a Flame Body into a Fire mon ===');
  // Flame Body brn into a Fire-type attacker (Charizard) — the randomChance draws, but brn
  // can\'t land (Fire immune to brn). Confirm the randomChance still draws + no shuffle.
  for (const seed of [[1, 2, 3, 4]]) {
    const r = await run(
      [mon('Charizard', ['bodyslam', 'bodyslam'], { nature: 'Adamant' })],
      [mon('Suicune', ['recover', 'recover'], { ability: 'Flame Body' })],
      seed, [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], 'gen3customgame');
    console.log(`  Fire attacker vs Flame Body, dec0: ${fmtCalls(r.per[0])}`);
    console.log(`    (Charizard is Fire → brn immune; the randomChance still draws, no status)`);
  }

  console.log('\n=== Static into an ALREADY-PARALYZED attacker: randomChance draws, no re-status ===');
  // Pre-para the attacker (Snorlax) via Thunder Wave then Body Slam. Actually easier: give a
  // Static mon on BOTH sides — no. Just Body Slam repeatedly; once par lands, later procs
  // draw randomChance but no re-apply.
  for (const seed of [[2, 2, 2, 2]]) {
    const r = await run(
      [mon('Snorlax', ['bodyslam', 'bodyslam'])],
      [mon('Suicune', ['recover', 'recover'], { ability: 'Static' })],
      seed, [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']], 'gen3customgame');
    for (let i = 0; i < r.per.length; i++) console.log(`  dec${i}: ${fmtCalls(r.per[i])}`);
    console.log(`  attacker status lines: ${JSON.stringify(r.lines.filter((l) => /-status\|p1a/.test(l)))}`);
  }
})();
