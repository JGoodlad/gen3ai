// probe_contact_proc_sub.js — does a CONTACT move into a SUBSTITUTE trigger the contact
// proc (Static)? The sim's runEvent('DamagingHit') fires with `damagedTargets` = targets
// whose damage is a NUMBER. A sub hit sets damage[i]=HIT_SUBSTITUTE then true (not a
// number) — so the target is likely NOT in damagedTargets → NO contact proc. PROBE it.
// Also: does the contact proc fire when the move KO's the target (DamagingHit still fires
// on the KO'd target)? And when the move MISSES (no — no damage). gen3customgame.
// Run: node .../probe_contact_proc_sub.js

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
  console.log('=== Contact (Body Slam) into a Static mon BEHIND A SUB — does the proc fire? ===');
  // p2 Suicune has Substitute + Static. Turn 1 it subs; turn 2 p1 Body Slams the sub.
  for (const seed of [[1, 2, 3, 4], [2, 2, 2, 2]]) {
    const r = await run(
      [mon('Snorlax', ['bodyslam', 'bodyslam'])],
      [mon('Suicune', ['substitute', 'recover'], { ability: 'Static' })],
      seed, [['move 1', 'move 1'], ['move 1', 'move 2']], 'gen3customgame');
    console.log(`  seed ${JSON.stringify(seed)}`);
    console.log(`    dec0 (p2 subs): ${fmtCalls(r.per[0])}`);
    console.log(`    dec1 (p1 BodySlam into sub): ${fmtCalls(r.per[1])}`);
    const proc = r.per[1].some((c) => c.site === 'onDamagingHit');
    console.log(`    => contact-proc randomChance fired behind the sub: ${proc}`);
  }

  console.log('\n=== Contact (Body Slam) that KOs the target — does the proc fire on the KO? ===');
  // A frail Static mon KO'd by Body Slam. The DamagingHit event fires on the KO'd target.
  for (const seed of [[2, 2, 2, 2], [386, 717, 278, 169]]) {
    const r = await run(
      [mon('Snorlax', ['bodyslam', 'bodyslam'], { nature: 'Adamant', evs: { atk: 252 } })],
      [mon('Pichu', ['recover', 'recover'], { ability: 'Static' })], // frail → KO'd
      seed, [['move 1', 'move 1']], 'gen3customgame');
    const proc = r.per[0].some((c) => c.site === 'onDamagingHit');
    const faint = r.lines.some((l) => /faint\|p2a: Pichu/.test(l));
    console.log(`  seed ${JSON.stringify(seed)} KO=${faint} proc-fired=${proc}: ${fmtCalls(r.per[0])}`);
  }
})();
