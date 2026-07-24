// probe_soundproof_damaging.js — settle the draw model + emission of a DAMAGING sound move
//   (Hyper Voice, flags.sound, bp 90) into a SOUNDPROOF holder. Does the blocked move draw
//   its accuracy roll THEN emit `-immune` (like a type-immune / Wonder-Guard block), or is it
//   blocked BEFORE accuracy? The sim is the only oracle.
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
  const seeds = [[1, 2, 3, 4], [7, 11, 13, 17], [42, 42, 42, 42], [99, 3, 55, 200]];
  // p1 Snorlax uses Hyper Voice; p2 target with Soundproof (Mr. Mime) vs control (No Ability Mr. Mime).
  const attacker = [mon('Snorlax', ['hypervoice', 'tackle'])];
  const soundproof = [mon('Mr. Mime', ['calmmind', 'tackle'], { ability: 'Soundproof' })];
  const control = [mon('Mr. Mime', ['calmmind', 'tackle'], { ability: 'No Ability' })];
  // choices: p1 Hyper Voice (move 1), p2 Calm Mind (move 1) — p2's move is draw-free/never-miss so
  // the delta isolates Hyper Voice's draws.
  const choices = [['move 1', 'move 1']];
  for (const seed of seeds) {
    const sp = await run(attacker, soundproof, seed, choices, 'gen3customgame');
    const ct = await run(attacker, control, seed, choices, 'gen3customgame');
    const spN = sp.per[0].length, ctN = ct.per[0].length;
    console.log(`seed ${JSON.stringify(seed)}: soundproof=${spN} control=${ctN}`);
    console.log('  soundproof draws:', fmtCalls(sp.per[0]));
    console.log('  control    draws:', fmtCalls(ct.per[0]));
    // emission around the hypervoice move:
    const emitSP = sp.lines.filter((l) => /Snorlax|Mr\. Mime|immune|damage|move\|/.test(l) && !/^\|t:/.test(l));
    console.log('  soundproof emission:', JSON.stringify(emitSP.filter((l) => /Hyper Voice|immune|-damage/.test(l))));
  }
})();
