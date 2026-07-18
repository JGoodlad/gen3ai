// probe_r2_repro_trace.js — drive a saved A/B repro through the sim, instrument
// prng.shuffle + speedSort, and dump the residual shuffle windows + -heal/-damage
// order per decision. Localizes the R2 Leftovers -heal permutation.
//   node harness/probe_r2_repro_trace.js <repro-dir> [decFocus]
'use strict';
const path = require('path');
const fs = require('fs');
const PS = path.resolve('/home/goodlad/dev/gen3ai/deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));

const dir = process.argv[2];
const focus = process.argv[3] !== undefined ? Number(process.argv[3]) : null;
const s = JSON.parse(fs.readFileSync(path.join(dir, 'summary.json'), 'utf8'));
const tick = () => new Promise((r) => setTimeout(r, 0));

function tok(t) {
  if (t === '-' || t === '' || t == null) return null;
  if (t[0] === 'm') return `move ${Number(t.slice(1)) + 1}`;
  if (t[0] === 's') return `switch ${Number(t.slice(1)) + 1}`;
  return null;
}
function hName(h) {
  const eff = h.effect ? (h.effect.name || h.effect.id) : (h.callback ? 'cb' : '?');
  const ident = h.effectHolder && h.effectHolder.fullname ? h.effectHolder.fullname
    : (h.effectHolder && h.effectHolder.name ? h.effectHolder.name : 'field');
  return `${eff}@${ident}(o=${h.order},so=${h.subOrder},sp=${h.speed})`;
}

(async () => {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) if (l) log.push(l); })();
  streams.omniscient.write(`>start {"formatid":"gen3ou","seed":${JSON.stringify(s.battle_seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: s.packed_teams.p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: s.packed_teams.p2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  let curDec = -1;
  const realShuffle = battle.prng.shuffle.bind(battle.prng);
  let shufLog = [];
  battle.prng.shuffle = function (items, start = 0, end = items.length) {
    const before = items.slice(start, end).map((x) => (x && x.effect !== undefined ? hName(x) : (x && x.name ? x.name : String(x))));
    realShuffle(items, start, end);
    const after = items.slice(start, end).map((x) => (x && x.effect !== undefined ? hName(x) : (x && x.name ? x.name : String(x))));
    shufLog.push({ start, end, before, after });
  };
  const realSpeedSort = battle.speedSort.bind(battle);
  battle.speedSort = function (list, cmp) {
    if (focus !== null && curDec === focus && list.length >= 2 && list[0] && list[0].order !== undefined) {
      console.log('  speedSort PRE :', list.map(hName).join(' | '));
    }
    return realSpeedSort(list, cmp);
  };

  const choices = s.choices;
  let i = 0, safety = 0;
  while (!battle.ended && safety < 200) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    curDec = i;
    const pair = choices[Math.min(i, choices.length - 1)];
    const llen = log.length;
    shufLog = [];
    const c1 = tok(pair[0]), c2 = tok(pair[1]);
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 20; k++) await tick();
    if (focus === null || i === focus) {
      const rel = log.slice(llen).filter((l) => /-heal|-damage|-weather|upkeep/.test(l));
      console.log(`--- decision ${i} p1=${pair[0]} p2=${pair[1]} seedAfter=${battle.prng.getSeed()} ---`);
      const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
      console.log(`    p1.speed=${a0 && a0.speed} p2.speed=${a1 && a1.speed}`);
      for (const l of rel) console.log(`    > ${l}`);
      for (const sh of shufLog) console.log(`    SHUFFLE [${sh.start},${sh.end}) pre=[${sh.before.join(', ')}] post=[${sh.after.join(', ')}]`);
    }
    i++;
    if (focus !== null && i > focus) break;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
})();
