// probe_e2e182_simtrace.js — replay the e2e_182 battle through the REAL Showdown
// sim by SEEDING `>start` at the golden's init_seed and submitting the RECORDED
// choice tokens (decoded from the e2e golden's DEC rows). Dump, per decision, the
// omniscient protocol log lines (esp. |-heal|/|-damage|/|-status|/|-curestatus|/
// |move|/|switch|) so the residual ORDER + the exact per-turn Blissey HP timeline
// are visible. The sim is the ONLY oracle.
//
// Usage: node harness/probe_e2e182_simtrace.js [decFrom] [decTo]  (default 26..32)
'use strict';
const path = require('path');
const fs = require('fs');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(PS + '/dist/sim/battle-stream');

const decFrom = process.argv[2] !== undefined ? Number(process.argv[2]) : 26;
const decTo = process.argv[3] !== undefined ? Number(process.argv[3]) : 32;

const golden = fs.readFileSync(path.join(__dirname, '../tests/vectors/e2e_fuzz_golden.txt'), 'utf8').split('\n');
let p1team, p2team, initSeed, decs = [];
for (const line of golden) {
  const f = line.split('\t');
  if (f[0] === 'TEAM' && f[1] === 'e2e_182' && f[2] === 'p1') p1team = f[3];
  if (f[0] === 'TEAM' && f[1] === 'e2e_182' && f[2] === 'p2') p2team = f[3];
  if (f[0] === 'INIT' && f[1] === 'e2e_182') initSeed = f[2];
  if (f[0] === 'DEC' && f[1] === 'e2e_182') decs.push({ idx: Number(f[2]), req: f[3], p1tok: f[6], p2tok: f[7], p2hp: f[22], p2max: f[23], p1hp: f[10] });
}
decs.sort((a, b) => a.idx - b.idx);

function tokToChoice(tok) {
  if (tok === '-') return null;
  const m = tok.match(/^m(\d+)$/); if (m) return `move ${Number(m[1]) + 1}`;
  const s = tok.match(/^s(\d+)$/); if (s) return `switch ${Number(s[1]) + 1}`;
  throw new Error('bad tok ' + tok);
}

const tick = () => new Promise(r => setImmediate(r));

(async () => {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();

  const seedArr = JSON.parse('[' + initSeed + ']');
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":${JSON.stringify(seedArr)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1team })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2team })}`);
  for (let i = 0; i < 12; i++) await tick();

  // The e2e golden's init_seed is the POST-construction seed the Rust engine starts
  // from (start_with_switchins skips the sim's construction draws). Reseed the sim's
  // prng to init_seed here so this replay is bit-identical to the golden generation.
  const { PRNG } = require(PS + '/dist/sim/prng');
  stream.battle.prng = new PRNG(seedArr.slice());

  for (const d of decs) {
    const battle = stream.battle;
    if (battle.ended) break;
    const logLenBefore = log.length;
    const bliss = () => { const s = battle.sides[1]; const a = s.active[0]; return a ? `${a.name} ${a.hp}/${a.maxhp} ${a.status||'-'}` : '?'; };
    const p1a = () => { const s = battle.sides[0]; const a = s.active[0]; return a ? `${a.name} ${a.hp}/${a.maxhp} ${a.status||'-'}` : '?'; };
    // dump p2's request move order (so we can see which slot m2 maps to)
    if (d.idx >= decFrom && d.idx <= decTo) {
      const req = battle.sides[1].activeRequest;
      const mv = (req && req.active && req.active[0] && req.active[0].moves) || [];
      console.log(`\n[dec ${d.idx}] p2 request moves: ${mv.map((m, i) => `${i}:${m.id || m.move}${m.disabled ? '(dis)' : ''}`).join(' ')}`);
    }
    const c1 = tokToChoice(d.p1tok), c2 = tokToChoice(d.p2tok);
    try { if (c1) streams.omniscient.write(`>p1 ${c1}`); } catch (e) {}
    try { if (c2) streams.omniscient.write(`>p2 ${c2}`); } catch (e) {}
    for (let i = 0; i < 16; i++) await tick();
    if (d.idx >= decFrom && d.idx <= decTo) {
      console.log(`\n=== dec ${d.idx} req=${d.req} p1=${d.p1tok}(${c1}) p2=${d.p2tok}(${c2}) -> golden p1hp=${d.p1hp} p2hp=${d.p2hp}/${d.p2max} ===`);
      for (let i = logLenBefore; i < log.length; i++) {
        const l = log[i];
        if (/^\|(move|switch|drag|-heal|-damage|-status|-curestatus|-activate|-cureteam|-fail|upkeep|turn|faint)/.test(l)) console.log('  ', l);
      }
      console.log(`  >> post-decision: p1active=${p1a()} | p2active=${bliss()}`);
    }
  }
  console.log('\n=== ended=', stream.battle.ended, 'winner=', stream.battle.winner, 'decisions replayed');
  try { streams.omniscient.destroy(); } catch (e) {}
})().catch(e => { console.error(e); process.exit(1); });
