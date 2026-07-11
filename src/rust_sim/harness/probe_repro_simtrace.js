// probe_repro_simtrace.js — replay ONE A/B-fuzzer repro dir through the REAL
// Showdown sim (byte-identical to the recording: same packed teams, battle
// seed, choose seed, same runBattle driver) while instrumenting the PRNG, and
// print every draw (method, args, result, call site) grouped by decision.
//
// The sim is the ONLY oracle: this shows exactly WHICH draws the sim makes in
// the diverging decision so a port desync can be localized by mechanism, not
// guessed. READ-ONLY with respect to the repro dir.
//
// Usage: node harness/probe_repro_simtrace.js <repro-dir> [decFrom] [decTo]
//   decFrom/decTo (inclusive) bound the decisions whose draws are printed
//   (default: all). Run from src/rust_sim.
'use strict';
const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
const e2e = require('./gen_e2e_fuzz.js');

const reproDir = process.argv[2];
if (!reproDir) { console.error('usage: probe_repro_simtrace.js <repro-dir> [decFrom] [decTo]'); process.exit(2); }
const decFrom = process.argv[3] !== undefined ? Number(process.argv[3]) : 0;
const decTo = process.argv[4] !== undefined ? Number(process.argv[4]) : Infinity;

const summary = JSON.parse(fs.readFileSync(path.join(reproDir, 'summary.json'), 'utf8'));

// ---- PRNG instrumentation (global, prototype-level) ----
let decisionNo = -1; // bumped externally via seed-boundary sniffing below
let traceOn = false;
let drawNo = 0;
function site() {
  const lines = new Error().stack.split('\n').slice(2);
  const keep = [];
  for (const l of lines) {
    if (l.includes('/dist/sim/') && !l.includes('prng.js')) {
      const m = l.match(/at ([^ ]+) .*\/dist\/sim\/([^)]+)\)/);
      keep.push(m ? `${m[1]}@${m[2]}` : l.trim());
      if (keep.length >= 4) break;
    }
  }
  return keep.join(' <- ');
}
for (const meth of ['random', 'randomChance', 'sample', 'shuffle']) {
  const orig = PRNG.prototype[meth];
  PRNG.prototype[meth] = function (...args) {
    const before = this.getSeed();
    const r = orig.apply(this, args);
    if (traceOn) {
      const shown = meth === 'shuffle' ? `len=${args[0] ? args[0].length : '?'}` : JSON.stringify(args);
      const res = meth === 'shuffle' ? '' : ` -> ${JSON.stringify(r)}`;
      console.log(`  [dec ${decisionNo}] draw#${drawNo++} ${meth}(${shown})${res} seed_before=${before} :: ${site()}`);
    }
    return r;
  };
}

// runBattle drives decisions internally; we sniff decision boundaries by
// patching console-free hooks is fragile — instead wrap the stream write.
// Simpler: reimplement the loop? No — reuse runBattle but flip traceOn by
// counting seed snapshots: gen_e2e_fuzz's runBattle calls prng.getSeed()
// exactly once BEFORE each decision (seedBefore) and once after. We patch
// getSeed to count "before" calls. Even indices = seedBefore.
const origGetSeed = PRNG.prototype.getSeed;
let getSeedCalls = 0;
PRNG.prototype.getSeed = function () {
  const s = origGetSeed.apply(this);
  return s;
};

(async () => {
  // Patch: bump decisionNo on each seedBefore by intercepting at the driver
  // level. gen_e2e_fuzz exposes runBattle only; we detect decision boundaries
  // by wrapping BattleStream writes of >p1/>p2 — a decision starts when the
  // driver snapshots. Practical approach: trace everything and tag draws by
  // the CURRENT decision derived from a counter we bump when we see the
  // known per-decision write pattern. The driver writes >p1 and/or >p2 then
  // ticks; all draws for the decision happen after those writes. We bump on
  // the FIRST write after a quiet period, which equals: bump on every write
  // of '>p1 ' OR a '>p2 ' not immediately preceded by '>p1 '.
  const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream'));
  const origWrite = BattleStream.prototype._write;
  let lastWasP1 = false;
  BattleStream.prototype._write = function (chunk) {
    const c = String(chunk);
    if (c.startsWith('>p1 ')) { decisionNo++; lastWasP1 = true; traceOn = decisionNo >= decFrom && decisionNo <= decTo; drawNo = 0; if (traceOn) console.log(`--- decision ${decisionNo}: ${c.trim()}`); }
    else if (c.startsWith('>p2 ')) {
      if (!lastWasP1) { decisionNo++; traceOn = decisionNo >= decFrom && decisionNo <= decTo; drawNo = 0; if (traceOn) console.log(`--- decision ${decisionNo}:`); }
      if (traceOn) console.log(`      ${c.trim()}`);
      lastWasP1 = false;
    } else { lastWasP1 = false; }
    return origWrite.apply(this, arguments);
  };

  const rec = await e2e.runBattle(
    summary.packed_teams.p1, summary.packed_teams.p2,
    summary.battle_seed, summary.choose_seed, 'modeled');
  traceOn = false;
  console.log('=== replay done. decisions=', rec.decisions.length, 'ended=', rec.ended, 'winner=', rec.winner, 'dropped=', rec.dropped);
  for (let i = Math.max(0, decFrom - 1); i <= Math.min(rec.decisions.length - 1, decTo === Infinity ? rec.decisions.length - 1 : decTo); i++) {
    const d = rec.decisions[i];
    console.log(`dec ${i}: req=${d.request} p1=${d.choiceP1} p2=${d.choiceP2} seedAfter=${d.seedAfter} fm=${d.firstMover}`);
  }
})().catch(e => { console.error(e); process.exit(1); });
