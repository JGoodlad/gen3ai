// probe_screen_residual_shuffle.js — SIM-oracle confirmation of the BOTH-SIDES-SAME-SCREEN
// residual tie-shuffle (`gen3_screen_residual_tie_shuffle_v1`, the residual sibling of the
// per-hit ModifyDamagePhase1 shuffle). Drives the REAL Showdown sim over the MC112 scenario
// (Cloyster + Snorlax, both set Light Screen turn 0, Icicle Spear turn 1) via the shared
// runBattle + replayChoices, instruments the PRNG, and prints every draw for decision 0.
//
// The tell-tale is a `shuffle(len=2)` at the END of decision 0 (fieldEvent('Residual')'s
// tie-group Fisher-Yates over the two tied Light Screen onSideResidual duration handlers).
// It runs TWO variants: (A) BOTH sides Light Screen, (B) only p1 Light Screen (p2 Splash),
// and asserts (A) draws EXACTLY ONE more `shuffle` at dec0 than (B) — the missing draw.
//
// Run from src/rust_sim:  node harness/probe_screen_residual_shuffle.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const e2e = require('./gen_e2e_fuzz.js');

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, ability, evs) {
  return {
    species, item: '', ability, moves, evs: { ...EV0, ...(evs || {}) }, ivs: IV31,
    nature: 'Serious', level: 100, gender: 'M', // pin gender → no construction draw
  };
}
// MC112 teams (single active mon each is enough — the residual is the whole point).
const P1 = Teams.pack([mon('Cloyster', ['iciclespear', 'lightscreen', 'splash'], 'No Guard', { spa: 252 })]);
const P2LS = Teams.pack([mon('Snorlax', ['lightscreen', 'splash'], 'Immunity', { hp: 252, def: 252 })]);

// ---- PRNG instrumentation (prototype-level), tagged by decision via BattleStream writes ----
let decisionNo = -1;
let traceOn = false;
const drawsAt = {}; // decisionNo -> [ 'random(...)', 'shuffle(len=2)', ... ]
for (const meth of ['random', 'randomChance', 'sample', 'shuffle']) {
  const orig = PRNG.prototype[meth];
  PRNG.prototype[meth] = function (...args) {
    const r = orig.apply(this, args);
    if (traceOn) {
      const shown = meth === 'shuffle' ? `len=${args[0] ? args[0].length : '?'}` : JSON.stringify(args);
      (drawsAt[decisionNo] = drawsAt[decisionNo] || []).push(`${meth}(${shown})`);
    }
    return r;
  };
}
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream'));
const origWrite = BattleStream.prototype._write;
let lastWasP1 = false;
BattleStream.prototype._write = function (chunk) {
  const c = String(chunk);
  if (c.startsWith('>p1 ')) { decisionNo++; lastWasP1 = true; traceOn = true; }
  else if (c.startsWith('>p2 ')) { if (!lastWasP1) { decisionNo++; traceOn = true; } lastWasP1 = false; }
  else lastWasP1 = false;
  return origWrite.apply(this, arguments);
};

async function runVariant(p2Choice0) {
  // reset per-run state
  for (const k of Object.keys(drawsAt)) delete drawsAt[k];
  decisionNo = -1; traceOn = false; lastWasP1 = false;
  // dec0: p1 Light Screen (m2 = slot idx 1), p2 (Light Screen m1 OR Splash m2).
  // dec1: p1 Icicle Spear (m1 = slot idx 0), p2 Splash (m2 = slot idx 1).
  const choices = [['m1', p2Choice0], ['m0', 'm1']];
  const rec = await e2e.runBattle(P1, P2LS, e2e.seedFrom(0xABCDEF), 0x1234, 'pool', { replayChoices: choices });
  return rec;
}

(async () => {
  console.log('=== VARIANT A: BOTH sides Light Screen (dec0 p1=LightScreen, p2=LightScreen) ===');
  const a = await runVariant('m0'); // p2 Light Screen (slot idx 0)
  const aDraws = (drawsAt[0] || []).slice();
  console.log('dec0 draws:', JSON.stringify(aDraws));
  const aShuffleLen2 = aDraws.filter((d) => d === 'shuffle(len=2)').length;
  console.log('dec0 shuffle(len=2) count =', aShuffleLen2, '| ended=', a.ended, 'dropped=', a.dropped);

  console.log('\n=== VARIANT B (control): ONLY p1 Light Screen (dec0 p1=LightScreen, p2=Splash) ===');
  const b = await runVariant('m1'); // p2 Splash (slot idx 1)
  const bDraws = (drawsAt[0] || []).slice();
  console.log('dec0 draws:', JSON.stringify(bDraws));
  const bShuffleLen2 = bDraws.filter((d) => d === 'shuffle(len=2)').length;
  console.log('dec0 shuffle(len=2) count =', bShuffleLen2, '| ended=', b.ended, 'dropped=', b.dropped);

  // A `shuffle(len=2)` internally calls ONE `random(start,end)` (prng.ts:150 — the while
  // loop runs once for a size-2 range), so the instrumentation logs BOTH the `shuffle`
  // wrapper AND its internal `random`. Count only the ACTUAL PRNG advancements
  // (random/randomChance/sample) — the `shuffle` entry is a wrapper, not a separate draw.
  const isDraw = (d) => !d.startsWith('shuffle(');
  const aDrawCount = aDraws.filter(isDraw).length;
  const bDrawCount = bDraws.filter(isDraw).length;
  const delta = aDrawCount - bDrawCount;
  console.log('\n=== RESULT ===');
  console.log('dec0 actual PRNG draws (excl. shuffle wrapper): A(both)=', aDrawCount, 'B(one)=', bDrawCount, 'delta=', delta);
  console.log('dec0 shuffle(len=2) tie-groups: A=', aShuffleLen2, 'B=', bShuffleLen2, 'delta=', aShuffleLen2 - bShuffleLen2);
  const pass = delta === 1 && (aShuffleLen2 - bShuffleLen2) === 1;
  console.log(pass
    ? 'PASS: both-Light-Screen draws EXACTLY ONE MORE residual shuffle (its random(0,2)) than one-Light-Screen — the sim DOES draw the tie-shuffle at the residual.'
    : 'FAIL: the extra draw is NOT a single residual shuffle(len=2) — re-examine.');
})().catch((e) => { console.error(e); process.exit(1); });
