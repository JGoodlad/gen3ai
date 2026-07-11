// trace_turn_rng.js — INSTRUMENTED Gen-3 single-turn PRNG draw tracer.
//
// Purpose (the CRUX investigation, NOT a golden generator): run ONE real gen3
// battle through the omniscient in-process BattleStream where BOTH sides use a
// damaging move, with the PRNG monkey-patched so EVERY draw is recorded with its
// HIGH-LEVEL method (+ args + result) AND the number of low-level rng.next()
// calls it consumed. We tag each draw with the call site (a captured stack frame
// filtered to sim/*) so we can see EXACTLY which line drew, in order.
//
// We replay the SAME scenario over many seeds and, per seed, print the draw trace
// bracketed by turn markers (parsed from the omniscient protocol's |turn|N lines
// via a queued boundary) so the per-turn draw ORDER + COUNT is unambiguous, and
// we confirm it is seed-INVARIANT in structure.
//
// Run:  node src/rust_sim/harness/trace_turn_rng.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };

function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Capture the FULL sim/* call chain for a draw (innermost few frames), skipping
// the prng wrappers (prng.js) AND the battle.js random/randomChance/shuffle/sample
// thin forwarders, so we see the TRUE caller (battle-actions.js / the turn loop /
// eachEvent / queue), plus the named JS function for clarity.
function isWrapperFrame(ln) {
  if (ln.includes('/sim/prng.js')) return true;
  if (ln.includes('trace_turn_rng.js')) return true;
  // battle.js thin forwarders Battle.random / Battle.randomChance / Battle.sample
  // / Battle.speedSort (speedSort is NOT a forwarder — keep it) — only skip the
  // 1-line random/randomChance/sample passthroughs at their known dist lines.
  if (/at Battle\.(random|randomChance|sample) /.test(ln)) return true;
  return false;
}
function frameStr(ln) {
  const fn = (ln.match(/at ([\w.<>]+) /) || [])[1] || '?';
  const loc = (ln.match(/\/(sim\/[^\s):]+:\d+):\d+/) || ln.match(/\/(data\/[^\s):]+:\d+):\d+/) || [])[1] || '?';
  return `${fn}@${loc}`;
}
function siteOf(depth = 3) {
  const e = {};
  Error.captureStackTrace(e, siteOf);
  const lines = (e.stack || '').split('\n').slice(1).filter((ln) => !isWrapperFrame(ln));
  const sim = lines.filter((ln) => /\/(sim|data)\//.test(ln)).slice(0, depth);
  return sim.map(frameStr).join(' <- ') || '?';
}

// Monkey-patch a PRNG instance: wrap the high-level methods to log (method,args,
// result, low-level-next-count, call-site). We count rng.next() calls per
// high-level call by wrapping rng.next. We also detect *direct* random() draws
// that aren't via randomChance/sample/shuffle.
function instrument(prng, sink) {
  const rng = prng.rng;
  let nextCount = 0;
  const origNext = rng.next.bind(rng);
  rng.next = function () { nextCount++; return origNext(); };

  const wrap = (name) => {
    const orig = prng[name].bind(prng);
    prng[name] = function (...args) {
      const site = siteOf();
      const before = nextCount;
      const result = orig(...args);
      const consumed = nextCount - before;
      sink.push({ method: name, args, result, nexts: consumed, site });
      return result;
    };
  };
  // randomChance/sample/shuffle internally call this.random — but they call the
  // ORIGINAL bound random captured at wrap time? No: they call `this.random`,
  // which we will also have wrapped. To avoid double-logging, wrap random LAST
  // and have it detect if it's being called from within another wrapped method
  // via a re-entrancy flag.
  let inHigh = false;
  for (const name of ['randomChance', 'sample', 'shuffle']) {
    const orig = prng[name].bind(prng);
    prng[name] = function (...args) {
      const site = siteOf();
      const before = nextCount;
      inHigh = true;
      let result;
      try { result = orig(...args); } finally { inHigh = false; }
      const consumed = nextCount - before;
      // For shuffle, result is the (mutated) array; summarize length instead.
      const logResult = name === 'shuffle' ? `[len=${args[0] && args[0].length}]` : result;
      sink.push({ method: name, args: args.filter((_, i) => name !== 'shuffle' || i > 0), result: logResult, nexts: consumed, site });
      return result;
    };
  }
  const origRandom = prng.random.bind(prng);
  prng.random = function (...args) {
    if (inHigh) return origRandom(...args); // counted under the parent high-level call
    const site = siteOf();
    const before = nextCount;
    const result = origRandom(...args);
    const consumed = nextCount - before;
    sink.push({ method: 'random', args, result, nexts: consumed, site });
    return result;
  };
}

function buildSeeds(n) {
  const out = [];
  let x = 0x12345 >>> 0;
  const step = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  for (let i = 0; i < n; i++) out.push(['sodium', null], [step() || 1, step() || 1, step() || 1, step() || 1]).pop && out.pop();
  // Build gen5 numeric seeds (4 words) for reproducibility.
  out.length = 0;
  x = 0x12345 >>> 0;
  for (let i = 0; i < n; i++) out.push([step() || 1, step() || 1, step() || 1, step() || 1]);
  return out;
}

// Two equal-speed-ish leads that BOTH use a damaging move with NO secondary and
// NO self-boost, so the only per-move draws are accuracy + crit + damage. We pick
// a SPEED TIE scenario (identical Speed) to force the action-order shuffle to
// fire, and contrast with a NON-tie scenario where it must NOT fire.
function scenarioTie() {
  // Two identical-stat mons (same species, same spread) → identical Speed → tie.
  return {
    id: 'speed_tie_both_attack',
    p1: mon('Tauros', ['bodyslam'], { nature: 'Hardy' }),
    p2: mon('Tauros', ['bodyslam'], { nature: 'Hardy' }),
    choices: [['p1', 'move 1'], ['p2', 'move 1']],
  };
}
function scenarioNoTie() {
  // Different Speed → NO action-order shuffle. Jolteon (130 base spe) vs Snorlax.
  // Use no-secondary damaging moves: Jolteon Shadow Ball? has 20% SpD drop. Use
  // Jolteon "doubleedge"? recoil but no secondary chance draw. Snorlax Earthquake.
  return {
    id: 'no_tie_both_attack',
    p1: mon('Jolteon', ['shadowball'], { nature: 'Hardy' }), // has secondary -> shows secondary draw
    p2: mon('Snorlax', ['earthquake'], { nature: 'Hardy' }), // no secondary
    choices: [['p1', 'move 1'], ['p2', 'move 1']],
  };
}
// A clean NO-secondary, NO-tie scenario to isolate the minimal per-move draws.
function scenarioCleanNoTie() {
  return {
    id: 'clean_no_tie_eq_vs_eq',
    p1: mon('Jolteon', ['doubleedge'], { nature: 'Hardy' }), // recoil, no secondary chance
    p2: mon('Snorlax', ['earthquake'], { nature: 'Hardy' }), // no secondary
    choices: [['p1', 'move 1'], ['p2', 'move 1']],
  };
}

async function runOnce(sc, seed, draws) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();

  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([sc.p1]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([sc.p2]) })}`);

  // The battle object exists synchronously after the writes are processed; tick a
  // few times so >start + both >player are consumed and the battle is built.
  for (let i = 0; i < 4; i++) await tick();

  // Instrument the live PRNG and mark the boundary: everything from now is the
  // turn-1 action resolution + the end-of-turn quick-claw roll for turn 2.
  instrument(stream.battle.prng, draws);
  draws.push({ marker: 'BEFORE_CHOICES (battle built, turn-1 quick-claw already drawn during setup)' });

  for (const [side, choice] of sc.choices) streams.omniscient.write(`>${side} ${choice}`);
  for (let i = 0; i < 8; i++) await tick();

  draws.push({ marker: 'AFTER_TURN_1 (both moves resolved; end-of-turn quick-claw for turn 2 drawn)' });

  const snap = (s) => {
    const a = s.active[0];
    return { species: a.species.name, hp: a.hp, maxhp: a.maxhp, spe: a.storedStats.spe, fainted: a.fainted };
  };
  const out = { p1: snap(stream.battle.sides[0]), p2: snap(stream.battle.sides[1]), log };
  try { streams.omniscient.destroy(); } catch (e) {}
  return out;
}

function fmtDraw(d) {
  if (d.marker) return `  --- ${d.marker} ---`;
  const args = JSON.stringify(d.args);
  let res = d.result;
  if (typeof res === 'number' && !Number.isInteger(res)) res = res.toFixed(6);
  return `  ${d.method.padEnd(12)} args=${args.padEnd(10)} -> ${String(res).padEnd(7)} nexts=${d.nexts}\n        @ ${d.site}`;
}

async function traceScenario(sc, seeds) {
  console.log(`\n================ SCENARIO: ${sc.id} ================`);
  const signatures = new Map(); // structural signature -> count
  let firstTrace = null;
  let firstSeed = null;
  for (const seed of seeds) {
    const draws = [];
    let res;
    try { res = await runOnce(sc, seed, draws); } catch (e) { console.log('  ERR', e.message); continue; }
    // Structural signature: ordered list of (method @ site) between the markers.
    const sig = draws.filter((d) => !d.marker).map((d) => `${d.method}@${d.site}`).join(' | ');
    signatures.set(sig, (signatures.get(sig) || 0) + 1);
    if (!firstTrace) { firstTrace = draws; firstSeed = seed; firstRes = res; }
  }
  // Print the first seed's full annotated trace.
  console.log(`First seed ${JSON.stringify(firstSeed)} full draw trace:`);
  for (const d of firstTrace) console.log(fmtDraw(d));
  console.log(`  p1=${firstRes.p1.species} hp=${firstRes.p1.hp}/${firstRes.p1.maxhp} spe=${firstRes.p1.spe} fnt=${firstRes.p1.fainted}`);
  console.log(`  p2=${firstRes.p2.species} hp=${firstRes.p2.hp}/${firstRes.p2.maxhp} spe=${firstRes.p2.spe} fnt=${firstRes.p2.fainted}`);
  console.log(`\n  Distinct structural draw signatures across ${seeds.length} seeds: ${signatures.size}`);
  let i = 0;
  for (const [sig, count] of [...signatures.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`   [${++i}] x${count}: ${sig}`);
  }
}

let firstRes = null;
async function main() {
  const seeds = buildSeeds(40);
  await traceScenario(scenarioTie(), seeds);
  await traceScenario(scenarioCleanNoTie(), seeds);
  await traceScenario(scenarioNoTie(), seeds);
}
main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
