// trace_status_secondary_rng.js — INSTRUMENTED Gen-3 PRNG draw tracer for the
// STATUS onBeforeMove draws + the DAMAGING-move SECONDARY draw (the CRUX facet of
// the "secondaries + onBeforeMove status" step).
//
// Goal: pin, for a move used by a STATUSED mon with a SECONDARY-bearing damaging
// move, the EXACT ordered PRNG draw sequence and where each new draw sits relative
// to the existing accuracy/crit/damage draws:
//
//   onBeforeMove status draws (BEFORE accuracy) → accuracy → crit → damage → secondary
//
// We monkey-patch the live PRNG so EVERY draw is logged with its high-level method
// (+ args + result), the number of low-level rng.next() calls it consumed, and a
// sim/* call-site stack — so we can read precisely which line drew, in order.
//
// To get a STATUSED attacker we play turn 1 (a secondary inflicts the status, e.g.
// Body Slam paralyzes), then on turn 2 the now-statused mon moves and we watch its
// onBeforeMove draw fire BEFORE accuracy. We ALSO directly inject status onto a
// fresh mon (`mon.setStatus`) so we can isolate each status (par / slp / frz / cnf)
// without the turn-1 confound, and assert the draw signature is seed-invariant.
//
// Run:  node src/rust_sim/harness/trace_status_secondary_rng.js
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

function isWrapperFrame(ln) {
  if (ln.includes('/sim/prng.js')) return true;
  if (ln.includes('trace_status_secondary_rng.js')) return true;
  if (/at Battle\.(random|randomChance|sample) /.test(ln)) return true;
  return false;
}
function frameStr(ln) {
  const fn = (ln.match(/at ([\w.<>]+) /) || [])[1] || '?';
  const loc = (ln.match(/\/(sim\/[^\s):]+:\d+):\d+/) || ln.match(/\/(data\/[^\s):]+:\d+):\d+/) || [])[1] || '?';
  return `${fn}@${loc}`;
}
function siteOf(depth = 4) {
  const e = {};
  Error.captureStackTrace(e, siteOf);
  const lines = (e.stack || '').split('\n').slice(1).filter((ln) => !isWrapperFrame(ln));
  const sim = lines.filter((ln) => /\/(sim|data)\//.test(ln)).slice(0, depth);
  return sim.map(frameStr).join(' <- ') || '?';
}

function instrument(prng, sink) {
  const rng = prng.rng;
  let nextCount = 0;
  const origNext = rng.next.bind(rng);
  rng.next = function () { nextCount++; return origNext(); };

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
      const logResult = name === 'shuffle' ? `[len=${args[0] && args[0].length}]` : result;
      sink.push({ method: name, args: args.filter((_, i) => name !== 'shuffle' || i > 0), result: logResult, nexts: consumed, site });
      return result;
    };
  }
  const origRandom = prng.random.bind(prng);
  prng.random = function (...args) {
    if (inHigh) return origRandom(...args);
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
  for (let i = 0; i < n; i++) out.push([step() || 1, step() || 1, step() || 1, step() || 1]);
  return out;
}

function startBattle(p1mons, p2mons, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1mons) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2mons) })}`);
  return { stream, streams, log };
}

function fmtDraw(d) {
  if (d.marker) return `  --- ${d.marker} ---`;
  const args = JSON.stringify(d.args);
  let res = d.result;
  if (typeof res === 'number' && !Number.isInteger(res)) res = res.toFixed(6);
  return `  ${d.method.padEnd(12)} args=${String(args).padEnd(11)} -> ${String(res).padEnd(8)} nexts=${d.nexts}\n        @ ${d.site}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// PART A. Direct-injected single status, isolated. We start a battle, inject ONE
// status onto p1's active via setStatus / addVolatile (draw-free apart from slp's
// onStart random(2,6) and confusion's onStart random(2,6), which we LOG), then on
// turn 1 the statused mon uses a SECONDARY damaging move so we see, in one trace:
//   [status onBeforeMove draw] → accuracy → crit → damage → [secondary random(100)]
// p2 is a passive never-miss no-secondary attacker (Swift) so its move adds only
// acc-free crit+dmg and never confounds the ordering we care about.
// ─────────────────────────────────────────────────────────────────────────────

async function traceInjected(label, statusSetup, p1move, seeds) {
  console.log(`\n================ INJECTED STATUS: ${label} ================`);
  const signatures = new Map();
  let firstTrace = null, firstSeed = null, firstInfo = null;
  for (const seed of seeds) {
    const { stream, streams } = startBattle(
      [mon('Snorlax', [p1move], { nature: 'Hardy' })],
      [mon('Snorlax', ['swift'], { nature: 'Hardy' })], // Swift = never-miss, no secondary
      seed,
    );
    for (let i = 0; i < 4; i++) await tick();
    const battle = stream.battle;
    const p1act = battle.sides[0].active[0];

    // Apply the status/volatile. setStatus is draw-free; slp/confusion onStart draw
    // their duration — we instrument AFTER setup so we don't log the duration draw
    // here (it is covered in PART C where we trace setStatus itself).
    const setupInfo = statusSetup(battle, p1act);

    const draws = [];
    instrument(battle.prng, draws);
    draws.push({ marker: `TURN 1 — p1 ${label} uses ${p1move}; p2 Snorlax uses Swift (never-miss, no-secondary)` });

    streams.omniscient.write('>p1 move 1');
    streams.omniscient.write('>p2 move 1');
    for (let i = 0; i < 10; i++) await tick();
    draws.push({ marker: `END TURN 1 (p1 status=${p1act.status || '-'} hp=${p1act.hp}/${p1act.maxhp}; p2 hp=${battle.sides[1].active[0].hp})` });

    const sig = draws.filter((d) => !d.marker).map((d) => `${d.method}@${d.site}`).join(' | ');
    signatures.set(sig, (signatures.get(sig) || 0) + 1);
    if (!firstTrace) { firstTrace = draws; firstSeed = seed; firstInfo = setupInfo; }
    try { streams.omniscient.destroy(); } catch (e) {}
  }
  console.log(`First seed ${JSON.stringify(firstSeed)} (${firstInfo}) full draw trace:`);
  for (const d of firstTrace) console.log(fmtDraw(d));
  console.log(`\n  Distinct structural draw signatures across ${seeds.length} seeds: ${signatures.size}`);
  let i = 0;
  for (const [sig, count] of [...signatures.entries()].sort((a, b) => b[1] - a[1])) {
    console.log(`   [${++i}] x${count}: ${sig || '(no draws)'}`);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// PART B. The SELF-CONTAINED two-turn loop the step targets: Body Slam (30% para
// secondary) inflicts paralysis on turn 1, then on turn 2 the PARALYZED mon draws
// full-para in onBeforeMove BEFORE its accuracy. We trace BOTH turns so the
// secondary-applies-status (turn 1) → status-draws-onBeforeMove (turn 2) chain is
// visible end to end in one battle.
// ─────────────────────────────────────────────────────────────────────────────

async function traceBodySlamLoop(seeds) {
  console.log(`\n================ SELF-CONTAINED: Body Slam para → paralyzed mon draws full-para ================`);
  // Find a seed where turn-1 Body Slam actually paralyzes (30% secondary), so turn 2
  // shows the para onBeforeMove draw. Slow defender (Shuckle) survives the hit.
  for (const seed of seeds) {
    const { stream, streams } = startBattle(
      [mon('Tauros', ['bodyslam'], { nature: 'Hardy' })],       // 30% para secondary
      [mon('Shuckle', ['swift'], { nature: 'Hardy', evs: { hp: 252, def: 252 } })],
      seed,
    );
    for (let i = 0; i < 4; i++) await tick();
    const battle = stream.battle;
    const p2act = battle.sides[1].active[0];

    const draws = [];
    instrument(battle.prng, draws);
    draws.push({ marker: 'TURN 1 — p1 Tauros Body Slam (30% para) vs p2 Shuckle Swift' });
    streams.omniscient.write('>p1 move 1');
    streams.omniscient.write('>p2 move 1');
    for (let i = 0; i < 10; i++) await tick();
    const paraed = p2act.status === 'par';
    draws.push({ marker: `END TURN 1 (p2 Shuckle status=${p2act.status || '-'})` });

    if (!paraed) { try { streams.omniscient.destroy(); } catch (e) {} continue; }

    draws.push({ marker: 'TURN 2 — p2 Shuckle is PARALYZED; both move (watch p2 para onBeforeMove BEFORE its accuracy)' });
    streams.omniscient.write('>p1 move 1');
    streams.omniscient.write('>p2 move 1');
    for (let i = 0; i < 10; i++) await tick();
    draws.push({ marker: `END TURN 2 (p2 status=${p2act.status || '-'} moved? see trace)` });

    console.log(`Seed ${JSON.stringify(seed)} (turn-1 Body Slam paralyzed the target):`);
    for (const d of draws) console.log(fmtDraw(d));
    try { streams.omniscient.destroy(); } catch (e) {}
    return;
  }
  console.log('  (no seed in the pool paralyzed on turn 1 — widen the pool)');
}

// ─────────────────────────────────────────────────────────────────────────────
// PART C. Trace setStatus / addVolatile('confusion') / the sleep+confusion onStart
// DURATION draws directly, so the gen3 sleep counter random(2,6) and confusion
// random(2,6) draw + the (draw-free) setStatus gates are explicit.
// ─────────────────────────────────────────────────────────────────────────────

async function traceStatusSetDraws(seeds) {
  console.log(`\n================ setStatus / onStart DURATION draws (sleep counter, confusion counter) ================`);
  const cases = [
    ['par  (setStatus draw-free)', (b, p) => { p.setStatus('par'); }],
    ['frz  (setStatus draw-free)', (b, p) => { p.setStatus('frz'); }],
    ['slp  (gen3 onStart random(2,6) = 1-4 turn counter)', (b, p) => { p.setStatus('slp'); }],
    ['confusion (onStart random(2,6) = 2-5 turn counter)', (b, p) => { p.addVolatile('confusion'); }],
  ];
  for (const [label, setup] of cases) {
    const sigs = new Map();
    let first = null, firstSeed = null;
    for (const seed of seeds) {
      const { stream, streams } = startBattle(
        [mon('Snorlax', ['swift'], { nature: 'Hardy' })],
        [mon('Snorlax', ['swift'], { nature: 'Hardy' })],
        seed,
      );
      for (let i = 0; i < 4; i++) await tick();
      const battle = stream.battle;
      const p1act = battle.sides[0].active[0];
      const draws = [];
      instrument(battle.prng, draws);
      draws.push({ marker: `apply: ${label}` });
      setup(battle, p1act);
      const sig = draws.filter((d) => !d.marker).map((d) => `${d.method}(${JSON.stringify(d.args)})@${d.site}`).join(' | ');
      sigs.set(sig, (sigs.get(sig) || 0) + 1);
      if (!first) { first = draws; firstSeed = seed; }
      try { streams.omniscient.destroy(); } catch (e) {}
    }
    console.log(`\n  ${label}  [seed ${JSON.stringify(firstSeed)}]`);
    for (const d of first) console.log(fmtDraw(d));
    console.log(`    distinct signatures across ${seeds.length} seeds: ${sigs.size}`);
  }
}

async function main() {
  const seeds = buildSeeds(40);

  // PART A — each status isolated, with a SECONDARY damaging move by the statused mon.
  // Ice Beam (10% freeze) so a frozen/normal mon's secondary is visible; Body Slam
  // (30% para); Rock Slide (30% flinch). p1 (statused) uses the secondary move.
  await traceInjected('paralysis', (b, p) => { p.setStatus('par'); return 'par injected'; }, 'icebeam', seeds);
  await traceInjected('sleep',     (b, p) => { p.setStatus('slp'); return 'slp injected'; }, 'icebeam', seeds);
  await traceInjected('freeze',    (b, p) => { p.setStatus('frz'); return 'frz injected'; }, 'icebeam', seeds);
  await traceInjected('confusion', (b, p) => { p.addVolatile('confusion'); return 'confusion injected'; }, 'icebeam', seeds);
  await traceInjected('flinch',    (b, p) => { p.addVolatile('flinch'); return 'flinch volatile injected'; }, 'icebeam', seeds);
  await traceInjected('none(baseline secondary)', (b, p) => 'no status', 'bodyslam', seeds);

  // PART B — self-contained Body Slam para loop (turn1 inflicts, turn2 the para draw).
  await traceBodySlamLoop(seeds);

  // PART C — the setStatus / onStart duration draws (sleep & confusion counters).
  await traceStatusSetDraws(seeds);
}
main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
