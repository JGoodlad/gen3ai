// trace_multiturn_rng.js — INSTRUMENTED Gen-3 MULTI-TURN PRNG draw tracer.
//
// Purpose (the CRUX investigation for the multi-turn step, NOT a golden
// generator): run ONE real gen3 battle through the omniscient in-process
// BattleStream where BOTH sides repeatedly use a damaging move between BULKY
// mons (so the battle lasts several turns before a faint), with the PRNG
// monkey-patched so EVERY draw is recorded with its HIGH-LEVEL method (+ args +
// result), the number of low-level rng.next() calls it consumed, AND a captured
// sim/* stack frame so we can see EXACTLY which line drew, in order.
//
// What the single-turn tracer (trace_turn_rng.js) deferred and this one closes:
//   (a) the per-action eachEvent('Update') / eachEvent('BeforeTurn') active-mon
//       speed-sort SHUFFLES (battle.ts go()/runAction), which fire ONLY on a
//       speed tie and draw shuffle(L) = L-1 nexts; and
//   (b) the END-OF-TURN RESIDUALS: Leftovers heal (1/16, NO draw), Sandstorm
//       weather chip (1/16, NO draw — but its onFieldResidual calls
//       eachEvent('Weather') which speed-sorts → a tie-shuffle draw), and major
//       status damage (burn 1/8, poison 1/8, Toxic n/16 ramp — all NO draw).
//   plus the per-turn gen3 Quick Claw randomChance(1,5) at endTurn (line 1795).
//
// We print the COMPLETE per-turn draw trace bracketed by turn markers, over
// several full turn cycles, so the per-turn draw ORDER + COUNT is unambiguous.
//
// Run:  node src/rust_sim/harness/trace_multiturn_rng.js
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

// ---- stack-frame capture (which sim/* line drew) -----------------------------
function isWrapperFrame(ln) {
  if (ln.includes('/sim/prng.js')) return true;
  if (ln.includes('trace_multiturn_rng.js')) return true;
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

// ---- PRNG instrumentation: log every high-level draw + its next() count ------
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
      const logResult = name === 'shuffle'
        ? `[shuffled ${args[0] && args[0].length}, range ${args[1] ?? 0}..${args[2] ?? (args[0] && args[0].length)}]`
        : result;
      const logArgs = name === 'shuffle' ? args.slice(1) : args;
      sink.push({ method: name, args: logArgs, result: logResult, nexts: consumed, site });
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

// gen5 numeric seeds for reproducibility
function buildSeeds(n) {
  const out = [];
  let x = 0x12345 >>> 0;
  const step = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  for (let i = 0; i < n; i++) out.push([step() || 1, step() || 1, step() || 1, step() || 1]);
  return out;
}

// ---- scenarios ---------------------------------------------------------------
// All use BULKY mons + a WEAK damaging move with NO secondary chance, so the
// battle survives several turns and the ONLY per-move draws are accuracy + crit
// + damage(16). We vary the residual + tie structure.

// (1) DISTINCT speed, Leftovers both, no weather/status. Isolates: per-action
//     eachEvent shuffles must be ABSENT (no tie) + Leftovers heal (no draw) +
//     endTurn QuickClaw. Snorlax(spe 30 base) vs Skarmory(spe 70) — distinct.
function scLeftoversDistinct() {
  return {
    id: 'leftovers_distinct_speed',
    p1: mon('Snorlax', ['tackle'], { item: 'Leftovers', nature: 'Hardy' }),
    p2: mon('Skarmory', ['tackle'], { item: 'Leftovers', nature: 'Hardy' }),
    choices: () => [['p1', 'move 1'], ['p2', 'move 1']],
    turns: 5,
  };
}

// (2) SPEED TIE, Leftovers both. Forces a tie-shuffle in EACH per-action
//     eachEvent('Update') AND eachEvent('BeforeTurn') AND the action-order sort.
//     Two identical bulky mons.
function scLeftoversTie() {
  return {
    id: 'leftovers_speed_tie',
    p1: mon('Snorlax', ['tackle'], { item: 'Leftovers', nature: 'Hardy' }),
    p2: mon('Snorlax', ['tackle'], { item: 'Leftovers', nature: 'Hardy' }),
    choices: () => [['p1', 'move 1'], ['p2', 'move 1']],
    turns: 5,
  };
}

// (3) SANDSTORM, distinct speed, Leftovers. The sand chip residual's
//     onFieldResidual calls eachEvent('Weather') → another speed-sort over the
//     2 actives (distinct here → NO shuffle draw) + the chip itself draws
//     nothing. Tyranitar's Sand Stream sets permanent sand on switch-in.
function scSandDistinct() {
  return {
    id: 'sand_distinct_speed',
    p1: mon('Tyranitar', ['tackle'], { item: 'Leftovers', ability: 'Sand Stream', nature: 'Hardy' }),
    p2: mon('Skarmory', ['tackle'], { item: 'Leftovers', nature: 'Hardy' }),
    choices: () => [['p1', 'move 1'], ['p2', 'move 1']],
    turns: 5,
  };
}

// (4) SANDSTORM, SPEED TIE. The eachEvent('Weather') speed-sort now ALSO ties →
//     a residual-phase tie-shuffle draw, on TOP of the per-action shuffles.
//     Two identical Tyranitar (one carries Sand Stream; both same speed).
function scSandTie() {
  return {
    id: 'sand_speed_tie',
    p1: mon('Tyranitar', ['tackle'], { item: 'Leftovers', ability: 'Sand Stream', nature: 'Hardy' }),
    p2: mon('Tyranitar', ['tackle'], { item: 'Leftovers', ability: 'Sand Stream', nature: 'Hardy' }),
    choices: () => [['p1', 'move 1'], ['p2', 'move 1']],
    turns: 5,
  };
}

// (5) STATUS residual, distinct speed. One bulky mon is pre-burned via Will-O-Wisp
//     on turn 1 (a status MOVE — deferred from the bit-port, but here we only need
//     the sim to APPLY a status so the burn-DoT residual fires on later turns). We
//     trace turns 2..N where the burn 1/8 chip + Leftovers heal both fire; the
//     point is to PROVE the burn residual draws NOTHING (no extra draw vs scLeftoversDistinct).
//     Gengar (fast) Will-O-Wisp turn 1, then both Tackle. Snorlax bulky target.
function scStatusBurnDistinct() {
  return {
    id: 'burn_residual_distinct',
    // Jolteon (fast, Electric — NOT immune to Normal) burns bulky Blissey turn 1,
    // then both use a NEVER-MISS, non-immune damaging move (Swift / Tackle) so
    // EVERY later turn draws acc+crit+dmg for BOTH + endTurn QuickClaw, and the
    // burn 1/8 chip + Leftovers heal both fire as DRAW-FREE residuals. The point:
    // the per-turn draw COUNT on a burn turn == the no-status turn (7), proving
    // the burn residual draws nothing.
    p1: mon('Jolteon', ['willowisp', 'tackle'], { item: 'Leftovers', nature: 'Timid', evs: { spe: 252 } }),
    p2: mon('Blissey', ['tackle'], { item: 'Leftovers', nature: 'Bold' }),
    choices: (t) => (t === 0 ? [['p1', 'move 1'], ['p2', 'move 1']] : [['p1', 'move 2'], ['p2', 'move 1']]),
    turns: 5,
  };
}

const SCENARIOS = [scLeftoversDistinct, scLeftoversTie, scSandDistinct, scSandTie, scStatusBurnDistinct];

async function runOnce(sc, seed, perTurn) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const protoLog = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) protoLog.push(l); } })();

  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([sc.p1]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([sc.p2]) })}`);
  for (let i = 0; i < 4; i++) await tick();

  // Instrument AFTER setup (the turn-1 quick-claw at the bottom of the >start
  // path is already drawn; the boundary is the first set of choices).
  const draws = [];
  instrument(stream.battle.prng, draws);

  for (let t = 0; t < sc.turns; t++) {
    if (stream.battle.ended) break;
    // mark a turn boundary in the draw stream so the per-turn cut is exact.
    draws.push({ marker: `TURN ${t + 1} CHOICES` });
    for (const [side, choice] of sc.choices(t)) {
      // a fainted/forced side has no move request; guard by checking ended.
      try { streams.omniscient.write(`>${side} ${choice}`); } catch (e) {}
    }
    for (let i = 0; i < 8; i++) await tick();
  }

  // Slice the flat draw stream into per-turn groups by the markers.
  let cur = null;
  for (const d of draws) {
    if (d.marker) { cur = []; perTurn.push({ label: d.marker, draws: cur }); continue; }
    if (cur) cur.push(d);
  }

  const snap = (s) => {
    const a = s.active[0];
    return { species: a.species.name, hp: a.hp, maxhp: a.maxhp, spe: a.storedStats.spe, status: a.status, fainted: a.fainted };
  };
  const out = { p1: snap(stream.battle.sides[0]), p2: snap(stream.battle.sides[1]),
    turn: stream.battle.turn, weather: stream.battle.field.weather, protoLog };
  try { streams.omniscient.destroy(); } catch (e) {}
  return out;
}

function fmtDraw(d) {
  const args = JSON.stringify(d.args);
  let res = d.result;
  if (typeof res === 'number' && !Number.isInteger(res)) res = res.toFixed(6);
  return `    ${d.method.padEnd(12)} args=${String(args).padEnd(10)} -> ${String(res).padEnd(28)} nexts=${d.nexts}\n          @ ${d.site}`;
}

async function traceScenario(scFn, seeds) {
  const sc = scFn();
  console.log(`\n================ SCENARIO: ${sc.id} ================`);

  // Structural signature per turn = ordered (method @ site) list; assert the
  // ORDER+COUNT is identical across seeds (the bit-for-bit-order claim).
  const turnSigSets = new Map(); // turnIndex -> Map(sig -> count)
  let firstPerTurn = null;
  let firstSeed = null;
  let firstRes = null;

  for (const seed of seeds) {
    const perTurn = [];
    let res;
    try { res = await runOnce(sc, seed, perTurn); } catch (e) { console.log('  ERR', e.message); continue; }
    perTurn.forEach((grp, ti) => {
      const sig = grp.draws.map((d) => `${d.method}@${d.site}`).join(' | ');
      if (!turnSigSets.has(ti)) turnSigSets.set(ti, new Map());
      const m = turnSigSets.get(ti);
      m.set(sig, (m.get(sig) || 0) + 1);
    });
    if (!firstPerTurn) { firstPerTurn = perTurn; firstSeed = seed; firstRes = res; }
  }

  console.log(`First seed ${JSON.stringify(firstSeed)} — full per-turn draw trace:`);
  firstPerTurn.forEach((grp) => {
    console.log(`\n  --- ${grp.label} (${grp.draws.length} draws) ---`);
    for (const d of grp.draws) console.log(fmtDraw(d));
  });
  console.log(`\n  Final: p1=${firstRes.p1.species} hp=${firstRes.p1.hp}/${firstRes.p1.maxhp} st=${firstRes.p1.status || '-'} fnt=${firstRes.p1.fainted}`);
  console.log(`         p2=${firstRes.p2.species} hp=${firstRes.p2.hp}/${firstRes.p2.maxhp} st=${firstRes.p2.status || '-'} fnt=${firstRes.p2.fainted}`);
  console.log(`         turn=${firstRes.turn} weather=${firstRes.weather || '-'}`);

  // Per-turn signature stability across seeds.
  console.log(`\n  Per-turn structural-signature stability across ${seeds.length} seeds:`);
  for (const [ti, m] of [...turnSigSets.entries()].sort((a, b) => a[0] - b[0])) {
    const entries = [...m.entries()].sort((a, b) => b[1] - a[1]);
    const label = firstPerTurn[ti] ? firstPerTurn[ti].label : `group ${ti}`;
    if (entries.length === 1) {
      console.log(`   ${label}: 1 signature (STABLE) x${entries[0][1]} — len ${entries[0][0].split(' | ').length}`);
    } else {
      console.log(`   ${label}: ${entries.length} DISTINCT signatures (varies!) ` +
        entries.map(([s, c]) => `[x${c} len ${s ? s.split(' | ').length : 0}]`).join(' '));
      // print the distinct sigs so we see WHAT varies (e.g. a faint truncates).
      entries.forEach(([s, c], i) => console.log(`      sig[${i}] x${c}: ${s}`));
    }
  }
}

async function main() {
  const seeds = buildSeeds(Number(process.argv[2] || 30));
  for (const scFn of SCENARIOS) await traceScenario(scFn, seeds);
}
main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
