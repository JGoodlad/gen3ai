// trace_switch_rng.js — INSTRUMENTED Gen-3 SWITCH-TURN PRNG draw tracer.
//
// Purpose (the CRUX investigation for the SWITCHING step, NOT a golden
// generator): run real gen3 battles through the omniscient in-process
// BattleStream where one or both sides SWITCH (voluntary) and where a mon FAINTS
// and is REPLACED (post-faint forced switch), with the PRNG monkey-patched so
// EVERY draw is recorded with its HIGH-LEVEL method (+ args + result), the
// number of low-level rng.next() calls it consumed, AND a captured sim/* stack
// frame so we can see EXACTLY which line drew, in order.
//
// What we want to nail down (the FACET = "switch action order + draws"):
//   (1) where SWITCHES sort relative to MOVES (battle-queue order: switch=103,
//       instaswitch=3, move=200 → switches resolve BEFORE moves);
//   (2) the switch speed-tie shuffle: two switches at EQUAL speed → does the
//       action-order speed_sort draw ONE shuffle (random(0,2))? does a switch
//       itself draw anything? what about the runSwitch SwitchIn speed_sort
//       (battle-actions.ts:182 speedSort(allActive)) — does it draw on a tie?
//   (3) the per-action eachEvent shuffles AROUND switch actions (the gen<5
//       end-of-runAction Update tail) — present only on a speed tie;
//   (4) the POST-FAINT replacement: does the forced (instaswitch) replacement
//       draw anything? where does it sort? does the turn resume / end?
//
// We compare three structures:
//   A. BOTH sides switch (voluntary) — speed-tie and distinct variants.
//   B. one side SWITCHES, the other MOVES — proves switch-before-move ordering.
//   C. one mon FAINTS to a move, then the forced replacement comes in — proves
//      the instaswitch replacement draw count + that run_battle continues.
//
// Run:  node src/rust_sim/harness/trace_switch_rng.js [nSeeds]
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
  if (ln.includes('trace_switch_rng.js')) return true;
  if (/at Battle\.(random|randomChance|sample) /.test(ln)) return true;
  return false;
}
function frameStr(ln) {
  const fn = (ln.match(/at ([\w.<>]+) /) || [])[1] || '?';
  const loc = (ln.match(/\/(sim\/[^\s):]+:\d+):\d+/) || ln.match(/\/(data\/[^\s):]+:\d+):\d+/) || [])[1] || '?';
  return `${fn}@${loc}`;
}
function siteOf(depth = 5) {
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
// Each scenario is a 2-or-3 mon team per side + a per-turn script. We trace the
// per-turn draw stream (markers) AND the protocol lines (so we can see the
// |switch|, |faint|, |request|forceSwitch and turn boundaries).

// (A1) BOTH sides voluntary-switch, DISTINCT lead speeds.
//   Tests: switches sort before any move (none here), the action-order speed_sort
//   over the two switch actions is DISTINCT (no shuffle draw), each runSwitch's
//   SwitchIn speedSort over the 2 NEW actives — distinct or tie depending on the
//   INCOMING mons. We make the incoming mons DISTINCT speed too → 0 shuffle draws.
function scBothSwitchDistinct() {
  return {
    id: 'both_switch_distinct',
    // p1 lead Snorlax(spe 30) → switch to Jolteon(spe 130). p2 lead Skarmory(spe 70)
    // → switch to Forretress(spe 40). All four distinct → no ties anywhere.
    p1: [mon('Snorlax', ['tackle']), mon('Jolteon', ['tackle'])],
    p2: [mon('Skarmory', ['tackle']), mon('Forretress', ['tackle'])],
    script: [
      { label: 'BOTH SWITCH', p1: 'switch 2', p2: 'switch 2' },
    ],
  };
}

// (A2) BOTH sides voluntary-switch, where the TWO OUTGOING leads TIE on speed
//   (forces the action-order speed_sort tie-shuffle: 2 equal-speed switch actions
//   → 1 random(0,2)) AND the TWO INCOMING mons TIE too (forces the runSwitch
//   SwitchIn speedSort tie-shuffle). Two Snorlax-lead, two Snorlax-incoming.
function scBothSwitchTie() {
  return {
    id: 'both_switch_speed_tie',
    p1: [mon('Snorlax', ['tackle']), mon('Snorlax', ['tackle'])],
    p2: [mon('Snorlax', ['tackle']), mon('Snorlax', ['tackle'])],
    script: [
      { label: 'BOTH SWITCH (tie)', p1: 'switch 2', p2: 'switch 2' },
    ],
  };
}

// (B) one side SWITCHES, the other MOVES. Proves the switch resolves BEFORE the
//   move (order 103 < 200) regardless of speed. p1 switches (slow Snorlax→Jolteon),
//   p2 attacks with fast Skarmory. The switch must come first.
function scSwitchVsMove() {
  return {
    id: 'switch_vs_move',
    p1: [mon('Snorlax', ['tackle']), mon('Jolteon', ['tackle'])],
    p2: [mon('Skarmory', ['tackle']), mon('Forretress', ['tackle'])],
    script: [
      { label: 'P1 SWITCH, P2 MOVE', p1: 'switch 2', p2: 'move 1' },
    ],
  };
}

// (C) POST-FAINT replacement. A FAST glass attacker OHKOs a FRAIL opp lead; the
//   opp must send in a replacement (forced instaswitch). We trace the move turn
//   (with the faint + makeRequest('switch')) AND the replacement submission, to
//   see whether the instaswitch draws and how the turn resumes/ends.
//   Jolteon (fast) Thunderbolt OHKOs a frail Magikarp; opp replaces with Snorlax.
function scPostFaintReplace() {
  return {
    id: 'post_faint_replace',
    p1: [mon('Jolteon', ['thunderbolt'], { evs: { spa: 252, spe: 252 }, nature: 'Timid' })],
    p2: [mon('Magikarp', ['splash']), mon('Snorlax', ['tackle'])],
    script: [
      // turn 1: p1 KOs p2's Magikarp. p2 will then get a forceSwitch request.
      { label: 'P1 KOs P2 LEAD', p1: 'move 1', p2: 'move 1' },
      // p2 replacement (forced). p1 has no choice (it already moved / mon alive).
      { label: 'P2 REPLACES (forced)', p2: 'switch 2' },
    ],
  };
}

// (D) DOUBLE-FAINT replacement (both leads KO each other same turn via a tie).
//   Two equal-speed frail mons both OHKO each other → both sides get a forceSwitch
//   the same turn. Tests the both-replace path + whether the replacement runSwitch
//   SwitchIn speedSort ties.
function scDoubleFaintReplace() {
  return {
    id: 'double_faint_replace',
    // Two Electrode (very fast, frail) using Explosion would be cleaner, but
    // Explosion is a status-ish self-KO; instead use two equal-speed frail mons
    // that each OHKO the other with a strong neutral move. Use two Deoxys-Attack
    // analogues? Keep it simple: two Glass cannons that tie + OHKO.
    p1: [mon('Electrode', ['explosion'], { evs: { atk: 252 } }), mon('Snorlax', ['tackle'])],
    p2: [mon('Electrode', ['explosion'], { evs: { atk: 252 } }), mon('Snorlax', ['tackle'])],
    script: [
      { label: 'EXPLOSION DOUBLE-KO', p1: 'move 1', p2: 'move 1' },
      { label: 'BOTH REPLACE', p1: 'switch 2', p2: 'switch 2' },
    ],
  };
}

const SCENARIOS = [
  scBothSwitchDistinct, scBothSwitchTie, scSwitchVsMove, scPostFaintReplace, scDoubleFaintReplace,
];

async function runOnce(sc, seed, perTurn, captureProto) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const protoLog = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) protoLog.push(l); } })();

  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(sc.p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(sc.p2) })}`);
  for (let i = 0; i < 4; i++) await tick();

  // Instrument AFTER setup (the turn-1 quick-claw at the bottom of the >start
  // path is already drawn; the boundary is the first set of choices).
  const draws = [];
  instrument(stream.battle.prng, draws);

  // Wrap runAction so each draw is tagged with the action that produced it, and
  // emit an explicit ACTION marker before each runAction (so the per-action
  // shuffle attribution is unambiguous).
  const battle = stream.battle;
  const origRunAction = battle.runAction.bind(battle);
  battle.runAction = function (action) {
    const who = action.pokemon ? `${action.pokemon.side.id}:${action.pokemon.species.name}` : '';
    draws.push({ actionMarker: `${action.choice}${who ? ' ' + who : ''}` });
    return origRunAction(action);
  };

  for (const step of sc.script) {
    if (stream.battle.ended) break;
    draws.push({ marker: step.label, requestState: stream.battle.requestState });
    if (step.p1) { try { streams.omniscient.write(`>p1 ${step.p1}`); } catch (e) {} }
    if (step.p2) { try { streams.omniscient.write(`>p2 ${step.p2}`); } catch (e) {} }
    for (let i = 0; i < 10; i++) await tick();
  }

  // Slice the flat draw stream into per-step groups by the markers; within a
  // group, keep the actionMarkers inline so we can see which runAction drew.
  let cur = null;
  for (const d of draws) {
    if (d.marker) { cur = []; perTurn.push({ label: d.marker, requestState: d.requestState, draws: cur }); continue; }
    if (cur) cur.push(d);
  }

  const snap = (s) => {
    const a = s.active[0];
    return a ? { species: a.species.name, hp: a.hp, maxhp: a.maxhp, spe: a.storedStats.spe, status: a.status, fainted: a.fainted } : null;
  };
  const out = {
    p1: snap(stream.battle.sides[0]), p2: snap(stream.battle.sides[1]),
    turn: stream.battle.turn, ended: stream.battle.ended,
    winner: stream.battle.winner, protoLog,
  };
  if (captureProto) out.proto = protoLog.slice();
  try { streams.omniscient.destroy(); } catch (e) {}
  return out;
}

function fmtDraw(d) {
  const args = JSON.stringify(d.args);
  let res = d.result;
  if (typeof res === 'number' && !Number.isInteger(res)) res = res.toFixed(6);
  return `    ${d.method.padEnd(12)} args=${String(args).padEnd(10)} -> ${String(res).padEnd(20)} nexts=${d.nexts}\n          @ ${d.site}`;
}

async function traceScenario(scFn, seeds) {
  const sc = scFn();
  console.log(`\n================ SCENARIO: ${sc.id} ================`);

  const turnSigSets = new Map(); // stepIndex -> Map(sig -> count)
  let firstPerTurn = null;
  let firstSeed = null;
  let firstRes = null;

  for (const seed of seeds) {
    const perTurn = [];
    let res;
    try { res = await runOnce(sc, seed, perTurn, !firstPerTurn); } catch (e) { console.log('  ERR', e.message); continue; }
    perTurn.forEach((grp, ti) => {
      const sig = grp.draws.filter((d) => !d.actionMarker).map((d) => `${d.method}@${d.site}`).join(' | ');
      if (!turnSigSets.has(ti)) turnSigSets.set(ti, new Map());
      const m = turnSigSets.get(ti);
      m.set(sig, (m.get(sig) || 0) + 1);
    });
    if (!firstPerTurn) { firstPerTurn = perTurn; firstSeed = seed; firstRes = res; }
  }

  console.log(`First seed ${JSON.stringify(firstSeed)} — full per-step draw trace:`);
  firstPerTurn.forEach((grp) => {
    const nDraws = grp.draws.filter((d) => !d.actionMarker).length;
    console.log(`\n  --- ${grp.label} [requestState before submit: '${grp.requestState}'] (${nDraws} draws) ---`);
    for (const d of grp.draws) {
      if (d.actionMarker) { console.log(`    >>> runAction: ${d.actionMarker}`); continue; }
      console.log(fmtDraw(d));
    }
  });
  const sp = (m) => (m ? `${m.species} hp=${m.hp}/${m.maxhp} spe=${m.spe} st=${m.status || '-'} fnt=${m.fainted}` : 'NONE');
  console.log(`\n  Final: p1=${sp(firstRes.p1)}`);
  console.log(`         p2=${sp(firstRes.p2)}`);
  console.log(`         turn=${firstRes.turn} ended=${firstRes.ended} winner=${JSON.stringify(firstRes.winner)}`);

  // Print the relevant protocol lines (switch/faint/request/turn) for the first seed.
  if (firstRes.proto) {
    console.log(`\n  Protocol (switch/faint/request/turn/win lines, first seed):`);
    for (const l of firstRes.proto) {
      if (/^\|(switch|drag|faint|turn|win|upkeep)\b/.test(l) || /forceSwitch/.test(l) || /^\|request\b/.test(l)) {
        console.log(`     ${l.length > 120 ? l.slice(0, 120) + '…' : l}`);
      }
    }
  }

  // Per-step signature stability across seeds.
  console.log(`\n  Per-step structural-signature stability across ${seeds.length} seeds:`);
  for (const [ti, m] of [...turnSigSets.entries()].sort((a, b) => a[0] - b[0])) {
    const entries = [...m.entries()].sort((a, b) => b[1] - a[1]);
    const label = firstPerTurn[ti] ? firstPerTurn[ti].label : `group ${ti}`;
    if (entries.length === 1) {
      console.log(`   ${label}: 1 signature (STABLE) x${entries[0][1]} — len ${entries[0][0] ? entries[0][0].split(' | ').length : 0}`);
    } else {
      console.log(`   ${label}: ${entries.length} DISTINCT signatures (varies!) ` +
        entries.map(([s, c]) => `[x${c} len ${s ? s.split(' | ').length : 0}]`).join(' '));
      entries.forEach(([s, c], i) => console.log(`      sig[${i}] x${c}: ${s || '(empty)'}`));
    }
  }
}

async function main() {
  const seeds = buildSeeds(Number(process.argv[2] || 30));
  for (const scFn of SCENARIOS) await traceScenario(scFn, seeds);
}
main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
