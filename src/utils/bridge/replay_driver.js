// replay_driver.js
// Offline replay / re-roll driver over a battle's reconstruction record — the
// Node half of utils/bridge/reconstruction.py's replay_battle()/reroll_turn().
// Batch, not streaming: ONE JSON request on stdin, ONE JSON response on stdout,
// then exit. No server, no live battle — this only ever re-runs recorded games.
//
// request:
//   { mode: "replay",  record: {format_id, prng_seed, input_log, commands} }
//   { mode: "reroll",  record: ..., turn: T,
//     seeds: ["a,b,c,d"|"sodium,<hex>"|"original", ...],   // "original" = NO PRNG swap
//     p1_action: "recorded"|"random"|"<explicit choice>", p2_action: ...,
//     followup: "random"|"default" }
//
// response (replay):
//   { p1_chunks, p2_chunks, outcome }
// response (reroll):
//   { turn, pre_state, requests, recorded_choices,
//     prefix_p1_chunks, prefix_p2_chunks,
//     rerolls: [{seed, choices_used, outcome, turn_log, p1_chunks, p2_chunks}] }
// or { error: "..." }.
//
// Mechanism: rebuild a fresh BattleStream from the record's `>start` line (which
// carries the RESOLVED seed) + `>player` lines, then feed the recorded raw
// commands. Because the sim is deterministic given (seed, teams, command
// sequence), the regenerated per-side streams are byte-identical to what the
// live bridge emitted — including |error| rounds for choices the sim refused
// (the raw `commands` log replays those attempts verbatim; battle.inputLog alone
// would not). Sole exception: |t:| lines carry the wall clock at emission time;
// they're in poke-env's MESSAGES_TO_IGNORE, so state/obs never see them. For a
// re-roll, the rebuild stops at the start of turn T and `battle.prng` is
// swapped for a fresh PRNG — every die in the sim routes through it — before
// resolving just that turn.
//
// The per-side chunks are the ONE-SIDED views (what each agent saw); the
// pre_state / outcome / turn_log fields are OMNISCIENT (referee view). The
// Python caller keeps those apart — see reconstruction.py's module header.
'use strict';

const path = require('path');
const psPath = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(psPath, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(psPath, 'dist/sim/prng'));

const tick = () => new Promise((r) => setImmediate(r));
const SIDES = ['p1', 'p2'];

// ---------------------------------------------------------------------------
// Session: a BattleStream + per-side chunk collectors
// ---------------------------------------------------------------------------

function buildSession() {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const chunks = { p1: [], p2: [] };
  let endedSides = 0;
  let resolveDone;
  const done = new Promise((r) => { resolveDone = r; });
  for (const side of SIDES) {
    (async () => {
      try {
        for await (const c of streams[side]) chunks[side].push(c);
      } catch (e) { /* a destroyed stream ends with an error; chunks stand */ }
      finally { if (++endedSides >= 2) resolveDone(); }
    })();
  }
  return { stream, streams, chunks, done };
}

function destroySession(sess) {
  try { sess.stream.destroy(); } catch (e) { /* already ended */ }
}

function writeStart(sess, record) {
  const startLine = record.input_log.find((l) => l.startsWith('>start '));
  const playerLines = record.input_log.filter((l) => l.startsWith('>player '));
  if (!startLine) throw new Error('record has no >start line');
  if (playerLines.length !== 2) throw new Error('record does not have exactly two >player lines');
  sess.streams.omniscient.write(startLine);
  for (const line of playerLines) sess.streams.omniscient.write(line);
}

function writeCmd(sess, cmd) {
  const [side, payload] = cmd;
  if (side === 'forcelose') {
    sess.streams.omniscient.write(`>forcelose ${payload}`);
  } else {
    sess.streams[side].write(payload);
  }
}

// ---------------------------------------------------------------------------
// Omniscient snapshots (referee view — for ground truth, never for obs)
// ---------------------------------------------------------------------------

function sideOutcome(b, i) {
  const side = b.sides[i];
  const a = side.active[0];
  return {
    active_species: a ? a.species.id : null,
    active_hp: a ? a.hp : null,
    active_maxhp: a ? a.maxhp : null,
    active_status: a ? a.status : null,
    active_fainted: a ? a.fainted : null,
    alive: side.pokemon.filter((p) => !p.fainted).length,
    team_hp: side.pokemon.reduce((acc, p) => acc + p.hp, 0),
    team_maxhp: side.pokemon.reduce((acc, p) => acc + p.maxhp, 0),
  };
}

function outcomeOf(b, extra) {
  return Object.assign({
    turn: b.turn,
    ended: b.ended,
    winner: b.winner || null,
    p1: sideOutcome(b, 0),
    p2: sideOutcome(b, 1),
  }, extra || {});
}

// Comparable pre-turn board snapshot (the spike's shape): proves identical
// reconstruction across re-rolls.
function preState(b) {
  const sideSnap = (side) => ({
    active: side.active.map((p) => p && ({
      species: p.species.id, hp: p.hp, maxhp: p.maxhp,
      status: p.status, fainted: p.fainted,
      boosts: Object.assign({}, p.boosts),
      item: p.item, ability: p.ability,
      volatiles: Object.keys(p.volatiles).sort(),
      moveSlots: p.moveSlots.map((m) => ({ id: m.id, pp: m.pp })),
    })),
    pokemon: side.pokemon.map((p) => ({
      species: p.species.id, hp: p.hp, maxhp: p.maxhp,
      status: p.status, fainted: p.fainted, position: p.position,
    })),
    sideConditions: Object.keys(side.sideConditions).sort(),
  });
  return {
    turn: b.turn,
    weather: b.field.weather,
    pseudoWeather: Object.keys(b.field.pseudoWeather).sort(),
    p1: sideSnap(b.sides[0]),
    p2: sideSnap(b.sides[1]),
  };
}

// ---------------------------------------------------------------------------
// Choice sources
// ---------------------------------------------------------------------------

// Aux PRNG (mulberry32 over an FNV-1a hash of the seed string) for "random"
// action/follow-up picks. Independent of the sim's PRNG on purpose: drawing our
// choices from battle.prng would entangle choice randomness with the dice under
// study. Deterministic per re-roll seed.
function auxRngFromSeed(seedStr) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < seedStr.length; i++) {
    h ^= seedStr.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  let a = h >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function pickUniform(rng, items) {
  return items[Math.floor(rng() * items.length)];
}

// Uniform-random legal choice for `side` given its CURRENT request: a legal
// move or (if not trapped) a legal switch. Falls back to the sim's 'default'.
function randomChoice(b, sideIdx, rng) {
  const side = b.sides[sideIdx];
  const req = side.activeRequest;
  if (!req || req.wait) return null;
  const benchSwitches = [];
  side.pokemon.forEach((p, i) => {
    if (!p.fainted && !p.isActive) benchSwitches.push(`switch ${i + 1}`);
  });
  if (req.forceSwitch) {
    return benchSwitches.length ? pickUniform(rng, benchSwitches) : 'default';
  }
  const options = [];
  const active = side.active[0];
  if (active) {
    active.moveSlots.forEach((m, i) => {
      if (!m.disabled && m.pp > 0) options.push(`move ${i + 1}`);
    });
  }
  // The REQUEST's trapped flag is the agent-visible truth (maybeTrapped still
  // allows the attempt — the sim refuses it and we fall through to a re-pick).
  if (!(req.active && req.active[0] && req.active[0].trapped)) options.push(...benchSwitches);
  return options.length ? pickUniform(rng, options) : 'default';
}

function followupChoice(b, sideIdx, followup, rng) {
  if (followup === 'default') return 'default';
  const c = randomChoice(b, sideIdx, rng);
  return c === null ? null : c;
}

// ---------------------------------------------------------------------------
// Reconstruction to the start of turn T
// ---------------------------------------------------------------------------

function atTurnStart(b, T) {
  return b && !b.ended && b.turn === T
    && b.sides[0].requestState === 'move' && b.sides[1].requestState === 'move';
}

// Replay recorded commands until the battle sits at the start-of-turn move
// round of turn T (checked BEFORE each command, so we stop exactly before
// turn T's first recorded choice). Returns the index of the first unapplied
// command. Throws if the battle ends (or the log runs out) before turn T.
function buildToTurn(sess, record, T) {
  writeStart(sess, record);
  const cmds = record.commands;
  let i = 0;
  while (i < cmds.length) {
    if (atTurnStart(sess.stream.battle, T)) return i;
    writeCmd(sess, cmds[i]);
    i += 1;
  }
  if (atTurnStart(sess.stream.battle, T)) return i;
  const b = sess.stream.battle;
  throw new Error(
    `battle never reached the start of turn ${T} `
    + `(ended=${b ? b.ended : '?'} at turn ${b ? b.turn : '?'})`);
}

// First recorded p1/p2 choice at/after command index `from` — the original
// turn-T picks. null when a side never chose again (e.g. the battle ended by
// forfeit that round).
function recordedTurnChoices(record, from) {
  const out = { p1: null, p2: null };
  for (let i = from; i < record.commands.length; i++) {
    const [side, payload] = record.commands[i];
    if (side === 'forcelose') break;
    if (out[side] === null) out[side] = payload;
    if (out.p1 !== null && out.p2 !== null) break;
  }
  return out;
}

// Per-side QUEUES of the remaining recorded commands (capped). The "recorded"
// action source pulls successive entries until the side's round-1 choice is
// ACCEPTED — so a live refused-then-corrected sequence (the maybe-trapped
// probe) replays faithfully under a fresh seed too: the pre-turn state is
// identical, the first attempt refuses identically, the recorded correction
// follows. Entries beyond the accepted one are never consumed (follow-up
// rounds in a re-rolled timeline are NEW decisions → the followup policy).
function recordedQueues(record, from, cap) {
  const out = { p1: [], p2: [] };
  for (let i = from; i < record.commands.length; i++) {
    const [side, payload] = record.commands[i];
    if (side === 'forcelose') break;
    if (out[side].length < (cap || 8)) out[side].push(payload);
  }
  return out;
}

// Resolve turn T EXACTLY as the original battle did: feed the remaining
// recorded commands verbatim (original interleaving, original follow-ups)
// until the turn advances or the battle ends. Used for the special seed
// "original" with both sides recorded — the realized line, scored through the
// same outcome pipeline as the re-rolls so margins are directly comparable.
async function resolveTurnExact(sess, record, fromIdx) {
  const b = sess.stream.battle;
  const startTurn = b.turn;
  const used = { p1: [], p2: [] };
  let i = fromIdx;
  while (i < record.commands.length && !b.ended && b.turn === startTurn) {
    const [side, payload] = record.commands[i];
    writeCmd(sess, record.commands[i]);
    if (side !== 'forcelose') used[side].push(payload);
    i += 1;
  }
  await tick();
  return { used, stuck: !b.ended && b.turn === startTurn };
}

// Resolve ONE whole turn (start-of-turn choices + any follow-up rounds a
// mid-turn faint forces). Routing invariant: within one turn a side gets at
// most ONE 'move' request — a refused choice re-asks as 'move' again, while
// every mid-turn follow-up (a forced switch after a faint) asks as 'switch'.
// So a 'move' request draws from the side's configured SOURCE and anything
// else from the follow-up policy. A "recorded" source is a queue — a refusal
// pulls the next recorded command, mirroring the live refused-then-corrected
// sequence; an explicit/random source is single-shot, so a refusal falls to
// the follow-up policy (callers detect it from the |error| line in the side's
// suffix chunks). Bounded; marks `stuck` if the turn cannot settle.
async function resolveTurn(sess, sourceChoice, followup, rng) {
  const b = sess.stream.battle;
  const startTurn = b.turn;
  const used = { p1: [], p2: [] };
  let guard = 0;
  let stuck = false;
  while (!b.ended && b.turn === startTurn) {
    if (guard++ > 40) { stuck = true; break; }
    let wrote = false;
    for (let i = 0; i < 2; i++) {
      const side = SIDES[i];
      const s = b.sides[i];
      if (!s.activeRequest || s.activeRequest.wait || s.isChoiceDone()) continue;
      let c = s.requestState === 'move' ? sourceChoice(side, i) : null;
      if (c === null) c = followupChoice(b, i, followup, rng);
      if (c === null) continue;
      used[side].push(c);
      sess.streams[side].write(c);
      wrote = true;
    }
    await tick();
    if (!wrote) break;
  }
  await tick();
  return { used, stuck };
}

// ---------------------------------------------------------------------------
// Modes
// ---------------------------------------------------------------------------

async function runReplay(req) {
  const record = req.record;
  const sess = buildSession();
  try {
    writeStart(sess, record);
    for (const cmd of record.commands) writeCmd(sess, cmd);
    await tick();
    const b = sess.stream.battle;
    if (!b) throw new Error('battle missing after replay');
    if (!b.ended) {
      throw new Error(`replayed all ${record.commands.length} commands but battle has not ended `
        + `(turn ${b.turn}) — corrupt or truncated record?`);
    }
    const outcome = outcomeOf(b);
    // Battle over → side streams end; wait for the pumps to flush the tails.
    await Promise.race([sess.done, new Promise((r) => setTimeout(r, 5000))]);
    return { p1_chunks: sess.chunks.p1, p2_chunks: sess.chunks.p2, outcome };
  } finally {
    destroySession(sess);
  }
}

async function runReroll(req) {
  const record = req.record;
  const T = req.turn;
  if (!Number.isInteger(T) || T < 1) throw new Error(`invalid turn ${T}`);
  const seeds = req.seeds || [];
  const followup = req.followup || 'random';
  const actionSpec = { p1: req.p1_action || 'recorded', p2: req.p2_action || 'recorded' };

  // Inspection pass: reconstruct once for the decision-point outputs.
  const inspect = buildSession();
  let restIdx, pre, requests, recorded, prefixCounts, prefixChunks;
  try {
    restIdx = buildToTurn(inspect, record, T);
    await tick();
    const b = inspect.stream.battle;
    pre = preState(b);
    requests = { p1: b.sides[0].activeRequest, p2: b.sides[1].activeRequest };
    recorded = recordedTurnChoices(record, restIdx);
    prefixCounts = { p1: inspect.chunks.p1.length, p2: inspect.chunks.p2.length };
    prefixChunks = { p1: inspect.chunks.p1.slice(), p2: inspect.chunks.p2.slice() };
  } finally {
    destroySession(inspect);
  }

  const rerolls = [];
  for (const seed of seeds) {
    const sess = buildSession();
    try {
      const fromIdx = buildToTurn(sess, record, T);
      await tick();
      const b = sess.stream.battle;
      const logStart = b.log.length;
      // The special seed "original" keeps the battle's own mid-game PRNG state
      // (no swap) — with both sides recorded that reproduces the REALIZED turn
      // exactly (recorded follow-ups included), scored through the same outcome
      // pipeline as the re-rolls so margins are directly comparable. With a
      // non-recorded action it answers "same dice stream, different action"
      // (common-random-numbers against the realized line).
      const isOriginal = seed === 'original';
      if (!isOriginal) {
        // THE SWAP: every die the sim rolls from here routes through the new PRNG.
        b.prng = new PRNG(seed);
      }
      const rng = auxRngFromSeed(seed);
      let resolved;
      if (isOriginal && actionSpec.p1 === 'recorded' && actionSpec.p2 === 'recorded') {
        resolved = await resolveTurnExact(sess, record, fromIdx);
      } else {
        const queues = recordedQueues(record, fromIdx);
        const singleShot = { p1: false, p2: false };
        const sourceChoice = (side, i) => {
          const spec = actionSpec[side];
          if (spec === 'recorded') return queues[side].length ? queues[side].shift() : null;
          if (singleShot[side]) return null;   // refused explicit/random → follow-up
          singleShot[side] = true;
          if (spec === 'random') return randomChoice(b, i, rng);
          return spec;                          // explicit choice string
        };
        resolved = await resolveTurn(sess, sourceChoice, followup, rng);
      }
      rerolls.push({
        seed,
        choices_used: resolved.used,
        outcome: outcomeOf(b, resolved.stuck ? { stuck: true } : null),
        turn_log: b.log.slice(logStart),
        p1_chunks: sess.chunks.p1.slice(prefixCounts.p1),
        p2_chunks: sess.chunks.p2.slice(prefixCounts.p2),
      });
    } finally {
      destroySession(sess);
    }
  }

  return {
    turn: T,
    pre_state: pre,
    requests,
    recorded_choices: recorded,
    prefix_p1_chunks: prefixChunks.p1,
    prefix_p2_chunks: prefixChunks.p2,
    rerolls,
  };
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

(async () => {
  let input = '';
  for await (const chunk of process.stdin) input += chunk;
  // Exit only from the write callback: a large response is async-buffered on a
  // pipe, and a bare process.exit() right after write() truncates it mid-flush.
  const finish = (obj, code) => process.stdout.write(JSON.stringify(obj), () => process.exit(code));
  let req;
  try {
    req = JSON.parse(input);
  } catch (e) {
    finish({ error: `bad request JSON: ${e.message}` }, 1);
    return;
  }
  try {
    const out = req.mode === 'replay' ? await runReplay(req)
      : req.mode === 'reroll' ? await runReroll(req)
        : (() => { throw new Error(`unknown mode ${req.mode}`); })();
    finish(out, 0);
  } catch (e) {
    finish({ error: e.stack || String(e) }, 1);
  }
})();
