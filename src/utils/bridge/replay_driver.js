// replay_driver.js
// Offline replay / re-roll driver over a battle's reconstruction record — the
// Node half of utils/bridge/reconstruction.py's replay_battle()/reroll_turn()/
// reroll_many(). Batch, not streaming: ONE JSON request on stdin, ONE JSON
// response on stdout, then exit. No server, no live battle — this only ever
// re-runs recorded games.
//
// request:
//   { mode: "replay",  record: {format_id, prng_seed, input_log, commands} }
//   { mode: "reroll",  record: ..., turn: T,
//     seeds: ["a,b,c,d"|"sodium,<hex>"|"original", ...],   // "original" = NO PRNG swap
//     p1_action: "recorded"|"random"|"<explicit choice>", p2_action: ...,
//     followup: "random"|"default" }
//   { mode: "reroll_many", record: ..., turn: T, followup: ...,
//     arms: [{p1_action, p2_action, seed, label}, ...] }   // N independent arms, ONE process
//
// response (replay):   { p1_chunks, p2_chunks, outcome }
// response (reroll):    { turn, pre_state, requests, recorded_choices,
//                         prefix_p1_chunks, prefix_p2_chunks,
//                         rerolls: [{seed, choices_used, outcome, turn_log, p1_chunks, p2_chunks}] }
// response (reroll_many): same head, with `arms: [{label, seed, ...same per-arm fields}]`
// or { error: "..." }.
//
// Mechanism: rebuild a fresh BattleStream from the record's `>start` line (which
// carries the RESOLVED seed) + `>player` lines, then feed the recorded raw
// commands (see replay_kernels.js for the shared sim kernels). Because the sim is
// deterministic given (seed, teams, command sequence), the regenerated per-side
// streams are byte-identical to what the live bridge emitted — including |error|
// rounds for choices the sim refused. Sole exception: |t:| lines carry the wall
// clock at emission time; they're in poke-env's MESSAGES_TO_IGNORE, so state/obs
// never see them. For a re-roll, the rebuild stops at the start of turn T and
// `battle.prng` is swapped for a fresh PRNG before resolving just that turn.
//
// The per-side chunks are the ONE-SIDED views (what each agent saw); the
// pre_state / outcome / turn_log fields are OMNISCIENT (referee view). The
// Python caller keeps those apart — see reconstruction.py's module header.
'use strict';

const K = require('./replay_kernels');
const {
  PRNG, tick,
  buildSession, destroySession, writeStart, writeCmd,
  outcomeOf, preState, auxRngFromSeed, randomChoice,
  buildToTurn, recordedTurnChoices, recordedQueues,
  resolveTurnExact, resolveTurn,
} = K;

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

// Resolve ONE arm — its own action spec + seed — at turn T in a FRESH session,
// returning the per-side SUFFIX chunks + outcome. This is `runReroll`'s exact
// inner-loop resolution lifted out verbatim, so a batched `reroll_many` produces
// BYTE-IDENTICAL suffixes to N separate `reroll` calls (the speed win is only
// that all arms share ONE Node process / module load, not a behaviour change).
// Guarded by reroll_many_parity_fuzz_test.
async function resolveArm(record, T, actionSpec, seed, followup) {
  const sess = buildSession();
  try {
    const fromIdx = buildToTurn(sess, record, T);
    await tick();
    const b = sess.stream.battle;
    const prefixCounts = { p1: sess.chunks.p1.length, p2: sess.chunks.p2.length };
    const logStart = b.log.length;
    const isOriginal = seed === 'original';
    if (!isOriginal) {
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
        if (singleShot[side]) return null;
        singleShot[side] = true;
        if (spec === 'random') return randomChoice(b, i, rng);
        return spec;
      };
      resolved = await resolveTurn(sess, sourceChoice, followup, rng);
    }
    return {
      choices_used: resolved.used,
      outcome: outcomeOf(b, resolved.stuck ? { stuck: true } : null),
      turn_log: b.log.slice(logStart),
      p1_chunks: sess.chunks.p1.slice(prefixCounts.p1),
      p2_chunks: sess.chunks.p2.slice(prefixCounts.p2),
    };
  } finally {
    destroySession(sess);
  }
}

// Batched re-roll: resolve N independent ARMS (each its own p1/p2 action source +
// seed) of turn T in ONE process, so a sweep over candidate actions (the lookahead)
// pays the ~677ms Node-spawn / pokemon-showdown require cost ONCE instead of once
// per candidate. Each arm is a fresh session (resolveArm), so the arms are fully
// independent and byte-identical to per-arm `reroll`. Shares one inspection pass
// for the decision-point outputs + prefix chunks.
async function runRerollMany(req) {
  const record = req.record;
  const T = req.turn;
  if (!Number.isInteger(T) || T < 1) throw new Error(`invalid turn ${T}`);
  const arms = req.arms || [];
  const followup = req.followup || 'random';

  const inspect = buildSession();
  let pre, requests, recorded, prefixChunks;
  try {
    const restIdx = buildToTurn(inspect, record, T);
    await tick();
    const b = inspect.stream.battle;
    pre = preState(b);
    requests = { p1: b.sides[0].activeRequest, p2: b.sides[1].activeRequest };
    recorded = recordedTurnChoices(record, restIdx);
    prefixChunks = { p1: inspect.chunks.p1.slice(), p2: inspect.chunks.p2.slice() };
  } finally {
    destroySession(inspect);
  }

  const results = [];
  for (const arm of arms) {
    const actionSpec = { p1: arm.p1_action || 'recorded', p2: arm.p2_action || 'recorded' };
    const r = await resolveArm(record, T, actionSpec, arm.seed, followup);
    results.push(Object.assign({ label: arm.label != null ? arm.label : null, seed: arm.seed }, r));
  }

  return {
    turn: T,
    pre_state: pre,
    requests,
    recorded_choices: recorded,
    prefix_p1_chunks: prefixChunks.p1,
    prefix_p2_chunks: prefixChunks.p2,
    arms: results,
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
        : req.mode === 'reroll_many' ? await runRerollMany(req)
          : (() => { throw new Error(`unknown mode ${req.mode}`); })();
    finish(out, 0);
  } catch (e) {
    finish({ error: e.stack || String(e) }, 1);
  }
})();
