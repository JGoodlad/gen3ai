// search_driver.js
// WARM, PERSISTENT clone-and-branch search-server — the Node half of
// utils/bridge/search_session.py. Unlike replay_driver.js (one request → one
// response → exit), this stays alive and holds a node-snapshot cache, so a
// multi-ply BEAM over candidate lines (the prober's `better_line` probe) can
// branch a search TREE from any explored node WITHOUT re-replaying the battle
// from turn 1 per node.
//
// THE LEVER: State.serializeBattle / deserializeBattle clone a mid-battle state
// in-process in ~1.7ms (vs ~26ms to rebuild from the record, which grows with
// depth) — verified round-trip of a paused-at-move-request battle (deserialize
// rebuilds the open requests via getRequests; PRNG continuity from the live
// counter). Each node stores its serialized snapshot; expanding a node
// deserializes a FRESH battle from that snapshot, applies the joint turn, and
// re-serializes the child. The per-side chunk splitting is REUSED verbatim from
// replay_kernels (buildSession + getPlayerStreams) — a deserialized battle is
// re-attached to a fresh BattleStream and `restart()`-ed with the same `send`
// wiring BattleStream._writeLine('start') uses, so the emitted one-sided chunks
// are byte-identical (modulo |t:|) to the replay_driver / reroll_many path
// (pinned by search_clone_parity_fuzz_test).
//
// PROTOCOL (newline-delimited JSON, request → one response line, stays alive):
//   {id, cmd:"open_root", record, turn}
//     → {id, ok, node_id, requests, recorded_choices, pre_state,
//        prefix_p1_chunks, prefix_p2_chunks}
//   {id, cmd:"expand_many", arms:[{node_id, p1_action, p2_action, seed,
//                                   recorded_exact?, followup?, label}]}
//     → {id, ok, arms:[{label, node_id, ended, stuck, outcome, requests,
//                       choices_used, p1_chunks, p2_chunks}]}   // p*_chunks are this turn's SUFFIX
//   {id, cmd:"close"}  → exits
//
// p*_chunks are this ply's one-sided SUFFIX (a deserialized battle can't re-emit
// the historical |request| lines — requests are sent out of band, not stored in
// battle.log — so re-emitting the prefix would lose the team; instead the suffix is
// returned and the Python caller composes the request/team-complete ROOT prefix +
// each ply's suffix, the same (prefix + suffix) shape reroll_many produces).
// pre_state / outcome are OMNISCIENT (referee view) — never fed to the obs encoder.
'use strict';

const readline = require('readline');
const K = require('./replay_kernels');
const {
  PRNG, State, tick,
  buildSession, destroySession,
  outcomeOf, preState, auxRngFromSeed, randomChoice,
  buildToTurn, recordedTurnChoices,
  resolveTurnExact, resolveTurn,
} = K;

// node_id -> { snapshot, record?, restIdx? }. record/restIdx only on the root,
// so a depth-1 expand can reproduce the realized turn EXACTLY (resolveTurnExact)
// for the value_crn faithfulness anchor.
const NODES = new Map();
let nodeCounter = 0;
function freshId() { return `n${nodeCounter++}`; }

// Re-attach a deserialized battle to a fresh per-side-collecting session, wiring
// `send` exactly as BattleStream._writeLine('start') does so updates route
// through pushMessage → getPlayerStreams demux → the chunk collectors.
function sessionFromSnapshot(snapshot) {
  const sess = buildSession();
  const battle = State.deserializeBattle(snapshot);
  battle.restart((t, data) => {
    if (Array.isArray(data)) data = data.join('\n');
    sess.stream.pushMessage(t, data);
    if (t === 'end' && !sess.stream.keepAlive) sess.stream.pushEnd();
  });
  sess.stream.battle = battle;
  return sess;
}

async function openRoot(req) {
  const record = req.record;
  const T = req.turn;
  if (!Number.isInteger(T) || T < 1) throw new Error(`invalid turn ${T}`);
  // A fresh root starts a fresh tree; drop the previous search's nodes so a WARM process reused
  // across many battles (the search-teacher workers) stays memory-bounded. ids stay monotonic.
  NODES.clear();
  const sess = buildSession();
  try {
    const restIdx = buildToTurn(sess, record, T);
    await tick();
    const b = sess.stream.battle;
    const snapshot = State.serializeBattle(b);
    const node_id = freshId();
    NODES.set(node_id, { snapshot, record, restIdx });
    return {
      node_id,
      requests: { p1: b.sides[0].activeRequest, p2: b.sides[1].activeRequest },
      recorded_choices: recordedTurnChoices(record, restIdx),
      pre_state: preState(b),
      prefix_p1_chunks: sess.chunks.p1.slice(),
      prefix_p2_chunks: sess.chunks.p2.slice(),
    };
  } finally {
    destroySession(sess);
  }
}

// Resolve ONE arm: clone the parent node, apply the joint turn, serialize the
// child. Returns THIS PLY's one-sided chunks (a suffix — see the baseline slice
// below) + outcome + open requests. A caller deepening past one ply must
// accumulate them; the driver cannot, because it never re-emits the historical
// |request| lines a materializer needs.
async function expandArm(arm) {
  const node = NODES.get(arm.node_id);
  if (!node) throw new Error(`unknown node ${arm.node_id}`);
  const followup = arm.followup || 'random';
  const seed = arm.seed || 'original';
  const sess = sessionFromSnapshot(node.snapshot);
  try {
    const b = sess.stream.battle;
    // A deserialized battle re-emits its WHOLE restored log on the first flush, but
    // that re-emission lacks the historical |request| lines (requests are sent out of
    // band, not stored in battle.log) — so the team is never re-established and a
    // materializer parsing the turn-T request fails. So we DON'T use the re-emitted
    // prefix: flush it, mark a baseline, and return only THIS turn's SUFFIX. The Python
    // caller composes the (request/team-complete) ROOT prefix + each ply's suffix — the
    // exact (prefix + suffix) shape reroll_many produces (parity-fuzzed bit-for-bit).
    b.sendUpdates();
    await tick();
    await tick();
    const baseline = { p1: sess.chunks.p1.length, p2: sess.chunks.p2.length };
    const isOriginal = seed === 'original';
    let resolved;
    if (arm.recorded_exact) {
      // Reproduce the realized turn EXACTLY (the value_crn anchor). Only the root
      // carries the record alignment; never swap the PRNG (realized dice).
      if (!node.record) throw new Error('recorded_exact requested on a non-root node');
      resolved = await resolveTurnExact(sess, node.record, node.restIdx);
    } else {
      if (!isOriginal) b.prng = new PRNG(seed);   // THE SWAP: fresh dice for this arm
      const rng = auxRngFromSeed(seed);
      const spec = { p1: arm.p1_action || 'recorded', p2: arm.p2_action || 'recorded' };
      const singleShot = { p1: false, p2: false };
      const sourceChoice = (side, i) => {
        const s = spec[side];
        if (s === 'recorded') return null;          // no record alignment off-root → follow-up
        if (singleShot[side]) return null;
        singleShot[side] = true;
        if (s === 'random') return randomChoice(b, i, rng);
        return s;                                    // explicit choice string
      };
      resolved = await resolveTurn(sess, sourceChoice, followup, rng);
    }
    const ended = !!b.ended;
    const child_id = ended ? null : freshId();
    if (child_id) NODES.set(child_id, { snapshot: State.serializeBattle(b) });
    return {
      label: arm.label != null ? arm.label : null,
      node_id: child_id,
      ended,
      stuck: !!resolved.stuck,
      outcome: outcomeOf(b, resolved.stuck ? { stuck: true } : null),
      requests: ended ? null
        : { p1: b.sides[0].activeRequest, p2: b.sides[1].activeRequest },
      choices_used: resolved.used,
      p1_chunks: sess.chunks.p1.slice(baseline.p1),   // THIS turn's suffix only
      p2_chunks: sess.chunks.p2.slice(baseline.p2),
    };
  } finally {
    destroySession(sess);
  }
}

async function expandMany(req) {
  const arms = req.arms || [];
  const out = [];
  for (const arm of arms) out.push(await expandArm(arm));
  return { arms: out };
}

// ---------------------------------------------------------------------------
// persistent stdin loop
// ---------------------------------------------------------------------------

const rl = readline.createInterface({ input: process.stdin });
let queue = Promise.resolve();      // serialize command handling (battles are async)

function respond(obj) { process.stdout.write(JSON.stringify(obj) + '\n'); }

rl.on('line', (line) => {
  const text = line.trim();
  if (!text) return;
  queue = queue.then(async () => {
    let req;
    try { req = JSON.parse(text); } catch (e) {
      respond({ error: `bad request JSON: ${e.message}` });
      return;
    }
    const id = req.id != null ? req.id : null;
    if (req.cmd === 'close') { respond({ id, ok: true, bye: true }); process.exit(0); }
    try {
      const out = req.cmd === 'open_root' ? await openRoot(req)
        : req.cmd === 'expand_many' ? await expandMany(req)
          : (() => { throw new Error(`unknown cmd ${req.cmd}`); })();
      respond(Object.assign({ id, ok: true }, out));
    } catch (e) {
      respond({ id, ok: false, error: e.stack || String(e) });
    }
  }).catch((e) => respond({ ok: false, error: e.stack || String(e) }));
});

rl.on('close', () => process.exit(0));
