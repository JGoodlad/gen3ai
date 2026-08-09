// local_sim_bridge.js
// Streaming bridge between Python (poke-env) and a local, in-process Pokémon
// Showdown BattleStream. ONE battle per process. Emits the per-side protocol
// text (the |...| lines each player sees) so poke-env's parser can consume it
// exactly as if it came off the websocket — but with no server, no port, no
// usernames, no matchmaking.
//
// This is the same subprocess-over-stdio pattern as validate_team.js, except it
// is a *streaming relay* (protocol text both ways) rather than request/response
// JSON. It deliberately does NOT read serializeBattle() state — the whole point
// is to feed poke-env the protocol stream.
//
// stdin (newline-delimited commands):
//   START <json>   {formatid, seed?, persistent?, p1:{name,team}, p2:{name,team}}
//   CHOOSE <side> <choice>   e.g.  CHOOSE p1 move 1   /   CHOOSE p2 switch 3
//   FORCELOSE <side>         e.g.  FORCELOSE p1   (poke-env /forfeit path)
//   END                      tear down and exit
//
// One battle per process by default. If a START carries `"persistent": true`, the
// process stays ALIVE after a battle ends (emits `__END__` and resets) so the SAME
// child can run a fresh battle on the next START — used by the RL env transport to
// avoid a Node spawn per episode. Non-persistent behaviour is unchanged (exit on
// battle end), so `run_local_battles` and the seed-repro test are unaffected.
//
// stdout (newline-delimited frames):
//   p1 <base64(chunk)>   one protocol chunk p1 saw (may be multi-line)
//   p2 <base64(chunk)>   one protocol chunk p2 saw
//   __RECON__ <base64(json)>  the battle's reconstruction record (see below),
//                             emitted once per battle, just before __END__
//   __END__              battle over, both side streams closed
//   __ERR__ <base64(msg)>  fatal error
//
// Base64 per chunk because protocol text contains \n, |, and arbitrary JSON in
// |request| — one stdout line == exactly one side-tagged chunk, unambiguous to
// demux on the Python side.
//
// Reconstruction record (__RECON__): everything needed to rebuild the battle
// bit-for-bit offline — {v, format_id, prng_seed, input_log, commands}.
//   input_log = battle.inputLog: the sim's own normalized record (>start with the
//     RESOLVED seed, >player with both packed teams, every COMMITTED choice).
//     State-faithful: replaying it reproduces every board state.
//   commands  = the raw choice lines this child processed, in order, INCLUDING
//     attempts the sim refused (an "[Unavailable choice]" maybe-trapped probe is
//     refused, never committed, so it is absent from input_log — but its |error|
//     + re-request round IS part of the per-side protocol the agent saw).
//     Protocol-faithful: replaying >start + >player from input_log, then these
//     commands, regenerates BYTE-IDENTICAL per-side streams (verified).
// This is FULL-INFORMATION (referee-view) data: both teams + the seed. It must
// only ever be persisted as a separate artifact at the bridge layer — never fed
// into the one-sided observation pipeline (the project's hard one-sided/omniscient
// wall; see utils/bridge/reconstruction.py).
'use strict';

const path = require('path');
const psPath = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(psPath, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(psPath, 'dist/sim/prng'));

let streams = null;
let rawStream = null;     // the underlying BattleStream — for inputLog/prngSeed at end
let formatId = null;
let cmdLog = [];          // raw [side, choice] / ['forcelose', side] lines, in processing order
let reconEmitted = false;
let endedSides = 0;
// Sticky: once any START asks for it, the process survives battle ends and waits for
// the next START instead of exiting.
let persistent = false;
// Counterfactual Monte-Carlo: {turn, seed} → swap the battle's PRNG for a fresh one at the START of
// `turn` (so the prefix replays under the recorded dice but the post-divergence dice are resampled).
// Mirrors replay_driver.js's `b.prng = new PRNG(seed)`, but inside the live streaming bridge.
let resumeReseed = null;
let reseeded = false;

// ── Reject-streak bound (`gen3_node_bridge_reject_bound_v1`) ────────────────────────────
// The mirror of the rust bridge's `REJECT_STREAK_CAP` (src/rust_sim/src/bridge.rs).
//
// A refused choice leaves the boundary OPEN — correct when the client then picks something
// else. But an RL policy is DETERMINISTIC given the same request, so if its action mask
// disagrees with the sim about legality it re-sends the SAME choice forever. Measured in a
// live run on the rust bridge: one child spinning at 46 MB/s of re-requests while its env's
// `step()` never returned, wedging the whole vec-env. The node bridge is the DEFAULT
// in-process transport and had no such bound.
//
// Probed (2026-08-03), the sim side is worse than "re-request forever": `Side.emitChoiceError`
// only re-emits a `|request|` when the refusal actually CHANGED the request (the
// "[Unavailable choice]" maybe-trapped reveal). A plain "[Invalid choice]" emits the error and
// NOTHING else — and poke-env answers `[Invalid choice]` by re-choosing immediately off the
// stale request (`player.py` -> `_handle_battle_request(maybe_default_order=True)`), so a
// deterministic client and the sim ping-pong with zero sim progress. Measured raw: 36 refusal
// rounds in 3 ms, unbounded.
//
// Cap 8 is generous on purpose — a handful of refusals is legitimate (the maybe-trapped probe
// is a normal two-exchange round) — it exists to turn an unbounded spin into a diagnosable
// error, not to police ordinary refusals. Measured over the gates: the max streak reached in
// normal play is 1.
const REJECT_STREAK_CAP = 8;
// Consecutive refusals per side since the last COMMITTED decision (see `noteOutboundChunk`).
let rejectStreak = { p1: 0, p2: 0 };
// The last choice each side sent — named in the fatal message so the loop is diagnosable.
let lastChoiceSent = { p1: null, p2: null };
// `battle.inputLog.length` at the last progress check — the committed-decision cursor.
let lastCommitted = 0;
// Set once a fatal condition is reported: stop relaying / accepting choices for this battle
// (the rust bridge's `stopped`). The process stays alive; the parent raises on `__ERR__`.
let stopped = false;

// Both refusal shapes the sim can emit (Side.emitChoiceError's `[Invalid choice]` /
// `[Unavailable choice]`). Multiline because a chunk may carry several protocol lines.
const REFUSAL_RE = /^\|error\|\[(?:Invalid|Unavailable) choice\]/m;

function out(line) {
  process.stdout.write(line + '\n');
}

// process.stdout.write() to a PIPE is ASYNCHRONOUS, and process.exit() discards whatever is
// still queued in Node userspace — which TRUNCATED large `__RECON__` lines whenever the reader
// was slow to drain (run_20260807_135637's final eval at --eval-concurrency 100: 224
// "failed to capture reconstruction: Incorrect padding" errors, every one in the exit-right-
// after-emit window; the persistent training path never exits, so it never truncated). A
// zero-length write's callback fires only after every previously queued write has been handed
// to the kernel pipe (write callbacks are FIFO), and data already IN the pipe survives child
// exit — so exiting from that callback can never cut a line. Blocks (keeps the process alive)
// until the reader drains the pipe, which is exactly the contract a framed protocol wants.
function exitWhenDrained(code) {
  process.exitCode = code;
  process.stdout.write('', () => process.exit(code));
}

function emit(side, chunk) {
  out(side + ' ' + Buffer.from(chunk, 'utf8').toString('base64'));
}

function fail(msg) {
  out('__ERR__ ' + Buffer.from(String(msg), 'utf8').toString('base64'));
}

// Emit the battle's reconstruction record (once). Best-effort: a capture failure
// must never cost the battle itself, so any error is swallowed silently — the
// Python side degrades gracefully when no __RECON__ arrives.
function emitRecon() {
  if (reconEmitted) return;
  reconEmitted = true;
  try {
    const b = rawStream && rawStream.battle;
    if (!b || !b.inputLog) return;
    const record = {
      v: 1,
      format_id: formatId,
      prng_seed: b.prngSeed,
      input_log: b.inputLog,
      commands: cmdLog,
    };
    out('__RECON__ ' + Buffer.from(JSON.stringify(record), 'utf8').toString('base64'));
  } catch (e) { /* never break the battle for the record */ }
}

// How many decisions the sim has COMMITTED. `Battle.commitChoices` is the only place a
// per-side choice reaches `inputLog` (`>p1 move 1`), and it runs only once every side's
// choice was ACCEPTED — so this is the node analogue of the rust bridge's "a choice was
// accepted" reset. `>start` / `>player` / `>forcelose` also land here; any growth is
// progress, which is exactly what we want.
function committedCount() {
  const b = rawStream && rawStream.battle;
  return b && b.inputLog ? b.inputLog.length : 0;
}

// Bound the refusal↔re-choice exchange. Returns true if this chunk tripped the cap (the
// caller must then NOT relay it — the `__ERR__` frame replaces it).
function noteOutboundChunk(side, chunk) {
  if (stopped) return true;
  // PROGRESS first: the sim committed a decision, so whatever refusals preceded it were part
  // of a boundary that DID advance. This is the ONLY reset — receiving a re-request must NOT
  // reset, or the "[Unavailable choice]" + re-request shape would make the cap unreachable.
  const n = committedCount();
  if (n !== lastCommitted) {
    lastCommitted = n;
    rejectStreak.p1 = 0;
    rejectStreak.p2 = 0;
  }
  if (!REFUSAL_RE.test(chunk)) return false;
  rejectStreak[side] += 1;
  if (process.env.GEN3_BRIDGE_REJECT_DEBUG) process.stderr.write(`REJSTREAK ${side} ${rejectStreak[side]}\n`);
  if (rejectStreak[side] <= REJECT_STREAK_CAP) return false;
  const refusal = chunk.split('\n').find((l) => REFUSAL_RE.test(l)) || chunk;
  stopped = true;
  fail(
    `no-progress reject loop on ${side}: ${rejectStreak[side]} consecutive refusals with no ` +
    `committed decision (turn ${(rawStream && rawStream.battle && rawStream.battle.turn) || '?'}). ` +
    `Last choice sent: "${lastChoiceSent[side]}". Refusal: "${refusal}". The client keeps ` +
    `re-sending a choice the sim refuses, so nothing can advance — failing loud instead of ` +
    `spinning (see REJECT_STREAK_CAP).`
  );
  return true;
}

function pumpSide(side) {
  (async () => {
    try {
      for await (const chunk of streams[side]) {
        if (noteOutboundChunk(side, chunk)) continue;
        emit(side, chunk);
      }
    } catch (e) {
      fail(`${side} stream: ${e && e.message}`);
    } finally {
      endedSides += 1;
      if (endedSides >= 2) {
        emitRecon();
        out('__END__');
        if (persistent) {
          // Reset for the next battle on the SAME process; the next START rebuilds
          // a fresh BattleStream. (A fresh sim per START → no cross-battle state.)
          streams = null;
          rawStream = null;
          endedSides = 0;
        } else {
          exitWhenDrained(0);   // NEVER bare process.exit — __RECON__/__END__ may be undrained
        }
      }
    }
  })();
}

function handleStart(json) {
  const msg = JSON.parse(json);
  if (msg.persistent) persistent = true;
  const stream = new BattleStream();
  streams = getPlayerStreams(stream);
  rawStream = stream;
  formatId = msg.formatid;
  cmdLog = [];
  reconEmitted = false;
  resumeReseed = msg.resumeReseed || null;   // {turn, seed} | null
  reseeded = false;
  // Fresh battle → fresh reject-streak state (persistent children reuse the process).
  rejectStreak = { p1: 0, p2: 0 };
  lastChoiceSent = { p1: null, p2: null };
  lastCommitted = 0;
  stopped = false;
  pumpSide('p1');
  pumpSide('p2');

  const seedClause = msg.seed ? `,"seed":${JSON.stringify(msg.seed)}` : '';
  streams.omniscient.write(`>start {"formatid":"${msg.formatid}"${seedClause}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify(msg.p1)}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify(msg.p2)}`);
}

function handleLine(line) {
  if (!line) return;
  const sp = line.indexOf(' ');
  const cmd = sp === -1 ? line : line.slice(0, sp);
  const rest = sp === -1 ? '' : line.slice(sp + 1);

  switch (cmd) {
    case 'START':
      handleStart(rest);
      break;
    case 'CHOOSE': {
      // rest = "<side> <choice>"
      const s2 = rest.indexOf(' ');
      const side = s2 === -1 ? rest : rest.slice(0, s2);
      const choice = s2 === -1 ? '' : rest.slice(s2 + 1);
      // A reported reject loop stops the session (the rust bridge's `stopped`): relaying
      // further choices would only resume the spin the `__ERR__` just reported.
      if (stopped) break;
      if (streams && streams[side]) {
        lastChoiceSent[side] = choice;
        // Reseed at the START of the divergence turn (battle.turn has already advanced to it after
        // the prior turn resolved), BEFORE this turn's choices commit — so the prefix keeps the
        // recorded dice and only the post-divergence resolution draws from the fresh PRNG. Once.
        if (resumeReseed && !reseeded && rawStream && rawStream.battle
            && rawStream.battle.turn === resumeReseed.turn) {
          rawStream.battle.prng = new PRNG(resumeReseed.seed);
          reseeded = true;
        }
        cmdLog.push([side, choice]);
        streams[side].write(choice);
      }
      break;
    }
    case 'FORCELOSE':
      if (streams) {
        cmdLog.push(['forcelose', rest.trim()]);
        streams.omniscient.write(`>forcelose ${rest.trim()}`);
      }
      break;
    case 'END':
      exitWhenDrained(0);
      break;
    default:
      fail(`unknown command: ${cmd}`);
  }
}

let buf = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  buf += chunk;
  let nl;
  while ((nl = buf.indexOf('\n')) !== -1) {
    const line = buf.slice(0, nl);
    buf = buf.slice(nl + 1);
    try {
      handleLine(line.trim());
    } catch (e) {
      fail(e && e.stack ? e.stack : String(e));
    }
  }
});
process.stdin.on('end', () => exitWhenDrained(0));
process.on('uncaughtException', (e) => {
  fail(e && e.stack ? e.stack : String(e));   // the __ERR__ line is as truncatable as __RECON__
  exitWhenDrained(1);
});
