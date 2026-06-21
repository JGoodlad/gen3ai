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

function out(line) {
  process.stdout.write(line + '\n');
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

function pumpSide(side) {
  (async () => {
    try {
      for await (const chunk of streams[side]) {
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
          process.exit(0);
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
      if (streams && streams[side]) {
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
      process.exit(0);
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
process.stdin.on('end', () => process.exit(0));
process.on('uncaughtException', (e) => {
  fail(e && e.stack ? e.stack : String(e));
  process.exit(1);
});
