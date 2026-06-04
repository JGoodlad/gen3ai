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
//   __END__              battle over, both side streams closed
//   __ERR__ <base64(msg)>  fatal error
//
// Base64 per chunk because protocol text contains \n, |, and arbitrary JSON in
// |request| — one stdout line == exactly one side-tagged chunk, unambiguous to
// demux on the Python side.
'use strict';

const path = require('path');
const psPath = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(psPath, 'dist/sim/battle-stream'));

let streams = null;
let endedSides = 0;
// Sticky: once any START asks for it, the process survives battle ends and waits for
// the next START instead of exiting.
let persistent = false;

function out(line) {
  process.stdout.write(line + '\n');
}

function emit(side, chunk) {
  out(side + ' ' + Buffer.from(chunk, 'utf8').toString('base64'));
}

function fail(msg) {
  out('__ERR__ ' + Buffer.from(String(msg), 'utf8').toString('base64'));
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
        out('__END__');
        if (persistent) {
          // Reset for the next battle on the SAME process; the next START rebuilds
          // a fresh BattleStream. (A fresh sim per START → no cross-battle state.)
          streams = null;
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
      if (streams && streams[side]) streams[side].write(choice);
      break;
    }
    case 'FORCELOSE':
      if (streams) streams.omniscient.write(`>forcelose ${rest.trim()}`);
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
