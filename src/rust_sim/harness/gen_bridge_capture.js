// gen_bridge_capture.js — the CAPTURE-ORACLE for the local-sim BRIDGE per-side streams.
//
// Phase 0 of turning the Rust `pokesim` crate into a drop-in replacement for the Node
// bridge `src/utils/bridge/local_sim_bridge.js`. That bridge drives a REAL in-process
// `BattleStream`, calls `getPlayerStreams(stream)` to SPLIT the omniscient protocol into
// PER-SIDE streams (`streams.p1` / `streams.p2` — each side sees only its OWN chunks,
// ending in a `|request|{...}` JSON line when that side must choose), and relays those
// chunks to poke-env. The Rust crate today emits only the OMNISCIENT stream and NO
// `|request|`. THIS harness captures the ground-truth PER-SIDE streams (esp. the
// `|request|` JSON) so later phases can be validated BYTE-FOR-BYTE.
//
// It MIRRORS `local_sim_bridge.js`'s machinery exactly (for byte-fidelity):
//   * `new BattleStream()` + `getPlayerStreams(stream)`,
//   * `>start {"formatid":...,"seed":...}` / `>player p1 {...}` / `>player p2 {...}`
//     written to `streams.omniscient`,
//   * both sides pumped with `for await (const chunk of streams[side])`,
//   * a choice written via `streams[side].write(choice)`.
// The CHOICES are driven from the omniscient `battle` object (reusing gen_e2e_fuzz.js's
// team loading + seeded random-legal-choice picking — ANY legal move/switch, since we are
// capturing the sim's OUTPUT, not restricting to modeled mechanics), but the CAPTURE is of
// the per-side `streams.p1`/`streams.p2` chunks — the identical chunks the bridge would
// base64-emit. We record RAW chunk text (not base64 — human-readable, line-addressable
// diffing).
//
// Format: gen3ou (the production training format — so the golden reflects the real
// deployment, incl. Sleep/Freeze Clause). Distinct real teams are paired; a battle seed +
// a separate seeded choice-RNG are ALL derived from a fixed MASTER_SEED (printed at start,
// env-overridable) so a plain regen reproduces the golden BYTE-FOR-BYTE.
//
// Record grammar (TAB-delimited; the raw-line payload is everything after the last field's
// tab, so it may itself contain any character except a newline):
//   SCEN  <id>
//   TEAM  <id>  <p1|p2>  <packed team>
//   INIT  <id>  <battleNo>  <seed m,n,o,p>  <formatid>   (the resolved battle seed + format)
//   CMD   <id>  <battleNo>  <cmdNo>  <side>  <choice>     (each `streams[side].write(choice)`)
//   CHUNK <id>  <battleNo>  <side>  <chunkNo>  <lineNo>  <raw line>
//                 (ONE record per raw LINE — a multi-line chunk is expanded so a byte-diff
//                  is line-addressable; `chunkNo` groups the lines of one yielded chunk)
//   END   <id>  <battleNo>  <ended:0|1>  <winner:p1|p2|tie|none>
//
// Output: tests/vectors/bridge_capture_golden.txt
// Run:    node src/rust_sim/harness/gen_bridge_capture.js
//         env knobs: BRIDGE_MASTER_SEED (hex or dec), BRIDGE_BATTLES (default 30)
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
// The SAME battle-stream the bridge (local_sim_bridge.js) requires.
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

// Reuse gen_e2e_fuzz.js's team loading + legal-choice enumeration + the seeded RNG
// helpers (ONE source of truth — never re-implement team validation / choice legality).
const e2e = require('./gen_e2e_fuzz.js');

const OUT = path.resolve(__dirname, '../tests/vectors/bridge_capture_golden.txt');

// gen3ou — the production training format (so the golden reflects the real deployment,
// incl. the Sleep/Freeze Clause SetStatus shuffles + the maybeTrapped request shapes).
const FORMAT = 'gen3ou';

// A fixed master seed (env-overridable), printed at start so the run is reproducible.
function parseSeed(s, dflt) {
  if (s === undefined) return dflt;
  const n = /^0x/i.test(s) ? parseInt(s, 16) : parseInt(s, 10);
  return Number.isFinite(n) ? (n >>> 0) : dflt;
}
const MASTER_SEED = parseSeed(process.env.BRIDGE_MASTER_SEED, 0x42524447); // 'BRDG'
const BATTLES = Number(process.env.BRIDGE_BATTLES || 30);
// Safety cap on decisions per battle (real gen3ou teams can grind; a stall aborts loud).
const SAFETY = 1000;

function tick() { return new Promise((r) => setTimeout(r, 0)); }

// ── The per-side capture driver ──────────────────────────────────────────────
// Mirrors local_sim_bridge.js: construct a BattleStream, split it with
// getPlayerStreams, write >start/>player to the omniscient side, pump BOTH per-side
// streams (recording every chunk), and — whenever a side's chunk carries an ACTIONABLE
// `|request|` — pick a random LEGAL choice from the OMNISCIENT battle object (the e2e
// legal-choice logic) and `streams[side].write(choice)`. Runs to game-end.
async function runBattle(p1Packed, p2Packed, seed, chooseSeed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);

  // Per-side capture: for EACH side, the ordered list of chunks getPlayerStreams yields,
  // each chunk its own array of raw lines (empty-string lines dropped like the bridge's
  // demux, which base64s whole chunks — we keep the chunk grouping via chunkNo).
  const chunks = { p1: [], p2: [] };
  // The ordered (side, choice) commands actually issued (for CMD rows + the Rust driver).
  const cmds = [];

  // Start pumping BOTH per-side streams BEFORE writing START (the bridge's pattern —
  // `for await (const chunk of streams[side])`), recording every yielded chunk in order.
  const pump = (side) => (async () => {
    for await (const chunk of streams[side]) {
      chunks[side].push(chunk);
    }
  })();
  pump('p1');
  pump('p2');

  // Mirror local_sim_bridge.js's START exactly.
  const seedClause = seed ? `,"seed":${JSON.stringify(seed)}` : '';
  streams.omniscient.write(`>start {"formatid":"${FORMAT}"${seedClause}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1Packed })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2Packed })}`);
  // Let the framing + the first requests flush.
  for (let i = 0; i < 16; i++) await tick();

  // `initSeed` is the POST-CONSTRUCTION seed (read after `>player`), so the Rust offline
  // replay resumes draw-free and SKIPS the turn-0 `endTurn` where gen3 rolls turn 1's Quick
  // Claw (`randomChance(1,5)`, read the next turn). That bit is unrecoverable from the seed,
  // so capture it alongside — `gen3_turn0_quick_claw_capture_v1`, the sibling of the
  // `ab_fuzz` INIT capture. (The LIVE bridge models the real construction and ignores this.)
  const rec = {
    initSeed: stream.battle.prng.getSeed(),
    quickClawRoll: !!stream.battle.quickClawRoll,
    chunks, cmds, ended: false, winner: null,
  };

  // The request-driven loop. We DRIVE choices off the omniscient `battle` object (the
  // e2e legal-choice logic reads `battle.requestState`/`sides[i].activeRequest`); we
  // CAPTURE the per-side chunks (already accumulating via pump). A side chooses whenever
  // it has an ACTIONABLE request — `requestState` is 'move' or 'switch' with that side
  // flagged (a `{"wait":true}` side is NOT flagged, so it is skipped).
  const rng = e2e.mulberry32(chooseSeed);
  let safety = 0;
  while (!stream.battle.ended && safety < SAFETY) {
    safety++;
    const battle = stream.battle;
    const reqState = battle.requestState;
    if (reqState !== 'move' && reqState !== 'switch') { await tick(); continue; }

    // Determine which side(s) must act this decision, and pick a legal choice for each.
    const toWrite = []; // [side, choice]
    if (reqState === 'switch') {
      // Forced replacement(s): the flagged side(s) pick a bench mon.
      for (let i = 0; i < 2; i++) {
        const req = battle.sides[i].activeRequest;
        if (req && req.forceSwitch && req.forceSwitch[0]) {
          const c = pickReplacement(battle, i, rng);
          if (!c) throw new Error(`no legal replacement for p${i + 1} (stall)`);
          toWrite.push([`p${i + 1}`, c]);
        }
      }
    } else {
      // Move request: BOTH sides pick (any legal move — or a switch when trapped/forced).
      for (let i = 0; i < 2; i++) {
        const c = pickAnyLegal(battle, i, rng);
        if (!c) throw new Error(`no legal choice for p${i + 1} (stall)`);
        toWrite.push([`p${i + 1}`, c]);
      }
    }
    if (toWrite.length === 0) throw new Error('actionable request but no side to write (stall)');

    // Write each side's choice to its OWN per-side stream (the bridge's pattern:
    // streams[side].write(choice)), recording the command.
    for (const [side, choice] of toWrite) {
      cmds.push([side, choice]);
      try { streams[side].write(choice); } catch (e) { throw new Error(`write ${side} ${choice}: ${e && e.message}`); }
    }
    for (let i = 0; i < 16; i++) await tick();
  }
  if (safety >= SAFETY) throw new Error('battle did not advance to game-end (safety cap)');

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  // Drain the streams (flush any tail chunks — e.g. the terminal |win| the per-side
  // streams push on the final update) before tearing down.
  for (let i = 0; i < 8; i++) await tick();
  try { streams.omniscient.destroy(); } catch (e) { /* best-effort teardown */ }
  return rec;
}

// ── Legal-choice helpers (any legal move/switch — the sim's OUTPUT is what we capture) ─
// Unlike gen_e2e_fuzz.js's pickMove (restricted to MODELED mechanics), we pick ANY legal
// move — we are recording the sim's per-side stream, so unmodeled mechanics are fine.
function pickAnyLegal(battle, side, rng) {
  const req = battle.sides[side].activeRequest;
  if (!req || !req.active || !req.active[0]) return null;
  const moves = req.active[0].moves || [];
  const legalMoveSlots = [];
  for (let k = 0; k < moves.length; k++) {
    if (!moves[k].disabled) legalMoveSlots.push(k);
  }
  // Respect the sim's trapped flag (a trapped mon's switch is rejected → stall).
  const active = battle.sides[side].active[0];
  const isTrapped = !!(active && active.trapped);
  const switchSlots = isTrapped ? [] : legalSwitchSlots(battle, side);

  if (legalMoveSlots.length === 0) {
    // No usable move (all disabled / 0-PP → the request offers Struggle): pick a switch,
    // else move 1 (the sim substitutes Struggle for a no-usable-move `move 1`).
    if (switchSlots.length > 0) return `switch ${switchSlots[e2e.randInt(rng, switchSlots.length)] + 1}`;
    return 'move 1';
  }
  // Mostly attack; ~1/6 voluntarily switch (to exercise switch-request + forceSwitch
  // request shapes on real teams).
  if (switchSlots.length > 0 && rng() < 1 / 6) {
    return `switch ${switchSlots[e2e.randInt(rng, switchSlots.length)] + 1}`;
  }
  return `move ${legalMoveSlots[e2e.randInt(rng, legalMoveSlots.length)] + 1}`;
}

function legalSwitchSlots(battle, side) {
  const s = battle.sides[side];
  const out = [];
  for (let k = 0; k < s.pokemon.length; k++) {
    if (s.pokemon[k] !== s.active[0] && !s.pokemon[k].fainted) out.push(k);
  }
  return out;
}
function pickReplacement(battle, side, rng) {
  const slots = legalSwitchSlots(battle, side);
  if (slots.length === 0) return null;
  return `switch ${slots[e2e.randInt(rng, slots.length)] + 1}`;
}

// A well-spread gen5 battle seed from a 32-bit state (mirrors gen_e2e_fuzz.js's seedFrom
// via the exported helper, so the seed pool shape matches the other goldens).
function winTok(rec) {
  if (!rec.ended) return 'none';
  if (rec.winner === 'P1') return 'p1';
  if (rec.winner === 'P2') return 'p2';
  if (rec.winner === '') return 'tie';
  return 'none';
}

async function main() {
  const t0 = Date.now();
  console.error(`bridge capture: MASTER_SEED 0x${MASTER_SEED.toString(16)} (${MASTER_SEED}), format ${FORMAT}, target ${BATTLES} battles`);

  const { teams, skipped, total } = e2e.loadTeams();
  console.error(`teams: loaded ${teams.length} / ${total} (.txt), skipped ${skipped} (import/validate)`);
  if (teams.length < 8) { console.error('too few valid teams loaded'); process.exit(1); }

  const pairRng = e2e.mulberry32(MASTER_SEED);

  const out = [];
  out.push('# bridge_capture_golden.txt — the PER-SIDE (p1/p2) protocol streams of the REAL Node BattleStream.');
  out.push('# The byte target for the Rust port\'s BRIDGE surface (getPlayerStreams split + the |request| JSON emitter).');
  out.push('# Captured EXACTLY as local_sim_bridge.js pumps them: >start/>player to omniscient, per-side chunks recorded');
  out.push('# in yield order, choices driven off the omniscient battle object (any legal move/switch), gen3ou, to game-end.');
  out.push('# Record grammar (TAB-delimited; the raw-line payload is everything after the final tab):');
  out.push('#   SCEN  <id>');
  out.push('#   TEAM  <id>  <p1|p2>  <packed team>');
  out.push('#   INIT  <id>  <battleNo>  <seed m,n,o,p>  <formatid>');
  out.push('#   CMD   <id>  <battleNo>  <cmdNo>  <side>  <choice>');
  out.push('#   CHUNK <id>  <battleNo>  <side>  <chunkNo>  <lineNo>  <raw line>');
  out.push('#   END   <id>  <battleNo>  <ended:0|1>  <winner:p1|p2|tie|none>');
  out.push(`# MASTER_SEED ${MASTER_SEED}  format ${FORMAT}  battles ${BATTLES}`);

  let battleNo = 0;
  let tries = 0;
  const MAX_TRIES = BATTLES * 40;
  let totalChunks = 0; let totalLines = 0; let requestFrames = 0; let forceSwitchFrames = 0;
  let trappedFrames = 0;
  const failures = [];
  // Collect the FIRST few distinct request JSONs of each kind for the schema report.
  const schemaSamples = { move: [], forceSwitch: [], trapped: [] };

  while (battleNo < BATTLES && tries < MAX_TRIES && teams.length >= 2) {
    tries++;
    const ia = e2e.randInt(pairRng, teams.length);
    let ib = e2e.randInt(pairRng, teams.length);
    if (ib === ia) ib = (ib + 1) % teams.length;
    if (ib === ia) continue;
    const seedState = (Math.imul(Math.floor(pairRng() * 4294967296), 1) ^ (tries * 2654435761)) >>> 0;
    const seed = e2e.seedFrom(seedState);
    const chooseSeed = (Math.floor(pairRng() * 4294967296) ^ 0x9e3779b9) >>> 0;

    let rec;
    try {
      rec = await runBattle(teams[ia].packed, teams[ib].packed, seed, chooseSeed);
    } catch (e) {
      // A stall / write error is a poisoned scenario — record it and try the next pair.
      failures.push(`battle try ${tries} (${teams[ia].file} vs ${teams[ib].file}): ${e && e.message}`);
      continue;
    }
    if (!rec.ended) { failures.push(`battle try ${tries}: did not end`); continue; }

    const id = `bridge_${battleNo}`;
    out.push(`SCEN\t${id}`);
    out.push(`TEAM\t${id}\tp1\t${teams[ia].packed}`);
    out.push(`TEAM\t${id}\tp2\t${teams[ib].packed}`);
    // `prng.getSeed()` returns the comma-joined seed string (e.g. "1,2,3,4") — the same
    // form the Rust replay must feed to its `>start` seed; recorded verbatim.
    // The 6th field is the turn-0 `quickClawRoll`; `parse_bridge_golden` treats it as
    // OPTIONAL (absent → false), so every pre-existing golden stays byte-replayable.
    out.push(['INIT', id, battleNo, rec.initSeed, FORMAT, rec.quickClawRoll ? 1 : 0].join('\t'));
    rec.cmds.forEach((c, ci) => {
      out.push(['CMD', id, battleNo, ci, c[0], c[1]].join('\t'));
    });
    for (const side of ['p1', 'p2']) {
      rec.chunks[side].forEach((chunk, chunkNo) => {
        totalChunks++;
        const lines = chunk.split('\n');
        lines.forEach((rawLine, lineNo) => {
          // Normalize the wall-clock `|t:|<unixtime>` lines (the ONE documented
          // non-deterministic protocol line — poke-env ignores them) so a plain regen
          // reproduces the golden byte-for-byte, exactly as gen_writeline_capture.js does.
          const raw = rawLine.startsWith('|t:|') ? '|t:|<NORMALIZED>' : rawLine;
          out.push(['CHUNK', id, battleNo, side, chunkNo, lineNo, raw].join('\t'));
          totalLines++;
          if (raw.startsWith('|request|')) {
            const payload = raw.slice('|request|'.length);
            if (payload && payload !== 'null') {
              let obj = null;
              try { obj = JSON.parse(payload); } catch (e) { obj = null; }
              if (obj && !obj.wait) {
                requestFrames++;
                if (obj.forceSwitch) {
                  forceSwitchFrames++;
                  if (schemaSamples.forceSwitch.length < 3) schemaSamples.forceSwitch.push(payload);
                } else if (obj.active) {
                  const a = obj.active[0] || {};
                  if (a.trapped || a.maybeTrapped) {
                    trappedFrames++;
                    if (schemaSamples.trapped.length < 3) schemaSamples.trapped.push(payload);
                  }
                  if (schemaSamples.move.length < 3) schemaSamples.move.push(payload);
                }
              }
            }
          }
        });
      });
    }
    out.push(['END', id, battleNo, rec.ended ? 1 : 0, winTok(rec)].join('\t'));
    battleNo++;
  }

  if (battleNo < BATTLES) {
    console.error(`bridge capture: only produced ${battleNo}/${BATTLES} battles after ${tries} tries`);
    if (failures.length) console.error('  failures:\n    ' + failures.slice(0, 20).join('\n    '));
    // A short golden is still better than none IF we hit the target; short of the target we fail loud.
    process.exit(1);
  }

  const body = out.join('\n') + '\n';
  fs.writeFileSync(OUT, body);
  const bytes = Buffer.byteLength(body, 'utf8');
  console.error(
    `bridge capture: ${battleNo} battles, ${totalChunks} chunks, ${totalLines} lines, ` +
    `${requestFrames} |request| frames (${forceSwitchFrames} forceSwitch, ${trappedFrames} trapped/maybeTrapped), ` +
    `${bytes} bytes -> ${OUT}`);
  if (failures.length) console.error(`  (${failures.length} pair(s) rejected while sampling — expected on real teams)`);
  console.error(`done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  // Emit the schema samples (Deliverable 2) to a sidecar so the report can pretty-print them.
  const schemaOut = path.resolve(__dirname, '../tests/vectors/bridge_request_schema_samples.json');
  fs.writeFileSync(schemaOut, JSON.stringify(schemaSamples, null, 2) + '\n');
  console.error(`request-schema samples -> ${schemaOut}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
