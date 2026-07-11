// gen_writeline_capture.js — the WRITE_LINE (streaming drop-in) differential capture.
//
// Drives the REAL Node `BattleStream` with the SAME scenario corpus as
// gen_protocol_capture.js (one scenario source — required from that module), but
// records the omniscient stream **PER WRITE**: after every single `>`-command write
// (`>start`, `>player p1`, `>player p2`, and each side's choice written SEPARATELY —
// the bridge's streaming pattern) it ticks the event loop and attributes every
// newly-flushed omniscient line to that write. `tests/writeline_test.rs` replays the
// identical command stream through the Rust `BattleStream::write_line` and byte-diffs
// the per-write chunks (filtered to the gated line types, `|t:|` normalized — the
// same discipline as protocol_test).
//
// SEED CONVENTION: the battle runs on a raw `>start` seed, but the Rust surface
// replays from the recorded pre-first-decision `initSeed` (the sim's turn-0
// construction window — gender samples, the turn-0 Quick Claw — is deliberately not
// modeled; identical to the protocol golden's convention). The golden stores BOTH:
// the W row for `>start` carries the ORIGINAL command; the INIT row carries the seed
// the Rust replay must use.
//
// Output: tests/vectors/writeline_capture_golden.txt
// Run:    node src/rust_sim/harness/gen_writeline_capture.js

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const { scenarios, buildSeeds, forceSwitchTable, FORMAT } = require('./gen_protocol_capture.js');

const OUT = path.resolve(__dirname, '../tests/vectors/writeline_capture_golden.txt');

// Its own master seed (a DIFFERENT slice of the seed space from the protocol golden,
// so the two gates jointly cover more stochastic branches).
const MASTER_SEED = 0x57524c4e; // 'WRLN'
const SEEDS_PER_SCEN = 2; // 19 scenarios × 2 = 38 battles (>= 20 mandated)

function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Run ONE scenario at one seed, writing every command SEPARATELY and attributing
// the omniscient lines flushed after each write.
async function runBattle(sc, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => {
    for await (const ch of streams.omniscient) {
      for (const l of ch.split('\n')) {
        lines.push(l.startsWith('|t:|') ? '|t:|<NORMALIZED>' : l);
      }
    }
  })();

  const writes = []; // { command, lines: [...] }
  let mark = 0;
  const doWrite = async (command) => {
    try { streams.omniscient.write(command); } catch (e) {}
    for (let i = 0; i < 16; i++) await tick();
    const chunk = lines.slice(mark);
    mark = lines.length;
    writes.push({ command, lines: chunk });
  };

  await doWrite(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  await doWrite(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(sc.p1) })}`);
  await doWrite(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(sc.p2) })}`);

  const script = sc.makeScript();
  const rec = { initSeed: stream.battle.prng.getSeed(), writes, ended: false, winner: null };

  let decisionNo = 0;
  let safety = 0;
  while (!stream.battle.ended && safety < 120) {
    safety++;
    const battle = stream.battle;
    const reqState = battle.requestState;
    if (reqState !== 'move' && reqState !== 'switch') { await tick(); continue; }
    const force = forceSwitchTable(battle);
    const choices = script(decisionNo, battle, reqState, force);
    if (!choices) break;
    // The bridge's streaming pattern: each side's choice is its OWN write.
    if (choices.p1) await doWrite(`>p1 ${choices.p1}`);
    if (choices.p2) await doWrite(`>p2 ${choices.p2}`);
    decisionNo++;
  }
  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

async function main() {
  const seeds = buildSeeds(SEEDS_PER_SCEN, MASTER_SEED);
  const out = [];
  out.push('# writeline_capture_golden.txt — the PER-WRITE omniscient chunks of the real BattleStream.');
  out.push('# The byte target for the Rust `BattleStream::write_line` (gen3_writeline_stream_v1).');
  out.push('# Record grammar (TAB-delimited; the L payload is everything after the 4th tab):');
  out.push('#   SCEN  <id>');
  out.push('#   TEAM  <id>  <p1|p2>  <packed team>');
  out.push('#   INIT  <id>  <battleNo>  <initSeed>   (the seed the Rust >start must use)');
  out.push('#   W     <id>  <battleNo>  <writeNo>  <command verbatim>');
  out.push('#   L     <id>  <battleNo>  <writeNo>  <lineNo>  <raw line>');
  out.push('#   END   <id>  <battleNo>  <ended:0|1>  <winner>');

  const S = scenarios();
  let battles = 0;
  let totalWrites = 0;
  let totalLines = 0;
  const failures = [];

  for (const sc of S) {
    out.push(`SCEN\t${sc.id}`);
    out.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    out.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);
    let battleNo = 0;
    for (const seed of seeds) {
      let rec;
      try { rec = await runBattle(sc, seed); }
      catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); battleNo++; continue; }
      out.push(['INIT', sc.id, battleNo, rec.initSeed].join('\t'));
      rec.writes.forEach((w, wi) => {
        out.push(['W', sc.id, battleNo, wi, w.command].join('\t'));
        totalWrites++;
        w.lines.forEach((raw, li) => {
          out.push(['L', sc.id, battleNo, wi, li, raw].join('\t'));
          totalLines++;
        });
      });
      out.push(['END', sc.id, battleNo, rec.ended ? 1 : 0, rec.winner || 'none'].join('\t'));
      battles++;
      battleNo++;
    }
  }

  if (failures.length) {
    console.error('WRITELINE CAPTURE FAILURES:\n  ' + failures.slice(0, 20).join('\n  '));
    process.exit(1);
  }
  fs.writeFileSync(OUT, out.join('\n') + '\n');
  console.error(`writeline capture: ${battles} battles, ${totalWrites} writes, ${totalLines} raw lines -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
