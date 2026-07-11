// gen_bridge_trapping_capture.js — the TRAPPED-state BRIDGE capture golden.
//
// Deliverable B of extending Phase 0 of the Rust-sim bridge integration: the Phase-0
// capture (`gen_bridge_capture.js`) records `maybeTrapped:true` but NEVER `trapped:true`,
// because its driver picks only LEGAL choices off the omniscient battle object, so it
// never ISSUES a switch the sim rejects — and `'hidden'`→`trapped:true` is patched in
// ONLY after a rejected switch attempt. This sibling captures the missing state: a small
// set of CONSTRUCTED trapping matchups (Arena Trap + Magnet Pull) driven by a SCRIPTED
// plan that deliberately issues a REJECTED switch, so the per-side stream records the
// `|error|[Unavailable choice] Can't switch: The active Pokémon is trapped` chunk + the
// re-request carrying `active[0].trapped:true` — VERBATIM, in the identical TAB grammar
// as `bridge_capture_golden.txt`. This gives Phase 1 a byte target that INCLUDES the
// trapped state machine.
//
// It MIRRORS `local_sim_bridge.js`'s machinery exactly (for byte-fidelity), like
// `gen_bridge_capture.js`: `new BattleStream()` + `getPlayerStreams`, `>start`/`>player`
// to the omniscient side, both per-side streams pumped, choices written via
// `streams[side].write(choice)`. UNLIKE the Phase-0 driver, the choices are a fixed
// SCRIPT (not legal-picked), and a rejected choice's `|error|` + re-request round IS
// part of the captured per-side protocol (the CMD row records the attempt too — the
// bridge relays the rejection + re-request the same way).
//
// FORMAT: gen3customgame (no clause noise — the trapping flags + `|error|` are format-
// independent, and a constructed matchup wants no Sleep/Freeze-Clause shuffles). A fixed
// SEED per scenario → byte-reproducible: `|t:|<unixtime>` normalized to `|t:|<NORMALIZED>`
// (the one documented non-deterministic protocol line), so a plain regen + md5 is stable.
//
// Record grammar (IDENTICAL to bridge_capture_golden.txt; TAB-delimited, the raw-line
// payload is everything after the final tab so it may contain any char except newline):
//   SCEN  <id>
//   TEAM  <id>  <p1|p2>  <packed team>
//   INIT  <id>  <battleNo>  <seed m,n,o,p>  <formatid>
//   CMD   <id>  <battleNo>  <cmdNo>  <side>  <choice>       (INCLUDING a rejected attempt)
//   CHUNK <id>  <battleNo>  <side>  <chunkNo>  <lineNo>  <raw line>   (incl. |error| + the
//                 trapped:true re-request — the whole point of this golden)
//   END   <id>  <battleNo>  <ended:0|1>  <winner:p1|p2|tie|none>
//
// Output: tests/vectors/bridge_trapping_golden.txt
// Run:    node src/rust_sim/harness/gen_bridge_trapping_capture.js
//         (byte-reproducible: run twice, md5 identical.)
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)
'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/bridge_trapping_golden.txt');
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

// The scenarios: constructed trapping matchups + a scripted plan that ISSUES a rejected
// switch (the trapped side) to force the `trapped:true` re-request into the stream.
// A step is { side, choice }; the plan runs until exhausted or the battle ends.
const SCENARIOS = [
  {
    id: 'arena_trap_reject',
    // Arena Trap traps the grounded Snorlax; p2 attempts a switch → rejected → trapped:true.
    seed: [7, 11, 13, 17],
    p1: [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap' })],
    p2: [mon('Snorlax', ['bodyslam', 'splash'], { gender: 'M' }),
         mon('Regice', ['icebeam', 'splash'])],
    plan: [
      { side: 'p2', choice: 'switch 2' },              // REJECTED → |error| + trapped:true re-request
      { side: 'p1', choice: 'move 2' },                // both splash — advance a turn
      { side: 'p2', choice: 'move 2' },
      { side: 'p2', choice: 'switch 2' },              // REJECTED again next turn (still trapped)
      { side: 'p1', choice: 'move 2' },
      { side: 'p2', choice: 'move 2' },
    ],
  },
  {
    id: 'magnet_pull_reject',
    // Magnet Pull traps the Steel Skarmory (Steel/Flying, groundedness irrelevant).
    seed: [23, 29, 31, 37],
    p1: [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull' })],
    p2: [mon('Skarmory', ['drillpeck', 'splash'], { ability: 'Keen Eye', gender: 'M' }),
         mon('Snorlax', ['bodyslam', 'splash'], { gender: 'M' })],
    plan: [
      { side: 'p2', choice: 'switch 2' },              // REJECTED → |error| + trapped:true re-request
      { side: 'p1', choice: 'move 2' },
      { side: 'p2', choice: 'move 2' },
      { side: 'p2', choice: 'switch 2' },              // REJECTED again
      { side: 'p1', choice: 'move 2' },
      { side: 'p2', choice: 'move 2' },
    ],
  },
  {
    id: 'shadow_tag_firm_trap',
    // Shadow Tag traps the foe UNCONDITIONALLY, and — the distinction the request/per-side
    // A/B fuzzer surfaced (`gen3_shadowtag_firm_trap_v1`) — the gen3 mod sets `pokemon.trapped
    // = true` DIRECTLY (NOT the `'hidden'` `tryTrap`), so the FIRST `move` request already
    // carries `trapped:true` (NO `maybeTrapped` phase). A rejected switch then draws
    // `|error|[Invalid choice]…` with NO `trapped:true` re-request (the update is a no-op —
    // the request was already `trapped`), unlike Arena-Trap / Magnet-Pull's `[Unavailable
    // choice]` + re-request above.
    seed: [41, 43, 47, 53],
    p1: [mon('Wobbuffet', ['splash'], { ability: 'Shadow Tag', gender: 'M' })],
    p2: [mon('Snorlax', ['bodyslam', 'splash'], { gender: 'M' }),
         mon('Regice', ['icebeam', 'splash'])],
    plan: [
      { side: 'p2', choice: 'switch 2' },              // REJECTED → |error|[Invalid choice], NO re-request
      { side: 'p1', choice: 'move 1' },
      { side: 'p2', choice: 'move 2' },
      { side: 'p2', choice: 'switch 2' },              // REJECTED again (still firm-trapped)
      { side: 'p1', choice: 'move 1' },
      { side: 'p2', choice: 'move 2' },
    ],
  },
];

// Run ONE scenario, driving the scripted plan and capturing per-side chunks (incl. the
// rejected-switch `|error|` + the re-request). Mirrors gen_bridge_capture.js's driver but
// writes SCRIPTED choices (so a rejected attempt is issued deliberately).
async function runScenario(scen) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const chunks = { p1: [], p2: [] };
  const cmds = [];

  const pump = (side) => (async () => {
    for await (const chunk of streams[side]) chunks[side].push(chunk);
  })();
  pump('p1'); pump('p2');

  const seedClause = scen.seed ? `,"seed":${JSON.stringify(scen.seed)}` : '';
  streams.omniscient.write(`>start {"formatid":"${FORMAT}"${seedClause}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(scen.p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(scen.p2) })}`);
  for (let i = 0; i < 16; i++) await tick();

  const initSeed = stream.battle.prng.getSeed();

  // Drive the scripted plan. Each step writes ONE side's choice (including a rejected
  // switch, which the sim answers with an `|error|` + a re-request on that side —
  // captured in `chunks[side]`). We do NOT gate on legality — the rejection round is the
  // captured artifact.
  for (const step of scen.plan) {
    if (stream.battle.ended) break;
    cmds.push([step.side, step.choice]);
    try { streams[step.side].write(step.choice); } catch (e) { throw new Error(`write ${step.side} ${step.choice}: ${e && e.message}`); }
    for (let i = 0; i < 16; i++) await tick();
  }

  const ended = !!stream.battle.ended;
  const winner = stream.battle.winner;
  for (let i = 0; i < 8; i++) await tick();
  try { streams.omniscient.destroy(); } catch (e) { /* teardown */ }
  return { initSeed, chunks, cmds, ended, winner };
}

function winTok(rec) {
  if (!rec.ended) return 'none';
  if (rec.winner === 'P1') return 'p1';
  if (rec.winner === 'P2') return 'p2';
  if (rec.winner === '') return 'tie';
  return 'none';
}

async function main() {
  const out = [];
  out.push('# bridge_trapping_golden.txt — the TRAPPED-state BRIDGE per-side streams (Deliverable B).');
  out.push('# Constructed Arena-Trap / Magnet-Pull matchups driven by a SCRIPTED plan that ISSUES a REJECTED');
  out.push('# switch, so the per-side stream records the |error|[Unavailable choice] Can\'t switch: The active');
  out.push('# Pokémon is trapped chunk + the re-request carrying active[0].trapped:true — the state gen_bridge_');
  out.push('# capture.js never reaches (its legal-only driver never issues a rejected switch). Same TAB grammar');
  out.push('# as bridge_capture_golden.txt; gen3customgame; fixed per-scenario seed; |t:| normalized → byte-reproducible.');
  out.push('# Record grammar (TAB-delimited; the raw-line payload is everything after the final tab):');
  out.push('#   SCEN  <id>');
  out.push('#   TEAM  <id>  <p1|p2>  <packed team>');
  out.push('#   INIT  <id>  <battleNo>  <seed m,n,o,p>  <formatid>');
  out.push('#   CMD   <id>  <battleNo>  <cmdNo>  <side>  <choice>   (INCLUDING a rejected attempt)');
  out.push('#   CHUNK <id>  <battleNo>  <side>  <chunkNo>  <lineNo>  <raw line>   (incl. |error| + trapped:true re-request)');
  out.push('#   END   <id>  <battleNo>  <ended:0|1>  <winner:p1|p2|tie|none>');
  out.push(`# format ${FORMAT}  scenarios ${SCENARIOS.length}`);

  let battleNo = 0;
  let totalChunks = 0, totalLines = 0, errorLines = 0, trappedFrames = 0, maybeFrames = 0;

  for (const scen of SCENARIOS) {
    const rec = await runScenario(scen);
    const id = scen.id;
    out.push(`SCEN\t${id}`);
    out.push(`TEAM\t${id}\tp1\t${Teams.pack(scen.p1)}`);
    out.push(`TEAM\t${id}\tp2\t${Teams.pack(scen.p2)}`);
    out.push(['INIT', id, battleNo, rec.initSeed, FORMAT].join('\t'));
    rec.cmds.forEach((c, ci) => out.push(['CMD', id, battleNo, ci, c[0], c[1]].join('\t')));
    for (const side of ['p1', 'p2']) {
      rec.chunks[side].forEach((chunk, chunkNo) => {
        totalChunks++;
        const lines = chunk.split('\n');
        lines.forEach((rawLine, lineNo) => {
          const raw = rawLine.startsWith('|t:|') ? '|t:|<NORMALIZED>' : rawLine;
          out.push(['CHUNK', id, battleNo, side, chunkNo, lineNo, raw].join('\t'));
          totalLines++;
          if (raw.startsWith('|error|')) errorLines++;
          if (raw.startsWith('|request|')) {
            const payload = raw.slice('|request|'.length);
            if (payload && payload !== 'null') {
              let obj = null; try { obj = JSON.parse(payload); } catch (e) { obj = null; }
              if (obj && obj.active && obj.active[0]) {
                if (obj.active[0].trapped) trappedFrames++;
                else if (obj.active[0].maybeTrapped) maybeFrames++;
              }
            }
          }
        });
      });
    }
    out.push(['END', id, battleNo, rec.ended ? 1 : 0, winTok(rec)].join('\t'));
    battleNo++;
  }

  // FAIL LOUD if the golden did not actually capture the trapped:true state (the point).
  if (trappedFrames === 0) { console.error('FATAL: no trapped:true frame captured — the golden is pointless'); process.exit(1); }
  if (errorLines === 0) { console.error('FATAL: no |error| line captured — the rejection round is missing'); process.exit(1); }

  const body = out.join('\n') + '\n';
  fs.writeFileSync(OUT, body);
  const bytes = Buffer.byteLength(body, 'utf8');
  console.error(
    `bridge trapping capture: ${battleNo} scenarios, ${totalChunks} chunks, ${totalLines} lines, ` +
    `${errorLines} |error|, ${trappedFrames} trapped:true frames, ${maybeFrames} maybeTrapped:true frames, ` +
    `${bytes} bytes -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
