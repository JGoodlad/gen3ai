// probe_bridge_trapping.js — the TRAPPING-REQUEST characterization probe for the
// BRIDGE per-side `|request|` surface. This is Deliverable A of extending Phase 0 of
// the Rust-sim bridge integration to fully pin the SUBTLEST part of the poke-env
// `|request|` JSON: the Arena-Trap / Magnet-Pull SWITCH-legality flags
// (`maybeTrapped` / `trapped`) + the `|error|` re-request round.
//
// WHY this exists (read gen_bridge_capture.js first): the Phase-0 capture harness
// drives a real in-process `BattleStream` + `getPlayerStreams` and records the
// per-side chunks INCLUDING the `|request|` JSON — but it only ever captured
// `maybeTrapped:true`, NEVER `trapped:true`, because the sim only patches
// `'hidden'`→`trapped:true` AFTER a side ATTEMPTS a switch the sim REJECTS, and the
// capture driver respected `active.trapped` so it never ISSUED a rejected switch.
// This probe deliberately ISSUES the rejected switch to trigger the reveal, and
// records the exact `active[0]` flags + the `|error|` text on both sides.
//
// The gen3 trapping SEMANTICS are already modeled in the Rust engine
// (`gen3_trapping_v1`, `src/rust_sim/CLAUDE.md` → "## Trapping") — this probe pins
// the REQUEST-DISPLAY state machine the Rust `|request|` EMITTER (a Phase-1
// deliverable) must reproduce so poke-env's `parse_request` derives the identical
// switch-legality (`src/poke_env/battle/battle.py`: `active[0].trapped` →
// `self._trapped` → `available_switches` empty; `maybeTrapped` → `self._maybe_trapped`,
// display-only). The sim is the ONLY oracle — every source read is a hypothesis.
//
// ── THE STATE MACHINE (what the Rust |request| emitter must implement) ────────────
//   Recompute trap-ness PER REQUEST from the engine's `is_trapped` (never sticky):
//     • trapped-WITH-live-bench + NO switch attempted yet  →  active[0].maybeTrapped:true
//       (the ability trap is 'hidden' → the sim marks it "maybe" until a rejection).
//     • the side ATTEMPTS a switch  →  the sim emits
//         |error|[Unavailable choice] Can't switch: The active Pokémon is trapped
//       to THAT side and RE-REQUESTS with active[0].trapped:true patched in
//       (maybeTrapped is DROPPED; the `moves` array is UNCHANGED — only SWITCH is
//       disallowed, every move stays legal).
//     • NO live bench (this is the last mon)  →  BOTH flags OMITTED (the
//       getMoveRequestData canSwitchIn / isLastActive gates: no bench to trap toward).
//     • the trap LIFTS (trapper faints / switches out)  →  the next request has
//       NEITHER flag (per-request recompute, not a sticky patch).
//   A NON-trapped foe's request never carries either flag.
//
// ── THE MATRIX (9 cases; each asserts the observed flags vs the expected) ─────────
//   1 Arena Trap vs a GROUNDED foe (+bench)  → maybeTrapped → (reject) → trapped
//   2 Arena Trap vs FLYING + vs LEVITATE     → NOT trapped, switch ACCEPTED (controls)
//   3 Arena Trap vs a grounded GHOST         → IS trapped in gen3 (no trap type-immunity)
//   4 Magnet Pull vs a STEEL foe (Skarmory)  → trapped (Steel/Flying still trapped)
//   5 Magnet Pull vs a NON-steel foe         → NOT trapped (control)
//   6 Dugtrio MIRROR (both Arena Trap)       → mutual maybeTrapped → trapped
//   7 Magneton MIRROR (both Magnet Pull)     → mutual trapped
//   8 trapped mon is the LAST mon (no bench) → NEITHER flag
//   9 trapper faints / leaves                → the flag LIFTS next request (non-sticky)
//
// Constructed gen3customgame teams (no clause noise), MODELED-in-gen3 species/
// abilities, genders PINNED explicitly (an unspecified gender on a ratio species
// makes the sim draw at construction — avoid it). Each case FAILS LOUD on a flag
// mismatch, so this probe doubles as a real-sim characterization GATE.
//
// Run:  node src/rust_sim/harness/probe_bridge_trapping.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)
'use strict';

const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const SEED = [7, 11, 13, 17];
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };

// A pinned-gender mon (default genderless 'N' — a ratio species MUST pass an
// explicit 'M'/'F' or the sim draws a construction-time gender `sample`).
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

// Read the ACTIVE-request `active[0]` flag form for a side, as one of the three
// canonical labels the emitter must reproduce.
function flagForm(a0) {
  if (!a0) return 'no-active-request';
  const t = Object.prototype.hasOwnProperty.call(a0, 'trapped') ? a0.trapped : undefined;
  const m = Object.prototype.hasOwnProperty.call(a0, 'maybeTrapped') ? a0.maybeTrapped : undefined;
  if (t === true) return 'trapped';
  if (m === true) return 'maybeTrapped';
  if (t === undefined && m === undefined) return 'neither';
  return `trapped=${JSON.stringify(t)} maybeTrapped=${JSON.stringify(m)}`;
}

// ── The per-side capture driver (mirrors gen_bridge_capture.js / local_sim_bridge.js) ──
// Pump BOTH per-side streams, write choices to streams[side], and expose the LATEST
// per-side `active[0]` request object + any `|error|` line each write produced.
async function makeBattle(p1team, p2team) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = { p1: [], p2: [] };            // ALL raw lines each side saw, in order
  const pump = (side) => (async () => {
    for await (const chunk of streams[side]) {
      for (const l of chunk.split('\n')) if (l) lines[side].push(l);
    }
  })();
  pump('p1'); pump('p2');
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(SEED)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 16; i++) await tick();
  return { stream, streams, lines };
}

// Latest `|request|` active[0] object a side currently sees (its pending decision).
function latestActive0(lines) {
  for (let i = lines.length - 1; i >= 0; i--) {
    if (lines[i].startsWith('|request|')) {
      const payload = lines[i].slice('|request|'.length);
      if (!payload || payload === 'null') continue;
      let obj = null; try { obj = JSON.parse(payload); } catch (e) { obj = null; }
      if (!obj || obj.wait) continue;
      return obj.active ? (obj.active[0] || null) : null;   // forceSwitch has no active
    }
  }
  return null;
}

// Write a choice to one side and return { error, active0After } observed on that side.
async function writeAndObserve(bat, side, choice) {
  const start = bat.lines[side].length;
  bat.streams[side].write(choice);
  for (let i = 0; i < 16; i++) await tick();
  const fresh = bat.lines[side].slice(start);
  const error = fresh.find((l) => l.startsWith('|error|')) || null;
  return { error, active0After: latestActive0(bat.lines[side]) };
}

// ── The assertion harness ─────────────────────────────────────────────────────────
let failures = 0;
const rows = [];   // { case, expectedFlag, observedFlag, error, switchAccepted, note }

function record(caseName, expectedFlag, observedFlag, errorText, switchAccepted, note) {
  const ok = expectedFlag === observedFlag;
  if (!ok) failures++;
  rows.push({ caseName, expectedFlag, observedFlag, errorText: errorText || '(none)', switchAccepted, note, ok });
  const tag = ok ? 'OK ' : '!!!';
  console.log(`  [${tag}] ${caseName}`);
  console.log(`         expected active[0] flag = ${expectedFlag}`);
  console.log(`         observed active[0] flag = ${observedFlag}`);
  console.log(`         |error| = ${errorText || '(none)'}`);
  console.log(`         switch accepted = ${switchAccepted}${note ? `   (${note})` : ''}`);
}

// Assert the `moves` array is preserved (unchanged legality) across the trap reveal.
function movesUnchanged(before, after) {
  const ids = (a0) => a0 && a0.moves ? a0.moves.map((m) => `${m.id}:${m.disabled ? 'D' : 'L'}`).join(',') : '?';
  return ids(before) === ids(after);
}

// A full "trapped-with-bench" case: capture the pre-attempt flag, ISSUE the switch to
// trigger the reveal, and record the `|error|` + the re-request flag.
// The trapped side is `side` (the FOE of the trapper), switching to bench slot 2.
async function trappedWithBenchCase(caseName, p1team, p2team, opts = {}) {
  const trappedSide = opts.trappedSide || 'p2';
  const otherSide = trappedSide === 'p2' ? 'p1' : 'p2';
  const bat = await makeBattle(p1team, p2team);
  // Advance to a real move-request boundary so trap-ness is settled (leads' trapped
  // is computed at the pre-turn-1 endTurn, so it's already present at the first
  // request; but issue a benign splash-splash turn first when opts.warm to prove the
  // per-request recompute across a turn).
  if (opts.warm) {
    await writeAndObserve(bat, 'p1', 'move 2');
    await writeAndObserve(bat, 'p2', 'move 2');
  }
  const preActive0 = latestActive0(bat.lines[trappedSide]);
  const preFlag = flagForm(preActive0);
  const attempt = await writeAndObserve(bat, trappedSide, 'switch 2');
  const postFlag = flagForm(attempt.active0After);
  const switchAccepted = !attempt.error;
  const movesOk = movesUnchanged(preActive0, attempt.active0After);
  // The FULL expectation for a trapped-with-bench case: preFlag maybeTrapped → the
  // reject → postFlag trapped, moves preserved, switch NOT accepted.
  const seq = `${preFlag}→${postFlag}`;
  record(caseName, 'maybeTrapped→trapped', seq, attempt.error, switchAccepted,
    `moves preserved=${movesOk}; the un-trapped side stays free`);
  if (!movesOk) { failures++; console.log('         !!! MOVES ARRAY CHANGED across the reveal (expected UNCHANGED)'); }
  // Capture the pretty-printed forms for the report (the first realized trapped case).
  if (!trappedWithBenchCase._captured && postFlag === 'trapped') {
    trappedWithBenchCase._captured = {
      caseName,
      trapped_active0: attempt.active0After,
      maybeTrapped_active0: preActive0,
      error: attempt.error,
    };
  }
  try { bat.stream.omniscient.destroy(); } catch (e) { /* teardown */ }
  return { bat, preFlag, postFlag, error: attempt.error };
}

// A control case: a NON-trapped foe's voluntary switch is ACCEPTED, neither flag.
async function freeCase(caseName, p1team, p2team, opts = {}) {
  const trappedSide = opts.trappedSide || 'p2';
  const bat = await makeBattle(p1team, p2team);
  const preActive0 = latestActive0(bat.lines[trappedSide]);
  const preFlag = flagForm(preActive0);
  // A free switch to slot 2 is accepted (no error); the other side moves.
  const otherSide = trappedSide === 'p2' ? 'p1' : 'p2';
  const attempt = await writeAndObserve(bat, trappedSide, 'switch 2');
  await writeAndObserve(bat, otherSide, 'move 2');
  const switchAccepted = !attempt.error;
  record(caseName, 'neither', preFlag, attempt.error, switchAccepted, opts.note || '');
  if (!switchAccepted) { failures++; console.log('         !!! a FREE switch was REJECTED'); }
  try { bat.stream.omniscient.destroy(); } catch (e) { /* teardown */ }
  return bat;
}

async function main() {
  const d = Dex.forFormat(FORMAT);
  console.log('=== resolved gen3 trapping ability facts (the mod-chain-resolved handlers) ===');
  for (const id of ['arenatrap', 'magnetpull']) {
    const ab = d.abilities.get(id);
    console.log(`  ${id}: onFoeTrapPokemon=${!!ab.onFoeTrapPokemon} onAnyTrapPokemon=${!!ab.onAnyTrapPokemon} ` +
      `onFoeMaybeTrapPokemon=${!!ab.onFoeMaybeTrapPokemon} onAnyMaybeTrapPokemon=${!!ab.onAnyMaybeTrapPokemon}`);
  }
  console.log(`  Ghost.damageTaken.trapped = ${d.types.get('Ghost').damageTaken['trapped']} ` +
    `(undefined ⇒ NO trap type-immunity in Showdown-gen3 → a grounded Ghost IS trapped)\n`);

  console.log('=== CASE 1 — Arena Trap vs a GROUNDED foe (+bench): maybeTrapped → (reject) → trapped ===');
  await trappedWithBenchCase('1. Arena Trap (Dugtrio) vs a grounded Snorlax (+bench)',
    [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap' })],
    [mon('Snorlax', ['bodyslam', 'splash'], { gender: 'M' }),
     mon('Regice', ['icebeam', 'splash'])]);

  console.log('\n=== CASE 2 — Arena Trap vs FLYING + vs LEVITATE (controls): neither, switch ACCEPTED ===');
  await freeCase('2a. Arena Trap vs a FLYING foe (Zapdos)',
    [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap' })],
    [mon('Zapdos', ['thunderbolt', 'splash'], { ability: 'Pressure' }),
     mon('Regice', ['icebeam', 'splash'])],
    { note: 'Flying escapes Arena Trap (not grounded)' });
  await freeCase('2b. Arena Trap vs a LEVITATE foe (Gengar)',
    [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap' })],
    [mon('Gengar', ['thunderbolt', 'splash'], { ability: 'Levitate', gender: 'M' }),
     mon('Snorlax', ['bodyslam', 'splash'], { gender: 'M' })],
    { note: 'Levitate escapes Arena Trap (not grounded)' });

  console.log('\n=== CASE 3 — Arena Trap vs a grounded GHOST (Banette): IS trapped in gen3 ===');
  await trappedWithBenchCase('3. Arena Trap (Dugtrio) vs a grounded GHOST (Banette)',
    [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap' })],
    [mon('Banette', ['shadowball', 'splash'], { ability: 'Insomnia', gender: 'M' }),
     mon('Snorlax', ['bodyslam', 'splash'], { gender: 'M' })]);

  console.log('\n=== CASE 4 — Magnet Pull vs a STEEL foe (Skarmory, Steel/Flying): trapped ===');
  await trappedWithBenchCase('4. Magnet Pull (Magneton) vs Skarmory (Steel/Flying — groundedness irrelevant)',
    [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull' })],
    [mon('Skarmory', ['drillpeck', 'splash'], { ability: 'Keen Eye', gender: 'M' }),
     mon('Snorlax', ['bodyslam', 'splash'], { gender: 'M' })]);

  console.log('\n=== CASE 5 — Magnet Pull vs a NON-steel foe (control): neither, switch ACCEPTED ===');
  await freeCase('5. Magnet Pull (Magneton) vs a non-Steel Snorlax',
    [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull' })],
    [mon('Snorlax', ['bodyslam', 'splash'], { gender: 'M' }),
     mon('Regice', ['icebeam', 'splash'])],
    { note: 'Snorlax is not Steel → Magnet Pull does not trap' });

  console.log('\n=== CASE 6 — Dugtrio MIRROR (both Arena Trap): mutual maybeTrapped → trapped ===');
  {
    // BOTH sides trapped: assert each side independently, driving each one's rejected switch.
    const p1team = [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap' }),
                    mon('Snorlax', ['bodyslam', 'splash'], { gender: 'M' })];
    const p2team = [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap' }),
                    mon('Regice', ['icebeam', 'splash'])];
    for (const s of ['p1', 'p2']) {
      const bat = await makeBattle(p1team, p2team);
      const pre = flagForm(latestActive0(bat.lines[s]));
      const attempt = await writeAndObserve(bat, s, 'switch 2');
      record(`6. Dugtrio MIRROR — ${s} side`, 'maybeTrapped→trapped',
        `${pre}→${flagForm(attempt.active0After)}`, attempt.error, !attempt.error, 'mutual trap');
      try { bat.stream.omniscient.destroy(); } catch (e) { /* teardown */ }
    }
  }

  console.log('\n=== CASE 7 — Magneton MIRROR (both Magnet Pull + Steel): mutual trapped ===');
  {
    const p1team = [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull' }),
                    mon('Forretress', ['spikes', 'splash'], { ability: 'Sturdy', gender: 'M' })];
    const p2team = [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull' }),
                    mon('Metagross', ['meteormash', 'splash'], { ability: 'Clear Body' })];
    for (const s of ['p1', 'p2']) {
      const bat = await makeBattle(p1team, p2team);
      const pre = flagForm(latestActive0(bat.lines[s]));
      const attempt = await writeAndObserve(bat, s, 'switch 2');
      record(`7. Magneton MIRROR — ${s} side`, 'maybeTrapped→trapped',
        `${pre}→${flagForm(attempt.active0After)}`, attempt.error, !attempt.error, 'mutual Steel trap');
      try { bat.stream.omniscient.destroy(); } catch (e) { /* teardown */ }
    }
  }

  console.log('\n=== CASE 8 — trapped mon is the LAST mon (no live bench): NEITHER flag ===');
  {
    const bat = await makeBattle(
      [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap' })],
      [mon('Snorlax', ['bodyslam', 'splash'], { gender: 'M' })]);   // no bench
    const a0 = latestActive0(bat.lines.p2);
    const flag = flagForm(a0);
    record('8. Arena Trap vs a LAST-mon foe (no live bench)', 'neither', flag, null, 'n/a (no bench to switch to)',
      'canSwitchIn/isLastActive gate omits BOTH flags even though the ability traps');
    // Capture the neither-flag active[0] for the report.
    main._lastMonActive0 = a0;
    try { bat.stream.omniscient.destroy(); } catch (e) { /* teardown */ }
  }

  console.log('\n=== CASE 9 — the trapper LEAVES: the flag LIFTS on the next request (non-sticky) ===');
  {
    // p1 Dugtrio traps p2 Snorlax; then p1 switches its Dugtrio OUT — p2's NEXT request
    // must carry NEITHER flag (the trap is recomputed per-request, not sticky).
    const bat = await makeBattle(
      [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap' }),
       mon('Regice', ['icebeam', 'splash'])],
      [mon('Snorlax', ['bodyslam', 'splash'], { gender: 'M' }),
       mon('Regice', ['icebeam', 'splash'])]);
    const trappedFlag = flagForm(latestActive0(bat.lines.p2));   // maybeTrapped while trapped
    // p1 switches its trapper out (free); p2 splashes.
    await writeAndObserve(bat, 'p1', 'switch 2');
    await writeAndObserve(bat, 'p2', 'move 2');
    const liftedFlag = flagForm(latestActive0(bat.lines.p2));    // now neither (trapper gone)
    record('9. trap LIFTS when the trapper switches out (per-request recompute)', 'maybeTrapped→neither',
      `${trappedFlag}→${liftedFlag}`, null, 'yes (a voluntary switch is now legal)',
      'the flag is recomputed each request, not a sticky patch');
    try { bat.stream.omniscient.destroy(); } catch (e) { /* teardown */ }
  }

  // ── The report table + the captured trapped:true / maybeTrapped:true / neither forms ──
  console.log('\n\n================ TRAPPING MATRIX SUMMARY ================');
  console.log('case                                              | expected           | observed             | switch-accepted | error');
  for (const r of rows) {
    const c = r.caseName.padEnd(48).slice(0, 48);
    const e = r.expectedFlag.padEnd(18);
    const o = r.observedFlag.padEnd(20);
    const sa = String(r.switchAccepted).padEnd(15);
    console.log(`${r.ok ? ' ' : '!'} ${c} | ${e} | ${o} | ${sa} | ${r.errorText}`);
  }

  const cap = trappedWithBenchCase._captured;
  if (cap) {
    console.log('\n---- the ACTUAL captured trapped:true active[0] (case: ' + cap.caseName + ') ----');
    console.log(JSON.stringify(cap.trapped_active0, null, 2));
    console.log('\n---- the maybeTrapped:true active[0] (BEFORE the rejected switch, same case) ----');
    console.log(JSON.stringify(cap.maybeTrapped_active0, null, 2));
    console.log('\n---- the |error| that triggered the trapped:true re-request ----');
    console.log(cap.error);
  }
  if (main._lastMonActive0) {
    console.log('\n---- the NEITHER-flag active[0] (last-mon, case 8 — both flags OMITTED) ----');
    console.log(JSON.stringify(main._lastMonActive0, null, 2));
  }

  console.log(`\n================ ${failures === 0 ? 'ALL CASES PASSED' : `${failures} CASE(S) FAILED`} ================`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
