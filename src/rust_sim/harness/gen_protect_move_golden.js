// gen_protect_move_golden.js — Gen-3 PROTECT / DETECT differential golden.
//
// Extends harness/gen_recovery_move_golden.js (the per-decision STATE+STATUS+BOOSTS+
// SEED+winner full-battle differential) to the NEW execution path this step adds:
// PROTECT and DETECT (identical full-turn protection). DEFERRED (fail-loud in the
// engine): Endure (survive-at-1-HP, a different onDamage mechanic), Quick Guard /
// Wide Guard / King's Shield / Spiky Shield (gen4+, none in gen3).
//
// THE DRAW MODEL (verified bit-for-bit vs the omniscient sim's PRNG probe,
// harness/probe_protect_rng.js + the resolved gen3 `stall` condition):
//
//   THE STALL / CONSECUTIVE-SUCCESS DRAW (the user's own Protect/Detect):
//     * Protect/Detect are NEVER-MISS (`accuracy: true`) → NO accuracy draw, and
//       PRIORITY 3 (gen3) so they resolve BEFORE the foe's attack (the `protect`
//       volatile is up when the foe's move runs).
//     * `onPrepareHit` → `runEvent('StallMove')`: the `stall` volatile's `onStallMove`
//       draws `randomChance(1, counter)` — but ONLY when the `stall` volatile is
//       ALREADY present. On the FIRST protect (no volatile yet) `runEvent('StallMove')`
//       has NO handler → returns true with NO DRAW → SHORT-CIRCUIT success.
//     * ON SUCCESS the `stall` volatile is (re)added (`onHit` → `addVolatile('stall')`):
//       from 0 it `onStart`s to counter 2, otherwise `onRestart`s `counter *= 2`, capped
//       at the gen3 `counterMax` 8 (the gen4 override gen3 inherits). So consecutive
//       successes give the floored denominator sequence 0→2→4→8→8→… = success
//       100%/50%/25%/12.5%/12.5% (the gen3 floor 1/8). DRAW-FREE apply.
//     * ON FAILURE (the stall roll failed) the `stall` volatile is DELETED → counter 0,
//       NO protection that turn. Draws nothing further.
//     * The `stall` volatile has `duration: 2` (reset to 2 by `onRestart`): it RESETS
//       (counter → 0) the first turn the user does NOT successfully protect (a different
//       move, a failed protect, or a switch-out clearVolatile).
//
//   THE MOVE-BLOCK DRAW (a foe move targeting the protected mon):
//     * In gen3 `tryMoveHit` the accuracy roll (`randomChance(accuracy,100)`) is drawn
//       FIRST; only if it PASSES does the protect `onTryHit` fire (at `runEvent('TryHit')`,
//       AFTER accuracy). So the BLOCKED foe move DRAWS its accuracy roll, then (if it
//       passes) is blocked (`-activate Protect`) — drawing NO crit / damage / secondary /
//       status. A move that MISSES its accuracy never reaches the block (`-miss`). The
//       block precedes the immunity report (EQ into a Flying/Levitate protector shows
//       `-activate Protect`, NOT `-immune`). Protect only blocks moves TARGETING the
//       protected mon (a self-target move is never blocked). DRAW-FREE block (only the
//       accuracy roll already happened).
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios that each ISOLATE a branch, capturing the running PRNG seed BEFORE the
// first decision (`initSeed`) and AFTER each DECISION BOUNDARY, plus each active's
// species/hp/maxhp/fainted/status(+inner counter) + boosts + confusion + pokemon_left +
// THE STALL COUNTER (`volatiles.stall.counter`) + first mover + winner. The Rust test
// seeds a BattleState at the init seed and runs `run_full_battle` WITHOUT re-seeding —
// so the post-decision seed must match at EVERY boundary, AND the per-decision HP (a
// block means the foe's damage never lands → HP unchanged) + the stall counter must
// match. An EXACT cross-decision seed+state match to game-end is the draw-ORDER+COUNT +
// block proof. (A wrong stall draw or block model desyncs the SEED; a wrong block desyncs
// the HP STATE.)
//
// FAIL-LOUD: each scenario declares the BRANCH it must realize (a block, a stall
// SUCCESS on a consecutive protect, a stall FAILURE on a consecutive protect, a counter
// reset, a protect-vs-status block); generation aborts if the sim run did NOT realize it
// — and CRUCIALLY a CONSECUTIVE-PROTECT FAILURE must realize (so the denominator is
// proven both ways). The output shares the recovery/setup TAB format + a 2-col tail
// (`stall.counter` for p1/p2) so the Rust gate reuses the parser with a small extension.
//
// Output: tests/vectors/protect_move_golden.txt
//
// Run:  node src/rust_sim/harness/gen_protect_move_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/protect_move_golden.txt');
// gen3customgame (NOT gen3ou) — matches the e2e capstone + recovery/setup format. No
// SetStatus clause shuffle; the Rust test passes `format_id: "gen3customgame"`.
const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };

function mon(species, moves, opts = {}) {
  return {
    species,
    item: opts.item || '',
    ability: opts.ability || 'No Ability',
    moves,
    evs: { ...EV0, ...(opts.evs || {}) },
    ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious',
    level: opts.level || 100,
    gender: opts.gender || 'N',
  };
}

function tick() { return new Promise((r) => setTimeout(r, 0)); }

function encodeChoice(c) {
  if (!c) return '-';
  const m = c.match(/^move\s+(\d+)$/);
  if (m) return `m${Number(m[1]) - 1}`;
  const s = c.match(/^switch\s+(\d+)$/);
  if (s) return `s${Number(s[1]) - 1}`;
  throw new Error(`unencodable choice ${JSON.stringify(c)}`);
}

function buildSeeds(n) {
  const out = [];
  let x = 0x2f6e51a3 >>> 0;
  const step = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  for (let i = 0; i < n; i++) out.push([step() || 1, step() || 1, step() || 1, step() || 1]);
  return out;
}

// The first ACTION to RUN this turn (matching the Rust `first_mover`). A mon that
// full-paras / is still-asleep / frozen / flinched emits `|cant|` and runs its action
// FIRST without a `|move|` line — so count `|cant|` + the confusion `-activate` too.
function firstMoverSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    const parts = log[i].split('|');
    const tag = parts[1];
    const isAction =
      tag === 'move' || tag === 'switch' || tag === 'cant' ||
      (tag === '-activate' && (parts[3] || '') === 'confusion');
    if (isAction && parts.length >= 3) {
      const actor = parts[2].trim();
      if (actor.startsWith('p1a:')) return 'p1';
      if (actor.startsWith('p2a:')) return 'p2';
    }
  }
  return 'none';
}

function statusOf(active) {
  const st = (active && active.status) || '';
  let stage = 0;
  if (st === 'tox') stage = active.statusState ? (active.statusState.stage || 0) : 0;
  if (st === 'slp') stage = active.statusState ? (active.statusState.time || 0) : 0;
  return { status: st || '-', stage };
}

function boostsOf(a) {
  const b = a && a.boosts ? a.boosts : {};
  return [b.atk | 0, b.def | 0, b.spa | 0, b.spd | 0, b.spe | 0];
}

function confusionOf(a) {
  return a && a.volatiles && a.volatiles['confusion'] ? (a.volatiles['confusion'].time | 0) : 0;
}

// The gen-3 PROTECT/DETECT stall counter at the decision boundary (the engine's
// `protect_counter`). `volatiles.stall.counter` is the value AFTER this turn's protect
// resolved (and after the residual decremented the volatile's duration to 1 — still
// alive); 0 == no stall volatile (deleted by a failed roll, expired by a non-protect
// turn, or cleared on switch).
function stallOf(a) {
  return a && a.volatiles && a.volatiles['stall'] ? (a.volatiles['stall'].counter | 0) : 0;
}

function snap(side) {
  const a = side.active[0];
  if (!a) return { species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', stage: 0, left: side.pokemonLeft, boosts: [0, 0, 0, 0, 0], confusion: 0, stall: 0 };
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status, stage, left: side.pokemonLeft,
    boosts: boostsOf(a), confusion: confusionOf(a), stall: stallOf(a),
  };
}

function forceSwitchTable(battle) {
  const out = [false, false];
  if (battle.requestState !== 'switch') return out;
  for (let i = 0; i < 2; i++) {
    const req = battle.sides[i].activeRequest;
    if (req && req.forceSwitch && req.forceSwitch[0]) out[i] = true;
  }
  return out;
}

// Scan the protocol log between two decision points for the per-side onBeforeMove
// outcome flags (shared format) + the PROTECT branch flags:
//   block       — a `|-activate|... move: Protect` (a foe move BLOCKED by protect)
//   protectUp   — a `|-singleturn|... Protect` (a protect SUCCEEDED, the volatile went up)
//   stallFail   — a protect move that did NOT put the volatile up (a failed consecutive
//                 stall roll). Detected in main() via the per-decision stall-counter delta
//                 + the absence of a `-singleturn` — here we flag a `-fail` on a stalling
//                 move user instead (the sim emits no `-fail` for a failed stall — it just
//                 doesn't `-singleturn`; the reliable signal is the counter, computed in
//                 main()). We still scan `miss` for completeness.
function outcomesSince(log, fromIdx) {
  const out = {
    p1: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    p2: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    block: false, protectUp: false, miss: false,
  };
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    const who = (p[2] || '').startsWith('p1a:') ? 'p1' : (p[2] || '').startsWith('p2a:') ? 'p2' : null;
    if (tag === 'cant' && who) {
      if (p[3] === 'par') out[who].fullpara = true;
      if (p[3] === 'flinch') out[who].flinch = true;
    }
    if (tag === '-curestatus' && who) {
      if ((p[3] || '') === 'slp') out[who].wake = true;
      if ((p[3] || '') === 'frz') out[who].thaw = true;
    }
    if (tag === '-damage' && who && (p[4] || '').includes('confusion')) out[who].selfhit = true;
    // A protect BLOCK: `|-activate|p1a: X|move: Protect`.
    if (tag === '-activate' && (p[3] || '').includes('Protect')) out.block = true;
    // A protect SUCCESS: `|-singleturn|p1a: X|Protect`.
    if (tag === '-singleturn' && (p[3] || '').includes('Protect')) out.protectUp = true;
    if (tag === '-miss') out.miss = true;
  }
  return out;
}

async function runBattle(sc, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();

  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(sc.p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(sc.p2) })}`);
  for (let i = 0; i < 10; i++) await tick();

  const script = sc.makeScript();
  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen, branchSeen: {} };

  let prevStall = [0, 0];
  let decisionNo = 0;
  let safety = 0;
  while (!stream.battle.ended && safety < 400) {
    safety++;
    const battle = stream.battle;
    const reqState = battle.requestState;
    if (reqState !== 'move' && reqState !== 'switch') { await tick(); continue; }
    const force = forceSwitchTable(battle);
    const seedBefore = battle.prng.getSeed();
    if (decisionNo === 0) rec.initSeed = seedBefore;

    const choices = script(decisionNo, battle, reqState, force);
    if (!choices) break;

    const logLenBefore = log.length;
    if (choices.p1) { try { streams.omniscient.write(`>p1 ${choices.p1}`); } catch (e) {} }
    if (choices.p2) { try { streams.omniscient.write(`>p2 ${choices.p2}`); } catch (e) {} }
    for (let i = 0; i < 16; i++) await tick();

    const seedAfter = battle.prng.getSeed();
    // STALL GUARD: a rejected choice (no advance in seed/log/request) → fail loud.
    if (seedAfter === seedBefore && log.length === logLenBefore && battle.requestState === reqState) {
      throw new Error(`STALL: choice ${JSON.stringify(choices)} did not advance the battle ` +
        `(scenario ${sc.id}, decision ${decisionNo}) — the sim rejected it. Fix the script.`);
    }
    const outcomes = outcomesSince(log, logLenBefore);
    const first = reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none';

    const p1 = snap(battle.sides[0]);
    const p2 = snap(battle.sides[1]);

    // A consecutive-protect STALL FAILURE (the proof the denominator works BOTH ways):
    // the user had a stall counter > 0 entering this turn, chose protect, and the volatile
    // is now GONE (counter dropped to 0) WITHOUT a `-singleturn` this turn → the stall
    // roll failed. (A counter that simply expired from a non-protect turn also drops to 0,
    // but those turns have protectUp=false AND no protect chosen — we only flag stallFail
    // when this turn's chooser USED protect, detected via the recorded choice tokens in
    // main(). Here we record the raw counters + protectUp so main() can decide.)
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1, p2,
      firstMover: first,
      outcomes,
      prevStall: [...prevStall],
    });
    prevStall = [p1.stall, p2.stall];
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

// ── Scenarios ──────────────────────────────────────────────────────────────

function scenarios() {
  const S = [];

  const firstLiveBench = (side, battle) => {
    const s = battle.sides[side];
    for (let k = 0; k < s.pokemon.length; k++) {
      const p = s.pokemon[k];
      if (p !== s.active[0] && !p.fainted) return `switch ${k + 1}`;
    }
    return 'pass';
  };
  const fromPlan = (plan) => () => {
    let i = 0;
    return (decisionNo, battle, reqState, force) => {
      if (reqState === 'switch') {
        const c = { p1: null, p2: null };
        if (force[0]) c.p1 = firstLiveBench(0, battle);
        if (force[1]) c.p2 = firstLiveBench(1, battle);
        return c;
      }
      const entry = plan[Math.min(i, plan.length - 1)];
      i++;
      return entry;
    };
  };

  // --- (1) SINGLE PROTECT blocking a foe ATTACK (the block + the foe's accuracy draw):
  //   a bulky Snorlax Protects vs a frail Dugtrio's Earthquake; the EQ draws its accuracy
  //   then is BLOCKED (`-activate Protect`) — Snorlax takes NO damage (HP unchanged that
  //   turn). Snorlax then Body Slams the frail foe out. The per-decision HP (no chip on the
  //   protect turn) + the seed (accuracy drawn, no crit/damage) must match. REQUIRES a
  //   block. ---
  S.push({
    id: 'single_protect_blocks_attack',
    p1: [mon('Snorlax', ['protect', 'bodyslam'], { ability: 'Thick Fat', item: 'Leftovers', nature: 'Careful', evs: { hp: 252, atk: 252 } })],
    // Dugtrio: frail, fast, Earthquake (100 acc — its accuracy draw is exercised on the
    // block). Arena Trap is a provable no-op (no switching forced). It dies to Body Slam.
    p2: [mon('Dugtrio', ['earthquake'], { ability: 'Arena Trap', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Protect → EQ BLOCKED (acc drawn, no damage)
      { p1: 'move 2', p2: 'move 1' }, // Body Slam (take an EQ now) — chip the foe
      { p1: 'move 1', p2: 'move 1' }, // Protect again → BLOCK
      { p1: 'move 2', p2: 'move 1' }, // Body Slam
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['block'],
  });

  // --- (2) CONSECUTIVE PROTECTS (the stall denominator BOTH ways): a bulky Blissey Protects
  //   EVERY turn vs a frail Diglett's Mud-Slap (never-miss chip), so the stall counter climbs
  //   2→4→8 and the success roll is drawn on turns 2+. Across the 80-seed sweep BOTH a
  //   consecutive-protect SUCCESS and a consecutive-protect FAILURE realize (the floored
  //   1/2, 1/4, 1/8 denominators). The per-decision STALL COUNTER + seed prove the draw model.
  //   Diglett's Mud-Slap is never-miss (so a blocked one still draws its accuracy:true =
  //   no draw — wait, Mud-Slap is 100 acc not true; pick a 100-acc move so accuracy IS
  //   drawn on the block). Blissey can't be KO'd by the frail chip, so the chain is long.
  //   REQUIRES: block + protectUp + stallFail (a consecutive-protect FAILURE). ---
  S.push({
    id: 'consecutive_protect_stall',
    p1: [mon('Blissey', ['protect', 'icebeam'], { ability: 'Natural Cure', item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    // Diglett: frail, fast; Mud-Shot (Ground, 95 acc — its accuracy IS drawn on a block).
    // Sand Veil N/A (no sand). It can't dent a 252/252 Bold Blissey; it dies to Ice Beam.
    p2: [mon('Diglett', ['mudshot', 'scratch'], { ability: 'Sand Veil', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Protect (1st — no draw, counter→2) BLOCK
      { p1: 'move 1', p2: 'move 1' }, // Protect (2nd — rc(1,2)) success→4 or fail→reset
      { p1: 'move 1', p2: 'move 1' }, // Protect (rc(1,2) or rc(1,4))
      { p1: 'move 1', p2: 'move 1' }, // Protect (escalate / reset)
      { p1: 'move 1', p2: 'move 1' }, // Protect
      { p1: 'move 1', p2: 'move 1' }, // Protect
      { p1: 'move 1', p2: 'move 1' }, // Protect
      { p1: 'move 1', p2: 'move 1' }, // Protect
      { p1: 'move 2', p2: 'move 1' }, // Ice Beam — start the KO grind
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['block', 'protectUp', 'stallFail'],
  });

  // --- (3) PROTECT then a NON-protect move then PROTECT (the counter RESET): a bulky
  //   Skarmory Protects (counter 2), Protects again (counter→4 on success or reset on fail),
  //   then uses a NON-protect move (Steel Wing) — the stall volatile EXPIRES — then Protects
  //   again, which is a FRESH first-protect (NO draw, counter→2). The per-decision stall
  //   counter proves the reset (a turn with no protect → counter 0 → the next protect draws
  //   nothing). REQUIRES: block + protectUp + a counter reset (asserted via the stall column
  //   in the Rust gate). ---
  S.push({
    id: 'protect_reset_then_protect',
    p1: [mon('Skarmory', ['protect', 'steelwing'], { ability: 'Keen Eye', item: 'Leftovers', nature: 'Impish', evs: { hp: 252, def: 252 } })],
    // Smeargle: frail; Tackle (95 acc — accuracy drawn on a block). It can't dent a bulky
    // Skarmory; dies to Steel Wing. (Own Tempo is a modeled no-op here.)
    p2: [mon('Smeargle', ['tackle'], { ability: 'Own Tempo', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Protect (counter→2) BLOCK
      { p1: 'move 1', p2: 'move 1' }, // Protect (rc(1,2)) → counter 4 or reset
      { p1: 'move 2', p2: 'move 1' }, // Steel Wing (NON-protect → stall volatile EXPIRES)
      { p1: 'move 1', p2: 'move 1' }, // Protect AGAIN — FRESH (NO draw, counter→2) BLOCK
      { p1: 'move 1', p2: 'move 1' }, // Protect (rc(1,2))
      { p1: 'move 2', p2: 'move 1' }, // Steel Wing — grind the foe out
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['block', 'protectUp'],
  });

  // --- (4) PROTECT vs a STATUS move (a status-move block): a bulky Snorlax Protects vs a
  //   foe's Thunder Wave — the TWave draws its accuracy then is BLOCKED (`-activate Protect`),
  //   so Snorlax is NOT paralyzed (status unchanged). Snorlax then Body Slams the foe. The
  //   per-decision STATUS (no par on the protect turn) + seed prove the status-move block.
  //   REQUIRES: block. ---
  S.push({
    id: 'protect_blocks_status_move',
    p1: [mon('Snorlax', ['protect', 'bodyslam'], { ability: 'Thick Fat', item: 'Leftovers', nature: 'Careful', evs: { hp: 252, atk: 252 } })],
    // Electrode: frail, fast; Thunder Wave (gen3 100 acc — accuracy drawn on the block) +
    // Thunder Shock chip. Soundproof is a provable no-op (no sound moves used). It dies to
    // Body Slam (a couple hits). After the TWave is blocked it falls back to Thunder Shock.
    p2: [mon('Electrode', ['thunderwave', 'thundershock'], { ability: 'Soundproof', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Protect → Thunder Wave BLOCKED (no par)
      { p1: 'move 2', p2: 'move 2' }, // Body Slam ; foe Thunder Shock chip
      { p1: 'move 1', p2: 'move 1' }, // Protect → Thunder Wave BLOCKED again
      { p1: 'move 2', p2: 'move 2' }, // Body Slam
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' },
    ]),
    require: ['block'],
  });

  // --- (5) DETECT (the identical-protection variant): a bulky Umbreon DETECTS vs a frail
  //   foe's never-miss Swift, blocking it (Detect draws the SAME stall model as Protect),
  //   then Faint-Attacks the foe out. Proves Detect routes through the same `is_protect`
  //   path. REQUIRES: block + protectUp (Detect emits `-singleturn ... Protect` too). ---
  S.push({
    id: 'detect_blocks_attack',
    p1: [mon('Umbreon', ['detect', 'feintattack'], { ability: 'Synchronize', item: 'Leftovers', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    // Spinarak: frail Bug; Swift (60 BP, NEVER-MISS — so a BLOCKED Swift draws NO accuracy,
    // the never-miss block path). Insomnia N/A. It dies to Faint Attack (never-miss Dark).
    p2: [mon('Spinarak', ['swift'], { ability: 'Insomnia', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Detect → Swift BLOCKED (never-miss: NO accuracy draw)
      { p1: 'move 2', p2: 'move 1' }, // Faint Attack (take a Swift) — chip the foe
      { p1: 'move 1', p2: 'move 1' }, // Detect again → BLOCK
      { p1: 'move 2', p2: 'move 1' }, // Faint Attack
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['block', 'protectUp'],
  });

  // --- (6) PROTECT INTO A REAL BATTLE TO GAME-END (the union: protect interleaved with a
  //   switch + the full move/residual/faint machinery): p1's Snorlax Protects a couple times
  //   (blocking a chipper), VOLUNTARILY PIVOTS to a hard-hitting Salamence that sweeps the
  //   FRAIL 2-mon foe team out to a clean win. Protect + a switch (which CLEARS the stall
  //   counter) + residuals + faints all the way to a win. REQUIRES: block + a win. ---
  S.push({
    id: 'protect_into_real_battle',
    p1: [mon('Snorlax', ['protect', 'bodyslam'], { ability: 'Thick Fat', item: 'Leftovers', nature: 'Careful', evs: { hp: 252, atk: 252 } }),
         mon('Salamence', ['dragonclaw', 'earthquake'], { ability: 'Intimidate', item: 'Leftovers', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    // Two FRAIL chippers using ONLY Quick Attack (40 BP Normal, never-miss — a blocked one
    // draws no accuracy; can't KO a 524-HP Thick-Fat Snorlax early so the scripted pivot is
    // always reached). A max-Atk Adamant Salamence OHKOs the frail pair with Dragon Claw.
    p2: [mon('Misdreavus', ['quickattack'], { ability: 'Levitate', nature: 'Jolly', evs: { atk: 252, spe: 252 } }),
         mon('Houndour', ['quickattack'], { ability: 'Flash Fire', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Snorlax Protect → Quick Attack BLOCKED
      { p1: 'move 1', p2: 'move 1' }, // Protect again (rc(1,2)) BLOCK / fail
      { p1: 'switch 2', p2: 'move 1' }, // VOLUNTARY pivot to Salamence (CLEARS the stall counter)
      { p1: 'move 1', p2: 'move 1' }, // Salamence Dragon Claw — KO the frail Misdreavus
      { p1: 'move 1', p2: 'move 1' }, // Dragon Claw the second frail foe (Houndour) out
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
    ]),
    require: ['block'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# protect_move_golden.txt — Gen-3 PROTECT / DETECT full-battle golden.');
  lines.push('# Per-decision-boundary STATE(+status+counter)+BOOSTS+STALL-COUNTER+SEED+first-mover differential to GAME-END.');
  lines.push('# (Extends the recovery/setup TAB format with a 2-col stall-counter tail.)');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1(fullpara wake thaw selfhit flinch) p2(...) block  p1Stall p2Stall');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0;
  const corpus = {};
  const scenSeen = {};

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);
    scenSeen[sc.id] = {};

    let scenDecs = 0;
    for (const seed of seeds) {
      let rec;
      try { rec = await runBattle(sc, seed); } catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); continue; }
      if (rec.gen !== 3) { failures.push(`${sc.id}: expected gen 3, got ${rec.gen}`); break; }
      if (!rec.initSeed || rec.decisions.length === 0) { failures.push(`${sc.id} seed ${seed}: no decisions`); continue; }
      const seedStr = seed.join(',');
      lines.push(['INIT', sc.id, rec.initSeed, seedStr].join('\t'));

      // Per-decision branch detection (block / protectUp / stallFail). `stallFail` =
      // this side CHOSE protect, had a stall counter > 0 ENTERING the turn (a consecutive
      // use, so the stall roll DREW), and the protect did NOT go up — the gen3 stall does
      // NOT delete the volatile on a failed roll, so the counter PERSISTS (does NOT escalate
      // to prev*2 — that only happens on a SUCCESS via onHit/onRestart). So a FAIL is:
      // chose-protect + prev>0 + the counter did NOT escalate (cur <= prev, i.e. cur < prev*2
      // capped at 8).
      rec.decisions.forEach((d) => {
        if (d.outcomes.block) { scenSeen[sc.id].block = true; corpus.block = (corpus.block || 0) + 1; }
        if (d.outcomes.protectUp) { scenSeen[sc.id].protectUp = true; corpus.protectUp = (corpus.protectUp || 0) + 1; }
        if (d.request === 'move') {
          for (const [side, ch, cur, prev] of [
            ['p1', d.choiceP1, d.p1.stall, d.prevStall[0]],
            ['p2', d.choiceP2, d.p2.stall, d.prevStall[1]],
          ]) {
            const choseProtect = ch === 'm0'; // slot-0 is Protect/Detect in every scenario
            const escalated = Math.min(prev * 2, 8);
            // A SUCCESS escalates the counter to min(prev*2, 8); a FAIL leaves it <= prev
            // (the gen3 stall persists, no onRestart). So prev>0 + chose-protect + cur != the
            // escalated value → the stall roll FAILED. (At prev==8 the escalated value is
            // also 8, so this can't distinguish there; the consecutive scenario realizes a
            // fail at prev=2/4 well before reaching 8.)
            if (choseProtect && prev > 0 && cur !== escalated) {
              scenSeen[sc.id].stallFail = true; corpus.stallFail = (corpus.stallFail || 0) + 1;
            }
          }
        }
      });

      rec.decisions.forEach((d) => {
        const sp = (s) => [
          s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status, s.stage, s.left,
          s.boosts[0], s.boosts[1], s.boosts[2], s.boosts[3], s.boosts[4], s.confusion,
        ].join('\t');
        const oc = (o) => [o.fullpara ? 1 : 0, o.wake ? 1 : 0, o.thaw ? 1 : 0, o.selfhit ? 1 : 0, o.flinch ? 1 : 0].join('\t');
        lines.push([
          'DEC', sc.id, seedStr, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
          sp(d.p1), sp(d.p2), d.firstMover,
          oc(d.outcomes.p1), oc(d.outcomes.p2), d.outcomes.block ? 1 : 0,
          d.p1.stall, d.p2.stall,
        ].join('\t'));
        decRows++; scenDecs++;
      });

      let winTok = 'none';
      if (rec.ended) {
        if (rec.winner === 'P1') winTok = 'p1';
        else if (rec.winner === 'P2') winTok = 'p2';
        else if (rec.winner === '') winTok = 'tie';
      }
      if (winTok === 'p1' || winTok === 'p2') winRows++;
      if (winTok === 'tie') tieRows++;
      lines.push(['END', sc.id, seedStr, rec.ended ? 1 : 0, winTok].join('\t'));
    }
    if (scenDecs === 0) failures.push(`${sc.id}: produced NO decision rows`);

    for (const need of (sc.require || [])) {
      if (!scenSeen[sc.id][need]) failures.push(`${sc.id}: REQUIRED branch ${need} never realized across the seed sweep`);
    }
    for (const bad of (sc.forbid || [])) {
      if (scenSeen[sc.id][bad]) failures.push(`${sc.id}: FORBIDDEN branch ${bad} realized (the scenario isolation is broken)`);
    }
  }

  if (failures.length) {
    console.error('PROTECT-MOVE GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`PROTECT GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('block decisions', 'block', 50);
  need('protect-up (success) decisions', 'protectUp', 50);
  need('consecutive-protect FAILURE decisions', 'stallFail', 10);
  if (winRows < 50) { console.error(`PROTECT GOLDEN: too few WIN rows (${winRows} < 50)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `protect-move golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: block=${corpus.block || 0} protectUp=${corpus.protectUp || 0} stallFail=${corpus.stallFail || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
