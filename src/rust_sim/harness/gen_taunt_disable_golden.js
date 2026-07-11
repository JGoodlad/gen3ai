// gen_taunt_disable_golden.js — Gen-3 TAUNT + DISABLE differential harness.
//
// Extends harness/gen_status_move_golden.js (the per-decision STATE+STATUS+SEED+winner
// full-battle differential) to the move-SELECTION-restriction layer this step adds:
//   TAUNT   (taunt):   Dark, Status, accuracy 100 (DRAWS randomChance(100,100)); disables
//                      every Status move; duration a CONSTANT 2 (gen3 override `duration:2`,
//                      the base onStart `duration++` NEVER manifests — verified vs the sim,
//                      see `probe_taunt_duration_branch.js`); NO duration draw; protect:1 /
//                      bypasssub:1. Residual tick at onResidualOrder 10, subOrder 15.
//   DISABLE (disable): Normal, Status, accuracy 55 (DRAWS randomChance(55,100), CAN miss);
//                      disables the target's lastMove for a stored duration = (the disabler
//                      moved FIRST / willMove(target) TRUE ? random(2,6) : random(2,6)+1)
//                      turns — VERIFIED vs the sim (`probe_disable_full_lifecycle.js`); the
//                      residual DisableDuration handler then ticks it down + frees the move.
//                      onTryHit FAILS draw-free with no lastMove; protect:1 / bypasssub:1.
//                      Residual tick at order NO_ORDER, subOrder 2.
//
// THE PROOF (the CRUX): drive the OMNISCIENT in-process BattleStream (no server) over
// CONSTRUCTED scenarios that each ISOLATE one draw/branch, capturing the running PRNG seed
// BEFORE the first decision (initSeed) and AFTER each DECISION BOUNDARY, plus each active's
// species/hp/maxhp/fainted/status + the TAUNT presence + the DISABLED slot + pokemon_left +
// first mover + winner. The Rust test seeds a BattleState at the init seed and runs
// run_full_battle WITHOUT re-seeding — so the post-decision seed must match the sim's at
// EVERY boundary, INCLUDING the taunt accuracy(100) draw and the disable accuracy(55) +
// random(2,6). An EXACT cross-decision seed match to game-end + the per-decision taunt/
// disable STATE + the winner is the draw-ORDER+COUNT proof.
//
// The KEY per-decision STATE the disable duration is proven by: `disabled` (WHICH slot is
// disabled, or -1). If the port stored the WRONG duration, the disabled slot would clear one
// boundary too early/late vs the sim → a STATE divergence at the free-up boundary. Scenario
// (3)/(3b) exercise BOTH the faster-disabler (rolled) and slower-disabler (rolled+1) branches
// AND their exact free-up turns, so a +1 (or -1) off-by-one on either branch FAILS this gate.
//
// The taunt/disable SELECTION restriction is proven by SCRIPTING the target into its
// remaining-usable moves (or a forced Struggle) — a choice the sim only accepts if the port's
// legality matches; a Rust replay that diverges on the disabled/taunted slot desyncs the seed.
//
// FAIL-LOUD: each scenario declares the BRANCH it must realize (taunt applied, disable
// applied + move disabled, disable MISS, disable into no-lastMove FAIL, forced Struggle, the
// free-up = disableEnd/tauntEnd).
//
// Output: tests/vectors/taunt_disable_golden.txt (same TAB format as status_move_golden).
//
// Run:  node src/rust_sim/harness/gen_taunt_disable_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/taunt_disable_golden.txt');
// gen3customgame (NO clauses -> no SetStatus handler-sort shuffle). Taunt/Disable are not
// status-INFLICTING moves so the Sleep/Freeze clauses never gate them; the format only
// affects the SetStatus shuffle (which taunt/disable never reach). The Rust test uses
// gen3customgame -> `sleep_clause` OFF.
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
  let x = 0x2f9a7c11 >>> 0;
  const step = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  for (let i = 0; i < n; i++) out.push([step() || 1, step() || 1, step() || 1, step() || 1]);
  return out;
}

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

// The DISABLED move slot for an active mon (the index into its moveSlots of the disabled
// move id), or -1 if not disabled. Mirrors the Rust `disabled_slot`.
function disabledSlotOf(a) {
  if (!a || !a.volatiles || !a.volatiles['disable']) return -1;
  const disId = a.volatiles['disable'].move;
  if (!disId) return -1;
  for (let k = 0; k < a.moveSlots.length; k++) {
    if (a.moveSlots[k].id === disId) return k;
  }
  return -1;
}

function snap(side) {
  const a = side.active[0];
  if (!a) return { species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', left: side.pokemonLeft, taunted: 0, disabled: -1 };
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status: a.status || '-', left: side.pokemonLeft,
    taunted: a.volatiles && a.volatiles['taunt'] ? 1 : 0,
    disabled: disabledSlotOf(a),
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

// Scan the protocol log between two decision points for the taunt/disable branch OUTCOMES
// (so the differential AND the per-scenario branch floor can assert WHICH branch fired).
function outcomesSince(log, fromIdx) {
  const out = {
    tauntStart: false, disableStart: false, miss: false, fail: false,
    struggle: false, tauntEnd: false, disableEnd: false, cantDisabled: false,
  };
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    if (tag === '-start') {
      if ((p[3] || '').includes('Taunt')) out.tauntStart = true;
      if ((p[3] || '') === 'Disable') out.disableStart = true;
    }
    if (tag === '-end') {
      if ((p[3] || '').includes('Taunt')) out.tauntEnd = true;
      if ((p[3] || '') === 'Disable') out.disableEnd = true;
    }
    if (tag === '-miss') out.miss = true;
    if (tag === '-fail') out.fail = true;
    if (tag === 'move' && (p[3] || '') === 'Struggle') out.struggle = true;
    if (tag === 'cant' && ((p[3] || '') === 'Disable' || (p[3] || '') === 'move: Taunt')) out.cantDisabled = true;
  }
  return out;
}

// Choose the target's remaining-usable move from its ACTIVE request (respecting the
// taunt/disable `disabled` flags the sim reports) — so the scripted choice is ALWAYS legal.
// Returns the 1-based `move K` for the first non-disabled move with PP, or `move 1` (the sim
// substitutes Struggle if all are disabled/0-PP).
function firstUsableMove(side, battle) {
  const req = battle.sides[side].activeRequest;
  if (req && req.active && req.active[0] && req.active[0].moves) {
    const moves = req.active[0].moves;
    for (let k = 0; k < moves.length; k++) {
      if (!moves[k].disabled && (moves[k].pp === undefined || moves[k].pp > 0)) return `move ${k + 1}`;
    }
    // all disabled / out of PP -> Struggle (the sim accepts `move 1`, substituting struggle).
    return 'move 1';
  }
  return 'move 1';
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
    // STALL GUARD: a REJECTED choice (no seed/log/request advance) would record a stuck
    // duplicate forever — fail loud rather than emit a poisoned golden.
    if (seedAfter === seedBefore && log.length === logLenBefore && battle.requestState === reqState) {
      throw new Error(`STALL: choice ${JSON.stringify(choices)} did not advance the battle ` +
        `(scenario ${sc.id}, decision ${decisionNo}) — the sim rejected it. Fix the script.`);
    }
    const outcomes = outcomesSince(log, logLenBefore);
    for (const k of Object.keys(outcomes)) if (outcomes[k]) rec.branchSeen[k] = true;
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      outcomes,
    });
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
  // A plan-driven p1 script; p2 always picks a forced switch, else its first remaining-usable
  // move (respecting taunt/disable/PP) so its choice is always legal even after the restriction.
  const planP1 = (p1plan) => () => {
    let i = 0;
    return (decisionNo, battle, reqState, force) => {
      if (reqState === 'switch') {
        const c = { p1: null, p2: null };
        if (force[0]) c.p1 = firstLiveBench(0, battle);
        if (force[1]) c.p2 = firstLiveBench(1, battle);
        return c;
      }
      const p1c = p1plan[Math.min(i, p1plan.length - 1)];
      i++;
      return { p1: p1c, p2: firstUsableMove(1, battle) };
    };
  };

  // --- (1) TAUNT LANDS + a Status move is BLOCKED, then FREES UP. A FAST, strong Alakazam
  //   (Psychic) Taunts a moderately-bulky Gengar (Shadow Ball + Will-O-Wisp): once Taunted,
  //   Will-O-Wisp is un-selectable -> Gengar must Shadow Ball. The FIXED-2 taunt ticks down +
  //   frees up (Will-O-Wisp usable again), captured across boundaries. Alakazam's Psychic
  //   (super-effective vs Gengar) KOs it decisively. REQUIRES: tauntStart + tauntEnd. ---
  S.push({
    id: 'taunt_lands_blocks_status_and_frees',
    // A FAST, strong PHYSICAL Aerodactyl (Earthquake) Taunts a Blissey (Ice Beam + Thunder
    // Wave). The taunt blocks Thunder Wave (a NON-defensive status, so no wall lock), ticks its
    // FIXED 2, and FREES UP (Thunder Wave selectable again) — captured across boundaries —
    // while Earthquake (physical, so Blissey's low physical Def bleeds it) KOs Blissey after the
    // free-up. Blissey Ice Beams. REQUIRES: tauntStart + tauntEnd. ---
    p1: [mon('Aerodactyl', ['taunt', 'earthquake'], { nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Blissey', ['icebeam', 'thunderwave'], { nature: 'Bold', evs: { hp: 252, def: 252 } })],
    makeScript: planP1(['move 1', 'move 2']),
    require: ['tauntStart', 'tauntEnd'],
  });

  // --- (2) TAUNT forces STRUGGLE. A fast Alakazam (Psychic) Taunts a Milotic whose ONLY moves
  //   are Status (Toxic + Recover) -> both un-selectable -> Milotic Struggles. Psychic grinds
  //   Milotic (and its Struggle recoil helps) to a KO. REQUIRES: tauntStart + struggle. ---
  S.push({
    id: 'taunt_forces_struggle',
    p1: [mon('Alakazam', ['taunt', 'psychic'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Milotic', ['toxic', 'recover'], { nature: 'Bold', evs: { hp: 252, def: 252 } })],
    makeScript: planP1(['move 1', 'move 2']),
    require: ['tauntStart', 'struggle'],
  });

  // --- (2b) TAUNTER SECOND on turn >= 2 (MINOR A — the base taunt onStart's `activeTurns &&
  //   !willMove` duration++ branch). A SLOWER Snorlax (base 30) Taunts a FASTER Alakazam (base
  //   120) on turn 2, AFTER Alakazam already moved (target.activeTurns>=1 AND willMove(target)
  //   FALSE). PROVES gen3 taunt is a CONSTANT duration 2 even here (the free-up matches the
  //   taunter-first case — verified vs the sim, `probe_taunt_duration_branch.js`). Snorlax's
  //   Body Slam grinds Alakazam; Alakazam's Calm Mind is Taunt-blocked (must Shadow Ball).
  //   REQUIRES: tauntStart (the slow-taunter branch) + tauntEnd (the free-up). ---
  S.push({
    id: 'taunt_second_turn2_minor_a',
    // Taunter = a SLOW, physical Snorlax (Taunt + Body Slam). Target = a bulky Suicune (Surf +
    // Calm Mind) that SURVIVES long enough to see the free-up. On turn 0 both attack (Suicune
    // moves first -> activeTurns), turn 1 Snorlax Taunts (slower -> willMove(Suicune) FALSE,
    // the activeTurns++ path). Body Slam (physical) bleeds Suicune UNWALLED by its Calm Mind
    // (an SpD boost), and the taunt frees up (Calm Mind selectable again) at the SAME turn a
    // taunter-first would — the CONSTANT-2 proof. REQUIRES: tauntStart + tauntEnd. ---
    p1: [mon('Snorlax', ['taunt', 'bodyslam'], { nature: 'Brave', ivs: { ...IV31, spe: 0 }, evs: { hp: 252, atk: 252 } })],
    p2: [mon('Suicune', ['surf', 'calmmind'], { nature: 'Bold', evs: { hp: 252, def: 252 } })],
    makeScript: planP1(['move 2', 'move 1', 'move 2']),
    require: ['tauntStart', 'tauntEnd'],
  });

  // --- (3) DISABLE by a FASTER disabler + the random(2,6) duration + the exact FREE-UP turn.
  //   A FAST Aerodactyl (base 130) Disables a slow Snorlax's lastMove (Body Slam) — willMove
  //   TRUE (Snorlax still to move) -> duration stored = rolled. The acc-55 disable MISSES on
  //   some seeds (acc-only, no random(2,6)); LANDS on others (draws random(2,6), Body Slam
  //   disabled -> Snorlax must Rest, then Body Slam frees up). Aerodactyl Rock Slides; Snorlax
  //   has bulk so the free-up is observed before a KO. REQUIRES: disableStart + miss +
  //   disableEnd (the free-up). ---
  S.push({
    id: 'disable_faster_disabler_free_up',
    p1: [mon('Aerodactyl', ['disable', 'rockslide'], { nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'rest'], { nature: 'Brave', ivs: { ...IV31, spe: 0 }, evs: { hp: 252, def: 252 } })],
    makeScript: planP1(['move 2', 'move 1', 'move 2']),
    require: ['disableStart', 'miss', 'disableEnd'],
  });

  // --- (3b) DISABLE by a SLOWER disabler (the rolled+1 duration branch) + the exact FREE-UP.
  //   A slow Blissey (base 55) Disables a FASTER target's lastMove — willMove FALSE (the target
  //   already moved) -> duration stored = rolled+1 (ONE turn longer than the faster-disabler
  //   case). Uses a Jolteon (base 130, faster than Blissey) whose lastMove (Thunderbolt) gets
  //   disabled -> it must Shadow Ball; the disable frees up. Blissey Ice Beams to grind Jolteon;
  //   Blissey is bulky so the free-up is observed. REQUIRES: disableStart + disableEnd (the
  //   +1-longer free-up). ---
  S.push({
    id: 'disable_slower_disabler_free_up',
    p1: [mon('Blissey', ['disable', 'icebeam'], { item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    p2: [mon('Jolteon', ['thunderbolt', 'shadowball'], { nature: 'Timid', evs: { hp: 252, spa: 252, spe: 252 } })],
    makeScript: planP1(['move 2', 'move 1', 'move 2']),
    require: ['disableStart', 'disableEnd'],
  });

  // --- (4) DISABLE into a mon that has NOT moved (fail, draw-free). A FAST Aerodactyl Disables
  //   a slow Snorlax on TURN 1 — Snorlax has no lastMove yet -> onTryHit FAILS (accuracy drawn,
  //   NO random(2,6)). Aerodactyl Rock Slides to a win. REQUIRES: fail. ---
  S.push({
    id: 'disable_into_no_last_move_fails',
    p1: [mon('Aerodactyl', ['disable', 'rockslide'], { nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'rest'], { nature: 'Brave', ivs: { ...IV31, spe: 0 }, evs: { hp: 252, atk: 252 } })],
    makeScript: planP1(['move 1', 'move 2']),
    require: ['fail'],
  });

  // --- (5) TAUNT + DISABLE stacked -> forced STRUGGLE. A fast Alakazam Taunts (a Snorlax's Rest
  //   gone), then Disables Body Slam (its lastMove) -> BOTH of Snorlax's moves un-usable ->
  //   Struggle. Alakazam has Taunt + Disable + Psychic. Snorlax (slow) Body Slams turn 0 (sets
  //   lastMove) and is Taunted; turn 1 Alakazam Disables Body Slam -> Struggle. Psychic +
  //   Struggle recoil KO Snorlax. REQUIRES: tauntStart + disableStart + struggle. ---
  S.push({
    id: 'taunt_plus_disable_forces_struggle',
    // A fast GENGAR (Ghost/Poison) Taunts a Blissey (Seismic Toss + Soft-Boiled): Taunt kills
    // Soft-Boiled, then Disable locks Seismic Toss (its lastMove) -> BOTH un-usable -> Blissey
    // STRUGGLES. Blissey's Seismic Toss is FIGHTING -> IMMUNE into Gengar (Ghost) so Gengar takes
    // ZERO from it; once Struggling only Struggle's tiny typeless chip (Blissey base Atk 10)
    // touches Gengar -> Gengar SURVIVES to grind Blissey down with Sludge Bomb (Poison, neutral
    // vs Normal), so the struggle lock is captured across MANY boundaries. Some seeds MISS the
    // acc-55 Disable; across 80 seeds it lands often -> struggle realizes. REQUIRES: tauntStart
    // + disableStart + struggle. ---
    p1: [mon('Gengar', ['taunt', 'disable', 'sludgebomb'], { item: 'Leftovers', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Blissey', ['seismictoss', 'softboiled'], { nature: 'Bold', evs: { hp: 252, def: 252 } })],
    // ADAPTIVE p1: keep the target Taunted (re-Taunt when the volatile lapses) AND retry the
    // acc-55 Disable on the target's lastMove until it lands — so BOTH restrictions overlap and
    // Struggle is FORCED regardless of the seed's Disable-miss luck; else Sludge Bomb.
    makeScript: (() => () => {
      return (decisionNo, battle, reqState, force) => {
        if (reqState === 'switch') {
          const c = { p1: null, p2: null };
          if (force[0]) c.p1 = firstLiveBench(0, battle);
          if (force[1]) c.p2 = firstLiveBench(1, battle);
          return c;
        }
        const foe = battle.sides[1].active[0];
        const taunted = foe && foe.volatiles && foe.volatiles['taunt'];
        const disabled = disabledSlotOf(foe) >= 0;
        const hasLastMove = foe && foe.lastMove && foe.lastMove.id && foe.lastMove.id !== 'struggle';
        let p1c;
        if (!taunted) p1c = 'move 1';            // (re-)Taunt to keep the status move locked
        else if (!disabled && hasLastMove) p1c = 'move 2'; // retry Disable on the lastMove
        else p1c = 'move 3';                     // both locked (or nothing to disable) -> Sludge Bomb
        return { p1: p1c, p2: firstUsableMove(1, battle) };
      };
    })(),
    require: ['tauntStart', 'disableStart', 'struggle'],
  });

  // --- (6) DISABLE clears on switch-out. Blissey Disables Snorlax's Body Slam; Snorlax switches
  //   to a bench Aerodactyl — the disable is gone. Blissey Ice Beams to grind. REQUIRES:
  //   disableStart + a switch. ---
  S.push({
    id: 'disable_clears_on_switch',
    p1: [mon('Blissey', ['disable', 'icebeam'], { item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'rest'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Aerodactyl', ['rockslide', 'earthquake'], { nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    makeScript: (() => () => {
      let i = 0;
      return (decisionNo, battle, reqState, force) => {
        if (reqState === 'switch') {
          const c = { p1: null, p2: null };
          if (force[0]) c.p1 = firstLiveBench(0, battle);
          if (force[1]) c.p2 = firstLiveBench(1, battle);
          return c;
        }
        // turn0: both attack (Snorlax Body Slams -> lastMove). turn1: p1 Disable, p2 attack.
        // turn2: p2 switch to Aerodactyl (disable clears). then attack.
        let p1c, p2c;
        if (i === 0) { p1c = 'move 2'; p2c = firstUsableMove(1, battle); }
        else if (i === 1) { p1c = 'move 1'; p2c = firstUsableMove(1, battle); }
        else if (i === 2) { p1c = 'move 2'; p2c = 'switch 2'; }
        else { p1c = 'move 2'; p2c = firstUsableMove(1, battle); }
        i++;
        return { p1: p1c, p2: p2c };
      };
    })(),
    require: ['disableStart'],
  });

  // --- (7) TAUNT/DISABLE INTO A REAL BATTLE (a longer mixed game to game-end). A fast Alakazam
  //   opens Taunt on a Suicune (blocks its Calm Mind), pivots to Aerodactyl, and grinds it out —
  //   so the restriction interleaves with a switch + the full move/residual/faint machinery to a
  //   win. REQUIRES: tauntStart + a win run. ---
  S.push({
    id: 'taunt_into_real_battle',
    p1: [mon('Alakazam', ['taunt', 'psychic'], { nature: 'Timid', evs: { spa: 252, spe: 252 } }),
         mon('Aerodactyl', ['rockslide', 'earthquake'], { nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Suicune', ['surf', 'calmmind'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
         mon('Jolteon', ['thunderbolt', 'shadowball'], { nature: 'Timid', evs: { hp: 4, spa: 252, spe: 252 } })],
    makeScript: (() => () => {
      let pivoted = false;
      let tauntedOnce = false;
      return (decisionNo, battle, reqState, force) => {
        if (reqState === 'switch') {
          const c = { p1: null, p2: null };
          if (force[0]) { c.p1 = firstLiveBench(0, battle); pivoted = true; }
          if (force[1]) c.p2 = firstLiveBench(1, battle);
          return c;
        }
        // Taunt Suicune once (while Alakazam is out), then pivot to Aerodactyl (only if it can),
        // then Rock Slide. Robust to Alakazam fainting early (a forced switch already pivoted).
        const p1active = battle.sides[0].active[0];
        const alakazamOut = p1active && p1active.species && /alakazam/i.test(p1active.species.name);
        let p1c;
        if (alakazamOut && !tauntedOnce) { p1c = 'move 1'; tauntedOnce = true; }
        else if (alakazamOut && !pivoted) { p1c = 'switch 2'; pivoted = true; }
        else p1c = 'move 1'; // Aerodactyl Rock Slide (or Alakazam Psychic if pivot impossible)
        return { p1: p1c, p2: firstUsableMove(1, battle) };
      };
    })(),
    require: ['tauntStart'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# taunt_disable_golden.txt — Gen-3 TAUNT + DISABLE full-battle golden.');
  lines.push('# Per-decision-boundary STATE(+taunt+disabled-slot)+SEED differential to GAME-END.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status left taunted disabled) p2(...) first \\');
  lines.push('#        (tauntStart disableStart miss fail struggle)');
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

      for (const k of Object.keys(rec.branchSeen)) { scenSeen[sc.id][k] = true; corpus[k] = (corpus[k] || 0) + 1; }

      rec.decisions.forEach((d) => {
        const sp = (s) => [
          s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status, s.left, s.taunted, s.disabled,
        ].join('\t');
        const oc = (o) => [
          o.tauntStart ? 1 : 0, o.disableStart ? 1 : 0, o.miss ? 1 : 0, o.fail ? 1 : 0, o.struggle ? 1 : 0,
        ].join('\t');
        lines.push([
          'DEC', sc.id, seedStr, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
          sp(d.p1), sp(d.p2), d.firstMover, oc(d.outcomes),
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
      if (scenSeen[sc.id][bad]) failures.push(`${sc.id}: FORBIDDEN branch ${bad} realized (isolation broken)`);
    }
  }

  if (failures.length) {
    console.error('TAUNT/DISABLE GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`TAUNT/DISABLE GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('taunt-applied runs', 'tauntStart', 5);
  need('taunt-free-up runs', 'tauntEnd', 5);
  need('disable-applied runs', 'disableStart', 5);
  need('disable-miss runs', 'miss', 3);
  need('disable-free-up runs', 'disableEnd', 5);
  need('fail runs', 'fail', 5);
  need('forced-Struggle runs', 'struggle', 5);
  if (winRows < 50) { console.error(`TAUNT/DISABLE GOLDEN: too few WIN rows (${winRows} < 50)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `taunt/disable golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: taunt=${corpus.tauntStart || 0} tauntEnd=${corpus.tauntEnd || 0} disable=${corpus.disableStart || 0} ` +
    `disableEnd=${corpus.disableEnd || 0} miss=${corpus.miss || 0} fail=${corpus.fail || 0} struggle=${corpus.struggle || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
