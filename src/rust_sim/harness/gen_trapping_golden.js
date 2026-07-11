// gen_trapping_golden.js — Gen-3 TRAPPING (Arena Trap / Magnet Pull) differential harness.
//
// Extends harness/gen_taunt_disable_golden.js (the per-decision STATE+SEED+winner
// full-battle differential) to the SWITCH-legality layer this step adds:
//   ARENA TRAP  (arenatrap):  the foe active traps every GROUNDED opposing mon (a
//       Flying-type / Levitate holder escapes; a grounded GHOST is trapped — Showdown-gen3
//       has NO `trapped` type-immunity, probe-verified). `onFoeTrapPokemon` (base data) →
//       1 handler per TrapPokemon event → NO draw, even in the Dugtrio MIRROR (mutual trap,
//       zero extra PRNG — probe-verified byte-identical seeds vs a Sand Veil control).
//   MAGNET PULL (magnetpull): traps STEEL-type foes (groundedness irrelevant — Skarmory,
//       Steel/Flying, is trapped). gen3 overrides it to `onAnyTrapPokemon`
//       (data/mods/gen3/abilities.ts) so BOTH actives' Magnet Pulls register on EVERY
//       TrapPokemon/MaybeTrapPokemon event: the speed-TIED MAGNETON MIRROR draws ONE
//       Fisher-Yates shuffle per event per mon = **4 draws per endTurn** (probe: 11 vs the
//       Sturdy control's 7); an Arena-Trap-vs-Magnet-Pull cross at equal speed draws 2
//       (both events on the Magnet Pull holder only). The draws sit INSIDE the endTurn
//       per-mon loop (DisableMove → TrapPokemon → MaybeTrapPokemon per mon), BEFORE the
//       gen3 quickClawRoll (battle.ts:1795).
//   TRAPPED = the sim's `pokemon.trapped` truthiness (both abilities `tryTrap(true)` →
//       'hidden'); it REJECTS a voluntary `switch N` at a `move` request DRAW-FREE. A
//       PHAZE (Roar) still drags a trapped mon; a fainted mon's forced replacement is
//       always accepted; the trapping mon itself switches freely.
//
// THE PROOF (the CRUX): drive the OMNISCIENT in-process BattleStream (no server) over
// CONSTRUCTED scenarios, capturing the running PRNG seed BEFORE the first decision
// (initSeed) and AFTER each DECISION BOUNDARY, plus each active's species/hp/maxhp/
// fainted/status + pokemon_left + the per-side TRAPPED flag (recorded only when the
// battle sits at the NEXT `move` request — where the sim's endTurn just recomputed it and
// the port's live `is_trapped` is definitionally the same instant; '-' at a mid-turn
// forced-switch pause / game end) + first mover + winner. The Rust test seeds a
// BattleState at the init seed and replays run_full_battle WITHOUT re-seeding — the
// per-boundary seed match catches a missing/extra trap-event shuffle draw (the Magneton
// mirror's 4/endTurn) and the trapped columns catch a wrong trapped computation. The
// scripts NEVER submit an illegal (trapped) switch — the switch-REJECTION path is pinned
// by tests/regression_test.rs (T1-T4) instead, mirroring how the PP layer pinned its
// reject-and-re-request gate.
//
// FAIL-LOUD: each scenario declares REQUIRED branches (trapped rows realized, a drag,
// wins); a rejected choice (no advance) throws STALL rather than emit a poisoned golden.
//
// Output: tests/vectors/trapping_golden.txt (TAB format, sibling of taunt_disable).
//
// Run:  node src/rust_sim/harness/gen_trapping_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/trapping_golden.txt');
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
  let x = 0x51e3a9b7 >>> 0;
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

function snap(side) {
  const a = side.active[0];
  if (!a) return { species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', left: side.pokemonLeft };
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status: a.status || '-', left: side.pokemonLeft,
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

// The AUTHORITATIVE trapped fact (the choice-legality gate): the sim's internal
// `pokemon.trapped` truthiness ('hidden' for both abilities — the request JSON only
// shows `maybeTrapped` until a rejected attempt patches it, and omits both flags when
// no bench is live; neither display nuance affects legality or draws).
function trappedOf(battle, side) {
  const a = battle.sides[side].active[0];
  return !!(a && a.trapped);
}

function outcomesSince(log, fromIdx) {
  const out = { drag: false, struggle: false };
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    if (p[1] === 'drag') out.drag = true;
    if (p[1] === 'move' && (p[3] || '') === 'Struggle') out.struggle = true;
  }
  return out;
}

function firstUsableMove(side, battle) {
  const req = battle.sides[side].activeRequest;
  if (req && req.active && req.active[0] && req.active[0].moves) {
    const moves = req.active[0].moves;
    for (let k = 0; k < moves.length; k++) {
      if (!moves[k].disabled && (moves[k].pp === undefined || moves[k].pp > 0)) return `move ${k + 1}`;
    }
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
    // STALL GUARD: a REJECTED choice (e.g. a trapped switch) would record a stuck duplicate
    // forever — fail loud rather than emit a poisoned golden. Scripts must respect `trapped`.
    if (seedAfter === seedBefore && log.length === logLenBefore && battle.requestState === reqState) {
      throw new Error(`STALL: choice ${JSON.stringify(choices)} did not advance the battle ` +
        `(scenario ${sc.id}, decision ${decisionNo}) — the sim rejected it. Fix the script.`);
    }
    const outcomes = outcomesSince(log, logLenBefore);
    for (const k of Object.keys(outcomes)) if (outcomes[k]) rec.branchSeen[k] = true;

    // Per-side TRAPPED at THIS boundary: meaningful only when the battle now sits at the
    // NEXT `move` request (the endTurn that closed this decision just recomputed it — the
    // exact instant the Rust port's live `is_trapped` reproduces). '-' at a mid-turn
    // forced-switch pause (the sim's flag is stale there) and at game end.
    const trapTok = battle.requestState === 'move'
      ? [trappedOf(battle, 0) ? 1 : 0, trappedOf(battle, 1) ? 1 : 0]
      : ['-', '-'];
    if (trapTok[0] === 1 || trapTok[1] === 1) rec.branchSeen.trapped = true;
    if (trapTok[0] === 1 && trapTok[1] === 1) rec.branchSeen.mutualTrap = true;

    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      trapped: trapTok,
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
  const liveBenchSlot = (side, battle) => {
    const s = battle.sides[side];
    for (let k = 0; k < s.pokemon.length; k++) {
      const p = s.pokemon[k];
      if (p !== s.active[0] && !p.fainted) return k + 1;
    }
    return 0;
  };
  // A plan-driven p1 script; p2 follows `p2mode`:
  //   'mirror'         — p2 submits the SAME plan entry as p1 (the mirror scenarios);
  //   'splash-or-switch' — SPLASH (`move 2` — every p2 mon's slot 2 is splash) while
  //       TRAPPED (the "must fight" fact is the submitted move; a trapped mon that
  //       one-shots the trapper would just end the battle before any trapped boundary),
  //       and SWITCH to the live bench whenever FREE (so a wrongly-free port desyncs on
  //       the recorded choice);
  //   'splash'         — always splash (the trapper does the damage).
  const planP1 = (p1plan, p2mode) => () => {
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
      let p2c;
      if (p2mode === 'mirror') {
        p2c = p1c;
      } else if (p2mode === 'splash-or-switch') {
        p2c = !trappedOf(battle, 1) && liveBenchSlot(1, battle) > 0
          ? `switch ${liveBenchSlot(1, battle)}`
          : 'move 2';
      } else {
        p2c = 'move 2'; // splash
      }
      return { p1: p1c, p2: p2c };
    };
  };

  // --- (1) ARENA TRAP traps a grounded foe → it MUST FIGHT to the end. Dugtrio (Arena
  //   Trap) vs Snorlax + Regice (both grounded): p2 WOULD switch when free (its script
  //   switches whenever un-trapped) but is trapped every boundary → it only ever fights;
  //   EQ grinds both to a win. The trapped columns pin the trap; the script's
  //   switch-when-free arm proves the trap held (a wrongly-free port would desync on the
  //   recorded choice). REQUIRES: trapped rows + a win. ---
  S.push({
    id: 'arena_trap_traps_grounded_foe_must_fight',
    p1: [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'splash'], { nature: 'Brave', ivs: { ...IV31, spe: 0 }, evs: { hp: 252, atk: 252 } }),
         mon('Regice', ['icebeam', 'splash'], { nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    makeScript: planP1(['move 1'], 'splash-or-switch'),
    require: ['trapped'],
  });

  // --- (2) FLYING + LEVITATE escape Arena Trap: Zapdos (Flying) and Gengar (Levitate)
  //   voluntarily switch OUT vs a Dugtrio every time they're in — every switch ACCEPTED
  //   (never trapped). Dugtrio Rock Slides (hits both) to a win. REQUIRES: a win (the
  //   free-switch churn is proven by the recorded s-choices replaying draw-compatibly). ---
  S.push({
    id: 'flying_and_levitate_switch_freely_vs_arena_trap',
    p1: [mon('Dugtrio', ['rockslide', 'splash'], { ability: 'Arena Trap', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Zapdos', ['drillpeck', 'splash'], { ability: 'Pressure', nature: 'Modest', evs: { hp: 252, spa: 252 } }),
         mon('Gengar', ['sludgebomb', 'splash'], { ability: 'Levitate', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    makeScript: planP1(['move 1'], 'splash-or-switch'),
    forbid: ['trapped'],
  });

  // --- (3) MAGNET PULL traps STEEL only (flying-irrelevant): p2's Snorlax (non-Steel)
  //   switches out FREELY on turn 0 (accepted); the entrant Skarmory (Steel/FLYING) is
  //   TRAPPED — its script would switch back when free but never can → it fights until
  //   Thunderbolt (2x vs Flying) KOs it; Snorlax replaces (forced, accepted) and fights
  //   (no bench). REQUIRES: trapped rows + a win. ---
  S.push({
    id: 'magnet_pull_traps_steel_only',
    // Magneton is deliberately WEAK special (Adamant, no SpA EVs) and Skarmory specially
    // BULKY (Careful hp/spd) so the trapped Skarmory survives several boundaries (a
    // Modest 252 Thunderbolt 2x OHKOs it before any trapped row records).
    p1: [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull', nature: 'Adamant', evs: { spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'splash'], { nature: 'Brave', ivs: { ...IV31, spe: 0 }, evs: { hp: 252, atk: 252 } }),
         mon('Skarmory', ['drillpeck', 'splash'], { ability: 'Keen Eye', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    makeScript: planP1(['move 1'], 'splash-or-switch'),
    require: ['trapped'],
  });

  // --- (4) THE MAGNETON MIRROR (mutual trap + THE DRAW MODEL): both actives Magnet Pull,
  //   IDENTICAL sets → equal cached speeds → EVERY endTurn draws 4 (TrapPokemon +
  //   MaybeTrapPokemon × 2 mons, 1 tie-shuffle each — the gen3 `onAny` override). Three
  //   splash-splash turns pin the +4 rhythm bit-for-bit (a port that drops ANY of them
  //   desyncs the boundary seed), then Thunderbolts (a para secondary that lands breaks
  //   the speed tie → those endTurns draw 0 — the cached-speed-dependent tie pinned both
  //   ways) to a KO; replacements fight to a win. Both trapped columns 1 while the mirror
  //   holds (Steel↔Steel). REQUIRES: trapped + mutualTrap rows + a win. ---
  S.push({
    id: 'magneton_mirror_mutual_trap_draws',
    p1: [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull', nature: 'Modest', evs: { spa: 252 } }),
         mon('Snorlax', ['bodyslam', 'splash'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull', nature: 'Modest', evs: { spa: 252 } }),
         mon('Regice', ['icebeam', 'splash'], { nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    makeScript: planP1(['move 2', 'move 2', 'move 2', 'move 1'], 'mirror'),
    require: ['trapped', 'mutualTrap'],
  });

  // --- (5) THE DUGTRIO MIRROR (mutual trap, ZERO extra draws): both actives Arena Trap →
  //   both grounded → MUTUALLY trapped, and (onFoe = 1 handler per event) the endTurn trap
  //   events draw NOTHING — probe-verified byte-identical to a no-trap control. Two splash
  //   turns pin the zero-draw rhythm, then the EQ mirror (action-order speed ties!) to a
  //   KO → replacement → win. REQUIRES: trapped + mutualTrap rows. ---
  S.push({
    id: 'dugtrio_mirror_mutual_trap_no_draws',
    p1: [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap', nature: 'Adamant', evs: { atk: 252 } }),
         mon('Snorlax', ['bodyslam', 'splash'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap', nature: 'Adamant', evs: { atk: 252 } }),
         mon('Regice', ['icebeam', 'splash'], { nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    makeScript: planP1(['move 2', 'move 2', 'move 1'], 'mirror'),
    require: ['trapped', 'mutualTrap'],
  });

  // --- (6) ROAR drags a TRAPPED mon out (phaze bypasses trapping): Dugtrio (Arena Trap)
  //   Roars the trapped Snorlax → the drag fires (acc + the n-bench `sample`) and Regice
  //   is dragged in — even though Snorlax could never have LEFT voluntarily. Then EQ to a
  //   win. REQUIRES: trapped + drag rows. ---
  S.push({
    id: 'roar_drags_a_trapped_mon',
    p1: [mon('Dugtrio', ['roar', 'earthquake'], { ability: 'Arena Trap', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'splash'], { nature: 'Brave', ivs: { ...IV31, spe: 0 }, evs: { hp: 252, atk: 252 } }),
         mon('Regice', ['icebeam', 'splash'], { nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    makeScript: planP1(['move 1', 'move 1', 'move 2'], 'splash-or-switch'),
    require: ['trapped', 'drag'],
  });

  // --- (7) ARENA TRAP vs MAGNET PULL CROSS at a full speed tie (the one-sided +2 draw):
  //   an Arena Trap Dugtrio vs a MAGNET PULL Dugtrio (hacked ability — customgame allows;
  //   same species/set → equal speeds). The MP-Dugtrio IS trapped (grounded vs Arena
  //   Trap); the AT-Dugtrio is NOT (Ground, not Steel — Magnet Pull can't hold it): the
  //   one-sided columns pin the asymmetry. Draw model: each event on the MP holder has 2
  //   tied handlers (own onAny + foe onFoe) → +2/endTurn; events on the AT holder have 1
  //   (foe's onAny) → 0 — a port that models the matrix wrong desyncs every boundary.
  //   Splash ×2 then EQ to a KO → replacement → win. REQUIRES: trapped rows; FORBIDS
  //   mutualTrap (the asymmetry). ---
  S.push({
    id: 'arenatrap_vs_magnetpull_cross_tie',
    p1: [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap', nature: 'Adamant', evs: { atk: 252 } })],
    p2: [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Magnet Pull', nature: 'Adamant', evs: { atk: 252 } }),
         mon('Regice', ['icebeam', 'splash'], { nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    makeScript: planP1(['move 2', 'move 2', 'move 1'], 'splash-or-switch'),
    require: ['trapped'],
    forbid: ['mutualTrap'],
  });

  // --- (8) TRAPPING INTO A REAL BATTLE (mixed, to game-end): Magneton (Magnet Pull) +
  //   Dugtrio (Arena Trap) vs Skarmory/Snorlax/Zapdos. p1 pivots Magneton → Dugtrio
  //   voluntarily (the TRAPPER side switches freely); p2 fights while trapped and
  //   switches when free — so trapped/free alternates with faints, replacements, and the
  //   full residual machinery to a win. REQUIRES: trapped rows + wins realized. ---
  S.push({
    id: 'trapping_into_a_real_battle',
    // Same weak-special Magneton / specially-bulky Skarmory tuning as (3) so the trapped
    // lead survives boundaries; Snorlax is un-invested (Relaxed) so Dugtrio isn't OHKO'd.
    p1: [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull', nature: 'Adamant', evs: { spe: 252 } }),
         mon('Dugtrio', ['earthquake', 'rockslide'], { ability: 'Arena Trap', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Skarmory', ['drillpeck', 'splash'], { ability: 'Keen Eye', nature: 'Careful', evs: { hp: 252, spd: 252 } }),
         mon('Snorlax', ['bodyslam', 'splash'], { nature: 'Relaxed', evs: { hp: 252 } }),
         mon('Zapdos', ['drillpeck', 'splash'], { ability: 'Pressure', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    makeScript: (() => () => {
      let pivoted = false;
      return (decisionNo, battle, reqState, force) => {
        if (reqState === 'switch') {
          const c = { p1: null, p2: null };
          if (force[0]) { c.p1 = firstLiveBench(0, battle); pivoted = true; }
          if (force[1]) c.p2 = firstLiveBench(1, battle);
          return c;
        }
        // p1: two Thunderbolt turns from Magneton, then pivot to Dugtrio (a free voluntary
        // switch by the TRAPPER side), then attack; robust to Magneton fainting first.
        const a0 = battle.sides[0].active[0];
        const magnetonOut = a0 && /magneton/i.test(a0.species.name);
        let p1c;
        if (magnetonOut && decisionNo >= 2 && !pivoted && liveBenchSlot(0, battle) > 0) {
          p1c = `switch ${liveBenchSlot(0, battle)}`;
          pivoted = true;
        } else {
          p1c = 'move 1';
        }
        // p2: fight while trapped; switch when free (rotates the trio through the trap).
        let p2c = firstUsableMove(1, battle);
        if (!trappedOf(battle, 1) && liveBenchSlot(1, battle) > 0 && decisionNo % 2 === 1) {
          p2c = `switch ${liveBenchSlot(1, battle)}`;
        }
        return { p1: p1c, p2: p2c };
      };
    })(),
    require: ['trapped'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# trapping_golden.txt — Gen-3 TRAPPING (Arena Trap / Magnet Pull) full-battle golden.');
  lines.push('# Per-decision-boundary STATE(+per-side TRAPPED)+SEED differential to GAME-END.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status left) p2(...) first trapP1 trapP2 drag');
  lines.push('#        (trapP1/trapP2: 0|1 at a move-request boundary, - at a forced-switch pause/game end)');
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
        const sp = (s) => [s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status, s.left].join('\t');
        lines.push([
          'DEC', sc.id, seedStr, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
          sp(d.p1), sp(d.p2), d.firstMover, d.trapped[0], d.trapped[1], d.outcomes.drag ? 1 : 0,
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
    console.error('TRAPPING GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`TRAPPING GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('trapped runs', 'trapped', 50);
  need('mutual-trap runs', 'mutualTrap', 20);
  need('phaze-drag runs', 'drag', 20);
  if (winRows < 50) { console.error(`TRAPPING GOLDEN: too few WIN rows (${winRows} < 50)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `trapping golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: trapped=${corpus.trapped || 0} mutualTrap=${corpus.mutualTrap || 0} drag=${corpus.drag || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
