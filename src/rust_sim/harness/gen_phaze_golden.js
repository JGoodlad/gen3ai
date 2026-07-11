// gen_phaze_golden.js — Gen-3 PHAZING (Roar / Whirlwind) differential golden.
//
// Extends harness/gen_spikes_golden.js (the per-decision STATE+STATUS+SPIKES-LAYERS+
// SEED+winner full-battle differential) to the NEW mechanic this step adds: the gen-3
// PHAZE moves **Roar** + **Whirlwind** (`forceSwitch: true`) — force the FOE to switch
// to a RANDOM eligible team member. DEFERRED (fail-loud in the engine): Haze (resets
// boosts — a DIFFERENT mechanic), Perish Song, Roar of Time (not gen3), and everything
// already deferred. Roar/Whirlwind are the only gen-3 phaze moves (`isPhaze` ==
// `forceSwitch`).
//
// THE DRAW MODEL (verified bit-for-bit vs the omniscient sim's PRNG probe,
// harness/probe_phaze_rng.js + .._draws/.._actions):
//
//   THE PHAZE MOVE (`forceSwitch: true`, `target: "normal"`, priority −6):
//     * PRIORITY −6 → the phazer almost always moves LAST (it sees the foe's move first).
//     * ACCURACY: gen-3 Roar/Whirlwind resolve to **`accuracy: 100`** (NOT `true`!), so
//       they DRAW `randomChance(100, 100)` — it ALWAYS passes but CONSUMES one draw. (The
//       base Showdown data lists `accuracy: true`; the gen-3 dex value is 100. So a phaze
//       is NOT never-miss — it draws the accuracy roll.)
//     * THE RANDOM TARGET DRAW: on a successful phaze, `forceSwitch` sets the foe's
//       `forceSwitchFlag` IFF `canSwitch(foe.side)` (the foe has >= 1 eligible non-active,
//       non-fainted bench mon). The runAction tail then `dragIn`s: `getRandomSwitchable`
//       → `sample(possibleSwitches)` → `this.random(n)` — ONE draw, EVEN when n == 1
//       (`random(1)` returns 0 but STILL calls `rng.next()` — the n=1 draw gotcha).
//     * THE FAIL CASE: a phaze with NO eligible target (the foe's last mon alive) →
//       `canSwitch` is false → NO forceSwitchFlag, NO drag, NO `sample` draw (the accuracy
//       roll is the ONLY draw the phaze move makes that turn).
//
//   THE DRAG (forced switch-in, `dragIn` → `switchIn(isDrag=true)`):
//     * The dragged-in mon switches in (NOT a player choice). It takes Spikes via the
//       existing runSwitch `runEvent('EntryHazard')` (drag → EntryHazard/Spikes → SwitchIn
//       → ability Start, IDENTICALLY to a normal switch-in), fires its switch-in ability
//       Start (Intimidate / Sand Stream), and a Spikes-KO on the dragged-in mon faints it
//       → forces a NORMAL replacement (the owner chooses). Boosts/volatiles of the phazed-
//       OUT mon are cleared (it left). The dragged mon does NOT get an action this turn
//       (the phaze is end-of-turn).
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios that each ISOLATE a branch, capturing the running PRNG seed BEFORE the first
// decision (`initSeed`) and AFTER each DECISION BOUNDARY, plus each active's species/hp/
// maxhp/fainted/status + boosts + confusion + pokemon_left + the per-side SPIKES LAYERS +
// the DRAGGED-IN species (per decision) + first mover + winner. The Rust test seeds a
// BattleState at the init seed and runs `run_full_battle` WITHOUT re-seeding — so the
// post-decision seed must match at EVERY boundary, AND the dragged-in mon (which active
// species ends up in the slot) + its post-drag HP (the spikes chip) must match. A
// DIFFERENT sampled mon → a STATE desync (active species); a wrong draw model → a SEED
// desync. The seed sweep makes DIFFERENT mons get dragged (the random-target proof).
//
// FAIL-LOUD: each scenario declares the BRANCH it must realize (a random drag, a 1-
// eligible n=1 drag, a phaze FAIL [foe last mon], a phaze-into-Spikes chip, a phaze-into-
// a-Spikes-KO → forced replacement); generation aborts if the sim run did NOT realize it.
// A scenario that sweeps the random-target additionally REQUIRES that >= 2 DISTINCT mons
// got dragged across the seed sweep (else the "random" isn't proven). The output shares
// the spikes TAB format with a 1-col dragged-species tail (the active species after the
// phaze already lands in the snapshot; the explicit `dragSpecies` col pins WHICH mon was
// pulled on THAT decision, surviving a same-turn chain).
//
// Output: tests/vectors/phaze_golden.txt
//
// Run:  node src/rust_sim/harness/gen_phaze_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/phaze_golden.txt');
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
  let x = 0x51ab37e9 >>> 0;
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

function spikesOf(side) {
  const sc = side.sideConditions && side.sideConditions['spikes'];
  return sc ? (sc.layers | 0) : 0;
}

function snap(side) {
  const a = side.active[0];
  if (!a) return { species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', stage: 0, left: side.pokemonLeft, boosts: [0, 0, 0, 0, 0], confusion: 0, spikes: spikesOf(side) };
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status, stage, left: side.pokemonLeft,
    boosts: boostsOf(a), confusion: confusionOf(a), spikes: spikesOf(side),
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

// Scan the protocol log between two decision points for the PHAZE + spikes branch flags.
//   phazeDrag  — a `|drag|` line (a phaze dragged a foe in this decision) + WHICH species
//   phazeFail  — a `|-fail|` on a phaze user with NO drag (foe's last mon)
//   spikesDamage / koEntry — reused from the spikes golden (a phaze-into-Spikes chip / KO)
function outcomesSince(log, fromIdx) {
  const out = {
    p1: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    p2: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    phazeDrag: false, phazeFail: false, spikesDamage: false, koEntry: false,
    dragSpecies: '-', dragSide: '-',
  };
  let lastSpikesDamageIdx = -100;
  let sawDrag = false;
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
    // A phaze DRAG: `|drag|p2a: Snorlax|Snorlax|524/524`. Record the dragged species + side.
    if (tag === 'drag' && who) {
      out.phazeDrag = true; sawDrag = true;
      // p[3] = the full details (e.g. "Snorlax" or "Sandshrew, L1"); take the species head.
      out.dragSpecies = (p[3] || '').split(',')[0].trim() || '-';
      out.dragSide = who;
    }
    // A phaze FAIL: `|-fail|p1a: Suicune` from a Roar with no eligible foe. Only count it as
    // a phaze-fail when NO drag happened this window (a `-fail` could also be a Spikes-at-max,
    // but the phaze scenarios never lay Spikes via the same user on the same fail decision).
    if (tag === '-fail' && !sawDrag) out.phazeFail = true;
    if (tag === '-damage' && (p[4] || '').includes('Spikes')) { out.spikesDamage = true; lastSpikesDamageIdx = i; }
    if (tag === 'faint' && i - lastSpikesDamageIdx <= 2) out.koEntry = true;
  }
  return out;
}

// The first live, non-active bench mon as a `switch N` choice (for forced replacements).
function firstLiveBench(side, battle) {
  const s = battle.sides[side];
  for (let k = 0; k < s.pokemon.length; k++) {
    const p = s.pokemon[k];
    if (p !== s.active[0] && !p.fainted) return `switch ${k + 1}`;
  }
  return 'pass';
}

// Clamp a desired 1-based move slot to a LEGAL one for `side`'s active (the request moves
// length) — falls back to the first usable move if the desired slot is out of range /
// disabled. Keeps the phaze scenarios robust as the dragged-in mon's moveset varies.
function legalMove(side, battle, want) {
  const req = battle.sides[side].activeRequest;
  const moves = req && req.active && req.active[0] ? req.active[0].moves : null;
  if (!moves) return 'move 1';
  const usable = [];
  for (let k = 0; k < moves.length; k++) if (!moves[k].disabled) usable.push(k + 1);
  if (usable.length === 0) return 'move 1';
  return `move ${usable.includes(want) ? want : usable[0]}`;
}

// Build a battle-aware script from a scenario's `intent(decisionNo, battle) -> {p1Want,
// p2Want}` (1-based desired move slots), clamping each to a legal move + auto-replacing on
// a forced switch. Avoids the brittleness of a fixed plan when a phaze drags an unknown mon.
function intentDriver(intent) {
  return (decisionNo, battle, reqState, force) => {
    if (reqState === 'switch') {
      const c = { p1: null, p2: null };
      if (force[0]) c.p1 = firstLiveBench(0, battle);
      if (force[1]) c.p2 = firstLiveBench(1, battle);
      return c;
    }
    const { p1Want, p2Want } = intent(decisionNo, battle);
    return { p1: legalMove(0, battle, p1Want), p2: legalMove(1, battle, p2Want) };
  };
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

  // A scenario provides EITHER an `intent` (the battle-aware move-slot driver) OR an
  // explicit `makeScript` closure (scenario 7's switch-once-then-sweep). Prefer `intent`.
  const script = sc.intent ? intentDriver(sc.intent) : sc.makeScript();
  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen };

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
    for (let i = 0; i < 18; i++) await tick();

    const seedAfter = battle.prng.getSeed();
    if (seedAfter === seedBefore && log.length === logLenBefore && battle.requestState === reqState) {
      throw new Error(`STALL: choice ${JSON.stringify(choices)} did not advance the battle ` +
        `(scenario ${sc.id}, decision ${decisionNo}) — the sim rejected it. Fix the script.`);
    }
    const outcomes = outcomesSince(log, logLenBefore);
    const first = reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none';

    const p1 = snap(battle.sides[0]);
    const p2 = snap(battle.sides[1]);

    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1, p2,
      firstMover: first,
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

  // The first live, non-active bench mon as a `switch N` choice (for forced replacements).
  const firstLiveBench = (side, battle) => {
    const s = battle.sides[side];
    for (let k = 0; k < s.pokemon.length; k++) {
      const p = s.pokemon[k];
      if (p !== s.active[0] && !p.fainted) return `switch ${k + 1}`;
    }
    return 'pass';
  };

  // Clamp a desired 1-based move slot to a LEGAL one for `side`'s active (the request
  // moves length) — falls back to move 1 if the desired slot is out of range / disabled.
  // Keeps the phaze scenarios robust as the dragged-in mon's moveset varies.
  const legalMove = (side, battle, want) => {
    const req = battle.sides[side].activeRequest;
    const moves = req && req.active && req.active[0] ? req.active[0].moves : null;
    if (!moves) return 'move 1';
    const usable = [];
    for (let k = 0; k < moves.length; k++) if (!moves[k].disabled) usable.push(k + 1);
    if (usable.length === 0) return 'move 1';
    return `move ${usable.includes(want) ? want : usable[0]}`;
  };

  // A battle-aware driver: `intent(decisionNo, battle)` returns the desired {p1Want, p2Want}
  // 1-based move slots for a `move` request; the driver clamps each to a legal move and
  // handles forced switches by auto-replacing with the first live bench. This avoids the
  // brittleness of a fixed plan when a phaze drags an unknown mon into the active slot.
  const driver = (intent) => () => (decisionNo, battle, reqState, force) => {
    if (reqState === 'switch') {
      const c = { p1: null, p2: null };
      if (force[0]) c.p1 = firstLiveBench(0, battle);
      if (force[1]) c.p2 = firstLiveBench(1, battle);
      return c;
    }
    const { p1Want, p2Want } = intent(decisionNo, battle);
    return { p1: legalMove(0, battle, p1Want), p2: legalMove(1, battle, p2Want) };
  };

  // --- (1) ROAR drags a RANDOM bench mon (the random-target proof). p1's FAST Starmie
  //   Roars (move 1) on the FIRST turn, then Surfs (move 2) the dragged-in mon to a sweep.
  //   p2's lead Blissey has 3 eligible bench (Snorlax / Skarmory / Vaporeon). Across the
  //   seed sweep DIFFERENT mons get dragged → the per-decision active species varies (the
  //   `distinctDrags` require below proves it). The foes are passive (Seismic Toss / weak
  //   physical) so a fast SpA Starmie Surf grinds them out — a clean win. REQUIRES:
  //   phazeDrag + distinctDrags + a win. ---
  S.push({
    id: 'roar_random_drag',
    p1: [mon('Starmie', ['roar', 'surf'], { ability: 'Illuminate', item: 'Leftovers', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Blissey', ['pound'], { ability: 'Natural Cure', item: 'Leftovers', nature: 'Bold', evs: { hp: 4 } }),
         mon('Snorlax', ['pound'], { ability: 'Immunity', item: 'Leftovers', nature: 'Careful', evs: { hp: 4 } }),
         mon('Skarmory', ['pound'], { ability: 'Keen Eye', item: 'Leftovers', nature: 'Impish', evs: { hp: 4 } }),
         mon('Sableye', ['pound'], { ability: 'Keen Eye', item: 'Leftovers', nature: 'Bold', evs: { hp: 4 } })],
    // Roar on turn 1 + every 4th turn (to drag fresh mons), else Surf. Foes use move 1.
    makeScript: () => (decisionNo, battle) => null, // replaced below
    intent: (decisionNo) => ({ p1Want: decisionNo % 4 === 0 ? 1 : 2, p2Want: 1 }),
    require: ['phazeDrag', 'distinctDrags'],
  });

  // --- (2) ROAR with ONE eligible bench (the n=1 draw gotcha): p2 has exactly ONE bench
  //   mon, so `sample([the one])` = `random(1)` (returns 0) — STILL a DRAW. The drag always
  //   pulls that one mon, but the SEED must advance by the sample draw (a wrong "n=1 is
  //   draw-free" model would desync the seed). REQUIRES: phazeDrag + a win. ---
  S.push({
    id: 'roar_n1_eligible_draws',
    p1: [mon('Starmie', ['roar', 'surf'], { ability: 'Illuminate', item: 'Leftovers', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Blissey', ['pound'], { ability: 'Natural Cure', item: 'Leftovers', nature: 'Bold', evs: { hp: 4 } }),
         mon('Snorlax', ['pound'], { ability: 'Immunity', item: 'Leftovers', nature: 'Careful', evs: { hp: 4 } })],
    makeScript: () => (decisionNo, battle) => null,
    intent: (decisionNo) => ({ p1Want: decisionNo === 0 ? 1 : 2, p2Want: 1 }),
    require: ['phazeDrag'],
  });

  // --- (3) ROAR FAILS (foe's last mon — no eligible target): p2 has only ONE mon, so
  //   `canSwitch` is FALSE → NO forceSwitchFlag, NO drag, NO `sample` draw — only the
  //   accuracy roll. The phaze is a `-fail` no-op. The per-decision SEED must reflect the
  //   accuracy-only draw (a wrong model that drew the sample anyway desyncs). p1 then Surfs
  //   the lone Blissey to death. REQUIRES: phazeFail + NO phazeDrag (forbid) + a win. ---
  S.push({
    id: 'roar_fails_last_mon',
    p1: [mon('Starmie', ['roar', 'surf'], { ability: 'Illuminate', item: 'Leftovers', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Blissey', ['pound'], { ability: 'Natural Cure', item: '', nature: 'Bold', evs: { hp: 4 } })],
    makeScript: () => (decisionNo, battle) => null,
    // Roar (fails) on turn 1 + occasionally, else Surf the lone Blissey out.
    intent: (decisionNo) => ({ p1Want: decisionNo % 3 === 0 ? 1 : 2, p2Want: 1 }),
    require: ['phazeFail'],
    forbid: ['phazeDrag'],
  });

  // --- (4) WHIRLWIND drags a random bench mon (the second phaze move — identical mechanic,
  //   different id). p1's fast Aerodactyl Whirlwinds (move 1) then Rock Slides (move 2) the
  //   dragged mons out. p2 has 3 eligible bench. REQUIRES: phazeDrag + distinctDrags + win. ---
  S.push({
    id: 'whirlwind_random_drag',
    p1: [mon('Aerodactyl', ['whirlwind', 'rockslide'], { ability: 'Pressure', item: 'Choice Band', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Blissey', ['pound'], { ability: 'Natural Cure', item: 'Leftovers', nature: 'Bold', evs: { hp: 4 } }),
         mon('Snorlax', ['pound'], { ability: 'Immunity', item: 'Leftovers', nature: 'Careful', evs: { hp: 4 } }),
         mon('Starmie', ['pound'], { ability: 'Illuminate', item: 'Leftovers', nature: 'Timid', evs: { hp: 4 } }),
         mon('Tyranitar', ['pound'], { ability: 'Sand Stream', item: 'Leftovers', nature: 'Adamant', evs: { hp: 4 } })],
    makeScript: () => (decisionNo, battle) => null,
    intent: (decisionNo) => ({ p1Want: decisionNo % 3 === 0 ? 1 : 2, p2Want: 1 }),
    require: ['phazeDrag', 'distinctDrags'],
  });

  // --- (5) ROAR INTO SPIKES (the dragged mon takes the hazard chip): p1's Skarmory lays
  //   2 layers of Spikes on the p2 side (move 1), then Roars (move 2) — the dragged GROUNDED
  //   bench mon takes floor(maxhp/6) (2 layers) on entry via the runSwitch EntryHazard. (All
  //   p2 bench are grounded so EVERY drag chips.) p1 then Drill-Pecks (move 3) the chipped
  //   foes out. REQUIRES: phazeDrag + spikesDamage + a win. ---
  S.push({
    id: 'roar_into_spikes',
    p1: [mon('Skarmory', ['spikes', 'roar', 'drillpeck'], { ability: 'Keen Eye', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Marowak', ['pound'], { ability: 'Rock Head', item: '', nature: 'Bold', evs: { hp: 4 } }),
         mon('Donphan', ['pound'], { ability: 'Sturdy', item: '', nature: 'Bold', evs: { hp: 4 } }),
         mon('Phanpy', ['pound'], { ability: 'Pickup', item: '', nature: 'Bold', evs: { hp: 4 } })],
    makeScript: () => (decisionNo, battle) => null,
    // Lay 2 spikes (turns 0,1), then Roar (turn 2) + every 3rd turn, else Drill Peck.
    intent: (decisionNo) => {
      if (decisionNo < 2) return { p1Want: 1, p2Want: 1 }; // Spikes ×2
      return { p1Want: decisionNo % 3 === 2 ? 2 : 3, p2Want: 1 }; // Roar then Drill Peck
    },
    require: ['phazeDrag', 'spikesDamage'],
  });

  // --- (6) ROAR repeatedly INTO 3-LAYER SPIKES (the dragged mons accumulate chip, and a
  //   late re-drag of an already-chipped lvl-1 mon is SPIKES-KO'd on entry → a NORMAL
  //   replacement): p1 lays 3 layers of Spikes (move 1), then Roars (move 2) over and over.
  //   The two GROUNDED lvl-1 bench mons (~11 HP) take floor(maxhp/4) every re-entry, so they
  //   whittle down (deterministically) until a re-drag at low HP is KO'd BY THE SPIKES ON
  //   ENTRY, forcing a NORMAL replacement (the survivor, also chipped on entry). The chain
  //   (re-drag → spikes KO → forced replace → spikes) tests the composition. The KO is
  //   STOCHASTIC across seeds (which mon gets re-dragged), so `koEntry` is a COVERAGE counter
  //   here (realized on SOME seeds), and the exact single-drag spikes-KO is ALSO pinned by a
  //   DETERMINISTIC regression test (`phaze_drag_into_a_spikes_ko_*`). REQUIRES: phazeDrag +
  //   spikesDamage + a win. ---
  S.push({
    id: 'roar_repeated_into_spikes',
    p1: [mon('Skarmory', ['spikes', 'roar', 'drillpeck'], { ability: 'Keen Eye', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Marowak', ['pound'], { ability: 'Rock Head', item: '', nature: 'Bold', evs: { hp: 4 } }),
         mon('Diglett', ['pound'], { level: 1, ability: 'Sand Veil', nature: 'Bold' }),
         mon('Sandshrew', ['pound'], { level: 1, ability: 'Sand Veil', nature: 'Bold' })],
    makeScript: () => (decisionNo, battle) => null,
    // Lay 3 spikes (turns 0,1,2), then Roar EVERY turn after (re-drag the lvl-1 chippers so
    // they accumulate spikes chip until a re-drag spikes-KOs the low one).
    intent: (decisionNo) => {
      if (decisionNo < 3) return { p1Want: 1, p2Want: 1 }; // Spikes ×3
      return { p1Want: 2, p2Want: 1 }; // Roar repeatedly
    },
    require: ['phazeDrag', 'spikesDamage'],
  });

  // --- (7) PHAZE IN A REAL BATTLE TO GAME-END (the union: phaze + spikes + a voluntary
  //   switch + the full move/residual/faint machinery): p1's Skarmory lays Spikes + Roars a
  //   grounded foe in (it takes chip), p1 pivots to a hard hitter (Salamence) that sweeps
  //   the chipped foe team out to a clean win — spikes layers PERSIST across the phaze +
  //   switches. REQUIRES: phazeDrag + spikesDamage + a win. ---
  S.push({
    id: 'phaze_into_real_battle',
    p1: [mon('Skarmory', ['spikes', 'roar', 'drillpeck'], { ability: 'Keen Eye', item: 'Leftovers', nature: 'Impish', evs: { hp: 252, def: 252 } }),
         mon('Salamence', ['rockslide', 'earthquake'], { ability: 'Intimidate', item: 'Leftovers', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    // FRAIL lvl-1 grounded foes (OHKO'd by ANY hit) so the battle ENDS in a few decisions —
    // well before any PP exhaustion (the engine doesn't model PP/Struggle, so a long slog
    // where the foe's move PP runs out and it Struggles would desync). Their pound can't dent
    // the bulky Skarmory, and a lvl-1 mon takes 3-layer spikes well below its tiny maxhp.
    p2: [mon('Diglett', ['pound'], { level: 1, ability: 'Sand Veil', nature: 'Bold' }),
         mon('Sandshrew', ['pound'], { level: 1, ability: 'Sand Veil', nature: 'Bold' }),
         mon('Cubone', ['pound'], { level: 1, ability: 'Rock Head', nature: 'Bold' })],
    makeScript: () => {
      // Spikes ×2, Roar (drag a grounded lvl-1 foe → spikes chip), switch to Salamence, then
      // sweep with Earthquake (move 2 — Ground OHKOs the frail lvl-1 grounded foes; Rock Slide
      // [move 1] is Ground-resisted-Rock-vs-Ground so Earthquake is the cleaner sweep).
      let switched = false;
      return (decisionNo, battle, reqState, force) => {
        if (reqState === 'switch') {
          const c = { p1: null, p2: null };
          if (force[0]) c.p1 = firstLiveBench(0, battle);
          if (force[1]) c.p2 = firstLiveBench(1, battle);
          return c;
        }
        const p1Active = battle.sides[0].active[0];
        const isSkarmory = p1Active && p1Active.species.name === 'Skarmory';
        let p1;
        if (isSkarmory && decisionNo < 2) p1 = legalMove(0, battle, 1); // Spikes ×2
        else if (isSkarmory && decisionNo === 2) p1 = legalMove(0, battle, 2); // Roar
        else if (isSkarmory && !switched) { p1 = 'switch 2'; switched = true; } // pivot to Salamence
        else p1 = legalMove(0, battle, 2); // Salamence Earthquake (move 2) — OHKOs the frail foes
        return { p1, p2: legalMove(1, battle, 1) };
      };
    },
    require: ['phazeDrag', 'spikesDamage'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# phaze_golden.txt — Gen-3 PHAZING (Roar / Whirlwind) full-battle golden.');
  lines.push('# Per-decision-boundary STATE(+status+spikes-layers)+BOOSTS+SEED+first-mover+dragSpecies differential to GAME-END.');
  lines.push('# (Extends the spikes TAB format with a 1-col dragged-species tail.)');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1(fullpara wake thaw selfhit flinch) p2(...) spikesDamage  p1Spikes p2Spikes  dragSpecies');
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
    const dragSpeciesSeen = new Set();

    let scenDecs = 0;
    for (const seed of seeds) {
      let rec;
      try { rec = await runBattle(sc, seed); } catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); continue; }
      if (rec.gen !== 3) { failures.push(`${sc.id}: expected gen 3, got ${rec.gen}`); break; }
      if (!rec.initSeed || rec.decisions.length === 0) { failures.push(`${sc.id} seed ${seed}: no decisions`); continue; }
      const seedStr = seed.join(',');
      lines.push(['INIT', sc.id, rec.initSeed, seedStr].join('\t'));

      rec.decisions.forEach((d) => {
        if (d.outcomes.phazeDrag) { scenSeen[sc.id].phazeDrag = true; corpus.phazeDrag = (corpus.phazeDrag || 0) + 1; dragSpeciesSeen.add(d.outcomes.dragSpecies); }
        if (d.outcomes.phazeFail) { scenSeen[sc.id].phazeFail = true; corpus.phazeFail = (corpus.phazeFail || 0) + 1; }
        if (d.outcomes.spikesDamage) { scenSeen[sc.id].spikesDamage = true; corpus.spikesDamage = (corpus.spikesDamage || 0) + 1; }
        if (d.outcomes.koEntry) { scenSeen[sc.id].koEntry = true; corpus.koEntry = (corpus.koEntry || 0) + 1; }
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
          oc(d.outcomes.p1), oc(d.outcomes.p2), d.outcomes.spikesDamage ? 1 : 0,
          d.p1.spikes, d.p2.spikes, d.outcomes.dragSpecies,
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
    // The "distinctDrags" pseudo-branch: >= 2 DISTINCT mons dragged across the sweep
    // (proving the random-target draw genuinely varied, not a fixed pick).
    if (dragSpeciesSeen.size >= 2) scenSeen[sc.id].distinctDrags = true;

    for (const need of (sc.require || [])) {
      if (!scenSeen[sc.id][need]) failures.push(`${sc.id}: REQUIRED branch ${need} never realized across the seed sweep (dragsSeen=${[...dragSpeciesSeen].join(',')})`);
    }
    for (const bad of (sc.forbid || [])) {
      if (scenSeen[sc.id][bad]) failures.push(`${sc.id}: FORBIDDEN branch ${bad} realized (the scenario isolation is broken)`);
    }
  }

  if (failures.length) {
    console.error('PHAZE GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`PHAZE GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('phaze DRAG decisions', 'phazeDrag', 50);
  need('phaze FAIL decisions', 'phazeFail', 50);
  need('phaze-into-Spikes DAMAGE decisions', 'spikesDamage', 50);
  // phaze-into-a-Spikes-KO is STOCHASTIC across seeds (which lvl-1 mon gets re-dragged low
  // enough); the EXACT single-drag spikes-KO is pinned by a DETERMINISTIC regression test
  // (phaze_drag_into_a_spikes_ko_*). Here it's a soft coverage counter (realized on SOME seeds).
  need('phaze-into-Spikes-KO decisions', 'koEntry', 1);
  if (winRows < 50) { console.error(`PHAZE GOLDEN: too few WIN rows (${winRows} < 50)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `phaze golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: phazeDrag=${corpus.phazeDrag || 0} phazeFail=${corpus.phazeFail || 0} ` +
    `spikesDamage=${corpus.spikesDamage || 0} koEntry=${corpus.koEntry || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
