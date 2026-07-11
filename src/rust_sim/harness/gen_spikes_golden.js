// gen_spikes_golden.js — Gen-3 SPIKES (entry hazard) differential golden.
//
// Extends harness/gen_protect_move_golden.js (the per-decision STATE+STATUS+BOOSTS+
// SEED+winner full-battle differential) to the NEW mechanic this step adds: the ENTRY
// HAZARD **Spikes** — the first SIDE CONDITION (a per-side persistent layer count) +
// the grounded switch-in damage. DEFERRED (excluded / fail-loud in the engine): Toxic
// Spikes + Stealth Rock (NOT gen3), Rapid Spin (the hazard-clear move). Spikes is the
// only gen-3 entry hazard.
//
// THE DRAW + DAMAGE MODEL (verified bit-for-bit vs the omniscient sim's PRNG probe,
// harness/probe_spikes_rng.js + the resolved gen3 `spikes` condition):
//
//   THE SPIKES MOVE (`sideCondition: "spikes"`, `target: "foeSide"`):
//     * NEVER-MISS (`accuracy: true`) → NO accuracy draw.
//     * Increments the CASTER's FOE side's `spikes` layer count by 1, CAPPED at 3
//       (`onSideRestart`: `if (layers >= 3) return false` → a Spikes at 3 FAILS, `-fail`).
//     * DRAW-FREE both ways (the onSideStart/onSideRestart consume NO PRNG); `landed`
//       FALSE (the in-tryMoveHit Update shuffle is skipped). So a Spikes-vs-move turn
//       draws ONLY the existing action-order/eachEvent shuffles.
//
//   THE SWITCH-IN DAMAGE (the gen-3 `runSwitch`'s `runEvent('EntryHazard')`, gen4-
//   inherited; ORDER: EntryHazard → SwitchIn → `if (!hp) return` → ability Start):
//     * GROUNDED-ONLY: a Flying-type / Levitate entrant takes ZERO (`isGrounded()`).
//     * Amount (the resolved `spikes.onEntryHazard`, `[_,3,4,6][layers]*maxhp/24` →
//       `damage()` → `clampIntRange(_,1)`): 1 layer `max(floor(maxhp/8),1)`, 2 layers
//       `max(floor(maxhp/6),1)`, 3 layers `max(floor(maxhp/4),1)`.
//     * DRAW-FREE (the deterministic `this.damage`; the nested `runEvent('Damage')` has
//       no drawing handler for the modeled abilities). A Spikes hit that KOs the entrant
//       faints it → forces ANOTHER replacement (which ALSO takes Spikes); no Quick Claw.
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios that each ISOLATE a branch, capturing the running PRNG seed BEFORE the first
// decision (`initSeed`) and AFTER each DECISION BOUNDARY, plus each active's species/hp/
// maxhp/fainted/status + boosts + confusion + pokemon_left + THE SPIKES LAYERS (per side)
// + first mover + winner. The Rust test seeds a BattleState at the init seed and runs
// `run_full_battle` WITHOUT re-seeding — so the post-decision seed must match at EVERY
// boundary, AND the per-decision HP (the spikes switch-in damage) + the per-side spikes
// layers must match. (A wrong layer count or switch-in damage desyncs the HP STATE; a
// wrong draw model desyncs the SEED.)
//
// FAIL-LOUD: each scenario declares the BRANCH it must realize (a grounded switch-in
// taking spikes damage, a 2-/3-layer hit, a Flying/Levitate IMMUNE entry, a Spikes-at-max
// FAIL, a spikes-KO-on-switch-in → forced replacement); generation aborts if the sim run
// did NOT realize it. The output shares the protect TAB format with a 2-col spikes-layers
// tail (p1Spikes, p2Spikes) so the Rust gate reuses the parser.
//
// Output: tests/vectors/spikes_golden.txt
//
// Run:  node src/rust_sim/harness/gen_spikes_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/spikes_golden.txt');
// gen3customgame (NOT gen3ou) — matches the e2e capstone + the other goldens. No
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
  let x = 0x51ab37e9 >>> 0;
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

// The Spikes layer count on a side (`side.sideConditions.spikes.layers`, 0 = absent).
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

// Scan the protocol log between two decision points for the SPIKES branch flags:
//   sideStart  — `|-sidestart|p2: ...|Spikes` (a Spikes layer was laid / stacked)
//   sideFail   — `|-fail|p1a: ...` on a Spikes user (a Spikes-at-max FAIL)
//   damage     — `|-damage|p2a: ...|[from] Spikes` (a grounded switch-in took spikes)
//   immune     — a switch-in onto a spiked side that took NO `-damage Spikes` (Flying/
//                Levitate) — detected in main() via the HP delta, not here.
//   koEntry    — a `|faint|` immediately after a switch-in spikes `-damage` (a spikes KO)
function outcomesSince(log, fromIdx) {
  const out = {
    p1: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    p2: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    sideStart: false, sideFail: false, spikesDamage: false, koEntry: false,
  };
  let lastSpikesDamageIdx = -100;
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
    // A Spikes layer laid / stacked.
    if (tag === '-sidestart' && (p[3] || '').includes('Spikes')) out.sideStart = true;
    // A Spikes-at-max FAIL (the move user gets `-fail`). (gen3 emits `|-fail|<user>` for a
    // failed Spikes-at-3.)
    if (tag === '-fail') out.sideFail = true;
    // A grounded switch-in took spikes damage.
    if (tag === '-damage' && (p[4] || '').includes('Spikes')) { out.spikesDamage = true; lastSpikesDamageIdx = i; }
    // A spikes KO on entry: a faint right after a spikes -damage on the same mon.
    if (tag === 'faint' && i - lastSpikesDamageIdx <= 2) out.koEntry = true;
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
        // For a forced replacement, prefer the SCRIPT's choice if it provided one for the
        // flagged side at this step; else fall back to the first live bench. We thread the
        // plan's next entry's switch fields through for forced steps that the plan
        // explicitly scripts (the KO-chain scenario), else auto-pick.
        const entry = plan[Math.min(i, plan.length - 1)];
        const c = { p1: null, p2: null };
        if (force[0]) c.p1 = (entry && entry.fp1) || firstLiveBench(0, battle);
        if (force[1]) c.p2 = (entry && entry.fp2) || firstLiveBench(1, battle);
        // A forced replacement consumes a plan slot only if the slot was a forced-switch
        // directive (carries fp1/fp2); otherwise it's auto and we do NOT advance i.
        if (entry && (entry.fp1 || entry.fp2)) i++;
        return c;
      }
      const entry = plan[Math.min(i, plan.length - 1)];
      i++;
      return entry;
    };
  };

  // --- (1) LAY 1 SPIKES → a GROUNDED switch-in takes maxhp/8: p1's Skarmory lays one
  //   layer of Spikes on the p2 side, then p2 voluntarily switches a GROUNDED but FRAIL
  //   Marowak in — it takes floor(maxhp/8) (262 maxhp → 32). The per-decision HP (the
  //   exact chip) + the p2 spikes-layer count (1) prove the lay + the 1-layer switch-in
  //   damage. p1 then sweeps the frail foe out (Drill Peck OHKOs the chipped Marowak; the
  //   lead Diglett is frail too). REQUIRES: sideStart + spikesDamage + a win. ---
  S.push({
    id: 'lay1_grounded_switchin',
    p1: [mon('Skarmory', ['spikes', 'drillpeck'], { ability: 'Keen Eye', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    // Frail GROUNDED foes: Diglett (lead) + Marowak (the spikes-taker). A max-Atk Adamant
    // Skarmory Drill Peck OHKOs/2HKOs both, so the battle ENDS in a win.
    p2: [mon('Diglett', ['mudshot'], { ability: 'Sand Veil', nature: 'Jolly', evs: { spe: 252 } }),
         mon('Marowak', ['earthquake'], { ability: 'Rock Head', nature: 'Adamant', evs: { atk: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Skarmory Spikes (p2 side → 1 layer) ; Diglett Mud-Shot
      { p1: 'move 2', p2: 'switch 2' }, // p2 → Marowak IN → takes floor(maxhp/8)
      { p1: 'move 2', p2: 'move 1' }, // Drill Peck the chipped Marowak
      { p1: 'move 2', p2: 'move 1' }, // KO Marowak → forced replace to Diglett (also on spikes)
      { p1: 'move 2', p2: 'move 1' }, // Drill Peck the frail Diglett out
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['sideStart', 'spikesDamage'],
  });

  // --- (2) STACK to 2 + 3 LAYERS (the increasing damage) + a Spikes-at-max FAIL: p1's
  //   Skarmory lays Spikes 3× (2 then 3 layers), and a 4th Spikes FAILS (already at 3,
  //   `-fail`, draw-free). Then p2 switches a GROUNDED Marowak in onto 3 layers → takes
  //   floor(maxhp/4). The per-decision spikes-layer count (1→2→3, stays 3 on the fail) +
  //   the 3-layer switch-in damage prove the stack + the cap + the FAIL. p1 then sweeps
  //   the frail foe out. REQUIRES: sideStart + sideFail + spikesDamage + a win. ---
  S.push({
    id: 'stack3_fail_then_grounded_switchin',
    p1: [mon('Skarmory', ['spikes', 'drillpeck'], { ability: 'Keen Eye', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Diglett', ['mudshot'], { ability: 'Sand Veil', nature: 'Jolly', evs: { spe: 252 } }),
         mon('Marowak', ['earthquake'], { ability: 'Rock Head', nature: 'Adamant', evs: { atk: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Spikes (1) ; Diglett Mud-Shot
      { p1: 'move 1', p2: 'move 1' }, // Spikes (2)
      { p1: 'move 1', p2: 'move 1' }, // Spikes (3)
      { p1: 'move 1', p2: 'move 1' }, // Spikes (already 3 → FAIL, draw-free)
      { p1: 'move 2', p2: 'switch 2' }, // Marowak IN onto 3 layers → floor(maxhp/4)
      { p1: 'move 2', p2: 'move 1' }, // Drill Peck the chipped Marowak → KO → replace to Diglett
      { p1: 'move 2', p2: 'move 1' }, // Drill Peck the frail Diglett out
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['sideStart', 'sideFail', 'spikesDamage'],
  });

  // --- (3) A FLYING / LEVITATE switch-in takes ZERO: p1 lays 2 layers, then p2 switches a
  //   FLYING Salamence in (ZERO spikes — Flying), then a LEVITATE Claydol in (ZERO spikes
  //   — Levitate), then back to the FRAIL grounded lead which the spikes DID chip; p1 then
  //   sweeps. The per-decision HP (NO spikes chip on the Flying/Levitate entries — only the
  //   move chip) + the spikes layers (2, unchanged) prove the grounded gate. REQUIRES:
  //   sideStart + immuneEntry + a win. ---
  S.push({
    id: 'flying_levitate_immune',
    p1: [mon('Skarmory', ['spikes', 'drillpeck'], { ability: 'Keen Eye', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    // A frail grounded lead (Diglett) + a FLYING Zapdos (immune) + a LEVITATE Claydol
    // (immune). The two immune mons bounce in (ZERO spikes), then p1 KOs the frail trio.
    p2: [mon('Diglett', ['mudshot'], { ability: 'Sand Veil', nature: 'Jolly', evs: { spe: 252 } }),
         mon('Zapdos', ['thunderbolt'], { ability: 'Pressure', nature: 'Modest', evs: { spa: 252 } }),       // Flying → immune
         mon('Claydol', ['psychic'], { ability: 'Levitate', nature: 'Modest', evs: { spa: 252 } })],          // Levitate → immune
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Spikes (1) ; Diglett Mud-Shot
      { p1: 'move 1', p2: 'switch 2' }, // Spikes (2) ; Zapdos (Flying) IN onto 1 layer → ZERO
      { p1: 'move 2', p2: 'switch 3' }, // Drill Peck ; Claydol (Levitate) IN onto 2 layers → ZERO
      { p1: 'move 2', p2: 'switch 2' }, // Drill Peck ; back to Zapdos → still ZERO spikes
      { p1: 'move 2', p2: 'move 1' }, // Drill Peck the Zapdos (4× weak) → KO → replace
      { p1: 'move 2', p2: 'move 1' }, // grind the rest out
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['sideStart', 'immuneEntry'],
  });

  // --- (4) A SPIKES HIT THAT KOs a LOW-HP switch-in → forced replacement (which ALSO
  //   takes spikes): p1 lays 3 layers; p2's GROUNDED chippers bounce in and out, each
  //   re-entry taking floor(maxhp/4) spikes, so they whittle down (deterministic — spikes
  //   is draw-free) until a re-entry at HP ≤ floor(maxhp/4) is KO'd BY THE SPIKES ON ENTRY,
  //   forcing ANOTHER replacement (which ALSO takes spikes on its entry). p1 never attacks
  //   the entrant (Splash) so the spikes hit is the SOLE damage — a pure spikes-KO-on-entry
  //   chain that ENDS the battle (both p2 mons die to spikes). The per-decision HP (the chip
  //   ladder) + the KO + the forced-replacement request boundaries prove the KO → replacement
  //   wiring. REQUIRES: spikesDamage + koEntry (a spikes KO on switch-in) + a win. ---
  S.push({
    id: 'spikes_ko_on_entry_forces_replacement',
    // Skarmory: Spikes (the layer) + Splash (a draw-free no-op while bouncing) + Drill Peck
    // (to FINISH the last mon, which can no longer bounce into spikes). Bulky so the chippers
    // can't dent it.
    p1: [mon('Skarmory', ['spikes', 'splash', 'drillpeck'], { ability: 'Keen Eye', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    // Two LOW-maxhp GROUNDED chippers (lvl-1 Diglett/Sandshrew): floor(maxhp/4) spikes chips
    // them every entry; bouncing them whittles each down so a later re-entry at HP ≤
    // floor(maxhp/4) is a PURE SPIKES KO → forces the OTHER in (also chipped on entry). Their
    // only move (Scratch) can't dent a bulky Skarmory. Once only ONE mon remains (can't
    // bounce), p1 Drill-Pecks it out → a clean p1 win. Spikes is draw-free + deterministic,
    // so the whole chip ladder + the spikes-KO are seed-independent.
    p2: [mon('Diglett', ['scratch'], { level: 1, ability: 'Sand Veil' }),
         mon('Sandshrew', ['scratch'], { level: 1, ability: 'Sand Veil' })],
    // CUSTOM script: lay 3 spikes, then pivot to the OTHER live mon each turn (so each entry
    // re-takes spikes) until only one mon remains, then Drill-Peck it out.
    makeScript: () => {
      let laid = 0;
      return (decisionNo, battle, reqState, force) => {
        if (reqState === 'switch') {
          const c = { p1: null, p2: null };
          if (force[0]) c.p1 = firstLiveBench(0, battle);
          if (force[1]) c.p2 = firstLiveBench(1, battle);
          return c;
        }
        const s2 = battle.sides[1];
        // p2: pivot to the FIRST live, non-active bench mon (so the entry re-takes spikes);
        // if none can switch (only the active is left), attack (Scratch) — a draw-free filler.
        let p2 = 'move 1';
        let canSwitch = false;
        for (let k = 0; k < s2.pokemon.length; k++) {
          const p = s2.pokemon[k];
          if (p !== s2.active[0] && !p.fainted) { p2 = `switch ${k + 1}`; canSwitch = true; break; }
        }
        const layers = (s2.sideConditions && s2.sideConditions.spikes) ? s2.sideConditions.spikes.layers : 0;
        // p1: lay Spikes until 3 layers; while the foe can still bounce, Splash (let spikes
        // do the work); once the foe can NO LONGER switch (a lone mon), Drill-Peck it out.
        let p1;
        if (laid < 3 || layers < 3) { p1 = 'move 1'; laid++; }
        else if (canSwitch) { p1 = 'move 2'; } // Splash — let the bounce + spikes chip
        else { p1 = 'move 3'; } // Drill Peck the lone survivor out
        return { p1, p2 };
      };
    },
    require: ['spikesDamage', 'koEntry'],
  });

  // --- (5) SPIKES IN A REAL BATTLE TO GAME-END (the union: Spikes + a switch + the full
  //   move/residual/faint machinery): p1's Skarmory lays Spikes, p2 pivots a grounded mon
  //   in (takes chip), p1 pivots to a hard hitter that sweeps the FRAIL foe team out to a
  //   clean win — Spikes layers PERSIST across both sides' switches the whole way. The
  //   union exercises spikes alongside a real switch + residual + faints to a win.
  //   REQUIRES: sideStart + spikesDamage + a win. ---
  S.push({
    id: 'spikes_into_real_battle',
    p1: [mon('Skarmory', ['spikes', 'drillpeck'], { ability: 'Keen Eye', item: 'Leftovers', nature: 'Impish', evs: { hp: 252, def: 252 } }),
         mon('Salamence', ['dragonclaw', 'earthquake'], { ability: 'Intimidate', item: 'Leftovers', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    // Two FRAIL grounded chippers (Quick Attack only) — they take spikes on entry, can't
    // dent a bulky Skarmory, and a max-Atk Adamant Salamence OHKOs the pair with Dragon Claw.
    p2: [mon('Marowak', ['quickattack'], { ability: 'Rock Head', item: 'Leftovers', nature: 'Jolly', evs: { atk: 252, spe: 252 } }),
         mon('Dugtrio', ['quickattack'], { ability: 'Arena Trap', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Skarmory Spikes (1) ; Marowak Quick Attack
      { p1: 'move 1', p2: 'switch 2' }, // Spikes (2) ; p2 → Dugtrio IN → takes floor(maxhp/8 or /6) spikes
      { p1: 'switch 2', p2: 'move 1' }, // p1 → Salamence IN (p1 side has NO spikes — clean)
      { p1: 'move 1', p2: 'move 1' }, // Dragon Claw — KO the frail Dugtrio
      { p1: 'move 1', p2: 'move 1' }, // Dragon Claw the second frail foe out
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
    ]),
    require: ['sideStart', 'spikesDamage'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# spikes_golden.txt — Gen-3 SPIKES (entry hazard) full-battle golden.');
  lines.push('# Per-decision-boundary STATE(+status+spikes-layers)+BOOSTS+SEED+first-mover differential to GAME-END.');
  lines.push('# (Extends the protect TAB format with a 2-col spikes-layers tail.)');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1(fullpara wake thaw selfhit flinch) p2(...) spikesDamage  p1Spikes p2Spikes');
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

      // Per-decision branch detection.
      let prevP2Spikes = 0;
      rec.decisions.forEach((d) => {
        if (d.outcomes.sideStart) { scenSeen[sc.id].sideStart = true; corpus.sideStart = (corpus.sideStart || 0) + 1; }
        if (d.outcomes.sideFail) { scenSeen[sc.id].sideFail = true; corpus.sideFail = (corpus.sideFail || 0) + 1; }
        if (d.outcomes.spikesDamage) { scenSeen[sc.id].spikesDamage = true; corpus.spikesDamage = (corpus.spikesDamage || 0) + 1; }
        if (d.outcomes.koEntry) { scenSeen[sc.id].koEntry = true; corpus.koEntry = (corpus.koEntry || 0) + 1; }
        // An IMMUNE entry: a switch-in onto a side that HAS spikes (>0 layers) where the
        // entrant took NO spikes -damage this decision. (We detect it via: this decision was
        // a switch-in choice onto a >0-layer side with no spikesDamage flag.) Approximated
        // here as: p2 chose a switch, p2 side has spikes > 0, and no spikes damage fired.
        if (d.choiceP2 && d.choiceP2.startsWith('s') && d.p2.spikes > 0 && !d.outcomes.spikesDamage) {
          scenSeen[sc.id].immuneEntry = true; corpus.immuneEntry = (corpus.immuneEntry || 0) + 1;
        }
        prevP2Spikes = d.p2.spikes;
      });
      void prevP2Spikes;

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
          d.p1.spikes, d.p2.spikes,
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
    console.error('SPIKES GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`SPIKES GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('spikes-lay (sideStart) decisions', 'sideStart', 50);
  need('spikes switch-in DAMAGE decisions', 'spikesDamage', 50);
  need('spikes-at-max FAIL decisions', 'sideFail', 50);
  need('Flying/Levitate IMMUNE-entry decisions', 'immuneEntry', 50);
  need('spikes-KO-on-entry decisions', 'koEntry', 50);
  if (winRows < 50) { console.error(`SPIKES GOLDEN: too few WIN rows (${winRows} < 50)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `spikes golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: sideStart=${corpus.sideStart || 0} spikesDamage=${corpus.spikesDamage || 0} ` +
    `sideFail=${corpus.sideFail || 0} immuneEntry=${corpus.immuneEntry || 0} koEntry=${corpus.koEntry || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
