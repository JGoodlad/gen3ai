// gen_status_move_golden.js — Gen-3 STANDALONE STATUS-MOVE differential harness.
//
// Extends harness/gen_secondary_golden.js (the per-decision STATE+STATUS+BOOSTS+
// CONFUSION+SEED+winner full-battle differential) to the NEW execution path this step
// adds: the STANDALONE STATUS-INFLICTING moves whose category is Status / bp 0 —
//   par: Thunder Wave / Stun Spore / Glare
//   psn: Poison Powder / Poison Gas
//   tox: Toxic
//   brn: Will-O-Wisp
//   slp: Spore / Sleep Powder / Hypnosis / Sing / Lovely Kiss / Grass Whistle
//
// THE DRAW MODEL (verified bit-for-bit vs the omniscient sim, `data/mods/gen3/
// scripts.ts::tryMoveHit`):
//   1. MOVE-TYPE IMMUNITY (DRAW-FREE) — only for the two `ignoreImmunity:false`
//      status moves, Thunder Wave (Electric→Ground immune) + Glare (Normal→Ghost
//      immune); every other status move IGNORES type immunity.
//   2. ACCURACY `randomChance(acc,100)` — ALWAYS drawn (unless never_miss), even on a
//      type-immune target (gen3 draws accuracy THEN reports `-immune`).
//   3. APPLY via trySetStatus → setStatus (DRAW-FREE gates: already-statused / status-
//      type immunity / ability immunity [Insomnia/Vital Spirit slp] / SLEEP CLAUSE
//      MOD), then the status onStart: SLEEP draws ONE `random(2,6)` duration (1-4
//      turns), TOXIC starts at stage 1 (NO draw). A status move has NO crit / damage /
//      secondary and NEVER fires the in-`tryMoveHit` `eachEvent('Update')` shuffle.
//
// THE PROOF (the CRUX): drive the OMNISCIENT in-process BattleStream (no server) over
// CONSTRUCTED scenarios that each ISOLATE one draw/branch, capturing the running PRNG
// seed BEFORE the first decision (`initSeed`) and AFTER each DECISION BOUNDARY, plus
// each active's species/hp/maxhp/fainted/STATUS(+the Sleep/Toxic inner counter via
// statusState.time/.stage)/boosts/confusion + pokemon_left + first mover + winner. The
// Rust test seeds a BattleState at the init seed and runs `run_full_battle` WITHOUT
// re-seeding — so the post-decision seed must match the sim's at EVERY boundary,
// INCLUDING the new accuracy draw and the sleep random(2,6). An EXACT cross-decision
// seed match to game-end + the per-decision STATUS (incl. the sleep counter + the
// Toxic stage ramp) + the final winner is the draw-ORDER+COUNT proof.
//
// FAIL-LOUD: each scenario declares the BRANCH it must realize (e.g. TWave LANDS para,
// TWave→Ground IMMUNE, Sleep Clause BLOCK); generation aborts if the sim run did NOT
// realize that branch (so a mis-built scenario can never silently pass an empty path).
//
// Output: tests/vectors/status_move_golden.txt (same TAB format as secondary_golden).
//
// Run:  node src/rust_sim/harness/gen_status_move_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/status_move_golden.txt');
// gen3ou (NOT gen3customgame) so the SLEEP CLAUSE MOD is active (it rides the
// `Standard` ruleset). The Rust test passes `format_id: "gen3ou"` → `sleep_clause`
// ON. (The secondary/fullbattle/e2e goldens use gen3customgame = no clause.)
const FORMAT = 'gen3ou';
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
  let x = 0x51ed2c19 >>> 0;
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

// The inner status counter: for `tox` the badly-poison STAGE (statusState.stage); for
// `slp` the remaining-turns counter (statusState.time, set by onStart's random(2,6),
// decremented in onBeforeMove). Matches the Rust Status::Toxic(stage) / Sleep(n).
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

function snap(side) {
  const a = side.active[0];
  if (!a) return { species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', stage: 0, left: side.pokemonLeft, boosts: [0, 0, 0, 0, 0], confusion: 0 };
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status, stage, left: side.pokemonLeft,
    boosts: boostsOf(a), confusion: confusionOf(a),
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

// Scan the protocol log between two decision points for the RNG / branch OUTCOMES the
// status-move path produces (so the differential AND the per-scenario branch floor can
// assert WHICH branch fired). Returns per-side + per-battle flags.
function outcomesSince(log, fromIdx) {
  const out = {
    p1: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    p2: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    statusLanded: false,   // a -status line (any status applied this decision)
    immune: false,         // a -immune line (type/ability/clause immunity)
    miss: false,           // a -miss line (accuracy miss)
    sleepClause: false,    // the Sleep Clause Mod activation message
    slpLanded: false, parLanded: false, brnLanded: false, psnLanded: false, toxLanded: false,
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
    if (tag === '-status') {
      out.statusLanded = true;
      const s = p[3] || '';
      if (s === 'slp') out.slpLanded = true;
      if (s === 'par') out.parLanded = true;
      if (s === 'brn') out.brnLanded = true;
      if (s === 'psn') out.psnLanded = true;
      if (s === 'tox') out.toxLanded = true;
    }
    if (tag === '-immune') out.immune = true;
    if (tag === '-miss') out.miss = true;
    if (tag === '-message' && (p[2] || '').includes('Sleep Clause')) out.sleepClause = true;
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
    // STALL GUARD: if a choice was REJECTED (the request state + seed + log did not
    // advance), the sim ignored it (e.g. an illegal switch) and the loop would record
    // a STUCK DUPLICATE decision forever. Fail loud at generation rather than emit a
    // poisoned golden (a Rust replay can never match a phantom decision).
    if (seedAfter === seedBefore && log.length === logLenBefore && battle.requestState === reqState) {
      throw new Error(`STALL: choice ${JSON.stringify(choices)} did not advance the battle ` +
        `(scenario ${sc.id}, decision ${decisionNo}) — the sim rejected it. Fix the script.`);
    }
    const outcomes = outcomesSince(log, logLenBefore);
    // Aggregate the branch flags this run realized (for the per-scenario floor).
    for (const k of ['statusLanded', 'immune', 'miss', 'sleepClause', 'slpLanded', 'parLanded', 'brnLanded', 'psnLanded', 'toxLanded']) {
      if (outcomes[k]) rec.branchSeen[k] = true;
    }
    for (const side of ['p1', 'p2']) {
      for (const k of ['wake', 'fullpara']) if (outcomes[side][k]) rec.branchSeen[`${side}_${k}`] = true;
    }
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
// Each is { id, p1[], p2[], makeScript, require } where `require` is the set of
// branch flags the SIM run must realize at least once (across the seed sweep) — else
// generation FAILS LOUD (the scenario didn't exercise its intended draw/branch).

function scenarios() {
  const S = [];

  const repeat = (entry, onForce) => () => (decisionNo, battle, reqState, force) => {
    if (reqState === 'switch') {
      const c = { p1: null, p2: null };
      if (force[0]) c.p1 = onForce(0, battle);
      if (force[1]) c.p2 = onForce(1, battle);
      return c;
    }
    return entry;
  };
  const fromPlan = (plan, onForce) => () => {
    let i = 0;
    return (decisionNo, battle, reqState, force) => {
      if (reqState === 'switch') {
        const c = { p1: null, p2: null };
        if (force[0]) c.p1 = onForce(0, battle);
        if (force[1]) c.p2 = onForce(1, battle);
        return c;
      }
      const entry = plan[Math.min(i, plan.length - 1)];
      i++;
      return entry;
    };
  };
  const firstLiveBench = (side, battle) => {
    const s = battle.sides[side];
    for (let k = 0; k < s.pokemon.length; k++) {
      const p = s.pokemon[k];
      if (p !== s.active[0] && !p.fainted) return `switch ${k + 1}`;
    }
    return 'pass';
  };

  // --- (1) THUNDER WAVE LANDS (par, 100 acc): Zapdos (no Ground) paralyzes a bulky
  //   Snorlax, then attacks it down with Thunderbolt. Once paralyzed, Snorlax draws
  //   randomChance(1,4) full-para in onBeforeMove. p2 Snorlax Body Slams back (its own
  //   30% para secondary may also fire — but the FOCUS is TWave's par). Grind to a win.
  //   REQUIRES: a par landed (the TWave par) + a full-para onBeforeMove fired. ---
  S.push({
    id: 'thunderwave_para_lands',
    p1: [mon('Zapdos', ['thunderwave', 'thunderbolt'], { nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['earthquake', 'crunch'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan([{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }], firstLiveBench),
    require: ['parLanded', 'p2_fullpara'],
  });

  // --- (2) THUNDER WAVE → GROUND IMMUNE (the move-type immunity, the new wrinkle):
  //   Zapdos TWave into Swampert (Water/GROUND) — Electric is 0× vs Ground → NO
  //   paralysis (accuracy is still drawn, then `-immune`). Then Zapdos Thunderbolts
  //   (also Electric → STILL damages a Water/Ground? no — Tbolt is 0× vs Ground too!
  //   use a SECOND damaging move). p1 also carries Ice Beam (SE on Swampert) to win.
  //   Swampert Earthquakes back (SE on Zapdos... no, Zapdos is Flying → immune; use
  //   Surf). REQUIRES: a -immune (the TWave Ground immunity) and NO par ever lands. ---
  S.push({
    id: 'thunderwave_ground_immune',
    p1: [mon('Zapdos', ['thunderwave', 'icebeam'], { nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Swampert', ['surf', 'icebeam'], { nature: 'Bold', evs: { hp: 252, def: 252 } })],
    makeScript: fromPlan([{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }], firstLiveBench),
    require: ['immune'],
    forbid: ['parLanded'],
  });

  // --- (3) TOXIC RAMP (tox, 85 acc): Gengar Toxics a bulky Snorlax → the badly-poison
  //   STAGE ramps each turn (maxhp/16 * stage, capped 15). Gengar then chips with Shadow
  //   Ball over MANY turns so the stage climbs past 2. The Toxic STAGE
  //   (statusState.stage) must match the Rust Toxic(stage) ramp. Snorlax attacks with
  //   BODY SLAM (Normal → Gengar is GHOST-IMMUNE) so Gengar is never hit and the game
  //   goes the distance (Toxic + Shadow Ball whittle Snorlax; the tox DoT bounds it to a
  //   win). REQUIRES: a tox landed. ---
  S.push({
    id: 'toxic_ramp',
    // Modest SpA Gengar: it walls Snorlax's Body Slam via the GHOST Normal-immunity (so
    // it never needs Def) AND its Shadow Ball does real damage, so the battle is BOUNDED
    // to a win even on a Toxic MISS (no weak-attacker stall that hits the turn cap).
    p1: [mon('Gengar', ['toxic', 'shadowball'], { item: 'Leftovers', nature: 'Modest', evs: { hp: 4, spa: 252, spe: 252 } })],
    // A lightly-invested Misdreavus (Ghost): Body Slam is NORMAL → Gengar Ghost-immune,
    // AND Shadow Ball is SUPER-EFFECTIVE on Ghost, so Gengar KOs it in a FEW turns (well
    // within Shadow Ball's 15 PP) while the tox DoT ramps the stage past 2. Bounded → win.
    p2: [mon('Misdreavus', ['bodyslam', 'shadowball'], { nature: 'Calm', evs: { hp: 252, spd: 252 } })],
    // turn 1 Toxic; then Shadow Ball forever. p2 uses Body Slam (move 1, NORMAL → Gengar
    // GHOST-immune, so Gengar is untouched and the tox stage ramps a few turns to a win).
    makeScript: fromPlan([{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }], firstLiveBench),
    require: ['toxLanded'],
  });

  // --- (4) TOXIC → STEEL/POISON IMMUNE: Gengar Toxic into Skarmory (Steel/Flying) —
  //   Steel is psn-immune (Toxic checks the psn immunity in trySetStatus) → NO tox
  //   (accuracy drawn, then `-immune`). Gengar wins with Thunderbolt (SE on Skarmory).
  //   Skarmory Drill Pecks back. REQUIRES: a -immune (the Steel psn-immunity) + NO tox
  //   ever lands. ---
  S.push({
    id: 'toxic_steel_immune',
    p1: [mon('Gengar', ['toxic', 'thunderbolt'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Skarmory', ['drillpeck', 'steelwing'], { nature: 'Impish', evs: { hp: 252, def: 252 } })],
    makeScript: fromPlan([{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }], firstLiveBench),
    require: ['immune'],
    forbid: ['toxLanded'],
  });

  // --- (5) WILL-O-WISP LANDS + MISS (brn, 75 acc): Gengar Will-O-Wisps a bulky
  //   non-Fire Snorlax. At 75% accuracy the move MISSES on some seeds (accuracy-only,
  //   no brn) and LANDS on others (brn → the residual burn DoT chips, already-modeled).
  //   Gengar then Shadow Balls. NO Leftovers → bounded. Snorlax Earthquakes back.
  //   REQUIRES: a brn landed AND a -miss (both branches across the seed sweep). ---
  S.push({
    id: 'willowisp_burn_and_miss',
    p1: [mon('Gengar', ['willowisp', 'shadowball'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['earthquake', 'crunch'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan([{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }], firstLiveBench),
    require: ['brnLanded', 'miss'],
  });

  // --- (6) SPORE → SLEEP + the random(2,6) DURATION + the onBeforeMove WAKE (slp, 100
  //   acc): Breloom Spores a bulky Snorlax → slp (the random(2,6) duration draw — a
  //   SEED-bearing draw; a missing draw desyncs seedAfter). The asleep Snorlax then
  //   draws DRAW-FREE counter decrements each turn and WAKES after the counter runs out
  //   (the onBeforeMove wake). Breloom Sky Uppercuts to whittle. NO Leftovers →
  //   bounded. REQUIRES: a slp landed + a wake (slp cured by onBeforeMove). ---
  S.push({
    id: 'spore_sleep_wake',
    p1: [mon('Breloom', ['spore', 'skyuppercut'], { nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['earthquake', 'crunch'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: repeat({ p1: 'move 2', p2: 'move 1' }, firstLiveBench),
    // turn 1 we Spore (move 1), then attack (move 2) — handled by a custom plan below.
    require: ['slpLanded', 'p2_wake'],
  });
  // Override the script: Spore once, then Sky Uppercut forever.
  S[S.length - 1].makeScript = (() => () => {
    let i = 0;
    return (decisionNo, battle, reqState, force) => {
      if (reqState === 'switch') { const c = { p1: null, p2: null }; if (force[0]) c.p1 = firstLiveBench(0, battle); if (force[1]) c.p2 = firstLiveBench(1, battle); return c; }
      const e = i === 0 ? { p1: 'move 1', p2: 'move 1' } : { p1: 'move 2', p2: 'move 1' };
      i++;
      return e;
    };
  })();

  // --- (7) SLEEP POWDER MISS (slp, 75 acc): Venusaur Sleep Powders a bulky Snorlax —
  //   at 75% accuracy it MISSES on some seeds (accuracy-only, no slp, NO random(2,6)).
  //   Venusaur then Sludge Bombs. NO Leftovers → bounded. REQUIRES: a -miss + a slp
  //   landed (so both the miss-no-draw and the land-with-random(2,6) draw counts are
  //   covered in ONE scenario). ---
  S.push({
    id: 'sleeppowder_miss',
    p1: [mon('Venusaur', ['sleeppowder', 'sludgebomb'], { nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['earthquake', 'crunch'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: (() => () => {
      let i = 0;
      return (decisionNo, battle, reqState, force) => {
        if (reqState === 'switch') { const c = { p1: null, p2: null }; if (force[0]) c.p1 = firstLiveBench(0, battle); if (force[1]) c.p2 = firstLiveBench(1, battle); return c; }
        const e = i % 3 === 0 ? { p1: 'move 1', p2: 'move 1' } : { p1: 'move 2', p2: 'move 1' };
        i++;
        return e;
      };
    })(),
    require: ['miss', 'slpLanded'],
  });

  // --- (8) STUN SPORE PARA (par, 75 acc): Venusaur Stun Spores a bulky Snorlax → par
  //   (Grass-type status move, ignoreImmunity default true so NO type block). Venusaur
  //   Sludge Bombs. The paralyzed Snorlax draws full-para in onBeforeMove + its ×0.25
  //   speed. NO Leftovers → bounded. REQUIRES: a par landed + a full-para. ---
  S.push({
    id: 'stunspore_para',
    p1: [mon('Venusaur', ['stunspore', 'sludgebomb'], { nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['earthquake', 'crunch'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: (() => () => {
      let i = 0;
      return (decisionNo, battle, reqState, force) => {
        if (reqState === 'switch') { const c = { p1: null, p2: null }; if (force[0]) c.p1 = firstLiveBench(0, battle); if (force[1]) c.p2 = firstLiveBench(1, battle); return c; }
        const e = i === 0 ? { p1: 'move 1', p2: 'move 1' } : { p1: 'move 2', p2: 'move 1' };
        i++;
        return e;
      };
    })(),
    require: ['parLanded', 'p2_fullpara'],
  });

  // --- (9) SLEEP CLAUSE BLOCK (the gen3ou Sleep Clause Mod): Breloom Spores p2's
  //   Snorlax (slp #1, the random(2,6) draws), p2 then switches to Blissey, and Breloom
  //   Spores AGAIN — the 2nd foe-sleep is BLOCKED by the clause (accuracy drawn, the
  //   move HITS, then setStatus fails at the SetStatus event → NO random(2,6), Blissey
  //   stays statusless). Breloom then Sky Uppercuts to a win. REQUIRES: a slp landed
  //   (the 1st) + the Sleep Clause activation. ---
  S.push({
    id: 'sleep_clause_block',
    p1: [mon('Breloom', ['spore', 'skyuppercut'], { nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['earthquake', 'crunch'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Blissey', ['icebeam', 'thunderbolt'], { nature: 'Calm', evs: { hp: 252, spa: 252 } })],
    makeScript: (() => () => {
      let i = 0;
      return (decisionNo, battle, reqState, force) => {
        if (reqState === 'switch') { const c = { p1: null, p2: null }; if (force[0]) c.p1 = firstLiveBench(0, battle); if (force[1]) c.p2 = firstLiveBench(1, battle); return c; }
        // turn 1: Spore Snorlax. turn 2: p2 switches Blissey, p1 Spore (BLOCKED). then Sky Uppercut.
        let e;
        if (i === 0) e = { p1: 'move 1', p2: 'move 1' };
        else if (i === 1) e = { p1: 'move 1', p2: 'switch 2' };
        else e = { p1: 'move 2', p2: 'move 1' };
        i++;
        return e;
      };
    })(),
    require: ['slpLanded', 'sleepClause'],
  });

  // --- (10) STATUS-MOVE-INTO-A-REAL-BATTLE (a longer mixed game to game-end): p1's
  //   lead opens with a status move, then VOLUNTARILY PIVOTS to a heavy attacker that
  //   grinds the foe out — so the status path interleaves with a switch + the full
  //   move/residual/faint machinery all the way to a win. p1 (Gengar Will-O-Wisp +
  //   Snorlax) vs p2 (a frail Jolteon → a bulky Suicune). Turn 1 Will-O-Wisp Jolteon
  //   (brn — the burn DoT then chips), turn 2 pivot Snorlax, then Body Slam to a win.
  //   The opening uses a SLOWER, bulky Gengar so it survives turn 1 to pivot cleanly
  //   (avoiding a turn-1 faint that would strand the opening). REQUIRES: a brn landed +
  //   a win run. ---
  const openStatusPivotGrind = () => () => {
    let movesSeen = 0;
    return (decisionNo, battle, reqState, force) => {
      if (reqState === 'switch') {
        const c = { p1: null, p2: null };
        if (force[0]) c.p1 = firstLiveBench(0, battle);
        if (force[1]) c.p2 = firstLiveBench(1, battle);
        return c;
      }
      // move 0: Will-O-Wisp; move 1: VOLUNTARY pivot to Snorlax; move 2+: Earthquake
      // (a SECONDARY-FREE attacker, so the grind doesn't re-enter the secondary draw
      // path — this scenario isolates the STANDALONE status move + a switch).
      let e;
      if (movesSeen === 0) e = { p1: 'move 1', p2: 'move 1' };
      else if (movesSeen === 1) e = { p1: 'switch 2', p2: 'move 1' };
      else e = { p1: 'move 2', p2: 'move 1' };
      movesSeen++;
      return e;
    };
  };
  S.push({
    id: 'status_into_real_battle',
    p1: [mon('Gengar', ['willowisp', 'shadowball'], { item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } }),
         mon('Snorlax', ['earthquake', 'crunch'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Jolteon', ['shadowball', 'irontail'], { nature: 'Timid', evs: { hp: 4, spa: 252, spe: 252 } }),
         mon('Suicune', ['surf', 'icebeam'], { nature: 'Bold', evs: { hp: 252, def: 252 } })],
    makeScript: openStatusPivotGrind(),
    require: ['brnLanded'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# status_move_golden.txt — Gen-3 STANDALONE STATUS-MOVE full-battle golden.');
  lines.push('# Per-decision-boundary STATE(+status+sleep/toxic counter)+SEED differential to GAME-END.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1(fullpara wake thaw selfhit flinch) p2(...) statusLanded');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0;
  // Branch floors across the whole corpus (each must realize ≥1).
  const corpus = {};
  // Per-scenario realization tracking (for the fail-loud `require`/`forbid`).
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
          s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status, s.stage, s.left,
          s.boosts[0], s.boosts[1], s.boosts[2], s.boosts[3], s.boosts[4], s.confusion,
        ].join('\t');
        const oc = (o) => [o.fullpara ? 1 : 0, o.wake ? 1 : 0, o.thaw ? 1 : 0, o.selfhit ? 1 : 0, o.flinch ? 1 : 0].join('\t');
        lines.push([
          'DEC', sc.id, seedStr, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
          sp(d.p1), sp(d.p2), d.firstMover,
          oc(d.outcomes.p1), oc(d.outcomes.p2), d.outcomes.statusLanded ? 1 : 0,
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

    // FAIL-LOUD: the scenario must realize its declared branches (and forbid the rest).
    for (const need of (sc.require || [])) {
      if (!scenSeen[sc.id][need]) failures.push(`${sc.id}: REQUIRED branch ${need} never realized across the seed sweep`);
    }
    for (const bad of (sc.forbid || [])) {
      if (scenSeen[sc.id][bad]) failures.push(`${sc.id}: FORBIDDEN branch ${bad} realized (the scenario isolation is broken)`);
    }
  }

  if (failures.length) {
    console.error('STATUS-MOVE GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  // Corpus floors: every status branch must realize SOMEWHERE.
  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`STATUS GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('par-landed runs', 'parLanded', 5);
  need('tox-landed runs', 'toxLanded', 5);
  need('brn-landed runs', 'brnLanded', 5);
  need('slp-landed runs', 'slpLanded', 5);
  need('immune runs', 'immune', 5);
  need('miss runs', 'miss', 5);
  need('Sleep-Clause runs', 'sleepClause', 5);
  if (winRows < 50) { console.error(`STATUS GOLDEN: too few WIN rows (${winRows} < 50)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `status-move golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: par=${corpus.parLanded || 0} tox=${corpus.toxLanded || 0} brn=${corpus.brnLanded || 0} ` +
    `slp=${corpus.slpLanded || 0} immune=${corpus.immune || 0} miss=${corpus.miss || 0} ` +
    `sleepClause=${corpus.sleepClause || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
