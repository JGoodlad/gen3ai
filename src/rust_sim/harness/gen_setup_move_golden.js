// gen_setup_move_golden.js — Gen-3 SELF-TARGETING SETUP / STAT-BOOST MOVE differential.
//
// Extends harness/gen_status_move_golden.js (the per-decision STATE+STATUS+BOOSTS+
// CONFUSION+SEED+winner full-battle differential) to the NEW execution path this step
// adds: the PURE SELF-BOOST setup moves (category Status / bp 0 / target self) whose
// ENTIRE effect is raising the USER'S stat stages —
//   Calm Mind (+1 SpA/+1 SpD), Dragon Dance (+1 Atk/+1 Spe), Swords Dance (+2 Atk),
//   Agility (+2 Spe), Bulk Up (+1 Atk/+1 Def), Amnesia (+2 SpD), Barrier/Acid Armor/
//   Iron Defense (+2 Def), Cosmic Power (+1 Def/+1 SpD), Tail Glow (+2 SpA),
//   Meditate/Sharpen/Howl (+1 Atk), Harden/Withdraw (+1 Def), Growth (+1 SpA).
// (Defense Curl / Minimize / Double Team [volatile or evasion], Belly Drum [HP cost],
//  Curse [type-conditional] are NOT modeled — they stay fail-loud in the engine.)
//
// THE DRAW MODEL (verified bit-for-bit vs the omniscient sim, `data/mods/gen3/
// scripts.ts::tryMoveHit` + battle-actions.ts → `this.boost(boosts, source)`):
//   1. ACCURACY — every modeled setup move is NEVER-MISS (`accuracy: true`) → NO
//      accuracy draw. (Defensive: draw `randomChance(acc,100)` iff NOT never_miss; the
//      modeled set is all never-miss, so this never draws.)
//   2. APPLY `boost()` on the USER, each (stat, stages) clamped to ±6. **DRAW-FREE** —
//      boost() consumes NO PRNG. Our OWN Clear Body / White Smoke do NOT block our own
//      self-boost (the onTryBoost immunity is for FOE drops). A boost into +6 is a no-op
//      but still "succeeds" (draws nothing).
//   3. `landed` is ALWAYS FALSE — a status `moveHit` returns `undefined`, so the
//      in-`tryMoveHit` `eachEvent('Update')` shuffle is SKIPPED.
//
// THE KEY INTERACTION (the real validation target): a +SPEED self-boost (Dragon Dance,
// Agility) raises `boosts.spe` MID-BATTLE, but Showdown re-establishes the CACHED
// `pokemon.speed` only at turn-start (commitChoices), residual-start, and switch-in —
// NOT live. So a Dragon Dance changes the boost STAGE immediately, but THIS turn's
// eachEvent tie-shuffles + the NEXT turn's action ORDER pick up the boosted speed only
// at the next re-cache point. The scenarios below STRESS this: a Dragon Dance / Agility
// that FLIPS the first-mover on the FOLLOWING turn (and is bit-exact in the seed because
// the cached-speed timing is exact). A wrong cached-speed model → a divergent first-mover
// AND/OR a divergent Fisher-Yates tie-shuffle draw COUNT → a seed desync the gate catches.
//
// THE PROOF (the CRUX): drive the OMNISCIENT in-process BattleStream (no server) over
// CONSTRUCTED scenarios that each ISOLATE one branch, capturing the running PRNG seed
// BEFORE the first decision (`initSeed`) and AFTER each DECISION BOUNDARY, plus each
// active's species/hp/maxhp/fainted/status + THE 5 BOOST STAGES + pokemon_left + first
// mover + winner. The Rust test seeds a BattleState at the init seed and runs
// `run_full_battle` WITHOUT re-seeding — so the post-decision seed must match the sim's
// at EVERY boundary, AND the per-decision boost stages + first-mover must match. An EXACT
// cross-decision seed+boost+first-mover match to game-end is the draw-ORDER+COUNT +
// cached-speed proof.
//
// FAIL-LOUD: each scenario declares the BRANCH it must realize (a boost applied, the +6
// cap reached, a first-mover FLIP from a +Spe setup); generation aborts if the sim run
// did NOT realize it (so a mis-built scenario can never silently pass an empty path).
//
// Output: tests/vectors/setup_move_golden.txt (same TAB format as status_move_golden).
//
// Run:  node src/rust_sim/harness/gen_setup_move_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/setup_move_golden.txt');
// gen3customgame (NOT gen3ou) — these setup scenarios inflict NO status, so the Sleep
// Clause / SetStatus shuffle never matters; gen3customgame matches the e2e capstone
// format the setup moves also feed. The Rust test passes `format_id: "gen3customgame"`.
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

// The inner status counter: `tox` stage / `slp` remaining-turns (none here, but kept
// for format parity with the status golden). Matches Rust Status::Toxic / Sleep.
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

// Scan the protocol log between two decision points for the branch OUTCOMES a setup move
// produces — a `-boost` line (a self-boost applied), the boost reaching the +6 cap
// (Showdown emits `-boost ... 0` when a boost is wholly capped — but we detect the cap
// via the recorded boost STATE, not the log; see the per-decision boost diff in main()).
function outcomesSince(log, fromIdx) {
  const out = {
    p1: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    p2: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    boosted: false,     // a -boost line (a self-boost applied this decision)
    capped: false,      // a -boost with amount 0 (a fully-capped, no-op-but-success boost)
    miss: false,        // a -miss line (should never fire for never-miss setup)
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
    if (tag === '-boost') {
      out.boosted = true;
      // A wholly-capped boost is logged with amount 0 (`|-boost|p1a: X|atk|0`).
      if ((p[4] || '').trim() === '0') out.capped = true;
    }
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

  let decisionNo = 0;
  let safety = 0;
  // Track the previous decision's first-mover so a +Spe-driven FLIP is recordable.
  let prevFirst = null;
  while (!stream.battle.ended && safety < 400) {
    safety++;
    const battle = stream.battle;
    const reqState = battle.requestState;
    if (reqState !== 'move' && reqState !== 'switch') { await tick(); continue; }
    const force = forceSwitchTable(battle);
    const seedBefore = battle.prng.getSeed();
    if (decisionNo === 0) rec.initSeed = seedBefore;

    // Capture the user's PRE-decision boost (so the cap-detector can tell a no-op).
    const preBoost = [boostsOf(battle.sides[0].active[0]), boostsOf(battle.sides[1].active[0])];

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

    // A first-mover FLIP vs the previous MOVE decision (the +Spe-driven re-order proof):
    // recorded as a corpus branch so a scenario can REQUIRE it realized at least once.
    if (reqState === 'move' && prevFirst && first !== 'none' && first !== prevFirst) {
      outcomes.firstMoverFlip = true;
    }
    if (reqState === 'move' && first !== 'none') prevFirst = first;

    // Detect a fully-capped no-op self-boost from the STATE (a boost that did not change
    // the user's stage despite the move being used) — more robust than the log amount.
    const postBoost = [boostsOf(battle.sides[0].active[0]), boostsOf(battle.sides[1].active[0])];
    for (const sideIdx of [0, 1]) {
      const used = encodeChoice(sideIdx === 0 ? choices.p1 : choices.p2).startsWith('m');
      if (used && JSON.stringify(preBoost[sideIdx]) === JSON.stringify(postBoost[sideIdx]) &&
          preBoost[sideIdx].some((v) => v === 6)) {
        outcomes.capped = true;
      }
    }

    for (const k of ['boosted', 'capped', 'miss', 'firstMoverFlip']) {
      if (outcomes[k]) rec.branchSeen[k] = true;
    }
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
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
// Each is { id, p1[], p2[], makeScript, require } where `require` is the set of branch
// flags the SIM run must realize at least once (across the seed sweep) — else generation
// FAILS LOUD (the scenario didn't exercise its intended branch).

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
  // A plan-driven script (one entry per MOVE decision; a forced switch picks the first
  // live bench). The plan's last entry repeats once exhausted.
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

  // --- (1) CALM MIND CLIMB TO +6 (the multi-turn climb + the cap): a bulky Suicune
  //   uses Calm Mind for 6+ turns vs a foe that can't break it, so SpA/SpD climb +1/+1
  //   each turn until BOTH hit +6 (the 7th Calm Mind is a no-op-but-success cap). p2 is
  //   a weak special attacker that can't dent a +SpD Suicune; Suicune then Surfs it down.
  //   REQUIRES: a boost applied + the +6 cap reached. ---
  S.push({
    id: 'calmmind_climb_cap',
    p1: [mon('Suicune', ['calmmind', 'surf'], { ability: 'Pressure', item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    // A frail special attacker whose Shadow Ball can't break a +SpD Suicune → the climb
    // is uninterrupted, then Surf KOs it. (Pressure / Levitate are provable no-ops /
    // modeled, so the engine's inert/Levitate handling matches the sim.)
    p2: [mon('Misdreavus', ['shadowball', 'thunderbolt'], { ability: 'Levitate', nature: 'Modest', evs: { hp: 4, spa: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // CM x7 (climbs +1/+1 each turn; 7th caps at +6)
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' }, // the cap (no-op success)
      { p1: 'move 2', p2: 'move 1' }, // Surf to the win
    ]),
    require: ['boosted', 'capped'],
  });

  // --- (2) SWORDS DANCE +2 ATK: a Flygon Swords Dances once (+2 Atk), then Earthquakes
  //   the foe MUCH harder (the boost STATE drives the damage roll, so the per-decision HP
  //   after the boosted EQ must match — a wrong boost apply diverges the HP). The foe
  //   attacks ONLY with Earthquake, which Flygon (LEVITATE, a MODELED ability) is IMMUNE
  //   to, so Flygon takes 0 and freely sweeps (no Thick-Club-style unmodeled item — plain
  //   Leftovers). REQUIRES: a boost applied. ---
  S.push({
    id: 'swordsdance_atk',
    // Flygon (Ground/Dragon, Levitate → Ground-immune) Adamant max-Atk — a clean SD
    // sweeper whose damage the engine fully prices (no Thick Club / item stat-mod gaps).
    p1: [mon('Flygon', ['swordsdance', 'earthquake'], { ability: 'Levitate', item: 'Leftovers', nature: 'Adamant', evs: { hp: 4, atk: 252, spe: 252 } })],
    // Donphan attacks ONLY with Earthquake → Flygon is Levitate-immune (takes 0), so the
    // boosted EQ sweep is uninterrupted and always a p1 win.
    p2: [mon('Donphan', ['earthquake'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Swords Dance (+2 Atk)
      { p1: 'move 2', p2: 'move 1' }, // boosted Earthquake
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['boosted'],
  });

  // --- (3) DRAGON DANCE → FIRST-MOVER FLIP (the cached-speed crux): an Adamant Salamence
  //   (236 Spe) is SLOWER than a max-Spe Timid Starmie (361 Spe). Turn 1 Salamence Dragon
  //   Dances (+1 Atk/+1 Spe, cached speed STALE → Starmie still moves first THIS turn);
  //   turn 2 Salamence's cached speed is +1 = 354, STILL < 361 (Starmie first again); turn
  //   3 the cached speed is +2 = 472 > 361 → the first-mover FLIPS to p1. The foe attacks
  //   ONLY with Earthquake (Salamence is FLYING → IMMUNE, so it takes 0 damage and never
  //   needs to race the foe's offense), so the climb-then-flip is clean and Salamence
  //   sweeps with boosted Dragon Claw. The seed must be bit-exact through the flip (the
  //   cached-speed timing + the per-turn eachEvent tie-shuffle draw count). REQUIRES: a
  //   boost applied + a first-mover flip. ---
  S.push({
    id: 'dragondance_speed_flip',
    // Adamant Salamence base 100 Spe, no Spe EVs (236) → slower than a max-Spe Starmie
    // (361) until +2 Dragon Dance (472). Intimidate is a MODELED ability (lead drop only).
    p1: [mon('Salamence', ['dragondance', 'dragonclaw'], { ability: 'Intimidate', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    // Max-Spe Timid Starmie outspeeds Salamence until +2; its ONLY move is Earthquake,
    // which Salamence (Flying) is IMMUNE to — so Salamence freely sets up and sweeps
    // (the battle always ends in a p1 win, no stall).
    p2: [mon('Starmie', ['earthquake'], { ability: 'Illuminate', nature: 'Timid', evs: { hp: 4, spa: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // DD #1 (cached stale → Starmie first this turn)
      { p1: 'move 1', p2: 'move 1' }, // DD #2 (cached +1 = 354 < 361 → Starmie first)
      { p1: 'move 2', p2: 'move 1' }, // boosted Dragon Claw — cached +2 = 472 → p1 FIRST (flip)
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['boosted', 'firstMoverFlip'],
  });

  // --- (4) AGILITY OUTSPEEDS A FASTER FOE NEXT TURN (the +2 Spe single jump): an Adamant
  //   Metagross (176 Spe) uses Agility (+2 Spe = 352). STILL the cached-speed timing: it
  //   does NOT outspeed THIS turn (cached 176 stale → the faster Heracross moves first on
  //   the Agility turn), but its cached speed is re-established to 352 at the NEXT turn-
  //   start, overtaking the Jolly Heracross (295) → the first-mover FLIPS to p1. The foe
  //   attacks ONLY with Earthquake (a non-STAB neutral hit a bulky Metagross survives);
  //   boosted Meteor Mash then grinds Heracross out. REQUIRES: a boost applied + a flip. ---
  S.push({
    id: 'agility_outspeed_flip',
    // Adamant 252-HP Metagross base 70 Spe (176) → slower than a Jolly Heracross (295)
    // until +2 Agility (352). Clear Body is MODELED (and does NOT block its own Agility).
    p1: [mon('Metagross', ['agility', 'meteormash'], { ability: 'Clear Body', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    // Jolly max-Spe Heracross (295) outspeeds an unboosted Metagross; its non-STAB
    // Earthquake (neutral on Steel/Psychic) can't break a 252-HP Metagross fast, so
    // Metagross survives to overtake and KO with boosted Meteor Mash.
    p2: [mon('Heracross', ['earthquake'], { ability: 'Guts', nature: 'Jolly', evs: { hp: 4, atk: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Agility (+2 Spe; cached stale → Heracross first)
      { p1: 'move 2', p2: 'move 1' }, // boosted Meteor Mash — Metagross now FIRST (the flip)
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['boosted', 'firstMoverFlip'],
  });

  // --- (5) A BOOST INTO THE +6 CAP (no-op success, isolated): an Aggron with +2 Atk
  //   Swords Dance — after 3 Swords Dances it's at +6; the 4th is a wholly-capped no-op
  //   that STILL "succeeds" (draws nothing, seed unchanged). A frail foe can't break the
  //   bulky Aggron, so the cap is reached uninterrupted; Aggron then Earthquakes to the
  //   win. Rock Head is a MODELED no-op (no recoil move here). REQUIRES: the +6 cap. ---
  S.push({
    id: 'swordsdance_cap',
    p1: [mon('Aggron', ['swordsdance', 'earthquake'], { ability: 'Rock Head', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    // A frail Houndoom (Dark/Fire, FIRE → 2× weak to Earthquake) whose special attack
    // can't dent a 252-HP Aggron → the cap is reached uninterrupted, then the boosted EQ
    // OHKOs. Early Bird is a provable no-op here (no sleep); NOT Levitate (Aggron sweeps
    // with Ground, so the foe must NOT be Ground-immune).
    p2: [mon('Houndoom', ['shadowball', 'thunderbolt'], { ability: 'Early Bird', nature: 'Modest', evs: { hp: 4, spa: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // SD -> +2
      { p1: 'move 1', p2: 'move 1' }, // SD -> +4
      { p1: 'move 1', p2: 'move 1' }, // SD -> +6
      { p1: 'move 1', p2: 'move 1' }, // SD -> +6 (CAP, no-op success)
      { p1: 'move 2', p2: 'move 1' }, // Earthquake to the win
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['boosted', 'capped'],
  });

  // --- (6) CALM MIND + BULK UP vs A REAL BATTLE (setup interleaved with a switch + the
  //   full move/residual/faint machinery to game-end): p1's Suicune Calm Minds twice
  //   (+SpA/+SpD climb) behind a frail foe, then VOLUNTARILY PIVOTS to a Snorlax that
  //   Bulk Ups twice (+Atk/+Def climb) and Body Slams the FRAIL 2-mon team out. The setup
  //   path interleaves with a switch + residuals + a faint all the way to a win; both foes
  //   are FRAIL (high offense, low bulk) so the boosted sweep ends in a clean p1 win in a
  //   few turns — well before any move's PP runs out. (Thick Fat is MODELED; Pressure /
  //   Inner Focus are provable no-ops here.) REQUIRES: a boost applied + a win run. ---
  S.push({
    id: 'setup_into_real_battle',
    p1: [mon('Suicune', ['calmmind', 'surf'], { ability: 'Pressure', item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } }),
         mon('Snorlax', ['bulkup', 'bodyslam'], { ability: 'Thick Fat', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    // Two FRAIL special attackers (no defensive EVs, no Leftovers) the boosted Snorlax
    // OHKOs/2HKOs — so the grind is short and ALWAYS terminates in a p1 win. Houndoom's
    // Crunch (Dark) and Glalie's Ice Beam can't break the bulky setup mons.
    p2: [mon('Houndoom', ['crunch', 'sludgebomb'], { ability: 'Early Bird', nature: 'Modest', evs: { hp: 4, spa: 252, spe: 252 } }),
         mon('Glalie', ['icebeam', 'crunch'], { ability: 'Inner Focus', nature: 'Modest', evs: { hp: 4, spa: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Suicune Calm Mind (+1 SpA/+1 SpD)
      { p1: 'move 1', p2: 'move 1' }, // Calm Mind again (+2/+2)
      { p1: 'switch 2', p2: 'move 1' }, // VOLUNTARY pivot to Snorlax (the switch interleave)
      { p1: 'move 1', p2: 'move 1' }, // Snorlax Bulk Up (+1 Atk/+1 Def)
      { p1: 'move 1', p2: 'move 1' }, // Bulk Up again (+2/+2)
      { p1: 'move 2', p2: 'move 1' }, // boosted Body Slam — sweeps the frail team out
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['boosted'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# setup_move_golden.txt — Gen-3 SELF-BOOST SETUP-MOVE full-battle golden.');
  lines.push('# Per-decision-boundary STATE(+status+counter)+BOOSTS+SEED+first-mover differential to GAME-END.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1(fullpara wake thaw selfhit flinch) p2(...) boosted');
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
          s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status, s.stage, s.left,
          s.boosts[0], s.boosts[1], s.boosts[2], s.boosts[3], s.boosts[4], s.confusion,
        ].join('\t');
        const oc = (o) => [o.fullpara ? 1 : 0, o.wake ? 1 : 0, o.thaw ? 1 : 0, o.selfhit ? 1 : 0, o.flinch ? 1 : 0].join('\t');
        lines.push([
          'DEC', sc.id, seedStr, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
          sp(d.p1), sp(d.p2), d.firstMover,
          oc(d.outcomes.p1), oc(d.outcomes.p2), d.outcomes.boosted ? 1 : 0,
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

    // FAIL-LOUD: the scenario must realize its declared branches.
    for (const need of (sc.require || [])) {
      if (!scenSeen[sc.id][need]) failures.push(`${sc.id}: REQUIRED branch ${need} never realized across the seed sweep`);
    }
    for (const bad of (sc.forbid || [])) {
      if (scenSeen[sc.id][bad]) failures.push(`${sc.id}: FORBIDDEN branch ${bad} realized (the scenario isolation is broken)`);
    }
  }

  if (failures.length) {
    console.error('SETUP-MOVE GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  // Corpus floors: every setup branch must realize SOMEWHERE.
  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`SETUP GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('boost-applied runs', 'boosted', 5);
  need('+6 cap runs', 'capped', 5);
  need('first-mover-flip runs', 'firstMoverFlip', 5);
  // NOTE: we do NOT assert `corpus.miss == 0`. A `-miss` line CAN appear in these battles
  // (a FOE's sub-100% move missing, e.g. Steel Wing 90), but a NEVER-MISS setup move is
  // never the misser — proven instead by the per-decision SEED parity (a setup move that
  // wrongly drew accuracy would desync the Rust replay's seed). The `miss` flag here is
  // not a setup-move outcome, so a non-zero count is benign.
  if (winRows < 50) { console.error(`SETUP GOLDEN: too few WIN rows (${winRows} < 50)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `setup-move golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: boosted=${corpus.boosted || 0} capped=${corpus.capped || 0} ` +
    `firstMoverFlip=${corpus.firstMoverFlip || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
