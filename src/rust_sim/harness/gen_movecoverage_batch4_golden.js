// gen_movecoverage_batch3_golden.js — Gen-3 MOVE-COVERAGE BATCH 3 differential golden
// (`gen3_move_coverage_batch3_v1`): the THREE STATEFUL, DRAW-FREE move classes —
// CURSE / WISH / BATON PASS. Curse + Baton Pass route through category-Status; Wish is a
// slot-keyed delayed heal.
//
// Extends the fullbattle/secondary TAB format with per-decision NEW VOLATILE columns:
//   - curse-flag per side (is the active mon cursed)
//   - wish-pending duration per side (the slot condition's remaining turns; 0 = none)
//   - sub-hp per side (the Substitute decoy's HP; 0 = no sub)
// on top of the per-decision STATE(+status+boosts+HP)+SEED+first-mover full-battle
// differential to GAME-END.
//
// THE THREE CLASSES (probe-settled by probe_batch3_{curse,wish,batonpass}.js):
//   CURSE       — a type-conditional move. NON-GHOST user → a self-boost {atk:+1, def:+1,
//                 spe:-1} (line order -Spe, +Atk, +Def; the -Spe can flip the first mover
//                 NEXT turn via the stale cached speed). Its `move.self={boosts}` rides the
//                 gen3 `selfDrops` path, which DRAWS ONE `random(100)` (discarded) — so the
//                 non-ghost curse is NOT draw-free (like Overheat's self-drop). GHOST user → pays
//                 floor(maxhp/2) HP + lays the `curse` volatile on the FOE (residual chip
//                 floor(maxhp/4)/turn at order 10 subOrder 8). Re-curse fails ([still]+-fail);
//                 curse-into-a-sub does nothing; a Ghost target is NOT immune.
//   WISH        — a slot-keyed DELAYED heal: heal floor(maxhp/2) at the END of the turn AFTER
//                 cast (duration 2). Slot-keyed (survives switch/faint/phaze). Residual ORDER
//                 7 (BEFORE the sand chip + all order-10 handlers); two Wishes at equal speed
//                 tie-shuffle. Double-Wish FAILS ([still], draw-free). Heal-at-full is silent.
//   BATON PASS  — a self-switch that PASSES the outgoing mon's boosts + copyable volatiles
//                 (substitute / leech-seed / confusion / curse) to the entrant. No eligible
//                 bench → FAILS ([still]+-fail, draw-free). The entrant's |switch| carries
//                 [from] Baton Pass.
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios that each ISOLATE a class, capturing initSeed + per-decision seedAfter, each
// active's species/hp/maxhp/fainted/status + boosts + confusion + pokemon_left + CURSE +
// WISH-PENDING + SUB-HP + first mover + winner. The Rust test seeds a BattleState at initSeed
// and runs `run_full_battle` WITHOUT re-seeding — so the post-decision seed must match at
// EVERY boundary (a wrong draw model → SEED desync), AND the cursed flag, the wish duration,
// the sub HP, and the passed boosts must match.
//
// Output: tests/vectors/movecoverage_batch3_golden.txt
//
// Run:  node src/rust_sim/harness/gen_movecoverage_batch3_golden.js

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/movecoverage_batch4_golden.txt');
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
  let x = 0x2f9c11ad >>> 0;
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

function curseOf(a) {
  return a && a.volatiles && a.volatiles['curse'] ? 1 : 0;
}

function subHpOf(a) {
  return a && a.volatiles && a.volatiles['substitute'] ? (a.volatiles['substitute'].hp | 0) : 0;
}

function wishOf(side) {
  const sc = side.slotConditions && side.slotConditions[0];
  const w = sc && sc.wish;
  return w ? (w.duration | 0) : 0;
}

function snap(side) {
  const a = side.active[0];
  const wish = wishOf(side);
  if (!a) {
    return {
      species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', stage: 0, left: side.pokemonLeft,
      boosts: [0, 0, 0, 0, 0], confusion: 0, curse: 0, wish, subHp: 0,
    };
  }
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status, stage, left: side.pokemonLeft,
    boosts: boostsOf(a), confusion: confusionOf(a),
    curse: curseOf(a), wish, subHp: subHpOf(a),
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

// Scan the protocol log between two decision points for the BATCH-4 branch flags
// (`gen3_move_coverage_batch4_v1`: FOCUS PUNCH + PURSUIT).
function outcomesSince(log, fromIdx) {
  const out = {
    fpSingleturn: false, // `-singleturn|…|move: Focus Punch` (the beforeTurnMove laid the volatile)
    fpLand: false, // a Focus Punch that HIT (`|move|…|Focus Punch|<target>` — not [still])
    fpCant: false, // `|cant|…|Focus Punch|Focus Punch` (lostFocus cancel)
    pursuitActivate: false, // `-activate|…|move: Pursuit` (the switch-interrupt fired)
    pursuitMove: false, // a Pursuit that RAN (`|move|…|Pursuit|…`) — normal OR interrupt
  };
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    if (tag === '-singleturn' && (p[3] || '') === 'move: Focus Punch') out.fpSingleturn = true;
    if (tag === 'cant' && (p[3] || '') === 'Focus Punch') out.fpCant = true;
    if (tag === 'move' && (p[3] || '') === 'Focus Punch') {
      // A landed FP has a target field + no [still]; a cancelled FP is `…|Focus Punch||[still]`.
      const still = (p[5] || '') === '[still]';
      if (!still && (p[4] || '') !== '') out.fpLand = true;
    }
    if (tag === '-activate' && (p[3] || '') === 'move: Pursuit') out.pursuitActivate = true;
    if (tag === 'move' && (p[3] || '') === 'Pursuit') out.pursuitMove = true;
  }
  return out;
}

function firstLiveBench(side, battle) {
  const s = battle.sides[side];
  for (let k = 0; k < s.pokemon.length; k++) {
    const p = s.pokemon[k];
    if (p !== s.active[0] && !p.fainted) return `switch ${k + 1}`;
  }
  return 'pass';
}

function legalMove(side, battle, want) {
  const req = battle.sides[side].activeRequest;
  const moves = req && req.active && req.active[0] ? req.active[0].moves : null;
  if (!moves) return 'move 1';
  const usable = [];
  for (let k = 0; k < moves.length; k++) if (!moves[k].disabled) usable.push(k + 1);
  if (usable.length === 0) return 'move 1';
  return `move ${usable.includes(want) ? want : usable[0]}`;
}

function intentDriver(intent) {
  return (decisionNo, battle, reqState, force) => {
    if (reqState === 'switch') {
      const c = { p1: null, p2: null };
      // A forced switch: for a BATON PASS the intent may name a target; else first live bench.
      const r = intent(decisionNo, battle) || {};
      if (force[0]) c.p1 = r.p1Switch ? `switch ${r.p1Switch}` : firstLiveBench(0, battle);
      if (force[1]) c.p2 = r.p2Switch ? `switch ${r.p2Switch}` : firstLiveBench(1, battle);
      return c;
    }
    const r = intent(decisionNo, battle);
    return {
      p1: r.p1Switch ? `switch ${r.p1Switch}` : legalMove(0, battle, r.p1Want),
      p2: r.p2Switch ? `switch ${r.p2Switch}` : legalMove(1, battle, r.p2Want),
    };
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

  // Optional one-time post-start injection (status / HP). STATE only (no PRNG) so the seed
  // parity is unaffected.
  if (sc.inject) {
    const battle = stream.battle;
    for (const inj of sc.inject) {
      if (inj.side !== undefined) {
        const s = battle.sides[inj.side];
        const m = s.active[0];
        if (inj.status) m.setStatus(inj.status, m, null, true);
        if (inj.hp !== undefined) m.hp = inj.hp;
      }
    }
  }

  const script = intentDriver(sc.intent);
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
    // First mover from the protocol log. A PURSUIT INTERRUPT turn is EXCLUDED (set 'none'):
    // the sim emits the Pursuit strike's `|move|…Pursuit` line INSIDE the switch's execution,
    // BEFORE the `|switch|` line, so the log-derived first mover is the pursuer — but the port's
    // action-order `first_mover` is the SWITCH (order 103 < move 200). Both are correct for their
    // definition; the divergence is purely the interrupt's line-vs-action ordering, so the seed +
    // state assertions carry the interrupt proof and the first-mover assertion is skipped here.
    let first = reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none';
    if (outcomes.pursuitActivate) first = 'none';

    const p1 = snap(battle.sides[0]);
    const p2 = snap(battle.sides[1]);

    rec.decisions.push({
      request: reqState, force,
      choiceP1: encodeChoice(choices.p1), choiceP2: encodeChoice(choices.p2),
      seedAfter, p1, p2, firstMover: first, outcomes,
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
  // Fodder used to force a quick game-end: a Level-1 mon that only Splashes/Softboils
  // (no damage) so the pursuer/FP-user can KO it once its real work is done.
  const L1 = (species, moves) => mon(species, moves, { level: 1, ability: 'No Ability' });

  // ═══════════════════════ FOCUS PUNCH ═══════════════════════

  // (1) FP LANDS: the foe uses a NON-damaging move → the user keeps focus → Focus
  //     Punch executes (the beforeTurnMove laid the volatile + `-singleturn`, the onTry
  //     did NOT cancel). REQUIRES: fpSingleturn + fpLand + a win.
  S.push({
    id: 'fp_land_vs_splash',
    p1: [mon('Machamp', ['focuspunch', 'seismictoss'], { evs: { atk: 252, hp: 252 } })],
    p2: [L1('Blissey', ['splash', 'softboiled'])],
    // The L1 Blissey Splashes; Focus Punch OHKOs it (150 BP Fighting vs Normal).
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['fpSingleturn', 'fpLand'],
  });

  // (2) FP CANCELLED: the foe DAMAGES the user first (FP is priority -3 → moves last)
  //     → lostFocus → the punch is cant'd (`|cant|…Focus Punch|Focus Punch`). Then the
  //     user Seismic-Tosses the fodder to a win. REQUIRES: fpSingleturn + fpCant + win.
  S.push({
    id: 'fp_cancel_vs_tackle',
    p1: [mon('Machamp', ['focuspunch', 'seismictoss'], { evs: { atk: 252, hp: 252 } })],
    p2: [L1('Snorlax', ['tackle'])],
    // Turn 1: Focus Punch — the L1 Snorlax Tackle chips Machamp → lostFocus → cancelled.
    // Turn 2: Seismic Toss the L1 Snorlax to a win.
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    require: ['fpSingleturn', 'fpCant'],
  });

  // (3) FP vs a STATUS move: a Thunder Wave HIT does NOT set lostFocus (onHit gates on
  //     `category !== 'Status'`), so Focus Punch LANDS even while newly paralyzed — and
  //     the para roll draws on the FP turn (onBeforeMove, before onTry). REQUIRES:
  //     fpSingleturn + fpLand.
  S.push({
    id: 'fp_vs_thunderwave',
    p1: [mon('Machamp', ['focuspunch', 'seismictoss'], { evs: { atk: 252, hp: 252 } })],
    p2: [L1('Jolteon', ['thunderwave', 'splash'])],
    // The L1 Jolteon Thunder-Waves Machamp; Focus Punch still OHKOs it (para permitting).
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['fpSingleturn', 'fpLand'],
  });

  // (4) FP behind a SUBSTITUTE: chip absorbed by the user's OWN sub does NOT set
  //     lostFocus (the sub's onTryPrimaryHit intercepts before the focuspunch onHit), so
  //     Focus Punch keeps focus + lands. REQUIRES: fpSingleturn + fpLand + a sub up.
  S.push({
    id: 'fp_behind_sub',
    p1: [mon('Machamp', ['substitute', 'focuspunch'], { evs: { atk: 252, hp: 252 } })],
    p2: [L1('Snorlax', ['tackle'])],
    // Turn 1: Substitute (the L1 Tackle can't break it). Turn 2: Focus Punch — the L1
    // Tackle hits the SUB → focus kept → FP KOs the L1 Snorlax.
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    require: ['fpSingleturn', 'fpLand'],
  });

  // (5) FP MIRROR at a SPEED TIE: both Focus Punch (the two beforeTurnMove order-5 actions
  //     tie → the mirror tie-shuffle; BOTH mons carry the focuspunch volatile at the
  //     residual → the +1 residual duration-handler tie-shuffle). The one that moves first
  //     lands (chips the other); the second is cancelled (it took damage). Grinds to a win.
  //     REQUIRES: fpSingleturn + fpLand + fpCant + win.
  S.push({
    id: 'fp_mirror_tie',
    p1: [mon('Machamp', ['focuspunch'], { evs: { atk: 252 } })],
    p2: [mon('Machamp', ['focuspunch'], { evs: { atk: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['fpSingleturn', 'fpLand', 'fpCant'],
  });

  // (6) FP + a FLINCH move that HITS: Rock Slide damages the FP user → lostFocus → FP
  //     CANCELLED, the flinch is BLOCKED by focuspunch.onTryAddVolatile (DRAW-RELEVANT: no
  //     extra residual duration handler — the flinch-secondary random(100) still draws).
  //     Then the user Seismic-Tosses the Aerodactyl to a win. REQUIRES: fpSingleturn +
  //     fpCant + win.
  S.push({
    id: 'fp_flinch_hit',
    p1: [mon('Machamp', ['focuspunch', 'seismictoss'], { evs: { hp: 252, def: 252 } })],
    p2: [mon('Aerodactyl', ['rockslide'], { level: 40, evs: { spe: 252, atk: 252 } })],
    // Turn 1: Focus Punch (cancelled by Rock Slide + flinch blocked). Turn 2+: Seismic Toss
    // the L40 Aerodactyl to a win.
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    require: ['fpSingleturn', 'fpCant'],
  });

  // ═══════════════════════ PURSUIT ═══════════════════════

  // (7) PURSUIT INTERRUPT: the foe VOLUNTARILY switches the same turn we Pursuit → the
  //     strike hits the SWITCHING mon at ×2 BP + never-miss BEFORE the switch resolves
  //     (`-activate move: Pursuit`), then the replacement comes in. REQUIRES:
  //     pursuitActivate + pursuitMove + win.
  S.push({
    id: 'pursuit_interrupt',
    p1: [mon('Snorlax', ['pursuit', 'bodyslam'], { evs: { atk: 252, hp: 252 } })],
    p2: [mon('Jolteon', ['thunderbolt', 'splash'], { evs: { spe: 252, hp: 252 } }),
         L1('Blissey', ['softboiled', 'splash'])],
    // Turn 1: Pursuit while p2 switches to the L1 Blissey → the strike hits the outgoing
    // Jolteon (×2). Turn 2: Body Slam the L1 Blissey → KO → forced switch back to Jolteon.
    // Turn 3+: Body Slam the (already-Pursuit-chipped) Jolteon to a win.
    intent: (n) => {
      if (n === 0) return { p1Want: 1, p2Switch: 2 };
      return { p1Want: 2, p2Want: 1 };
    },
    require: ['pursuitActivate', 'pursuitMove'],
  });

  // (8) NORMAL PURSUIT (the foe STAYS in): an ordinary bp-40 Dark hit (acc + crit + dmg),
  //     NO ×2, NO interrupt. REQUIRES: pursuitMove + FORBID pursuitActivate.
  S.push({
    id: 'pursuit_normal_stay',
    p1: [mon('Snorlax', ['pursuit', 'bodyslam'], { evs: { atk: 252, hp: 252 } })],
    p2: [L1('Gengar', ['splash', 'nightshade'])],
    // The L1 Gengar STAYS in and Splashes; Pursuit is a plain bp-40 Dark hit (super-
    // effective vs Ghost) that KOs the L1 Gengar. No switch → no interrupt.
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['pursuitMove'],
    forbid: ['pursuitActivate'],
  });

  // (9) PURSUIT KOs the SWITCHER: a low-HP mon switches into a Pursuit that KOs it — the
  //     ALREADY-CHOSEN switch STILL brings in the replacement (the gen 2-4 `-hint`), and
  //     the turn completes (Quick Claw drawn). REQUIRES: pursuitActivate + a p2 faint.
  S.push({
    id: 'pursuit_ko_switcher',
    p1: [mon('Snorlax', ['pursuit', 'bodyslam'], { evs: { atk: 252, hp: 252 } })],
    p2: [mon('Gengar', ['shadowball', 'splash'], { evs: { spe: 252 } }),
         L1('Blissey', ['softboiled', 'splash'])],
    inject: [{ side: 1, hp: 30 }],
    // Turn 1: Pursuit while the 30-HP Gengar switches out → the ×2 Dark interrupt KOs it →
    // the L1 Blissey still switches in. Turn 2+: Body Slam the L1 Blissey to a win.
    intent: (n) => {
      if (n === 0) return { p1Want: 1, p2Switch: 2 };
      return { p1Want: 2, p2Want: 1 };
    },
    require: ['pursuitActivate'],
  });

  // (10) PURSUIT into a switching GHOST: Dark hits Ghost (super-effective), the ×2
  //      interrupt. REQUIRES: pursuitActivate.
  S.push({
    id: 'pursuit_ghost_switcher',
    p1: [mon('Snorlax', ['pursuit', 'bodyslam'], { evs: { atk: 252, hp: 252 } })],
    p2: [mon('Misdreavus', ['shadowball', 'splash'], { evs: { spe: 252, hp: 252 } }),
         L1('Blissey', ['softboiled', 'splash'])],
    intent: (n) => {
      if (n === 0) return { p1Want: 1, p2Switch: 2 };
      return { p1Want: 2, p2Want: 1 };
    },
    require: ['pursuitActivate'],
  });

  // (11) PURSUIT into a switching SUBSTITUTE holder: the ×2 interrupt hits the switcher's
  //      SUBSTITUTE (absorbed). REQUIRES: pursuitActivate + a sub up at some decision.
  S.push({
    id: 'pursuit_into_sub',
    p1: [mon('Snorlax', ['pursuit', 'bodyslam'], { evs: { atk: 252, hp: 252 } })],
    p2: [mon('Gengar', ['substitute', 'splash'], { ability: 'No Ability', evs: { hp: 252, spe: 252 } }),
         L1('Blissey', ['softboiled', 'splash'])],
    // Turn 1: the Gengar Substitutes (Snorlax Pursuits — a normal bp-40 hit into the sub).
    // Turn 2: Pursuit while the SUB-holder Gengar switches out → the ×2 interrupt hits the
    // SUB. Turn 3+: Body Slam the L1 Blissey to a win.
    intent: (n) => {
      if (n === 0) return { p1Want: 1, p2Want: 1 }; // Gengar subs
      if (n === 1) return { p1Want: 1, p2Switch: 2 }; // Pursuit interrupts the switching sub holder
      return { p1Want: 2, p2Want: 1 };
    },
    require: ['pursuitActivate'],
  });

  // (12) PURSUIT target FASTER: the switching foe outspeeds the pursuer — the interrupt
  //      fires at SWITCH time regardless of speed. REQUIRES: pursuitActivate.
  S.push({
    id: 'pursuit_target_faster',
    p1: [mon('Snorlax', ['pursuit', 'bodyslam'], { evs: { atk: 252, hp: 252 } })],
    p2: [mon('Jolteon', ['thunderbolt', 'splash'], { evs: { spe: 252, hp: 252 } }),
         L1('Blissey', ['softboiled', 'splash'])],
    // The fast Jolteon switches; the slow Snorlax's Pursuit still interrupts it.
    intent: (n) => {
      if (n === 0) return { p1Want: 1, p2Switch: 2 };
      return { p1Want: 2, p2Want: 1 };
    },
    require: ['pursuitActivate'],
  });

  // (13) PURSUIT MIRROR at a SPEED TIE (both Pursuit): the two beforeTurnMove order-5
  //      actions tie → the mirror tie-shuffle. Neither foe switches (both stay), so both
  //      Pursuits are plain bp-40 hits + both leave a pursuit volatile up at the residual
  //      (the +1 residual duration-handler tie-shuffle on the tie). Grinds to a win.
  //      REQUIRES: pursuitMove + win; FORBID pursuitActivate (no switch → no interrupt).
  S.push({
    id: 'pursuit_mirror_tie',
    p1: [mon('Umbreon', ['pursuit'], { evs: { atk: 252 } })],
    p2: [mon('Umbreon', ['pursuit'], { evs: { atk: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['pursuitMove'],
    forbid: ['pursuitActivate'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# movecoverage_batch4_golden.txt — Gen-3 MOVE-COVERAGE BATCH 4 full-battle golden.');
  lines.push('# Per-decision STATE(+status+boosts+HP+SUB-HP)+SEED+first-mover differential to GAME-END.');
  lines.push('# Classes: FOCUS PUNCH / PURSUIT (the beforeTurnMove order-5 queue action; the pursuit');
  lines.push('#   switch-interrupt at x2 BP; the focuspunch onTry cancel + flinch block).');
  lines.push('#   The CURSE/WISH columns are always 0 here (reused DEC format); SUB-HP is live.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INJECT <id>  <json array of {side?,status?,hp?}>  ([] if none)');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1Curse p1Wish p1SubHp  p2Curse p2Wish p2SubHp');
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
    lines.push(`INJECT\t${sc.id}\t${JSON.stringify(sc.inject || [])}`);
    scenSeen[sc.id] = {};

    let scenDecs = 0;
    for (const seed of seeds) {
      let rec;
      try { rec = await runBattle(sc, seed); } catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); continue; }
      if (rec.gen !== 3) { failures.push(`${sc.id}: expected gen 3, got ${rec.gen}`); break; }
      if (!rec.initSeed || rec.decisions.length === 0) { failures.push(`${sc.id} seed ${seed}: no decisions`); continue; }
      const seedStr = seed.join(',');
      lines.push(['INIT', sc.id, rec.initSeed, seedStr].join('\t'));

      rec.decisions.forEach((d) => {
        for (const k of Object.keys(d.outcomes)) {
          if (d.outcomes[k]) { scenSeen[sc.id][k] = true; corpus[k] = (corpus[k] || 0) + 1; }
        }
      });

      rec.decisions.forEach((d) => {
        const sp = (s) => [
          s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status, s.stage, s.left,
          s.boosts[0], s.boosts[1], s.boosts[2], s.boosts[3], s.boosts[4], s.confusion,
        ].join('\t');
        lines.push([
          'DEC', sc.id, seedStr, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
          sp(d.p1), sp(d.p2), d.firstMover,
          d.p1.curse, d.p1.wish, d.p1.subHp,
          d.p2.curse, d.p2.wish, d.p2.subHp,
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
    console.error('MOVECOVERAGE BATCH4 GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`BATCH4 GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('fp-singleturn decisions', 'fpSingleturn', 40);
  need('fp-land decisions', 'fpLand', 40);
  need('fp-cant decisions', 'fpCant', 40);
  need('pursuit-activate decisions', 'pursuitActivate', 40);
  need('pursuit-move decisions', 'pursuitMove', 40);
  if (winRows < 40) { console.error(`BATCH4 GOLDEN: too few WIN rows (${winRows} < 40)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `movecoverage batch4 golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: fpSingleturn=${corpus.fpSingleturn || 0} fpLand=${corpus.fpLand || 0} fpCant=${corpus.fpCant || 0} ` +
    `pursuitActivate=${corpus.pursuitActivate || 0} pursuitMove=${corpus.pursuitMove || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
