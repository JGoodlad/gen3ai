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

const OUT = path.resolve(__dirname, '../tests/vectors/movecoverage_batch3_golden.txt');
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

// Scan the protocol log between two decision points for the BATCH-3 branch flags.
function outcomesSince(log, fromIdx) {
  const out = {
    curseStart: false, curseFail: false, curseDamage: false, curseBoost: false,
    wishHeal: false, wishCast: false, wishFail: false,
    bpPass: false, bpFail: false,
  };
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    if (tag === '-start' && (p[3] || '') === 'Curse') out.curseStart = true;
    if (tag === '-damage' && (p[4] || '').includes('Curse')) out.curseDamage = true;
    if (tag === '-heal' && (p[4] || '').includes('move: Wish')) out.wishHeal = true;
    if (tag === 'switch' && (p[5] || '').includes('Baton Pass')) out.bpPass = true;
    // A non-ghost curse emits -boost/-unboost with no [from] tag (from a Curse move).
    if (tag === 'move' && (p[3] || '') === 'Curse') {
      // a curse move used; classify by the FOLLOWING lines
      const next = log[i + 1] ? log[i + 1].split('|') : [];
      if (next[1] === '-boost' || next[1] === '-unboost') out.curseBoost = true;
    }
    if (tag === 'move' && (p[3] || '') === 'Wish') out.wishCast = true;
    if (tag === '-fail') {
      // a bare -fail on the caster after a Curse/Baton Pass [still] move announce
      const prev = log[i - 1] ? log[i - 1].split('|') : [];
      if (prev[1] === 'move' && (prev[3] || '') === 'Curse') out.curseFail = true;
      if (prev[1] === 'move' && (prev[3] || '') === 'Baton Pass') out.bpFail = true;
    }
    // double-wish [still] fail: a `|move|…|Wish||[still]` line with no -fail after.
    if (tag === 'move' && (p[3] || '') === 'Wish' && (p[5] || '') === '[still]') out.wishFail = true;
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
    const first = reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none';

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

  // --- (1) CURSE non-ghost: Snorlax self +1 Atk / +1 Def / -1 Spe. The -Spe drops its
  //   cached speed for the NEXT turn (already the slower mon here, so no flip, but the boost
  //   columns pin the mixed +/- apply). REQUIRES: curseBoost. ---
  S.push({
    id: 'curse_nonghost_boost',
    p1: [mon('Snorlax', ['curse', 'bodyslam'], { ability: 'Immunity', nature: 'Serious', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Diglett', ['mudslap'], { level: 1, ability: 'Sand Veil', nature: 'Bold' })],
    // Snorlax curses turns 1-2 (the +Atk/+Def/-Spe apply proof), then Body Slams the L1
    // Diglett to a win. (The -Spe leaves Snorlax slower, but Diglett is L1 so it dies fast.)
    intent: (n) => ({ p1Want: n < 2 ? 1 : 2, p2Want: 1 }),
    require: ['curseBoost'],
  });

  // --- (2) CURSE non-ghost first-mover FLIP: a FAST curser drops below a slower foe NEXT
  //   turn after two -Spe stacks. Persian (fast) curses twice → its -2 Spe lets Snorlax
  //   move first. REQUIRES: curseBoost. ---
  S.push({
    id: 'curse_nonghost_speed_flip',
    p1: [mon('Persian', ['curse', 'slash'], { ability: 'Limber', nature: 'Jolly', evs: { spe: 252, atk: 252 } })],
    p2: [mon('Sandshrew', ['scratch'], { level: 1, ability: 'Sand Veil', nature: 'Bold' })],
    // Persian curses turns 1-2 (each -1 Spe — the FIRST-MOVER column + SEED prove the
    // cached-speed timing), then Slashes the L1 Sandshrew to a win.
    intent: (n) => ({ p1Want: n < 2 ? 1 : 2, p2Want: 1 }),
    require: ['curseBoost'],
  });

  // --- (3) CURSE ghost: Gengar pays maxhp/2 + curses the foe; the curse residual chips
  //   maxhp/4/turn. REQUIRES: curseStart + curseDamage. ---
  S.push({
    id: 'curse_ghost_lay_and_residual',
    p1: [mon('Gengar', ['curse', 'shadowball'], { ability: 'Levitate', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['splash', 'bodyslam'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    require: ['curseStart', 'curseDamage'],
  });

  // --- (4) CURSE ghost residual ORDER vs Leftovers + burn: the curse chip (sub 8) fires
  //   AFTER Leftovers (sub 4) and the burn DoT (sub 6). p2 Snorlax holds Leftovers + is
  //   burned (injected). REQUIRES: curseStart + curseDamage. ---
  S.push({
    id: 'curse_residual_order',
    p1: [mon('Gengar', ['curse', 'shadowball'], { ability: 'Levitate', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['splash', 'bodyslam'], { ability: 'Immunity', item: 'leftovers', nature: 'Careful', evs: { hp: 252 } })],
    inject: [{ side: 1, status: 'brn' }],
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    require: ['curseStart', 'curseDamage'],
  });

  // --- (5) CURSE ghost re-curse FAIL: a 2nd Curse into an already-cursed foe fails
  //   ([still]+-fail), no HP cost. REQUIRES: curseFail. ---
  S.push({
    id: 'curse_recurse_fail',
    p1: [mon('Gengar', ['curse', 'shadowball'], { ability: 'Levitate', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['splash'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    // Turn 1 curse (lay), turn 2 curse again (fail), then Shadow Ball.
    intent: (n) => ({ p1Want: n < 2 ? 1 : 2, p2Want: 1 }),
    require: ['curseFail'],
  });

  // --- (6) CURSE ghost into a SUBSTITUTE: does nothing ([still]+-fail), no HP cost, no
  //   curse. p2 Snorlax subs turn 1; Gengar's curse into the sub fails. REQUIRES: curseFail
  //   + FORBID curseStart (no curse laid through the sub). ---
  S.push({
    id: 'curse_ghost_into_sub',
    p1: [mon('Gengar', ['curse', 'sludgebomb'], { ability: 'Levitate', nature: 'Modest', evs: { spa: 252 } })],
    p2: [mon('Snorlax', ['substitute', 'splash'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 4 } })],
    // p2 subs turn 1; Curse turn 2 into the up sub → does nothing ([still]+-fail, no curse).
    // Then Gengar Sludge Bombs (a Poison move — breaks the sub, then KOs the near-0-HP-EV
    // Snorlax) so the battle terminates. FORBID curseStart (no curse laid through the sub).
    intent: (n) => (n === 0 ? { p1Want: 2, p2Want: 1 } : (n === 1 ? { p1Want: 1, p2Want: 2 } : { p1Want: 2, p2Want: 2 })),
    require: ['curseFail'],
    forbid: ['curseStart'],
  });

  // --- (7) WISH basic: cast turn N, heals maxhp/2 at end of N+1. p1 Charizard (odd maxhp)
  //   proves the floor. REQUIRES: wishHeal. ---
  S.push({
    id: 'wish_basic_heal',
    p1: [mon('Charizard', ['wish', 'splash', 'seismictoss'], { ability: 'Blaze', nature: 'Timid', evs: { spe: 252 } })],
    p2: [mon('Diglett', ['harden'], { level: 1, ability: 'Sand Veil', nature: 'Bold' })],
    inject: [{ side: 0, hp: 100 }],
    // Turn 1: Wish. Turn 2: Splash (the Wish resolves this turn's end onto the injured
    // Charizard — the maxhp/2 heal proof). Then Seismic Toss the L1 Diglett to a win.
    intent: (n) => ({ p1Want: n === 0 ? 1 : (n === 1 ? 2 : 3), p2Want: 1 }),
    require: ['wishHeal'],
  });

  // --- (8) WISH residual ORDER: the Wish heal (order 7) fires BEFORE Leftovers (order 10) +
  //   burn. p1 Blissey holds Leftovers + is burned; Wishes then Splashes. REQUIRES: wishHeal. ---
  S.push({
    id: 'wish_residual_order',
    p1: [mon('Blissey', ['wish', 'seismictoss'], { ability: 'Natural Cure', item: 'leftovers', nature: 'Bold', evs: { hp: 252 } })],
    p2: [mon('Diglett', ['mudslap'], { level: 1, ability: 'Sand Veil', nature: 'Bold' })],
    inject: [{ side: 0, status: 'brn', hp: 100 }],
    // Blissey is burned + holds Leftovers. Turn 1 Wish (heal at order 7 fires BEFORE the
    // Leftovers order-10 heal + the burn DoT — the residual-order proof); then Seismic Toss
    // the L1 Diglett to a win.
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    require: ['wishHeal'],
  });

  // --- (9) WISH slot-keyed across a SWITCH: p1 Wishes, switches out; the incoming mon gets
  //   healed floor(ITS maxhp/2). REQUIRES: wishHeal. ---
  S.push({
    id: 'wish_slot_keyed_switch',
    p1: [mon('Blissey', ['wish', 'splash'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252 } }),
         mon('Chansey', ['splash'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252 } })],
    p2: [mon('Snorlax', ['bodyslam'], { ability: 'Immunity', nature: 'Adamant', evs: { atk: 252, hp: 252 } })],
    inject: [{ side: 0, hp: 100 }],
    // Turn 1: Wish. Turn 2: switch to Chansey (the Wish resolves this turn's end onto Chansey).
    // Then Chansey Splashes while Snorlax Body Slams it to a game-end (so the battle terminates).
    intent: (n) => (n === 0 ? { p1Want: 1, p2Want: 1 } : (n === 1 ? { p1Switch: 2, p2Want: 1 } : { p1Want: 1, p2Want: 1 })),
    require: ['wishHeal'],
  });

  // --- (10) WISH double-Wish FAIL: a 2nd Wish while one is pending FAILS ([still],
  //   draw-free). REQUIRES: wishFail + wishHeal. ---
  S.push({
    id: 'wish_double_fail',
    p1: [mon('Blissey', ['wish', 'seismictoss'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252 } })],
    p2: [mon('Diglett', ['mudslap'], { level: 1, ability: 'Sand Veil', nature: 'Bold' })],
    inject: [{ side: 0, hp: 100 }],
    // Turn 1 Wish, turn 2 Wish again (fails — a Wish is pending; the turn-2 Wish heal
    // resolves that end), then Seismic Toss the L1 Diglett to a win.
    intent: (n) => ({ p1Want: n < 2 ? 1 : 2, p2Want: 1 }),
    require: ['wishFail', 'wishHeal'],
  });

  // --- (11) WISH heal-at-FULL: a full-HP mon's Wish resolves SILENTLY (no -heal line).
  //   REQUIRES: wishCast + FORBID wishHeal (the resolve is silent → no -heal). ---
  S.push({
    id: 'wish_heal_at_full',
    p1: [mon('Blissey', ['wish', 'seismictoss'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    p2: [mon('Diglett', ['harden'], { level: 1, ability: 'Sand Veil', nature: 'Bold' })],
    // Blissey is at FULL HP (the L1 Diglett only Hardens itself — 0 damage), so the Wish
    // resolves SILENTLY (no -heal line). Then Seismic Toss the L1 Diglett to a win.
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    require: ['wishCast'],
    forbid: ['wishHeal'],
  });

  // --- (12) BATON PASS boost transfer: Jolteon Swords Dances (+2 Atk), Baton Passes to
  //   Snorlax which inherits the +2. REQUIRES: bpPass. ---
  S.push({
    id: 'bp_boost_transfer',
    p1: [mon('Jolteon', ['swordsdance', 'batonpass'], { ability: 'Volt Absorb', nature: 'Jolly', evs: { spe: 252, atk: 252 } }),
         mon('Snorlax', ['bodyslam'], { ability: 'Immunity', nature: 'Adamant', evs: { atk: 252, hp: 252 } })],
    p2: [mon('Blissey', ['softboiled', 'splash'], { level: 1, ability: 'Natural Cure', nature: 'Bold' })],
    // Turn 1: Swords Dance. Turn 2: Baton Pass (→ forced switch to Snorlax). Then Snorlax with
    // +2 Atk Body Slams the L1 Blissey to a win (the +2 boost carried across the pass).
    intent: (n) => {
      if (n === 0) return { p1Want: 1, p2Want: 2 };
      if (n === 1) return { p1Want: 2, p2Want: 2 }; // Baton Pass
      return { p1Want: 1, p2Want: 1 };
    },
    require: ['bpPass'],
  });

  // --- (13) BATON PASS substitute transfer: Jolteon Subs, Baton Passes; the SUB HP transfers
  //   to Snorlax. REQUIRES: bpPass. ---
  S.push({
    id: 'bp_sub_transfer',
    p1: [mon('Jolteon', ['substitute', 'batonpass'], { ability: 'Volt Absorb', nature: 'Jolly', evs: { spe: 252, hp: 252 } }),
         mon('Snorlax', ['bodyslam'], { ability: 'Immunity', nature: 'Adamant', evs: { atk: 252, hp: 252 } })],
    p2: [mon('Blissey', ['pound', 'splash'], { level: 1, ability: 'Natural Cure', nature: 'Bold' })],
    // Turn 1: Substitute (the L1 Blissey's Pound can't break it). Turn 2: Baton Pass → the SUB
    // HP transfers to Snorlax, which then Body Slams the L1 Blissey to a win.
    intent: (n) => {
      if (n === 0) return { p1Want: 1, p2Want: 1 }; // Substitute
      if (n === 1) return { p1Want: 2, p2Want: 1 }; // Baton Pass
      return { p1Want: 1, p2Want: 1 };
    },
    require: ['bpPass'],
  });

  // --- (14) BATON PASS leech-seed transfer: a leech-seeded Jolteon Baton Passes; the SEED
  //   transfers to Snorlax (the seeder keeps draining). REQUIRES: bpPass. ---
  S.push({
    id: 'bp_leech_transfer',
    p1: [mon('Jolteon', ['agility', 'batonpass'], { ability: 'Volt Absorb', nature: 'Jolly', evs: { spe: 252, hp: 252 } }),
         mon('Snorlax', ['bodyslam'], { ability: 'Immunity', nature: 'Adamant', evs: { atk: 252, hp: 252 } })],
    p2: [mon('Meganium', ['leechseed', 'splash'], { level: 30, ability: 'Overgrow', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    // Turn 1: Jolteon Agilities (an idle self-boost) while the L30 Meganium leech-seeds it.
    // Turn 2: Jolteon Baton Passes → the SEED (+ the +2 Spe) transfers to Snorlax (the seeder
    // keeps draining the new mon). Then Snorlax Body Slams the L30 Meganium to a win.
    intent: (n) => {
      if (n === 0) return { p1Want: 1, p2Want: 1 }; // Agility + Meganium leech-seeds Jolteon
      if (n === 1) return { p1Want: 2, p2Want: 2 }; // Jolteon Baton Passes (seed transfers)
      return { p1Want: 1, p2Want: 2 };
    },
    require: ['bpPass'],
  });

  // --- (15) BATON PASS no-bench FAIL: a last-mon Baton Pass fails ([still]+-fail),
  //   draw-free. REQUIRES: bpFail. ---
  S.push({
    id: 'bp_no_bench_fail',
    p1: [mon('Jolteon', ['batonpass', 'thunderbolt'], { ability: 'Volt Absorb', nature: 'Jolly', evs: { spe: 252, spa: 252 } })],
    p2: [mon('Blissey', ['pound'], { level: 1, ability: 'Natural Cure', nature: 'Bold' })],
    // Jolteon's ONLY mon → Baton Pass fails ([still]+-fail, draw-free); then Thunderbolt the
    // L1 Blissey to a win.
    intent: (n) => ({ p1Want: n < 2 ? 1 : 2, p2Want: 1 }),
    require: ['bpFail'],
  });

  // --- (16) BATON PASS into a real battle to a WIN: Jolteon Swords Dances, Baton Passes to
  //   a boosted Snorlax that sweeps. REQUIRES: bpPass + win. ---
  S.push({
    id: 'bp_into_a_real_battle',
    p1: [mon('Jolteon', ['swordsdance', 'batonpass'], { ability: 'Volt Absorb', nature: 'Jolly', evs: { spe: 252, atk: 252 } }),
         mon('Snorlax', ['bodyslam'], { ability: 'Immunity', nature: 'Adamant', evs: { atk: 252, hp: 252 } })],
    p2: [mon('Diglett', ['mudslap'], { level: 1, ability: 'Sand Veil', nature: 'Bold' }),
         mon('Sandshrew', ['scratch'], { level: 1, ability: 'Sand Veil', nature: 'Bold' })],
    intent: (n) => {
      if (n === 0) return { p1Want: 1, p2Want: 1 }; // Swords Dance
      if (n === 1) return { p1Want: 2, p2Want: 1 }; // Baton Pass
      return { p1Want: 1, p2Want: 1 };
    },
    require: ['bpPass'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# movecoverage_batch3_golden.txt — Gen-3 MOVE-COVERAGE BATCH 3 full-battle golden.');
  lines.push('# Per-decision STATE(+status+boosts+HP+CURSE+WISH-PENDING+SUB-HP)+SEED+first-mover differential to GAME-END.');
  lines.push('# Classes: CURSE / WISH / BATON PASS.');
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
    console.error('MOVECOVERAGE BATCH3 GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`BATCH3 GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('curse-boost decisions', 'curseBoost', 40);
  need('curse-start decisions', 'curseStart', 40);
  need('curse-damage decisions', 'curseDamage', 40);
  need('curse-fail decisions', 'curseFail', 40);
  need('wish-heal decisions', 'wishHeal', 40);
  need('wish-fail decisions', 'wishFail', 40);
  need('bp-pass decisions', 'bpPass', 40);
  need('bp-fail decisions', 'bpFail', 40);
  if (winRows < 40) { console.error(`BATCH3 GOLDEN: too few WIN rows (${winRows} < 40)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `movecoverage batch3 golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: curseBoost=${corpus.curseBoost || 0} curseStart=${corpus.curseStart || 0} curseDamage=${corpus.curseDamage || 0} ` +
    `curseFail=${corpus.curseFail || 0} wishHeal=${corpus.wishHeal || 0} wishFail=${corpus.wishFail || 0} ` +
    `bpPass=${corpus.bpPass || 0} bpFail=${corpus.bpFail || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
