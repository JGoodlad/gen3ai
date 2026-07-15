// gen_movecoverage_batch4c_golden.js — Gen-3 MOVE-COVERAGE BATCH 4c differential golden
// (`gen3_move_coverage_batch4c_v1`): the TURN-SPANNING move classes — the LAST MISMODELED
// cluster. HYPER BEAM (mustrecharge) / SOLAR BEAM (two-turn charge + sun skip) /
// DOOM DESIRE + FUTURE SIGHT (the slot-keyed future strike).
//
//   HYPER BEAM   — 150-BP Physical (gen3 Normal→Physical), acc 90. A SUCCESSFUL damaging
//                  hit (plain / sub-absorb / sub-break / target-KO) applies `mustrecharge`
//                  DRAW-FREE: the NEXT turn's request offers ONLY `Recharge` (trapped) and
//                  the turn is spent as `|cant|…|recharge` — the user's action draws ZERO
//                  and costs NO PP; a statused user draws NO status roll (the recharge
//                  onBeforeMove is priority 11, before slp/frz/truant/flinch/para). A MISS
//                  / IMMUNE / PROTECT-block does NOT trigger the lock (PP still consumed).
//   SOLAR BEAM   — 120-BP Special Grass. CHARGE turn: PP deducted (−1; −2 vs Pressure),
//                  `[still]` + `-prepare`, ZERO move draws → the `twoturnmove` volatile
//                  (a NO_ORDER/subOrder-2 residual duration handler on BOTH residuals).
//                  FIRE turn: locked single-move request, NO PP, accuracy 100 DRAWN → crit
//                  → damage. SUN skips the charge (3 draws like a normal move); an abort
//                  on the fire turn LOSES the charge; rain/sand/hail HALVE the BP
//                  (state-only, read at damage time).
//   DOOM DESIRE / FUTURE SIGHT — the CAST draws exactly ONE random(16) (the cast-time
//                  damage SNAPSHOT — typeless, no STAB/chart, willCrit false, cast-time
//                  stats), `|-start|<caster>|<Name>`; the pending slot condition registers
//                  an order-11 residual handler every end-of-turn; at the end of turn N+2
//                  it resolves — ONE accuracy roll (85/90) then the STORED number lands on
//                  WHOEVER occupies the slot (sub absorbs, no carry; a resolve KO defers
//                  the Quick Claw past the forced replacement). A double-cast FAILS with a
//                  bare |move| line, zero draws, PP still deducted.
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios that each ISOLATE a branch, capturing initSeed + per-decision seedAfter, each
// active's species/hp/maxhp/fainted/status + boosts + confusion + pokemon_left + CURSE +
// WISH-PENDING + SUB-HP + per-side FUTURE-PENDING + first mover + winner. The Rust test
// seeds a BattleState at initSeed and runs `run_full_battle` WITHOUT re-seeding — so the
// post-decision seed must match at EVERY boundary (a wrong draw model → SEED desync), AND
// the recharge-lock / charge-fire / future-resolve HP+timing must match.
//
// EXTENDS the batch-3/4 42-field DEC format with 2 trailing per-side FUTURE-PENDING
// columns → 44 fields. WISH is live in ONE scenario (the residual-order composition);
// CURSE stays 0.
//
// Output: tests/vectors/movecoverage_batch4c_golden.txt
//
// Run:  node src/rust_sim/harness/gen_movecoverage_batch4c_golden.js

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/movecoverage_batch4c_golden.txt');
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
  let x = 0x4c7e21b9 >>> 0;
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

// The pending FUTUREMOVE slot condition's remaining duration on this side's slot
// (`side.slotConditions[0].futuremove.duration` — 3 at cast, post-residual 2 → 1 → gone).
function futureOf(side) {
  const sc = side.slotConditions && side.slotConditions[0];
  const f = sc && sc.futuremove;
  return f ? (f.duration | 0) : 0;
}

function snap(side) {
  const a = side.active[0];
  const wish = wishOf(side);
  const future = futureOf(side);
  if (!a) {
    return {
      species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', stage: 0, left: side.pokemonLeft,
      boosts: [0, 0, 0, 0, 0], confusion: 0, curse: 0, wish, subHp: 0, future,
    };
  }
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status, stage, left: side.pokemonLeft,
    boosts: boostsOf(a), confusion: confusionOf(a),
    curse: curseOf(a), wish, subHp: subHpOf(a), future,
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

// Scan the protocol log between two decision points for the BATCH-4c branch flags.
function outcomesSince(log, fromIdx) {
  const out = {
    hbMove: false,        // a Hyper Beam move RAN
    hbMiss: false,        // a Hyper Beam that MISSED ([miss])
    hbImmune: false,      // a Hyper Beam into an immune target (-immune)
    mustRecharge: false,  // the |-mustrecharge| line (the lock applied)
    rechargeCant: false,  // |cant|…|recharge (the locked turn)
    truantCant: false,    // |cant|…|ability: Truant (must NEVER fire in the HB-Truant scenario)
    sbPrepare: false,     // |-prepare| (a Solar Beam charge turn)
    sbAnim: false,        // |-anim| (the SUN skip)
    sbFire: false,        // a Solar Beam FIRE (the [from]lockedmove move line)
    protectBlock: false,  // |-activate|…|Protect (a blocked hit)
    subHit: false,        // a sub absorbed/broke (-activate Substitute [damage] / -end Substitute)
    fmStart: false,       // |-start|…|Doom Desire / Future Sight (a future-move CAST)
    fmEnd: false,         // |-end|…|move: Doom Desire / Future Sight (a RESOLVE)
    fmMiss: false,        // a resolve that MISSED (a -miss in a window with fmEnd)
    fmCastFail: false,    // a re-cast while pending (a |move|DD/FS window with NO -start)
    fmKo: false,          // a resolve window that fainted a mon (fmEnd + faint)
    cantAny: false,       // any |cant| (the SB abort scenarios' para/flinch cants)
    wishHeal: false,      // a Wish heal resolved (the residual-order scenario)
  };
  let sawFmMove = false;
  let sawFmStart = false;
  let sawMiss = false;
  let sawFaint = false;
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    if (tag === 'move') {
      const name = p[3] || '';
      if (name === 'Hyper Beam') {
        out.hbMove = true;
        if ((p[5] || '') === '[miss]') out.hbMiss = true;
      }
      if (name === 'Solar Beam' && (p[5] || '').includes('lockedmove')) out.sbFire = true;
      if (name === 'Doom Desire' || name === 'Future Sight') sawFmMove = true;
    }
    if (tag === '-immune' && log[i - 1] && log[i - 1].includes('Hyper Beam')) out.hbImmune = true;
    if (tag === '-mustrecharge') out.mustRecharge = true;
    if (tag === 'cant') {
      out.cantAny = true;
      if ((p[3] || '') === 'recharge') out.rechargeCant = true;
      if ((p[3] || '') === 'ability: Truant') out.truantCant = true;
    }
    if (tag === '-prepare') out.sbPrepare = true;
    if (tag === '-anim') out.sbAnim = true;
    if (tag === '-activate' && (p[3] || '') === 'Protect') out.protectBlock = true;
    if ((tag === '-activate' && (p[3] || '') === 'Substitute') ||
        (tag === '-end' && (p[3] || '') === 'Substitute')) out.subHit = true;
    if (tag === '-start' && /^(Doom Desire|Future Sight)$/.test(p[3] || '')) {
      out.fmStart = true;
      sawFmStart = true;
    }
    if (tag === '-end' && /^move: (Doom Desire|Future Sight)$/.test(p[3] || '')) out.fmEnd = true;
    if (tag === '-miss') sawMiss = true;
    if (tag === 'faint') sawFaint = true;
    if (tag === '-heal' && (p[4] || '').includes('move: Wish')) out.wishHeal = true;
  }
  if (out.fmEnd && sawMiss) out.fmMiss = true;
  if (out.fmEnd && sawFaint) out.fmKo = true;
  if (sawFmMove && !sawFmStart) out.fmCastFail = true;
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

// The 1-based bench slot of the first live non-active mon (for a scripted voluntary
// switch), or 0 when none.
function benchSlot(battle, side) {
  const s = battle.sides[side];
  for (let k = 0; k < s.pokemon.length; k++) {
    const p = s.pokemon[k];
    if (p !== s.active[0] && !p.fainted) return k + 1;
  }
  return 0;
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

  // Optional one-time post-start injection (status / HP). STATE only (no PRNG) so seed parity
  // is unaffected.
  if (sc.inject) {
    const battle = stream.battle;
    for (const inj of sc.inject) {
      if (inj.side !== undefined) {
        const idx = inj.slot !== undefined ? inj.slot : 0;
        const m = idx === 0 ? battle.sides[inj.side].active[0] : battle.sides[inj.side].pokemon[idx];
        if (inj.status) m.setStatus(inj.status, m, null, true);
        if (inj.hp !== undefined) m.hp = inj.hp;
      }
    }
  }

  const script = intentDriver(sc.intent);
  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen };

  let decisionNo = 0;
  let safety = 0;
  while (!stream.battle.ended && safety < 600) {
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
    for (let i = 0; i < 20; i++) await tick();

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

  // ═══════════════════════ HYPER BEAM ═══════════════════════

  // (1) HB hit → recharge → clear CYCLE (+ natural acc-90 misses across the sweep —
  //     a MISS must NOT trigger the lock). Snorlax HBs every turn (the locked turns'
  //     `move 1` resolves to Recharge); the bulky walls grind down to a win.
  S.push({
    id: 'hb_hit_recharge_cycle',
    p1: [
      mon('Snorlax', ['hyperbeam', 'seismictoss'], { evs: { atk: 252, hp: 252 } }),
      mon('Blissey', ['seismictoss'], { evs: { hp: 252 } }),
    ],
    p2: [
      mon('Skarmory', ['splash', 'seismictoss'], { evs: { hp: 252 } }),
      mon('Forretress', ['splash', 'seismictoss'], { evs: { hp: 252 } }),
    ],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['hbMove', 'mustRecharge', 'rechargeCant', 'hbMiss'],
  });

  // (2) HB KOs the target — the |-mustrecharge| precedes the |faint|, the lock PERSISTS
  //     across the opponent's force-switch (the next turn is still locked). Frail L50
  //     foes so a landed HB is a guaranteed KO.
  S.push({
    id: 'hb_ko_lock_persists',
    p1: [mon('Snorlax', ['hyperbeam', 'seismictoss'], { evs: { atk: 252, hp: 252 } })],
    p2: [
      mon('Alakazam', ['seismictoss', 'splash'], { level: 50 }),
      mon('Jolteon', ['seismictoss', 'splash'], { level: 50 }),
      mon('Espeon', ['seismictoss', 'splash'], { level: 50 }),
    ],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['hbMove', 'mustRecharge', 'rechargeCant'],
  });

  // (3) HB into a PROTECT — a blocked HB does NOT trigger the lock; the user HBs again
  //     the very next turn. The protect stall-denominator interplay also realizes.
  S.push({
    id: 'hb_into_protect',
    p1: [mon('Snorlax', ['hyperbeam', 'seismictoss'], { evs: { atk: 252, hp: 252 } })],
    p2: [mon('Skarmory', ['protect', 'splash', 'seismictoss'], { evs: { hp: 252 } })],
    // p2 alternates Protect/Splash so blocked AND landed HBs both realize.
    intent: (n) => ({ p1Want: 1, p2Want: (n % 2 === 0) ? 1 : 2 }),
    require: ['hbMove', 'protectBlock', 'mustRecharge', 'rechargeCant'],
  });

  // (4) HB into a SUBSTITUTE — BOTH a sub-absorb and a sub-break DO trigger the lock.
  S.push({
    id: 'hb_into_sub',
    p1: [mon('Snorlax', ['hyperbeam', 'seismictoss'], { evs: { atk: 252, hp: 252 } })],
    p2: [mon('Snorlax', ['substitute', 'splash', 'seismictoss'], { evs: { hp: 252, spe: 252 } })],
    intent: (n) => ({ p1Want: 1, p2Want: (n % 3 === 0) ? 1 : 2 }),
    require: ['hbMove', 'subHit', 'mustRecharge', 'rechargeCant'],
  });

  // (5) HB into a GHOST — IMMUNE: the accuracy draw is consumed, NO lock, PP consumed;
  //     Crunch (a modeled secondary move) finishes the battle.
  S.push({
    id: 'hb_into_ghost_immune',
    p1: [mon('Snorlax', ['hyperbeam', 'crunch'], { evs: { atk: 252, hp: 252 } })],
    p2: [mon('Gengar', ['nightshade', 'splash'], { ability: 'No Ability', evs: { hp: 252 } })],
    // Alternate HB (immune — proves no lock) and Crunch (real damage to a win).
    intent: (n) => ({ p1Want: (n % 2 === 0) ? 1 : 2, p2Want: 1 }),
    require: ['hbMove', 'hbImmune'],
    forbid: ['mustRecharge', 'rechargeCant'],
  });

  // (6) STATUSED user on the locked turn — p2's Thunder Wave lands across the sweep;
  //     the recharge cant (priority 11) WINS over par (NO para roll on the locked turn —
  //     a draw-count crux the seed assertion pins).
  S.push({
    id: 'hb_recharge_statused',
    p1: [mon('Snorlax', ['hyperbeam', 'seismictoss'], { evs: { atk: 252, hp: 252 } })],
    p2: [mon('Skarmory', ['thunderwave', 'splash', 'seismictoss'], { evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: 1, p2Want: (n % 2 === 0) ? 1 : 2 }),
    require: ['hbMove', 'mustRecharge', 'rechargeCant'],
  });

  // (7) HB MIRROR at a SPEED TIE — the action-order tie + the cast-turn mustrecharge
  //     duration-handler residual tie (both locked/unlocked in phase) all draw.
  S.push({
    id: 'hb_mirror_tie',
    p1: [
      mon('Snorlax', ['hyperbeam', 'seismictoss'], { evs: { atk: 252, hp: 252 } }),
      mon('Blissey', ['seismictoss'], { evs: { hp: 252 } }),
    ],
    p2: [mon('Snorlax', ['hyperbeam', 'seismictoss'], { evs: { atk: 252, hp: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['hbMove', 'mustRecharge', 'rechargeCant'],
  });

  // (8) TRUANT Slaking + HB — a LANDED HB's recharge turn consumes the loaf (HB /
  //     recharge / HB cadence with no truant cant on that path); a MISSED HB leaves no
  //     lock, so the next turn's loaf legitimately shows `|cant|…|ability: Truant`
  //     (across the seed sweep BOTH interleavings realize — the parity clock composition
  //     the seed assertion pins).
  S.push({
    id: 'hb_truant_slaking',
    p1: [mon('Slaking', ['hyperbeam', 'seismictoss'], { ability: 'Truant', evs: { atk: 252, hp: 252 } })],
    p2: [
      mon('Skarmory', ['splash', 'seismictoss'], { evs: { hp: 252 } }),
      mon('Forretress', ['splash', 'seismictoss'], { evs: { hp: 252 } }),
    ],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['hbMove', 'mustRecharge', 'rechargeCant'],
  });

  // ═══════════════════════ SOLAR BEAM ═══════════════════════

  // (9) SB charge → fire CYCLE (no weather): the charge turn draws ZERO move draws, the
  //     fire turn draws acc+crit+dmg; the locked fire request; grind to a win.
  S.push({
    id: 'sb_charge_fire',
    p1: [mon('Venusaur', ['solarbeam', 'gigadrain'], { evs: { spa: 252, hp: 252 } })],
    p2: [mon('Swampert', ['surf', 'splash'], { evs: { hp: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['sbPrepare', 'sbFire'],
  });

  // (10) SB SUN SKIP (Drought foe): [still] + -prepare + -anim then IMMEDIATE normal
  //      execution — 3 draws, PP −1, NO volatile.
  S.push({
    id: 'sb_sun_skip',
    p1: [mon('Venusaur', ['solarbeam', 'gigadrain'], { evs: { spa: 252, hp: 252 } })],
    p2: [mon('Groudon', ['earthquake', 'splash'], { ability: 'Drought', evs: { hp: 4 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['sbPrepare', 'sbAnim'],
    forbid: ['sbFire'],
  });

  // (11) SB in RAIN (Drizzle foe): charges normally, the fire's BP is HALVED (state-only
  //      — the damage differs, the draws do not).
  S.push({
    id: 'sb_rain_halved',
    p1: [mon('Venusaur', ['solarbeam', 'gigadrain'], { evs: { spa: 252, hp: 252 } })],
    p2: [mon('Kyogre', ['calmmind', 'splash'], { ability: 'Drizzle', evs: { hp: 4 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['sbPrepare', 'sbFire'],
  });

  // (12) FIRE-turn ABORT — p2's Thunder Wave paralyzes; a full-para cant on the fire (or
  //      charge) turn LOSES the charge (a fresh charge re-pays PP). The para roll on the
  //      CHARGE turn (drawn) is also exercised.
  S.push({
    id: 'sb_para_abort',
    p1: [mon('Venusaur', ['solarbeam', 'gigadrain'], { evs: { spa: 252, hp: 252 } })],
    p2: [mon('Swampert', ['thunderwave', 'surf', 'splash'], { evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: 1, p2Want: (n === 0) ? 1 : 2 }),
    require: ['sbPrepare', 'sbFire'],
  });

  // (13) FIRE into a PROTECT — the accuracy is drawn then blocked; the charge is
  //      CONSUMED (the next SB re-charges, re-paying PP).
  S.push({
    id: 'sb_fire_into_protect',
    p1: [mon('Venusaur', ['solarbeam', 'gigadrain'], { evs: { spa: 252, hp: 252 } })],
    p2: [mon('Swampert', ['protect', 'surf', 'splash'], { evs: { hp: 252 } })],
    // p2 protects on ODD decisions (the fire turns of a clean charge/fire cadence).
    intent: (n) => ({ p1Want: 1, p2Want: (n % 2 === 1) ? 1 : 3 }),
    require: ['sbPrepare', 'sbFire', 'protectBlock'],
  });

  // (14) SB MIRROR at a SPEED TIE — both charge/fire in phase: the action-order tie, the
  //      per-residual twoturnmove duration-handler ties, and the fire-turn in-tryMoveHit
  //      Update ties all draw.
  S.push({
    id: 'sb_mirror_tie',
    p1: [
      mon('Venusaur', ['solarbeam', 'gigadrain'], { evs: { spa: 252, hp: 252, spe: 252 } }),
      mon('Blissey', ['seismictoss'], { evs: { hp: 252 } }),
    ],
    p2: [mon('Venusaur', ['solarbeam', 'gigadrain'], { evs: { spa: 252, hp: 252, spe: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['sbPrepare', 'sbFire'],
  });

  // (15) SB into a SUBSTITUTE — the fire absorbs into the sub (break, no carry).
  S.push({
    id: 'sb_into_sub',
    p1: [mon('Venusaur', ['solarbeam', 'gigadrain'], { evs: { spa: 252, hp: 252 } })],
    p2: [mon('Snorlax', ['substitute', 'splash', 'seismictoss'], { evs: { hp: 252, spe: 252 } })],
    intent: (n) => ({ p1Want: 1, p2Want: (n % 3 === 0) ? 1 : 2 }),
    require: ['sbPrepare', 'sbFire', 'subHit'],
  });

  // ═══════════════════════ DOOM DESIRE / FUTURE SIGHT ═══════════════════════

  // (16) DD cast → idle → RESOLVE at N+2 (+ the acc-85 resolve MISS + the DOUBLE-CAST
  //      fail — Jirachi re-casts every turn, so every second cast fails draw-free with
  //      PP still deducted). Seismic Toss keeps chip pressure for a win.
  S.push({
    id: 'dd_cast_resolve',
    p1: [mon('Jirachi', ['doomdesire', 'seismictoss'], { evs: { spa: 252, hp: 252 } })],
    p2: [mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: (n % 2 === 0) ? 1 : 2, p2Want: 1 }),
    require: ['fmStart', 'fmEnd', 'fmMiss'],
  });

  // (17) DD DOUBLE-CAST every turn — the fail is a bare |move| with ZERO draws (PP still
  //      deducted; the pending strike resolves on schedule).
  S.push({
    id: 'dd_double_cast',
    p1: [mon('Jirachi', ['doomdesire', 'seismictoss'], { evs: { spa: 252, hp: 252 } })],
    p2: [mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['fmStart', 'fmEnd', 'fmCastFail'],
  });

  // (18) FUTURE SIGHT — the same mechanic at bp 80 / acc 90 (Special).
  S.push({
    id: 'fs_cast_resolve',
    p1: [mon('Jirachi', ['futuresight', 'seismictoss'], { evs: { spa: 252, hp: 252 } })],
    p2: [mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: (n % 2 === 0) ? 1 : 2, p2Want: 1 }),
    require: ['fmStart', 'fmEnd'],
  });

  // (19) SLOT SEMANTICS — p2 switches between the cast and the resolve: the strike hits
  //      the ENTRANT with the OLD stored (cast-time) damage.
  S.push({
    id: 'dd_slot_switch',
    p1: [mon('Jirachi', ['doomdesire', 'seismictoss'], { evs: { spa: 252, hp: 252 } })],
    p2: [
      mon('Skarmory', ['splash', 'seismictoss'], { evs: { hp: 252 } }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    ],
    // p2 pivots every 3rd decision so casts resolve onto a switched-in slot occupant.
    intent: (n, battle) => {
      const b = benchSlot(battle, 1);
      if (n % 3 === 1 && b) return { p1Want: 2, p2Switch: b };
      return { p1Want: (n % 3 === 0) ? 1 : 2, p2Want: 1 };
    },
    require: ['fmStart', 'fmEnd'],
  });

  // (20) DD into a SUBSTITUTE at resolve — the sub absorbs the stored number (accuracy
  //      still drawn; break, no carry).
  S.push({
    id: 'dd_into_sub',
    p1: [mon('Jirachi', ['doomdesire', 'seismictoss'], { evs: { spa: 252, hp: 252 } })],
    p2: [mon('Snorlax', ['substitute', 'splash', 'seismictoss'], { evs: { hp: 252, spe: 252 } })],
    intent: (n) => ({ p1Want: (n % 3 === 0) ? 1 : 2, p2Want: (n % 3 === 0) ? 1 : 2 }),
    require: ['fmStart', 'fmEnd', 'subHit'],
  });

  // (21) A RESOLVE KO — the strike faints a frail foe at the residual: the deferred-faint
  //      protocol (NO Quick Claw on the resolve turn; the QC lands after the forced
  //      replacement), then the battle grinds to a win.
  S.push({
    id: 'dd_resolve_ko',
    p1: [mon('Jirachi', ['doomdesire', 'splash'], { evs: { atk: 252, hp: 252 } })],
    p2: [
      mon('Gengar', ['nightshade', 'splash'], { ability: 'No Ability', level: 50 }),
      mon('Alakazam', ['nightshade', 'splash'], { level: 50 }),
      mon('Jolteon', ['nightshade', 'splash'], { level: 50 }),
    ],
    // Cast whenever possible (a pending re-cast fails draw-free); each resolve OHKOs
    // the frail L50 occupant — the DD strikes alone grind p2 to a win (typeless hits
    // the Ghost Gengar too).
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['fmStart', 'fmEnd', 'fmKo'],
  });

  // (22) FS MIRROR at a SPEED TIE — both cast/resolve in phase: the order-11 residual
  //      handler tie EVERY end-of-turn + the landed-resolve double-Update ties.
  S.push({
    id: 'fs_mirror_tie',
    p1: [
      mon('Jirachi', ['futuresight', 'seismictoss'], { evs: { spa: 252, hp: 252, spe: 252 } }),
      mon('Blissey', ['seismictoss'], { evs: { hp: 252 } }),
    ],
    p2: [mon('Jirachi', ['futuresight', 'seismictoss'], { evs: { spa: 252, hp: 252, spe: 252 } })],
    intent: (n) => ({ p1Want: (n % 2 === 0) ? 1 : 2, p2Want: (n % 2 === 0) ? 1 : 2 }),
    require: ['fmStart', 'fmEnd'],
  });

  // (23) THE RESIDUAL-ORDER COMPOSITION — Wish(7) → sand chip(8) → Leftovers(10.4) →
  //      futuremove(11): a Celebi under Sand Stream weaves Wish + Doom Desire; every
  //      residual layer fires in one turn (the life/death ordering the seed+HP pin).
  S.push({
    id: 'dd_residual_order',
    p1: [mon('Celebi', ['doomdesire', 'wish', 'gigadrain'], { item: 'Leftovers', evs: { spa: 252, hp: 252 } })],
    p2: [mon('Tyranitar', ['splash', 'seismictoss'], { ability: 'Sand Stream', item: 'Leftovers', evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: (n % 3 === 0) ? 1 : ((n % 3 === 1) ? 2 : 3), p2Want: 2 }),
    require: ['fmStart', 'fmEnd', 'wishHeal'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# movecoverage_batch4c_golden.txt — Gen-3 MOVE-COVERAGE BATCH 4c full-battle golden.');
  lines.push('# Per-decision STATE(+status+boosts+HP+SUB-HP+WISH+FUTURE-PENDING)+SEED+first-mover');
  lines.push('# differential to GAME-END. Classes: HYPER BEAM (mustrecharge) / SOLAR BEAM (two-turn');
  lines.push('#   charge + sun skip) / DOOM DESIRE + FUTURE SIGHT (the slot-keyed future strike).');
  lines.push('# EXTENDS the batch-3/4 DEC format with 2 trailing per-side FUTURE-PENDING columns (44 fields).');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INJECT <id>  <json array of {side?,slot?,status?,hp?}>  ([] if none)');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1Curse p1Wish p1SubHp  p2Curse p2Wish p2SubHp  p1Future p2Future');
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
          d.p1.future, d.p2.future,
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
    console.error('MOVECOVERAGE BATCH4c GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`BATCH4c GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('hyper-beam decisions', 'hbMove', 40);
  need('mustrecharge decisions', 'mustRecharge', 40);
  need('recharge-cant decisions', 'rechargeCant', 40);
  need('hyper-beam miss decisions', 'hbMiss', 5);
  need('solar-beam charge decisions', 'sbPrepare', 40);
  need('solar-beam fire decisions', 'sbFire', 40);
  need('sun-skip decisions', 'sbAnim', 20);
  need('future-move cast decisions', 'fmStart', 40);
  need('future-move resolve decisions', 'fmEnd', 40);
  need('future-move resolve-miss decisions', 'fmMiss', 5);
  need('future-move double-cast-fail decisions', 'fmCastFail', 10);
  need('future-move resolve-KO decisions', 'fmKo', 5);
  if (winRows < 40) { console.error(`BATCH4c GOLDEN: too few WIN rows (${winRows} < 40)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `movecoverage batch4c golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: hb=${corpus.hbMove || 0} mustrecharge=${corpus.mustRecharge || 0} rechargeCant=${corpus.rechargeCant || 0} ` +
    `hbMiss=${corpus.hbMiss || 0} hbImmune=${corpus.hbImmune || 0} sbPrepare=${corpus.sbPrepare || 0} sbFire=${corpus.sbFire || 0} ` +
    `sbAnim=${corpus.sbAnim || 0} fmStart=${corpus.fmStart || 0} fmEnd=${corpus.fmEnd || 0} fmMiss=${corpus.fmMiss || 0} ` +
    `fmCastFail=${corpus.fmCastFail || 0} fmKo=${corpus.fmKo || 0} protectBlock=${corpus.protectBlock || 0} subHit=${corpus.subHit || 0} ` +
    `wishHeal=${corpus.wishHeal || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
