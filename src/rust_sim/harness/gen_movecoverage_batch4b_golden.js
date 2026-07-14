// gen_movecoverage_batch4b_golden.js — Gen-3 MOVE-COVERAGE BATCH 4b differential golden
// (`gen3_move_coverage_batch4b_v1`): the THREE remaining MISMODELED single-turn damaging
// moves — BEAT UP / THUNDER / WATER SPOUT.
//
//   BEAT UP      — a MULTI-STRIKE move: ONE strike PER healthy (non-fainted, NON-STATUSED)
//                  party member of the USER's side, each a TYPELESS flat-BP-10 Special hit
//                  with the STAT SWAP (ally dex base-atk → attacker SpA, target dex base-def
//                  → defender SpD, modifier=1). The whole move draws ONE accuracy roll (acc
//                  100) BEFORE the multi-strike loop, then PER STRIKE crit `randomChance(1,16)`
//                  + damage `random(16)`. The multihit STOPS at the target's faint. A statused/
//                  fainted teammate (incl. a statused ACTIVE user) skips its strike; a sub
//                  break lets later strikes hit the mon directly.
//   THUNDER      — a 120-BP Special Electric move (base acc 70, 30% para) whose id-gated
//                  onModifyMove REWRITES the base accuracy by the target's effective weather:
//                  RAIN → never-miss (the accuracy `random(100)` is SKIPPED — ONE FEWER draw),
//                  SUN → base 50, else (none / sand / hail / Cloud-Nine-or-Air-Lock-suppressed)
//                  → base 70. The draw-count crux: rain removes EXACTLY the accuracy draw.
//   WATER SPOUT  — a variable-BP Special Water move: `bp = max(floor(150·hp/maxhp), 1)` — a
//                  deterministic STATE read of the USER's CURRENT hp, DRAW-NEUTRAL (only the
//                  damage magnitude changes, not any roll).
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios that each ISOLATE a class, capturing initSeed + per-decision seedAfter, each
// active's species/hp/maxhp/fainted/status + boosts + confusion + pokemon_left + CURSE +
// WISH-PENDING + SUB-HP + first mover + winner. The Rust test seeds a BattleState at initSeed
// and runs `run_full_battle` WITHOUT re-seeding — so the post-decision seed must match at
// EVERY boundary (a wrong draw model → SEED desync), AND the multi-strike HP + the
// Thunder-weather-accuracy hit/miss + the Water-Spout variable-BP damage must match.
//
// Reuses the batch-3/4 42-field DEC format (CURSE/WISH columns are always 0 here; SUB-HP is
// live — Beat Up / Water Spout into a sub).
//
// Output: tests/vectors/movecoverage_batch4b_golden.txt
//
// Run:  node src/rust_sim/harness/gen_movecoverage_batch4b_golden.js

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/movecoverage_batch4b_golden.txt');
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

// Scan the protocol log between two decision points for the BATCH-4b branch flags.
function outcomesSince(log, fromIdx) {
  const out = {
    beatUp: false,          // a Beat Up move RAN (`|move|…|Beat Up`)
    beatUpHitcount: false,  // a `|-hitcount|` line (the multi-strike completed)
    beatUpSubBreak: false,  // a Beat Up strike BROKE a sub (`|-end|…|Substitute`)
    thunder: false,         // a Thunder move RAN
    thunderMiss: false,     // a Thunder that MISSED (`|move|…|Thunder|…|[miss]`)
    thunderPara: false,     // a Thunder that PARALYZED (`|-status|…|par`)
    waterSpout: false,      // a Water Spout move RAN
  };
  let lastBeatUp = false;
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    if (tag === 'move') {
      const name = p[3] || '';
      lastBeatUp = name === 'Beat Up';
      if (name === 'Beat Up') out.beatUp = true;
      if (name === 'Thunder') {
        out.thunder = true;
        if ((p[5] || '') === '[miss]') out.thunderMiss = true;
      }
      if (name === 'Water Spout') out.waterSpout = true;
    }
    if (tag === '-hitcount') out.beatUpHitcount = true;
    if (tag === '-end' && (p[3] || '') === 'Substitute' && lastBeatUp) out.beatUpSubBreak = true;
    if (tag === '-status' && (p[3] || '') === 'par') out.thunderPara = true;
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
  const L1 = (species, moves) => mon(species, moves, { level: 1, ability: 'No Ability' });

  // ═══════════════════════ BEAT UP ═══════════════════════

  // (1) BEAT UP full healthy side: all 6 party mons strike a BULKY foe that SURVIVES the
  //     turn (so a 6-strike turn is realized), then p1 grinds it down to a win.
  S.push({
    id: 'beatup_full_side',
    p1: [
      mon('Slaking', ['beatup', 'seismictoss'], { evs: { atk: 252, hp: 252 } }),
      mon('Machamp', ['seismictoss']),
      mon('Alakazam', ['seismictoss']),
      mon('Snorlax', ['seismictoss']),
      mon('Blissey', ['seismictoss']),
      mon('Gengar', ['seismictoss'], { ability: 'No Ability' }),
    ],
    p2: [mon('Skarmory', ['roost', 'splash'], { evs: { hp: 4 } })],
    // p1 Beat Up every turn (6 strikes); the bulky Skarmory Splashes (draw-free) until it dies.
    intent: () => ({ p1Want: 1, p2Want: 2 }),
    require: ['beatUp', 'beatUpHitcount'],
  });

  // (2) BEAT UP with STATUSED teammates: two teammates injected brn/par → their strikes are
  //     SKIPPED (fewer strikes → fewer draws).
  S.push({
    id: 'beatup_statused_skip',
    p1: [
      mon('Slaking', ['beatup', 'seismictoss'], { evs: { atk: 252, hp: 252 } }),
      mon('Machamp', ['seismictoss']),
      mon('Snorlax', ['seismictoss']),
      mon('Blissey', ['seismictoss']),
      mon('Gengar', ['seismictoss'], { ability: 'No Ability' }),
      mon('Alakazam', ['seismictoss']),
    ],
    p2: [mon('Skarmory', ['roost', 'splash'], { evs: { hp: 4 } })],
    inject: [
      { side: 0, slot: 1, status: 'brn' }, // Machamp burned → skips its strike
      { side: 0, slot: 3, status: 'par' }, // Blissey para'd → skips its strike
    ],
    intent: () => ({ p1Want: 1, p2Want: 2 }),
    require: ['beatUp', 'beatUpHitcount'],
  });

  // (3) BEAT UP KOs the target MID-SEQUENCE: a low-HP foe is KO'd on an early strike → the
  //     loop STOPS, later strikes + the Quick Claw skip. Grinds no further (the foe is its
  //     only mon) → p1 wins turn 1.
  S.push({
    id: 'beatup_ko_midsequence',
    p1: [
      mon('Slaking', ['beatup', 'seismictoss'], { evs: { atk: 252 } }),
      mon('Snorlax', ['seismictoss']),
      mon('Blissey', ['seismictoss']),
    ],
    p2: [mon('Gengar', ['shadowball', 'splash'], { ability: 'No Ability', evs: { spe: 252 } })],
    inject: [{ side: 1, hp: 15 }], // low HP → an early strike KOs it, stopping the multihit
    intent: () => ({ p1Want: 1, p2Want: 2 }),
    require: ['beatUp', 'beatUpHitcount'],
  });

  // (4) BEAT UP into a SUBSTITUTE: the foe subs first, Beat Up strikes into the sub — a
  //     strike BREAKS it, later strikes hit the mon directly. Then p1 grinds to a win.
  S.push({
    id: 'beatup_into_sub',
    p1: [
      mon('Slaking', ['beatup', 'seismictoss'], { evs: { atk: 252, hp: 252 } }),
      mon('Snorlax', ['seismictoss']),
      mon('Blissey', ['seismictoss']),
    ],
    p2: [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252, spe: 252 } })],
    // Turn 0: the faster p2 Snorlax Substitutes; p1 Beat Up strikes the fresh sub. Turn 1+:
    // Beat Up keeps striking (sub break + later strikes hit the mon), grinding to a win.
    intent: (n) => (n === 0 ? { p1Want: 1, p2Want: 1 } : { p1Want: 1, p2Want: 2 }),
    require: ['beatUp', 'beatUpHitcount'],
  });

  // (5) BEAT UP into a GHOST: the TYPELESS '???' strikes hit Ghost at 1× (no Dark super-
  //     effective, no immunity). Grinds to a win.
  S.push({
    id: 'beatup_into_ghost',
    p1: [
      mon('Slaking', ['beatup', 'seismictoss'], { evs: { atk: 252, hp: 252 } }),
      mon('Machamp', ['seismictoss']),
      mon('Snorlax', ['seismictoss']),
    ],
    p2: [mon('Gengar', ['splash', 'nightshade'], { ability: 'No Ability', evs: { hp: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['beatUp', 'beatUpHitcount'],
  });

  // (5b) BEAT UP at a SPEED TIE — the attacker and defender tie on speed, so the gen3
  //      multihit loop's PER-STRIKE `eachEvent('Update')` DRAWS a shuffle after each strike
  //      (zero at distinct speed — the gap the other Beat Up scenarios miss). Both Charizard
  //      at 252 Spe / Modest tie; the foe Splashes (draw-free). p1 grinds to a win.
  S.push({
    id: 'beatup_speed_tie',
    p1: [
      mon('Charizard', ['beatup', 'seismictoss'], { ability: 'Blaze', evs: { spa: 252, spe: 252 }, nature: 'Modest' }),
      mon('Snorlax', ['seismictoss']),
      mon('Blissey', ['seismictoss']),
    ],
    p2: [mon('Charizard', ['splash', 'roost'], { ability: 'Blaze', evs: { spa: 252, spe: 252 }, nature: 'Modest' })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['beatUp', 'beatUpHitcount'],
  });

  // (5c) BEAT UP MIRROR at a SPEED TIE — BOTH actives Beat Up at equal speed: the action-order
  //      tie-shuffle + BOTH multi-strikes' per-strike Updates all draw. p1's stronger bench wins.
  S.push({
    id: 'beatup_mirror_tie',
    p1: [
      mon('Charizard', ['beatup', 'seismictoss'], { ability: 'Blaze', evs: { spa: 252, spe: 252 }, nature: 'Modest' }),
      mon('Slaking', ['seismictoss'], { evs: { atk: 252 } }),
      mon('Machamp', ['seismictoss'], { evs: { atk: 252 } }),
    ],
    p2: [
      mon('Charizard', ['beatup', 'roost'], { ability: 'Blaze', evs: { spa: 252, spe: 252 }, nature: 'Modest' }),
      mon('Magikarp', ['splash'], { level: 5 }),
    ],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['beatUp', 'beatUpHitcount'],
  });

  // ═══════════════════════ THUNDER ═══════════════════════

  // (6) THUNDER in RAIN (Drizzle foe): NEVER-MISS → the accuracy `random(100)` is SKIPPED
  //     (ONE FEWER draw) + 30% para. The foe survives (Soft-Boiled is draw-free) so the
  //     para + the foe's full-para roll + Quick Claw all draw; p1 grinds to a win.
  S.push({
    id: 'thunder_rain',
    p1: [mon('Zapdos', ['thunder', 'seismictoss'], { evs: { spa: 252, spe: 252 } })],
    p2: [mon('Blissey', ['splash', 'seismictoss'], { ability: 'Drizzle', evs: { hp: 4 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['thunder'],
  });

  // (7) THUNDER in SUN (Drought foe): base accuracy 50 → the accuracy roll DRAWS at a lower
  //     threshold (more misses). p1 grinds to a win.
  S.push({
    id: 'thunder_sun',
    p1: [mon('Zapdos', ['thunder', 'seismictoss'], { evs: { spa: 252, spe: 252 } })],
    p2: [mon('Blissey', ['splash', 'seismictoss'], { ability: 'Drought', evs: { hp: 4 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['thunder'],
  });

  // (8) THUNDER with NO weather: base accuracy 70 (the control). p1 grinds to a win.
  S.push({
    id: 'thunder_none',
    p1: [mon('Zapdos', ['thunder', 'seismictoss'], { evs: { spa: 252, spe: 252 } })],
    p2: [mon('Blissey', ['splash', 'seismictoss'], { ability: 'No Ability', evs: { hp: 4 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['thunder'],
  });

  // (9) THUNDER in SAND (Sand Stream foe): non-rain/sun weather does NOT touch Thunder →
  //     base accuracy 70 (proven-by-inference for hail too). p1 grinds to a win.
  S.push({
    id: 'thunder_sand',
    p1: [mon('Zapdos', ['thunder', 'seismictoss'], { evs: { spa: 252, spe: 252 } })],
    p2: [mon('Tyranitar', ['splash', 'seismictoss'], { ability: 'Sand Stream', evs: { hp: 4 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['thunder'],
  });

  // ═══════════════════════ WATER SPOUT ═══════════════════════

  // (10) WATER SPOUT at FULL HP: bp 150. p1 grinds a bulky foe to a win.
  S.push({
    id: 'waterspout_full_hp',
    p1: [mon('Kyogre', ['waterspout', 'seismictoss'], { evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['splash', 'roost'], { evs: { hp: 4 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['waterSpout'],
  });

  // (11) WATER SPOUT at LOW HP: the user injected to low HP → bp floors (the min-BP-1 edge is
  //      reachable at 1 HP). Same draws, smaller damage. p1 uses Seismic Toss to finish.
  S.push({
    id: 'waterspout_low_hp',
    p1: [mon('Kyogre', ['waterspout', 'seismictoss'], { evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['splash', 'roost'], { evs: { hp: 4 } })],
    inject: [{ side: 0, hp: 40 }], // low HP → floored bp
    // Turn 0: Water Spout at low HP (small damage). Then Seismic Toss the foe to a win.
    intent: (n) => (n === 0 ? { p1Want: 1, p2Want: 1 } : { p1Want: 2, p2Want: 1 }),
    require: ['waterSpout'],
  });

  // (12) WATER SPOUT into a SUBSTITUTE (full HP, bp 150): the variable BP hits the sub
  //      normally (break, no carry). Then p1 grinds to a win.
  S.push({
    id: 'waterspout_into_sub',
    p1: [mon('Kyogre', ['waterspout', 'seismictoss'], { evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252, spe: 252 } })],
    intent: (n) => (n === 0 ? { p1Want: 1, p2Want: 1 } : { p1Want: 1, p2Want: 2 }),
    require: ['waterSpout'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# movecoverage_batch4b_golden.txt — Gen-3 MOVE-COVERAGE BATCH 4b full-battle golden.');
  lines.push('# Per-decision STATE(+status+boosts+HP+SUB-HP)+SEED+first-mover differential to GAME-END.');
  lines.push('# Classes: BEAT UP (multi-strike stat swap) / THUNDER (weather accuracy) / WATER SPOUT');
  lines.push('#   (variable BP). The CURSE/WISH columns are always 0 here (reused DEC format); SUB-HP is live.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INJECT <id>  <json array of {side?,slot?,status?,hp?}>  ([] if none)');
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
    console.error('MOVECOVERAGE BATCH4b GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`BATCH4b GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('beat-up decisions', 'beatUp', 40);
  need('beat-up hitcount decisions', 'beatUpHitcount', 40);
  need('beat-up sub-break decisions', 'beatUpSubBreak', 5);
  need('thunder decisions', 'thunder', 40);
  need('thunder miss decisions', 'thunderMiss', 5);
  need('thunder para decisions', 'thunderPara', 5);
  need('water-spout decisions', 'waterSpout', 40);
  if (winRows < 40) { console.error(`BATCH4b GOLDEN: too few WIN rows (${winRows} < 40)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `movecoverage batch4b golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: beatUp=${corpus.beatUp || 0} hitcount=${corpus.beatUpHitcount || 0} subBreak=${corpus.beatUpSubBreak || 0} ` +
    `thunder=${corpus.thunder || 0} thunderMiss=${corpus.thunderMiss || 0} thunderPara=${corpus.thunderPara || 0} ` +
    `waterSpout=${corpus.waterSpout || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
