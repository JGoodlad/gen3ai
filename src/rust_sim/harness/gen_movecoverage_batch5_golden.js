// gen_movecoverage_batch5_golden.js — Gen-3 MOVE-COVERAGE BATCH 5 differential golden
// (`gen3_move_coverage_batch5_v1`): the REACTIVE FIXED-DAMAGE family (COUNTER / MIRROR
// COAT / ENDEAVOR), the VARIABLE-BP family (RETURN / FRUSTRATION / FLAIL / REVERSAL /
// LOW KICK), and SLEEP TALK (the move-sampler).
//
//   COUNTER / MIRROR COAT — a beforeTurnMove (order 5) volatile whose onStart RESETS
//       {slot:null, damage:0} each selection turn; the onDamage recorder (priority -101)
//       arms it with 2x each qualifying DIRECT foe Move hit (counter: Physical or bare
//       hiddenpower; mirrorcoat: Special and not hiddenpower), OVERWRITING per hit
//       (multihit -> 2x the LAST strike). Execution: un-armed -> a bare |move| line with
//       ZERO draws (no -fail); armed -> ONE accuracy draw (acc 100, NOT never-miss) then
//       type immunity (Counter Fighting->Ghost, MC Psychic->Dark -> `-immune`), NO
//       crit/damage rolls; the hit is landed=true (the in-tryMoveHit Update at a tie).
//       duration:1 -> a NO_ORDER/subOrder-2 residual duration handler (the mirror tie).
//   ENDEAVOR — onTry fails at pokemon.hp >= target.hp (EQUALITY INCLUDED, `|-fail|user`,
//       ZERO draws); else ONE accuracy draw, Normal->Ghost `-immune` after it, NO
//       crit/damage; amount = target.hp - user.hp (reads the MON's hp behind a sub, the
//       number lands on the SUB with no carry).
//   RETURN / FRUSTRATION / FLAIL / REVERSAL / LOW KICK — engine-computed BP (happiness /
//       the 48*hp/maxhp ladder / the weighthg ladder), DRAW-IDENTICAL to a plain
//       single-hit physical move (acc-100 drawn, crit 1/16, damage random(16), QC).
//   SLEEP TALK — usable only while ASLEEP (the slp cant prints, the move proceeds via
//       sleepUsable, skippedTime++); exactly ONE sample = random(pool_len) — drawn EVEN
//       at n=1 — over the slot-order pool of !nosleeptalk && !charge moves (NO pp
//       filter); a 0-PP pick -> |cant|nopp|<id> and STOP; else the picked move runs its
//       FULL normal draw chain ([from] Sleep Talk announce, its PP untouched); empty
//       pool -> [still] + -fail, zero draws; awake/wake-turn -> a SILENT fail; a
//       PREVIOUS-turn choicelock -> [still]+-fail before the sample. skippedTime is
//       restored (time += skipped) at switch-in — the slp `stage` column asserts it.
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios that each ISOLATE a branch, capturing initSeed + per-decision seedAfter, each
// active's species/hp/maxhp/fainted/status(+slp-time/tox-stage)/boosts/confusion +
// pokemon_left + CURSE + WISH + SUB-HP + FUTURE-PENDING + first mover + winner. The Rust
// test seeds a BattleState at initSeed and runs `run_full_battle` WITHOUT re-seeding.
//
// REUSES the batch-4c 44-field DEC format (CURSE/WISH/FUTURE columns asserted 0 by the
// Rust gate; the slp `stage` column carries the SLEEP COUNTER — the skippedTime-restore
// proof). INJECT gains an optional per-slot `pp` set (STATE-only, no PRNG).
//
// Output: tests/vectors/movecoverage_batch5_golden.txt
//
// Run:  node src/rust_sim/harness/gen_movecoverage_batch5_golden.js

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/movecoverage_batch5_golden.txt');
const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };

function mon(species, moves, opts = {}) {
  const m = {
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
  if (opts.happiness !== undefined) m.happiness = opts.happiness;
  return m;
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
  let x = 0x5b8e31c7 >>> 0;
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

// Scan the protocol log between two decision points for the BATCH-5 branch flags.
// Reactive outcomes are attributed by ADJACENCY: a |move|Counter line followed
// immediately by |-damage| = a landed return; by |-immune| = the immune branch; by
// anything else = the zero-draw un-armed fail.
function outcomesSince(log, fromIdx) {
  const out = {
    ctHit: false,       // a landed Counter return
    ctFail: false,      // an un-armed Counter fail (bare |move|, zero draws)
    ctImmune: false,    // Counter -> Ghost `-immune`
    mcHit: false,       // a landed Mirror Coat return
    mcFail: false,      // an un-armed Mirror Coat fail
    mcImmune: false,    // Mirror Coat -> Dark `-immune`
    evHit: false,       // a landed Endeavor (the delta)
    evFail: false,      // the hp >= target.hp `-fail`
    evImmune: false,    // Endeavor -> Ghost `-immune`
    vbHit: false,       // a landed variable-BP hit (Return/Frustration/Flail/Reversal/Low Kick)
    vbImmune: false,    // a variable-BP move into a Ghost `-immune`
    vbMiss: false,      // (accuracy is 100 -- never expected; a canary)
    stTalk: false,      // a |move|Sleep Talk line while ASLEEP (the sample turn)
    stCalled: false,    // a called move line ([from] Sleep Talk)
    stCalledRest: false,// the called move was Rest (the silent no-op)
    stStill: false,     // Sleep Talk [still]+fail (empty pool / choicelock)
    stNopp: false,      // |cant|...|nopp| (a 0-PP pick)
    stSilent: false,    // an AWAKE Sleep Talk (normal move line, nothing after)
    slpCant: false,     // |cant|...|slp
    subHit: false,      // a sub absorbed/broke
    beatUp: false,      // a Beat Up multi-strike ran (the MC last-strike scenario)
    struggleMove: false,// a Struggle ran (the counter-vs-struggle composition)
    koTurn: false,      // a faint in the window
  };
  const vbNames = /^(Return|Frustration|Flail|Reversal|Low Kick)$/;
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    // The adjacency read skips `|debug|` lines — gen3customgame is a debug-true
    // format, and a basePowerCallback (Flail/Reversal/Low Kick/Beat Up) interposes a
    // `|debug|BP: N` line between the `|move|` announce and its `|-damage|`.
    let ni = i + 1;
    while (log[ni] && log[ni].split('|')[1] === 'debug') ni++;
    const next = log[ni] ? log[ni].split('|') : [];
    const nextTag = next[1] || '';
    if (tag === 'move') {
      const name = p[3] || '';
      const attrs = (p[5] || '') + (p[6] || '');
      if (name === 'Counter') {
        if (nextTag === '-damage') out.ctHit = true;
        else if (nextTag === '-immune') out.ctImmune = true;
        else out.ctFail = true;
      }
      if (name === 'Mirror Coat') {
        if (nextTag === '-damage') out.mcHit = true;
        else if (nextTag === '-immune') out.mcImmune = true;
        else out.mcFail = true;
      }
      if (name === 'Endeavor') {
        if (nextTag === '-damage' || nextTag === '-activate' || nextTag === '-end') out.evHit = true;
        else if (nextTag === '-immune') out.evImmune = true;
        else if (nextTag === '-fail') out.evFail = true;
      }
      if (vbNames.test(name)) {
        if (nextTag === '-damage' || nextTag === '-crit' || nextTag === '-supereffective' ||
            nextTag === '-resisted' || nextTag === '-activate' || nextTag === '-end') out.vbHit = true;
        else if (nextTag === '-immune') out.vbImmune = true;
        else if ((p[5] || '') === '[miss]') out.vbMiss = true;
      }
      if (name === 'Sleep Talk') {
        if ((p[4] || '') === '' && attrs.includes('[still]')) out.stStill = true;
        else if (nextTag === 'cant' && (next[3] || '') === 'nopp') out.stNopp = true;
        else if (nextTag === 'move' && (next[5] || next[6] || '').includes('Sleep Talk')) out.stTalk = true;
        else out.stSilent = true;
      }
      if ((p[5] || '').includes('[from] Sleep Talk') || (p[6] || '').includes('[from] Sleep Talk')) {
        out.stCalled = true;
        out.stTalk = true;
        if (name === 'Rest') out.stCalledRest = true;
      }
      if (name === 'Struggle') out.struggleMove = true;
      if (name === 'Beat Up') out.beatUp = true;
    }
    if (tag === 'cant') {
      if ((p[3] || '') === 'slp') out.slpCant = true;
      if ((p[3] || '') === 'nopp') out.stNopp = true;
    }
    if ((tag === '-activate' && (p[3] || '') === 'Substitute') ||
        (tag === '-end' && (p[3] || '') === 'Substitute')) out.subHit = true;
    if (tag === 'faint') out.koTurn = true;
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

  // Optional one-time post-start injection (status / HP / per-slot PP). STATE only
  // (no PRNG) so seed parity is unaffected.
  if (sc.inject) {
    const battle = stream.battle;
    for (const inj of sc.inject) {
      if (inj.side !== undefined) {
        const idx = inj.slot !== undefined ? inj.slot : 0;
        const m = idx === 0 ? battle.sides[inj.side].active[0] : battle.sides[inj.side].pokemon[idx];
        if (inj.status) m.setStatus(inj.status, m, null, true);
        if (inj.hp !== undefined) m.hp = inj.hp;
        if (inj.pp) m.moveSlots[inj.pp.moveSlot].pp = inj.pp.val;
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

  // ═══════════════════════ COUNTER ═══════════════════════

  // (1) Counter baseline: 2x a landed physical / a splash-turn fail / a special fail —
  //     the foe cycles Drill Peck / Splash / Surf; Snorlax alternates Counter and
  //     Seismic Toss (chip → game end).
  S.push({
    id: 'ct_basic',
    p1: [mon('Snorlax', ['counter', 'seismictoss'], { evs: { hp: 252 } })],
    p2: [mon('Skarmory', ['drillpeck', 'splash', 'surf'], { evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: (n % 3 === 2) ? 2 : 1, p2Want: (n % 3) + 1 }),
    require: ['ctHit', 'ctFail'],
  });

  // (2) Counter INTO a Ghost — Gengar's Shadow Ball (Ghost = PHYSICAL in gen3) arms it;
  //     the Fighting return fire is `-immune` (acc drawn then immune). Crunch (modeled
  //     secondary) is the win path.
  S.push({
    id: 'ct_ghost_immune',
    p1: [mon('Machamp', ['counter', 'crunch'], { evs: { hp: 252, atk: 252 } })],
    p2: [mon('Gengar', ['shadowball', 'splash'], { ability: 'Levitate', evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: (n % 2 === 0) ? 1 : 2, p2Want: (n % 2 === 0) ? 1 : 2 }),
    require: ['ctImmune'],
    forbid: ['ctHit'],
  });

  // (3) Counter vs a SUBSTITUTE-absorbed hit: the foe's Drill Peck lands on OUR sub →
  //     NOT recorded → the counter fails zero-draw. (Alternating sub/counter.)
  S.push({
    id: 'ct_sub_not_recorded',
    p1: [mon('Snorlax', ['counter', 'substitute', 'seismictoss'], { evs: { hp: 252 } })],
    p2: [mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: (n % 3 === 0) ? 2 : ((n % 3 === 1) ? 1 : 3), p2Want: (n % 2) + 1 }),
    require: ['ctFail', 'subHit'],
  });

  // (4) Counter vs SEISMIC TOSS (fixed-damage, category Physical) — armed, returns 200.
  S.push({
    id: 'ct_seismic',
    p1: [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    p2: [mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: (n % 4 === 3) ? 2 : 1, p2Want: (n % 4 === 3) ? 2 : 1 }),
    require: ['ctHit'],
  });

  // (5) Counter vs STRUGGLE — the foe exhausts a 1-slot Tackle (35 raw PP → inject 2)
  //     then Struggles (typeless PHYSICAL) into the counter.
  S.push({
    id: 'ct_struggle',
    p1: [mon('Snorlax', ['counter', 'seismictoss'], { evs: { hp: 252 } })],
    p2: [mon('Skarmory', ['tackle'], { evs: { hp: 252 } })],
    inject: [{ side: 1, pp: { moveSlot: 0, val: 2 } }],
    intent: (n) => ({ p1Want: (n % 3 === 2) ? 2 : 1, p2Want: 1 }),
    require: ['ctHit', 'struggleMove'],
  });

  // (6) COUNTER MIRROR at an equal speed — both-counter turns (both fail, the +4 draw
  //     delta: the order-5 pair tie + 2 trailing Updates + the residual duration tie)
  //     interleaved with seismic-toss turns (a LANDED counter at the tie fires the
  //     in-tryMoveHit Update — the CT probe's 13-draw turn).
  S.push({
    id: 'ct_mirror_tie',
    p1: [mon('Snorlax', ['counter', 'seismictoss'], { evs: { hp: 252 } })],
    p2: [mon('Snorlax', ['counter', 'seismictoss'], { evs: { hp: 252 } })],
    intent: (n) => ({
      p1Want: (n % 3 === 0) ? 1 : ((n % 3 === 1) ? 2 : 1),
      p2Want: (n % 3 === 0) ? 1 : ((n % 3 === 1) ? 1 : 2),
    }),
    require: ['ctHit', 'ctFail'],
  });

  // ═══════════════════════ MIRROR COAT ═══════════════════════

  // (7) Mirror Coat baseline: 2x a landed special (Psychic) / a physical fail (Return)
  //     / a no-damage fail (Splash).
  S.push({
    id: 'mc_basic',
    p1: [mon('Blissey', ['mirrorcoat', 'seismictoss'], { evs: { hp: 252 } })],
    p2: [mon('Alakazam', ['psychic', 'return', 'splash'], { evs: { hp: 252, spa: 252 } })],
    intent: (n) => ({ p1Want: (n % 4 === 3) ? 2 : 1, p2Want: (n % 3) + 1 }),
    require: ['mcHit', 'mcFail'],
  });

  // (8) Mirror Coat INTO a Dark — Tyranitar's Crunch (Dark = SPECIAL in gen3) arms it;
  //     the Psychic return fire is `-immune`.
  S.push({
    id: 'mc_dark_immune',
    p1: [mon('Blissey', ['mirrorcoat', 'seismictoss'], { evs: { hp: 252 } })],
    p2: [mon('Tyranitar', ['crunch', 'splash'], { ability: 'No Ability', evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: (n % 2 === 0) ? 1 : 2, p2Want: (n % 2 === 0) ? 1 : 2 }),
    require: ['mcImmune'],
    forbid: ['mcHit'],
  });

  // (9) Mirror Coat vs BEAT UP — the multi-strike arms it with 2x the LAST strike
  //     (Smeargle is Normal, so the Psychic return LANDS). The multihit-overwrite pin.
  S.push({
    id: 'mc_beatup_last_strike',
    p1: [mon('Blissey', ['mirrorcoat', 'seismictoss'], { evs: { hp: 252 } })],
    p2: [
      mon('Smeargle', ['beatup', 'splash'], { evs: { hp: 252 } }),
      mon('Snorlax', ['splash'], { evs: { hp: 252 } }),
      mon('Blissey', ['splash'], { evs: { hp: 252 } }),
    ],
    intent: (n) => ({ p1Want: (n % 3 === 2) ? 2 : 1, p2Want: (n % 3 === 2) ? 2 : 1 }),
    require: ['mcHit', 'beatUp'],
  });

  // ═══════════════════════ ENDEAVOR ═══════════════════════

  // (10) Endeavor baseline: the foe's Drill Peck chips Swellow low → Endeavor sets the
  //      Snorlax to Swellow's hp; the follow-up Endeavor at equal hp FAILS (`-fail`,
  //      zero draws). Return is the win path.
  S.push({
    id: 'ev_delta_and_equality_fail',
    p1: [mon('Swellow', ['endeavor', 'return'], { evs: { hp: 252, atk: 252 }, happiness: 255 })],
    p2: [mon('Snorlax', ['drillpeck', 'splash'], { evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: (n % 3 === 2) ? 2 : 1, p2Want: (n % 2) + 1 }),
    require: ['evHit', 'evFail'],
  });

  // (11) Endeavor INTO a Ghost — `-immune` after the accuracy draw. Steel Wing (a
  //      modeled self-boost-secondary move) is the win path.
  S.push({
    id: 'ev_ghost_immune',
    p1: [mon('Swellow', ['endeavor', 'steelwing'], { evs: { hp: 252, atk: 252 } })],
    // Seismic Toss (Fighting) chips the Normal/Flying Swellow below Gengar's hp so the
    // Endeavor onTry PASSES and the Normal->Ghost immunity branch is reached (Night
    // Shade, Ghost, would be IMMUNE into Swellow and never chip).
    p2: [mon('Gengar', ['seismictoss', 'splash'], { ability: 'Levitate', evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: (n % 2 === 0) ? 1 : 2, p2Want: (n % 2 === 0) ? 1 : 2 }),
    require: ['evImmune'],
    forbid: ['evHit'],
  });

  // (12) Endeavor into a SUBSTITUTE — the delta reads the MON's hp, the number lands on
  //      the SUB (break, no carry).
  S.push({
    id: 'ev_into_sub',
    p1: [mon('Swellow', ['endeavor', 'return'], { evs: { hp: 252, atk: 252 }, happiness: 255 })],
    p2: [mon('Snorlax', ['substitute', 'drillpeck', 'splash'], { evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: (n % 3 === 2) ? 2 : 1, p2Want: (n % 3) + 1 }),
    require: ['evHit', 'subHit'],
  });

  // ═══════════════════════ VARIABLE BP ═══════════════════════

  // (13) RETURN at both happiness extremes — the h255 lead (bp 102) then a pivot to the
  //      h3 bench mon (bp 1; `|| 1`-clamped, a normal 4-draw hit, NOT a fail). The foe
  //      chips with Seismic Toss.
  S.push({
    id: 'vb_return_happiness',
    p1: [
      mon('Tauros', ['return', 'splash'], { evs: { hp: 252, atk: 252 }, happiness: 255 }),
      mon('Tauros', ['return', 'splash'], { evs: { hp: 252, atk: 252 }, happiness: 3 }),
    ],
    p2: [mon('Skarmory', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    intent: (n, battle) => {
      const b = benchSlot(battle, 0);
      if (n % 7 === 3 && b) return { p1Switch: b, p2Want: 1 };
      return { p1Want: 1, p2Want: (n % 2) + 1 };
    },
    require: ['vbHit'],
  });

  // (14) FRUSTRATION at both extremes — h0 (bp 102) lead + h252 (bp 1) bench.
  S.push({
    id: 'vb_frustration_happiness',
    p1: [
      mon('Tauros', ['frustration', 'splash'], { evs: { hp: 252, atk: 252 }, happiness: 0 }),
      mon('Tauros', ['frustration', 'splash'], { evs: { hp: 252, atk: 252 }, happiness: 252 }),
    ],
    p2: [mon('Skarmory', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    intent: (n, battle) => {
      const b = benchSlot(battle, 0);
      if (n % 7 === 3 && b) return { p1Switch: b, p2Want: 1 };
      return { p1Want: 1, p2Want: (n % 2) + 1 };
    },
    require: ['vbHit'],
  });

  // (15) FLAIL ladder — Blissey's Seismic Toss chips Snorlax in EXACT 100-HP steps, so
  //      the flail BP crosses the 48*hp/maxhp bands deterministically across the battle
  //      (each band boundary shows in the foe's HP delta at an un-changed seed).
  S.push({
    id: 'vb_flail_ladder',
    p1: [mon('Snorlax', ['flail', 'splash'], { evs: { hp: 252, atk: 252 } })],
    p2: [mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['vbHit'],
  });

  // (16) REVERSAL + LOW KICK + the Fighting->Ghost immunity — Machamp Low-Kicks the
  //      HEAVY Snorlax (4600 hg -> bp 120) then the foe pivots to Gengar (`-immune`,
  //      405 hg irrelevant) and back; Reversal mixes the hp-ladder in.
  S.push({
    id: 'vb_lowkick_reversal',
    p1: [mon('Machamp', ['lowkick', 'reversal'], { ability: 'Guts', evs: { hp: 252, atk: 252 } })],
    p2: [
      mon('Snorlax', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
      mon('Gengar', ['nightshade', 'splash'], { ability: 'Levitate', evs: { hp: 252 } }),
    ],
    intent: (n, battle) => {
      const b = benchSlot(battle, 1);
      if (n % 5 === 2 && b) return { p1Want: 1, p2Switch: b };
      return { p1Want: (n % 2) + 1, p2Want: 1 };
    },
    require: ['vbHit', 'vbImmune'],
  });

  // (17) LOW KICK weight ladder — a light foe (Pichu 20 hg -> bp 20) vs a mid foe
  //      (Gengar-class 405 -> 60) vs heavy (Snorlax 4600 -> 120): distinct damage at
  //      the SAME draw count (draw-neutrality on the weight axis).
  S.push({
    id: 'vb_lowkick_weights',
    p1: [mon('Blissey', ['lowkick', 'seismictoss'], { evs: { hp: 252, atk: 252 } })],
    p2: [
      mon('Snorlax', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
      mon('Wobbuffet', ['splash'], { ability: 'No Ability', evs: { hp: 252 } }),
      mon('Pichu', ['splash'], { evs: { hp: 252 } }),
    ],
    intent: (n, battle) => {
      const b = benchSlot(battle, 1);
      if (n % 6 === 3 && b) return { p1Want: 1, p2Switch: b };
      return { p1Want: (n % 3 === 2) ? 2 : 1, p2Want: 1 };
    },
    require: ['vbHit'],
  });

  // ═══════════════════════ SLEEP TALK ═══════════════════════

  // (18) RESTTALK — the classic set: Rest when hurt, Sleep Talk while asleep (the
  //      sample over [rest, bodyslam, curse]: a picked Rest silently no-ops, a picked
  //      Body Slam runs its full chain incl. the par secondary, a picked Curse rolls
  //      its selfDrops random(100)). The slp `stage` column asserts the counter.
  S.push({
    id: 'st_resttalk',
    p1: [mon('Snorlax', ['sleeptalk', 'rest', 'bodyslam', 'curse'], { evs: { hp: 252, atk: 252 } })],
    // Skarmory: bulky vs Body Slam (resisted + high Def) and chips hard enough
    // (Drill Peck + Seismic Toss) that the Rest→Sleep-Talk loop actually runs
    // (a frail foe dies before the sleeper ever needs Rest).
    p2: [mon('Skarmory', ['drillpeck', 'seismictoss'], { evs: { hp: 252, def: 252 } })],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      if (me && me.status === 'slp') return { p1Want: 1, p2Want: 1 };
      if (me && me.hp < me.maxhp * 0.6) return { p1Want: 2, p2Want: 1 };
      return { p1Want: 3, p2Want: (n % 2) + 1 };
    },
    require: ['stTalk', 'stCalled', 'slpCant'],
  });

  // (19) RESTTALK MIRROR at an equal speed — the sample + called-move draws compose
  //      with the tie shuffles (the called landed move's in-tryMoveHit Update at a tie).
  S.push({
    id: 'st_resttalk_mirror_tie',
    p1: [mon('Snorlax', ['sleeptalk', 'rest', 'bodyslam'], { evs: { hp: 252, atk: 252 } })],
    p2: [mon('Snorlax', ['sleeptalk', 'rest', 'bodyslam'], { evs: { hp: 252, atk: 252 } })],
    intent: (n, battle) => {
      const pick = (side) => {
        const me = battle.sides[side].active[0];
        if (me && me.status === 'slp') return 1;
        if (me && me.hp < me.maxhp * 0.6) return 2;
        return 3;
      };
      return { p1Want: pick(0), p2Want: pick(1) };
    },
    require: ['stTalk', 'stCalled'],
  });

  // (20) EMPTY POOL — [sleeptalk, solarbeam]: solarbeam is `charge`-flagged, sleeptalk
  //      excludes itself → the pool is EMPTY → `[still]` + `-fail`, zero draws. The
  //      awake turns select Sleep Talk too (the SILENT awake fail realizes). Breloom's
  //      Spore supplies the sleep; Seismic Toss grinds to the end.
  S.push({
    id: 'st_empty_pool',
    p1: [mon('Snorlax', ['sleeptalk', 'solarbeam'], { evs: { hp: 252 } })],
    p2: [mon('Breloom', ['spore', 'seismictoss'], { evs: { hp: 252 } })],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      const asleep = me && me.status === 'slp';
      return { p1Want: 1, p2Want: asleep ? 2 : 1 };
    },
    require: ['stStill', 'stSilent', 'slpCant'],
    forbid: ['stCalled'],
  });

  // (21) CHOICE-LOCKED Sleep Talk — a Choice Band RestTalker: the FIRST Sleep Talk of a
  //      lock samples + executes; every LATER one fails `[still]`+`-fail` BEFORE the
  //      sample (the lock records Sleep Talk itself). The lock clears on switch-out
  //      (the pivot mixes fresh locks in).
  S.push({
    id: 'st_choicelock',
    p1: [
      mon('Snorlax', ['sleeptalk', 'bodyslam', 'rest'], { item: 'Choice Band', evs: { hp: 252, atk: 252 } }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    ],
    p2: [mon('Breloom', ['spore', 'seismictoss', 'splash'], { evs: { hp: 252 } })],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      const asleep = me && me.status === 'slp';
      if (me && me.species.name === 'Snorlax') {
        if (n % 9 === 7) {
          const b = benchSlot(battle, 0);
          if (b) return { p1Switch: b, p2Want: 2 };
        }
        return { p1Want: 1, p2Want: asleep ? 2 : 1 };
      }
      if (n % 9 === 8) {
        const b = benchSlot(battle, 0);
        if (b) return { p1Switch: b, p2Want: 2 };
      }
      return { p1Want: 1, p2Want: 2 };
    },
    require: ['stTalk', 'stStill'],
  });

  // (22) 0-PP PICK — the pool is exactly [bodyslam] (n=1: the sample STILL draws) with
  //      its PP injected to 0 → `|cant|…|nopp|bodyslam`, the turn wasted.
  S.push({
    id: 'st_nopp_pick',
    p1: [mon('Snorlax', ['sleeptalk', 'bodyslam'], { evs: { hp: 252 } })],
    p2: [mon('Breloom', ['spore', 'seismictoss'], { evs: { hp: 252 } })],
    inject: [{ side: 0, pp: { moveSlot: 1, val: 0 } }],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      const asleep = me && me.status === 'slp';
      return { p1Want: 1, p2Want: asleep ? 2 : 1 };
    },
    require: ['stNopp', 'slpCant'],
    forbid: ['stCalled'],
  });

  // (23) skippedTime RESTORE — the RestTalker sleep-talks twice (time 3 → 1,
  //      skipped 2), pivots out and back: the slp `stage` column reads 3 again on
  //      re-entry (the switch-in restore), so the later wake timing shifts.
  S.push({
    id: 'st_skipped_time_switch',
    p1: [
      mon('Snorlax', ['sleeptalk', 'rest', 'bodyslam'], { evs: { hp: 252, atk: 252 } }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    ],
    p2: [mon('Skarmory', ['drillpeck', 'splash', 'seismictoss'], { evs: { hp: 252 } })],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      if (me && me.species.name === 'Snorlax') {
        if (me.status === 'slp') {
          const t = me.statusState ? (me.statusState.time | 0) : 0;
          const skipped = me.statusState ? (me.statusState.skippedTime | 0) : 0;
          // After two sleep-talks (skipped >= 2), pivot out while still asleep.
          if (skipped >= 2 && t >= 1) {
            const b = benchSlot(battle, 0);
            if (b) return { p1Switch: b, p2Want: 1 };
          }
          return { p1Want: 1, p2Want: 1 };
        }
        if (me.hp < me.maxhp * 0.7) return { p1Want: 2, p2Want: 1 };
        return { p1Want: 3, p2Want: (n % 2) + 1 };
      }
      // Blissey stalls a turn or two then hands back to the sleeper.
      if (n % 3 === 2) {
        const b = benchSlot(battle, 0);
        if (b) return { p1Switch: b, p2Want: 2 };
      }
      return { p1Want: 1, p2Want: 2 };
    },
    require: ['stTalk', 'slpCant'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# movecoverage_batch5_golden.txt — Gen-3 MOVE-COVERAGE BATCH 5 full-battle golden.');
  lines.push('# Per-decision STATE(+status+slp-time+boosts+HP+SUB-HP)+SEED+first-mover differential');
  lines.push('# to GAME-END. Classes: COUNTER / MIRROR COAT / ENDEAVOR (reactive fixed damage),');
  lines.push('#   RETURN / FRUSTRATION / FLAIL / REVERSAL / LOW KICK (variable BP), SLEEP TALK.');
  lines.push('# REUSES the batch-3/4/4c 44-field DEC format (CURSE/WISH/FUTURE columns stay 0).');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INJECT <id>  <json array of {side?,slot?,status?,hp?,pp:{moveSlot,val}?}>  ([] if none)');
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
    console.error('MOVECOVERAGE BATCH5 GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`BATCH5 GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('landed Counter returns', 'ctHit', 40);
  need('un-armed Counter fails', 'ctFail', 40);
  need('Counter->Ghost immunes', 'ctImmune', 20);
  need('landed Mirror Coat returns', 'mcHit', 40);
  need('un-armed Mirror Coat fails', 'mcFail', 20);
  need('Mirror Coat->Dark immunes', 'mcImmune', 20);
  need('Beat-Up-armed MC decisions', 'beatUp', 20);
  need('landed Endeavors', 'evHit', 40);
  need('Endeavor equality fails', 'evFail', 20);
  need('Endeavor->Ghost immunes', 'evImmune', 20);
  need('variable-BP hits', 'vbHit', 200);
  need('variable-BP Ghost immunes', 'vbImmune', 20);
  need('Sleep Talk sample turns', 'stTalk', 100);
  need('called-move executions', 'stCalled', 80);
  need('called Rest no-ops', 'stCalledRest', 10);
  need('empty-pool/choicelock [still] fails', 'stStill', 40);
  need('0-PP nopp cants', 'stNopp', 20);
  need('awake silent fails', 'stSilent', 20);
  need('counter-vs-struggle turns', 'struggleMove', 10);
  if (winRows < 40) { console.error(`BATCH5 GOLDEN: too few WIN rows (${winRows} < 40)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `movecoverage batch5 golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: ctHit=${corpus.ctHit || 0} ctFail=${corpus.ctFail || 0} ctImmune=${corpus.ctImmune || 0} ` +
    `mcHit=${corpus.mcHit || 0} mcFail=${corpus.mcFail || 0} mcImmune=${corpus.mcImmune || 0} beatUp=${corpus.beatUp || 0} ` +
    `evHit=${corpus.evHit || 0} evFail=${corpus.evFail || 0} evImmune=${corpus.evImmune || 0} ` +
    `vbHit=${corpus.vbHit || 0} vbImmune=${corpus.vbImmune || 0} vbMiss=${corpus.vbMiss || 0} ` +
    `stTalk=${corpus.stTalk || 0} stCalled=${corpus.stCalled || 0} stCalledRest=${corpus.stCalledRest || 0} ` +
    `stStill=${corpus.stStill || 0} stNopp=${corpus.stNopp || 0} stSilent=${corpus.stSilent || 0} ` +
    `struggle=${corpus.struggleMove || 0} subHit=${corpus.subHit || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
