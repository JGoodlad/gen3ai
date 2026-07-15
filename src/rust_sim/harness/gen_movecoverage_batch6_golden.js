// gen_movecoverage_batch6_golden.js — Gen-3 MOVE-COVERAGE BATCH 6 differential golden
// (`gen3_move_coverage_batch6_v1`): the FINAL UNMODELED tail — twelve moves.
//
//   GROUP A (volatile locks + reactive):
//     ENCORE — acc-100 draw + the durationCallback random(3,7) INSIDE addVolatile
//         (already-encored fails BEFORE the durationCallback — accuracy only; the
//         no-lastMove / failencore / 0-PP-lastMove fails draw BOTH), stored =
//         willMove(target) ? rolled : rolled+1 (the Disable branch), the onOverrideAction
//         execution override (a queued DIFFERENT move runs AS the encored move, the
//         ENCORED slot's PP deducts), the order-10/subOrder-14 residual tick + the
//         0-PP early `-end`.
//     DESTINY BOND — a ZERO-draw cast (re-cast succeeds draw-free); the volatile
//         persists until the user's NEXT move attempt (onBeforeMove −1 / onMoveAborted);
//         a FOE-Move KO while up → the killer faints too (|faint| victim → -activate →
//         |faint| killer); a residual KO does NOT trigger; both-last-mons → the gen-3 TIE.
//     ENDURE — the protect/stall family (priority 4): first use draw-free, consecutive
//         uses roll the SHARED randomChance(1,counter) ladder 2→4→8; survive any MOVE
//         damage at 1 HP (incl. fixed damage); residual damage still kills; a foe
//         SWITCH fails it via the willAct gate; a SUCCESS turn adds the endure+stall
//         intra-mon residual duration tie (ONE shuffle at ANY speed).
//   GROUP B (field/trap):
//     PERISH SONG — draw-free in EVERY branch; all actives (incl. the caster) get
//         perish3 → 2 → 1 → faint at 0 (the order-12 residual, LAST in the ladder);
//         Soundproof immune (the re-cast with >=1 immune is a SILENT success; the
//         all-counted re-cast fails [still]); switch-out clears; Baton Pass PASSES it.
//     MEAN LOOK / SPIDER WEB / BLOCK — draw-free linked trap volatiles: the target is
//         FIRM-trapped (trapped:true request, no maybeTrapped) until the TRAPPER leaves
//         ANY way; a Ghost IS trapped; Baton Pass moves the mon but PASSES the trap;
//         a phaze drags the trapped mon normally (and the drag clears the link); a
//         SUBSTITUTE blocks; re-application fails.
//   GROUP C (utility/self) — ALL draw-free in every branch:
//     BELLY DRUM (pay floor(maxhp/2), SET atk to +6; fails at 2*hp<=maxhp or atk>=6),
//     CHARGE (×2 the next Electric move's BP; consumed by the user's NEXT move attempt
//         of any kind; NO gen3 SpD boost), MEMENTO (never-miss self-faint + foe
//         −2 Atk/−2 SpA; blocked by Protect/Sub → NO faint; gen3 faint-cancels-all +
//         no Quick Claw on the landed turn), MIMIC (copy the target's lastMove over the
//         Mimic slot, pp min(5, base), maxpp calculatePP(copied,3); reverts on
//         switch-out; fails on sub/no-lastMove/failmimic/already-known), PAIN SPLIT
//         (floor-average both actives' HP, each clamped at its own maxhp; sub blocks),
//     PSYCH UP (copy ALL boost stages verbatim; NO protect flag; bypasssub).
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios that each ISOLATE a branch, capturing initSeed + per-decision seedAfter, each
// active's species/hp/maxhp/fainted/status(+slp-time/tox-stage)/boosts/confusion +
// pokemon_left + CURSE + WISH + SUB-HP + FUTURE-PENDING + ENCORE + PERISH + TRAPPED +
// first mover + winner. The Rust test seeds a BattleState at initSeed and runs
// `run_full_battle` WITHOUT re-seeding.
//
// EXTENDS the batch-4c/5 44-field DEC format with SIX appended columns (→ 50 fields):
//   p1Encore p1Perish p1Trapped  p2Encore p2Perish p2Trapped
// (encore = the volatile's remaining duration, perish = the counter's remaining
// duration, trapped = the live `trapped` VOLATILE presence — see trappedOf: the
// engine's `pokemon.trapped` flag goes stale at mid-turn faint-pause boundaries).
//
// Output: tests/vectors/movecoverage_batch6_golden.txt
//
// Run:  node src/rust_sim/harness/gen_movecoverage_batch6_golden.js

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/movecoverage_batch6_golden.txt');
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
  let x = 0x6c1d92e5 >>> 0;
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

function encoreOf(a) {
  return a && a.volatiles && a.volatiles['encore'] ? (a.volatiles['encore'].duration | 0) : 0;
}

function perishOf(a) {
  return a && a.volatiles && a.volatiles['perishsong'] ? (a.volatiles['perishsong'].duration | 0) : 0;
}

function trappedOf(a) {
  // The trap-move `trapped` VOLATILE presence — NOT the engine's `pokemon.trapped`
  // flag, which is recomputed only at endTurn and goes STALE at a mid-turn
  // faint-pause boundary (probe MC91 dec5: the volatile is gone the moment the
  // trapper faints, but the flag still reads true until the next endTurn). The
  // volatile is the live state the port models (`MonState::trapped_by`), so the
  // column stays comparable at EVERY boundary.
  return a && a.volatiles && a.volatiles['trapped'] ? 1 : 0;
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
      encore: 0, perish: 0, trapped: 0,
    };
  }
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status, stage, left: side.pokemonLeft,
    boosts: boostsOf(a), confusion: confusionOf(a),
    curse: curseOf(a), wish, subHp: subHpOf(a), future,
    encore: encoreOf(a), perish: perishOf(a), trapped: trappedOf(a),
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

// Scan the protocol log between two decision points for the BATCH-6 branch flags.
function outcomesSince(log, fromIdx) {
  const out = {
    enLand: false,       // |-start|…|Encore (a landed encore)
    enOverride: false,   // a queued different move executed as the encored one — flagged
                         // when a |-start|Encore was seen EARLIER (state asserts cover it)
    enFail: false,       // Encore [still] + -fail (any fail form)
    enEnd: false,        // |-end|…|Encore (expiry OR the 0-PP early end)
    dbCast: false,       // |-singlemove|…|Destiny Bond
    dbMutual: false,     // |-activate|…|move: Destiny Bond (the mutual-faint chain)
    edUp: false,         // |-singleturn|…|move: Endure
    edSurvive: false,    // |-activate|…|move: Endure (the survive-at-1 clamp fired)
    edStallFail: false,  // Endure [still] + -fail (a failed stall roll)
    psCast: false,       // |-fieldactivate|move: Perish Song
    psImmune: false,     // |-immune|…|[from] ability: Soundproof on a Perish Song turn
    psTick: false,       // |-start|…|perish1/2/3 (a residual tick)
    psFaint: false,      // |-start|…|perish0 (the 0-tick faint)
    psFail: false,       // Perish Song [still] + -fail (all-already-counted)
    psSilent: false,     // a Perish Song |move| with -immune but NO fieldactivate/fail
    tmTrap: false,       // |-activate|…|trapped (a trap-move volatile landed)
    tmFail: false,       // Mean Look/Spider Web/Block [still] + -fail (sub / re-apply)
    bdSet: false,        // |-setboost|…|atk|6|[from] move: Belly Drum
    bdFail: false,       // Belly Drum [still] + -fail
    chSet: false,        // |-start|…|Charge
    meFaint: false,      // Memento landed (the user's |faint| follows)
    meDrop: false,       // the memento -unboost lines
    meBlocked: false,    // Memento into a Protect (-activate Protect; NO faint)
    miCopy: false,       // |-activate|…|move: Mimic|<Move>
    miFail: false,       // Mimic [still] + -fail
    pspHit: false,       // |-sethp|…|[from] move: Pain Split
    pspFail: false,      // Pain Split [still] + -fail (sub block)
    puCopy: false,       // |-copyboost|…|[from] move: Psych Up
    bpPass: false,       // a |switch|…|[from] Baton Pass
    subHit: false,       // a sub absorbed/broke
    koTurn: false,       // a faint in the window
    tie: false,
  };
  let sawEncoreStartBefore = fromIdx; // (unused adjacency marker)
  const trapNames = /^(Mean Look|Spider Web|Block)$/;
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    let ni = i + 1;
    while (log[ni] && log[ni].split('|')[1] === 'debug') ni++;
    const next = log[ni] ? log[ni].split('|') : [];
    const nextTag = next[1] || '';
    if (tag === 'move') {
      const name = p[3] || '';
      const attrs = (p[5] || '') + (p[6] || '');
      const still = attrs.includes('[still]');
      if (name === 'Encore') {
        if (nextTag === '-start' && (next[3] || '') === 'Encore') out.enLand = true;
        else if (still || nextTag === '-fail') out.enFail = true;
      }
      if (name === 'Destiny Bond') out.dbCast = true;
      if (name === 'Endure' && still) out.edStallFail = true;
      if (name === 'Perish Song') {
        if (still || nextTag === '-fail') out.psFail = true;
        else if (nextTag === '-immune') {
          // immune first — the silent-success vs fieldactivate split resolves below
          let j = ni;
          let sawField = false;
          while (log[j]) {
            const q = log[j].split('|');
            if (q[1] === '-fieldactivate') { sawField = true; break; }
            if (q[1] === 'move' || q[1] === 'turn' || q[1] === 'upkeep') break;
            j++;
          }
          if (!sawField) out.psSilent = true;
        }
      }
      if (trapNames.test(name) && (still || nextTag === '-fail')) out.tmFail = true;
      if (name === 'Belly Drum' && (still || nextTag === '-fail')) out.bdFail = true;
      if (name === 'Memento') {
        if (nextTag === '-activate' && (next[3] || '') === 'Protect') out.meBlocked = true;
        else if (nextTag === '-unboost' || nextTag === '-fail' || nextTag === 'faint') {
          if (nextTag !== '-fail') out.meFaint = true;
        }
      }
      if (name === 'Mimic' && (still || (nextTag === '-fail'))) out.miFail = true;
      if (name === 'Pain Split' && (still || nextTag === '-fail')) out.pspFail = true;
    }
    if (tag === '-start') {
      const what = p[3] || '';
      if (what === 'Encore') out.enLand = true;
      if (what === 'Charge') out.chSet = true;
      if (/^perish[123]$/.test(what)) out.psTick = true;
      if (what === 'perish0') out.psFaint = true;
    }
    if (tag === '-end' && (p[3] || '') === 'Encore') out.enEnd = true;
    if (tag === '-singlemove' && (p[3] || '') === 'Destiny Bond') out.dbCast = true;
    if (tag === '-singleturn' && (p[3] || '') === 'move: Endure') out.edUp = true;
    if (tag === '-activate') {
      const what = p[3] || '';
      if (what === 'move: Destiny Bond') out.dbMutual = true;
      if (what === 'move: Endure') out.edSurvive = true;
      if (what === 'trapped') out.tmTrap = true;
      if (what === 'move: Mimic') out.miCopy = true;
      if (what === 'Substitute') out.subHit = true;
    }
    if (tag === '-fieldactivate' && (p[2] || '') === 'move: Perish Song') out.psCast = true;
    if (tag === '-immune' && (p[3] || '').includes('Soundproof')) out.psImmune = true;
    if (tag === '-setboost' && (p[5] || '').includes('Belly Drum')) out.bdSet = true;
    if (tag === '-unboost' && log.slice(Math.max(fromIdx, i - 3), i).some((l) => l.includes('|Memento|'))) out.meDrop = true;
    if (tag === '-sethp') out.pspHit = true;
    if (tag === '-copyboost') out.puCopy = true;
    if (tag === 'switch' && (p[5] || p[6] || '').includes('Baton Pass')) out.bpPass = true;
    if (tag === '-end' && (p[3] || '') === 'Substitute') out.subHit = true;
    if (tag === 'faint') out.koTurn = true;
    if (tag === 'tie') out.tie = true;
  }
  void sawEncoreStartBefore;
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

function isTrapped(battle, side) {
  const a = battle.sides[side].active[0];
  return !!(a && a.trapped);
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

  // ═══════════════════════ ENCORE ═══════════════════════

  // (1) FASTER encore user (the target has NOT moved yet → stored = rolled): Jolteon
  //     encores Snorlax. The first-turn encore (no lastMove) FAILS with acc+duration
  //     draws; a landed encore locks Snorlax (the queued OTHER move is OVERRIDDEN —
  //     EN7); re-encore into an already-encored foe fails accuracy-only; the lock
  //     expires at the residual. Thunderbolt grinds to the end.
  S.push({
    id: 'en_faster_lock',
    p1: [mon('Jolteon', ['encore', 'thunderbolt'], { evs: { hp: 252, spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['splash', 'bodyslam'], { evs: { hp: 252 } })],
    intent: (n, battle) => {
      const foe = battle.sides[1].active[0];
      const encored = !!(foe && foe.volatiles && foe.volatiles['encore']);
      // p2 alternates splash/bodyslam (so the encored lastMove varies and the
      // OVERRIDE fires when the queued move differs from the lock).
      const p2Want = (n % 2) + 1;
      // p1: encore turn-0 (the no-lastMove FAIL) and whenever un-encored on an even
      // rhythm; thunderbolt otherwise (incl. one re-encore into the lock: n%5==4).
      let p1Want = 2;
      if (n === 0 || (!encored && n % 3 === 1) || (encored && n % 5 === 4)) p1Want = 1;
      return { p1Want, p2Want };
    },
    require: ['enLand', 'enFail', 'enEnd'],
  });

  // (2) SLOWER encore user (the target ALREADY moved → stored = rolled + 1): Snorlax
  //     encores Jolteon's splash/thunderbolt.
  S.push({
    id: 'en_slower_lock',
    p1: [mon('Snorlax', ['encore', 'seismictoss', 'splash'], { evs: { hp: 252 } })],
    p2: [mon('Jolteon', ['splash', 'thunderbolt'], { evs: { hp: 252, spa: 252, spe: 252 } })],
    intent: (n, battle) => {
      const foe = battle.sides[1].active[0];
      const encored = !!(foe && foe.volatiles && foe.volatiles['encore']);
      const p2Want = (n % 3 === 2) ? 2 : 1;
      // While the foe is LOCKED, p1 idles (splash) so the encore runs its FULL
      // duration to the natural `-end` (the enEnd realization); it chips only when
      // the foe is free.
      let p1Want = encored ? 3 : 2;
      if (!encored && n % 3 === 1) p1Want = 1;
      return { p1Want, p2Want };
    },
    require: ['enLand', 'enEnd'],
  });

  // (3) The 0-PP EARLY END + the 0-PP-lastMove FAIL: p2's splash is injected to 2 PP;
  //     the encored slot exhausts → the encore `-end`s EARLY at that residual; a
  //     re-encore into the (still-lastMove) 0-PP splash FAILS with both draws.
  S.push({
    id: 'en_zero_pp',
    p1: [mon('Jolteon', ['encore', 'thunderbolt'], { evs: { hp: 252, spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['splash', 'bodyslam'], { evs: { hp: 252 } })],
    inject: [{ side: 1, pp: { moveSlot: 0, val: 2 } }],
    intent: (n, battle) => {
      const foe = battle.sides[1].active[0];
      const encored = !!(foe && foe.volatiles && foe.volatiles['encore']);
      const splashPP = foe ? foe.moveSlots[0].pp : 0;
      // p2 leads with splash (establish lastMove) then bodyslam once out of the lock.
      const p2Want = 1;
      // p1: encore at n=1 (locks splash, 2→1→0 PP) then re-encore once un-encored
      // (the 0-PP-lastMove fail while splash is still the lastMove), else thunderbolt.
      let p1Want = 2;
      if (n === 1 && !encored) p1Want = 1;
      if (n > 1 && !encored && splashPP === 0 && n % 3 === 0) p1Want = 1;
      return { p1Want, p2Want };
    },
    require: ['enLand', 'enEnd', 'enFail'],
  });

  // (4) ENCORE MIRROR at an equal speed — both-encore turns tie (the eachEvent + the
  //     encore-vs-encore residual duration ties at order 10/subOrder 14).
  S.push({
    id: 'en_mirror_tie',
    p1: [mon('Snorlax', ['encore', 'seismictoss'], { evs: { hp: 252 } })],
    p2: [mon('Snorlax', ['encore', 'seismictoss'], { evs: { hp: 252 } })],
    intent: (n, battle) => {
      const a = battle.sides[0].active[0];
      const b = battle.sides[1].active[0];
      const aEnc = !!(a && a.volatiles && a.volatiles['encore']);
      const bEnc = !!(b && b.volatiles && b.volatiles['encore']);
      return {
        p1Want: (!bEnc && n % 4 === 1) ? 1 : 2,
        p2Want: (!aEnc && n % 4 === 1) ? 1 : 2,
      };
    },
    require: ['enLand'],
  });

  // ═══════════════════════ DESTINY BOND ═══════════════════════

  // (5) DB MUTUAL FAINT + double replacement: Gengar bonds when low; Snorlax's Body
  //     Slam KOs it → BOTH faint → double replacement; the benches grind to a win.
  S.push({
    id: 'db_mutual_replace',
    p1: [
      mon('Gengar', ['destinybond', 'thunderbolt'], { ability: 'Levitate', evs: { spa: 252, spe: 252 } }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    ],
    p2: [
      mon('Snorlax', ['shadowball', 'splash'], { evs: { hp: 252, atk: 252 } }),
      mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } }),
    ],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      const isGengar = me && me.species.name === 'Gengar';
      if (isGengar) {
        // Bond every other turn once chipped; thunderbolt otherwise. (The killer's
        // Shadow Ball is GHOST — physical in gen3 — so it actually hits Gengar.)
        const low = me.hp < me.maxhp * 0.75;
        return { p1Want: low ? 1 : 2, p2Want: 1 };
      }
      return { p1Want: 1, p2Want: 1 };
    },
    require: ['dbCast', 'dbMutual'],
  });

  // (6) DB WINDOW CLOSED — Gengar bonds, then SPLASHES (the next move attempt removes
  //     the volatile at onBeforeMove −1), then is KO'd → NO mutual faint. Also the
  //     draw-free RE-CAST (consecutive bonds succeed, PP −1 each).
  S.push({
    id: 'db_window_closed',
    p1: [
      mon('Gengar', ['destinybond', 'splash'], { ability: 'Levitate', evs: {} }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    ],
    p2: [mon('Snorlax', ['shadowball', 'splash'], { evs: { hp: 252, atk: 252 } })],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      const isGengar = me && me.species.name === 'Gengar';
      if (isGengar) {
        // bond, bond (the re-cast), splash, splash… — p2 SPLASHES on the bond turns
        // (so the KO can only ever land on a window-CLOSED turn) and Shadow Balls on
        // the splash turns.
        const bonding = n % 4 <= 1;
        return { p1Want: bonding ? 1 : 2, p2Want: bonding ? 2 : 1 };
      }
      return { p1Want: 1, p2Want: 1 };
    },
    require: ['dbCast', 'koTurn'],
    forbid: ['dbMutual'],
  });

  // (7) DB TIE — both LAST mons: the mutual faint empties both sides → the gen-3 TIE.
  S.push({
    id: 'db_tie',
    p1: [mon('Gengar', ['destinybond', 'nightshade'], { ability: 'Levitate', evs: {} })],
    p2: [mon('Snorlax', ['shadowball', 'splash'], { evs: { hp: 252, atk: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['dbCast'],
  });

  // (8) DB vs a RESIDUAL KO — the sand chip (not a foe Move) KOs the bond holder →
  //     NO mutual faint (the whole cast+chip turn is draw-free at distinct speeds).
  S.push({
    id: 'db_residual_no_trigger',
    p1: [
      mon('Gengar', ['destinybond', 'splash'], { ability: 'Levitate', evs: {} }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    ],
    p2: [mon('Tyranitar', ['splash', 'crunch'], { ability: 'Sand Stream', evs: { hp: 252 } })],
    inject: [{ side: 0, hp: 18 }],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      const isGengar = me && me.species.name === 'Gengar';
      return { p1Want: isGengar ? 1 : 1, p2Want: 1 };
    },
    require: ['dbCast', 'koTurn'],
    forbid: ['dbMutual'],
  });

  // ═══════════════════════ ENDURE ═══════════════════════

  // (9) ENDURE SURVIVE LADDER — Snorlax endures Tauros' Double-Edge at 1 HP; the
  //     consecutive stall rolls escalate 2→4→8 and eventually FAIL → the KO; the
  //     bench continues to a win.
  S.push({
    id: 'ed_survive_ladder',
    p1: [
      mon('Snorlax', ['endure', 'seismictoss'], { evs: { hp: 252 } }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    ],
    p2: [mon('Tauros', ['doubleedge', 'splash'], { evs: { atk: 252, spe: 252 } })],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      const isLax = me && me.species.name === 'Snorlax';
      if (isLax) return { p1Want: 1, p2Want: 1 };
      return { p1Want: 1, p2Want: 1 };
    },
    require: ['edUp', 'edSurvive', 'edStallFail'],
  });

  // (10) The SHARED stall counter — Endure and Protect ALTERNATE on the same mon: an
  //      endure escalates a later Protect's denominator and vice-versa (ED3/ED4).
  S.push({
    id: 'ed_shared_counter',
    p1: [
      mon('Snorlax', ['endure', 'protect', 'seismictoss'], { evs: { hp: 252 } }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    ],
    p2: [mon('Tauros', ['doubleedge', 'splash'], { evs: { atk: 252, spe: 252 } })],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      const isLax = me && me.species.name === 'Snorlax';
      if (isLax) return { p1Want: (n % 2) + 1, p2Want: 1 };
      return { p1Want: 1, p2Want: 1 };
    },
    require: ['edUp', 'edStallFail'],
  });

  // (11) ENDURE vs FIXED damage + the RESIDUAL kill — a BURNED endurer survives the
  //      Seismic Toss at 1 HP (ED6-class) then the burn chip KOs it the same turn
  //      (ED5: endure guards MOVE damage only).
  S.push({
    id: 'ed_residual_death',
    p1: [
      mon('Snorlax', ['endure', 'splash'], { evs: { hp: 252 } }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    ],
    p2: [mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } })],
    inject: [{ side: 0, status: 'brn' }],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      const isLax = me && me.species.name === 'Snorlax';
      if (isLax) return { p1Want: 1, p2Want: 1 };
      return { p1Want: 1, p2Want: 1 };
    },
    require: ['edUp', 'edSurvive', 'koTurn'],
  });

  // (12) ENDURE vs a foe SWITCH — the willAct gate fails it draw-free (ED7).
  S.push({
    id: 'ed_vs_switch',
    p1: [mon('Snorlax', ['endure', 'seismictoss'], { evs: { hp: 252 } })],
    p2: [
      mon('Tauros', ['doubleedge', 'splash'], { evs: { atk: 252, spe: 252 } }),
      mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } }),
    ],
    intent: (n, battle) => {
      const b = benchSlot(battle, 1);
      // The foe pivots on a rhythm; p1 endures those turns (the willAct fail) and
      // chips otherwise.
      if (n % 5 === 2 && b) return { p1Want: 1, p2Switch: b };
      return { p1Want: (n % 3 === 0) ? 1 : 2, p2Want: 1 };
    },
    require: ['edUp'],
  });

  // ═══════════════════════ PERISH SONG ═══════════════════════

  // (13) PERISH BASIC — Celebi sings; it pivots out at perish1 (the switch clears its
  //      own counter) while Snorlax stays and FAINTS at 0; the all-counted RE-CAST
  //      fails ([still]). Seismic Toss grinds to the end.
  S.push({
    id: 'ps_basic',
    p1: [
      mon('Celebi', ['perishsong', 'seismictoss'], { evs: { hp: 252 } }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    ],
    p2: [
      mon('Snorlax', ['splash', 'bodyslam'], { evs: { hp: 252 } }),
      mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } }),
    ],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      const isCelebi = me && me.species.name === 'Celebi';
      if (isCelebi) {
        const myPerish = me.volatiles && me.volatiles['perishsong'];
        const d = myPerish ? myPerish.duration | 0 : 0;
        if (!myPerish && n % 7 === 0) return { p1Want: 1, p2Want: 1 };
        if (myPerish && n % 7 === 1) return { p1Want: 1, p2Want: 1 }; // the all-counted re-cast FAIL
        if (d === 1) {
          const b = benchSlot(battle, 0);
          if (b) return { p1Switch: b, p2Want: 1 }; // pivot out at perish1 (cleared)
        }
        return { p1Want: 2, p2Want: 1 };
      }
      return { p1Want: 1, p2Want: 1 };
    },
    require: ['psCast', 'psTick', 'psFaint', 'psFail'],
  });

  // (14) PERISH MIRROR at an equal speed — both sides perish out on the SAME residual
  //      (a double faint → the tied-entrant double replacement); the per-residual
  //      order-12 pair tie draws ONE shuffle (P5).
  S.push({
    id: 'ps_mirror_tie',
    p1: [
      mon('Snorlax', ['perishsong', 'splash'], { evs: { hp: 252 } }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    ],
    p2: [
      mon('Snorlax', ['perishsong', 'splash'], { evs: { hp: 252 } }),
      mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } }),
    ],
    intent: (n, battle) => {
      const a = battle.sides[0].active[0];
      const isLax = a && a.species.name === 'Snorlax';
      if (isLax) return { p1Want: n === 0 ? 1 : 2, p2Want: 2 };
      return { p1Want: 1, p2Want: 1 };
    },
    require: ['psCast', 'psFaint'],
  });

  // (15) PERISH vs SOUNDPROOF — Electrode is immune (everyone else INCLUDING the
  //      caster is counted; the re-cast with the immune foe is a SILENT success).
  S.push({
    id: 'ps_soundproof',
    p1: [
      mon('Misdreavus', ['perishsong', 'nightshade'], { evs: { hp: 252 } }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    ],
    p2: [mon('Electrode', ['splash', 'thunderbolt'], { ability: 'Soundproof', evs: { hp: 252 } })],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      const isMis = me && me.species.name === 'Misdreavus';
      if (isMis) {
        if (n % 6 === 0 || n % 6 === 1) return { p1Want: 1, p2Want: 1 }; // cast + silent re-cast
        return { p1Want: 2, p2Want: (n % 2) + 1 };
      }
      return { p1Want: 1, p2Want: 2 };
    },
    require: ['psImmune', 'psSilent', 'psFaint'],
  });

  // ═══════════════════════ MEAN LOOK / SPIDER WEB / BLOCK ═══════════════════════

  // (16) MEAN LOOK LIFECYCLE — the GHOST Gengar is firm-trapped (its intent stops
  //      pivoting while trapped); the re-application fails; when the TRAPPER pivots
  //      out the link ends and Gengar's next pivot is ACCEPTED.
  S.push({
    id: 'tm_meanlook_lifecycle',
    p1: [
      mon('Umbreon', ['meanlook', 'shadowball'], { evs: { hp: 252 } }),
      mon('Blissey', ['shadowball', 'splash'], { evs: { hp: 252 } }),
    ],
    p2: [
      mon('Gengar', ['thunderbolt', 'splash'], { ability: 'Levitate', evs: { hp: 252 } }),
      mon('Misdreavus', ['psychic', 'splash'], { evs: { hp: 252 } }),
    ],
    intent: (n, battle) => {
      const foeTrapped = isTrapped(battle, 1);
      const me = battle.sides[0].active[0];
      const isUmb = me && me.species.name === 'Umbreon';
      // p2 pivots on a rhythm — but ONLY when not trapped.
      if (!foeTrapped && n % 4 === 3) {
        const b = benchSlot(battle, 1);
        if (b) return { p1Want: isUmb ? 2 : 1, p2Switch: b };
      }
      if (isUmb) {
        // trap when the foe is free; re-apply once into the trap (the fail); pivot
        // out at n%9==7 (the link-end proof: the foe's next pivot is accepted).
        if (n % 9 === 7) {
          const b = benchSlot(battle, 0);
          if (b) return { p1Switch: b, p2Want: 1 };
        }
        const want = !foeTrapped || n % 5 === 2 ? 1 : 2;
        return { p1Want: want, p2Want: 1 };
      }
      if (n % 6 === 5) {
        const b = benchSlot(battle, 0);
        if (b) return { p1Switch: b, p2Want: 1 };
      }
      return { p1Want: 1, p2Want: 1 };
    },
    require: ['tmTrap', 'tmFail'],
  });

  // (17) SPIDER WEB + BATON PASS INHERIT — the trapped Celebi Baton-Passes (LEGAL —
  //      selfSwitch bypasses the trap gate) and the ENTRANT inherits the firm trap
  //      (noCopy false); the trapper's later faint frees it.
  S.push({
    id: 'tm_bp_inherit',
    p1: [
      mon('Celebi', ['batonpass', 'seismictoss'], { evs: { hp: 252 } }),
      mon('Snorlax', ['bodyslam', 'splash'], { evs: { hp: 252, atk: 252 } }),
    ],
    p2: [
      mon('Ariados', ['spiderweb', 'splash'], { evs: { hp: 252 } }),
      mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } }),
    ],
    intent: (n, battle) => {
      const meTrapped = isTrapped(battle, 0);
      const me = battle.sides[0].active[0];
      const isCelebi = me && me.species.name === 'Celebi';
      const foe = battle.sides[1].active[0];
      const isAriados = foe && foe.species.name === 'Ariados';
      const p2Want = isAriados ? ((n % 5 === 0) ? 1 : 2) : 1;
      if (isCelebi && meTrapped && n % 4 === 2) {
        return { p1Want: 1, p2Want }; // Baton Pass OUT of the trap (the entrant inherits)
      }
      return { p1Want: 2, p2Want };
    },
    require: ['tmTrap', 'bpPass'],
  });

  // (18) BLOCK vs a SUBSTITUTE + the PHAZE bypass — a subbed Snorlax blocks the trap
  //      move ([still]+fail, still free); once the sub breaks, Block lands and the
  //      trapped Snorlax is dragged out by Roar anyway (phazing bypasses trapping and
  //      the drag clears the holder's link).
  S.push({
    id: 'tm_block_sub_phaze',
    p1: [
      mon('Golem', ['block', 'seismictoss'], { evs: { hp: 252 } }),
      mon('Skarmory', ['roar', 'drillpeck', 'seismictoss'], { evs: { hp: 252 } }),
    ],
    p2: [
      mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    ],
    intent: (n, battle) => {
      const foe = battle.sides[1].active[0];
      const subbed = !!(foe && foe.volatiles && foe.volatiles['substitute']);
      const foeTrapped = isTrapped(battle, 1);
      const me = battle.sides[0].active[0];
      const isGolem = me && me.species.name === 'Golem';
      const p2Want = subbed ? 2 : 1;
      if (isGolem) {
        if (foeTrapped && n % 6 === 4) {
          const b = benchSlot(battle, 0);
          if (b) return { p1Switch: b, p2Want }; // hand to Skarmory for the Roar
        }
        return { p1Want: subbed && n % 2 === 0 ? 1 : (foeTrapped ? 2 : 1), p2Want };
      }
      // Skarmory: Roar the trapped mon (the drag bypass), else chip.
      return { p1Want: foeTrapped ? 1 : 3, p2Want };
    },
    require: ['tmTrap', 'tmFail', 'subHit'],
  });

  // ═══════════════════════ BELLY DRUM ═══════════════════════

  // (19) BELLY DRUM — the full-HP drum (pay half, SET +6), the immediate re-drum FAIL
  //      (atk >= 6), and the low-HP FAIL (2*hp <= maxhp) once chipped; Return sweeps.
  S.push({
    id: 'bd_bellydrum',
    p1: [mon('Snorlax', ['bellydrum', 'return'], { evs: { hp: 252, atk: 252 }, happiness: 255 })],
    p2: [
      mon('Skarmory', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    ],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      const atk6 = me && me.boosts.atk >= 6;
      // Drum at n=0 (success), n=1 (the atk>=6 fail), then Return; once the foe's
      // chip drops us below half, a late drum (n%9==8) realizes the low-HP fail.
      let p1Want = 2;
      if (n <= 1 || (n % 9 === 8 && !atk6) || (n % 9 === 8 && atk6)) p1Want = 1;
      return { p1Want, p2Want: 1 };
    },
    require: ['bdSet', 'bdFail'],
  });

  // ═══════════════════════ CHARGE ═══════════════════════

  // (20) CHARGE — charge → Thunderbolt (×2, the HP delta proves it), charge → Surf
  //      (consumed with NO boost), charge → charge (the onRestart re-add) → tbolt.
  S.push({
    id: 'ch_charge',
    p1: [mon('Lanturn', ['charge', 'thunderbolt', 'surf'], { evs: { hp: 252, spa: 252 } })],
    p2: [
      mon('Gyarados', ['splash', 'bodyslam'], { ability: 'No Ability', evs: { hp: 252 } }),
      mon('Snorlax', ['splash', 'bodyslam'], { evs: { hp: 252 } }),
    ],
    intent: (n) => {
      // the 8-turn cycle: charge, tbolt, tbolt(control), charge, surf, tbolt(control),
      // charge, charge(restart) → tbolt…
      const seq = [1, 2, 2, 1, 3, 2, 1, 1];
      return { p1Want: seq[n % 8], p2Want: (n % 2) + 1 };
    },
    require: ['chSet'],
  });

  // ═══════════════════════ MEMENTO ═══════════════════════

  // (21) MEMENTO — the landed memento (self-faint + foe −2/−2 + the foe's queued move
  //      CANCELLED + no Quick Claw) → replacement; a memento INTO A PROTECT is blocked
  //      and the user does NOT faint; the LAST-mon memento hands p2 the win.
  S.push({
    id: 'me_memento',
    p1: [
      mon('Dugtrio', ['memento', 'splash'], { evs: { spe: 252 } }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
      mon('Misdreavus', ['memento', 'splash'], { evs: { hp: 252 } }),
    ],
    p2: [mon('Snorlax', ['bodyslam', 'protect'], { evs: { hp: 252, atk: 252 } })],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      const name = me ? me.species.name : '';
      // p2 protects on a rhythm (the blocked memento realizes across the sweep).
      const p2Want = (n % 4 === 1) ? 2 : 1;
      if (name === 'Dugtrio') return { p1Want: (n % 4 === 1 || n % 4 === 2) ? 1 : 2, p2Want };
      if (name === 'Blissey') return { p1Want: 1, p2Want };
      return { p1Want: 1, p2Want }; // Misdreavus: the last-mon memento
    },
    require: ['meFaint', 'meDrop', 'meBlocked'],
  });

  // ═══════════════════════ MIMIC ═══════════════════════

  // (22) MIMIC — the first-turn no-lastMove FAIL; the copy (Psychic into the Mimic
  //      slot at pp 5); the copied slot RUNS (its own PP decrements); the pivot
  //      out+back REVERTS the slot to Mimic; the re-mimic copies fresh.
  S.push({
    id: 'mi_mimic',
    p1: [
      // p1 KNOWS splash — a Mimic pointed at a splash-lastMove foe FAILS with the
      // already-known gate (the miFail realization); a psychic-lastMove copy lands.
      mon('Snorlax', ['mimic', 'splash'], { evs: { hp: 252 } }),
      mon('Blissey', ['seismictoss', 'splash'], { evs: { hp: 252 } }),
    ],
    p2: [mon('Alakazam', ['psychic', 'splash'], { evs: { hp: 252, spa: 252 } })],
    intent: (n, battle) => {
      const me = battle.sides[0].active[0];
      const isLax = me && me.species.name === 'Snorlax';
      const foe = battle.sides[1].active[0];
      const foeLast = foe && foe.lastMove ? foe.lastMove.id : '';
      if (isLax) {
        const slot0 = me.moveSlots[0] ? me.moveSlots[0].id : '';
        const copied = slot0 !== 'mimic';
        // Mimic on n∈{1,2}: the faster Alakazam moves FIRST, so its THIS-turn move is
        // the lastMove Mimic sees — (n%2)+1 gives splash at n=1 (the already-known
        // FAIL, since p1 knows Splash) and psychic at n=2 (the COPY). Deterministic
        // realization of BOTH branches before the chip can KO the mimicker.
        if (!copied && (n === 1 || n === 2)) {
          return { p1Want: 1, p2Want: (n % 2) + 1 };
        }
        if (copied && n % 7 === 5) {
          const b = benchSlot(battle, 0);
          if (b) return { p1Switch: b, p2Want: 1 }; // pivot → the overlay reverts
        }
        return { p1Want: copied ? 1 : 2, p2Want: (n % 2) + 1 };
      }
      if (n % 5 === 4) {
        const b = benchSlot(battle, 0);
        if (b) return { p1Switch: b, p2Want: 1 };
      }
      return { p1Want: 1, p2Want: 1 };
    },
    require: ['miCopy', 'miFail'],
  });

  // ═══════════════════════ PAIN SPLIT ═══════════════════════

  // (23) PAIN SPLIT — the clamp case (Gengar 261 maxhp vs Blissey 714: the average
  //      caps at Gengar's maxhp while Blissey takes the FULL loss), the normal split,
  //      and the SUB block ([still]+fail).
  S.push({
    id: 'psp_painsplit',
    p1: [mon('Gengar', ['painsplit', 'thunderbolt'], { ability: 'Levitate', evs: {} })],
    p2: [
      mon('Blissey', ['icebeam', 'substitute', 'splash'], { evs: { hp: 252 } }),
      mon('Snorlax', ['shadowball', 'splash'], { evs: { hp: 252 } }),
    ],
    intent: (n, battle) => {
      const foe = battle.sides[1].active[0];
      const subbed = !!(foe && foe.volatiles && foe.volatiles['substitute']);
      // Blissey subs on a rhythm; Gengar pain-splits into it (the block) and
      // otherwise splits/chips.
      const p2Want = (n % 5 === 2 && !subbed) ? 2 : 1;
      // While the sub is up, alternate the BLOCKED pain split and a Thunderbolt INTO
      // the sub (the subHit realization); else pain-split on a rhythm.
      const p1Want = subbed ? ((n % 2 === 0) ? 1 : 2) : ((n % 3 === 0) ? 1 : 2);
      return { p1Want, p2Want };
    },
    require: ['pspHit', 'pspFail', 'subHit'],
  });

  // ═══════════════════════ PSYCH UP ═══════════════════════

  // (24) PSYCH UP — copies Suicune's Calm Mind stages VERBATIM (the boost columns
  //      assert it), WORKS THROUGH the substitute (bypasssub), and an all-zero copy
  //      (post-KO replacement) still succeeds. Body Slam grinds to a win.
  S.push({
    id: 'pu_psychup',
    p1: [mon('Snorlax', ['psychup', 'bodyslam'], { evs: { hp: 252, atk: 252 } })],
    p2: [
      mon('Suicune', ['calmmind', 'substitute', 'surf'], { evs: { hp: 252 } }),
      mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } }),
    ],
    intent: (n, battle) => {
      const foe = battle.sides[1].active[0];
      const isCune = foe && foe.species.name === 'Suicune';
      if (isCune) {
        const spa = foe.boosts.spa | 0;
        const subbed = !!(foe.volatiles && foe.volatiles['substitute']);
        const p2Want = (!subbed && n % 6 === 3) ? 2 : (spa < 4 ? 1 : 3);
        return { p1Want: (n % 4 === 2 && spa > 0) ? 1 : 2, p2Want };
      }
      return { p1Want: (n % 6 === 5) ? 1 : 2, p2Want: 1 };
    },
    require: ['puCopy', 'subHit'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# movecoverage_batch6_golden.txt — Gen-3 MOVE-COVERAGE BATCH 6 full-battle golden.');
  lines.push('# Per-decision STATE(+status+slp-time+boosts+HP+SUB-HP+ENCORE+PERISH+TRAPPED)+SEED');
  lines.push('# +first-mover differential to GAME-END. The FINAL UNMODELED tail: ENCORE /');
  lines.push('#   DESTINY BOND / ENDURE / PERISH SONG / MEAN LOOK / SPIDER WEB / BLOCK /');
  lines.push('#   BELLY DRUM / CHARGE / MEMENTO / MIMIC / PAIN SPLIT / PSYCH UP.');
  lines.push('# EXTENDS the batch-3/4/4c/5 44-field DEC format with SIX appended columns (50):');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INJECT <id>  <json array of {side?,slot?,status?,hp?,pp:{moveSlot,val}?}>  ([] if none)');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1Curse p1Wish p1SubHp  p2Curse p2Wish p2SubHp  p1Future p2Future \\');
  lines.push('#        p1Encore p1Perish p1Trapped  p2Encore p2Perish p2Trapped');
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
          d.p1.encore, d.p1.perish, d.p1.trapped,
          d.p2.encore, d.p2.perish, d.p2.trapped,
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
    console.error('MOVECOVERAGE BATCH6 GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`BATCH6 GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('landed encores', 'enLand', 60);
  need('encore fails', 'enFail', 40);
  need('encore ends', 'enEnd', 40);
  need('destiny bond casts', 'dbCast', 60);
  need('destiny bond mutual faints', 'dbMutual', 30);
  need('endure singleturns', 'edUp', 60);
  need('endure survive clamps', 'edSurvive', 40);
  need('endure stall fails', 'edStallFail', 30);
  need('perish song fieldactivates', 'psCast', 60);
  need('perish ticks', 'psTick', 60);
  need('perish faints', 'psFaint', 40);
  need('perish all-counted fails', 'psFail', 20);
  need('perish soundproof immunes', 'psImmune', 20);
  need('perish silent re-casts', 'psSilent', 20);
  need('trap-move lands', 'tmTrap', 60);
  need('trap-move fails (sub/re-apply)', 'tmFail', 30);
  need('baton passes', 'bpPass', 20);
  need('belly drum sets', 'bdSet', 40);
  need('belly drum fails', 'bdFail', 30);
  need('charge starts', 'chSet', 60);
  need('landed mementos', 'meFaint', 40);
  need('memento drops', 'meDrop', 40);
  need('memento protect-blocks', 'meBlocked', 15);
  need('mimic copies', 'miCopy', 40);
  need('mimic fails', 'miFail', 30);
  need('pain splits', 'pspHit', 40);
  need('pain split sub-fails', 'pspFail', 20);
  need('psych up copies', 'puCopy', 40);
  if (winRows < 40) { console.error(`BATCH6 GOLDEN: too few WIN rows (${winRows} < 40)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `movecoverage batch6 golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: ${Object.keys(corpus).sort().map((k) => `${k}=${corpus[k]}`).join(' ')} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
