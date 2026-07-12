// gen_movecoverage_batch1_golden.js — Gen-3 MOVE-COVERAGE BATCH 1 differential golden
// (`gen3_move_coverage_batch1_v1`): the DRAW-FREE post-hit effects a damaging move drops
// after a landed hit — RECOIL / DRAIN / SELF STAT-DROP / ITEM REMOVAL / RAPID SPIN.
//
// Extends the leechseed TAB format with a per-side ITEM column (for Knock Off / Thief)
// and per-side LEECH-SEEDED + SPIKES columns (for Rapid Spin's clear), on top of the
// per-decision STATE(+status)+BOOSTS(self-drop)+HP(recoil/drain)+SEED+first-mover full-
// battle differential to GAME-END.
//
// THE FIVE CLASSES (FOUR draw-free + SELF-DROP draws ONE random(100) — settled by
// probe_batch1_movecoverage.js + a per-call-site PRNG trace):
//   RECOIL     — Double-Edge `recoil:[1,3]` (Take Down / Submission `[1,4]`): the USER
//                takes max(floor(dmgDealt·num/den),1) HP; Rock Head negates; fires behind
//                a SUBSTITUTE too (on the sub damage). DRAW-FREE.
//   DRAIN      — Giga Drain / Absorb / Mega Drain `drain:[1,2]`: the USER heals a fraction
//                of the damage dealt; heal-at-full fails; fires behind a sub (ceil vs floor).
//                DRAW-FREE.
//   SELF-DROP  — Overheat (self −2 SpA) / Superpower (self −1 Atk/−1 Def): `move.self.boosts`
//                on the USER, ±6 clamp. gen3 `selfDrops` DRAWS ONE `random(100)` (the
//                `secondaryRoll`) then applies UNCONDITIONALLY (`self.chance === undefined`),
//                so it is NOT draw-free. Fires behind a sub.
//   ITEM       — Knock Off (removes; gen3 no dmg boost) / Thief / Covet (steal iff attacker
//                itemless); Sticky Hold blocks; onAfterHit → ONLY when the MON was damaged
//                (NOT behind a sub, the target keeps its item). DRAW-FREE.
//   RAPID SPIN — clears the USER's own Spikes + Leech Seed (onAfterHit + onAfterSubDamage,
//                so it clears behind a sub too). DRAW-FREE.
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios that each ISOLATE a class, capturing the running PRNG seed BEFORE the first
// decision (`initSeed`) and AFTER each DECISION BOUNDARY, plus each active's species/hp/
// maxhp/fainted/status + boosts + confusion + ITEM + pokemon_left + per-side SPIKES layers
// + per-side LEECH-SEEDED + first mover + winner. The Rust test seeds a BattleState at the
// init seed and runs `run_full_battle` WITHOUT re-seeding — so the post-decision seed must
// match at EVERY boundary (a wrong draw model → a SEED desync), AND the recoil/drain HP,
// the self-drop boosts, the removed/stolen item, and the cleared spikes/leech must match
// (a wrong effect → an HP / boost / item / spikes / leech desync).
//
// Output: tests/vectors/movecoverage_batch1_golden.txt
//
// Run:  node src/rust_sim/harness/gen_movecoverage_batch1_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/movecoverage_batch1_golden.txt');
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

function leechSeededOf(a) {
  return !!(a && a.volatiles && a.volatiles['leechseed']);
}

// The CURRENT held item id (or '-' when none). Matches the port's MonState::item.
function itemOf(a) {
  return (a && a.item) ? a.item : '-';
}

function snap(side) {
  const a = side.active[0];
  if (!a) {
    return {
      species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', stage: 0, left: side.pokemonLeft,
      boosts: [0, 0, 0, 0, 0], confusion: 0, spikes: spikesOf(side), leechSeeded: false, item: '-',
    };
  }
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status, stage, left: side.pokemonLeft,
    boosts: boostsOf(a), confusion: confusionOf(a), spikes: spikesOf(side),
    leechSeeded: leechSeededOf(a), item: itemOf(a),
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

// Scan the protocol log between two decision points for the BATCH-1 branch flags.
function outcomesSince(log, fromIdx) {
  const out = {
    recoil: false, drain: false, selfDrop: false,
    knockOff: false, thiefSteal: false, covetSteal: false, stickyBlock: false,
    rapidClearSpikes: false, rapidClearLeech: false,
  };
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    if (tag === '-damage' && (p[4] || '').includes('Recoil')) out.recoil = true;
    if (tag === '-heal' && (p[4] || '').includes('drain')) out.drain = true;
    // A -unboost line (`|-unboost|<mon>|<stat>|<mag>`) — the SELF-DROP scenarios are the
    // only ones producing an unboost (on the mover). Detect any -unboost.
    if (tag === '-unboost') out.selfDrop = true;
    if (tag === '-enditem' && (p[4] || '').includes('Knock Off')) out.knockOff = true;
    if (tag === '-item' && (p[4] || '').includes('Thief')) out.thiefSteal = true;
    if (tag === '-item' && (p[4] || '').includes('Covet')) out.covetSteal = true;
    if (tag === '-activate' && (p[3] || '').includes('Sticky Hold')) out.stickyBlock = true;
    if (tag === '-sideend' && (p[4] || '').includes('Rapid Spin')) out.rapidClearSpikes = true;
    if (tag === '-end' && (p[4] || '').includes('Rapid Spin')) out.rapidClearLeech = true;
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
      if (force[0]) c.p1 = firstLiveBench(0, battle);
      if (force[1]) c.p2 = firstLiveBench(1, battle);
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

  // Optional one-time post-start injection (weather / status / HP / spikes / leech). STATE
  // only (no PRNG) so the seed parity is unaffected. Recorded as an INJECT line so the Rust
  // test reproduces the board.
  if (sc.inject) {
    const battle = stream.battle;
    for (const inj of sc.inject) {
      if (inj.weather) { battle.field.setWeather(inj.weather, battle.sides[0].active[0]); battle.field.weatherState.duration = 0; }
      if (inj.side !== undefined) {
        const m = battle.sides[inj.side].active[0];
        if (inj.spikes) for (let k = 0; k < inj.spikes; k++) battle.sides[inj.side].addSideCondition('spikes', battle.sides[1 - inj.side].active[0]);
        if (inj.leechseed) m.addVolatile('leechseed', battle.sides[1 - inj.side].active[0]);
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

  // --- (1) RECOIL: Double-Edge recoil floor(dmg/3), grinding to a win. p1 Tauros repeatedly
  //   Double-Edges the p2 Snorlax; each hit recoils floor(dmg/3) onto Tauros (visible in
  //   Tauros's HP). REQUIRES: recoil + win. ---
  S.push({
    id: 'recoil_doubleedge',
    p1: [mon('Tauros', ['doubleedge', 'bodyslam'], { ability: 'Sturdy', item: '', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', item: '', nature: 'Careful', evs: { hp: 252, def: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['recoil'],
  });

  // --- (2) RECOIL negated by ROCK HEAD: Aggron Double-Edge with Rock Head takes NO recoil
  //   (Aggron HP stays full-ish, chipped only by the foe). REQUIRES: NO recoil (forbid). ---
  S.push({
    id: 'recoil_rockhead_negates',
    p1: [mon('Aggron', ['doubleedge', 'tackle'], { ability: 'Rock Head', item: '', nature: 'Adamant', evs: { atk: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', item: '', nature: 'Careful', evs: { hp: 252, def: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    forbid: ['recoil'],
  });

  // --- (3) DRAIN: Giga Drain heals floor(dmg/2). p1 Sceptile is INJURED (inject) so the
  //   heal is visible in its HP; it Giga Drains the p2 Snorlax to a win. REQUIRES: drain. ---
  S.push({
    id: 'drain_gigadrain',
    p1: [mon('Sceptile', ['gigadrain', 'absorb'], { ability: 'Overgrow', item: '', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', item: '', nature: 'Careful', evs: { hp: 252 } })],
    inject: [{ side: 0, hp: 80 }],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['drain'],
  });

  // --- (4) SELF-DROP: Overheat self −2 SpA, climbing to the −6 floor. p1 Charizard Overheats
  //   the p2 Snorlax every turn; SpA drops −2/turn until the −6 floor (into-floor emits
  //   nothing). REQUIRES: selfDrop. ---
  S.push({
    id: 'selfdrop_overheat',
    p1: [mon('Charizard', ['overheat', 'ember'], { ability: 'Blaze', item: '', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', item: '', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['selfDrop'],
  });

  // --- (5) SELF-DROP: Superpower self −1 Atk/−1 Def. p1 Machamp Superpowers the p2 Snorlax;
  //   Atk/Def drop −1 each per use. REQUIRES: selfDrop. ---
  S.push({
    id: 'selfdrop_superpower',
    p1: [mon('Machamp', ['superpower', 'karatechop'], { ability: 'Guts', item: '', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', item: '', nature: 'Careful', evs: { hp: 252, def: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['selfDrop'],
  });

  // --- (6) ITEM: Knock Off removes the target's Leftovers (gen3 no dmg boost). p2 Snorlax
  //   holds Leftovers; after the first Knock Off its item is gone (item column). REQUIRES:
  //   knockOff. ---
  S.push({
    id: 'item_knockoff',
    p1: [mon('Tyranitar', ['knockoff', 'crunch'], { ability: 'Sand Stream', item: '', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', item: 'Leftovers', nature: 'Careful', evs: { hp: 252, spd: 252, def: 4 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['knockOff'],
  });

  // --- (7) ITEM: Knock Off BLOCKED by Sticky Hold (Muk keeps Leftovers). REQUIRES:
  //   stickyBlock + NO knockOff (forbid). ---
  S.push({
    id: 'item_knockoff_stickyhold',
    p1: [mon('Tyranitar', ['knockoff', 'crunch'], { ability: 'Sand Stream', item: '', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Muk', ['pound'], { ability: 'Sticky Hold', item: 'Leftovers', nature: 'Careful', evs: { hp: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['stickyBlock'],
    forbid: ['knockOff'],
  });

  // --- (8) ITEM: Thief STEALS (attacker itemless). p1 Gengar (no item) Thiefs the p2
  //   Snorlax's Leftovers → Gengar GAINS it (item columns flip). A 2nd Thief does nothing
  //   (attacker now holds it). REQUIRES: thiefSteal. ---
  S.push({
    id: 'item_thief_steals',
    p1: [mon('Gengar', ['thief', 'shadowball'], { ability: 'Levitate', item: '', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', item: 'Leftovers', nature: 'Careful', evs: { hp: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['thiefSteal'],
  });

  // --- (9) ITEM: Thief does NOT steal (attacker HOLDS an item). p1 Gengar holds Leftovers →
  //   Thief just damages, no steal (both items unchanged). REQUIRES: NO thiefSteal (forbid). ---
  S.push({
    id: 'item_thief_holds_no_steal',
    p1: [mon('Gengar', ['thief', 'shadowball'], { ability: 'Levitate', item: 'Leftovers', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', item: 'Choice Band', nature: 'Careful', evs: { hp: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    forbid: ['thiefSteal'],
  });

  // --- (10) ITEM: Covet STEALS (attacker itemless) — like Thief but no -enditem line. p1
  //   Persian Covets the p2 Snorlax's Leftovers → Persian GAINS it. REQUIRES: covetSteal. ---
  S.push({
    id: 'item_covet_steals',
    p1: [mon('Persian', ['covet', 'slash'], { ability: 'Limber', item: '', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', item: 'Leftovers', nature: 'Careful', evs: { hp: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['covetSteal'],
  });

  // --- (11) RAPID SPIN: clears the USER's OWN Spikes (injected 3 layers) + Leech Seed
  //   (injected). p1 Forretress Rapid Spins the p2 Snorlax; after the first spin its side
  //   spikes go 3→0 and its leech clears (spikes/leech columns). REQUIRES:
  //   rapidClearSpikes + rapidClearLeech. ---
  S.push({
    id: 'rapidspin_clears',
    p1: [mon('Forretress', ['rapidspin', 'tackle'], { ability: 'Sturdy', item: '', nature: 'Relaxed', evs: { hp: 252, def: 252 } })],
    p2: [mon('Snorlax', ['pound', 'leechseed'], { ability: 'Immunity', item: '', nature: 'Careful', evs: { hp: 252 } })],
    // Inject 3 spikes on p1's side + leech-seed p1's Forretress (seeded by p2).
    inject: [{ side: 0, spikes: 3, leechseed: true }],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['rapidClearSpikes', 'rapidClearLeech'],
  });

  // --- (12) RAPID SPIN clears behind a SUB too (onAfterSubDamage). Inject spikes on p1's
  //   side; p2 Snorlax is behind a SUB (Substitute move first). Rapid Spin still clears the
  //   user's spikes even when the sub absorbs the hit. REQUIRES: rapidClearSpikes. ---
  S.push({
    id: 'rapidspin_through_sub',
    p1: [mon('Forretress', ['rapidspin', 'tackle'], { ability: 'Sturdy', item: '', nature: 'Relaxed', evs: { hp: 252, def: 252 } })],
    p2: [mon('Snorlax', ['substitute', 'pound'], { ability: 'Immunity', item: '', nature: 'Careful', evs: { hp: 252 } })],
    inject: [{ side: 0, spikes: 2 }],
    // p2 subs turn 1 (move 1), then p1 spins into the sub; spikes still clear.
    intent: (decisionNo) => ({ p1Want: 1, p2Want: decisionNo === 0 ? 1 : 2 }),
    require: ['rapidClearSpikes'],
  });

  // --- (13) RECOIL/DRAIN/SELF-DROP INTO A REAL BATTLE to game-end (the union: recoil +
  //   drain + self-drop + switching + faints all the way to a win). REQUIRES: win. ---
  S.push({
    id: 'batch1_into_a_real_battle',
    p1: [mon('Salamence', ['doubleedge', 'dragonclaw'], { ability: 'Sturdy', item: '', nature: 'Adamant', evs: { atk: 252, spe: 252 } }),
         mon('Sceptile', ['gigadrain', 'leafblade'], { ability: 'Overgrow', item: '', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Diglett', ['pound'], { level: 1, ability: 'Sand Veil', nature: 'Bold' }),
         mon('Sandshrew', ['pound'], { level: 1, ability: 'Sand Veil', nature: 'Bold' }),
         mon('Cubone', ['pound'], { level: 1, ability: 'Rock Head', nature: 'Bold' })],
    intent: (decisionNo, battle) => {
      const p1Active = battle.sides[0].active[0];
      const isMence = p1Active && p1Active.species.name === 'Salamence';
      // Salamence Double-Edges (recoil), then pivot to Sceptile (Giga Drain) mid-battle.
      if (isMence && decisionNo >= 2) return { p1Switch: 2, p2Want: 1 };
      return { p1Want: 1, p2Want: 1 };
    },
    require: ['recoil'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# movecoverage_batch1_golden.txt — Gen-3 MOVE-COVERAGE BATCH 1 full-battle golden.');
  lines.push('# Per-decision STATE(+status+spikes+leechSeeded+ITEM)+BOOSTS+SEED+first-mover differential to GAME-END.');
  lines.push('# Classes: RECOIL / DRAIN / SELF-DROP / ITEM-REMOVAL / RAPID-SPIN — all DRAW-FREE.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INJECT <id>  <json array of {weather?,side?,status?,hp?,spikes?,leechseed?}>  ([] if none)');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1Spikes p2Spikes  p1LeechSeeded p2LeechSeeded  p1Item p2Item');
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
        for (const k of ['recoil', 'drain', 'selfDrop', 'knockOff', 'thiefSteal', 'covetSteal', 'stickyBlock', 'rapidClearSpikes', 'rapidClearLeech']) {
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
          d.p1.spikes, d.p2.spikes,
          d.p1.leechSeeded ? 1 : 0, d.p2.leechSeeded ? 1 : 0,
          d.p1.item, d.p2.item,
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
    console.error('MOVECOVERAGE BATCH1 GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`BATCH1 GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('recoil decisions', 'recoil', 50);
  need('drain decisions', 'drain', 50);
  need('self-drop decisions', 'selfDrop', 50);
  need('knock-off decisions', 'knockOff', 40);
  need('thief-steal decisions', 'thiefSteal', 40);
  need('covet-steal decisions', 'covetSteal', 40);
  need('sticky-hold-block decisions', 'stickyBlock', 40);
  need('rapid-spin-clear-spikes decisions', 'rapidClearSpikes', 40);
  need('rapid-spin-clear-leech decisions', 'rapidClearLeech', 40);
  if (winRows < 50) { console.error(`BATCH1 GOLDEN: too few WIN rows (${winRows} < 50)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `movecoverage batch1 golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: recoil=${corpus.recoil || 0} drain=${corpus.drain || 0} selfDrop=${corpus.selfDrop || 0} ` +
    `knockOff=${corpus.knockOff || 0} thiefSteal=${corpus.thiefSteal || 0} covetSteal=${corpus.covetSteal || 0} ` +
    `stickyBlock=${corpus.stickyBlock || 0} rapidClearSpikes=${corpus.rapidClearSpikes || 0} ` +
    `rapidClearLeech=${corpus.rapidClearLeech || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
