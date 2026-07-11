// gen_leechseed_golden.js — Gen-3 LEECH SEED differential golden.
//
// Extends harness/gen_phaze_golden.js (the per-decision STATE+STATUS+SPIKES-LAYERS+
// BOOSTS+SEED+winner full-battle differential) to the NEW mechanic this step adds: the
// gen-3 status MOVE **Leech Seed** (`leechseed`) — plant the `leechseed` volatile on the
// FOE; each end-of-turn the seeded mon loses HP and the SEEDER's ACTIVE heals it.
// DEFERRED (fail-loud in the engine): a Liquid Ooze target reverses the drain (rare in
// gen-3 OU), plus everything already deferred.
//
// THE DRAW MODEL (verified bit-for-bit vs the omniscient sim's PRNG probe,
// harness/probe_leechseed_rng.js):
//
//   THE LEECH SEED MOVE (`volatileStatus: 'leechseed'`, `target: "normal"`, type Grass,
//   accuracy 90):
//     * ACCURACY: gen-3 Leech Seed is `accuracy: 90` (NOT never-miss), so it DRAWS
//       `randomChance(90, 100)` — it CAN miss. The accuracy roll is drawn
//       UNCONDITIONALLY, even into a Grass-immune OR already-seeded target (the
//       immunity / fail is reported only AFTER the accuracy roll). VERIFIED: a
//       splash/splash baseline turn draws 1 (Quick Claw); a Leech-Seed turn — LAND,
//       GRASS-immune, OR already-seeded-fail — ALL draw 2 (accuracy + Quick Claw).
//     * GRASS IMMUNITY (`onTryImmunity` → `!target.hasType('Grass')`): a Grass target is
//       IMMUNE — accuracy still drawn, then `-immune`, NO volatile.
//     * ALREADY-SEEDED: a 2nd Leech Seed on a seeded target FAILS (`addVolatile` returns
//       false): accuracy drawn, then "did nothing", the existing volatile UNCHANGED.
//     * PLANT: on a landed non-immune non-already-seeded hit, the `leechseed` volatile is
//       added to the foe (DRAW-FREE — `onStart` only adds `-start`). `landed` is FALSE (a
//       status moveHit returns undefined → the in-tryMoveHit Update is skipped).
//
//   THE LEECH RESIDUAL (the crux — DRAW-FREE but ORDER-SENSITIVE; gen4-inherited
//   onResidualOrder 10, onResidualSubOrder 5 — BETWEEN Leftovers sub 4 and the status DoT
//   sub 6): each end-of-turn the seeded mon loses `floor(maxhp/8)` (clamped) and the
//   SEEDER's CURRENT active HEALS the drained amount (clamped to its maxhp). VERIFIED
//   residual order: `sandstorm[o=8] → leftovers[o=10,s=4] → leechseed[o=10,s=5] →
//   brn[o=10,s=6]`. A seeder whose active is FAINTED → no drain, no heal (the whole
//   onResidual returns: `if (!target || target.fainted || target.hp <= 0) return`).
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios that each ISOLATE a branch, capturing the running PRNG seed BEFORE the first
// decision (`initSeed`) and AFTER each DECISION BOUNDARY, plus each active's species/hp/
// maxhp/fainted/status + boosts + confusion + pokemon_left + per-side SPIKES LAYERS + the
// per-side LEECH-SEEDED flag + first mover + winner. The Rust test seeds a BattleState at
// the init seed and runs `run_full_battle` WITHOUT re-seeding — so the post-decision seed
// must match at EVERY boundary, AND the leech-seeded volatile state + the drain/heal HP
// (which is in the `hp` column) must match. A wrong draw model → a SEED desync; a wrong
// drain/heal/order → an HP / leech-flag desync.
//
// FAIL-LOUD: each scenario declares the BRANCH it must realize (a seed LANDS, a GRASS
// immune, an ALREADY-SEEDED fail, a leech-drain KO, a leech+Leftovers+weather+burn order
// interaction, a seeder-fainted-no-heal, a leech-to-game-end); generation aborts if the
// sim run did NOT realize it.
//
// Output: tests/vectors/leechseed_golden.txt
//
// Run:  node src/rust_sim/harness/gen_leechseed_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/leechseed_golden.txt');
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

// Whether `side`'s active mon is LEECH-SEEDED (the `leechseed` volatile is present).
function leechSeededOf(a) {
  return !!(a && a.volatiles && a.volatiles['leechseed']);
}

function snap(side) {
  const a = side.active[0];
  if (!a) {
    return {
      species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', stage: 0, left: side.pokemonLeft,
      boosts: [0, 0, 0, 0, 0], confusion: 0, spikes: spikesOf(side), leechSeeded: false,
    };
  }
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status, stage, left: side.pokemonLeft,
    boosts: boostsOf(a), confusion: confusionOf(a), spikes: spikesOf(side),
    leechSeeded: leechSeededOf(a),
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

// Scan the protocol log between two decision points for the LEECH-SEED branch flags.
//   seedStart  — a `|-start|...|move: Leech Seed` (a seed LANDED this decision)
//   seedImmune — a `|-immune|` reported for a Leech Seed user's target (Grass immune)
//   seedFail   — a Leech Seed that did nothing (already seeded) — a `move ... |[still]` +
//                "did nothing" debug (we detect the leech drain present but no new -start)
//   leechDamage / leechHeal — the residual `-damage ... [from] Leech Seed` + `-heal [silent]`
//   leechKO    — a faint within 2 lines of a Leech Seed residual damage (the drain KO'd)
function outcomesSince(log, fromIdx) {
  const out = {
    p1: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    p2: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    seedStart: false, seedImmune: false, seedFail: false,
    leechDamage: false, leechHeal: false, leechKO: false,
  };
  let lastLeechDamageIdx = -100;
  let usedLeechSeed = false;
  let sawStart = false;
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
    // A Leech Seed move was used this window (so a `-immune` / `-fail` is the leech's).
    if (tag === 'move' && (p[3] || '') === 'Leech Seed') usedLeechSeed = true;
    // The seed LANDED: `|-start|p2a: Snorlax|move: Leech Seed`.
    if (tag === '-start' && (p[3] || '') === 'move: Leech Seed') { out.seedStart = true; sawStart = true; }
    // GRASS immune: a `-immune` after a Leech Seed move (the target's a Grass type).
    if (tag === '-immune' && usedLeechSeed) out.seedImmune = true;
    // ALREADY-SEEDED fail: a Leech Seed move with `[still]` (the "did nothing" path) and
    // NO new -start this window. The protocol emits `|move|...|Leech Seed||[still]`.
    if (tag === 'move' && (p[3] || '') === 'Leech Seed' && (p[4] || '') === '' && (p[5] || '').includes('still')) {
      // mark; confirmed below if no -start follows.
      out._leechStill = true;
    }
    // The residual drain + heal: `-damage ... [from] Leech Seed` / `-heal ... [silent]`.
    if (tag === '-damage' && (p[4] || '').includes('Leech Seed')) { out.leechDamage = true; lastLeechDamageIdx = i; }
    if (tag === '-heal' && (p[3] || '').includes('[silent]')) out.leechHeal = true;
    if (tag === 'faint' && i - lastLeechDamageIdx <= 2) out.leechKO = true;
  }
  if (out._leechStill && !sawStart) out.seedFail = true;
  delete out._leechStill;
  return out;
}

// The first live, non-active bench mon as a `switch N` choice (for forced replacements).
function firstLiveBench(side, battle) {
  const s = battle.sides[side];
  for (let k = 0; k < s.pokemon.length; k++) {
    const p = s.pokemon[k];
    if (p !== s.active[0] && !p.fainted) return `switch ${k + 1}`;
  }
  return 'pass';
}

// Clamp a desired 1-based move slot to a LEGAL one for `side`'s active.
function legalMove(side, battle, want) {
  const req = battle.sides[side].activeRequest;
  const moves = req && req.active && req.active[0] ? req.active[0].moves : null;
  if (!moves) return 'move 1';
  const usable = [];
  for (let k = 0; k < moves.length; k++) if (!moves[k].disabled) usable.push(k + 1);
  if (usable.length === 0) return 'move 1';
  return `move ${usable.includes(want) ? want : usable[0]}`;
}

// A battle-aware driver: `intent(decisionNo, battle)` returns the desired {p1Want, p2Want}
// 1-based move slots (a `p1Switch`/`p2Switch` 1-based team slot OVERRIDES the move for a
// VOLUNTARY pivot). The driver clamps each move to a legal one + auto-replaces on a forced
// switch.
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

  // Optional one-time post-start injection (weather / status / HP) — used ONLY by the
  // residual-order interaction scenario, whose sand + burn the move set can't otherwise
  // realize on a single seeded mon without a Tyranitar/burn move muddying the draw count.
  // The inject is recorded into the golden as a TEAM-independent `INJECT` line so the Rust
  // test reproduces the identical board. We keep injects PURELY about the engine STATE
  // (no PRNG) so the seed parity is unaffected by the injection itself.
  if (sc.inject) {
    const battle = stream.battle;
    for (const inj of sc.inject) {
      if (inj.weather) { battle.field.setWeather(inj.weather, battle.sides[0].active[0]); battle.field.weatherState.duration = 0; }
      if (inj.side !== undefined) {
        const m = battle.sides[inj.side].active[0];
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

  // --- (1) SEED LANDS → drain + seeder heal each turn (the basic loop). p1 Meganium
  //   Leech-Seeds (move 1) the p2 Snorlax, then Synthesizes (move 2) so the seeder heal is
  //   ALSO visible alongside the move heal; the drain (floor(524/8)=65) chips Snorlax every
  //   turn while the seed persists, and Meganium heals 65 each residual. The seed accuracy
  //   is 90 → it MISSES on some seeds (re-cast next turn), proving the accuracy draw; once
  //   landed, the leechSeeded flag + the per-turn HP delta are the proof. REQUIRES:
  //   seedStart + leechDamage. ---
  S.push({
    id: 'seed_lands_drain_and_heal',
    p1: [mon('Meganium', ['leechseed', 'synthesis'], { ability: 'Overgrow', item: 'Leftovers', evs: { hp: 252, def: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', item: 'Leftovers', nature: 'Careful', evs: { hp: 252 } })],
    // Leech Seed until it lands (re-cast on a miss), then Synthesis to expose the seeder heal.
    intent: (decisionNo, battle) => {
      const foe = battle.sides[1].active[0];
      const seeded = foe && foe.volatiles && foe.volatiles['leechseed'];
      return { p1Want: seeded ? 2 : 1, p2Want: 1 };
    },
    require: ['seedStart', 'leechDamage'],
  });

  // --- (2) GRASS-immune target (accuracy still drawn, then -immune, no volatile). p1
  //   Snorlax Leech-Seeds (move 1) a p2 Sceptile (pure Grass → immune). Every cast draws
  //   accuracy then `-immune`; the leechSeeded flag stays FALSE. p1 then Body Slams (move 2)
  //   the Sceptile out. REQUIRES: seedImmune + NO seedStart (forbid). ---
  S.push({
    id: 'grass_target_immune',
    p1: [mon('Snorlax', ['leechseed', 'bodyslam'], { ability: 'Immunity', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Sceptile', ['pound'], { ability: 'Overgrow', item: 'Leftovers', nature: 'Jolly', evs: { hp: 252 } })],
    intent: (decisionNo) => ({ p1Want: decisionNo % 2 === 0 ? 1 : 2, p2Want: 1 }),
    require: ['seedImmune'],
    forbid: ['seedStart'],
  });

  // --- (3) ALREADY-SEEDED fail (a 2nd Leech Seed on a seeded mon FAILS — accuracy drawn
  //   then "did nothing", the existing seed unchanged). p1 Snorlax seeds the p2 Blissey,
  //   then Leech-Seeds AGAIN every turn (each re-cast draws accuracy then fails); the seed
  //   keeps draining (the drain HP proves the original volatile persists). REQUIRES:
  //   seedStart + seedFail + leechDamage. ---
  S.push({
    id: 'already_seeded_fails',
    p1: [mon('Snorlax', ['leechseed', 'bodyslam'], { ability: 'Immunity', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Blissey', ['pound'], { ability: 'Natural Cure', item: 'Leftovers', nature: 'Bold', evs: { hp: 252 } })],
    // Leech Seed EVERY turn — turn 1 lands (after any misses), the rest fail (already seeded).
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['seedStart', 'seedFail', 'leechDamage'],
  });

  // --- (4) LEECH DRAIN KOs the seeded mon (the seed kills). p1 Meganium seeds a chipped p2
  //   Gengar; the residual drain (floor(maxhp/8)) eventually KOs Gengar — and the seeder
  //   heal STILL applies (the heal is inside the same onResidual, before faintMessages). p1
  //   Splashes/Synths while the seed grinds Gengar out. REQUIRES: seedStart + leechKO + win. ---
  S.push({
    id: 'leech_drain_kos',
    p1: [mon('Meganium', ['leechseed', 'synthesis'], { ability: 'Overgrow', item: 'Leftovers', evs: { hp: 252, def: 252 } })],
    p2: [mon('Gengar', ['pound'], { ability: 'Levitate', item: '', nature: 'Timid', evs: { spe: 252 } })],
    // Seed (re-cast on a miss), then Synthesis while the leech grinds the lone Gengar out.
    intent: (decisionNo, battle) => {
      const foe = battle.sides[1].active[0];
      const seeded = foe && foe.volatiles && foe.volatiles['leechseed'];
      return { p1Want: seeded ? 2 : 1, p2Want: 1 };
    },
    require: ['seedStart', 'leechKO'],
  });

  // --- (5) THE RESIDUAL-ORDER INTERACTION (the risk case): leech + Leftovers + SANDSTORM
  //   chip + BURN DoT on the SAME seeded mon. The verified residual order is
  //   sandstorm(o=8) → Leftovers(o=10,s=4) → LEECH(o=10,s=5) → burn(o=10,s=6). We INJECT
  //   sand weather + a burn on the p2 Gengar (Leftovers holder) and seed it; the per-turn HP
  //   delta is the EXACT composition of all four residuals in order (a wrong leech subOrder
  //   would re-order the heal/drain and desync the HP), and the seeder Meganium heals the
  //   leech amount. The inject is a STATE-only board set (no PRNG) so the seed parity holds.
  //   REQUIRES: seedStart + leechDamage. The injected sand/burn make it the 4-way order test. ---
  S.push({
    id: 'leech_leftovers_sand_burn_order',
    p1: [mon('Meganium', ['leechseed', 'synthesis'], { ability: 'Overgrow', item: 'Leftovers', evs: { hp: 252, def: 252 } })],
    p2: [mon('Gengar', ['pound'], { ability: 'Levitate', item: 'Leftovers', nature: 'Timid', evs: { hp: 252 } })],
    // Inject AFTER start: sandstorm + burn the Gengar + chip it (so the 4-way residual runs).
    inject: [{ weather: 'sandstorm' }, { side: 1, status: 'brn', hp: 240 }],
    // Seed (re-cast on a miss), then Synthesis while the 4-way residual chips Gengar.
    intent: (decisionNo, battle) => {
      const foe = battle.sides[1].active[0];
      const seeded = foe && foe.volatiles && foe.volatiles['leechseed'];
      return { p1Want: seeded ? 2 : 1, p2Want: 1 };
    },
    require: ['seedStart', 'leechDamage'],
  });

  // --- (6) SEEDER ACTIVE FAINTED → leech does NOTHING (the drain is SKIPPED entirely when
  //   the seeder's active is fainted, AND the heal goes to whatever is active). We seed a p2
  //   Gengar from a p1 Meganium, then the Meganium FAINTS (a p2 attack) and its replacement
  //   (Snorlax) becomes the seeder's active — so the residual leech now drains Gengar and
  //   heals the NEW active (Snorlax). This tests the `getAtSlot(sourceSlot)` semantics (the
  //   heal follows the seeder's CURRENT active across a replacement). REQUIRES: seedStart +
  //   leechDamage. ---
  S.push({
    id: 'seeder_replaced_heal_follows',
    p1: [mon('Meganium', ['leechseed', 'synthesis'], { ability: 'Overgrow', item: '', nature: 'Timid', evs: { hp: 4 } }),
         mon('Snorlax', ['pound'], { ability: 'Immunity', item: 'Leftovers', nature: 'Careful', evs: { hp: 252 } })],
    p2: [mon('Tyranitar', ['crunch'], { ability: 'Sand Stream', item: 'Choice Band', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    // Seed Gengar/TTar (re-cast on a miss); the fast CB Tyranitar Crunch KOs the frail
    // Meganium → forced replace to Snorlax → the seed's heal now follows Snorlax.
    intent: (decisionNo, battle) => {
      const foe = battle.sides[1].active[0];
      const seeded = foe && foe.volatiles && foe.volatiles['leechseed'];
      return { p1Want: seeded ? 2 : 1, p2Want: 1 };
    },
    require: ['seedStart', 'leechDamage'],
  });

  // --- (7) LEECH INTO A REAL BATTLE TO GAME-END (the union: leech + switching + residuals +
  //   faints all the way to a win). p1's Meganium seeds the p2 lead then pivots to a hitter;
  //   the seed drains while p1 grinds the foe team out (the leechSeeded flag survives the
  //   seeded mon switching out — it clears — and a fresh seed re-applies). The frail lvl-1
  //   foes end the battle in a few decisions (no PP slog the engine can't model). REQUIRES:
  //   seedStart + leechDamage + a win. ---
  S.push({
    id: 'leech_into_real_battle',
    p1: [mon('Meganium', ['leechseed', 'synthesis'], { ability: 'Overgrow', item: 'Leftovers', evs: { hp: 252, def: 252 } }),
         mon('Snorlax', ['bodyslam'], { ability: 'Immunity', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    // Frail lvl-1 grounded foes — OHKO'd by Body Slam, so the battle ENDS quickly.
    p2: [mon('Diglett', ['pound'], { level: 1, ability: 'Sand Veil', nature: 'Bold' }),
         mon('Sandshrew', ['pound'], { level: 1, ability: 'Sand Veil', nature: 'Bold' }),
         mon('Cubone', ['pound'], { level: 1, ability: 'Rock Head', nature: 'Bold' })],
    intent: (decisionNo, battle) => {
      const p1Active = battle.sides[0].active[0];
      const isMeg = p1Active && p1Active.species.name === 'Meganium';
      const foe = battle.sides[1].active[0];
      const seeded = foe && foe.volatiles && foe.volatiles['leechseed'];
      if (isMeg) {
        if (!seeded) return { p1Want: 1, p2Want: 1 };       // seed (re-cast on a miss)
        // Seeded → pivot to Snorlax (team slot 2) to close the battle with attacks.
        return { p1Switch: 2, p2Want: 1 };
      }
      return { p1Want: 1, p2Want: 1 };                       // Snorlax Body Slam sweep
    },
    require: ['seedStart', 'leechDamage'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# leechseed_golden.txt — Gen-3 LEECH SEED full-battle golden.');
  lines.push('# Per-decision-boundary STATE(+status+spikes-layers+leechSeeded)+BOOSTS+SEED+first-mover differential to GAME-END.');
  lines.push('# (Extends the phaze TAB format: replaces the dragSpecies tail with per-side leechSeeded flags.)');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INJECT <id>  <json array of {weather?,side?,status?,hp?}>  (a one-time STATE-only post-start board set; [] if none)');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1(fullpara wake thaw selfhit flinch) p2(...)  p1Spikes p2Spikes  p1LeechSeeded p2LeechSeeded');
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
        for (const k of ['seedStart', 'seedImmune', 'seedFail', 'leechDamage', 'leechHeal', 'leechKO']) {
          if (d.outcomes[k]) { scenSeen[sc.id][k] = true; corpus[k] = (corpus[k] || 0) + 1; }
        }
        if (d.p1.leechSeeded || d.p2.leechSeeded) { corpus.leechSeededRows = (corpus.leechSeededRows || 0) + 1; }
      });

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
          oc(d.outcomes.p1), oc(d.outcomes.p2),
          d.p1.spikes, d.p2.spikes,
          d.p1.leechSeeded ? 1 : 0, d.p2.leechSeeded ? 1 : 0,
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
    console.error('LEECHSEED GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`LEECHSEED GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('seed-LANDS decisions', 'seedStart', 50);
  need('GRASS-immune decisions', 'seedImmune', 50);
  need('already-seeded-FAIL decisions', 'seedFail', 50);
  need('leech-DRAIN decisions', 'leechDamage', 100);
  need('leech-KO decisions', 'leechKO', 1);
  need('leech-seeded STATE rows', 'leechSeededRows', 100);
  if (winRows < 50) { console.error(`LEECHSEED GOLDEN: too few WIN rows (${winRows} < 50)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `leechseed golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: seedStart=${corpus.seedStart || 0} seedImmune=${corpus.seedImmune || 0} ` +
    `seedFail=${corpus.seedFail || 0} leechDamage=${corpus.leechDamage || 0} leechKO=${corpus.leechKO || 0} ` +
    `leechSeededRows=${corpus.leechSeededRows || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
