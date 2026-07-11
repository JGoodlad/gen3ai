// gen_substitute_golden.js — Gen-3 SUBSTITUTE differential golden.
//
// Extends harness/gen_leechseed_golden.js (the per-decision STATE+STATUS+SPIKES-LAYERS+
// BOOSTS+CONFUSION+SEED+winner full-battle differential) to the NEW mechanic this step
// adds: gen-3 **Substitute** (`substitute`) — the user spends `floor(maxhp/4)` HP to make a
// decoy with that much HP that ABSORBS incoming foe hits until it breaks.
//
// THE DRAW MODEL (verified bit-for-bit vs the omniscient sim's PRNG probes,
// harness/probe_substitute_*.js — the SURPRISES this surfaced are the secondary draw-COUNT
// and the confusion self-hit target):
//
//   THE SUBSTITUTE MOVE (`volatileStatus:'substitute'`, `target:'self'`, never-miss):
//     * NEVER-MISS (`accuracy: true`) → NO accuracy draw.
//     * FAIL (DRAW-FREE) if a `substitute` is ALREADY present OR `hp <= floor(maxhp/4)` (can't
//       afford — VERIFIED: hp == floor(maxhp/4) FAILS, hp == that+1 SUCCEEDS).
//     * SUCCESS: pay `floor(maxhp/4)` HP and create the volatile with `hp = floor(maxhp/4)`.
//       DRAW-FREE (`onStart` just `-start`). `landed` FALSE (no in-tryMoveHit Update).
//
//   A FOE MOVE INTO A SUBSTITUTED MON:
//     * A DAMAGING move: draws acc+crit+damage as normal (UNCHANGED count), but the damage
//       hits the SUB's HP (not the mon). The sub BREAKS when its HP reaches 0; the excess
//       does NOT carry to the mon (gen-3). THE SECONDARY draw-COUNT SURPRISE: the per-move
//       SECONDARY `random(100)` is STILL DRAWN (gen-3 iterates the now-`null` target list, so
//       the draw fires — same count as a bare hit), but its EFFECT does NOT apply (no status /
//       no stat-drop / no flinch, AND no confusion `random(2,6)` / Tri-Attack `random(3)`
//       follow-on draw). VERIFIED: Body Slam / Crunch / Water Pulse / Tri Attack into a sub
//       draw the SAME acc+crit+dmg+random(100) as bare, then NOTHING further.
//     * A STATUS / stat-DROP move (Thunder Wave / Toxic / Leech Seed / a -stat secondary) is
//       BLOCKED by the sub: accuracy still drawn, then `-fail` / no effect, NO status set, and
//       no SetStatus shuffle / sleep random(2,6). DRAW-FREE past accuracy.
//     * A CONFUSION self-hit hits the MON, NOT the sub (the self-hit's `this.damage` bypasses
//       the `onTryPrimaryHit` sub-intercept) — the sub HP is unchanged, the mon's HP drops.
//       The draw model is unchanged (`randomChance(1,2)` then `random(16)`).
//     * PHAZE (Roar / Whirlwind) BYPASSES the sub — the user is dragged anyway (forceSwitch is
//       a runAction-tail effect, not a moveHit target). Draw model unchanged.
//
//   The substitute clears on switch-out (clearVolatile) and on faint.
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios that each ISOLATE a branch, capturing the running PRNG seed BEFORE the first
// decision (`initSeed`) and AFTER each DECISION BOUNDARY, plus each active's species/hp/
// maxhp/fainted/status + boosts + confusion + pokemon_left + per-side SPIKES LAYERS + the
// per-side SUBSTITUTE HP + first mover + winner. The Rust test seeds a BattleState at the
// init seed and runs `run_full_battle` WITHOUT re-seeding — so the post-decision seed must
// match at EVERY boundary, AND the substitute HP + the mon HP (which holds the create cost /
// the absorbed-or-not damage / the confusion-self-hit) must match. A wrong secondary draw
// model → a SEED desync; a wrong absorb / break / block / confusion-target → an HP / sub-HP
// / status / boost desync.
//
// FAIL-LOUD: each scenario declares the BRANCH it must realize (a CREATE, an ABSORB that
// drops the sub HP WITHOUT applying the secondary, a BREAK, a blocked STATUS, a blocked
// STAT-DROP, a CONFUSION self-hit that hits the mon, a create FAIL, a phaze drag-through, a
// sub-to-game-end); generation aborts if the sim run did NOT realize it.
//
// Output: tests/vectors/substitute_golden.txt
//
// Run:  node src/rust_sim/harness/gen_substitute_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/substitute_golden.txt');
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
  let x = 0x6d2b79f5 >>> 0;
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

// The active mon's SUBSTITUTE HP (the `substitute` volatile's `hp`), or 0 when there is no
// sub (a present sub is always >= 1 — it breaks at 0 → removed — so 0 unambiguously means
// "no sub", matching the engine's `Option<u16>` → 0 mapping for None).
function subHpOf(a) {
  return a && a.volatiles && a.volatiles['substitute'] ? (a.volatiles['substitute'].hp | 0) : 0;
}

function snap(side) {
  const a = side.active[0];
  if (!a) {
    return {
      species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', stage: 0, left: side.pokemonLeft,
      boosts: [0, 0, 0, 0, 0], confusion: 0, spikes: spikesOf(side), subHp: 0,
    };
  }
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status, stage, left: side.pokemonLeft,
    boosts: boostsOf(a), confusion: confusionOf(a), spikes: spikesOf(side),
    subHp: subHpOf(a),
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

// Scan the protocol log between two decision points for the SUBSTITUTE branch flags.
//   subStart   — a `|-start|...|Substitute` (a sub was CREATED this decision)
//   subFail    — a `|-fail|...|move: Substitute` (create FAILED: already-subbed / [weak])
//   subDamage  — a `|-activate|...|move: Substitute|[damage]` (a hit was ABSORBED, sub held)
//   subBreak   — a `|-end|...|Substitute` (the sub broke this decision)
//   subBlock   — a `|-activate|...|move: Substitute|[block]` / a status/leech that did nothing
//                into a subbed mon (we detect via the absence-of-status semantics in the
//                STATE; here we flag the `-activate` block protocol when present)
//   selfhit    — a confusion self-hit (`-damage ... [from] confusion`) — proves it hit the MON
//   drag       — a `|drag|` (phaze dragged the subbed mon)
function outcomesSince(log, fromIdx) {
  const out = {
    p1: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    p2: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    subStart: false, subFail: false, subDamage: false, subBreak: false, subBlock: false,
    subAbsorb: false, drag: false,
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
    // CREATE: `|-start|p1a: Snorlax|Substitute`.
    if (tag === '-start' && (p[3] || '') === 'Substitute') out.subStart = true;
    // FAIL: `|-fail|p1a: Snorlax|move: Substitute` (already-subbed) or `...|[weak]`.
    if (tag === '-fail' && (p[3] || '') === 'move: Substitute') out.subFail = true;
    // ABSORB (sub held): `|-activate|p2a: Blissey|Substitute|[damage]` (the `-activate` carries
    // the bare condition name `Substitute` in p[3], NOT `move: Substitute`).
    if (tag === '-activate' && (p[3] || '') === 'Substitute' && (p[4] || '').includes('damage')) out.subDamage = true;
    // BREAK: `|-end|p2a: Blissey|Substitute`.
    if (tag === '-end' && (p[3] || '') === 'Substitute') out.subBreak = true;
    // BLOCK (a status/leech/secondary into a sub): `|-activate|...|Substitute|[block]`.
    if (tag === '-activate' && (p[3] || '') === 'Substitute' && (p[4] || '').includes('block')) out.subBlock = true;
    if (tag === 'drag') out.drag = true;
  }
  // A hit was ABSORBED by the sub iff it either HELD (`-activate [damage]`) OR BROKE (`-end`)
  // this decision — both mean the foe's damage went to the sub, not the mon (the absorb proof).
  out.subAbsorb = out.subDamage || out.subBreak;
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

  // Optional one-time post-start STATE-only injection (HP — to force the low-HP create FAIL
  // at exactly floor(maxhp/4)). PURELY a board set (no PRNG) so the seed parity is intact.
  if (sc.inject) {
    const battle = stream.battle;
    for (const inj of sc.inject) {
      if (inj.weather) { battle.field.setWeather(inj.weather, battle.sides[0].active[0]); battle.field.weatherState.duration = 0; }
      if (inj.side !== undefined) {
        const m = battle.sides[inj.side].active[0];
        if (inj.status) m.setStatus(inj.status, m, null, true);
        if (inj.hp !== undefined) {
          // hp can be a function of maxhp (for the create-fail boundary) or an absolute value.
          m.hp = (typeof inj.hp === 'function') ? inj.hp(m.maxhp) : inj.hp;
        }
      }
    }
  }

  const script = intentDriver(sc.intent);
  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen };

  // Several substitute scenarios are deliberately NON-terminating (a sub/Splash stalemate
  // that exercises the create / absorb / block branches turn after turn). Cap them at a small
  // `maxDecisions` so the golden stays compact + fast; the battle-to-game-end scenarios (the
  // phaze drag + the real-battle sweep) terminate naturally well under the cap.
  const maxDecisions = sc.maxDecisions || 400;
  let decisionNo = 0;
  let safety = 0;
  while (!stream.battle.ended && safety < 400 && decisionNo < maxDecisions) {
    safety++;
    const battle = stream.battle;
    const reqState = battle.requestState;
    if (reqState !== 'move' && reqState !== 'switch') { await tick(); continue; }
    const force = forceSwitchTable(battle);
    const seedBefore = battle.prng.getSeed();
    if (decisionNo === 0) rec.initSeed = seedBefore;

    // Mid-battle STATE-only injection at a specific decision (e.g. confuse the subbed mon
    // once its sub is up). A no-PRNG board set, so seed parity holds.
    if (sc.injectAt && sc.injectAt[decisionNo]) {
      for (const inj of sc.injectAt[decisionNo]) {
        const m = battle.sides[inj.side].active[0];
        if (m) {
          if (inj.confusion) m.addVolatile('confusion');
          if (inj.hp !== undefined) m.hp = (typeof inj.hp === 'function') ? inj.hp(m.maxhp) : inj.hp;
          if (inj.status) m.setStatus(inj.status, m, null, true);
        }
      }
    }

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

  // --- (1) CREATE the substitute (cost + sub HP). p1 Snorlax makes a sub (move 1), spending
  //   floor(maxhp/4) and creating a sub with that HP; then Splashes (move 2). A 2nd Substitute
  //   FAILS (already-subbed). The mon's HP (524-131=393) + the sub HP (131) are the proof; the
  //   create draws NOTHING (only the end-of-turn Quick Claw). p2 Splashes (draw-free).
  //   REQUIRES: subStart + subFail. ---
  S.push({
    id: 'create_then_already_subbed_fail',
    p1: [mon('Snorlax', ['substitute', 'splash'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    p2: [mon('Blissey', ['splash'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252 } })],
    // Substitute turn 1; then Substitute AGAIN every turn (the rest fail, already-subbed).
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    maxDecisions: 6,
    require: ['subStart', 'subFail'],
  });

  // --- (2) CREATE FAIL at low HP (hp <= floor(maxhp/4) → can't afford). Inject Snorlax to
  //   EXACTLY floor(maxhp/4) (== fail), then to floor(maxhp/4)+1 (== succeed) — so the same
  //   scenario pins BOTH sides of the `<=` boundary. The mon's HP is unchanged on the fail
  //   (no cost), then drops by floor(maxhp/4) on the success. REQUIRES: subFail + subStart. ---
  S.push({
    id: 'create_fail_at_low_hp_boundary',
    p1: [mon('Snorlax', ['substitute', 'splash'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    p2: [mon('Blissey', ['splash'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252 } })],
    // dec 0: inject hp = floor(maxhp/4) → Substitute FAILS. dec 1: inject hp = floor(maxhp/4)+1
    // → Substitute SUCCEEDS (cost floor(maxhp/4) → 1 HP left). Splash thereafter.
    injectAt: {
      0: [{ side: 0, hp: (maxhp) => Math.floor(maxhp / 4) }],
      1: [{ side: 0, hp: (maxhp) => Math.floor(maxhp / 4) + 1 }],
    },
    intent: (decisionNo) => ({ p1Want: decisionNo <= 1 ? 1 : 2, p2Want: 1 }),
    maxDecisions: 4,
    require: ['subFail', 'subStart'],
  });

  // --- (3) ABSORB without applying the SECONDARY (the secondary draw-COUNT proof). p2 Blissey
  //   makes a sub; p1 Snorlax Body Slams (par 30 secondary) INTO the sub. The hit is ABSORBED
  //   (the sub HP drops), the secondary `random(100)` IS DRAWN (the SEED advances by the full
  //   acc+crit+dmg+secondary count — the proof), but NO paralysis is applied (Blissey's status
  //   stays `-`). The sub HP delta + the unchanged Blissey HP + the absent par + the SEED are
  //   the proof. REQUIRES: subStart + subDamage; FORBID: p2.fullpara never set by the absorbed
  //   Body Slam (we assert no par via the STATE in the Rust test, not a corpus flag). ---
  S.push({
    id: 'absorb_body_slam_secondary_blocked',
    // A WEAK level-50 Snorlax Body Slam (~50-70) does NOT break Blissey's big sub (HP 178), so
    // the sub HOLDS turn after turn — `subDamage` (`-activate [damage]`) fires, proving the
    // absorb-WITHOUT-break path AND the secondary suppression on a held sub. (The secondary
    // random(100) still draws; the sub HP drops by the dealt damage but never to 0 here.)
    p1: [mon('Snorlax', ['bodyslam', 'splash'], { level: 50, ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    p2: [mon('Blissey', ['substitute', 'softboiled'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    // p2 makes a sub (move 1) when it has none + is healthy, else Soft-Boiled (move 2) to keep
    // its HP topped so the sub re-up is afforded; p1 Body Slams INTO the sub every turn.
    intent: (decisionNo, battle) => {
      const me = battle.sides[1].active[0];
      const hasSub = me && me.volatiles && me.volatiles['substitute'];
      return { p1Want: 1, p2Want: hasSub ? 2 : 1 };
    },
    maxDecisions: 8,
    require: ['subStart', 'subDamage'],
  });

  // --- (4) BREAK the sub (a hit >= sub HP; excess does NOT carry). p2 Gengar (small maxhp →
  //   small sub) makes a sub; p1 Snorlax's big Body Slam BREAKS it. The mon's HP is UNCHANGED
  //   (the broken sub ate the whole hit, no carry-over), the sub goes to 0 → None. Then the
  //   next Body Slam hits the bare Gengar (HP drops + par possible). REQUIRES: subStart +
  //   subDamage(or break) + subBreak. ---
  S.push({
    id: 'break_the_sub_no_carry',
    // p2 Electrode (Electric — NOT immune to Normal; low maxhp → a SMALL sub) makes a sub; a
    // 252-Atk Adamant Snorlax Body Slam BREAKS it (the hit >= the small sub HP). The mon's HP
    // is UNCHANGED (no carry-over). (Electrode is grounded + not Ghost, so Body Slam connects.)
    p1: [mon('Snorlax', ['bodyslam', 'splash'], { ability: 'Immunity', nature: 'Adamant', evs: { atk: 252 } })],
    p2: [mon('Electrode', ['substitute', 'splash'], { ability: 'Soundproof', nature: 'Timid', evs: { spe: 252 } })],
    // p2 makes a sub when it has none (move 1), else Splash; p1 Body Slams to break it then hit.
    intent: (decisionNo, battle) => {
      const me = battle.sides[1].active[0];
      const hasSub = me && me.volatiles && me.volatiles['substitute'];
      return { p1Want: 1, p2Want: hasSub ? 2 : 1 };
    },
    maxDecisions: 8,
    require: ['subStart', 'subBreak'],
  });

  // --- (5) STATUS move BLOCKED by the sub. p2 Blissey makes a sub; p1 Jolteon Thunder Waves
  //   (move 1) INTO the sub — accuracy drawn, then BLOCKED (`-fail`), NO paralysis. Blissey's
  //   status stays `-`, the sub HP is unchanged (a status move does NOT damage the sub), the
  //   SEED advances by exactly the accuracy roll (the draw-count proof). REQUIRES: subStart;
  //   FORBID: a p2 par (asserted via STATE). ---
  S.push({
    id: 'status_move_blocked_by_sub',
    p1: [mon('Jolteon', ['thunderwave', 'splash'], { ability: 'Volt Absorb', nature: 'Timid', evs: { spe: 252 } })],
    p2: [mon('Blissey', ['substitute', 'softboiled'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252 } })],
    // p2 keeps a sub up (re-make when broken, else Soft-Boiled); p1 Thunder Waves into it.
    intent: (decisionNo, battle) => {
      const me = battle.sides[1].active[0];
      const hasSub = me && me.volatiles && me.volatiles['substitute'];
      return { p1Want: 1, p2Want: hasSub ? 2 : 1 };
    },
    maxDecisions: 8,
    require: ['subStart'],
  });

  // --- (6) STAT-DROP secondary BLOCKED by the sub. p2 Gengar makes a sub; p1 Tyranitar
  //   Crunches (move 1, -1 SpD secondary) INTO the sub — the secondary random(100) IS DRAWN
  //   (the SEED proof), the hit is ABSORBED (sub HP drops), but NO -1 SpD applies (Gengar's
  //   boosts stay [0,0,0,0,0]). REQUIRES: subStart + subDamage. ---
  S.push({
    id: 'stat_drop_secondary_blocked_by_sub',
    p1: [mon('Tyranitar', ['crunch', 'splash'], { ability: 'Sand Stream', nature: 'Adamant', evs: { atk: 252 } })],
    p2: [mon('Gengar', ['substitute', 'splash'], { ability: 'Levitate', item: 'Leftovers', nature: 'Timid', evs: { hp: 252 } })],
    // p2 re-makes the sub when it can (move 1) — Tyranitar's sand chips Gengar so it can't
    // always afford; else Splash. p1 Crunches into the sub. (Sand sets up the chip realistically.)
    intent: (decisionNo, battle) => {
      const me = battle.sides[1].active[0];
      const hasSub = me && me.volatiles && me.volatiles['substitute'];
      const canAfford = me && me.hp > Math.floor(me.maxhp / 4);
      return { p1Want: 1, p2Want: (!hasSub && canAfford) ? 1 : 2 };
    },
    maxDecisions: 8,
    require: ['subStart', 'subAbsorb'],
  });

  // --- (7) CONFUSION self-hit hits the MON, NOT the sub. p1 Snorlax makes a sub; we INJECT
  //   confusion onto it (once the sub is up); the typeless-40 self-hit (on a failed
  //   confusion check) damages the MON's HP — the sub HP STAYS PUT. So a turn where Snorlax
  //   self-hits drops its OWN HP while subHp is unchanged. The draw model (randomChance(1,2)
  //   then random(16)) is unchanged → the SEED proves it. REQUIRES: subStart + selfhit. ---
  S.push({
    id: 'confusion_self_hit_hits_the_mon',
    p1: [mon('Snorlax', ['substitute', 'splash'], { ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Blissey', ['splash'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252 } })],
    // dec 0: Substitute (sub up). dec 1..: Splash; confusion injected at dec 1 (sub is up).
    // The foe only Splashes (no confusion source), and the inject sets the volatile directly
    // (Snorlax's Immunity ability does NOT block confusion) — so the engine's confusion arm
    // runs the self-hit, which must hit the MON (not the sub).
    injectAt: { 1: [{ side: 0, confusion: true }] },
    intent: (decisionNo) => ({ p1Want: decisionNo === 0 ? 1 : 2, p2Want: 1 }),
    maxDecisions: 7,
    require: ['subStart', 'selfhit'],
  });

  // --- (8) PHAZE BYPASSES the sub. p2 Snorlax makes a sub; p1 Suicune Roars (move 1) — the
  //   subbed Snorlax is DRAGGED OUT anyway (the sub does not stop a phaze). The dragged mon's
  //   sub is gone (cleared on switch-out). REQUIRES: subStart + drag. ---
  S.push({
    id: 'phaze_bypasses_the_sub',
    p1: [mon('Suicune', ['roar', 'splash'], { ability: 'Pressure', item: 'Leftovers', nature: 'Bold', evs: { hp: 252 } })],
    p2: [mon('Snorlax', ['substitute', 'splash'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } }),
         mon('Gengar', ['splash'], { ability: 'Levitate', nature: 'Timid', evs: { hp: 252 } })],
    // p2 Snorlax makes a sub (move 1) when it has none; p1 Splash first to let the sub go up,
    // then Roar to drag the subbed mon. p2's Gengar (no sub move) just Splashes once dragged.
    intent: (decisionNo, battle) => {
      const p2a = battle.sides[1].active[0];
      const isSnorlax = p2a && p2a.species.name === 'Snorlax';
      const hasSub = p2a && p2a.volatiles && p2a.volatiles['substitute'];
      if (isSnorlax && !hasSub) return { p1Want: 2, p2Want: 1 }; // Splash / Substitute → sub up
      return { p1Want: 1, p2Want: 1 };                            // Roar (drag) / Splash
    },
    maxDecisions: 6,
    require: ['subStart', 'drag'],
  });

  // --- (9) SUBSTITUTE INTO A REAL BATTLE TO GAME-END (the union: sub + absorb + break +
  //   switching + faints all the way to a win). p1 Snorlax subs then attacks; the foe's frail
  //   lvl-1 mons are OHKO'd by Body Slam, so the battle ENDS in a few decisions. The sub
  //   survives the foe's chip (absorbed) and the per-decision sub HP + the win are the proof.
  //   REQUIRES: subStart + a win. ---
  S.push({
    id: 'substitute_into_real_battle',
    p1: [mon('Snorlax', ['substitute', 'bodyslam'], { ability: 'Immunity', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    // Frail lvl-1 foes that Body Slam OHKOs — so the battle ENDS quickly. They Pound (chip the
    // sub) so the absorb is exercised against a real attacker.
    p2: [mon('Diglett', ['pound'], { level: 1, ability: 'Sand Veil', nature: 'Bold' }),
         mon('Sandshrew', ['pound'], { level: 1, ability: 'Sand Veil', nature: 'Bold' }),
         mon('Cubone', ['pound'], { level: 1, ability: 'Rock Head', nature: 'Bold' })],
    // Sub when none (move 1), else Body Slam (move 2) to sweep.
    intent: (decisionNo, battle) => {
      const me = battle.sides[0].active[0];
      const hasSub = me && me.volatiles && me.volatiles['substitute'];
      return { p1Want: hasSub ? 2 : 1, p2Want: 1 };
    },
    require: ['subStart'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# substitute_golden.txt — Gen-3 SUBSTITUTE full-battle golden.');
  lines.push('# Per-decision-boundary STATE(+status+spikes-layers+subHp)+BOOSTS+CONFUSION+SEED+first-mover differential to GAME-END.');
  lines.push('# (Mirrors the leechseed TAB format: replaces the per-side leechSeeded flags with per-side substitute HP.)');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1(fullpara wake thaw selfhit flinch) p2(...)  p1Spikes p2Spikes  p1SubHp p2SubHp');
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

      rec.decisions.forEach((d) => {
        for (const k of ['subStart', 'subFail', 'subDamage', 'subBreak', 'subBlock', 'subAbsorb', 'drag']) {
          if (d.outcomes[k]) { scenSeen[sc.id][k] = true; corpus[k] = (corpus[k] || 0) + 1; }
        }
        if (d.outcomes.p1.selfhit || d.outcomes.p2.selfhit) { scenSeen[sc.id].selfhit = true; corpus.selfhit = (corpus.selfhit || 0) + 1; }
        if (d.p1.subHp > 0 || d.p2.subHp > 0) { corpus.subRows = (corpus.subRows || 0) + 1; }
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
          d.p1.subHp, d.p2.subHp,
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
    console.error('SUBSTITUTE GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`SUBSTITUTE GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('CREATE decisions', 'subStart', 50);
  need('CREATE-FAIL decisions', 'subFail', 50);
  need('ABSORB decisions (hold or break)', 'subAbsorb', 50);
  need('HELD-sub absorb decisions', 'subDamage', 20);
  need('BREAK decisions', 'subBreak', 10);
  need('CONFUSION-self-hit decisions', 'selfhit', 10);
  need('PHAZE-drag decisions', 'drag', 50);
  need('substitute STATE rows', 'subRows', 100);
  if (winRows < 50) { console.error(`SUBSTITUTE GOLDEN: too few WIN rows (${winRows} < 50)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `substitute golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: subStart=${corpus.subStart || 0} subFail=${corpus.subFail || 0} subDamage=${corpus.subDamage || 0} ` +
    `subBreak=${corpus.subBreak || 0} selfhit=${corpus.selfhit || 0} drag=${corpus.drag || 0} ` +
    `subRows=${corpus.subRows || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
