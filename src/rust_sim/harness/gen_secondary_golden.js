// gen_secondary_golden.js — Gen-3 SECONDARY-EFFECTS + onBeforeMove STATUS draws
// FULL-BATTLE differential harness.
//
// Extends harness/gen_fullbattle_golden.js (a SECONDARY-FREE move+switch+
// replacement → game-end per-decision STATE+SEED+winner differential) to the two
// NEW draw sites this step adds:
//
//   (A) the onBeforeMove STATUS draws (BEFORE accuracy) — para randomChance(1,4),
//       freeze randomChance(1,5) thaw, confusion randomChance(1,2)+self-hit
//       random(16), flinch/sleep DRAW-FREE; and
//   (B) the per-move SECONDARY random(100) AFTER a landed hit (Body Slam par30,
//       Ice Beam frz10, Thunderbolt par10, Rock Slide flinch30, Sludge Bomb psn30).
//
// THE PROOF (the CRUX): drive the OMNISCIENT in-process BattleStream (no server) a
// FULL battle with REAL secondary moves so mons get statused IN-ENGINE (no status
// injection) and the onBeforeMove draws fire on later turns — capturing the running
// PRNG seed BEFORE the first decision (`initSeed`) and AFTER each DECISION BOUNDARY
// (each `move` turn AND each forced-`switch` replacement sub-step). The Rust test
// seeds a BattleState at the init seed and runs `run_full_battle` WITHOUT
// re-seeding — so the post-decision seed must match the sim's `seedAfter` at EVERY
// boundary, INCLUDING the new onBeforeMove status draw (the NEW LEADING draw before
// accuracy) and the secondary random(100) (the NEW TRAILING draw after damage). An
// EXACT cross-decision seed match to game-end + the per-decision STATUS + the final
// winner is the draw-ORDER+COUNT proof: one extra/missing/mis-ordered status or
// secondary draw desyncs the LCG and the seed diverges on some seed.
//
// THE DRAW-COUNT CASES exercised:
//   - a LANDED secondary move draws random(100) (Body Slam/Ice Beam/Thunderbolt/
//     Rock Slide/Sludge Bomb);
//   - a status-IMMUNE-but-DAMAGED target (Thunderbolt→Electric, the para no-ops but
//     the random(100) STILL draws);
//   - a DAMAGE-immune target (Thunderbolt→Ground: immune short-circuits BEFORE
//     moveHit so the secondary random(100) is NOT drawn — a distinct draw count);
//   - the SELF-CONTAINED loop: Body Slam paralyzes turn 1, then the paralyzed mon
//     draws randomChance(1,4) full-para in onBeforeMove BEFORE its own accuracy.
//
// Output: tests/vectors/secondary_golden.txt (same TAB format as fullbattle_golden).
//
// Run:  node src/rust_sim/harness/gen_secondary_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/secondary_golden.txt');
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
  let x = 0x3d8f1c27 >>> 0;
  const step = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  for (let i = 0; i < n; i++) out.push([step() || 1, step() || 1, step() || 1, step() || 1]);
  return out;
}

// The first ACTION to RUN this turn (matching the Rust `first_mover` = the first
// queued move/switch action's side), NOT the first to successfully MOVE. A mon that
// full-paras / is still-asleep / frozen / flinched / confusion-self-hits emits
// `|cant|` (or `|-activate|...|confusion`) and runs its action FIRST without a
// `|move|` line — so we must count `|cant|` and the confusion `-activate` too, else
// a faster-but-cancelled mon would be skipped and the recorded first-mover would
// disagree with the Rust action order. `|switch|` and `|move|` are the normal cases.
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
  return { status: st || '-', stage };
}

// The 5 stat-stage boosts (atk/def/spa/spd/spe) in the SAME index order the Rust
// `MonState.boosts` uses ([atk,def,spa,spd,spe,accuracy,evasion]); we record the five
// STAT stages a secondary can drop/raise. `active.boosts` is a BoostsTable, always
// present (pokemon.ts:419), ints. Accuracy/evasion are out of scope for these moves
// (none of the boost scenarios drop accuracy/evasion) so we record the 5 stat stages.
function boostsOf(a) {
  const b = a && a.boosts ? a.boosts : {};
  return [b.atk | 0, b.def | 0, b.spa | 0, b.spd | 0, b.spe | 0];
}

// The CONFUSION volatile's remaining-turn counter (set by onStart's random(2,6),
// decremented in onBeforeMove). Presence + `.time`. 0 = not confused — matching the
// Rust `MonState.confusion: Option<u8>` (Some(t) iff t>0).
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

// Scan the protocol log between two decision points for the RNG OUTCOMES the new
// draws produce — so the differential can assert WHICH outcome fired (not just the
// post seed). Returns flags per side.
function rngOutcomesSince(log, fromIdx) {
  const out = {
    p1: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false, frozen: false },
    p2: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false, frozen: false },
    secondaryLanded: false,
  };
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    const who = (p[2] || '').startsWith('p1a:') ? 'p1' : (p[2] || '').startsWith('p2a:') ? 'p2' : null;
    if (tag === 'cant' && who) {
      if (p[3] === 'par') out[who].fullpara = true;
      if (p[3] === 'frz') out[who].frozen = true;
      if (p[3] === 'flinch') out[who].flinch = true;
    }
    if (tag === '-curestatus' && who) {
      // a wake (slp) or thaw (frz) shows -curestatus; disambiguate via the [from].
      if ((p[3] || '') === 'slp') out[who].wake = true;
      if ((p[3] || '') === 'frz') out[who].thaw = true;
    }
    if (tag === '-activate' && who && p[3] === 'confusion') {
      // self-hit shows a following |-damage| with [from] confusion.
    }
    if (tag === '-damage' && who && (p[4] || '').includes('confusion')) out[who].selfhit = true;
    if ((tag === '-status' || tag === '-start') && who) out.secondaryLanded = true;
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
    for (let i = 0; i < 16; i++) await tick();

    const seedAfter = battle.prng.getSeed();
    const outcomes = rngOutcomesSince(log, logLenBefore);
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
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
// Each is { id, p1[], p2[], makeScript }. All use REAL SECONDARY moves so status
// is inflicted IN-ENGINE and the onBeforeMove draws fire across the battle.

function scenarios() {
  const S = [];

  const fromPlan = (plan, onForce) => () => {
    let i = 0;
    return (decisionNo, battle, reqState, force) => {
      if (reqState === 'switch') {
        const c = { p1: null, p2: null };
        if (force[0]) c.p1 = onForce(0, battle);
        if (force[1]) c.p2 = onForce(1, battle);
        return c;
      }
      const entry = plan[i] || { p1: 'move 1', p2: 'move 1' };
      i++;
      return entry;
    };
  };
  // A repeating per-move plan (last entry repeats forever).
  const repeat = (entry, onForce) => () => {
    return (decisionNo, battle, reqState, force) => {
      if (reqState === 'switch') {
        const c = { p1: null, p2: null };
        if (force[0]) c.p1 = onForce(0, battle);
        if (force[1]) c.p2 = onForce(1, battle);
        return c;
      }
      return entry;
    };
  };
  const firstLiveBench = (side, battle) => {
    const s = battle.sides[side];
    for (let k = 0; k < s.pokemon.length; k++) {
      const p = s.pokemon[k];
      if (p !== s.active[0] && !p.fainted) return `switch ${k + 1}`;
    }
    return 'pass';
  };

  // --- (1) BODY SLAM (par 30) SELF-CONTAINED LOOP: a fast Tauros Body Slams a bulky
  //   Snorlax every turn — once Snorlax is paralyzed it draws full-para in
  //   onBeforeMove BEFORE its own Body Slam. Both grind to a KO → win. Exercises the
  //   secondary random(100) AND the para onBeforeMove draw end-to-end. ---
  S.push({
    id: 'bodyslam_para_loop',
    p1: [mon('Tauros', ['bodyslam', 'earthquake'], { ability: 'Intimidate', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: repeat({ p1: 'move 1', p2: 'move 1' }, firstLiveBench),
  });

  // --- (2) ICE BEAM (frz 10) — a frozen mon draws the 20% thaw roll in
  //   onBeforeMove. A full-SpA Suicune Ice Beams a bulky SpD Snorlax (NEUTRAL to
  //   Ice → survives ~5-6 Ice Beams, so the 10% freeze realizes across seeds before
  //   the KO, and the frozen Snorlax draws randomChance(1,5) thaw rolls). Snorlax
  //   has NO Leftovers (no heal-stalemate) so the battle is BOUNDED to a win; it
  //   Body Slams back (its own 30% para secondary also fires — extra coverage). ---
  S.push({
    id: 'icebeam_freeze_thaw',
    p1: [mon('Suicune', ['icebeam', 'surf'], { item: 'Leftovers', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'earthquake'], { nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    makeScript: repeat({ p1: 'move 1', p2: 'move 1' }, firstLiveBench),
  });

  // --- (3) SLUDGE BOMB (psn 30) vs a status-IMMUNE-but-DAMAGED target: the
  //   random(100) STILL draws, the poison NO-OPS (Poison & Steel are psn-immune in
  //   gen3). The target must take DAMAGE (so the move HITS and reaches the
  //   secondary): p1 Gengar Sludge Bombs p2 Muk (pure POISON → Poison resists
  //   Poison 0.5× so it's DAMAGED, AND Muk is psn-immune). So every landed Sludge
  //   Bomb draws the secondary random(100) but the psn never applies — the
  //   status-immune-but-damaged draw-count case (DISTINCT from a Steel target where
  //   Poison is 0× → immune short-circuits → NO secondary draw). Gengar's
  //   Thunderbolt finishes; Muk Sludge Bombs back. ---
  S.push({
    id: 'sludgebomb_psn_immune_damaged',
    p1: [mon('Gengar', ['sludgebomb', 'thunderbolt'], { nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Muk', ['sludgebomb', 'shadowball'], { nature: 'Modest', evs: { hp: 252, spd: 252 } })],
    makeScript: repeat({ p1: 'move 1', p2: 'move 1' }, firstLiveBench),
  });

  // --- (3b) THUNDERBOLT (par 10) PARALYZES a NON-Electric target — gen3 Electric is
  //   NOT para-immune, but a clean para-lands case uses a WATER target (SE Electric,
  //   real damage + para). p1 Jolteon Thunderbolt vs p2 bulky Suicune (Water,
  //   2× weak to Electric, NO Leftovers → bounded). Once paralyzed, Suicune draws
  //   randomChance(1,4) full-para + its ×0.25 speed re-orders the turn. Suicune
  //   Surfs back (could be resisted/neutral). Grind to a win as para + full-para
  //   fire across seeds. ---
  S.push({
    id: 'tbolt_para_lands',
    p1: [mon('Jolteon', ['thunderbolt', 'shadowball'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Suicune', ['surf', 'icebeam'], { nature: 'Bold', evs: { hp: 252, def: 252 } })],
    makeScript: repeat({ p1: 'move 1', p2: 'move 1' }, firstLiveBench),
  });

  // --- (4) THUNDERBOLT vs a DAMAGE-immune target (Ground): the move is IMMUNE so it
  //   short-circuits BEFORE moveHit → the secondary random(100) is NOT drawn (a
  //   distinct draw count). p1 Jolteon Thunderbolt vs p2 Flygon (Ground/Dragon,
  //   Levitate too) — immune. p1 also carries Shadow Ball to actually win; p2
  //   Earthquakes (Ground, hits Jolteon). ---
  S.push({
    id: 'tbolt_ground_immune_no_secondary',
    p1: [mon('Jolteon', ['thunderbolt', 'shadowball'], { nature: 'Timid', evs: { spa: 252, spe: 252 } }),
         mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Flygon', ['earthquake', 'rockslide'], { ability: 'Levitate', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'move 1', p2: 'move 1' }, // Tbolt (immune, no secondary) vs EQ
       { p1: 'move 1', p2: 'move 1' }],
      firstLiveBench),
  });

  // --- (5) ROCK SLIDE (flinch 30): a fast Aerodactyl Rock Slides a slower bulky
  //   target every turn — when the flinch lands the target is cant'd (flinch
  //   onBeforeMove, DRAW-FREE — so the seed must NOT advance for the flinch itself,
  //   only the secondary random(100) drew). Grind to a win. Rockhead so no recoil
  //   (Double-Edge unused). ---
  S.push({
    id: 'rockslide_flinch',
    p1: [mon('Aerodactyl', ['rockslide', 'earthquake'], { item: 'Choice Band', ability: 'Rock Head', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: repeat({ p1: 'move 1', p2: 'move 1' }, firstLiveBench),
  });

  // --- (6) SLUDGE BOMB (psn 30): a Gengar Sludge Bombs a bulky Snorlax → poison;
  //   then the residual poison DoT chips (already-modeled), AND a 2nd Sludge Bomb on
  //   the now-poisoned mon STILL draws random(100) but the psn no-ops (already
  //   statused). Exercises both the secondary draw AND the already-statused gate. ---
  S.push({
    id: 'sludgebomb_psn',
    p1: [mon('Gengar', ['sludgebomb', 'thunderbolt'], { nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: repeat({ p1: 'move 1', p2: 'move 1' }, firstLiveBench),
  });

  // --- (7) A LONGER mixed battle: secondary moves + a SWITCH + a post-faint
  //   replacement, all the way to a win. p1 (Snorlax Body Slam + Suicune Ice Beam)
  //   vs p2 (two frail attackers). Switch turn 1, then grind with secondary moves so
  //   status + onBeforeMove draws fire across switches + replacements. ---
  S.push({
    id: 'mixed_secondary_switch_grind',
    p1: [mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Suicune', ['icebeam', 'surf'], { item: 'Leftovers', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Jolteon', ['thunderbolt', 'shadowball'], { nature: 'Timid', evs: { spa: 252, spe: 252 } }),
         mon('Gengar', ['sludgebomb', 'thunderbolt'], { nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    makeScript: fromPlan(
      [{ p1: 'switch 2', p2: 'move 1' }, // p1 pivots Snorlax→Suicune, p2 Tbolt
       { p1: 'move 1', p2: 'move 1' }],  // Ice Beam grind (freeze/para fire)
      firstLiveBench),
  });

  // ───── NEW: STAT-DROP / SELF-BOOST secondaries (the boost-STATE coverage) ─────
  // The boost APPLY is DRAW-FREE (one secondary random(100), then boost() consumes no
  // PRNG), so these prove the boost STAGE state — a wrong/missing apply or wrong
  // stat/target is caught by the per-decision boosts[] assertion (NOT the seed).

  // --- (8) CRUNCH (-1 SpD FOE, gen3 override, 20%): a bulky Tyranitar Crunches a
  //   special wall Suicune. The 20% SpD-drop accumulates across the grind →
  //   the DEFENDER's spd stage goes negative mid-battle. Tyranitar's Sand Stream
  //   weather chips (already-modeled). Suicune Surfs back (SE on Ttar). ---
  S.push({
    id: 'crunch_spd_drop',
    p1: [mon('Tyranitar', ['crunch', 'rockslide'], { item: 'Leftovers', ability: 'Sand Stream', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Suicune', ['surf', 'icebeam'], { item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    makeScript: repeat({ p1: 'move 1', p2: 'move 1' }, firstLiveBench),
  });

  // --- (9) PSYCHIC (-1 SpD FOE, 10%) + SHADOW BALL (-1 SpD FOE, 20%): a Starmie
  //   Psychics a Snorlax (the spd-drop accumulates); Gengar/Snorlax exchange. The
  //   10%/20% SpD drops realize across seeds. NO Leftovers on the wall → bounded. ---
  S.push({
    id: 'psychic_shadowball_spd_drop',
    p1: [mon('Starmie', ['psychic', 'surf'], { item: 'Leftovers', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['shadowball', 'bodyslam'], { nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    makeScript: repeat({ p1: 'move 1', p2: 'move 1' }, firstLiveBench),
  });

  // --- (10) METEOR MASH (+1 Atk SELF, 20%): a Metagross Meteor Mashes a Skarmory.
  //   The 20% self-Atk-boost raises the ATTACKER's atk stage POSITIVE mid-battle (the
  //   self-boost direction, distinct from the foe stat-drop). Skarmory Drill Pecks
  //   back (no Leftovers → bounded). Clear Steel typing so neither is psn/etc. ---
  S.push({
    id: 'meteormash_self_atk',
    p1: [mon('Metagross', ['meteormash', 'earthquake'], { ability: 'Clear Body', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Skarmory', ['drillpeck', 'steelwing'], { nature: 'Impish', evs: { hp: 252, def: 252 } })],
    makeScript: repeat({ p1: 'move 1', p2: 'move 1' }, firstLiveBench),
  });

  // --- (11) WATER PULSE (confusion 20%): a Starmie Water Pulses a bulky non-Water,
  //   non-Own-Tempo, non-Ghost Snorlax. The 20% confusion is inflicted IN-ENGINE → the
  //   random(2,6) duration draw fires (the NEW SEED-bearing draw — a missing draw
  //   desyncs seedAfter), then the confused Snorlax runs its own onBeforeMove confusion
  //   loop (randomChance(1,2) + a self-hit random(16)) on later turns. NO Leftovers on
  //   Snorlax → bounded to a win. Snorlax Body Slams back. ---
  S.push({
    id: 'waterpulse_confusion',
    p1: [mon('Starmie', ['waterpulse', 'thunderbolt'], { item: 'Leftovers', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'earthquake'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: repeat({ p1: 'move 1', p2: 'move 1' }, firstLiveBench),
  });

  return S;
}

const STATUS_TOKENS = { '-': 0, brn: 1, par: 2, slp: 3, frz: 4, psn: 5, tox: 6 };

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# secondary_golden.txt — Gen-3 SECONDARY + onBeforeMove STATUS full-battle golden.');
  lines.push('# Per-decision-boundary STATE(+status)+SEED differential to GAME-END.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <di> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1(fullpara wake thaw selfhit flinch) p2(...) secondaryLanded');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0;
  // Floors: prove the new branches actually realize across the corpus.
  let fullparaRows = 0, wakeRows = 0, thawRows = 0, selfhitRows = 0, flinchRows = 0;
  let secondaryLandedRows = 0, statusedDecRows = 0;
  // NEW boost/confusion-STATE floors: a row where ANY stat stage is non-zero (a foe
  // stat-drop OR a self stat-raise realized), a row where the ATTACKER side has a
  // POSITIVE stat stage (the self-boost direction), and a row where a mon is confused.
  let boostedRows = 0, selfBoostRows = 0, confusionRows = 0;

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);

    let scenDecs = 0;
    for (const seed of seeds) {
      let rec;
      try { rec = await runBattle(sc, seed); } catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); continue; }
      if (rec.gen !== 3) { failures.push(`${sc.id}: expected gen 3, got ${rec.gen}`); break; }
      if (!rec.initSeed || rec.decisions.length === 0) { failures.push(`${sc.id} seed ${seed}: no decisions`); continue; }
      const seedStr = seed.join(',');
      lines.push(['INIT', sc.id, rec.initSeed, seedStr].join('\t'));

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
          oc(d.outcomes.p1), oc(d.outcomes.p2), d.outcomes.secondaryLanded ? 1 : 0,
        ].join('\t'));
        decRows++; scenDecs++;
        if (d.outcomes.p1.fullpara || d.outcomes.p2.fullpara) fullparaRows++;
        if (d.outcomes.p1.wake || d.outcomes.p2.wake) wakeRows++;
        if (d.outcomes.p1.thaw || d.outcomes.p2.thaw) thawRows++;
        if (d.outcomes.p1.selfhit || d.outcomes.p2.selfhit) selfhitRows++;
        if (d.outcomes.p1.flinch || d.outcomes.p2.flinch) flinchRows++;
        if (d.outcomes.secondaryLanded) secondaryLandedRows++;
        if (d.p1.status !== '-' || d.p2.status !== '-') statusedDecRows++;
        const anyBoost = (s) => s.boosts.some((v) => v !== 0);
        if (anyBoost(d.p1) || anyBoost(d.p2)) boostedRows++;
        // A POSITIVE stage on EITHER side proves the self-RAISE direction realized
        // (Meteor Mash +Atk / Ancient Power); foe stat-drops are only ever negative.
        const anyPositive = (s) => s.boosts.some((v) => v > 0);
        if (anyPositive(d.p1) || anyPositive(d.p2)) selfBoostRows++;
        if (d.p1.confusion > 0 || d.p2.confusion > 0) confusionRows++;
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
  }

  if (failures.length) {
    console.error('SECONDARY GOLDEN FAILURES:\n  ' + failures.slice(0, 30).join('\n  '));
    process.exit(1);
  }
  // Floors — every new branch MUST realize across the corpus (else a branch is
  // untested). These are conservative given 30/20/10% events × 60 seeds × 7 scen.
  const need = (label, n, min) => { if (n < min) { console.error(`SECONDARY GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('landed-secondary rows', secondaryLandedRows, 100);
  need('statused-decision rows', statusedDecRows, 100);
  need('full-para rows', fullparaRows, 20);
  need('thaw rows', thawRows, 5);
  need('flinch rows', flinchRows, 10);
  need('WIN rows', winRows, 50);
  // NEW boost/confusion-STATE floors: each new branch must provably realize.
  need('boosted-decision rows', boostedRows, 30);
  need('self-boost (positive-stage) rows', selfBoostRows, 10);
  need('confusion-active rows', confusionRows, 20);
  void STATUS_TOKENS; void wakeRows; void selfhitRows;

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `secondary golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: ${secondaryLandedRows} secondary-landed, ${statusedDecRows} statused, ${fullparaRows} full-para, ` +
    `${wakeRows} wake, ${thawRows} thaw, ${selfhitRows} self-hit, ${flinchRows} flinch, ` +
    `${boostedRows} boosted, ${selfBoostRows} self-boost, ${confusionRows} confused -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
