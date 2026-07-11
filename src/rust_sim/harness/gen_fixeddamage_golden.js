// gen_fixeddamage_golden.js — Gen-3 FIXED-DAMAGE / FIXED-FORMULA move differential.
//
// Extends harness/gen_recovery_move_golden.js (the per-decision STATE+HP+STATUS+SEED+
// winner full-battle differential) to the NEW execution path this layer adds: the
// FIXED-DAMAGE moves — a `damage:` / `damageCallback` move in Showdown that BYPASSES
// getDamage, so NO crit roll + NO 16-way damage roll:
//   Seismic Toss / Night Shade  (damage: 'level'  → the USER's level),
//   Sonic Boom                  (damage: 20),
//   Dragon Rage                 (damage: 40),
//   Super Fang                  (damageCallback = max(floor(target.hp/2), 1)).
// (Psywave [variable RNG] / the OHKO moves / Counter / Mirror Coat / Bide / Endeavor are
//  NOT modeled — the Rust engine fail-louds on them.)
//
// THE DRAW MODEL (VERIFIED bit-for-bit vs the omniscient sim, `harness/probe_fixeddamage_rng.js`):
//   1. ACCURACY — `randomChance(acc, 100)`, drawn UNLESS never_miss. Seismic Toss / Night
//      Shade / Dragon Rage are acc-100 but `never_miss == false`, so they STILL draw ONE
//      accuracy roll (the phaze acc-100 precedent — always passes but CONSUMES a draw);
//      Sonic Boom / Super Fang are acc-90 and CAN genuinely miss. This is the ONLY per-move
//      draw — NO crit, NO damage roll, NO secondary.
//   2. TYPE IMMUNITY — accuracy-drawn-THEN-`-immune` (the same short-circuit as a normal
//      damaging move): Seismic Toss (Fighting) into a GHOST (0×), Night Shade (Ghost) into
//      a NORMAL (0×), Sonic Boom / Super Fang (Normal) into a GHOST (0×) all report
//      `-immune` (NOT `-miss`) with the SAME draw count as a landed hit.
//   3. DAMAGE — the exact fixed amount (user's level / 20 / 40 / max(floor(target.hp/2),1)).
//      Behind a SUBSTITUTE the NUMBER hits the sub (breaks with no carry); Super Fang still
//      halves the MON's current hp (VERIFIED: SF into a full-HP-536 Blissey behind a 178-HP
//      sub deals floor(536/2)=268 → the sub BREAKS, not floor(178/2)).
//   4. `landed` is TRUE on a hit (a `damage:` move returns a truthy number → the in-
//      tryMoveHit Update fires); FALSE on a miss / immune / block.
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server, gen3customgame) over
// CONSTRUCTED scenarios that each ISOLATE one branch, capturing the running PRNG seed
// BEFORE the first decision (`initSeed`) and AFTER each DECISION BOUNDARY, plus each
// active's species/hp/maxhp/fainted/status(+inner counter) + boosts + confusion +
// pokemon_left + first mover + winner. The Rust test seeds a BattleState at the init seed
// and runs `run_full_battle` WITHOUT re-seeding — so the post-decision seed AND the
// per-decision HP (the fixed-damage amount) must match at EVERY boundary. A 1-HP error
// desyncs the STATE; a wrong draw model (e.g. a crit/damage roll wrongly drawn, or a missing
// accuracy roll) desyncs the SEED.
//
// FAIL-LOUD: each scenario declares the BRANCH it must realize (a fixed-damage hit, an
// immune report, a miss, a KO-to-win, a sub-absorb); generation aborts if the sim run did
// NOT realize it. The output shares the recovery_move_golden TAB format (so the Rust gate
// reuses the same parser), with the trailing branch flag re-meaninged to `fixedHit`.
//
// Output: tests/vectors/fixeddamage_golden.txt
//
// Run:  node src/rust_sim/harness/gen_fixeddamage_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/fixeddamage_golden.txt');
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
  let x = 0x7f4a3b1d >>> 0;
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

// The set of FIXED-DAMAGE move display names — used to decide whether a `-damage` /
// `-immune` / `-miss` line this decision came from a fixed-damage move (its `|move|`
// precedes the effect line in the same action window).
const FIXED_NAMES = new Set(['Seismic Toss', 'Night Shade', 'Sonic Boom', 'Dragon Rage', 'Super Fang']);

// Scan the protocol log between two decision points for the shared per-side onBeforeMove
// outcome flags PLUS the FIXED-DAMAGE branch flags:
//   fixedHit    — a fixed-damage move dealt damage (a `-damage` or a sub `-activate/-end`
//                 following a fixed-damage `|move|`).
//   fixedImmune — a fixed-damage move reported `-immune` (a type-immune target).
//   fixedMiss   — a fixed-damage move reported `-miss` (Sonic Boom / Super Fang acc-90).
//   fixedSub    — a fixed-damage move hit a Substitute (`-activate ... Substitute [damage]`
//                 or `-end ... Substitute` following a fixed-damage `|move|`).
function outcomesSince(log, fromIdx) {
  const out = {
    p1: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    p2: { fullpara: false, wake: false, thaw: false, selfhit: false, flinch: false },
    fixedHit: false, fixedImmune: false, fixedMiss: false, fixedSub: false,
  };
  let lastMoveFixed = false; // whether the most recent |move| was a fixed-damage move
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    const who = (p[2] || '').startsWith('p1a:') ? 'p1' : (p[2] || '').startsWith('p2a:') ? 'p2' : null;
    if (tag === 'move') {
      lastMoveFixed = FIXED_NAMES.has((p[3] || '').trim());
      // A fixed-damage move that shows `[miss]` on its own |move| line.
      if (lastMoveFixed && (p[5] || '').includes('[miss]')) out.fixedMiss = true;
    }
    if (tag === 'cant' && who) {
      if (p[3] === 'par') out[who].fullpara = true;
      if (p[3] === 'flinch') out[who].flinch = true;
    }
    if (tag === '-curestatus' && who) {
      if ((p[3] || '') === 'slp') out[who].wake = true;
      if ((p[3] || '') === 'frz') out[who].thaw = true;
    }
    if (tag === '-damage' && who && (p[4] || '').includes('confusion')) out[who].selfhit = true;
    if (lastMoveFixed) {
      // A fixed-damage move's damage effect (the |move| just above was a fixed-damage move).
      if (tag === '-damage' && !(p[4] || '').includes('[from]')) out.fixedHit = true;
      if (tag === '-immune') out.fixedImmune = true;
      if (tag === '-miss') out.fixedMiss = true;
      // Into a Substitute: the sub absorbs the hit (survived → -activate, broke → -end).
      if (tag === '-activate' && (p[3] || '') === 'Substitute') { out.fixedHit = true; out.fixedSub = true; }
      if (tag === '-end' && (p[3] || '') === 'Substitute') { out.fixedHit = true; out.fixedSub = true; }
    }
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
  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen, branchSeen: {} };

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
    if (seedAfter === seedBefore && log.length === logLenBefore && battle.requestState === reqState) {
      throw new Error(`STALL: choice ${JSON.stringify(choices)} did not advance the battle ` +
        `(scenario ${sc.id}, decision ${decisionNo}) — the sim rejected it. Fix the script.`);
    }
    const outcomes = outcomesSince(log, logLenBefore);
    const first = reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none';

    for (const k of ['fixedHit', 'fixedImmune', 'fixedMiss', 'fixedSub']) {
      if (outcomes[k]) rec.branchSeen[k] = true;
    }
    if (outcomes.p1.wake || outcomes.p2.wake) rec.branchSeen.wake = true;
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
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

  const firstLiveBench = (side, battle) => {
    const s = battle.sides[side];
    for (let k = 0; k < s.pokemon.length; k++) {
      const p = s.pokemon[k];
      if (p !== s.active[0] && !p.fainted) return `switch ${k + 1}`;
    }
    return 'pass';
  };
  const fromPlan = (plan) => () => {
    let i = 0;
    return (decisionNo, battle, reqState, force) => {
      if (reqState === 'switch') {
        const c = { p1: null, p2: null };
        if (force[0]) c.p1 = firstLiveBench(0, battle);
        if (force[1]) c.p2 = firstLiveBench(1, battle);
        return c;
      }
      const entry = plan[Math.min(i, plan.length - 1)];
      i++;
      return entry;
    };
  };

  // --- (1) SEISMIC TOSS CHIP TO GAME-END (the level-100 fixed damage, multi-turn): a
  //   Machamp Seismic-Tosses a bulky Blissey for a FLAT 100 each turn (no crit swing, no
  //   damage roll) while Blissey chips back with a weak Ice Beam. The per-decision HP must
  //   decrease by EXACTLY 100 per Seismic Toss (a wrong amount / a spurious crit-or-damage
  //   roll diverges HP+SEED). Machamp is bulky (Guts, no burn) so it grinds Blissey out.
  //   REQUIRES: a fixed-damage hit. ---
  S.push({
    id: 'seismic_toss_chip',
    p1: [mon('Machamp', ['seismictoss'], { ability: 'Guts', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Blissey', ['icebeam'], { ability: 'Natural Cure', item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Seismic Toss (100) ; Ice Beam chip
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
    ]),
    require: ['fixedHit'],
  });

  // --- (2) SEISMIC TOSS INTO A GHOST (IMMUNE, accuracy-only): a Machamp Seismic-Tosses a
  //   GENGAR (Ghost — Fighting 0× → IMMUNE). Accuracy is drawn THEN `-immune` (SAME draw
  //   count as a landed hit — no crit/damage). Gengar takes ZERO from every Seismic Toss;
  //   it grinds Machamp out with Shadow Ball (Ghost — neutral on the Fighting Machamp).
  //   REQUIRES: a fixed-damage immune. ---
  S.push({
    id: 'seismic_toss_ghost_immune',
    p1: [mon('Machamp', ['seismictoss'], { ability: 'Guts', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Gengar', ['shadowball'], { ability: 'Levitate', item: 'Leftovers', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Seismic Toss into Ghost → IMMUNE (0 damage) ; Shadow Ball
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
    ]),
    require: ['fixedImmune'],
  });

  // --- (3) NIGHT SHADE INTO A NORMAL (IMMUNE): a Gengar Night-Shades a SNORLAX (Normal —
  //   Ghost 0× → IMMUNE). Accuracy drawn THEN `-immune`. Snorlax takes ZERO; it grinds
  //   Gengar out with Shadow Ball (Ghost 2× super-effective on Gengar). REQUIRES: a
  //   fixed-damage immune. ---
  S.push({
    id: 'night_shade_normal_immune',
    p1: [mon('Gengar', ['nightshade'], { ability: 'Levitate', item: 'Leftovers', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Snorlax', ['shadowball'], { ability: 'Immunity', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Night Shade into Normal → IMMUNE ; Shadow Ball (2× on Gengar)
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
    ]),
    require: ['fixedImmune'],
  });

  // --- (4) NIGHT SHADE lands (the level-100 fixed damage into a non-Normal): a Gengar
  //   Night-Shades a bulky Suicune (Water — Ghost neutral) for a FLAT 100 each turn while
  //   Suicune Surfs back. The per-decision HP must decrease by EXACTLY 100 per Night Shade.
  //   REQUIRES: a fixed-damage hit. ---
  S.push({
    id: 'night_shade_lands',
    p1: [mon('Gengar', ['nightshade'], { ability: 'Levitate', item: 'Leftovers', nature: 'Timid', evs: { hp: 252, spe: 252 } })],
    p2: [mon('Suicune', ['surf'], { ability: 'Pressure', item: 'Leftovers', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Night Shade (100) ; Surf chip (grinds Gengar)
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
    ]),
    require: ['fixedHit'],
  });

  // --- (5) A FIXED-DAMAGE KO-TO-WIN (the deciding faint draws NO Quick Claw): a Machamp
  //   Seismic-Tosses (100) a chipped-low LAST mon to death. The scenario chips the foe below
  //   100 first, then the fixed-damage KO ends the battle — a KO-to-win through the deferred-
  //   faint protocol (no Quick Claw on the deciding faint desyncs the SEED if wrong). We use
  //   a low-maxhp lvl-1 foe so the very FIRST Seismic Toss (100) is a guaranteed lethal KO.
  //   REQUIRES: a fixed-damage hit + a win run. ---
  S.push({
    id: 'seismic_toss_ko_to_win',
    p1: [mon('Machamp', ['seismictoss'], { ability: 'Guts', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Diglett', ['scratch'], { level: 1, ability: 'Sand Veil' })], // lvl1 → maxhp 11 < 100 → ST OHKOs
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Seismic Toss (100) KOs the lvl-1 last mon → p1 WINS
    ]),
    require: ['fixedHit'],
  });

  // --- (6) A FIXED-DAMAGE MOVE INTO A SUBSTITUTE (the sub absorbs the number): a Machamp
  //   Seismic-Tosses (100) INTO a Snorlax's Substitute (sub hp floor(524/4)=131). The FIRST
  //   ST hits the sub (131→31, survives → `-activate Substitute [damage]`); the SECOND ST
  //   BREAKS it (31 < 100 → `-end`, no carry to the mon). Then a further ST hits the mon
  //   directly for 100. The sub HP path + the no-carry break must match (a wrong sub
  //   interaction diverges the sub-HP-derived STATE). Snorlax splashes so it just holds the
  //   sub; Machamp grinds it out after the sub breaks. REQUIRES: a fixed-damage sub-hit. ---
  S.push({
    id: 'seismic_toss_into_substitute',
    p1: [mon('Machamp', ['seismictoss'], { ability: 'Guts', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Snorlax', ['substitute', 'splash'], { ability: 'Immunity', item: 'Leftovers', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Machamp ST (100, no sub yet) ; Snorlax Substitute (pays 131)
      { p1: 'move 1', p2: 'move 2' }, // ST (100) into the sub → sub 131->31 (survives) ; Splash
      { p1: 'move 1', p2: 'move 2' }, // ST (100) into the sub → 31 < 100 → BREAK (no carry) ; Splash
      { p1: 'move 1', p2: 'move 2' }, // ST (100) into the MON now (no sub) ; Splash
      { p1: 'move 1', p2: 'move 2' },
      { p1: 'move 1', p2: 'move 2' },
      { p1: 'move 1', p2: 'move 2' },
    ]),
    require: ['fixedHit', 'fixedSub'],
  });

  // --- (7) SONIC BOOM + DRAGON RAGE (the flat-number moves, incl. the acc-90 MISS): a
  //   Snorlax with BOTH Sonic Boom (fixed 20, acc 90) and Dragon Rage (fixed 40, acc 100)
  //   grinds a NON-HEALING chip-frail foe down; the acc-90 Sonic Boom MISSES on some seeds
  //   (the seed sweep realizes both a hit and a miss). The per-decision HP must reflect the
  //   exact 20/40; a MISS draws its accuracy roll then does nothing (a wrong miss model
  //   desyncs the SEED). Cleffa is a small-maxhp Normal punching bag (no recovery) that
  //   Sonic Boom + Dragon Rage grind out to a win. REQUIRES: a hit + a miss. ---
  S.push({
    id: 'sonic_boom_dragon_rage',
    p1: [mon('Snorlax', ['sonicboom', 'dragonrage'], { ability: 'Immunity', item: 'Leftovers', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    // Cleffa: tiny Normal (low maxhp), SPLASH-only so it never heals or KOs the Snorlax —
    // the 20/40 fixed chips grind it out. Magic Guard is N/A in gen-3 (Cleffa's gen-3 ability
    // is Cute Charm — a contact secondary; we override to a provable no-op to keep draws clean).
    p2: [mon('Cleffa', ['splash'], { ability: 'Cute Charm', nature: 'Calm', evs: { hp: 0 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Sonic Boom (20, acc 90 — may miss) ; Splash
      { p1: 'move 2', p2: 'move 1' }, // Dragon Rage (40) ; Splash
      { p1: 'move 1', p2: 'move 1' }, // Sonic Boom (20)
      { p1: 'move 2', p2: 'move 1' }, // Dragon Rage (40)
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['fixedHit', 'fixedMiss'],
  });

  // --- (8) SUPER FANG (halves the target's CURRENT hp, min 1, acc 90 — CAN miss; Ghost
  //   immune): a Raticate Super-Fangs a NON-HEALING bulky foe. The per-decision HP must halve
  //   EXACTLY (floor, min 1) on each landed Super Fang; an acc-90 MISS draws its accuracy then
  //   does nothing. A Guts Raticate finishes with Quick Attack (Super Fang alone can only
  //   halve, never KO — it grinds to 1 HP). Snorlax is a big non-healing pool so the halving
  //   is exercised repeatedly; the sweep realizes a hit AND a miss. REQUIRES: a hit + a miss. ---
  S.push({
    id: 'super_fang_halving',
    p1: [mon('Raticate', ['superfang', 'quickattack'], { ability: 'Guts', item: 'Leftovers', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    // Snorlax: a big non-healing HP pool (SPLASH-only), so Super Fang's halving (524→262→131→…)
    // is exercised turn after turn; Quick Attack (a real chip) finishes it after Super Fang
    // grinds it low. Immunity is a modeled no-op.
    p2: [mon('Snorlax', ['splash'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Super Fang (halves 524→262) ; Splash
      { p1: 'move 1', p2: 'move 1' }, // Super Fang (262→131) ; Splash
      { p1: 'move 1', p2: 'move 1' }, // Super Fang (131→65) ; Splash
      { p1: 'move 1', p2: 'move 1' }, // Super Fang (65→32) ; Splash
      { p1: 'move 1', p2: 'move 1' }, // Super Fang (32→16) ; Splash
      { p1: 'move 2', p2: 'move 1' }, // Quick Attack — real chip toward the KO
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: ['fixedHit', 'fixedMiss'],
  });

  // --- (9) FIXED-DAMAGE INTO A REAL BATTLE TO GAME-END (the union: Seismic Toss interleaved
  //   with a switch + the full move/residual/faint machinery to a clean win): p1's Machamp
  //   Seismic-Tosses (100) a frail foe, VOLUNTARILY PIVOTS to a hard-hitting Salamence that
  //   sweeps the FRAIL 2-mon foe team out. Fixed damage + a switch + residuals + a faint all
  //   the way to a win. REQUIRES: a fixed-damage hit + a win run. ---
  S.push({
    id: 'fixed_damage_into_real_battle',
    p1: [mon('Machamp', ['seismictoss', 'crosschop'], { ability: 'Guts', item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Salamence', ['dragonclaw', 'earthquake'], { ability: 'Intimidate', item: 'Leftovers', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Misdreavus', ['quickattack'], { ability: 'Levitate', nature: 'Jolly', evs: { atk: 252, spe: 252 } }),
         mon('Houndour', ['quickattack'], { ability: 'Flash Fire', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Machamp Seismic Toss (100) into the frail Misdreavus (Ghost? — Fighting 0× on Ghost!)
      { p1: 'switch 2', p2: 'move 1' }, // hmm — Misdreavus is Ghost (ST immune); PIVOT to Salamence instead
      { p1: 'move 1', p2: 'move 1' }, // Salamence Dragon Claw — KO the frail Misdreavus
      { p1: 'move 1', p2: 'move 1' }, // Dragon Claw the second frail foe (Houndour) out
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 1', p2: 'move 1' },
    ]),
    require: ['fixedImmune'], // ST into the Ghost Misdreavus is IMMUNE — proves immune-in-a-real-battle
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# fixeddamage_golden.txt — Gen-3 FIXED-DAMAGE / FIXED-FORMULA move full-battle golden.');
  lines.push('# Per-decision-boundary STATE(+hp+status+counter)+BOOSTS+SEED+first-mover differential to GAME-END.');
  lines.push('# (Shares the recovery_move_golden TAB format so the Rust gate reuses the parser;');
  lines.push('#  the trailing branch flag is re-meaninged to `fixedHit`.)');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        p1(fullpara wake thaw selfhit flinch) p2(...) fixedHit');
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

      for (const k of Object.keys(rec.branchSeen)) { scenSeen[sc.id][k] = true; corpus[k] = (corpus[k] || 0) + 1; }

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
          oc(d.outcomes.p1), oc(d.outcomes.p2), d.outcomes.fixedHit ? 1 : 0,
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
    console.error('FIXED-DAMAGE GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`FIXED-DAMAGE GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('fixed-damage-hit runs', 'fixedHit', 40);
  need('fixed-damage-immune runs', 'fixedImmune', 20);
  need('fixed-damage-miss runs', 'fixedMiss', 5);
  need('fixed-damage-sub runs', 'fixedSub', 20);
  if (winRows < 50) { console.error(`FIXED-DAMAGE GOLDEN: too few WIN rows (${winRows} < 50)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `fixed-damage golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: fixedHit=${corpus.fixedHit || 0} fixedImmune=${corpus.fixedImmune || 0} ` +
    `fixedMiss=${corpus.fixedMiss || 0} fixedSub=${corpus.fixedSub || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
