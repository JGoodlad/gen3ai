// gen_ability_batch4_golden.js — the BATCH-4 CLASS-SWEEP golden (`gen3_ability_batch4_v1`):
// the final mechanics tail, each proven bit-for-bit over full battles to game-end.
//
//   TRUANT — onBeforeMove priority 9 (after slp/frz 10, before flinch 8): `|cant|…|ability:
//     Truant` iff `truantTurn`, DRAW-FREE (a loaf turn draws NOTHING for the loafer — no para
//     roll either); onSwitchIn arms `truantTurn = turn !== 0`; the order-27 residual toggles it
//     (a mid-turn entrant is toggled back the same turn and MOVES its first full turn). A
//     speed-tied Truant MIRROR adds ONE residual tie-shuffle draw (the order-27 group).
//   INNER FOCUS — blocks the flinch volatile at the APPLY (`onTryAddVolatile` → null): the
//     secondary random(100) STILL DRAWS (draw-count-identical to a landed flinch — the control
//     pair pins it), the flinch just never sticks. CONTRAST Shield Dust (filters the draw).
//   SHADOW TAG — `onFoeTrapPokemon` traps UNCONDITIONALLY (no grounded/type gate; a MIRROR is
//     mutually trapped), DRAW-FREE (0 extra draws). Asserted via the per-decision TRAPPED columns.
//   CUTE CHARM + ATTRACT — a damaging CONTACT hit into the holder draws randomChance(1,3)
//     UNCONDITIONALLY (the gender gate lives inside attract.onStart — a same-gender pair still
//     draws, the volatile fails draw-free); on a pass the ATTACKER is attracted: onBeforeMove
//     priority 2 (confusion 3 > attract 2 > par 1) emits `-activate` ALWAYS then draws
//     randomChance(1,2) (cant on pass); cleared when the SOURCE leaves / the HOLDER switches out.
//   COLOR CHANGE — onDamagingHit type override (`types_override = [move.type]`): DRAW-FREE, NOT
//     behind a sub, not on the KO hit, never for typeless ???; feeds STAB/chart/status-immunity/
//     sand-immunity reads; switch-out reverts.
//   KING'S ROCK — onModifyMove appends a `{chance:10, flinch}` TRAILING secondary to LISTED
//     moves: rolled AFTER the move's own secondary, BEFORE the foe's contact proc; Serene Grace
//     doubles to 20; Shield Dust filters (no draw); a fixed-damage listed move (Seismic Toss)
//     procs too.
//   FOCUS BAND — onDamage `randomChance(1,10)` draws FIRST on EVERY Damage event into the holder
//     (move hits, burn chips, Spikes, …; NOT sub-absorbed hits); survive-at-1 only on a lethal
//     MOVE hit.
//
// THE PROOF (the batch-2 per-decision STATE+HP+STATUS+SEED differential + the trapping golden's
// per-side TRAPPED columns): drive the OMNISCIENT in-process BattleStream over constructed full
// battles to GAME-END; the Rust test replays from the init seed WITHOUT re-seeding.
//
// Output: tests/vectors/ability_batch4_golden.txt
// Run:  node src/rust_sim/harness/gen_ability_batch4_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/ability_batch4_golden.txt');
const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };

// GENDERS ARE ALWAYS EXPLICIT ('N' unless a scenario needs M/F): an unspecified gender on a
// ratio species makes the sim DRAW `battle.sample(['M','F'])` at construction — an init draw
// the port does not model (probe-verified: the post-start seed differs).
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

// 60 fixed seeds — enough for the 1/10 procs (King's Rock, Focus Band) to realize repeatedly
// and for the 1/3 Cute Charm + 1/2 attract-immobilize to exercise both branches.
const seeds = [];
{
  let s = 0x4d5e6f70 >>> 0;
  const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s; };
  for (let i = 0; i < 60; i++) seeds.push([rng() % 65536, rng() % 65536, rng() % 65536, rng() % 65536]);
}

// The AUTHORITATIVE trapped fact (the trapping golden's source): the sim's internal
// `pokemon.trapped` truthiness.
function trappedOf(battle, side) {
  const a = battle.sides[side].active[0];
  return !!(a && a.trapped);
}

function snap(battle, side) {
  const a = battle.sides[side].active[0];
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status: a.status || '-', left: battle.sides[side].pokemonLeft,
    trapped: trappedOf(battle, side),
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

// First ACTOR of the turn: the first |move|/|cant|/|switch| line. `switch` is included
// (unlike the earlier batch goldens, whose plans never voluntarily switched on a move
// request): a voluntary switch ACTS FIRST in the sim's action queue — exactly what the
// port's `first_mover` records — so a switch-turn's first mover is the switching side.
function firstMoverSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    if ((p[1] === 'move' || p[1] === 'cant' || p[1] === 'switch') && p.length >= 3) {
      const a = (p[2] || '').trim();
      if (a.startsWith('p1a:')) return 'p1';
      if (a.startsWith('p2a:')) return 'p2';
    }
  }
  return 'none';
}

// Per-scenario coverage: did THIS class's observable effect fire this decision?
function coverageMarker(log, fromIdx, sc) {
  const has = (re) => {
    for (let i = fromIdx; i < log.length; i++) if (re.test(log[i])) return true;
    return false;
  };
  switch (sc.cover) {
    case 'truant_cant': return has(/\|cant\|.*\|ability: Truant/);
    case 'flinch_cant': return has(/\|cant\|.*\|flinch/);
    case 'attract_start': return has(/\|-start\|.*\|Attract\|\[from\] ability: Cute Charm/);
    case 'attract_cant': return has(/\|cant\|.*\|Attract/);
    case 'typechange': return has(/\|-start\|.*\|typechange\|.*Color Change/);
    case 'focusband': return has(/\|-activate\|.*\|item: Focus Band/);
    case 'burn_chip': return has(/\|-damage\|.*\|\[from\] brn/);
    case 'spikes_chip': return has(/\|-damage\|.*\|\[from\] Spikes/);
    case 'none': return false;
    default: throw new Error(`unknown cover kind ${sc.cover}`);
  }
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

  const rec = { initSeed: null, decisions: [], winner: null, ended: false, coverRows: 0 };

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

    let choices;
    if (reqState === 'switch') {
      choices = { p1: force[0] ? 'switch 2' : null, p2: force[1] ? 'switch 2' : null };
    } else {
      // Sanitize a planned voluntary switch whose bench target is FAINTED (the sim would
      // reject it): fall back to move 1. Keeps cycling plans legal after a bench loss.
      const sanitize = (c, sideIdx) => {
        const m = c && c.match(/^switch\s+(\d+)$/);
        if (!m) return c;
        const target = battle.sides[sideIdx].pokemon[Number(m[1]) - 1];
        return target && !target.fainted ? c : 'move 1';
      };
      const p1c = sanitize(sc.plan1[decisionNo % sc.plan1.length], 0);
      const p2c = sanitize(sc.plan2[decisionNo % sc.plan2.length], 1);
      choices = { p1: p1c, p2: p2c };
    }

    const logLenBefore = log.length;
    if (choices.p1) { try { streams.omniscient.write(`>p1 ${choices.p1}`); } catch (e) {} }
    if (choices.p2) { try { streams.omniscient.write(`>p2 ${choices.p2}`); } catch (e) {} }
    for (let i = 0; i < 16; i++) await tick();

    const seedAfter = battle.prng.getSeed();
    if (seedAfter === seedBefore && log.length === logLenBefore && battle.requestState === reqState) {
      throw new Error(`STALL: choice ${JSON.stringify(choices)} did not advance the battle ` +
        `(scenario ${sc.id}, decision ${decisionNo}) — the sim rejected it (a trapped switch?). Fix the plan.`);
    }
    const p1s = snap(battle, 0);
    const p2s = snap(battle, 1);
    const covered = reqState === 'move' && coverageMarker(log, logLenBefore, sc);
    if (covered) rec.coverRows++;
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: p1s,
      p2: p2s,
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      covered,
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

function encodeChoice(c) {
  if (!c) return '-';
  const m = c.match(/^move\s+(\d+)$/);
  if (m) return `m${Number(m[1]) - 1}`;
  const s = c.match(/^switch\s+(\d+)$/);
  if (s) return `s${Number(s[1]) - 1}`;
  throw new Error(`unencodable choice ${JSON.stringify(c)}`);
}

// ── Scenarios ────────────────────────────────────────────────────────────────
function scenarios() {
  const S = [];

  // ── TRUANT ──
  S.push({
    id: 'truant_alternates',
    // Slaking Body Slams every other turn (loaf turns draw NOTHING); Swampert chips back.
    p1: [mon('Slaking', ['bodyslam', 'earthquake'], { ability: 'Truant', nature: 'Adamant', evs: { atk: 252 } })],
    p2: [mon('Swampert', ['surf', 'surf'], { nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'truant_cant',
  });
  S.push({
    id: 'truant_mirror_tie',
    // A speed-TIED Truant mirror: the order-27 residual group draws ONE extra tie-shuffle
    // every turn (probe Q4's 9-vs-8) — any mismatch desyncs the whole stream.
    p1: [mon('Slaking', ['bodyslam', 'bodyslam'], { ability: 'Truant', nature: 'Adamant', evs: { atk: 252 } })],
    p2: [mon('Slaking', ['bodyslam', 'bodyslam'], { ability: 'Truant', nature: 'Adamant', evs: { atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'truant_cant',
  });
  S.push({
    id: 'truant_switch_rearm',
    // A voluntary pivot re-arms truantTurn (onSwitchIn true → the same-turn residual toggles
    // it back → Slaking MOVES its first full turn after re-entry — probe Q3).
    p1: [mon('Slaking', ['bodyslam', 'earthquake'], { ability: 'Truant', nature: 'Adamant', evs: { atk: 252 } }),
         mon('Zangoose', ['scratch', 'scratch'], { ability: 'Immunity' })],
    p2: [mon('Suicune', ['surf', 'surf'], { nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    plan1: ['move 1', 'switch 2', 'move 1', 'switch 2'], plan2: ['move 1'],
    cover: 'truant_cant',
  });
  S.push({
    id: 'truant_paralyzed_loaf',
    // A paralyzed Slaking's LOAF turn draws NO para roll (truant 9 > par 1 short-circuits —
    // probe Q2b); its MOVE turns draw the para roll first.
    p1: [mon('Slaking', ['bodyslam', 'earthquake'], { ability: 'Truant', nature: 'Adamant', evs: { atk: 252 } })],
    p2: [mon('Jolteon', ['thunderwave', 'thunderbolt'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    plan1: ['move 1'], plan2: ['move 1', 'move 2'],
    cover: 'truant_cant',
  });

  // ── INNER FOCUS (+ the flinch control pair) ──
  S.push({
    id: 'innerfocus_blocks_flinch',
    // Bite's 30% flinch secondary DRAWS but never sticks (Inner Focus blocks at the apply):
    // Snorlax always moves. The seed timeline must match the sim's (the roll IS drawn).
    p1: [mon('Jolteon', ['bite', 'thunderbolt'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'bodyslam'], { ability: 'Inner Focus', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'flinch_cant', // MUST stay 0 — asserted below
  });
  S.push({
    id: 'flinch_control_thickfat',
    // The SAME board with a non-blocking ability: the flinch lands (cant rows > 0) on the
    // SAME seeds — the pair pins block-at-the-apply vs filter-the-draw.
    p1: [mon('Jolteon', ['bite', 'thunderbolt'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'bodyslam'], { ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'flinch_cant',
  });

  // ── SHADOW TAG (the TRAPPED columns are the assert) ──
  S.push({
    id: 'shadowtag_traps_unconditionally',
    // p1's Golduck holds Shadow Tag (customgame — no ability legality): p2's SKARMORY (Flying!)
    // is trapped anyway (no grounded gate). p2 has a bench it can never voluntarily reach.
    p1: [mon('Golduck', ['surf', 'surf'], { ability: 'Shadow Tag', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Skarmory', ['drillpeck', 'drillpeck'], { ability: 'Keen Eye', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Snorlax', ['bodyslam', 'bodyslam'], { ability: 'Thick Fat' })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'none',
  });
  S.push({
    id: 'shadowtag_mirror_mutual',
    // A Shadow-Tag MIRROR is MUTUALLY trapped (no fellow-holder exemption on the trap itself)
    // and adds ZERO draws (the seed parity is the draw-freeness proof).
    p1: [mon('Golduck', ['surf', 'surf'], { ability: 'Shadow Tag', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Golduck', ['surf', 'surf'], { ability: 'Shadow Tag', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'none',
  });

  // ── CUTE CHARM + ATTRACT ──
  S.push({
    id: 'cutecharm_attracts_the_attacker',
    // M Zangoose Scratches F Miltank (Cute Charm): the 1/3 roll draws on every damaging
    // contact hit; a pass attracts Zangoose (the -activate + 1/2 immobilize follow).
    p1: [mon('Zangoose', ['scratch', 'scratch'], { ability: 'Immunity', gender: 'M', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Miltank', ['bodyslam', 'bodyslam'], { ability: 'Cute Charm', gender: 'F', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'attract_start',
  });
  S.push({
    id: 'cutecharm_gender_control',
    // F-into-F: the 1/3 roll STILL DRAWS (the seed timeline shifts identically) but the
    // volatile fails draw-free — attract_start must stay 0.
    p1: [mon('Zangoose', ['scratch', 'scratch'], { ability: 'Immunity', gender: 'F', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Miltank', ['bodyslam', 'bodyslam'], { ability: 'Cute Charm', gender: 'F', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'attract_start', // MUST stay 0 — asserted below
  });
  S.push({
    id: 'cutecharm_source_leaves_clears',
    // Miltank (the attract SOURCE) pivots out — the attract onUpdate removes the volatile;
    // Zangoose is free again (its later turns draw no attract 1/2).
    p1: [mon('Zangoose', ['scratch', 'scratch'], { ability: 'Immunity', gender: 'M', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Miltank', ['bodyslam', 'bodyslam'], { ability: 'Cute Charm', gender: 'F', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Chansey', ['seismictoss', 'seismictoss'], { ability: 'Natural Cure', gender: 'F', evs: { hp: 252 } })],
    plan1: ['move 1'], plan2: ['move 1', 'switch 2', 'move 1', 'switch 2'],
    cover: 'attract_cant',
  });

  // ── COLOR CHANGE ──
  S.push({
    id: 'colorchange_chart_reads',
    // TBolt → Kecleon becomes Electric; the alternating Earthquake is then SUPER-EFFECTIVE
    // (the chart read through the override) and re-overrides to Ground. Kecleon Surfs back.
    p1: [mon('Jolteon', ['thunderbolt', 'earthquake'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Kecleon', ['surf', 'surf'], { ability: 'Color Change', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    plan1: ['move 1', 'move 2'], plan2: ['move 1'],
    cover: 'typechange',
  });
  S.push({
    id: 'colorchange_status_immunity',
    // Sludge Bomb → Kecleon becomes Poison → the later Toxic FAILS on the Poison type (the
    // status type-immunity through the override).
    p1: [mon('Gengar', ['sludgebomb', 'toxic'], { ability: 'Levitate', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Kecleon', ['surf', 'surf'], { ability: 'Color Change', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    plan1: ['move 1', 'move 2'], plan2: ['move 1'],
    cover: 'typechange',
  });

  // ── KING'S ROCK ──
  S.push({
    id: 'kingsrock_flinches',
    // Slash (listed, NO own secondary) + KR: ONE extra trailing random(100); a <10 roll
    // flinches the slower Snorlax (its Body Slam is cant'd).
    p1: [mon('Zangoose', ['slash', 'slash'], { ability: 'Immunity', item: "King's Rock", nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'bodyslam'], { ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'flinch_cant',
  });
  S.push({
    id: 'kingsrock_serene_grace_doubles',
    // Serene Grace doubles the appended chance to 20 (the probe's 10/15-land split).
    p1: [mon('Blissey', ['slash', 'slash'], { ability: 'Serene Grace', gender: 'F', nature: 'Adamant', item: "King's Rock", evs: { hp: 252, atk: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'bodyslam'], { ability: 'Thick Fat', nature: 'Adamant', evs: { atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'flinch_cant',
  });
  S.push({
    id: 'kingsrock_own_secondary_order',
    // Muddy Water (listed, own 30% acc-drop): TWO trailing rolls — [own][KR] in list order.
    p1: [mon('Zangoose', ['muddywater', 'muddywater'], { ability: 'Immunity', item: "King's Rock", nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'bodyslam'], { ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'flinch_cant',
  });
  S.push({
    id: 'kingsrock_seismictoss_procs',
    // A fixed-damage LISTED move still procs (the appended secondary rolls right after the
    // damage apply — no crit/randomizer precede it).
    p1: [mon('Zangoose', ['seismictoss', 'seismictoss'], { ability: 'Immunity', item: "King's Rock", nature: 'Adamant', evs: { spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'bodyslam'], { ability: 'Thick Fat', nature: 'Adamant', evs: { atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'flinch_cant',
  });
  S.push({
    id: 'kingsrock_control_no_item',
    // The SAME Slash board WITHOUT the item: no extra roll, no flinch (cover must stay 0).
    p1: [mon('Zangoose', ['slash', 'slash'], { ability: 'Immunity', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'bodyslam'], { ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'flinch_cant', // MUST stay 0 — asserted below
  });

  // ── FOCUS BAND ──
  S.push({
    id: 'focusband_survives_lethal',
    // A lv-5 FB Rattata under Cross Chop: the 1/10 roll draws on every hit; a pass on a
    // lethal MOVE hit survives at 1 HP (the -activate line).
    p1: [mon('Machamp', ['crosschop', 'crosschop'], { ability: 'Guts', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Rattata', ['scratch', 'scratch'], { ability: 'Guts', item: 'Focus Band', level: 5 })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'focusband',
  });
  S.push({
    id: 'focusband_burn_chip_draws',
    // A BURNED FB Snorlax: the burn chip draws the 1/10 roll every residual (no survive —
    // not a Move); the seed parity is the draw proof. Dusclops burns then Surfs; Snorlax
    // SPLASHES (1 PP/use, self-targeting so Pressure never double-deducts — the first cut
    // of this scenario exhausted Body Slam's 24 PP under Pressure's x2 and hit the sim's
    // accept-then-`|cant|nopp` path, which the port's strict choice gate deliberately
    // rejects; scripted goldens must stay within request-legal choices).
    p1: [mon('Dusclops', ['willowisp', 'surf'], { ability: 'Pressure', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Snorlax', ['splash', 'splash'], { ability: 'Thick Fat', item: 'Focus Band', nature: 'Adamant', evs: { atk: 252 } })],
    plan1: ['move 1', 'move 2'], plan2: ['move 1'],
    cover: 'burn_chip',
  });
  S.push({
    id: 'focusband_spikes_draws',
    // The Spikes switch-in chip into an FB holder draws the 1/10 roll (no survive).
    p1: [mon('Skarmory', ['spikes', 'drillpeck'], { ability: 'Keen Eye', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Zangoose', ['scratch', 'scratch'], { ability: 'Immunity', nature: 'Adamant', evs: { atk: 252 } }),
         mon('Snorlax', ['bodyslam', 'bodyslam'], { ability: 'Thick Fat', item: 'Focus Band', nature: 'Adamant', evs: { atk: 252 } })],
    plan1: ['move 1', 'move 2'], plan2: ['move 1', 'switch 2', 'move 1', 'switch 2'],
    cover: 'spikes_chip',
  });

  return S;
}

// ── Driver ────────────────────────────────────────────────────────────────────
(async () => {
  const S = scenarios();
  const lines = [];
  lines.push('# ability_batch4_golden (gen3_ability_batch4_v1) — TRUANT / INNER FOCUS / SHADOW TAG / CUTE CHARM+ATTRACT / COLOR CHANGE / KING\'S ROCK / FOCUS BAND');
  lines.push(`# format=${FORMAT} seeds=${seeds.length}`);
  lines.push('# SCEN <id> <cover>');
  lines.push('# TEAM p1|p2 <packed>');
  lines.push('# INIT <seed4>');
  lines.push('# DEC <request> <fP1> <fP2> <cP1> <cP2> <seed4> ' +
    '<p1 species,hp,maxhp,fainted,status,left,trapped> <p2 ...> <firstMover> <covered>');
  lines.push('# END <winner|none> <ended>');

  let totalDecisions = 0;
  let totalCover = 0;
  const coverById = {};

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}\t${sc.cover}`);
    lines.push(`TEAM\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\tp2\t${Teams.pack(sc.p2)}`);
    let scCover = 0;
    for (const seed of seeds) {
      const rec = await runBattle(sc, seed);
      if (!rec.initSeed || rec.decisions.length === 0) {
        throw new Error(`scenario ${sc.id} seed ${JSON.stringify(seed)} never reached a decision — fix the plan/teams`);
      }
      lines.push(`INIT\t${rec.initSeed}`);
      for (const d of rec.decisions) {
        const enc = (p) => `${p.species},${p.hp},${p.maxhp},${p.fainted ? 1 : 0},${p.status},${p.left},${p.trapped ? 1 : 0}`;
        lines.push(`DEC\t${d.request}\t${d.force[0] ? 1 : 0}\t${d.force[1] ? 1 : 0}\t${d.choiceP1}\t${d.choiceP2}\t` +
          `${d.seedAfter}\t${enc(d.p1)}\t${enc(d.p2)}\t${d.firstMover}\t${d.covered ? 1 : 0}`);
        totalDecisions++;
        if (d.covered) { totalCover++; scCover++; }
      }
      lines.push(`END\t${rec.winner || 'none'}\t${rec.ended ? 1 : 0}`);
    }
    coverById[sc.id] = scCover;
  }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  const crypto = require('crypto');
  const md5 = crypto.createHash('md5').update(lines.join('\n') + '\n').digest('hex');
  console.error(`wrote ${OUT}`);
  console.error(`scenarios=${S.length} decisions=${totalDecisions} coverRows=${totalCover} md5=${md5}`);
  console.error('per-scenario cover:', JSON.stringify(coverById));

  // FAIL LOUD if a class did not realize its effect (the golden would be vacuous) — and if a
  // CONTROL realized one it must not (Inner Focus never flinch-cants; F-into-F never attracts;
  // no-item never flinches).
  const positive = ['truant_alternates', 'truant_mirror_tie', 'truant_switch_rearm', 'truant_paralyzed_loaf',
    'flinch_control_thickfat', 'cutecharm_attracts_the_attacker', 'cutecharm_source_leaves_clears',
    'colorchange_chart_reads', 'colorchange_status_immunity', 'kingsrock_flinches',
    'kingsrock_serene_grace_doubles', 'kingsrock_own_secondary_order', 'kingsrock_seismictoss_procs',
    'focusband_survives_lethal', 'focusband_burn_chip_draws', 'focusband_spikes_draws'];
  const zero = ['innerfocus_blocks_flinch', 'cutecharm_gender_control', 'kingsrock_control_no_item'];
  const failures = [];
  for (const id of positive) if (!(coverById[id] > 0)) failures.push(`${id}: 0 cover rows (effect never realized)`);
  for (const id of zero) if (coverById[id] !== 0) failures.push(`${id}: expected 0 cover rows, got ${coverById[id]}`);
  if (failures.length) {
    console.error('\nFAIL — coverage:\n  ' + failures.join('\n  '));
    process.exit(1);
  }
  console.error('OK — every batch-4 class realized its effect + the controls stayed inert.');
})();
