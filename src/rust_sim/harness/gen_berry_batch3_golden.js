// gen_berry_batch3_golden.js — the BATCH-3 CLASS-SWEEP golden
// (`gen3_berry_trace_shedskin_v1`): the BERRY item classes (ONE eatItem consumption
// mechanism + parameter rows) + TRACE + SHED SKIN, each proven bit-for-bit over full
// battles to game-end.
//
//   CURE_BERRY (cheri par / chesto slp / pecha psn+tox / rawst brn / aspear frz /
//     persim confusion / lum all+confusion) — eats at the FIRST eachEvent('Update')
//     after the condition (BEFORE the holder's own move — it never rolls full-para that
//     turn); LUM additionally eats IMMEDIATELY inside setStatus (onAfterSetStatus -1,
//     AFTER a Synchronize reflect) — incl. on Rest's self-sleep (LumRest). DRAW-FREE.
//   HEAL_BERRY (oran +10 / sitrus +30 / figy,wiki,mago,aguav,iapapa floor(maxhp/8) +
//     nature-gated confusion) — the RESIDUAL order 10 subOrder 4 (the Leftovers slot),
//     at 2*hp <= maxhp exactly. The Figy-family confusion draws random(2,6).
//   PINCH_BERRY (liechi atk / ganlon def / salac spe / petaya spa / apicot spd +1;
//     starf sample→+2; lansat focusenergy crit+2) — the residual slot at 4*hp <= maxhp.
//     Starf's sample is the ONLY new berry draw besides the figy confusion.
//   PP_BERRY (leppa) — eats at the Update when a slot hits 0 PP; +10 capped at maxpp.
//   TRACE — a MID-BATTLE switch-in draws ONE `sample` (random(1) even for a single foe)
//     and copies the foe's CURRENT ability (no copied-onStart fire; passive effects
//     LIVE; switch-out reverts). A LEAD trace's draw is a >start-window draw (before
//     the seeded start).
//   SHED SKIN — the residual order 10 subOrder 3 (the ability slot): ONE
//     randomChance(33,100) per STATUSED residual; a pass cures BEFORE the DoT chips.
//
// THE PROOF: the established per-decision STATE+HP+STATUS+**ITEM**+SEED differential —
// the omniscient BattleStream drives constructed battles to game-end; the Rust test
// replays from the init seed WITHOUT re-seeding. The eat timeline (item → NONE at the
// right decision), the cure/heal/boost effects, Starf's sample, the figy confusion
// random(2,6), the Shed Skin randomChance and the Trace sample must all land in the
// exact draw positions or the LCG desyncs.
//
// Output: tests/vectors/berry_batch3_golden.txt (the batch2 TAB format + an item token
// and a boosts CSV in each side snapshot).
//
// Run:  node src/rust_sim/harness/gen_berry_batch3_golden.js

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/berry_batch3_golden.txt');
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

// 40 fixed seeds per scenario — enough for the freeze-secondary (aspear) and the
// Shed Skin 33% to realize, and for Starf's sample to spread over the stat pool.
const seeds = [];
{
  let s = 0x3d4e5f6a >>> 0;
  const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s; };
  for (let i = 0; i < 40; i++) seeds.push([rng() % 65536, rng() % 65536, rng() % 65536, rng() % 65536]);
}

function snap(side) {
  const a = side.active[0];
  const b = a.boosts;
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status: a.status || '-', left: side.pokemonLeft,
    item: a.item || '-',
    boosts: `${b.atk}:${b.def}:${b.spa}:${b.spd}:${b.spe}`,
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

function firstMoverSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    if ((p[1] === 'move' || p[1] === 'cant') && p.length >= 3) {
      const a = (p[2] || '').trim();
      if (a.startsWith('p1a:')) return 'p1';
      if (a.startsWith('p2a:')) return 'p2';
    }
  }
  return 'none';
}

// Coverage: did THIS scenario's class effect fire this decision?
//   eat       — a `|-enditem|…|[eat]` line (any berry consumption).
//   leppa     — the `|-activate|…|item: Leppa Berry|…|[consumed]` line.
//   shedskin  — a `|-activate|…|ability: Shed Skin` line (the cure landed).
//   trace     — a `|-ability|…|…|[from] ability: Trace` line (the copy fired).
function coverageMarker(log, fromIdx, sc) {
  const has = (re) => {
    for (let i = fromIdx; i < log.length; i++) if (re.test(log[i])) return true;
    return false;
  };
  switch (sc.cover) {
    case 'eat': return has(/\|-enditem\|.*\|\[eat\]/);
    case 'leppa': return has(/\|-activate\|.*\|item: Leppa Berry\|/);
    case 'shedskin': return has(/\|-activate\|.*\|ability: Shed Skin/);
    case 'trace': return has(/\|-ability\|.*\[from\] ability: Trace/);
    default: return false;
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
  const maxDecisions = sc.maxDecisions || 400;
  while (!stream.battle.ended && safety < maxDecisions) {
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
      const p1c = sc.plan1[decisionNo % sc.plan1.length];
      const p2c = sc.plan2[decisionNo % sc.plan2.length];
      choices = { p1: p1c, p2: p2c };
    }

    const logLenBefore = log.length;
    if (choices.p1) { try { streams.omniscient.write(`>p1 ${choices.p1}`); } catch (e) {} }
    if (choices.p2) { try { streams.omniscient.write(`>p2 ${choices.p2}`); } catch (e) {} }
    for (let i = 0; i < 16; i++) await tick();

    const seedAfter = battle.prng.getSeed();
    if (seedAfter === seedBefore && log.length === logLenBefore && battle.requestState === reqState) {
      throw new Error(`STALL: choice ${JSON.stringify(choices)} did not advance the battle ` +
        `(scenario ${sc.id}, decision ${decisionNo}) — the sim rejected it. Fix the plan.`);
    }
    const covered = reqState === 'move' && coverageMarker(log, logLenBefore, sc);
    if (covered) rec.coverRows++;
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
      // A decision where either side CHOSE a voluntary switch reports firstMover
      // 'none': the golden derives the mover from the first |move| line, while the
      // port's first_mover counts the switch ACTION — apples-to-oranges (the batch2
      // format never mixed them; batch3's trace scenarios do). The SEED + state still
      // pin the full ordering.
      firstMover: (reqState === 'move' && !/^s/.test(encodeChoice(choices.p1)) && !/^s/.test(encodeChoice(choices.p2)))
        ? firstMoverSince(log, logLenBefore) : 'none',
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
// Chip pattern: a Blissey Seismic Toss (exact 100/turn, draw-light) grinds the holder
// down so the heal/pinch threshold crossings + the eventual KO (game end) are
// deterministic; the holder attacks back so most seeds end in a win either way.
function scenarios() {
  const S = [];
  const chipFoe = () => mon('Blissey', ['seismictoss', 'softboiled'], { evs: { hp: 252, spe: 252 } });

  // ── CURE berries ──
  S.push({
    id: 'cheri_cures_par_at_update',
    // Jolteon TWaves turn 1 (then attacks); the cheri eats at the Update BEFORE
    // Snorlax's own move (no full-para roll that turn). The SECOND TWave (item gone)
    // sticks — full-para rolls resume.
    p1: [mon('Jolteon', ['thunderwave', 'thunderbolt'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'cheriberry', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1', 'move 2'], plan2: ['move 1'],
    cover: 'eat',
  });
  S.push({
    id: 'chesto_rest_full_heal_awake',
    // ChestoRest: Rest → slp → the post-action Update cures it → full HP, awake. The
    // holder splashes between rests (a rest-sustain loop never ends — capped at 16
    // decisions; the Rust replay matches the un-ended tail exactly).
    p1: [chipFoe()],
    p2: [mon('Snorlax', ['rest', 'splash'], { item: 'chestoberry', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 2', 'move 2', 'move 1'],
    cover: 'eat', maxDecisions: 16,
  });
  S.push({
    id: 'pecha_cures_tox',
    p1: [mon('Starmie', ['toxic', 'surf'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'pechaberry', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1', 'move 2'], plan2: ['move 1'],
    cover: 'eat',
  });
  S.push({
    id: 'rawst_cures_brn',
    p1: [mon('Moltres', ['willowisp', 'flamethrower'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Golem', ['rockslide', 'earthquake'], { item: 'rawstberry', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1', 'move 2'], plan2: ['move 1'],
    cover: 'eat',
  });
  S.push({
    id: 'aspear_cures_frz',
    // Freeze is Ice Beam's 10% secondary — over 40 seeds a few battles freeze and the
    // aspear cures at the next Update (the eat rows are the cover; a no-freeze battle
    // is still a valid no-eat control row).
    p1: [mon('Lapras', ['icebeam', 'icebeam'], { nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'aspearberry', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'eat',
  });
  S.push({
    id: 'persim_cures_confusion',
    // Water Pulse's 20% confusion secondary → the persim eats at the next Update.
    p1: [mon('Starmie', ['waterpulse', 'waterpulse'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'persimberry', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'eat',
  });
  S.push({
    id: 'lum_immediate_on_twave',
    // Lum eats INSIDE setStatus (before the Update) — the para never sticks at any
    // boundary; the second TWave (item gone) sticks.
    p1: [mon('Jolteon', ['thunderwave', 'thunderbolt'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'lumberry', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1', 'move 2'], plan2: ['move 1'],
    cover: 'eat',
  });
  S.push({
    id: 'lum_rest_instant_wake',
    // LumRest: Rest sets slp → lum eats immediately (before the heal) → full HP, awake,
    // no sleep counter — the classic CM-LumRest line. Capped like the chesto loop.
    p1: [chipFoe()],
    p2: [mon('Snorlax', ['rest', 'splash'], { item: 'lumberry', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 2', 'move 2', 'move 1'],
    cover: 'eat', maxDecisions: 16,
  });
  // CONTROL: the WRONG cure berry for the status — a cheri holder that gets BURNED
  // keeps its berry (no eat, no cure).
  S.push({
    id: 'cure_control_wrong_status_no_eat',
    p1: [mon('Moltres', ['willowisp', 'flamethrower'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Golem', ['rockslide', 'earthquake'], { item: 'cheriberry', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1', 'move 2'], plan2: ['move 1'],
    cover: 'eat', expectZeroCover: true,
  });

  // ── HEAL berries (threshold 1/2 at the residual) ──
  for (const [id, item, nature] of [
    ['oran_heals_10', 'oranberry', 'Adamant'],
    ['sitrus_heals_30', 'sitrusberry', 'Adamant'],
    // The figy family: the nature MINUSES the flavor stat → the heal ALSO confuses
    // (the random(2,6) draw). figy=atk (Modest), wiki=spa (Adamant), mago=spe (Brave),
    // aguav=spd (Naughty), iapapa=def (Lonely).
    ['figy_heals_and_confuses_minus_atk', 'figyberry', 'Modest'],
    ['wiki_heals_and_confuses_minus_spa', 'wikiberry', 'Adamant'],
    ['mago_heals_and_confuses_minus_spe', 'magoberry', 'Brave'],
    ['aguav_heals_and_confuses_minus_spd', 'aguavberry', 'Naughty'],
    ['iapapa_heals_and_confuses_minus_def', 'iapapaberry', 'Lonely'],
    // CONTROL: figy with a NON-minus-atk nature — heals, NO confusion (no draw).
    ['figy_control_no_minus_no_confusion', 'figyberry', 'Careful'],
  ]) {
    S.push({
      id,
      p1: [chipFoe()],
      p2: [mon('Snorlax', ['splash', 'bodyslam'], { item, nature, evs: { hp: 252, atk: 252 } })],
      plan1: ['move 1'], plan2: ['move 1'],
      cover: 'eat',
    });
  }

  // ── PINCH berries (threshold 1/4 at the residual) ──
  for (const [id, item] of [
    ['liechi_pinch_atk', 'liechiberry'],
    ['ganlon_pinch_def', 'ganlonberry'],
    ['salac_pinch_spe', 'salacberry'],
    ['petaya_pinch_spa', 'petayaberry'],
    ['apicot_pinch_spd', 'apicotberry'],
    ['starf_pinch_sample_plus2', 'starfberry'],
    ['lansat_pinch_focus_energy', 'lansatberry'],
  ]) {
    S.push({
      id,
      p1: [chipFoe()],
      p2: [mon('Snorlax', ['splash', 'bodyslam'], { item, nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
      plan1: ['move 1'], plan2: ['move 1'],
      cover: 'eat',
    });
  }
  // SUBSTITUTE interaction: sub-absorbed hits leave hp untouched → no pinch trigger
  // until the REAL hp crosses (after the sub cost / breaks).
  S.push({
    id: 'salac_behind_substitute',
    p1: [chipFoe()],
    p2: [mon('Snorlax', ['substitute', 'splash'], { item: 'salacberry', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'eat',
  });

  // ── PP berry (Leppa) ──
  S.push({
    id: 'leppa_restores_the_zeroed_slot',
    // Blizzard (8 maxpp) is spammed to 0 → the leppa eats at that Update and restores
    // min(0+10, 8) = 8. The foe alternates toss/softboiled so the battle still ends
    // (Snorlax dies to chip) shortly after the leppa fires.
    p1: [mon('Blissey', ['seismictoss', 'softboiled'], { evs: { hp: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['blizzard', 'bodyslam'], { item: 'leppaberry', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    plan1: ['move 1', 'move 1', 'move 2'], plan2: ['move 1'],
    cover: 'leppa',
  });

  // ── SHED SKIN ──
  S.push({
    id: 'shedskin_cures_burn',
    // Will-O-Wisp burns the Shed Skin holder → one randomChance(33,100) per residual
    // until the cure (a cure turn takes NO burn chip — subOrder 3 < 6).
    p1: [mon('Moltres', ['willowisp', 'flamethrower'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Arbok', ['sludgebomb', 'earthquake'], { ability: 'Shed Skin', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1', 'move 2'], plan2: ['move 1'],
    cover: 'shedskin',
  });
  S.push({
    id: 'shedskin_cures_toxic',
    // A NON-Poison Shed Skin holder (Dragonair) — an Arbok would be tox-immune.
    p1: [mon('Starmie', ['toxic', 'surf'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Dragonair', ['thunderbolt', 'surf'], { ability: 'Shed Skin', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    plan1: ['move 1', 'move 2'], plan2: ['move 1'],
    cover: 'shedskin',
  });
  // CONTROL: an UNSTATUSED Shed Skin holder draws NOTHING at the residual (but its
  // handler still ties the residual shuffle — the seed stream proves both).
  S.push({
    id: 'shedskin_unstatused_no_draw',
    p1: [chipFoe()],
    p2: [mon('Arbok', ['sludgebomb', 'earthquake'], { ability: 'Shed Skin', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'shedskin', expectZeroCover: true,
  });

  // ── TRACE ──
  S.push({
    id: 'trace_midbattle_copies_immunity',
    // p1 leads Machamp, switches Gardevoir (Trace) in mid-battle → ONE sample draw +
    // the copy of Snorlax's Immunity; the battle then plays out (Psychic vs Body Slam).
    p1: [mon('Machamp', ['crosschop', 'crosschop'], { ability: 'Guts', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Gardevoir', ['psychic', 'thunderbolt'], { ability: 'Trace', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'earthquake'], { ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['switch 2', ...Array(60).fill('move 1')], plan2: ['move 1'],
    cover: 'trace',
  });
  S.push({
    id: 'trace_copies_flashfire_live',
    // Trace copies Houndoom's Flash Fire mid-battle → the traced FF is LIVE (the foe's
    // Flamethrower is absorbed: no damage + the flashfire volatile arms).
    p1: [mon('Machamp', ['crosschop', 'crosschop'], { ability: 'Guts', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
         mon('Gardevoir', ['psychic', 'thunderbolt'], { ability: 'Trace', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Houndoom', ['flamethrower', 'crunch'], { ability: 'Flash Fire', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    plan1: ['switch 2', ...Array(60).fill('move 2')], plan2: ['move 1', 'move 2'],
    cover: 'trace',
  });
  S.push({
    id: 'trace_lead_copies_at_start',
    // A LEAD Trace: the copy happens in the >start window (its sample draw pre-dates
    // the seeded start — the port applies the copy draw-free). What this pins is the
    // lead-trace >start SEED-NEUTRALITY: every post-start decision boundary (state +
    // seed) must match the sim with the copy already applied. (No mon here carries a
    // status move, so the traced Immunity's block is NOT exercised — the live-copy
    // behaviour is pinned by BR5 + the mid-battle trace scenarios.)
    p1: [mon('Gardevoir', ['psychic', 'thunderbolt'], { ability: 'Trace', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'earthquake'], { ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    cover: 'trace', expectZeroCover: true, // the -ability line lands BEFORE decision 0's window
  });

  return S;
}

// ── Driver ────────────────────────────────────────────────────────────────────
(async () => {
  const S = scenarios();
  const lines = [];
  lines.push('# berry_batch3_golden (gen3_berry_trace_shedskin_v1) — BERRY classes / TRACE / SHED SKIN');
  lines.push(`# format=${FORMAT} seeds=${seeds.length}`);
  lines.push('# SCEN <id> <cover> <expectZeroCover>');
  lines.push('# TEAM p1|p2 <packed>');
  lines.push('# INIT <seed4>');
  lines.push('# DEC <request> <fP1> <fP2> <cP1> <cP2> <seed4> ' +
    '<p1 species,hp,maxhp,fainted,status,left,item,boosts(a:d:sa:sd:sp)> <p2 ...> <firstMover> <covered>');
  lines.push('# END <winner|none> <ended>');

  let totalDecisions = 0;
  let totalCover = 0;
  const coverById = {};

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}\t${sc.cover}\t${sc.expectZeroCover ? 1 : 0}`);
    lines.push(`TEAM\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\tp2\t${Teams.pack(sc.p2)}`);
    let scCover = 0;
    for (const seed of seeds) {
      const rec = await runBattle(sc, seed);
      if (!rec.initSeed || rec.decisions.length === 0) {
        throw new Error(`scenario ${sc.id} seed ${JSON.stringify(seed)} never reached a decision`);
      }
      lines.push(`INIT\t${rec.initSeed}`);
      for (const d of rec.decisions) {
        const enc = (s) => `${s.species},${s.hp},${s.maxhp},${s.fainted ? 1 : 0},${s.status},${s.left},${s.item},${s.boosts}`;
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
  console.log(`wrote ${OUT}`);
  console.log(`scenarios=${S.length} runs=${S.length * seeds.length} decisions=${totalDecisions} coverRows=${totalCover}`);
  for (const [id, n] of Object.entries(coverById)) console.log(`  cover ${id}: ${n}`);
})().catch((e) => { console.error(e); process.exit(1); });
