// gen_movecoverage_batch2_golden.js — Gen-3 MOVE-COVERAGE BATCH 2 differential golden
// (`gen3_move_coverage_batch2_v1`): the DRAW-FRIENDLY status-move classes —
// STATUS-CURE / WEATHER-SET / STAT-DROP / SCREENS. All are category-Status moves.
//
// Extends the batch-1 TAB format with per-decision WEATHER (+ turns) and per-side SCREEN
// (light_screen / reflect) counters, on top of the per-decision STATE(+status+boosts+HP)+
// SEED+first-mover full-battle differential to GAME-END.
//
// THE FOUR CLASSES (probe-settled by probe_batch2_movecoverage.js):
//   STATUS-CURE — Refresh (self par/psn/brn), Heal Bell + Aromatherapy (whole-team major-
//                 status cure incl. bench). NEVER-MISS + DRAW-FREE onHit. Heal Bell skips a
//                 Soundproof ally; Refresh fails on none/slp/frz.
//   WEATHER-SET — Rain Dance (Rain) / Sunny Day (Sun): a 5-turn TIMED weather. NEVER-MISS,
//                 DRAW-FREE at the move (the eachEvent('WeatherChange') tie-shuffle draws
//                 only on a speed tie). setWeather FAILS (draw-free) if the SAME weather is
//                 already active; a DIFFERENT weather overwrites. 5-turn upkeep + expiry.
//   STAT-DROP   — Screech (-2 Def) / Charm (-2 Atk) / Metal Sound (-2 SpD) / Feather Dance
//                 (-2 Atk) / Tickle (-1/-1) / Fake Tears (-2 SpD). accuracy draw + boost()
//                 (draw-free, +-6 clamp, Clear Body/White Smoke/Hyper Cutter immunity;
//                 Screech/Metal Sound are SOUND → Soundproof-immune).
//   SCREENS     — Light Screen / Reflect (5 turns, halve special/physical). NEVER-MISS,
//                 DRAW-FREE. Re-use while up FAILS. 5-turn side-residual countdown + expiry.
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios that each ISOLATE a class, capturing initSeed + per-decision seedAfter, each
// active's species/hp/maxhp/fainted/status + boosts + confusion + pokemon_left + WEATHER
// (+turns) + per-side SCREEN counters + first mover + winner. The Rust test seeds a
// BattleState at initSeed and runs `run_full_battle` WITHOUT re-seeding — so the post-
// decision seed must match at EVERY boundary (a wrong draw model → SEED desync), AND the
// cured status, the weather/screen counters, and the stat-drop boosts must match.
//
// Output: tests/vectors/movecoverage_batch2_golden.txt
//
// Run:  node src/rust_sim/harness/gen_movecoverage_batch2_golden.js

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/movecoverage_batch2_golden.txt');
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

function screenOf(side, id) {
  const sc = side.sideConditions && side.sideConditions[id];
  return sc ? (sc.duration | 0) : 0;
}

function snap(side) {
  const a = side.active[0];
  const lightScreen = screenOf(side, 'lightscreen');
  const reflect = screenOf(side, 'reflect');
  if (!a) {
    return {
      species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', stage: 0, left: side.pokemonLeft,
      boosts: [0, 0, 0, 0, 0], confusion: 0, lightScreen, reflect,
    };
  }
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status, stage, left: side.pokemonLeft,
    boosts: boostsOf(a), confusion: confusionOf(a), lightScreen, reflect,
  };
}

// Field weather: the id ('' | 'raindance' | 'sunnyday' | 'sandstorm' | 'hail') + turns.
function weatherOf(battle) {
  const w = battle.field.weather || '';
  const dur = battle.field.weatherState ? (battle.field.weatherState.duration | 0) : 0;
  return { id: w || '-', turns: dur };
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

// Scan the protocol log between two decision points for the BATCH-2 branch flags.
function outcomesSince(log, fromIdx) {
  const out = {
    cure: false, healBell: false, aromatherapy: false, soundproofBlockCure: false,
    weatherSet: false, weatherFail: false, statDrop: false, statBlock: false,
    screenSet: false, screenFail: false, screenEnd: false, weatherEnd: false,
  };
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    if (tag === '-curestatus') out.cure = true;
    if (tag === '-activate' && (p[3] || '') === 'move: Heal Bell') out.healBell = true;
    if (tag === '-cureteam' && (p[3] || '').includes('Aromatherapy')) out.aromatherapy = true;
    if (tag === '-immune' && (p[3] || '').includes('Soundproof')) out.soundproofBlockCure = true;
    if (tag === '-weather' && p[2] === 'none') out.weatherEnd = true;
    else if (tag === '-weather' && !(/upkeep/.test(log[i]))) out.weatherSet = true;
    if (tag === '-unboost') out.statDrop = true;
    if (tag === '-fail' && (p[3] || '').includes('unboost')) out.statBlock = true;
    if (tag === '-sidestart' && /Light Screen|Reflect/.test(log[i])) out.screenSet = true;
    if (tag === '-sideend' && /Light Screen|Reflect/.test(log[i])) out.screenEnd = true;
    // A bare `-fail` on a caster after a weather/screen announce (overwrite / re-use fail).
    if (tag === '-fail' && p.length === 3) { /* ambiguous — tallied via require lists per scen */ }
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

  // Optional one-time post-start injection (weather / status / HP / bench status). STATE
  // only (no PRNG) so the seed parity is unaffected. Recorded as an INJECT line so the
  // Rust test reproduces the board.
  if (sc.inject) {
    const battle = stream.battle;
    for (const inj of sc.inject) {
      if (inj.weather) { battle.field.setWeather(inj.weather, battle.sides[0].active[0]); battle.field.weatherState.duration = 0; }
      if (inj.side !== undefined) {
        const s = battle.sides[inj.side];
        const m = s.active[0];
        if (inj.status) m.setStatus(inj.status, m, null, true);
        if (inj.benchStatus) { const b = s.pokemon[1]; if (b) b.setStatus(inj.benchStatus, b, null, true); }
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
    const weather = weatherOf(battle);

    rec.decisions.push({
      request: reqState, force,
      choiceP1: encodeChoice(choices.p1), choiceP2: encodeChoice(choices.p2),
      seedAfter, p1, p2, weather, firstMover: first, outcomes,
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

  // --- (1) STATUS-CURE: Refresh self-cures paralysis (injected). p1 Vaporeon is para'd,
  //   Refreshes it away, then grinds to a win. REQUIRES: cure. ---
  S.push({
    id: 'cure_refresh',
    p1: [mon('Vaporeon', ['refresh', 'surf'], { ability: 'Water Absorb', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    inject: [{ side: 0, status: 'par' }],
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    require: ['cure'],
  });

  // --- (2) STATUS-CURE: Heal Bell cures the WHOLE team (active tox + bench par). p1 Miltank
  //   Heal Bells; its own tox + the benched Blissey's par both clear. REQUIRES: healBell + cure. ---
  S.push({
    id: 'cure_healbell_team',
    p1: [mon('Miltank', ['healbell', 'bodyslam'], { ability: 'Thick Fat', nature: 'Impish', evs: { hp: 252 } }),
         mon('Blissey', ['softboiled'], { ability: 'Thick Fat', nature: 'Bold', evs: { hp: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    inject: [{ side: 0, status: 'tox', benchStatus: 'par' }],
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    require: ['healBell', 'cure'],
  });

  // --- (3) STATUS-CURE: Heal Bell SKIPS a Soundproof ally on the bench. p1 Miltank Heal Bells;
  //   the benched Soundproof Electrode's par is NOT cured (the ally-skip). REQUIRES: healBell. ---
  S.push({
    id: 'cure_healbell_soundproof_ally',
    p1: [mon('Miltank', ['healbell', 'bodyslam'], { ability: 'Thick Fat', nature: 'Impish', evs: { hp: 252 } }),
         mon('Electrode', ['thunderbolt'], { ability: 'Soundproof', nature: 'Timid', evs: { hp: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    inject: [{ side: 0, status: 'par', benchStatus: 'par' }],
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    require: ['healBell', 'cure'],
  });

  // --- (4) STATUS-CURE: Aromatherapy cures the whole team (active brn + bench slp). NO
  //   Soundproof gate. REQUIRES: aromatherapy. ---
  S.push({
    id: 'cure_aromatherapy',
    p1: [mon('Vileplume', ['aromatherapy', 'gigadrain'], { ability: 'Chlorophyll', nature: 'Modest', evs: { hp: 252, spa: 4 } }),
         mon('Snorlax', ['bodyslam'], { ability: 'Thick Fat', nature: 'Careful', evs: { hp: 252 } })],
    p2: [mon('Regice', ['pound'], { ability: 'Clear Body', nature: 'Calm', evs: { hp: 252 } })],
    inject: [{ side: 0, status: 'brn', benchStatus: 'slp' }],
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    // Aromatherapy uses clearStatus() (a `-cureteam` banner, NO per-mon `-curestatus`),
    // so we require the banner, not the `cure` flag.
    require: ['aromatherapy'],
  });

  // --- (5) WEATHER-SET: Rain Dance sets Rain for 5 turns; it upkeeps then EXPIRES. The
  //   weather (+turns) column tracks 5→…→0. REQUIRES: weatherSet + weatherEnd. ---
  S.push({
    id: 'weather_raindance',
    p1: [mon('Electrode', ['raindance', 'thunderbolt'], { ability: 'Static', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    require: ['weatherSet', 'weatherEnd'],
  });

  // --- (6) WEATHER-SET: Sunny Day sets Sun for 5 turns; upkeep + expiry. REQUIRES:
  //   weatherSet + weatherEnd. ---
  S.push({
    id: 'weather_sunnyday',
    p1: [mon('Ninetales', ['sunnyday', 'splash'], { ability: 'Flash Fire', nature: 'Timid', evs: { hp: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['splash'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    // Both Splash after the set → the battle never ends, so the 5-turn sun upkeeps + EXPIRES.
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    require: ['weatherSet', 'weatherEnd'],
  });

  // --- (7) WEATHER-SET: Rain Dance while permanent Rain (injected Drizzle-like) is ALREADY
  //   active → FAILS (draw-free), the weather stays permanent (turns 0). REQUIRES: weatherSet
  //   (the announce). ---
  S.push({
    id: 'weather_raindance_already_rain',
    p1: [mon('Electrode', ['raindance', 'thunderbolt'], { ability: 'Static', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    inject: [{ weather: 'raindance' }],
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    require: ['weatherSet'],
  });

  // --- (8) WEATHER-SET: Rain Dance OVERWRITES a permanent Sun (injected). The weather flips
  //   sun→rain with a 5-turn timer. REQUIRES: weatherSet. ---
  S.push({
    id: 'weather_raindance_overwrites_sun',
    p1: [mon('Electrode', ['raindance', 'thunderbolt'], { ability: 'Static', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    inject: [{ weather: 'sunnyday' }],
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    require: ['weatherSet'],
  });

  // --- (9) WEATHER-SET SPEED-TIE: Rain Dance in a Snorlax mirror — the eachEvent
  //   ('WeatherChange') tie-shuffle DRAWS on the set turn (the SEED parity proves it), and the
  //   per-turn eachEvent('Weather') shuffle draws while up. REQUIRES: weatherSet + weatherEnd. ---
  S.push({
    id: 'weather_raindance_speed_tie',
    p1: [mon('Snorlax', ['raindance', 'pound'], { ability: 'Immunity', nature: 'Serious', evs: { hp: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Serious', evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: n === 0 ? 1 : 2, p2Want: 1 }),
    require: ['weatherSet', 'weatherEnd'],
  });

  // --- (10) STAT-DROP: Screech (-2 Def, acc 85) — climbs the foe to the -6 floor across the
  //   seed sweep (misses on some seeds, lands on others). REQUIRES: statDrop. ---
  S.push({
    id: 'statdrop_screech',
    p1: [mon('Persian', ['screech', 'slash'], { ability: 'Limber', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['statDrop'],
  });

  // --- (11) STAT-DROP: Charm (-2 Atk, acc 100). REQUIRES: statDrop. ---
  S.push({
    id: 'statdrop_charm',
    p1: [mon('Clefable', ['charm', 'bodyslam'], { ability: 'Cute Charm', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 4 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['statDrop'],
  });

  // --- (12) STAT-DROP: Screech BLOCKED by Clear Body (accuracy still drawn, no drop). The
  //   foe's Def column stays 0. REQUIRES: statBlock + NO statDrop (forbid). ---
  S.push({
    id: 'statdrop_clearbody_block',
    p1: [mon('Persian', ['screech', 'slash'], { ability: 'Limber', nature: 'Jolly', evs: { atk: 252, spe: 252 } })],
    p2: [mon('Metagross', ['meteormash'], { ability: 'Clear Body', nature: 'Adamant', evs: { hp: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['statBlock'],
    forbid: ['statDrop'],
  });

  // --- (13) STAT-DROP: Metal Sound (-2 SpD, acc 85) IMMUNE vs Soundproof (a sound move). No
  //   drop. REQUIRES: NO statDrop (forbid). ---
  S.push({
    id: 'statdrop_metalsound_soundproof',
    p1: [mon('Registeel', ['metalsound', 'irontail'], { ability: 'Clear Body', nature: 'Sassy', evs: { hp: 252 } })],
    p2: [mon('Electrode', ['thunderbolt'], { ability: 'Soundproof', nature: 'Timid', evs: { hp: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    forbid: ['statDrop'],
  });

  // --- (14) SCREENS: Light Screen + Reflect (5 turns each, side-residual countdown + expiry).
  //   p1 Blissey sets Light Screen, then Reflect, then Splash-grinds; both count 5→0 + end.
  //   REQUIRES: screenSet + screenEnd. ---
  S.push({
    id: 'screens_lightscreen_reflect',
    p1: [mon('Blissey', ['lightscreen', 'reflect', 'softboiled'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    intent: (n) => ({ p1Want: n === 0 ? 1 : (n === 1 ? 2 : 3), p2Want: 1 }),
    require: ['screenSet', 'screenEnd'],
  });

  // --- (15) SCREENS: Light Screen re-use while UP FAILS (draw-free), the timer unchanged.
  //   REQUIRES: screenSet. ---
  S.push({
    id: 'screens_reuse_fail',
    p1: [mon('Blissey', ['lightscreen', 'softboiled'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    p2: [mon('Snorlax', ['pound'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['screenSet'],
  });

  // --- (16) SCREENS HALVE damage: Reflect halves a physical hit to the p1 side. p2 Snorlax
  //   Body Slams into a Reflect; the reduced damage shows in p1's HP. Also proves the screen
  //   is consumed by the damage calc. REQUIRES: screenSet. ---
  S.push({
    id: 'screens_reflect_halves_physical',
    p1: [mon('Blissey', ['reflect', 'softboiled'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252, def: 4 } })],
    p2: [mon('Snorlax', ['bodyslam'], { ability: 'Thick Fat', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    intent: () => ({ p1Want: 1, p2Want: 1 }),
    require: ['screenSet'],
  });

  // --- (17) BATCH-2 INTO A REAL BATTLE to game-end (the union: weather-set + screens +
  //   stat-drop + a cure + switching + faints all the way to a win). REQUIRES: win. ---
  S.push({
    id: 'batch2_into_a_real_battle',
    p1: [mon('Blissey', ['lightscreen', 'seismictoss', 'softboiled'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252, def: 252 } }),
         mon('Tyranitar', ['crunch', 'rockslide'], { ability: 'Sand Stream', nature: 'Adamant', evs: { atk: 252, hp: 252 } })],
    p2: [mon('Diglett', ['mudslap'], { level: 1, ability: 'Sand Veil', nature: 'Bold' }),
         mon('Sandshrew', ['scratch'], { level: 1, ability: 'Sand Veil', nature: 'Bold' }),
         mon('Cubone', ['pound'], { level: 1, ability: 'Rock Head', nature: 'Bold' })],
    intent: (n, battle) => {
      const p1Active = battle.sides[0].active[0];
      const isBliss = p1Active && p1Active.species.name === 'Blissey';
      // Blissey sets a screen, chips with Seismic Toss, then pivots to Tyranitar (sand) mid-battle.
      if (isBliss && n >= 2) return { p1Switch: 2, p2Want: 1 };
      return { p1Want: n === 0 ? 1 : 2, p2Want: 1 };
    },
    require: ['screenSet'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# movecoverage_batch2_golden.txt — Gen-3 MOVE-COVERAGE BATCH 2 full-battle golden.');
  lines.push('# Per-decision STATE(+status+boosts)+HP+WEATHER(+turns)+per-side SCREENS+SEED+first-mover differential to GAME-END.');
  lines.push('# Classes: STATUS-CURE / WEATHER-SET / STAT-DROP / SCREENS.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INJECT <id>  <json array of {weather?,side?,status?,benchStatus?,hp?}>  ([] if none)');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left atk def spa spd spe confusion) p2(...) first \\');
  lines.push('#        weatherId weatherTurns  p1LightScreen p1Reflect  p2LightScreen p2Reflect');
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
          d.weather.id, d.weather.turns,
          d.p1.lightScreen, d.p1.reflect,
          d.p2.lightScreen, d.p2.reflect,
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
    console.error('MOVECOVERAGE BATCH2 GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`BATCH2 GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('cure decisions', 'cure', 40);
  need('heal-bell decisions', 'healBell', 40);
  need('aromatherapy decisions', 'aromatherapy', 40);
  need('weather-set decisions', 'weatherSet', 40);
  need('weather-end decisions', 'weatherEnd', 40);
  need('stat-drop decisions', 'statDrop', 40);
  need('stat-block decisions', 'statBlock', 40);
  need('screen-set decisions', 'screenSet', 40);
  need('screen-end decisions', 'screenEnd', 40);
  if (winRows < 40) { console.error(`BATCH2 GOLDEN: too few WIN rows (${winRows} < 40)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `movecoverage batch2 golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: cure=${corpus.cure || 0} healBell=${corpus.healBell || 0} aromatherapy=${corpus.aromatherapy || 0} ` +
    `weatherSet=${corpus.weatherSet || 0} weatherEnd=${corpus.weatherEnd || 0} statDrop=${corpus.statDrop || 0} ` +
    `statBlock=${corpus.statBlock || 0} screenSet=${corpus.screenSet || 0} screenEnd=${corpus.screenEnd || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
