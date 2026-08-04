// gen_forecast_golden.js — the FORECAST (Castform forme swap) class-sweep golden
// (`gen3_forecast_v1`, ROUND 35).
//
// Forecast swaps Castform's forme + TYPE to follow the EFFECTIVE weather, DRAW-FREE, at every
// `eachEvent('WeatherChange')` site + the entrant's own `onStart` + the start window, reverting
// silently at `clearVolatile`. Because it is draw-free, the per-decision SEED must match the sim
// bit-for-bit; the **SPECIES column IS the forme** (the port's `DecisionRecord.active_species`
// reports the LIVE `species_id`, so `castformrainy`/`castformsunny`/`castformsnowy`/`castform`
// come through for free) and the WEATHER columns pin the timed-weather countdown the formes key
// off. This sweep is the breadth complement to the deterministic FC1-FC8 pins in
// `regression_test.rs` — many seeds × the wiring sites, rather than one board per site.
//
//   COVERS (each a DECISIVE full battle in gen3customgame; the foe NEVER attacks — Splash
//   forever — so P1 always wins and the plan can never misalign on an unexpected faint):
//     fc_rain_cycle   — Castform casts RAIN DANCE (the weather-set-MOVE site): formes Rainy, holds
//                       through the upkeep turns, REVERTS at the 5-turn expiry (the UNCONDITIONAL
//                       clearWeather WeatherChange), then Ice Beams to the win.
//     fc_hail_snowy   — the same cycle on HAIL (`gen3_forecast_v1` also modeled hail-the-MOVE):
//                       formes SNOWY (ICE → hail-chip IMMUNE while the foe chips every turn — the
//                       forme's TYPE is load-bearing for the HP columns, not just cosmetic).
//     fc_sand_base    — SANDSTORM maps to the DEFAULT arm: the Castform stays BASE `castform`
//                       (Normal → it DOES take the sand chip). The counter-intuitive case, and the
//                       one a "any weather → a forme" model would get wrong. NO formechange fires.
//     fc_ability_in   — a DRIZZLE Politoed switches in mid-battle (the `run_switch` WeatherChange
//                       site): the standing Castform formes Rainy off the ability-set weather.
//     fc_pivot        — a FORMED Castform pivots OUT (silent `clearVolatile` revert — it is base
//                       `castform` on the bench) and back IN (re-formes via its own `onStart`).
//     fc_suppressed   — a CLOUD NINE Psyduck suppresses the rain (Castform stays BASE), then
//                       FAINTS to Ice Beam → the negater's `onEnd` WeatherChange fires with the
//                       dying holder EXCLUDED (`effective_weather_excluding`) → the Castform formes
//                       Rainy on the faint boundary. The `process_faints` half of the exclusion
//                       (FC5 pins the voluntary-switch half).
//     fc_start_window — a Castform LEAD vs a DROUGHT Ninetales lead: formed `castformsunny` (Fire)
//                       by the START window, before decision 0.
//
// Output: tests/vectors/forecast_golden.txt (the whiteherb TAB format with the item columns
// swapped for the per-decision WEATHER id + turns, plus a formechange-fired marker).
//
// Run:  node src/rust_sim/harness/gen_forecast_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/forecast_golden.txt');
const FORMAT = 'gen3customgame';
const SEEDS = 40;
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
  let x = 0x35c0_f00d >>> 0;
  const step = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  for (let i = 0; i < n; i++) out.push([step() || 1, step() || 1, step() || 1, step() || 1]);
  return out;
}

function firstMoverSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    const parts = log[i].split('|');
    const tag = parts[1];
    const isAction =
      tag === 'move' || tag === 'switch' || tag === 'drag' || tag === 'cant' ||
      (tag === '-activate' && (parts[3] || '') === 'confusion');
    if (isAction && parts.length >= 3) {
      const actor = parts[2].trim();
      if (actor.startsWith('p1a:')) return 'p1';
      if (actor.startsWith('p2a:')) return 'p2';
    }
  }
  return 'none';
}

function boostStr(a) {
  const b = a ? a.boosts : {};
  return [b.atk || 0, b.def || 0, b.spa || 0, b.spd || 0, b.spe || 0, b.accuracy || 0, b.evasion || 0].join(',');
}

function snap(side) {
  const a = side.active[0];
  if (!a) return { species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', left: side.pokemonLeft, boosts: '0,0,0,0,0,0,0' };
  return {
    // `a.species.id` is the LIVE species — for a Forecast Castform this IS the forme
    // (`castformrainy` etc.); a non-permanent formeChange never touches `baseSpecies`.
    species: a.species.id, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status: a.status || '-', left: side.pokemonLeft, boosts: boostStr(a),
  };
}

// The FIELD weather id ('' when clear) + its remaining duration (0 = permanent/none).
function weatherSnap(battle) {
  const w = battle.field.weather || '';
  const d = (battle.field.weatherState && battle.field.weatherState.duration) || 0;
  return { id: w, turns: d };
}

function formeFiredSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    if (log[i].startsWith('|-formechange')) return true;
  }
  return false;
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

async function runBattle(sc, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();

  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(sc.p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(sc.p2) })}`);
  for (let i = 0; i < 10; i++) await tick();

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

    let choices;
    if (reqState === 'switch') {
      choices = { p1: force[0] ? 'switch 2' : null, p2: force[1] ? 'switch 2' : null };
    } else {
      choices = {
        p1: sc.plan1[Math.min(decisionNo, sc.plan1.length - 1)],
        p2: sc.plan2[Math.min(decisionNo, sc.plan2.length - 1)],
      };
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
    const w = weatherSnap(battle);
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
      weather: w.id,
      weatherTurns: w.turns,
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      forme: formeFiredSince(log, logLenBefore),
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

// ── Scenarios ────────────────────────────────────────────────────────────────
// A Modest 252-SpA Castform so Ice Beam kills on schedule (the plans are clamped, but a foe that
// outlives the plan would stretch the battle past the weather cycle we want to observe).
function castform(moves) {
  return mon('Castform', moves, { ability: 'Forecast', evs: { hp: 4, spa: 252, spe: 252 }, nature: 'Modest' });
}
function laxFoe(moves) {
  return mon('Snorlax', moves, { ability: 'Immunity', evs: { hp: 252, def: 252 }, nature: 'Careful' });
}
function frail(species, moves, ability) {
  return mon(species, moves, { ability, evs: { hp: 4 }, nature: 'Serious' });
}

function scenarios() {
  const S = [];
  // The weather-MOVE cycle: cast (t1), hold (t2-4), EXPIRE (t5), then Ice Beam to the win.
  const cyclePlan1 = ['move 1', 'move 2', 'move 2', 'move 2', 'move 2', 'move 3'];
  const splashForever = ['move 1'];

  S.push({
    id: 'fc_rain_cycle',
    p1: [castform(['raindance', 'splash', 'icebeam'])],
    p2: [laxFoe(['splash'])],
    plan1: cyclePlan1, plan2: splashForever,
  });
  S.push({
    id: 'fc_hail_snowy',
    p1: [castform(['hail', 'splash', 'icebeam'])],
    p2: [laxFoe(['splash'])],
    plan1: cyclePlan1, plan2: splashForever,
  });
  S.push({
    id: 'fc_sand_base',
    p1: [castform(['sandstorm', 'splash', 'icebeam'])],
    p2: [laxFoe(['splash'])],
    plan1: cyclePlan1, plan2: splashForever,
  });
  // A Drizzle Politoed switches in at decision 1 → the standing Castform formes Rainy.
  S.push({
    id: 'fc_ability_in',
    p1: [castform(['splash', 'splash', 'icebeam'])],
    p2: [frail('Abra', ['splash'], 'Inner Focus'), frail('Politoed', ['splash'], 'Drizzle')],
    plan1: ['move 1', 'move 3'], plan2: ['move 1', 'switch 2', 'move 1'],
  });
  // A FORMED Castform pivots out (silent revert on the bench) and back in (re-formes onStart).
  S.push({
    id: 'fc_pivot',
    p1: [castform(['raindance', 'splash', 'icebeam']), frail('Rattata', ['splash'], 'Guts')],
    p2: [laxFoe(['splash'])],
    plan1: ['move 1', 'switch 2', 'switch 2', 'move 3'], plan2: splashForever,
  });
  // Cloud Nine suppresses the rain; the holder then FAINTS to Ice Beam → the negater's onEnd
  // WeatherChange fires with the DYING holder excluded → the Castform formes on that boundary.
  S.push({
    id: 'fc_suppressed',
    p1: [castform(['raindance', 'splash', 'icebeam'])],
    p2: [frail('Psyduck', ['splash'], 'Cloud Nine'), frail('Abra', ['splash'], 'Inner Focus')],
    plan1: ['move 1', 'move 3'], plan2: splashForever,
  });
  // The START window: a Drought lead formes the Castform before decision 0.
  S.push({
    id: 'fc_start_window',
    p1: [castform(['splash', 'splash', 'icebeam'])],
    p2: [frail('Ninetales', ['splash'], 'Drought')],
    plan1: ['move 3'], plan2: splashForever,
  });
  return S;
}

async function main() {
  const S = scenarios();
  const seeds = buildSeeds(SEEDS);
  const lines = [];
  lines.push('# forecast_golden.txt — gen3_forecast_v1 (ROUND 35) class sweep.');
  lines.push('# DEC fields: scen seed request f1 f2 c1 c2 seedAfter | p1(species hp maxhp fnt status left) p1boosts |');
  lines.push('#             p2(...) p2boosts | weather weatherTurns firstMover formeFired');
  lines.push(`# format=${FORMAT} seeds=${SEEDS}`);

  const failures = [];
  let decRows = 0; let formeRows = 0; let winRows = 0; let tieRows = 0;

  // Non-vacuity: which scenarios MUST forme, which must NOT, and which must be formed at dec0.
  const mustForme = new Set(['fc_rain_cycle', 'fc_hail_snowy', 'fc_ability_in', 'fc_pivot', 'fc_suppressed']);
  const mustNotForme = new Set(['fc_sand_base']);
  const formedAtDec0 = new Map([['fc_start_window', 'castformsunny']]);
  // The forme each cycling scenario must REACH at some decision (the type-carrying proof).
  const mustReach = new Map([
    ['fc_rain_cycle', 'castformrainy'],
    ['fc_hail_snowy', 'castformsnowy'],
    ['fc_ability_in', 'castformrainy'],
    ['fc_pivot', 'castformrainy'],
    ['fc_suppressed', 'castformrainy'],
  ]);
  // …and which of those must also REVERT to base afterwards. NOT `fc_ability_in`: Drizzle sets
  // PERMANENT (duration-0) ability weather, so the rain never expires and that Castform never
  // pivots — it is Rainy for the rest of the battle, correctly. (Requiring a revert there is what
  // this floor caught on the first run — the guard working, not a bug.)
  const mustRevert = new Set(['fc_rain_cycle', 'fc_hail_snowy', 'fc_pivot', 'fc_suppressed']);

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);

    let scenDecs = 0; let scenForme = 0; let reached = 0; let dec0Formed = 0; let reverts = 0;
    for (const seed of seeds) {
      let rec;
      try { rec = await runBattle(sc, seed); } catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); continue; }
      if (rec.gen !== 3) { failures.push(`${sc.id}: expected gen 3, got ${rec.gen}`); break; }
      if (!rec.initSeed || rec.decisions.length === 0) { failures.push(`${sc.id} seed ${seed}: no decisions`); continue; }
      const seedStr = seed.join(',');
      lines.push(['INIT', sc.id, rec.initSeed, seedStr].join('\t'));

      const want = mustReach.get(sc.id);
      const p1Species = rec.decisions.map((d) => d.p1.species);
      if (want && p1Species.includes(want)) reached++;
      // A cycling scenario must also REVERT to base at some LATER decision (the expiry /
      // clearVolatile proof) — a model that formes but never reverts would else pass.
      if (want && mustRevert.has(sc.id)) {
        const first = p1Species.indexOf(want);
        if (first >= 0 && p1Species.slice(first + 1).includes('castform')) reverts++;
      }
      const d0 = formedAtDec0.get(sc.id);
      if (d0 && rec.decisions[0].p1.species === d0) dec0Formed++;

      rec.decisions.forEach((d) => {
        const sp = (s) => [s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status, s.left].join('\t');
        lines.push([
          'DEC', sc.id, seedStr, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
          sp(d.p1), d.p1.boosts, sp(d.p2), d.p2.boosts,
          d.weather || '-', d.weatherTurns, d.firstMover, d.forme ? 1 : 0,
        ].join('\t'));
        decRows++; scenDecs++;
        if (d.forme) { formeRows++; scenForme++; }
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
    if (mustForme.has(sc.id) && scenForme < 10) {
      failures.push(`${sc.id}: only ${scenForme} formechange rows (<10) — the forme barely fires`);
    }
    if (mustNotForme.has(sc.id) && scenForme > 0) {
      failures.push(`${sc.id}: expected 0 formechange rows (SAND is the DEFAULT arm), got ${scenForme}`);
    }
    if (mustReach.has(sc.id) && reached < 10) {
      failures.push(`${sc.id}: only ${reached} runs reached ${mustReach.get(sc.id)} (<10)`);
    }
    if (mustRevert.has(sc.id) && reverts < 10) {
      failures.push(`${sc.id}: only ${reverts} runs REVERTED to base after forming (<10) — a forme-but-never-revert model would pass without this`);
    }
    if (formedAtDec0.has(sc.id) && dec0Formed < 10) {
      failures.push(`${sc.id}: only ${dec0Formed} runs were START-window formed at dec0 (<10)`);
    }
  }

  if (failures.length) {
    console.error('FORECAST GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }
  if (winRows < 100) { console.error(`FORECAST GOLDEN: too few WIN rows (${winRows} < 100)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `forecast golden: ${S.length} scenarios, ${decRows} decision rows, ${formeRows} formechange rows, ` +
    `${winRows} wins + ${tieRows} ties -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
