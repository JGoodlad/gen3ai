// gen_yawn_golden.js — the YAWN delayed-sleep golden (`gen3_yawn_v1`).
//
// gen-3 Yawn (`volatileStatus: 'yawn'`, `accuracy: true`) is a category-Status foe-target move whose
// CRUX is that the sleep `random(2,6)` fires at RESOLVE (the residual `onEnd`), not at cast — the
// CAST is entirely DRAW-FREE. The `yawn` condition (`duration: 2`, `onResidualOrder: 10,
// onResidualSubOrder: 19`) decrements 2 → 1 (end of the cast turn) then 1 → 0 (end of the NEXT
// turn); on the 1 → 0 tick the `onEnd` emits `|-end|<t>|move: Yawn|[silent]` then
// `target.trySetStatus('slp', source)` — so the sleep lands at the END of the turn AFTER cast, and
// the `random(2,6)` duration is drawn THERE. This golden makes the delayed sleep OBSERVABLE on a
// per-decision STATUS + SLEEP-COUNTER column: the target is status-free while the yawn is pending,
// then becomes `slp` with the exact random(2,6) counter, which then decrements to the wake.
//
//   COVERS (each a DECISIVE full battle in gen3customgame — p1 = a mono Snorlax [yawn, earthquake,
//   thunderwave, splash] (Immunity, so the caster is never statused), p2 mostly a mono Blissey that
//   Splashes every turn and never attacks → p1 never takes damage → guaranteed decisive P1):
//     yawn_resolve_sleep_wake  — Yawn (dec0) → RESOLVES at end of dec1 → Blissey sleeps (the
//                                random(2,6) counter set), stays asleep while Snorlax Splashes, WAKES
//                                N turns later, then Earthquake to the win. The sleep-lands + the
//                                counter + the wake are all OBSERVED on the sleepTime column.
//     yawn_into_statused       — Thunder Wave (dec0 → par) THEN Yawn into the par'd foe (dec1 →
//                                `onTryHit` FAILS: `[still]` + `-fail`, NO volatile, DRAW-FREE); the
//                                foe stays par + NEVER sleeps, then Earthquake to the win.
//     yawn_into_vitalspirit    — Yawn (dec0) into a Vital Spirit foe → `-immune`, NO volatile,
//                                DRAW-FREE; the foe NEVER sleeps, then Earthquake to the win.
//     yawn_statused_between    — Yawn (dec0 cast) THEN Thunder Wave (dec1 → par); the yawn RESOLVES
//                                at end of dec1 but the foe is now par → `-end [silent]` but NO sleep
//                                (DRAW-FREE resolve); the foe stays par (never slp), then EQ to win.
//     yawn_real_battle         — a MULTI-mon game: Yawn Blissey-A, Earthquake it down (asleep),
//                                forced switch to Blissey-B, Earthquake it down → the yawn volatile
//                                CLEARS on the phazed/replaced mon + composes in a full game.
//
// Output: tests/vectors/yawn_golden.txt (a per-side sleepTime column + a `yawnResolved` marker).
//
// Run:  node src/rust_sim/harness/gen_yawn_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/yawn_golden.txt');
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
  let x = 0x7a1c_3d5f >>> 0;
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

// The sleep/Toxic inner counter (`statusState.time` for slp = remaining turns; else 0).
function sleepTime(a) {
  if (!a || a.status !== 'slp') return 0;
  return (a.statusState && a.statusState.time) || 0;
}

function snap(side) {
  const a = side.active[0];
  if (!a) return { species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', left: side.pokemonLeft, sleep: 0 };
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status: a.status || '-', left: side.pokemonLeft, sleep: sleepTime(a),
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

// Coverage marker: did a Yawn RESOLVE (`|-end|...|move: Yawn|[silent]`) fire this decision?
function yawnResolvedSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    const l = log[i];
    if (l.startsWith('|-end|') && l.includes('move: Yawn')) return true;
  }
  return false;
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
      const p1c = sc.plan1[Math.min(decisionNo, sc.plan1.length - 1)];
      const p2c = sc.plan2[Math.min(decisionNo, sc.plan2.length - 1)];
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
    const yawnResolved = yawnResolvedSince(log, logLenBefore);
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      yawnResolved,
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

// ── Scenarios ────────────────────────────────────────────────────────────────
// p1 = a mono Snorlax (Immunity, max Atk/Spe) with [yawn, earthquake, thunderwave, splash]. Earthquake
// (no secondary) is the clean KO'er; Thunder Wave sets up the statused scenarios; Splash is the
// non-lethal filler that lets the target sleep + wake. p2 mostly a mono Blissey (Natural Cure — a
// mono never switches so it never fires) that Splashes forever → p1 never takes damage → guaranteed
// decisive P1.
function snorlax() {
  return mon('Snorlax', ['yawn', 'earthquake', 'thunderwave', 'splash'], {
    ability: 'Immunity', nature: 'Adamant', evs: { atk: 252, spe: 252, hp: 4 },
  });
}
function blissey() {
  return mon('Blissey', ['splash'], { ability: 'Natural Cure', nature: 'Careful', evs: { hp: 252, spd: 252 } });
}
// A Vital-Spirit foe (sleep-immune) — Primeape Splashes, Yawn is `-immune`, then EQ to the win.
function vitalSpirit() {
  return mon('Primeape', ['splash'], { ability: 'Vital Spirit', nature: 'Careful', evs: { hp: 252, spd: 252 } });
}
const P2_SPLASH = ['move 1'];

function scenarios() {
  const S = [];

  // yawn_resolve_sleep_wake — Yawn (dec0) → RESOLVES at end of dec1 → Blissey sleeps (the
  //   random(2,6) counter set); Snorlax Splashes while it stays asleep + WAKES N turns later; then
  //   Earthquake to the win. The sleepTime column climbs to the counter at dec1, decrements each
  //   turn to 0 (wake), and the Earthquakes KO the (huge-HP) Blissey.
  S.push({
    id: 'yawn_resolve_sleep_wake',
    p1: [snorlax()], p2: [blissey()],
    plan1: ['move 1', 'move 4', 'move 4', 'move 4', 'move 4', 'move 4', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2'],
    plan2: P2_SPLASH,
  });

  // yawn_into_statused — Thunder Wave (dec0 → par) THEN Yawn into the par'd foe (dec1 → the
  //   `onTryHit` FAILS: `[still]` + `-fail`, NO volatile, DRAW-FREE — the foe stays par + never
  //   sleeps), then Earthquake to the win.
  S.push({
    id: 'yawn_into_statused',
    p1: [snorlax()], p2: [blissey()],
    plan1: ['move 3', 'move 1', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2'],
    plan2: P2_SPLASH,
  });

  // yawn_into_vitalspirit — Yawn (dec0) into a Vital Spirit foe → `-immune`, NO volatile, DRAW-FREE;
  //   the foe NEVER sleeps, then Earthquake to the win.
  S.push({
    id: 'yawn_into_vitalspirit',
    p1: [snorlax()], p2: [vitalSpirit()],
    plan1: ['move 1', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2'],
    plan2: P2_SPLASH,
  });

  // yawn_statused_between — Yawn (dec0 cast) THEN Thunder Wave (dec1 → par); the yawn RESOLVES at end
  //   of dec1 but the foe is now par → `-end [silent]` fires but NO sleep sets (DRAW-FREE resolve —
  //   `trySetStatus` no-ops on the already-par'd foe). The foe stays par (never slp), then EQ to win.
  S.push({
    id: 'yawn_statused_between',
    p1: [snorlax()], p2: [blissey()],
    plan1: ['move 1', 'move 3', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2'],
    plan2: P2_SPLASH,
  });

  // yawn_real_battle — a MULTI-mon game: Yawn Blissey-A, Earthquake it down (asleep), forced switch
  //   to Blissey-B, Earthquake it down → the yawn volatile CLEARS on the KO'd/replaced mon + Yawn
  //   composes in a full multi-mon game with a forced replacement.
  S.push({
    id: 'yawn_real_battle',
    p1: [snorlax()], p2: [blissey(), blissey()],
    plan1: ['move 1', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2'],
    plan2: P2_SPLASH,
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(40);
  const lines = [];
  lines.push('# yawn_golden.txt — the gen3_yawn_v1 delayed-sleep golden.');
  lines.push('# Per-decision-boundary STATE+HP+STATUS+SLEEP-COUNTER+SEED differential to GAME-END:');
  lines.push('# Yawn is DRAW-FREE at cast; the sleep random(2,6) fires at the RESOLVE (end of the');
  lines.push('# turn AFTER cast), routed through try_set_status → the SAME slp counter machinery.');
  lines.push('# sleepTime = the slp remaining-turn counter (statusState.time); 0 else.');
  lines.push('# yawnResolved = a `|-end|...|move: Yawn` fired this decision (coverage marker).');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status left) sleep1 p2(...) sleep2 first yawnResolved');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0, resolveTotal = 0, sleepRowsTotal = 0;
  // Scenarios whose target must actually SLEEP (the resolve must fire) vs must NOT.
  const mustSleep = new Set(['yawn_resolve_sleep_wake', 'yawn_real_battle']);
  const mustNotSleep = new Set(['yawn_into_statused', 'yawn_into_vitalspirit', 'yawn_statused_between']);

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);

    let scenDecs = 0;
    let scenSleepRows = 0;
    let scenResolveRows = 0;
    for (const seed of seeds) {
      let rec;
      try { rec = await runBattle(sc, seed); } catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); continue; }
      if (rec.gen !== 3) { failures.push(`${sc.id}: expected gen 3, got ${rec.gen}`); break; }
      if (!rec.initSeed || rec.decisions.length === 0) { failures.push(`${sc.id} seed ${seed}: no decisions`); continue; }
      const seedStr = seed.join(',');
      lines.push(['INIT', sc.id, rec.initSeed, seedStr].join('\t'));

      rec.decisions.forEach((d) => {
        const sp = (s) => [s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status, s.left].join('\t');
        lines.push([
          'DEC', sc.id, seedStr, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
          sp(d.p1), d.p1.sleep, sp(d.p2), d.p2.sleep, d.firstMover, d.yawnResolved ? 1 : 0,
        ].join('\t'));
        decRows++; scenDecs++;
        if (d.p1.status === 'slp' || d.p2.status === 'slp') scenSleepRows++;
        if (d.yawnResolved) { resolveTotal++; scenResolveRows++; }
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
    sleepRowsTotal += scenSleepRows;
    if (scenDecs === 0) failures.push(`${sc.id}: produced NO decision rows`);
    if (mustSleep.has(sc.id)) {
      if (scenResolveRows < 10) failures.push(`${sc.id}: only ${scenResolveRows} Yawn-resolve rows (<10) — the resolve barely fires`);
      if (scenSleepRows < 10) failures.push(`${sc.id}: only ${scenSleepRows} asleep rows (<10) — the sleep barely lands`);
    }
    if (mustNotSleep.has(sc.id) && scenSleepRows > 0) {
      failures.push(`${sc.id}: expected 0 asleep rows (Yawn must NOT sleep here), got ${scenSleepRows}`);
    }
  }

  if (failures.length) {
    console.error('YAWN GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }
  if (winRows < 100) { console.error(`YAWN GOLDEN: too few WIN rows (${winRows} < 100)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `yawn golden: ${S.length} scenarios, ${decRows} decision rows, ${resolveTotal} yawn-resolve rows, ` +
    `${sleepRowsTotal} asleep rows, ${winRows} wins + ${tieRows} ties -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
