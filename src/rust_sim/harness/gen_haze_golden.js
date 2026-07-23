// gen_haze_golden.js — the HAZE boost-reset FIELD-move golden (`gen3_haze_v1`).
//
// gen-3 Haze (`onHitField`: `this.add('-clearallboost'); for (const p of this.getAllActive())
// p.clearBoosts();`) is a category-Status FIELD move (`type Ice`, `accuracy: true` [never-miss],
// `target: all`, `priority 0`, resolved at the user's speed slot — NOT a residual) that emits ONE
// `|-clearallboost` line and zeroes BOTH actives' boost stages INCLUDING the USER's own. DRAW-FREE
// (probe-settled: a Haze turn draws the SAME count as a Splash control — only the endTurn Quick
// Claw). This golden makes the boost-clear OBSERVABLE on a per-decision 7-stage BOOST column for
// BOTH sides: both actives climb their boosts over a few turns, then the caster HAZES → every stage
// (atk/def/spa/spd/spe/acc/eva) drops to 0 on BOTH sides at once. The revert-verified HZ1/HZ2 pins
// are the wrong-clear discriminator (reverting the clear leaves the boosts standing → the pin fails).
//
//   COVERS (each a DECISIVE full battle in gen3customgame; p1 = mono Snorlax, p2 = mono Alakazam
//   that Calm-Minds EVERY turn and never attacks — so p1 never takes damage → guaranteed P1 win,
//   and both mono → no bench-switch can misalign the clamped plan):
//     haze_clears_both   — Snorlax SD, SD (+4 Atk) while Alakazam CMs (+2 SpA/SpD), then HAZE
//                          (dec2) -> BOTH reset to all-0 (Snorlax's OWN +4 Atk wiped too — the
//                          getAllActive() proof), then Body Slam the de-boosted Alakazam down.
//     haze_no_boost_noop — Haze cast at dec0 with NO boosts up: a draw-free no-op (both all-0);
//                          the seed matches a plain move. Then Body Slam to the win.
//
// Output: tests/vectors/haze_golden.txt (the naturalcure TAB format + a 7-stage BOOST column/side).
//
// Run:  node src/rust_sim/harness/gen_haze_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/haze_golden.txt');
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
  let x = 0x51ed_2c0b >>> 0;
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

// The 7 boost stages in the port's BOOST order (atk,def,spa,spd,spe,acc,eva).
function boostStr(a) {
  const b = a ? a.boosts : {};
  return [b.atk || 0, b.def || 0, b.spa || 0, b.spd || 0, b.spe || 0, b.accuracy || 0, b.evasion || 0].join(',');
}

function snap(side) {
  const a = side.active[0];
  if (!a) return { species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', left: side.pokemonLeft, boosts: '0,0,0,0,0,0,0' };
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status: a.status || '-', left: side.pokemonLeft, boosts: boostStr(a),
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

// Coverage marker: did a Haze `|-clearallboost` fire this decision?
function hazeFiredSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) if (log[i].startsWith('|-clearallboost')) return true;
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

  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen, hazeRows: 0 };

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
    const hazed = hazeFiredSince(log, logLenBefore);
    if (hazed) rec.hazeRows++;
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      hazed,
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

// ── Scenarios ────────────────────────────────────────────────────────────────
// p1 = mono Snorlax (Immunity, max HP/SpD bulk) with [swordsdance, haze, bodyslam, splash]. p2 =
// mono Alakazam (Synchronize) that CALM-MINDS EVERY turn and NEVER attacks — so p1 never takes
// damage (guaranteed decisive P1) while p2 still stacks SpA/SpD boosts that Haze must clear. Both
// mono → no bench-switch can misalign the clamped plan. Snorlax Swords-Dances (its own Atk climbs),
// HAZES (BOTH sides reset to all-0 — the user's own Atk wiped too, the getAllActive() proof), then
// Body-Slams the de-boosted Alakazam down (physical, so p2's CM SpD does nothing).
function snorlax() {
  return mon('Snorlax', ['swordsdance', 'haze', 'bodyslam', 'splash'], {
    ability: 'Immunity', nature: 'Careful', evs: { hp: 252, spd: 252, atk: 4 },
  });
}
function calmMinder() {
  return mon('Alakazam', ['calmmind', 'psychic'], {
    ability: 'Synchronize', nature: 'Modest', evs: { spa: 252, spe: 252, hp: 4 },
  });
}
// p2 CALM MINDS forever (never Psychic) — it never hurts Snorlax, so p1 always wins.
const HAZE_P2_PLAN = ['move 1'];

function scenarios() {
  const S = [];

  // haze_clears_both — Snorlax SD, SD (+4 Atk) while Alakazam CMs (+2 SpA/SpD), then HAZE (dec2) ->
  //   BOTH reset to all-0 (Snorlax's own +4 Atk wiped too), then Body Slam to the win. The BOOST
  //   columns climb p1 Atk 2->4 / p2 SpA+SpD 1->2, then BOTH zero at dec2 (and p2 re-climbs after).
  S.push({
    id: 'haze_clears_both',
    p1: [snorlax()], p2: [calmMinder()],
    plan1: ['move 1', 'move 1', 'move 2', 'move 3'],
    plan2: HAZE_P2_PLAN,
  });

  // haze_no_boost_noop — Haze cast at dec0 with NO boosts up: a draw-free no-op (both all-0), the
  //   seed matches a plain Splash. Then Body Slam to the win (p2 keeps CMing → its boosts climb but
  //   never get re-Hazed, the un-cleared contrast to haze_clears_both).
  S.push({
    id: 'haze_no_boost_noop',
    p1: [snorlax()], p2: [calmMinder()],
    plan1: ['move 2', 'move 3'],
    plan2: HAZE_P2_PLAN,
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(40);
  const lines = [];
  lines.push('# haze_golden.txt — the gen3_haze_v1 boost-reset FIELD-move golden.');
  lines.push('# Per-decision-boundary STATE+HP+STATUS+BOOSTS(7-stage/side)+SEED differential to');
  lines.push('# GAME-END: Haze zeroes BOTH actives\' boosts (incl. the user\'s own) + emits');
  lines.push('# |-clearallboost, DRAW-FREE (the per-decision seed matches a Splash bit-for-bit).');
  lines.push('# `hazed`=a Haze -clearallboost fired this decision (coverage marker).');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status left) p1boosts(a,d,sa,sd,sp,ac,ev) p2(...) p2boosts first hazed');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0, hazeTotal = 0;
  const hazeScenarios = new Set(['haze_clears_both', 'haze_no_boost_noop']);
  const noHazeScenarios = new Set();

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);

    let scenDecs = 0;
    let scenHazes = 0;
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
          sp(d.p1), d.p1.boosts, sp(d.p2), d.p2.boosts, d.firstMover, d.hazed ? 1 : 0,
        ].join('\t'));
        decRows++; scenDecs++;
        if (d.hazed) { hazeTotal++; scenHazes++; }
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
    if (hazeScenarios.has(sc.id) && scenHazes < 10) {
      failures.push(`${sc.id}: only ${scenHazes} Haze rows (<10) — the clear barely fires`);
    }
    if (noHazeScenarios.has(sc.id) && scenHazes > 0) {
      failures.push(`${sc.id}: expected 0 Haze rows (a Splash control), got ${scenHazes}`);
    }
  }

  if (failures.length) {
    console.error('HAZE GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }
  if (winRows < 60) { console.error(`HAZE GOLDEN: too few WIN rows (${winRows} < 60)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `haze golden: ${S.length} scenarios, ${decRows} decision rows, ${hazeTotal} Haze rows, ` +
    `${winRows} wins + ${tieRows} ties -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
