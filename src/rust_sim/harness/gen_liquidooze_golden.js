// gen_liquidooze_golden.js — the LIQUID OOZE heal-reversal golden (`gen3_liquid_ooze_v1`).
//
// gen-3 Liquid Ooze (`onSourceTryHeal`: a `drain`/`leechseed` heal into the holder is turned into
// DAMAGE on the HEALER — `this.damage(damage, null, null, null, true); return 0`) REVERSES the two
// heal paths the port models: a DRAIN move (Giga/Mega/Absorb/Leech-Life) and the Leech Seed
// residual. Instead of healing, the drainer/seeder takes the would-be heal as damage
// (`|-damage|<healer>|<HP>|[from] ability: Liquid Ooze|[of] <ooze-mon>`), and it can KO the healer.
// DRAW-FREE (probe `harness/probe_batch89_abilities_items.js`: the drain move's normal draws only;
// the leech residual draws nothing extra). Dream Eater is EXCLUDED in gen3 (moot — not a modeled
// drain move). This golden makes the reversal OBSERVABLE on the per-decision HP timeline: the
// healer's HP goes DOWN (not up) after a drain / on the leech residual.
//
//   COVERS (each a DECISIVE full battle in gen3customgame; p2 = a Liquid Ooze Tentacruel that
//   never attacks [Barrier / Splash] — so p1 only ever takes the LO reversal → guaranteed P1 win):
//     lo_drain_reversal — p1 Venusaur GIGA-DRAINS the Liquid Ooze Tentacruel: Venusaur's HP DROPS
//                         (the reversal) while Tentacruel takes the drain damage. Then Sludge Bomb
//                         to the win.
//     lo_control_heal   — the SAME plan/teams but Tentacruel has CLEAR BODY (not Liquid Ooze):
//                         Venusaur HEALS from Giga Drain (the discriminator — only Liquid Ooze
//                         makes lo_drain_reversal's HP timeline differ).
//     lo_leech_reversal — p1 Venusaur LEECH-SEEDS the Liquid Ooze Tentacruel: each residual the
//                         SEEDER (Venusaur) takes the drained amount as DAMAGE (Tentacruel's own
//                         Leech-Seed `-damage` fires first, then Venusaur's Liquid-Ooze `-damage`).
//                         Then Sludge Bomb to the win.
//
// Output: tests/vectors/liquidooze_golden.txt (the naturalcure TAB format).
//
// Run:  node src/rust_sim/harness/gen_liquidooze_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/liquidooze_golden.txt');
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
  let x = 0x7b4a_19c3 >>> 0;
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

function snap(side) {
  const a = side.active[0];
  if (!a) return { species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', left: side.pokemonLeft };
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status: a.status || '-', left: side.pokemonLeft,
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

// Coverage marker: did a Liquid Ooze reversal `-damage ... [from] ability: Liquid Ooze` fire?
function oozeFiredSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    if (log[i].includes('|-damage|') && log[i].includes('[from] ability: Liquid Ooze')) return true;
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

  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen, oozeRows: 0 };

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
    const oozed = oozeFiredSince(log, logLenBefore);
    if (oozed) rec.oozeRows++;
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      oozed,
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

// ── Scenarios ────────────────────────────────────────────────────────────────
// p1 Venusaur (max HP/SpA) draws vs a Liquid Ooze Tentacruel that NEVER attacks (Barrier /
// Splash) so p1 only ever takes the LO reversal → guaranteed decisive P1 (Sludge Bomb finishes).
function venusaur(moves) {
  return mon('Venusaur', moves, { ability: 'Overgrow', nature: 'Modest', evs: { hp: 252, spa: 252, def: 4 } });
}
function tentacruel(ability) {
  return mon('Tentacruel', ['barrier', 'splash'], { ability, nature: 'Bold', evs: { hp: 252, def: 252 } });
}

function scenarios() {
  const S = [];

  // lo_drain_reversal — Giga Drain the Liquid Ooze Tentacruel: Venusaur's HP DROPS (the reversal)
  //   while Tentacruel takes the drain damage. Two Giga Drains show the reversal, then Sludge Bomb.
  S.push({
    id: 'lo_drain_reversal',
    p1: [venusaur(['gigadrain', 'sludgebomb'])], p2: [tentacruel('Liquid Ooze')],
    plan1: ['move 1', 'move 1', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2'],
    plan2: ['move 2'],
  });

  // lo_control_heal — the SAME plan/teams but Tentacruel has CLEAR BODY (not Liquid Ooze): Venusaur
  //   HEALS from Giga Drain (the discriminator — the reversal never fires).
  S.push({
    id: 'lo_control_heal',
    p1: [venusaur(['gigadrain', 'sludgebomb'])], p2: [tentacruel('Clear Body')],
    plan1: ['move 1', 'move 1', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2'],
    plan2: ['move 2'],
  });

  // lo_leech_reversal — Leech Seed the Liquid Ooze Tentacruel: each residual the SEEDER (Venusaur)
  //   takes the drained amount as DAMAGE. Then Sludge Bomb to the win.
  S.push({
    id: 'lo_leech_reversal',
    p1: [venusaur(['leechseed', 'sludgebomb'])], p2: [tentacruel('Liquid Ooze')],
    plan1: ['move 1', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2'],
    plan2: ['move 2'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(40);
  const lines = [];
  lines.push('# liquidooze_golden.txt — the gen3_liquid_ooze_v1 heal-reversal golden.');
  lines.push('# Per-decision-boundary STATE+HP+STATUS+SEED differential to GAME-END: Liquid Ooze');
  lines.push('# turns a drain / Leech Seed heal into DAMAGE on the HEALER (the healer\'s HP goes DOWN,');
  lines.push('# emitted |-damage|…|[from] ability: Liquid Ooze|[of] <ooze>). DRAW-FREE (the per-decision');
  lines.push('# seed matches the drain move / leech residual bit-for-bit). `oozed`=a Liquid-Ooze');
  lines.push('# reversal -damage fired this decision (coverage marker).');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status left) p2(...) first oozed');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0, oozeTotal = 0;
  const oozeScenarios = new Set(['lo_drain_reversal', 'lo_leech_reversal']);
  const noOozeScenarios = new Set(['lo_control_heal']);

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);

    let scenDecs = 0;
    let scenOoze = 0;
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
          sp(d.p1), sp(d.p2), d.firstMover, d.oozed ? 1 : 0,
        ].join('\t'));
        decRows++; scenDecs++;
        if (d.oozed) { oozeTotal++; scenOoze++; }
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
    if (oozeScenarios.has(sc.id) && scenOoze < 10) {
      failures.push(`${sc.id}: only ${scenOoze} Liquid-Ooze reversal rows (<10) — the reversal barely fires`);
    }
    if (noOozeScenarios.has(sc.id) && scenOoze > 0) {
      failures.push(`${sc.id}: expected 0 reversal rows (a non-LO control), got ${scenOoze}`);
    }
  }

  if (failures.length) {
    console.error('LIQUID OOZE GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }
  if (winRows < 90) { console.error(`LIQUID OOZE GOLDEN: too few WIN rows (${winRows} < 90)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `liquidooze golden: ${S.length} scenarios, ${decRows} decision rows, ${oozeTotal} reversal rows, ` +
    `${winRows} wins + ${tieRows} ties -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
