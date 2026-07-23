// gen_wonderguard_golden.js — the WONDER GUARD SE-only damage-gate golden (`gen3_wonder_guard_v1`).
//
// gen-3 Wonder Guard (the gen4-override `onTryHit`, inherited): a DAMAGING move into a Wonder Guard
// holder (Shedinja, Bug/Ghost, 1 HP) CONNECTS only if it is STRICTLY super-effective
// (`runEffectiveness > 0`) AND not type-immune; every neutral / resisted / 0×-immune move is
// BLOCKED with `|-immune|<t>|[from] ability: Wonder Guard`. The gate runs AFTER the accuracy roll
// (already drawn) and BEFORE crit/damage/secondary — so a BLOCKED move draws ONLY its accuracy roll
// (EXACTLY a type-immune move's draw count). BYPASSED by Status moves, self-target, typeless
// (`???`/Struggle), and ALL residual damage (a MOVE hook only — so a residual can KO the 1-HP
// Shedinja). This golden makes it OBSERVABLE on the per-decision HP/STATUS/winner timeline.
//
//   COVERS (each a DECISIVE full battle in gen3customgame; p2 Shedinja only Splashes, so p1's
//   choices fully drive the outcome):
//     wg_block_forms_then_ko — p1 Charizard throws BODY SLAM (Normal → Ghost 0×: WG-immune, a
//                              DISTINCT byte form from a plain type -immune), WATER GUN (neutral:
//                              WG-immune), MAGICAL LEAF (Grass resisted, NEVER-MISS: WG-immune with
//                              NO accuracy draw), then EMBER (Fire SE: CONNECTS → KO). The
//                              per-decision seed proves the draw-count crux (a blocked acc-move
//                              draws its accuracy; a never-miss blocked move draws NOTHING).
//     wg_control_hits        — the SAME Charizard vs a COMPOUND-EYES (no-WG) Shedinja: Body Slam
//                              shows a PLAIN 0× -immune, then Water Gun (neutral) HITS the 1-HP
//                              Shedinja → KO at dec1 (the discriminator — only Wonder Guard makes a
//                              neutral move fail; wgBlocked == 0).
//     wg_leech_bypass_ko     — p1 Venusaur LEECH-SEEDS the WG Shedinja (a Status move → WG bypass →
//                              plants); the end-of-turn Leech Seed RESIDUAL drains the 1-HP Shedinja
//                              (clampIntRange(_,1)) → it FAINTS (a residual bypasses WG). If Leech
//                              Seed MISSES (acc 90), an Ember (Fire SE) finishes it — either way
//                              decisive; the residual-KO is the bypass proof.
//     wg_status_bypass_ko    — p1 Gengar THUNDER-WAVES the WG Shedinja (a Status move → WG bypass →
//                              Shedinja PARALYZED), then SHADOW BALL (Ghost SE vs Ghost 2×) → KO.
//                              The dec0 status column (par) proves the status-move bypass.
//
// Output: tests/vectors/wonderguard_golden.txt (the liquidooze TAB format; last DEC col = wgBlocked).
//
// Run:  node src/rust_sim/harness/gen_wonderguard_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/wonderguard_golden.txt');
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
  let x = 0x51ed_2c07 >>> 0;
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

// Coverage marker: did a Wonder Guard block `-immune ... [from] ability: Wonder Guard` fire?
function wgBlockedSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    if (log[i].includes('|-immune|') && log[i].includes('[from] ability: Wonder Guard')) return true;
  }
  return false;
}

// Coverage marker: did a Leech Seed RESIDUAL `-damage` (the WG bypass) fire?
function leechResidualSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    if (log[i].includes('|-damage|') && log[i].includes('[from] Leech Seed')) return true;
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
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      wgBlocked: wgBlockedSince(log, logLenBefore),
      leechResidual: leechResidualSince(log, logLenBefore),
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

// ── Scenarios ────────────────────────────────────────────────────────────────
function shedinja(ability, moves) {
  return mon('Shedinja', moves || ['splash'], { ability });
}

function scenarios() {
  const S = [];

  // wg_block_forms_then_ko — the 4 byte forms in one battle: 0×-immune WG / neutral WG / resisted
  //   never-miss WG / SE connect KO. p2 Shedinja only Splashes → p1 fully drives.
  S.push({
    id: 'wg_block_forms_then_ko',
    p1: [mon('Charizard', ['bodyslam', 'watergun', 'magicalleaf', 'ember'], { ability: 'Blaze' })],
    p2: [shedinja('Wonder Guard')],
    plan1: ['move 1', 'move 2', 'move 3', 'move 4'], // BodySlam, WaterGun, MagicalLeaf, Ember(KO)
    plan2: ['move 1'],
  });

  // wg_control_hits — the discriminator: a COMPOUND-EYES (no WG) Shedinja. Body Slam → PLAIN 0×
  //   -immune (dec0), Water Gun (neutral) HITS the 1-HP Shedinja → KO (dec1). wgBlocked == 0.
  S.push({
    id: 'wg_control_hits',
    p1: [mon('Charizard', ['bodyslam', 'watergun', 'magicalleaf', 'ember'], { ability: 'Blaze' })],
    p2: [shedinja('Compound Eyes')],
    plan1: ['move 1', 'move 2', 'move 3', 'move 4'],
    plan2: ['move 1'],
  });

  // wg_leech_bypass_ko — Leech Seed (Status → WG bypass) plants; the residual KOs the 1-HP Shedinja
  //   (a residual bypasses WG). Ember is the fallback if Leech Seed (acc 90) misses.
  S.push({
    id: 'wg_leech_bypass_ko',
    p1: [mon('Venusaur', ['leechseed', 'ember'], { ability: 'Overgrow', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [shedinja('Wonder Guard')],
    plan1: ['move 1', 'move 2', 'move 2', 'move 2'],
    plan2: ['move 1'],
  });

  // wg_status_bypass_ko — Thunder Wave (Status → WG bypass) paralyzes Shedinja, then Shadow Ball
  //   (Ghost SE vs Ghost) KOs. The dec0 status column (par) proves the status-move bypass.
  S.push({
    id: 'wg_status_bypass_ko',
    p1: [mon('Gengar', ['thunderwave', 'shadowball'], { ability: 'Levitate', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [shedinja('Wonder Guard')],
    plan1: ['move 1', 'move 2', 'move 2'],
    plan2: ['move 1'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(40);
  const lines = [];
  lines.push('# wonderguard_golden.txt — the gen3_wonder_guard_v1 SE-only damage-gate golden.');
  lines.push('# Per-decision-boundary STATE+HP+STATUS+SEED differential to GAME-END: a DAMAGING move');
  lines.push('# into a Wonder Guard Shedinja (Bug/Ghost, 1 HP) connects ONLY if STRICTLY SE, else it is');
  lines.push('# blocked (-immune [from] ability: Wonder Guard) drawing ONLY its accuracy roll. Status');
  lines.push('# moves + residuals BYPASS WG (so a residual KOs the 1-HP Shedinja). `wgBlocked`=a WG');
  lines.push('# block fired this decision (coverage marker).');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status left) p2(...) first wgBlocked');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0, wgTotal = 0;
  const blockScenarios = new Set(['wg_block_forms_then_ko']);
  const noBlockScenarios = new Set(['wg_control_hits', 'wg_leech_bypass_ko', 'wg_status_bypass_ko']);

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);

    let scenDecs = 0;
    let scenBlocks = 0;
    let scenLeechKO = 0;
    let scenPar = 0;
    let scenConnectKO = 0;
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
          sp(d.p1), sp(d.p2), d.firstMover, d.wgBlocked ? 1 : 0,
        ].join('\t'));
        decRows++; scenDecs++;
        if (d.wgBlocked) { wgTotal++; scenBlocks++; }
        if (d.leechResidual && d.p2.fainted) scenLeechKO++;
        if (d.p2.status === 'par') scenPar++;
        // A super-effective CONNECT that KO'd Shedinja (its HP hit 0 by a direct hit, not a residual).
        if (d.p2.fainted && !d.leechResidual && d.choiceP1 === 'm3') scenConnectKO++; // Ember (idx 3)
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
    // Coverage: the block scenario blocks 3 moves per run (>=100 blocks over 40 seeds); the SE Ember
    // connects+KOs every run; the leech scenario's residual KOs on most seeds; the status scenario
    // paralyzes every run.
    if (blockScenarios.has(sc.id)) {
      if (scenBlocks < 100) failures.push(`${sc.id}: only ${scenBlocks} WG blocks (<100) — the gate barely fires`);
      if (scenConnectKO < 30) failures.push(`${sc.id}: only ${scenConnectKO} SE-connect KOs (<30) — the SE move never connects`);
    }
    if (noBlockScenarios.has(sc.id) && scenBlocks > 0) {
      failures.push(`${sc.id}: expected 0 WG-block rows, got ${scenBlocks}`);
    }
    if (sc.id === 'wg_leech_bypass_ko' && scenLeechKO < 20) {
      failures.push(`${sc.id}: only ${scenLeechKO} Leech-Seed residual KOs (<20) — the residual bypass barely fires`);
    }
    if (sc.id === 'wg_status_bypass_ko' && scenPar < 30) {
      failures.push(`${sc.id}: only ${scenPar} paralyzed-Shedinja rows (<30) — Thunder Wave never landed through WG`);
    }
  }

  if (failures.length) {
    console.error('WONDER GUARD GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }
  if (winRows < 120) { console.error(`WONDER GUARD GOLDEN: too few WIN rows (${winRows} < 120)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `wonderguard golden: ${S.length} scenarios, ${decRows} decision rows, ${wgTotal} WG-block rows, ` +
    `${winRows} wins + ${tieRows} ties -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
