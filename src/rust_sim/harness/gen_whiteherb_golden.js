// gen_whiteherb_golden.js — the WHITE HERB (BOOST_RESTORE) golden (`gen3_white_herb_v1`).
//
// White Herb (`ItemData.boost_restore`) is a single-use item that restores ALL of the holder's
// NEGATIVE boost stages to 0 (positives untouched) then consumes itself, DRAW-FREE. The resolved
// gen3 item fires from `onAnyAfterMove` / `onAnySwitchIn` / `onResidual(order 29)`, so it triggers
// IMMEDIATELY after the causing stat-drop — the holder's OWN self-drop move (Superpower), a foe's
// stat-drop MOVE (Charm), or a foe's Intimidate-on-switch-in. Since it is draw-free, the
// per-decision SEED matches a control bit-for-bit; the BOOST columns + the ITEM column (whiteherb →
// "" on consumption) are the effect proof.
//
//   COVERS (each a DECISIVE full battle in gen3customgame; the WH holder does the interesting
//   thing, the foe NEVER attacks — Splash forever — so P1 always wins and both are effectively
//   mono → no bench-switch can misalign the clamped plan):
//     wh_self_drop   — p1 Snorlax (White Herb) SUPERPOWERS the foe every turn: the 1st drops its
//                      own Atk/Def −1 → White Herb RESTORES (dec0: atk/def 0, item ""); every LATER
//                      Superpower drops −1/−1 with NO restore (item gone — the SINGLE-USE proof).
//     wh_foe_charm   — the foe CHARMS p1 Snorlax (White Herb) −2 Atk turn 1 → restore (dec0: atk 0,
//                      item "") — the foe stat-drop MOVE (`apply_secondary_boost`) path.
//     wh_intimidate  — p1 Salamence's LEAD Intimidate drops p2 Snorlax (White Herb) −1 Atk →
//                      restore during construction (dec0: p2 atk 0, item "") — the switch-in path.
//     wh_net_positive— p1 Snorlax (White Herb) SWORDS-DANCES (+2 Atk) turn 1, then the foe CHARMS
//                      it (−2) turn 2 → NET 0, NO negative stage → NO trigger, item RETAINED (the
//                      no-trigger-when-net-non-negative proof: WH restores only negatives).
//
// Output: tests/vectors/whiteherb_golden.txt (the haze TAB format + a per-side item column).
//
// Run:  node src/rust_sim/harness/gen_whiteherb_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/whiteherb_golden.txt');
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
  let x = 0x7a3b_91c5 >>> 0;
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

// The active's held item id (or "" once consumed). The sim's `pokemon.item` is '' after a
// White Herb useItem(); "-" is the placeholder for a fainted (absent) active.
function itemStr(a) {
  return a && a.item ? a.item : '';
}

function snap(side) {
  const a = side.active[0];
  if (!a) return { species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', left: side.pokemonLeft, boosts: '0,0,0,0,0,0,0', item: '-' };
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status: a.status || '-', left: side.pokemonLeft, boosts: boostStr(a), item: itemStr(a),
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

// Coverage marker: did a White Herb `|-enditem|…|White Herb` fire this decision?
function whiteHerbFiredSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    if (log[i].startsWith('|-enditem') && log[i].includes('White Herb')) return true;
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

  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen, whRows: 0 };

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
    const wh = whiteHerbFiredSince(log, logLenBefore);
    if (wh) rec.whRows++;
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      wh,
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

// ── Scenarios ────────────────────────────────────────────────────────────────
// The foe NEVER attacks (Splash) so p1 always wins; the WH holder is bulky enough to observe the
// restore over a few turns before the foe dies.
function whSnorlax(moves, opts = {}) {
  return mon('Snorlax', moves, { ability: 'Own Tempo', item: 'White Herb', evs: { hp: 252, atk: 252, def: 4 }, nature: 'Adamant', ...opts });
}
function bulkyFoe(moves, opts = {}) {
  // A bulky Own-Tempo Snorlax foe that only Splashes / Charms — never attacks.
  return mon('Snorlax', moves, { ability: 'Own Tempo', evs: { hp: 252, def: 252, spd: 4 }, nature: 'Careful', ...opts });
}
function frailFoe(moves, opts = {}) {
  // A frail foe that Charms / Splashes then dies to Body Slam.
  return mon('Alakazam', moves, { ability: 'Own Tempo', evs: { spe: 252, hp: 4 }, nature: 'Timid', ...opts });
}

function scenarios() {
  const S = [];

  // wh_self_drop — p1 Snorlax (White Herb) Superpowers the bulky foe every turn: the 1st self-drop
  //   (Atk/Def −1) is RESTORED (dec0 atk/def 0, item ""); each LATER Superpower drops −1/−1 with no
  //   restore (SINGLE-USE). The foe (bulky Snorlax) just Splashes → p1 wins.
  S.push({
    id: 'wh_self_drop',
    p1: [whSnorlax(['superpower', 'splash'])],
    p2: [bulkyFoe(['splash'])],
    plan1: ['move 1'],  // Superpower forever
    plan2: ['move 1'],  // Splash forever
  });

  // wh_foe_charm — the foe CHARMS p1 Snorlax (White Herb) −2 Atk turn 1 → restore (dec0 atk 0,
  //   item ""); then p1 Body Slams the frail foe down. The apply_secondary_boost trigger path.
  S.push({
    id: 'wh_foe_charm',
    p1: [whSnorlax(['bodyslam', 'splash'])],
    p2: [frailFoe(['charm', 'splash'])],
    plan1: ['move 2', 'move 1'],       // splash t1 (so the Charm lands cleanly), bodyslam after
    plan2: ['move 1', 'move 2'],       // charm t1, splash after
  });

  // wh_intimidate — p1 Salamence's LEAD Intimidate drops p2 Snorlax (White Herb) −1 Atk → restore
  //   during construction (dec0 p2 atk 0, item ""); then Salamence EQs p2 down. The switch-in path.
  S.push({
    id: 'wh_intimidate',
    p1: [mon('Salamence', ['earthquake', 'splash'], { ability: 'Intimidate', evs: { atk: 252, spe: 252, hp: 4 }, nature: 'Adamant' })],
    p2: [whSnorlax(['splash'], { evs: { hp: 252, def: 252, spd: 4 }, nature: 'Careful' })],
    plan1: ['move 1'],  // Earthquake forever
    plan2: ['move 1'],  // Splash forever
  });

  // wh_net_positive — p1 Snorlax (White Herb) SDs (+2 Atk) turn 1, then the foe CHARMS it (−2) turn
  //   2 → NET 0, NO negative → NO trigger, item RETAINED. Then Body Slam to the win. The
  //   no-trigger-when-net-non-negative proof (WH restores only negatives).
  S.push({
    id: 'wh_net_positive',
    p1: [whSnorlax(['swordsdance', 'bodyslam', 'splash'])],
    p2: [frailFoe(['charm', 'splash'])],
    plan1: ['move 1', 'move 3', 'move 2'],  // SD t1, splash t2 (while foe Charms), bodyslam after
    plan2: ['move 2', 'move 1', 'move 2'],  // splash t1, charm t2, splash after
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(40);
  const lines = [];
  lines.push('# whiteherb_golden.txt — the gen3_white_herb_v1 BOOST_RESTORE golden.');
  lines.push('# Per-decision-boundary STATE+HP+STATUS+BOOSTS(7-stage/side)+ITEM(/side)+SEED');
  lines.push('# differential to GAME-END. White Herb restores the holder\'s NEGATIVE boosts to 0 +');
  lines.push('# consumes the item (whiteherb -> ""), DRAW-FREE (the per-decision seed matches a');
  lines.push('# control bit-for-bit). `wh`=a White Herb -enditem fired this decision (coverage).');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status left) p1boosts(a,d,sa,sd,sp,ac,ev) p1item \\');
  lines.push('#        p2(...) p2boosts p2item first wh');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0, whTotal = 0;
  // In-decision restore fire (the `wh` -enditem marker within a decision's log window).
  const whTriggerScenarios = new Set(['wh_self_drop', 'wh_foe_charm']);
  const noTriggerScenarios = new Set(['wh_net_positive']);
  // The LEAD-Intimidate restore fires during CONSTRUCTION (before any decision's log window), so
  // its proof is the ITEM column: dec0's WH holder (p2) must already read item "" + atk 0.
  const constructionRestoreScenarios = new Set(['wh_intimidate']);

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);

    let scenDecs = 0;
    let scenWh = 0;
    let ctorRestores = 0;
    for (const seed of seeds) {
      let rec;
      try { rec = await runBattle(sc, seed); } catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); continue; }
      if (rec.gen !== 3) { failures.push(`${sc.id}: expected gen 3, got ${rec.gen}`); break; }
      if (!rec.initSeed || rec.decisions.length === 0) { failures.push(`${sc.id} seed ${seed}: no decisions`); continue; }
      const seedStr = seed.join(',');
      lines.push(['INIT', sc.id, rec.initSeed, seedStr].join('\t'));

      // The construction-restore proof: for a lead-Intimidate scenario, dec0's WH holder (p2)
      // must read item "" + a restored atk (0 or positive) — the item consumed during ctor.
      if (constructionRestoreScenarios.has(sc.id)) {
        const d0 = rec.decisions[0];
        const atk = Number(d0.p2.boosts.split(',')[0]);
        if (d0.p2.item === '' && atk >= 0) ctorRestores++;
      }

      rec.decisions.forEach((d) => {
        const sp = (s) => [s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status, s.left].join('\t');
        lines.push([
          'DEC', sc.id, seedStr, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
          sp(d.p1), d.p1.boosts, d.p1.item, sp(d.p2), d.p2.boosts, d.p2.item, d.firstMover, d.wh ? 1 : 0,
        ].join('\t'));
        decRows++; scenDecs++;
        if (d.wh) { whTotal++; scenWh++; }
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
    if (whTriggerScenarios.has(sc.id) && scenWh < 10) {
      failures.push(`${sc.id}: only ${scenWh} White Herb rows (<10) — the restore barely fires`);
    }
    if (noTriggerScenarios.has(sc.id) && scenWh > 0) {
      failures.push(`${sc.id}: expected 0 White Herb rows (net non-negative), got ${scenWh}`);
    }
    if (constructionRestoreScenarios.has(sc.id) && ctorRestores < 10) {
      failures.push(`${sc.id}: only ${ctorRestores} construction restores (<10) — the lead Intimidate White Herb didn't fire`);
    }
  }

  if (failures.length) {
    console.error('WHITE HERB GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }
  if (winRows < 100) { console.error(`WHITE HERB GOLDEN: too few WIN rows (${winRows} < 100)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `whiteherb golden: ${S.length} scenarios, ${decRows} decision rows, ${whTotal} White Herb rows, ` +
    `${winRows} wins + ${tieRows} ties -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
