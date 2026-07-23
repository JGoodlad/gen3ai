// gen_trick_golden.js — the TRICK (item swap) golden (`gen3_trick_v1`).
//
// Trick (`trick`, num 271) is a category-Status ITEM-SWAP move (type Psychic, accuracy 100 →
// draws ONE randomChance(100,100), target normal, NO bypasssub). The DRAW MODEL is ONE accuracy
// draw then a DRAW-FREE swap, so the per-decision SEED matches bit-for-bit; the per-side ITEM-NUM
// columns (the swap identity) + HP/STATUS are the effect proof. Probe-settled vs the omniscient
// sim (`harness/probe_batch89_trick_edges.js` + `probe_batch89_haze_trick_yawn.js` +
// `probe_trick_open_qs{,2}.js`).
//
//   COVERS (each a DECISIVE full battle in gen3customgame; the foe never attacks — Splash — so P1
//   always wins):
//     trick_two_swap    — p1 Alakazam (Silk Scarf) Tricks a Leftovers foe → BOTH items swap
//                         (dec0: p1 leftovers / p2 silkscarf); then Psychic to the win.
//     trick_one_sided   — p1 Alakazam (ITEMLESS) Tricks a Leftovers foe → one-sided swap: the foe
//                         loses its item (`-enditem [silent]`), p1 gains it (dec0: p1 leftovers /
//                         p2 itemless num 0).
//     trick_sticky_hold — p1 Alakazam (Silk Scarf) Tricks a STICKY HOLD Muk → PLAIN `-immune`, NO
//                         swap (items UNCHANGED at every decision); Psychic (SE vs Poison) KOs Muk.
//     trick_substitute  — the foe SUBSTITUTES then p1 Tricks into the sub → `[still]`+`-fail`, NO
//                         swap (no bypasssub); Psychic breaks the sub + KOs (items UNCHANGED).
//     trick_both_itemless — both sides ITEMLESS → Trick FAILS (`[still]`+`-fail`, both num 0).
//     trick_cb_release  — p1 Alakazam (CHOICE BAND) Tricks its Band away → the CB user is UNLOCKED
//                         (uses Psychic — a DIFFERENT slot — the NEXT turn; if the port kept the
//                         lock the decision count would diverge). The RECEIVER (foe) locks on its
//                         own next move (automatic). dec0: p1 leftovers / p2 choiceband.
//     trick_real_battle — p1 Alakazam Tricks its CB into the foe lead, p1 SWITCHES to Snorlax, and
//                         Snorlax KOs both foes (a forced replacement between them) — Trick composed
//                         with the real switch/replacement machinery, to game-end.
//
// Output: tests/vectors/trick_golden.txt (the whiteherb TAB format; the item column is the item
// dex NUM, 0 = itemless, so the swap identity is asserted).
//
// Run:  node src/rust_sim/harness/gen_trick_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/trick_golden.txt');
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
  let x = 0x51ed_2c9f >>> 0;
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

// The active's held item dex NUM (0 = itemless — the collision-free sentinel; no gen3 item is 0).
function itemNum(battle, a) {
  if (!a || !a.item) return 0;
  const it = battle.dex.items.get(a.item);
  return it && it.num ? it.num : 0;
}

function snap(battle, side) {
  const a = side.active[0];
  if (!a) return { species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', left: side.pokemonLeft, boosts: '0,0,0,0,0,0,0', item: 0 };
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status: a.status || '-', left: side.pokemonLeft, boosts: boostStr(a), item: itemNum(battle, a),
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

// Coverage markers: did a Trick RESOLVE (a `|move|…|Trick|…` line) and did it SWAP (a
// `|-…|…|[from] move: Trick` or `-activate|…|move: Trick` line) this decision?
function trickMarkersSince(log, fromIdx) {
  let used = false, swap = false;
  for (let i = fromIdx; i < log.length; i++) {
    const l = log[i];
    if (l.startsWith('|move|') && /\|Trick(\||$)/.test(l)) used = true;
    if (l.includes('move: Trick')) swap = true;
  }
  return { used, swap };
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
    const mk = trickMarkersSince(log, logLenBefore);
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle, battle.sides[0]),
      p2: snap(battle, battle.sides[1]),
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      used: mk.used,
      swap: mk.swap,
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

// ── Scenarios ────────────────────────────────────────────────────────────────
function zam(moves, opts = {}) {
  // Alakazam: fast, Special sweeper (Psychic KOs frail foes fast). Synchronize (inert here).
  return mon('Alakazam', moves, { ability: 'Synchronize', evs: { spa: 252, spe: 252, hp: 4 }, nature: 'Timid', ...opts });
}
function frailFoe(species, moves, opts = {}) {
  return mon(species, moves, { ability: 'Own Tempo', evs: { spe: 252, hp: 4 }, nature: 'Timid', ...opts });
}

function scenarios() {
  const S = [];

  // trick_two_swap — a full two-item swap (Silk Scarf <-> Leftovers), then Psychic to the win.
  S.push({
    id: 'trick_two_swap',
    p1: [zam(['trick', 'psychic'], { item: 'Silk Scarf' })],
    p2: [frailFoe('Ledian', ['splash'], { item: 'Leftovers' })],
    plan1: ['move 1', 'move 2'],   // Trick t1, Psychic after
    plan2: ['move 1'],             // Splash forever
  });

  // trick_one_sided — p1 itemless -> the foe loses its Leftovers (`-enditem [silent]`), p1 gains it.
  S.push({
    id: 'trick_one_sided',
    p1: [zam(['trick', 'psychic'], { item: '' })],
    p2: [frailFoe('Ledian', ['splash'], { item: 'Leftovers' })],
    plan1: ['move 1', 'move 2'],
    plan2: ['move 1'],
  });

  // trick_sticky_hold — Sticky Hold Muk → PLAIN `-immune`, NO swap. Psychic (SE vs Poison) KOs Muk.
  S.push({
    id: 'trick_sticky_hold',
    p1: [zam(['trick', 'psychic'], { item: 'Silk Scarf' })],
    p2: [mon('Muk', ['splash'], { ability: 'Sticky Hold', item: 'Leftovers', evs: { hp: 4, spe: 252 }, nature: 'Timid' })],
    plan1: ['move 1', 'move 2'],
    plan2: ['move 1'],
  });

  // trick_substitute — the foe SUBs, then p1 Tricks into the sub → `[still]`+`-fail`, NO swap.
  //   Psychic then breaks the sub + KOs.
  S.push({
    id: 'trick_substitute',
    p1: [zam(['trick', 'psychic', 'splash'], { item: 'Silk Scarf' })],
    p2: [frailFoe('Furret', ['substitute', 'splash'], { item: 'Leftovers' })],
    plan1: ['move 3', 'move 1', 'move 2'],   // splash t1 (foe subs), Trick t2 (into sub → fail), Psychic after
    plan2: ['move 1', 'move 2'],             // substitute t1, splash after
  });

  // trick_both_itemless — both sides itemless → Trick FAILS (`[still]`+`-fail`, both num 0).
  S.push({
    id: 'trick_both_itemless',
    p1: [zam(['trick', 'psychic'], { item: '' })],
    p2: [frailFoe('Furret', ['splash'], { item: '' })],
    plan1: ['move 1', 'move 2'],
    plan2: ['move 1'],
  });

  // trick_cb_release — p1 Alakazam (CHOICE BAND) Tricks its Band away, then uses a DIFFERENT slot
  //   (Psychic) the NEXT turn: the CB user is UNLOCKED (else the decision count diverges). The foe
  //   receives the Band + locks on its own next move (Splash — automatic).
  S.push({
    id: 'trick_cb_release',
    p1: [zam(['trick', 'psychic', 'splash'], { item: 'Choice Band' })],
    p2: [frailFoe('Ledian', ['splash'], { item: 'Leftovers' })],
    plan1: ['move 1', 'move 2'],   // Trick t1 (CB away → locked to trick, then RELEASED), Psychic t2 (a DIFFERENT slot)
    plan2: ['move 1'],
  });

  // trick_real_battle — Trick composed with switching + a forced replacement, to game-end.
  //   p1 Alakazam Tricks its CB into the foe lead, then SWITCHES to Snorlax; Snorlax KOs both foes.
  S.push({
    id: 'trick_real_battle',
    p1: [
      zam(['trick', 'psychic'], { item: 'Choice Band' }),
      mon('Snorlax', ['bodyslam', 'splash'], { ability: 'Own Tempo', item: 'Leftovers', evs: { hp: 252, atk: 252, def: 4 }, nature: 'Adamant' }),
    ],
    p2: [
      frailFoe('Ledian', ['splash'], { item: 'Leftovers' }),
      frailFoe('Furret', ['splash'], { item: 'Silk Scarf' }),
    ],
    plan1: ['move 1', 'switch 2', 'move 1'],   // Trick t1, switch to Snorlax t2, BodySlam after
    plan2: ['move 1'],                          // Splash forever
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(40);
  const lines = [];
  lines.push('# trick_golden.txt — the gen3_trick_v1 ITEM-SWAP golden.');
  lines.push('# Per-decision-boundary STATE+HP+STATUS+BOOSTS(7-stage/side)+ITEM-NUM(/side)+SEED');
  lines.push('# differential to GAME-END. Trick draws ONE accuracy roll then a DRAW-FREE swap, so');
  lines.push('# the per-decision seed matches bit-for-bit; the per-side item dex NUM (0=itemless)');
  lines.push('# proves the swap identity. used=a `|move|…|Trick|` resolved; swap=a `move: Trick`');
  lines.push('# line fired (a successful swap) this decision (coverage markers).');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status left) p1boosts(a,d,sa,sd,sp,ac,ev) p1itemnum \\');
  lines.push('#        p2(...) p2boosts p2itemnum first used swap');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0, usedTotal = 0, swapTotal = 0;
  const swapScenarios = new Set(['trick_two_swap', 'trick_one_sided', 'trick_cb_release', 'trick_real_battle']);
  const noSwapScenarios = new Set(['trick_sticky_hold', 'trick_substitute', 'trick_both_itemless']);

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);

    let scenDecs = 0, scenUsed = 0, scenSwap = 0;
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
          sp(d.p1), d.p1.boosts, d.p1.item, sp(d.p2), d.p2.boosts, d.p2.item, d.firstMover,
          d.used ? 1 : 0, d.swap ? 1 : 0,
        ].join('\t'));
        decRows++; scenDecs++;
        if (d.used) { usedTotal++; scenUsed++; }
        if (d.swap) { swapTotal++; scenSwap++; }
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
    // Every scenario USES Trick at least once per seed.
    if (scenUsed < 30) failures.push(`${sc.id}: only ${scenUsed} Trick-used rows (<30) — Trick barely fired`);
    if (swapScenarios.has(sc.id) && scenSwap < 30) {
      failures.push(`${sc.id}: only ${scenSwap} swap rows (<30) — the swap never fired`);
    }
    if (noSwapScenarios.has(sc.id) && scenSwap > 0) {
      failures.push(`${sc.id}: expected 0 swap rows (blocked/failed Trick), got ${scenSwap}`);
    }
  }

  if (failures.length) {
    console.error('TRICK GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }
  if (winRows < 200) { console.error(`TRICK GOLDEN: too few WIN rows (${winRows} < 200)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `trick golden: ${S.length} scenarios, ${decRows} decision rows, ${usedTotal} Trick-used rows, ` +
    `${swapTotal} swap rows, ${winRows} wins + ${tieRows} ties -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
