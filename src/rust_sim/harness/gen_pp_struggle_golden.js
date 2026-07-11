// gen_pp_struggle_golden.js — Gen-3 PP-TRACKING + STRUGGLE differential.
//
// Extends the recovery/setup golden (the per-decision STATE+SEED+winner full-battle
// differential) to the NEW execution path this step adds: per-move PP counters, the
// forced-STRUGGLE substitution, and the gen-3 Struggle move. It ADDS a per-mon PP column
// (the 4 move slots' current PP) so a wrong decrement / a wrong Struggle trigger diverges
// the STATE, and asserts the exact per-decision SEED so a wrong draw model diverges too.
//
// THE DRAW MODEL (verified bit-for-bit vs the omniscient sim — `harness/probe_pp_struggle_rng.js`):
//   1. PP INIT: a moveslot's in-battle PP is `calculatePP(move, 3) = pp*8/5` (the Pokemon
//      ctor's default 3 PP-ups) for a normal move, or the raw `pp` for a `noPPBoosts` move.
//   2. PP DECREMENT: −1 per USE, DRAW-FREE, and ONLY when the mon actually MOVES (a full-
//      para / sleep / flinch / frozen / confusion-self-hit turn deducts NOTHING — deductPP
//      runs AFTER runEvent('BeforeMove') passes). A MISS / an IMMUNE hit STILL decrement.
//   3. PRESSURE −2: a move TARGETING a Pressure holder deducts 2 PP (DRAW-FREE, no RNG).
//   4. FORCED STRUGGLE: when the mon has NO usable move (all slots 0 PP, OR a Choice-Band
//      lock leaves only its locked slot which is at 0 PP) `side.choose` substitutes
//      `moveid:'struggle'` — the scripted `move K` on the exhausted slot auto-becomes
//      Struggle.
//   5. STRUGGLE: type '???' typeless (no STAB, hits Ghosts), BP 50, PHYSICAL, accuracy 100
//      → DRAWS accuracy (NOT never-miss), then crit + damage like a normal move; recoil =
//      max(floor(damageDealt/4), 1) via the gen-3 `recoil:[1,4]` path (NOT struggleRecoil=
//      maxhp/4), applied DRAW-FREE to the user. Struggle consumes no PP.
//   6. PP does NOT reset on switch-out (gen-3) — it PERSISTS.
//
// THE PROOF: drive the OMNISCIENT in-process BattleStream (no server) over CONSTRUCTED
// scenarios, capturing the running PRNG seed BEFORE the first decision (initSeed) and AFTER
// each DECISION BOUNDARY, plus each active's species/hp/maxhp/fainted/status(+counter) +
// per-slot PP + pokemon_left + first mover + winner. The Rust test seeds a BattleState at
// the init seed and runs `run_full_battle` WITHOUT re-seeding — so the post-decision seed +
// per-decision HP + STATUS + PP must match at EVERY boundary. An EXACT cross-decision
// seed+state match to game-end is the draw-ORDER+COUNT + PP + Struggle-recoil proof.
//
// FAIL-LOUD: each scenario declares the BRANCH it must realize (a forced Struggle, a
// Struggle recoil, a Pressure −2 decrement, a Struggle-into-a-Ghost hit); generation aborts
// if the sim run did NOT realize it.
//
// Output: tests/vectors/pp_struggle_golden.txt
// Run:  node src/rust_sim/harness/gen_pp_struggle_golden.js

'use strict';
const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/pp_struggle_golden.txt');
const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };

function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N',
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
  let x = 0x2f6ac13b >>> 0;
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

// The 4 move slots' current PP (padded to 4 with -1 for a mon with fewer moves), so the
// Rust test can compare per-slot. Struggle is NOT a slot (it never appears here).
function ppOf(active) {
  const pp = [-1, -1, -1, -1];
  if (active && active.moveSlots) {
    for (let k = 0; k < active.moveSlots.length && k < 4; k++) pp[k] = active.moveSlots[k].pp;
  }
  return pp;
}

function snap(side) {
  const a = side.active[0];
  if (!a) return { species: '-', hp: 0, maxhp: 0, fainted: true, status: '-', stage: 0, left: side.pokemonLeft, pp: [-1, -1, -1, -1] };
  const { status, stage } = statusOf(a);
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status, stage, left: side.pokemonLeft, pp: ppOf(a),
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

// Scan the protocol log between two decisions for the branch flags a scenario can REQUIRE.
function outcomesSince(log, fromIdx) {
  const out = { struggle: false, recoil: false, pressure2: false, miss: false, immune: false };
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    if (tag === 'move' && (p[3] || '') === 'Struggle') out.struggle = true;
    if (tag === '-damage' && (p[4] || '').includes('[from] Recoil')) out.recoil = true;
    if (tag === '-miss') out.miss = true;
    if (tag === '-immune') out.immune = true;
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

  // Track the PP decrement across a Pressure holder to flag the −2 branch.
  const script = sc.makeScript();
  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen, branchSeen: {} };

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

    // Pre-decision PP snapshot of BOTH actives (to detect a Pressure −2 drop).
    const ppBefore = [ppOf(battle.sides[0].active[0]), ppOf(battle.sides[1].active[0])];

    const choices = script(decisionNo, battle, reqState, force);
    if (!choices) break;

    const logLenBefore = log.length;
    if (choices.p1) { try { streams.omniscient.write(`>p1 ${choices.p1}`); } catch (e) {} }
    if (choices.p2) { try { streams.omniscient.write(`>p2 ${choices.p2}`); } catch (e) {} }
    for (let i = 0; i < 16; i++) await tick();

    const seedAfter = battle.prng.getSeed();
    // STALL GUARD: a rejected choice (no advance in seed/log/request) → fail loud.
    if (seedAfter === seedBefore && log.length === logLenBefore && battle.requestState === reqState) {
      throw new Error(`STALL: choice ${JSON.stringify(choices)} did not advance the battle ` +
        `(scenario ${sc.id}, decision ${decisionNo}) — the sim rejected it. Fix the script.`);
    }
    const outcomes = outcomesSince(log, logLenBefore);
    // Pressure −2: a side whose USED move slot dropped by 2 this decision (the foe holds
    // Pressure). Detect a -2 drop on the choosing side's chosen slot.
    for (let s = 0; s < 2; s++) {
      const c = s === 0 ? choices.p1 : choices.p2;
      const mm = c && c.match(/^move\s+(\d+)$/);
      if (mm) {
        const k = Number(mm[1]) - 1;
        const after = ppOf(battle.sides[s].active[0]);
        if (ppBefore[s][k] >= 0 && after[k] >= 0 && ppBefore[s][k] - after[k] === 2) outcomes.pressure2 = true;
      }
    }
    const first = reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none';

    for (const k of ['struggle', 'recoil', 'pressure2', 'miss', 'immune']) {
      if (outcomes[k]) rec.branchSeen[k] = true;
    }
    rec.decisions.push({
      request: reqState, force,
      choiceP1: encodeChoice(choices.p1), choiceP2: encodeChoice(choices.p2),
      seedAfter, p1: snap(battle.sides[0]), p2: snap(battle.sides[1]),
      firstMover: first, outcomes,
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

  const firstLiveBench = (side, battle) => {
    const s = battle.sides[side];
    for (let k = 0; k < s.pokemon.length; k++) {
      const p = s.pokemon[k];
      if (p !== s.active[0] && !p.fainted) return `switch ${k + 1}`;
    }
    return 'pass';
  };
  const fromPlan = (plan) => () => {
    let i = 0;
    return (decisionNo, battle, reqState, force) => {
      if (reqState === 'switch') {
        const c = { p1: null, p2: null };
        if (force[0]) c.p1 = firstLiveBench(0, battle);
        if (force[1]) c.p2 = firstLiveBench(1, battle);
        return c;
      }
      const entry = plan[Math.min(i, plan.length - 1)];
      i++;
      return entry;
    };
  };

  // --- (1) CHOICE-LOCK FORCED STRUGGLE + RECOIL (the protocol-battle shape, kept SHORT via
  //   an IMMUNE locked move so the foe survives the exhaustion): a Choice-Band Snorlax with
  //   Earthquake (m0, 16 PP) + fillers LOCKS to Earthquake, spams it 16× into a LEVITATE
  //   Gengar (EQ → `-immune`, 0 damage, but PP still −1) → 0 PP → FORCED Struggle (the other
  //   slots are Choice-disabled). Struggle then chips the Ghost Gengar (typeless → HITS) with
  //   recoil. The per-decision PP (16→0), the immune decrement, the forced Struggle, AND the
  //   Struggle-into-a-Ghost recoil all in ~18 turns, deterministic. Gengar's Night Shade (a
  //   fixed 100 chip) can't out-race a 524-HP Snorlax over the exhaustion. REQUIRES: a forced
  //   Struggle + a Struggle recoil. ---
  S.push({
    id: 'cb_lock_forced_struggle',
    p1: [mon('Snorlax', ['earthquake', 'bodyslam', 'crunch', 'shadowball'],
      { item: 'Choice Band', ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    // Gengar: Ghost/Poison, LEVITATE → immune to Earthquake (Ground). Night Shade deals a
    // fixed 100 (chip) — a 524-HP Snorlax survives all 16 EQ + the Struggle turns. Struggle
    // (typeless) HITS the Ghost.
    p2: [mon('Gengar', ['nightshade'], { ability: 'Levitate', nature: 'Timid', evs: { hp: 252, spe: 252 } })],
    makeScript: fromPlan([{ p1: 'move 1', p2: 'move 1' }]), // always m0 → EQ (immune), then Struggle
    require: ['struggle', 'recoil', 'immune'],
  });

  // --- (2) SINGLE-MOVE FORCED STRUGGLE + STRUGGLE-INTO-A-GHOST (typeless hits): a mon with
  //   ONLY Extreme Speed (m0, 8 PP — a 5-base-PP move) spams it 8× → 0 PP → FORCED Struggle
  //   (all slots exhausted, no Choice lock needed). The foe is a GHOST Misdreavus — Extreme
  //   Speed (Normal) is IMMUNE to it (draws acc, `-immune`, PP still −1), and STRUGGLE
  //   (typeless '???') HITS the Ghost (a typeless move has no type-chart row → 1×). So the
  //   scenario proves BOTH: an immune hit still decrements PP, AND Struggle hits a Ghost.
  //   REQUIRES: a forced Struggle + a Struggle-into-a-Ghost hit + an immune decrement.
  S.push({
    id: 'single_move_struggle_into_ghost',
    p1: [mon('Snorlax', ['extremespeed'], { ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    // Misdreavus: frail GHOST (immune to Normal Extreme Speed). Uses Quick Attack (Normal —
    // also immune-ish? No: QA is Normal into Normal Snorlax = neutral, tiny chip). It can't
    // KO a max-HP Snorlax, so the 8 Extreme Speeds (all `-immune`, PP still −1) then Struggle
    // (which HITS the Ghost) all complete. Levitate is irrelevant.
    p2: [mon('Misdreavus', ['quickattack'], { ability: 'Levitate', nature: 'Jolly', evs: { hp: 252, spe: 252 } })],
    makeScript: fromPlan([{ p1: 'move 1', p2: 'move 1' }]),
    require: ['struggle', 'immune'],
  });

  // --- (3) PRESSURE −2 PP DECREMENT: a Snorlax Body-Slams (m0, 24 PP) a Pressure Suicune —
  //   each use deducts 2 PP (not 1). Over the battle the PP falls 24→22→20→…; the per-
  //   decision PP must match the −2 cadence. Body Slam (85 BP) grinds the bulky Suicune out
  //   before Snorlax runs out of PP (so no Struggle here — this isolates the −2 decrement).
  //   REQUIRES: a Pressure −2 decrement.
  S.push({
    id: 'pressure_minus_two',
    p1: [mon('Snorlax', ['bodyslam', 'splash'], { ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Suicune', ['splash'], { ability: 'Pressure', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    makeScript: fromPlan([{ p1: 'move 1', p2: 'move 1' }]),
    require: ['pressure2'],
  });

  // --- (4) PP PERSISTS ACROSS SWITCH-OUT (gen-3 no reset) + a MISS still decrements: a
  //   Zapdos with Thunder (m0, 16 PP, 70% acc — CAN miss) fires it, PIVOTS to Snorlax, then
  //   pivots back — Thunder's PP must be UNCHANGED across the switch (persist), and each
  //   Thunder use (hit OR miss) decrements. A Choice lock is absent (no CB), so the switch is
  //   free. The per-decision PP proves the no-reset + miss-decrement. REQUIRES: nothing forced
  //   (the PP-state assertions carry it); the miss branch is opportunistic across seeds.
  S.push({
    id: 'pp_persists_switch_and_miss',
    p1: [mon('Zapdos', ['thunder', 'thunderbolt'], { nature: 'Modest', evs: { hp: 252, spa: 252 } }),
         mon('Snorlax', ['bodyslam', 'splash'], { ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Regirock', ['splash'], { nature: 'Impish', evs: { hp: 252, def: 252 } })],
    makeScript: fromPlan([
      { p1: 'move 1', p2: 'move 1' }, // Thunder (may miss) — pp 16->15
      { p1: 'switch 2', p2: 'move 1' }, // pivot to Snorlax
      { p1: 'switch 2', p2: 'move 1' }, // pivot BACK to Zapdos — Thunder pp STILL 15?
      { p1: 'move 1', p2: 'move 1' }, // Thunder again — pp 15->14
      { p1: 'move 1', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' }, // Thunderbolt — grind the frail-ish foe
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
      { p1: 'move 2', p2: 'move 1' },
    ]),
    require: [],
  });

  // --- (5) PP INTO A REAL BATTLE TO GAME-END (the union: PP + Struggle + a switch + the full
  //   move/residual/faint machinery to a win). A single-move Snorlax (Extreme Speed only, m0,
  //   8 PP) into a 2-mon foe: it exhausts ES in 8 (into a Ghost Misdreavus — immune, PP still
  //   −1) → Struggles; the Struggle (typeless) chips + KOs the frail Ghost, then a 2nd frail
  //   foe, all the way to a win — with the Struggle recoil taxing Snorlax the whole time. PP
  //   + immune-decrement + forced Struggle + a switch (the foe's post-faint replacement) + a
  //   win. REQUIRES: a forced Struggle + a win. ---
  S.push({
    id: 'pp_into_real_battle',
    p1: [mon('Snorlax', ['extremespeed'], { ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    // Two frail GHOSTs (immune to Normal Extreme Speed → 8 immune ES, PP −1 each), each using
    // a tiny chip. Struggle (typeless) hits them; the recoil + the Ghosts' chip race is decided
    // well within the cap. Levitate irrelevant.
    p2: [mon('Misdreavus', ['nightshade'], { ability: 'Levitate', nature: 'Timid', evs: { hp: 4, spe: 252 } }),
         mon('Gengar', ['nightshade'], { ability: 'Levitate', nature: 'Timid', evs: { hp: 4, spe: 252 } })],
    makeScript: fromPlan([{ p1: 'move 1', p2: 'move 1' }]),
    require: ['struggle'],
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(80);
  const lines = [];
  lines.push('# pp_struggle_golden.txt — Gen-3 PP-tracking + STRUGGLE full-battle golden.');
  lines.push('# Per-decision-boundary STATE(+status+counter+PP)+SEED+first-mover differential to GAME-END.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status stage left pp0 pp1 pp2 pp3) p2(...) first \\');
  lines.push('#        struggle recoil pressure2 immune');
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
    scenSeen[sc.id] = {};

    let scenDecs = 0;
    for (const seed of seeds) {
      let rec;
      try { rec = await runBattle(sc, seed); } catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); continue; }
      if (rec.gen !== 3) { failures.push(`${sc.id}: expected gen 3, got ${rec.gen}`); break; }
      if (!rec.initSeed || rec.decisions.length === 0) { failures.push(`${sc.id} seed ${seed}: no decisions`); continue; }
      const seedStr = seed.join(',');
      lines.push(['INIT', sc.id, rec.initSeed, seedStr].join('\t'));

      for (const k of Object.keys(rec.branchSeen)) { scenSeen[sc.id][k] = true; corpus[k] = (corpus[k] || 0) + 1; }

      rec.decisions.forEach((d) => {
        const sp = (s) => [
          s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status, s.stage, s.left,
          s.pp[0], s.pp[1], s.pp[2], s.pp[3],
        ].join('\t');
        lines.push([
          'DEC', sc.id, seedStr, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
          sp(d.p1), sp(d.p2), d.firstMover,
          d.outcomes.struggle ? 1 : 0, d.outcomes.recoil ? 1 : 0,
          d.outcomes.pressure2 ? 1 : 0, d.outcomes.immune ? 1 : 0,
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
  }

  if (failures.length) {
    console.error('PP-STRUGGLE GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }

  const need = (label, key, min) => { const n = corpus[key] || 0; if (n < min) { console.error(`PP-STRUGGLE GOLDEN: too few ${label} (${n} < ${min})`); process.exit(1); } };
  need('forced-Struggle runs', 'struggle', 40);
  need('Struggle-recoil runs', 'recoil', 40);
  need('Pressure −2 runs', 'pressure2', 40);
  need('immune-decrement runs', 'immune', 40);
  if (winRows < 40) { console.error(`PP-STRUGGLE GOLDEN: too few WIN rows (${winRows} < 40)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `pp-struggle golden: ${S.length} scenarios, ${decRows} decision rows, ${winRows} wins + ${tieRows} ties; ` +
    `branches: struggle=${corpus.struggle || 0} recoil=${corpus.recoil || 0} ` +
    `pressure2=${corpus.pressure2 || 0} immune=${corpus.immune || 0} miss=${corpus.miss || 0} -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
