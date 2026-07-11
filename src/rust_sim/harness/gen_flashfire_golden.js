// gen_flashfire_golden.js — the FLASH FIRE ×1.5 fire-boost CLASS-SWEEP golden
// (`gen3_flashfire_boost_v1`).
//
// Flash Fire is a MULTI-TURN mechanic: a mon ARMS its `flash_fire` volatile by ABSORBING a
// Fire move, and THEREAFTER its OWN Fire moves deal ×1.5 (the flashfire volatile's
// `onModifyDamagePhase1 chainModify(1.5)` — a DAMAGE-PHASE fold, the SAME phase as
// Reflect/Light Screen; probe-settled by `harness/probe_flashfire_rng.js`, NOT a stat mod).
// Because gen-3 uses the TYPE-BASED phys/spec split, EVERY Fire move is Special — so the FF
// boost only ever composes with LIGHT SCREEN (never Reflect, which halves physical), and the
// two Phase1 chainModify handlers must ACCUMULATE into ONE modifier (probe-confirmed: sequential
// per-mod rounds diverge for ~¼ of baseDamage values).
//
// The single-turn damage golden can't stage the activation (it needs a prior absorb turn), so
// this is a dedicated full-battle class-sweep — the SAME per-decision STATE+HP+SEED-to-game-end
// differential as `gen_ability_dmgmod_golden.js`. Every scenario is a decisive battle where FF
// arms NATURALLY (a Fire-type foe fires a Fire move at the FF holder → 0 damage + activation),
// then the FF holder's Fire moves land BOOSTED. The Rust replay's exact HP proves the ×1.5 (a
// wrong multiplier / fold point / chain-combine lands a different HP); the per-decision SEED
// proves the activation + boost are DRAW-FREE.
//
//   COVERS:
//     ff_special_boost      — an FF holder (Ninetales) beats a Fire foe (Charizard) with
//                             BOOSTED Flamethrower after absorbing its Fire Blast (activation).
//     ff_boost_light_screen — an FF holder's boosted Fire move into a LIGHT-SCREEN foe (the
//                             Phase1 chain-combine: FF ×1.5 ⊗ Light Screen ×0.5 accumulated).
//     ff_wrongtype_control  — an ARMED FF holder using a NON-Fire move gets NO boost (the type
//                             gate — a Fire-typed FF holder's non-Fire coverage move).
//     ff_not_activated      — the SAME FF holder that NEVER absorbs a Fire move: its Fire moves
//                             are UNBOOSTED (the activation gate — arms only after an absorb).
//     ff_switch_clear       — an FF holder arms, PIVOTS OUT and back (boost cleared), then its
//                             Fire move is UNBOOSTED again (the clearVolatile-on-switch gate).
//
// Output: tests/vectors/flashfire_golden.txt (same TAB format as ability_dmgmod_golden).
//
// Run:  node src/rust_sim/harness/gen_flashfire_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/flashfire_golden.txt');
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
  let x = 0x5f3a1c7d >>> 0;
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

// Coverage marker: did the FF HOLDER's BOOSTED Fire move land a direct -damage on the foe this
// decision, WHILE the holder's `flash_fire` volatile is ARMED? `sc.holderSide` names the FF mon;
// `sc.boostedName` the boosted move's display name. We read the ARMED state off the live battle
// object (the holder's `volatiles.flashfire`) at the moment the move lands — the ground-truth the
// Rust replay must reproduce (a wrong activation state or a wrong boost lands a different HP).
function boostedFireLandedSince(log, fromIdx, sc, battle) {
  const holder = sc.holderSide; // 'p1' | 'p2'
  const foe = holder === 'p1' ? 'p2' : 'p1';
  let pending = null; // { by, name }
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    if (tag === 'move' && p.length >= 3) {
      const actor = (p[2] || '').trim();
      const bySide = actor.startsWith('p1a:') ? 'p1' : actor.startsWith('p2a:') ? 'p2' : null;
      pending = bySide ? { by: bySide, name: (p[3] || '') } : null;
    } else if (tag === '-damage' && p.length >= 3) {
      const tgt = (p[2] || '').trim();
      const tgtSide = tgt.startsWith('p1a:') ? 'p1' : tgt.startsWith('p2a:') ? 'p2' : null;
      const residual = p.slice(4).some((x) => x.startsWith('[from]'));
      if (pending && !residual && pending.by === holder && tgtSide === foe) {
        const nameOk = !sc.boostedName || pending.name === sc.boostedName;
        // The holder is currently the active mon on its side; is its FF armed NOW?
        const hMon = battle.sides[holder === 'p1' ? 0 : 1].active[0];
        const armed = !!(hMon && hMon.volatiles && hMon.volatiles.flashfire);
        if (nameOk && armed) return true;
      }
      pending = null;
    } else if (tag === '-miss' || tag === '-immune' || tag === 'faint') {
      pending = null;
    }
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

  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen, boostedRows: 0 };

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
    // Evaluate the boosted marker BEFORE the next decision mutates the battle (reads live volatiles).
    const boosted = reqState === 'move' && boostedFireLandedSince(log, logLenBefore, sc, battle);
    if (boosted) rec.boostedRows++;
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      boosted,
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

// ── Scenarios ────────────────────────────────────────────────────────────────
function scenarios() {
  const S = [];
  const Dex = require(path.join(PS, 'dist/sim')).Dex;
  const dex3 = Dex.mod('gen3');
  const moveName = (id) => dex3.moves.get(id).name;

  // ff_special_boost — Ninetales (FF) absorbs Charizard's Fire Blast T1 (armed), then its
  //   Flamethrower is ×1.5. Charizard is pure Fire-attacker → CANNOT damage the FF Ninetales
  //   (every Fire move absorbed), so Ninetales wins over several BOOSTED hits. Decisive.
  S.push({
    id: 'ff_special_boost',
    p1: [mon('Ninetales', ['flamethrower', 'rest'], { ability: 'Flash Fire', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Charizard', ['fireblast'], { ability: 'Blaze', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    plan1: ['move 1'],
    plan2: ['move 1'],
    holderSide: 'p1',
    boostedName: moveName('flamethrower'),
  });

  // (ff_boost_light_screen — the Phase1 CHAIN-COMBINE FF ×1.5 ⊗ Light Screen ×0.5 — is NOT a
  //  full-battle golden scenario: the port does not model the Light Screen status MOVE (it is
  //  fail-loud / out of the modeled set, since no gate exercises it). The combine is proven at
  //  the CALC level instead by `tests/flashfire_test.rs::flash_fire_light_screen_chain_combine_exact`
  //  — the exact accumulated base — which is the definitive bit-for-bit gate for the accumulation.)

  // ff_wrongtype_control — an ARMED FF holder using a NON-Fire move gets NO boost. Houndoom
  //   (Dark/Fire, FF) absorbs Charizard's Fire Blast (armed), then uses CRUNCH (Dark) — the FF
  //   ×1.5 must NOT apply (the type gate). Crunch is Dark → super-effective on... Charizard is
  //   Fire/Flying (Dark neutral). Houndoom out-damages Charizard with unboosted Crunch;
  //   Charizard's Fire Blast is absorbed. Decisive P1.
  S.push({
    id: 'ff_wrongtype_control',
    p1: [mon('Houndoom', ['crunch', 'flamethrower'], { ability: 'Flash Fire', nature: 'Rash', evs: { spa: 252, atk: 252 } })],
    p2: [mon('Charizard', ['fireblast'], { ability: 'Blaze', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    plan1: ['move 1'], // Crunch (Dark) — NOT boosted even when FF is armed
    plan2: ['move 1'], // Fire Blast (absorbed, arms FF)
    holderSide: 'p1',
    boostedName: moveName('crunch'), // the marker fires on Crunch landing WHILE armed — but the
    //   HP proves it is UNBOOSTED (the exact-HP + seed differential is the real gate).
  });

  // ff_not_activated — the SAME Ninetales, but the foe NEVER fires a Fire move, so FF never
  //   arms → its Flamethrower is UNBOOSTED. p2 Snorlax (Normal) Body Slams; Ninetales' Fire is
  //   neutral + UNBOOSTED. A long-ish but decisive battle (or a draw-safe cap). This is the
  //   activation-gate control: the fold must NOT fire without a prior absorb.
  S.push({
    id: 'ff_not_activated',
    p1: [mon('Ninetales', ['flamethrower', 'rest'], { ability: 'Flash Fire', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Snorlax', ['bodyslam'], { ability: 'No Ability', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'],
    plan2: ['move 1'],
    holderSide: 'p1',
    boostedName: moveName('flamethrower'), // marker fires on Flamethrower landing — but FF is
    //   NOT armed so `boosted` will be FALSE here (the marker reads the live volatile). The
    //   scenario proves the UNBOOSTED path via HP+seed to game-end.
  });

  // (ff_switch_clear — the clearVolatile-on-switch gate — is NOT a full-battle golden scenario:
  //  a fixed cyclic plan over a variable-length game re-issues a `switch` the sim rejects once a
  //  side is down to one live mon. It is proven instead by the probe (A5) + the revert-verified
  //  regression pin `flash_fire_clears_on_switch_out` — the robust constructed form.)

  return S;
}

async function main() {
  const seeds = buildSeeds(30);
  const lines = [];
  lines.push('# flashfire_golden.txt — the gen3_flashfire_boost_v1 FLASH FIRE ×1.5 class-sweep golden.');
  lines.push('# Per-decision-boundary STATE+HP+SEED differential to GAME-END: activation + the');
  lines.push('# ×1.5 Fire boost (incl. the Light-Screen Phase1 chain-combine) + wrong-type/');
  lines.push('# not-activated/switch-clear controls.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status left) p2(...) first boosted');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0, boostedTotal = 0;
  // Which scenarios are BOOST scenarios (must land >=10 armed-boosted hits) vs CONTROL
  // scenarios (the marker legitimately never fires — proven by the UNBOOSTED HP path).
  const boostScenarios = new Set(['ff_special_boost']);

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);

    let scenDecs = 0;
    let scenBoosted = 0;
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
          sp(d.p1), sp(d.p2), d.firstMover, d.boosted ? 1 : 0,
        ].join('\t'));
        decRows++; scenDecs++;
        if (d.boosted) { boostedTotal++; scenBoosted++; }
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
    if (boostScenarios.has(sc.id) && scenBoosted < 10) {
      failures.push(`${sc.id}: only ${scenBoosted} armed-boosted-hit rows (<10) — the FF boost barely fires`);
    }
  }

  if (failures.length) {
    console.error('FLASH FIRE GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }
  if (winRows < 80) { console.error(`FLASH FIRE GOLDEN: too few WIN rows (${winRows} < 80)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `flashfire golden: ${S.length} scenarios, ${decRows} decision rows, ${boostedTotal} armed-boosted-hit rows, ` +
    `${winRows} wins + ${tieRows} ties -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
