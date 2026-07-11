// gen_accuracy_golden.js — the ACCURACY pipeline CLASS-SWEEP golden
// (`gen3_accuracy_pipeline_v1`).
//
// The to-hit twin of `gen_ability_dmgmod_golden.js`. This phase is DRAW-RELEVANT
// (unlike the P1/P2 stat/BP folds): a wrong effective accuracy → a hit/miss FLIP →
// the accuracy `randomChance` is drawn, then crit/damage draws follow ONLY on a hit,
// so the SEED diverges. The exact gen3 to-hit MATH **and DRAW-ORDER** must be
// bit-for-bit — the SEED at every decision boundary is the teeth here.
//
// Covers the wired accuracy classes the engine now folds into `turn.rs::
// effective_accuracy` (`ItemData.acc_mod` / `AbilityData.acc_mod` + the acc/eva
// STAGE TABLE):
//   STAGE FOLD  — Mud-Slap (a 100%-accuracy-drop secondary) lowers the FOE's accuracy
//     each hit; the foe's own attacks then roll at 95×(3/4)^n. The realized hit-RATE
//     across seeds must match the sim (the fuzzer's exact acc-stage cluster).
//   ACCURACY_ITEM — Bright Powder (×0.9) / Lax Incense (×0.95) on the DEFENDER: the
//     attacker's shaky move misses more often.
//   ACCURACY ability — Compound Eyes (×1.3 attacker) makes a shaky move land more;
//     Sand Veil (×0.8 defender in sand) makes the attacker miss more; Hustle (×0.8
//     attacker, physical-TYPE move) drops a physical hit's accuracy — with a SPECIAL
//     control (Hustle unaffected on a special-type move).
//   No-mod CONTROLS — the same shaky move with NO item/ability/stage (the empty path,
//     byte-identical to the pre-pipeline raw roll).
//
// THE PROOF (the established per-decision STATE+HP+SEED differential): drive the
// OMNISCIENT in-process BattleStream over constructed full battles to GAME-END,
// capturing the PRNG seed at every decision boundary + each side's species/hp/maxhp/
// fainted/status/left + the first mover + the winner. The Rust test replays from the
// init seed WITHOUT re-seeding — every hit/miss AND the cross-decision draw stream must
// match. A single mis-computed effAcc flips a roll and desyncs the LCG here.
//
// Output: tests/vectors/accuracy_golden.txt (same TAB format as ability_dmgmod_golden).
//
// Run:  node src/rust_sim/harness/gen_accuracy_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/accuracy_golden.txt');
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
  let x = 0x51ed270b >>> 0; // a distinct stream from the item/ability goldens
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

// A -miss on any side, since the last decision boundary — the coverage signal the
// accuracy classes are meant to produce (a modified effAcc flips some rolls to misses).
function missSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    if (p[1] === '-miss') return true;
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

  // Coverage: hook runEvent('ModifyAccuracy') — an accMod handler FIRED this decision if
  // it changed the relayVar (in != out). This proves the accMod class is exercised
  // (distinct from a stage, which shows up as a -miss / hit-rate shift).
  const battle = stream.battle;
  let accModFiredThisDecision = false;
  const realRun = battle.runEvent.bind(battle);
  battle.runEvent = function (eventid, target, source, effect, relayVar, ...rest) {
    const out = realRun(eventid, target, source, effect, relayVar, ...rest);
    if (eventid === 'ModifyAccuracy' && typeof relayVar === 'number' && typeof out === 'number' && out !== relayVar) {
      accModFiredThisDecision = true;
    }
    return out;
  };

  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: battle.gen, accModRows: 0, missRows: 0 };

  let decisionNo = 0;
  let safety = 0;
  while (!battle.ended && safety < 400) {
    safety++;
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
    accModFiredThisDecision = false;
    if (choices.p1) { try { streams.omniscient.write(`>p1 ${choices.p1}`); } catch (e) {} }
    if (choices.p2) { try { streams.omniscient.write(`>p2 ${choices.p2}`); } catch (e) {} }
    for (let i = 0; i < 16; i++) await tick();

    const seedAfter = battle.prng.getSeed();
    if (seedAfter === seedBefore && log.length === logLenBefore && battle.requestState === reqState) {
      throw new Error(`STALL: choice ${JSON.stringify(choices)} did not advance the battle ` +
        `(scenario ${sc.id}, decision ${decisionNo}) — the sim rejected it. Fix the plan.`);
    }
    const missed = reqState === 'move' && missSince(log, logLenBefore);
    const accMod = reqState === 'move' && accModFiredThisDecision;
    if (missed) rec.missRows++;
    if (accMod) rec.accModRows++;
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      missed,
      accMod,
    });
    decisionNo++;
  }

  rec.ended = !!battle.ended;
  rec.winner = battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

// ── Scenarios ────────────────────────────────────────────────────────────────
// Each scenario is a full decisive battle where an accuracy modifier is in effect,
// producing a hit-rate shift the SEED differential pins bit-for-bit. `covers` names
// the marker to require ≥N of ('miss' | 'accMod').
function scenarios() {
  const S = [];

  // — STAGE FOLD: Muddy Water (bp 95, acc 85, 30% accuracy-drop secondary) both KOs the
  //   foe (a decisive strong move, PP-safe at 12 PP) AND stacks a −1 accuracy stage on it
  //   30% of the time — the foe (Cross Chop, acc 80) then rolls at 80×(3/4)^n, missing more
  //   as its accuracy erodes. The fuzzer's acc-stage cluster, end-to-end + decisive. —
  S.push({
    id: 'muddywater_stage_drop',
    // Swampert (Water/Ground) Muddy Water into a Salamence: strong hits KO in a few turns
    // AND drop Salamence's accuracy; Salamence Cross Chop (acc 80) whiffs more each drop.
    p1: [mon('Swampert', ['muddywater', 'rest'], { ability: 'No Ability', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Salamence', ['crosschop', 'rest'], { ability: 'No Ability', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], // Muddy Water (12 PP — plenty for a decisive battle)
    plan2: ['move 1'], // Cross Chop (acc 80) — erodes with each Muddy Water acc-drop
    covers: 'miss',
  });

  // — ACCURACY_ITEM: Bright Powder on the DEFENDER makes the attacker's shaky move miss
  //   more (95×0.9 = 85.5, and a lower-acc move drops further). —
  S.push({
    id: 'brightpowder_defender',
    // p1 Tauros Cross Chop (acc 80) into a Bright Powder Snorlax: 80×0.9 = 72 → ~28% miss.
    p1: [mon('Tauros', ['crosschop', 'rest'], { ability: 'No Ability', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'rest'], { ability: 'No Ability', item: 'brightpowder', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'],
    plan2: ['move 1'],
    covers: 'accMod',
  });
  // Lax Incense variant (×0.95).
  S.push({
    id: 'laxincense_defender',
    p1: [mon('Tauros', ['crosschop', 'rest'], { ability: 'No Ability', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'rest'], { ability: 'No Ability', item: 'laxincense', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'],
    plan2: ['move 1'],
    covers: 'accMod',
  });

  // — ACCURACY ability: Compound Eyes on the ATTACKER lifts a shaky move (70×1.3 = 91). —
  S.push({
    id: 'compoundeyes_attacker',
    // Butterfree (Compound Eyes) Thunder (acc 70 → 91) into a bulky Blissey.
    p1: [mon('Butterfree', ['thunder', 'rest'], { ability: 'Compound Eyes', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Blissey', ['softboiled', 'seismictoss'], { ability: 'No Ability', nature: 'Calm', evs: { hp: 252, spd: 252 } })],
    plan1: ['move 1'],
    plan2: ['move 2', 'move 1'], // Seismic Toss then Soft-Boiled (chip + heal → long)
    covers: 'accMod',
  });

  // — Sand Veil on the DEFENDER in sand (×0.8): the attacker misses more. Sand from a
  //   Tyranitar lead (Sand Stream) so the weather is permanent. —
  S.push({
    id: 'sandveil_defender_sand',
    // Tyranitar (Sand Stream) sets sand; p2 Cacturne (Sand Veil) is hit by Tyranitar's
    // Cross Chop (acc 80 → 80×0.8 = 64 → ~36% miss). Cacturne idles (Rest) to stay alive.
    p1: [mon('Tyranitar', ['crosschop', 'rest'], { ability: 'Sand Stream', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Cacturne', ['rest', 'spikes'], { ability: 'Sand Veil', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    plan1: ['move 1'], // Cross Chop into the Sand-Veil mon
    plan2: ['move 1'], // Rest (idle survivor)
    covers: 'accMod',
  });

  // — Hustle on the ATTACKER: a PHYSICAL-type move drops (Cross Chop 80×0.8 = 64), with a
  //   SPECIAL-type-move control (Hustle does NOT touch an Electric/Water/etc. special). —
  S.push({
    id: 'hustle_physical',
    // Nincada? No — use a mon that gets Hustle + a physical-type move. Delibird has Hustle;
    // give it Cross Chop (Fighting, a physical-type in the gen3 list). Into a bulky wall.
    p1: [mon('Delibird', ['crosschop', 'rest'], { ability: 'Hustle', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'rest'], { ability: 'No Ability', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'], // Cross Chop (Fighting = physical-type) → ×0.8 acc
    plan2: ['move 1'],
    covers: 'accMod',
  });

  // — NO-MOD CONTROLS (the empty path): the same shaky move with no item/ability/stage.
  //   Must be byte-identical to the pre-pipeline raw roll (0 accMod rows expected). —
  S.push({
    id: 'nomod_control_crosschop',
    p1: [mon('Tauros', ['crosschop', 'rest'], { ability: 'No Ability', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Snorlax', ['bodyslam', 'rest'], { ability: 'No Ability', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['move 1'],
    plan2: ['move 1'],
    covers: 'none',
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(30);
  const lines = [];
  lines.push('# accuracy_golden.txt — the gen3_accuracy_pipeline_v1 ACCURACY class-sweep golden.');
  lines.push('# Per-decision-boundary STATE+HP+SEED differential to GAME-END, every wired acc class.');
  lines.push('# DRAW-RELEVANT: a hit/miss flip desyncs the SEED (the accuracy randomChance is drawn,');
  lines.push('# then crit/damage draws follow ONLY on a hit). The seedAfter column is the teeth.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status left) p2(...) first miss accMod');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0, missTotal = 0, accModTotal = 0;

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);

    let scenDecs = 0, scenMiss = 0, scenAccMod = 0;
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
          sp(d.p1), sp(d.p2), d.firstMover, d.missed ? 1 : 0, d.accMod ? 1 : 0,
        ].join('\t'));
        decRows++; scenDecs++;
        if (d.missed) { missTotal++; scenMiss++; }
        if (d.accMod) { accModTotal++; scenAccMod++; }
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
    // Coverage floor per the scenario's declared marker.
    if (sc.covers === 'miss' && scenMiss < 10) {
      failures.push(`${sc.id}: only ${scenMiss} miss rows (<10) — the stage fold barely produces misses`);
    }
    if (sc.covers === 'accMod' && scenAccMod < 10) {
      failures.push(`${sc.id}: only ${scenAccMod} accMod rows (<10) — the acc item/ability fold barely fires`);
    }
    if (sc.covers === 'none' && scenAccMod > 0) {
      failures.push(`${sc.id}: control produced ${scenAccMod} accMod rows (>0) — a no-mod scenario must not fire ModifyAccuracy`);
    }
  }

  if (failures.length) {
    console.error('ACCURACY GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }
  if (winRows < 150) { console.error(`ACCURACY GOLDEN: too few WIN rows (${winRows} < 150)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `accuracy golden: ${S.length} scenarios, ${decRows} decision rows, ${missTotal} miss rows, ` +
    `${accModTotal} accMod rows, ${winRows} wins + ${tieRows} ties -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
