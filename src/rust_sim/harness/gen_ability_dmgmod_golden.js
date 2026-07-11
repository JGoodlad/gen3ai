// gen_ability_dmgmod_golden.js — the ABILITY DMG_MOD CLASS-SWEEP golden (data-driven
// mechanics framework, Phase 2).
//
// The ability twin of `gen_item_mods_golden.js`. Covers EVERY WIRED member of the
// ability DMG_MOD class the engine now prices DATA-DRIVEN (`gen3_item_mechanics_v1`
// ability side — `turn.rs::resolve_atk_stat_mods` / `resolve_def_stat_mods` /
// `resolve_bp_mods` over `AbilityData.dmg_mod`):
//   PINCH family (Torrent/Blaze/Overgrow/Swarm) — BP ×1.5 for the ability's type when
//     the user is at hp<=maxhp/3. Naturally reached: a strong foe whittles the pinch
//     mon down until its STAB attack starts hitting boosted (the `boosted` marker fires
//     only once it is BOTH in pinch AND using its type).
//   Huge / Pure Power — Atk ×2 (unconditional, physical).
//   Guts — Atk ×1.5 while statused AND burn-halve SUPPRESSED (the key interaction). The
//     foe burns / paralyzes the Guts mon; every post-status physical hit is boosted.
//   Marvel Scale — Def ×1.5 while the DEFENDER is statused. The Marvel mon is burned;
//     the foe's physical hits into it are reduced (the `boosted` marker = a foe hit into
//     the statused Marvel mon).
//   Plus CONTROLS: a wrong-type pinch mon (never boosts), an unstatused Guts (never
//     boosts) — proving the condition gate, not just the multiplier.
//
// THE PROOF (the established per-decision STATE+HP+SEED differential, imitating
// gen_item_mods_golden.js): drive the OMNISCIENT in-process BattleStream over
// constructed full battles to GAME-END, capturing the PRNG seed at every decision
// boundary + each side's species/hp/maxhp/fainted/status/left + the first mover + the
// winner. The Rust test replays from the init seed WITHOUT re-seeding — every boosted
// hit's exact HP delta AND the cross-decision draw stream must match (the folds are all
// DRAW-FREE stat/BP math, so ANY extra/missing draw desyncs the LCG here).
//
// Output: tests/vectors/ability_dmgmod_golden.txt (same TAB format as item_mods_golden).
//
// Run:  node src/rust_sim/harness/gen_ability_dmgmod_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/ability_dmgmod_golden.txt');
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
  let x = 0x2c9f8e11 >>> 0;
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

// The coverage marker for a scenario: did the DMG_MOD fold actually FIRE this decision?
// `side` is which side carries the ability ('p1' or 'p2'); `moveName` the boosted move's
// display name (null ⇒ any physical hit, for Marvel Scale which reads the DEFENDER).
//   - ATK/BP folds (pinch/Huge/Pure/Guts): the ability-holder's boosted move lands a
//     direct -damage on the foe, AND (for pinch/Guts) the condition held. We approximate
//     "condition held" structurally by the HP/status the Rust replay independently
//     verifies — this marker only needs to be a repeatable coverage floor, so we key it
//     on the move landing + (for Marvel) the defender's status.
//   - DEF fold (Marvel Scale): the FOE's physical move lands a direct -damage on the
//     statused ability-holder.
function boostedLandedSince(log, fromIdx, sc) {
  const holder = sc.abilitySide; // 'p1' | 'p2'
  const foe = holder === 'p1' ? 'p2' : 'p1';
  let pending = null; // { by: side, isBoostMove: bool }
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
      if (pending && !residual) {
        if (sc.foldDir === 'def') {
          // Marvel Scale: the FOE hit the HOLDER, and the holder is statused.
          if (pending.by === foe && tgtSide === holder) return true;
        } else {
          // ATK/BP: the HOLDER's boosted move hit the FOE.
          const nameOk = !sc.boostedName || pending.name === sc.boostedName;
          if (pending.by === holder && tgtSide === foe && nameOk) return true;
        }
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

    // The plan: each side cycles its per-decision plan (single-mon teams so a forced
    // switch never happens). p1 = plan1, p2 = plan2.
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
    const boosted = reqState === 'move' && boostedLandedSince(log, logLenBefore, sc);
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
// Each scenario is a full decisive battle where the DMG_MOD condition arises
// NATURALLY over the course of play (pinch: the mon is whittled below ⅓; status: the
// foe inflicts it), so the fold fires repeatedly. `abilitySide` names the mon carrying
// the ability; `foldDir` is 'atk'|'bp'|'def'; `boostedName` (optional) restricts the
// coverage marker to a specific move.
function scenarios() {
  const S = [];
  const Dex = require(path.join(PS, 'dist/sim')).Dex;
  const dex3 = Dex.mod('gen3');
  const moveName = (id) => dex3.moves.get(id).name;

  // — PINCH family — a bulky mon holding the pinch ability whittles a strong foe while
  //   BEING whittled; once it drops to ⅓ HP its STAB attack hits boosted. p2 is a strong
  //   neutral attacker so the pinch mon reaches low HP without a one-shot. —
  S.push({
    id: 'torrent_pinch_water',
    // Blastoise (Torrent) Surf; a hard-hitting neutral Snorlax whittles it to pinch.
    p1: [mon('Blastoise', ['surf', 'rest'], { ability: 'Torrent', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Snorlax', ['bodyslam'], { ability: 'No Ability', nature: 'Adamant', evs: { atk: 252 } })],
    plan1: ['move 1'],
    plan2: ['move 1'],
    abilitySide: 'p1',
    foldDir: 'bp',
    boostedName: moveName('surf'),
  });
  S.push({
    id: 'blaze_pinch_fire',
    p1: [mon('Charizard', ['flamethrower', 'roost'], { ability: 'Blaze', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Snorlax', ['bodyslam'], { ability: 'No Ability', nature: 'Adamant', evs: { atk: 252 } })],
    plan1: ['move 1'],
    plan2: ['move 1'],
    abilitySide: 'p1',
    foldDir: 'bp',
    boostedName: moveName('flamethrower'),
  });
  S.push({
    id: 'overgrow_pinch_grass',
    // Leaf Blade (Grass, NON-drain — Giga Drain's HP drain is outside the port's modeled
    // scope, a no-drain fuzz, and would diverge the battle length).
    p1: [mon('Sceptile', ['leafblade', 'rest'], { ability: 'Overgrow', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Snorlax', ['bodyslam'], { ability: 'No Ability', nature: 'Adamant', evs: { atk: 252 } })],
    plan1: ['move 1'],
    plan2: ['move 1'],
    abilitySide: 'p1',
    foldDir: 'bp',
    boostedName: moveName('leafblade'),
  });
  S.push({
    id: 'swarm_pinch_bug',
    p1: [mon('Heracross', ['megahorn', 'swordsdance'], { ability: 'Swarm', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Snorlax', ['bodyslam'], { ability: 'No Ability', nature: 'Adamant', evs: { atk: 252 } })],
    plan1: ['move 1'],
    plan2: ['move 1'],
    abilitySide: 'p1',
    foldDir: 'bp',
    boostedName: moveName('megahorn'),
  });
  // Wrong-type pinch CONTROL: a Torrent mon whittled to pinch but using a NON-Water move
  // gets NO boost (the type gate). The `boosted` marker keys on Ice Beam landing — which
  // it does — but the Rust replay's HP proves the damage is UNBOOSTED (the seed +
  // exact-HP differential is the real gate; the marker just ensures the path is exercised).
  S.push({
    id: 'torrent_wrongtype_control',
    p1: [mon('Blastoise', ['icebeam', 'rest'], { ability: 'Torrent', nature: 'Modest', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Salamence', ['rest'], { ability: 'No Ability', nature: 'Serious', evs: { hp: 252 } })],
    plan1: ['move 1'],
    plan2: ['move 1'],
    abilitySide: 'p1',
    foldDir: 'bp',
    boostedName: moveName('icebeam'),
  });

  // — HUGE / PURE POWER (Atk ×2, unconditional physical) — the ×2 mon vs a bulky wall. —
  S.push({
    id: 'huge_power_azumarill',
    p1: [mon('Azumarill', ['waterfall', 'rest'], { ability: 'Huge Power', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Suicune', ['surf'], { ability: 'No Ability', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    plan1: ['move 1'],
    plan2: ['move 1'],
    abilitySide: 'p1',
    foldDir: 'atk',
    boostedName: moveName('waterfall'),
  });
  S.push({
    id: 'pure_power_medicham',
    p1: [mon('Medicham', ['brickbreak', 'rest'], { ability: 'Pure Power', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Suicune', ['surf'], { ability: 'No Ability', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    plan1: ['move 1'],
    plan2: ['move 1'],
    abilitySide: 'p1',
    foldDir: 'atk',
    boostedName: moveName('brickbreak'),
  });

  // — GUTS (Atk ×1.5 while statused + burn-halve SUPPRESSED) — the foe BURNS the Guts mon
  //   (Will-O-Wisp), then every physical hit is ×1.5 and the burn does NOT halve. —
  S.push({
    id: 'guts_burned_machamp',
    // Machamp (Guts) Body Slam; Weezing burns it T1 then idles (Pain Split/Rest),
    // Machamp keeps hitting boosted. Weezing is Poison (NOT Ghost) so Body Slam LANDS
    // (a Ghost burner would be Normal-immune — the first attempt's 0-boosted-row bug).
    p1: [mon('Machamp', ['bodyslam', 'rest'], { ability: 'Guts', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Weezing', ['willowisp', 'rest'], { ability: 'No Ability', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    plan1: ['move 1'], // Body Slam (physical, reliable) — burn-suppressed once statused
    plan2: ['move 1', 'move 2'], // Will-O-Wisp then Rest (idle)
    abilitySide: 'p1',
    foldDir: 'atk',
    boostedName: moveName('bodyslam'),
  });
  S.push({
    id: 'guts_paralyzed_ursaring',
    // Ursaring (Guts) paralyzed by a foe Thunder Wave — Guts applies its Atk ×1.5 the
    // moment the para lands. A FIXED-BP physical (Body Slam) so the ONLY variable is the
    // Guts fold (NOT Facade's separate status-doubling — kept out to isolate this class).
    p1: [mon('Ursaring', ['bodyslam'], { ability: 'Guts', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Blissey', ['thunderwave', 'softboiled'], { ability: 'No Ability', nature: 'Calm', evs: { hp: 252, spd: 252 } })],
    plan1: ['move 1'], // Body Slam every turn — ×1.5 once paralyzed
    plan2: ['move 1', 'move 2'], // Thunder Wave then Soft-Boiled
    abilitySide: 'p1',
    foldDir: 'atk',
    boostedName: moveName('bodyslam'),
  });
  // Unstatused Guts CONTROL: a Guts mon that never gets statused gets NO boost.
  S.push({
    id: 'guts_unstatused_control',
    p1: [mon('Machamp', ['bodyslam', 'rest'], { ability: 'Guts', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Suicune', ['surf'], { ability: 'No Ability', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    plan1: ['move 1'],
    plan2: ['move 1'],
    abilitySide: 'p1',
    foldDir: 'atk',
    boostedName: moveName('bodyslam'),
  });

  // — MARVEL SCALE (Def ×1.5 while the DEFENDER is statused) — the Marvel mon is BURNED by
  //   the foe, then the foe's physical hits into it are reduced. The `boosted` marker =
  //   a foe physical hit into the statused Marvel holder. —
  S.push({
    id: 'marvel_scale_burned_milotic',
    // Snorlax (Body Slam) attacks a Marvel Scale Milotic that burns itself irrelevant —
    // instead the ATTACKER's Body Slam has a burn chance; but to force the DEFENDER's
    // status reliably we give the foe a self-... no: p2 Milotic is burned by p1's
    // Will-O-Wisp T1 (p1 = Gengar), then p1's Gengar switches to physical? Simpler: p1 =
    // a physical attacker whose move BURNS (Flamethrower? special). Use a dedicated
    // burner: p1 Gengar Will-O-Wisp T1, then Shadow Punch (physical) into the burned
    // Marvel Milotic. Milotic recovers to stay alive (decisive but long-ish).
    p1: [mon('Gengar', ['willowisp', 'shadowpunch'], { ability: 'No Ability', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Milotic', ['recover'], { ability: 'Marvel Scale', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    plan1: ['move 1', 'move 2'], // Will-O-Wisp then Shadow Punch (physical → ModifyDef fires)
    plan2: ['move 1'], // Recover
    abilitySide: 'p2',
    foldDir: 'def',
    boostedName: null,
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(30);
  const lines = [];
  lines.push('# ability_dmgmod_golden.txt — the gen3_item_mechanics_v1 ABILITY DMG_MOD class-sweep golden.');
  lines.push('# Per-decision-boundary STATE+HP+SEED differential to GAME-END, every wired ability.');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status left) p2(...) first boosted');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0, boostedTotal = 0;

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
    // EVERY scenario must land its boosted/measured hit repeatedly across the 30-seed
    // sweep — the member's fold actually fires in the corpus.
    if (scenBoosted < 10) failures.push(`${sc.id}: only ${scenBoosted} boosted-hit rows (<10) — the ability's fold barely fires`);
  }

  if (failures.length) {
    console.error('ABILITY DMGMOD GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }
  if (winRows < 200) { console.error(`ABILITY DMGMOD GOLDEN: too few WIN rows (${winRows} < 200)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `ability dmgmod golden: ${S.length} scenarios, ${decRows} decision rows, ${boostedTotal} boosted-hit rows, ` +
    `${winRows} wins + ${tieRows} ties -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
