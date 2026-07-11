// gen_item_mods_golden.js — the ITEM-MODIFIER CLASS-SWEEP golden (data-driven
// mechanics framework, Phase 1).
//
// Covers EVERY member of the classes the engine now prices DATA-DRIVEN
// (`gen3_item_mechanics_v1` — `turn.rs::resolve_atk_stat_mods` / `resolve_def_stat_mods`
// / `resolve_bp_mods` over `ItemData`):
//   TYPE_BOOST  — all 24: the 18 stat-fold members (×1.1 family + Sea Incense ×1.05),
//                 the 2 gen2 bows (DIRECT ×1.1 base-power float), the 4 gen4-named
//                 incenses (×4915/4096 base-power chain — NOT ×1.1). Each holder
//                 ALTERNATES a matching-type move (boosted) with a non-matching
//                 control move (NOT boosted) — a wrong fold/multiplier/rounding
//                 diverges the per-decision HP on the first boosted hit; a boost
//                 leaking onto the control move diverges on the control hit.
//   SPECIES_STAT — all 6 (Thick Club, gen3-SpA-ONLY Light Ball, DeepSeaTooth,
//                 DeepSeaScale [def-side], Metal Powder [def-side], Soul Dew
//                 [both sides]), plus WRONG-SPECIES holder controls.
//   CHOICE      — Choice Band (the ×1.5 fold + the move lock, locked plan).
//
// THE PROOF (the established per-decision STATE+HP+SEED differential, imitating
// gen_taunt_disable_golden.js): drive the OMNISCIENT in-process BattleStream over
// constructed full battles to GAME-END, capturing the PRNG seed at every decision
// boundary + each side's species/hp/maxhp/fainted/status/left + the first mover +
// the winner. The Rust test replays from the init seed WITHOUT re-seeding — every
// boosted hit's exact HP delta AND the cross-decision draw stream must match.
//
// Output: tests/vectors/item_mods_golden.txt (TAB format, a simplified
// taunt_disable_golden minus the volatile columns, plus a boosted-hit marker).
//
// Run:  node src/rust_sim/harness/gen_item_mods_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/item_mods_golden.txt');
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
  let x = 0x51a3e9b7 >>> 0;
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

// Did p1's BOOSTED move (display name `boostedName`) LAND a direct (non-residual)
// -damage on p2 in this decision's log window? The coverage marker: the Rust side
// tallies these rows, guaranteeing every item's boosted fold actually fired.
function boostedLandedSince(log, fromIdx, boostedName) {
  let pending = false;
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    const tag = p[1];
    if (tag === 'move' && (p[2] || '').trim().startsWith('p1a:')) {
      pending = (p[3] || '') === boostedName;
    } else if (tag === '-damage' && (p[2] || '').trim().startsWith('p2a:')) {
      const residual = p.slice(4).some((x) => x.startsWith('[from]'));
      if (pending && !residual) return true;
    } else if (tag === '-miss' || tag === '-immune' || tag === 'faint') {
      if (pending && tag !== 'faint') pending = false;
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

    // The plan: p1 cycles its per-turn move plan (or is choice-locked to move 1);
    // p2 always uses its move 1. Forced switches never happen (single-mon teams).
    let choices;
    if (reqState === 'switch') {
      choices = { p1: force[0] ? 'switch 2' : null, p2: force[1] ? 'switch 2' : null };
    } else {
      const p1c = sc.plan[decisionNo % sc.plan.length];
      choices = { p1: p1c, p2: 'move 1' };
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
    const boosted = reqState === 'move' && boostedLandedSince(log, logLenBefore, sc.boostedName);
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
// Each: p1 = the item holder alternating [boosted move, control move] (CB: locked),
// p2 = a no-EV defender that attacks back (decisive real battles to game-end).
function scenarios() {
  const S = [];
  const Dex = require(path.join(PS, 'dist/sim')).Dex;
  const dex3 = Dex.mod('gen3');
  const moveName = (id) => dex3.moves.get(id).name;

  // A TYPE_BOOST member: holder alternates the matching move with a control move.
  const typeItem = (id, item, attacker, boostedMove, controlMove, defender) => {
    S.push({
      id,
      p1: [mon(attacker, [boostedMove, controlMove], { item, nature: 'Serious', evs: { atk: 252, spa: 252 } })],
      p2: [defender],
      plan: ['move 1', 'move 2'],
      boostedName: moveName(boostedMove),
    });
  };
  const wall = () => mon('Snorlax', ['bodyslam'], { nature: 'Serious' });
  const suicune = () => mon('Suicune', ['icebeam'], { nature: 'Serious' });

  // — the 17 gen3-mod ×1.1 stat-fold members + Sea Incense ×1.05 —
  typeItem('magnet_electric', 'magnet', 'Salamence', 'thunderbolt', 'strength', wall());
  typeItem('charcoal_fire', 'charcoal', 'Salamence', 'flamethrower', 'strength', wall());
  typeItem('mysticwater_water', 'mysticwater', 'Salamence', 'surf', 'strength', wall());
  typeItem('miracleseed_grass', 'miracleseed', 'Salamence', 'razorleaf', 'strength', wall());
  typeItem('nevermeltice_ice', 'nevermeltice', 'Salamence', 'icebeam', 'strength', wall());
  typeItem('blackbelt_fighting', 'blackbelt', 'Salamence', 'crosschop', 'strength', wall());
  typeItem('poisonbarb_poison', 'poisonbarb', 'Salamence', 'sludgebomb', 'strength', wall());
  typeItem('softsand_ground', 'softsand', 'Salamence', 'earthquake', 'strength', wall());
  typeItem('sharpbeak_flying', 'sharpbeak', 'Salamence', 'drillpeck', 'strength', wall());
  typeItem('twistedspoon_psychic', 'twistedspoon', 'Salamence', 'psychic', 'strength', wall());
  typeItem('silverpowder_bug', 'silverpowder', 'Salamence', 'signalbeam', 'strength', wall());
  typeItem('hardstone_rock', 'hardstone', 'Salamence', 'rockslide', 'strength', wall());
  // (Ghost vs the Snorlax wall is 0× — use Suicune so the boosted hit LANDS.)
  typeItem('spelltag_ghost', 'spelltag', 'Salamence', 'shadowball', 'strength', suicune());
  typeItem('dragonfang_dragon', 'dragonfang', 'Salamence', 'dragonclaw', 'strength', wall());
  typeItem('blackglasses_dark', 'blackglasses', 'Salamence', 'crunch', 'strength', wall());
  typeItem('metalcoat_steel', 'metalcoat', 'Salamence', 'irontail', 'strength', wall());
  typeItem('silkscarf_normal', 'silkscarf', 'Tauros', 'bodyslam', 'earthquake', suicune());
  typeItem('seaincense_water_105', 'seaincense', 'Salamence', 'surf', 'strength', wall());
  // — the gen2 bows: DIRECT ×1.1 base-power float (Body Slam BP 85 → floor 93) —
  typeItem('pinkbow_normal_direct', 'pinkbow', 'Snorlax', 'bodyslam', 'earthquake', suicune());
  typeItem('polkadotbow_normal_direct', 'polkadotbow', 'Tauros', 'bodyslam', 'earthquake', suicune());
  // — the 4 gen4-named incenses: ×4915/4096 base-power chain (NOT ×1.1) —
  typeItem('oddincense_psychic_12', 'oddincense', 'Salamence', 'psychic', 'strength', wall());
  typeItem('rockincense_rock_12', 'rockincense', 'Salamence', 'rockslide', 'strength', wall());
  typeItem('roseincense_grass_12', 'roseincense', 'Salamence', 'razorleaf', 'strength', wall());
  typeItem('waveincense_water_12', 'waveincense', 'Salamence', 'surf', 'strength', wall());

  // — SPECIES_STAT members (+ wrong-species controls). The def-side items sit on
  //   the DEFENDER (p2) — the "boosted" marker tracks p1's attacking move landing
  //   on the item holder, i.e. every marked row exercised the def fold. —
  S.push({
    id: 'thickclub_marowak_atk2',
    p1: [mon('Marowak', ['earthquake', 'rockslide'], { item: 'thickclub', nature: 'Adamant', evs: { atk: 252 } })],
    p2: [mon('Suicune', ['surf'], { nature: 'Serious' })],
    plan: ['move 1', 'move 2'],
    boostedName: moveName('earthquake'),
  });
  S.push({
    id: 'thickclub_wrong_species_control',
    p1: [mon('Snorlax', ['earthquake', 'bodyslam'], { item: 'thickclub', nature: 'Adamant', evs: { atk: 252 } })],
    p2: [mon('Suicune', ['surf'], { nature: 'Serious' })],
    plan: ['move 1', 'move 2'],
    boostedName: moveName('earthquake'),
  });
  S.push({
    id: 'lightball_pikachu_spa2_only',
    // gen3 Light Ball is SpA-ONLY: Thunderbolt (boosted) alternates with the
    // PHYSICAL Strength (must NOT be boosted — the gen4 Atk half doesn't exist).
    // A slow Snorlax defender lets frail Pikachu land several boosted bolts.
    p1: [mon('Pikachu', ['thunderbolt', 'strength'], { item: 'lightball', nature: 'Serious', evs: { hp: 252, spa: 252 } })],
    p2: [mon('Snorlax', ['bodyslam'], { nature: 'Serious' })],
    plan: ['move 1', 'move 2'],
    boostedName: moveName('thunderbolt'),
  });
  S.push({
    id: 'deepseatooth_clamperl_spa2',
    p1: [mon('Clamperl', ['surf', 'strength'], { item: 'deepseatooth', nature: 'Modest', evs: { spa: 252 } })],
    p2: [mon('Aerodactyl', ['rockslide'], { nature: 'Serious' })],
    plan: ['move 1', 'move 2'],
    boostedName: moveName('surf'),
  });
  S.push({
    id: 'deepseascale_clamperl_spd2_def',
    // The DEF-side fold: Alakazam's Psychic into a DeepSeaScale Clamperl (SpD ×2).
    p1: [mon('Alakazam', ['psychic'], { nature: 'Modest', evs: { spa: 252 } })],
    p2: [mon('Clamperl', ['surf'], { item: 'deepseascale', nature: 'Serious', evs: { hp: 252 } })],
    plan: ['move 1'],
    boostedName: moveName('psychic'),
  });
  S.push({
    id: 'metalpowder_ditto_def2_def',
    // Metal Powder Ditto (Def ×2, untransformed): Snorlax's Body Slam into it.
    p1: [mon('Snorlax', ['bodyslam'], { nature: 'Adamant', evs: { atk: 252 } })],
    p2: [mon('Ditto', ['strength'], { item: 'metalpowder', nature: 'Serious', evs: { hp: 252 } })],
    plan: ['move 1'],
    boostedName: moveName('bodyslam'),
  });
  S.push({
    id: 'deepseascale_wrong_species_def_control',
    p1: [mon('Alakazam', ['psychic'], { nature: 'Modest', evs: { spa: 252 } })],
    p2: [mon('Snorlax', ['bodyslam'], { item: 'deepseascale', nature: 'Serious' })],
    plan: ['move 1'],
    boostedName: moveName('psychic'),
  });
  S.push({
    id: 'souldew_latios_spa_and_spd',
    // Soul Dew both directions in ONE battle: Latios' Psychic (SpA ×1.5 out) vs
    // Suicune's Ice Beam (SpD ×1.5 in, SE vs the Dragon).
    p1: [mon('Latios', ['psychic', 'thunderbolt'], { item: 'souldew', nature: 'Modest', evs: { spa: 252 } })],
    p2: [mon('Suicune', ['icebeam'], { nature: 'Serious' })],
    plan: ['move 1', 'move 2'],
    boostedName: moveName('psychic'),
  });

  // — CHOICE: the Band's ×1.5 fold under its move LOCK (the plan stays on move 1;
  //   the engine's choice_locked_move makes move 2 illegal anyway). —
  S.push({
    id: 'choiceband_locked_15',
    p1: [mon('Aerodactyl', ['rockslide', 'earthquake'], { item: 'choiceband', nature: 'Adamant', evs: { atk: 252 } })],
    p2: [mon('Suicune', ['surf'], { nature: 'Serious' })],
    plan: ['move 1'],
    boostedName: moveName('rockslide'),
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(30);
  const lines = [];
  lines.push('# item_mods_golden.txt — the gen3_item_mechanics_v1 ITEM-MODIFIER class-sweep golden.');
  lines.push('# Per-decision-boundary STATE+HP+SEED differential to GAME-END, every wired item.');
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
    // EVERY scenario must land its boosted/measured hit repeatedly — the class
    // member's fold actually fires in the corpus (>=10 across the 30-seed sweep).
    if (scenBoosted < 10) failures.push(`${sc.id}: only ${scenBoosted} boosted-hit rows (<10) — the item's fold barely fires`);
  }

  if (failures.length) {
    console.error('ITEM MODS GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }
  if (winRows < 300) { console.error(`ITEM MODS GOLDEN: too few WIN rows (${winRows} < 300)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `item mods golden: ${S.length} scenarios, ${decRows} decision rows, ${boostedTotal} boosted-hit rows, ` +
    `${winRows} wins + ${tieRows} ties -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
