// gen_ability_batch2_golden.js — the ABILITY BATCH-2 CLASS-SWEEP golden
// (`gen3_ability_batch2_v1`): the DRAW-BEARING "reactive" ability classes + the block tail,
// each proven bit-for-bit over full battles to game-end.
//
//   CONTACT_PROC (Static par / Poison Point psn / Flame Body brn / Effect Spore slp|par|psn) —
//     an onDamagingHit that, when the HOLDER is hit by a CONTACT move, draws `randomChance(chance)`
//     (Static/PP/FB = randomChance(1,3); Effect Spore = randomChance(1,10) then a sample(3) on a
//     pass) and, on a pass, STATUSES THE ATTACKER. The contact-proc `randomChance` draws INSIDE
//     runEvent('DamagingHit') (gen<5) which is AFTER the move's OWN secondary random(100) — so a
//     contact move with a secondary (Body Slam par) shows [move secondary] THEN [contact proc] in
//     the draw stream. It does NOT fire behind a SUBSTITUTE (a sub-absorbed hit never reaches the
//     mon, so its onDamagingHit doesn't fire — the `!absorbed` gate, review-probe-verified) but DOES
//     fire on a KO. The ATTACKER's status timeline
//     (captured per-decision) proves the proc; ANY extra/missing/mis-ordered draw desyncs the seed.
//     Control: the SAME defender with a no-op ability (Insomnia) — the attacker stays un-statused
//     (the contact-proc draw is absent → the whole draw stream shifts).
//   CONTACT recoil (Rough Skin) — a DRAW-FREE onDamagingHit that deals baseMaxhp/16 recoil to the
//     ATTACKER on a contact hit. Observed via the attacker's HP (the recoil chip). Draw-free (the
//     seed stays identical to a no-op control — the STATE differs).
//   BLOCK — Damp (Explosion cancelled at TryMove: the user does NOT self-KO, the move draws NOTHING
//     — a big draw-count drop), Soundproof (a sound move — Sing / Grass Whistle / Roar — is immune:
//     accuracy drawn, then -immune, no status / no drag / no sample), Suction Cups (a phaze into
//     the holder draws its accuracy then -activate — NO drag sample; the holder stays).
//   SYNCHRONIZE — reflect a foe-inflicted major status back to the SOURCE (slp/frz exempt; tox→psn).
//     Draw-free in gen3customgame (this golden's format). Observed via the SOURCE's status timeline
//     (a Thunder Wave into a Synchronize holder → the caster ALSO gets paralyzed).
//
// THE PROOF (the established per-decision STATE+HP+**STATUS**+SEED differential): drive the
// OMNISCIENT in-process BattleStream over constructed full battles to GAME-END, capturing the PRNG
// seed at every decision boundary + each side's species/hp/maxhp/fainted/**status**/left + the
// first mover + the winner. The Rust test replays from the init seed WITHOUT re-seeding — every
// proc'd status (on the ATTACKER / the reflect SOURCE), every Rough-Skin recoil HP, every blocked
// Explosion (no self-KO), every immune sound move, every un-dragged phaze, AND the whole
// cross-decision draw stream must match (the contact-proc randomChance / Effect Spore sample are
// the ONLY new draws — ANY extra/missing/mis-ordered one desyncs the LCG here).
//
// Output: tests/vectors/ability_batch2_golden.txt (the batch1 TAB format).
//
// Run:  node src/rust_sim/harness/gen_ability_batch2_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/ability_batch2_golden.txt');
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

// 80 fixed seeds — enough for the contact-proc randomChance to PASS repeatedly (1/3 for
// Static/PP/FB, 1/10 for Effect Spore) so the proc'd-status timeline + its draw fire, and for the
// Effect-Spore sample to realize each of slp/par/psn.
const seeds = [];
{
  let s = 0x2b3c4d5e >>> 0;
  const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s; };
  for (let i = 0; i < 80; i++) seeds.push([rng() % 65536, rng() % 65536, rng() % 65536, rng() % 65536]);
}

function snap(side) {
  const a = side.active[0];
  return {
    species: a.species.name, hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted,
    status: a.status || '-', left: side.pokemonLeft,
    speBoost: a.boosts.spe || 0,
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

function firstMoverSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    const p = log[i].split('|');
    if ((p[1] === 'move' || p[1] === 'cant') && p.length >= 3) {
      const a = (p[2] || '').trim();
      if (a.startsWith('p1a:')) return 'p1';
      if (a.startsWith('p2a:')) return 'p2';
    }
  }
  return 'none';
}

// The per-scenario coverage marker: did THIS class's observable effect fire this decision?
//   contact_proc  — a `|-status|<attacker>|<st>|[from] ability: <Name>` line (the proc landed).
//   rough_skin    — a `|-damage|<attacker>|…|[from] ability: Rough Skin` line.
//   damp_block    — a `|cant|…|ability: Damp` line (the Explosion cancelled).
//   sound_immune  — a `|-immune|…|[from] ability: Soundproof` line.
//   suction_block — a `|-activate|…|ability: Suction Cups` line.
//   synchronize   — a `|-status|…|[from] ability: Synchronize` line.
function coverageMarker(log, fromIdx, sc) {
  const has = (re) => {
    for (let i = fromIdx; i < log.length; i++) if (re.test(log[i])) return true;
    return false;
  };
  switch (sc.cover) {
    case 'contact_proc': return has(/\|-status\|.*\|\[from\] ability: (Static|Poison Point|Flame Body|Effect Spore)/);
    case 'rough_skin': return has(/\|-damage\|.*\|\[from\] ability: Rough Skin/);
    case 'damp_block': return has(/\|cant\|.*\|ability: Damp/);
    case 'sound_immune': return has(/\|-immune\|.*\|\[from\] ability: Soundproof/);
    case 'suction_block': return has(/\|-activate\|.*\|ability: Suction Cups/);
    case 'synchronize': return has(/\|-status\|.*\|\[from\] ability: Synchronize/);
    default: return false;
  }
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

  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen, coverRows: 0 };

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
    const p1s = snap(battle.sides[0]);
    const p2s = snap(battle.sides[1]);
    const covered = reqState === 'move' && coverageMarker(log, logLenBefore, sc);
    if (covered) rec.coverRows++;
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: p1s,
      p2: p2s,
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      covered,
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

function encodeChoice(c) {
  if (!c) return '-';
  const m = c.match(/^move\s+(\d+)$/);
  if (m) return `m${Number(m[1]) - 1}`;
  const s = c.match(/^switch\s+(\d+)$/);
  if (s) return `s${Number(s[1]) - 1}`;
  throw new Error(`unencodable choice ${JSON.stringify(c)}`);
}

// ── Scenarios ────────────────────────────────────────────────────────────────
function scenarios() {
  const S = [];

  // ── CONTACT_PROC — the ATTACKER gets statused by a contact move into the holder ──
  // A bulky Body-Slam (contact + its OWN 30% par secondary) user hits the contact-proc holder
  // repeatedly; the proc statuses the ATTACKER (Body Slam CAN'T self-para its own par, but Static
  // paralyzes it; Poison Point poisons it; Flame Body burns it). The attacker's status timeline
  // proves the proc; the draw stream proves the position + count.
  //   Static: par the attacker. Use a Normal attacker (Snorlax) so Body Slam's own par CAN'T land
  //   on the Static holder in a way that confuses the ATTACKER timeline (we watch the ATTACKER).
  // A HARD-HITTING Body-Slam attacker vs a FRAIL contact-proc holder that chips back — the holder
  // is KO'd within a handful of turns (so the battle ENDS), and the proc fires on the contact hits
  // along the way. The attacker's status timeline (par/psn/brn/sample) proves the proc.
  S.push({
    id: 'static_paras_the_attacker',
    // p1 Snorlax Body Slams p2 Electabuzz (Static, frail) → Static paras Snorlax before Electabuzz
    // dies. Electabuzz Thunderbolts back so the battle progresses to a win.
    p1: [mon('Snorlax', ['bodyslam', 'earthquake'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Electabuzz', ['thunderbolt', 'thunderbolt'], { ability: 'Static', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p2', cover: 'contact_proc',
  });
  S.push({
    id: 'poisonpoint_poisons_the_attacker',
    // Tackle (contact, NO secondary) into a frail Poison Point holder (Nidoking) → the attacker
    // gets poisoned. Nidoking hits back with Ice Beam.
    p1: [mon('Snorlax', ['tackle', 'earthquake'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Nidoking', ['icebeam', 'icebeam'], { ability: 'Poison Point', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p2', cover: 'contact_proc',
  });
  S.push({
    id: 'flamebody_burns_the_attacker',
    // Body Slam into a frail Flame Body holder (Magmar) → the attacker gets burned (non-Fire
    // attacker). Magmar Fire Blasts back.
    p1: [mon('Snorlax', ['bodyslam', 'earthquake'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Magmar', ['fireblast', 'fireblast'], { ability: 'Flame Body', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p2', cover: 'contact_proc',
  });
  S.push({
    id: 'effectspore_samples_a_status',
    // Body Slam into a frail Effect Spore holder (Vileplume) → randomChance(1,10) then
    // sample(slp/par/psn). Vileplume Sludge Bombs back so the battle ends; over 80 seeds each of
    // slp/par/psn realizes at least once (the sample is exercised).
    p1: [mon('Snorlax', ['bodyslam', 'earthquake'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Vileplume', ['sludgebomb', 'sludgebomb'], { ability: 'Effect Spore', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p2', cover: 'contact_proc',
  });
  // CONTROL: the SAME Effect-Spore defender with a NO-OP ability (Insomnia) — the attacker stays
  // un-statused (the contact-proc draw is ABSENT, so the whole draw stream shifts vs the proc case).
  S.push({
    id: 'contact_proc_control_noop',
    p1: [mon('Snorlax', ['bodyslam', 'earthquake'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Vileplume', ['sludgebomb', 'sludgebomb'], { ability: 'Insomnia', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p2', cover: 'contact_proc',
  });
  // NON-CONTACT control: Earthquake (NOT contact) into a Static holder → NO proc, NO draw.
  S.push({
    id: 'noncontact_no_proc',
    p1: [mon('Snorlax', ['earthquake', 'bodyslam'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Electabuzz', ['thunderbolt', 'thunderbolt'], { ability: 'Static', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p2', cover: 'contact_proc',
  });

  // ── CONTACT recoil (Rough Skin) — the attacker takes baseMaxhp/16 recoil (DRAW-FREE) ──
  S.push({
    id: 'roughskin_recoils_the_attacker',
    // Body Slam into a Rough Skin holder → the attacker loses maxhp/16 each contact hit. Draw-free.
    // Sharpedo (a real Rough Skin mon) chips back with Surf so the battle ends.
    p1: [mon('Snorlax', ['bodyslam', 'earthquake'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    p2: [mon('Sharpedo', ['surf', 'surf'], { ability: 'Rough Skin', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p2', cover: 'rough_skin',
  });

  // ── BLOCK: Damp — Explosion is cancelled at TryMove; the user does NOT self-KO ──
  S.push({
    id: 'damp_blocks_explosion',
    // p1 Snorlax tries Explosion (move 1, CANCELLED by p2 Golduck's Damp — Snorlax does NOT faint),
    // then Body Slams (move 2, real damage) — so the Damp block is demonstrated every odd turn AND
    // the battle progresses to a win. Golduck (frail) Surfs back. The plan ALTERNATES so it ends.
    p1: [mon('Snorlax', ['explosion', 'bodyslam'], { nature: 'Adamant', evs: { atk: 252 } })],
    p2: [mon('Golduck', ['surf', 'surf'], { ability: 'Damp', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
    plan1: ['move 1', 'move 2'], plan2: ['move 1'],
    abilitySide: 'p2', cover: 'damp_block',
  });

  // ── BLOCK: Soundproof — a sound move (Sing) is immune ──
  S.push({
    id: 'soundproof_immune_to_sing',
    // p1 Jynx Sings (sound sleep) into p2 Electrode (Soundproof) → immune, no sleep. Both then
    // trade (Ice Beam / Thunderbolt) to a win so the battle ends.
    p1: [mon('Jynx', ['sing', 'icebeam'], { nature: 'Modest', evs: { spa: 252 } })],
    p2: [mon('Electrode', ['thunderbolt', 'rest'], { ability: 'Soundproof', nature: 'Timid', evs: { hp: 252, spe: 252 } })],
    plan1: ['move 1', 'move 2'], plan2: ['move 1'],
    abilitySide: 'p2', cover: 'sound_immune',
  });

  // ── BLOCK: Suction Cups — a Roar into the holder draws no sample; the holder stays ──
  S.push({
    id: 'suctioncups_blocks_roar',
    // p2 Suicune Roars (priority -6, into the up p1 Cradily [Suction Cups] which has a bench).
    // The drag is blocked (no sample); Cradily stays active. Both trade to a win.
    p1: [mon('Cradily', ['surf', 'rest'], { ability: 'Suction Cups', nature: 'Bold', evs: { hp: 252, def: 252 } }), mon('Snorlax', ['bodyslam', 'rest'])],
    p2: [mon('Suicune', ['roar', 'surf'], { nature: 'Bold', evs: { hp: 252, spa: 252 } })],
    plan1: ['move 1'], plan2: ['move 1'],
    abilitySide: 'p1', cover: 'suction_block',
  });

  // ── SYNCHRONIZE — reflect a foe status back to the source (draw-free in customgame) ──
  S.push({
    id: 'synchronize_reflects_paralysis',
    // p1 Jolteon Thunder Waves p2 Alakazam (Synchronize) → Alakazam is para'd AND Jolteon (the
    // source) is para'd too (reflected). Both then trade to a win.
    p1: [mon('Jolteon', ['thunderwave', 'thunderbolt'], { nature: 'Timid', evs: { spe: 252, spa: 252 } })],
    p2: [mon('Alakazam', ['psychic', 'recover'], { ability: 'Synchronize', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    plan1: ['move 1', 'move 2'], plan2: ['move 1'],
    abilitySide: 'p2', cover: 'synchronize',
  });
  S.push({
    id: 'synchronize_reflects_toxic_as_psn',
    // p1 Starmie Toxics p2 Alakazam (Synchronize) → Alakazam is badly poisoned AND Starmie (the
    // source, Water/Psychic → NOT Poison-immune) gets regular poison (tox→psn reflect). A Poison
    // Toxic user (Gengar) would be psn-IMMUNE to the reflect — so use a non-Poison caster. Both
    // trade to a win. (Toxic is 85% acc → the reflect realizes over 80 seeds.)
    p1: [mon('Starmie', ['toxic', 'surf'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    p2: [mon('Alakazam', ['psychic', 'recover'], { ability: 'Synchronize', nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    plan1: ['move 1', 'move 2'], plan2: ['move 1'],
    abilitySide: 'p2', cover: 'synchronize',
  });

  return S;
}

// ── Driver ────────────────────────────────────────────────────────────────────
(async () => {
  const S = scenarios();
  const lines = [];
  lines.push('# ability_batch2_golden (gen3_ability_batch2_v1) — CONTACT_PROC / recoil / BLOCK / SYNCHRONIZE');
  lines.push(`# format=${FORMAT} seeds=${seeds.length}`);
  lines.push('# SCEN <id> <abilitySide> <cover>');
  lines.push('# TEAM p1|p2 <packed>');
  lines.push('# INIT <seed4>');
  lines.push('# DEC <request> <fP1> <fP2> <cP1> <cP2> <seed4> ' +
    '<p1 species,hp,maxhp,fainted,status,left,speBoost> <p2 ...> <firstMover> <covered>');
  lines.push('# END <winner|none> <ended>');

  let totalDecisions = 0;
  let totalCover = 0;
  const coverById = {};
  const seenEffectSporeStatuses = new Set();

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}\t${sc.abilitySide}\t${sc.cover}`);
    lines.push(`TEAM\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\tp2\t${Teams.pack(sc.p2)}`);
    let scCover = 0;
    for (const seed of seeds) {
      const rec = await runBattle(sc, seed);
      if (!rec.initSeed || rec.decisions.length === 0) {
        throw new Error(`scenario ${sc.id} seed ${JSON.stringify(seed)} never reached a decision ` +
          `(decisions=${rec.decisions.length}, ended=${rec.ended}) — fix the plan/teams`);
      }
      // `getSeed()` returns a COMMA-STRING (e.g. "55250,62519,52978,42619"), not an array.
      lines.push(`INIT\t${rec.initSeed}`);
      for (const d of rec.decisions) {
        const p1 = d.p1, p2 = d.p2;
        const c1 = `${p1.species},${p1.hp},${p1.maxhp},${p1.fainted ? 1 : 0},${p1.status},${p1.left},${p1.speBoost}`;
        const c2 = `${p2.species},${p2.hp},${p2.maxhp},${p2.fainted ? 1 : 0},${p2.status},${p2.left},${p2.speBoost}`;
        lines.push(`DEC\t${d.request}\t${d.force[0] ? 1 : 0}\t${d.force[1] ? 1 : 0}\t${d.choiceP1}\t${d.choiceP2}\t` +
          `${d.seedAfter}\t${c1}\t${c2}\t${d.firstMover}\t${d.covered ? 1 : 0}`);
        totalDecisions++;
        if (d.covered) { totalCover++; scCover++; }
        // Track which Effect-Spore statuses realized (the sample exercise).
        if (sc.id === 'effectspore_samples_a_status') {
          if (p1.status !== '-') seenEffectSporeStatuses.add(p1.status);
        }
      }
      lines.push(`END\t${rec.winner || 'none'}\t${rec.ended ? 1 : 0}`);
    }
    coverById[sc.id] = scCover;
  }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  const crypto = require('crypto');
  const md5 = crypto.createHash('md5').update(lines.join('\n') + '\n').digest('hex');
  console.error(`wrote ${OUT}`);
  console.error(`scenarios=${S.length} decisions=${totalDecisions} coverRows=${totalCover} md5=${md5}`);
  console.error('per-scenario cover:', JSON.stringify(coverById));
  console.error('Effect-Spore sampled statuses:', JSON.stringify([...seenEffectSporeStatuses].sort()));

  // FAIL LOUD if a class did not realize its effect (the golden would be vacuous).
  const need = {
    static_paras_the_attacker: 'contact_proc',
    poisonpoint_poisons_the_attacker: 'contact_proc',
    flamebody_burns_the_attacker: 'contact_proc',
    effectspore_samples_a_status: 'contact_proc',
    roughskin_recoils_the_attacker: 'rough_skin',
    damp_blocks_explosion: 'damp_block',
    soundproof_immune_to_sing: 'sound_immune',
    suctioncups_blocks_roar: 'suction_block',
    synchronize_reflects_paralysis: 'synchronize',
    synchronize_reflects_toxic_as_psn: 'synchronize',
  };
  const failures = [];
  for (const [id, _] of Object.entries(need)) {
    if (!(coverById[id] > 0)) failures.push(`${id}: 0 cover rows (effect never realized)`);
  }
  // The non-contact + no-op controls MUST have 0 contact-proc cover (the proof they don't proc).
  if (coverById.noncontact_no_proc !== 0) failures.push('noncontact_no_proc: should have 0 contact-proc cover (EQ is non-contact)');
  if (coverById.contact_proc_control_noop !== 0) failures.push('contact_proc_control_noop: should have 0 contact-proc cover (Insomnia is a no-op)');
  if (failures.length) {
    console.error('\nFAIL — a class did not realize its effect:\n  ' + failures.join('\n  '));
    process.exit(1);
  }
  console.error('OK — every batch-2 class realized its effect + the controls stayed inert.');
})();
