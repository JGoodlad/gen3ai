// gen_statusimmune_golden.js — the STATUS_IMMUNE ability CLASS-SWEEP golden
// (`gen3_status_immune_v1`).
//
// The gen-3 abilities that grant immunity to a specific MAJOR status. PROBE-settled membership +
// draw model (`harness/probe_statusimmune_*.js`, drift-gated by dump_gen3_mechanics.js --check):
//   Limber (par) / Insomnia + Vital Spirit (slp) / Immunity (psn,tox) / Water Veil (brn) block
//     via `onSetStatus` (phase=setStatus, INSIDE runEvent('SetStatus'), after the clause shuffle);
//   Magma Armor (frz) blocks via `onImmunity` (phase=immunity, at runStatusImmunity BEFORE the
//     SetStatus event).
// In gen3customgame (this golden's format) EVERY member is DRAW-FREE — the immune mon simply is
// NOT statused, and the per-decision seed matches a status-lands battle bit-for-bit at the
// application point (a spurious block draw would desync).
//
// The immunity is made OBSERVABLE on the existing ACTIVE-mon STATUS timeline: a foe FIRES the
// matching status move at the immune holder EVERY relevant turn; the immune mon STAYS `-`
// (unstatused) all battle, while a NON-immune CONTROL (a plain-ability twin on the IDENTICAL
// plan/teams) gets STATUSED → its trajectory DIVERGES (a paralyzed/asleep mon acts differently,
// a burned/poisoned/toxic'd mon takes DoT and often LOSES). So `ok` on the immune scenario's
// timeline is 100% the ability block, and the control's persistent status is the clean
// discriminator. WRONG-STATUS controls (a Limber mon takes a BURN normally) prove the block is
// status-SPECIFIC, not a blanket status wall.
//
//   COVERS (each a DECISIVE full battle; the foe status-moves the immune holder in-engine):
//     si_limber_par        — Thunder Wave into Limber Snorlax: STAYS unparalyzed all battle.
//     si_insomnia_slp      — Spore into Insomnia Snorlax: STAYS awake.
//     si_vitalspirit_slp   — Hypnosis into Vital Spirit Snorlax: STAYS awake.
//     si_immunity_tox      — Toxic into Immunity Snorlax: STAYS un-poisoned.
//     si_waterveil_brn     — Will-O-Wisp into Water Veil Snorlax: STAYS unburned.
//     si_magmaarmor_frz    — Ice Beam (10% frz secondary) into Magma Armor Snorlax: NEVER frozen
//                            (the immunity-phase block; the secondary random(100) STILL draws).
//     si_control_par_none  — the SAME si_limber_par plan/teams but a NO-ability Snorlax: it gets
//                            PARALYZED (the discriminator; the battle diverges).
//     si_control_slp_none  — the SAME si_insomnia_slp plan but a no-ability Snorlax: it SLEEPS.
//     si_control_tox_none  — the SAME si_immunity_tox plan but no-ability: it gets TOXIC'd.
//     si_limber_takes_brn  — a Limber Snorlax hit by Will-O-Wisp: BURNS (Limber blocks par ONLY).
//     si_immunity_takes_brn— an Immunity Snorlax hit by Will-O-Wisp: BURNS (blocks psn/tox ONLY).
//
// Output: tests/vectors/statusimmune_golden.txt (same TAB format as naturalcure_golden).
//
// Run:  node src/rust_sim/harness/gen_statusimmune_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/statusimmune_golden.txt');
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
  let x = 0x51ed_2a13 >>> 0;
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

// Coverage marker: did the target ability BLOCK a status this decision? Read the sim-truth
// protocol `|-immune|<mon>|[from] ability: <A>` line (the setStatus/immunity block's `-immune`
// reveal for Limber/Insomnia/Immunity/Water Veil/Vital Spirit/Magma Armor). This just COUNTS the
// block events for the coverage floor; the real STATE gate is the immune mon's status column
// staying `-` (asserted bit-for-bit by the Rust replay).
const IMMUNE_ABILITY_NAMES = ['Limber', 'Insomnia', 'Immunity', 'Water Veil', 'Vital Spirit', 'Magma Armor'];
function immuneBlockedSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    const l = log[i];
    if (l.includes('|-immune|') && IMMUNE_ABILITY_NAMES.some((n) => l.includes(`[from] ability: ${n}`))) return true;
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

  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen, blockRows: 0 };

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
    const blocked = immuneBlockedSince(log, logLenBefore);
    if (blocked) rec.blockRows++;
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      blocked,
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

// ── Scenarios ────────────────────────────────────────────────────────────────
// p1 = the immune holder (or a control twin). It simply ATTACKS the foe with Seismic Toss (a
// FIXED 100 chip — draw-clean: no crit/damage roll, no status secondary — so the holder's own
// move never confounds the foe's status column). p2 = a mid-bulk foe that RE-FIRES the matching
// STATUS move at the holder EVERY turn (foePlan = always slot 0). The immune holder STAYS `-` all
// battle and eventually Seismic-Tosses the foe down (WIN); a NON-immune control gets STATUSED →
// its trajectory diverges (a paralyzed/asleep mon acts differently; a burned/poisoned mon takes
// DoT). The holder does NOT Rest (a self-Rest sleep would confound the "stays clean" invariant).
//
// The holder is **Snorlax** (Normal — no par/frz/psn/brn type-immunity, so the ONLY status block
// is the ability under test). The foe is **Miltank** (a mid-bulk Normal, ~523 HP) so it survives
// the ~5 Seismic Tosses needed to re-fire the status move enough for the block coverage floor,
// then dies (the battle ENDS with a p1 win). Miltank's ability is Scrappy? No — a modeled no-op:
// Thick Fat is modeled but affects only Ice/Fire; the foe takes only Normal Seismic Toss (typeless
// fixed damage), so Thick Fat is inert. Use Scrappy? not modeled. Use **Sturdy** (modeled no-op).
function foeMiltank(statusMove) {
  return [mon('Miltank', [statusMove, 'bodyslam'], { ability: 'Sturdy', nature: 'Careful', evs: { hp: 252, spd: 252, def: 4 } })];
}

function holderPlan() {
  // holder: Seismic Toss the foe every turn (a fixed-100 chip; ~5-6 turns to KO Miltank). NON-
  // cyclic (clamped to Seismic Toss, 16 PP). No Rest → the holder's OWN status column is never
  // self-set. On the immune lines the holder acts every turn (never statused) → ~6 turns to KO.
  return ['move 1'];
}
function foePlan() {
  // foe: RE-FIRE the STATUS move (slot 0) for the first 6 turns — enough re-fires to clear the
  // block coverage floor (>=10 across 40 seeds) — THEN Body Slam (slot 1) so the battle ENDS even
  // when the control is fully crippled (a frozen/asleep control can't attack, so the FOE must be
  // able to win those lines; the immune holder is already Seismic-Tossing the foe down by then).
  // NON-cyclic (clamps to Body Slam).
  return ['move 1', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1', 'move 2'];
}

function holderTeam(ability) {
  // Snorlax: Seismic Toss (fixed-100 chip to kill the foe) + Body Slam (unused; a 2-move set).
  // Immune ability under test. Bulky so it outlasts the foe's Body Slam chip on the control lines.
  return [mon('Snorlax', ['seismictoss', 'bodyslam'], { ability, nature: 'Careful', evs: { hp: 252, spd: 128, def: 128 } })];
}

function scenarios() {
  const S = [];
  // Each immune scenario: the foe RE-FIRES the matching status move at the immune Snorlax every
  // turn. The immune Snorlax Seismic-Tosses the foe down. The status column MUST stay `-`.
  S.push({ id: 'si_limber_par', p1: holderTeam('Limber'), p2: foeMiltank('thunderwave'), plan1: holderPlan(), plan2: foePlan() });
  S.push({ id: 'si_insomnia_slp', p1: holderTeam('Insomnia'), p2: foeMiltank('spore'), plan1: holderPlan(), plan2: foePlan() });
  S.push({ id: 'si_vitalspirit_slp', p1: holderTeam('Vital Spirit'), p2: foeMiltank('hypnosis'), plan1: holderPlan(), plan2: foePlan() });
  S.push({ id: 'si_immunity_tox', p1: holderTeam('Immunity'), p2: foeMiltank('toxic'), plan1: holderPlan(), plan2: foePlan() });
  S.push({ id: 'si_waterveil_brn', p1: holderTeam('Water Veil'), p2: foeMiltank('willowisp'), plan1: holderPlan(), plan2: foePlan() });
  // Magma Armor: the foe uses Ice Beam (10% frz secondary) — the ONLY freeze source in gen3.
  // Snorlax is Normal (Ice Beam neutral, no type-immunity), so the ONLY frz block is Magma Armor.
  // NOTE: a SECONDARY-freeze block is SILENT (no `-immune` line — `setStatus` emits `-immune`
  // only for a top-level `move.status`, which Ice Beam's frz is NOT), so the block COVERAGE is
  // proven by the STATE differential vs the paired `si_control_frz_none` (which DOES freeze on
  // some seeds) rather than by a `-immune` marker.
  S.push({ id: 'si_magmaarmor_frz', p1: holderTeam('Magma Armor'), p2: foeMiltank('icebeam'), plan1: holderPlan(), plan2: foePlan() });

  // CONTROLS (a NO-ability twin on the IDENTICAL plan/teams — it gets STATUSED, timeline diverges):
  S.push({ id: 'si_control_par_none', p1: holderTeam('No Ability'), p2: foeMiltank('thunderwave'), plan1: holderPlan(), plan2: foePlan() });
  S.push({ id: 'si_control_slp_none', p1: holderTeam('No Ability'), p2: foeMiltank('spore'), plan1: holderPlan(), plan2: foePlan() });
  S.push({ id: 'si_control_tox_none', p1: holderTeam('No Ability'), p2: foeMiltank('toxic'), plan1: holderPlan(), plan2: foePlan() });
  // The Magma Armor STATE discriminator: a NO-ability Snorlax vs the SAME Ice-Beam foe FREEZES
  // on some seeds (the block coverage proof — the immune holder freezes on NONE).
  S.push({ id: 'si_control_frz_none', p1: holderTeam('No Ability'), p2: foeMiltank('icebeam'), plan1: holderPlan(), plan2: foePlan() });

  // WRONG-STATUS controls (an immune ability does NOT block a DIFFERENT status → it lands):
  S.push({ id: 'si_limber_takes_brn', p1: holderTeam('Limber'), p2: foeMiltank('willowisp'), plan1: holderPlan(), plan2: foePlan() });
  S.push({ id: 'si_immunity_takes_brn', p1: holderTeam('Immunity'), p2: foeMiltank('willowisp'), plan1: holderPlan(), plan2: foePlan() });
  return S;
}

async function main() {
  const seeds = buildSeeds(40);
  const lines = [];
  lines.push('# statusimmune_golden.txt — the gen3_status_immune_v1 STATUS_IMMUNE class-sweep golden.');
  lines.push('# Per-decision-boundary STATE+HP+STATUS+SEED differential to GAME-END: the block is');
  lines.push('# observable on the ACTIVE-STATUS timeline (an immune mon status-moved every turn STAYS');
  lines.push('# `-`), vs a non-immune control that gets STATUSED. In gen3customgame the block is');
  lines.push('# DRAW-FREE (the per-decision seed matches a status-lands battle bit-for-bit at the');
  lines.push('# application point). `blocked`=an immunity -immune fired this decision (coverage marker).');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status left) p2(...) first blocked');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0, blockTotal = 0;
  // The `setStatus`-phase immune holders MUST land >=N `-immune` block events (the status MOVE
  // sets `move.status`, so the block emits `-immune`). Magma Armor is NOT here — a secondary-frz
  // block is SILENT (no `-immune`), so it is proven by the STATE differential (below), not a
  // block-marker floor. The controls must NEVER block.
  const blockScenarios = new Set([
    'si_limber_par', 'si_insomnia_slp', 'si_vitalspirit_slp', 'si_immunity_tox', 'si_waterveil_brn',
  ]);
  const noBlockScenarios = new Set([
    'si_control_par_none', 'si_control_slp_none', 'si_control_tox_none', 'si_control_frz_none',
    'si_limber_takes_brn', 'si_immunity_takes_brn', 'si_magmaarmor_frz',
  ]);
  // Immune scenarios whose holder STATUS column must be `-` at EVERY decision (the STATE proof) —
  // ALL 6 members incl. Magma Armor (Snorlax never frozen since MA blocks it).
  const mustStayClean = new Set([
    'si_limber_par', 'si_insomnia_slp', 'si_vitalspirit_slp', 'si_immunity_tox', 'si_waterveil_brn',
    'si_magmaarmor_frz',
  ]);
  // Wrong-status controls whose holder MUST become burned at some point (the block is specific).
  const mustGetBurned = new Set(['si_limber_takes_brn', 'si_immunity_takes_brn']);
  // The Magma Armor frz discriminator: the NO-ability control MUST freeze on >=1 seed (proving
  // the Ice Beam CAN freeze this target — so the immune holder staying frz-free is the block).
  let frzControlFroze = false;

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);

    let scenDecs = 0;
    let scenBlocks = 0;
    let anyBurn = false;
    let cleanViolation = null;
    for (const seed of seeds) {
      let rec;
      try { rec = await runBattle(sc, seed); } catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); continue; }
      if (rec.gen !== 3) { failures.push(`${sc.id}: expected gen 3, got ${rec.gen}`); break; }
      if (!rec.initSeed || rec.decisions.length === 0) { failures.push(`${sc.id} seed ${seed}: no decisions`); continue; }
      const seedStr = seed.join(',');
      lines.push(['INIT', sc.id, rec.initSeed, seedStr].join('\t'));

      rec.decisions.forEach((d) => {
        // Track the immune-scenario STATE invariant: the holder (p1) status column must be `-`.
        if (mustStayClean.has(sc.id) && d.p1.status !== '-' && !cleanViolation) {
          cleanViolation = `${sc.id} seed ${seedStr}: immune holder became status='${d.p1.status}'`;
        }
        if (mustGetBurned.has(sc.id) && d.p1.status === 'brn') anyBurn = true;
        // The frz-control discriminator: the NO-ability Snorlax freezes on some seeds.
        if (sc.id === 'si_control_frz_none' && d.p1.status === 'frz') frzControlFroze = true;
        const sp = (s) => [s.species, s.hp, s.maxhp, s.fainted ? 1 : 0, s.status, s.left].join('\t');
        lines.push([
          'DEC', sc.id, seedStr, d.request, d.force[0] ? 1 : 0, d.force[1] ? 1 : 0,
          d.choiceP1, d.choiceP2, d.seedAfter,
          sp(d.p1), sp(d.p2), d.firstMover, d.blocked ? 1 : 0,
        ].join('\t'));
        decRows++; scenDecs++;
        if (d.blocked) { blockTotal++; scenBlocks++; }
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
    if (blockScenarios.has(sc.id) && scenBlocks < 10) {
      failures.push(`${sc.id}: only ${scenBlocks} immunity-block rows (<10) — the block barely fires`);
    }
    if (noBlockScenarios.has(sc.id) && scenBlocks > 0) {
      failures.push(`${sc.id}: expected 0 immunity-block rows (a control), got ${scenBlocks}`);
    }
    if (cleanViolation) failures.push(cleanViolation);
    if (mustGetBurned.has(sc.id) && !anyBurn) {
      failures.push(`${sc.id}: the wrong-status control NEVER burned — the block is over-broad or the plan is wrong`);
    }
  }

  // The Magma Armor STATE-discriminator gate: the NO-ability control MUST have frozen on >=1
  // seed (else Ice Beam can't freeze this target at all → the immune scenario proves nothing).
  if (!frzControlFroze) {
    failures.push('si_control_frz_none: the NO-ability control NEVER froze across 40 seeds — ' +
      'Ice Beam cannot freeze the target, so the Magma Armor block is unproven. Widen the seeds.');
  }

  if (failures.length) {
    console.error('STATUS_IMMUNE GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }
  if (winRows < 120) { console.error(`STATUS_IMMUNE GOLDEN: too few WIN rows (${winRows} < 120)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `statusimmune golden: ${S.length} scenarios, ${decRows} decision rows, ${blockTotal} immunity-block rows, ` +
    `${winRows} wins + ${tieRows} ties -> ${OUT}`);
  process.exit(0);
}

main();
