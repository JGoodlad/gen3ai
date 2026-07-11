// gen_naturalcure_golden.js — the NATURAL CURE switch-out-cure CLASS-SWEEP golden
// (`gen3_natural_cure_v1`, the sole gen-3 SWITCH_OUT-cure ability).
//
// Natural Cure cures the holder's MAJOR STATUS when it SWITCHES OUT (voluntary pivot OR
// phaze-DRAG-out), DRAW-FREE (probe-settled by `harness/probe_naturalcure_rng.js`: the cure +
// its `[silent]` `-curestatus` reveal consume ZERO PRNG — `onSwitchOut`, `onCheckShow` is
// undefined). The cure is made OBSERVABLE on the existing ACTIVE-mon STATUS timeline: an NC mon
// is statused IN-ENGINE by the foe, PIVOTS OUT (cured), PIVOTS BACK — and RETURNS UNSTATUSED.
// The Rust replay must reproduce that active-status column bit-for-bit; a non-NC control RETURNS
// STILL statused (the wrong-target discriminator — the SAME plan, only the ability differs, so
// the timelines diverge at the return boundary iff the cure is modeled).
//
//   COVERS (each a DECISIVE full battle, in-engine status via a modeled foe status move):
//     nc_cure_return_tox   — NC Starmie is TOXIC'd, pivots to Skarmory, RETURNS UNSTATUSED.
//     nc_cure_return_brn   — NC Starmie is BURNED (Will-O-Wisp), pivots, RETURNS UNBURNED.
//     nc_cure_return_par   — NC Starmie is PARALYZED (Thunder Wave — a Water mon is NOT immune),
//                            pivots, RETURNS UN-PARALYZED (and its raw Speed restored).
//     nc_cure_return_slp   — NC Starmie is put to SLEEP (Hypnosis), pivots, RETURNS AWAKE (the
//                            sleep COUNTER reset too — a re-slept mon draws a fresh random(2,6)).
//     nc_cure_phaze_drag   — NC Starmie is TOXIC'd, then the foe ROARS it OUT (a phaze DRAG cures
//                            the dragged-out mon — SwitchOut fires on isDrag), it later returns clean.
//     nc_control_tox_nonnc — the SAME nc_cure_return_tox plan/teams but a NON-NC Starmie
//                            (Illuminate): it RETURNS STILL TOXIC (the cure is ability-gated).
//     nc_unstatused_noop   — an UNSTATUSED NC Starmie pivots freely (the onSwitchOut early-return
//                            no-op — a switch is draw-clean regardless; the seed/state timeline is
//                            byte-identical to a non-NC unstatused pivot, proven by the replay).
//
// Because the cure is DRAW-FREE, the per-decision SEED must match a plain switch bit-for-bit —
// the golden's per-decision `seedAfter` gate proves it (a spurious cure draw would desync).
//
// Output: tests/vectors/naturalcure_golden.txt (same TAB format as flashfire_golden).
//
// Run:  node src/rust_sim/harness/gen_naturalcure_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/naturalcure_golden.txt');
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
  let x = 0x2c9e_b17f >>> 0;
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

// Coverage marker: did a NATURAL-CURE cure fire this decision? Read the sim-truth protocol line
// `|-curestatus|<mon>|<status>|[from] ability: Natural Cure|[silent]` in the decision's log
// window — the exact ground-truth event the Rust cure must produce (the Rust replay reproduces
// the cure's EFFECT on the active-status timeline, which is the real STATE gate; this marker just
// COUNTS the cure events for the coverage floor, like the flashfire `boosted` marker).
function naturalCureFiredSince(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    const l = log[i];
    if (l.includes('|-curestatus|') && l.includes('[from] ability: Natural Cure')) return true;
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

  const rec = { initSeed: null, decisions: [], winner: null, ended: false, gen: stream.battle.gen, cureRows: 0 };

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
      // NON-cyclic plans: CLAMP to the last element (which is a safe attack) as the game length
      // varies by seed — a modulo would re-issue an early `switch` the sim rejects once the game
      // outran the scripted pivots (the STALL trap).
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
    const cured = naturalCureFiredSince(log, logLenBefore);
    if (cured) rec.cureRows++;
    rec.decisions.push({
      request: reqState,
      force,
      choiceP1: encodeChoice(choices.p1),
      choiceP2: encodeChoice(choices.p2),
      seedAfter,
      p1: snap(battle.sides[0]),
      p2: snap(battle.sides[1]),
      firstMover: reqState === 'move' ? firstMoverSince(log, logLenBefore) : 'none',
      cured,
    });
    decisionNo++;
  }

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  try { streams.omniscient.destroy(); } catch (e) {}
  return rec;
}

// ── Scenarios ────────────────────────────────────────────────────────────────
// A shared p1 pair: NC Starmie (the cure holder) + a Snorlax pivot partner (bulky Normal that
// survives Seismic Toss + a Toxic tick, so it does not confound the return boundary by fainting).
// The plan pivots ONCE early — Surf (get statused) → pivot out (CURED) → pivot BACK (RETURN
// boundary, dec 2) → then finishes with attacks. It is NON-cyclic (the tail clamps to Surf), so
// the fixed plan never re-issues an illegal switch as the game length varies by seed. The foe is
// **Misdreavus** (frail Ghost, dies to a few neutral Surfs; **Seismic Toss** deals a fixed 100
// with NO status secondary + NO super-effective typing vs Water/Psychic) so the RETURN boundary
// is UNCONFOUNDED — `ok` there is 100% the cure, and the non-NC control's persistent status is
// the clean discriminator (a non-NC Starmie stays statused → the tox ramp FAINTS it → the whole
// battle diverges: different winner AND length AND HP timeline). Starmie Recovers to outlast the
// foe. Same-length pivot-once plan reused across statuses; only the foe's status move differs.
function ncPair(ability) {
  return [
    mon('Starmie', ['surf', 'recover'], { ability, nature: 'Timid', evs: { spa: 252, spe: 252, hp: 4 } }),
    mon('Snorlax', ['bodyslam', 'rest'], { ability: 'Own Tempo', nature: 'Careful', evs: { hp: 252, spd: 252 } }),
  ];
}
// The pivot-ONCE plans (non-cyclic — clamped to the last element, which is Surf/Seismic-Toss):
//   p1: Surf (get statused), switch (cured), switch back (RETURN @dec2), then Surf/Recover to win.
//   p2: status move ONCE (dec0), then Seismic Toss forever (no re-status → the return is clean).
const NC_P1_PLAN = ['move 1', 'switch 2', 'switch 2', 'move 1', 'move 2', 'move 1', 'move 1', 'move 2', 'move 1', 'move 1'];
const NC_P2_PLAN = ['move 1', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2', 'move 2'];
function foeMisd(statusMove) {
  return [mon('Misdreavus', [statusMove, 'seismictoss'], { ability: 'Levitate', nature: 'Calm', evs: { hp: 252, def: 252 } })];
}

function scenarios() {
  const S = [];

  // nc_cure_return_tox — Misdreavus Toxics (dec0); the pivot-back boundary (dec2) is a Seismic-
  //   Toss turn → Starmie RETURNS UNSTATUSED. Decisive P1 in ~6 decisions.
  S.push({ id: 'nc_cure_return_tox', p1: ncPair('Natural Cure'), p2: foeMisd('toxic'), plan1: NC_P1_PLAN, plan2: NC_P2_PLAN, holderSide: 'p1' });

  // nc_control_tox_nonnc — the SAME plan/teams but a NON-NC Starmie (Illuminate): it RETURNS
  //   STILL TOXIC (dec2), the tox ramp then FAINTS it — the whole battle DIVERGES (P2 wins). The
  //   wrong-target discriminator: only the ability differs, so a modeled cure is the ONLY thing
  //   that makes nc_cure_return_tox's timeline differ from this one.
  S.push({ id: 'nc_control_tox_nonnc', p1: ncPair('Illuminate'), p2: foeMisd('toxic'), plan1: NC_P1_PLAN, plan2: NC_P2_PLAN, holderSide: 'p1' });

  // nc_cure_return_brn — Will-O-Wisp burns (dec0); RETURN (dec2) is UNBURNED. (brn carries no
  //   switch-in reset, so `ok`-on-return is unambiguously the cure.)
  S.push({ id: 'nc_cure_return_brn', p1: ncPair('Natural Cure'), p2: foeMisd('willowisp'), plan1: NC_P1_PLAN, plan2: NC_P2_PLAN, holderSide: 'p1' });

  // nc_cure_return_par — Thunder Wave paralyzes (a Water Starmie is NOT Ground → not immune,
  //   dec0); RETURN (dec2) is UN-PARALYZED (its raw Speed restored — par is a pure application).
  S.push({ id: 'nc_cure_return_par', p1: ncPair('Natural Cure'), p2: foeMisd('thunderwave'), plan1: NC_P1_PLAN, plan2: NC_P2_PLAN, holderSide: 'p1' });

  // nc_cure_return_slp — Hypnosis sleeps (dec0); RETURN (dec2) is AWAKE (the sleep COUNTER reset
  //   too — clearing status drops the whole Status::Sleep(n) variant).
  S.push({ id: 'nc_cure_return_slp', p1: ncPair('Natural Cure'), p2: foeMisd('hypnosis'), plan1: NC_P1_PLAN, plan2: NC_P2_PLAN, holderSide: 'p1' });

  // nc_cure_phaze_drag — the foe TOXICs (dec0), then ROARS the statused NC Starmie OUT (dec1): a
  //   phaze DRAG cures the dragged-out mon (runEvent('SwitchOut') fires on isDrag too). p2 Suicune
  //   (0-Spe Bold, slower → Roars AFTER Starmie moves; Roar priority -6): Toxic then Roar. The
  //   drag replaces Starmie with the only eligible bench mon (Snorlax, n=1 sample) and CURES
  //   Starmie's tox as it leaves. Starmie Surfs / Snorlax Body-Slams Suicune down. Decisive P1.
  S.push({
    id: 'nc_cure_phaze_drag',
    p1: ncPair('Natural Cure'),
    p2: [mon('Suicune', ['toxic', 'roar'], { ability: 'Pressure', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    plan1: ['move 1', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1'],
    plan2: ['move 1', 'move 2', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1'],
    holderSide: 'p1',
  });

  // nc_unstatused_noop — an UNSTATUSED NC Starmie pivots freely: the onSwitchOut early-return
  //   no-op. The foe (Own-Tempo Snorlax) uses Earthquake (NO status secondary) so no status is
  //   ever inflicted → every switch-out has nothing to cure. NO -curestatus ever fires; the
  //   seed/state timeline is byte-identical to a plain pivot (proven by the replay). Pivots twice
  //   early (draw-clean), then attacks.
  S.push({
    id: 'nc_unstatused_noop',
    p1: ncPair('Natural Cure'),
    p2: [mon('Snorlax', ['earthquake', 'bodyslam'], { ability: 'Own Tempo', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
    plan1: ['switch 2', 'switch 2', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1'],
    plan2: ['move 1', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1', 'move 1'],
    holderSide: 'p1',
  });

  return S;
}

async function main() {
  const seeds = buildSeeds(40);
  const lines = [];
  lines.push('# naturalcure_golden.txt — the gen3_natural_cure_v1 SWITCH_OUT-cure class-sweep golden.');
  lines.push('# Per-decision-boundary STATE+HP+STATUS+SEED differential to GAME-END: the cure is');
  lines.push('# observable on the ACTIVE-STATUS timeline (an NC mon statused -> pivots out CURED ->');
  lines.push('# RETURNS UNSTATUSED), vs a non-NC control that RETURNS STILL statused. Draw-free (the');
  lines.push('# per-decision seed matches a plain switch bit-for-bit). `cured`=a Natural-Cure');
  lines.push('# -curestatus fired this decision (coverage marker).');
  lines.push('# SCEN  <id>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>');
  lines.push('# DEC   <id> <seed> <req> <fP1> <fP2> <cP1> <cP2> <seedAfter> \\');
  lines.push('#        p1(species hp max fnt status left) p2(...) first cured');
  lines.push('# END   <id>  <seed>  <ended:0|1>  <winner:p1|p2|tie|none>');

  const S = scenarios();
  const failures = [];
  let decRows = 0, winRows = 0, tieRows = 0, cureTotal = 0;
  // Which scenarios MUST land >=N cure events (the cure holders) vs CONTROL scenarios (the
  // marker legitimately never fires — non-NC control + the unstatused no-op).
  const cureScenarios = new Set(['nc_cure_return_tox', 'nc_cure_return_brn', 'nc_cure_return_par', 'nc_cure_return_slp', 'nc_cure_phaze_drag']);
  const noCureScenarios = new Set(['nc_control_tox_nonnc', 'nc_unstatused_noop']);

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${Teams.pack(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${Teams.pack(sc.p2)}`);

    let scenDecs = 0;
    let scenCures = 0;
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
          sp(d.p1), sp(d.p2), d.firstMover, d.cured ? 1 : 0,
        ].join('\t'));
        decRows++; scenDecs++;
        if (d.cured) { cureTotal++; scenCures++; }
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
    if (cureScenarios.has(sc.id) && scenCures < 10) {
      failures.push(`${sc.id}: only ${scenCures} Natural-Cure cure rows (<10) — the cure barely fires`);
    }
    if (noCureScenarios.has(sc.id) && scenCures > 0) {
      failures.push(`${sc.id}: expected 0 cure rows (a non-NC / unstatused control), got ${scenCures}`);
    }
  }

  if (failures.length) {
    console.error('NATURAL CURE GOLDEN FAILURES:\n  ' + failures.slice(0, 40).join('\n  '));
    process.exit(1);
  }
  if (winRows < 120) { console.error(`NATURAL CURE GOLDEN: too few WIN rows (${winRows} < 120)`); process.exit(1); }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(
    `naturalcure golden: ${S.length} scenarios, ${decRows} decision rows, ${cureTotal} Natural-Cure cure rows, ` +
    `${winRows} wins + ${tieRows} ties -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
