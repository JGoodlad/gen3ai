// gen_battle_golden.js — Gen-3 MULTI-TURN move-execution differential harness.
//
// Extends harness/gen_turn_golden.js from ONE turn to an N-turn LOOP of scripted
// damaging moves between BULKY mons, exercising the FULL per-turn cycle the
// single-turn step deferred:
//   (a) the per-action eachEvent('Update')/'BeforeTurn'/'Weather' speed-tie
//       SHUFFLES (drawn only on a speed tie, in the exact place/count), and
//   (b) the END-OF-TURN RESIDUALS — Leftovers heal (1/16), Sandstorm chip (1/16 to
//       non-Rock/Ground/Steel), and major-status damage (burn 1/8, poison 1/8,
//       Toxic n/16 ramp) — all DRAW-FREE arithmetic (the only residual draws are
//       the handler-sort + nested-Weather tie-shuffles).
//
// THE PROOF (the crux): drive the OMNISCIENT in-process BattleStream (no server)
// N turns, capturing the running PRNG seed BEFORE the first recorded turn
// (`seed_before` for the Rust to seed once) and AFTER each turn (`seed_after`).
// The Rust test seeds a BattleState at the first recorded turn's pre-turn state and
// runs `run_battle` WITHOUT re-seeding — so the post-turn seed must match the sim's
// `seed_after` at EVERY turn boundary. An EXACT cross-turn seed match across MANY
// seeds is the draw-ORDER+COUNT proof over the FULL multi-turn cycle (a single
// extra/missing/mis-ordered draw on turn k desyncs every turn >= k).
//
// SCENARIO CLASSES (all distinct-speed for seed parity, plus one tie class):
//   * leftovers (no weather)            — Leftovers heal both, draw-free, 7 draws/turn.
//   * sandstorm (Tyranitar Sand Stream) — sand chip to non-Rock/Ground/Steel +
//     Leftovers, both draw-free; distinct speed ⇒ still 7 draws/turn.
//   * burn / poison / Toxic             — status applied by a status MOVE on turn 1
//     (deferred from the bit-port), so we START RECORDING AT TURN 2: the golden
//     captures the post-turn-1 status (+ Toxic stage) and the Rust seeds at turn-2's
//     pre-turn state with that status injected, then chains turns 2..N (the status
//     DoT residual is draw-free, so cross-turn seed parity still holds).
//   * speed-TIE                         — identical mons: every eachEvent + action-
//     order + residual shuffle FIRES across the turns (16 draws/turn, 17 in sand).
//
// SEEDING (sidestepping the >start setup draws): we capture `seed_before` at the
// boundary just BEFORE the first recorded turn's choices, so the Rust seeds its
// BattleState prng with that exact state and replays only the move turns.
//
// Output: tests/vectors/battle_golden.txt, TAB-delimited, std-parseable. See the
// header block written to the file for the record grammar.
//
// Run:  node src/rust_sim/harness/gen_battle_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/battle_golden.txt');
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

// A well-spread deterministic gen5 seed pool (4 16-bit words), same generator the
// other goldens use so the corpus is reproducible.
function buildSeeds(n) {
  const out = [];
  let x = 0x5bd1e995 >>> 0;
  const step = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  for (let i = 0; i < n; i++) out.push([step() || 1, step() || 1, step() || 1, step() || 1]);
  return out;
}

// Who moved FIRST in the most recent turn (the LAST |move| after the latest |turn|).
function firstMoverThisTurn(log, fromIdx) {
  for (let i = fromIdx; i < log.length; i++) {
    const parts = log[i].split('|');
    if (parts[1] === 'move' && parts.length >= 3) {
      const actor = parts[2].trim();
      if (actor.startsWith('p1a:')) return 'p1';
      if (actor.startsWith('p2a:')) return 'p2';
    }
  }
  return 'none';
}

// Normalize the sim's status string to the golden's token + a toxic stage.
function statusOf(active) {
  const st = active.status || '';
  let stage = 0;
  if (st === 'tox') stage = active.statusState ? (active.statusState.stage || 0) : 0;
  return { status: st || '-', stage };
}

// Run ONE scenario at one seed: drive `turns` turns, recording per-turn the
// pre/post seed + both actives' state. `startTurn` is the 1-based turn at which
// recording begins (so a status-applying turn 1 can be skipped for the bit-port).
async function runBattle(sc, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();

  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([sc.p1]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([sc.p2]) })}`);
  for (let i = 0; i < 8; i++) await tick();

  const snap = (s) => {
    const a = s.active[0];
    const { status, stage } = statusOf(a);
    return { hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted, status, stage };
  };

  const startTurn = sc.startTurn || 1;
  const rec = {
    // The pre-turn STATE + seed at the boundary just before the first RECORDED turn.
    initSeed: null,
    initP1: null,
    initP2: null,
    turns: [], // per recorded turn: { seedBefore, seedAfter, p1, p2, firstMover, ended }
    gen: stream.battle.gen,
  };

  for (let t = 1; t <= sc.turns; t++) {
    if (stream.battle.ended) break;
    const choices = sc.choices(t); // [['p1','move k'], ['p2','move k']]
    // The action speeds for THIS turn's actives (the tie key); used by the harness
    // to GUARD the distinct/tie class invariant.
    const aspeed = [
      stream.battle.sides[0].active[0].getActionSpeed(),
      stream.battle.sides[1].active[0].getActionSpeed(),
    ];
    const seedBefore = stream.battle.prng.getSeed();

    // At the start-turn boundary, record the pre-turn state the Rust seeds with.
    if (t === startTurn) {
      rec.initSeed = seedBefore;
      rec.initP1 = snap(stream.battle.sides[0]);
      rec.initP2 = snap(stream.battle.sides[1]);
    }

    const logLenBefore = log.length;
    for (const [side, choice] of choices) {
      try { streams.omniscient.write(`>${side} ${choice}`); } catch (e) { /* forced/ended */ }
    }
    for (let i = 0; i < 12; i++) await tick();
    const seedAfter = stream.battle.prng.getSeed();

    if (t >= startTurn) {
      rec.turns.push({
        seedBefore,
        seedAfter,
        aspeed,
        p1: snap(stream.battle.sides[0]),
        p2: snap(stream.battle.sides[1]),
        firstMover: firstMoverThisTurn(log, logLenBefore),
        ended: !!(stream.battle.sides[0].active[0].fainted || stream.battle.sides[1].active[0].fainted),
      });
    }
    if (stream.battle.sides[0].active[0].fainted || stream.battle.sides[1].active[0].fainted) break;
  }

  try { streams.omniscient.destroy(); } catch (e) { /* best effort */ }
  return rec;
}

// ── Scenarios: bulky mons + weak no-secondary damaging moves so battles last
//    several turns and residuals fire. `tie` => speed-tie class. `startTurn` skips
//    a status-applying turn 1. `slots` is the per-turn move index used. ──
function scenarios() {
  const S = [];
  // A scenario: { id, p1, p2, tie, startTurn, p1Slot, p2Slot, choices(t) }.
  // For the no-status scenarios the same slot every turn; the status scenarios use a
  // status move on turn 1 (slot differs) then a damaging move on slots p1Slot/p2Slot.
  const addDmg = (id, p1, p2, p1Slot, p2Slot, opts = {}) => {
    S.push({
      id, p1, p2,
      tie: !!opts.tie,
      startTurn: 1,
      p1Slot, p2Slot,
      turns: opts.turns || 6,
      choices: () => [['p1', `move ${p1Slot + 1}`], ['p2', `move ${p2Slot + 1}`]],
    });
  };
  // A status scenario: turn 1 applies the status (status move at slot statusSlot of
  // the inflicter side), then both use a damaging move (slot p1Slot/p2Slot) from
  // turn 2 on. Recording starts at turn 2.
  const addStatus = (id, p1, p2, p1Slot, p2Slot, statusSide, statusSlot, opts = {}) => {
    S.push({
      id, p1, p2,
      tie: false,
      startTurn: 2,
      p1Slot, p2Slot,
      turns: opts.turns || 6,
      choices: (t) => {
        if (t === 1) {
          // The inflicter uses its status move; the other uses its damaging move.
          const c = [['p1', `move ${p1Slot + 1}`], ['p2', `move ${p2Slot + 1}`]];
          c[statusSide] = [statusSide === 0 ? 'p1' : 'p2', `move ${statusSlot + 1}`];
          return c;
        }
        return [['p1', `move ${p1Slot + 1}`], ['p2', `move ${p2Slot + 1}`]];
      },
    });
  };

  // --- (1) LEFTOVERS, no weather, distinct speed (the long cross-turn carry). ---
  // Suicune (spe 85 base) Surf vs Snorlax (spe 30) Earthquake — both bulky, both
  // heal 1/16 each turn (draw-free). 6 clean 7-draw turns.
  addDmg('leftovers_distinct',
    mon('Suicune', ['surf', 'icebeam'], { item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } }),
    mon('Snorlax', ['earthquake', 'bodyslam'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    0, 0);

  // --- (2) LEFTOVERS, no weather, distinct speed, weaker move (Tackle) so it lasts
  //     the full 8 turns. Skarmory (spe 70) Tackle vs Snorlax (spe 30) Tackle. ---
  addDmg('leftovers_tackle_long',
    mon('Skarmory', ['tackle', 'drillpeck'], { item: 'Leftovers', nature: 'Impish', evs: { hp: 252, def: 252 } }),
    mon('Snorlax', ['tackle', 'earthquake'], { item: 'Leftovers', nature: 'Careful', evs: { hp: 252, spd: 252 } }),
    0, 0, { turns: 8 });

  // --- (3) SANDSTORM, distinct speed: Tyranitar Sand Stream sets permanent sand on
  //     switch-in. Snorlax (Normal — takes the chip) vs Tyranitar (Rock/Dark —
  //     IMMUNE to its own sand). Both Leftovers. Distinct speed ⇒ 7 draws/turn. ---
  addDmg('sand_ttar_vs_snorlax',
    mon('Tyranitar', ['rockslide', 'earthquake'], { item: 'Leftovers', ability: 'Sand Stream', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    mon('Snorlax', ['earthquake', 'bodyslam'], { item: 'Leftovers', nature: 'Careful', evs: { hp: 252, spd: 252 } }),
    1, 0);
  // NOTE: Tyranitar slot 0 = Rock Slide (FLINCH secondary — draws random(100)); use
  // slot 1 (Earthquake, no secondary). Snorlax slot 0 = Earthquake (no secondary).

  // --- (4) SANDSTORM, distinct speed, BOTH take the chip: Tyranitar Sand Stream
  //     lead vs Suicune (Water — takes sand) + Starmie? Keep 2 mons. Use Milotic
  //     (Water) so BOTH the non-Rock/Ground/Steel Suicune AND ... actually Tyranitar
  //     is sand-immune. Use a Tyranitar that does NOT take chip + a Suicune that
  //     does — already covered by Snorlax in (3). Add a Steel-immune check:
  //     Skarmory (Steel/Flying — sand IMMUNE) vs sand from a Tyranitar that itself
  //     is immune; the chip only hits neither active → no chip at all. Instead test a
  //     mon that DOES take it: Gengar? frail. Use Suicune (bulky Water). ---
  addDmg('sand_chip_on_suicune',
    mon('Tyranitar', ['earthquake', 'rockslide'], { item: 'Leftovers', ability: 'Sand Stream', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    mon('Suicune', ['surf', 'icebeam'], { item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } }),
    0, 0);

  // --- (5) POISON residual: Gengar (fast) Toxic? No — Toxic = badly poisoned ramp.
  //     Use regular poison via... there's no clean damaging-only poison. We apply
  //     poison with a status move (Toxic→tox ramp; or a poison move). Use a bulky
  //     poison: Crobat? Keep it simple — inflict TOXIC (the ramp) with the Toxic
  //     move turn 1, then both Tackle. Skarmory (immune to Ground) Toxic'd? Skarmory
  //     is Steel — IMMUNE to poison. Use Snorlax (poisonable) badly-poisoned by a
  //     fast Gengar's Toxic, then both attack. ---
  addStatus('toxic_ramp_snorlax',
    mon('Gengar', ['toxic', 'shadowball'], { item: 'Leftovers', nature: 'Timid', evs: { hp: 4, spa: 252, spe: 252 } }),
    mon('Snorlax', ['tackle', 'bodyslam'], { item: 'Leftovers', nature: 'Careful', evs: { hp: 252, spd: 252 } }),
    1, 0, /*statusSide=*/0, /*statusSlot=*/0, { turns: 6 });
  // Gengar slot 1 = Shadow Ball (SpD-drop secondary draws random(100)) — but Gengar
  // only uses it from turn 2; that's a SECONDARY draw the bit-port doesn't model →
  // would desync. Replace Gengar's damaging move with Tackle-equivalent: give Gengar
  // a Normal no-secondary move it can learn. Gengar can't learn Tackle; use a
  // never-miss / no-secondary special: 'swift' (Normal, no secondary, never-miss).
  S[S.length - 1].p1 = mon('Gengar', ['toxic', 'swift'], { item: 'Leftovers', nature: 'Timid', evs: { hp: 4, spa: 252, spe: 252 } });

  // --- (6) BURN residual: a fast burner Will-O-Wisps a bulky physical mon turn 1,
  //     then both attack. Jolteon (fast, Electric) Will-O-Wisp → Snorlax; then
  //     Jolteon Swift (never-miss, no secondary) + Snorlax Tackle. Burn = 1/8. ---
  addStatus('burn_snorlax',
    mon('Jolteon', ['willowisp', 'swift'], { item: 'Leftovers', nature: 'Timid', evs: { hp: 4, spa: 252, spe: 252 } }),
    mon('Snorlax', ['tackle', 'bodyslam'], { item: 'Leftovers', nature: 'Careful', evs: { hp: 252, spd: 252 } }),
    1, 0, /*statusSide=*/0, /*statusSlot=*/0, { turns: 6 });

  // --- (7) POISON (regular, not toxic): inflict via Poison Powder (a status move).
  //     A fast Venomoth poisons a bulky Blissey, then both attack. poison = 1/8. ---
  addStatus('poison_blissey',
    mon('Crobat', ['poisonpowder', 'swift'], { item: 'Leftovers', nature: 'Jolly', evs: { hp: 4, atk: 252, spe: 252 } }),
    mon('Blissey', ['tackle', 'icebeam'], { item: 'Leftovers', nature: 'Calm', evs: { hp: 252, spd: 252 } }),
    1, 0, /*statusSide=*/0, /*statusSlot=*/0, { turns: 6 });

  // --- (8) SPEED-TIE (identical mons): every action-order + eachEvent + residual
  //     shuffle FIRES across the turns. Two identical Snorlax Earthquake, both
  //     Leftovers. 16 draws/turn (residual handler-sort ties the dual Leftovers). ---
  addDmg('tie_snorlax_eq',
    mon('Snorlax', ['earthquake', 'bodyslam'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    mon('Snorlax', ['earthquake', 'bodyslam'], { item: 'Leftovers', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    0, 0, { tie: true });

  // --- (9) SPEED-TIE in SANDSTORM: two identical Tyranitar (both Sand Stream, both
  //     Rock/Dark = sand-immune), Earthquake. The residual phase adds the nested
  //     eachEvent('Weather') tie-shuffle on top → 17 draws/turn. ---
  addDmg('tie_ttar_sand_eq',
    mon('Tyranitar', ['earthquake', 'rockslide'], { item: 'Leftovers', ability: 'Sand Stream', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    mon('Tyranitar', ['earthquake', 'rockslide'], { item: 'Leftovers', ability: 'Sand Stream', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    0, 0, { tie: true });

  // --- (10) SPEED-TIE, no Leftovers: two identical Tauros Earthquake (so the
  //     residual handler-sort has NO tied Leftovers → no residual handler-sort draw,
  //     only the per-action shuffles). Confirms the residual-tie is conditional on
  //     the dual-Leftovers, not unconditional. ---
  addDmg('tie_tauros_no_lefto',
    mon('Tauros', ['earthquake', 'bodyslam'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    mon('Tauros', ['earthquake', 'bodyslam'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    0, 0, { tie: true });

  // --- (11) LEFTOVERS distinct, both at FULL HP early (heal is a no-op until they
  //     take damage) — a weak Tackle pair that barely dents, so several turns where
  //     Leftovers heals back to full (cap-at-maxhp path). Registeel (Steel, bulky)
  //     Tackle vs Skarmory Tackle. ---
  addDmg('leftovers_full_hp_cap',
    mon('Registeel', ['tackle', 'ironhead'], { item: 'Leftovers', nature: 'Impish', evs: { hp: 252, def: 252 } }),
    mon('Skarmory', ['tackle', 'drillpeck'], { item: 'Leftovers', nature: 'Careful', evs: { hp: 252, spd: 252 } }),
    0, 0, { turns: 7 });

  // --- (12) RAIN (Kyogre Drizzle): rain has NO end-of-turn chip (so residuals are
  //     just Leftovers); confirms the rain-weather branch adds no residual draw.
  //     Kyogre Surf (rain ×1.5) vs Snorlax Earthquake. ---
  addDmg('rain_no_chip',
    mon('Kyogre', ['surf', 'icebeam'], { item: 'Leftovers', ability: 'Drizzle', nature: 'Modest', evs: { hp: 252, spa: 252 } }),
    mon('Snorlax', ['earthquake', 'bodyslam'], { item: 'Leftovers', nature: 'Careful', evs: { hp: 252, spd: 252 } }),
    0, 0);

  return S;
}

function packTeam(m) { return Teams.pack([m]); }

const STATUS_TOKENS = { '-': 0, brn: 1, par: 2, slp: 3, frz: 4, psn: 5, tox: 6 };

async function main() {
  const seeds = buildSeeds(40);
  const lines = [];
  lines.push('# battle_golden.txt — Gen-3 MULTI-TURN move-execution golden (per-seed cross-turn STATE+SEED differential).');
  lines.push('# SCEN  <id>  <tie:0|1>  <startTurn>  <p1Slot>  <p2Slot>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# INIT  <id>  <seed>  <m,n,o,p>  p1(hp max fnt status stage)  p2(hp max fnt status stage)');
  lines.push('#   = the pre-turn STATE + seed at the boundary just before the first RECORDED turn');
  lines.push('#     (startTurn). The Rust SEEDS its BattleState prng with <seed> and injects the');
  lines.push('#     status (+ toxic stage) onto each active, then runs run_battle WITHOUT re-seeding.');
  lines.push('#     status token: - brn par slp frz psn tox.');
  lines.push('# TURN  <id>  <turn#>  <seed_before>  <m,n,o,p seed>  <seed_after>  \\');
  lines.push('#        p1_hp p1_max p1_fnt p1_status p1_stage  p2_hp p2_max p2_fnt p2_status p2_stage  \\');
  lines.push('#        first_mover  ended_on_faint');
  lines.push('# seed_after asserted EXACTLY for tie=0; tie=1 rows assert ONLY hp/status/fainted +');
  lines.push('#   first_mover (a tie also draws per-action eachEvent shuffles the Rust DOES model,');
  lines.push('#   but the FULL tie-cycle seed parity is asserted for the no-status tie scenarios too).');

  const S = scenarios();
  const failures = [];
  let rows = 0;

  for (const sc of S) {
    lines.push(`SCEN\t${sc.id}\t${sc.tie ? 1 : 0}\t${sc.startTurn || 1}\t${sc.p1Slot}\t${sc.p2Slot}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${packTeam(sc.p1)}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${packTeam(sc.p2)}`);

    let scenTurnRows = 0;
    for (const seed of seeds) {
      let rec;
      try { rec = await runBattle(sc, seed); } catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); continue; }
      if (rec.gen !== 3) { failures.push(`${sc.id}: expected gen 3, got ${rec.gen}`); break; }
      if (!rec.initSeed || rec.turns.length === 0) {
        failures.push(`${sc.id} seed ${seed}: no recorded turns (startTurn ${sc.startTurn} not reached)`);
        continue;
      }
      // Guard the distinct/tie class invariant on the FIRST recorded turn's speeds.
      const a0 = rec.turns[0].aspeed;
      const speedsEqual = a0[0] === a0[1];
      if (!sc.tie && speedsEqual) {
        failures.push(`${sc.id}: marked distinct-speed but actives TIE on action speed (${a0.join(' vs ')})`);
        break;
      }
      if (sc.tie && !speedsEqual) {
        failures.push(`${sc.id}: marked speed-tie but actives have DISTINCT speeds (${a0.join(' vs ')})`);
        break;
      }

      const seedStr = seed.join(',');
      const fmtStatus = (s) => `${s.status}\t${s.stage}`;
      lines.push([
        'INIT', sc.id, rec.initSeed, seedStr,
        rec.initP1.hp, rec.initP1.maxhp, rec.initP1.fainted ? 1 : 0, rec.initP1.status, rec.initP1.stage,
        rec.initP2.hp, rec.initP2.maxhp, rec.initP2.fainted ? 1 : 0, rec.initP2.status, rec.initP2.stage,
      ].join('\t'));
      let turnNo = (sc.startTurn || 1);
      for (const trn of rec.turns) {
        lines.push([
          'TURN', sc.id, turnNo, trn.seedBefore, seedStr, trn.seedAfter,
          trn.p1.hp, trn.p1.maxhp, trn.p1.fainted ? 1 : 0, trn.p1.status, trn.p1.stage,
          trn.p2.hp, trn.p2.maxhp, trn.p2.fainted ? 1 : 0, trn.p2.status, trn.p2.stage,
          trn.firstMover, trn.ended ? 1 : 0,
        ].join('\t'));
        turnNo++;
        rows++;
        scenTurnRows++;
      }
    }
    if (scenTurnRows === 0) failures.push(`${sc.id}: produced NO turn rows`);
    // Sanity: a status scenario must actually show the status on the init line for
    // at least some seeds (else the status move whiffed / target immune).
  }

  if (failures.length) {
    console.error('BATTLE GOLDEN FAILURES:\n  ' + failures.slice(0, 30).join('\n  '));
    process.exit(1);
  }

  // Validate status scenarios applied their status (scan the emitted INIT lines).
  const needStatus = { toxic_ramp_snorlax: 'tox', burn_snorlax: 'brn', poison_blissey: 'psn' };
  for (const [id, tok] of Object.entries(needStatus)) {
    const ok = lines.some((l) => l.startsWith(`INIT\t${id}\t`) && l.includes(`\t${tok}\t`));
    if (!ok) {
      console.error(`BATTLE GOLDEN: status scenario ${id} never shows status ${tok} on its INIT line — the status move whiffed or the target is immune.`);
      process.exit(1);
    }
  }
  // Reference the token map so an editor keeps it in sync with the Rust parser.
  void STATUS_TOKENS;

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(`battle golden: ${S.length} scenarios, ${rows} (scenario,seed,turn) rows -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
