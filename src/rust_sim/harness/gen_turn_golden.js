// gen_turn_golden.js — Gen-3 SINGLE-TURN move-execution differential harness.
//
// Drives REAL gen3 battles over an in-process OMNISCIENT BattleStream (the
// damage_probe.js / gen_damage_golden.js pattern — NO server). Per (scenario,
// seed) it submits ONE damaging move per side, lets the turn resolve, and reads
// the POST-TURN STATE off the live battle object (NOT protocol bytes):
//   * each active mon's hp / maxhp / fainted,
//   * whether each attacker CRIT (|-crit|) or MISSED (|-miss|), and
//   * the battle PRNG seed right BEFORE the turn AND right AFTER it.
//
// THE PROOF (the crux): a single both-attack turn's HP/faint is a deterministic
// function of (which mover went first [the action-order speed-tie shuffle],
// accuracy outcome, crit outcome, the 16-way damage roll, and the gen3
// faint-skips-second-move quirk). The harness emits the sim's PRNG state right
// BEFORE the turn (`SEED_BEFORE`) — so the Rust test seeds its BattleState prng
// with the IDENTICAL pre-turn state (sidestepping the >start setup draws — gender
// sample, turn-1 Quick Claw — which this bounded step does NOT model) — then runs
// `run_turn` and asserts BOTH:
//   (a) post-turn hp/fainted/crit/miss match per side, AND
//   (b) the post-turn PRNG seed equals the sim's `SEED_AFTER`.
// An EXACT post-turn-seed match across MANY seeds is the draw-ORDER+COUNT proof:
// the only way the Rust prng lands on the sim's exact post-turn state every seed
// is to draw the same RNG values in the same order and count (a single extra /
// missing / mis-ordered draw shifts the LCG stream and the seed diverges).
//
// SCENARIO CLASSES:
//   * DISTINCT-SPEED, both-survive, no-secondary (the clean 7-draw turn:
//     [acc,crit,dmg] x2 movers + quickclaw(1,5)). The FULL prng-state differential
//     (a) AND (b) — `tie=0`.
//   * GUARANTEED-FAINT (faster OHKOs slower): the slower move is cancelled (gen3
//     faint-skip) → it draws NOTHING, and the trailing Quick Claw is NOT drawn (a
//     faint defers endTurn). Validates the truncated draw count. (a) AND (b).
//   * SPEED-TIE (identical mons): the action-order Fisher-Yates shuffle DRAWS
//     (one random(0,2)) at the front, deciding who moves first. On a tie the sim
//     ALSO draws per-action eachEvent('Update')/'BeforeTurn' shuffles that this
//     bounded step does NOT model, so for the tie class we assert ONLY (a) the
//     post-turn hp/fainted/crit/miss + WHO moved first (the shuffle outcome is on
//     a production path), NOT the post-turn seed. `tie=1` marks these rows.
//
// MOVES ARE NO-SECONDARY damaging moves (Earthquake / Surf / Double-Edge / Hydro
// Pump / Megahorn / Tackle) so the per-move secondary random(100) is never drawn
// (this step defers secondaries). Hydro Pump (80%) / Megahorn (85%) / Tackle (95%)
// exercise the MISS branch; never_miss moves (Swift) exercise the no-accuracy-draw
// branch.
//
// Output: tests/vectors/turn_golden.txt, TAB-delimited, std-parseable. See the
// header block written to the file for the record grammar.
//
// Run:  node src/rust_sim/harness/gen_turn_golden.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const fs = require('fs');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const OUT = path.resolve(__dirname, '../tests/vectors/turn_golden.txt');
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

// A well-spread deterministic gen5 seed pool (4 16-bit words), same generator as
// gen_damage_golden.js so the corpus is reproducible.
function buildSeeds(n) {
  const out = [];
  let x = 0x9e3779b1 >>> 0;
  const step = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  for (let i = 0; i < n; i++) out.push([step() || 1, step() || 1, step() || 1, step() || 1]);
  return out;
}

// Walk the omniscient log for a side's attacker: did its (last) move CRIT / MISS?
// We attribute by the |move|<actor> line then the following |-crit| / |-miss|.
function critMiss(log) {
  // Returns { p1: {crit,miss}, p2: {crit,miss} } for each side's active attacker.
  const out = { p1: { crit: false, miss: false, moved: false }, p2: { crit: false, miss: false, moved: false } };
  let pendingSide = null; // 'p1' / 'p2' of the in-flight move
  for (const line of log) {
    const parts = line.split('|');
    if (parts.length < 2) continue;
    const tag = parts[1];
    if (tag === 'move' && parts.length >= 3) {
      const actor = parts[2].trim();
      if (actor.startsWith('p1a:')) { pendingSide = 'p1'; out.p1.moved = true; }
      else if (actor.startsWith('p2a:')) { pendingSide = 'p2'; out.p2.moved = true; }
      else pendingSide = null;
    } else if (tag === '-crit' && pendingSide) {
      out[pendingSide].crit = true;
    } else if (tag === '-miss') {
      // |-miss|<source>|<target> — attribute to the source's side.
      const src = (parts[2] || '').trim();
      if (src.startsWith('p1a:')) out.p1.miss = true;
      else if (src.startsWith('p2a:')) out.p2.miss = true;
    }
  }
  return out;
}

// Who moved FIRST (the first |move| line's side) — the action-order outcome (for
// the speed-tie scenarios, this is the shuffle's decision).
function firstMover(log) {
  for (const line of log) {
    const parts = line.split('|');
    if (parts[1] === 'move' && parts.length >= 3) {
      const actor = parts[2].trim();
      if (actor.startsWith('p1a:')) return 'p1';
      if (actor.startsWith('p2a:')) return 'p2';
    }
  }
  return 'none';
}

async function runTurn(sc, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();

  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([sc.p1]) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([sc.p2]) })}`);

  // Quiesce so >start + both >player are consumed and the battle is built + at the
  // first move request, then capture the PRNG state right BEFORE the turn.
  for (let i = 0; i < 8; i++) await tick();
  const seedBefore = stream.battle.prng.getSeed();
  // The two actives' resolved action speeds (the tie key). A distinct-speed
  // scenario MUST have these differ (else it silently becomes a tie that draws the
  // per-action eachEvent shuffles, breaking seed parity).
  const aspeed = [
    stream.battle.sides[0].active[0].getActionSpeed(),
    stream.battle.sides[1].active[0].getActionSpeed(),
  ];

  streams.omniscient.write(`>p1 ${sc.choices.p1}`);
  streams.omniscient.write(`>p2 ${sc.choices.p2}`);
  for (let i = 0; i < 12; i++) await tick();
  const seedAfter = stream.battle.prng.getSeed();

  const snap = (s) => {
    const a = s.active[0];
    return { hp: a.hp, maxhp: a.maxhp, fainted: !!a.fainted };
  };
  const cm = critMiss(log);
  const out = {
    seedBefore,
    seedAfter,
    aspeed,
    turn: stream.battle.turn,
    gen: stream.battle.gen,
    p1: snap(stream.battle.sides[0]),
    p2: snap(stream.battle.sides[1]),
    crit: { p1: cm.p1.crit, p2: cm.p2.crit },
    miss: { p1: cm.p1.miss, p2: cm.p2.miss },
    moved: { p1: cm.p1.moved, p2: cm.p2.moved },
    firstMover: firstMover(log),
  };
  try { streams.omniscient.destroy(); } catch (e) { /* best effort */ }
  return out;
}

// ── Scenarios: each picks ONE damaging move per side; `tie` marks the speed-tie
//    class (seed-parity NOT asserted — extra eachEvent shuffles deferred). ──
function scenarios() {
  const S = [];
  // p1/p2 are mons; p1Slot/p2Slot are the 0-based move indexes (the move actually
  // used). `tie` => speed-tie class. `forceFaint` => expect a KO (a sanity gate).
  const add = (id, p1, p2, p1Slot, p2Slot, opts = {}) =>
    S.push({ id, p1, p2, p1Slot, p2Slot, tie: !!opts.tie, forceFaint: !!opts.forceFaint,
             choices: { p1: `move ${p1Slot + 1}`, p2: `move ${p2Slot + 1}` } });

  // --- DISTINCT-SPEED, both-survive, no-secondary (the clean 7-draw turn). ---
  // 1. STAB special vs neutral physical (Suicune Surf / Snorlax Earthquake).
  add('stab_surf_vs_eq',
    mon('Suicune', ['surf', 'icebeam'], { nature: 'Modest', evs: { hp: 252, spa: 252 } }),
    mon('Snorlax', ['earthquake', 'bodyslam'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    0, 0);
  // 2. STAB-physical Tackle (Normal, no recoil/secondary) vs Surf, both bulky.
  add('tackle_stab_vs_surf',
    mon('Snorlax', ['tackle', 'earthquake'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    mon('Suicune', ['surf', 'icebeam'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
    0, 0);
  // 3. Super-effective (Starmie Surf vs Marowak [Water 2x Ground]; Marowak
  //    Earthquake vs Starmie [neutral]). NO weather setter (sand/hail would add an
  //    end-of-turn chip residual this step defers), NO item (Thick Club's Atk
  //    doubling is a ModifyAtk this step doesn't model). NOTE: the SE Surf actually
  //    OHKOs Marowak in all 60 seeds, so this doubles as faster-attacker faint-skip
  //    coverage on a distinct-speed seed-parity scenario (a bulkier SE survivor is
  //    a follow-up for SE-damage-on-survivor coverage).
  add('se_surf_vs_eq',
    mon('Starmie', ['surf', 'icebeam'], { nature: 'Modest', evs: { hp: 252, spa: 252 } }),
    mon('Marowak', ['earthquake', 'rockslide'], { nature: 'Adamant', evs: { hp: 252, atk: 252 }, ability: 'Rock Head' }),
    0, 0);
  // 4. Quad-effective (Cloyster Ice Beam vs Salamence [Ice 4x Dragon/Flying]) — but
  //    Ice Beam has a freeze secondary (draws random(100)); use a NO-secondary
  //    quad: there is no no-secondary Ice move that's 4x... use Megahorn (Bug) on a
  //    Psychic/Grass? Keep it clean: Aerodactyl Rock Slide is flinch-secondary.
  //    Instead test 4x via Earthquake (Ground 2x) ... not quad. Use a SE neutral
  //    pairing that's no-secondary: Flygon Earthquake (2x vs Tyranitar) — done in 3.
  //    Add a resist case: Skarmory Drill Peck? has no... use Forretress Earthquake.
  add('resisted_eq_vs_megahorn',
    mon('Flygon', ['earthquake', 'rockslide'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    mon('Forretress', ['earthquake', 'spikes'], { nature: 'Relaxed', evs: { hp: 252, def: 252 }, ability: 'Sturdy' }),
    0, 0);
  // 5. Choice Band physical (Earthquake — no recoil/secondary). CB ×1.5 on the
  //    attacker's Atk stat.
  add('choiceband_earthquake',
    mon('Snorlax', ['earthquake', 'tackle'], { item: 'Choice Band', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    mon('Suicune', ['surf', 'icebeam'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
    0, 0);
  // 6. Weather: Kyogre Drizzle ⇒ rain ×1.5 Surf vs Snorlax Earthquake. (Drizzle is
  //    a switch-in ability our engine sets, so the Rust side gets rain too.)
  add('rain_surf_vs_eq',
    mon('Kyogre', ['surf', 'icebeam'], { ability: 'Drizzle', nature: 'Modest', evs: { hp: 252, spa: 252 } }),
    mon('Snorlax', ['earthquake', 'bodyslam'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    0, 0);
  // 7. never_miss accuracy (both Swift — NO accuracy draw, just crit+dmg). Distinct
  //    speed (Jolteon vs Snorlax).
  add('swift_no_accuracy_draw',
    mon('Jolteon', ['swift', 'thunderbolt'], { nature: 'Modest', evs: { hp: 4, spa: 252, spe: 252 } }),
    mon('Snorlax', ['swift', 'bodyslam'], { nature: 'Modest', evs: { hp: 252, spa: 252 } }),
    0, 0);
  // 8. Sub-100 accuracy (Hydro Pump 80% / Megahorn 85%) — exercises the MISS draw
  //    branch (a miss ends that move after its accuracy draw → fewer downstream
  //    draws → seed parity still holds, proving the miss path is counted right).
  //    DISTINCT speed: Starmie (spe 115 base, max-invested) clearly outspeeds the
  //    Heracross, so no action-order tie. (Starmie usually faints to Megahorn —
  //    ~51/60 seeds — adding faint-row coverage to a seed-parity scenario.)
  add('hydropump_vs_megahorn',
    mon('Starmie', ['hydropump', 'surf'], { nature: 'Timid', evs: { hp: 4, spa: 252, spe: 252 } }),
    mon('Heracross', ['megahorn', 'earthquake'], { nature: 'Brave', evs: { hp: 252, atk: 252 } }),
    0, 0);
  // 9. High-crit move (Crabhammer, critRatio 2 ⇒ randomChance(1,8)) — the crit draw
  //    uses the 1/8 denominator; verifies the crit ratio mapping. Crabhammer is
  //    no-secondary. Kingdra Crabhammer vs Snorlax Earthquake.
  add('highcrit_crabhammer',
    mon('Kingdra', ['crabhammer', 'icebeam'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    mon('Snorlax', ['earthquake', 'bodyslam'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    0, 0);
  // 10. Low-level attacker (level scaling) — level-50 Snorlax Earthquake vs a bulky
  //     Suicune; both survive.
  add('low_level_earthquake',
    mon('Snorlax', ['earthquake', 'tackle'], { nature: 'Adamant', evs: { hp: 252, atk: 252 }, level: 50 }),
    mon('Suicune', ['surf', 'icebeam'], { nature: 'Bold', evs: { hp: 252, def: 252 } }),
    0, 0);

  // 11b. TYPE-IMMUNE move (the draw-COUNT crux): Flygon (Ground/Dragon) Earthquake
  //      into Skarmory (Steel/Flying — Ground-immune via Flying) draws ONLY accuracy
  //      then `-immune` (NO crit/damage draw); Skarmory's Drill Peck... has no
  //      secondary but Skarmory is slower. Skarmory Steel Wing (no secondary? it has
  //      a 10% Def-boost secondary) — use a clean physical: Skarmory has few clean
  //      moves; give it Earthquake (Ground 1x vs Flygon... Flygon is Ground/Dragon,
  //      Ground 1x). Flygon faster than Skarmory. Both bulky → survive. The Rust
  //      side MUST short-circuit Flygon's immune EQ after accuracy (or the post-turn
  //      seed desyncs), so this row is a strong draw-count test.
  add('immune_eq_vs_skarmory',
    mon('Flygon', ['earthquake', 'rockslide'], { nature: 'Jolly', evs: { atk: 252, spe: 252 } }),
    mon('Skarmory', ['earthquake', 'drillpeck'], { nature: 'Impish', evs: { hp: 252, def: 252 }, ability: 'Keen Eye' }),
    0, 0);

  // --- GUARANTEED-FAINT (faster OHKOs slower): faint-skip + no Quick Claw. ---
  // 11. Aerodactyl (fast) Earthquake OHKOs Magikarp (frail); Magikarp's Tackle is
  //     cancelled (draws nothing). No Quick Claw (faint defers endTurn).
  add('faster_ohko_magikarp',
    mon('Aerodactyl', ['earthquake', 'doubleedge'], { item: 'Choice Band', nature: 'Adamant', evs: { atk: 252, spe: 252 }, ability: 'Rock Head' }),
    mon('Magikarp', ['tackle', 'splash'], { nature: 'Adamant', evs: { atk: 252 } }),
    0, 0, { forceFaint: true });
  // 12. Strong special OHKO: Kyogre (no rain needed) Surf OHKOs a frail Diglett.
  add('faster_ohko_diglett',
    mon('Kyogre', ['surf', 'icebeam'], { nature: 'Modest', evs: { spa: 252, spe: 252 } }),
    mon('Diglett', ['earthquake', 'rockslide'], { nature: 'Adamant', evs: { atk: 252 } }),
    0, 0, { forceFaint: true });

  // --- SPEED-TIE (identical mons): the action-order shuffle DRAWS; seed parity NOT
  //     asserted (eachEvent shuffles deferred), only (a) hp/fainted/crit/miss +
  //     who-moved-first. ---
  // 13. Two identical Tauros Earthquake (no recoil/secondary): a true speed tie. The
  //     action-order shuffle decides who hits first; both survive one neutral hit.
  add('tie_tauros_eq',
    mon('Tauros', ['earthquake', 'tackle'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    mon('Tauros', ['earthquake', 'tackle'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    0, 0, { tie: true });
  // 14. Two identical Snorlax Earthquake (no secondary) — another tie, both survive.
  add('tie_snorlax_eq',
    mon('Snorlax', ['earthquake', 'bodyslam'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    mon('Snorlax', ['earthquake', 'bodyslam'], { nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
    0, 0, { tie: true });

  return S;
}

function packTeam(m) { return Teams.pack([m]); }

async function main() {
  const seeds = buildSeeds(60); // 60 seeds × 14 scenarios = 840 (scenario,seed) rows
  const lines = [];
  lines.push('# turn_golden.txt — Gen-3 SINGLE-TURN move-execution golden (per-seed STATE differential).');
  lines.push('# SCEN  <id>  <tie:0|1>  <forceFaint:0|1>');
  lines.push('# TEAM  <id>  <p1|p2>  <packed team string>');
  lines.push('# SLOT  <id>  <p1Slot>  <p2Slot>   (0-based move index used by each side)');
  lines.push('# TURN  <id>  <seed_before>  <m,n,o,p seed>  <seed_after>  \\');
  lines.push('#        p1_hp p1_max p1_fnt p1_crit p1_miss p1_moved  \\');
  lines.push('#        p2_hp p2_max p2_fnt p2_crit p2_miss p2_moved  first_mover');
  lines.push('# seed_before = the sim PRNG state right BEFORE the turn (the Rust test SEEDS its');
  lines.push('#   BattleState prng with this, sidestepping the >start setup draws this step omits).');
  lines.push('# seed_after  = the sim PRNG state right AFTER the turn (asserted EXACTLY for tie=0; the');
  lines.push('#   tie=1 rows assert ONLY the hp/fainted/crit/miss + first_mover, NOT seed_after,');
  lines.push('#   because a speed tie also draws per-action eachEvent shuffles this step defers).');

  const S = scenarios();
  const failures = [];
  let rows = 0;

  for (const sc of S) {
    const p1packed = packTeam(sc.p1);
    const p2packed = packTeam(sc.p2);
    lines.push(`SCEN\t${sc.id}\t${sc.tie ? 1 : 0}\t${sc.forceFaint ? 1 : 0}`);
    lines.push(`TEAM\t${sc.id}\tp1\t${p1packed}`);
    lines.push(`TEAM\t${sc.id}\tp2\t${p2packed}`);
    lines.push(`SLOT\t${sc.id}\t${sc.p1Slot}\t${sc.p2Slot}`);

    let faintSeen = false;
    for (const seed of seeds) {
      let r;
      try { r = await runTurn(sc, seed); } catch (e) { failures.push(`${sc.id} seed ${seed}: ${e.message}`); continue; }
      if (r.gen !== 3) { failures.push(`${sc.id}: expected gen 3, got ${r.gen}`); break; }
      // Guard the class invariant: a distinct-speed scenario MUST have distinct
      // action speeds (else it's a silent tie → eachEvent shuffles → seed parity
      // would break); a tie scenario MUST have equal speeds (else the shuffle never
      // fires and the action-order draw isn't exercised).
      const speedsEqual = r.aspeed[0] === r.aspeed[1];
      if (!sc.tie && speedsEqual) {
        failures.push(`${sc.id}: marked distinct-speed but actives TIE on action speed (${r.aspeed.join(' vs ')}) — pick clearly-different-speed mons`);
        break;
      }
      if (sc.tie && !speedsEqual) {
        failures.push(`${sc.id}: marked speed-tie but actives have DISTINCT speeds (${r.aspeed.join(' vs ')}) — the action-order shuffle won't fire`);
        break;
      }
      // Sanity: the turn resolved (turn advanced to 2, OR a faint ended/paused it).
      const seedStr = seed.join(',');
      lines.push([
        'TURN', sc.id, r.seedBefore, seedStr, r.seedAfter,
        r.p1.hp, r.p1.maxhp, r.p1.fainted ? 1 : 0, r.crit.p1 ? 1 : 0, r.miss.p1 ? 1 : 0, r.moved.p1 ? 1 : 0,
        r.p2.hp, r.p2.maxhp, r.p2.fainted ? 1 : 0, r.crit.p2 ? 1 : 0, r.miss.p2 ? 1 : 0, r.moved.p2 ? 1 : 0,
        r.firstMover,
      ].join('\t'));
      if (r.p1.fainted || r.p2.fainted) faintSeen = true;
      rows++;
    }
    if (sc.forceFaint && !faintSeen) {
      failures.push(`${sc.id}: expected a guaranteed faint but none occurred across ${seeds.length} seeds`);
    }
  }

  if (failures.length) {
    console.error('TURN GOLDEN FAILURES:\n  ' + failures.slice(0, 20).join('\n  '));
    process.exit(1);
  }

  fs.writeFileSync(OUT, lines.join('\n') + '\n');
  console.error(`turn golden: ${S.length} scenarios, ${rows} (scenario,seed) rows -> ${OUT}`);
  process.exit(0);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
