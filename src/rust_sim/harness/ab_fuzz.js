// ab_fuzz.js — the A/B DIFFERENTIAL FUZZER: a continuous, hours-long divergence
// hunter driving the REAL Showdown sim and the Rust port side-by-side.
//
// Per chunk it (1) generates team pairs + battle seeds + random legal choices by
// driving the OMNISCIENT in-process BattleStream to game-end (REUSING the e2e
// capstone recorder — the same per-decision TAB golden format: species/hp/status/
// boosts/confusion/spikes/seed per boundary + winner), (2) writes the chunk file,
// (3) invokes the Rust chunk REPLAYER (`src/bin/ab_replay.rs` — per-battle JSON
// verdicts, never panicking on a divergence), and (4) tallies verdicts, saving a
// SELF-CONTAINED, standalone-replayable repro dir for every divergence.
//
// TEAM MODES (--mode):
//   randbats (default) — Showdown's OWN gen3 Random Battle generator
//                        (`Teams.generate('gen3randombattle', {seed})`, backed by
//                        dist/data/random-battles/gen3/teams). Sets are made
//                        port-replayable at the SET level: an unmodeled ITEM →
//                        Leftovers; an unmodeled ABILITY → the species' modeled/
//                        no-op ability if one exists, else the whole team is
//                        rejection-sampled. Movesets/levels/EVs stay UNTOUCHED
//                        (the choice picker simply never PICKS an unmodeled move).
//                        The adjustment rate is LOGGED per chunk — adjusted teams
//                        are "randbats-derived", not pure randbats.
//   random             — the MODELED-UNIVERSE generator (the coverage multiplier):
//                        teams of 6 sampled from EVERY gen3-dex species whose
//                        gen3 learnset ∩ the modeled move universe has >=4 moves
//                        (>=1 modeled DAMAGING move forced per mon), a modeled/
//                        no-op ability, a modeled item, random nature/EVs/IVs,
//                        level 100 — packed with the real Teams.pack.
//   pool               — the e2e capstone's 22 filter-clean data/teams/ teams.
//
// The modeled-universe predicates (isModeledMove / abilityAllowed / itemAllowed)
// and the battle driver (runBattle / emitBattle) are REQUIRED from
// gen_e2e_fuzz.js — one source of truth, no copy-paste drift.
//
// USAGE
//   node src/rust_sim/harness/ab_fuzz.js [--mode randbats|random|pool|ourandom]
//        [--battles N | --hours H]     (default: run until killed)
//        [--master-seed S]             (default: from time; ALWAYS printed)
//        [--chunk N]                   (default 25 battles per chunk)
//        [--out DIR]                   (default harness/ab_fuzz_out/)
//        [--keep-chunks]               (default: clean chunk files are deleted)
//
// Results land under --out:
//   ab_fuzz.log                    one stats line per chunk (append-only)
//   divergences/<runid>_<battle>/  battle.txt (standalone-replayable single-battle
//                                  chunk) + summary.json (mode, seeds, teams,
//                                  choices, first-divergence kind/index/exp/got)
//   chunks/                        the last (or all, with --keep-chunks) chunk files
//
// SIGINT (ctrl-c): finishes the current chunk, prints the cumulative summary,
// exits 0. A second SIGINT aborts immediately.
//
// Reproducibility: every run prints its --master-seed; the same mode + master
// seed + code replays the same battles. Every divergence repro is additionally
// self-contained (the exact packed teams + init seed + choice tokens ride in
// battle.txt), so repros stay replayable regardless of generator drift.

'use strict';

const path = require('path');
const fs = require('fs');
const { execFileSync, spawnSync } = require('child_process');

const e2e = require('./gen_e2e_fuzz.js');
const ouRandom = require('./ou_random_teams.js');
const {
  runBattle, emitBattle, isModeledMove, abilityAllowed, itemAllowed,
  teamFilterClean, loadTeams, mulberry32, randInt, seedFrom, toId, dex3,
  REJECT_ABILITIES, REJECT_SPECIES, REJECT_MOVES,
} = e2e;

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Teams } = require(path.join(PS, 'dist/sim'));

const ROOT = path.resolve(__dirname, '../../..');
const CRATE = path.resolve(__dirname, '..');
// The isolated byte-fuzz build (CARGO_TARGET_DIR=/tmp/pokesim_target_bytefuzz) is
// preferred via POKESIM_AB_REPLAY_BIN so a byte run never touches the shared target/
// used by the live fuzzers (project law). Falls back to the shared target build.
const REPLAYER = process.env.POKESIM_AB_REPLAY_BIN || path.join(CRATE, 'target/release/ab_replay');

const rustMoves = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/pokemon/gen3_moves.json'), 'utf8'));
const portSpecies = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/pokemon/gen3_species.json'), 'utf8'));
const learnset = JSON.parse(fs.readFileSync(path.join(ROOT, 'data/pokemon/gen3_learnset.json'), 'utf8'));

// ── Flags ─────────────────────────────────────────────────────────────────────
function parseFlags(argv) {
  const f = {
    mode: 'randbats',
    battles: null,
    hours: null,
    masterSeed: null,
    chunk: 25,
    out: path.join(__dirname, 'ab_fuzz_out'),
    keepChunks: false,
    // `--protocol` turns on the OMNISCIENT BYTE differential (gen3_omniscient_byte_fuzz_v1):
    // capture the REAL filtered `|...|` log per battle + byte-diff it against the port's
    // run_full_battle_logged output → a new `protocol` divergence kind. `--format`
    // {gen3customgame,gen3ou} threads the run format (gen3ou = the clause-shuffle draw path
    // the gen3customgame e2e capstone never hits).
    protocol: false,
    format: 'gen3customgame',
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--mode') f.mode = next();
    else if (a === '--battles') f.battles = Number(next());
    else if (a === '--hours') f.hours = Number(next());
    else if (a === '--master-seed') f.masterSeed = Number(next()) >>> 0;
    else if (a === '--chunk') f.chunk = Number(next());
    else if (a === '--out') f.out = path.resolve(next());
    else if (a === '--keep-chunks') f.keepChunks = true;
    else if (a === '--protocol') f.protocol = true;
    else if (a === '--format') f.format = next();
    else { console.error(`unknown flag ${a}`); process.exit(2); }
  }
  if (!['gen3customgame', 'gen3ou'].includes(f.format)) {
    console.error(`--format must be gen3customgame|gen3ou, got ${f.format}`);
    process.exit(2);
  }
  if (!['randbats', 'random', 'pool', 'ourandom'].includes(f.mode)) {
    console.error(`--mode must be randbats|random|pool|ourandom, got ${f.mode}`);
    process.exit(2);
  }
  if (f.masterSeed === null) f.masterSeed = (Date.now() ^ (process.pid * 2654435761)) >>> 0;
  return f;
}

// ── Randbats provider (Showdown's OWN gen3 random-team generator) ────────────
// Probe result (pinned dist): `Teams.generate('gen3randombattle', {seed})` →
// PokemonSet[6] via dist/data/random-battles/gen3/teams; a gen5-style
// [n,n,n,n] seed makes it DETERMINISTIC (verified: same seed → identical team).
const RANDBATS_FORMAT = 'gen3randombattle';

function speciesAllowedAbility(speciesId) {
  const sp = dex3.species.get(speciesId);
  if (!sp || !sp.exists) return null;
  const abils = [...new Set(Object.values(sp.abilities || {}).map(toId))];
  const ok = abils.filter((a) => abilityAllowed(a));
  return ok.length ? ok : null;
}

// Post-process ONE randbats team to be port-replayable at the SET level.
// Returns { team, adjustedSets } or { reject: why }.
function adaptRandbatsTeam(team) {
  let adjustedSets = 0;
  let naturesNormalized = 0;
  for (const set of team) {
    let touched = false;
    const sid = toId(set.species || set.name);
    // The DEFERRAL rejects read the SHARED `REJECT_MOVES` / `REJECT_ABILITIES` /
    // `REJECT_SPECIES` sets (gen_e2e_fuzz.js — one source of truth), tallying
    // `move:<id>` / `reject:<key>` in `drop_reasons`.
    //   * `REJECT_ABILITIES` / `REJECT_SPECIES` are now EMPTY: FORECAST/Castform — the last
    //     ability member — is MODELED (`gen3_forecast_v1`, ROUND 35), so **gen3-randbats
    //     Castform teams now reach the port** (they were rejection-sampled out before,
    //     exactly the coverage hole that hid the Transform no-op from every offline gate).
    //   * `REJECT_MOVES` is NOT empty. Its ORIGINAL members — Transform (ROUND 33) and the
    //     wrap family (ROUND 32) — are modeled and LEFT (so Ditto and Mew reach the port
    //     too), but the CURRENT members are the ROUND-40 audit's 16 silent-desync moves
    //     (`gen3_unmodeled_move_failloud_v2`). ZERO of those appear in the curated
    //     gen3randombattle movepool (measured exhaustively), so this reject never fires on
    //     randbats today; it exists so a future randbats data update cannot feed a guarded
    //     carrier to the construction panic.
    // Both loops stay as the seam for the next deferral.
    for (const mv of (set.moves || [])) {
      if (REJECT_MOVES.has(toId(mv))) return { reject: `move:${toId(mv)}` };
    }
    if (REJECT_ABILITIES.has(toId(set.ability)) || REJECT_SPECIES.has(sid)) {
      return { reject: `reject:${toId(set.ability)}|${sid}` };
    }
    // Port-data ingest guards (safety net — gen3 randbats should always pass):
    if (!portSpecies[sid]) return { reject: `species:${sid}` };
    for (const mv of set.moves) {
      if (!rustMoves[toId(mv)]) return { reject: `move:${toId(mv)}` };
    }
    // Randbats sets carry NO nature (the sim treats '' as neutral); the port's
    // compute_stats fail-louds on an unknown nature. Hardy is NEUTRAL — stat-
    // identical in BOTH engines (both read the same packed team) — so this is a
    // serialization normalization, not a semantic adjustment (counted apart).
    if (!set.nature) { set.nature = 'Hardy'; naturesNormalized++; }
    // Unspecified GENDER on a ratio species: the sim SAMPLES one at construction
    // (`battle.sample(['M','F'])`) — a construction-time value the port cannot
    // re-derive, and the Cute Charm attract gender-compare fail-louds on it
    // (batch-4, gen3_ability_batch4: the 469-panic 2026-07-09 spike). Pin 'M'
    // explicitly — like the Hardy nature, a SERIALIZATION normalization: both
    // engines read the same packed team, and 'M' is valid for every true-ratio
    // species ('' from the dex means both genders possible; fixed-M/F/N species
    // return their fixed letter and are left untouched).
    if (!set.gender && dex3.species.get(sid).gender === '') { set.gender = 'M'; naturesNormalized++; }
    // Unmodeled ITEM → Leftovers (universally holdable, modeled).
    if (!itemAllowed(set.item)) { set.item = 'Leftovers'; touched = true; }
    // Unmodeled ABILITY → the species' modeled/no-op ability if one exists,
    // else the whole team is rejection-sampled.
    if (!abilityAllowed(set.ability)) {
      const ok = speciesAllowedAbility(sid);
      if (!ok) return { reject: `ability:${toId(set.ability)}` };
      // Deterministic pick (first allowed) — determinism must not depend on rng state here.
      const disp = dex3.abilities.get(ok[0]);
      set.ability = disp && disp.exists ? disp.name : ok[0];
      touched = true;
    }
    if (touched) adjustedSets++;
  }
  return { team, adjustedSets, naturesNormalized };
}

function makeRandbatsProvider(rng, stats) {
  return function nextTeam() {
    // Real gen3 species are saturated with unmodeled abilities, so the team-level
    // rejection rate is ~95-97% — the cap must be far above the ~3% keep rate's
    // long-run tail (P(2000 consecutive rejections) ≈ 1e-25; generation is ~1-2ms).
    for (let tries = 0; tries < 2000; tries++) {
      const genSeed = seedFrom((Math.floor(rng() * 4294967296)) >>> 0);
      let team;
      try {
        team = Teams.generate(RANDBATS_FORMAT, { seed: genSeed });
      } catch (e) { stats.genErrors++; continue; }
      if (!team || team.length !== 6) { stats.genErrors++; continue; }
      const r = adaptRandbatsTeam(team);
      if (r.reject) {
        stats.teamsRejected++;
        stats.rejectReasons.set(r.reject, (stats.rejectReasons.get(r.reject) || 0) + 1);
        continue;
      }
      // setsTotal counts KEPT teams only — the disclosed adjustment rate is
      // "sets touched / sets actually fielded" (rejected teams never battle).
      stats.setsTotal += team.length;
      stats.setsAdjusted += r.adjustedSets;
      stats.naturesNormalized += r.naturesNormalized || 0;
      stats.teamsKept++;
      const packed = Teams.pack(r.team);
      const re = Teams.unpack(packed);
      if (!re || re.length !== 6) { stats.genErrors++; continue; }
      return { packed, genSeed };
    }
    throw new Error('randbats provider: 2000 consecutive rejections — check the adapter');
  };
}

// ── Random modeled-universe provider (the coverage multiplier) ────────────────
const NATURES = [
  'Adamant', 'Bashful', 'Bold', 'Brave', 'Calm', 'Careful', 'Docile', 'Gentle',
  'Hardy', 'Hasty', 'Impish', 'Jolly', 'Lax', 'Lonely', 'Mild', 'Modest',
  'Naive', 'Naughty', 'Quiet', 'Quirky', 'Rash', 'Relaxed', 'Sassy', 'Serious', 'Timid',
];
// Item ids the port models (mirrors gen_e2e_fuzz's MODELED_ITEMS, as pickable ids).
const ITEM_CHOICES = [...e2e.MODELED_ITEMS];

// The modeled-universe: every gen3 species whose learnset ∩ modeled moves has
// >=4 options with >=1 modeled DAMAGING move, and >=1 modeled/no-op ability.
function buildRandomUniverse() {
  const eligible = [];
  const allModeledMoves = new Set();
  for (const sid of Object.keys(learnset).sort()) {
    if (!portSpecies[sid]) continue;
    const sp = dex3.species.get(sid);
    if (!sp || !sp.exists || sp.gen > 3) continue;
    const pool = [...new Set(learnset[sid].map(toId))].filter((m) => rustMoves[m] && isModeledMove(m));
    const dmg = pool.filter((m) => {
      const md = dex3.moves.get(m);
      return md && md.exists && md.category !== 'Status';
    });
    if (pool.length < 4 || dmg.length < 1) continue;
    const abil = speciesAllowedAbility(sid);
    if (!abil) continue;
    for (const m of pool) allModeledMoves.add(m);
    eligible.push({ sid, name: sp.name, pool, dmg, abil });
  }
  return { eligible, allModeledMoves };
}

function sampleDistinct(rng, arr, n) {
  const idx = arr.map((_, i) => i);
  const out = [];
  for (let k = 0; k < n && idx.length; k++) {
    const j = randInt(rng, idx.length);
    out.push(arr[idx[j]]);
    idx.splice(j, 1);
  }
  return out;
}

function genRandomSet(entry, rng) {
  // 1 forced modeled DAMAGING move + 3 others from the remaining pool.
  const first = entry.dmg[randInt(rng, entry.dmg.length)];
  const rest = entry.pool.filter((m) => m !== first);
  const moves = [first, ...sampleDistinct(rng, rest, 3)];
  // Random EVs (0..252 step 4 per stat), trimmed to the 510 total.
  const keys = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
  const evs = {};
  let total = 0;
  for (const k of keys) { const v = randInt(rng, 64) * 4; evs[k] = v; total += v; }
  let guard = 0;
  while (total > 510 && guard++ < 1000) {
    const k = keys[randInt(rng, 6)];
    if (evs[k] >= 4) { evs[k] -= 4; total -= 4; }
  }
  const ivs = {};
  for (const k of keys) ivs[k] = randInt(rng, 32);
  return {
    name: entry.name,
    species: entry.name,
    moves,
    ability: entry.abil[randInt(rng, entry.abil.length)],
    item: ITEM_CHOICES[randInt(rng, ITEM_CHOICES.length)],
    nature: NATURES[randInt(rng, NATURES.length)],
    evs,
    ivs,
    level: 100,
  };
}

function makeRandomProvider(rng, universe, stats) {
  return function nextTeam() {
    for (let tries = 0; tries < 50; tries++) {
      const entries = sampleDistinct(rng, universe.eligible, 6);
      const team = entries.map((en) => genRandomSet(en, rng));
      stats.setsTotal += team.length;
      const packed = Teams.pack(team);
      const re = Teams.unpack(packed);
      if (!re || re.length !== 6) { stats.genErrors++; continue; }
      stats.teamsKept++;
      return { packed, genSeed: null };
    }
    throw new Error('random provider: 50 consecutive pack failures');
  };
}

// ── R3 RESOLVED: Hidden Power is now IV-derived, no quarantine ─────────────────
// R3 (the flat-BP-70 Hidden Power divergence) is FIXED in the engine
// (`gen3_iv_derived_hidden_power_bp_v1`): the port now computes each HP mon's base power
// from its actual IVs (`MonState::hidden_power_bp`, `⌊hpPowerX·40/63⌋+30`, range 30..=70),
// matching the sim bit-for-bit. So EVERY real gen3ou HP team is now byte-safe — the former
// `teamHpBp70Clean` picker quarantine (which dropped the ~14 non-BP-70-spread teams) is
// REMOVED. `random` mode still excludes HP by default (its random IVs give arbitrary types).

// ── Pool provider (the e2e capstone's filter-clean real teams) ────────────────
function makePoolProvider(rng, stats) {
  const { teams } = loadTeams();
  // Filter-clean (modeled abilities/items/moves) — HP teams no longer need a BP-70 filter
  // (R3 is fixed: the engine damages Hidden Power at its IV-true BP for every spread).
  const clean = teams.filter((t) => teamFilterClean(t.packed).ok);
  if (clean.length < 2) throw new Error(`pool mode: only ${clean.length} filter-clean teams`);
  console.error(`[pool] ${clean.length} filter-clean teams of ${teams.length} loaded` +
    ` (R3 fixed: IV-derived Hidden Power BP — all HP teams byte-safe)`);
  return function nextTeam() {
    stats.teamsKept++;
    return { packed: clean[randInt(rng, clean.length)].packed, genSeed: null };
  };
}

// Pin every UNSET gender to an explicit letter (mirrors bridge_ab_fuzz.js::pinGenders):
// a ratio species ('' → 'M') / a fixed-gender species (its letter). The byte path needs
// this because an UNSET gender makes the SIM draw `battle.sample(['M','F'])` at
// construction (a pre-initSeed draw the port cannot see → the `|switch|` DETAILS gender
// diverges — the construction-window gap). Pinning makes BOTH engines read the SAME
// explicit gender (no draw), so the switch details match + the batch emission forms
// downstream become visible. Idempotent on already-gender-safe teams. `gen3_omniscient_byte_fuzz_v1`.
function pinGenders(packed) {
  const team = Teams.unpack(packed);
  if (!team) return packed;
  let touched = false;
  for (const set of team) {
    if (set.gender) continue;
    const sid = toId(set.species || set.name);
    const sp = dex3.species.get(sid);
    if (!sp || !sp.exists) continue;
    set.gender = sp.gender === '' ? 'M' : sp.gender;
    touched = true;
  }
  return touched ? Teams.pack(team) : packed;
}

// ── Repro saving ──────────────────────────────────────────────────────────────
function chunkHeader(flags, runId, chunkIdx) {
  return [
    '# ab_fuzz chunk — A/B differential fuzzer (real Showdown sim vs the Rust port).',
    `# mode=${flags.mode} master_seed=${flags.masterSeed} run_id=${runId} chunk=${chunkIdx}`,
    '# Format identical to tests/vectors/e2e_fuzz_golden.txt (SCEN/TEAM/INIT/DEC/END).',
    '# Replay: target/release/ab_replay <this-file>   (one JSON verdict per battle)',
  ];
}

function saveRepro(outDir, runId, flags, battleMeta, verdict, chunkIdx) {
  // Allowlisted residuals go to a SEPARATE `allowlisted/` tree (auditable + a corpus-fixture
  // source) so `divergences/` stays the gate-FAIL set only. A real (non-allowlisted) divergence
  // lands in `divergences/`.
  const sub = verdict.allowlisted ? 'allowlisted' : 'divergences';
  const dir = path.join(outDir, sub, `${runId}_${battleMeta.id}`);
  fs.mkdirSync(dir, { recursive: true });
  const lines = chunkHeader(flags, runId, chunkIdx);
  emitBattle(lines, battleMeta.id, battleMeta.p1Packed, battleMeta.p2Packed, battleMeta.rec, { protocol: flags.protocol });
  fs.writeFileSync(path.join(dir, 'battle.txt'), lines.join('\n') + '\n');
  const summary = {
    mode: flags.mode,
    master_seed: flags.masterSeed,
    run_id: runId,
    battle_id: battleMeta.id,
    battle_seed: battleMeta.battleSeed,
    init_seed: battleMeta.rec.initSeed,
    choose_seed: battleMeta.rec.chooseSeed,
    gen_seeds: battleMeta.genSeeds,
    packed_teams: { p1: battleMeta.p1Packed, p2: battleMeta.p2Packed },
    choices: battleMeta.rec.decisions.map((d) => [d.choiceP1, d.choiceP2]),
    dropped: battleMeta.rec.dropped,
    ended: battleMeta.rec.ended,
    winner: battleMeta.rec.winner,
    first_divergence: {
      verdict: verdict.verdict,
      kind: verdict.kind || verdict.verdict,
      decision_index: verdict.decision === undefined ? null : verdict.decision,
      expected: verdict.expected === undefined ? null : verdict.expected,
      got: verdict.got === undefined ? null : verdict.got,
      detail: verdict.detail === undefined ? null : verdict.detail,
    },
    replay_cmd: `src/rust_sim/target/release/ab_replay ${path.relative(ROOT, dir)}`,
  };
  fs.writeFileSync(path.join(dir, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
  return dir;
}

// ── Main loop ─────────────────────────────────────────────────────────────────
async function main() {
  const flags = parseFlags(process.argv);
  const runId = `r${Date.now().toString(36)}`;
  const t0 = Date.now();

  fs.mkdirSync(path.join(flags.out, 'chunks'), { recursive: true });
  fs.mkdirSync(path.join(flags.out, 'divergences'), { recursive: true });
  const logPath = path.join(flags.out, 'ab_fuzz.log');

  console.error(`[ab_fuzz] run_id=${runId} mode=${flags.mode} master_seed=${flags.masterSeed} ` +
    `chunk=${flags.chunk} battles=${flags.battles ?? '∞'} hours=${flags.hours ?? '∞'} out=${flags.out}`);
  console.error('[ab_fuzz] reproduce with: ' +
    `node src/rust_sim/harness/ab_fuzz.js --mode ${flags.mode} --master-seed ${flags.masterSeed}` +
    (flags.battles ? ` --battles ${flags.battles}` : ''));

  // Build the Rust replayer once (fast if fresh) — UNLESS an explicit binary was
  // provided via POKESIM_AB_REPLAY_BIN (the isolated byte-fuzz build), in which case we
  // must NOT rebuild into the shared target/.
  if (!process.env.POKESIM_AB_REPLAY_BIN) {
    const env = { ...process.env, PATH: `${process.env.HOME}/.cargo/bin:${process.env.PATH}` };
    const r = spawnSync('cargo', ['build', '--release', '--bin', 'ab_replay'], { cwd: CRATE, env, stdio: 'inherit' });
    if (r.status !== 0) { console.error('[ab_fuzz] cargo build failed'); process.exit(1); }
  } else {
    console.error(`[ab_fuzz] using replayer ${REPLAYER} (no rebuild)`);
  }

  // Deterministic streams, all derived from the master seed.
  const teamRng = mulberry32(flags.masterSeed);
  const battleRng = mulberry32((flags.masterSeed ^ 0xabcdef01) >>> 0);

  const genStats = {
    setsTotal: 0, setsAdjusted: 0, naturesNormalized: 0, teamsKept: 0, teamsRejected: 0, genErrors: 0,
    rejectReasons: new Map(),
  };
  let provider;
  if (flags.mode === 'randbats') provider = makeRandbatsProvider(teamRng, genStats);
  else if (flags.mode === 'pool') provider = makePoolProvider(teamRng, genStats);
  else if (flags.mode === 'ourandom') {
    // gen3ou-RANDBATS: random teams drawn from the REAL gen3ou distribution (Smogon usage +
    // the teammate joint + per-species move/item/ability/spread priors), validated by
    // Showdown's own TeamValidator('gen3ou'). See harness/ou_random_teams.js.
    const ouUniverse = ouRandom.buildOuUniverse({
      isModeledMove: (m) => isModeledMove(m, true), // HP is BP-70 by construction here
      modeledItem: (it) => e2e.MODELED_ITEMS.has(toId(it)),
      allowedAbility: (sid, a) => {
        const ok = speciesAllowedAbility(sid);
        return !!ok && ok.includes(toId(a));
      },
      portSpecies,
    });
    console.error(ouRandom.describeCoverage(ouUniverse));
    provider = ouRandom.makeOuRandomProvider(teamRng, genStats, ouUniverse);
  }
  else {
    const universe = buildRandomUniverse();
    console.error(`[random] modeled universe: ${universe.eligible.length} species, ` +
      `${universe.allModeledMoves.size} modeled moves, ` +
      `${e2e.MODELED_ABILITIES.size}+${e2e.NOOP_ABILITIES.size} abilities (modeled+no-op), ` +
      `${ITEM_CHOICES.length} items`);
    provider = makeRandomProvider(teamRng, universe, genStats);
  }

  // Cumulative tallies.
  const cum = {
    battles: 0, ok: 0, diverged: 0, panic: 0, parseError: 0, empty: 0,
    // `--protocol` GREEN GATE: a `diverged` verdict carrying an `allowlisted` reason
    // (a documented, non-gen3ou-impacting residual — see ab_replay's classify_known_residual)
    // is counted HERE, NOT in `diverged`. The gate fails ONLY on a non-allowlisted divergence.
    allowlisted: 0,
    allowlistReasons: new Map(),
    prefix: 0, // dropped-mid-battle prefixes (still replayed + compared)
    ended: 0,
    decisions: 0,
    kinds: new Map(),
    dropReasons: new Map(),
    speciesRoster: new Set(),
    speciesActive: new Set(),
    movesUsed: new Set(),
    reproDirs: [],
    chunkErrors: 0,
  };

  let stopRequested = false;
  process.on('SIGINT', () => {
    if (stopRequested) { console.error('\n[ab_fuzz] second SIGINT — aborting now'); process.exit(130); }
    stopRequested = true;
    console.error('\n[ab_fuzz] SIGINT — finishing the current chunk, then summarizing…');
  });

  const shouldStop = () => {
    if (stopRequested) return true;
    if (flags.battles !== null && cum.battles >= flags.battles) return true;
    if (flags.hours !== null && (Date.now() - t0) / 3600000 >= flags.hours) return true;
    return false;
  };

  let chunkIdx = 0;
  let lastChunkFile = null;
  while (!shouldStop()) {
    const chunkT0 = Date.now();
    const lines = chunkHeader(flags, runId, chunkIdx);
    const metas = new Map(); // battle id -> meta (for repro saving)
    let produced = 0;
    const setsAdjBefore = genStats.setsAdjusted;
    const setsTotBefore = genStats.setsTotal;

    try {
      const target = flags.battles !== null
        ? Math.min(flags.chunk, Math.max(1, flags.battles - cum.battles))
        : flags.chunk;
      for (let i = 0; i < target && !stopRequested; i++) {
        const t1 = provider();
        let t2 = provider();
        // Byte mode: pin genders so the sim never draws one at construction (the
        // switch-details construction-window gap). State mode is unchanged.
        if (flags.protocol) { t1.packed = pinGenders(t1.packed); t2.packed = pinGenders(t2.packed); }
        // pool mode can hand back the same packed team — that's a legal mirror.
        const battleSeed = seedFrom((Math.floor(battleRng() * 4294967296)) >>> 0);
        const chooseSeed = (Math.floor(battleRng() * 4294967296) ^ 0x9e3779b9) >>> 0;
        let rec;
        try {
          // `pool` mode's teams are gen3ou-VALIDATED (BP-70 HP), so admit typed Hidden
          // Power there; `random` mode's random IVs could yield a non-70 HP BP so it stays
          // excluded (see isModeledMove). `--format` threads the run format for the byte path.
          rec = await runBattle(t1.packed, t2.packed, battleSeed, chooseSeed, 'modeled', {
            format: flags.format,
            // `ourandom` PINS Hidden Power to BP 70 by construction (its IV solver), so HP is
            // byte-safe there exactly as it is on the gen3ou-validated pool.
            allowHiddenPower: flags.mode === 'pool' || flags.mode === 'ourandom',
          });
        } catch (e) {
          cum.empty++;
          cum.dropReasons.set('sim-exception', (cum.dropReasons.get('sim-exception') || 0) + 1);
          continue;
        }
        if (!rec.initSeed || rec.decisions.length === 0) {
          cum.empty++;
          const r = rec.dropped || 'empty';
          cum.dropReasons.set(r.split(':')[0], (cum.dropReasons.get(r.split(':')[0]) || 0) + 1);
          continue;
        }
        // A battle dropped mid-run (forced-unmodeled) is kept as a PREFIX: every
        // recorded boundary is still a committed decision the port must match
        // (END carries ended=0/none, which run_full_battle reproduces exactly).
        if (rec.dropped) {
          cum.prefix++;
          cum.dropReasons.set(rec.dropped.split(':')[0], (cum.dropReasons.get(rec.dropped.split(':')[0]) || 0) + 1);
        }
        if (rec.ended) cum.ended++;
        const id = `ab_${chunkIdx}_${i}`;
        emitBattle(lines, id, t1.packed, t2.packed, rec, { protocol: flags.protocol });
        metas.set(id, {
          id, p1Packed: t1.packed, p2Packed: t2.packed, rec, battleSeed,
          genSeeds: [t1.genSeed, t2.genSeed],
        });
        produced++;
        cum.battles++;
        cum.decisions += rec.decisions.length;
        // Coverage tallies.
        for (const packed of [t1.packed, t2.packed]) {
          for (const set of Teams.unpack(packed)) cum.speciesRoster.add(toId(set.species || set.name));
        }
        for (const d of rec.decisions) {
          if (d.p1 && d.p1.species && d.p1.species !== '-') cum.speciesActive.add(toId(d.p1.species));
          if (d.p2 && d.p2.species && d.p2.species !== '-') cum.speciesActive.add(toId(d.p2.species));
        }
        for (const m of Object.keys(rec.movesUsed || {})) cum.movesUsed.add(m);
      }

      if (produced === 0) {
        if (stopRequested) break;
        cum.chunkErrors++;
        appendLog(logPath, `${new Date().toISOString()} chunk=${chunkIdx} EMPTY (no replayable battles)`);
        chunkIdx++;
        continue;
      }

      // Write the chunk + replay it through the Rust port.
      const chunkFile = path.join(flags.out, 'chunks', `${runId}_chunk${chunkIdx}.txt`);
      fs.writeFileSync(chunkFile, lines.join('\n') + '\n');

      let verdicts = [];
      const replayArgs = flags.protocol ? ['--protocol', chunkFile] : [chunkFile];
      const stdout = execFileSync(REPLAYER, replayArgs, { maxBuffer: 256 * 1024 * 1024 }).toString();
      for (const line of stdout.split('\n')) {
        if (!line.trim() || line.startsWith('{"chunk_summary"')) continue;
        try { verdicts.push(JSON.parse(line)); } catch (e) { /* non-JSON noise */ }
      }

      let chunkOk = 0; let chunkDiv = 0;
      for (const v of verdicts) {
        if (v.verdict === 'ok') { cum.ok++; chunkOk++; continue; }
        // GREEN GATE: an ALLOWLISTED divergence (a documented non-gen3ou residual) is NOT
        // a gate failure — count it by reason, don't save a repro, don't tally it as diverged.
        if (v.allowlisted) {
          cum.allowlisted++;
          cum.allowlistReasons.set(v.allowlisted, (cum.allowlistReasons.get(v.allowlisted) || 0) + 1);
          const meta = metas.get(v.battle);
          if (meta) saveRepro(flags.out, runId, flags, meta, v, chunkIdx); // → allowlisted/ tree
          continue;
        }
        chunkDiv++;
        if (v.verdict === 'panic') cum.panic++;
        else if (v.verdict === 'parse_error') cum.parseError++;
        else cum.diverged++;
        const kind = v.kind || v.verdict;
        cum.kinds.set(kind, (cum.kinds.get(kind) || 0) + 1);
        const meta = metas.get(v.battle);
        if (meta) {
          const dir = saveRepro(flags.out, runId, flags, meta, v, chunkIdx);
          cum.reproDirs.push(dir);
          console.error(`[DIVERGED] ${v.battle} kind=${kind} dec=${v.decision ?? '-'} → ${dir}`);
        }
      }

      // Housekeeping: clean chunk files are deleted unless --keep-chunks (the
      // divergent battles are already saved standalone under divergences/).
      if (!flags.keepChunks) {
        if (lastChunkFile && fs.existsSync(lastChunkFile)) fs.unlinkSync(lastChunkFile);
        lastChunkFile = chunkFile;
      }

      const hrs = (Date.now() - t0) / 3600000;
      const bph = hrs > 0 ? Math.round(cum.battles / hrs) : 0;
      const adjRate = flags.mode === 'randbats' && genStats.setsTotal > setsTotBefore
        ? ((genStats.setsAdjusted - setsAdjBefore) / (genStats.setsTotal - setsTotBefore)).toFixed(3)
        : '-';
      const kindsStr = [...cum.kinds.entries()].map(([k, c]) => `${k}=${c}`).join(',') || '-';
      appendLog(logPath,
        `${new Date().toISOString()} run=${runId} mode=${flags.mode} chunk=${chunkIdx} ` +
        `battles=${produced} ok=${chunkOk} diverged=${chunkDiv} ` +
        `cum_battles=${cum.battles} cum_ok=${cum.ok} cum_diverged=${cum.diverged} cum_panic=${cum.panic} ` +
        `cum_allowlisted=${cum.allowlisted} ` +
        `cum_prefix=${cum.prefix} cum_empty=${cum.empty} kinds=${kindsStr} ` +
        `species=${cum.speciesRoster.size} active_species=${cum.speciesActive.size} moves=${cum.movesUsed.size} ` +
        `adj_rate=${adjRate} bph=${bph} chunk_s=${((Date.now() - chunkT0) / 1000).toFixed(1)}`);
    } catch (e) {
      // A chunk that errors (sim crash, replayer I/O, …) logs + continues — the
      // loop must survive hours.
      cum.chunkErrors++;
      appendLog(logPath, `${new Date().toISOString()} chunk=${chunkIdx} ERROR ${String(e && e.message || e).slice(0, 300)}`);
      console.error(`[chunk ${chunkIdx} ERROR]`, e && e.stack ? e.stack.split('\n').slice(0, 4).join('\n') : e);
    }
    chunkIdx++;
  }

  // ── Cumulative summary ──────────────────────────────────────────────────────
  const hrs = (Date.now() - t0) / 3600000;
  const bph = hrs > 0 ? Math.round(cum.battles / hrs) : 0;
  const summary = {
    run_id: runId,
    mode: flags.mode,
    master_seed: flags.masterSeed,
    elapsed_hours: Number(hrs.toFixed(3)),
    battles: cum.battles,
    battles_per_hour: bph,
    decisions: cum.decisions,
    ok: cum.ok,
    diverged: cum.diverged,
    panic: cum.panic,
    parse_error: cum.parseError,
    allowlisted: cum.allowlisted,
    allowlisted_by_reason: Object.fromEntries(cum.allowlistReasons),
    prefix_battles: cum.prefix,
    ended_battles: cum.ended,
    empty_skipped: cum.empty,
    chunk_errors: cum.chunkErrors,
    divergence_kinds: Object.fromEntries(cum.kinds),
    drop_reasons: Object.fromEntries(cum.dropReasons),
    coverage: {
      distinct_species_rostered: cum.speciesRoster.size,
      distinct_species_active: cum.speciesActive.size,
      distinct_moves_used: cum.movesUsed.size,
    },
    randbats_adjustment: flags.mode === 'randbats' ? {
      sets_total: genStats.setsTotal,
      sets_adjusted: genStats.setsAdjusted,
      natures_normalized: genStats.naturesNormalized,
      set_adjust_rate: genStats.setsTotal ? Number((genStats.setsAdjusted / genStats.setsTotal).toFixed(4)) : 0,
      teams_kept: genStats.teamsKept,
      teams_rejected: genStats.teamsRejected,
      reject_reasons: Object.fromEntries(genStats.rejectReasons),
    } : undefined,
    repro_dirs: cum.reproDirs,
  };
  const sumPath = path.join(flags.out, `summary_${runId}.json`);
  fs.writeFileSync(sumPath, JSON.stringify(summary, null, 2) + '\n');
  appendLog(logPath, `${new Date().toISOString()} run=${runId} DONE ${JSON.stringify({
    battles: cum.battles, ok: cum.ok, diverged: cum.diverged, panic: cum.panic, bph,
  })}`);
  console.error('\n[ab_fuzz] SUMMARY');
  console.error(JSON.stringify(summary, null, 2));

  // GREEN GATE (`--protocol`): FAIL (exit non-zero) on ANY non-allowlisted divergence —
  // a NEW divergence kind/form that matches NO documented residual. Allowlisted residuals
  // (reported by reason above) + ok battles PASS. A non-protocol run keeps exit 0 (it's a
  // hunter, not a gate). `gen3_omniscient_byte_fuzz_v1`.
  if (flags.protocol) {
    const hardFails = cum.diverged + cum.panic + cum.parseError;
    if (hardFails > 0) {
      console.error(`\n[ab_fuzz] GREEN-GATE FAIL: ${hardFails} NON-allowlisted divergence(s) ` +
        `(diverged=${cum.diverged} panic=${cum.panic} parse_error=${cum.parseError}); ` +
        `allowlisted=${cum.allowlisted} by reason ${JSON.stringify(Object.fromEntries(cum.allowlistReasons))}`);
      process.exit(1);
    }
    console.error(`\n[ab_fuzz] GREEN-GATE PASS: 0 non-allowlisted divergences ` +
      `(ok=${cum.ok}, allowlisted=${cum.allowlisted} by reason ` +
      `${JSON.stringify(Object.fromEntries(cum.allowlistReasons))}).`);
  }
  process.exit(0);
}

function appendLog(logPath, line) {
  fs.appendFileSync(logPath, line + '\n');
  console.error(line);
}

// ── Exports (one source of truth for the bridge A/B fuzzer) ──────────────────
// The team providers + the modeled-universe builder are REUSED by
// harness/bridge_ab_fuzz.js (the per-side / request A/B fuzzer), so team
// generation (modeled moves + explicit genders + the randbats adapter) lives in
// exactly ONE place. Additive — `require.main` still runs the omniscient fuzzer.
module.exports = {
  parseFlags, adaptRandbatsTeam, makeRandbatsProvider,
  buildRandomUniverse, makeRandomProvider, makePoolProvider,
  genRandomSet, sampleDistinct, speciesAllowedAbility,
  NATURES, ITEM_CHOICES, RANDBATS_FORMAT,
};

if (require.main === module) {
  main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
}
