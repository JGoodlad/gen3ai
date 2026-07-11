// bridge_ab_fuzz.js — the REQUEST / PER-PLAYER A/B DIFFERENTIAL FUZZER: the
// per-side SIBLING of harness/ab_fuzz.js (which A/Bs the OMNISCIENT stream). It
// verifies, over RANDOM teams, that the Rust crate's PER-SIDE (p1/p2) protocol
// streams + the `|request|` JSON (the poke-env legal-action requests, incl. the
// maybeTrapped/trapped switch-legality state machine) are BYTE-IDENTICAL to the
// real Node Showdown `getPlayerStreams`.
//
// Per chunk it:
//   1. NODE ORACLE — drives a real in-process BattleStream + getPlayerStreams
//      (the local_sim_bridge.js pattern, mirroring gen_bridge_capture.js) to
//      game-end, picking random LEGAL + MODELED choices from a seeded choice-RNG,
//      capturing BOTH per-side chunk streams + the ordered command stream.
//   2. TRAPPING PROBES — when a side's active is TRAPPED (Arena Trap / Magnet Pull
//      / Shadow Tag, detected via the sim's `pokemon.trapped`), sometimes issues a
//      REJECTED `switch` first (→ `|error|` + the `trapped:true` re-request) before
//      the legal move — so the maybeTrapped→trapped machine is exercised over random
//      teams. The `trapping` mode + weighted providers make trapping matchups occur.
//   3. RUST REPLAY + DIFF — drives `bridge_replay` over the identical teams+cmds and
//      diffs the Rust per-side chunks against the Node oracle BYTE-FOR-BYTE, with a
//      first-divergence taxonomy (preamble / perside / privacy / request / error /
//      chunk_count / panic).
//   4. OUTPUT — one stats line per chunk to harness/bridge_ab_fuzz_out/bridge_ab_fuzz.log,
//      a self-contained standalone-replayable repro dir per divergence.
//
// Team generation REUSES ab_fuzz.js's providers (modeled moves + explicit genders +
// the randbats adapter — one source of truth); the TRAPPING provider is new here.
//
// USAGE
//   node src/rust_sim/harness/bridge_ab_fuzz.js
//        [--mode trapping|randbats|random|pool]   (default trapping)
//        [--format gen3customgame|gen3ou]         (default gen3customgame)
//        [--battles N | --hours H]                (default: run until killed)
//        [--master-seed S]                        (default: from time; ALWAYS printed)
//        [--chunk N]                              (default 25 battles per chunk)
//        [--out DIR]                              (default harness/bridge_ab_fuzz_out/)
//        [--keep-chunks]
//        [--trap-prob P]                          (P(issue a rejected switch when trapped), default 0.5)
//
// SIGINT (ctrl-c): finishes the current chunk, prints the cumulative summary. A
// second SIGINT aborts immediately.

'use strict';

const path = require('path');
const fs = require('fs');
const { spawnSync } = require('child_process');

const e2e = require('./gen_e2e_fuzz.js');
const {
  isModeledMove, abilityAllowed, itemAllowed, teamFilterClean, loadTeams,
  mulberry32, randInt, seedFrom, toId, dex3,
} = e2e;
const ab = require('./ab_fuzz.js');

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const ROOT = path.resolve(__dirname, '../../..');
const CRATE = path.resolve(__dirname, '..');
// ISOLATED target dir — never rebuild the shared target/ (the live ab_replay fuzzer).
const BRIDGE_TARGET = '/tmp/pokesim_target_bridge';
const REPLAYER = path.join(BRIDGE_TARGET, 'release/bridge_replay');

function tick() { return new Promise((r) => setTimeout(r, 0)); }

// ── Flags ─────────────────────────────────────────────────────────────────────
function parseFlags(argv) {
  const f = {
    mode: 'trapping',
    format: 'gen3customgame',
    battles: null,
    hours: null,
    masterSeed: null,
    chunk: 25,
    out: path.join(__dirname, 'bridge_ab_fuzz_out'),
    keepChunks: false,
    trapProb: 0.5,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === '--mode') f.mode = next();
    else if (a === '--format') f.format = next();
    else if (a === '--battles') f.battles = Number(next());
    else if (a === '--hours') f.hours = Number(next());
    else if (a === '--master-seed') f.masterSeed = Number(next()) >>> 0;
    else if (a === '--chunk') f.chunk = Number(next());
    else if (a === '--out') f.out = path.resolve(next());
    else if (a === '--keep-chunks') f.keepChunks = true;
    else if (a === '--trap-prob') f.trapProb = Number(next());
    else { console.error(`unknown flag ${a}`); process.exit(2); }
  }
  if (!['trapping', 'randbats', 'random', 'pool'].includes(f.mode)) {
    console.error(`--mode must be trapping|randbats|random|pool, got ${f.mode}`);
    process.exit(2);
  }
  if (!['gen3customgame', 'gen3ou'].includes(f.format)) {
    console.error(`--format must be gen3customgame|gen3ou, got ${f.format}`);
    process.exit(2);
  }
  if (f.masterSeed === null) f.masterSeed = (Date.now() ^ (process.pid * 2654435761)) >>> 0;
  return f;
}

// ── The TRAPPING team provider (new) ─────────────────────────────────────────
// Weights team generation so trapping matchups actually occur: one side leads a
// trapper (Arena Trap Dugtrio/Diglett OR Magnet Pull Magneton/Nosepass OR Shadow
// Tag Wobbuffet/Wynaut), the other side a random modeled-universe team of varied
// FOE TYPES (grounded / Flying / Levitate / Steel / grounded-Ghost) + varied bench
// sizes. Genders pinned explicitly. Movesets are modeled-only (Splash + one modeled
// damaging move) so the ENGINE never fail-louds — the point is the REQUEST layer.
const TRAPPERS = [
  { species: 'Dugtrio', ability: 'Arena Trap', moves: ['earthquake', 'splash'], gender: 'M' },
  { species: 'Diglett', ability: 'Arena Trap', moves: ['earthquake', 'splash'], gender: 'M' },
  { species: 'Magneton', ability: 'Magnet Pull', moves: ['thunderbolt', 'splash'], gender: 'N' },
  { species: 'Nosepass', ability: 'Magnet Pull', moves: ['rockslide', 'splash'], gender: 'M' },
  { species: 'Wobbuffet', ability: 'Shadow Tag', moves: ['splash'], gender: 'M' },
  { species: 'Wynaut', ability: 'Shadow Tag', moves: ['splash'], gender: 'M' },
];
// A spread of foe archetypes to make trap/escape branches realize (grounded /
// Flying-escape-Arena-Trap / Levitate-escape / Steel-trapped-by-MP / grounded Ghost).
const TRAP_FOES = [
  { species: 'Snorlax', moves: ['bodyslam', 'splash'], gender: 'M' },      // grounded → Arena Trap
  { species: 'Zapdos', moves: ['thunderbolt', 'splash'], gender: 'N' },    // Flying → escapes Arena Trap
  { species: 'Gengar', moves: ['icebeam', 'splash'], ability: 'Levitate', gender: 'M' }, // Levitate → escapes
  { species: 'Skarmory', moves: ['drillpeck', 'splash'], ability: 'Keen Eye', gender: 'M' }, // Steel → Magnet Pull
  { species: 'Banette', moves: ['shadowball', 'splash'], gender: 'M' },    // grounded Ghost → Arena Trap traps
  { species: 'Regice', moves: ['icebeam', 'splash'], gender: 'N' },        // grounded, bulky
  { species: 'Metagross', moves: ['meteormash', 'splash'], ability: 'Clear Body', gender: 'N' }, // Steel → MP
  { species: 'Salamence', moves: ['rockslide', 'splash'], ability: 'Intimidate', gender: 'M' }, // Flying
  { species: 'Jirachi', moves: ['bodyslam', 'splash'], ability: 'Serene Grace', gender: 'N' }, // Steel → MP
  { species: 'Suicune', moves: ['surf', 'splash'], ability: 'Pressure', gender: 'N' },     // grounded
];

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mkSet(spec) {
  return {
    name: spec.species, species: spec.species, item: spec.item || 'Leftovers',
    ability: spec.ability || 'No Ability', moves: spec.moves,
    evs: EV0, ivs: IV31, nature: 'Serious', level: 100, gender: spec.gender || 'N',
  };
}

// Build ONE trapping matchup: a 1-mon trapper vs a foe team of 2..4 (varied bench,
// incl. a 1-mon last-mon case so the both-flags-omitted branch realizes). Sometimes
// BOTH sides are trappers (a mirror), so mutual-trap draws are exercised.
function makeTrappingProvider(rng) {
  // The provider hands back a PAIR (both sides) so it can build a coordinated matchup.
  return function nextPair() {
    const trapper = TRAPPERS[randInt(rng, TRAPPERS.length)];
    const mirror = rng() < 0.25; // ~1/4 mirror (mutual trap)
    // Foe bench size 1..4 (1 = last-mon-no-bench branch).
    const nFoes = 1 + randInt(rng, 4);
    const chosen = ab.sampleDistinct(rng, TRAP_FOES, nFoes);
    const p1 = mirror
      ? [mkSet(trapper), ...chosen.slice(0, Math.max(1, nFoes - 1)).map(mkSet)]
      : [mkSet(trapper)];
    const p2 = mirror
      ? [mkSet(TRAPPERS[randInt(rng, TRAPPERS.length)]), ...chosen.slice(0, Math.max(1, nFoes - 1)).map(mkSet)]
      : chosen.map(mkSet);
    // De-dup species within a team (Species Clause is off in customgame, but packed
    // duplicate species confuse the choice/switch slot math — keep teams distinct-species).
    const dedup = (team) => {
      const seen = new Set();
      return team.filter((s) => { const id = toId(s.species); if (seen.has(id)) return false; seen.add(id); return true; });
    };
    const t1 = dedup(p1), t2 = dedup(p2);
    return {
      p1: Teams.pack(t1.length ? t1 : [mkSet(trapper)]),
      p2: Teams.pack(t2.length ? t2 : [mkSet(TRAP_FOES[0])]),
    };
  };
}

// ── Per-side capture driver (the Node oracle) ────────────────────────────────
// Mirrors gen_bridge_capture.js's getPlayerStreams driver, but restricts choices to
// MODELED moves (so the ENGINE stays in scope) and issues TRAPPING PROBES: when a
// side's active is trapped and has live bench, with probability trapProb it first
// writes a REJECTED `switch` (→ |error| + trapped:true re-request) before the legal
// move. Captures both per-side chunk streams + the ordered command list.
async function runBridgeBattle(p1Packed, p2Packed, seed, chooseSeed, format, trapProb) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const chunks = { p1: [], p2: [] };
  const cmds = []; // [side, choice] in write order (includes rejected attempts)

  const pump = (side) => (async () => { for await (const ch of streams[side]) chunks[side].push(ch); })();
  pump('p1'); pump('p2');

  const seedClause = seed ? `,"seed":${JSON.stringify(seed)}` : '';
  streams.omniscient.write(`>start {"formatid":"${format}"${seedClause}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1Packed })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2Packed })}`);
  for (let i = 0; i < 16; i++) await tick();

  const rec = {
    initSeed: stream.battle.prng.getSeed(), chunks, cmds, ended: false, winner: null,
    counts: { trapped: 0, maybeTrapped: 0, trappedTrue: 0, errorFrames: 0, forceSwitch: 0, requests: 0 },
  };

  const rng = mulberry32(chooseSeed);
  const SAFETY = 400;
  let safety = 0;
  while (!stream.battle.ended && safety < SAFETY) {
    safety++;
    const battle = stream.battle;
    const reqState = battle.requestState;
    if (reqState !== 'move' && reqState !== 'switch') { await tick(); continue; }

    if (reqState === 'switch') {
      // Forced replacement(s): the flagged side(s) pick a bench mon.
      for (let i = 0; i < 2; i++) {
        const req = battle.sides[i].activeRequest;
        if (req && req.forceSwitch && req.forceSwitch[0]) {
          const c = pickReplacement(battle, i, rng);
          if (!c) throw new Error(`no legal replacement for p${i + 1} (stall)`);
          cmds.push([`p${i + 1}`, c]);
          try { streams[`p${i + 1}`].write(c); } catch (e) { throw new Error(`write p${i + 1} ${c}: ${e && e.message}`); }
        }
      }
    } else {
      // Move request: BOTH sides pick. A TRAPPED side with live bench may first issue
      // a REJECTED switch probe (the maybeTrapped→trapped machine).
      for (let i = 0; i < 2; i++) {
        const side = `p${i + 1}`;
        const active = battle.sides[i].active[0];
        const isTrapped = !!(active && active.trapped);
        const benchSlots = legalSwitchSlots(battle, i);
        if (isTrapped && benchSlots.length > 0 && rng() < trapProb) {
          // Issue a REJECTED switch to a live bench mon (the sim answers |error| +
          // trapped:true re-request on THIS side; the choice is NOT committed).
          const rejSlot = benchSlots[randInt(rng, benchSlots.length)];
          const rej = `switch ${rejSlot + 1}`;
          cmds.push([side, rej]);
          try { streams[side].write(rej); } catch (e) { throw new Error(`write ${side} ${rej}: ${e && e.message}`); }
          // Let the `|error|` + the `trapped:true` re-request (emitRequest(update=true))
          // fully flush before the legal choice — matching gen_bridge_trapping_capture.js's
          // 16-tick settle (a shorter settle races the re-request → an `[Invalid choice]`
          // parse-time reject with NO re-request, a driver artifact, not a port bug).
          for (let k = 0; k < 16; k++) await tick();
        }
        const c = pickModeledLegal(battle, i, rng, isTrapped);
        if (!c) throw new Error(`no legal choice for p${i + 1} (stall)`);
        cmds.push([side, c]);
        try { streams[side].write(c); } catch (e) { throw new Error(`write ${side} ${c}: ${e && e.message}`); }
      }
    }
    for (let i = 0; i < 16; i++) await tick();
  }
  if (safety >= SAFETY) throw new Error('battle did not advance to game-end (safety cap)');

  rec.ended = !!stream.battle.ended;
  rec.winner = stream.battle.winner;
  for (let i = 0; i < 8; i++) await tick();
  try { streams.omniscient.destroy(); } catch (e) {}
  // Tally the request-shape coverage from the captured streams (both sides).
  for (const side of ['p1', 'p2']) {
    for (const chunk of chunks[side]) {
      for (const raw of chunk.split('\n')) {
        if (raw.startsWith('|error|')) rec.counts.errorFrames++;
        if (!raw.startsWith('|request|')) continue;
        const payload = raw.slice('|request|'.length);
        if (!payload || payload === 'null') continue;
        let obj = null; try { obj = JSON.parse(payload); } catch (e) { continue; }
        if (obj.wait) continue;
        rec.counts.requests++;
        if (obj.forceSwitch) { rec.counts.forceSwitch++; continue; }
        const a = obj.active && obj.active[0];
        if (a && a.trapped) { rec.counts.trapped++; rec.counts.trappedTrue++; }
        else if (a && a.maybeTrapped) { rec.counts.trapped++; rec.counts.maybeTrapped++; }
      }
    }
  }
  return rec;
}

// Pick a MODELED legal choice for one side (respecting the sim's trapped flag).
function pickModeledLegal(battle, side, rng, isTrapped) {
  const req = battle.sides[side].activeRequest;
  if (!req || !req.active || !req.active[0]) return null;
  const moves = req.active[0].moves || [];
  // Legal + MODELED move slots (so the port's ENGINE never fail-louds; the request
  // bytes are what we A/B, but a battle must play to game-end to exercise them).
  const modeledSlots = [];
  for (let k = 0; k < moves.length; k++) {
    if (moves[k].disabled) continue;
    const id = toId(moves[k].id || moves[k].move);
    if (id === 'struggle' || isModeledMove(id)) modeledSlots.push(k);
  }
  const benchSlots = isTrapped ? [] : legalSwitchSlots(battle, side);
  if (modeledSlots.length === 0) {
    // No modeled move usable — switch if we can, else move 1 (Struggle substitute).
    if (benchSlots.length > 0) return `switch ${benchSlots[randInt(rng, benchSlots.length)] + 1}`;
    return 'move 1';
  }
  // Mostly attack; ~1/6 voluntary switch (exercise the switch-request shape).
  if (benchSlots.length > 0 && rng() < 1 / 6) {
    return `switch ${benchSlots[randInt(rng, benchSlots.length)] + 1}`;
  }
  return `move ${modeledSlots[randInt(rng, modeledSlots.length)] + 1}`;
}

function legalSwitchSlots(battle, side) {
  const s = battle.sides[side];
  const out = [];
  for (let k = 0; k < s.pokemon.length; k++) {
    if (s.pokemon[k] !== s.active[0] && !s.pokemon[k].fainted) out.push(k);
  }
  return out;
}
function pickReplacement(battle, side, rng) {
  const slots = legalSwitchSlots(battle, side);
  if (slots.length === 0) return null;
  return `switch ${slots[randInt(rng, slots.length)] + 1}`;
}

// ── Golden emission (the bridge TAB grammar — same as gen_bridge_capture.js) ──
function emitBridgeBattle(lines, id, battleNo, p1Packed, p2Packed, format, rec) {
  lines.push(`SCEN\t${id}`);
  lines.push(`TEAM\t${id}\tp1\t${p1Packed}`);
  lines.push(`TEAM\t${id}\tp2\t${p2Packed}`);
  lines.push(['INIT', id, battleNo, rec.initSeed, format].join('\t'));
  rec.cmds.forEach((c, ci) => lines.push(['CMD', id, battleNo, ci, c[0], c[1]].join('\t')));
  for (const side of ['p1', 'p2']) {
    rec.chunks[side].forEach((chunk, chunkNo) => {
      let lineNo = 0;
      chunk.split('\n').forEach((rawLine) => {
        // `|debug|` is a poke-env-IGNORED free-form sim line (gen3customgame sets
        // debug:true → it emits `|debug|doubling secondary chance` etc.). The bridge
        // emitter (bridge.rs) deliberately never emits it, and poke-env drops it, so it
        // is filtered from the golden here (the same class as the `|t:|` normalization —
        // bridge_test.rs's gated-line convention). A blank line is also dropped.
        if (rawLine.startsWith('|debug|') || rawLine === '') return;
        const raw = rawLine.startsWith('|t:|') ? '|t:|<NORMALIZED>' : rawLine;
        lines.push(['CHUNK', id, battleNo, side, chunkNo, lineNo, raw].join('\t'));
        lineNo++;
      });
    });
  }
  lines.push(['END', id, battleNo, rec.ended ? 1 : 0, winTok(rec)].join('\t'));
}
function winTok(rec) {
  if (!rec.ended) return 'none';
  if (rec.winner === 'P1') return 'p1';
  if (rec.winner === 'P2') return 'p2';
  if (rec.winner === '') return 'tie';
  return 'none';
}

function chunkHeader(flags, runId, chunkIdx) {
  return [
    '# bridge_ab_fuzz chunk — REQUEST / PER-SIDE A/B fuzzer (real Node getPlayerStreams vs the Rust port).',
    `# mode=${flags.mode} format=${flags.format} master_seed=${flags.masterSeed} run_id=${runId} chunk=${chunkIdx}`,
    '# Format identical to tests/vectors/bridge_trapping_golden.txt (SCEN/TEAM/INIT/CMD/CHUNK/END).',
    '# Replay: /tmp/pokesim_target_bridge/release/bridge_replay <this-file>',
  ];
}

function saveRepro(outDir, runId, flags, meta, verdict, chunkIdx) {
  const dir = path.join(outDir, 'divergences', `${runId}_${meta.id}`);
  fs.mkdirSync(dir, { recursive: true });
  const lines = chunkHeader(flags, runId, chunkIdx);
  emitBridgeBattle(lines, meta.id, 0, meta.p1Packed, meta.p2Packed, flags.format, meta.rec);
  fs.writeFileSync(path.join(dir, 'battle.txt'), lines.join('\n') + '\n');
  const summary = {
    mode: flags.mode, format: flags.format, master_seed: flags.masterSeed, run_id: runId,
    battle_id: meta.id, init_seed: meta.rec.initSeed, choose_seed: meta.chooseSeed,
    packed_teams: { p1: meta.p1Packed, p2: meta.p2Packed },
    cmds: meta.rec.cmds, counts: meta.rec.counts,
    first_divergence: {
      verdict: verdict.verdict, kind: verdict.kind || verdict.verdict, side: verdict.side ?? null,
      line_index: verdict.line === undefined ? null : verdict.line,
      expected: verdict.expected === undefined ? null : verdict.expected,
      got: verdict.got === undefined ? null : verdict.got,
      detail: verdict.detail === undefined ? null : verdict.detail,
    },
    replay_cmd: `/tmp/pokesim_target_bridge/release/bridge_replay ${path.join(dir, 'battle.txt')}`,
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
  const logPath = path.join(flags.out, 'bridge_ab_fuzz.log');

  console.error(`[bridge_ab_fuzz] run_id=${runId} mode=${flags.mode} format=${flags.format} ` +
    `master_seed=${flags.masterSeed} chunk=${flags.chunk} battles=${flags.battles ?? '∞'} ` +
    `trap_prob=${flags.trapProb} out=${flags.out}`);
  console.error('[bridge_ab_fuzz] reproduce: ' +
    `node src/rust_sim/harness/bridge_ab_fuzz.js --mode ${flags.mode} --format ${flags.format} ` +
    `--master-seed ${flags.masterSeed}` + (flags.battles ? ` --battles ${flags.battles}` : ''));

  // Build the Rust replayer ONCE into the ISOLATED target dir (never the shared target/).
  {
    const env = { ...process.env, PATH: `${process.env.HOME}/.cargo/bin:${process.env.PATH}`, CARGO_TARGET_DIR: BRIDGE_TARGET };
    const r = spawnSync('cargo', ['build', '--release', '--bin', 'bridge_replay'], { cwd: CRATE, env, stdio: 'inherit' });
    if (r.status !== 0) { console.error('[bridge_ab_fuzz] cargo build failed'); process.exit(1); }
  }
  if (!fs.existsSync(REPLAYER)) { console.error(`[bridge_ab_fuzz] replayer missing: ${REPLAYER}`); process.exit(1); }

  // Deterministic streams from the master seed.
  const teamRng = mulberry32(flags.masterSeed);
  const battleRng = mulberry32((flags.masterSeed ^ 0xabcdef01) >>> 0);

  const genStats = {
    setsTotal: 0, setsAdjusted: 0, naturesNormalized: 0, teamsKept: 0, teamsRejected: 0, genErrors: 0,
    rejectReasons: new Map(),
  };
  // A pair provider hands back { p1, p2 } packed teams for one battle. trapping mode
  // builds a coordinated matchup; the others reuse ab_fuzz.js's single-team providers.
  let pairProvider;
  if (flags.mode === 'trapping') {
    const tp = makeTrappingProvider(teamRng);
    pairProvider = () => tp();
  } else {
    let single;
    if (flags.mode === 'randbats') single = ab.makeRandbatsProvider(teamRng, genStats);
    else if (flags.mode === 'pool') single = ab.makePoolProvider(teamRng, genStats);
    else {
      const universe = ab.buildRandomUniverse();
      console.error(`[random] modeled universe: ${universe.eligible.length} species`);
      single = ab.makeRandomProvider(teamRng, universe, genStats);
    }
    // Pin an EXPLICIT gender on EVERY set whose pack leaves it empty (the task's
    // CRITICAL LESSON). Two sim behaviours the port doesn't re-derive from a
    // gender-less pack:
    //   * a TRUE-ratio species (`sp.gender === ''`) → the sim DRAWS a construction-time
    //     `sample(['M','F'])`, an unmodeled init draw → pin 'M'.
    //   * a FIXED-gender species (`sp.gender === 'M'|'F'|'N'`, e.g. Nidoking=M, Latios=M,
    //     Magneton=N) → the sim CANONICALIZES `details` to `<Species>, M` while the port
    //     reads the empty pack field → pin the species' fixed letter so both agree.
    // The `random` provider (unlike `randbats`) leaves gender unset; this pass is
    // idempotent on the already-gender-safe `randbats`/`pool`/`trapping` teams.
    const pinGenders = (packed) => {
      const team = Teams.unpack(packed);
      if (!team) return packed;
      let touched = false;
      for (const set of team) {
        if (set.gender) continue;
        const sid = toId(set.species || set.name);
        const sp = dex3.species.get(sid);
        if (!sp || !sp.exists) continue;
        set.gender = sp.gender === '' ? 'M' : sp.gender; // '' → 'M'; else the fixed letter
        touched = true;
      }
      return touched ? Teams.pack(team) : packed;
    };
    pairProvider = () => ({ p1: pinGenders(single().packed), p2: pinGenders(single().packed) });
  }

  const cum = {
    battles: 0, ok: 0, diverged: 0, panic: 0, parseError: 0, empty: 0, ended: 0,
    kinds: new Map(), dropReasons: new Map(), reproDirs: [], chunkErrors: 0,
    cov: { trapped: 0, maybeTrapped: 0, trappedTrue: 0, errorFrames: 0, forceSwitch: 0, requests: 0 },
    speciesRoster: new Set(),
  };

  let stopRequested = false;
  process.on('SIGINT', () => {
    if (stopRequested) { console.error('\n[bridge_ab_fuzz] second SIGINT — aborting'); process.exit(130); }
    stopRequested = true;
    console.error('\n[bridge_ab_fuzz] SIGINT — finishing the current chunk, then summarizing…');
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
    const metas = new Map();
    let produced = 0;

    try {
      const target = flags.battles !== null
        ? Math.min(flags.chunk, Math.max(1, flags.battles - cum.battles))
        : flags.chunk;
      for (let i = 0; i < target && !stopRequested; i++) {
        const pair = pairProvider();
        const battleSeed = seedFrom((Math.floor(battleRng() * 4294967296)) >>> 0);
        const chooseSeed = (Math.floor(battleRng() * 4294967296) ^ 0x9e3779b9) >>> 0;
        let rec;
        try {
          rec = await runBridgeBattle(pair.p1, pair.p2, battleSeed, chooseSeed, flags.format, flags.trapProb);
        } catch (e) {
          cum.empty++;
          cum.dropReasons.set(String(e && e.message || e).split(' ')[0], (cum.dropReasons.get(String(e && e.message || e).split(' ')[0]) || 0) + 1);
          continue;
        }
        if (!rec.initSeed || (rec.chunks.p1.length === 0 && rec.chunks.p2.length === 0)) {
          cum.empty++;
          continue;
        }
        if (rec.ended) cum.ended++;
        const id = `bab_${chunkIdx}_${i}`;
        emitBridgeBattle(lines, id, 0, pair.p1, pair.p2, flags.format, rec);
        metas.set(id, { id, p1Packed: pair.p1, p2Packed: pair.p2, rec, chooseSeed });
        produced++;
        cum.battles++;
        for (const k of Object.keys(cum.cov)) cum.cov[k] += rec.counts[k] || 0;
        for (const packed of [pair.p1, pair.p2]) {
          for (const set of Teams.unpack(packed)) cum.speciesRoster.add(toId(set.species || set.name));
        }
      }

      if (produced === 0) {
        if (stopRequested) break;
        cum.chunkErrors++;
        appendLog(logPath, `${new Date().toISOString()} chunk=${chunkIdx} EMPTY (no replayable battles)`);
        chunkIdx++;
        continue;
      }

      const chunkFile = path.join(flags.out, 'chunks', `${runId}_chunk${chunkIdx}.txt`);
      fs.writeFileSync(chunkFile, lines.join('\n') + '\n');

      let verdicts = [];
      // spawnSync (NOT execFileSync) — the replayer exits NON-ZERO whenever a battle
      // diverged, and execFileSync THROWS on non-zero (losing stdout). We need the
      // per-battle verdict lines regardless of exit code.
      const proc = spawnSync(REPLAYER, [chunkFile, '--ab'], { maxBuffer: 256 * 1024 * 1024 });
      if (proc.error) throw proc.error;
      const stdout = (proc.stdout || Buffer.from('')).toString();
      for (const line of stdout.split('\n')) {
        if (!line.trim() || !line.startsWith('{')) continue;
        try { const v = JSON.parse(line); if (v.chunk_summary) continue; verdicts.push(v); } catch (e) {}
      }

      let chunkOk = 0; let chunkDiv = 0;
      for (const v of verdicts) {
        if (v.verdict === 'ok') { cum.ok++; chunkOk++; continue; }
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
          console.error(`[DIVERGED] ${v.battle} kind=${kind} side=${v.side ?? '-'} line=${v.line ?? '-'} → ${dir}`);
        }
      }

      if (!flags.keepChunks) {
        if (lastChunkFile && fs.existsSync(lastChunkFile)) fs.unlinkSync(lastChunkFile);
        lastChunkFile = chunkFile;
      }

      const hrs = (Date.now() - t0) / 3600000;
      const bph = hrs > 0 ? Math.round(cum.battles / hrs) : 0;
      const kindsStr = [...cum.kinds.entries()].map(([k, c]) => `${k}=${c}`).join(',') || '-';
      appendLog(logPath,
        `${new Date().toISOString()} run=${runId} mode=${flags.mode} fmt=${flags.format} chunk=${chunkIdx} ` +
        `battles=${produced} ok=${chunkOk} diverged=${chunkDiv} ` +
        `cum_battles=${cum.battles} cum_ok=${cum.ok} cum_diverged=${cum.diverged} cum_panic=${cum.panic} ` +
        `cum_empty=${cum.empty} kinds=${kindsStr} ` +
        `trapped=${cum.cov.trapped} maybeTrapped=${cum.cov.maybeTrapped} trappedTrue=${cum.cov.trappedTrue} ` +
        `error=${cum.cov.errorFrames} forceSwitch=${cum.cov.forceSwitch} requests=${cum.cov.requests} ` +
        `species=${cum.speciesRoster.size} bph=${bph} chunk_s=${((Date.now() - chunkT0) / 1000).toFixed(1)}`);
    } catch (e) {
      cum.chunkErrors++;
      appendLog(logPath, `${new Date().toISOString()} chunk=${chunkIdx} ERROR ${String(e && e.message || e).slice(0, 300)}`);
      console.error(`[chunk ${chunkIdx} ERROR]`, e && e.stack ? e.stack.split('\n').slice(0, 4).join('\n') : e);
    }
    chunkIdx++;
  }

  const hrs = (Date.now() - t0) / 3600000;
  const bph = hrs > 0 ? Math.round(cum.battles / hrs) : 0;
  const summary = {
    run_id: runId, mode: flags.mode, format: flags.format, master_seed: flags.masterSeed,
    elapsed_hours: Number(hrs.toFixed(3)), battles: cum.battles, battles_per_hour: bph,
    ok: cum.ok, diverged: cum.diverged, panic: cum.panic, parse_error: cum.parseError,
    ended_battles: cum.ended, empty_skipped: cum.empty, chunk_errors: cum.chunkErrors,
    divergence_kinds: Object.fromEntries(cum.kinds),
    drop_reasons: Object.fromEntries(cum.dropReasons),
    trapping_coverage: cum.cov,
    distinct_species_rostered: cum.speciesRoster.size,
    repro_dirs: cum.reproDirs,
  };
  fs.writeFileSync(path.join(flags.out, `summary_${runId}.json`), JSON.stringify(summary, null, 2) + '\n');
  appendLog(logPath, `${new Date().toISOString()} run=${runId} DONE ${JSON.stringify({
    battles: cum.battles, ok: cum.ok, diverged: cum.diverged, panic: cum.panic, bph,
  })}`);
  console.error('\n[bridge_ab_fuzz] SUMMARY');
  console.error(JSON.stringify(summary, null, 2));
  process.exit(0);
}

function appendLog(logPath, line) {
  fs.appendFileSync(logPath, line + '\n');
  console.error(line);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
