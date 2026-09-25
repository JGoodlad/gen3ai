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
//        [--mode trapping|randbats|random|pool|ladder]   (default trapping)
//        [--ladder-tier commit|milestone|full]   (`--mode ladder`: the LADDER-USAGE corpus tier, default milestone)
//        [--format gen3customgame|gen3ou]         (default gen3customgame)
//        [--battles N | --hours H]                (default: run until killed)
//        [--master-seed S]                        (default: from time; ALWAYS printed)
//        [--chunk N]                              (default 25 battles per chunk)
//        [--out DIR]                              (default harness/bridge_ab_fuzz_out/)
//        [--keep-chunks]
//        [--trap-prob P]                          (P(issue a rejected switch when trapped), default 0.5)
//        [--selftest]                             (allowlist GATE-INTEGRITY injections through the
//                                                  real replayer, then exit — see selftest() below)
//
// SIGINT (ctrl-c): finishes the current chunk, prints the cumulative summary. A
// second SIGINT aborts immediately.

'use strict';

const path = require('path');
const fs = require('fs');
const os = require('os');
const { spawnSync } = require('child_process');

const e2e = require('./gen_e2e_fuzz.js');
const {
  isModeledMove, abilityAllowed, itemAllowed, teamFilterClean, loadTeams,
  mulberry32, randInt, seedFrom, toId, dex3,
} = e2e;
const ab = require('./ab_fuzz.js');
const ladderCorpus = require('./ladder_corpus.js');
// `--mode ladder` widens the picker to typed Hidden Power: the corpus is gen3ou-VALIDATED and the
// engine prices HP at its IV-true BP (R3). Every other mode keeps `isModeledMove`'s default.
let PICK_HIDDEN_POWER = false;

const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const ROOT = path.resolve(__dirname, '../../..');
const CRATE = path.resolve(__dirname, '..');
// ISOLATED target dir — never rebuild the shared target/ (the live ab_replay fuzzer).
// OVERRIDABLE via `POKESIM_BRIDGE_TARGET` (`gen3_transform_v1`, ROUND 33): the hard-coded
// path is a SHARED /tmp dir, so a second worktree running this concurrently would rebuild it
// from ITS source — which is exactly why ROUND 32 had to SKIP this gate and record the miss
// as honest scope. An env override lets each worktree run the per-side gate in its own dir;
// the default is unchanged, so existing invocations are byte-identical.
const BRIDGE_TARGET = process.env.POKESIM_BRIDGE_TARGET || '/tmp/pokesim_target_bridge';
// The EMISSION SELF-CHECK build (`gen3_core_emission_selfcheck_v1`, `src/emission_check.rs`): every
// emitted line is checked at the moment it is emitted, a failure is a `panic` verdict. Its own
// profile directory (`selfcheck/`), so it never overwrites a production `release/` binary.
const REPLAYER = path.join(BRIDGE_TARGET, 'selfcheck/bridge_replay');

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
    ladderTier: 'milestone',
    selftest: false,
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
    else if (a === '--ladder-tier') f.ladderTier = next();
    else if (a === '--selftest') f.selftest = true;
    else { console.error(`unknown flag ${a}`); process.exit(2); }
  }
  if (!['trapping', 'randbats', 'random', 'pool', 'ladder'].includes(f.mode)) {
    console.error(`--mode must be trapping|randbats|random|pool|ladder, got ${f.mode}`);
    process.exit(2);
  }
  if (!ladderCorpus.TIERS.includes(f.ladderTier)) {
    console.error(`--ladder-tier must be ${ladderCorpus.TIERS.join('|')}, got ${f.ladderTier}`);
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
    initSeed: stream.battle.prng.getSeed(),
    quickClawRoll: !!stream.battle.quickClawRoll,
    chunks, cmds, ended: false, winner: null,
    // SEED ANCHOR: the omniscient PRNG seed AFTER each RESOLVED decision boundary (one per
    // committed decision, in order) — the independent post-decision seed assertion the Rust
    // `bridge_replay --ab` checks against `run_full_battle`'s per-decision engine seed BEFORE
    // the per-side byte diff, so a per-side/request divergence is partitioned into an upstream
    // engine desync vs a genuine per-side/request-serializer bug.
    seeds: [],
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
    // Capture the post-decision seed for THIS resolved boundary (one per committed decision,
    // aligned 1:1 with the Rust bridge's ScriptDecision list / run_full_battle decisions).
    rec.seeds.push(stream.battle.prng.getSeed());
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
    if (id === 'struggle' || isModeledMove(id, PICK_HIDDEN_POWER)) modeledSlots.push(k);
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
  // Turn-0 `quickClawRoll` (`gen3_turn0_quick_claw_capture_v1`): `initSeed` is the
  // POST-construction seed, so the offline Rust replay skips the turn-0 endTurn that decides
  // turn 1's Quick Claw. Optional 6th INIT field (absent -> false) so old goldens still replay.
  lines.push(['INIT', id, battleNo, rec.initSeed, format, rec.quickClawRoll ? 1 : 0].join('\t'));
  rec.cmds.forEach((c, ci) => lines.push(['CMD', id, battleNo, ci, c[0], c[1]].join('\t')));
  // SEED ANCHOR rows — one per RESOLVED decision boundary, in order.
  (rec.seeds || []).forEach((s, di) => lines.push(['SEED', id, battleNo, di, s].join('\t')));
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
    `# Replay: ${REPLAYER} <this-file>`,
  ];
}

function saveRepro(outDir, runId, flags, meta, verdict, chunkIdx, subdir) {
  const dir = path.join(outDir, subdir || 'divergences', `${runId}_${meta.id}`);
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
    replay_cmd: `${REPLAYER} ${path.join(dir, 'battle.txt')}`,
  };
  fs.writeFileSync(path.join(dir, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
  return dir;
}

// Build the Rust replayer ONCE into the ISOLATED target dir (never the shared target/).
function buildReplayer() {
  const env = { ...process.env, PATH: `${process.env.HOME}/.cargo/bin:${process.env.PATH}`, CARGO_TARGET_DIR: BRIDGE_TARGET };
  const r = spawnSync('cargo', ['build', '--profile', 'selfcheck', '--features', 'emission-selfcheck', '--bin', 'bridge_replay'], { cwd: CRATE, env, stdio: 'inherit' });
  if (r.status !== 0) { console.error('[bridge_ab_fuzz] cargo build failed'); process.exit(1); }
  if (!fs.existsSync(REPLAYER)) { console.error(`[bridge_ab_fuzz] replayer missing: ${REPLAYER}`); process.exit(1); }
}

// The GREEN-GATE accounting of ONE replayer verdict: `ok`, `allowlisted` (a documented residual
// the Rust replayer NARROWLY classified — counted separately, never a failure), or `hard` (every
// other outcome, which fails the run). Only a `diverge` verdict can be allowlisted: an
// `allowlisted` field on a panic / error / parse_error is NOT honoured.
function verdictClass(v) {
  if (v.verdict === 'ok') return 'ok';
  if (v.verdict === 'diverge' && typeof v.allowlisted === 'string' && v.allowlisted.length > 0) return 'allowlisted';
  return 'hard';
}

// ── Main loop ─────────────────────────────────────────────────────────────────
async function main() {
  const flags = parseFlags(process.argv);
  if (flags.selftest) return selftest();
  const runId = `r${Date.now().toString(36)}`;
  const t0 = Date.now();

  fs.mkdirSync(path.join(flags.out, 'chunks'), { recursive: true });
  fs.mkdirSync(path.join(flags.out, 'divergences'), { recursive: true });
  fs.mkdirSync(path.join(flags.out, 'allowlisted'), { recursive: true });
  const logPath = path.join(flags.out, 'bridge_ab_fuzz.log');

  console.error(`[bridge_ab_fuzz] run_id=${runId} mode=${flags.mode} format=${flags.format} ` +
    `master_seed=${flags.masterSeed} chunk=${flags.chunk} battles=${flags.battles ?? '∞'} ` +
    `trap_prob=${flags.trapProb} out=${flags.out}`);
  console.error('[bridge_ab_fuzz] reproduce: ' +
    `node src/rust_sim/harness/bridge_ab_fuzz.js --mode ${flags.mode} --format ${flags.format} ` +
    `--master-seed ${flags.masterSeed}` + (flags.mode === 'ladder' ? ` --ladder-tier ${flags.ladderTier}` : '') +
    (flags.battles ? ` --battles ${flags.battles}` : ''));

  buildReplayer();

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
    else if (flags.mode === 'ladder') {
      single = ladderCorpus.makeLadderProvider(teamRng, genStats, flags.ladderTier, teamFilterClean);
      PICK_HIDDEN_POWER = true;
    }
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
    battles: 0, ok: 0, diverged: 0, allowlisted: 0, panic: 0, parseError: 0, empty: 0, ended: 0,
    kinds: new Map(), allowKinds: new Map(), dropReasons: new Map(), reproDirs: [], chunkErrors: 0,
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

      let chunkOk = 0; let chunkDiv = 0; let chunkAllow = 0;
      for (const v of verdicts) {
        const cls = verdictClass(v);
        if (cls === 'ok') { cum.ok++; chunkOk++; continue; }
        // A documented residual (a request-DISPLAY deferral, or one of the turn-0 construction
        // speed-tie per-side keys) is `allowlisted` NARROWLY by the Rust replayer — counted
        // SEPARATELY, saved under allowlisted/, and NOT a gate failure (mirrors ab_fuzz.js --protocol).
        if (cls === 'allowlisted') {
          cum.allowlisted++; chunkAllow++;
          cum.allowKinds.set(v.allowlisted, (cum.allowKinds.get(v.allowlisted) || 0) + 1);
          const meta = metas.get(v.battle);
          if (meta) saveRepro(flags.out, runId, flags, meta, v, chunkIdx, 'allowlisted');
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
          const dir = saveRepro(flags.out, runId, flags, meta, v, chunkIdx, 'divergences');
          cum.reproDirs.push(dir);
          console.error(`[DIVERGED] ${v.battle} kind=${kind} side=${v.side ?? '-'} line=${v.line ?? '-'} dec=${v.decision ?? '-'} → ${dir}`);
        }
      }

      if (!flags.keepChunks) {
        if (lastChunkFile && fs.existsSync(lastChunkFile)) fs.unlinkSync(lastChunkFile);
        lastChunkFile = chunkFile;
      }

      const hrs = (Date.now() - t0) / 3600000;
      const bph = hrs > 0 ? Math.round(cum.battles / hrs) : 0;
      const kindsStr = [...cum.kinds.entries()].map(([k, c]) => `${k}=${c}`).join(',') || '-';
      const allowStr = [...cum.allowKinds.entries()].map(([k, c]) => `${k}=${c}`).join(',') || '-';
      appendLog(logPath,
        `${new Date().toISOString()} run=${runId} mode=${flags.mode} fmt=${flags.format} chunk=${chunkIdx} ` +
        `battles=${produced} ok=${chunkOk} diverged=${chunkDiv} allowlisted=${chunkAllow} ` +
        `cum_battles=${cum.battles} cum_ok=${cum.ok} cum_diverged=${cum.diverged} cum_allowlisted=${cum.allowlisted} cum_panic=${cum.panic} ` +
        `cum_empty=${cum.empty} kinds=${kindsStr} allow=${allowStr} ` +
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
    ok: cum.ok, diverged: cum.diverged, allowlisted: cum.allowlisted, panic: cum.panic, parse_error: cum.parseError,
    ended_battles: cum.ended, empty_skipped: cum.empty, chunk_errors: cum.chunkErrors,
    divergence_kinds: Object.fromEntries(cum.kinds),
    allowlisted_kinds: Object.fromEntries(cum.allowKinds),
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
  // GREEN GATE: exit non-zero ONLY on a NON-allowlisted diverge / panic / parse_error
  // (mirrors ab_fuzz.js --protocol). Allowlisted request-DISPLAY deferrals do NOT fail.
  const hardFail = cum.diverged + cum.panic + cum.parseError;
  process.exit(hardFail > 0 ? 1 : 0);
}

// ── ALLOWLIST GATE-INTEGRITY SELF-TEST (`node bridge_ab_fuzz.js --selftest`) ─────────────────
// The per-side allowlist lives in the Rust replayer (`bridge_replay.rs` + its
// `bridge_replay/switchin_block_swap.rs`), and its `#[cfg(test)]` modules pin each clause in
// isolation. This is the END-TO-END half: FAULT INJECTIONS into real, committed repro fixtures,
// replayed through the SAME `bridge_replay --ab` binary and the SAME `verdictClass` accounting the
// fuzzer uses. The POSITIVES show every tagged construction-window fixture still resolves to its
// reason; the NEGATIVES — the load-bearing half — show that a CONTENT change inside the reordered
// window, a dropped / extra / internally-reordered line, a non-tie lead, and a divergence LATER in
// an otherwise-allowlisted battle each still FAIL the gate. Run it after ANY change to a per-side
// allowlist clause. Every mangled golden is written to a temp dir, never into the corpus.
function selftest() {
  buildReplayer();
  const corpus = path.join(CRATE, 'tests/vectors/bridge_corpus');
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'bridge_ab_selftest_'));
  const BLOCK_SWAP = 'turn0-construction-speed-tie-switchin-block-swap';
  let fail = 0; let n = 0;
  const check = (label, got, want) => {
    n++;
    const ok = JSON.stringify(got) === JSON.stringify(want);
    if (!ok) { fail++; console.error(`  FAIL ${label}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`); }
    else console.error(`  ok   ${label}`);
  };
  // Replay ONE single-battle golden text and reduce its verdict to what the gate sees.
  let k = 0;
  const replay = (text) => {
    const file = path.join(tmp, `case${k++}.txt`);
    fs.writeFileSync(file, text);
    const proc = spawnSync(REPLAYER, [file, '--ab'], { maxBuffer: 256 * 1024 * 1024 });
    const lines = (proc.stdout || Buffer.from('')).toString().split('\n').filter((l) => l.startsWith('{') && !l.includes('"chunk_summary"'));
    if (lines.length !== 1) return { cls: 'no-verdict', n: lines.length };
    const v = JSON.parse(lines[0]);
    return { cls: verdictClass(v), kind: v.kind || v.verdict, allowlisted: v.allowlisted ?? null, side: v.side ?? null, line: v.line ?? null };
  };
  const read = (name) => fs.readFileSync(path.join(corpus, name), 'utf8');

  // ── Row helpers over the SCEN/TEAM/INIT/CMD/SEED/CHUNK/END grammar ──
  const rawOf = (row) => row.split('\t').slice(6).join('\t');
  const withRaw = (row, raw) => [...row.split('\t').slice(0, 6), raw].join('\t');
  const isChunk = (row, side) => row.startsWith('CHUNK\t') && row.split('\t')[3] === side;
  // Indices (into `rows`) of one side's CHUNK rows, split at that side's first `|turn|` row.
  const sideRows = (rows, side) => {
    const idx = rows.map((r, i) => (isChunk(r, side) ? i : -1)).filter((i) => i >= 0);
    const t = idx.findIndex((i) => rawOf(rows[i]).startsWith('|turn|'));
    if (t < 0) throw new Error(`selftest: no |turn| row on ${side}`);
    return { window: idx.slice(0, t), rest: idx.slice(t) };
  };
  // Apply `edit(rows)` to a fixture's rows; `edit` mutates in place.
  const mangle = (text, edit) => { const rows = text.split('\n'); edit(rows); return rows.join('\n'); };
  const findRow = (rows, idxs, pred, what) => {
    const i = idxs.find((j) => pred(rawOf(rows[j])));
    if (i === undefined) throw new Error(`selftest: precondition — no row matching ${what}`);
    return i;
  };

  const f21 = read('21_construction_switchin_block_swap_p1_ou.txt');
  const f22 = read('22_construction_switchin_block_swap_p2_ou.txt');
  const INTIM = '|-ability|p2a: Salamence|Intimidate|boost';
  const UNBOOST = '|-unboost|p1a: Zapdos|atk|1';
  const PRESSURE = '|-ability|p1a: Zapdos|Pressure|[silent]';

  // ── The accounting itself ──
  check('verdictClass: ok', verdictClass({ verdict: 'ok' }), 'ok');
  check('verdictClass: a diverge carrying a reason is allowlisted', verdictClass({ verdict: 'diverge', allowlisted: BLOCK_SWAP }), 'allowlisted');
  check('verdictClass: a diverge with NO reason is a hard failure', verdictClass({ verdict: 'diverge', kind: 'perside' }), 'hard');
  check('verdictClass: an EMPTY reason is a hard failure', verdictClass({ verdict: 'diverge', allowlisted: '' }), 'hard');
  check('verdictClass: a panic is never allowlisted', verdictClass({ verdict: 'panic', allowlisted: BLOCK_SWAP }), 'hard');

  // ── POSITIVES: every tagged construction-window fixture still resolves to its reason ──
  for (const name of fs.readdirSync(corpus).filter((f) => f.endsWith('.txt')).sort()) {
    const text = read(name);
    const tag = (text.split('\n').find((l) => l.trim().startsWith('# ALLOWLIST ')) || '').trim().slice('# ALLOWLIST '.length).trim();
    if (!tag) continue;
    const r = replay(text);
    check(`POSITIVE ${name} -> ${tag}`, [r.cls, r.allowlisted], ['allowlisted', tag]);
  }
  // The precondition every negative below leans on: fixture 21's p1 window really is the
  // [Intimidate, unboost, Pressure] block order the sim wrote.
  {
    const rows = f21.split('\n');
    const w = sideRows(rows, 'p1').window.map((i) => rawOf(rows[i]));
    check('precondition: fixture 21 p1 window ends [INTIM, UNBOOST, PRESSURE]', w.slice(-3), [INTIM, UNBOOST, PRESSURE]);
  }

  // ── NEGATIVES: each must FAIL the gate (a `diverge` with NO allowlisted reason) ──
  const HARD = (r) => [r.cls, r.allowlisted];
  const WANT = ['hard', null];
  const neg = (label, text) => {
    const r = replay(text);
    check(`NEGATIVE ${label}  [kind=${r.kind} side=${r.side} line=${r.line}]`, HARD(r), WANT);
  };

  neg('content change inside the reordered window (-unboost atk|1 -> atk|2)', mangle(f21, (rows) => {
    const { window } = sideRows(rows, 'p1');
    const i = findRow(rows, window, (r) => r === UNBOOST, 'the p1 -unboost');
    rows[i] = withRaw(rows[i], '|-unboost|p1a: Zapdos|atk|2');
  }));
  neg('content change inside the reordered window (Pressure -> Insomnia)', mangle(f21, (rows) => {
    const { window } = sideRows(rows, 'p1');
    const i = findRow(rows, window, (r) => r === PRESSURE, 'the p1 Pressure line');
    rows[i] = withRaw(rows[i], '|-ability|p1a: Zapdos|Insomnia|[silent]');
  }));
  neg('mis-targeted -unboost (the Intimidator\'s own slot)', mangle(f21, (rows) => {
    const { window } = sideRows(rows, 'p1');
    const i = findRow(rows, window, (r) => r === UNBOOST, 'the p1 -unboost');
    rows[i] = withRaw(rows[i], '|-unboost|p2a: Salamence|atk|1');
  }));
  neg('a DROPPED window line (the golden loses Pressure)', mangle(f21, (rows) => {
    const { window } = sideRows(rows, 'p1');
    rows.splice(findRow(rows, window, (r) => r === PRESSURE, 'the p1 Pressure line'), 1);
  }));
  neg('an EXTRA window line (a duplicated -unboost)', mangle(f21, (rows) => {
    const { window } = sideRows(rows, 'p1');
    const i = findRow(rows, window, (r) => r === UNBOOST, 'the p1 -unboost');
    rows.splice(i + 1, 0, rows[i]);
  }));
  neg('the Intimidate pair internally REORDERED (multiset preserved: -unboost before its announce)', mangle(f21, (rows) => {
    const { window } = sideRows(rows, 'p1');
    const a = findRow(rows, window, (r) => r === INTIM, 'the p1 Intimidate line');
    const b = findRow(rows, window, (r) => r === UNBOOST, 'the p1 -unboost');
    const ra = rawOf(rows[a]); rows[a] = withRaw(rows[a], rawOf(rows[b])); rows[b] = withRaw(rows[b], ra);
  }));
  neg('the two lead |switch| lines swapped (a non-framing reorder)', mangle(f21, (rows) => {
    const { window } = sideRows(rows, 'p1');
    const a = findRow(rows, window, (r) => r.startsWith('|switch|p1a: '), 'the p1a switch');
    const b = findRow(rows, window, (r) => r.startsWith('|switch|p2a: '), 'the p2a switch');
    const ra = rawOf(rows[a]); rows[a] = withRaw(rows[a], rawOf(rows[b])); rows[b] = withRaw(rows[b], ra);
  }));
  neg('a content change in the OTHER side\'s window (p2 -unboost atk|1 -> atk|2)', mangle(f21, (rows) => {
    const { window } = sideRows(rows, 'p2');
    const i = findRow(rows, window, (r) => r === UNBOOST, 'the p2 -unboost');
    rows[i] = withRaw(rows[i], '|-unboost|p1a: Zapdos|atk|2');
  }));
  neg('a NON-TIE lead (Salamence Spe EV 252 -> 248: 327 vs Zapdos 328)', mangle(f21, (rows) => {
    const i = rows.findIndex((r) => r.startsWith('TEAM\t') && r.split('\t')[2] === 'p2');
    const before = rows[i];
    rows[i] = before.replace('Naive|,4,,252,,252|', 'Naive|,4,,252,,248|');
    if (rows[i] === before) throw new Error('selftest: precondition — Salamence EV spread not found');
  }));
  // The MASKING guard — the construction reorder must not hide a real bug later in the battle.
  neg('a LATER divergence on the SAME side (a p1 -damage HP after |turn|1)', mangle(f21, (rows) => {
    const { rest } = sideRows(rows, 'p1');
    const i = findRow(rows, rest, (r) => r.startsWith('|-damage|'), 'a p1 -damage after turn 1');
    rows[i] = withRaw(rows[i], rawOf(rows[i]).replace(/\|(\d+)\//, (m, hp) => `|${Number(hp) === 0 ? 1 : Number(hp) - 1}/`));
  }));
  neg('a LATER divergence on the OTHER side (a p2 -damage HP after |turn|1)', mangle(f21, (rows) => {
    const { rest } = sideRows(rows, 'p2');
    const i = findRow(rows, rest, (r) => r.startsWith('|-damage|'), 'a p2 -damage after turn 1');
    rows[i] = withRaw(rows[i], rawOf(rows[i]).replace(/\|(\d+)\//, (m, hp) => `|${Number(hp) === 0 ? 1 : Number(hp) - 1}/`));
  }));
  neg('a LATER |request| divergence on the SAME side (a pp value)', mangle(f21, (rows) => {
    const { rest } = sideRows(rows, 'p1');
    const i = findRow(rows, rest, (r) => r.startsWith('|request|') && /"pp":\d+/.test(r), 'a p1 request with a pp');
    rows[i] = withRaw(rows[i], rawOf(rows[i]).replace(/"pp":(\d+)/, (m, pp) => `"pp":${Number(pp) + 1}`));
  }));
  neg('the LAST p1 line dropped (a truncated golden)', mangle(f21, (rows) => {
    const { rest } = sideRows(rows, 'p1');
    rows.splice(rest[rest.length - 1], 1);
  }));
  // Fixture 22 (the p2-side orientation): its own content-change negative.
  neg('fixture 22: content change inside the reordered p2 window (atk|1 -> atk|2)', mangle(f22, (rows) => {
    const { window } = sideRows(rows, 'p2');
    const i = findRow(rows, window, (r) => r === '|-unboost|p2a: Zapdos|atk|1', 'the p2 -unboost');
    rows[i] = withRaw(rows[i], '|-unboost|p2a: Zapdos|atk|2');
  }));

  try { fs.rmSync(tmp, { recursive: true, force: true }); } catch (e) {}
  console.error(fail ? `\n[selftest] ${fail} of ${n} FAILURE(S)` : `\n[selftest] all ${n} allowlist gate-integrity cases pass`);
  process.exit(fail ? 1 : 0);
}

function appendLog(logPath, line) {
  fs.appendFileSync(logPath, line + '\n');
  console.error(line);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
