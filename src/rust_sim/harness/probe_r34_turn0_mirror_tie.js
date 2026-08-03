// probe_r34_turn0_mirror_tie.js — settle, DRAW BY DRAW, what the REAL sim does in the
// turn-0 CONSTRUCTION WINDOW for the `sbd_msdd8698_b293` repro: a same-species
// **Masquerain mirror** (both Intimidate) at a speed-TIED lead, where node emits the
// Intimidate block `p2a` first and the rust port emits `p1a` first.
//
// THE QUESTION this settles: is the port's turn-0 construction (`gen3_turn0_construction_v1`)
// consuming/ordering the `insertChoice` tie draw differently (a REAL engine bug), or is the
// flip an unobservable attribution ambiguity (the ROUND-18 benign class)?
//
// METHOD (the sim is the ORACLE): instrument `battle.prng.rng.next` BEFORE the players are
// added (so no construction draw escapes), replay the repro's EXACT `>start` seed + packed
// teams through the real BattleStream, and print
//   * every turn-0 raw draw with the seed BEFORE it and the CALL SITE (from the stack), and
//   * the emitted turn-0 protocol lines.
// The port's half of the same picture is `POKESIM_PRNG_TRACE=1 ./sim_bridge` (see
// `--rust` below, which drives the built binary with the identical START line and prints
// its `[prng] #n -> v seed_before=...` stream). Line the two up by draw index.
//
// NON-VACUITY GUARDS (a probe where the mechanic never fires silently tests nothing):
//   G1 both leads are the SAME species,
//   G2 both leads have Intimidate,
//   G3 the two leads' raw Speed is EQUAL (the tie is what makes the draw happen),
//   G4 the emitted turn-0 window actually CONTAINS 2 `-ability|...|Intimidate|boost` lines.
// Any guard failing exits 2 (INCONCLUSIVE), never a silent pass.
//
// Run:
//   node src/rust_sim/harness/probe_r34_turn0_mirror_tie.js            # sim only
//   node src/rust_sim/harness/probe_r34_turn0_mirror_tie.js --rust     # sim + port A/B
//   ... --repro <dir>   # take start_json from a saved gen_sim_bridge_diff repro
//
// Exit: 0 = the sim's own trace printed + guards held; 2 = a guard failed.

'use strict';
const path = require('path');
const fs = require('fs');
const { spawnSync } = require('child_process');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Dex } = require(path.join(PS, 'dist/sim'));

const DEFAULT_REPRO = path.join(__dirname, 'sim_bridge_diff_out/divergences/sbd_msdd8698_b293');
const RUST_TARGET = process.env.POKESIM_SIMBRIDGE_TARGET || '/tmp/pokesim_masq_build';
const RUST_BIN = path.join(RUST_TARGET, 'release/sim_bridge');

function tick() { return new Promise((r) => setTimeout(r, 0)); }

function parseArgs(argv) {
  const f = { repro: DEFAULT_REPRO, rust: false };
  for (let i = 2; i < argv.length; i++) {
    if (argv[i] === '--rust') f.rust = true;
    else if (argv[i] === '--repro') f.repro = path.resolve(argv[++i]);
  }
  return f;
}

// The call site of a draw, as a compact tag: the first sim frame that is not prng/battle.random.
function callSite() {
  const st = new Error().stack.split('\n').slice(1);
  const frames = [];
  for (const l of st) {
    const m = l.match(/at ([^\s(]+)?\s*\(?([^\s)]*\.js):(\d+):(\d+)\)?/);
    if (!m) continue;
    const file = path.basename(m[2]);
    const fn = m[1] || '?';
    if (fn.includes('next') && file === 'prng.js') continue;
    if (file === 'probe_r34_turn0_mirror_tie.js') continue;
    frames.push(`${file}:${m[3]}:${fn}`);
    if (frames.length >= 4) break;
  }
  return frames.join(' <- ');
}

async function simTrace(startMsg) {
  const stream = new BattleStream();
  const lines = [];
  (async () => {
    for await (const ch of stream) {
      for (const l of String(ch).split('\n')) if (l) lines.push(l);
    }
  })();
  const seedClause = startMsg.seed ? `,"seed":${JSON.stringify(startMsg.seed)}` : '';
  stream.write(`>start {"formatid":"${startMsg.formatid}"${seedClause}}`);
  await tick();
  const battle = stream.battle;
  if (!battle) throw new Error('no battle after >start');

  // Patch the RAW rng source BEFORE the players are added, so the per-mon gender
  // `sample`s (which happen in `Side`/`Pokemon` construction at `>player` time) are
  // captured too.
  const draws = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = () => {
    const before = battle.prng.getSeed();
    const v = realNext();
    draws.push({ n: draws.length, before: String(before), v, site: callSite() });
    return v;
  };

  stream.write(`>player p1 ${JSON.stringify(startMsg.p1)}`);
  stream.write(`>player p2 ${JSON.stringify(startMsg.p2)}`);
  for (let i = 0; i < 20; i++) await tick();

  return { battle, draws, lines };
}

// Drive the built rust sim_bridge with the identical START line, capturing its
// POKESIM_PRNG_TRACE draw stream + the p1 chunks it emits.
function rustTrace(startMsg) {
  if (!fs.existsSync(RUST_BIN)) {
    const env = { ...process.env, PATH: `${process.env.HOME}/.cargo/bin:${process.env.PATH}`, CARGO_TARGET_DIR: RUST_TARGET };
    const b = spawnSync('cargo', ['build', '--release', '--bin', 'sim_bridge'],
      { cwd: path.resolve(__dirname, '..'), env, stdio: 'inherit' });
    if (b.status !== 0) throw new Error('cargo build failed');
  }
  const r = spawnSync(RUST_BIN, [], {
    input: `START ${JSON.stringify(startMsg)}\n`,
    env: { ...process.env, POKESIM_PRNG_TRACE: '1' },
    encoding: 'utf8', timeout: 60000,
  });
  const draws = [];
  for (const l of String(r.stderr || '').split('\n')) {
    const m = l.match(/^\[prng\] #(\d+) -> (\d+) seed_before=(.*)$/);
    if (m) draws.push({ n: Number(m[1]), v: Number(m[2]), before: m[3] });
  }
  const lines = [];
  for (const l of String(r.stdout || '').split('\n')) {
    const m = l.match(/^p1 (.*)$/);
    if (m) for (const x of Buffer.from(m[1], 'base64').toString('utf8').split('\n')) if (x) lines.push(x);
  }
  return { draws, lines };
}

(async () => {
  const flags = parseArgs(process.argv);
  const summary = JSON.parse(fs.readFileSync(path.join(flags.repro, 'summary.json'), 'utf8'));
  const startMsg = JSON.parse(summary.start_json);
  console.log(`[probe] repro=${path.basename(flags.repro)} format=${startMsg.formatid} seed=${JSON.stringify(startMsg.seed)}`);

  const { battle, draws, lines } = await simTrace(startMsg);

  // ── Non-vacuity guards ──────────────────────────────────────────────────────
  const a1 = battle.sides[0].active[0];
  const a2 = battle.sides[1].active[0];
  const fail = (m) => { console.error(`[probe] GUARD FAILED (inconclusive): ${m}`); process.exit(2); };
  if (a1.species.name !== a2.species.name) fail(`G1 leads are not the same species (${a1.species.name} vs ${a2.species.name})`);
  if (a1.ability !== 'intimidate' || a2.ability !== 'intimidate') fail(`G2 leads are not both Intimidate (${a1.ability}/${a2.ability})`);
  if (a1.storedStats.spe !== a2.storedStats.spe) fail(`G3 leads are not raw-Speed tied (${a1.storedStats.spe} vs ${a2.storedStats.spe})`);
  const win = lines.slice(lines.findIndex((l) => l === '|start'));
  const intim = win.filter((l) => /^\|-ability\|p\da: .*\|Intimidate\|boost$/.test(l));
  if (intim.length !== 2) fail(`G4 the turn-0 window has ${intim.length} Intimidate lines, expected 2`);
  console.log(`[probe] guards OK: mirror=${a1.species.name} ability=intimidate spe=${a1.storedStats.spe} (TIED), 2 Intimidate lines emitted`);

  console.log('\n=== SIM turn-0 protocol window ===');
  for (const l of win) { console.log('  ' + l); if (l.startsWith('|turn|')) break; }

  console.log(`\n=== SIM turn-0 draws (up to |turn|1) : ${draws.length} total captured ===`);
  for (const d of draws.slice(0, 12)) console.log(`  #${d.n} -> ${d.v}  seed_before=${d.before}\n       ${d.site}`);

  if (flags.rust) {
    const rt = rustTrace(startMsg);
    const rwin = rt.lines.slice(rt.lines.findIndex((l) => l === '|start'));
    console.log('\n=== PORT turn-0 protocol window ===');
    for (const l of rwin) { console.log('  ' + l); if (l.startsWith('|turn|')) break; }
    console.log(`\n=== PORT turn-0 draws : ${rt.draws.length} captured ===`);
    for (const d of rt.draws.slice(0, 12)) console.log(`  #${d.n} -> ${d.v}  seed_before=${d.before}`);

    console.log('\n=== DRAW-STREAM A/B (sim vs port) ===');
    const n = Math.max(Math.min(draws.length, 12), Math.min(rt.draws.length, 12));
    let firstMismatch = -1;
    for (let i = 0; i < n; i++) {
      const s = draws[i]; const p = rt.draws[i];
      const same = s && p && s.v === p.v && s.before === p.before;
      if (!same && firstMismatch < 0) firstMismatch = i;
      console.log(`  #${i}  sim=${s ? s.v : '-'} (${s ? s.before : '-'})   port=${p ? p.v : '-'} (${p ? p.before : '-'})   ${same ? 'OK' : 'DIFF'}`);
    }
    console.log(firstMismatch < 0
      ? '\n[probe] draw streams AGREE over the compared prefix — the flip is NOT a draw-value divergence.'
      : `\n[probe] FIRST DRAW MISMATCH at index ${firstMismatch}.`);
  }
})().catch((e) => { console.error(e); process.exit(1); });
