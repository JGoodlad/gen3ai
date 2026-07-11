// trace_confusion_boost_tri.js — REVIEW instrumentation for the THREE secondary
// completion paths this step adds: the CONFUSION secondary (land + already-confused),
// a STAT-DROP secondary (Crunch -1 SpD), and TRI ATTACK (1 random(100) + 1 sample(3)).
//
// For each case it runs ONE isolated turn in the live omniscient BattleStream,
// monkey-patches the PRNG to log EVERY draw (method + args + result + sim call-site),
// captures (seedBefore, seedAfter), and prints the trace so a human can read the
// EXACT draw order. It ALSO emits compact `CMP <case> <seed> <seedBefore> <seedAfter>`
// lines that the Rust review test (`tests/review_confusion_boost_tri.rs`) replays —
// running `run_move` from seedBefore and asserting its post-move seed == seedAfter.
//
// The CRUX lenses (from the review prompt):
//   - Water Pulse confuse-LAND draws random(2,6) at the right point (inside addVolatile);
//   - Water Pulse on an ALREADY-CONFUSED target draws the secondary random(100) but
//     NOT random(2,6) (the addVolatile-returns-false gate);
//   - a stat-drop (Crunch) draws its secondary random(100) and applies -1 (draw-free);
//   - Tri Attack draws ONE random(100) then ONE random(3) (sample), NOT three random(100)s.
//
// Run:  node src/rust_sim/harness/trace_confusion_boost_tri.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

'use strict';

const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };

function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

function isWrapperFrame(ln) {
  if (ln.includes('/sim/prng.js')) return true;
  if (ln.includes('trace_confusion_boost_tri.js')) return true;
  if (/at Battle\.(random|randomChance|sample) /.test(ln)) return true;
  return false;
}
function frameStr(ln) {
  const fn = (ln.match(/at ([\w.<>]+) /) || [])[1] || '?';
  const loc = (ln.match(/\/(sim\/[^\s):]+:\d+):\d+/) || ln.match(/\/(data\/[^\s):]+:\d+):\d+/) || [])[1] || '?';
  return `${fn}@${loc}`;
}
function siteOf(depth = 5) {
  const e = {};
  Error.captureStackTrace(e, siteOf);
  const lines = (e.stack || '').split('\n').slice(1).filter((ln) => !isWrapperFrame(ln));
  const sim = lines.filter((ln) => /\/(sim|data)\//.test(ln)).slice(0, depth);
  return sim.map(frameStr).join(' <- ') || '?';
}

function instrument(prng, sink) {
  const rng = prng.rng;
  let nextCount = 0;
  const origNext = rng.next.bind(rng);
  rng.next = function () { nextCount++; return origNext(); };
  let inHigh = false;
  for (const name of ['randomChance', 'sample', 'shuffle']) {
    const orig = prng[name].bind(prng);
    prng[name] = function (...args) {
      const site = siteOf();
      const before = nextCount;
      inHigh = true;
      let result;
      try { result = orig(...args); } finally { inHigh = false; }
      const consumed = nextCount - before;
      const logResult = name === 'shuffle' ? `[len=${args[0] && args[0].length}]` : result;
      sink.push({ method: name, args: args.filter((_, i) => name !== 'shuffle' || i > 0), result: logResult, nexts: consumed, site });
      return result;
    };
  }
  const origRandom = prng.random.bind(prng);
  prng.random = function (...args) {
    if (inHigh) return origRandom(...args);
    const site = siteOf();
    const before = nextCount;
    const result = origRandom(...args);
    const consumed = nextCount - before;
    sink.push({ method: 'random', args, result, nexts: consumed, site });
    return result;
  };
}

function buildSeeds(n, salt) {
  const out = [];
  let x = (0x12345 ^ salt) >>> 0;
  const step = () => { x = (Math.imul(x, 1103515245) + 12345) >>> 0; return x & 0xffff; };
  for (let i = 0; i < n; i++) out.push([step() || 1, step() || 1, step() || 1, step() || 1]);
  return out;
}

function startBattle(p1mons, p2mons, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1mons) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2mons) })}`);
  return { stream, streams, log };
}

function fmtDraw(d) {
  if (d.marker) return `  --- ${d.marker} ---`;
  const args = JSON.stringify(d.args);
  let res = d.result;
  if (typeof res === 'number' && !Number.isInteger(res)) res = res.toFixed(6);
  return `  ${d.method.padEnd(12)} args=${String(args).padEnd(11)} -> ${String(res).padEnd(8)} nexts=${d.nexts}\n        @ ${d.site}`;
}

// Run ONE turn. `setup(battle)` runs after the leads are in (draw-free injection of
// confusion etc.). p1 uses `move 1` (the move under test); p2 uses `move 1` too.
// Returns { seedBefore, seedAfter, draws, info }.
async function oneTurn(p1mons, p2mons, seed, setup, p1choice = 'move 1', p2choice = 'move 1') {
  const { stream, streams } = startBattle(p1mons, p2mons, seed);
  for (let i = 0; i < 4; i++) await tick();
  const battle = stream.battle;
  if (setup) setup(battle);
  const draws = [];
  instrument(battle.prng, draws);
  const seedBefore = battle.prng.getSeed();
  streams.omniscient.write(`>p1 ${p1choice}`);
  streams.omniscient.write(`>p2 ${p2choice}`);
  for (let i = 0; i < 12; i++) await tick();
  const seedAfter = battle.prng.getSeed();
  const p1 = battle.sides[0].active[0];
  const p2 = battle.sides[1].active[0];
  const info = {
    p1status: p1.status || '-', p2status: p2.status || '-',
    p1hp: `${p1.hp}/${p1.maxhp}`, p2hp: `${p2.hp}/${p2.maxhp}`,
    p2conf: (p2.volatiles && p2.volatiles.confusion) ? p2.volatiles.confusion.time : 0,
    p1conf: (p1.volatiles && p1.volatiles.confusion) ? p1.volatiles.confusion.time : 0,
    p2boosts: JSON.stringify(p2.boosts), p1boosts: JSON.stringify(p1.boosts),
  };
  try { streams.omniscient.destroy(); } catch (e) {}
  return { seedBefore, seedAfter, draws, info };
}

const cmpLines = [];

// Find N seeds where the secondary LANDS (the move-under-test is p1 move 1, p2 is a
// passive defender). `landed(info)` decides. Returns the matching turn records.
// Emits REPLAYABLE CMP rows: `CMP <case> <seedBefore> <seedAfter> <p1pack> <p2pack>` so
// the Rust review test seeds at seedBefore, runs run_full_battle([move/move]), and
// asserts the post-turn seed == seedAfter (NO pre-injection needed for these cases).
async function findLanding(caseName, p1mons, p2mons, salt, setup, landed, want = 3, p1choice = 'move 1') {
  console.log(`\n================ ${caseName} ================`);
  const seeds = buildSeeds(800, salt);
  const p1pack = Teams.pack(p1mons);
  const p2pack = Teams.pack(p2mons);
  const hits = [];
  for (const seed of seeds) {
    const rec = await oneTurn(p1mons, p2mons, seed, setup, p1choice);
    if (landed(rec.info, rec.draws)) {
      hits.push({ seed, ...rec });
      if (hits.length >= want) break;
    }
  }
  if (hits.length === 0) { console.log('  (NO landing seed found — widen the pool)'); return hits; }
  // Print the FIRST hit's full trace; emit CMP lines for all hits.
  const h0 = hits[0];
  console.log(`First landing seed ${JSON.stringify(h0.seed)} | before=${h0.seedBefore} after=${h0.seedAfter}`);
  console.log(`  info: ${JSON.stringify(h0.info)}`);
  for (const d of h0.draws) console.log(fmtDraw(d));
  // boosts as a compact 7-int csv [atk,def,spa,spd,spe,accuracy,evasion] for p1 (user)
  // and p2 (foe), so the Rust review test asserts the boost STATE too (a draw-free
  // divergence the seed can't catch — wrong stat / missed stat in a multi-stat Ancient
  // Power, wrong target, or the accuracy index for Muddy Water).
  const b7 = (j) => { const o = JSON.parse(j); return [o.atk, o.def, o.spa, o.spd, o.spe, o.accuracy, o.evasion].join(','); };
  for (const h of hits) {
    cmpLines.push(`CMP\t${caseName}\t${h.seedBefore}\t${h.seedAfter}\t${p1pack}\t${p2pack}\t${b7(h.info.p1boosts)}\t${b7(h.info.p2boosts)}`);
  }
  return hits;
}

async function main() {
  // ── 1. WATER PULSE confuse-LAND: p1 Starmie Water Pulse vs a bulky Snorlax (survives,
  //      Normal so not Water-immune). Look for confusion landing (p2conf > 0).
  await findLanding(
    'waterpulse_confuse_land',
    [mon('Starmie', ['waterpulse'], { nature: 'Modest', evs: { spa: 252 } })],
    [mon('Snorlax', ['swift'], { nature: 'Hardy', evs: { hp: 252, def: 252 } })],
    1, null,
    (info) => info.p2conf > 0, 3,
  );

  // ── 2. WATER PULSE on an ALREADY-CONFUSED target: pre-inject confusion onto p2, then
  //      Water Pulse. Expect the secondary random(100) BUT NO random(2,6) (gate). We
  //      look for a seed where the secondary roll LANDS (would-add) yet is gated. Since
  //      we can't read the secondary roll directly, we accept ALL seeds and rely on the
  //      trace + the CMP replay (the Rust must NOT draw random(2,6) on a gated land).
  //      To MAXIMISE signal we also assert in the trace there is no addVolatile-onStart
  //      random(2,6) (range [2,6)) AFTER a secondary random(100).
  {
    console.log(`\n================ waterpulse_already_confused (pre-injected) ================`);
    const seeds = buildSeeds(800, 2);
    const hits = [];
    for (const seed of seeds) {
      // setup: inject a DISTINCT confusion counter (time=4) so a wrong re-draw shows.
      const setup = (b) => {
        const p2 = b.sides[1].active[0];
        p2.addVolatile('confusion');
        // overwrite to a fixed time so any re-add (wrong) would change it
        if (p2.volatiles.confusion) p2.volatiles.confusion.time = 4;
      };
      const rec = await oneTurn(
        [mon('Starmie', ['waterpulse'], { nature: 'Modest', evs: { spa: 252 } })],
        [mon('Snorlax', ['swift'], { nature: 'Hardy', evs: { hp: 252, def: 252 } })],
        seed, setup,
      );
      // We want a seed where the WATER PULSE secondary random(100) LANDED (roll<20) so
      // the gate is actually exercised. Detect: a random(100) drawn at the secondaries
      // call-site with result < 20.
      const secRoll = rec.draws.find((d) => d.method === 'random' && Array.isArray(d.args) && d.args[0] === 100 && /secondaries/.test(d.site));
      const landed = secRoll && secRoll.result < 20;
      // Confirm NO random(2,6) (the duration) was drawn this turn — range arg [2,6).
      const dur = rec.draws.find((d) => d.method === 'random' && Array.isArray(d.args) && d.args[0] === 2 && d.args[1] === 6);
      if (landed) {
        hits.push({ seed, ...rec, gatedNoDur: !dur });
        if (hits.length >= 3) break;
      }
    }
    if (hits.length === 0) { console.log('  (no landed-secondary seed found)'); }
    else {
      const h0 = hits[0];
      console.log(`First gated-land seed ${JSON.stringify(h0.seed)} | before=${h0.seedBefore} after=${h0.seedAfter}`);
      console.log(`  info: ${JSON.stringify(h0.info)}  | NO random(2,6) drawn? ${h0.gatedNoDur}`);
      for (const d of h0.draws) console.log(fmtDraw(d));
      let allGated = true;
      for (const h of hits) {
        if (!h.gatedNoDur) { allGated = false; console.log(`  !! seed ${h.seed} DREW random(2,6) on an already-confused target — GATE BUG`); }
      }
      // This case PRE-INJECTS confusion, so it is NOT replayable via run_full_battle;
      // the GATE (no random(2,6) on an already-confused land) is asserted HERE on the
      // live sim and cross-checked by the Rust unit test
      // `confusion_secondary_already_confused_skips_the_duration_draw`.
      console.log(`  GATE (no random(2,6) on any already-confused LAND across ${hits.length} seeds): ${allGated ? 'PASS' : 'FAIL'}`);
    }
  }

  // ── 3. STAT-DROP: Tyranitar Crunch (-1 SpD foe, 20%) vs a bulky Snorlax. Landing =
  //      p2 SpD boost (index 'spd') == -1.
  await findLanding(
    'crunch_spd_drop',
    [mon('Tyranitar', ['crunch'], { nature: 'Adamant', evs: { atk: 252 } })],
    [mon('Snorlax', ['swift'], { nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    3, null,
    (info) => { try { return JSON.parse(info.p2boosts).spd === -1; } catch (e) { return false; } }, 3,
  );

  // ── 3b. ANCIENT POWER self +1 ALL (the only MULTI-STAT self-boost, 10%): the apply
  //       touches 5 stats at once (a STATE divergence if any are missed). Seed parity
  //       proves the ONE random(100) draw; the Rust review test ALSO asserts the
  //       user's 5 boost stages == +1 each on a land.
  await findLanding(
    'ancientpower_self_all',
    [mon('Aerodactyl', ['ancientpower'], { nature: 'Adamant', evs: { atk: 252 } })],
    [mon('Snorlax', ['swift'], { nature: 'Hardy', evs: { hp: 252, def: 252 } })],
    7, null,
    (info) => { try { return JSON.parse(info.p1boosts).atk > 0; } catch (e) { return false; } }, 3,
  );

  // ── 3c. MUDDY WATER foe −1 ACCURACY (30%): the ACCURACY boost-array index (5), which
  //       the damage Combatant ignores — a STATE-only apply. Seed parity proves the draw.
  await findLanding(
    'muddywater_acc_drop',
    [mon('Swampert', ['muddywater'], { nature: 'Modest', evs: { spa: 252 } })],
    [mon('Snorlax', ['swift'], { nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    8, null,
    (info) => { try { return JSON.parse(info.p2boosts).accuracy < 0; } catch (e) { return false; } }, 3,
  );

  // ── 4. TRI ATTACK: Porygon2 Tri Attack (20% → sample brn/par/frz) vs a bulky Snorlax.
  //      Landing = p2 gets a status (Normal, not immune to brn/par/frz). Watch for ONE
  //      random(100) then ONE random(3) (the sample), NOT three random(100)s.
  await findLanding(
    'triattack_status',
    [mon('Porygon2', ['triattack'], { nature: 'Modest', evs: { spa: 252 } })],
    [mon('Snorlax', ['swift'], { nature: 'Hardy', evs: { hp: 252, def: 252 } })],
    4, null,
    (info) => info.p2status !== '-', 3,
  );

  // ── 5. SPEED-TIE confusion-land (the HIGHEST-signal ordering case): two identical
  //      Starmies (mirror = a true action-speed TIE), p1 Water Pulse, p2 Swift. On a
  //      tie the per-action eachEvent('Update') shuffles DRAW, so this proves the
  //      secondary random(100) + confusion random(2,6) fire in the RIGHT place RELATIVE
  //      to the in-tryMoveHit [7] shuffle (a mis-ordered secondary vs shuffle desyncs).
  await findLanding(
    'tie_waterpulse_confuse',
    [mon('Starmie', ['waterpulse'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    [mon('Starmie', ['swift'], { nature: 'Timid', evs: { spa: 252, spe: 252 } })],
    9, null,
    (info) => info.p2conf > 0, 3,
  );

  // Emit the machine-readable CMP block for the Rust replay test.
  console.log('\n================ CMP (machine-readable: case seedBefore seedAfter p1pack p2pack) ================');
  for (const ln of cmpLines) console.log(ln);

  // Also write to a file the Rust test reads.
  const fs = require('fs');
  const outPath = path.join(__dirname, '..', 'tests', 'vectors', 'confusion_boost_tri_cmp.txt');
  fs.writeFileSync(outPath, cmpLines.join('\n') + '\n');
  console.log(`\nwrote ${cmpLines.length} CMP rows -> ${outPath}`);
}
main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
