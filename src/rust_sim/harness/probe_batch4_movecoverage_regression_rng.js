// probe_batch4_movecoverage_regression_rng.js — GROUND TRUTH for the MOVE-COVERAGE BATCH 4
// regression pins (MC30…MC35, `gen3_move_coverage_batch4_v1`: FOCUS PUNCH + PURSUIT) in
// tests/regression_test.rs.
//
// Each scenario drives the OMNISCIENT in-process BattleStream over a CONSTRUCTED gen3customgame
// board whose EXACT packed teams + seed the Rust regression test replays. It RESEEDS the sim's
// prng to the RAW seed right before the first decision (so the decision draws line up with the
// Rust's draw-free `start_with_switchins`), then prints the per-decision seedAfter + observable
// STATE — copied verbatim into the pins as the real-Showdown ground truth. Each pin FAILS if its
// class's engine wiring is reverted.
//
// Run:  node src/rust_sim/harness/probe_batch4_movecoverage_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

function tick() { return new Promise((r) => setTimeout(r, 0)); }

const RAW = [44317, 42357, 9927, 48760]; // the shared raw seed (the port constructs draw-free with it)

async function run(label, p1, p2, plan, rawSeed = RAW) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  b.prng = new PRNG(rawSeed.slice());
  const a0 = () => b.sides[0].active[0], a1 = () => b.sides[1].active[0];
  console.log(`\n=== ${label} (raw seed ${rawSeed.join(',')}) ===`);
  let i = 0, safety = 0;
  while (!b.ended && safety < 60 && i < plan.length) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = log.length;
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const fmt = (m) => (m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.status ? ' ' + m.status : ''}` : '-');
    console.log(`  dec ${i - 1} [${rs}] ${JSON.stringify(entry)}  seedAfter=${b.prng.getSeed()}`);
    console.log(`      p1=${fmt(a0())}  left=${b.sides[0].pokemonLeft}`);
    console.log(`      p2=${fmt(a1())}  left=${b.sides[1].pokemonLeft}`);
    log.slice(before)
      .filter((l) => /\|move\||-damage|-activate|cant|switch|faint|-crit|-supereffective/.test(l))
      .forEach((l) => console.log(`      L ${l}`));
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  const machamp = 'Machamp||||focuspunch,seismictoss|Serious|252,252,,,,|||||';
  const machampSplash = 'Machamp||||focuspunch,splash|Serious|252,252,,,,|||||';
  const snorlaxTackle = 'Snorlax|||immunity|tackle|Serious|,252,,,,|||||';
  const blisseySplash = 'Blissey|||naturalcure|splash|Serious|252,,,,,|||||';

  // MC30: FOCUS PUNCH CANCELLED — the foe Tackles the FP user first (FP is priority -3 → moves
  //   last) → lostFocus → the onTry cancels draw-free BEFORE accuracy. Snorlax takes NO damage
  //   (FP did not execute). One turn (both survive).
  await run('MC30 FP cancelled by a prior hit',
    machamp, snorlaxTackle, [{ p1: 'move 1', p2: 'move 1' }]);

  // MC31: FOCUS PUNCH LANDS — the foe Splashes (non-damaging) → the user keeps focus → FP
  //   executes (the beforeTurnMove Update + the FP acc/crit/dmg). Machamp FP hits Blissey.
  await run('MC31 FP lands (foe Splash)',
    machampSplash, blisseySplash, [{ p1: 'move 1', p2: 'move 1' }]);

  // MC32: PURSUIT INTERRUPT ×2 — the foe voluntarily switches → the strike hits the SWITCHING
  //   mon at ×2 BP + never-miss (crit + dmg, NO accuracy) BEFORE the switch resolves.
  const ttar = 'Tyranitar|||pressure|pursuit,crunch|Serious|,252,,252,,|||||';
  const jolteonSnorlax =
    'Jolteon|||voltabsorb|thunderbolt|Serious|,,,,,252|||||]Snorlax|||immunity|bodyslam|Serious|252,,,,,|||||';
  await run('MC32 Pursuit interrupt (foe switches → x2, no acc)',
    ttar, jolteonSnorlax, [{ p1: 'move 1', p2: 'switch 2' }]);

  // MC33: NORMAL PURSUIT — the foe STAYS in and attacks → a plain bp-40 Dark hit (acc + crit +
  //   dmg), NO ×2, NO interrupt (the contrast to MC32).
  const snorlaxBlissey =
    'Snorlax|||immunity|bodyslam|Serious|252,252,,,,|||||]Blissey|||naturalcure|softboiled|Serious|252,,,,,|||||';
  await run('MC33 Pursuit normal (foe stays → plain bp40)',
    ttar, snorlaxBlissey, [{ p1: 'move 1', p2: 'move 1' }]);

  // MC34: PURSUIT KOs the SWITCHER — a low-HP Gengar switches into a Pursuit that KOs it (Dark
  //   super-effective vs Ghost); the already-chosen switch STILL brings in Snorlax (the gen 2-4
  //   `-hint`), and the turn completes (Quick Claw drawn).
  const gengarLow = 'Gengar|||levitate|shadowball|Serious|,,,,,252|||||]Snorlax|||immunity|bodyslam|Serious|252,,,,,|||||';
  // Inject Gengar to 40 HP after start by scripting: give it a 1-turn setup? Simpler: use a
  // low-HP Gengar via a Focus Sash-less low level. Instead we inject HP in the probe below.
  {
    const stream = new BattleStream();
    const streams = getPlayerStreams(stream);
    const log = [];
    (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
    streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
    streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: ttar })}`);
    streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: gengarLow })}`);
    for (let i = 0; i < 12; i++) await tick();
    const b = stream.battle;
    b.sides[1].active[0].hp = 40; // inject Gengar to 40 HP (STATE-only, no PRNG)
    b.prng = new PRNG(RAW.slice());
    const before = log.length;
    streams.omniscient.write('>p1 move 1');
    streams.omniscient.write('>p2 switch 2');
    for (let k = 0; k < 18; k++) await tick();
    const fmt = (m) => (m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.status ? ' ' + m.status : ''}` : '-');
    console.log(`\n=== MC34 Pursuit KOs the switcher (Gengar injected to 40 HP) (raw ${RAW.join(',')}) ===`);
    console.log(`  seedAfter=${b.prng.getSeed()}`);
    console.log(`  p1=${fmt(b.sides[0].active[0])}  left=${b.sides[0].pokemonLeft}`);
    console.log(`  p2=${fmt(b.sides[1].active[0])}  left=${b.sides[1].pokemonLeft}`);
    log.slice(before)
      .filter((l) => /\|move\||-damage|-activate|cant|switch|faint|-hint|-supereffective/.test(l))
      .forEach((l) => console.log(`  L ${l}`));
    try { streams.omniscient.destroy(); } catch (e) {}
  }

  // MC35: FOCUS PUNCH MIRROR at a SPEED TIE — both Focus Punch (the two beforeTurnMove order-5
  //   actions tie; both mons carry the focuspunch volatile at the residual → the +1 residual
  //   duration-handler tie-shuffle). The first FP lands; the second is cancelled draw-free.
  const machampMirror = 'Machamp||||focuspunch|Serious|,252,,,,|||||';
  await run('MC35 FP mirror speed-tie (btm tie + residual mirror tie)',
    machampMirror, machampMirror, [{ p1: 'move 1', p2: 'move 1' }]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
