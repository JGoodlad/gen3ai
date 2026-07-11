// probe_batch4_regression_rng.js — GROUND TRUTH for the `gen3_ability_batch4_v1` regression
// pins (B4-1 Truant loaf / B4-2 Inner Focus / B4-3 Shadow Tag / B4-4 Cute Charm+attract /
// B4-5 Color Change / B4-6 King's Rock / B4-7 Focus Band) in tests/regression_test.rs.
//
// Each scenario drives the OMNISCIENT in-process BattleStream over a CONSTRUCTED
// gen3customgame board whose EXACT packed teams + seed the Rust regression test replays. It
// RESEEDS the sim's prng to the RAW seed right before the first decision (so the decision
// draws line up with the Rust's draw-free `start_with_switchins`), then prints the
// per-decision seedAfter + observable STATE — copied verbatim into the pins as the
// real-Showdown ground truth. Each pin FAILS if its member's engine wiring is reverted.
//
// Run:  node src/rust_sim/harness/probe_batch4_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1, p2, rawSeed, plan, opts = {}) {
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
  if (!opts.quiet) console.log(`\n=== ${label} (raw seed ${rawSeed.join(',')}) ===`);
  let i = 0, safety = 0;
  const decs = [];
  while (!b.ended && safety < 60 && i < plan.length) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = log.length;
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const fmt = (m) => `${m.species.name} ${m.hp}/${m.maxhp}${m.status ? ' ' + m.status : ''}`;
    const seedAfter = b.prng.getSeed();
    decs.push({ seedAfter, p1: fmt(a0()), p2: fmt(a1()), lines: log.slice(before) });
    if (!opts.quiet) {
      console.log(`  dec ${i - 1} [${rs}] ${JSON.stringify(entry)}`);
      console.log(`    seedAfter=${seedAfter}`);
      console.log(`    p1=${fmt(a0())} (trapped=${!!a0().trapped}) | p2=${fmt(a1())} (trapped=${!!a1().trapped})`);
      const notable = log.slice(before).filter((l) => /-status|ability:|cant|-immune|-activate|typechange|Attract|Focus Band|flinch/.test(l));
      if (notable.length) console.log(`    lines: ${JSON.stringify(notable)}`);
    }
  }
  try { streams.omniscient.destroy(); } catch (e) {}
  return { decs, ended: b.ended, log };
}

// Scan raw seeds for one whose FIRST decision log matches `re`.
async function findSeed(p1, p2, re, base, plan) {
  let s = base >>> 0;
  const rng = () => { s = (s * 1664525 + 1013904223) >>> 0; return s; };
  for (let n = 0; n < 500; n++) {
    const seed = [rng() % 65536, rng() % 65536, rng() % 65536, rng() % 65536];
    const r = await run('scan', p1, p2, seed, plan || [{ p1: 'move 1', p2: 'move 1' }], { quiet: true });
    if (r.decs.length && re.test(r.decs.map((d) => d.lines.join('\n')).join('\n'))) return seed;
  }
  return null;
}

async function main() {
  // ── B4-1 TRUANT: Slaking Body Slams turn 1, LOAFS turn 2 (draw-free cant — the foe's HP is
  //    unchanged and the loafer deducts no PP).
  const slaking = 'Slaking|||Truant|bodyslam,earthquake|Adamant|,252,,,,|N||||';
  const swampert = 'Swampert|||Torrent|surf,surf|Modest|252,,,252,,|N||||';
  const truantSeed = [21001, 4383, 9902, 61177];
  await run('B4-1 TRUANT (moves t1, loafs t2)', slaking, swampert, truantSeed,
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);

  // ── B4-2 INNER FOCUS: Bite's 30% flinch PASSES → the Thick-Fat control is cant'd, the
  //    Inner-Focus holder MOVES (the roll drawn either way — block at the APPLY).
  const jolt = 'Jolteon|||Static|bite,thunderbolt|Timid|,,,252,,252|N||||';
  const laxIF = 'Snorlax|||InnerFocus|bodyslam,bodyslam|Adamant|252,252,,,,|N||||';
  const laxTF = 'Snorlax|||ThickFat|bodyslam,bodyslam|Adamant|252,252,,,,|N||||';
  const ifSeed = await findSeed(jolt, laxTF, /\|cant\|p2a: Snorlax\|flinch/, 0x24681357);
  console.log('\nB4-2 flinch-pass seed:', ifSeed);
  if (ifSeed) {
    await run('B4-2 INNER FOCUS (roll drawn, flinch blocked — Snorlax moves)', jolt, laxIF, ifSeed, [{ p1: 'move 1', p2: 'move 1' }]);
    await run('B4-2 CONTROL (Thick Fat — the flinch lands, Snorlax cant)', jolt, laxTF, ifSeed, [{ p1: 'move 1', p2: 'move 1' }]);
  }

  // ── B4-3 SHADOW TAG: a FLYING foe is trapped unconditionally; the trap adds ZERO draws
  //    (the Keen-Eye control's seed timeline is IDENTICAL).
  const golduckST = 'Golduck|||ShadowTag|surf,surf|Modest|252,,,252,,|N||||';
  const golduckKE = 'Golduck|||KeenEye|surf,surf|Modest|252,,,252,,|N||||';
  const skarm = 'Skarmory|||KeenEye|drillpeck,drillpeck|Adamant|252,252,,,,|N||||]Snorlax|||ThickFat|bodyslam,bodyslam|Serious||N||||';
  const stSeed = [30303, 11111, 47474, 5252];
  await run('B4-3 SHADOW TAG (Skarmory trapped, 0 extra draws)', golduckST, skarm, stSeed, [{ p1: 'move 1', p2: 'move 1' }]);
  await run('B4-3 CONTROL (Keen Eye — untrapped, IDENTICAL seeds)', golduckKE, skarm, stSeed, [{ p1: 'move 1', p2: 'move 1' }]);

  // ── B4-4 CUTE CHARM: the 1/3 roll PASSES → the M attacker is attracted; the next turn's
  //    attract 1/2 CANTS him. The F-into-F control draws the SAME 1/3 roll (dec-0 seed
  //    IDENTICAL) but never attracts.
  const zangM = 'Zangoose|||Immunity|scratch,scratch|Adamant|,252,,,,252|M||||';
  const zangF = 'Zangoose|||Immunity|scratch,scratch|Adamant|,252,,,,252|F||||';
  const miltank = 'Miltank|||CuteCharm|bodyslam,bodyslam|Adamant|252,252,,,,|F||||';
  const ccSeed = await findSeed(zangM, miltank, /\|-start\|p1a: Zangoose\|Attract[\s\S]*\|cant\|p1a: Zangoose\|Attract/, 0x99881122,
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);
  console.log('\nB4-4 attract-land+cant seed:', ccSeed);
  if (ccSeed) {
    await run('B4-4 CUTE CHARM (M attracted, then attract-cant)', zangM, miltank, ccSeed,
      [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);
    await run('B4-4 CONTROL (F-into-F — the 1/3 roll STILL DRAWS, no attract)', zangF, miltank, ccSeed,
      [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1' }]);
  }

  // ── B4-5 COLOR CHANGE: TBolt → Kecleon becomes Electric (typechange), the next EQ is
  //    SUPER-EFFECTIVE through the override; a switch-out reverts (not exercised here).
  const joltEq = 'Jolteon|||Static|thunderbolt,earthquake|Timid|,,,252,,252|N||||';
  const kecleon = 'Kecleon|||ColorChange|surf,surf|Modest|252,,,252,,|N||||';
  const ccgSeed = [42424, 3141, 59265, 35897];
  await run('B4-5 COLOR CHANGE (TBolt→Electric; EQ is then SE)', joltEq, kecleon, ccgSeed,
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }]);

  // ── B4-6 KING'S ROCK: the appended 10% flinch PASSES on Slash → the slower Snorlax is
  //    cant'd. The no-item control on the SAME seed never draws the extra roll (its seed
  //    timeline differs).
  const zangKR = "Zangoose||kingsrock|Immunity|slash,slash|Adamant|,252,,,,252|N||||";
  const zangNo = 'Zangoose|||Immunity|slash,slash|Adamant|,252,,,,252|N||||';
  const lax = 'Snorlax|||ThickFat|bodyslam,bodyslam|Adamant|252,252,,,,|N||||';
  const krSeed = await findSeed(zangKR, lax, /\|cant\|p2a: Snorlax\|flinch/, 0x31415926);
  console.log('\nB4-6 KR flinch-pass seed:', krSeed);
  if (krSeed) {
    await run("B4-6 KING'S ROCK (the appended roll passes — Snorlax flinched)", zangKR, lax, krSeed, [{ p1: 'move 1', p2: 'move 1' }]);
    await run('B4-6 CONTROL (no item — no extra roll, Snorlax moves)', zangNo, lax, krSeed, [{ p1: 'move 1', p2: 'move 1' }]);
  }

  // ── B4-7 FOCUS BAND: a lethal Cross Chop into the lv-5 FB Rattata — the 1/10 PASSES →
  //    survive at 1 HP (`-activate item: Focus Band`). The no-item control dies on a
  //    DIFFERENT seed timeline (the onDamage roll absent).
  const machamp = 'Machamp|||Guts|crosschop,crosschop|Adamant|,252,,,,252|N||||';
  const rattaFB = 'Rattata||focusband|Guts|scratch,scratch|Serious||N|||5|';
  const rattaNo = 'Rattata|||Guts|scratch,scratch|Serious||N|||5|';
  const fbSeed = await findSeed(machamp, rattaFB, /\|-activate\|p2a: Rattata\|item: Focus Band/, 0x27182818);
  console.log('\nB4-7 FB survive seed:', fbSeed);
  if (fbSeed) {
    await run('B4-7 FOCUS BAND (lethal hit survived at 1 HP)', machamp, rattaFB, fbSeed, [{ p1: 'move 1', p2: 'move 1' }]);
    await run('B4-7 CONTROL (no item — the roll absent, Rattata faints)', machamp, rattaNo, fbSeed, [{ p1: 'move 1', p2: 'move 1' }]);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
