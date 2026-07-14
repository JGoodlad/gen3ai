// probe_batch4b_regression_rng.js — GROUND TRUTH for the MOVE-COVERAGE BATCH 4b regression
// pins (MC39…MC43, `gen3_move_coverage_batch4b_v1`: BEAT UP + THUNDER + WATER SPOUT) in
// tests/regression_test.rs.
//
// Each scenario drives the OMNISCIENT in-process BattleStream over a CONSTRUCTED gen3customgame
// board whose EXACT packed teams + seed the Rust regression test replays. It RESEEDS the sim's
// prng to the RAW seed right before the first decision (so the decision draws line up with the
// Rust's draw-free `start_with_switchins`), then prints the per-decision seedAfter + observable
// STATE — copied verbatim into the pins as the real-Showdown ground truth. Each pin FAILS if its
// class's engine wiring is reverted.
//
// Run:  node src/rust_sim/harness/probe_batch4b_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

function tick() { return new Promise((r) => setTimeout(r, 0)); }

const RAW = [44317, 42357, 9927, 48760]; // the shared raw seed (the port constructs draw-free with it)

async function run(label, p1, p2, plan, opts = {}) {
  const rawSeed = opts.rawSeed || RAW;
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  // STATE-only injections (HP), then reseed to the raw seed (draw-free construction parity).
  if (opts.inject) for (const inj of opts.inject) {
    const m = b.sides[inj.side].pokemon[inj.slot || 0];
    if (inj.hp !== undefined) m.hp = inj.hp;
  }
  b.prng = new PRNG(rawSeed.slice());
  const a0 = () => b.sides[0].active[0], a1 = () => b.sides[1].active[0];
  console.log(`\n=== ${label} (raw seed ${rawSeed.join(',')})  weather=${b.field.weather || '-'} ===`);
  let i = 0, safety = 0;
  while (!b.ended && safety < 60 && i < plan.length) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = log.length;
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const fmt = (m) => (m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.status ? ' ' + m.status : ''}` : '-');
    console.log(`  dec ${i - 1} [${rs}] ${JSON.stringify(entry)}  seedAfter=${b.prng.getSeed()}`);
    console.log(`      p1=${fmt(a0())}  left=${b.sides[0].pokemonLeft}`);
    console.log(`      p2=${fmt(a1())}  left=${b.sides[1].pokemonLeft}`);
    log.slice(before)
      .filter((l) => /\|move\||-damage|-activate|-hitcount|cant|switch|faint|-crit|-immune|-status|-miss/.test(l))
      .forEach((l) => console.log(`      L ${l}`));
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // ── MC39: BEAT UP full 6-strike (the target SURVIVES → all strikes fire) ──────────────
  //   Slaking + 5 healthy bench each strike the bulky Skarmory once (typeless BP-10 stat swap).
  //   ONE accuracy roll + 6*(crit+damage) + Quick Claw. Skarmory splashes (draw-free).
  const beatupSide =
    'Slaking|||keeneye|beatup,seismictoss|Serious|252,252,,,,|||||' +
    ']Machamp|||noguard|splash|Serious|,,,,,|||||' +
    ']Alakazam|||synchronize|splash|Serious|,,,,,|||||' +
    ']Snorlax|||immunity|splash|Serious|,,,,,|||||' +
    ']Blissey|||naturalcure|splash|Serious|,,,,,|||||' +
    ']Gengar|||keeneye|splash|Serious|,,,,,|||||';
  const skarmSplash = 'Skarmory|||keeneye|splash|Serious|,,,,,|||||';
  await run('MC39 Beat Up 6 strikes (survive)', beatupSide, skarmSplash, [{ p1: 'move 1', p2: 'move 1' }]);

  // ── MC40: BEAT UP KOs the target MID-SEQUENCE (loop stops, NO Quick Claw) ─────────────
  //   Gengar injected to 15 HP → an early strike KOs it; the multihit STOPS. Gengar (faster)
  //   Splashes first (draw-free). p2's only mon → p1 wins (no Quick Claw on the deciding faint).
  const beatupSmall = 'Slaking|||keeneye|beatup,seismictoss|Serious|252,,,,,|||||]Snorlax|||immunity|splash|Serious|,,,,,|||||]Blissey|||naturalcure|splash|Serious|,,,,,|||||';
  const gengarSplash = 'Gengar|||keeneye|splash|Serious|,,,,,252|||||';
  await run('MC40 Beat Up KO mid-sequence', beatupSmall, gengarSplash,
    [{ p1: 'move 1', p2: 'move 1' }], { inject: [{ side: 1, slot: 0, hp: 30 }] });

  // ── MC41: THUNDER in RAIN — never-miss → the accuracy random(100) is SKIPPED ──────────
  //   Zapdos Thunder into a Drizzle Blissey (bulky, survives). NEVER-MISS: 0 accuracy draw.
  const zapThunder = 'Zapdos|||keeneye|thunder,seismictoss|Serious|,,252,,,252|||||';
  const blisseyDrizzle = 'Blissey|||drizzle|splash|Serious|252,,,,,|||||';
  await run('MC41 Thunder RAIN (never-miss, no acc draw)', zapThunder, blisseyDrizzle, [{ p1: 'move 1', p2: 'move 1' }]);

  //   CONTROL: THUNDER with NO weather (base acc 70) — same board, No Ability Blissey. The
  //   accuracy roll DRAWS (at 44317 it MISSES) → a DIFFERENT post-turn seed than rain.
  const blisseyPlain = 'Blissey|||naturalcure|splash|Serious|252,,,,,|||||';
  await run('MC41c Thunder BASE (acc 70 drawn)', zapThunder, blisseyPlain, [{ p1: 'move 1', p2: 'move 1' }]);

  // ── MC42: THUNDER in SUN — base accuracy 50 (a lower threshold) ──────────────────────
  //   Sweep for a seed where random(100) ∈ [50,70): SUN (thresh 50) MISSES while BASE (70)
  //   HITS — the threshold pin (revert of the sun→50 makes sun behave like base and HIT).
  const groudonDrought = 'Blissey|||drought|splash|Serious|252,,,,,|||||';
  // find a raw seed whose FIRST random(100) is in [50,70)
  let sunSeed = null;
  for (let s = 1; s < 4000 && !sunSeed; s++) {
    const p = new PRNG([s, s * 3 + 1, s * 7 + 2, s * 11 + 3]);
    const roll = p.random(100);
    if (roll >= 50 && roll < 70) sunSeed = { seed: [s, s * 3 + 1, s * 7 + 2, s * 11 + 3], roll };
  }
  console.log(`\n[sun-threshold seed search] ${JSON.stringify(sunSeed)}`);
  await run('MC42 Thunder SUN (thresh 50 → MISS at mid-roll)', zapThunder, groudonDrought,
    [{ p1: 'move 1', p2: 'move 1' }], { rawSeed: sunSeed.seed });
  await run('MC42c Thunder BASE (thresh 70 → HIT at the same mid-roll)', zapThunder, blisseyPlain,
    [{ p1: 'move 1', p2: 'move 1' }], { rawSeed: sunSeed.seed });

  // ── MC43: WATER SPOUT variable BP — full HP (bp 150) vs low HP (bp 74) at the SAME seed ─
  //   Draw-NEUTRAL: same post-turn seed, DIFFERENT damage. Kyogre Water Spout into a bulky
  //   Snorlax (survives), which Splashes (draw-free).
  const kyogreWS = 'Kyogre|||keeneye|waterspout,seismictoss|Serious|,,252,,,252|||||';
  const snorlaxSplash = 'Snorlax|||immunity|splash|Serious|252,,,,,|||||';
  await run('MC43 Water Spout FULL HP (bp 150)', kyogreWS, snorlaxSplash, [{ p1: 'move 1', p2: 'move 1' }]);
  await run('MC43b Water Spout LOW HP (bp ~74)', kyogreWS, snorlaxSplash,
    [{ p1: 'move 1', p2: 'move 1' }], { inject: [{ side: 0, slot: 0, hp: 179 }] }); // 179/341 → floor(150*179/341)=78

  // ── MC44: BEAT UP MIRROR at a SPEED TIE — BOTH Charizards Beat Up at equal speed. This
  //   exercises TWO tie-only draws the distinct-speed scenarios hide: (a) the multihit loop's
  //   PER-STRIKE eachEvent('Update') (drawn after each strike on a tie), and (b) the `beatup`
  //   `duration: 1` volatile's residual DURATION handler (two beatup volatiles tie → one
  //   residual shuffle). Reverting EITHER desyncs the post-turn seed.
  const charBU1 = 'Charizard|||Blaze|beatup,seismictoss|Modest|,,,252,,252|N||||]Slaking|||keeneye|splash|Serious|,,,,,|N||||]Machamp|||noguard|splash|Serious|,,,,,|N||||';
  const charBU2 = 'Charizard|||Blaze|beatup,roost|Modest|,,,252,,252|N||||]Magikarp|||keeneye|splash|Serious|,,,,,|N|||5|';
  await run('MC44 Beat Up mirror at a speed tie', charBU1, charBU2, [{ p1: 'move 1', p2: 'move 1' }]);

  // ── MC45: BEAT UP CANCELS A FOCUS PUNCH — p2 Charizard Beat Ups (priority 0) into p1
  //   Charizard, which chose Focus Punch (priority -3 → moves LAST). The direct strikes set
  //   lostFocus → the Focus Punch is CANCELLED (0 damage to p2). Reverting the lostFocus set
  //   in run_beat_up wrongly lets the Focus Punch land.
  const fpUser = 'Charizard|||Blaze|focuspunch,seismictoss|Modest|,4,,,,252|N||||';
  const buFoe = 'Charizard|||Blaze|beatup,roost|Modest|,,,252,,252|N||||]Slaking|||keeneye|splash|Serious|,,,,,|N||||]Machamp|||noguard|splash|Serious|,,,,,|N||||';
  await run('MC45 Beat Up cancels a Focus Punch', fpUser, buFoe, [{ p1: 'move 1', p2: 'move 1' }]);
}

main().catch((e) => { console.error(e && e.stack ? e.stack : String(e)); process.exit(1); });
