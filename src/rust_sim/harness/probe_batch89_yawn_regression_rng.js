// probe_batch89_yawn_regression_rng.js — GROUND TRUTH for the `gen3_yawn_v1` YAWN regression pins
// (Y1 the resolve draws random(2,6) at the RIGHT turn; Y2 the yawn MIRROR residual (10,19) tie;
// Y3 the gen3ou Sleep-Clause-at-resolve) in tests/regression_test.rs.
//
// Drives the OMNISCIENT in-process BattleStream (no server), RESEEDED to a RAW seed right before the
// first decision (matching the Rust's draw-free `start_with_switchins`), and prints each decision's
// seedAfter + both actives' status/sleep-counter + a clause/resolve marker. Yawn's CAST is DRAW-FREE;
// the sleep `random(2,6)` fires at the RESOLVE (end of the turn AFTER cast). A gen3ou clause-blocked
// resolve draws the SetStatus 2-clause shuffle but NO random(2,6).
//
// Run:  node src/rust_sim/harness/probe_batch89_yawn_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }

const st = (a) => {
  if (!a) return '-';
  const t = a.status === 'slp' ? `slp:${(a.statusState && a.statusState.time) || 0}` : (a.status || '-');
  return `hp=${a.hp}/${a.maxhp} ${t}`;
};

async function run(label, format, p1, p2, rawSeed, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${format}","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  b.prng = new PRNG(rawSeed.slice());
  console.log(`\n=== ${label} [${format}] (raw seed ${rawSeed.join(',')}) ===`);
  let i = 0;
  for (const entry of plan) {
    const before = log.length;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const chunk = log.slice(before);
    const resolved = chunk.some((l) => l.startsWith('|-end|') && l.includes('move: Yawn'));
    const clause = chunk.some((l) => l.startsWith('|-message|Sleep Clause'));
    const a0 = b.sides[0].active[0], a1 = b.sides[1].active[0];
    console.log(`  dec ${i} ${JSON.stringify(entry)} yawnResolved=${resolved} sleepClause=${clause}`);
    console.log(`    seedAfter=${b.prng.getSeed()}`);
    console.log(`    p1 ${st(a0)} | p2 ${st(a1)}`);
    i++;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // Y1: the resolve draws random(2,6) at the RIGHT turn. Yawn (dec0, DRAW-FREE cast) → Splash (dec1,
  //   the yawn RESOLVES → Blissey sleeps, the random(2,6) draws HERE). A Splash control has an
  //   IDENTICAL dec0 (cast draw-free) but a DIFFERENT dec1 (no resolve draw).
  const lax = "Snorlax|||Immunity|yawn,splash,earthquake,thunderwave|Adamant|4,252,,,252,|N||||";
  const bliss = "Blissey|||NaturalCure|splash|Careful|252,,,252,,|N||||";
  const seed1 = [13, 27, 41, 55];
  await run('Y1 yawn cast->resolve', 'gen3customgame', lax, bliss, seed1, [
    { p1: 'move 1', p2: 'move 1' },  // dec0: Yawn cast (DRAW-FREE)
    { p1: 'move 2', p2: 'move 1' },  // dec1: Splash — the yawn RESOLVES (random(2,6) here)
  ]);
  await run('Y1 splash control', 'gen3customgame', lax, bliss, seed1, [
    { p1: 'move 2', p2: 'move 1' },  // dec0: Splash — SAME draws as the Yawn cast (draw-free)
    { p1: 'move 2', p2: 'move 1' },  // dec1: Splash — NO resolve draw (the +random(2,6) discriminator)
  ]);

  // Y2: the yawn MIRROR residual (10,19) tie. BOTH Snorlax cast Yawn on each other (dec0) at EQUAL
  //   speed; both resolve at end of dec1 → the two yawn residual handlers TIE at (order 10, subOrder
  //   19, equal speed) → ONE Fisher-Yates tie-shuffle random(0,2), PLUS two random(2,6). A control
  //   with only ONE side yawning (equal-speed board) resolves ONE yawn (one random(2,6), NO tie).
  const laxA = "Snorlax|||Immunity|yawn,splash|Adamant|4,252,,,252,|N||||";
  const laxB = "Snorlax|||Immunity|yawn,splash|Adamant|4,252,,,252,|N||||";
  const seed2 = [17, 33, 49, 61];
  await run('Y2 yawn MIRROR (equal speed tie)', 'gen3customgame', laxA, laxB, seed2, [
    { p1: 'move 1', p2: 'move 1' },  // dec0: BOTH cast Yawn (draw-free)
    { p1: 'move 2', p2: 'move 2' },  // dec1: BOTH Splash — BOTH yawns resolve (tie-shuffle + 2×random(2,6))
  ]);
  await run('Y2 single-yawn control (equal speed, no tie)', 'gen3customgame', laxA, laxB, seed2, [
    { p1: 'move 1', p2: 'move 2' },  // dec0: p1 Yawn / p2 Splash
    { p1: 'move 2', p2: 'move 2' },  // dec1: BOTH Splash — ONLY p2's mon has a yawn → ONE random(2,6), NO tie
  ]);

  // Y3: the gen3ou Sleep-Clause-at-resolve. Smeargle Spores p2's Blissey-A (dec0 → asleep), p2
  //   switches A→B while p1 Yawns B (dec1), then Splash (dec2) — the yawn on B RESOLVES but p2's side
  //   already has a (foe-inflicted) sleeper (benched A) → the Sleep Clause Mod BLOCKS the sleep at
  //   the SetStatus event: it DRAWS the 2-clause shuffle but NO random(2,6) (B stays awake). A
  //   no-prior-sleeper control (p2 never had A slept) resolves normally: the shuffle + random(2,6).
  const smeargle = "Smeargle|||Own Tempo|spore,yawn,splash,earthquake|Jolly|4,,,,,252|N||||";
  // Two distinct Blisseys (nickname|species) with SERENE GRACE (NOT Natural Cure — else the slept
  // Blissey-A would be CURED on switch-out and the clause would never see a benched sleeper).
  const blissA = "A|Blissey||Serene Grace|splash|Careful|252,,,252,,|N||||";
  const blissB = "B|Blissey||Serene Grace|splash|Careful|252,,,252,,|N||||";
  const p2two = `${blissA}]${blissB}`;
  const seed3 = [23, 37, 51, 63];
  await run('Y3 gen3ou clause-block at resolve', 'gen3ou', smeargle, p2two, seed3, [
    { p1: 'move 1', p2: 'move 1' },   // dec0: Spore -> Blissey-A asleep
    { p1: 'move 2', p2: 'switch 2' }, // dec1: p2 switch A->B; p1 Yawn B (cast)
    { p1: 'move 3', p2: 'move 1' },   // dec2: Splash — the yawn on B RESOLVES -> CLAUSE BLOCK (no random(2,6))
  ]);
  await run('Y3 gen3ou control (no prior sleeper)', 'gen3ou', smeargle, p2two, seed3, [
    { p1: 'move 3', p2: 'move 1' },   // dec0: Splash (no Spore)
    { p1: 'move 2', p2: 'switch 2' }, // dec1: p2 switch A->B; p1 Yawn B (cast)
    { p1: 'move 3', p2: 'move 1' },   // dec2: Splash — the yawn on B RESOLVES normally (shuffle + random(2,6))
  ]);
}
main();
