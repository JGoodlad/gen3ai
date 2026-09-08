// GOLDEN GENERATOR — the SPREAD STAT-DROP family differential vector
// (`gen3_spread_stat_drop_v1`, ROUND 58).
//
// The five gen-3 stat-drop status moves whose target is `allAdjacentFoes` — leer / growl /
// tailwhip / stringshot / sweetscent. They reached the engine as a pure DATA change (the
// extractor's `_stat_drop_boosts` target gate widened to admit `allAdjacentFoes`, which in
// SINGLES resolves to the one foe); no engine line was written for them. **That is exactly why
// they need a differential**: "the existing arm should serve them" is a claim about behaviour,
// and the only thing that settles it is running both engines side by side.
//
//   node src/rust_sim/harness/gen_spread_stat_drop_golden.js
//     -> tests/vectors/spread_stat_drop_golden.txt
//   then `cargo test --test spread_stat_drop_golden_test` re-pins the port against it.
//
// env: CF_SEEDS (per-scenario seed count, default 24), CF_OUT (output path).
//
// 🚨 The `screech` CONTROL scenario is a `target: normal` member of the SAME arm. If a spread
// move diverges while Screech does not, the defect is in the target widening; if both diverge,
// it is in the arm. Without it a failure cannot be attributed to either.
// 🚨 The `-protect` scenarios use the ONE-decision script on purpose: Protect lasts only the turn
// it is used, so the blocker-on-turn-1 shape used for Substitute/Safeguard would leave the caster
// hitting an UNPROTECTED foe — a scenario that is perfectly green and perfectly vacuous. ROUND 57
// shipped that bug into its first draft and only the enforced floor caught it.
'use strict';
const fs = require('fs');
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream.js'));

const N_SEEDS = parseInt(process.env.CF_SEEDS || '24', 10);
const OUT = process.env.CF_OUT || path.resolve(__dirname, '../tests/vectors/spread_stat_drop_golden.txt');

const GENGAR = (m) =>
  `|Gengar|Leftovers|Levitate|${m},splash,splash,splash|Hardy|85,85,85,85,85,85|M||||`;
const FOES = {
  inert: '|Snorlax|Leftovers|Immunity|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||',
  subber: '|Snorlax|Leftovers|Immunity|substitute,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||',
  clearbody: '|Regirock|Leftovers|ClearBody|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||',
  hypercutter: '|Pinsir|Leftovers|HyperCutter|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||',
  keeneye: '|Hitmonchan|Leftovers|KeenEye|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||',
  soundproof: '|Voltorb|Leftovers|Soundproof|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||',
  protect: '|Snorlax|Leftovers|Immunity|protect,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||',
};

// One decision: p1 casts from slot 1.
const T1 = ['>p1 move 1\n>p2 move 1'];
// Two decisions: p2 raises its blocker on turn 1 (p1 splashes from slot 2), cast on turn 2.
const T2 = ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2'];
// Two decisions, both casts — reaches the ALREADY-CONFUSED branch on the second.
// Seven casts — the last lands on a stat already AT the −6 floor.
const T7 = Array(7).fill('>p1 move 1\n>p2 move 1');

const MOVES = ['leer', 'growl', 'tailwhip', 'stringshot', 'sweetscent'];
const SCENARIOS = [];
// The CONTROLS — a draw-free move, and a `target: normal` member of the SAME arm.
SCENARIOS.push({ name: 'control-splash', move: 'splash', foe: 'inert', script: T1 });
SCENARIOS.push({ name: 'control-screech-normaltarget', move: 'screech', foe: 'inert', script: T1 });
SCENARIOS.push({ name: 'control-screech-clearbody', move: 'screech', foe: 'clearbody', script: T1 });
for (const mv of MOVES) {
  SCENARIOS.push({ name: `${mv}-plain`, move: mv, foe: 'inert', script: T1 });
  // SEVEN casts drive the stat to the −6 floor, so the vector carries the DELTA-0 `|-unboost|…|0`
  // line — the cap-before-TryBoost form ROUND 55 pinned for Intimidate.
  SCENARIOS.push({ name: `${mv}-to-the-floor`, move: mv, foe: 'inert', script: T7 });
  SCENARIOS.push({ name: `${mv}-clearbody`, move: mv, foe: 'clearbody', script: T1 });
  SCENARIOS.push({ name: `${mv}-hypercutter`, move: mv, foe: 'hypercutter', script: T1 });
  SCENARIOS.push({ name: `${mv}-keeneye`, move: mv, foe: 'keeneye', script: T1 });
  SCENARIOS.push({ name: `${mv}-soundproof`, move: mv, foe: 'soundproof', script: T1 });
  SCENARIOS.push({ name: `${mv}-substitute`, move: mv, foe: 'subber', script: T2 });
  SCENARIOS.push({ name: `${mv}-protect`, move: mv, foe: 'protect', script: T1 });
}

// The gated line types — the same shape `protocol_test.rs` filters to. `|t:|` is wall-clock and
// poke-env ignores it; `debug`/`error` are dropped from BOTH sides everywhere in this project.
const KEEP = /^\|(move|-start|-end|-fail|-immune|-activate|-miss|-boost|-unboost|-status|-damage|-heal|turn)\|/;

// 🚨 READ THE **OMNISCIENT** STREAM, never the raw `BattleStream` chunks.
// The raw stream carries `|split|pN` markers: each HP-revealing event is emitted TWICE — the
// owner's exact `x/y` line and the spectator's percentage line. A naive chunk join therefore
// records every `-damage` / `-heal` twice, and the resulting golden accuses a CORRECT port of
// dropping lines. `getPlayerStreams(...).omniscient` is the referee view that resolves the split,
// and it is what `harness/gen_protocol_capture.js` has always used.
async function runOne(p1, p2, script, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const all = [];
  (async () => {
    for await (const ch of streams.omniscient) {
      for (const l of ch.split('\n')) all.push(l);
    }
  })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[${seed.join(',')}]}`);
  streams.omniscient.write(`>player p1 {"name":"P1","team":"${p1}"}`);
  streams.omniscient.write(`>player p2 {"name":"P2","team":"${p2}"}`);
  await new Promise((r) => setTimeout(r, 120));
  const before = stream.battle.prng.getSeed();
  for (const c of script) {
    for (const part of c.split('\n')) streams.omniscient.write(part);
    await new Promise((r) => setTimeout(r, 120));
  }
  const after = stream.battle.prng.getSeed();
  const i = all.indexOf('|turn|1');
  if (i < 0) throw new Error('no |turn|1 framing marker in the omniscient stream');
  const lines = all.slice(i + 1).filter((l) => KEEP.test(l));
  return { before, after, lines };
}

(async () => {
  const rows = [];
  let n = 0;
  for (const sc of SCENARIOS) {
    for (let k = 0; k < N_SEEDS; k++) {
      // A spread of start seeds; the ACCURACY branch varies with them, which is the point.
      const seed = [1000 + k * 7, 2000 + k * 13, 3000 + k * 29, 4000 + k * 51];
      const r = await runOne(GENGAR(sc.move), FOES[sc.foe], sc.script, seed);
      rows.push({
        name: sc.name,
        move: sc.move,
        p1: GENGAR(sc.move),
        p2: FOES[sc.foe],
        script: sc.script,
        seedBefore: r.before,
        seedAfter: r.after,
        lines: r.lines,
      });
      n++;
    }
  }
  // A DISCLOSED coverage census — the vector must actually contain both accuracy branches, or
  // the miss path it exists to cover is untested and nobody would know.
  const missRows = rows.filter((r) => r.lines.some((l) => l.includes('|-miss|')));
  const hitRows = rows.filter((r) => r.lines.some((l) => /^\|-unboost\|/.test(l)));
  const floorRows = rows.filter((r) => r.lines.some((l) => /^\|-unboost\|.*\|0$/.test(l)));
  const blockRows = rows.filter((r) => r.lines.some((l) => l.includes('|-fail|') && l.includes('ability:')));
  const header = [
    '# SPREAD STAT-DROP GOLDEN (gen3_spread_stat_drop_v1)',
    `# generated by harness/gen_spread_stat_drop_golden.js  seeds/scenario=${N_SEEDS}`,
    `# rows=${n} scenarios=${SCENARIOS.length}`,
    `# COVERAGE: |-miss| rows = ${missRows.length}; |-unboost| rows = ${hitRows.length}; ` +
      `DELTA-0 floor rows = ${floorRows.length}; ability-blocked rows = ${blockRows.length}`,
    '# Each row is one JSON object: the board, the script, the sim SEED_BEFORE/SEED_AFTER and the',
    '# filtered protocol lines. The port is seeded at SEED_BEFORE (the pre-first-decision state).',
    '',
  ].join('\n');
  fs.writeFileSync(OUT, header + rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
  console.log(`wrote ${OUT}`);
  console.log(`rows=${n}  miss=${missRows.length}  unboost=${hitRows.length}  floor=${floorRows.length}  blocked=${blockRows.length}`);
  if (floorRows.length === 0) {
    console.error('REFUSING: no DELTA-0 floor row — the cap-before-TryBoost form is uncovered.');
    process.exit(1);
  }
  if (missRows.length === 0) {
    console.error('REFUSING: the vector contains NO miss row — the accuracy branch it exists to cover is absent.');
    process.exit(1);
  }
})();
