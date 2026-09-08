// GOLDEN GENERATOR — the CONFUSION-MOVE FAMILY differential vector
// (`gen3_confusion_move_family_v1`, ROUND 57).
//
// The hand-written gate in `tests/confusion_family_test.rs` pins ONE seed per disposition, and
// one seed can never reach the MISS path: Supersonic is accuracy 55, so ~45% of its casts miss,
// and a single fixed seed either misses or does not. This generator sweeps **many seeds per
// scenario** so the vector contains both branches of every accuracy, and records the sim's
// SEED_BEFORE / SEED_AFTER plus the filtered `|...|` lines for each.
//
//   node src/rust_sim/harness/gen_confusion_family_golden.js
//     -> tests/vectors/confusion_family_golden.txt
//   then `cargo test --test confusion_family_golden_test` re-pins the port against it.
//
// env: CF_SEEDS (per-scenario seed count, default 24), CF_OUT (output path).
//
// 🚨 THE `splash` CONTROL SCENARIO IS PART OF THE VECTOR AND MUST STAY. A seed row that matches
// for a control move but not for a family member localizes a defect to the move; without it a
// mismatch cannot be told apart from a harness or seeding-convention error, which is exactly the
// ambiguity that made the first read of the ROUND-57 Confuse Ray accuracy bug wrong.
'use strict';
const fs = require('fs');
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream.js'));

const N_SEEDS = parseInt(process.env.CF_SEEDS || '24', 10);
const OUT = process.env.CF_OUT || path.resolve(__dirname, '../tests/vectors/confusion_family_golden.txt');

const GENGAR = (m) =>
  `|Gengar|Leftovers|Levitate|${m},splash,splash,splash|Hardy|85,85,85,85,85,85|M||||`;
const FOES = {
  inert: '|Snorlax|Leftovers|Immunity|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||',
  subber: '|Snorlax|Leftovers|Immunity|substitute,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||',
  owntempo: '|Slowbro|Leftovers|OwnTempo|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||',
  soundproof: '|Voltorb|Leftovers|Soundproof|splash,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||',
  safeguard: '|Snorlax|Leftovers|Immunity|safeguard,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||',
  protect: '|Snorlax|Leftovers|Immunity|protect,splash,splash,splash|Hardy|85,85,85,85,85,85|M||||',
};

// One decision: p1 casts from slot 1.
const T1 = ['>p1 move 1\n>p2 move 1'];
// Two decisions: p2 raises its blocker on turn 1 (p1 splashes from slot 2), cast on turn 2.
const T2 = ['>p1 move 2\n>p2 move 1', '>p1 move 1\n>p2 move 2'];
// Two decisions, both casts — reaches the ALREADY-CONFUSED branch on the second.
const T2CAST = ['>p1 move 1\n>p2 move 1', '>p1 move 1\n>p2 move 1'];

const MOVES = ['confuseray', 'supersonic', 'sweetkiss', 'teeterdance'];
const SCENARIOS = [];
// The CONTROL — a draw-free move on the plain board.
SCENARIOS.push({ name: 'control-splash', move: 'splash', foe: 'inert', script: T1 });
for (const mv of MOVES) {
  SCENARIOS.push({ name: `${mv}-plain`, move: mv, foe: 'inert', script: T1 });
  SCENARIOS.push({ name: `${mv}-twice`, move: mv, foe: 'inert', script: T2CAST });
  SCENARIOS.push({ name: `${mv}-owntempo`, move: mv, foe: 'owntempo', script: T1 });
  SCENARIOS.push({ name: `${mv}-soundproof`, move: mv, foe: 'soundproof', script: T1 });
  SCENARIOS.push({ name: `${mv}-substitute`, move: mv, foe: 'subber', script: T2 });
  SCENARIOS.push({ name: `${mv}-safeguard`, move: mv, foe: 'safeguard', script: T2 });
  // ⚠️ PROTECT GETS T1, NOT T2. Protect lasts only the turn it is used, so the T2 shape
  // (blocker on turn 1, cast on turn 2) leaves the caster hitting an UNPROTECTED foe — the
  // scenario would be named `-protect` and test nothing. The enforced Protect FLOOR in
  // `confusion_family_golden_test.rs` is what caught this: the first draft produced 0 Protect
  // rows and every one of those rows still passed the byte comparison.
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
  const hitRows = rows.filter((r) => r.lines.some((l) => l.includes('|-start|') && l.includes('confusion')));
  const header = [
    '# CONFUSION-MOVE FAMILY GOLDEN (gen3_confusion_move_family_v1)',
    `# generated by harness/gen_confusion_family_golden.js  seeds/scenario=${N_SEEDS}`,
    `# rows=${n} scenarios=${SCENARIOS.length}`,
    `# COVERAGE: rows containing a |-miss| = ${missRows.length}; rows landing a confusion = ${hitRows.length}`,
    '# Each row is one JSON object: the board, the script, the sim SEED_BEFORE/SEED_AFTER and the',
    '# filtered protocol lines. The port is seeded at SEED_BEFORE (the pre-first-decision state).',
    '',
  ].join('\n');
  fs.writeFileSync(OUT, header + rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
  console.log(`wrote ${OUT}`);
  console.log(`rows=${n}  miss-rows=${missRows.length}  confusion-landed-rows=${hitRows.length}`);
  if (missRows.length === 0) {
    console.error('REFUSING: the vector contains NO miss row — the accuracy branch it exists to cover is absent.');
    process.exit(1);
  }
})();
